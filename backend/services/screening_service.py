"""Screening orchestration: request in, persisted clinical result out.

This service owns the *request* concerns -- pseudonymous patient records, persistence,
audit logging, FHIR export -- and delegates every numerical decision to
:class:`~backend.services.inference_service.InferenceService`. It does not build
circuits, does not calibrate, and does not choose thresholds.

How the image reaches the model
-------------------------------
Two accepted forms, checked in this order:

1. ``features`` -- the on-device image descriptor. Preferred (PART 19): the
   photograph stays on the phone. Must match the served model's descriptor length
   exactly; a mismatch is a 422 that names the expected length.
2. ``image_path`` -- a file the backend can read. The backend then runs quality
   control, ROI extraction and feature extraction itself.

Neither is fabricated. If ``features`` is absent and ``image_path`` does not resolve
to a readable image, the request fails with a 422 rather than being scored against a
synthesised feature vector -- which is what the previous implementation did, and what
the FINAL EXECUTION RULE forbids.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from backend.db import crud
from backend.ml.features_clinical import row_from_flags
from backend.schemas.fhir import (
    FHIRCodeableConcept,
    FHIRCoding,
    FHIRObservation,
    FHIRRiskAssessment,
)
from backend.schemas.inference import InferenceResult
from backend.schemas.result import AssessmentResultResponse
from backend.schemas.screening import (
    AssessmentResponse,
    HistoryEntryResponse,
    ScreeningCreate,
)
from backend.services.inference_service import (
    InferenceError,
    InferenceService,
    get_inference_service,
)
from backend.services.mock_inference import mock_inference

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 25 * 1024 * 1024


class ImageUnavailableError(InferenceError):
    """No usable image reached the backend: no descriptor, and no readable file."""

    status_code = 422


class ScreeningService:
    """Coordinates inference, persistence, audit, and FHIR interoperability."""

    def __init__(self, inference: Optional[InferenceService] = None):
        # Injected rather than constructed so tests can supply a service pointed at a
        # fixture model version. Resolved lazily -- constructing this service must not
        # load artifacts, because the FastAPI app builds it at import time.
        self._inference = inference

    @property
    def inference(self) -> InferenceService:
        if self._inference is None:
            self._inference = get_inference_service()
        return self._inference

    def analyze_screening(
        self,
        db: Session,
        screening_in: ScreeningCreate,
        actor_id: str = "system",
    ) -> AssessmentResultResponse:
        """Run the end-to-end screening pipeline and persist the result.

        Raises:
            InferenceError: any failure whose ``str()`` is safe to return to the
                client and whose ``status_code`` the route should use.
        """
        patient = crud.get_or_create_patient(db, patient_id=screening_in.patient_id)
        screening = crud.create_screening(db, screening_in, patient)

        try:
            inference = self._run_inference(screening_in)
        except InferenceError:
            # The screening row stays, marked FAILED, so a clinician sees that an
            # attempt was made rather than nothing at all.
            screening.status = "FAILED"
            db.add(screening)
            db.commit()
            crud.create_audit_log(
                db=db,
                action="SCREENING_FAILED",
                entity_type="screening",
                entity_id=screening.id,
                actor_id=actor_id,
                details="Inference did not complete.",
            )
            raise

        result = crud.record_inference(db=db, screening=screening, inference=inference)

        crud.create_audit_log(
            db=db,
            action="SCREENING_ANALYZE",
            entity_type="screening",
            entity_id=screening.id,
            actor_id=actor_id,
            # Model version and inference ID, not the probability: an audit trail
            # should identify the computation, and the result row already holds
            # the numbers.
            details=(
                f"model={inference.model_version} inference={inference.inference_id} "
                f"mode={inference.quantum.execution_mode} mock={inference.is_mock}"
            ),
        )

        return AssessmentResultResponse.model_validate(result)

    # -------------------------------------------------------------- inference
    def _run_inference(self, screening_in: ScreeningCreate) -> InferenceResult:
        """Route the request to the mock stub, the descriptor path, or the image path."""
        if screening_in.is_mock:
            logger.info("Serving a labelled mock result by request.")
            return mock_inference(
                seed_material=(
                    f"{screening_in.patient_id}|{screening_in.image_path}"
                    f"|{screening_in.scan_type}"
                )
            )

        clinical_row = row_from_flags(
            smoking=screening_in.smoking_history,
            alcohol=screening_in.alcohol_consumption,
            betel_quid=screening_in.betel_quid,
            age=screening_in.age,
            sex=screening_in.sex,
        )

        if screening_in.features:
            return self.inference.infer_from_descriptor(
                screening_in.features, clinical_row=clinical_row
            )

        path = self._resolve_image(screening_in.image_path)
        size = path.stat().st_size
        try:
            with Image.open(path) as image:
                image.load()
                return self.inference.infer_from_image(
                    image, clinical_row=clinical_row, file_size_bytes=size
                )
        except (UnidentifiedImageError, OSError) as exc:
            logger.warning("Could not decode %s: %s", path.name, exc)
            raise ImageUnavailableError(
                "The image could not be read. Please capture or upload it again."
            ) from exc

    @staticmethod
    def _resolve_image(image_path: str) -> Path:
        """Validate that the path names a readable image file of a sane size.

        Validation at the boundary (PART 34), and the error text never echoes the
        path back -- it may contain a filename a patient supplied.
        """
        if not image_path:
            raise ImageUnavailableError(
                "No image was supplied. Capture a photograph and try again."
            )
        path = Path(image_path)
        if not path.is_file():
            logger.warning("Requested image is not present on this server: %r", image_path)
            raise ImageUnavailableError(
                "The image is not available on the server. Upload it via "
                "/api/screening/upload first, or send the on-device descriptor."
            )
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ImageUnavailableError(
                "The image is too large to process. Capture it at a lower resolution."
            )
        return path

    def get_patient_history(self, db: Session, patient_id: str) -> List[HistoryEntryResponse]:
        """
        Retrieve screening history for a patient matching Group 1 Flutter structure.
        """
        screenings = crud.get_patient_screenings(db, patient_id)
        history_entries = []

        for s in screenings:
            res = crud.get_screening_result(db, s.id)
            res_schema = AssessmentResultResponse.model_validate(res) if res else None
            assessment_schema = AssessmentResponse.model_validate(s)
            
            history_entries.append(
                HistoryEntryResponse(
                    assessment=assessment_schema,
                    result=res_schema,
                )
            )

        return history_entries

    def build_fhir_resources(self, db: Session, screening_id: str) -> Optional[Dict[str, Any]]:
        """
        Generate HL7 FHIR R4 Observation and RiskAssessment resources for a screening.
        """
        screening = crud.get_screening(db, screening_id)
        if not screening:
            return None

        result = crud.get_screening_result(db, screening_id)
        if not result:
            return None

        # 1. FHIR Observation
        observation = FHIRObservation(
            id=f"obs-{screening.id}",
            subject={"reference": f"Patient/{screening.patient_id}"},
            effectiveDateTime=screening.created_at.isoformat(),
            valueQuantity={
                "value": round(result.final_probability, 4),
                "unit": "probability",
                "system": "http://unitsofmeasure.org",
                "code": "1",
            },
            interpretation=[
                FHIRCodeableConcept(
                    coding=[
                        FHIRCoding(
                            code="POS" if result.classification == "screening_positive" else "NEG",
                            display="Positive" if result.classification == "screening_positive" else "Negative",
                        )
                    ],
                    text=result.risk_level,
                )
            ],
            note=[{"text": result.details}],
        )

        # 2. FHIR RiskAssessment
        risk_assessment = FHIRRiskAssessment(
            id=f"risk-{screening.id}",
            subject={"reference": f"Patient/{screening.patient_id}"},
            occurrenceDateTime=result.created_at.isoformat(),
            basis=[{"reference": f"Observation/obs-{screening.id}"}],
            prediction=[
                {
                    "outcome": {
                        "text": "Oral Cavity Malignant / Premalignant Lesion Risk",
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "363349007",
                                "display": "Malignant tumor of oral cavity",
                            }
                        ],
                    },
                    "probabilityDecimal": round(result.final_probability, 4),
                    "qualitativeRisk": {
                        "text": result.risk_level,
                    },
                }
            ],
            note=[{"text": "AI-assisted screening risk estimate only. Histopathological confirmation required."}],
        )

        return {
            "observation": observation.model_dump(),
            "risk_assessment": risk_assessment.model_dump(),
        }
