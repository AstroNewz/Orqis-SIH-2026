"""HTTP surface for screening.

Path convention
---------------
PART 18 suggests ``/api/v1/inference``. This repository already ships a Flutter
client against ``/api/screening/analyze``, ``/api/results/{id}`` and
``/api/patients/{id}/history`` (see ARCHITECTURE.md and
``lib/data/datasources/assessment_api_datasource.dart``), and PART 18 also says not to
adopt the suggested paths blindly where the existing architecture specifies another
contract. The existing paths are kept and ``/api/model/info`` is added alongside them;
DEC-020 records the reconciliation.

Error mapping
-------------
Inference failures carry their own ``status_code`` and a message written to be shown
to a patient. They are translated here rather than collapsed into a 500, so a poor
photograph reads as "retake it" (422) and an unloaded model reads as "try shortly"
(503). Anything *not* an :class:`InferenceError` is an unexpected fault: it is logged
with a traceback server-side and returned as a bare 500 with no detail, because
PART 34 forbids exposing stack traces to the client.
"""

import logging
import os
import shutil
import uuid
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from backend.db import crud
from backend.db.session import get_db
from backend.schemas.result import AssessmentResultResponse
from backend.schemas.screening import (
    AssessmentResponse,
    HistoryEntryResponse,
    ScreeningCreate,
)
from backend.services.inference_service import InferenceError, get_inference_service
from backend.services.screening_service import ScreeningService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Screening"])
screening_service = ScreeningService()

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Extensions the image pipeline can actually decode. Whitelisted rather than
# blacklisted, and the *extension the server chooses* is derived from this list --
# never from the client-supplied filename, which is attacker-controlled.
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


@router.post("/screening/upload", status_code=status.HTTP_201_CREATED)
async def upload_screening_image(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
) -> Dict[str, str]:
    """Accept an intra-oral photograph and return a server-side path for analysis.

    Streamed in bounded chunks and cut off at :data:`MAX_UPLOAD_BYTES`, so a large
    body cannot exhaust memory or disk. The stored filename is a fresh UUID -- the
    client's filename is used only to pick an extension, and only after the extension
    has been checked against :data:`ALLOWED_IMAGE_EXTENSIONS`.
    """
    original_ext = os.path.splitext(file.filename or "")[1].lower()
    if original_ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Unsupported image format. Supported: "
                + ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
            ),
        )

    safe_filename = f"{uuid.uuid4()}{original_ext}"
    dest_path = os.path.join(UPLOAD_DIR, safe_filename)

    written = 0
    try:
        with open(dest_path, "wb") as buffer:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            "Image exceeds the "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit. "
                            "Capture at a lower resolution."
                        ),
                    )
                buffer.write(chunk)
    except HTTPException:
        # Do not leave a truncated file behind for the analyze call to trip over.
        _discard(dest_path)
        raise
    except OSError as exc:
        _discard(dest_path)
        logger.exception("Could not store upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The image could not be stored. Please try again.",
        ) from exc
    finally:
        await file.close()

    if written == 0:
        _discard(dest_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image was empty. Please capture it again.",
        )

    logger.info("Stored upload %s (%d bytes)", safe_filename, written)
    return {
        "status": "uploaded",
        "image_path": dest_path,
        "patient_id": patient_id or str(uuid.uuid4()),
        "size_bytes": str(written),
    }


def _discard(path: str) -> None:
    """Remove a partially written upload, ignoring a missing file."""
    try:
        os.unlink(path)
    except OSError:
        pass


@router.post("/screening/analyze", response_model=AssessmentResultResponse)
def analyze_screening(
    payload: ScreeningCreate,
    db: Session = Depends(get_db),
) -> AssessmentResultResponse:
    """Run the end-to-end screening pipeline: quality control, feature extraction,
    amplitude encoding, VQC inference, calibration, risk classification, persistence.

    Returns 422 when the submission itself cannot be scored (unusable photograph,
    wrong descriptor length) and 503 when the model or quantum backend is not
    available. Both carry a message intended for the patient.
    """
    try:
        return screening_service.analyze_screening(db=db, screening_in=payload)
    except InferenceError as exc:
        # str(exc) is patient-safe by construction -- every subclass of InferenceError
        # is documented to carry a message that can be shown to a client.
        logger.info("Screening rejected (%d): %s", exc.status_code, exc)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected failure during screening analysis: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Screening analysis failed unexpectedly. Please try again.",
        ) from exc


@router.get("/model/info")
def model_info() -> Dict[str, Any]:
    """Describe the served model: version, qubits, depth, calibration, feature contract.

    The client needs ``expected_descriptor_dimension`` from here before it can send an
    on-device descriptor. Reports ``ready: false`` rather than failing when no trained
    artifacts are present, so a fresh checkout can still start the backend.
    """
    return get_inference_service().describe()


@router.get("/screening/{screening_id}", response_model=AssessmentResponse)
def get_screening(
    screening_id: str,
    db: Session = Depends(get_db),
) -> AssessmentResponse:
    """Retrieve screening session metadata by ID."""
    screening = crud.get_screening(db, screening_id)
    if not screening:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Screening session '{screening_id}' not found",
        )
    return AssessmentResponse.model_validate(screening)


@router.get("/results/{screening_id}", response_model=AssessmentResultResponse)
def get_screening_result(
    screening_id: str,
    db: Session = Depends(get_db),
) -> AssessmentResultResponse:
    """Retrieve screening assessment results by screening ID."""
    result = crud.get_screening_result(db, screening_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result for screening '{screening_id}' not found",
        )
    return AssessmentResultResponse.model_validate(result)


@router.get("/patients/{patient_id}/history", response_model=List[HistoryEntryResponse])
def get_patient_history(
    patient_id: str,
    db: Session = Depends(get_db),
) -> List[HistoryEntryResponse]:
    """Retrieve chronological screening history for a specific patient."""
    return screening_service.get_patient_history(db, patient_id)


@router.get("/screening/{screening_id}/fhir")
def get_fhir_screening_report(
    screening_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Export screening result as interoperable HL7 FHIR R4 resources."""
    fhir_data = screening_service.build_fhir_resources(db, screening_id)
    if not fhir_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"FHIR resource for screening '{screening_id}' not found",
        )
    return fhir_data
