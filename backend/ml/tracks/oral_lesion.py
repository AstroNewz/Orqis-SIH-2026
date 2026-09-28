"""The oral-lesion track: an adapter over the existing inference service.

This file adds a second way to *reach* :class:`~backend.services.inference_service.
InferenceService`. It does not change it. No import runs in the other direction, the
service has no idea this module exists, and ``/api/analyze`` continues to return
:class:`~backend.schemas.inference.InferenceResult` byte for byte. That constraint is the
whole point of making :class:`~backend.ml.track.DiseaseTrack` a protocol rather than a base
class: the oldest, most-tested path in the repository should not have to move to make room
for a second condition.

The translation that matters here is the verdict. ``InferenceResult`` carries two models'
opinions with a documented rule for which one headlines (DEC-034: the validated classical
baseline outranks the VQC on this dataset, so the classical *band* leads while the
calibrated *probability* stays the quantum one, because it is the only calibrated number
available). :class:`~backend.ml.track.TrackAssessment` has to say the same thing in the
platform's shared vocabulary without quietly promoting the uncalibrated classical score to
a percentage along the way -- hence ``probability_is_calibrated``, which carries that
warning as data instead of as a docstring nobody reads at render time.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from backend.ml.track import (
    InputSpec,
    Modality,
    RiskBand,
    TrackAssessment,
    TrackDescriptor,
    TrackInputError,
    TrackNotReadyError,
    ValidationSummary,
)

__all__ = ["OralLesionTrack", "ORAL_TRACK_ID"]

ORAL_TRACK_ID = "oral_lesion"

_DISCLAIMER = (
    "Research prototype. Not a medical device and not a diagnosis. "
    "Any concerning finding must be confirmed by a clinician."
)


def _band(label: Optional[str]) -> RiskBand:
    """Map a service risk label onto the shared vocabulary.

    An unrecognised label becomes ``INDETERMINATE`` rather than defaulting to ``LOW``.
    If the service ever grows a band this module has not been taught, the honest failure
    is "we cannot say", not a reassuring answer produced by a lookup miss.
    """
    if not label:
        return RiskBand.INDETERMINATE
    normalised = label.strip().upper()
    for member in RiskBand:
        if member.value == normalised:
            return member
    if normalised.startswith("HIGH"):
        return RiskBand.HIGH
    if normalised.startswith("MODERATE") or normalised.startswith("MEDIUM"):
        return RiskBand.MODERATE
    if normalised.startswith("LOW"):
        return RiskBand.LOW
    return RiskBand.INDETERMINATE


class OralLesionTrack:
    """Oral-lesion screening from a photograph, via the existing MobileNetV3 + VQC path."""

    track_id = ORAL_TRACK_ID

    def __init__(self) -> None:
        # Nothing is loaded here. The service is a lazily-initialised singleton and
        # resolving it at import time would make a missing artifact a startup crash
        # instead of one unready card in the UI.
        pass

    # ----------------------------------------------------------------- introspection
    @staticmethod
    def _service():
        from backend.services.inference_service import get_inference_service

        return get_inference_service()

    def describe(self) -> TrackDescriptor:
        try:
            info: Dict[str, Any] = self._service().describe()
        except Exception as error:  # noqa: BLE001 - describe() must never raise
            info = {"ready": False, "reason": f"{type(error).__name__}: {error}"}

        ready = bool(info.get("ready"))
        quantum = info.get("quantum") or {}
        primary = info.get("classical_reference")

        return TrackDescriptor(
            track_id=self.track_id,
            display_name="Oral Lesion Screening",
            condition="Potentially malignant oral lesions",
            modality=Modality.IMAGE,
            input_spec=InputSpec(
                modality=Modality.IMAGE,
                content_types=["image/jpeg", "image/png"],
                description=(
                    "A well-lit, in-focus photograph of the oral cavity with the lesion "
                    "clearly visible and filling a reasonable part of the frame."
                ),
                max_bytes=10 * 1024 * 1024,
            ),
            validation=ValidationSummary(
                dataset="SMART-OM oral lesion benchmark",
                task="Lesion vs non-lesion screening",
                primary_metric="ROC-AUC",
                partition_scored="held-out validation split",
                # The honest statement of where this track stands. The classical baseline
                # outranks the VQC here and the operating point was chosen on validation,
                # so this is a development estimate and the platform says so rather than
                # letting a caller infer clinical validation from a number.
                frozen_test_evaluated=False,
                partition_reused_for_selection=True,
                notes=(
                    "The validated classical baseline headlines the verdict; the VQC ranks "
                    "below it on this dataset (DEC-034). The calibrated probability is the "
                    "quantum one because it is the only calibrated score available."
                ),
            ),
            model_version=str(info.get("model_version") or ""),
            ready=ready,
            unready_reason=None if ready else str(info.get("reason") or "unavailable"),
            primary_model=str(primary or "quantum_vqc_calibrated"),
            uses_quantum=True,
            quantum_role=(
                f"A {quantum.get('n_qubits', '?')}-qubit variational classifier over the "
                "reduced image descriptor, reported alongside the classical baseline "
                "rather than in place of it."
            ),
            disclaimer=_DISCLAIMER,
        )

    # ---------------------------------------------------------------------- inference
    def analyze(self, payload: Any, **options: Any) -> TrackAssessment:
        """Screen one image.

        Args:
            payload: raw image bytes, or an object the service's ``infer_from_image``
                already accepts.
        """
        from backend.services.inference_service import (
            ImageRejectedError,
            InferenceError,
            ModelNotReadyError,
        )

        if payload is None:
            raise TrackInputError("An image is required.")

        started = time.perf_counter()
        try:
            result = self._service().infer_from_image(payload)
        except ModelNotReadyError as error:
            raise TrackNotReadyError(str(error)) from error
        except ImageRejectedError as error:
            raise TrackInputError(str(error)) from error
        except InferenceError as error:
            raise TrackNotReadyError(str(error)) from error

        return self.to_assessment(result, elapsed_ms=(time.perf_counter() - started) * 1000.0)

    @classmethod
    def to_assessment(cls, result: Any, *, elapsed_ms: float = 0.0) -> TrackAssessment:
        """Translate an ``InferenceResult`` into the platform vocabulary.

        Split out from :meth:`analyze` so the mapping can be tested against a constructed
        result without a loaded model -- this is where a silent semantic error would live,
        so it should be reachable without 200 MB of artifacts.
        """
        # DEC-034: the headline band comes from ``primary_*`` when the service filled it
        # in, falling back to the quantum band otherwise. ``primary_probability`` is a
        # *ranking* score when the classical baseline leads, which is precisely why
        # ``primary_calibrated`` travels with it.
        headline_band = result.primary_risk_level or result.risk_level
        headline_prob = (
            result.primary_probability
            if result.primary_probability is not None
            else result.probability
        )
        calibrated = bool(
            result.primary_calibrated if result.primary_probability is not None else True
        )

        return TrackAssessment(
            track_id=ORAL_TRACK_ID,
            model_version=result.model_version,
            created_at=result.created_at,
            risk_band=_band(headline_band),
            probability=headline_prob,
            probability_is_calibrated=calibrated,
            threshold=result.primary_threshold,
            threshold_selected_on="validation split",
            primary_model=result.primary_model,
            classical_probability=result.classical_probability,
            # The calibrated quantum probability -- the value persisted as
            # ``final_probability`` and exported to FHIR. Kept distinct from the headline
            # so neither number has to pretend to be the other.
            quantum_probability=result.probability,
            detail={
                "inference_id": result.inference_id,
                "classification": result.classification,
                "quantum_risk_level": result.risk_level,
                "quantum_probability_calibrated": result.probability,
                "quantum_probability_uncalibrated": result.probability_uncalibrated,
                "expectation_value": result.expectation_value,
                "classical_model": result.classical_model,
                "details": result.details,
                "quantum": result.quantum.model_dump() if result.quantum else None,
                "calibration": (
                    result.calibration.model_dump() if result.calibration else None
                ),
                "quality": result.quality.model_dump() if result.quality else None,
                "quantum_visual": (
                    result.quantum_visual.model_dump() if result.quantum_visual else None
                ),
                "quantum_time_ms": result.quantum_time_ms,
                "preprocessing_time_ms": result.preprocessing_time_ms,
            },
            execution_time_ms=result.execution_time_ms or elapsed_ms,
            is_mock=result.is_mock,
            disclaimer=result.disclaimer or _DISCLAIMER,
        )
