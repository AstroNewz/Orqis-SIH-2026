"""Real inference orchestration: pipeline, quantum circuit, calibration, bands.

This is the module the FINAL EXECUTION RULE is about. It does not fabricate a
probability, and it does not return a hardcoded clinical result. Every call loads
the persisted artifacts for a model version and runs the actual stages:

    descriptor or image
        -> quality control (image path only)
        -> ROI + normalisation (image path only)
        -> feature extraction (image path only)
        -> clinical encoding
        -> fusion
        -> dimensionality reduction
        -> amplitude encoding
        -> variational circuit
        -> <Z_0> -> raw score
        -> fitted calibrator
        -> risk band

Artifacts are loaded once and cached
------------------------------------
Loading a pipeline and rebuilding a quantum circuit per request would dominate the
latency of an inference that is otherwise milliseconds. They are loaded lazily on
first use, behind a lock, and reused. Lazily rather than at import so that a
backend with no trained model still starts, serves ``/health``, and returns a clear
"model not ready" error instead of crashing on boot (PART 23, PART 29).

Two entry points, one pipeline
------------------------------
:meth:`InferenceService.infer_from_image` is the full path. :meth:`infer_from_descriptor`
takes an image descriptor the device computed itself, which is the flow PART 19
prefers -- the raw clinical photograph never leaves the phone. Both converge on the
same fusion, reduction, circuit and calibrator, so the two paths cannot drift.

Errors are typed and patient-safe
---------------------------------
Every failure raises an :class:`InferenceError` subclass whose message is safe to
show a patient. Technical detail goes to the log, never to the response (PART 23,
PART 34).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.ml.artifacts import (
    BASELINES_FILE,
    CALIBRATION_FILE,
    QUANTUM_FILE,
    ArtifactError,
    ArtifactStore,
)
from backend.ml.baselines import BaselineError, LogisticBaseline
from backend.ml.pipeline import ClassicalPipeline
from backend.ml.types import ImageQualityRejected, PreprocessingError
from backend.schemas.inference import (
    CalibrationSummary,
    FeatureSummary,
    InferenceResult,
    QualitySummary,
    QuantumSummary,
    QuantumVisualSummary,
)
from quantum_ml.backends import build_backend
from quantum_ml.calibration import ProbabilityCalibrator, RiskLevel, ScoreCalibrator
from quantum_ml.results import ExecutionMode
from quantum_ml.vqc_classifier import VariationalQuantumClassifier

logger = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """Base class. ``str(exc)`` is always safe to return to a client."""

    status_code = 500


class ModelNotReadyError(InferenceError):
    """No trained artifacts are available for the requested model version."""

    status_code = 503


class FeatureContractError(InferenceError):
    """The supplied feature vector does not match the model's input contract."""

    status_code = 422


class ImageRejectedError(InferenceError):
    """Quality control rejected the capture; the patient should retake it."""

    status_code = 422


class QuantumExecutionError(InferenceError):
    """The quantum stage failed even after backend fallback."""

    status_code = 503


class InferenceService:
    """Loads a model version's artifacts and runs real inferences with them."""

    def __init__(
        self,
        *,
        model_version: Optional[str] = None,
        extractor_name: str = "handcrafted_lab",
        config: Optional[Settings] = None,
    ):
        self._cfg = config or default_settings
        self._requested_version = model_version
        self._extractor_name = extractor_name
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None

        self._version: str = ""
        self._pipeline: Optional[ClassicalPipeline] = None
        self._model: Optional[VariationalQuantumClassifier] = None
        self._calibrator: Optional[ScoreCalibrator] = None
        self._bands: Optional[ProbabilityCalibrator] = None
        self._backend: Any = None
        self._calibration_payload: Dict[str, Any] = {}
        self._classical: Optional[LogisticBaseline] = None
        self._classical_name: str = ""
        self._classical_threshold: Optional[float] = None

    # ------------------------------------------------------------ artifacts
    def _load(self) -> None:
        """Load artifacts once. Idempotent, thread-safe, and never partially applied.

        Everything is built into locals and assigned to ``self`` only at the end, so
        a failure halfway through leaves the service cleanly unloaded rather than
        holding a pipeline with no model behind it.
        """
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            store = ArtifactStore.from_settings(self._cfg)
            try:
                version = store.resolve_version(self._requested_version)
                pipeline = store.load_pipeline(version)
                quantum_payload = store.read_component(version, QUANTUM_FILE)
                calibration_payload = store.read_component(version, CALIBRATION_FILE)
            except ArtifactError as exc:
                self._load_error = str(exc)
                raise ModelNotReadyError(
                    "The screening model is not available on this server yet."
                ) from exc

            mode = ExecutionMode(self._cfg.QUANTUM_EXECUTION_MODE)
            backend = build_backend(
                mode=mode,
                shots=self._cfg.QUANTUM_SHOTS,
                seed=self._cfg.RANDOM_SEED,
                noise_device=self._cfg.QUANTUM_NOISE_MODEL,
                ibm_backend_name=self._cfg.IBMQ_BACKEND_NAME,
                ibm_api_key=self._cfg.IBMQ_API_KEY,
                ibm_instance=self._cfg.IBMQ_INSTANCE,
                ibm_channel=self._cfg.IBMQ_CHANNEL,
                allow_fallback=self._cfg.QUANTUM_ALLOW_HARDWARE_FALLBACK,
            )
            if backend.resolution.fell_back:
                # Logged, not raised: PART 12 requires the client never to break
                # because hardware is unavailable.
                logger.warning(
                    "Quantum backend fell back from %s to %s: %s",
                    mode.value,
                    backend.effective_mode.value,
                    backend.resolution.fallback_reason,
                )

            model = VariationalQuantumClassifier.from_dict(quantum_payload, backend=backend)
            if not model.trained:
                raise ModelNotReadyError(
                    "The screening model on this server has not finished training."
                )
            calibrator = ScoreCalibrator.from_dict(calibration_payload)
            bands = ProbabilityCalibrator.from_calibration_payload(
                calibration_payload, config=self._cfg
            )
            classical, classical_name = _load_classical_reference(store, version)
            classical_threshold = (
                _load_classical_operating_point(store, version, classical_name)
                if classical is not None
                else None
            )
            scored_in = calibration_payload.get("scored_in_mode")
            if scored_in and scored_in != backend.effective_mode.value:
                # Not fatal, but the calibrated numbers are then fitted to a
                # different score distribution than the one being served.
                logger.warning(
                    "Calibrator for %s was fitted on %s scores but this server runs "
                    "%s. Re-run backend.training.calibrate --mode %s.",
                    version,
                    scored_in,
                    backend.effective_mode.value,
                    backend.effective_mode.value,
                )

            self._version = version
            self._pipeline = pipeline
            self._model = model
            self._calibrator = calibrator
            self._bands = bands
            self._backend = backend
            self._calibration_payload = calibration_payload
            self._classical = classical
            self._classical_name = classical_name
            self._classical_threshold = classical_threshold
            self._load_error = None
            self._loaded = True
            logger.info(
                "Inference ready: %s | %d qubits, %d layers | %s on %s | bands %.4f/%.4f (%s)",
                version,
                model.num_qubits,
                model.num_layers,
                backend.effective_mode.value,
                backend.backend_name,
                bands.threshold,
                bands.high_risk_threshold,
                bands.bands_source,
            )
            if classical is not None and classical_threshold is not None:
                logger.info(
                    "Headline verdict served by classical baseline %s at "
                    "validation-selected threshold %.4f; quantum VQC calibrated score "
                    "retained as a secondary experimental readout (DEC-034).",
                    classical_name,
                    classical_threshold,
                )

    @property
    def ready(self) -> bool:
        """Whether artifacts can be loaded, without raising."""
        try:
            self._load()
            return True
        except InferenceError:
            return False

    @property
    def model_version(self) -> str:
        """The resolved model version. Empty until artifacts have loaded."""
        return self._version

    @property
    def execution_mode(self) -> Optional[str]:
        """The mode inference *actually* runs in, or ``None`` before loading.

        This is the effective mode, not the configured one: if hardware was requested
        and was unreachable, this reports the simulator that took over.
        """
        return self._backend.effective_mode.value if self._backend is not None else None

    def describe(self) -> Dict[str, Any]:
        """Provenance for ``/model/info``. Safe when no model is loaded."""
        try:
            self._load()
        except InferenceError as exc:
            return {
                "ready": False,
                "reason": str(exc),
                "detail": self._load_error,
                "configured_execution_mode": self._cfg.QUANTUM_EXECUTION_MODE,
            }
        assert self._pipeline and self._model and self._calibrator and self._bands
        return {
            "ready": True,
            "model_version": self._version,
            "pipeline": self._pipeline.describe(),
            "quantum": {
                "n_qubits": self._model.num_qubits,
                "n_ansatz_layers": self._model.num_layers,
                "n_parameters": self._model.num_params,
                "circuit_depth": self._model.ansatz_depth,
                "amplitude_dimension": 2**self._model.num_qubits,
                "execution_mode": self._backend.effective_mode.value,
                "backend_name": self._backend.backend_name,
                "shots": self._backend.effective_shots,
                "is_exact": self._backend.is_exact,
                "fell_back": self._backend.resolution.fell_back,
                "fallback_reason": self._backend.resolution.fallback_reason,
            },
            "calibration": {
                "method": self._calibrator.method.value,
                "is_calibrated": self._calibrator.is_calibrated,
                "fitted_on": self._calibrator.fitted_on,
                "scored_in_mode": self._calibration_payload.get("scored_in_mode"),
                **self._bands.describe(),
            },
            "expected_descriptor_dimension": self._expected_descriptor_dimension(),
            "clinical_features_expected": bool(self._pipeline.fusion.uses_clinical),
            "classical_reference": self._classical_name or None,
        }

    def _expected_descriptor_dimension(self) -> int:
        """Image-descriptor length a client must supply, pre-fusion."""
        assert self._pipeline is not None
        return int(self._pipeline.fusion.n_image_features)

    # ------------------------------------------------------------- inference
    def infer_from_descriptor(
        self,
        descriptor: Sequence[float],
        *,
        clinical_row: Optional[Mapping[str, Any]] = None,
        quality: Optional[QualitySummary] = None,
    ) -> InferenceResult:
        """Infer from an image descriptor the client computed on-device.

        The preferred flow: the photograph stays on the phone and only a compact
        numerical vector is transmitted (PART 19).

        Raises:
            ModelNotReadyError: no usable artifacts.
            FeatureContractError: wrong descriptor length, or non-finite values.
            QuantumExecutionError: the circuit could not be executed.
        """
        self._load()
        assert self._pipeline is not None

        started = time.perf_counter()
        vector = np.asarray(descriptor, dtype=np.float64).ravel()
        expected = self._expected_descriptor_dimension()
        if vector.size != expected:
            # A precise, actionable 422 rather than letting the amplitude encoder
            # raise deep in the quantum layer with an internal message.
            raise FeatureContractError(
                f"This model expects an image descriptor of {expected} values "
                f"({self._extractor_name}); {vector.size} were supplied."
            )
        if not np.all(np.isfinite(vector)):
            raise FeatureContractError(
                "The supplied feature vector contains missing or non-finite values."
            )

        try:
            reduced = self._pipeline.transform_matrix(
                image_features=vector.reshape(1, -1),
                clinical_rows=[clinical_row] if self._pipeline.fusion.uses_clinical else None,
            )[0]
        except PreprocessingError as exc:
            logger.warning("Feature transform failed: %s", exc)
            raise FeatureContractError(
                "The supplied features could not be processed by this model."
            ) from exc
        preprocessing_ms = (time.perf_counter() - started) * 1000.0

        return self._infer_from_reduced(
            reduced,
            clinical_supplied=clinical_row is not None,
            quality=quality,
            preprocessing_ms=preprocessing_ms,
        )

    def infer_from_image(
        self,
        image,
        *,
        clinical_row: Optional[Mapping[str, Any]] = None,
        file_size_bytes: Optional[int] = None,
        strict_quality: bool = True,
    ) -> InferenceResult:
        """Infer from an open PIL image, running quality control first.

        Raises:
            ImageRejectedError: quality control rejected the capture. The message is
                the report's patient guidance, e.g. asking for a sharper photograph.
        """
        self._load()
        assert self._pipeline is not None

        started = time.perf_counter()
        try:
            outcome = self._pipeline.run(
                image,
                clinical_row=clinical_row,
                file_size_bytes=file_size_bytes,
                strict_quality=strict_quality,
                config=self._cfg,
            )
        except ImageQualityRejected as exc:
            # Rejection is a normal outcome, not a server error: PART 5 requires
            # poor captures to be sent back for reacquisition rather than inferred on.
            raise ImageRejectedError(exc.report.patient_message) from exc
        except PreprocessingError as exc:
            logger.warning("Preprocessing failed: %s", exc)
            raise InferenceError(
                "The image could not be processed. Please try capturing it again."
            ) from exc
        preprocessing_ms = (time.perf_counter() - started) * 1000.0

        if outcome.fused is None:
            raise ImageRejectedError(
                outcome.quality.patient_message
                if outcome.quality
                else "The image could not be analysed. Please capture it again."
            )

        # Additive, experimental E2 secondary signal (DEC-035). Best-effort and fully
        # isolated: any failure returns None and the classical primary path below is
        # served unchanged. Uses this request's own ROI, so no test data is involved.
        quantum_visual = self._quantum_visual_summary(image, outcome.roi)

        return self._infer_from_reduced(
            np.asarray(outcome.fused.values, dtype=np.float64),
            clinical_supplied=clinical_row is not None,
            quality=_quality_summary(outcome.quality),
            preprocessing_ms=preprocessing_ms,
            quantum_visual=quantum_visual,
        )

    def _quantum_visual_summary(self, image, roi) -> Optional[QuantumVisualSummary]:
        """Best-effort E2 quantum-visual secondary signal for one image. Never raises.

        Additive and experimental (DEC-035): on ANY failure -- torch or the E2 artifact
        absent, a dimension mismatch, a circuit error -- this returns ``None`` and the
        caller serves the classical primary result unchanged. Nothing here touches
        ``probability`` / ``final_probability`` / the risk band / FHIR values.

        The MobileNet ROI embedding is recomputed from the *same* normalised crop the
        classical pipeline used for this image (identical ROI box and target size), so the
        runtime embedding matches the representation E2 was fitted on. Imports are local so
        even a broken E2 dependency cannot affect module load or the classical path.
        """
        try:
            from backend.ml.features_image import extract_mobilenet
            from backend.ml.preprocessing import normalise_crop
            from backend.ml.roi import crop_to_roi
            from backend.services.quantum_visual_service import (
                get_quantum_visual_service,
            )

            service = get_quantum_visual_service(self._cfg)
            if not service.ready:
                return None
            assert self._pipeline is not None
            normalised = normalise_crop(
                crop_to_roi(image, roi), target_size=self._pipeline.target_size
            )
            embedding = extract_mobilenet(normalised)
            return service.analyze_embedding(embedding)
        except Exception as exc:  # noqa: BLE001 - E2 is optional; never break inference
            logger.info("E2 quantum-visual secondary signal skipped: %s", exc)
            return None

    def _infer_from_reduced(
        self,
        reduced: np.ndarray,
        *,
        clinical_supplied: bool,
        quality: Optional[QualitySummary],
        preprocessing_ms: float,
        quantum_visual: Optional[QuantumVisualSummary] = None,
    ) -> InferenceResult:
        """Shared tail: circuit, calibration, band, assembly.

        ``quantum_visual`` is the optional, additive E2 secondary signal supplied only by
        the image path; it is ``None`` for the descriptor path (which has no image) and on
        any E2 failure. It never affects any primary value assembled below.
        """
        assert self._pipeline and self._model and self._calibrator and self._bands

        quantum_started = time.perf_counter()
        try:
            raw_score, metadata = self._model.predict_probability(reduced)
        except Exception as exc:  # noqa: BLE001 - any circuit failure becomes typed
            logger.exception("Quantum execution failed")
            raise QuantumExecutionError(
                "The analysis engine is temporarily unavailable. Please try again."
            ) from exc
        quantum_ms = (time.perf_counter() - quantum_started) * 1000.0

        calibrated = float(self._calibrator.transform_one(raw_score))

        classical_probability: Optional[float] = None
        if self._classical is not None:
            try:
                classical_probability = self._classical.predict_one(reduced)
            except BaselineError as exc:
                # A dimension mismatch here means the baseline was fitted on a
                # different feature space than the one being served. Report nothing
                # rather than a number from the wrong model.
                logger.warning("Classical reference unavailable: %s", exc)

        # The coherent, calibrated headline number stays the quantum result: it is the
        # only calibrated probability available, it is safe to persist and to export as
        # a FHIR probability, and the risk band it produces is self-consistent with it.
        risk_level, classification, details = self._bands.categorize_risk(
            probability=calibrated, calibrated=self._calibrator.is_calibrated
        )

        # The *displayed* verdict, however, headlines the strongest validated model's
        # band. On this dataset that is the classical baseline -- the quantum VQC ranks
        # last of four (ISS-008 / DEC-034) -- so when the baseline and its
        # validation-selected operating point are both available, the client headlines
        # its two-band ranking verdict. Only the band is promoted: the classical score
        # is uncalibrated and must not be shown as a percentage, so the calibrated
        # number above is left untouched. With no classical baseline the headline falls
        # back to the quantum band, exactly as before.
        if classical_probability is not None and self._classical_threshold is not None:
            primary_model = self._classical_name or "logistic_regression"
            primary_probability: Optional[float] = classical_probability
            primary_threshold: Optional[float] = self._classical_threshold
            primary_calibrated = False
            primary_risk_level = (
                RiskLevel.MODERATE.value
                if classical_probability >= self._classical_threshold
                else RiskLevel.LOW.value
            )
        else:
            primary_model = "quantum_vqc_calibrated"
            primary_probability = calibrated
            primary_threshold = self._bands.threshold
            primary_calibrated = self._calibrator.is_calibrated
            primary_risk_level = risk_level

        pipeline = self._pipeline
        return InferenceResult(
            inference_id=str(uuid.uuid4()),
            model_version=self._version,
            created_at=datetime.now(timezone.utc),
            probability=calibrated,
            probability_uncalibrated=raw_score,
            raw_score=raw_score,
            expectation_value=float(metadata.get("expectation_value", 1.0 - 2.0 * raw_score)),
            risk_level=risk_level,
            classification=classification,
            details=details,
            classical_probability=classical_probability,
            classical_model=self._classical_name or None,
            primary_model=primary_model,
            primary_risk_level=primary_risk_level,
            primary_probability=primary_probability,
            primary_threshold=primary_threshold,
            primary_calibrated=primary_calibrated,
            calibration=CalibrationSummary(
                method=self._calibrator.method.value,
                is_calibrated=self._calibrator.is_calibrated,
                fitted_on=self._calibrator.fitted_on,
                screening_threshold=self._bands.threshold,
                high_risk_threshold=self._bands.high_risk_threshold,
                bands_source=self._bands.bands_source,
            ),
            quantum=QuantumSummary(
                n_qubits=self._model.num_qubits,
                circuit_depth=self._model.ansatz_depth,
                n_ansatz_layers=self._model.num_layers,
                n_parameters=self._model.num_params,
                amplitude_dimension=2**self._model.num_qubits,
                execution_mode=self._backend.effective_mode.value,
                backend_name=self._backend.backend_name,
                shots=self._backend.effective_shots,
                requested_mode=self._backend.resolution.requested_mode.value,
                fell_back=self._backend.resolution.fell_back,
                fallback_reason=self._backend.resolution.fallback_reason,
                is_exact=self._backend.is_exact,
            ),
            features=FeatureSummary(
                feature_mode=pipeline.feature_mode,
                extractor=self._extractor_name,
                n_image_features=int(pipeline.fusion.n_image_features),
                n_clinical_features=int(pipeline.fusion.n_clinical_features),
                n_reduced_features=int(reduced.size),
                reduction_method=pipeline.reducer.method,
                preprocessing_version=pipeline.preprocessing_version,
                pipeline_version=pipeline.version,
                clinical_features_supplied=clinical_supplied,
            ),
            quality=quality,
            quantum_visual=quantum_visual,
            execution_time_ms=preprocessing_ms + quantum_ms,
            quantum_time_ms=quantum_ms,
            preprocessing_time_ms=preprocessing_ms,
            is_mock=False,
        )


def _load_classical_reference(
    store: ArtifactStore, version: str
) -> Tuple[Optional[LogisticBaseline], str]:
    """Load the logistic-regression baseline for side-by-side reporting.

    Optional by design: a model version can be served before baselines exist, and a
    missing comparison must not block a screening. Only logistic regression is
    restorable without scikit-learn -- the tree ensembles are reproducible-by-refit,
    which belongs in evaluation, not in a request path.
    """
    payload = store.read_component(version, BASELINES_FILE, required=False)
    record = ((payload or {}).get("baselines") or {}).get("logistic_regression")
    if not record or not record.get("weights"):
        return None, ""
    try:
        return LogisticBaseline.from_dict(record), "logistic_regression"
    except (BaselineError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Could not restore the classical reference for %s: %s", version, exc)
        return None, ""


def _load_classical_operating_point(
    store: ArtifactStore, version: str, classical_name: str
) -> Optional[float]:
    """Validation-selected screening threshold for the classical baseline, or ``None``.

    Read from this model version's ``evaluation.json`` ``operating_points`` -- but
    only when that entry is explicitly ``chosen_on == "validation"``. The held-out
    test partition must never influence a served threshold, so a test-derived
    operating point, a missing provenance marker, or a missing/unreadable artifact all
    yield ``None`` and the caller falls back to the quantum band. Read-only: the
    frozen evaluation artifact is never modified.
    """
    if not classical_name:
        return None
    try:
        payload = store.read_component(version, "evaluation.json", required=False)
    except (ArtifactError, OSError, ValueError, TypeError):
        return None
    point = ((payload or {}).get("operating_points") or {}).get(classical_name)
    if not isinstance(point, dict) or point.get("chosen_on") != "validation":
        return None
    threshold = point.get("threshold")
    try:
        value = float(threshold)
    except (TypeError, ValueError):
        return None
    return value if 0.0 <= value <= 1.0 else None


def _quality_summary(report) -> Optional[QualitySummary]:
    if report is None:
        return None
    return QualitySummary(
        verdict=report.verdict.value,
        passed=report.acceptable,
        reasons=list(report.messages),
        metrics={
            "mean_luminance": round(report.mean_luminance, 4),
            "laplacian_variance": round(report.laplacian_variance, 4),
            "clipped_fraction": round(report.clipped_fraction, 6),
            "bytes_per_pixel": round(report.bytes_per_pixel, 6),
            "roi_fraction": round(report.roi_fraction, 6),
        },
    )


_default_service: Optional[InferenceService] = None
_default_lock = threading.Lock()


def get_inference_service(config: Optional[Settings] = None) -> InferenceService:
    """Process-wide service, so artifacts load once rather than per request."""
    global _default_service
    if _default_service is None:
        with _default_lock:
            if _default_service is None:
                _default_service = InferenceService(config=config)
    return _default_service


def reset_inference_service() -> None:
    """Drop the cached service. For tests that swap artifacts or settings."""
    global _default_service
    with _default_lock:
        _default_service = None
