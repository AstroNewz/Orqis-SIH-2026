"""The ECG track: 12-lead screening on the PTB-XL arena.

**This track is deliberately shipped before its model.** Everything a caller needs -- the
input contract, the lead order, the provenance block, the failure vocabulary -- is fixed
here now, and the track reports ``ready=False`` naming exactly which artifact is missing.
That ordering is not an accident of scheduling. The alternative, waiting for a number
before deciding what the platform is allowed to say about it, is how a development ROC-AUC
turns into a clinical claim: the honest framing has to exist before there is a result that
would benefit from a dishonest one.

What is genuinely established (E4 classical baselines, DEC-044 then DEC-046) is the
**ceiling**: 0.946293 ROC-AUC for NORM-vs-abnormal on ``strat_fold`` 9, from a logistic
fusion of a gradient-boosted model over the 97-dimensional ``ecg-v1`` vector (0.940234) and
a 1D-CNN over the raw 12-lead waveform (0.940476), with patient-level partitioning verified
and ``strat_fold`` 10 never read. Any model served here must beat or match that, and the
quantum head is not wired in until it does so under the pre-registered standard.

The two single arms **tie** -- paired delta +0.000242, CI [-0.006101, +0.006374], spanning
zero -- while their fusion beats both with intervals that exclude zero. They therefore see
different things despite scoring the same, which is why the bar is the fusion and not the
better of the two, and why DEC-046 pre-registered a *classical*-fusion control for any
future quantum arm: beating a single classical model is no longer evidence of anything.

Serving is a two-step contract, and the second step is the one that is outstanding:

1. :class:`~backend.ml.ecg_transform.EcgFeatureTransform` -- fitted on TRAIN only, already
   persisted under ``backend/artifacts/models/ecg_transform``.
2. A classifier bundle at :data:`CLASSIFIER_PATH`, which the evaluation harness does not
   currently persist because selection is still open. Dropping one there makes this track
   live with no change to this file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

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

__all__ = ["EcgTrack", "ECG_TRACK_ID", "CLASSIFIER_PATH", "STANDARD_LEAD_ORDER"]

ECG_TRACK_ID = "ecg_12lead"

CLASSIFIER_PATH = Path("backend/artifacts/models/ecg_classifier/current.joblib")
"""The servable bundle. A joblib dict with keys ``estimator``, ``feature_set_version``,
``transform`` (or ``None`` for a model over raw features), ``threshold``,
``threshold_selected_on``, ``model_version``, ``is_calibrated`` and ``metrics``.

Kept as a single file with its metadata inside it so a model can never be served with
another model's threshold -- the pairing is the part that would fail silently."""

SAMPLING_FREQUENCY_HZ = 100.0
RECORD_SECONDS = 10
EXPECTED_SHAPE = (12, 1000)

STANDARD_LEAD_ORDER: List[str] = [
    "I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6",
]
"""PTB-XL's channel order. Published rather than inferred: a caller that sends the
precordial leads first gets a confident and completely wrong answer, and nothing
downstream can detect it."""

_DISCLAIMER = (
    "Research prototype. Not a medical device, not a diagnosis, and not an arrhythmia "
    "monitor. Performance is reported on a development partition only."
)

# The measured classical ceiling this track must clear: the fusion arm (DEC-046), not the
# tabular arm alone (DEC-044, 0.940234). A served model that beat the tabular number but not
# this one would be beating a baseline the phase has already exceeded.
CLASSICAL_CEILING_ROC_AUC = 0.946293
CLASSICAL_CEILING_CI = [0.9364, 0.9550]
CLASSICAL_CEILING_ARM = "fusion@cnn+gbm"


class EcgTrack:
    """NORM-vs-abnormal screening from a 10-second 12-lead ECG."""

    track_id = ECG_TRACK_ID

    def __init__(self, classifier_path: Optional[Path] = None) -> None:
        self._path = Path(classifier_path) if classifier_path else CLASSIFIER_PATH
        self._bundle: Optional[Dict[str, Any]] = None
        self._load_error: Optional[str] = None

    # ----------------------------------------------------------------------- loading
    def _load(self) -> Dict[str, Any]:
        if self._bundle is not None:
            return self._bundle
        if not self._path.exists():
            raise TrackNotReadyError(
                f"No ECG classifier at {self._path}. The E4 arena has a measured classical "
                f"ceiling of {CLASSICAL_CEILING_ROC_AUC:.6f} ROC-AUC "
                f"({CLASSICAL_CEILING_ARM}, fold 9, DEC-046), but no model has been "
                "persisted for serving yet: every number measured so far was selected on "
                "a development partition, and freezing a servable model on one would ship "
                "an operating point chosen against the data it is reported on."
            )
        try:
            import joblib

            bundle = joblib.load(self._path)
        except Exception as error:  # noqa: BLE001
            self._load_error = f"{type(error).__name__}: {error}"
            raise TrackNotReadyError(f"Could not load {self._path}: {error}") from error

        missing = {"estimator", "feature_set_version", "threshold", "model_version"} - set(bundle)
        if missing:
            raise TrackNotReadyError(
                f"{self._path} is missing {sorted(missing)}. A bundle without its threshold "
                "and feature-set version is not servable: the estimator alone does not say "
                "what it expects or where its operating point came from."
            )
        self._bundle = bundle
        return bundle

    # ------------------------------------------------------------------ introspection
    def describe(self) -> TrackDescriptor:
        try:
            bundle = self._load()
            ready, reason = True, None
        except TrackNotReadyError as error:
            bundle, ready, reason = {}, False, str(error)

        return TrackDescriptor(
            track_id=self.track_id,
            display_name="12-Lead ECG Screening",
            condition="Abnormal electrocardiogram (any of MI, ST/T change, conduction disturbance, hypertrophy)",
            modality=Modality.SIGNAL,
            input_spec=InputSpec(
                modality=Modality.SIGNAL,
                content_types=["application/json", "application/octet-stream"],
                description=(
                    "A 10-second 12-lead ECG sampled at 100 Hz, in millivolts, as a "
                    "12 x 1000 array in the standard lead order."
                ),
                shape=list(EXPECTED_SHAPE),
                units="mV",
                sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
                channel_names=list(STANDARD_LEAD_ORDER),
            ),
            validation=ValidationSummary(
                dataset="PTB-XL v1.0.3 (PhysioNet, CC BY 4.0) -- 21,799 records / 18,869 patients",
                task="NORM vs abnormal",
                primary_metric="ROC-AUC",
                primary_metric_value=bundle.get("metrics", {}).get("roc_auc")
                if bundle
                else CLASSICAL_CEILING_ROC_AUC,
                confidence_interval=list(CLASSICAL_CEILING_CI),
                partition_scored="strat_fold 9 (validation)",
                n_records=2146,
                n_patients=1917,
                # strat_fold 10 has never been read on this machine. Until it is, every
                # number here is a development estimate and the platform must not imply
                # otherwise -- which is exactly what this flag prevents.
                frozen_test_evaluated=False,
                partition_reused_for_selection=True,
                notes=(
                    "Folds 1-8 fit, fold 9 scored, fold 10 frozen and never read. Splits are "
                    "patient-disjoint by construction and verified. Intervals are "
                    "bootstrapped over patients, not records. The reported value is the "
                    f"measured classical ceiling ({CLASSICAL_CEILING_ROC_AUC:.6f}, "
                    f"{CLASSICAL_CEILING_ARM}); no quantum component is served on this "
                    "track."
                ),
            ),
            model_version=str(bundle.get("model_version") or ""),
            ready=ready,
            unready_reason=reason,
            primary_model=str(bundle.get("primary_model") or "gradient_boosted_trees"),
            # Stated as a fact about what is deployed, not about what is planned. The
            # quantum head is gated behind beating the classical ceiling under the
            # pre-registered standard, and it has not been evaluated on this arena yet.
            uses_quantum=False,
            quantum_role=None,
            disclaimer=_DISCLAIMER,
        )

    # ---------------------------------------------------------------------- inference
    @staticmethod
    def _as_array(payload: Any) -> np.ndarray:
        """Coerce and validate one record against the published :class:`InputSpec`."""
        if payload is None:
            raise TrackInputError("An ECG record is required.")
        if isinstance(payload, (bytes, bytearray)):
            try:
                payload = json.loads(bytes(payload).decode("utf-8"))
            except Exception as error:  # noqa: BLE001
                raise TrackInputError(f"Could not parse the ECG payload as JSON: {error}") from error
        if isinstance(payload, dict):
            for key in ("signal", "signals", "data", "leads"):
                if key in payload:
                    payload = payload[key]
                    break
            else:
                raise TrackInputError(
                    "Expected an object with a 'signal' key holding a 12 x 1000 array."
                )
        try:
            array = np.asarray(payload, dtype=np.float32)
        except Exception as error:  # noqa: BLE001
            raise TrackInputError(f"ECG payload is not numeric: {error}") from error

        if array.shape == (EXPECTED_SHAPE[1], EXPECTED_SHAPE[0]):
            # Time-major is the other common convention; accept it, but transpose
            # explicitly rather than letting a (1000, 12) array reach a model that will
            # happily consume it as twelve 1000-sample leads' worth of nonsense.
            array = array.T
        if array.shape != EXPECTED_SHAPE:
            raise TrackInputError(
                f"Expected a {EXPECTED_SHAPE[0]} x {EXPECTED_SHAPE[1]} array "
                f"({RECORD_SECONDS} s at {SAMPLING_FREQUENCY_HZ:g} Hz, leads "
                f"{', '.join(STANDARD_LEAD_ORDER)}); got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise TrackInputError("The ECG contains NaN or infinite samples.")
        return array

    def analyze(self, payload: Any, **options: Any) -> TrackAssessment:
        # Validate before loading, so a malformed request gets a 4xx naming the real
        # problem rather than a 503 about a missing artifact.
        array = self._as_array(payload)
        bundle = self._load()

        started = time.perf_counter()
        from backend.dataset.wfdb_reader import EcgSignal
        from backend.ml.features_ecg import extract_features

        # ``extract_features`` is record-local by construction -- it fits nothing, which is
        # what makes the same code path legitimate for a live request and for a TRAIN fold.
        # It expects time-major samples, so the lead-major contract is transposed here and
        # nowhere else.
        signal = EcgSignal(
            samples=np.asarray(array.T, dtype=np.float64),
            sampling_frequency=SAMPLING_FREQUENCY_HZ,
            lead_names=tuple(STANDARD_LEAD_ORDER),
            units=("mV",) * EXPECTED_SHAPE[0],
            record_name="request",
        )
        extracted = extract_features(signal)
        if not extracted.usable:
            # A well-formed but unreadable record is not a bad request, so this is not an
            # error: it is a verdict of "cannot say". Banding it anyway would produce LOW
            # RISK for a lead-off or motion-corrupted strip, which is the single most
            # dangerous thing a screening tool can do.
            return TrackAssessment(
                track_id=self.track_id,
                model_version=str(bundle["model_version"]),
                risk_band=RiskBand.INDETERMINATE,
                probability=None,
                probability_is_calibrated=False,
                primary_model=str(bundle.get("primary_model") or "gradient_boosted_trees"),
                detail={
                    "reason": "signal_quality",
                    "template_beats": extracted.template_beats,
                    "n_beats": extracted.n_beats,
                    "n_missing_features": extracted.n_missing,
                    "n_features": extracted.dimension,
                    "quality_flags": list(extracted.quality_flags),
                },
                execution_time_ms=(time.perf_counter() - started) * 1000.0,
                disclaimer=_DISCLAIMER,
            )
        features = np.asarray(extracted.vector, dtype=np.float64).reshape(1, -1)

        expected_version = bundle["feature_set_version"]
        from backend.ml.features_ecg import FEATURE_SET_VERSION

        if expected_version != FEATURE_SET_VERSION:
            raise TrackNotReadyError(
                f"{self._path} was fitted on feature set {expected_version!r} but this build "
                f"extracts {FEATURE_SET_VERSION!r}. The vectors have different meanings "
                "position by position, so scoring across them would be silently wrong."
            )

        transform = bundle.get("transform")
        represented = transform.transform(features) if transform is not None else features

        estimator = bundle["estimator"]
        if hasattr(estimator, "predict_proba"):
            probability = float(estimator.predict_proba(represented)[0, 1])
        else:
            probability = float(estimator.decision_function(represented)[0])

        threshold = float(bundle["threshold"])
        calibrated = bool(bundle.get("is_calibrated", False))
        band = RiskBand.MODERATE if probability >= threshold else RiskBand.LOW

        return TrackAssessment(
            track_id=self.track_id,
            model_version=str(bundle["model_version"]),
            risk_band=band,
            probability=probability if calibrated or 0.0 <= probability <= 1.0 else None,
            probability_is_calibrated=calibrated,
            threshold=threshold,
            threshold_selected_on=str(bundle.get("threshold_selected_on") or "TRAIN folds 1-8"),
            primary_model=str(bundle.get("primary_model") or "gradient_boosted_trees"),
            classical_probability=probability,
            quantum_probability=None,
            detail={
                "feature_set_version": bundle["feature_set_version"],
                "n_features": int(features.shape[1]),
                "representation_dimension": int(represented.shape[1]),
                "lead_order": list(STANDARD_LEAD_ORDER),
                "sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
            },
            execution_time_ms=(time.perf_counter() - started) * 1000.0,
            is_mock=False,
            disclaimer=_DISCLAIMER,
        )
