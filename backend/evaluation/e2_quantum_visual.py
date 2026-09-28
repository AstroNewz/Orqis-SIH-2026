"""E2: the fixed 8-qubit quantum visual demonstrator, measured against its controls.

This is the evaluation driver for architecture Candidate B (DEC-035). It answers one
question honestly and refuses to answer any other: **on the frozen MobileNet ROI
descriptor, does a fixed 0-trainable-parameter 8-qubit angle-encoded reservoir plus a
small logistic head produce a validation signal that exceeds its own patient-blocked
null -- and does it do so any better than two matched classical controls at equal
input and equal head?**

It is *not* an advantage experiment. E1.1 measured no classical bottleneck on the
primary condition (DEC-033/DEC-034), so no quantum-advantage claim is authorised. E2 is
the SIH fallback *demonstrator*: a real, executable quantum visual computation whose
numbers are reported beside honest controls, never in place of the classical primary
clinical path.

Three representations, identical rows / identical head / identical evaluation
-----------------------------------------------------------------------------
=========  ============================================================  ==============
Name       Feature vector                                                Dimension
=========  ============================================================  ==============
quantum    fixed 8-qubit angle-encoded reservoir, 16 local Z observables 16
rff        random Fourier features of the same PCA-8 input (DEC-033)      16
pca        the raw PCA-8 scores handed straight to the head              8
=========  ============================================================  ==============

The quantum result is *never* reported without ``rff`` and ``pca`` beside it: an RBF
random-feature map at equal output dimension is the honest "what would any fixed
nonlinear feature map get?" control, and the raw PCA-8 is the "does the map add anything
over the numbers it was built from?" control. All three are fitted with the *same*
``fit_readout`` (L2 logistic, patient-grouped CV inside train, class-weighted), scored
once on validation, and gated against the *same* patient-blocked column-permutation null.

Discipline, asserted in code and in the payload
-----------------------------------------------
* **The test partition is never read.** No code path in this module can name it: the
  joins iterate ``("train", "validation")`` only, ``partitions_read`` says so, and
  ``test_partition_used`` is ``False`` in every payload. PCA, the angle scaler, the RFF
  bandwidth, the head coefficients and the null all see train rows only; validation is
  scored exactly once per representation.
* **Nothing is searched.** The 16 observables are frozen by structure in
  :mod:`quantum_ml.visual_circuit`; the head's only freedom is its L2 ``C``, chosen by
  grouped CV inside train. There is no observable selection and no feature selection.
* **The fast readout is validated against Aer.** ``validate_against_aer`` asserts the
  ``|psi|^2 @ diag^T`` contraction equals Aer's ``save_expectation_value`` to 1e-10 on a
  train sample before any feature is reported.
* **KEEP/KILL is measured, not asserted.** K1 latency, K2 non-degeneracy and K3
  null-exceedance are computed and reported; a KILL is stated plainly and the quantum
  stage is preserved as telemetry rather than dressed up as a success.

Usage::

    python -m backend.evaluation.e2_quantum_visual --out reports_e2_quantum_visual.json
    python -m backend.evaluation.e2_quantum_visual --permutations 200 --n-jobs -1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings, settings as default_settings
# Reused, never reimplemented: the fold rule lives inside ``fit_readout`` and the metric,
# null-summary and rank helpers are the exact ones E0.1 used, so an E2 number sits on the
# same scale as the E0/E1 numbers it will be compared against.
from backend.evaluation.e0_controls import _patient_blocked_column_shuffle
from backend.evaluation.e0_readout import (
    FIXED_THRESHOLD,
    VARIANCE_FLOOR,
    _metrics,
    _null_summary,
    _rank_in_null,
    fit_readout,
    patient_blocked_permutations,
)
from backend.ml.artifacts import ArtifactStore
from backend.ml.quantum_visual import (
    PCA_COMPONENTS,
    QUANTUM_VISUAL_VERSION,
    LogisticHead,
    QuantumVisualPreprocessor,
    RandomFourierControl,
)
from backend.training.data import load_partitions
from backend.training.pixel_data import (
    CONDITION_DESCRIPTIONS,
    CONDITION_ORACLE_ALL,
    CONDITION_ORACLE_LESION,
    condition_rows,
    load_pixel_partitions,
)
from backend.training.prepare_e1_features import e1_cache_path, load_e1_cache
from quantum_ml.visual_circuit import (
    E2_N_FEATURES,
    E2_SEED,
    VISUAL_CIRCUIT_VERSION,
    QuantumVisualFeatureMap,
    validate_against_aer,
)

logger = logging.getLogger(__name__)

E2_VERSION = "v1-e2-quantum-visual-driver-1"

#: The artifact "model version" directory the fitted E2 objects persist under. It is not
#: a clinical model version -- it is isolated so the E2 demonstrator can never be picked
#: up as ``current`` and headline a screening.
E2_ARTIFACT_VERSION = "e2_quantum_visual"
E2_ARTIFACT_FILE = "e2_quantum_visual.json"

#: Primary first, secondary second -- the same two oracle conditions E1.1 reports, in the
#: same order, and there is deliberately no way to name a predicted condition or the test
#: partition here.
PRIMARY = CONDITION_ORACLE_LESION
CONDITIONS: Tuple[str, ...] = (PRIMARY, CONDITION_ORACLE_ALL)

#: DEC-024 geometry-leak bar: on ``A_all`` ROI area alone scores validation ROC-AUC 0.72,
#: so every number on the secondary condition is read against 0.72, not 0.5.
GEOMETRY_LEAK_ROC_AUC = 0.72

#: Gating-null draws. 200 gives a p95 that is not itself resampling noise and a finest
#: empirical p of 1/201.
DEFAULT_PERMUTATIONS = 200

#: Above this many train rows each refit is expensive enough that the null budget is
#: reduced -- and the reduction is recorded per condition, never applied silently. Mirrors
#: E0.1's control exactly. The oracle primary (215 train rows) runs the full budget; the
#: secondary ``A_all`` (~1,692 rows) runs the reduced one.
LARGE_CONDITION_TRAIN_ROWS = 400
LARGE_CONDITION_PERMUTATIONS = 50

#: KEEP/KILL K1: wall-clock budget per image for the quantum feature extraction, cold and
#: sustained. The demonstrator has to be live in the app, not a batch job.
K1_LATENCY_MS_PER_IMAGE = 500.0

#: Images timed one-at-a-time for the sustained latency figure.
LATENCY_SUSTAINED_SAMPLES = 32

#: Train rows checked in the Aer cross-validation of the fast contraction.
AER_CHECK_SAMPLES = 24

SEED = 42


class E2Error(RuntimeError):
    """Raised when an E2 representation cannot be measured on aligned train/val rows."""


# --------------------------------------------------------------------------- inputs
class E2Inputs:
    """One condition's train/validation MobileNet matrices and labels, row-aligned.

    The MobileNet descriptor comes from the E1 feature cache and the labels/patients/ids
    from the oracle pixel partition; the two are joined by image id and cross-checked
    against the handcrafted descriptor cache's independent label derivation. Any image
    missing from either cache is dropped and counted. Only ``train`` and ``validation``
    are ever enumerated.
    """

    def __init__(
        self,
        *,
        condition: str,
        mobilenet_train: np.ndarray,
        mobilenet_validation: np.ndarray,
        y_train: np.ndarray,
        y_validation: np.ndarray,
        patients_train: List[str],
        patients_validation: List[str],
        ids_train: List[str],
        ids_validation: List[str],
    ) -> None:
        self.condition = condition
        self.mobilenet_train = mobilenet_train
        self.mobilenet_validation = mobilenet_validation
        self.y_train = y_train
        self.y_validation = y_validation
        self.patients_train = patients_train
        self.patients_validation = patients_validation
        self.ids_train = ids_train
        self.ids_validation = ids_validation

    @property
    def shape_note(self) -> Dict[str, Any]:
        return {
            "n_train": int(self.y_train.size),
            "n_validation": int(self.y_validation.size),
            "n_train_positive": int(self.y_train.sum()),
            "n_validation_positive": int(self.y_validation.sum()),
            "train_prevalence": round(float(self.y_train.mean()), 6),
            "validation_prevalence": round(float(self.y_validation.mean()), 6),
            "n_train_patients": len(set(self.patients_train)),
            "n_validation_patients": len(set(self.patients_validation)),
            "patient_overlap_train_validation": len(
                set(self.patients_train) & set(self.patients_validation)
            ),
        }


def build_inputs(
    *, config: Optional[Settings] = None, conditions: Sequence[str] = CONDITIONS
) -> Tuple[Dict[str, E2Inputs], Dict[str, Any]]:
    """Load the MobileNet descriptor for each condition on train + validation only.

    The join is the same firewall-safe one E1.1 uses: the MobileNet matrix is indexed by
    image id from the E1 cache, the labels/patients come from the oracle pixel partition,
    and the two label derivations are cross-checked so a stale cache cannot put a wrong
    label into a metric. ``load_partitions(transform=False)`` supplies the second,
    independent label derivation; its features are never used here -- E2 consumes the
    576-d MobileNet embedding, and the PCA-8 compression happens later, TRAIN-only, inside
    :class:`QuantumVisualPreprocessor`.
    """
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)

    descriptors = load_partitions(config=cfg, transform=False)
    pixels = load_pixel_partitions(config=cfg, roi_mode="oracle")
    e1 = load_e1_cache(e1_cache_path(store))

    e1_ids = [str(v) for v in e1["image_ids"]]
    e1_index = {image_id: position for position, image_id in enumerate(e1_ids)}
    if "matrix_mobilenet" not in e1:
        raise E2Error(
            "The E1 feature cache has no 'matrix_mobilenet'; rebuild it with "
            "`python -m backend.training.prepare_e1_features` before running E2."
        )
    mobilenet = np.asarray(e1["matrix_mobilenet"], dtype=np.float64)

    manifest_sha = hashlib.sha256(
        Path(store.split_manifest_path).read_bytes()
    ).hexdigest()

    per_condition: Dict[str, E2Inputs] = {}
    dropped: Dict[str, Dict[str, int]] = {}
    for condition in conditions:
        bundle: Dict[str, Dict[str, Any]] = {}
        drops: Dict[str, int] = {}
        # Iterating this fixed pair is the firewall: "test" is not in it and cannot be
        # reached from here.
        for partition_name in ("train", "validation"):
            pixel_partition = pixels.partitions[partition_name]
            descriptor_partition = descriptors.partitions[partition_name]
            rows = condition_rows(pixel_partition).get(condition)
            if rows is None or not rows.size:
                raise E2Error(
                    f"Condition {condition!r} selects no rows in {partition_name}."
                )
            descriptor_index = {
                image_id: position
                for position, image_id in enumerate(descriptor_partition.image_ids)
            }

            keep_pixel: List[int] = []
            keep_descriptor: List[int] = []
            keep_e1: List[int] = []
            missing = 0
            for row in rows:
                image_id = pixel_partition.image_ids[int(row)]
                if image_id not in descriptor_index or image_id not in e1_index:
                    missing += 1
                    continue
                keep_pixel.append(int(row))
                keep_descriptor.append(descriptor_index[image_id])
                keep_e1.append(e1_index[image_id])
            if not keep_pixel:
                raise E2Error(
                    f"No {partition_name} row of condition {condition!r} is present in "
                    "both the pixel and E1 caches; rebuild them from the same dataset "
                    "state."
                )
            drops[partition_name] = missing

            pixel_rows = np.asarray(keep_pixel, dtype=int)
            descriptor_rows = np.asarray(keep_descriptor, dtype=int)
            e1_rows = np.asarray(keep_e1, dtype=int)
            labels = np.asarray(pixel_partition.labels, dtype=int)[pixel_rows]
            descriptor_labels = np.asarray(descriptor_partition.labels, dtype=int)[
                descriptor_rows
            ]
            if not np.array_equal(labels, descriptor_labels):
                raise E2Error(
                    f"Label mismatch between the pixel and descriptor caches on "
                    f"{partition_name}/{condition}; rebuild both caches."
                )
            bundle[partition_name] = {
                "ids": [pixel_partition.image_ids[int(r)] for r in pixel_rows],
                "patients": [pixel_partition.patient_ids[int(r)] for r in pixel_rows],
                "labels": labels,
                "mobilenet": mobilenet[e1_rows],
            }

        train, validation = bundle["train"], bundle["validation"]
        overlap = set(train["patients"]) & set(validation["patients"])
        if overlap:
            raise E2Error(
                f"{len(overlap)} patients appear in both train and validation for "
                f"condition {condition!r} (first: {sorted(overlap)[0]}). Patient-level "
                "isolation is violated; nothing further is measured."
            )
        dropped[condition] = drops
        per_condition[condition] = E2Inputs(
            condition=condition,
            mobilenet_train=train["mobilenet"],
            mobilenet_validation=validation["mobilenet"],
            y_train=train["labels"],
            y_validation=validation["labels"],
            patients_train=train["patients"],
            patients_validation=validation["patients"],
            ids_train=train["ids"],
            ids_validation=validation["ids"],
        )

    provenance = {
        "split_manifest_sha256": manifest_sha,
        "mobilenet_source": "E1 feature cache matrix_mobilenet (MobileNetV3-Small, 576-d)",
        "mobilenet_dimension": int(mobilenet.shape[1]),
        "e1_cache_version": str(e1["metadata"]["cache_version"]),
        "e1_cache_rows": len(e1_ids),
        "pixel_cache_version": str(pixels.cache_metadata.get("cache_version", "unknown")),
        "rows_dropped_for_cache_misalignment": dropped,
        "join": (
            "MobileNet descriptor by image id from the E1 cache; labels/patients from the "
            "oracle pixel partition; the two label derivations cross-checked. Rows missing "
            "from either cache are dropped and counted."
        ),
        "label_cross_check": (
            "pixel-cache labels asserted equal to the handcrafted-descriptor cache labels "
            "on the joined rows; a mismatch raises rather than picks one silently"
        ),
    }
    return per_condition, provenance


# ----------------------------------------------------------------- representations
def _quantum_features(
    preprocessor: QuantumVisualPreprocessor,
    feature_map: QuantumVisualFeatureMap,
    embeddings: np.ndarray,
) -> np.ndarray:
    """MobileNet embeddings -> TRAIN-fitted angles -> exact 16-d quantum feature vector."""
    angles = preprocessor.to_angles(embeddings)
    return feature_map.transform(angles)


def build_representations(
    inputs: E2Inputs,
    preprocessor: QuantumVisualPreprocessor,
    feature_map: QuantumVisualFeatureMap,
    rff: RandomFourierControl,
) -> Dict[str, Dict[str, Any]]:
    """The three matched representations as ``(train, validation)`` matrix pairs.

    Frozen by construction: quantum (16 local observables), rff (16 random Fourier
    features of the same PCA-8 input), pca (the raw PCA-8 scores). Nothing is selected;
    the only fittable object downstream is the shared logistic head.
    """
    pca_train = preprocessor.pca_scores(inputs.mobilenet_train)
    pca_validation = preprocessor.pca_scores(inputs.mobilenet_validation)

    quantum_train = _quantum_features(preprocessor, feature_map, inputs.mobilenet_train)
    quantum_validation = _quantum_features(
        preprocessor, feature_map, inputs.mobilenet_validation
    )

    observable_labels = list(feature_map.observable_set.labels)
    return {
        "quantum": {
            "label": "fixed 8-qubit angle-encoded reservoir, 16 local Z observables",
            "labels": observable_labels,
            "train": quantum_train,
            "validation": quantum_validation,
            "is_quantum": True,
        },
        "rff": {
            "label": "random Fourier features (RBF) of the same PCA-8 input, 16-d",
            "labels": [f"rff{i}" for i in range(rff.n_features)],
            "train": rff.transform(pca_train),
            "validation": rff.transform(pca_validation),
            "is_quantum": False,
        },
        "pca": {
            "label": "raw PCA-8 scores handed straight to the head",
            "labels": [f"pca{i}" for i in range(PCA_COMPONENTS)],
            "train": pca_train,
            "validation": pca_validation,
            "is_quantum": False,
        },
    }


# --------------------------------------------------------------------------- nulls
def _column_null_draw(
    x_train: np.ndarray,
    y_train: np.ndarray,
    patients_train: Sequence[str],
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    patients_validation: Sequence[str],
    labels: Sequence[str],
    variant: str,
    seed: int,
    draw_seed: int,
) -> Optional[float]:
    """One column-permutation draw: shuffle every feature column in whole patient blocks
    (train and validation independently, labels untouched), refit the head, rescore.

    Module-level so a process pool can pickle it. Shuffling each column by patient blocks
    destroys the joint feature->label relationship while preserving each column's marginal
    and its patient-scale structure -- an i.i.d. row shuffle would make the null easier
    than reality. Refitting the whole head (grouped CV included) rather than reusing the
    real-label ``C`` keeps the null as flexible as the observed model it is compared to.
    """
    rng = np.random.default_rng(draw_seed)
    shuffled_train = np.column_stack(
        [
            _patient_blocked_column_shuffle(x_train[:, j], patients_train, rng)
            for j in range(x_train.shape[1])
        ]
    )
    shuffled_validation = np.column_stack(
        [
            _patient_blocked_column_shuffle(x_validation[:, j], patients_validation, rng)
            for j in range(x_validation.shape[1])
        ]
    )
    try:
        readout = fit_readout(
            shuffled_train, y_train, list(patients_train),
            variant=variant, labels=list(labels), seed=seed,
        )
    except Exception:  # noqa: BLE001 -- a degenerate draw is dropped, never zeroed
        return None
    metrics = _metrics(y_validation, readout.scores(shuffled_validation))
    if not metrics:
        return None
    value = metrics.get("pr_auc")
    return None if value is None else float(value)


def _label_null_draw(
    x_train: np.ndarray,
    permuted: np.ndarray,
    patients_train: Sequence[str],
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    labels: Sequence[str],
    variant: str,
    seed: int,
) -> Optional[float]:
    """One label-permutation draw: refit the head on patient-blocked shuffled labels."""
    try:
        readout = fit_readout(
            x_train, permuted, list(patients_train),
            variant=variant, labels=list(labels), seed=seed,
        )
    except Exception:  # noqa: BLE001 -- a degenerate permutation is dropped, not faked
        return None
    metrics = _metrics(y_validation, readout.scores(x_validation))
    if not metrics:
        return None
    value = metrics.get("pr_auc")
    return None if value is None else float(value)


def _dispatch(tasks: List[Callable[[], Optional[float]]], *, n_jobs: int) -> List[float]:
    """Run the draw closures serially (``n_jobs == 1``) or over a process pool.

    Draws are independent and each is a whole grouped-CV search, so a process pool earns
    its dispatch cost here -- the same granularity E0.1 parallelises at. ``None`` draws
    (degenerate refits) are counted out rather than coerced to a number.
    """
    if n_jobs == 1:
        values = [task() for task in tasks]
    else:
        from joblib import Parallel, delayed

        values = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(task.func)(*task.args) for task in tasks  # type: ignore[attr-defined]
        )
    return [float(v) for v in values if v is not None and np.isfinite(v)]


class _Task:
    """A picklable ``(func, args)`` pair; process pools cannot pickle a lambda closure."""

    __slots__ = ("func", "args")

    def __init__(self, func: Callable[..., Optional[float]], args: Tuple[Any, ...]) -> None:
        self.func = func
        self.args = args

    def __call__(self) -> Optional[float]:
        return self.func(*self.args)


def column_permutation_null(
    x_train: np.ndarray,
    y_train: np.ndarray,
    patients_train: Sequence[str],
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    patients_validation: Sequence[str],
    labels: Sequence[str],
    *,
    variant: str,
    draws: int,
    seed: int,
    n_jobs: int,
) -> List[float]:
    tasks = [
        _Task(
            _column_null_draw,
            (
                x_train, y_train, list(patients_train),
                x_validation, y_validation, list(patients_validation),
                list(labels), variant, seed, seed + 977 + i,
            ),
        )
        for i in range(int(draws))
    ]
    return _dispatch(tasks, n_jobs=n_jobs)


def label_permutation_null(
    x_train: np.ndarray,
    y_train: np.ndarray,
    patients_train: Sequence[str],
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    labels: Sequence[str],
    *,
    variant: str,
    draws: int,
    seed: int,
    n_jobs: int,
) -> List[float]:
    permutations = patient_blocked_permutations(
        y_train, list(patients_train), n=int(draws), seed=seed
    )
    tasks = [
        _Task(
            _label_null_draw,
            (
                x_train, permuted, list(patients_train),
                x_validation, y_validation, list(labels), variant, seed,
            ),
        )
        for permuted in permutations
    ]
    return _dispatch(tasks, n_jobs=n_jobs)


# ------------------------------------------------------------- one representation
def evaluate_representation(
    name: str,
    spec: Dict[str, Any],
    inputs: E2Inputs,
    *,
    draws: int,
    seed: int,
    n_jobs: int,
    with_label_null: bool,
) -> Dict[str, Any]:
    """Fit the shared head on one representation, score validation, gate on the null.

    Returns the fitted head as a portable :class:`LogisticHead` dict (so the backend can
    re-apply it without scikit-learn), the validation metrics, the K2 degeneracy of the
    train feature matrix, and the column-permutation gating null. The label-permutation
    null is added for the quantum representation only, reported but non-gating (its power
    ceiling makes it a weaker instrument than the column null for this decision).
    """
    x_train = np.asarray(spec["train"], dtype=np.float64)
    x_validation = np.asarray(spec["validation"], dtype=np.float64)
    labels = list(spec["labels"])
    variant = f"e2_{name}"

    readout = fit_readout(
        x_train, inputs.y_train, list(inputs.patients_train),
        variant=variant, labels=labels, seed=seed,
    )
    head = LogisticHead.from_readout_dict(readout.to_dict())

    # Re-apply the portable head and assert it reproduces the sklearn readout, so the
    # persisted inference object is validated here rather than trusted.
    validation_scores = readout.scores(x_validation)
    head_scores = head.scores(x_validation)
    head_max_deviation = float(np.max(np.abs(validation_scores - head_scores))) if validation_scores.size else 0.0
    train_metrics = _metrics(inputs.y_train, readout.scores(x_train))
    validation_metrics = _metrics(inputs.y_validation, validation_scores)
    observed_pr = None if not validation_metrics else validation_metrics.get("pr_auc")

    # --- K2 degeneracy on the TRAIN feature matrix (variance floor, rank, constants).
    variances = x_train.var(axis=0, ddof=0)
    n_constant = int((variances <= VARIANCE_FLOOR).sum())
    centred = x_train - x_train.mean(axis=0)
    rank = int(np.linalg.matrix_rank(centred))
    degeneracy = {
        "n_features": int(x_train.shape[1]),
        "n_constant_observables": n_constant,
        "variance_floor": VARIANCE_FLOOR,
        "min_feature_variance": float(variances.min()),
        "max_feature_variance": float(variances.max()),
        "effective_rank_centred": rank,
        "non_degenerate": bool(n_constant == 0 and rank == int(x_train.shape[1])),
    }

    # --- Column-permutation gating null.
    column_null = column_permutation_null(
        x_train, inputs.y_train, inputs.patients_train,
        x_validation, inputs.y_validation, inputs.patients_validation,
        labels, variant=variant, draws=draws, seed=seed, n_jobs=n_jobs,
    )
    column_rank = _rank_in_null(observed_pr, column_null)
    survives_gating_null = (
        None
        if column_rank is None
        else bool(column_rank["exceeds_null_p95"] and column_rank["empirical_p"] <= 0.05)
    )

    result: Dict[str, Any] = {
        "representation": name,
        "label": spec["label"],
        "is_quantum": bool(spec["is_quantum"]),
        "input_dimension": int(x_train.shape[1]),
        "n_train": int(x_train.shape[0]),
        "n_validation": int(x_validation.shape[0]),
        "readout": {
            "kind": readout.kind,
            "n_trainable_parameters": readout.n_trainable,
            "regularisation_C": readout.regularisation_c,
            "selection": readout.cv_note,
            "n_excluded_below_variance_floor": (
                None if readout.scaler is None else readout.scaler.n_excluded
            ),
        },
        "portable_head_max_deviation_vs_sklearn": head_max_deviation,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_pr_auc": observed_pr,
        "degeneracy": degeneracy,
        "column_permutation_null": {
            "n_requested": int(draws),
            "n_usable": len(column_null),
            "blocking": (
                "every feature column shuffled in whole patient blocks, TRAIN and "
                "VALIDATION independently; labels untouched; head refit per draw"
            ),
            "pr_auc": _null_summary(column_null),
            "observed_vs_null": column_rank,
            "gates_the_claim": True,
        },
        "survives_gating_null": survives_gating_null,
        "_head": head,  # stripped before serialisation; kept for persistence
    }

    if with_label_null:
        label_null = label_permutation_null(
            x_train, inputs.y_train, inputs.patients_train,
            x_validation, inputs.y_validation,
            labels, variant=variant, draws=draws, seed=seed, n_jobs=n_jobs,
        )
        result["label_permutation_null"] = {
            "n_requested": int(draws),
            "n_usable": len(label_null),
            "blocking": "patient-level TRAIN label permutation; head refit per draw",
            "pr_auc": _null_summary(label_null),
            "observed_vs_null": _rank_in_null(observed_pr, label_null),
            "gates_the_claim": False,
            "power_note": (
                "reported, not gated. Refitting on permuted labels while scoring against "
                "real labels randomises the head and floors the empirical p near the base "
                "rate, so it is a weaker instrument than the column-permutation null for "
                "this decision."
            ),
        }
    return result


# ------------------------------------------------------------------- latency (K1)
def measure_latency(
    preprocessor: QuantumVisualPreprocessor,
    train_embeddings: np.ndarray,
    *,
    sustained_samples: int = LATENCY_SUSTAINED_SAMPLES,
) -> Dict[str, Any]:
    """K1: cold and sustained wall-clock per image for the quantum feature extraction.

    Cold uses a *fresh* feature map so the Aer simulator construction and first-circuit
    cost are inside the number. Sustained times single images one at a time on a warm map
    -- the app's real access pattern. Batch throughput is reported for context. Timed on
    TRAIN angles only.
    """
    angles = preprocessor.to_angles(train_embeddings)
    n = int(angles.shape[0])
    sample = min(sustained_samples, n)

    cold_map = QuantumVisualFeatureMap()
    started = time.perf_counter()
    cold_map.transform(angles[:1])
    cold_ms = (time.perf_counter() - started) * 1000.0

    warm_map = QuantumVisualFeatureMap()
    warm_map.transform(angles[:1])  # discard the warm-up
    per_image_ms: List[float] = []
    for i in range(sample):
        started = time.perf_counter()
        warm_map.transform(angles[i : i + 1])
        per_image_ms.append((time.perf_counter() - started) * 1000.0)

    started = time.perf_counter()
    warm_map.transform(angles)
    batch_ms = (time.perf_counter() - started) * 1000.0

    sustained = float(np.mean(per_image_ms)) if per_image_ms else float("nan")
    passes = bool(
        cold_ms <= K1_LATENCY_MS_PER_IMAGE and sustained <= K1_LATENCY_MS_PER_IMAGE
    )
    return {
        "budget_ms_per_image": K1_LATENCY_MS_PER_IMAGE,
        "cold_ms_first_image": round(cold_ms, 3),
        "sustained_ms_per_image_mean": round(sustained, 3),
        "sustained_ms_per_image_p95": (
            round(float(np.percentile(per_image_ms, 95)), 3) if per_image_ms else None
        ),
        "batch_ms_per_image": round(batch_ms / max(n, 1), 3),
        "n_sustained_samples": sample,
        "passes": passes,
    }


# ------------------------------------------------------------------------ per condition
def _condition_note(condition: str) -> str:
    base = CONDITION_DESCRIPTIONS.get(condition, condition)
    if condition == PRIMARY:
        return (
            base + " PRIMARY condition: one ROI source, so the DEC-024 area channel is "
            "removed by construction and PR-AUC is read against the base rate."
        )
    return (
        base + " SECONDARY condition: the DEC-024 geometry leak is PRESENT -- ROI area "
        f"alone scores validation ROC-AUC {GEOMETRY_LEAK_ROC_AUC}, so read every number "
        "against that, not 0.5."
    )


def run_condition(
    inputs: E2Inputs,
    *,
    draws: int,
    seed: int,
    n_jobs: int,
    fit_artifacts: bool,
) -> Dict[str, Any]:
    """Fit the preprocessor + RFF control TRAIN-only, evaluate all three representations.

    ``fit_artifacts`` marks the condition whose fitted objects (preprocessor, RFF, the
    three heads, the Aer validation) are persisted for backend integration -- the primary
    condition. The secondary condition is measured and reported but its objects are not
    the ones the demonstrator ships.
    """
    started = time.perf_counter()

    preprocessor = QuantumVisualPreprocessor.fit(inputs.mobilenet_train, seed=seed)
    feature_map = QuantumVisualFeatureMap()
    rff = RandomFourierControl.fit(
        preprocessor.pca_scores(inputs.mobilenet_train),
        n_features=E2_N_FEATURES, seed=seed,
    )

    # Validate the fast contraction against Aer on a TRAIN angle sample before any feature
    # is reported. Never on validation or test.
    train_angles = preprocessor.to_angles(inputs.mobilenet_train)
    aer_validation = validate_against_aer(
        feature_map, train_angles[: min(AER_CHECK_SAMPLES, train_angles.shape[0])]
    )

    representations = build_representations(inputs, preprocessor, feature_map, rff)
    evaluated: Dict[str, Any] = {}
    for name, spec in representations.items():
        evaluated[name] = evaluate_representation(
            name, spec, inputs,
            draws=draws, seed=seed, n_jobs=n_jobs,
            with_label_null=spec["is_quantum"],
        )

    # Connected-correlation telemetry (entanglement witness) on TRAIN quantum features.
    quantum_train_features = representations["quantum"]["train"]
    correlations = feature_map.connected_correlations(quantum_train_features)
    correlation_telemetry = {
        "note": (
            "connected correlations C_i = <Z_iZ_{i+1}> - <Z_i><Z_{i+1}> per NN ring edge, "
            "averaged over TRAIN rows. Nonzero values are direct evidence the CZ ring "
            "produced correlations a separable angle encoding could not. Entanglement "
            "witness only -- NOT an advantage metric."
        ),
        "mean_abs_connected_correlation_per_edge": [
            round(float(v), 8) for v in np.abs(correlations).mean(axis=0)
        ],
        "max_abs_connected_correlation": round(float(np.max(np.abs(correlations))), 8),
    }

    latency = measure_latency(preprocessor, inputs.mobilenet_train)

    quantum = evaluated["quantum"]
    k2_pass = bool(quantum["degeneracy"]["non_degenerate"])
    k3_pass = bool(quantum["survives_gating_null"]) if quantum["survives_gating_null"] is not None else False
    k1_pass = bool(latency["passes"])
    keep = k1_pass and k2_pass and k3_pass
    keep_kill = {
        "K1_latency": {
            "passes": k1_pass,
            "budget_ms_per_image": K1_LATENCY_MS_PER_IMAGE,
            "cold_ms_first_image": latency["cold_ms_first_image"],
            "sustained_ms_per_image_mean": latency["sustained_ms_per_image_mean"],
        },
        "K2_non_degenerate": {
            "passes": k2_pass,
            "effective_rank_centred": quantum["degeneracy"]["effective_rank_centred"],
            "n_constant_observables": quantum["degeneracy"]["n_constant_observables"],
        },
        "K3_exceeds_own_null_p95": {
            "passes": k3_pass,
            "observed_vs_null": quantum["column_permutation_null"]["observed_vs_null"],
            "null_p95": (quantum["column_permutation_null"]["pr_auc"] or {}).get("p95"),
        },
        "decision": "KEEP" if keep else "KILL",
        "decision_note": (
            "KEEP: the quantum stage runs within budget, its features are non-degenerate, "
            "and its validation PR-AUC exceeds its own patient-blocked null p95 (p<=0.05)."
            if keep
            else "KILL as a predictor: at least one gate failed. The circuit is preserved "
            "as telemetry (execution provenance + connected-correlation witness); it is "
            "NOT reported as a working risk model and the classical primary path is "
            "untouched."
        ),
    }

    # Matched-control read: quantum PR-AUC never stands alone.
    matched_controls = {
        name: {
            "validation_pr_auc": evaluated[name]["validation_pr_auc"],
            "survives_gating_null": evaluated[name]["survives_gating_null"],
        }
        for name in ("rff", "pca")
    }

    fitted_objects = None
    if fit_artifacts:
        fitted_objects = {
            "preprocessor": preprocessor,
            "rff": rff,
            "feature_map": feature_map,
            "heads": {name: evaluated[name]["_head"] for name in evaluated},
            "aer_validation": aer_validation,
        }

    # Strip the non-serialisable head objects from the reported payload.
    for name in evaluated:
        evaluated[name].pop("_head", None)

    duration = time.perf_counter() - started
    payload = {
        "condition": inputs.condition,
        "condition_note": _condition_note(inputs.condition),
        "is_primary": inputs.condition == PRIMARY,
        "rows": inputs.shape_note,
        "geometry_leak_bar": {
            "area_only_validation_roc_auc": GEOMETRY_LEAK_ROC_AUC,
            "applies": inputs.condition != PRIMARY,
        },
        "preprocessor": {
            "version": preprocessor.version,
            "artifact_hash": preprocessor.artifact_hash,
            "pca_explained_variance_ratio": [
                round(float(v), 6) for v in preprocessor.pca.explained_variance_ratio
            ],
            "pca_total_explained_variance": round(
                float(preprocessor.pca.explained_variance_ratio.sum()), 6
            ),
            "angle_quantile": preprocessor.scaler.quantile,
        },
        "rff_control": {
            "gamma": round(float(rff.gamma), 8),
            "median_squared_distance": round(float(rff.median_squared_distance), 8),
            "n_features": rff.n_features,
            "seed": rff.seed,
        },
        "aer_validation": aer_validation,
        "representations": evaluated,
        "matched_controls_vs_quantum": {
            "quantum_validation_pr_auc": quantum["validation_pr_auc"],
            "controls": matched_controls,
            "note": (
                "the quantum PR-AUC is reported here beside its equal-dimension RFF "
                "control and the raw PCA-8 control on identical rows and the identical "
                "head; it is never reported alone."
            ),
        },
        "connected_correlation_telemetry": correlation_telemetry,
        "latency": latency,
        "keep_kill": keep_kill,
        "duration_seconds": round(duration, 3),
    }
    return payload, fitted_objects


# --------------------------------------------------------------------------- persistence
def _persist_artifacts(
    store: ArtifactStore, condition: str, fitted: Dict[str, Any], environment: Dict[str, Any]
) -> Dict[str, Any]:
    """Persist the primary condition's fitted E2 objects as one JSON bundle (no pickle).

    The backend re-applies these at inference: ``preprocessor`` -> 8 angles, the fixed
    circuit -> 16 features, ``heads.quantum`` -> the experimental secondary probability.
    The controls' heads are stored too so the report's matched comparison is reproducible.
    """
    preprocessor: QuantumVisualPreprocessor = fitted["preprocessor"]
    rff: RandomFourierControl = fitted["rff"]
    feature_map: QuantumVisualFeatureMap = fitted["feature_map"]
    heads: Dict[str, LogisticHead] = fitted["heads"]

    bundle = {
        "e2_driver_version": E2_VERSION,
        "quantum_visual_version": QUANTUM_VISUAL_VERSION,
        "visual_circuit_version": VISUAL_CIRCUIT_VERSION,
        "fitted_on_condition": condition,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": E2_SEED,
        "test_partition_used": False,
        "fitted_on": "train rows of the primary oracle condition only",
        "preprocessor": preprocessor.to_dict(),
        "preprocessor_artifact_hash": preprocessor.artifact_hash,
        "circuit_telemetry": feature_map.telemetry(),
        "observable_labels": list(feature_map.observable_set.labels),
        "rff_control": rff.to_dict(),
        "heads": {name: head.to_dict() for name, head in heads.items()},
        "aer_validation": fitted["aer_validation"],
        "environment": environment,
    }
    path = store.write_component(E2_ARTIFACT_VERSION, E2_ARTIFACT_FILE, bundle)
    return {"path": str(path), "preprocessor_artifact_hash": preprocessor.artifact_hash}


# -------------------------------------------------------------------------------- driver
def run(
    *,
    config: Optional[Settings] = None,
    conditions: Sequence[str] = CONDITIONS,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = SEED,
    n_jobs: int = 1,
    persist: bool = True,
) -> Dict[str, Any]:
    """Run E2 on every condition and return the full payload. Reads train + validation."""
    started = time.perf_counter()
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)
    per_condition, provenance = build_inputs(config=cfg, conditions=conditions)

    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "seed": seed,
        "fixed_threshold": FIXED_THRESHOLD,
    }

    results: Dict[str, Any] = {}
    persistence: Optional[Dict[str, Any]] = None
    for condition in conditions:
        inputs = per_condition[condition]
        n_train_rows = int(inputs.y_train.size)
        draws = n_permutations
        budget_note: Optional[str] = None
        if n_train_rows > LARGE_CONDITION_TRAIN_ROWS and n_permutations > LARGE_CONDITION_PERMUTATIONS:
            draws = LARGE_CONDITION_PERMUTATIONS
            budget_note = (
                f"reduced from {n_permutations} to {draws} draws: {n_train_rows} train "
                f"rows exceeds the {LARGE_CONDITION_TRAIN_ROWS}-row threshold and each draw "
                "refits the whole head including its inner grouped CV. Finest resolvable "
                f"empirical p is {1 / (draws + 1):.4f}."
            )
            logger.info("  %s null budget: %s", condition, budget_note)

        logger.info(
            "Condition %s: %d train / %d validation rows (%d / %d positive)",
            condition, inputs.y_train.size, inputs.y_validation.size,
            int(inputs.y_train.sum()), int(inputs.y_validation.sum()),
        )
        payload, fitted = run_condition(
            inputs, draws=draws, seed=seed, n_jobs=n_jobs,
            fit_artifacts=(condition == PRIMARY),
        )
        payload["null_budget_note"] = budget_note
        results[condition] = payload
        if fitted is not None and persist:
            persistence = _persist_artifacts(store, condition, fitted, environment)

    primary = results.get(PRIMARY, {})
    primary_keep_kill = primary.get("keep_kill", {})
    return {
        "e2_version": E2_VERSION,
        "architecture": "DEC-035 Candidate B: fixed 8-qubit angle-encoded visual reservoir",
        "role": (
            "EXPERIMENTAL quantum visual demonstrator (SIH fallback). Secondary signal "
            "only; the classical baseline remains the primary clinical decision-maker "
            "(DEC-033/DEC-034). No quantum-advantage claim is made or authorised."
        ),
        "test_partition_used": False,
        "partitions_read": ["train", "validation"],
        "test_partition_note": (
            "E2 reads TRAIN and VALIDATION only. No code path in this module can select "
            "the test partition; PCA, the angle scaler, the RFF bandwidth, every head and "
            "every null saw train rows only, and validation was scored once per "
            "representation."
        ),
        "primary_condition": PRIMARY,
        "provenance": provenance,
        "conditions": results,
        "primary_keep_kill_decision": primary_keep_kill.get("decision"),
        "persistence": persistence,
        "environment": environment,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="E2 fixed quantum visual demonstrator.")
    parser.add_argument("--out", type=Path, default=None, help="Payload destination.")
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Process pool size for the permutation nulls; 1 (default) runs serially.",
    )
    parser.add_argument(
        "--condition", action="append", default=None,
        help="Restrict to a named oracle condition (repeatable). Primary if omitted.",
    )
    parser.add_argument(
        "--no-persist", action="store_true",
        help="Skip writing the fitted-artifact bundle (report-only run).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    for noisy in ("qiskit", "stevedore", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    conditions = tuple(args.condition) if args.condition else CONDITIONS
    payload = run(
        conditions=conditions,
        n_permutations=args.permutations,
        seed=args.seed,
        n_jobs=args.n_jobs,
        persist=not args.no_persist,
    )
    destination = args.out or Path("reports_e2_quantum_visual.json")
    destination.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    logger.info("Wrote %s", destination)
    logger.info(
        "Primary KEEP/KILL: %s", payload.get("primary_keep_kill_decision")
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
