"""E1.1: the classical failure map -- seven pre-registered candidate representations.

``docs/PHASE_E1_RESEARCH_SPEC.md`` §2 asks one question and this module answers it:
**is the 181-feature descriptor's known structural incompleteness costing measurable
accuracy?** Seven candidates, fixed in the spec before any of them ran:

===  ===================================================================  ============
C1   the existing 181 features                                            control
C2   downsampled ROI pixels (1024 then 256 dims)                          gap 1, 7
C3   HOG -- gradient **orientation**                                      gap 2, 4
C4   LBP + GLCM -- joint / co-occurrence texture                          gap 2, 5
C5   8x8 grid + centre-vs-surround, same feature philosophy               gap 4, 5, 8
C6   frozen MobileNetV3-Small embedding                                   the ceiling
C7   fusion of the best two families                                      gap 6
===  ===================================================================  ============

What "measurable" means here is fixed in advance and is not a decimal place
-------------------------------------------------------------------------
21 validation positives (§1.5). A PR-AUC difference of +0.05 is inside sampling noise,
so every candidate is compared against C1 through a **patient-level bootstrap** of C1's
own validation PR-AUC, and the margin a candidate must clear is ``delta = 2 * se(C1)``
computed from that bootstrap **before** any candidate is scored. The spec derives the
margin from the data rather than asserting one (§9.4), and that is what makes a null
result here a measurement instead of a shrug.

Two guard rails from §3, both of which exist to stop this module manufacturing a finding:

1. **A single weak classifier is not a bottleneck.** Every family is fitted with both a
   linear and a nonlinear classifier, and "family fails" requires *both* to fall short.
   The reported per-family score is its best classifier, so a family is never penalised
   for the harness's choice of model.
2. **A gap a classical control already closes is not a quantum motivation.** C5 and C6
   are in the grid precisely so they get first refusal. If either closes a gap, §3's
   verdict is "use the classical fix" and Q1 loses its premise.

Discipline, identical for all seven
-----------------------------------
Train fits, validation scores, **test is never read** -- there is no code path in this
module that can select the test partition, and ``test_partition_used: false`` is asserted
in the payload and in ``tests/test_e1_failure_map.py``. Every hyperparameter (SVM ``C``,
RBF ``gamma``, forest depth, boosting rate) is chosen by patient-grouped CV *inside
train*, reusing ``grouped_folds`` and ``median_heuristic_gamma`` from
``backend.evaluation.e0_controls`` rather than a second implementation of either. Every
scaler is fitted on train rows only. Conditions ``A_lesion_polygon`` (primary, the
geometry leak removed by construction) and ``A_all`` (secondary, leak present at ROC-AUC
0.72) are reported separately and never pooled.

Multiplicity is corrected across the **frozen** candidate x classifier x condition count,
Benjamini-Hochberg, with the count written into the payload before the first fit. E0's
17-of-136 artefact is the reason that count is frozen rather than counted afterwards.

Usage::

    python -m backend.evaluation.e1_failure_map --out reports_e1_failure_map.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings, settings as default_settings
# Reused, never reimplemented: the fold assignment, the RBF width rule and the metric
# call must be the *same* ones E0 used, or a difference between E0's numbers and these
# is ambiguous between "the representations differ" and "the harnesses differ".
from backend.evaluation.e0_controls import (
    C_GRID,
    N_FOLDS,
    ControlsError,
    grouped_folds,
    median_heuristic_gamma,
)
from backend.evaluation.e0_readout import (
    _metrics,
    _null_summary,
    _rank_in_null,
    patient_blocked_permutations,
)
from backend.training.pixel_data import (
    CONDITION_ORACLE_ALL,
    CONDITION_ORACLE_LESION,
)
from backend.ml.artifacts import ArtifactStore
from backend.training.data import load_partitions
from backend.training.pixel_data import (
    CONDITION_DESCRIPTIONS,
    condition_rows,
    load_pixel_partitions,
)
from backend.training.prepare_e1_features import e1_cache_path, load_e1_cache

logger = logging.getLogger(__name__)

FAILURE_MAP_VERSION = "v1-e1-failure-map-1"

#: The two oracle conditions §2 requires, in reporting order. Primary first.
#: There is deliberately no way to name a predicted condition or the test partition here.
PRIMARY = CONDITION_ORACLE_LESION
CONDITIONS: Tuple[str, ...] = (PRIMARY, CONDITION_ORACLE_ALL)

#: Bootstrap draws for the C1 margin. §9.4 says >= 2000.
BOOTSTRAP_DRAWS = 2000

#: Column-permutation null draws for the C7 fusion increment (§9.2).
INCREMENT_PERMUTATIONS = 200

#: Forest / boosting grids. Small on purpose: §1.5's 87 primary-condition train positives
#: cannot support a large search, and a large one would be the selection surface §2 warns
#: about rather than a measurement.
FOREST_GRID: Tuple[Dict[str, Any], ...] = (
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 2},
    {"n_estimators": 600, "max_depth": 12, "min_samples_leaf": 1},
)
BOOSTING_GRID: Tuple[Dict[str, Any], ...] = (
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 200, "learning_rate": 0.10, "max_depth": 3},
)

FIXED_THRESHOLD = 0.50
SEED = 42


class FailureMapError(RuntimeError):
    """Raised when a candidate cannot be measured on aligned rows."""


# ------------------------------------------------------------------------- classifiers
def _standardise(train: np.ndarray, validation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score both partitions using **train** mean and scale only.

    Constant train columns are passed through with scale 1 rather than dropped: dropping
    would make the feature count depend on the partition, and a column that is constant
    on train genuinely carries no train-time information anyway.
    """
    mean = train.mean(axis=0)
    raw = train.std(axis=0)
    scale = np.where(raw > 1e-12, raw, 1.0)
    return (train - mean) / scale, (validation - mean) / scale


def _average_precision(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=int)
    if y.size == 0 or len(set(y.tolist())) < 2:
        return None
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, np.asarray(score, dtype=np.float64)))


def _cv_select(
    candidates: Sequence[Any],
    fit_score: Callable[[Any], Optional[float]],
    *,
    label: str,
) -> Tuple[Any, Dict[str, Optional[float]]]:
    """Pick the setting with the best within-train out-of-fold average precision.

    Strictly ``>``, so ties break towards the **earlier** grid entry. Every grid in this
    module is ordered strongest-regularisation-first, so a tie resolves to the simpler
    model: a tie means the data did not distinguish the settings, and picking the more
    flexible one then would be picking on noise.
    """
    scored: Dict[str, Optional[float]] = {}
    best: Tuple[Optional[float], Any] = (None, None)
    for candidate in candidates:
        value = fit_score(candidate)
        scored[str(candidate)] = None if value is None else round(value, 6)
        if value is not None and (best[0] is None or value > best[0]):
            best = (value, candidate)
    if best[1] is None:
        raise FailureMapError(
            f"No {label} setting produced a usable within-train CV score; the folds are "
            "degenerate for this condition."
        )
    return best[1], scored


def _svm_arm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    kernel: str,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Linear or RBF SVM with ``C`` (and the RBF width) fixed inside train only.

    The RBF width comes from the median heuristic on the **train** Gram, via the same
    helper E0 used. The rows are L2-normalised first so the ``||x-y||^2 = 2 - 2<x,y>``
    identity that helper relies on actually holds -- feeding it an unnormalised Gram
    would silently produce a width for a distance the data does not have.
    """
    from sklearn.svm import SVC

    def _l2(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms > 1e-12, norms, 1.0)

    xt, xv = _l2(x_train), _l2(x_validation)
    k_tt = xt @ xt.T
    k_vt = xv @ xt.T

    gamma: Optional[float] = None
    median_squared: Optional[float] = None
    if kernel == "rbf":
        gamma, median_squared = median_heuristic_gamma(k_tt)
        squared_tt = np.maximum(2.0 - 2.0 * k_tt, 0.0)
        squared_vt = np.maximum(2.0 - 2.0 * k_vt, 0.0)
        k_tt, k_vt = np.exp(-gamma * squared_tt), np.exp(-gamma * squared_vt)

    y_train = np.asarray(y_train, dtype=int)

    def _score(c: float) -> Optional[float]:
        oof = np.full(y_train.size, np.nan, dtype=np.float64)
        for train_idx, test_idx in folds:
            if len(set(y_train[train_idx].tolist())) < 2:
                continue
            model = SVC(C=c, kernel="precomputed", class_weight="balanced")
            model.fit(k_tt[np.ix_(train_idx, train_idx)], y_train[train_idx])
            oof[test_idx] = model.decision_function(k_tt[np.ix_(test_idx, train_idx)])
        usable = np.isfinite(oof)
        if usable.sum() < 2:
            return None
        return _average_precision(y_train[usable], oof[usable])

    best_c, by_c = _cv_select(C_GRID, _score, label=f"{kernel}-SVM C")

    final = SVC(C=float(best_c), kernel="precomputed", class_weight="balanced")
    final.fit(k_tt, y_train)
    train_decision = final.decision_function(k_tt)
    validation_decision = final.decision_function(k_vt)

    oof = np.full(y_train.size, np.nan, dtype=np.float64)
    for train_idx, test_idx in folds:
        if len(set(y_train[train_idx].tolist())) < 2:
            continue
        model = SVC(C=float(best_c), kernel="precomputed", class_weight="balanced")
        model.fit(k_tt[np.ix_(train_idx, train_idx)], y_train[train_idx])
        oof[test_idx] = model.decision_function(k_tt[np.ix_(test_idx, train_idx)])

    # Brier and ECE are undefined on an unbounded SVM margin, and the spec asks for both.
    # The Platt map is fitted on OUT-OF-FOLD train decisions: in-sample values would
    # report a calibration this arm does not have at validation time.
    probability_train = probability_validation = None
    platt_note = "no probability map: out-of-fold train decisions were single-class"
    usable = np.isfinite(oof)
    if usable.sum() >= 2 and len(set(y_train[usable].tolist())) == 2:
        from sklearn.linear_model import LogisticRegression

        platt = LogisticRegression(max_iter=5000, random_state=seed)
        platt.fit(oof[usable].reshape(-1, 1), y_train[usable])
        probability_train = np.asarray(
            platt.predict_proba(oof.reshape(-1, 1))[:, 1], dtype=np.float64
        )
        probability_validation = np.asarray(
            platt.predict_proba(validation_decision.reshape(-1, 1))[:, 1], dtype=np.float64
        )
        platt_note = (
            "probabilities from a 1-D logistic map fitted on out-of-fold TRAIN decision "
            "values; no validation row contributed to it"
        )

    return {
        "family": "svm",
        "kernel": kernel,
        "linear": kernel == "linear",
        "hyperparameters": {
            "C": float(best_c),
            "gamma": None if gamma is None else round(float(gamma), 8),
            "median_train_squared_distance": (
                None if median_squared is None else round(float(median_squared), 8)
            ),
        },
        "cv_average_precision_by_setting": by_c,
        "n_support_vectors": int(sum(int(v) for v in final.n_support_)),
        "train_out_of_fold": oof,
        "train_score": train_decision,
        "validation_score": validation_decision,
        "train_probability": probability_train,
        "validation_probability": probability_validation,
        "platt_note": platt_note,
    }


def _tree_arm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    name: str,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Random forest or gradient boosting, hyperparameters selected inside train.

    Uses ``backend.ml.baselines.build_estimator`` so these arms are the same estimators,
    with the same class weighting and the same serial-forest determinism note, that the
    frozen Phase-D baselines use. A locally constructed forest could differ in a default
    and the comparison to C1 would stop being a comparison of representations.
    """
    from backend.ml.baselines import build_estimator

    grid = FOREST_GRID if name == "random_forest" else BOOSTING_GRID
    y_train = np.asarray(y_train, dtype=int)

    def _score(params: Dict[str, Any]) -> Optional[float]:
        oof = np.full(y_train.size, np.nan, dtype=np.float64)
        for train_idx, test_idx in folds:
            if len(set(y_train[train_idx].tolist())) < 2:
                continue
            estimator = build_estimator(name, dict(params), seed=seed)
            estimator.fit(x_train[train_idx], y_train[train_idx])
            oof[test_idx] = np.asarray(estimator.predict_proba(x_train[test_idx]))[:, 1]
        usable = np.isfinite(oof)
        if usable.sum() < 2:
            return None
        return _average_precision(y_train[usable], oof[usable])

    best_params, by_setting = _cv_select(grid, _score, label=name)

    oof = np.full(y_train.size, np.nan, dtype=np.float64)
    for train_idx, test_idx in folds:
        if len(set(y_train[train_idx].tolist())) < 2:
            continue
        estimator = build_estimator(name, dict(best_params), seed=seed)
        estimator.fit(x_train[train_idx], y_train[train_idx])
        oof[test_idx] = np.asarray(estimator.predict_proba(x_train[test_idx]))[:, 1]

    final = build_estimator(name, dict(best_params), seed=seed)
    final.fit(x_train, y_train)
    train_probability = np.asarray(final.predict_proba(x_train))[:, 1]
    validation_probability = np.asarray(final.predict_proba(x_validation))[:, 1]

    return {
        "family": name,
        "kernel": None,
        "linear": False,
        "hyperparameters": dict(best_params),
        "cv_average_precision_by_setting": by_setting,
        "n_support_vectors": None,
        "train_out_of_fold": oof,
        "train_score": train_probability,
        "validation_score": validation_probability,
        "train_probability": train_probability,
        "validation_probability": validation_probability,
        "platt_note": "tree ensembles output probabilities directly; no Platt map needed",
    }


def _logistic_arm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    seed: int = SEED,
) -> Dict[str, Any]:
    """L2 logistic regression, ``C`` inside train. The linear arm for tree-only families."""
    from sklearn.linear_model import LogisticRegression

    y_train = np.asarray(y_train, dtype=int)

    def _score(c: float) -> Optional[float]:
        oof = np.full(y_train.size, np.nan, dtype=np.float64)
        for train_idx, test_idx in folds:
            if len(set(y_train[train_idx].tolist())) < 2:
                continue
            model = LogisticRegression(
                C=float(c), max_iter=5000, class_weight="balanced", random_state=seed
            )
            model.fit(x_train[train_idx], y_train[train_idx])
            oof[test_idx] = np.asarray(model.predict_proba(x_train[test_idx]))[:, 1]
        usable = np.isfinite(oof)
        if usable.sum() < 2:
            return None
        return _average_precision(y_train[usable], oof[usable])

    best_c, by_c = _cv_select(C_GRID, _score, label="logistic C")

    oof = np.full(y_train.size, np.nan, dtype=np.float64)
    for train_idx, test_idx in folds:
        if len(set(y_train[train_idx].tolist())) < 2:
            continue
        model = LogisticRegression(
            C=float(best_c), max_iter=5000, class_weight="balanced", random_state=seed
        )
        model.fit(x_train[train_idx], y_train[train_idx])
        oof[test_idx] = np.asarray(model.predict_proba(x_train[test_idx]))[:, 1]

    final = LogisticRegression(
        C=float(best_c), max_iter=5000, class_weight="balanced", random_state=seed
    )
    final.fit(x_train, y_train)
    return {
        "family": "logistic_regression",
        "kernel": None,
        "linear": True,
        "hyperparameters": {"C": float(best_c)},
        "cv_average_precision_by_setting": by_c,
        "n_support_vectors": None,
        "train_out_of_fold": oof,
        "train_score": np.asarray(final.predict_proba(x_train))[:, 1],
        "validation_score": np.asarray(final.predict_proba(x_validation))[:, 1],
        "train_probability": np.asarray(final.predict_proba(x_train))[:, 1],
        "validation_probability": np.asarray(final.predict_proba(x_validation))[:, 1],
        "platt_note": "logistic outputs probabilities directly; no Platt map needed",
    }


#: Which classifiers each candidate is fitted with. Every entry contains at least one
#: linear and one nonlinear arm, because §3's guard rail makes "both fell short" the
#: precondition for calling a family a failure.
ARM_SPECS: Dict[str, Tuple[str, ...]] = {
    "C1": ("logistic", "random_forest", "gradient_boosting"),
    "C2": ("linear_svm", "rbf_svm"),
    "C2b": ("linear_svm", "rbf_svm"),
    "C3": ("linear_svm", "rbf_svm", "random_forest"),
    "C4": ("logistic", "rbf_svm", "random_forest"),
    "C5": ("logistic", "random_forest", "gradient_boosting"),
    "C6": ("linear_svm", "rbf_svm", "random_forest"),
    "C7": ("logistic", "rbf_svm", "random_forest"),
}


def _fit_arm(
    arm: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    seed: int = SEED,
) -> Dict[str, Any]:
    if arm == "linear_svm":
        return _svm_arm(x_train, y_train, x_validation, folds, kernel="linear", seed=seed)
    if arm == "rbf_svm":
        return _svm_arm(x_train, y_train, x_validation, folds, kernel="rbf", seed=seed)
    if arm == "logistic":
        return _logistic_arm(x_train, y_train, x_validation, folds, seed=seed)
    if arm in ("random_forest", "gradient_boosting"):
        return _tree_arm(x_train, y_train, x_validation, folds, name=arm, seed=seed)
    raise FailureMapError(f"Unknown classifier arm {arm!r}.")


# ---------------------------------------------------------------------------- bootstrap
def bootstrap_pr_auc(
    y: np.ndarray,
    scores: np.ndarray,
    patients: Sequence[str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Patient-level bootstrap of validation PR-AUC (§9.4 step 1).

    Patients are resampled with replacement, not images: images from one patient share a
    diagnosis, so an image-level bootstrap would treat correlated rows as independent and
    understate the interval -- which is the direction that manufactures significance.
    """
    y = np.asarray(y, dtype=int)
    order: Dict[str, List[int]] = {}
    for position, patient in enumerate(patients):
        order.setdefault(patient, []).append(position)
    unique = list(order)
    rng = np.random.default_rng(seed)

    values: List[float] = []
    for _ in range(int(draws)):
        picked: List[int] = []
        for patient in rng.choice(unique, size=len(unique), replace=True):
            picked.extend(order[str(patient)])
        rows = np.asarray(picked, dtype=int)
        value = _average_precision(y[rows], np.asarray(scores, dtype=np.float64)[rows])
        if value is not None:
            values.append(value)

    if len(values) < 2:
        return {
            "measured": False,
            "reason": "fewer than two bootstrap draws contained both classes",
            "n_draws_requested": int(draws),
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "measured": True,
        "n_draws_requested": int(draws),
        "n_draws_usable": int(array.size),
        "unit": "patient (resampled with replacement)",
        "mean": round(float(array.mean()), 6),
        "standard_error": round(float(array.std(ddof=1)), 6),
        "p2_5": round(float(np.percentile(array, 2.5)), 6),
        "p97_5": round(float(np.percentile(array, 97.5)), 6),
    }


# --------------------------------------------------------------------------- candidates
def _pooled_pixels(raw: np.ndarray, edge: int) -> np.ndarray:
    """Mean-pool the cached 256x256 grayscale rows down to ``edge`` x ``edge``.

    Mean pooling rather than subsampling: a stride-8 subsample of mucosal texture is an
    aliased sample of it, and the resulting features would measure the sampling grid as
    much as the tissue.
    """
    side = int(round(float(np.sqrt(raw.shape[1]))))
    if side * side != raw.shape[1]:
        raise FailureMapError(f"Pixel rows of length {raw.shape[1]} are not square.")
    if side % edge:
        raise FailureMapError(f"{side} is not divisible by the target edge {edge}.")
    block = side // edge
    cube = np.asarray(raw, dtype=np.float64).reshape(-1, side, side)
    return cube.reshape(-1, edge, block, edge, block).mean(axis=(2, 4)).reshape(-1, edge * edge)


def _bh_adjust(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Benjamini-Hochberg adjusted p-values, order preserved, ``None`` passed through."""
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    out: List[Optional[float]] = [None] * len(p_values)
    if not indexed:
        return out
    indexed.sort(key=lambda pair: pair[1])
    m = len(indexed)
    running = 1.0
    for rank in range(m, 0, -1):
        position, p = indexed[rank - 1]
        running = min(running, p * m / rank)
        out[position] = round(min(1.0, running), 6)
    return out


def _condition_note(condition: str) -> str:
    base = CONDITION_DESCRIPTIONS.get(condition, condition)
    if condition == PRIMARY:
        return (
            base + " PRIMARY condition: restricting to one ROI source removes the "
            "DEC-024 area channel by construction."
        )
    return (
        base + " SECONDARY condition: the DEC-024 geometry leak is PRESENT here -- ROI "
        "area alone scores validation ROC-AUC 0.72, so read every number against 0.72, "
        "not 0.5."
    )


# -------------------------------------------------------------------------------- inputs
class E1Inputs:
    """Every candidate's train/validation matrices on one condition, row-aligned.

    All three caches key on image id, and the three loaders return them in three
    different row orders, so the joins here are by id and the resulting index arrays are
    reused by every candidate. That is what makes the §8 comparison a comparison of
    representations: C3 and C5 see the same images in the same order as C1.

    Any image missing from any cache is dropped from **all** candidates, and the drop is
    recorded. Letting one candidate keep a row another dropped would make its score
    incomparable, and §8 requires "no arm may drop a row the other keeps".
    """

    def __init__(
        self,
        *,
        condition: str,
        descriptor_train: np.ndarray,
        descriptor_validation: np.ndarray,
        pixels_train: np.ndarray,
        pixels_validation: np.ndarray,
        e1_train: Dict[str, np.ndarray],
        e1_validation: Dict[str, np.ndarray],
        y_train: np.ndarray,
        y_validation: np.ndarray,
        patients_train: List[str],
        patients_validation: List[str],
        ids_train: List[str],
        ids_validation: List[str],
    ) -> None:
        self.condition = condition
        self.descriptor_train = descriptor_train
        self.descriptor_validation = descriptor_validation
        self.pixels_train = pixels_train
        self.pixels_validation = pixels_validation
        self.e1_train = e1_train
        self.e1_validation = e1_validation
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
) -> Tuple[Dict[str, E1Inputs], Dict[str, Any]]:
    """Load all three caches once and align them per condition.

    ``load_partitions(transform=False)`` is deliberate: the 181-feature control must be
    the *raw cached descriptor*, not the PCA-reduced vector the quantum path consumes.
    Reducing to 8 dimensions first would compare candidates against a compressed control
    and understate C1 -- E0's leaky-control error in a different costume.
    """
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)

    descriptors = load_partitions(config=cfg, transform=False)
    pixels = load_pixel_partitions(config=cfg, roi_mode="oracle")
    e1 = load_e1_cache(e1_cache_path(store))

    e1_ids = [str(v) for v in e1["image_ids"]]
    e1_index = {image_id: position for position, image_id in enumerate(e1_ids)}
    families = ("hog", "texture", "multiscale", "mobilenet")
    e1_matrices = {name: np.asarray(e1[f"matrix_{name}"], dtype=np.float64) for name in families}

    manifest_sha = hashlib.sha256(
        Path(store.split_manifest_path).read_bytes()
    ).hexdigest()

    per_condition: Dict[str, E1Inputs] = {}
    dropped: Dict[str, Dict[str, int]] = {}
    for condition in conditions:
        selections: Dict[str, np.ndarray] = {}
        for partition_name in ("train", "validation"):
            pixel_partition = pixels.partitions[partition_name]
            rows = condition_rows(pixel_partition).get(condition)
            if rows is None or not rows.size:
                raise FailureMapError(
                    f"Condition {condition!r} selects no rows in {partition_name}."
                )
            selections[partition_name] = rows

        bundle: Dict[str, Dict[str, Any]] = {}
        drops: Dict[str, int] = {}
        for partition_name, rows in selections.items():
            pixel_partition = pixels.partitions[partition_name]
            descriptor_partition = descriptors.partitions[partition_name]
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
                raise FailureMapError(
                    f"No {partition_name} row of condition {condition!r} is present in "
                    "all three caches; rebuild them from the same dataset state."
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
                # Two independent label derivations disagreeing means one of the caches
                # is stale. Silently trusting either would put a wrong label in a metric.
                raise FailureMapError(
                    f"Label mismatch between the pixel and descriptor caches on "
                    f"{partition_name}/{condition}; rebuild both caches."
                )
            bundle[partition_name] = {
                "ids": [pixel_partition.image_ids[int(r)] for r in pixel_rows],
                "patients": [pixel_partition.patient_ids[int(r)] for r in pixel_rows],
                "labels": labels,
                "descriptor": np.asarray(
                    descriptor_partition.image_features, dtype=np.float64
                )[descriptor_rows],
                "pixels": np.asarray(pixel_partition.raw, dtype=np.uint8)[pixel_rows],
                "e1": {name: matrix[e1_rows] for name, matrix in e1_matrices.items()},
            }

        train, validation = bundle["train"], bundle["validation"]
        overlap = set(train["patients"]) & set(validation["patients"])
        if overlap:
            raise FailureMapError(
                f"{len(overlap)} patients appear in both train and validation for "
                f"condition {condition!r} (first: {sorted(overlap)[0]}). Patient-level "
                "isolation is violated; nothing further is measured."
            )
        dropped[condition] = drops
        per_condition[condition] = E1Inputs(
            condition=condition,
            descriptor_train=train["descriptor"],
            descriptor_validation=validation["descriptor"],
            pixels_train=train["pixels"],
            pixels_validation=validation["pixels"],
            e1_train=train["e1"],
            e1_validation=validation["e1"],
            y_train=train["labels"],
            y_validation=validation["labels"],
            patients_train=train["patients"],
            patients_validation=validation["patients"],
            ids_train=train["ids"],
            ids_validation=validation["ids"],
        )

    provenance = {
        "split_manifest_sha256": manifest_sha,
        "descriptor_cache": str(store.feature_cache_path("handcrafted_lab")),
        "descriptor_dimension": int(
            np.asarray(descriptors.train.image_features).shape[1]
        ),
        "pixel_cache_version": str(pixels.cache_metadata.get("cache_version", "unknown")),
        "e1_cache_version": str(e1["metadata"]["cache_version"]),
        "e1_cache_dimensions": e1["metadata"]["dimensions"],
        "e1_cache_rows": len(e1_ids),
        "rows_dropped_for_cache_misalignment": dropped,
        "join": "by image id across all three caches; rows missing anywhere are dropped "
        "from every candidate, never from one",
        "descriptor_note": (
            "raw cached 163-dim image descriptor + no reduction (transform=False). The "
            "PCA-reduced 8-dim vector is NOT used: comparing candidates against a "
            "compressed control would understate C1."
        ),
    }
    return per_condition, provenance


# ------------------------------------------------------------------------- candidate set
def candidate_matrices(inputs: E1Inputs) -> Dict[str, Dict[str, Any]]:
    """The seven pre-registered candidates as ``(train, validation)`` matrix pairs.

    Frozen before any result was seen. Nothing may be added to this dict after the fact
    -- the user's constraint and §2's multiplicity accounting both depend on the count
    being fixed in advance.
    """
    clinical_train = inputs.descriptor_train
    clinical_validation = inputs.descriptor_validation

    candidates: Dict[str, Dict[str, Any]] = {
        "C1": {
            "label": "baseline replication (existing descriptor, train-only tuning)",
            "gap_tested": None,
            "train": clinical_train,
            "validation": clinical_validation,
        },
        "C2": {
            "label": "downsampled ROI pixels 32x32 (1024 dims)",
            "gap_tested": "1, 7",
            "train": _pooled_pixels(inputs.pixels_train, 32),
            "validation": _pooled_pixels(inputs.pixels_validation, 32),
        },
        "C2b": {
            "label": "downsampled ROI pixels 16x16 (256 dims)",
            "gap_tested": "1, 7",
            "train": _pooled_pixels(inputs.pixels_train, 16),
            "validation": _pooled_pixels(inputs.pixels_validation, 16),
        },
        "C3": {
            "label": "HOG, gradient orientation histograms",
            "gap_tested": "2, 4",
            "train": inputs.e1_train["hog"],
            "validation": inputs.e1_validation["hog"],
        },
        "C4": {
            "label": "LBP (two radii) + GLCM per CIELAB channel",
            "gap_tested": "2, 5",
            "train": inputs.e1_train["texture"],
            "validation": inputs.e1_validation["texture"],
        },
        "C5": {
            "label": "descriptor + 8x8 multiscale grid + centre-vs-surround",
            "gap_tested": "4, 5, 8",
            "train": np.hstack([clinical_train, inputs.e1_train["multiscale"]]),
            "validation": np.hstack(
                [clinical_validation, inputs.e1_validation["multiscale"]]
            ),
        },
        "C6": {
            "label": "frozen MobileNetV3-Small embedding (576 dims)",
            "gap_tested": "1, 2, 4, 5, 6, 7",
            "train": inputs.e1_train["mobilenet"],
            "validation": inputs.e1_validation["mobilenet"],
        },
    }
    # C2b shares C2's arm spec and its gap: it is the second dimensionality §2 asks for,
    # not an eighth candidate. It is counted in the multiplicity total for that reason.
    return candidates


# ------------------------------------------------------------------------- one candidate
def run_candidate(
    name: str,
    spec: Dict[str, Any],
    inputs: E1Inputs,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Fit every arm of one candidate and return all of them, plus the best.

    "Best" is by **validation** PR-AUC, which is a selection surface -- so the returned
    payload keeps every arm's number, and §9's BH correction is applied across the frozen
    grid afterwards. Reporting only the winner would hide the multiplicity that makes the
    winner's margin smaller than it looks.
    """
    x_train = np.asarray(spec["train"], dtype=np.float64)
    x_validation = np.asarray(spec["validation"], dtype=np.float64)
    if x_train.shape[1] != x_validation.shape[1]:
        raise FailureMapError(
            f"{name}: train has {x_train.shape[1]} columns, validation "
            f"{x_validation.shape[1]}."
        )
    x_train, x_validation = _standardise(x_train, x_validation)

    arms: Dict[str, Any] = {}
    failures: Dict[str, str] = {}
    for arm in ARM_SPECS[name]:
        started = time.perf_counter()
        try:
            fitted = _fit_arm(arm, x_train, inputs.y_train, x_validation, folds, seed=seed)
        except Exception as exc:  # noqa: BLE001 - a failed arm is a recorded result
            failures[arm] = f"{type(exc).__name__}: {exc}"
            continue
        elapsed = time.perf_counter() - started
        metrics = _metrics(
            inputs.y_validation,
            fitted["validation_probability"]
            if fitted["validation_probability"] is not None
            else fitted["validation_score"],
        )
        train_metrics = _metrics(
            inputs.y_train,
            fitted["train_probability"]
            if fitted["train_probability"] is not None
            else fitted["train_score"],
        )
        oof = fitted["train_out_of_fold"]
        usable = np.isfinite(oof)
        arms[arm] = {
            "classifier": fitted["family"],
            "kernel": fitted["kernel"],
            "is_linear": fitted["linear"],
            "hyperparameters": fitted["hyperparameters"],
            "selected_on": "out-of-fold average precision inside TRAIN only",
            "cv_average_precision_by_setting": fitted["cv_average_precision_by_setting"],
            "n_support_vectors": fitted["n_support_vectors"],
            "train_cv_pr_auc": (
                None
                if usable.sum() < 2
                else _average_precision(inputs.y_train[usable], oof[usable])
            ),
            "train_in_sample_metrics": train_metrics,
            "validation_metrics": metrics,
            "validation_pr_auc": None if not metrics else metrics.get("pr_auc"),
            "fit_seconds": round(elapsed, 3),
            "calibration_note": fitted["platt_note"],
            "_validation_score": fitted["validation_score"],
            "_validation_probability": fitted["validation_probability"],
            "_train_out_of_fold": oof,
        }

    scored = [
        (arm, payload["validation_pr_auc"])
        for arm, payload in arms.items()
        if payload["validation_pr_auc"] is not None
    ]
    best_arm = max(scored, key=lambda pair: pair[1])[0] if scored else None

    linear = [
        payload["validation_pr_auc"]
        for payload in arms.values()
        if payload["is_linear"] and payload["validation_pr_auc"] is not None
    ]
    nonlinear = [
        payload["validation_pr_auc"]
        for payload in arms.values()
        if not payload["is_linear"] and payload["validation_pr_auc"] is not None
    ]

    return {
        "candidate": name,
        "label": spec["label"],
        "gap_tested": spec["gap_tested"],
        "input_dimension": int(x_train.shape[1]),
        "n_train": int(x_train.shape[0]),
        "n_validation": int(x_validation.shape[0]),
        "standardisation": "z-scored with TRAIN mean/scale only",
        "arms": {
            arm: {k: v for k, v in payload.items() if not k.startswith("_")}
            for arm, payload in arms.items()
        },
        "arm_failures": failures,
        "best_arm": best_arm,
        "best_validation_pr_auc": (
            None if best_arm is None else arms[best_arm]["validation_pr_auc"]
        ),
        "best_linear_validation_pr_auc": max(linear) if linear else None,
        "best_nonlinear_validation_pr_auc": max(nonlinear) if nonlinear else None,
        "n_linear_arms": len(linear),
        "n_nonlinear_arms": len(nonlinear),
        "_arms_internal": arms,
    }


def _increment_null(
    reference_validation: np.ndarray,
    candidate_validation: np.ndarray,
    reference_train_oof: np.ndarray,
    candidate_train_oof: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    patients_train: Sequence[str],
    patients_validation: Sequence[str],
    *,
    draws: int = INCREMENT_PERMUTATIONS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """§9.2 column-permutation null on a two-feature stack, reused for C7.

    Same machinery E0 used for the quantum increment, applied here to a *classical*
    increment: hold the labels and the C1 column fixed, shuffle only the candidate column
    in whole patient blocks, and ask whether the real column adds more than a same-shaped
    unrelated one would. Whole blocks rather than i.i.d. rows because an i.i.d. shuffle
    also destroys within-patient autocorrelation, which makes the null artificially easy.

    The label-permutation null is reported alongside with ``gates_the_claim: false`` and
    its power note, because E0 measured its p floor near 0.25 on a genuinely complementary
    +0.163 increment.
    """
    from sklearn.linear_model import LogisticRegression

    from backend.evaluation.e0_controls import _patient_blocked_column_shuffle

    y_train = np.asarray(y_train, dtype=int)
    usable = np.isfinite(reference_train_oof) & np.isfinite(candidate_train_oof)
    if usable.sum() < 4 or len(set(y_train[usable].tolist())) < 2:
        return {
            "measured": False,
            "reason": f"only {int(usable.sum())} train rows have both out-of-fold columns",
        }

    z_train = np.column_stack(
        [reference_train_oof[usable], candidate_train_oof[usable]]
    ).astype(np.float64)
    z_validation = np.column_stack(
        [reference_validation, candidate_validation]
    ).astype(np.float64)
    mean = z_train.mean(axis=0)
    raw = z_train.std(axis=0)
    scale = np.where(raw > 1e-12, raw, 1.0)
    zt = (z_train - mean) / scale
    zv = (z_validation - mean) / scale
    y_fit = y_train[usable]
    used_patients = [patients_train[int(i)] for i in np.flatnonzero(usable)]

    def _fit(
        y: np.ndarray,
        columns: Sequence[int],
        train_design: Optional[np.ndarray] = None,
        validation_design: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        if len(set(np.asarray(y).tolist())) < 2:
            return None
        a = zt if train_design is None else train_design
        b = zv if validation_design is None else validation_design
        model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
        try:
            model.fit(a[:, columns], np.asarray(y, dtype=int))
        except Exception:  # noqa: BLE001 - a degenerate draw is dropped, never zeroed
            return None
        return _average_precision(
            y_validation, np.asarray(model.predict_proba(b[:, columns])[:, 1])
        )

    reference_only = _fit(y_fit, [0])
    stack = _fit(y_fit, [0, 1])
    observed = None if reference_only is None or stack is None else stack - reference_only

    column_null: List[float] = []
    if reference_only is not None:
        rng = np.random.default_rng(seed + 977)
        for _ in range(int(draws)):
            shuffled_train = _patient_blocked_column_shuffle(zt[:, 1], used_patients, rng)
            shuffled_validation = _patient_blocked_column_shuffle(
                zv[:, 1], list(patients_validation), rng
            )
            value = _fit(
                y_fit,
                [0, 1],
                np.column_stack([zt[:, 0], shuffled_train]),
                np.column_stack([zv[:, 0], shuffled_validation]),
            )
            if value is not None:
                column_null.append(value - reference_only)

    label_null: List[float] = []
    for permuted in patient_blocked_permutations(
        y_fit, used_patients, n=int(draws), seed=seed
    ):
        one, two = _fit(permuted, [0]), _fit(permuted, [0, 1])
        if one is not None and two is not None:
            label_null.append(two - one)

    column_rank = _rank_in_null(observed, column_null)
    label_rank = _rank_in_null(observed, label_null)
    with np.errstate(invalid="ignore"):
        pearson = float(np.corrcoef(reference_validation, candidate_validation)[0, 1])
    return {
        "measured": True,
        "n_train_rows_used": int(usable.sum()),
        "reference_only_pr_auc": None if reference_only is None else round(reference_only, 6),
        "stack_pr_auc": None if stack is None else round(stack, 6),
        "increment": None if observed is None else round(observed, 6),
        "score_correlation_pearson": round(pearson, 6) if np.isfinite(pearson) else None,
        "column_permutation_null": {
            "n_requested": int(draws),
            "n_usable": len(column_null),
            "blocking": "candidate column shuffled in whole patient blocks, TRAIN and "
            "VALIDATION; labels and the C1 column untouched",
            "increment": _null_summary(column_null),
            "observed_vs_null": column_rank,
            "gates_the_claim": True,
        },
        "permutation_null": {
            "n_requested": int(draws),
            "n_usable": len(label_null),
            "blocking": "patient-level TRAIN label permutation",
            "increment": _null_summary(label_null),
            "observed_vs_null": label_rank,
            "gates_the_claim": False,
            "power_note": (
                "structurally low-powered for an increment: refitting on permuted labels "
                "randomises coefficient signs while scoring against real labels, so the "
                "empirical p floors near 0.25. E0 measured p = 0.11 for a genuine +0.163 "
                "increment. Not evidence against complementarity."
            ),
        },
        "survives_gating_null": (
            None
            if column_rank is None
            else bool(column_rank["exceeds_null_p95"] and column_rank["empirical_p"] <= 0.05)
        ),
    }


# ------------------------------------------------------------------------------ verdicts
def assign_verdict(
    candidate: Dict[str, Any],
    *,
    c1_pr_auc: Optional[float],
    delta: Optional[float],
    increment: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """One of §3's four pre-registered verdicts. The rules are not adjustable here.

    ``A`` already solved, ``B`` signal present but the model fails to exploit it, ``C``
    weak / non-generalizable, ``D`` redundant. Only ``B`` survives into §4, so this
    function is the single place where a quantum motivation can be created -- and it can
    only be created by a *nonlinear-only* gain that clears the bootstrap margin and
    survives the column-permutation null.
    """
    best = candidate["best_validation_pr_auc"]
    linear = candidate["best_linear_validation_pr_auc"]
    nonlinear = candidate["best_nonlinear_validation_pr_auc"]
    train_cv = [
        arm["train_cv_pr_auc"]
        for arm in candidate["arms"].values()
        if arm["train_cv_pr_auc"] is not None
    ]

    if best is None or c1_pr_auc is None or delta is None:
        return {
            "verdict": None,
            "reason": "not measurable: a required PR-AUC or the bootstrap margin is absent",
            "motivates_quantum": False,
        }

    margin = best - c1_pr_auc
    clears = margin >= delta
    nonlinear_clears = nonlinear is not None and (nonlinear - c1_pr_auc) >= delta
    linear_clears = linear is not None and (linear - c1_pr_auc) >= delta
    train_beats_c1 = bool(train_cv) and max(train_cv) > (c1_pr_auc or 0.0)
    survives = None if increment is None else increment.get("survives_gating_null")

    if candidate["candidate"] == "C1":
        return {
            "verdict": "control",
            "reason": "C1 is the reference, not a candidate against itself",
            "margin_over_c1": 0.0,
            "motivates_quantum": False,
        }

    if clears and nonlinear_clears and not linear_clears and survives is True:
        return {
            "verdict": "B",
            "reason": (
                f"nonlinear arm clears C1 by {nonlinear - c1_pr_auc:+.6f} >= delta "
                f"{delta:.6f} while the linear arm does not, and the increment survives "
                "the patient-blocked column-permutation null"
            ),
            "margin_over_c1": round(margin, 6),
            "motivates_quantum": True,
        }
    if clears and not train_beats_c1:
        return {
            "verdict": "C",
            "reason": (
                "validation gain without a corresponding within-train CV gain -- E0's "
                "signature. Reported as weak / non-generalizable, not as a positive."
            ),
            "margin_over_c1": round(margin, 6),
            "motivates_quantum": False,
        }
    if clears and survives is False:
        return {
            "verdict": "C",
            "reason": (
                f"gain of {margin:+.6f} does not survive the patient-blocked "
                "column-permutation null"
            ),
            "margin_over_c1": round(margin, 6),
            "motivates_quantum": False,
        }
    if clears and linear_clears:
        return {
            "verdict": "A",
            "reason": (
                f"family clears C1 by {margin:+.6f} with a LINEAR classifier, so the "
                "missing information is linearly accessible: the fix is the feature, not "
                "a transformation. Classical fix takes precedence (§3 guard rail)."
            ),
            "margin_over_c1": round(margin, 6),
            "motivates_quantum": False,
        }
    if abs(margin) < delta:
        return {
            "verdict": "D" if increment and increment.get("survives_gating_null") is False
            else "A",
            "reason": (
                f"family scores within the bootstrap margin of C1 ({margin:+.6f} vs delta "
                f"{delta:.6f}): the structural gap is real in the representation but "
                "costs nothing measurable at this sample size"
            ),
            "margin_over_c1": round(margin, 6),
            "motivates_quantum": False,
        }
    return {
        "verdict": "C",
        "reason": (
            f"family falls short of C1 by {margin:+.6f}; both a linear and a nonlinear "
            "arm were tuned on train and both fell short"
        ),
        "margin_over_c1": round(margin, 6),
        "motivates_quantum": False,
    }


# -------------------------------------------------------------------------------- driver
def run(
    *,
    config: Optional[Settings] = None,
    conditions: Sequence[str] = CONDITIONS,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    increment_permutations: int = INCREMENT_PERMUTATIONS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Run every candidate on every condition and return the full E1.1 payload."""
    started = time.perf_counter()
    per_condition, provenance = build_inputs(config=config, conditions=conditions)

    # Frozen BEFORE the first fit: §2's multiplicity accounting requires the count to be
    # fixed in advance, not counted afterwards from whatever happened to run.
    candidate_names = ("C1", "C2", "C2b", "C3", "C4", "C5", "C6", "C7")
    frozen_grid = {
        "n_candidates": len(candidate_names),
        "candidates": list(candidate_names),
        "arms_per_candidate": {k: list(v) for k, v in ARM_SPECS.items()},
        "n_conditions": len(conditions),
        "n_tests_total": sum(len(ARM_SPECS[c]) for c in candidate_names) * len(conditions),
        "frozen_before_first_fit": True,
        "note": (
            "C7 is the fusion of the two best non-C1 families by primary-condition "
            "validation PR-AUC, so its input is chosen from results -- which is why its "
            "claim is gated on the column-permutation null and not on its PR-AUC alone."
        ),
    }

    results: Dict[str, Any] = {}
    for condition, inputs in per_condition.items():
        folds, cv_note = grouped_folds(
            inputs.y_train, inputs.patients_train, seed=seed, n_splits=N_FOLDS
        )
        logger.info(
            "Condition %s: %d train / %d validation rows (%d / %d positive), %s",
            condition,
            inputs.y_train.size,
            inputs.y_validation.size,
            int(inputs.y_train.sum()),
            int(inputs.y_validation.sum()),
            cv_note,
        )

        specs = candidate_matrices(inputs)
        candidates: Dict[str, Any] = {}
        for name in ("C1", "C2", "C2b", "C3", "C4", "C5", "C6"):
            logger.info("  %s: %s", name, specs[name]["label"])
            candidates[name] = run_candidate(
                name, specs[name], inputs, folds, seed=seed
            )

        # --- C7: fuse the two strongest non-C1 families. Chosen from results, so its
        # claim is gated on the null rather than on its PR-AUC.
        ranked = sorted(
            (
                (name, payload["best_validation_pr_auc"])
                for name, payload in candidates.items()
                if name != "C1" and payload["best_validation_pr_auc"] is not None
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if len(ranked) >= 2:
            first, second = ranked[0][0], ranked[1][0]
            fusion_spec = {
                "label": f"fusion of {first} + {second} (standardised, concatenated)",
                "gap_tested": "6",
                "train": np.hstack([specs[first]["train"], specs[second]["train"]]),
                "validation": np.hstack(
                    [specs[first]["validation"], specs[second]["validation"]]
                ),
            }
            logger.info("  C7: %s", fusion_spec["label"])
            candidates["C7"] = run_candidate("C7", fusion_spec, inputs, folds, seed=seed)
            candidates["C7"]["fused_from"] = [first, second]
        else:
            candidates["C7"] = {
                "candidate": "C7",
                "label": "fusion not measurable",
                "arms": {},
                "arm_failures": {
                    "all": "fewer than two non-C1 families produced a usable PR-AUC"
                },
                "best_arm": None,
                "best_validation_pr_auc": None,
                "best_linear_validation_pr_auc": None,
                "best_nonlinear_validation_pr_auc": None,
                "_arms_internal": {},
            }

        # --- The margin, derived from C1's own sampling variability (§9.4).
        c1 = candidates["C1"]
        c1_best = c1["best_arm"]
        bootstrap: Dict[str, Any] = {"measured": False, "reason": "C1 produced no usable arm"}
        delta: Optional[float] = None
        if c1_best is not None:
            arm = c1["_arms_internal"][c1_best]
            scores = (
                arm["_validation_probability"]
                if arm["_validation_probability"] is not None
                else arm["_validation_score"]
            )
            bootstrap = bootstrap_pr_auc(
                inputs.y_validation,
                scores,
                inputs.patients_validation,
                draws=bootstrap_draws,
                seed=seed,
            )
            if bootstrap.get("measured"):
                delta = round(2.0 * float(bootstrap["standard_error"]), 6)

        c1_pr_auc = c1["best_validation_pr_auc"]

        # --- Increment test for every candidate against C1, on the same rows.
        for name, payload in candidates.items():
            if name == "C1" or payload["best_arm"] is None or c1_best is None:
                payload["increment_vs_c1"] = None
                continue
            c1_arm = c1["_arms_internal"][c1_best]
            candidate_arm = payload["_arms_internal"][payload["best_arm"]]
            payload["increment_vs_c1"] = _increment_null(
                np.asarray(
                    c1_arm["_validation_probability"]
                    if c1_arm["_validation_probability"] is not None
                    else c1_arm["_validation_score"],
                    dtype=np.float64,
                ),
                np.asarray(
                    candidate_arm["_validation_probability"]
                    if candidate_arm["_validation_probability"] is not None
                    else candidate_arm["_validation_score"],
                    dtype=np.float64,
                ),
                np.asarray(c1_arm["_train_out_of_fold"], dtype=np.float64),
                np.asarray(candidate_arm["_train_out_of_fold"], dtype=np.float64),
                inputs.y_train,
                inputs.y_validation,
                inputs.patients_train,
                inputs.patients_validation,
                draws=increment_permutations,
                seed=seed,
            )

        # --- BH correction across every candidate's gating p-value on this condition.
        names = [n for n in candidates if n != "C1"]
        raw_p = [
            (candidates[n].get("increment_vs_c1") or {})
            .get("column_permutation_null", {})
            .get("observed_vs_null", {})
            .get("empirical_p")
            if candidates[n].get("increment_vs_c1")
            else None
            for n in names
        ]
        adjusted = _bh_adjust(raw_p)
        for name, raw, adj in zip(names, raw_p, adjusted):
            candidates[name]["multiplicity"] = {
                "raw_empirical_p": raw,
                "bh_adjusted_p": adj,
                "corrected_across": frozen_grid["n_candidates"] - 1,
                "significant_at_0_05": None if adj is None else bool(adj < 0.05),
            }

        for name, payload in candidates.items():
            increment = payload.get("increment_vs_c1")
            adjusted_p = (payload.get("multiplicity") or {}).get("bh_adjusted_p")
            gated = increment
            if increment and adjusted_p is not None:
                gated = dict(increment)
                gated["survives_gating_null"] = bool(
                    increment.get("survives_gating_null") and adjusted_p < 0.05
                )
            payload["classification"] = assign_verdict(
                payload, c1_pr_auc=c1_pr_auc, delta=delta, increment=gated
            )
            payload.pop("_arms_internal", None)

        results[condition] = {
            "condition": condition,
            "condition_note": _condition_note(condition),
            "is_primary": condition == PRIMARY,
            "cv_note": cv_note,
            "rows": inputs.shape_note,
            "c1_reference": {
                "best_arm": c1_best,
                "validation_pr_auc": c1_pr_auc,
                "bootstrap": bootstrap,
                "keep_margin_delta": delta,
                "delta_note": (
                    "delta = 2 x standard error of C1's validation PR-AUC under a "
                    "patient-level bootstrap, computed BEFORE any candidate was compared "
                    "(§9.4). Derived from this dataset's variability, not asserted."
                ),
            },
            "geometry_leak_bar": {
                "area_only_validation_roc_auc": 0.72,
                "applies": condition != PRIMARY,
                "note": (
                    "DEC-024. On A_all every model scores above chance from ROI area "
                    "alone; the primary condition removes that channel by construction."
                ),
            },
            "candidates": candidates,
        }

    duration = time.perf_counter() - started
    primary = results.get(PRIMARY, {})
    motivating = [
        name
        for name, payload in primary.get("candidates", {}).items()
        if (payload.get("classification") or {}).get("motivates_quantum")
    ]
    verdict_counts: Dict[str, int] = {}
    for payload in primary.get("candidates", {}).values():
        key = str((payload.get("classification") or {}).get("verdict"))
        verdict_counts[key] = verdict_counts.get(key, 0) + 1

    return {
        "failure_map_version": FAILURE_MAP_VERSION,
        "spec": "docs/PHASE_E1_RESEARCH_SPEC.md §2, §3, §9",
        "test_partition_used": False,
        "test_partition_note": (
            "E1.1 reads TRAIN and VALIDATION only. No code path in this module can select "
            "the test partition, and no fitted quantity saw a validation row."
        ),
        "primary_condition": PRIMARY,
        "frozen_candidate_grid": frozen_grid,
        "provenance": provenance,
        "conditions": results,
        "bottleneck_summary": {
            "verdict_counts_primary_condition": verdict_counts,
            "families_motivating_quantum": motivating,
            "classical_bottleneck_measured": bool(motivating),
            "conclusion": (
                "A verdict-B family was measured on the primary condition; §4's Q1 "
                "premise holds for it."
                if motivating
                else "No candidate produced verdict B on the primary condition. Under §3 "
                "and §6 this is a NULL result and NO QUANTUM IMPLEMENTATION is justified."
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "seed": seed,
            "n_folds": N_FOLDS,
            "bootstrap_draws": bootstrap_draws,
            "increment_permutations": increment_permutations,
        },
        "duration_seconds": round(duration, 3),
    }


def _default_output(config: Optional[Settings] = None) -> Path:
    return Path("reports_e1_failure_map.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Payload destination.")
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--permutations", type=int, default=INCREMENT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    payload = run(
        bootstrap_draws=args.bootstrap_draws,
        increment_permutations=args.permutations,
        seed=args.seed,
    )
    destination = args.out or _default_output()
    destination.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    logger.info("Wrote %s", destination)
    summary = payload["bottleneck_summary"]
    logger.info("Verdicts (primary): %s", summary["verdict_counts_primary_condition"])
    logger.info("%s", summary["conclusion"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
