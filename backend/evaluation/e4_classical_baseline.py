"""E4.1: the classical failure map on PTB-XL -- the gate on every quantum experiment.

``docs/PHASE_E4_DATA_EXPANSION.md`` §8 item 5 makes this module a precondition, not a
milestone: *no quantum model may be fitted on PTB-XL until the classical ceiling is
measured.* E3 produced four honest nulls against a single classical number (C7 PR-AUC
0.913038) and the nulls were trustworthy but uninformative, because nothing established
what the classical ceiling was in the *compact* regime the circuits actually consumed.
This module measures that ceiling, at every dimension a near-term circuit could plausibly
take, before any circuit exists.

What is measured, and why each cell is here
-------------------------------------------
A model grid crossed with a representation grid, both frozen in this file before the first
fit:

=========  ===============================================================  ==============
``f97``    all 97 ``ecg-v1`` features, imputed and standardised             the ceiling
``pca32``  TRAIN-fitted PCA, 92.8% of the representation's variance         matched control
``pca16``  TRAIN-fitted PCA, 77.5%                                         matched control
``pca08``  TRAIN-fitted PCA, 60.0%                                         matched control
=========  ===============================================================  ==============

The three compact rows exist **only** because §7.4 condition 3 demands them: a quantum
model that consumes an 8-dimensional projection and beats a 97-feature classical model has
proved nothing about quantum computation until the 8-dimensional *classical* control has
been run on the same projection with the same fitting budget. Running those controls now,
before any quantum result could bias the choice of grid, is the whole point. If a circuit
later beats ``pca08`` classical arms, that is a measurement; if it only beats ``f97``, it
is a dimensionality artefact with a quantum label on it.

Three models per representation, for E1's reason (§3 guard rail 1): a single weak
classifier is not a bottleneck, so "classical fails here" must survive a linear model, a
modern gradient-boosted ensemble and a kernel machine. The kernel arm is also the specific
classical control for any future fidelity-kernel / QSVM experiment.

Discipline
----------
* **Folds 1-8 fit, fold 9 scores, fold 10 is never read.** There is no code path in this
  module that can select the test partition: fold-10 rows are refused on load, and the
  fold-10 signal files are not on this machine at all (§7.2).
* **Every hyperparameter, threshold and calibration comes from inside TRAIN.** Selection
  uses leave-one-``strat_fold``-out cross-validation *within folds 1-8*, which is
  patient-disjoint by construction (§7.1) -- PTB-XL's own folds are reused rather than a
  fresh grouping invented here, so the inner split carries the same verified guarantee as
  the outer one.
* **An unknown label is never a negative.** Records with no diagnostic superclass are
  excluded from fitting and from every metric, and are reported separately (§7.4).
* **Bootstrap resampling is over patients, not records.** Patients contribute multiple
  ECGs; resampling records would treat correlated rows as independent and understate the
  interval, which is the direction that manufactures significance.
* The secondary ``MI`` vs ``NORM`` task is reported and is **never** used for selection;
  every payload entry carries ``used_for_selection: false``.

Usage::

    python -m backend.evaluation.e4_classical_baseline \\
        --cache backend/artifacts/dataset/ptbxl/features_ecg-v1.npz \\
        --out backend/artifacts/reports/e4_classical_baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from backend.dataset.ptbxl import TEST_FOLDS, TRAIN_FOLDS, VALIDATION_FOLDS
from backend.evaluation.metrics import evaluate_predictions, sensitivity_at_specificity
from backend.ml.ecg_transform import (
    EcgFeatureTransform,
    assert_no_fit_eval_patient_overlap,
    fit_ecg_transform,
)
from backend.training.prepare_ecg_features import (
    ECG_CACHE_VERSION,
    FEATURE_SET_VERSION,
    LABEL_UNKNOWN,
    EcgCohort,
    load_cohort,
)

logger = logging.getLogger(__name__)

E4_BASELINE_VERSION = "v1-e4-classical-1"

#: Bootstrap draws over patients. §7.4 fixes 2,000 and percentile 95% CIs.
BOOTSTRAP_DRAWS = 2000

#: Label-permutation draws for the pipeline null (§7.4 condition 4). Frozen here so the
#: count cannot be chosen after seeing a p-value.
PERMUTATION_DRAWS = 200

#: §7.4's secondary operating point: "if we accept flagging 10% of healthy people, what
#: fraction of abnormal ECGs do we catch?"
FIXED_SPECIFICITY = 0.90

SEED = 42

#: The representation grid: ``(name, reduction, n_components)``. ``identity`` keeps every
#: feature that survives the coverage and variance filters; the PCA rows are §7.4
#: condition 3's matched classical controls, at the dimensions a near-term circuit could
#: plausibly encode.
REPRESENTATIONS: Tuple[Tuple[str, str, Optional[int]], ...] = (
    ("f97", "identity", None),
    ("pca32", "pca", 32),
    ("pca16", "pca", 16),
    ("pca08", "pca", 8),
)

#: Model grids. Deliberately small: with 17,084 labeled TRAIN records a large search would
#: be a selection surface rather than a measurement (E1 §2), and every configuration here
#: is evaluated by 8-fold inner CV, so the grid size multiplies the fitting budget eight
#: times over.
MODEL_GRIDS: Dict[str, Tuple[Dict[str, Any], ...]] = {
    "logistic": (
        {"C": 0.01},
        {"C": 0.1},
        {"C": 1.0},
        {"C": 10.0},
    ),
    "gbm": (
        {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31},
        {"learning_rate": 0.10, "max_iter": 200, "max_leaf_nodes": 31},
        {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 15},
    ),
    "rbf_svm": (
        {"C": 1.0},
        {"C": 10.0},
    ),
}

MODELS: Tuple[str, ...] = tuple(MODEL_GRIDS)

#: A floor, not a competitor. An arm that cannot beat the prevalence predictor is broken,
#: and its ROC-AUC of exactly 0.5 is the cheapest available check that the label vector and
#: the score vector are aligned.
TRIVIAL_ARM = "prevalence"

SUPERCLASS_ABNORMAL: Tuple[str, ...] = ("MI", "STTC", "CD", "HYP")


class ClassicalBaselineError(RuntimeError):
    """Raised when a baseline cannot be measured on honestly partitioned rows."""


# --------------------------------------------------------------------------- metrics
def roc_auc(y: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Rank-based ROC-AUC, ``None`` when the sample holds a single class.

    The Mann-Whitney form with tie-averaged ranks, rather than a trapezoid over a sampled
    curve: this is called ~2,000 times per arm inside the bootstrap, and returning ``None``
    rather than 0.5 keeps an undefined value from reading as a measured one (the same
    convention as :mod:`backend.evaluation.metrics`).
    """
    y_array = np.asarray(y, dtype=int)
    score_array = np.asarray(scores, dtype=np.float64)
    if y_array.size != score_array.size:
        raise ClassicalBaselineError(
            f"{y_array.size} labels against {score_array.size} scores."
        )
    n_positive = int(np.count_nonzero(y_array == 1))
    n_negative = int(np.count_nonzero(y_array == 0))
    if not n_positive or not n_negative:
        return None
    ranks = rankdata(score_array)
    positive_rank_sum = float(ranks[y_array == 1].sum())
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )


def _patient_row_index(patients: Sequence[int]) -> Tuple[List[int], List[np.ndarray]]:
    """Group row positions by patient, preserving first-appearance order."""
    order: Dict[int, List[int]] = {}
    for position, patient in enumerate(patients):
        order.setdefault(int(patient), []).append(position)
    keys = list(order)
    return keys, [np.asarray(order[key], dtype=int) for key in keys]


def _resample_rows(
    blocks: Sequence[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """One patient-level bootstrap draw: whole patients with replacement."""
    picked = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[index] for index in picked])


def bootstrap_roc_auc(
    y: np.ndarray,
    scores: np.ndarray,
    patients: Sequence[int],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Patient-level bootstrap of validation ROC-AUC (§7.4's primary metric).

    Patients are resampled with replacement, not records: one patient contributes several
    ECGs that share a diagnosis, so a record-level bootstrap would count correlated rows as
    independent evidence and return an interval that is too narrow.
    """
    y_array = np.asarray(y, dtype=int)
    score_array = np.asarray(scores, dtype=np.float64)
    _, blocks = _patient_row_index(patients)
    rng = np.random.default_rng(seed)

    values: List[float] = []
    for _ in range(int(draws)):
        rows = _resample_rows(blocks, rng)
        value = roc_auc(y_array[rows], score_array[rows])
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


def paired_bootstrap_delta(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    patients: Sequence[int],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Paired patient-level bootstrap of ``ROC-AUC(a) - ROC-AUC(b)`` (§7.4 condition 2).

    Both models are scored on the **same** resampled patients in every draw. Two
    independent bootstraps would fold each model's sampling variance into the difference
    and widen the interval by roughly a factor of sqrt(2) -- which here would hide a real
    difference rather than invent one, but is wrong in either direction. This is the exact
    machinery a future quantum-vs-classical comparison has to clear, exercised now between
    classical arms so it is not written for the first time when something is at stake.
    """
    y_array = np.asarray(y, dtype=int)
    a_array = np.asarray(scores_a, dtype=np.float64)
    b_array = np.asarray(scores_b, dtype=np.float64)
    _, blocks = _patient_row_index(patients)
    rng = np.random.default_rng(seed)

    deltas: List[float] = []
    for _ in range(int(draws)):
        rows = _resample_rows(blocks, rng)
        auc_a = roc_auc(y_array[rows], a_array[rows])
        auc_b = roc_auc(y_array[rows], b_array[rows])
        if auc_a is not None and auc_b is not None:
            deltas.append(auc_a - auc_b)

    if len(deltas) < 2:
        return {
            "measured": False,
            "reason": "fewer than two paired draws contained both classes",
            "n_draws_requested": int(draws),
        }
    array = np.asarray(deltas, dtype=np.float64)
    low = float(np.percentile(array, 2.5))
    high = float(np.percentile(array, 97.5))
    return {
        "measured": True,
        "n_draws_requested": int(draws),
        "n_draws_usable": int(array.size),
        "unit": "patient (paired, same resamples for both models)",
        "observed_delta": round(float(roc_auc(y_array, a_array) - roc_auc(y_array, b_array)), 6),
        "mean_delta": round(float(array.mean()), 6),
        "standard_error": round(float(array.std(ddof=1)), 6),
        "p2_5": round(low, 6),
        "p97_5": round(high, 6),
        "excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def _balanced_accuracy_curve(
    y: np.ndarray, scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Balanced accuracy at every threshold that changes the confusion matrix.

    Vectorised over the sorted scores rather than looped: threshold selection runs on the
    17,084 TRAIN out-of-fold predictions, where a per-candidate confusion matrix would mean
    17,084 passes over the vector.
    """
    order = np.argsort(scores, kind="mergesort")[::-1]
    y_sorted = np.asarray(y, dtype=int)[order]
    score_sorted = np.asarray(scores, dtype=np.float64)[order]
    n_positive = int(np.count_nonzero(y_sorted == 1))
    n_negative = int(np.count_nonzero(y_sorted == 0))
    if not n_positive or not n_negative:
        raise ClassicalBaselineError("Threshold selection needs both classes.")

    true_positive = np.cumsum(y_sorted == 1)
    false_positive = np.cumsum(y_sorted == 0)
    # Only the last row of each run of equal scores is a distinct threshold.
    distinct = np.r_[np.diff(score_sorted) != 0, True]
    sensitivity = true_positive[distinct] / n_positive
    specificity = 1.0 - false_positive[distinct] / n_negative
    return score_sorted[distinct], (sensitivity + specificity) / 2.0


def select_threshold(y: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    """The threshold maximising balanced accuracy, chosen on TRAIN rows only.

    §7.4 reports balanced accuracy "at a TRAIN-selected threshold". The threshold is taken
    from the out-of-fold TRAIN predictions -- not from in-sample TRAIN predictions, whose
    score distribution is shifted by the model having memorised those rows, and not from
    fold 9, which would make every fold-9 number partly self-selected.
    """
    thresholds, balanced = _balanced_accuracy_curve(y, scores)
    best = int(np.argmax(balanced))
    return {
        "threshold": round(float(thresholds[best]), 6),
        "train_out_of_fold_balanced_accuracy": round(float(balanced[best]), 6),
        "selected_on": "TRAIN folds 1-8, out-of-fold predictions",
    }


# ------------------------------------------------------------------------ estimators
def build_estimator(model: str, params: Dict[str, Any], *, seed: int = SEED) -> Any:
    """Instantiate one grid cell. The only place a model name becomes an object."""
    if model == "logistic":
        return LogisticRegression(
            C=float(params["C"]), max_iter=2000, solver="lbfgs", random_state=seed
        )
    if model == "gbm":
        return HistGradientBoostingClassifier(
            learning_rate=float(params["learning_rate"]),
            max_iter=int(params["max_iter"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            early_stopping=False,
            random_state=seed,
        )
    if model == "rbf_svm":
        # No ``probability=True``: it is deprecated in scikit-learn 1.9, and its internal
        # cross-validated Platt fit would refit the SVM five more times per cell. The
        # decision function is calibrated below from the TRAIN out-of-fold values instead,
        # which is both cheaper and fitted in the partition where fitting is allowed.
        return SVC(
            C=float(params["C"]),
            kernel="rbf",
            gamma="scale",
            cache_size=512,
            random_state=seed,
        )
    raise ClassicalBaselineError(f"Unknown model arm {model!r}.")


def raw_scores(estimator: Any, features: np.ndarray) -> np.ndarray:
    """A higher-is-more-positive score, probability where the estimator offers one."""
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(features), dtype=np.float64)[:, 1]
    return np.asarray(estimator.decision_function(features), dtype=np.float64)


def emits_probability(model: str) -> bool:
    return model in ("logistic", "gbm")


def _platt_calibrator(
    train_raw: np.ndarray, y_train: np.ndarray, *, seed: int = SEED
) -> Any:
    """A 1-D logistic map from decision value to probability, fitted on TRAIN OOF values.

    Being monotone, it cannot change ROC-AUC or PR-AUC by even a rounding step -- it exists
    so that Brier score, log loss, ECE and the selected threshold are defined for the
    kernel arm, and so the reported probabilities mean something. Fitted on out-of-fold
    TRAIN decision values, which is the only partition where fitting anything is allowed.
    """
    calibrator = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    calibrator.fit(np.asarray(train_raw, dtype=np.float64).reshape(-1, 1), y_train)
    return calibrator


def _apply_calibrator(calibrator: Optional[Any], values: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.asarray(values, dtype=np.float64)
    return np.asarray(
        calibrator.predict_proba(np.asarray(values, dtype=np.float64).reshape(-1, 1)),
        dtype=np.float64,
    )[:, 1]


# --------------------------------------------------------------------------- partitions
@dataclass(frozen=True)
class Partition:
    """One partition's labeled rows, already reduced to the arrays a model consumes."""

    name: str
    folds: Tuple[int, ...]
    features: np.ndarray
    y: np.ndarray
    patients: np.ndarray
    ecg_ids: np.ndarray
    strat_folds: np.ndarray
    superclasses: np.ndarray
    ages: np.ndarray
    sexes: np.ndarray
    n_beats: np.ndarray
    quality_flagged: np.ndarray

    @property
    def n_records(self) -> int:
        return int(self.y.size)

    @property
    def n_patients(self) -> int:
        return int(np.unique(self.patients).size)

    def summary(self) -> Dict[str, Any]:
        return {
            "partition": self.name,
            "strat_folds": list(self.folds),
            "n_records": self.n_records,
            "n_patients": self.n_patients,
            "n_positive": int(np.count_nonzero(self.y == 1)),
            "n_negative": int(np.count_nonzero(self.y == 0)),
            "prevalence": round(float(np.mean(self.y == 1)), 6),
        }


def primary_labels(cohort: EcgCohort) -> np.ndarray:
    """§7.4's primary task, taken from the cache rather than recomputed.

    The cohort builder already applied the frozen rule (any of MI/STTC/CD/HYP -> positive;
    NORM and none of them -> negative; no diagnostic superclass -> unknown). Recomputing it
    here would create a second definition of the primary task that could silently drift
    from the pre-registered one.
    """
    return np.asarray(cohort.labels, dtype=np.int8)


def secondary_labels(cohort: EcgCohort) -> np.ndarray:
    """§7.4's secondary ``MI`` vs ``NORM`` task. Reported, never used for selection.

    ``MI`` present -> positive, including when ``NORM`` is also present: the
    screening-conservative direction, identical to the primary rule. ``NORM`` with no
    abnormal superclass -> negative. Everything else -- notably a record whose only
    abnormality is STTC, CD or HYP -- is *unknown here*, because it is neither an MI nor a
    normal ECG, and folding it into either class would be a different task than the one
    that was pre-registered.
    """
    columns = {name: index for index, name in enumerate(cohort.superclass_order)}
    missing = [name for name in ("NORM",) + SUPERCLASS_ABNORMAL if name not in columns]
    if missing:
        raise ClassicalBaselineError(f"Cohort is missing superclasses {missing}.")
    present = np.asarray(cohort.superclasses, dtype=bool)
    infarction = present[:, columns["MI"]]
    any_abnormal = present[:, [columns[name] for name in SUPERCLASS_ABNORMAL]].any(axis=1)
    normal = present[:, columns["NORM"]] & ~any_abnormal
    labels = np.full(present.shape[0], LABEL_UNKNOWN, dtype=np.int8)
    labels[normal] = 0
    labels[infarction] = 1
    return labels


TASKS: Dict[str, Any] = {
    "primary_norm_vs_abnormal": primary_labels,
    "secondary_mi_vs_norm": secondary_labels,
}


def build_partition(
    cohort: EcgCohort,
    transform: EcgFeatureTransform,
    labels: np.ndarray,
    folds: Sequence[int],
    name: str,
) -> Partition:
    """Reduce a cohort to one partition's *labeled* rows under one representation."""
    fold_array = np.asarray(cohort.strat_folds, dtype=int)
    mask = np.isin(fold_array, list(folds)) & (np.asarray(labels) != LABEL_UNKNOWN)
    if not mask.any():
        raise ClassicalBaselineError(f"No labeled records in folds {sorted(folds)}.")
    features = transform.transform(np.asarray(cohort.features, dtype=np.float64)[mask])
    if not np.isfinite(features).all():
        raise ClassicalBaselineError(
            f"Non-finite values reached the model matrix for partition {name!r}."
        )
    flags = np.asarray(
        [bool(str(flag).strip()) for flag in cohort.quality_flags], dtype=bool
    )
    return Partition(
        name=name,
        folds=tuple(sorted(folds)),
        features=features,
        y=np.asarray(labels, dtype=int)[mask],
        patients=np.asarray(cohort.patient_ids, dtype=np.int64)[mask],
        ecg_ids=np.asarray(cohort.ecg_ids, dtype=np.int64)[mask],
        strat_folds=fold_array[mask],
        superclasses=np.asarray(cohort.superclasses, dtype=np.uint8)[mask],
        ages=np.asarray(cohort.ages, dtype=np.float64)[mask],
        sexes=np.asarray(cohort.sexes, dtype=int)[mask],
        n_beats=np.asarray(cohort.n_beats, dtype=int)[mask],
        quality_flagged=flags[mask],
    )


def guard_partitions(train: Partition, validation: Partition) -> Dict[str, Any]:
    """Re-prove the guarantees at the point of use rather than trusting §7.1.

    The fold audit ran once on the CSV; this runs on the exact arrays about to be fitted and
    scored. A leak introduced by a later refactor would show up here, not in a document.
    """
    shared = np.intersect1d(np.unique(train.patients), np.unique(validation.patients))
    if shared.size:
        raise ClassicalBaselineError(
            f"{shared.size} patient(s) appear in both TRAIN and VALIDATION, "
            f"first few: {shared[:5].tolist()}."
        )
    shared_ecgs = np.intersect1d(train.ecg_ids, validation.ecg_ids)
    if shared_ecgs.size:
        raise ClassicalBaselineError(
            f"{shared_ecgs.size} ECG id(s) appear in both partitions."
        )
    forbidden = sorted(set(train.folds) | set(validation.folds) & set(TEST_FOLDS))
    if set(train.folds) & set(TEST_FOLDS) or set(validation.folds) & set(TEST_FOLDS):
        raise ClassicalBaselineError(
            f"A test fold reached a fitted partition: {forbidden}."
        )
    return {
        "patient_overlap": 0,
        "ecg_id_overlap": 0,
        "train_folds": list(train.folds),
        "validation_folds": list(validation.folds),
        "test_partition_used": False,
    }


# ---------------------------------------------------------------------------- one arm
def out_of_fold_scores(
    model: str,
    params: Dict[str, Any],
    train: Partition,
    *,
    seed: int = SEED,
) -> Tuple[np.ndarray, List[Optional[float]]]:
    """Leave-one-``strat_fold``-out predictions inside TRAIN.

    The inner folds are PTB-XL's own folds 1-8, not a fresh split: they are already verified
    patient-disjoint (§7.1), so the inner CV inherits that guarantee instead of asserting a
    new one. Every hyperparameter and threshold in this module is chosen from these
    predictions, which is why they must be out-of-fold rather than in-sample.
    """
    inner_folds = sorted(set(train.strat_folds.tolist()))
    if len(inner_folds) < 2:
        raise ClassicalBaselineError(
            f"Inner CV needs at least two TRAIN folds, found {inner_folds}."
        )
    scores = np.full(train.n_records, np.nan, dtype=np.float64)
    per_fold: List[Optional[float]] = []
    for fold in inner_folds:
        held = train.strat_folds == fold
        estimator = build_estimator(model, params, seed=seed)
        estimator.fit(train.features[~held], train.y[~held])
        scores[held] = raw_scores(estimator, train.features[held])
        per_fold.append(roc_auc(train.y[held], scores[held]))
    if not np.isfinite(scores).all():
        raise ClassicalBaselineError("Some TRAIN rows received no out-of-fold prediction.")
    return scores, per_fold


def select_configuration(
    model: str,
    train: Partition,
    *,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Choose one grid cell by mean out-of-fold ROC-AUC inside TRAIN.

    The selection criterion is the pre-registered primary metric, computed where selection
    is allowed. Every candidate's score is kept in the payload so the margin between the
    chosen cell and the runner-up is visible: a grid whose cells are indistinguishable is a
    grid that did not matter, and that is worth reporting rather than hiding behind a winner.
    """
    candidates: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for params in MODEL_GRIDS[model]:
        started = time.perf_counter()
        scores, per_fold = out_of_fold_scores(model, params, train, seed=seed)
        pooled = roc_auc(train.y, scores)
        measured = [value for value in per_fold if value is not None]
        entry = {
            "params": dict(params),
            "train_out_of_fold_roc_auc_pooled": None if pooled is None else round(pooled, 6),
            "train_out_of_fold_roc_auc_mean_over_folds": (
                round(float(np.mean(measured)), 6) if measured else None
            ),
            "train_out_of_fold_roc_auc_per_fold": [
                None if value is None else round(value, 6) for value in per_fold
            ],
            "seconds": round(time.perf_counter() - started, 2),
        }
        candidates.append(entry)
        criterion = entry["train_out_of_fold_roc_auc_mean_over_folds"]
        if criterion is not None and (best is None or criterion > best["criterion"]):
            best = {"criterion": criterion, "params": dict(params), "scores": scores}
        logger.info(
            "    %-8s %-46s train-OOF ROC-AUC %.6f (%.1fs)",
            model,
            json.dumps(params, sort_keys=True),
            criterion if criterion is not None else float("nan"),
            entry["seconds"],
        )
    if best is None:
        raise ClassicalBaselineError(f"No {model} configuration could be scored.")
    return {
        "candidates": candidates,
        "selected_params": best["params"],
        "selection_criterion": "mean out-of-fold ROC-AUC over TRAIN folds 1-8",
        "selected_criterion_value": best["criterion"],
        "out_of_fold_scores": best["scores"],
    }


def run_arm(
    model: str,
    representation: str,
    train: Partition,
    validation: Partition,
    *,
    seed: int = SEED,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> Dict[str, Any]:
    """Select inside TRAIN, refit on all of TRAIN, score VALIDATION once."""
    if model == TRIVIAL_ARM:
        return _run_trivial_arm(
            representation, train, validation, bootstrap_draws=bootstrap_draws, seed=seed
        )

    selection = select_configuration(model, train, seed=seed)
    params = selection["selected_params"]
    train_out_of_fold = selection.pop("out_of_fold_scores")

    calibrator = (
        None
        if emits_probability(model)
        else _platt_calibrator(train_out_of_fold, train.y, seed=seed)
    )
    train_probability = _apply_calibrator(calibrator, train_out_of_fold)
    threshold = select_threshold(train.y, train_probability)

    started = time.perf_counter()
    estimator = build_estimator(model, params, seed=seed)
    estimator.fit(train.features, train.y)
    fit_seconds = time.perf_counter() - started
    validation_probability = _apply_calibrator(
        calibrator, raw_scores(estimator, validation.features)
    )

    metrics = evaluate_predictions(
        validation.y, validation_probability, threshold=threshold["threshold"]
    )
    return {
        "model": model,
        "representation": representation,
        "arm": f"{model}@{representation}",
        "input_dimension": int(train.features.shape[1]),
        "selection": selection,
        "selected_params": params,
        "probability_source": "native" if calibrator is None else "train_out_of_fold_platt",
        "threshold": threshold,
        "fit_seconds": round(fit_seconds, 2),
        "validation": metrics.to_dict(),
        "validation_roc_auc": metrics.roc_auc,
        "validation_bootstrap_roc_auc": bootstrap_roc_auc(
            validation.y, validation_probability, validation.patients,
            draws=bootstrap_draws, seed=seed,
        ),
        "sensitivity_at_fixed_specificity": sensitivity_at_specificity(
            validation.y, validation_probability, FIXED_SPECIFICITY
        ),
        "validation_scores": validation_probability,
    }


def _run_trivial_arm(
    representation: str,
    train: Partition,
    validation: Partition,
    *,
    bootstrap_draws: int,
    seed: int,
) -> Dict[str, Any]:
    """Predict the TRAIN prevalence for every record. The floor, and an alignment check."""
    prevalence = float(np.mean(train.y == 1))
    scores = np.full(validation.n_records, prevalence, dtype=np.float64)
    metrics = evaluate_predictions(validation.y, scores, threshold=0.5)
    return {
        "model": TRIVIAL_ARM,
        "representation": representation,
        "arm": f"{TRIVIAL_ARM}@{representation}",
        "input_dimension": 0,
        "selected_params": {"train_prevalence": round(prevalence, 6)},
        "probability_source": "train prevalence, constant",
        "threshold": {"threshold": 0.5, "selected_on": "fixed; nothing to select"},
        "fit_seconds": 0.0,
        "validation": metrics.to_dict(),
        "validation_roc_auc": metrics.roc_auc,
        "validation_bootstrap_roc_auc": {
            "measured": False,
            "reason": "a constant score has no rank information; ROC-AUC is 0.5 by "
                      "construction and an interval around it would be meaningless",
            "n_draws_requested": int(bootstrap_draws),
        },
        "sensitivity_at_fixed_specificity": sensitivity_at_specificity(
            validation.y, scores, FIXED_SPECIFICITY
        ),
        "validation_scores": scores,
    }


# ------------------------------------------------------------------------ permutation
def permutation_null(
    model: str,
    params: Dict[str, Any],
    train: Partition,
    validation: Partition,
    observed_roc_auc: float,
    *,
    draws: int = PERMUTATION_DRAWS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Label-permutation null for the strongest arm (§7.4 condition 4).

    Labels are permuted **at the patient level** inside TRAIN: a patient's whole block of
    ECGs receives another patient's label. Permuting records independently would break the
    within-patient label correlation that the real data has, so the null distribution would
    come from an easier problem than the observed fit and the p-value would be optimistic.

    The model is refitted from scratch on every draw. Reusing the fitted model and only
    permuting the scores would test nothing about the fitting pipeline, which is exactly
    what this condition exists to test.
    """
    if draws <= 0:
        return {"measured": False, "reason": "permutation draws set to zero"}

    patients, blocks = _patient_row_index(train.patients)
    patient_labels = np.asarray([int(train.y[block[0]]) for block in blocks])
    consistent = all(
        np.unique(train.y[block]).size == 1 for block in blocks
    )
    rng = np.random.default_rng(seed)

    values: List[float] = []
    started = time.perf_counter()
    for draw in range(int(draws)):
        shuffled = rng.permutation(patient_labels)
        y_permuted = np.empty(train.n_records, dtype=int)
        for block, label in zip(blocks, shuffled):
            y_permuted[block] = label
        if np.unique(y_permuted).size < 2:
            continue
        estimator = build_estimator(model, params, seed=seed)
        estimator.fit(train.features, y_permuted)
        value = roc_auc(validation.y, raw_scores(estimator, validation.features))
        if value is not None:
            values.append(value)
        if (draw + 1) % 25 == 0:
            logger.info(
                "    permutation %d/%d, null ROC-AUC so far mean %.4f max %.4f (%.0fs)",
                draw + 1, draws, float(np.mean(values)), float(np.max(values)),
                time.perf_counter() - started,
            )

    if len(values) < 2:
        return {"measured": False, "reason": "fewer than two usable permutation draws"}
    array = np.asarray(values, dtype=np.float64)
    # The +1 in numerator and denominator is the standard conservative correction: with a
    # finite number of draws, an unbeaten observation supports "p < 1/(draws+1)", not "p = 0".
    exceedances = int(np.count_nonzero(array >= observed_roc_auc))
    return {
        "measured": True,
        "unit": "patient-blocked label permutation inside TRAIN, model refitted each draw",
        "n_draws_requested": int(draws),
        "n_draws_usable": int(array.size),
        "within_patient_labels_consistent": bool(consistent),
        "observed_roc_auc": round(float(observed_roc_auc), 6),
        "null_mean": round(float(array.mean()), 6),
        "null_standard_deviation": round(float(array.std(ddof=1)), 6),
        "null_p97_5": round(float(np.percentile(array, 97.5)), 6),
        "null_max": round(float(array.max()), 6),
        "n_null_at_or_above_observed": exceedances,
        "p_value": round((exceedances + 1) / (array.size + 1), 6),
        "seconds": round(time.perf_counter() - started, 1),
    }


# ------------------------------------------------------------------------ failure map
def _group_errors(
    y: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    label: str,
) -> Dict[str, Any]:
    """Sensitivity / specificity / error rate inside one subgroup."""
    if not mask.any():
        return {"group": label, "n_records": 0, "measured": False}
    y_group = y[mask]
    predicted_group = predicted[mask]
    positive = y_group == 1
    negative = ~positive
    return {
        "group": label,
        "n_records": int(mask.sum()),
        "n_positive": int(positive.sum()),
        "prevalence": round(float(positive.mean()), 4),
        "error_rate": round(float(np.mean(predicted_group != y_group)), 4),
        "sensitivity": (
            round(float(predicted_group[positive].mean()), 4) if positive.any() else None
        ),
        "specificity": (
            round(float(1.0 - predicted_group[negative].mean()), 4)
            if negative.any() else None
        ),
        "measured": True,
    }


def failure_map(
    arm: Dict[str, Any],
    validation: Partition,
    cohort: EcgCohort,
    transform: EcgFeatureTransform,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Where the strongest classical arm fails -- the output this phase actually needs.

    A leaderboard says which arm won. This says what the winner cannot do, which is the only
    part that can motivate a quantum experiment: a circuit proposed against "classical is at
    0.9x" is a fishing expedition, while a circuit proposed against a named, quantified,
    reproducible failure mode is a hypothesis.
    """
    scores = np.asarray(arm["validation_scores"], dtype=np.float64)
    threshold = float(arm["threshold"]["threshold"])
    predicted = (scores >= threshold).astype(int)
    y = validation.y
    columns = {name: index for index, name in enumerate(cohort.superclass_order)}
    present = validation.superclasses.astype(bool)

    by_superclass: List[Dict[str, Any]] = []
    for name in cohort.superclass_order:
        mask = present[:, columns[name]]
        entry = _group_errors(y, predicted, mask, f"superclass:{name}")
        # For an abnormal superclass the interesting number is recall among its records;
        # a record may carry several superclasses, so these groups deliberately overlap.
        entry["note"] = (
            "overlapping group: a record can carry several superclasses"
            if name != "NORM"
            else "NORM records that are positive also carry an abnormal superclass"
        )
        by_superclass.append(entry)

    # Abnormal records carrying exactly one superclass: the cleanest available read on which
    # single pathology the classical arm misses, without multi-label confounding.
    single_label: List[Dict[str, Any]] = []
    n_superclasses = present[:, [columns[n] for n in SUPERCLASS_ABNORMAL]].sum(axis=1)
    for name in SUPERCLASS_ABNORMAL:
        mask = present[:, columns[name]] & (n_superclasses == 1) & (y == 1)
        single_label.append(_group_errors(y, predicted, mask, f"only:{name}"))

    age = validation.ages
    age_bands = [
        ("age:<40", age < 40),
        ("age:40-59", (age >= 40) & (age < 60)),
        ("age:60-74", (age >= 60) & (age < 75)),
        ("age:75+", age >= 75),
        ("age:missing", ~np.isfinite(age)),
    ]
    beats = validation.n_beats
    subgroups = (
        [_group_errors(y, predicted, mask, name) for name, mask in age_bands]
        + [
            _group_errors(y, predicted, validation.sexes == 0, "sex:0"),
            _group_errors(y, predicted, validation.sexes == 1, "sex:1"),
            _group_errors(y, predicted, validation.quality_flagged, "quality:flagged"),
            _group_errors(y, predicted, ~validation.quality_flagged, "quality:clean"),
            _group_errors(y, predicted, beats < 8, "beats:<8"),
            _group_errors(y, predicted, beats >= 8, "beats:>=8"),
        ]
    )

    # The records the task cannot score. Reported so the exclusion is visible as a number
    # rather than a sentence, and never folded into any metric.
    fold_array = np.asarray(cohort.strat_folds, dtype=int)
    unknown_mask = np.isin(fold_array, list(validation.folds)) & (
        np.asarray(labels) == LABEL_UNKNOWN
    )
    unknown: Dict[str, Any] = {
        "n_records": int(unknown_mask.sum()),
        "scored": False,
        "note": "no diagnostic superclass: the label is unknown, never negative (§7.4)",
    }
    if unknown_mask.any():
        unknown_scores = np.asarray(
            arm.get("unknown_scores", np.empty(0)), dtype=np.float64
        )
        if unknown_scores.size == int(unknown_mask.sum()):
            unknown["mean_score"] = round(float(unknown_scores.mean()), 6)
            unknown["fraction_above_threshold"] = round(
                float(np.mean(unknown_scores >= threshold)), 4
            )

    confusion = arm["validation"].get("confusion_matrix", {})
    return {
        "arm": arm["arm"],
        "threshold": round(threshold, 6),
        "overall": {
            "n_records": validation.n_records,
            "n_patients": validation.n_patients,
            "error_rate": round(float(np.mean(predicted != y)), 4),
            "confusion_matrix": confusion,
        },
        "missed_positives": int(np.count_nonzero((y == 1) & (predicted == 0))),
        "false_alarms": int(np.count_nonzero((y == 0) & (predicted == 1))),
        "by_superclass": by_superclass,
        "by_single_superclass": single_label,
        "by_subgroup": subgroups,
        "unknown_label_records": unknown,
    }


# ------------------------------------------------------------------------------- run
def _representation_transform(
    cohort: EcgCohort, reduction: str, n_components: Optional[int]
) -> EcgFeatureTransform:
    """Fit one representation on TRAIN folds only."""
    kwargs: Dict[str, Any] = {"reduction": reduction}
    if n_components is not None:
        kwargs["n_components"] = int(n_components)
    transform = fit_ecg_transform(cohort, **kwargs)
    if set(transform.fitted_on_folds) - set(TRAIN_FOLDS):
        raise ClassicalBaselineError(
            f"Representation fitted on non-TRAIN folds {transform.fitted_on_folds}."
        )
    return transform


def run(
    cohort: EcgCohort,
    *,
    models: Sequence[str] = MODELS,
    representations: Sequence[str] = tuple(name for name, _, _ in REPRESENTATIONS),
    tasks: Sequence[str] = tuple(TASKS),
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    permutation_draws: int = PERMUTATION_DRAWS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """The full classical failure map. Fits on folds 1-8, scores fold 9, never reads 10."""
    fold_array = np.asarray(cohort.strat_folds, dtype=int)
    if np.isin(fold_array, sorted(TEST_FOLDS)).any():
        raise ClassicalBaselineError(
            "The cohort contains frozen-test rows. This module has no honest use for "
            "them; rebuild the cache without fold 10."
        )
    unknown_models = sorted(set(models) - set(MODELS) - {TRIVIAL_ARM})
    if unknown_models:
        raise ClassicalBaselineError(f"Unknown model arms {unknown_models}.")
    layout = {name: (reduction, size) for name, reduction, size in REPRESENTATIONS}
    unknown_representations = sorted(set(representations) - set(layout))
    if unknown_representations:
        raise ClassicalBaselineError(f"Unknown representations {unknown_representations}.")
    unknown_tasks = sorted(set(tasks) - set(TASKS))
    if unknown_tasks:
        raise ClassicalBaselineError(f"Unknown tasks {unknown_tasks}.")

    started = time.perf_counter()
    report: Dict[str, Any] = {
        "baseline_version": E4_BASELINE_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "cache_version": ECG_CACHE_VERSION,
        "seed": seed,
        "bootstrap_draws": int(bootstrap_draws),
        "permutation_draws": int(permutation_draws),
        "fixed_specificity": FIXED_SPECIFICITY,
        "protocol": {
            "train_folds": sorted(TRAIN_FOLDS),
            "validation_folds": sorted(VALIDATION_FOLDS),
            "test_folds": sorted(TEST_FOLDS),
            "test_partition_used": False,
            "primary_metric": "ROC-AUC (§7.4, frozen before any model)",
            "bootstrap_unit": "patient",
            "selection": "leave-one-strat-fold-out CV inside TRAIN folds 1-8",
        },
        "cohort": {
            "n_records": int(cohort.n_records),
            "n_patients": int(cohort.n_patients),
            "dimension": int(cohort.dimension),
            "folds_present": sorted(set(fold_array.tolist())),
        },
        "representations": [],
        "tasks": {},
    }

    transforms: Dict[str, EcgFeatureTransform] = {}
    for name in representations:
        reduction, size = layout[name]
        transform = _representation_transform(cohort, reduction, size)
        transforms[name] = transform
        report["representations"].append(
            {
                "name": name,
                "reduction": reduction,
                "output_dimension": transform.output_dimension,
                "total_explained_variance_ratio": (
                    None
                    if transform.total_explained_variance_ratio is None
                    else round(float(transform.total_explained_variance_ratio), 6)
                ),
                "fitted_on_folds": list(transform.fitted_on_folds),
                "n_train_patients": transform.n_train_patients,
            }
        )
        logger.info(
            "representation %-6s -> %2d dims (variance %s)",
            name,
            transform.output_dimension,
            "n/a" if transform.total_explained_variance_ratio is None
            else f"{transform.total_explained_variance_ratio:.4f}",
        )

    for task_name in tasks:
        labels = TASKS[task_name](cohort)
        is_primary = task_name.startswith("primary")
        task_report: Dict[str, Any] = {
            "task": task_name,
            "used_for_selection": bool(is_primary),
            "n_unknown_label": int(np.count_nonzero(np.asarray(labels) == LABEL_UNKNOWN)),
            "arms": [],
        }
        logger.info(
            "task %s (%s)", task_name,
            "selection allowed" if is_primary else "reported only, never selected on",
        )

        arm_scores: Dict[str, np.ndarray] = {}
        first_validation: Optional[Partition] = None
        for representation in representations:
            transform = transforms[representation]
            train = build_partition(
                cohort, transform, labels, sorted(TRAIN_FOLDS), "TRAIN"
            )
            validation = build_partition(
                cohort, transform, labels, sorted(VALIDATION_FOLDS), "VALIDATION"
            )
            guards = guard_partitions(train, validation)
            assert_no_fit_eval_patient_overlap(transform, validation.patients.tolist())
            if first_validation is None:
                first_validation = validation
                task_report["partitions"] = [train.summary(), validation.summary()]
                task_report["guards"] = guards

            arms = list(models)
            if is_primary and representation == representations[0]:
                arms = [TRIVIAL_ARM] + arms
            for model in arms:
                logger.info("  arm %s@%s", model, representation)
                arm = run_arm(
                    model, representation, train, validation,
                    seed=seed, bootstrap_draws=bootstrap_draws,
                )
                arm_scores[arm["arm"]] = np.asarray(arm.pop("validation_scores"))
                arm["selection"] = arm.get("selection", {})
                task_report["arms"].append(arm)
                logger.info(
                    "  -> %s validation ROC-AUC %s",
                    arm["arm"],
                    "undefined" if arm["validation_roc_auc"] is None
                    else f"{arm['validation_roc_auc']:.6f}",
                )

        measured = [
            arm for arm in task_report["arms"]
            if arm["validation_roc_auc"] is not None and arm["model"] != TRIVIAL_ARM
        ]
        if not measured:
            raise ClassicalBaselineError(f"No arm could be scored for task {task_name!r}.")
        strongest = max(measured, key=lambda arm: arm["validation_roc_auc"])
        task_report["strongest_arm"] = {
            "arm": strongest["arm"],
            "model": strongest["model"],
            "representation": strongest["representation"],
            "input_dimension": strongest["input_dimension"],
            "selected_params": strongest["selected_params"],
            "validation_roc_auc": strongest["validation_roc_auc"],
            "bootstrap": strongest["validation_bootstrap_roc_auc"],
            "note": (
                "This is the number §7.4 condition 1 requires a quantum model to exceed."
                if is_primary
                else "Secondary task: never a selection target and never a quantum bar."
            ),
        }

        # Paired comparisons against the strongest arm: the same machinery §7.4 condition 2
        # will apply to a quantum arm, exercised here so the compact controls' distance from
        # the ceiling is an interval rather than a difference of two point estimates.
        assert first_validation is not None
        best_scores = arm_scores[strongest["arm"]]
        task_report["paired_vs_strongest"] = [
            {
                "arm": arm["arm"],
                "input_dimension": arm["input_dimension"],
                **paired_bootstrap_delta(
                    first_validation.y, best_scores, arm_scores[arm["arm"]],
                    first_validation.patients, draws=bootstrap_draws, seed=seed,
                ),
            }
            for arm in task_report["arms"]
            if arm["arm"] != strongest["arm"] and arm["model"] != TRIVIAL_ARM
        ]

        if is_primary:
            strongest_validation = build_partition(
                cohort, transforms[strongest["representation"]], labels,
                sorted(VALIDATION_FOLDS), "VALIDATION",
            )
            strongest_train = build_partition(
                cohort, transforms[strongest["representation"]], labels,
                sorted(TRAIN_FOLDS), "TRAIN",
            )
            strongest_with_scores = dict(strongest)
            strongest_with_scores["validation_scores"] = best_scores
            task_report["failure_map"] = failure_map(
                strongest_with_scores, strongest_validation, cohort,
                transforms[strongest["representation"]], labels,
            )
            task_report["permutation_null"] = permutation_null(
                strongest["model"], strongest["selected_params"],
                strongest_train, strongest_validation,
                float(strongest["validation_roc_auc"]),
                draws=permutation_draws, seed=seed,
            )

        report["tasks"][task_name] = task_report

    report["total_seconds"] = round(time.perf_counter() - started, 1)
    report["test_partition_used"] = False
    return report


def summary_lines(report: Dict[str, Any]) -> List[str]:
    """A console-legible leaderboard plus the failure map's headline numbers."""
    lines: List[str] = []
    for task_name, task in report["tasks"].items():
        marker = "selection allowed" if task["used_for_selection"] else "reported only"
        lines.append(f"{task_name} [{marker}]")
        for partition in task.get("partitions", []):
            lines.append(
                f"  {partition['partition']:<11} {partition['n_records']:>6} rec  "
                f"{partition['n_patients']:>6} pt  "
                f"{partition['n_positive']:>6}+ / {partition['n_negative']:>6}-  "
                f"prevalence {partition['prevalence']:.4f}"
            )
        ranked = sorted(
            task["arms"],
            key=lambda arm: (-1.0 if arm["validation_roc_auc"] is None
                             else -arm["validation_roc_auc"]),
        )
        for arm in ranked:
            bootstrap = arm["validation_bootstrap_roc_auc"]
            interval = (
                f"[{bootstrap['p2_5']:.4f}, {bootstrap['p97_5']:.4f}]"
                if bootstrap.get("measured") else "[not measured]"
            )
            value = arm["validation_roc_auc"]
            pr_auc = arm["validation"].get("pr_auc")
            lines.append(
                f"  {arm['arm']:<22} d={arm['input_dimension']:>3}  "
                f"ROC-AUC {'   n/a  ' if value is None else f'{value:.6f}'} {interval}  "
                f"PR-AUC {'n/a' if pr_auc is None else f'{pr_auc:.4f}'}  "
                f"bal-acc {arm['validation'].get('balanced_accuracy'):.4f}"
            )
        best = task.get("strongest_arm", {})
        if best:
            lines.append(
                f"  strongest: {best['arm']} at ROC-AUC {best['validation_roc_auc']:.6f}"
            )
        null = task.get("permutation_null", {})
        if null.get("measured"):
            lines.append(
                f"  permutation null: mean {null['null_mean']:.4f} max {null['null_max']:.4f}"
                f" -> p = {null['p_value']}"
            )
        fmap = task.get("failure_map")
        if fmap:
            lines.append(
                f"  failure map: {fmap['missed_positives']} missed positives, "
                f"{fmap['false_alarms']} false alarms at threshold {fmap['threshold']:.4f}"
            )
            for entry in fmap["by_single_superclass"]:
                if entry.get("measured"):
                    lines.append(
                        f"    {entry['group']:<12} n={entry['n_records']:>4}  "
                        f"recall {entry['sensitivity']}"
                    )
    return lines


def _strip_arrays(value: Any) -> Any:
    """Make the report JSON-serialisable without silently dropping a number."""
    if isinstance(value, dict):
        return {key: _strip_arrays(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strip_arrays(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="E4.1 classical failure map on PTB-XL (folds 1-8 fit, 9 score)."
    )
    parser.add_argument("--cache", type=Path, required=True, help="ecg-v1 feature cache")
    parser.add_argument("--out", type=Path, default=None, help="JSON report destination")
    parser.add_argument(
        "--models", default=",".join(MODELS), help=f"subset of {','.join(MODELS)}"
    )
    parser.add_argument(
        "--representations",
        default=",".join(name for name, _, _ in REPRESENTATIONS),
        help="subset of " + ",".join(name for name, _, _ in REPRESENTATIONS),
    )
    parser.add_argument("--tasks", default=",".join(TASKS), help="subset of " + ",".join(TASKS))
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--permutation-draws", type=int, default=PERMUTATION_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cohort = load_cohort(args.cache)
    report = run(
        cohort,
        models=tuple(name for name in args.models.split(",") if name),
        representations=tuple(name for name in args.representations.split(",") if name),
        tasks=tuple(name for name in args.tasks.split(",") if name),
        bootstrap_draws=args.bootstrap_draws,
        permutation_draws=args.permutation_draws,
        seed=args.seed,
    )

    for line in summary_lines(report):
        print(line)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(_strip_arrays(report), indent=2, sort_keys=False), encoding="utf-8"
        )
        print(f"\nreport -> {args.out}  ({args.out.stat().st_size} bytes)")
    print(f"total {report['total_seconds']}s, test partition used: {report['test_partition_used']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
