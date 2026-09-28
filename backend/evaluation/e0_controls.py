"""Classical controls and the complementarity test for the E0 decision gate.

``docs/PHASE_E0_DIAGNOSTIC_SPEC.md`` §9 is explicit that a readout improvement does not
validate the 65,536-amplitude encoding. The encoding has to beat a strong classical
control **on identical rows**, or provide information the classical control does not
already have. This module measures both, so the decision gate is entered against
numbers rather than against an argument.

Four controls, per ROI condition
--------------------------------

=================================  ==================================================
Control                            Discipline
=================================  ==================================================
``random_forest``                  refit from the frozen ``BaselineRecord`` on TRAIN
``linear_svm_on_amplitudes``       linear kernel on the same 65,536-dim states,
                                   ``C`` by patient-grouped CV **inside TRAIN**
``rbf_svm_on_amplitudes``          same Gram, ``gamma`` by median heuristic on TRAIN
                                   pairwise distances, ``C`` the same way
``phase_d_vqc``                    the frozen artifact re-scored on these rows
=================================  ==================================================

The linear control is deliberately **redone** rather than quoted. The existing figures
(PR-AUC 0.128 oracle / 0.263 predicted) had ``C`` chosen on the evaluation partition,
which makes them optimistic for the control and therefore understates the bar the
quantum model has to clear. A control that is allowed to cheat is not a control.

One Gram serves both kernels
----------------------------
The amplitude vectors are unit-norm by construction, so
``||x - y||^2 = 2 - 2 <x, y>``: a single ``X X^T`` gives the linear kernel and every
squared distance the RBF kernel needs. That is 1,692^2 = 2.9 M entries at the largest
condition -- tractable, which is what §9.1 asks be established rather than assumed. If a
condition were not tractable it would be reported as such, not dropped.

Why the stack uses out-of-fold train scores, and why its null is on the increment
--------------------------------------------------------------------------------
§9.2 wants a logistic regression on ``[RF score, quantum score]`` fitted on train only.
Fitting it on *in-sample* train scores would be a trap: the random forest is near-perfect
on its own training rows, so the stack would learn "trust the forest" from a signal that
does not exist at validation time, and the quantum feature could never earn a
coefficient. Both features are therefore generated **out of fold** on train, under the
same patient-grouped splitter, and the validation features come from the full-train
fits. The stack then gets a fair look at what the quantum score adds.

The spec asks whether the stack's *gain* survives its own null, and that word is
load-bearing. A null over the stack's absolute PR-AUC would be cleared by the RF feature
on its own -- the forest carries real signal, so any model containing it beats a
label-permuted refit -- and the test would certify "complementarity" for a quantum column
of pure noise. So the null quantity is ``stack - rf_only`` with both sides refitted under
the same handicap on every draw.

Two nulls are reported for that increment, because the E0-standard one cannot answer
"yes". Permuting the train labels also randomises the refitted coefficient *signs*, and
the resulting model is still scored against the real validation labels, so about a quarter
of draws land on a sign configuration that reproduces the observed increment by luck --
flooring the empirical p near 0.25 however real the effect is. Measured on a synthetic
cohort constructed to be genuinely complementary, a +0.163 increment scored p = 0.11 there.
The gating null therefore holds the labels and the RF column fixed and permutes only the
quantum column, patient-blocked, which asks the actual question: does *this* column add
more than an unrelated column of the same shape? Both appear in the payload, with
``gates_the_claim`` marking which is which.

The test partition is not read. Train and validation only, enforced by source inspection
in ``tests/test_e0_controls.py`` exactly as Phase D and E0.1 enforce it.

Usage::

    python -m backend.evaluation.e0_controls --roi oracle --e0 reports_e0_oracle.json
    python -m backend.evaluation.e0_controls --roi predicted --condition B_localized
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings, settings as default_settings
# Imported rather than re-implemented: the controls must be scored by the same metric
# call, summarised against the same null machinery and standardised by the same
# train-only scaler as the quantum readouts they are compared against. A second
# implementation of any of those would make a difference in the numbers ambiguous
# between "the models differ" and "the harnesses differ".
from backend.evaluation.e0_readout import (
    DEFAULT_PERMUTATIONS,
    LARGE_CONDITION_PERMUTATIONS,
    LARGE_CONDITION_TRAIN_ROWS,
    PRIMARY_CONDITION,
    E0Error,
    _metrics,
    _null_summary,
    _rank_in_null,
    fit_readout,
    load_frozen_model,
    patient_blocked_permutations,
)
from backend.ml.artifacts import ArtifactStore, BASELINES_FILE
from backend.ml.baselines import BaselineRecord, build_estimator, refit_from_record
from backend.ml.pixel_pipeline import V1_PIXEL_COUNT, V1_QUBIT_COUNT
from backend.training.data import load_partitions
from backend.training.pixel_data import (
    CONDITION_DESCRIPTIONS,
    LoadedPixelDataset,
    PixelPartition,
    condition_rows,
    load_pixel_partitions,
)
from quantum_ml.readout import measure_batch, observable_set

logger = logging.getLogger(__name__)

CONTROLS_VERSION = "v1-e0-controls-1"

#: Regularisation grid, identical to the one the E0.1 readout uses. Shared so that
#: "the control had less hyperparameter freedom than the quantum model" cannot be an
#: explanation for a difference between them.
C_GRID: Tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)

#: Folds for every within-train selection and for the out-of-fold stack features.
N_FOLDS = 5

#: Above this many train rows the Gram matrix and its CV are reported as intractable
#: rather than attempted. 1,692^2 float64 is 23 MB, so nothing in this project reaches
#: it -- the guard exists so that a future larger condition produces a stated refusal
#: instead of a silent swap.
MAX_GRAM_ROWS = 6000

FIXED_THRESHOLD = 0.50


class ControlsError(RuntimeError):
    """Raised when a control cannot be measured on aligned rows."""


# ------------------------------------------------------------------------ CV plumbing
def grouped_folds(
    y: np.ndarray, groups: Sequence[str], *, seed: int, n_splits: int = N_FOLDS
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str]:
    """Patient-grouped stratified folds inside train, with a stated fallback.

    Several images share a patient. An ungrouped fold would put the same patient on
    both sides of a split, so any hyperparameter chosen against it -- ``C``, ``gamma``,
    a stack coefficient -- would be chosen against a leak. When grouping is impossible
    the fallback is used and *named in the returned note*, never substituted quietly.
    """
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        k = min(n_splits, int(counts.min()), len(set(groups)))
        if k < 2:
            raise ValueError("not enough groups or minority-class rows for grouped CV")
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.zeros((y.size, 1)), y, groups=list(groups)))
        return folds, (
            f"{k}-fold StratifiedGroupKFold on TRAIN, grouped by patient, "
            "scoring average_precision"
        )
    except Exception as exc:  # noqa: BLE001 -- degrade to a stated fallback, not silently
        from sklearn.model_selection import StratifiedKFold

        k = max(2, min(n_splits, int(counts.min())))
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.zeros((y.size, 1)), y))
        return folds, (
            f"{k}-fold StratifiedKFold on TRAIN (patient grouping unavailable: "
            f"{type(exc).__name__}), scoring average_precision"
        )


def _average_precision(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=int)
    if y.size == 0 or len(set(y.tolist())) < 2:
        return None
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, np.asarray(score, dtype=np.float64)))


# --------------------------------------------------------------------- kernel controls
def median_heuristic_gamma(k_tt: np.ndarray) -> Tuple[float, float]:
    """``(gamma, median_squared_distance)`` from **train** pairwise distances only.

    For unit-norm rows ``||x - y||^2 = 2 - 2 <x, y>``, so the Gram already contains
    every distance. The median is taken over the strict upper triangle: the diagonal is
    zero by definition and including it would drag the median towards a scale no pair of
    distinct images actually has.
    """
    n = k_tt.shape[0]
    if n < 2:
        raise ControlsError("The median heuristic needs at least two train rows.")
    iu = np.triu_indices(n, k=1)
    squared = np.maximum(2.0 - 2.0 * k_tt[iu], 0.0)
    median = float(np.median(squared))
    if not np.isfinite(median) or median <= 0.0:
        raise ControlsError(
            f"Median train squared distance is {median!r}; the amplitude states are "
            "degenerate and an RBF width cannot be set from them."
        )
    return 1.0 / median, median


def _fit_precomputed_svm(
    k_tt: np.ndarray,
    y_train: np.ndarray,
    k_vt: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> Dict[str, Any]:
    """Select ``C`` inside train on a precomputed kernel, then score validation.

    Returns the decision function on train (in-sample), out of fold on train, and on
    validation, plus a Platt map fitted on the **out-of-fold** train values. The Platt
    step exists because PR-AUC and ROC-AUC are rank metrics an SVM decision function
    satisfies fine, while Brier and ECE are not defined on an unbounded margin -- and
    the spec asks for all four. Fitting the map on out-of-fold values keeps it a
    train-only object; fitting it in sample would report a calibration the control does
    not have.
    """
    from sklearn.svm import SVC

    y_train = np.asarray(y_train, dtype=int)
    scores: Dict[float, Optional[float]] = {}
    for c in C_GRID:
        oof = np.full(y_train.size, np.nan, dtype=np.float64)
        for train_idx, test_idx in folds:
            if len(set(y_train[train_idx].tolist())) < 2:
                continue
            model = SVC(C=c, kernel="precomputed", class_weight="balanced")
            model.fit(k_tt[np.ix_(train_idx, train_idx)], y_train[train_idx])
            oof[test_idx] = model.decision_function(k_tt[np.ix_(test_idx, train_idx)])
        usable = np.isfinite(oof)
        scores[c] = (
            None if usable.sum() < 2 else _average_precision(y_train[usable], oof[usable])
        )

    ranked = [(c, ap) for c, ap in scores.items() if ap is not None]
    if not ranked:
        raise ControlsError(
            "No value of C produced a usable within-train CV score; the folds are "
            "degenerate for this condition."
        )
    # Ties broken towards the *stronger* penalty (smaller C). At 65,536 features and a
    # few hundred rows the grid is mostly interpolating, so a tie means the data did not
    # distinguish the settings and the simpler model is the honest pick.
    best_c = min((c for c, ap in ranked if ap == max(a for _, a in ranked)))

    final = SVC(C=best_c, kernel="precomputed", class_weight="balanced")
    final.fit(k_tt, y_train)
    train_decision = final.decision_function(k_tt)
    validation_decision = final.decision_function(k_vt)

    oof = np.full(y_train.size, np.nan, dtype=np.float64)
    for train_idx, test_idx in folds:
        if len(set(y_train[train_idx].tolist())) < 2:
            continue
        model = SVC(C=best_c, kernel="precomputed", class_weight="balanced")
        model.fit(k_tt[np.ix_(train_idx, train_idx)], y_train[train_idx])
        oof[test_idx] = model.decision_function(k_tt[np.ix_(test_idx, train_idx)])

    platt = None
    usable = np.isfinite(oof)
    if usable.sum() >= 2 and len(set(y_train[usable].tolist())) == 2:
        from sklearn.linear_model import LogisticRegression

        platt = LogisticRegression(max_iter=5000, random_state=seed)
        platt.fit(oof[usable].reshape(-1, 1), y_train[usable])

    def _probability(decision: np.ndarray) -> Optional[np.ndarray]:
        if platt is None:
            return None
        return np.asarray(
            platt.predict_proba(np.asarray(decision, dtype=np.float64).reshape(-1, 1))[:, 1],
            dtype=np.float64,
        )

    return {
        "selected_C": float(best_c),
        "cv_average_precision_by_C": {
            str(c): (None if ap is None else round(float(ap), 6)) for c, ap in scores.items()
        },
        "n_support_vectors": int(sum(int(v) for v in final.n_support_)),
        "train_decision": train_decision,
        "train_decision_out_of_fold": oof,
        "validation_decision": validation_decision,
        "train_probability": _probability(train_decision),
        "validation_probability": _probability(validation_decision),
        "platt_note": (
            "probabilities from a 1-D logistic map fitted on out-of-fold TRAIN decision "
            "values; no validation row contributed to it"
            if platt is not None
            else "no probability map: out-of-fold train decisions were single-class"
        ),
    }


def amplitude_controls(
    amplitudes_train: np.ndarray,
    y_train: np.ndarray,
    amplitudes_validation: np.ndarray,
    y_validation: np.ndarray,
    groups: Sequence[str],
    *,
    seed: int,
) -> Dict[str, Any]:
    """Linear and RBF kernel controls on the identical 65,536-dim amplitude vectors."""
    n_train = amplitudes_train.shape[0]
    if n_train > MAX_GRAM_ROWS:
        return {
            "linear_svm_on_amplitudes": {
                "measured": False,
                "reason": (
                    f"{n_train} train rows exceeds the {MAX_GRAM_ROWS}-row Gram "
                    "threshold; reported as not measured rather than substituted."
                ),
            },
            "rbf_svm_on_amplitudes": {"measured": False, "reason": "same as linear"},
        }

    folds, cv_note = grouped_folds(y_train, groups, seed=seed)

    started = time.perf_counter()
    k_tt = amplitudes_train @ amplitudes_train.T
    k_vt = amplitudes_validation @ amplitudes_train.T
    gram_seconds = time.perf_counter() - started

    out: Dict[str, Any] = {}

    linear = _fit_precomputed_svm(k_tt, y_train, k_vt, folds, seed=seed)
    out["linear_svm_on_amplitudes"] = {
        "measured": True,
        "kernel": "linear (precomputed Gram on the unit-norm amplitude vectors)",
        "input_dimension": int(amplitudes_train.shape[1]),
        "selection": cv_note,
        "redone_note": (
            "C selected inside TRAIN only. The previously reported linear figures had C "
            "chosen on the evaluation partition and are optimistic for the control; this "
            "number is the bar the quantum readout actually has to clear."
        ),
        "gram_seconds": round(gram_seconds, 3),
        **{k: v for k, v in linear.items() if not isinstance(v, np.ndarray)},
    }

    gamma, median_squared = median_heuristic_gamma(k_tt)
    squared_vt = np.maximum(2.0 - 2.0 * k_vt, 0.0)
    squared_tt = np.maximum(2.0 - 2.0 * k_tt, 0.0)
    rbf_tt = np.exp(-gamma * squared_tt)
    rbf_vt = np.exp(-gamma * squared_vt)
    rbf = _fit_precomputed_svm(rbf_tt, y_train, rbf_vt, folds, seed=seed)
    out["rbf_svm_on_amplitudes"] = {
        "measured": True,
        "kernel": "RBF from the same Gram via ||x-y||^2 = 2 - 2<x,y>",
        "input_dimension": int(amplitudes_train.shape[1]),
        "gamma": round(float(gamma), 8),
        "gamma_selection": (
            "median heuristic: gamma = 1 / median of the strict-upper-triangle TRAIN "
            f"squared distances (median = {median_squared:.6f}). No validation row "
            "contributed to the width."
        ),
        "selection": cv_note,
        **{k: v for k, v in rbf.items() if not isinstance(v, np.ndarray)},
    }

    out["_arrays"] = {"linear": linear, "rbf": rbf}
    return out


# --------------------------------------------------------------------------- RF control
def random_forest_scores(
    *, config: Optional[Settings] = None, model_version: Optional[str] = None, seed: int = 42
) -> Dict[str, Any]:
    """Frozen classical reference: full-train fit plus out-of-fold train scores.

    The full-train fit is what scores validation, exactly as
    :mod:`backend.evaluation.pixel_comparison` does. The out-of-fold scores exist only
    for the §9.2 stack, which cannot be fitted on in-sample forest scores without
    learning a separation that does not survive to validation.
    """
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)
    version = store.resolve_version(model_version)
    payload = store.read_component(version, BASELINES_FILE)
    if not payload or "baselines" not in payload:
        raise ControlsError(
            f"No baselines artifact for model version {version!r}. Train the classical "
            "reference with `python -m backend.training.train_classical` first."
        )
    if "random_forest" not in payload["baselines"]:
        raise ControlsError(
            "The baselines artifact has no random_forest record; §9.1 names it as the "
            "current classical reference and it is not substituted."
        )

    data = load_partitions(config=cfg)
    if data.train.features is None or data.validation.features is None:
        raise ControlsError(
            "load_partitions returned unreduced features; the classical reference "
            "cannot be rebuilt without the fitted pipeline."
        )
    record = BaselineRecord.from_dict(payload["baselines"]["random_forest"]["record"])
    x_train, y_train = data.train.features, data.train.labels
    fitted = refit_from_record(record, x_train, y_train)
    validation_probability = np.asarray(
        fitted.predict_proba(data.validation.features), dtype=np.float64
    )

    folds, cv_note = grouped_folds(y_train, data.train.patient_ids, seed=seed)
    oof = np.full(y_train.size, np.nan, dtype=np.float64)
    for train_idx, test_idx in folds:
        estimator = build_estimator(
            record.name, dict(record.hyperparameters), seed=record.random_seed
        )
        estimator.fit(x_train[train_idx], np.asarray(y_train)[train_idx])
        oof[test_idx] = np.asarray(estimator.predict_proba(x_train[test_idx]))[:, 1]

    return {
        "record": {
            "name": record.name,
            "model_version": version,
            "hyperparameters": record.hyperparameters,
            "seed": record.random_seed,
            "n_train_samples": record.n_train_samples,
            "feature_dimension": record.feature_dimension,
            "input": "181 fused descriptor + clinical features, PCA-reduced",
            "rebuilt_from": BASELINES_FILE,
        },
        "out_of_fold_note": (
            "Train-side scores are out of fold under " + cv_note + ", used only as a "
            "stack feature. Validation scores come from the full-train refit."
        ),
        "validation_by_id": {
            image_id: float(p)
            for image_id, p in zip(data.validation.image_ids, validation_probability)
        },
        "train_oof_by_id": {
            image_id: float(p)
            for image_id, p in zip(data.train.image_ids, oof)
            if np.isfinite(p)
        },
    }


# ------------------------------------------------------------------- quantum best score
def best_e0_setting(
    e0_payload: Optional[Dict[str, Any]], condition: str
) -> Optional[Dict[str, Any]]:
    """The variant x weight setting with the highest **validation** PR-AUC for a condition.

    Selection on validation is what the spec permits and what E0.1 already does for
    model selection; the point of §9.2 is to check whether that selected score adds
    anything to the classical reference, so it is the selected one that gets stacked.
    Returns ``None`` when no E0.1 payload was supplied, in which case the caller
    measures the controls alone.
    """
    if not e0_payload:
        return None
    for entry in e0_payload.get("conditions", []):
        if entry.get("condition") != condition:
            continue
        best: Optional[Dict[str, Any]] = None
        for block in entry.get("results", []):
            metrics = block.get("validation_metrics") or {}
            pr = metrics.get("pr_auc")
            if pr is None:
                continue
            if best is None or float(pr) > float(best["pr_auc"]):
                best = {
                    "variant": block["variant"],
                    "weight_setting": block["weight_setting"],
                    "pr_auc": float(pr),
                }
        return best
    return None


def quantum_scores_for(
    *,
    variant: str,
    weight_setting: str,
    amplitudes_train: np.ndarray,
    y_train: np.ndarray,
    amplitudes_validation: np.ndarray,
    patients_train: Sequence[str],
    frozen: Any,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    chunk: int,
) -> Dict[str, Any]:
    """Re-measure and refit one E0.1 readout, returning train-OOF and validation scores.

    Refitted here rather than carried over from the E0.1 payload: that payload records
    metrics, not per-row scores, and re-deriving them proves the reported readout is
    reproducible from the same inputs instead of taken on trust.
    """
    oset = observable_set(variant, V1_QUBIT_COUNT)
    diagonals = oset.diagonals(V1_QUBIT_COUNT)
    n_layers = 1 if frozen is None else frozen.n_layers
    if weight_setting == "W-identity":
        weights: Optional[np.ndarray] = None
    elif frozen is None:
        raise ControlsError(
            f"Weight setting {weight_setting!r} needs the frozen Phase D artifact and "
            "none was found for this condition."
        )
    elif weight_setting == "W-untrained":
        weights = frozen.untrained_weights
    else:
        weights = frozen.weights

    x_train = measure_batch(
        amplitudes_train, diagonals, weights=weights,
        n_qubits=V1_QUBIT_COUNT, n_layers=n_layers, chunk=chunk,
    )
    x_validation = measure_batch(
        amplitudes_validation, diagonals, weights=weights,
        n_qubits=V1_QUBIT_COUNT, n_layers=n_layers, chunk=chunk,
    )
    readout = fit_readout(
        x_train, y_train, patients_train,
        variant=variant, labels=list(oset.labels), seed=seed,
    )
    oof = np.full(y_train.size, np.nan, dtype=np.float64)
    for train_idx, test_idx in folds:
        if len(set(np.asarray(y_train)[train_idx].tolist())) < 2:
            continue
        try:
            fold_readout = fit_readout(
                x_train[train_idx], np.asarray(y_train)[train_idx],
                [patients_train[int(i)] for i in train_idx],
                variant=variant, labels=list(oset.labels), seed=seed,
            )
        except Exception:  # noqa: BLE001 -- a degenerate fold is dropped, never faked
            continue
        oof[test_idx] = fold_readout.scores(x_train[test_idx])

    return {
        "variant": variant,
        "weight_setting": weight_setting,
        "n_observables": len(oset),
        "n_readout_parameters": readout.n_trainable,
        "readout_selection": readout.cv_note,
        "train_out_of_fold": oof,
        "train_in_sample": readout.scores(x_train),
        "validation": readout.scores(x_validation),
    }


# ------------------------------------------------------------------------- the stack
def _patient_blocked_column_shuffle(
    values: np.ndarray, patients: Sequence[str], rng: np.random.Generator
) -> np.ndarray:
    """Permute one feature column, shuffling whole patient blocks rather than rows.

    Rows are grouped by patient, the block *order* is shuffled, and the concatenated
    result is written back in the original row order. An i.i.d. row shuffle would also
    destroy the column's relationship to the label, but it would additionally destroy the
    column's within-patient correlation -- making the shuffled column noisier than any
    real feature and the resulting null easier to beat than it should be.

    Patients contribute different numbers of images, so a block boundary can straddle two
    donors. The result is still a bijection on rows and still preserves patient-scale
    structure, which is what the null needs.
    """
    order = np.argsort(np.asarray(patients, dtype=object), kind="stable")
    grouped = np.asarray(values, dtype=np.float64)[order]
    boundaries = np.flatnonzero(
        np.asarray(patients, dtype=object)[order][1:]
        != np.asarray(patients, dtype=object)[order][:-1]
    ) + 1
    blocks = np.split(grouped, boundaries)
    rng.shuffle(blocks)
    out = np.empty_like(grouped)
    out[:] = np.concatenate(blocks) if blocks else grouped
    restored = np.empty_like(out)
    restored[order] = out
    return restored


def complementarity(
    rf_train_oof: np.ndarray,
    quantum_train_oof: np.ndarray,
    y_train: np.ndarray,
    rf_validation: np.ndarray,
    quantum_validation: np.ndarray,
    y_validation: np.ndarray,
    patients_train: Sequence[str],
    *,
    seed: int,
    n_permutations: int,
    patients_validation: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """§9.2: does ``[RF, quantum]`` beat RF alone, and does the **gain** survive its null?

    Complementarity is claimed only if both hold. The score correlation is reported
    alongside because a quantum score that merely re-derives the forest's ranking adds
    nothing even when its standalone PR-AUC looks respectable.

    The null is on the *increment*, not on the stack's absolute PR-AUC, and that
    distinction decides whether the test means anything. A null over the stack's own
    PR-AUC would be exceeded by the RF feature alone -- the forest carries real signal, so
    any model containing it beats a label-permuted refit, and the test would return
    "complementarity" for a quantum column of pure noise. So every draw refits **both** a
    one-feature RF-only model and the two-feature stack under the same handicap and
    records ``stack - rf_only``. Both models are then penalised identically and the null
    answers the question actually being asked: how much does adding a second feature buy
    when that feature carries nothing?

    Two nulls, because one of them cannot answer "yes"
    --------------------------------------------------
    ``permutation_null`` permutes the TRAIN labels, patient-blocked, as everywhere else in
    E0. It is reported because it is the E0-standard null, but it is **structurally
    low-powered here** and that has to be said rather than discovered later: a stack
    fitted on permuted labels has essentially random coefficient *signs*, and it is then
    scored against the real validation labels, so the roughly one draw in four that lands
    on ``(rf+, quantum+)`` reproduces close to the observed increment by luck. That floors
    the empirical p near 0.25 no matter how real the effect is. Measured on a synthetic
    cohort built so that RF and the second feature are genuinely complementary
    (``tests/test_e0_controls.py``), a +0.163 increment scored p = 0.11 against this null.

    ``column_permutation_null`` is therefore the one the claim is gated on. It holds the
    labels and the RF column fixed and permutes only the quantum column, patient-blocked,
    in both partitions. That destroys the column's relationship to the label while
    preserving its marginal distribution and RF's real relationship, which is exactly the
    question: does *this* column add more than an unrelated column of the same shape?
    Both readings appear in the payload; neither is hidden behind the other.
    """

    from sklearn.linear_model import LogisticRegression

    usable = np.isfinite(rf_train_oof) & np.isfinite(quantum_train_oof)
    y_train = np.asarray(y_train, dtype=int)
    if usable.sum() < 4 or len(set(y_train[usable].tolist())) < 2:
        return {
            "measured": False,
            "reason": (
                f"only {int(usable.sum())} train rows have both out-of-fold features and "
                "at least one class is missing among them; the stack is not fitted."
            ),
        }

    z_train = np.column_stack([
        np.asarray(rf_train_oof[usable], dtype=np.float64),
        np.asarray(quantum_train_oof[usable], dtype=np.float64),
    ])
    z_validation = np.column_stack([
        np.asarray(rf_validation, dtype=np.float64),
        np.asarray(quantum_validation, dtype=np.float64),
    ])
    mean = z_train.mean(axis=0)
    raw_std = z_train.std(axis=0)
    std = np.where(raw_std > 1e-12, raw_std, 1.0)
    z_train_scaled = (z_train - mean) / std
    z_validation_scaled = (z_validation - mean) / std

    def _fit_and_score(
        y: np.ndarray,
        columns: Sequence[int],
        z_t: Optional[np.ndarray] = None,
        z_v: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Validation PR-AUC of a logistic on ``columns``, fitted on ``y``.

        ``columns=[0]`` is the RF-only model and ``[0, 1]`` the stack. Same solver, same
        standardisation, same class weighting -- the only difference between them is the
        second column, which is the whole point. ``z_t``/``z_v`` override the standardised
        design matrices, which is how the column null substitutes a shuffled quantum
        column without changing anything else.
        """
        if len(set(np.asarray(y).tolist())) < 2:
            return None
        train_design = z_train_scaled if z_t is None else z_t
        validation_design = z_validation_scaled if z_v is None else z_v
        model = LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed
        )
        try:
            model.fit(train_design[:, columns], np.asarray(y, dtype=int))
        except Exception:  # noqa: BLE001 -- degenerate draw dropped, never zeroed
            return None
        scores = np.asarray(
            model.predict_proba(validation_design[:, columns])[:, 1], dtype=np.float64
        )
        metrics = _metrics(y_validation, scores)
        if not metrics or metrics.get("pr_auc") is None:
            return None
        return float(metrics["pr_auc"])

    stack_model = LogisticRegression(
        max_iter=5000, class_weight="balanced", random_state=seed
    )
    stack_model.fit(z_train_scaled, y_train[usable])
    stack_validation = np.asarray(
        stack_model.predict_proba(z_validation_scaled)[:, 1], dtype=np.float64
    )

    rf_raw = _metrics(y_validation, rf_validation)
    rf_model_pr = _fit_and_score(y_train[usable], [0])
    stack_pr = _fit_and_score(y_train[usable], [0, 1])
    stacked = _metrics(y_validation, stack_validation)
    quantum_alone = _metrics(y_validation, quantum_validation)

    used_patients = [patients_train[int(i)] for i in np.flatnonzero(usable)]

    null: List[float] = []
    if n_permutations:
        permutations = patient_blocked_permutations(
            y_train[usable], used_patients, n=n_permutations, seed=seed
        )
        for permuted in permutations:
            one = _fit_and_score(permuted, [0])
            two = _fit_and_score(permuted, [0, 1])
            if one is not None and two is not None:
                null.append(two - one)

    # The gating null: labels and RF held fixed, only the quantum column shuffled. The
    # RF-only model is invariant to that shuffle, so its score is constant across draws
    # and each draw measures purely what a same-shaped unrelated column would have added.
    column_null: List[float] = []
    column_null_note = (
        "quantum column permuted patient-blocked in TRAIN and (when patient labels for "
        "the evaluation partition are supplied) in VALIDATION; labels and the RF column "
        "untouched. This is the null the claim is gated on -- see the docstring for why "
        "the label-permutation null cannot answer 'yes' here."
    )
    if n_permutations and rf_model_pr is not None:
        rng = np.random.default_rng(seed + 977)
        validation_patients = (
            list(patients_validation)
            if patients_validation is not None
            and len(patients_validation) == z_validation.shape[0]
            else None
        )
        if validation_patients is None:
            column_null_note += (
                " VALIDATION patient labels were not available, so the validation column "
                "was shuffled i.i.d.; that is a slightly easier null than the "
                "patient-blocked version, so it is the conservative direction for a "
                "negative result and the optimistic one for a positive."
            )
        for _ in range(int(n_permutations)):
            shuffled_train = _patient_blocked_column_shuffle(
                z_train_scaled[:, 1], used_patients, rng
            )
            if validation_patients is None:
                shuffled_validation = rng.permutation(z_validation_scaled[:, 1])
            else:
                shuffled_validation = _patient_blocked_column_shuffle(
                    z_validation_scaled[:, 1], validation_patients, rng
                )
            z_t = np.column_stack([z_train_scaled[:, 0], shuffled_train])
            z_v = np.column_stack([z_validation_scaled[:, 0], shuffled_validation])
            two = _fit_and_score(y_train[usable], [0, 1], z_t, z_v)
            if two is not None:
                column_null.append(two - rf_model_pr)

    observed_increment = (
        None if stack_pr is None or rf_model_pr is None else stack_pr - rf_model_pr
    )
    rf_pr = None if not rf_raw else rf_raw.get("pr_auc")
    with np.errstate(invalid="ignore"):
        pearson = float(np.corrcoef(rf_validation, quantum_validation)[0, 1])
    order_rf = np.argsort(np.argsort(rf_validation))
    order_q = np.argsort(np.argsort(quantum_validation))
    spearman = float(np.corrcoef(order_rf, order_q)[0, 1])

    beats_rf = (
        None if stack_pr is None or rf_model_pr is None else bool(stack_pr > rf_model_pr)
    )
    rank = _rank_in_null(observed_increment, null)
    label_null_survives = None if rank is None else bool(
        rank["exceeds_null_p95"] and rank["empirical_p"] <= 0.05
    )
    column_rank = _rank_in_null(observed_increment, column_null)
    survives = None if column_rank is None else bool(
        column_rank["exceeds_null_p95"] and column_rank["empirical_p"] <= 0.05
    )
    return {
        "measured": True,
        "n_train_rows_used": int(usable.sum()),
        "features": ["rf_score", "quantum_readout_score"],
        "fitted_on": (
            "TRAIN only, on out-of-fold values of both features; standardised with "
            "train statistics"
        ),
        "degenerate_feature_columns": [
            name for name, s in zip(("rf_score", "quantum_readout_score"), raw_std)
            if not s > 1e-12
        ],
        "coefficients": {
            "rf_score": round(float(stack_model.coef_.ravel()[0]), 6),
            "quantum_readout_score": round(float(stack_model.coef_.ravel()[1]), 6),
            "note": (
                "on standardised features, so the magnitudes are directly comparable. A "
                "quantum coefficient near zero means the stack found nothing to use."
            ),
        },
        "validation_metrics": {
            "rf_score_alone": rf_raw,
            "quantum_score_alone": quantum_alone,
            "stack": stacked,
        },
        "increment": {
            "rf_only_logistic_pr_auc": None if rf_model_pr is None else round(rf_model_pr, 6),
            "stack_pr_auc": None if stack_pr is None else round(stack_pr, 6),
            "stack_minus_rf_only": (
                None if observed_increment is None else round(observed_increment, 6)
            ),
            "note": (
                "Both sides are logistic models fitted on the same standardised "
                "out-of-fold train features; they differ only in whether the quantum "
                "column is present. `rf_only_logistic_pr_auc` should equal the raw RF "
                "score's PR-AUC whenever the fitted coefficient is positive, since a "
                "one-feature logistic is monotone in its feature."
            ),
        },
        "raw_rf_score_pr_auc": rf_pr,
        "score_correlation": {
            "pearson": round(pearson, 6) if np.isfinite(pearson) else None,
            "spearman": round(spearman, 6) if np.isfinite(spearman) else None,
            "note": (
                "computed on validation scores. High correlation means the quantum score "
                "is re-deriving the forest's ranking rather than adding to it."
            ),
        },
        "permutation_null": {
            "n_requested": int(n_permutations),
            "n_usable": len(null),
            "quantity": "stack PR-AUC minus RF-only PR-AUC, both refitted per draw",
            "blocking": "patient-level label permutation within TRAIN",
            "increment": _null_summary(null),
            "observed_increment_vs_null": rank,
            "gates_the_claim": False,
            "power_note": (
                "Reported for consistency with every other E0 null, but structurally "
                "low-powered for an increment: a stack refitted on permuted labels has "
                "random coefficient signs and is then scored against the real validation "
                "labels, so roughly one draw in four reproduces the observed increment by "
                "luck and the empirical p is floored near 0.25. A p above 0.05 here is "
                "therefore not evidence against complementarity. "
                "`column_permutation_null` is the gating test."
            ),
        },
        "column_permutation_null": {
            "n_requested": int(n_permutations),
            "n_usable": len(column_null),
            "quantity": (
                "stack PR-AUC with a permuted quantum column, minus the unchanged "
                "RF-only PR-AUC"
            ),
            "blocking": column_null_note,
            "increment": _null_summary(column_null),
            "observed_increment_vs_null": column_rank,
            "gates_the_claim": True,
        },
        "stack_beats_rf_alone": beats_rf,
        "gain_survives_its_own_null": survives,
        "gain_survives_label_permutation_null": label_null_survives,
        "complementarity_claimed": (
            None if beats_rf is None or survives is None else bool(beats_rf and survives)
        ),
    }


# ------------------------------------------------------------------------ per condition
def _restrict(metrics_rows: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(metrics_rows, dtype=np.float64)[mask]


def run_condition_controls(
    *,
    condition: str,
    train: PixelPartition,
    validation: PixelPartition,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    rf: Dict[str, Any],
    frozen: Any,
    best: Optional[Dict[str, Any]],
    seed: int,
    chunk: int,
    n_permutations: int,
    permutation_note: Optional[str],
) -> Dict[str, Any]:
    """Every §9 control for one ROI condition, on aligned rows."""
    y_train = train.labels[train_rows]
    y_validation = validation.labels[validation_rows]
    patients_train = [train.patient_ids[int(r)] for r in train_rows]
    train_ids = [train.image_ids[int(r)] for r in train_rows]
    validation_ids = [validation.image_ids[int(r)] for r in validation_rows]

    entry: Dict[str, Any] = {
        "condition": condition,
        "description": CONDITION_DESCRIPTIONS.get(condition, ""),
        "train": {
            "n_samples": int(train_rows.size),
            "n_positive": int(y_train.sum()),
            "n_patients": len(set(patients_train)),
            "prevalence": round(float(y_train.mean()), 6) if train_rows.size else None,
        },
        "validation": {
            "n_samples": int(validation_rows.size),
            "n_positive": int(y_validation.sum()),
            "prevalence": (
                round(float(y_validation.mean()), 6) if validation_rows.size else None
            ),
        },
        "permutation_budget_note": permutation_note,
        "controls": {},
    }
    if y_train.size == 0 or len(set(y_train.tolist())) < 2:
        entry["skipped"] = (
            "Train rows for this condition are single-class; no control can be fitted "
            "and metrics are undefined. Reported as counts rather than as a number."
        )
        return entry

    amplitudes_train = train.amplitudes(train_rows)
    amplitudes_validation = validation.amplitudes(validation_rows)

    # ---- amplitude-space controls, on exactly the rows E0.1 used
    controls = amplitude_controls(
        amplitudes_train, y_train, amplitudes_validation, y_validation,
        patients_train, seed=seed,
    )
    arrays = controls.pop("_arrays", {})
    for name, key in (("linear_svm_on_amplitudes", "linear"), ("rbf_svm_on_amplitudes", "rbf")):
        block = controls[name]
        if not block.get("measured") or key not in arrays:
            entry["controls"][name] = block
            continue
        raw = arrays[key]
        probability = raw["validation_probability"]
        block["metrics_rank"] = _metrics(
            y_validation, raw["validation_decision"]
        )
        block["metrics"] = (
            None if probability is None else _metrics(y_validation, probability)
        )
        block["metrics_note"] = (
            "`metrics_rank` is computed on the raw SVM decision function: PR-AUC and "
            "ROC-AUC are rank statistics and are exact there, while Brier and ECE in "
            "that block are meaningless and should be ignored. `metrics` is the same "
            "model through the train-only Platt map, where all four are defined."
        )
        entry["controls"][name] = block

    # ---- classical reference, on the intersection with the descriptor cache
    rf_validation_by_id = rf["validation_by_id"]
    rf_train_by_id = rf["train_oof_by_id"]
    validation_mask = np.asarray([i in rf_validation_by_id for i in validation_ids])
    train_mask = np.asarray([i in rf_train_by_id for i in train_ids])
    rf_validation = np.asarray(
        [rf_validation_by_id[i] for i in validation_ids if i in rf_validation_by_id],
        dtype=np.float64,
    )
    entry["controls"]["random_forest"] = {
        "measured": True,
        **rf["record"],
        "n_scored": int(validation_mask.sum()),
        "n_not_in_descriptor_cache": int((~validation_mask).sum()),
        "metrics": _metrics(y_validation[validation_mask], rf_validation),
    }

    # ---- Phase D VQC on these rows, from the frozen artifact
    if frozen is None:
        entry["controls"]["phase_d_vqc"] = {
            "measured": False,
            "reason": "no frozen pixel-VQC artifact for this ROI mode / condition",
        }
    else:
        from quantum_ml.readout import z_string_diagonal

        z0 = measure_batch(
            amplitudes_validation, z_string_diagonal(V1_QUBIT_COUNT, 1)[None, :],
            weights=frozen.weights, n_qubits=V1_QUBIT_COUNT,
            n_layers=frozen.n_layers, chunk=chunk,
        ).ravel()
        phase_d_scores = np.clip((1.0 - z0) / 2.0, 0.0, 1.0)
        entry["controls"]["phase_d_vqc"] = {
            "measured": True,
            "artifact": frozen.artifact.name,
            "sha256": frozen.sha256,
            "weights_trained_on_condition": frozen.condition,
            "weights_borrowed": frozen.condition != condition,
            "n_circuit_parameters": int(frozen.weights.size),
            "score_map": "(1 - <Z0>)/2, uncalibrated",
            "metrics": _metrics(y_validation, phase_d_scores),
            "z0_range": [round(float(z0.min()), 7), round(float(z0.max()), 7)],
            "score_band_width": round(
                float(phase_d_scores.max() - phase_d_scores.min()), 7
            ),
        }

    # ---- head-to-head on the intersection, so prevalence is identical
    aligned = {
        "n_validation_rows": int(validation_mask.sum()),
        "n_dropped_not_in_descriptor_cache": int((~validation_mask).sum()),
        "prevalence": (
            round(float(y_validation[validation_mask].mean()), 6)
            if validation_mask.any() else None
        ),
        "note": (
            "PR-AUC is prevalence-sensitive, so the amplitude controls are re-reported "
            "here on exactly the rows the random forest could score. The unrestricted "
            "numbers above are the ones directly comparable to E0.1."
        ),
        "random_forest_pr_auc": None,
        "amplitude_controls_pr_auc": {},
    }
    rf_aligned_metrics = _metrics(y_validation[validation_mask], rf_validation)
    aligned["random_forest_pr_auc"] = (
        None if not rf_aligned_metrics else rf_aligned_metrics.get("pr_auc")
    )
    for name, key in (("linear_svm_on_amplitudes", "linear"), ("rbf_svm_on_amplitudes", "rbf")):
        if key not in arrays:
            continue
        restricted = _metrics(
            y_validation[validation_mask],
            _restrict(arrays[key]["validation_decision"], validation_mask),
        )
        aligned["amplitude_controls_pr_auc"][name] = (
            None if not restricted else restricted.get("pr_auc")
        )
    entry["row_alignment"] = aligned

    # ---- §9.2 complementarity
    if best is None:
        entry["complementarity"] = {
            "measured": False,
            "reason": (
                "no E0.1 payload was supplied, so there is no selected quantum readout "
                "to stack. Pass --e0 <reports_e0_*.json>."
            ),
        }
        return entry

    folds, fold_note = grouped_folds(y_train, patients_train, seed=seed)
    quantum = quantum_scores_for(
        variant=best["variant"], weight_setting=best["weight_setting"],
        amplitudes_train=amplitudes_train, y_train=y_train,
        amplitudes_validation=amplitudes_validation, patients_train=patients_train,
        frozen=frozen, folds=folds, seed=seed, chunk=chunk,
    )
    entry["selected_quantum_readout"] = {
        k: v for k, v in quantum.items() if not isinstance(v, np.ndarray)
    }
    entry["selected_quantum_readout"]["selected_on"] = (
        "highest validation PR-AUC among the pre-registered E0.1 variant x weight "
        f"settings for this condition ({best['pr_auc']:.6f} in the E0.1 payload)"
    )
    reproduced = _metrics(y_validation, quantum["validation"])
    entry["selected_quantum_readout"]["reproduced_validation_pr_auc"] = (
        None if not reproduced else reproduced.get("pr_auc")
    )
    entry["selected_quantum_readout"]["stack_fold_note"] = fold_note

    rf_train_oof = np.asarray(
        [rf_train_by_id.get(i, np.nan) for i in train_ids], dtype=np.float64
    )
    patients_validation = [
        validation.patient_ids[int(r)] for r in validation_rows[validation_mask]
    ]
    entry["complementarity"] = complementarity(
        rf_train_oof, quantum["train_out_of_fold"], y_train,
        rf_validation, _restrict(quantum["validation"], validation_mask),
        y_validation[validation_mask], patients_train,
        seed=seed, n_permutations=n_permutations,
        patients_validation=patients_validation,
    )
    entry["complementarity"]["row_note"] = (
        f"Fitted on the {int(train_mask.sum())} of {train_rows.size} train rows present "
        f"in both caches and evaluated on the {int(validation_mask.sum())} of "
        f"{validation_rows.size} validation rows the forest could score. Both models see "
        "identical rows."
    )
    return entry


# --------------------------------------------------------------------------- the sweep
def run(
    *,
    roi_mode: str = "oracle",
    config: Optional[Settings] = None,
    dataset: Optional[LoadedPixelDataset] = None,
    model_version: Optional[str] = None,
    vqc_version: Optional[str] = None,
    e0_payload: Optional[Dict[str, Any]] = None,
    conditions: Optional[Sequence[str]] = None,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 42,
    chunk: int = 128,
) -> Dict[str, Any]:
    """Run the §9 controls for one ROI mode. Reads train and validation only."""
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)
    data = dataset or load_pixel_partitions(config=cfg, roi_mode=roi_mode)
    if data.qubit_count != V1_QUBIT_COUNT:
        raise E0Error(
            f"Pixel cache reports {data.qubit_count} qubits; V1 is fixed at "
            f"{V1_QUBIT_COUNT}."
        )

    from backend.training.train_pixel_vqc import PIXEL_VQC_VERSION

    directory = store.pixel_vqc_dir(vqc_version or PIXEL_VQC_VERSION)
    train_conditions = condition_rows(data.train)
    validation_conditions = condition_rows(data.validation)
    primary = PRIMARY_CONDITION.get(roi_mode)

    names = list(conditions) if conditions else sorted(
        set(train_conditions) & set(validation_conditions)
    )
    if primary in names:
        names = [primary] + [n for n in names if n != primary]

    logger.info("Refitting the classical reference and its out-of-fold train scores...")
    rf = random_forest_scores(config=cfg, model_version=model_version, seed=seed)

    out: List[Dict[str, Any]] = []
    budget_notes: Dict[str, str] = {}
    for condition in names:
        if condition not in train_conditions or condition not in validation_conditions:
            continue
        frozen = load_frozen_model(directory, roi_mode, condition)
        if frozen is None and primary:
            frozen = load_frozen_model(directory, roi_mode, primary)

        n_train_rows = int(train_conditions[condition].size)
        condition_permutations = n_permutations
        note: Optional[str] = None
        if (
            n_train_rows > LARGE_CONDITION_TRAIN_ROWS
            and n_permutations > LARGE_CONDITION_PERMUTATIONS
        ):
            condition_permutations = LARGE_CONDITION_PERMUTATIONS
            note = (
                f"reduced from {n_permutations} to {condition_permutations} draws: "
                f"{n_train_rows} train rows exceeds the "
                f"{LARGE_CONDITION_TRAIN_ROWS}-row threshold. Finest resolvable "
                f"empirical p is {1 / (condition_permutations + 1):.4f}. Same threshold "
                "as the E0.1 sweep, so the two nulls are read on the same footing."
            )
            budget_notes[condition] = note
            logger.info("  %-12s stack null %s", condition, note)

        started = time.perf_counter()
        entry = run_condition_controls(
            condition=condition,
            train=data.train,
            validation=data.validation,
            train_rows=train_conditions[condition],
            validation_rows=validation_conditions[condition],
            rf=rf,
            frozen=frozen,
            best=best_e0_setting(e0_payload, condition),
            seed=seed,
            chunk=chunk,
            n_permutations=condition_permutations,
            permutation_note=note,
        )
        entry["runtime_seconds"] = round(time.perf_counter() - started, 3)
        logger.info("  %-16s controls in %.1fs", condition, entry["runtime_seconds"])
        out.append(entry)

    return {
        "controls_version": CONTROLS_VERSION,
        "experiment": "E0 section 9 classical controls and complementarity test",
        "roi_mode": roi_mode,
        "partitions_read": ["train", "validation"],
        "test_partition_used": False,
        "n_qubits": V1_QUBIT_COUNT,
        "amplitudes_per_state": V1_PIXEL_COUNT,
        "controls": {
            "random_forest": "frozen BaselineRecord, refit on TRAIN, 181 fused features",
            "linear_svm_on_amplitudes": (
                "linear kernel on the identical 65,536-dim amplitude vectors, C by "
                "patient-grouped CV inside TRAIN -- redone, not quoted"
            ),
            "rbf_svm_on_amplitudes": (
                "same Gram, gamma by median heuristic on TRAIN distances, C the same way"
            ),
            "phase_d_vqc": "the frozen artifact re-scored on these rows, sha256 recorded",
        },
        "hyperparameter_discipline": {
            "C_grid": list(C_GRID),
            "selection": "within-TRAIN patient-grouped CV, scoring average_precision",
            "no_validation_selection": (
                "no control's C, gamma, threshold or calibration map was chosen using a "
                "validation row. The only validation-based selection anywhere in this "
                "module is which E0.1 variant to stack, which the spec permits."
            ),
            "threshold": {
                "value": FIXED_THRESHOLD,
                "selection": "fixed, not swept",
            },
        },
        "e0_payload_supplied": bool(e0_payload),
        "permutation_budget": {
            "requested": int(n_permutations),
            "large_condition_train_rows_threshold": LARGE_CONDITION_TRAIN_ROWS,
            "large_condition_permutations": LARGE_CONDITION_PERMUTATIONS,
            "reduced_conditions": budget_notes,
        },
        "seeds": {"experiment": seed, "permutation_base": seed},
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "conditions": out,
    }


def verdict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The §9 reading: best control per condition, and whether the quantum score adds.

    Stated as measured comparisons with margins. Where something was not measured the
    answer is "not measured", never a default that happens to favour one side.
    """
    rows: List[Dict[str, Any]] = []
    for entry in payload["conditions"]:
        if entry.get("skipped"):
            rows.append({
                "condition": entry["condition"],
                "skipped": entry["skipped"],
            })
            continue
        best_name, best_pr = None, None
        for name, block in entry.get("controls", {}).items():
            metrics = block.get("metrics") or block.get("metrics_rank")
            if not metrics or metrics.get("pr_auc") is None:
                continue
            pr = float(metrics["pr_auc"])
            if best_pr is None or pr > best_pr:
                best_name, best_pr = name, pr
        quantum = entry.get("selected_quantum_readout") or {}
        quantum_pr = quantum.get("reproduced_validation_pr_auc")
        comp = entry.get("complementarity") or {}
        rows.append({
            "condition": entry["condition"],
            "validation_n": entry["validation"]["n_samples"],
            "validation_prevalence": entry["validation"]["prevalence"],
            "best_control": best_name,
            "best_control_pr_auc": None if best_pr is None else round(best_pr, 6),
            "control_pr_auc": {
                name: (
                    (block.get("metrics") or block.get("metrics_rank") or {}).get("pr_auc")
                )
                for name, block in entry.get("controls", {}).items()
            },
            "selected_quantum": (
                None if not quantum
                else f"{quantum.get('variant')} / {quantum.get('weight_setting')}"
            ),
            "quantum_pr_auc": quantum_pr,
            "quantum_beats_best_control": (
                None if quantum_pr is None or best_pr is None
                else bool(float(quantum_pr) > best_pr)
            ),
            "margin_vs_best_control": (
                None if quantum_pr is None or best_pr is None
                else round(float(quantum_pr) - best_pr, 6)
            ),
            "complementarity_claimed": comp.get("complementarity_claimed"),
            "stack_minus_rf_only_pr_auc": (comp.get("increment") or {}).get(
                "stack_minus_rf_only"
            ),
            # The gating p is the column null's. The label-permutation p is carried
            # beside it rather than dropped, so a reader can see both and is not asked to
            # take the gate's choice of null on trust.
            "increment_empirical_p": (
                (comp.get("column_permutation_null") or {}).get(
                    "observed_increment_vs_null"
                ) or {}
            ).get("empirical_p"),
            "increment_empirical_p_label_null": (
                (comp.get("permutation_null") or {}).get("observed_increment_vs_null") or {}
            ).get("empirical_p"),
            "quantum_rf_spearman": (comp.get("score_correlation") or {}).get("spearman"),
        })
    return {
        "per_condition": rows,
        "gate_rule": (
            "Per the governing spec, a readout improvement does not validate the "
            "amplitude encoding. The encoding is credited only where the quantum score "
            "beats the best control on identical rows, or where the [RF, quantum] stack "
            "beats an RF-only logistic AND that increment survives its null. The null is "
            "on the increment, not on the stack's absolute PR-AUC, because the RF feature "
            "alone would clear the latter. The gating null permutes the quantum column "
            "patient-blocked and leaves labels and RF intact; the label-permutation null "
            "is reported beside it but does not gate, because refitting on permuted "
            "labels randomises the coefficient signs and floors its p near 0.25. Both "
            "conditions are reported above; neither is inferred from the other."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="E0 section 9 classical controls and the complementarity test."
    )
    parser.add_argument("--roi", dest="roi_mode", default="oracle",
                        choices=("oracle", "predicted"))
    parser.add_argument("--condition", action="append", default=None,
                        help="Restrict to one condition; repeatable.")
    parser.add_argument("--e0", default=None,
                        help="E0.1 payload JSON, used to pick the readout to stack.")
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--vqc-version", default=None)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--out", default=None, help="Write the full payload here.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("qiskit", "stevedore", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    e0_payload = None
    if args.e0:
        e0_payload = json.loads(Path(args.e0).read_text(encoding="utf-8"))
        if e0_payload.get("test_partition_used"):
            raise ControlsError(
                f"{args.e0} reports test_partition_used=True; refusing to build the "
                "decision gate on a payload that read the held-out set."
            )

    payload = run(
        roi_mode=args.roi_mode,
        model_version=args.model_version,
        vqc_version=args.vqc_version,
        e0_payload=e0_payload,
        conditions=args.condition,
        n_permutations=args.permutations,
        seed=args.seed,
        chunk=args.chunk,
    )
    payload["verdict"] = verdict(payload)
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Full payload -> %s", args.out)
    print(json.dumps(payload["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
