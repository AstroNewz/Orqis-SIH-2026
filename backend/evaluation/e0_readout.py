"""Phase E0.1 -- does the 16-qubit state hold signal that :math:`Z_0` discarded?

Implements the experiment specified in ``docs/PHASE_E0_DIAGNOSTIC_SPEC.md`` and recorded
as a controlled research decision in DEC-031. **Only the measurement/readout varies.**
The encoding, the 16 qubits, the L1 ansatz, the 32 circuit parameters, the patient
split, the preprocessing, the loss and the seed are all held at their frozen Phase D
values, and the invariants are asserted rather than assumed.

The design in one table:

=========  =============================  =====  ===================================
Variant    Observables                    Count  Readout
=========  =============================  =====  ===================================
``A0``     ``Z_0``                            1  fixed ``(1 - <Z_0>)/2`` (Phase D)
``A1``     ``Z_0``                            1  standardised -> logistic
``B``      ``Z_k``, k = 0..15                16  standardised -> logistic
``C``      ``B`` + ``Z_i Z_{i+8}``           24  standardised -> logistic
``D``      ``B`` + all ``Z_i Z_j``          136  standardised -> logistic
=========  =============================  =====  ===================================

``A1`` is not decoration. Without it, any gain of ``B`` over ``A0`` is confounded
between *having more observables* and *the readout merely becoming trainable*, so
``B - A1`` is the quantity that attributes cleanly to observable count.

Three weight settings, because "the state contains signal" has to be separated from
"the trained circuit found some":

``W-identity``
    No circuit at all -- observables read straight off the encoded amplitudes. This is
    the cleanest separation of a representation failure from a readout failure
    available: if nothing separates the classes here, no readout on top of this
    encoding can, and ``Z_0`` was not the binding constraint.
``W-untrained``
    The seeded initial draw, so a positive result cannot be credited to training.
``W-frozen``
    The selected Phase D weights, sha256-pinned per condition.

Everything is measured against a **patient-blocked permutation null**. One FWHT would
hand back all 65,536 Z-string expectations, so cheap access to observables is a
multiple-comparisons trap, and variant D puts 136 features against 103 train positives.
Observable sets are therefore pre-registered constants in :mod:`quantum_ml.readout`,
never chosen here, and a variant whose observed PR-AUC lands inside its own null is
reported as *no detected signal* however it compares to ``A0``.

Discipline this module enforces in code rather than in review:

- The scaler and the readout see **train rows only**; validation is scored once.
- Regularisation strength is chosen by **patient-grouped** CV *inside train*, so a
  patient cannot span a CV fold any more than it can span a partition.
- Observables whose train sd falls below :data:`VARIANCE_FLOOR` are dropped and
  **counted**, because standardising a degenerate observable amplifies float noise into
  an apparent feature.
- The **test partition is never read.** ``test_partition_used`` is ``False`` in the
  payload and a source-inspection test asserts the absence.

Usage::

    python -m backend.evaluation.e0_readout --roi oracle
    python -m backend.evaluation.e0_readout --roi predicted --permutations 200
    python -m backend.evaluation.e0_readout --roi oracle --gradients --plateau
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.evaluation.metrics import evaluate_predictions
from backend.ml.artifacts import ArtifactStore
from backend.ml.pixel_pipeline import V1_PIXEL_COUNT, V1_QUBIT_COUNT
from backend.training.pixel_data import (
    CONDITION_DESCRIPTIONS,
    LoadedPixelDataset,
    PixelPartition,
    condition_rows,
    load_pixel_partitions,
)
from quantum_ml.readout import (
    VARIANT_SIZES,
    ObservableSet,
    measure_batch,
    observable_set,
)

logger = logging.getLogger(__name__)

E0_VERSION = "v1-e0-readout-1"

#: Variants in report order. ``A0`` first because every other variant is read as a
#: difference from it.
VARIANTS: Tuple[str, ...] = ("A0", "A1", "B", "C", "D")

#: Weight settings in increasing order of "how much circuit is involved".
WEIGHT_SETTINGS: Tuple[str, ...] = ("W-identity", "W-untrained", "W-frozen")

#: Train-sd below which an observable is excluded rather than standardised. Dividing by
#: a sd of order 1e-16 turns floating-point residue into a unit-variance "feature" that
#: a logistic fit will happily use, and the resulting validation number would be noise
#: wearing a coefficient.
VARIANCE_FLOOR = 1e-9

#: Permutation-null size. 200 gives a p95/p99 that is not itself dominated by
#: resampling noise while keeping the whole sweep inside a few minutes.
DEFAULT_PERMUTATIONS = 200

#: Permutation-null cost control for the large conditions.
#:
#: Each permutation refits the whole readout pipeline, inner grouped CV included, so the
#: cost scales with (train rows) x (observables) x (grid) x (folds). At the primary
#: conditions that is trivial -- 215 train rows at oracle -- but ``A_all`` has 1,692 and
#: ``B_and_C_all`` 1,620, where the full 200 draws take hours per weight setting. Those
#: are the *secondary* conditions (``A_all`` carries the DEC-024 oracle-selection leak
#: and can only ever be read with that caveat), and a 50-draw null still resolves
#: p = 1/51 = 0.0196, finer than any decision this experiment makes.
#:
#: So the budget is reduced above a stated row threshold. It is logged at INFO and
#: recorded per condition in the payload as ``permutation_budget``: a cap that is not
#: visible in the output reads as "we ran the full protocol" when we did not.
LARGE_CONDITION_TRAIN_ROWS = 400
LARGE_CONDITION_PERMUTATIONS = 50

#: Fixed decision threshold for the fitted readouts. Deliberately *not* selected: a
#: swept threshold is one more thing chosen on validation, and the logistic boundary at
#: 0.5 is the natural operating point of a balanced fit. Reported as fixed.
FIXED_THRESHOLD = 0.50

#: Conditions the report leads with, per ROI mode: the cheap near-balanced honest one,
#: then the deployable one.
PRIMARY_CONDITION = {"oracle": "A_lesion_polygon", "predicted": "B_localized"}


class E0Error(RuntimeError):
    """Raised when a frozen invariant does not hold, or an artifact is unusable."""


# ------------------------------------------------------------------ standardisation
@dataclass
class Scaler:
    """Train-only feature standardiser with an explicit degeneracy floor.

    Holds the mean and sd of the **train** rows and nothing else. Constructed by
    :meth:`fit` from train data; :meth:`transform` is the only way validation rows
    reach a readout, so "the scaler never saw validation" is a structural property of
    this class rather than a claim about call order.
    """

    mean: np.ndarray
    std: np.ndarray
    keep: np.ndarray
    n_excluded: int
    excluded_labels: List[str] = field(default_factory=list)

    @classmethod
    def fit(cls, x: np.ndarray, labels: Sequence[str]) -> "Scaler":
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        mean = x.mean(axis=0)
        std = x.std(axis=0, ddof=0)
        keep = std > VARIANCE_FLOOR
        if not keep.any():
            # Every observable is degenerate on these rows. Refusing beats returning a
            # zero-width design matrix that a downstream fit would silently accept.
            raise E0Error(
                "Every observable has train sd below the variance floor "
                f"({VARIANCE_FLOOR:g}); there is nothing to standardise."
            )
        return cls(
            mean=mean,
            std=std,
            keep=keep,
            n_excluded=int((~keep).sum()),
            excluded_labels=[
                name for name, kept in zip(labels, keep) if not kept
            ],
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return (x[:, self.keep] - self.mean[self.keep]) / self.std[self.keep]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "kept": [bool(v) for v in self.keep],
            "n_excluded_below_variance_floor": self.n_excluded,
            "excluded_labels": self.excluded_labels,
            "variance_floor": VARIANCE_FLOOR,
            "fitted_on": "train rows only",
        }


# ------------------------------------------------------------------------- readouts
@dataclass
class Readout:
    """A fitted readout: either the frozen Phase D map or a logistic head.

    ``kind == "fixed_z0"`` is variant ``A0`` and has no parameters at all -- it is the
    exact ``(1 - <Z_0>)/2`` clip Phase D used, reproduced here so every other variant is
    measured as a difference from the thing that actually failed.
    """

    kind: str
    scaler: Optional[Scaler] = None
    coefficients: Optional[np.ndarray] = None
    intercept: float = 0.0
    regularisation_c: Optional[float] = None
    cv_note: Optional[str] = None

    @property
    def n_trainable(self) -> int:
        if self.kind == "fixed_z0":
            return 0
        return int(self.coefficients.size + 1) if self.coefficients is not None else 0

    def scores(self, observables: np.ndarray) -> np.ndarray:
        observables = np.atleast_2d(np.asarray(observables, dtype=np.float64))
        if self.kind == "fixed_z0":
            return np.clip((1.0 - observables[:, 0]) / 2.0, 0.0, 1.0)
        assert self.scaler is not None and self.coefficients is not None
        eta = self.scaler.transform(observables) @ self.coefficients + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -60.0, 60.0)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "n_trainable_parameters": self.n_trainable,
            "intercept": float(self.intercept),
            "coefficients": (
                None if self.coefficients is None
                else [float(c) for c in self.coefficients]
            ),
            "regularisation_C": self.regularisation_c,
            "selection": self.cv_note,
            "scaler": None if self.scaler is None else self.scaler.to_dict(),
        }


def fit_readout(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: Sequence[str],
    *,
    variant: str,
    labels: Sequence[str],
    seed: int,
    n_jobs: int = 1,
) -> Readout:
    """Fit the readout for ``variant`` on train rows only.

    ``A0`` returns the frozen map untouched. Everything else standardises with train
    statistics and fits an L2 logistic head whose ``C`` is chosen by **patient-grouped**
    stratified CV inside train. Grouping matters: several images share a patient, so an
    ungrouped fold would put the same patient on both sides of the CV split and pick a
    ``C`` that flatters itself.

    ``n_jobs`` defaults to 1 deliberately. The grid is 6 x 5 = 30 fits on a few hundred
    rows, and dispatching those through a process pool costs more than running them --
    measured at 2.00 s per search with ``n_jobs=-1`` against 0.55 s serial on the primary
    condition. The parallelism that pays is one level up, over permutation draws, where
    each task is a whole search.
    """
    if variant == "A0":
        return Readout(kind="fixed_z0", cv_note="frozen Phase D map; nothing fitted")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    scaler = Scaler.fit(x_train, labels)
    z_train = scaler.transform(x_train)

    grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    base = LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=5000,
        class_weight="balanced", random_state=seed,
    )
    cv_note: str
    splitter: Any
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        n_splits = min(5, int(np.bincount(np.asarray(y_train, dtype=int)).min()),
                       len(set(groups)))
        if n_splits < 2:
            raise ValueError("not enough groups or positives for grouped CV")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv_note = (
            f"L2 logistic; C chosen by {n_splits}-fold StratifiedGroupKFold on TRAIN, "
            "grouped by patient, scoring average_precision"
        )
        folds = list(splitter.split(z_train, y_train, groups=list(groups)))
    except Exception as exc:  # noqa: BLE001 -- degrade to a stated fallback, never silently
        from sklearn.model_selection import StratifiedKFold

        n_splits = max(2, min(5, int(np.bincount(np.asarray(y_train, dtype=int)).min())))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(z_train, y_train))
        cv_note = (
            f"L2 logistic; C chosen by {n_splits}-fold StratifiedKFold on TRAIN "
            f"(patient grouping unavailable: {type(exc).__name__}), "
            "scoring average_precision"
        )

    search = GridSearchCV(
        base, grid, scoring="average_precision", cv=folds, refit=True, n_jobs=n_jobs
    )
    search.fit(z_train, y_train)
    best = search.best_estimator_
    return Readout(
        kind="logistic",
        scaler=scaler,
        coefficients=np.asarray(best.coef_, dtype=np.float64).ravel(),
        intercept=float(np.asarray(best.intercept_).ravel()[0]),
        regularisation_c=float(search.best_params_["C"]),
        cv_note=cv_note,
    )


# --------------------------------------------------------------------- permutations
def patient_blocked_permutations(
    labels: np.ndarray, patients: Sequence[str], *, n: int, seed: int
) -> List[np.ndarray]:
    """``n`` label vectors with the patient block structure preserved.

    Images are not independent -- a patient contributes several, and within a patient
    the diagnosis is constant. Shuffling image labels would break that and produce a
    null that is *easier* than reality, making a chance result look significant. So the
    permutation acts on the patient-level label vector and is broadcast back to images.
    """
    labels = np.asarray(labels, dtype=int)
    unique: List[str] = []
    seen: Dict[str, int] = {}
    for p in patients:
        if p not in seen:
            seen[p] = len(unique)
            unique.append(p)
    index = np.asarray([seen[p] for p in patients], dtype=int)
    per_patient = np.zeros(len(unique), dtype=int)
    for slot in range(len(unique)):
        rows = labels[index == slot]
        per_patient[slot] = int(round(float(rows.mean()))) if rows.size else 0

    rng = np.random.default_rng(seed)
    out: List[np.ndarray] = []
    for _ in range(n):
        out.append(rng.permutation(per_patient)[index])
    return out


def _null_draw(
    x_train: np.ndarray,
    permuted: np.ndarray,
    patients_train: Sequence[str],
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    variant: str,
    labels: Sequence[str],
    seed: int,
) -> Tuple[Optional[float], Optional[float]]:
    """One permutation draw: refit the readout on shuffled labels, score validation.

    Module-level so a process pool can pickle it. The *whole* pipeline is refitted,
    inner grouped CV included, rather than reusing the ``C`` selected on the real
    labels: a null that skips hyperparameter selection is less flexible than the
    observed model it is compared against, which biases towards calling chance a
    finding. Refitting is the expensive choice and the correct one.

    A degenerate draw -- one where every observable collapses below the variance floor,
    or the shuffled labels leave a fold single-class -- returns ``(None, None)`` and is
    counted out of ``n_usable`` instead of silently becoming a zero.
    """
    try:
        shuffled = fit_readout(
            x_train, permuted, patients_train,
            variant=variant, labels=labels, seed=seed,
        )
    except Exception:  # noqa: BLE001 -- a degenerate permutation is dropped, not faked
        return None, None
    metrics = _metrics(y_validation, shuffled.scores(x_validation))
    if not metrics:
        return None, None
    pr = metrics.get("pr_auc")
    roc = metrics.get("roc_auc")
    return (
        None if pr is None else float(pr),
        None if roc is None else float(roc),
    )


def _null_summary(values: List[float]) -> Optional[Dict[str, Any]]:
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    if finite.size == 0:
        return None
    return {
        "n": int(finite.size),
        "mean": round(float(finite.mean()), 6),
        "sd": round(float(finite.std(ddof=1)) if finite.size > 1 else 0.0, 6),
        "p50": round(float(np.percentile(finite, 50)), 6),
        "p95": round(float(np.percentile(finite, 95)), 6),
        "p99": round(float(np.percentile(finite, 99)), 6),
        "max": round(float(finite.max()), 6),
    }


def _rank_in_null(observed: Optional[float], null: List[float]) -> Optional[Dict[str, Any]]:
    if observed is None or not np.isfinite(observed):
        return None
    finite = np.asarray([v for v in null if v is not None and np.isfinite(v)])
    if finite.size == 0:
        return None
    n_ge = int((finite >= observed).sum())
    return {
        "n_null_at_or_above_observed": n_ge,
        # (n_ge + 1) / (n + 1) is the standard permutation p-value; it cannot be 0,
        # which is the honest thing at 200 permutations.
        "empirical_p": round(float((n_ge + 1) / (finite.size + 1)), 6),
        "exceeds_null_p95": bool(observed > float(np.percentile(finite, 95))),
        "z_vs_null": (
            None if finite.size < 2 or finite.std(ddof=1) == 0
            else round(float((observed - finite.mean()) / finite.std(ddof=1)), 4)
        ),
    }


# ----------------------------------------------------------------------- separability
def _auc(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    """ROC-AUC by rank, or ``None`` when the sample is single-class."""
    y = np.asarray(y, dtype=int)
    if y.size == 0 or len(set(y.tolist())) < 2:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    sorted_scores = score[order]
    i = 0
    while i < score.size:
        j = i
        while j + 1 < score.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def observable_report(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    oset: ObservableSet,
) -> List[Dict[str, Any]]:
    """Per-observable distribution and univariate separability, train and validation."""
    out: List[Dict[str, Any]] = []
    for j, label in enumerate(oset.labels):
        col_t = x_train[:, j]
        col_v = x_validation[:, j]
        pos, neg = col_t[y_train == 1], col_t[y_train == 0]
        std = float(col_t.std(ddof=0))
        out.append({
            "label": label,
            "mask": int(oset.masks[j]),
            "weight": int(bin(int(oset.masks[j])).count("1")),
            "spatial_meaning": oset.spatial_notes[j],
            "train": {
                "n": int(col_t.size),
                "mean": float(col_t.mean()),
                "sd": std,
                "min": float(col_t.min()),
                "p05": float(np.percentile(col_t, 5)),
                "median": float(np.median(col_t)),
                "p95": float(np.percentile(col_t, 95)),
                "max": float(col_t.max()),
                "mean_positive": float(pos.mean()) if pos.size else None,
                "mean_negative": float(neg.mean()) if neg.size else None,
                "standardised_class_gap": (
                    None if std <= VARIANCE_FLOOR or not pos.size or not neg.size
                    else round(float((pos.mean() - neg.mean()) / std), 6)
                ),
                "roc_auc": _auc(y_train, col_t),
            },
            "validation": {
                "n": int(col_v.size),
                "mean": float(col_v.mean()),
                "sd": float(col_v.std(ddof=0)),
                "min": float(col_v.min()),
                "p05": float(np.percentile(col_v, 5)),
                "median": float(np.median(col_v)),
                "p95": float(np.percentile(col_v, 95)),
                "max": float(col_v.max()),
                "roc_auc": _auc(y_validation, col_v),
            },
            "below_variance_floor": bool(std <= VARIANCE_FLOOR),
        })
    return out


# -------------------------------------------------------------------------- gradients
def loss_and_gradient(
    states: np.ndarray,
    labels: np.ndarray,
    readout: Readout,
    oset: ObservableSet,
    weights: np.ndarray,
    *,
    n_qubits: int,
    n_layers: int,
    class_weights: Tuple[float, float],
    chunk: int,
) -> Tuple[float, np.ndarray]:
    """Class-weighted BCE and its exact gradient over the circuit parameters.

    The gradient is taken with the **readout coefficients held fixed** at their
    train-fitted values. That is a deliberate definition, not a convenience: with the
    head refitted at every step the derivative would be implicit and would not answer
    the question E0.1 asks, which is whether a richer measurement *creates a trainable
    gradient where* ``A0`` *had none*. Stated in the payload alongside the number.

    Observable derivatives use the parameter-shift rule. Every gate here is
    ``exp(-i theta P / 2)`` with ``P`` Pauli, so
    ``d<O>/dtheta = (<O>(theta + pi/2) - <O>(theta - pi/2)) / 2`` is exact, not a finite
    difference, and is asserted against a finite difference in the test suite.
    """
    diagonals = oset.diagonals(n_qubits)
    theta = np.asarray(weights, dtype=np.float64).copy()
    labels = np.asarray(labels, dtype=int)
    n = labels.size
    weight_negative, weight_positive = class_weights
    sample_weight = np.where(labels == 1, weight_positive, weight_negative)

    observables = measure_batch(
        states, diagonals, weights=theta, n_qubits=n_qubits,
        n_layers=n_layers, chunk=chunk,
    )
    probabilities = np.clip(readout.scores(observables), 1e-12, 1.0 - 1e-12)
    loss = float(
        -(sample_weight * (labels * np.log(probabilities)
                           + (1 - labels) * np.log(1.0 - probabilities))).mean()
    )

    # dL/d<O_ij>. For the logistic head the chain through eta collapses to the standard
    # (p - y) residual times the coefficient over the train sd. For the fixed A0 map,
    # p = (1 - o)/2 so dp/do = -1/2 and the clip is inactive wherever |o| < 1.
    if readout.kind == "fixed_z0":
        d_probability = -(sample_weight / n) * (
            labels / probabilities - (1 - labels) / (1.0 - probabilities)
        )
        d_observable = np.zeros_like(observables)
        d_observable[:, 0] = d_probability * -0.5
    else:
        assert readout.scaler is not None and readout.coefficients is not None
        residual = (sample_weight / n) * (probabilities - labels)
        scale = readout.coefficients / readout.scaler.std[readout.scaler.keep]
        d_observable = np.zeros_like(observables)
        d_observable[:, readout.scaler.keep] = residual[:, None] * scale[None, :]

    gradient = np.zeros(theta.size, dtype=np.float64)
    shift = np.pi / 2.0
    for k in range(theta.size):
        plus, minus = theta.copy(), theta.copy()
        plus[k] += shift
        minus[k] -= shift
        o_plus = measure_batch(states, diagonals, weights=plus, n_qubits=n_qubits,
                               n_layers=n_layers, chunk=chunk)
        o_minus = measure_batch(states, diagonals, weights=minus, n_qubits=n_qubits,
                                n_layers=n_layers, chunk=chunk)
        gradient[k] = float((d_observable * (o_plus - o_minus) / 2.0).sum())
    return loss, gradient


def gradient_block(
    states: np.ndarray,
    labels: np.ndarray,
    readout: Readout,
    oset: ObservableSet,
    weights: np.ndarray,
    *,
    n_qubits: int,
    n_layers: int,
    class_weights: Tuple[float, float],
    chunk: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    loss, gradient = loss_and_gradient(
        states, labels, readout, oset, weights, n_qubits=n_qubits, n_layers=n_layers,
        class_weights=class_weights, chunk=chunk,
    )
    magnitude = np.abs(gradient)
    return {
        "loss": round(loss, 8),
        "loss_minus_ln2": round(loss - float(np.log(2.0)), 8),
        "grad_l2_norm": float(np.linalg.norm(gradient)),
        "grad_max_abs": float(magnitude.max()),
        "grad_median_abs": float(np.median(magnitude)),
        "grad_min_abs": float(magnitude.min()),
        "per_parameter_abs": [float(v) for v in magnitude],
        "n_parameters": int(gradient.size),
        "method": "exact parameter-shift, readout coefficients held fixed",
        "circuit_evaluations": int(2 * gradient.size + 1),
        "seconds": round(time.perf_counter() - started, 3),
    }


def plateau_probe(
    states: np.ndarray,
    labels: np.ndarray,
    readout: Readout,
    oset: ObservableSet,
    *,
    n_qubits: int,
    n_layers: int,
    class_weights: Tuple[float, float],
    chunk: int,
    n_draws: int,
    seed: int,
) -> Dict[str, Any]:
    """Variance of each ``dL/dtheta_k`` across independent random weight draws.

    The standard barren-plateau indicator. A gradient variance that collapses towards
    zero says the objective is flat almost everywhere in parameter space, which is a
    property of the architecture-plus-readout rather than of any one optimiser -- so it
    bears directly on whether E0.2 is worth running at all.
    """
    rng = np.random.default_rng(seed)
    n_params = 2 * n_qubits * n_layers
    gradients = []
    for _ in range(int(n_draws)):
        draw = rng.uniform(-np.pi, np.pi, n_params)
        _, gradient = loss_and_gradient(
            states, labels, readout, oset, draw, n_qubits=n_qubits, n_layers=n_layers,
            class_weights=class_weights, chunk=chunk,
        )
        gradients.append(gradient)
    stacked = np.stack(gradients)
    variance = stacked.var(axis=0, ddof=1)
    return {
        "n_draws": int(n_draws),
        "seed": int(seed),
        "per_parameter_variance": [float(v) for v in variance],
        "mean_variance": float(variance.mean()),
        "max_variance": float(variance.max()),
        "min_variance": float(variance.min()),
        "mean_abs_gradient": float(np.abs(stacked).mean()),
        "note": (
            "Var_theta[dL/dtheta_k] over uniform(-pi, pi) draws, readout fixed. A "
            "vanishing variance indicates a flat objective across parameter space, not "
            "merely a bad starting point."
        ),
    }


# --------------------------------------------------------------------- frozen weights
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class FrozenModel:
    """The Phase D weights for one condition, with the provenance to prove which."""

    artifact: Path
    sha256: str
    weights: np.ndarray
    n_layers: int
    seed: int
    condition: str
    label: str

    @property
    def untrained_weights(self) -> np.ndarray:
        """The seeded initial draw.

        Reproduces :class:`~quantum_ml.vqc_classifier.VariationalQuantumClassifier`'s
        own initialisation -- a local ``default_rng(seed)`` uniform on ``[-pi, pi]`` --
        so ``W-untrained`` is the state the frozen run actually started from rather than
        an arbitrary random point.
        """
        return np.random.default_rng(self.seed).uniform(
            -np.pi, np.pi, self.weights.size
        )


def load_frozen_model(directory: Path, roi_mode: str, condition: str) -> Optional[FrozenModel]:
    path = directory / f"{roi_mode}_{condition}_pixel_vqc.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    selection = payload["validation_selection"]
    selected = selection["selected_config"]
    weights = np.asarray(payload["selected_weights"], dtype=np.float64)
    n_layers = int(selected["n_layers"])
    expected = 2 * V1_QUBIT_COUNT * n_layers
    if weights.size != expected:
        raise E0Error(
            f"{path.name} carries {weights.size} weights but {V1_QUBIT_COUNT} qubits x "
            f"{n_layers} layers needs {expected}. Refusing to reshape a frozen artifact."
        )
    return FrozenModel(
        artifact=path,
        sha256=_sha256(path),
        weights=weights,
        n_layers=n_layers,
        seed=int(selected["seed"]),
        condition=str(selected["condition"]),
        label=str(selection["selected_label"]),
    )


def class_weights_for(labels: np.ndarray) -> Tuple[float, float]:
    """Inverse-frequency weights normalised to mean 1, as Phase D's loss uses.

    Mirrors :meth:`VariationalQuantumClassifier._class_weights` so the loss values this
    module reports sit on the same scale as the frozen training artifacts' objectives.
    """
    labels = np.asarray(labels, dtype=int)
    n = labels.size
    n_positive = int(labels.sum())
    n_negative = n - n_positive
    if n == 0 or n_positive == 0 or n_negative == 0:
        return 1.0, 1.0
    weight_negative = n / (2.0 * n_negative)
    weight_positive = n / (2.0 * n_positive)
    mean = (weight_negative * n_negative + weight_positive * n_positive) / n
    return weight_negative / mean, weight_positive / mean


# ------------------------------------------------------------------------- the sweep
def _metrics(labels: np.ndarray, scores: np.ndarray) -> Optional[Dict[str, Any]]:
    if labels.size == 0 or len(set(np.asarray(labels).tolist())) < 2:
        return None
    return evaluate_predictions(labels, scores, threshold=FIXED_THRESHOLD).to_dict()


def run_condition(
    *,
    condition: str,
    train: PixelPartition,
    validation: PixelPartition,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    frozen: Optional[FrozenModel],
    n_permutations: int,
    seed: int,
    chunk: int,
    with_gradients: bool,
    with_plateau: bool,
    plateau_draws: int,
    permutation_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Every variant x weight setting for one ROI condition."""
    y_train = train.labels[train_rows]
    y_validation = validation.labels[validation_rows]
    patients_train = [train.patient_ids[int(r)] for r in train_rows]

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
            "n_patients": len({validation.patient_ids[int(r)] for r in validation_rows}),
            "prevalence": (
                round(float(y_validation.mean()), 6) if validation_rows.size else None
            ),
        },
        "frozen_artifact": None if frozen is None else {
            "name": frozen.artifact.name,
            "sha256": frozen.sha256,
            "selected_label": frozen.label,
            "weights_trained_on_condition": frozen.condition,
            "weights_borrowed": frozen.condition != condition,
            "n_layers": frozen.n_layers,
            "n_circuit_parameters": int(frozen.weights.size),
            "seed": frozen.seed,
        },
        "class_weights_negative_positive": [
            round(v, 6) for v in class_weights_for(y_train)
        ],
        "results": [],
    }
    if y_train.size == 0 or len(set(y_train.tolist())) < 2:
        entry["skipped"] = (
            "Train rows for this condition are single-class; a readout cannot be fitted "
            "and metrics are undefined. Reported as counts rather than as a number."
        )
        return entry

    n_layers = frozen.n_layers if frozen else 1
    class_weights = class_weights_for(y_train)
    superset = observable_set("D", V1_QUBIT_COUNT)
    superset_diagonals = superset.diagonals(V1_QUBIT_COUNT)

    amplitudes_train = train.amplitudes(train_rows)
    amplitudes_validation = validation.amplitudes(validation_rows)
    permutations = patient_blocked_permutations(
        y_train, patients_train, n=n_permutations, seed=seed
    ) if n_permutations else []

    for setting in WEIGHT_SETTINGS:
        if setting == "W-identity":
            weights: Optional[np.ndarray] = None
        elif frozen is None:
            continue
        elif setting == "W-untrained":
            weights = frozen.untrained_weights
        else:
            weights = frozen.weights

        # Measured once at the 136-observable superset, then sliced per variant. Every
        # variant is a subset of D, so one forward pass serves all five and the reported
        # runtime separates the shared measurement from the per-variant readout.
        started = time.perf_counter()
        big_train = measure_batch(
            amplitudes_train, superset_diagonals, weights=weights,
            n_qubits=V1_QUBIT_COUNT, n_layers=n_layers, chunk=chunk,
        )
        big_validation = measure_batch(
            amplitudes_validation, superset_diagonals, weights=weights,
            n_qubits=V1_QUBIT_COUNT, n_layers=n_layers, chunk=chunk,
        )
        measure_seconds = time.perf_counter() - started
        column_of = {m: i for i, m in enumerate(superset.masks)}

        for variant in VARIANTS:
            oset = observable_set(variant, V1_QUBIT_COUNT)
            columns = [column_of[m] for m in oset.masks]
            x_train = big_train[:, columns]
            x_validation = big_validation[:, columns]

            fit_started = time.perf_counter()
            readout = fit_readout(
                x_train, y_train, patients_train,
                variant=variant, labels=list(oset.labels), seed=seed,
            )
            fit_seconds = time.perf_counter() - fit_started

            train_metrics = _metrics(y_train, readout.scores(x_train))
            validation_scores = readout.scores(x_validation)
            validation_metrics = _metrics(y_validation, validation_scores)

            null_pr: List[float] = []
            null_roc: List[float] = []
            if permutations:
                # Parallel over draws, serial inside each fit. Draws are independent and
                # each is a whole grouped-CV search, so this is the granularity where a
                # process pool earns its dispatch cost -- unlike inside GridSearchCV,
                # where it measured 3.6x slower than serial on these row counts.
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=-1, prefer="processes")(
                    delayed(_null_draw)(
                        x_train, permuted, patients_train, x_validation, y_validation,
                        variant, list(oset.labels), seed,
                    )
                    for permuted in permutations
                )
                for pr, roc in results:
                    if pr is not None:
                        null_pr.append(pr)
                    if roc is not None:
                        null_roc.append(roc)

            observed_pr = (
                None if not validation_metrics else validation_metrics.get("pr_auc")
            )
            observed_roc = (
                None if not validation_metrics else validation_metrics.get("roc_auc")
            )

            block: Dict[str, Any] = {
                "variant": variant,
                "weight_setting": setting,
                "n_observables": len(oset),
                "n_readout_parameters": readout.n_trainable,
                "n_circuit_parameters": 0 if weights is None else int(weights.size),
                "readout": readout.to_dict(),
                "observables": oset.describe() if variant in ("A0", "C") else None,
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "permutation_null": {
                    "n_requested": int(n_permutations),
                    "n_usable": len(null_pr),
                    "blocking": "patient-level label permutation within TRAIN",
                    "budget_note": permutation_note,
                    "pr_auc": _null_summary(null_pr),
                    "roc_auc": _null_summary(null_roc),
                    "observed_pr_auc_vs_null": _rank_in_null(observed_pr, null_pr),
                    "observed_roc_auc_vs_null": _rank_in_null(observed_roc, null_roc),
                },
                "runtime": {
                    "measurement_seconds_shared_across_variants": round(
                        measure_seconds, 3
                    ),
                    "readout_fit_seconds": round(fit_seconds, 3),
                },
            }
            if variant in ("A0", "B", "D"):
                block["observable_report"] = observable_report(
                    x_train, y_train, x_validation, y_validation, oset
                )
            if with_gradients and weights is not None and variant in ("A0", "C"):
                block["gradient"] = gradient_block(
                    amplitudes_train, y_train, readout, oset, weights,
                    n_qubits=V1_QUBIT_COUNT, n_layers=n_layers,
                    class_weights=class_weights, chunk=chunk,
                )
            if (with_plateau and setting == "W-untrained" and variant in ("A0", "C")):
                block["barren_plateau_probe"] = plateau_probe(
                    amplitudes_train, y_train, readout, oset,
                    n_qubits=V1_QUBIT_COUNT, n_layers=n_layers,
                    class_weights=class_weights, chunk=chunk,
                    n_draws=plateau_draws, seed=seed,
                )
            entry["results"].append(block)

        logger.info(
            "  %-12s %-12s measured %d obs on %d+%d rows in %.1fs",
            condition, setting, len(superset), x_train.shape[0],
            x_validation.shape[0], measure_seconds,
        )
    return entry


def run(
    *,
    roi_mode: str = "oracle",
    config: Optional[Settings] = None,
    dataset: Optional[LoadedPixelDataset] = None,
    vqc_version: Optional[str] = None,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 42,
    chunk: int = 128,
    conditions: Optional[Sequence[str]] = None,
    with_gradients: bool = False,
    with_plateau: bool = False,
    plateau_draws: int = 20,
) -> Dict[str, Any]:
    """Run E0.1 for one ROI mode. Reads train and validation only."""
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)
    data = dataset or load_pixel_partitions(config=cfg, roi_mode=roi_mode)

    if data.qubit_count != V1_QUBIT_COUNT:
        raise E0Error(
            f"Pixel cache reports {data.qubit_count} qubits; V1 is fixed at "
            f"{V1_QUBIT_COUNT} and E0 changes only the readout."
        )

    from backend.training.train_pixel_vqc import PIXEL_VQC_VERSION

    directory = store.pixel_vqc_dir(vqc_version or PIXEL_VQC_VERSION)
    train_conditions = condition_rows(data.train)
    validation_conditions = condition_rows(data.validation)
    primary = PRIMARY_CONDITION.get(roi_mode)

    names = list(conditions) if conditions else sorted(
        set(train_conditions) & set(validation_conditions)
    )
    # Run the primary condition first. It is the one the decision gate reads, it is the
    # cheapest, and a sweep that dies partway through should have produced the headline
    # result rather than only the caveated secondary ones.
    if primary in names:
        names = [primary] + [n for n in names if n != primary]

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
                f"{LARGE_CONDITION_TRAIN_ROWS}-row threshold and each draw refits the "
                "full pipeline including its inner grouped CV. Finest resolvable "
                f"empirical p is {1 / (condition_permutations + 1):.4f}."
            )
            budget_notes[condition] = note
            logger.info("  %-12s permutation null %s", condition, note)

        out.append(run_condition(
            condition=condition,
            train=data.train,
            validation=data.validation,
            train_rows=train_conditions[condition],
            validation_rows=validation_conditions[condition],
            frozen=frozen,
            n_permutations=condition_permutations,
            seed=seed,
            chunk=chunk,
            with_gradients=with_gradients,
            with_plateau=with_plateau,
            plateau_draws=plateau_draws,
            permutation_note=note,
        ))

    return {
        "e0_diagnostics_version": E0_VERSION,
        "experiment": "E0.1 multi-observable readout",
        "roi_mode": roi_mode,
        "partitions_read": ["train", "validation"],
        "test_partition_used": False,
        "n_qubits": V1_QUBIT_COUNT,
        "amplitudes_per_state": V1_PIXEL_COUNT,
        "frozen_invariants": {
            "encoding": "256x256 grayscale ROI -> uint8 -> /255 -> L2 normalise",
            "qubits": V1_QUBIT_COUNT,
            "amplitudes": V1_PIXEL_COUNT,
            "ansatz": "unchanged L1 ry+rz per qubit, linear CNOT cascade, circular close",
            "loss": "class-weighted BCE, inverse-frequency weights normalised to mean 1",
            "split_manifest_sha256": _sha256(store.split_manifest_path),
            "split_seed": data.manifest.seed,
            "preprocessing": dict(data.cache_metadata),
            "changed_in_this_experiment": "the measurement/readout only",
        },
        "variants": {
            v: {"n_observables": VARIANT_SIZES[v], "pre_registered": True}
            for v in VARIANTS
        },
        "permutation_budget": {
            "requested": int(n_permutations),
            "large_condition_train_rows_threshold": LARGE_CONDITION_TRAIN_ROWS,
            "large_condition_permutations": LARGE_CONDITION_PERMUTATIONS,
            "reduced_conditions": budget_notes,
            "note": (
                "Conditions above the row threshold ran a smaller null for cost. Listed "
                "here and per condition rather than applied silently. This reaches the "
                "oracle secondary conditions (A_all, A_region_polygon) and also the "
                "predicted primary condition B_localized at 1,620 train rows -- so the "
                "predicted headline is read against a 50-draw null, whose finest "
                "resolvable p is 0.0196. The oracle primary (A_lesion_polygon, 215 rows) "
                "runs the full requested budget."
            ),
        },
        "weight_settings": {
            "W-identity": "no circuit; observables read off the encoded amplitudes",
            "W-untrained": "seeded initial draw, uniform(-pi, pi) at the artifact seed",
            "W-frozen": "the selected Phase D weights, sha256 recorded per condition",
        },
        "threshold": {
            "value": FIXED_THRESHOLD,
            "selection": "fixed, not swept; no threshold was chosen on any partition",
        },
        "seeds": {"experiment": seed, "permutation_base": seed},
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "conditions": out,
    }


# --------------------------------------------------------------------------- verdict
def verdict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The reading E0.1 supports, stated as measured comparisons.

    Deliberately comparative -- against ``A0``, against each variant's own permutation
    null -- because no absolute go/no-go threshold exists in the project documents and
    inventing one here would be exactly the kind of invented criterion the governing
    spec forbids.
    """
    rows: List[Dict[str, Any]] = []
    for condition in payload["conditions"]:
        baseline = None
        for block in condition["results"]:
            if block["variant"] == "A0" and block["weight_setting"] == "W-frozen":
                metrics = block.get("validation_metrics") or {}
                baseline = metrics.get("pr_auc")
        for block in condition["results"]:
            metrics = block.get("validation_metrics") or {}
            pr = metrics.get("pr_auc")
            rank = block["permutation_null"].get("observed_pr_auc_vs_null") or {}
            rows.append({
                "condition": condition["condition"],
                "variant": block["variant"],
                "weight_setting": block["weight_setting"],
                "n_observables": block["n_observables"],
                "validation_pr_auc": None if pr is None else round(float(pr), 6),
                "validation_roc_auc": (
                    None if metrics.get("roc_auc") is None
                    else round(float(metrics["roc_auc"]), 6)
                ),
                "prevalence": condition["validation"]["prevalence"],
                "null_p95_pr_auc": (
                    (block["permutation_null"].get("pr_auc") or {}).get("p95")
                ),
                "exceeds_null_p95": rank.get("exceeds_null_p95"),
                "empirical_p": rank.get("empirical_p"),
                "delta_vs_A0_frozen": (
                    None if pr is None or baseline is None
                    else round(float(pr) - float(baseline), 6)
                ),
            })
    signal = [
        r for r in rows
        if r["exceeds_null_p95"] and r["empirical_p"] is not None
        and r["empirical_p"] <= 0.05
    ]
    return {
        "per_variant": rows,
        "n_variant_settings_above_null_p95_at_p05": len(signal),
        "settings_with_signal": [
            {k: r[k] for k in ("condition", "variant", "weight_setting",
                               "validation_pr_auc", "empirical_p")}
            for r in signal
        ],
        "critical_question": (
            "Does the existing quantum state contain useful information that Z0 was "
            "failing to expose?"
        ),
        "answer_rule": (
            "Yes only where a pre-registered variant's validation PR-AUC exceeds its own "
            "patient-blocked permutation null (p95 and empirical p <= 0.05). A gain over "
            "A0 that sits inside the null is capacity, not signal. A gain present only at "
            "W-frozen and absent at W-untrained and W-identity belongs to the trained "
            "circuit, not to the readout."
        ),
        "encoding_caveat": (
            "A readout improvement does not validate the amplitude encoding. Per the "
            "governing spec it must still beat strong classical controls on identical "
            "rows or provide complementary information."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase E0.1: vary only the readout on the frozen 16-qubit pixel VQC."
    )
    parser.add_argument("--roi", dest="roi_mode", default="oracle",
                        choices=("oracle", "predicted"))
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--condition", action="append", default=None,
                        help="Restrict to a named ROI condition; repeatable.")
    parser.add_argument("--gradients", action="store_true",
                        help="Parameter-shift gradient magnitudes (64 circuit "
                             "evaluations per measured variant).")
    parser.add_argument("--plateau", action="store_true",
                        help="Barren-plateau gradient-variance probe.")
    parser.add_argument("--plateau-draws", type=int, default=20)
    parser.add_argument("--out", default=None, help="Write the full payload here.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("qiskit", "stevedore", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    payload = run(
        roi_mode=args.roi_mode,
        n_permutations=args.permutations,
        seed=args.seed,
        chunk=args.chunk,
        conditions=args.condition,
        with_gradients=args.gradients,
        with_plateau=args.plateau,
        plateau_draws=args.plateau_draws,
    )
    payload["verdict"] = verdict(payload)
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Full payload -> %s", args.out)
    print(json.dumps(payload["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
