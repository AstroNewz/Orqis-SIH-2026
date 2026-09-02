"""Probability calibration and risk categorisation.

The quantum output is **not** a calibrated probability
--------------------------------------------------------
:math:`(1 - \\langle Z_0 \\rangle) / 2` is a bounded, monotone function of a
measurement. Nothing about it makes it a frequency: if the model emits 0.30 for a
group of cases, roughly 30% of them should actually be positive, and there is no
reason a raw expectation value would satisfy that. On a dataset that is ~6%
positive, the raw scores cluster hard and reading one as "a 30% chance of cancer"
would be a straightforward misstatement.

Calibration is therefore a separate, *fitted* stage with two implementations:

**Platt scaling** fits :math:`p = \\sigma(a s + b)` -- two parameters. It assumes
the score-to-log-odds relationship is linear, which is restrictive but very hard
to overfit. Fitted with Platt's own target smoothing (``t+ = (N+ + 1)/(N+ + 2)``,
``t- = 1/(N- + 2)``) rather than hard 0/1 targets; with ~100 positives that
smoothing is the difference between a usable fit and one that saturates.

**Isotonic regression** fits an arbitrary monotone step function. Strictly more
expressive, and correspondingly free to overfit: it can carve out a step for a
handful of validation points. Preferred only when it actually wins on held-out
data.

``method="auto"`` picks between them by **validation** Brier score, and records
which won and by how much.

Fitting discipline
------------------
CRITICAL: the calibrator is fitted on training/validation data only. The held-out
test partition is never used to fit it. Fitting calibration on the test set makes
the reported calibration metrics meaningless -- they would be measuring the fit,
not generalisation. :func:`fit_calibrator` takes the fitting data explicitly and
has no access to anything else, so the discipline is structural rather than a
convention someone has to remember.

Persistence
-----------
Both calibrators serialise to plain JSON: Platt as two floats, isotonic as its
breakpoint arrays. Isotonic is *applied* with :func:`numpy.interp`, so loading a
saved calibrator does not require scikit-learn and involves no pickle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_VERSION = "calibration-1"

# Clamp for log-loss and for the sigmoid's output. Keeps a confidently wrong
# prediction's contribution finite without letting one sample dominate.
EPSILON = 1.0e-12

# Newton iterations for the Platt fit. The objective is convex in two parameters,
# so this converges in well under 10 iterations; the cap is a safety net.
PLATT_MAX_ITER = 100
PLATT_TOLERANCE = 1.0e-9


class CalibrationError(ValueError):
    """Raised when a calibrator cannot be fitted or applied."""


class CalibrationMethod(str, Enum):
    PLATT = "platt"
    ISOTONIC = "isotonic"
    IDENTITY = "identity"
    """Pass-through. Used only when calibration data is too small to fit anything,
    and flagged as such so the result is not presented as calibrated."""


class RiskLevel(str, Enum):
    """Risk bands. Values match the existing API contract exactly."""

    LOW = "LOW RISK"
    MODERATE = "MODERATE RISK"
    HIGH = "HIGH RISK"


# ---------------------------------------------------------------- metrics
def brier_score(y_true: Sequence[float], y_prob: Sequence[float]) -> float:
    """Mean squared error of the probabilities. Lower is better; 0 is perfect."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.size != p.size:
        raise CalibrationError(f"Length mismatch: {y.size} labels vs {p.size} probabilities.")
    if y.size == 0:
        raise CalibrationError("Cannot compute a Brier score on an empty sample.")
    return float(np.mean((p - y) ** 2))


def log_loss(y_true: Sequence[float], y_prob: Sequence[float]) -> float:
    """Mean binary cross-entropy."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    if y.size == 0:
        raise CalibrationError("Cannot compute log loss on an empty sample.")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error(
    y_true: Sequence[float], y_prob: Sequence[float], n_bins: int = 10
) -> float:
    """Bin-weighted mean gap between confidence and accuracy.

    Complements the Brier score, which mixes calibration and discrimination into
    one number. ECE isolates the calibration part: it asks, per confidence bin,
    whether the predicted rate matches the observed rate. Empty bins are skipped
    rather than counted as perfect.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.size == 0:
        raise CalibrationError("Cannot compute ECE on an empty sample.")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        # Left-closed bins, with the last bin closed on the right so p == 1 lands
        # somewhere rather than being dropped.
        in_bin = (p >= lower) & ((p < upper) if upper < 1.0 else (p <= upper))
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        total += (count / y.size) * abs(float(np.mean(y[in_bin]) - np.mean(p[in_bin])))
    return float(total)


# ---------------------------------------------------------------- Platt
@dataclass
class PlattCalibrator:
    """Two-parameter sigmoid calibration: :math:`p = \\sigma(a s + b)`."""

    a: float = 1.0
    b: float = 0.0
    n_samples: int = 0
    n_positive: int = 0
    converged: bool = False
    n_iterations: int = 0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        """Fit by Newton's method with Platt's target smoothing.

        Smoothed targets replace hard 0/1: with a small positive count, hard
        targets drive ``a`` toward infinity to separate the classes, producing a
        calibrator that outputs only near-0 and near-1 -- the opposite of
        calibrated.
        """
        s = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        if s.size != y.size:
            raise CalibrationError(f"Length mismatch: {s.size} scores vs {y.size} labels.")
        if s.size < 2:
            raise CalibrationError(f"Platt scaling needs at least 2 samples, got {s.size}.")
        if not np.all(np.isfinite(s)):
            raise CalibrationError("Scores contain non-finite values.")

        n_positive = float(np.sum(y == 1))
        n_negative = float(y.size - n_positive)
        if n_positive == 0 or n_negative == 0:
            raise CalibrationError(
                "Platt scaling needs both classes present; got "
                f"{int(n_positive)} positive and {int(n_negative)} negative."
            )

        target_positive = (n_positive + 1.0) / (n_positive + 2.0)
        target_negative = 1.0 / (n_negative + 2.0)
        targets = np.where(y == 1, target_positive, target_negative)

        # Initialise at the prior's log-odds with a == 0, i.e. "ignore the score",
        # which is the safest starting point and always feasible.
        a = 0.0
        b = float(np.log((n_negative + 1.0) / (n_positive + 1.0)))

        converged = False
        iterations = 0
        for iterations in range(1, PLATT_MAX_ITER + 1):
            # Platt's original parameterisation: p = sigmoid(-(a*s + b)). For a
            # score where higher means more likely positive, the fit returns a
            # *negative* ``a`` -- that is the convention, not a sign error.
            z = a * s + b
            p = _sigmoid(-z)
            gradient_common = p - targets
            grad_a = float(np.sum(gradient_common * -s))
            grad_b = float(np.sum(gradient_common * -1.0))

            w = p * (1.0 - p)
            h_aa = float(np.sum(w * s * s))
            h_ab = float(np.sum(w * s))
            h_bb = float(np.sum(w))

            # Ridge term keeps the 2x2 Hessian invertible when the scores are
            # nearly constant, which happens when the model has not learnt much.
            ridge = 1.0e-10
            det = (h_aa + ridge) * (h_bb + ridge) - h_ab * h_ab
            if abs(det) < 1.0e-300:
                break
            step_a = -((h_bb + ridge) * grad_a - h_ab * grad_b) / det
            step_b = -(-h_ab * grad_a + (h_aa + ridge) * grad_b) / det

            a += step_a
            b += step_b
            if max(abs(step_a), abs(step_b)) < PLATT_TOLERANCE:
                converged = True
                break

        self.a = float(a)
        self.b = float(b)
        self.n_samples = int(s.size)
        self.n_positive = int(n_positive)
        self.converged = converged
        self.n_iterations = iterations
        if not converged:
            logger.warning(
                "Platt scaling did not converge in %d iterations (a=%.4f, b=%.4f); "
                "the fit is still usable but the score distribution may be degenerate.",
                PLATT_MAX_ITER,
                self.a,
                self.b,
            )
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        return np.clip(_sigmoid(-(self.a * s + self.b)), 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": CalibrationMethod.PLATT.value,
            "a": self.a,
            "b": self.b,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlattCalibrator":
        return cls(
            a=float(payload["a"]),
            b=float(payload["b"]),
            n_samples=int(payload.get("n_samples", 0)),
            n_positive=int(payload.get("n_positive", 0)),
            converged=bool(payload.get("converged", False)),
            n_iterations=int(payload.get("n_iterations", 0)),
        )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic function.

    ``np.exp`` of a large positive argument overflows to inf and warns; branching
    on the sign keeps every exponent negative.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


# -------------------------------------------------------------- isotonic
@dataclass
class IsotonicCalibrator:
    """Monotone step-function calibration via pool-adjacent-violators.

    Fitted with scikit-learn's ``IsotonicRegression`` -- PAVA plus tie handling is
    exactly the kind of algorithm where a hand-rolled version is subtly wrong.
    Applied with :func:`numpy.interp` over the fitted knots, so inference needs
    neither scikit-learn nor a pickle.
    """

    x_knots: List[float] = field(default_factory=list)
    y_knots: List[float] = field(default_factory=list)
    n_samples: int = 0
    n_positive: int = 0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError as exc:  # pragma: no cover - sklearn is a dependency
            raise CalibrationError(
                "scikit-learn is required to fit isotonic calibration. Use "
                "method='platt', which has no such dependency."
            ) from exc

        s = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        if s.size != y.size:
            raise CalibrationError(f"Length mismatch: {s.size} scores vs {y.size} labels.")
        if s.size < 2:
            raise CalibrationError(f"Isotonic regression needs at least 2 samples, got {s.size}.")
        if not np.all(np.isfinite(s)):
            raise CalibrationError("Scores contain non-finite values.")

        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        model.fit(s, y)
        self.x_knots = [float(value) for value in np.asarray(model.X_thresholds_)]
        self.y_knots = [float(value) for value in np.asarray(model.y_thresholds_)]
        self.n_samples = int(s.size)
        self.n_positive = int(np.sum(y == 1))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not self.x_knots:
            raise CalibrationError("IsotonicCalibrator is not fitted.")
        s = np.asarray(scores, dtype=np.float64)
        # np.interp clamps to the end values outside the fitted range, matching
        # IsotonicRegression(out_of_bounds="clip").
        return np.clip(np.interp(s, self.x_knots, self.y_knots), 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": CalibrationMethod.ISOTONIC.value,
            "x_knots": self.x_knots,
            "y_knots": self.y_knots,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IsotonicCalibrator":
        return cls(
            x_knots=[float(v) for v in payload["x_knots"]],
            y_knots=[float(v) for v in payload["y_knots"]],
            n_samples=int(payload.get("n_samples", 0)),
            n_positive=int(payload.get("n_positive", 0)),
        )


@dataclass
class IdentityCalibrator:
    """Pass-through, for when there is not enough data to fit anything.

    Exists so the pipeline has a defined behaviour in that case instead of
    crashing or -- worse -- presenting a raw score as calibrated. Any result built
    on it must report ``calibrated=False``.
    """

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IdentityCalibrator":
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"method": CalibrationMethod.IDENTITY.value}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IdentityCalibrator":
        return cls()


# ----------------------------------------------------------- score calibrator
@dataclass
class ScoreCalibrator:
    """A fitted calibrator plus the evidence for why this method was chosen."""

    method: CalibrationMethod
    model: Any
    version: str = CALIBRATION_VERSION
    fit_metrics: Dict[str, Any] = field(default_factory=dict)
    """Honest out-of-sample calibration metrics, plus ``estimated_on`` recording how
    they were estimated. Never in-sample."""
    candidate_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    selection_basis: str = ""
    n_fit_samples: int = 0
    n_fit_positive: int = 0
    fitted_on: str = ""
    """Which partitions were used. Recorded so an auditor can confirm the test set
    was not among them."""
    held_out_probabilities: Optional[np.ndarray] = None
    """Out-of-sample calibrated probabilities for the fitting rows, from the
    cross-validation (or from the explicit selection sample).

    Kept in memory but deliberately **not** serialised: it is per-row data about
    training patients, and an artifact that travels between machines should not
    carry it. Its purpose is choosing an operating threshold without using the test
    set -- picking a threshold from in-sample probabilities would understate how
    many positives are missed. ``None`` when the identity fallback was used.
    """
    held_out_labels: Optional[np.ndarray] = None
    """Labels aligned with :attr:`held_out_probabilities`. Not serialised."""

    @property
    def is_calibrated(self) -> bool:
        """``False`` for the identity fallback, which calibrates nothing."""
        return self.method is not CalibrationMethod.IDENTITY

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Uncalibrated scores -> calibrated probabilities."""
        return self.model.transform(scores)

    def transform_one(self, score: float) -> float:
        return float(self.transform(np.asarray([score], dtype=np.float64))[0])

    def evaluate(self, scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calibration metrics on a held-out sample."""
        calibrated = self.transform(scores)
        y = np.asarray(labels, dtype=np.float64)
        return {
            "brier": brier_score(y, calibrated),
            "log_loss": log_loss(y, calibrated),
            "ece": expected_calibration_error(y, calibrated),
            "brier_uncalibrated": brier_score(y, np.asarray(scores, dtype=np.float64)),
            "mean_predicted": float(np.mean(calibrated)),
            "observed_rate": float(np.mean(y)),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "method": self.method.value,
            "model": self.model.to_dict(),
            "fit_metrics": self.fit_metrics,
            "candidate_metrics": self.candidate_metrics,
            "selection_basis": self.selection_basis,
            "n_fit_samples": self.n_fit_samples,
            "n_fit_positive": self.n_fit_positive,
            "fitted_on": self.fitted_on,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScoreCalibrator":
        method = CalibrationMethod(payload["method"])
        loaders = {
            CalibrationMethod.PLATT: PlattCalibrator,
            CalibrationMethod.ISOTONIC: IsotonicCalibrator,
            CalibrationMethod.IDENTITY: IdentityCalibrator,
        }
        return cls(
            method=method,
            model=loaders[method].from_dict(payload["model"]),
            version=str(payload.get("version", CALIBRATION_VERSION)),
            fit_metrics=dict(payload.get("fit_metrics", {})),
            candidate_metrics=dict(payload.get("candidate_metrics", {})),
            selection_basis=str(payload.get("selection_basis", "")),
            n_fit_samples=int(payload.get("n_fit_samples", 0)),
            n_fit_positive=int(payload.get("n_fit_positive", 0)),
            fitted_on=str(payload.get("fitted_on", "")),
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "is_calibrated": self.is_calibrated,
            "selection_basis": self.selection_basis,
            "n_fit_samples": self.n_fit_samples,
            "n_fit_positive": self.n_fit_positive,
            "fitted_on": self.fitted_on,
            "metrics": self.fit_metrics,
        }


def fit_calibrator(
    fit_scores: Sequence[float],
    fit_labels: Sequence[int],
    *,
    method: str = "auto",
    selection_scores: Optional[Sequence[float]] = None,
    selection_labels: Optional[Sequence[int]] = None,
    fitted_on: str = "train+validation",
    cv_seed: int = 42,
) -> ScoreCalibrator:
    """Fit a calibrator on scores that came from data the model did not train on.

    Args:
        fit_scores / fit_labels: uncalibrated model scores and true labels used to
            fit the mapping. Must NOT come from the held-out test partition.
        method: ``"platt"``, ``"isotonic"``, or ``"auto"``.
        selection_scores / selection_labels: an optional further held-out sample
            used to compare methods and to report ``fit_metrics``. When absent,
            both are done by stratified cross-validation *within* the fitting data,
            so no sample is ever scored by a model that saw it. Passing a separate
            sample is preferable when one is affordable, but on this dataset the
            validation partition has ~20 positives and splitting it again would
            leave too few to fit anything.
        fitted_on: human-readable note on the provenance of the fitting data.
        cv_seed: seed for the cross-validation fold assignment.

    Raises:
        CalibrationError: the data cannot support a fit.
    """
    scores = np.asarray(fit_scores, dtype=np.float64).ravel()
    labels = np.asarray(fit_labels).astype(int).ravel()
    if scores.size != labels.size:
        raise CalibrationError(
            f"Length mismatch: {scores.size} scores vs {labels.size} labels."
        )
    n_positive = int(np.sum(labels == 1))
    n_negative = int(labels.size - n_positive)

    # Below this there is nothing to fit: a mapping estimated from a handful of
    # points is noise, and presenting its output as a calibrated probability would
    # be worse than admitting the score is uncalibrated.
    if labels.size < 10 or n_positive < 2 or n_negative < 2:
        logger.warning(
            "Not enough data to fit calibration (n=%d, positive=%d, negative=%d); "
            "falling back to the identity calibrator. Results must be reported as "
            "UNCALIBRATED.",
            labels.size,
            n_positive,
            n_negative,
        )
        return ScoreCalibrator(
            method=CalibrationMethod.IDENTITY,
            model=IdentityCalibrator(),
            selection_basis=(
                f"insufficient data: n={labels.size}, positive={n_positive}, "
                f"negative={n_negative}"
            ),
            n_fit_samples=int(labels.size),
            n_fit_positive=n_positive,
            fitted_on=fitted_on,
        )

    requested = method.lower()
    if requested not in {"platt", "isotonic", "auto"}:
        raise CalibrationError(
            f"Unknown calibration method {method!r}. Expected platt, isotonic, or auto."
        )

    # Method selection and the reported metrics both happen on predictions the
    # candidate did not see, either from a supplied held-out sample or from
    # cross-validation within the fitting data. Scoring a candidate on its own
    # fitting data would always favour isotonic, which can drive its in-sample
    # Brier score arbitrarily low without generalising at all.
    wanted = list(_CANDIDATE_FACTORIES) if requested == "auto" else [requested]

    if selection_scores is not None and selection_labels is not None:
        eval_scores = np.asarray(selection_scores, dtype=np.float64).ravel()
        eval_labels = np.asarray(selection_labels).astype(int).ravel()
        if eval_scores.size != eval_labels.size:
            raise CalibrationError(
                f"Selection length mismatch: {eval_scores.size} scores vs "
                f"{eval_labels.size} labels."
            )
        held_out_predictions = {
            name: _CANDIDATE_FACTORIES[name]().fit(scores, labels).transform(eval_scores)
            for name in wanted
        }
        basis_source = "held-out selection sample"
    else:
        eval_scores, eval_labels = scores, labels
        held_out_predictions, n_folds = _out_of_fold_predictions(
            scores, labels, names=wanted, seed=cv_seed
        )
        basis_source = f"{n_folds}-fold cross-validation within the fitting data"

    if requested == "auto":
        best_name, selection_basis = _select_method(
            held_out_predictions, eval_labels, basis_source
        )
    else:
        best_name = requested
        selection_basis = f"explicitly requested {best_name}"

    candidate_metrics = {
        name: {
            "brier": brier_score(eval_labels, predictions),
            "log_loss": log_loss(eval_labels, predictions),
            "ece": expected_calibration_error(eval_labels, predictions),
        }
        for name, predictions in held_out_predictions.items()
    }

    # Refit the winner on the whole fitting set: cross-validation chose the method,
    # and the final model should use every sample available. With ~20 positives in
    # a validation partition, holding any of them out of the final fit is costly.
    best = _CANDIDATE_FACTORIES[best_name]().fit(scores, labels)

    calibrator = ScoreCalibrator(
        method=CalibrationMethod(best_name),
        model=best,
        candidate_metrics=candidate_metrics,
        selection_basis=selection_basis,
        n_fit_samples=int(labels.size),
        n_fit_positive=n_positive,
        fitted_on=fitted_on,
        held_out_probabilities=np.asarray(held_out_predictions[best_name], dtype=np.float64),
        held_out_labels=np.asarray(eval_labels, dtype=int),
    )
    # Reported from the held-out predictions, not from the refitted model applied
    # to its own fitting data -- the latter would understate the error.
    chosen = held_out_predictions[best_name]
    calibrator.fit_metrics = {
        "brier": brier_score(eval_labels, chosen),
        "log_loss": log_loss(eval_labels, chosen),
        "ece": expected_calibration_error(eval_labels, chosen),
        "brier_uncalibrated": brier_score(eval_labels, eval_scores),
        "mean_predicted": float(np.mean(chosen)),
        "observed_rate": float(np.mean(eval_labels)),
        "estimated_on": basis_source,
    }
    return calibrator


_CANDIDATE_FACTORIES: Dict[str, Any] = {
    CalibrationMethod.PLATT.value: PlattCalibrator,
    CalibrationMethod.ISOTONIC.value: IsotonicCalibrator,
}


def _stratified_folds(labels: np.ndarray, n_folds: int, seed: int) -> List[np.ndarray]:
    """Deterministic stratified fold assignment.

    Hand-rolled rather than pulled from scikit-learn so the Platt-only path keeps
    working without it, and so the assignment is reproducible from ``seed`` alone.
    Each class is shuffled then dealt round-robin across folds, which keeps the
    positive rate as even as an integer split allows -- important when there are
    only ~20 positives in total.
    """
    rng = np.random.default_rng(seed)
    assignment = np.empty(labels.size, dtype=int)
    for class_value in (0, 1):
        members = np.flatnonzero(labels == class_value)
        rng.shuffle(members)
        assignment[members] = np.arange(members.size) % n_folds
    return [np.flatnonzero(assignment == fold) for fold in range(n_folds)]


def _out_of_fold_predictions(
    scores: np.ndarray, labels: np.ndarray, *, names: Sequence[str], seed: int
) -> Tuple[Dict[str, np.ndarray], int]:
    """Cross-validated predictions for each candidate, one prediction per sample.

    Every sample is predicted by a model fitted without it, so the resulting Brier
    scores are honest estimates of out-of-sample performance rather than in-sample
    fits. A fold whose training portion is degenerate -- one class only, too few
    points -- falls back to the prior for that fold, which is the correct
    prediction when nothing can be fitted.
    """
    n_positive = int(np.sum(labels == 1))
    n_negative = int(labels.size - n_positive)
    n_folds = int(min(5, n_positive, n_negative))
    if n_folds < 2:
        raise CalibrationError(
            f"Cross-validated method selection needs at least 2 of each class, got "
            f"{n_positive} positive and {n_negative} negative."
        )

    folds = _stratified_folds(labels, n_folds, seed)
    predictions = {name: np.empty(labels.size, dtype=np.float64) for name in names}
    for held_out in folds:
        mask = np.ones(labels.size, dtype=bool)
        mask[held_out] = False
        fit_labels, fit_scores = labels[mask], scores[mask]
        for name in names:
            try:
                model = _CANDIDATE_FACTORIES[name]().fit(fit_scores, fit_labels)
                predictions[name][held_out] = model.transform(scores[held_out])
            except CalibrationError:
                predictions[name][held_out] = float(np.mean(fit_labels))
    return predictions, n_folds


def _select_method(
    predictions: Dict[str, np.ndarray],
    eval_labels: np.ndarray,
    basis_source: str,
) -> Tuple[str, str]:
    """Choose Platt vs isotonic, defaulting to Platt unless isotonic clearly wins.

    Comparing two Brier scores and taking the smaller one treats any difference as
    real, however tiny. Measured on synthetic data with a 250-sample selection
    slice: isotonic won by 0.00027 on validation and then *lost* on the test
    partition, 0.1525 against Platt's 0.1466. The validation margin was noise.

    So the comparison is paired and the difference is judged against its own
    standard error. Isotonic is chosen only when its advantage exceeds one standard
    error of the per-sample Brier difference; otherwise the two-parameter model
    wins the tie, because with ~20 positives the expressive one has far more room
    to fit noise. Both numbers go into ``selection_basis``.
    """
    platt_name, isotonic_name = CalibrationMethod.PLATT.value, CalibrationMethod.ISOTONIC.value
    if platt_name not in predictions or isotonic_name not in predictions:
        only = next(iter(predictions))
        return only, f"auto: {only} was the only fitted candidate"

    y = np.asarray(eval_labels, dtype=np.float64)
    difference = (predictions[isotonic_name] - y) ** 2 - (predictions[platt_name] - y) ** 2
    mean_difference = float(np.mean(difference))  # negative favours isotonic
    standard_error = (
        float(np.std(difference, ddof=1) / np.sqrt(difference.size))
        if difference.size > 1
        else 0.0
    )

    if mean_difference < -standard_error:
        return isotonic_name, (
            f"auto: isotonic chosen -- Brier(isotonic) - Brier(Platt) = "
            f"{mean_difference:+.6f}, beyond 1 SE ({standard_error:.6f}) on the "
            f"{basis_source}"
        )
    return platt_name, (
        f"auto: Platt chosen -- Brier(isotonic) - Brier(Platt) = "
        f"{mean_difference:+.6f}, within 1 SE ({standard_error:.6f}) on the "
        f"{basis_source}, so the two-parameter model wins the tie"
    )


# ------------------------------------------------------- risk categorisation
class ProbabilityCalibrator:
    """Risk categorisation and optional classical/quantum score fusion.

    Distinct from :class:`ScoreCalibrator`, which is the *fitted* calibration
    stage. This class turns an already-calibrated probability into a risk band and
    the accompanying patient-facing text, and it holds the score-fusion helper.

    The thresholds are configuration, not medicine
    ----------------------------------------------
    ``threshold`` (screening positive) and ``high_risk_threshold`` are
    **experimental engineering thresholds**. No clinically validated cut-off for
    this model exists, and every message this class produces says so. They default
    to the values in :mod:`backend.core.config` and are overridable per instance.
    """

    def __init__(
        self,
        threshold: float = 0.50,
        quantum_weight: float = 0.50,
        *,
        high_risk_threshold: float = 0.70,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise CalibrationError(f"threshold must be in [0, 1], got {threshold}.")
        if not 0.0 <= high_risk_threshold <= 1.0:
            raise CalibrationError(
                f"high_risk_threshold must be in [0, 1], got {high_risk_threshold}."
            )
        if not 0.0 <= quantum_weight <= 1.0:
            raise CalibrationError(f"quantum_weight must be in [0, 1], got {quantum_weight}.")
        self.threshold = threshold
        self.quantum_weight = quantum_weight
        self.high_risk_threshold = high_risk_threshold
        self.bands_source = "explicit"
        """Where the band boundaries came from. Reported in inference metadata so a
        result carrying the configured fallback is distinguishable from one carrying
        the model's own derived bands."""

    @classmethod
    def from_settings(cls, config: Optional[Any] = None) -> "ProbabilityCalibrator":
        from backend.core.config import settings as default_settings

        cfg = config or default_settings
        return cls(
            threshold=cfg.SCREENING_THRESHOLD,
            high_risk_threshold=cfg.HIGH_RISK_THRESHOLD,
        )

    @classmethod
    def from_calibration_payload(
        cls,
        payload: Optional[Mapping[str, Any]],
        *,
        config: Optional[Any] = None,
    ) -> "ProbabilityCalibrator":
        """Build the bands from a persisted ``calibration.json``, else from settings.

        Prefer this over :meth:`from_settings` anywhere a model version is known.
        Band boundaries are a property of a *particular* trained model's calibrated
        score distribution, not a global constant: on ``v1-handcrafted`` the
        configured 0.50/0.70 defaults both sat above every probability the model can
        emit, so the screen flagged nothing and the HIGH band was unreachable.
        Reading them from the artifact keeps them tied to the model version they
        were derived for.

        Falls back field-by-field, so a partially-populated artifact -- one where a
        band could not be derived -- still yields a usable instance rather than
        raising during inference.
        """
        bands = (payload or {}).get("resolved_bands") or {}
        fallback = cls.from_settings(config)
        threshold = bands.get("screening_threshold")
        high_risk = bands.get("high_risk_threshold")
        instance = cls(
            threshold=fallback.threshold if threshold is None else float(threshold),
            high_risk_threshold=(
                fallback.high_risk_threshold if high_risk is None else float(high_risk)
            ),
            quantum_weight=fallback.quantum_weight,
        )
        instance.bands_source = (
            "calibration_artifact"
            if threshold is not None or high_risk is not None
            else "configured_fallback"
        )
        return instance

    def fuse_probabilities(
        self,
        quantum_prob: float,
        classical_prob: Optional[float] = None,
    ) -> float:
        """Weighted average of a quantum and a classical probability.

        The weight is a **configuration choice, not a learned parameter**. Fusion
        is off by default in the inference path: an arbitrary 50/50 blend is not a
        model, and a fused score is not calibrated even when both inputs are.
        Retained because the ablation comparing quantum, classical, and blended
        scores needs it.
        """
        if classical_prob is None:
            return float(np.clip(quantum_prob, 0.0, 1.0))
        fused = (self.quantum_weight * quantum_prob) + (
            (1.0 - self.quantum_weight) * classical_prob
        )
        return float(np.clip(fused, 0.0, 1.0))

    def evaluate_brier_score(self, y_true: List[int], y_prob: List[float]) -> float:
        """Brier score. See :func:`brier_score`."""
        return brier_score(y_true, y_prob)

    def categorize_risk(
        self,
        probability: float,
        is_mock: bool = False,
        *,
        calibrated: bool = True,
    ) -> Tuple[str, str, str]:
        """Map a probability to ``(risk_level, classification, details)``.

        Args:
            probability: the calibrated probability, where one is available.
            is_mock: mark the output as simulated test data.
            calibrated: when ``False``, the detail text states that the score is
                an uncalibrated model output. Silently presenting a raw score as a
                percentage risk would misrepresent it.
        """
        if is_mock:
            if probability >= self.threshold:
                return (
                    "MOCK HIGH RISK",
                    "screening_positive",
                    "TEST DATA: AI-assisted screening risk estimate is elevated. "
                    "This is a simulated screening result for software testing only.",
                )
            return (
                "MOCK LOW RISK",
                "screening_negative",
                "TEST DATA: AI-assisted screening risk estimate is low. "
                "This is a simulated screening result for software testing only.",
            )

        score_text = (
            f"AI-assisted screening risk score: {probability:.1%}."
            if calibrated
            else f"Uncalibrated AI-assisted screening score: {probability:.3f} "
            "(no calibration model was available, so this is not a probability)."
        )
        disclaimer = (
            "Notice: This is an AI-assisted screening estimate, not a definitive "
            "clinical diagnosis. It does not replace professional clinical "
            "assessment or histopathological confirmation."
        )

        if probability >= self.high_risk_threshold:
            return (
                RiskLevel.HIGH.value,
                "screening_positive",
                f"{score_text} Preliminary risk estimate exceeds the experimental "
                f"screening threshold. {disclaimer} Professional clinical evaluation "
                "is recommended.",
            )
        if probability >= self.threshold:
            return (
                RiskLevel.MODERATE.value,
                "screening_positive",
                f"{score_text} Preliminary risk estimate is borderline/moderate. "
                f"{disclaimer}",
            )
        return (
            RiskLevel.LOW.value,
            "screening_negative",
            f"{score_text} Preliminary risk estimate is below the experimental "
            f"screening threshold. {disclaimer}",
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "screening_threshold": self.threshold,
            "high_risk_threshold": self.high_risk_threshold,
            "quantum_weight": self.quantum_weight,
            "bands_source": self.bands_source,
            "thresholds_are_clinically_validated": False,
        }
