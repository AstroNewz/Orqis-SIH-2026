"""Evaluation metrics for screening classifiers.

PART 15 names accuracy, precision, recall, F1, specificity, sensitivity, ROC-AUC,
PR-AUC, calibration metrics and the confusion matrix. All of them are computed
here, once, so a quantum model and a classical baseline are never scored by
subtly different code.

Which numbers actually matter here
----------------------------------
The dataset is ~6% positive. Accuracy is close to useless at that prevalence -- a
model that answers "negative" every time scores about 0.94 -- so it is reported
for completeness and should not be quoted on its own.

**PR-AUC (average precision)** is the headline discrimination metric. ROC-AUC is
reported too, but ROC curves are insensitive to class imbalance: a large absolute
number of false positives barely moves the false-positive rate when negatives
dominate, while it wrecks precision. PR-AUC sees that; ROC-AUC does not.

**Sensitivity at a fixed operating point** matters more than either for a
screening tool, because the cost of a missed lesion is not the cost of a false
alarm. :func:`threshold_sweep` reports the full trade-off rather than asserting
one threshold is correct, and :func:`sensitivity_at_specificity` answers the
question a clinician would actually ask.

Ties
----
ROC-AUC and average precision come from scikit-learn rather than being
hand-rolled. Quantum scores tie more often than continuous classical ones -- a
shot-based expectation value takes discrete values -- and naive rank-based AUC
silently mishandles ties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from quantum_ml.calibration import brier_score, expected_calibration_error, log_loss


class MetricsError(ValueError):
    """Raised when metrics cannot be computed from the supplied arrays."""


@dataclass
class ConfusionMatrix:
    """Counts at one operating point."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    threshold: float

    @property
    def n(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "threshold": self.threshold,
        }


@dataclass
class ClassificationMetrics:
    """Every metric PART 15 asks for, at one threshold, plus provenance."""

    n_samples: int
    n_positive: int
    n_negative: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    sensitivity: float
    specificity: float
    f1: float
    balanced_accuracy: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    brier: float
    log_loss: float
    ece: float
    confusion_matrix: ConfusionMatrix
    prevalence: float
    mean_predicted: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "prevalence": round(self.prevalence, 6),
            "threshold": self.threshold,
            "accuracy": _round(self.accuracy),
            "precision": _round(self.precision),
            "recall": _round(self.recall),
            "sensitivity": _round(self.sensitivity),
            "specificity": _round(self.specificity),
            "f1": _round(self.f1),
            "balanced_accuracy": _round(self.balanced_accuracy),
            "roc_auc": _round(self.roc_auc),
            "pr_auc": _round(self.pr_auc),
            "brier": _round(self.brier),
            "log_loss": _round(self.log_loss),
            "ece": _round(self.ece),
            "mean_predicted": _round(self.mean_predicted),
            "confusion_matrix": self.confusion_matrix.to_dict(),
        }
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload

    def summary_line(self) -> str:
        return (
            f"n={self.n_samples} pos={self.n_positive} | "
            f"PR-AUC {_fmt(self.pr_auc)} ROC-AUC {_fmt(self.roc_auc)} | "
            f"sens {self.sensitivity:.3f} spec {self.specificity:.3f} "
            f"prec {self.precision:.3f} F1 {self.f1:.3f} | "
            f"Brier {self.brier:.4f} ECE {self.ece:.4f} @ t={self.threshold:.2f}"
        )


def _round(value: Optional[float], places: int = 6) -> Optional[float]:
    return None if value is None else round(float(value), places)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _validate(y_true: Sequence[int], y_prob: Sequence[float]) -> tuple:
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    if y.size != p.size:
        raise MetricsError(f"Length mismatch: {y.size} labels vs {p.size} probabilities.")
    if y.size == 0:
        raise MetricsError("Cannot compute metrics on an empty sample.")
    if not np.all(np.isin(y, (0, 1))):
        raise MetricsError("Labels must be binary 0/1.")
    if not np.all(np.isfinite(p)):
        raise MetricsError("Probabilities contain non-finite values.")
    return y, p


def confusion_at(y_true: Sequence[int], y_prob: Sequence[float], threshold: float) -> ConfusionMatrix:
    """Confusion matrix at ``threshold``, with ``>=`` counting as positive."""
    y, p = _validate(y_true, y_prob)
    predicted = (p >= threshold).astype(int)
    return ConfusionMatrix(
        true_positive=int(np.sum((predicted == 1) & (y == 1))),
        false_positive=int(np.sum((predicted == 1) & (y == 0))),
        true_negative=int(np.sum((predicted == 0) & (y == 0))),
        false_negative=int(np.sum((predicted == 0) & (y == 1))),
        threshold=float(threshold),
    )


def evaluate_predictions(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    threshold: float = 0.50,
) -> ClassificationMetrics:
    """Compute the full metric set at one operating point.

    ``roc_auc`` and ``pr_auc`` are ``None`` when the sample contains only one
    class -- both are undefined then, and returning 0.5 or 0.0 would look like a
    measurement rather than an absence of one.
    """
    y, p = _validate(y_true, y_prob)
    matrix = confusion_at(y, p, threshold)
    tp, fp, tn, fn = (
        matrix.true_positive,
        matrix.false_positive,
        matrix.true_negative,
        matrix.false_negative,
    )
    notes: List[str] = []

    # Zero denominators are reported as 0.0 with a note, rather than nan: a model
    # that predicts no positives has no precision to speak of, and a silent nan
    # propagates into every aggregate downstream.
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    if not (tp + fp):
        notes.append(f"No sample scored >= {threshold}; precision is undefined, reported as 0.")
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    n_positive, n_negative = int(np.sum(y == 1)), int(np.sum(y == 0))
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    if n_positive and n_negative:
        from sklearn.metrics import average_precision_score, roc_auc_score

        roc_auc = float(roc_auc_score(y, p))
        pr_auc = float(average_precision_score(y, p))
    else:
        notes.append(
            f"Only one class present (positive={n_positive}, negative={n_negative}); "
            "ROC-AUC and PR-AUC are undefined."
        )

    if n_positive < 20:
        notes.append(
            f"Only {n_positive} positive samples: every metric here has a wide "
            "confidence interval and small differences between models are not "
            "meaningful."
        )

    return ClassificationMetrics(
        n_samples=int(y.size),
        n_positive=n_positive,
        n_negative=n_negative,
        threshold=float(threshold),
        accuracy=(tp + tn) / y.size,
        precision=precision,
        recall=recall,
        sensitivity=recall,
        specificity=specificity,
        f1=f1,
        balanced_accuracy=(recall + specificity) / 2.0,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier=brier_score(y, p),
        log_loss=log_loss(y, p),
        ece=expected_calibration_error(y, p),
        confusion_matrix=matrix,
        prevalence=float(np.mean(y)),
        mean_predicted=float(np.mean(p)),
        notes=notes,
    )


def threshold_sweep(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Sensitivity/specificity/precision across thresholds.

    Reported instead of asserting that one threshold is right. The screening
    threshold is an experimental engineering choice (see
    :class:`~quantum_ml.calibration.ProbabilityCalibrator`), and the sweep is what
    lets someone else choose a different one for a reason.
    """
    y, p = _validate(y_true, y_prob)
    grid = thresholds if thresholds is not None else np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows: List[Dict[str, Any]] = []
    for threshold in grid:
        matrix = confusion_at(y, p, float(threshold))
        tp, fp, tn, fn = (
            matrix.true_positive,
            matrix.false_positive,
            matrix.true_negative,
            matrix.false_negative,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
                "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
                "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
                "n_flagged": tp + fp,
                "true_positive": tp,
                "false_negative": fn,
            }
        )
    return rows


def sensitivity_at_specificity(
    y_true: Sequence[int], y_prob: Sequence[float], target_specificity: float = 0.90
) -> Dict[str, Any]:
    """Highest sensitivity achievable at or above ``target_specificity``.

    The clinically legible summary: "if we accept flagging 10% of healthy people,
    what fraction of lesions do we catch?" Returns the threshold that achieves it
    so the number is reproducible.
    """
    y, p = _validate(y_true, y_prob)
    if not (np.any(y == 1) and np.any(y == 0)):
        return {"target_specificity": target_specificity, "sensitivity": None,
                "achieved_specificity": None, "threshold": None,
                "note": "Undefined: the sample contains only one class."}

    best: Dict[str, Any] = {
        "target_specificity": target_specificity,
        "sensitivity": None,
        "achieved_specificity": None,
        "threshold": None,
    }
    # Candidate thresholds are the observed scores: no other value changes the
    # confusion matrix, so this is exact rather than a grid approximation.
    for threshold in np.unique(p):
        matrix = confusion_at(y, p, float(threshold))
        tn, fp = matrix.true_negative, matrix.false_positive
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        if specificity < target_specificity:
            continue
        tp, fn = matrix.true_positive, matrix.false_negative
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        if best["sensitivity"] is None or sensitivity > best["sensitivity"]:
            best = {
                "target_specificity": target_specificity,
                "sensitivity": round(sensitivity, 4),
                "achieved_specificity": round(specificity, 4),
                "threshold": float(threshold),
            }
    if best["sensitivity"] is None:
        best["note"] = (
            f"No threshold reaches specificity {target_specificity}; the model does "
            "not separate the classes well enough at this operating point."
        )
    return best


def threshold_for_sensitivity(
    y_true: Sequence[int], y_prob: Sequence[float], target_sensitivity: float = 0.85
) -> Dict[str, Any]:
    """Highest threshold that still reaches ``target_sensitivity``.

    Why this exists
    ---------------
    A *calibrated* model at 6% prevalence almost never emits a probability above
    0.50, so a 0.50 cut-off on calibrated output misses nearly every positive --
    measured on this dataset, 20 of 21. That is not a calibration failure; it is
    what being calibrated means. The threshold has to be chosen for the operating
    point a screening tool wants, which is high sensitivity, and then reported as
    the experimental choice it is.

    The highest such threshold is returned rather than the lowest, because among
    thresholds that all meet the sensitivity target, the highest is the one that
    flags fewest healthy people.

    This must be evaluated on out-of-sample probabilities. Choosing a threshold
    from probabilities the calibrator was fitted on will pick one that is too high,
    because in-sample positives score better than they will in the field.
    """
    y, p = _validate(y_true, y_prob)
    if not (np.any(y == 1) and np.any(y == 0)):
        return {
            "target_sensitivity": target_sensitivity,
            "threshold": None,
            "sensitivity": None,
            "specificity": None,
            "precision": None,
            "note": "Undefined: the sample contains only one class.",
        }

    best: Dict[str, Any] = {
        "target_sensitivity": target_sensitivity,
        "threshold": None,
        "sensitivity": None,
        "specificity": None,
        "precision": None,
    }
    # Only observed scores change the confusion matrix, so this is exact. Zero is
    # included so a target that needs "flag everything" is still reachable.
    for threshold in np.concatenate([[0.0], np.unique(p)]):
        matrix = confusion_at(y, p, float(threshold))
        tp, fp, tn, fn = (
            matrix.true_positive,
            matrix.false_positive,
            matrix.true_negative,
            matrix.false_negative,
        )
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        if sensitivity < target_sensitivity:
            continue
        if best["threshold"] is None or threshold > best["threshold"]:
            best = {
                "target_sensitivity": target_sensitivity,
                "threshold": float(threshold),
                "sensitivity": round(sensitivity, 4),
                "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
                "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
                "n_flagged": tp + fp,
                "false_negative": fn,
            }
    if best["threshold"] is None:
        best["note"] = (
            f"No threshold reaches sensitivity {target_sensitivity}; not even "
            "flagging every sample does, which means the labels and scores are "
            "misaligned."
        )
    return best


def compare_models(results: Dict[str, ClassificationMetrics]) -> List[Dict[str, Any]]:
    """Rank models by PR-AUC, falling back to Brier score when PR-AUC is undefined.

    PR-AUC first because it is the imbalance-aware discrimination metric; Brier as
    the tiebreak because a model that ranks well but is badly calibrated is not
    usable for a reported risk percentage.
    """
    rows = [
        {"model": name, **metrics.to_dict()} for name, metrics in results.items()
    ]
    rows.sort(
        key=lambda row: (
            row["pr_auc"] is None,
            -(row["pr_auc"] or 0.0),
            row["brier"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows
