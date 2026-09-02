"""Classical baselines: logistic regression, random forest, gradient boosting.

Why these three
---------------
PART 15 names them, and the choice is defensible rather than arbitrary. They span
the model-complexity range on the same features the quantum model sees:

- **Logistic regression** is the linear baseline. If it matches the VQC, the
  problem is linearly separable in the reduced space and the circuit is decoration.
  It is also the only one of the three that is a *calibrated-by-construction*
  probability model, which makes it the fair comparison for calibration metrics.
- **Random forest** is the variance-reduction baseline: bagged deep trees, high
  capacity, resistant to overfitting through averaging.
- **Gradient boosting** is the bias-reduction baseline: shallow trees fitted
  sequentially to residuals. On tabular features of this size it is usually the
  strongest classical option, so it is the hardest honest target for the VQC.

Giving the baselines a fair fight
---------------------------------
A quantum model looks good against a badly-configured baseline for no interesting
reason. Each baseline therefore gets a small hyperparameter grid selected on the
**validation** partition by PR-AUC -- never on test. The VQC's own hyperparameters
are chosen the same way, so the comparison is symmetric. Whatever the outcome, it
is not an artefact of one side being tuned and the other not.

Class imbalance
---------------
~6% of images are positive. Every baseline is weighted inversely to class
frequency (``class_weight="balanced"``, or explicit ``sample_weight`` for gradient
boosting, which has no such parameter). Unweighted, all three collapse to
predicting "negative" everywhere -- which scores 94% accuracy and finds nothing.

Persistence, and why there is no pickle
---------------------------------------
Logistic regression persists to JSON exactly: it *is* a coefficient vector and an
intercept, and :meth:`LogisticBaseline.from_dict` reconstructs it without
scikit-learn. That is why it, and not a tree ensemble, is the classical model the
backend can serve.

The tree ensembles have no compact honest serialisation, and pickling a fitted
estimator means loading an artifact can execute code and breaks across library
versions. They are instead reproducible by construction: identical
``(seed, hyperparameters, fitted pipeline, training partition)`` gives an identical
model, and :func:`refit_from_record` rebuilds one from its persisted record. They
exist to be compared against, not deployed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BASELINE_VERSION = "baselines-1"

LOGISTIC_REGRESSION = "logistic_regression"
RANDOM_FOREST = "random_forest"
GRADIENT_BOOSTING = "gradient_boosting"
AVAILABLE_BASELINES: Tuple[str, ...] = (
    LOGISTIC_REGRESSION,
    RANDOM_FOREST,
    GRADIENT_BOOSTING,
)


class BaselineError(RuntimeError):
    """Raised when a baseline cannot be built, fitted, or applied."""


# Grids are deliberately small. With 104 training positives, a large grid searched
# against a 21-positive validation partition selects noise; a handful of
# well-separated values is the most this data can support.
HYPERPARAMETER_GRIDS: Dict[str, List[Dict[str, Any]]] = {
    LOGISTIC_REGRESSION: [
        {"C": 0.01},
        {"C": 0.1},
        {"C": 1.0},
        {"C": 10.0},
    ],
    RANDOM_FOREST: [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 2},
        {"n_estimators": 600, "max_depth": 12, "min_samples_leaf": 4},
    ],
    GRADIENT_BOOSTING: [
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 200, "learning_rate": 0.10, "max_depth": 3},
        {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 3},
    ],
}

DESCRIPTIONS: Dict[str, str] = {
    LOGISTIC_REGRESSION: (
        "L2-regularised linear model on the reduced feature vector. The linear "
        "reference point, and the only baseline that is a calibrated probability "
        "model by construction."
    ),
    RANDOM_FOREST: (
        "Bagged decision trees with balanced class weights. High-capacity, "
        "variance-reducing reference."
    ),
    GRADIENT_BOOSTING: (
        "Sequentially boosted shallow trees with per-sample class weights. Usually "
        "the strongest classical option on tabular features of this size."
    ),
}


@dataclass
class BaselineRecord:
    """Everything needed to reproduce and audit one fitted baseline."""

    name: str
    version: str = BASELINE_VERSION
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    candidate_scores: List[Dict[str, Any]] = field(default_factory=list)
    """Validation PR-AUC per grid point, so the selection is inspectable."""
    selection_metric: str = "validation_pr_auc"
    random_seed: int = 42
    n_train_samples: int = 0
    n_train_positive: int = 0
    feature_dimension: int = 0
    feature_mode: str = ""
    extractor: str = ""
    preprocessing_version: str = ""
    pipeline_version: str = ""
    class_weighting: str = "balanced"
    training_duration_seconds: float = 0.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "hyperparameters": self.hyperparameters,
            "candidate_scores": self.candidate_scores,
            "selection_metric": self.selection_metric,
            "random_seed": self.random_seed,
            "n_train_samples": self.n_train_samples,
            "n_train_positive": self.n_train_positive,
            "feature_dimension": self.feature_dimension,
            "feature_mode": self.feature_mode,
            "extractor": self.extractor,
            "preprocessing_version": self.preprocessing_version,
            "pipeline_version": self.pipeline_version,
            "class_weighting": self.class_weighting,
            "training_duration_seconds": round(self.training_duration_seconds, 3),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BaselineRecord":
        return cls(
            name=str(payload["name"]),
            version=str(payload.get("version", BASELINE_VERSION)),
            hyperparameters=dict(payload.get("hyperparameters", {})),
            candidate_scores=list(payload.get("candidate_scores", [])),
            selection_metric=str(payload.get("selection_metric", "validation_pr_auc")),
            random_seed=int(payload.get("random_seed", 42)),
            n_train_samples=int(payload.get("n_train_samples", 0)),
            n_train_positive=int(payload.get("n_train_positive", 0)),
            feature_dimension=int(payload.get("feature_dimension", 0)),
            feature_mode=str(payload.get("feature_mode", "")),
            extractor=str(payload.get("extractor", "")),
            preprocessing_version=str(payload.get("preprocessing_version", "")),
            pipeline_version=str(payload.get("pipeline_version", "")),
            class_weighting=str(payload.get("class_weighting", "balanced")),
            training_duration_seconds=float(payload.get("training_duration_seconds", 0.0)),
            description=str(payload.get("description", "")),
        )


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights, normalised to mean 1.

    Matches ``class_weight="balanced"`` so gradient boosting -- which has no
    ``class_weight`` -- is weighted identically to the other two. Mean-1
    normalisation keeps the loss on the same scale as an unweighted fit, so
    objective values remain comparable.
    """
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise BaselineError(
            f"Both classes are required to weight a baseline; got {counts.tolist()}."
        )
    weights = y.size / (2.0 * counts)
    per_sample = weights[y]
    return per_sample / float(np.mean(per_sample))


def build_estimator(name: str, hyperparameters: Dict[str, Any], *, seed: int = 42):
    """Construct an unfitted scikit-learn estimator for one baseline."""
    if name == LOGISTIC_REGRESSION:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            # L2 is the default in every supported scikit-learn version. Passing
            # penalty="l2" explicitly is deprecated from 1.8 onward, so it is left
            # implicit rather than triggering a FutureWarning on every fit.
            class_weight="balanced",
            # lbfgs on 181 standardised features converges well inside this cap;
            # the explicit value keeps a convergence warning from being silent.
            max_iter=5000,
            random_state=seed,
            **hyperparameters,
        )
    if name == RANDOM_FOREST:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            class_weight="balanced",
            random_state=seed,
            # Deliberately serial. With n_jobs=-1 the per-tree probabilities are
            # accumulated by several threads, so the summation order -- and hence
            # the last bit of every predicted probability -- depends on thread
            # scheduling. The trees themselves are identical either way, but
            # "refitting from the record reproduces the model exactly" is worth
            # more in an auditable pipeline than the few seconds parallelism saves
            # at this dataset size.
            n_jobs=None,
            **hyperparameters,
        )
    if name == GRADIENT_BOOSTING:
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=seed, **hyperparameters)
    raise BaselineError(
        f"Unknown baseline {name!r}. Available: {', '.join(AVAILABLE_BASELINES)}."
    )


@dataclass
class FittedBaseline:
    """A fitted baseline and its record."""

    record: BaselineRecord
    estimator: Any

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Positive-class probabilities, ``(n,)``."""
        matrix = np.asarray(x, dtype=np.float64)
        if matrix.ndim != 2:
            raise BaselineError(f"Expected a 2-D feature matrix, got shape {matrix.shape}.")
        if matrix.shape[1] != self.record.feature_dimension:
            raise BaselineError(
                f"{self.record.name} was fitted on {self.record.feature_dimension} "
                f"features, got {matrix.shape[1]}."
            )
        return np.asarray(self.estimator.predict_proba(matrix)[:, 1], dtype=np.float64)

    def predict(self, x: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)

    @property
    def serialisable(self) -> bool:
        """Whether this baseline can be persisted exactly, without pickle."""
        return self.record.name == LOGISTIC_REGRESSION

    def to_dict(self) -> Dict[str, Any]:
        """JSON payload. Includes weights only for logistic regression.

        For the tree ensembles the payload is the record alone -- enough to refit
        an identical model, not enough to skip refitting. That is deliberate: see
        the module docstring.
        """
        payload: Dict[str, Any] = {"record": self.record.to_dict()}
        if self.serialisable:
            payload["weights"] = {
                "coefficients": np.asarray(self.estimator.coef_).ravel().tolist(),
                "intercept": float(np.asarray(self.estimator.intercept_).ravel()[0]),
            }
        return payload


@dataclass
class LogisticBaseline:
    """A logistic regression restored from JSON, with no scikit-learn dependency.

    Used when the backend serves a classical probability alongside the quantum one:
    the inference path should not have to import scikit-learn or unpickle anything
    to evaluate a dot product.
    """

    coefficients: np.ndarray
    intercept: float
    record: BaselineRecord

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != self.coefficients.size:
            raise BaselineError(
                f"Expected {self.coefficients.size} features, got {matrix.shape[1]}."
            )
        z = matrix @ self.coefficients + self.intercept
        # Branch on sign so a large positive exponent never overflows.
        out = np.empty_like(z)
        positive = z >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        out[~positive] = exp_z / (1.0 + exp_z)
        return out

    def predict_one(self, x: np.ndarray) -> float:
        return float(self.predict_proba(np.asarray(x, dtype=np.float64).reshape(1, -1))[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "weights": {
                "coefficients": self.coefficients.tolist(),
                "intercept": self.intercept,
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LogisticBaseline":
        weights = payload.get("weights")
        if not weights:
            raise BaselineError(
                "This payload has no weights. Only logistic regression is persisted "
                "with weights; tree ensembles must be refitted via refit_from_record."
            )
        return cls(
            coefficients=np.asarray(weights["coefficients"], dtype=np.float64),
            intercept=float(weights["intercept"]),
            record=BaselineRecord.from_dict(payload["record"]),
        )


def fit_baseline(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    metadata: Optional[Dict[str, Any]] = None,
) -> FittedBaseline:
    """Fit one baseline, selecting hyperparameters on validation PR-AUC.

    Args:
        name: one of :data:`AVAILABLE_BASELINES`.
        x_train / y_train: the **training partition only**.
        validation_data: ``(x, y)`` for hyperparameter selection. When omitted the
            first grid point is used unchanged and the record says so -- selecting
            on training data would pick the highest-capacity option every time.
        hyperparameters: pin the hyperparameters and skip selection entirely.
        seed: threaded into every estimator that has a ``random_state``.
        metadata: provenance fields for the record (extractor, feature mode, ...).

    Raises:
        BaselineError: the baseline is unknown or the data is unusable.
    """
    if name not in AVAILABLE_BASELINES:
        raise BaselineError(
            f"Unknown baseline {name!r}. Available: {', '.join(AVAILABLE_BASELINES)}."
        )
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train).astype(int).ravel()
    if x.ndim != 2:
        raise BaselineError(f"Expected a 2-D training matrix, got shape {x.shape}.")
    if x.shape[0] != y.size:
        raise BaselineError(f"Shape mismatch: {x.shape[0]} rows vs {y.size} labels.")
    if not np.all(np.isin(y, (0, 1))):
        raise BaselineError("Labels must be binary 0/1.")
    if y.size == 0 or np.unique(y).size < 2:
        raise BaselineError(
            f"Both classes are required to fit a baseline; got {np.bincount(y, minlength=2).tolist()}."
        )

    started = time.perf_counter()
    candidate_scores: List[Dict[str, Any]] = []

    if hyperparameters is not None:
        chosen = dict(hyperparameters)
        selection_metric = "pinned"
    elif validation_data is None:
        chosen = dict(HYPERPARAMETER_GRIDS[name][0])
        selection_metric = "grid_default_no_validation_data"
        logger.warning(
            "No validation data supplied for %s; using the first grid point %s "
            "without selection.",
            name,
            chosen,
        )
    else:
        from sklearn.metrics import average_precision_score

        x_validation = np.asarray(validation_data[0], dtype=np.float64)
        y_validation = np.asarray(validation_data[1]).astype(int).ravel()
        if np.unique(y_validation).size < 2:
            raise BaselineError(
                "Validation data must contain both classes to select hyperparameters."
            )
        best_score = -np.inf
        chosen = dict(HYPERPARAMETER_GRIDS[name][0])
        for grid_point in HYPERPARAMETER_GRIDS[name]:
            estimator = _fit_one(name, grid_point, x, y, seed=seed)
            score = float(
                average_precision_score(
                    y_validation, estimator.predict_proba(x_validation)[:, 1]
                )
            )
            candidate_scores.append({"hyperparameters": dict(grid_point), "validation_pr_auc": round(score, 6)})
            if score > best_score:
                best_score, chosen = score, dict(grid_point)
        selection_metric = "validation_pr_auc"
        logger.info("  %s selected %s (validation PR-AUC %.4f)", name, chosen, best_score)

    estimator = _fit_one(name, chosen, x, y, seed=seed)
    provenance = metadata or {}
    record = BaselineRecord(
        name=name,
        hyperparameters=chosen,
        candidate_scores=candidate_scores,
        selection_metric=selection_metric,
        random_seed=seed,
        n_train_samples=int(x.shape[0]),
        n_train_positive=int(np.sum(y == 1)),
        feature_dimension=int(x.shape[1]),
        feature_mode=str(provenance.get("feature_mode", "")),
        extractor=str(provenance.get("extractor", "")),
        preprocessing_version=str(provenance.get("preprocessing_version", "")),
        pipeline_version=str(provenance.get("pipeline_version", "")),
        class_weighting="balanced",
        training_duration_seconds=time.perf_counter() - started,
        description=DESCRIPTIONS[name],
    )
    return FittedBaseline(record=record, estimator=estimator)


def _fit_one(
    name: str, hyperparameters: Dict[str, Any], x: np.ndarray, y: np.ndarray, *, seed: int
):
    estimator = build_estimator(name, hyperparameters, seed=seed)
    if name == GRADIENT_BOOSTING:
        # No class_weight parameter, so the same balancing is applied by hand.
        estimator.fit(x, y, sample_weight=_balanced_sample_weight(y))
    else:
        estimator.fit(x, y)
    return estimator


def fit_all_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    names: Sequence[str] = AVAILABLE_BASELINES,
    seed: int = 42,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, FittedBaseline]:
    """Fit each requested baseline on the training partition."""
    fitted: Dict[str, FittedBaseline] = {}
    for name in names:
        logger.info("Fitting %s...", name)
        fitted[name] = fit_baseline(
            name,
            x_train,
            y_train,
            validation_data=validation_data,
            seed=seed,
            metadata=metadata,
        )
    return fitted


def refit_from_record(
    record: BaselineRecord, x_train: np.ndarray, y_train: np.ndarray
) -> FittedBaseline:
    """Rebuild a baseline from its persisted record.

    Reproducibility check as much as a loader: refitting on the same training
    partition with the same seed and hyperparameters must give the same model. If
    it does not, something outside the record is influencing the fit.
    """
    estimator = _fit_one(
        record.name,
        dict(record.hyperparameters),
        np.asarray(x_train, dtype=np.float64),
        np.asarray(y_train).astype(int).ravel(),
        seed=record.random_seed,
    )
    return FittedBaseline(record=record, estimator=estimator)
