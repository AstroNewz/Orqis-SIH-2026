"""E2 train-only preprocessing, matched control, and inference-safe head (DEC-035).

The quantum circuit in :mod:`quantum_ml.visual_circuit` consumes 8 angles in ``[0, pi]``.
This module is everything between the frozen MobileNet ROI descriptor (576-d) and those
angles, plus the honest classical control the demonstrator is never reported without:

* :class:`PCAModel` -- 576 -> 8 PCA, **fitted on TRAIN only**, re-applied as a pure-numpy
  affine map so inference never imports scikit-learn (the DEC-025 "no pickle, re-apply as
  arithmetic" discipline the logistic baseline already follows).
* :class:`AngleScaler` -- per-component robust **train**-quantile affine map to ``[0, pi]``
  with clipping, so a validation or inference value outside the train range is clamped,
  never allowed to fit a new scale.
* :class:`QuantumVisualPreprocessor` -- ties the two together, carries provenance
  (version, seed, source descriptor, PCA explained variance) and a content
  :pyattr:`~QuantumVisualPreprocessor.artifact_hash` so a persisted 8-vector can never be
  silently reinterpreted.
* :class:`RandomFourierControl` -- DEC-033's named honest analogue for a fixed quantum
  feature map: random Fourier features at **equal output dimension** (16), from the
  **same PCA-8 input**, fitted TRAIN-only (RBF bandwidth by the E0 median heuristic).
* :class:`LogisticHead` -- a fitted logistic head re-applied as pure numpy, so the E2
  secondary probability can be produced at inference without scikit-learn.

Everything fittable exposes ``fit`` (TRAIN rows only), ``to_dict``/``from_dict`` (JSON,
no pickle), and a deterministic transform. No test partition, no validation statistic,
and no label-dependent selection enters any ``fit`` here except the head's own
coefficients, which are what a classifier is.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

#: Bump when the preprocessing *semantics* change so a stored artifact/8-vector cannot be
#: silently reinterpreted against a new pipeline.
QUANTUM_VISUAL_VERSION = "v1-e2-quantum-visual-1"

#: One PCA component per qubit.
PCA_COMPONENTS = 8

#: MobileNetV3-Small embedding width (E1.1 C6 family).
MOBILENET_DIM = 576

#: Robust train quantile for the angle map: the 2nd/98th percentile per component define
#: the encoded range, and out-of-range values clip. Deterministic from train stats alone.
ANGLE_QUANTILE = 0.02

#: Upper bound of the encoded angle range ``[0, ANGLE_MAX]``. With two Ry re-uploading
#: blocks the effective single-qubit rotation folds like ``cos(2*theta)``; over ``[0, pi]``
#: that ``2*theta in [0, 2*pi]`` wraps and is NON-monotonic, so two well-separated PCA
#: values collapse onto the same observable and the head loses linear separability. The
#: E2 TRAIN-internal grouped-CV angle sweep (``backend/evaluation/e2_angle_experiment``)
#: measured this directly and selected ``pi/2`` -- keeping ``2*theta in [0, pi]`` monotonic
#: while the sandwiched CZ ring still entangles (nonzero connected correlations). This is a
#: deterministic TRAIN-only scaling constant; it introduces no trainable parameter. See
#: DEC-036. The default records the selected value; the scaler still carries its own so a
#: persisted artifact is never reinterpreted against a changed module constant.
ANGLE_MAX = float(np.pi / 2.0)

_EPS = 1e-12


class QuantumVisualError(RuntimeError):
    """Raised when an E2 preprocessing/control/head object or input is malformed."""


def _sha256_of(payload: Dict[str, Any]) -> str:
    """Stable content hash of a JSON-able numeric payload (sorted keys, compact)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- PCA
@dataclass(frozen=True)
class PCAModel:
    """A frozen PCA projection re-applied as ``(x - mean) @ components^T``.

    ``components`` is ``(n_components, n_features)`` -- scikit-learn's orientation -- so
    the transform matches ``sklearn.decomposition.PCA.transform`` exactly (mean-centre,
    then project). Persisted as JSON; scikit-learn is needed only to *fit*.
    """

    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    n_components: int
    n_features_in: int
    seed: int

    @classmethod
    def fit(cls, x_train: np.ndarray, *, n_components: int = PCA_COMPONENTS, seed: int) -> "PCAModel":
        from sklearn.decomposition import PCA

        x = np.asarray(x_train, dtype=np.float64)
        if x.ndim != 2:
            raise QuantumVisualError(f"PCA expects a 2-D train matrix, got shape {x.shape}.")
        if x.shape[0] <= n_components:
            raise QuantumVisualError(
                f"PCA needs more train rows ({x.shape[0]}) than components ({n_components})."
            )
        model = PCA(n_components=n_components, svd_solver="full", random_state=seed)
        model.fit(x)
        return cls(
            mean=np.asarray(model.mean_, dtype=np.float64),
            components=np.asarray(model.components_, dtype=np.float64),
            explained_variance_ratio=np.asarray(model.explained_variance_ratio_, dtype=np.float64),
            n_components=int(n_components),
            n_features_in=int(x.shape[1]),
            seed=int(seed),
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if arr.shape[1] != self.n_features_in:
            raise QuantumVisualError(
                f"PCA fitted on {self.n_features_in} features but got {arr.shape[1]}."
            )
        return (arr - self.mean) @ self.components.T

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "components": self.components.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "n_components": self.n_components,
            "n_features_in": self.n_features_in,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PCAModel":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            components=np.asarray(payload["components"], dtype=np.float64),
            explained_variance_ratio=np.asarray(payload["explained_variance_ratio"], dtype=np.float64),
            n_components=int(payload["n_components"]),
            n_features_in=int(payload["n_features_in"]),
            seed=int(payload["seed"]),
        )


# ------------------------------------------------------------------- angle scaler
@dataclass(frozen=True)
class AngleScaler:
    """Per-component robust-quantile affine map of PCA scores to ``[0, max_angle]`` (clipped).

    ``lo``/``hi`` are the ``quantile``/``1-quantile`` percentiles per component on TRAIN.
    A component with ``hi <= lo`` (constant on train, which real 576->8 PCA never
    produces) maps to a constant ``0.0`` -- honest, and it would then be caught by the K2
    degeneracy check downstream rather than hidden. ``max_angle`` is a fixed TRAIN-only
    scaling constant (no trainable parameter); it defaults to :data:`ANGLE_MAX`, the value
    the TRAIN-internal angle sweep selected to keep the two-block ``cos(2*theta)`` response
    monotonic (see DEC-036).
    """

    lo: np.ndarray
    hi: np.ndarray
    quantile: float
    max_angle: float = ANGLE_MAX

    @classmethod
    def fit(
        cls,
        scores_train: np.ndarray,
        *,
        quantile: float = ANGLE_QUANTILE,
        max_angle: float = ANGLE_MAX,
    ) -> "AngleScaler":
        s = np.atleast_2d(np.asarray(scores_train, dtype=np.float64))
        lo = np.percentile(s, 100.0 * quantile, axis=0)
        hi = np.percentile(s, 100.0 * (1.0 - quantile), axis=0)
        return cls(
            lo=np.asarray(lo, dtype=np.float64),
            hi=np.asarray(hi, dtype=np.float64),
            quantile=float(quantile),
            max_angle=float(max_angle),
        )

    def transform(self, scores: np.ndarray) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(scores, dtype=np.float64))
        if arr.shape[1] != self.lo.size:
            raise QuantumVisualError(
                f"AngleScaler fitted on {self.lo.size} components but got {arr.shape[1]}."
            )
        width = self.hi - self.lo
        safe = np.where(width > _EPS, width, 1.0)
        unit = np.clip((arr - self.lo) / safe, 0.0, 1.0)
        # Constant train components carry no information: pin them to angle 0.
        unit = np.where(width > _EPS, unit, 0.0)
        return unit * self.max_angle

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lo": self.lo.tolist(),
            "hi": self.hi.tolist(),
            "quantile": self.quantile,
            "max_angle": self.max_angle,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AngleScaler":
        return cls(
            lo=np.asarray(payload["lo"], dtype=np.float64),
            hi=np.asarray(payload["hi"], dtype=np.float64),
            quantile=float(payload["quantile"]),
            max_angle=float(payload.get("max_angle", ANGLE_MAX)),
        )


# ------------------------------------------------------------- full preprocessor
@dataclass(frozen=True)
class QuantumVisualPreprocessor:
    """TRAIN-fitted MobileNet-576 -> PCA-8 -> angles-in-[0, pi], with provenance."""

    pca: PCAModel
    scaler: AngleScaler
    version: str = QUANTUM_VISUAL_VERSION
    seed: int = 0
    source_descriptor: str = "mobilenet_v3_small_576"

    @classmethod
    def fit(
        cls,
        train_embeddings: np.ndarray,
        *,
        seed: int,
        n_components: int = PCA_COMPONENTS,
        quantile: float = ANGLE_QUANTILE,
        max_angle: float = ANGLE_MAX,
    ) -> "QuantumVisualPreprocessor":
        pca = PCAModel.fit(train_embeddings, n_components=n_components, seed=seed)
        scaler = AngleScaler.fit(
            pca.transform(train_embeddings), quantile=quantile, max_angle=max_angle
        )
        return cls(pca=pca, scaler=scaler, seed=int(seed))

    def pca_scores(self, embeddings: np.ndarray) -> np.ndarray:
        return self.pca.transform(embeddings)

    def to_angles(self, embeddings: np.ndarray) -> np.ndarray:
        """``(n, 8)`` angles in ``[0, pi]`` -- the quantum circuit's only input."""
        return self.scaler.transform(self.pca.transform(embeddings))

    @property
    def feature_dim(self) -> int:
        return self.pca.n_components

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "source_descriptor": self.source_descriptor,
            "feature_dim": self.feature_dim,
            "pca": self.pca.to_dict(),
            "angle_scaler": self.scaler.to_dict(),
        }

    @property
    def artifact_hash(self) -> str:
        return _sha256_of(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "QuantumVisualPreprocessor":
        return cls(
            pca=PCAModel.from_dict(payload["pca"]),
            scaler=AngleScaler.from_dict(payload["angle_scaler"]),
            version=str(payload.get("version", QUANTUM_VISUAL_VERSION)),
            seed=int(payload.get("seed", 0)),
            source_descriptor=str(payload.get("source_descriptor", "mobilenet_v3_small_576")),
        )


# ------------------------------------------------------------ RFF matched control
def _l2_normalise(x: np.ndarray) -> np.ndarray:
    arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norm > _EPS, norm, 1.0)


@dataclass(frozen=True)
class RandomFourierControl:
    """RFF approximation of an RBF kernel -- DEC-033's honest control at equal dimension.

    Operates on the **same PCA-8 input** as the quantum map (L2-normalised, matching the
    E0 median-heuristic convention), emits ``n_features`` (16, equal to the quantum
    vector), and feeds the **same head**. ``z(x) = sqrt(2/D) cos(x_norm W + b)`` with
    ``W ~ N(0, 2 gamma I)`` and ``b ~ U(0, 2 pi)``, ``gamma`` from the TRAIN median
    heuristic. Fitted TRAIN-only; a fixed random map, never learned.
    """

    weights: np.ndarray  # (n_in, n_features)
    bias: np.ndarray  # (n_features,)
    gamma: float
    median_squared_distance: float
    n_features: int
    seed: int

    @classmethod
    def fit(
        cls,
        train_pca_scores: np.ndarray,
        *,
        n_features: int = 16,
        seed: int,
    ) -> "RandomFourierControl":
        from backend.evaluation.e0_controls import median_heuristic_gamma

        x = _l2_normalise(train_pca_scores)
        gram = x @ x.T
        gamma, median_sq = median_heuristic_gamma(gram)
        rng = np.random.default_rng(seed)
        weights = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(x.shape[1], n_features))
        bias = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
        return cls(
            weights=np.asarray(weights, dtype=np.float64),
            bias=np.asarray(bias, dtype=np.float64),
            gamma=float(gamma),
            median_squared_distance=float(median_sq),
            n_features=int(n_features),
            seed=int(seed),
        )

    def transform(self, pca_scores: np.ndarray) -> np.ndarray:
        x = _l2_normalise(pca_scores)
        if x.shape[1] != self.weights.shape[0]:
            raise QuantumVisualError(
                f"RFF fitted on {self.weights.shape[0]} inputs but got {x.shape[1]}."
            )
        projection = x @ self.weights + self.bias
        return np.sqrt(2.0 / self.n_features) * np.cos(projection)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "gamma": self.gamma,
            "median_squared_distance": self.median_squared_distance,
            "n_features": self.n_features,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RandomFourierControl":
        return cls(
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=np.asarray(payload["bias"], dtype=np.float64),
            gamma=float(payload["gamma"]),
            median_squared_distance=float(payload["median_squared_distance"]),
            n_features=int(payload["n_features"]),
            seed=int(payload["seed"]),
        )


# ---------------------------------------------------------- inference-safe head
@dataclass(frozen=True)
class LogisticHead:
    """A fitted logistic head re-applied as pure numpy (sklearn-free inference).

    ``sigmoid(((x - mean) / scale) @ coefficients + intercept)``. Built from an
    :class:`e0_readout.Readout` via :meth:`from_readout_dict` so the E2 driver fits the
    head once (with grouped CV, class weighting) and the backend re-applies it without
    importing scikit-learn -- the same split the logistic baseline uses.
    """

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    n_features_in: int

    def scores(self, x: np.ndarray) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if arr.shape[1] != self.n_features_in:
            raise QuantumVisualError(
                f"Head fitted on {self.n_features_in} features but got {arr.shape[1]}."
            )
        eta = ((arr - self.mean) / self.scale) @ self.coefficients + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -60.0, 60.0)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": float(self.intercept),
            "n_features_in": self.n_features_in,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LogisticHead":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            intercept=float(payload["intercept"]),
            n_features_in=int(payload["n_features_in"]),
        )

    @classmethod
    def from_readout_dict(cls, readout: Dict[str, Any]) -> "LogisticHead":
        """Build from :meth:`e0_readout.Readout.to_dict` output.

        The readout's ``Scaler`` stores ``mean``/``std`` at **full** input width and a
        boolean ``kept`` mask; its ``coefficients`` are over the **kept** columns only
        (``LogisticRegression`` was fitted on ``scaler.transform(x)``, which drops the
        degenerate columns). We rebuild a full-width head: on kept columns
        ``mean``/``std`` and the fitted coefficient; on dropped columns ``mean`` as-is,
        ``scale = 1`` (a degenerate column's ``std`` can be ~0 -- dividing would give
        ``inf`` and ``inf * 0 = nan``), and ``coefficient = 0``. A dropped column then
        contributes ``((x - mean) / 1) * 0 = 0`` exactly, so this reproduces
        :meth:`e0_readout.Readout.scores` to floating point.
        """
        scaler = readout.get("scaler")
        coefficients = readout.get("coefficients")
        if scaler is None or coefficients is None:
            raise QuantumVisualError("Readout dict has no fitted scaler/coefficients to convert.")
        kept_mask = np.asarray(scaler["kept"], dtype=bool)
        full_mean = np.asarray(scaler["mean"], dtype=np.float64)
        full_std = np.asarray(scaler["std"], dtype=np.float64)
        kept_coef = np.asarray(coefficients, dtype=np.float64)
        n_in = int(kept_mask.size)
        if full_mean.size != n_in or full_std.size != n_in:
            raise QuantumVisualError(
                f"Scaler mean/std width ({full_mean.size}/{full_std.size}) does not match "
                f"the kept mask width ({n_in})."
            )
        if int(kept_mask.sum()) != kept_coef.size:
            raise QuantumVisualError(
                f"{int(kept_mask.sum())} kept columns but {kept_coef.size} coefficients."
            )
        mean = full_mean.copy()
        scale = full_std.copy()
        scale[~kept_mask] = 1.0
        coef = np.zeros(n_in, dtype=np.float64)
        coef[kept_mask] = kept_coef
        return cls(
            mean=mean,
            scale=scale,
            coefficients=coef,
            intercept=float(readout.get("intercept", 0.0)),
            n_features_in=n_in,
        )
