"""Dimensionality reduction: xfused -> xreduced, sized for amplitude encoding.

Amplitude encoding on ``n`` qubits represents exactly ``2 ** n`` amplitudes, so
the vector handed to the quantum layer must be shaped to that dimension. PART 9
prescribes the two cases explicitly:

- ``d < 2**n``  -> zero-pad. Handled downstream in the quantum encoder, since
  padding is an encoding concern, not a learned transform. This module reports
  ``identity`` and passes the vector through unchanged.
- ``d > 2**n``  -> reduce first, with a **documented** method. That is this
  module: PCA fitted on the training partition only.

Why PCA
-------
PCA is chosen over random projection or feature selection for three reasons that
matter specifically for amplitude encoding:

1. It is the linear map that preserves the most variance for a given output
   dimension, and amplitude encoding is itself a norm-preserving linear
   embedding -- so variance retained is signal retained in the state vector.
2. It is deterministic given the training matrix. Random projection would need a
   seed persisted alongside the model and would discard variance arbitrarily.
3. Its components are inspectable, so the explained-variance ratio quantifies
   exactly what the qubit budget costs. With the 594-dimensional multimodal
   MobileNetV3 vector reduced to 256 amplitudes (8 qubits) that number is
   reportable rather than hand-waved.

Implemented by SVD on the centred training matrix rather than pulled from
scikit-learn, so the fitted state persists as plain arrays with no pickle or
library-version coupling -- an inference artifact must still load in a year.

Whitening is **off** by default. Dividing each component by its singular value
would amplify the low-variance tail, which after L2 normalisation for amplitude
encoding means near-noise directions get the same weight as the leading
components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np

ReductionMethod = Literal["identity", "pca"]
REDUCTION_METHODS: tuple = ("identity", "pca")

REDUCTION_VERSION = "reduction-1"


class ReductionError(ValueError):
    """Raised when reduction is misconfigured or used before fitting."""


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def qubits_for_dimension(dimension: int) -> int:
    """Qubit count whose amplitude space is exactly ``dimension``."""
    if not is_power_of_two(dimension):
        raise ReductionError(
            f"Amplitude-encoding dimension must be a power of two, got {dimension}."
        )
    return int(dimension).bit_length() - 1


def target_dimension_for_qubits(n_qubits: int) -> int:
    """Amplitude-space dimension for ``n_qubits`` (2**n)."""
    if n_qubits < 1:
        raise ReductionError(f"Qubit count must be >= 1, got {n_qubits}.")
    return 2**n_qubits


@dataclass
class DimensionalityReducer:
    """PCA (or pass-through) sized to the quantum encoding dimension.

    The reducer decides its own method at ``fit`` time from the relationship
    between the fused feature dimension and the target: reduction only happens
    when it is actually needed. ``method`` after fitting records what was chosen,
    and that choice is persisted so inference cannot silently differ.
    """

    target_dimension: int
    version: str = REDUCTION_VERSION
    whiten: bool = False
    method: ReductionMethod = "identity"
    mean: Optional[np.ndarray] = None
    components: Optional[np.ndarray] = None
    """``(n_components, n_features)``: rows are the retained principal axes."""
    explained_variance: Optional[np.ndarray] = None
    explained_variance_ratio: Optional[np.ndarray] = None
    n_features_in: int = 0
    n_components: int = 0
    n_samples_seen: int = 0
    input_feature_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.target_dimension < 1:
            raise ReductionError(
                f"target_dimension must be >= 1, got {self.target_dimension}."
            )
        if self.method not in REDUCTION_METHODS:
            raise ReductionError(
                f"Unknown reduction method {self.method!r}. Expected one of {REDUCTION_METHODS}."
            )

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        matrix: np.ndarray,
        *,
        input_feature_names: Optional[List[str]] = None,
    ) -> "DimensionalityReducer":
        """Fit on the **training partition only**.

        Args:
            matrix: ``(n_samples, n_features)`` fused training features.
            input_feature_names: names of the input columns, kept for traceability.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ReductionError(
                f"DimensionalityReducer.fit expects a non-empty 2-D matrix, got {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ReductionError("Training matrix contains non-finite values.")

        n_samples, n_features = matrix.shape
        self.n_features_in = n_features
        self.n_samples_seen = n_samples
        self.input_feature_names = list(input_feature_names or [])

        if n_features <= self.target_dimension:
            # Nothing to reduce: the encoder zero-pads to 2**n. Recording this as
            # an explicit fitted state (rather than "unfitted") keeps the artifact
            # honest about what the pipeline actually does.
            self.method = "identity"
            self.n_components = n_features
            self.mean = None
            self.components = None
            self.explained_variance = None
            self.explained_variance_ratio = None
            return self

        # PCA cannot retain more components than the rank of the centred matrix,
        # which is bounded by n_samples - 1. Truncating here rather than failing
        # keeps small-sample experiments runnable; the shortfall is zero-padded by
        # the encoder and reported by `n_components`.
        max_components = min(self.target_dimension, n_features, max(1, n_samples - 1))
        self.method = "pca"
        self.n_components = int(max_components)

        self.mean = matrix.mean(axis=0)
        centred = matrix - self.mean
        # full_matrices=False keeps the SVD economical: (n, k) x (k,) x (k, d).
        _, singular_values, vt = np.linalg.svd(centred, full_matrices=False)

        self.components = vt[: self.n_components].copy()
        variances = (singular_values**2) / max(1, n_samples - 1)
        total_variance = float(variances.sum())
        self.explained_variance = variances[: self.n_components].copy()
        self.explained_variance_ratio = (
            self.explained_variance / total_variance
            if total_variance > 0
            else np.zeros_like(self.explained_variance)
        )
        return self

    # ----------------------------------------------------------- properties
    @property
    def fitted(self) -> bool:
        if self.n_features_in == 0:
            return False
        if self.method == "identity":
            return True
        return self.components is not None and self.mean is not None

    @property
    def output_dimension(self) -> int:
        """Dimension of the vector this reducer emits (before zero-padding)."""
        return self.n_components

    @property
    def total_explained_variance_ratio(self) -> float:
        """Fraction of training variance retained. ``1.0`` for pass-through."""
        if self.method == "identity":
            return 1.0
        if self.explained_variance_ratio is None:
            return 0.0
        return float(self.explained_variance_ratio.sum())

    @property
    def pads_to_target(self) -> bool:
        """True when the encoder still has to zero-pad to reach the target."""
        return self.n_components < self.target_dimension

    # ------------------------------------------------------------ transform
    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """Project a ``(n_samples, n_features)`` matrix onto the retained axes."""
        if not self.fitted:
            raise ReductionError("DimensionalityReducer used before fit().")
        matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
        if matrix.shape[1] != self.n_features_in:
            raise ReductionError(
                f"Reducer expects {self.n_features_in} features, got {matrix.shape[1]}."
            )
        if self.method == "identity":
            return matrix

        projected = (matrix - self.mean) @ self.components.T
        if self.whiten:
            scale = np.sqrt(np.maximum(self.explained_variance, 1e-12))
            projected = projected / scale
        return projected

    def transform_one(self, vector: np.ndarray) -> np.ndarray:
        """Reduce a single sample to a 1-D vector."""
        return self.transform(np.atleast_2d(vector))[0]

    def inverse_transform(self, reduced: np.ndarray) -> np.ndarray:
        """Map back to feature space. Diagnostic only -- lossy for ``pca``."""
        if not self.fitted:
            raise ReductionError("DimensionalityReducer used before fit().")
        reduced = np.atleast_2d(np.asarray(reduced, dtype=np.float64))
        if self.method == "identity":
            return reduced
        projected = reduced
        if self.whiten:
            projected = projected * np.sqrt(np.maximum(self.explained_variance, 1e-12))
        return projected @ self.components + self.mean

    # -------------------------------------------------------------- summary
    def summary(self) -> Dict[str, Any]:
        """Reportable description of what the reduction actually does."""
        return {
            "version": self.version,
            "method": self.method,
            "n_features_in": self.n_features_in,
            "n_components": self.n_components,
            "target_dimension": self.target_dimension,
            "qubits": qubits_for_dimension(self.target_dimension)
            if is_power_of_two(self.target_dimension)
            else None,
            "pads_to_target": self.pads_to_target,
            "zero_padded_slots": max(0, self.target_dimension - self.n_components),
            "explained_variance_ratio": round(self.total_explained_variance_ratio, 6),
            "whiten": self.whiten,
            "n_samples_seen": self.n_samples_seen,
        }

    # ---------------------------------------------------------- persistence
    def to_dict(self, *, include_arrays: bool = True) -> Dict[str, Any]:
        """JSON-serialisable state.

        ``include_arrays=False`` yields metadata only, for cases where the
        component matrix is stored beside the JSON as ``.npz`` (see
        :meth:`save_arrays`). A 576->256 component matrix is ~1.2 MB as JSON,
        which is acceptable but not free.
        """
        payload: Dict[str, Any] = {
            "version": self.version,
            "method": self.method,
            "target_dimension": self.target_dimension,
            "whiten": self.whiten,
            "n_features_in": self.n_features_in,
            "n_components": self.n_components,
            "n_samples_seen": self.n_samples_seen,
            "input_feature_names": self.input_feature_names,
            "total_explained_variance_ratio": self.total_explained_variance_ratio,
        }
        if include_arrays:
            payload["mean"] = None if self.mean is None else self.mean.tolist()
            payload["components"] = (
                None if self.components is None else self.components.tolist()
            )
            payload["explained_variance"] = (
                None if self.explained_variance is None else self.explained_variance.tolist()
            )
            payload["explained_variance_ratio"] = (
                None
                if self.explained_variance_ratio is None
                else self.explained_variance_ratio.tolist()
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DimensionalityReducer":
        reducer = cls(
            target_dimension=int(payload["target_dimension"]),
            version=str(payload.get("version", REDUCTION_VERSION)),
            whiten=bool(payload.get("whiten", False)),
            method=payload.get("method", "identity"),
        )
        reducer.n_features_in = int(payload.get("n_features_in", 0))
        reducer.n_components = int(payload.get("n_components", 0))
        reducer.n_samples_seen = int(payload.get("n_samples_seen", 0))
        reducer.input_feature_names = list(payload.get("input_feature_names", []))
        for attribute in ("mean", "components", "explained_variance", "explained_variance_ratio"):
            value = payload.get(attribute)
            if value is not None:
                setattr(reducer, attribute, np.asarray(value, dtype=np.float64))
        return reducer

    def save_arrays(self, path: Path) -> Path:
        """Write the fitted arrays to a compressed ``.npz``.

        ``numpy.savez_compressed`` is used rather than pickle: it stores raw
        arrays with no executable payload, so loading an artifact cannot run code.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: getattr(self, name)
            for name in ("mean", "components", "explained_variance", "explained_variance_ratio")
            if getattr(self, name) is not None
        }
        np.savez_compressed(path, **arrays)
        return path

    def load_arrays(self, path: Path) -> "DimensionalityReducer":
        """Load arrays previously written by :meth:`save_arrays`."""
        with np.load(Path(path)) as data:
            for name in data.files:
                setattr(self, name, np.asarray(data[name], dtype=np.float64))
        return self


def build_reducer(
    *,
    n_qubits: Optional[int] = None,
    target_dimension: Optional[int] = None,
    whiten: bool = False,
) -> DimensionalityReducer:
    """Construct a reducer from a qubit count or an explicit target dimension."""
    if target_dimension is None:
        if n_qubits is None:
            raise ReductionError("Provide either n_qubits or target_dimension.")
        target_dimension = target_dimension_for_qubits(n_qubits)
    return DimensionalityReducer(target_dimension=target_dimension, whiten=whiten)
