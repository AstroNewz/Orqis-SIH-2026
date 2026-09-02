"""Amplitude encoding: xreduced -> |psi> on n qubits.

Amplitude encoding represents a ``D``-dimensional real vector as the amplitudes
of an ``n``-qubit state, so ``D = 2**n`` exactly. The reference configurations are

    n =  8  ->  D =    256
    n = 10  ->  D =  1,024
    n = 12  ->  D =  4,096
    n = 16  ->  D = 65,536

and ``n`` is configurable rather than fixed.

PART 9's two shape cases are handled distinctly and reported distinctly:

- ``d < D`` -- **zero-pad**. Padding is done here rather than upstream because it
  is an encoding concern: the padded slots are amplitudes that must be zero, not
  features with value zero.
- ``d > D`` -- **reduce first**. Reduction is a *learned* transform that must be
  fitted on training data only, so it lives in
  :mod:`backend.ml.reduction` and is applied before the vector reaches this
  module. If an oversized vector still arrives here, that is a configuration
  error and the default policy raises rather than truncating: silently dropping
  the tail of a PCA-ordered vector would discard the components carrying the
  least variance without anyone knowing, and dropping the tail of a *non*-PCA
  vector would discard arbitrary features.

  ``oversize_policy="truncate"`` opts into the lossy behaviour explicitly, for
  callers that genuinely want a best-effort state from an oversized vector. The
  distinction is between a documented, requested loss and a silent one.

Validity of the state
---------------------
A quantum state requires :math:`\\sum_i |c_i|^2 = 1`. That is checked explicitly
before the circuit is constructed, not assumed, because ``QuantumCircuit.initialize``
raises a Qiskit-internal error for an invalid state and that error is much harder
to diagnose than a clear one from here.

The all-zero vector is a genuine edge case: it has no direction, so no
normalisation exists. It is treated as an error by default. Callers that must
degrade gracefully can opt into the uniform superposition, which is the only
basis-independent choice, but they get a documented flag rather than a silent
substitution -- a "prediction" from a uniform state carries no information about
the patient and must not be mistaken for one.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit

# Tolerance on sum(|c_i|^2) == 1. Well above float64 round-off for D = 65,536
# (which accumulates ~1e-12 at worst) and far below any real normalisation bug.
NORM_TOLERANCE = 1.0e-9

REFERENCE_QUBIT_COUNTS: Tuple[int, ...] = (8, 10, 12, 16)

ArrayLike = Union[np.ndarray, Sequence[float], List[float]]

OversizePolicy = str  # "error" | "truncate"
OVERSIZE_POLICIES: Tuple[str, ...] = ("error", "truncate")


class EncodingError(ValueError):
    """Raised when a feature vector cannot be amplitude-encoded."""


class ZeroVectorError(EncodingError):
    """Raised when an all-zero vector is encoded without an explicit fallback."""


class QuantumEncoder:
    """Amplitude encoder for an ``n``-qubit register.

    Args:
        num_qubits: ``n``. The state dimension is ``2**n``.
        use_pca: retained for backwards compatibility with the original
            constructor signature. Reduction now lives in
            :mod:`backend.ml.reduction`; :meth:`fit_pca` still works and is used by
            the standalone quantum tests, but the production pipeline fits its
            reducer on the training partition and passes an already-reduced
            vector.
        allow_zero_vector: when ``True``, an all-zero vector becomes the uniform
            superposition instead of raising. Off by default.
        oversize_policy: ``"error"`` (default) rejects a vector longer than
            ``2**n``; ``"truncate"`` keeps the leading ``2**n`` entries and records
            the loss on :attr:`last_truncated_features`.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        use_pca: bool = True,
        *,
        allow_zero_vector: bool = False,
        oversize_policy: OversizePolicy = "error",
    ) -> None:
        if num_qubits < 1:
            raise EncodingError(f"num_qubits must be >= 1, got {num_qubits}.")
        if oversize_policy not in OVERSIZE_POLICIES:
            raise EncodingError(
                f"Unknown oversize_policy {oversize_policy!r}. Expected one of "
                f"{OVERSIZE_POLICIES}."
            )
        self.num_qubits = int(num_qubits)
        self.dim = 2**self.num_qubits
        self.use_pca = use_pca
        self.allow_zero_vector = allow_zero_vector
        self.oversize_policy = oversize_policy
        self.pca = None
        self.last_padded_slots = 0
        """Number of zero-padded amplitudes in the most recent :meth:`transform`."""
        self.last_truncated_features = 0
        """Number of features discarded by the most recent :meth:`transform`."""

    # ------------------------------------------------------------------ shape
    def fit_pca(self, x_train: np.ndarray) -> "QuantumEncoder":
        """Fit an internal PCA for standalone use.

        The production path does **not** use this: reduction belongs to the
        classical pipeline, where it is fitted on the training partition alongside
        the fusion scalers and persisted with them. Kept because it is part of the
        module's existing public surface.
        """
        x_train = np.asarray(x_train, dtype=np.float64)
        if x_train.ndim != 2:
            raise EncodingError(f"fit_pca expects a 2-D matrix, got shape {x_train.shape}.")
        if x_train.shape[1] > self.dim:
            from sklearn.decomposition import PCA

            n_components = min(self.dim, x_train.shape[0], x_train.shape[1])
            self.pca = PCA(n_components=n_components)
            self.pca.fit(x_train)
        return self

    def pad(self, vector: np.ndarray) -> np.ndarray:
        """Fit the vector to ``2**n``: zero-pad if short, apply the oversize policy if long."""
        length = vector.shape[0]
        self.last_truncated_features = 0
        if length > self.dim:
            if self.oversize_policy == "error":
                raise EncodingError(
                    f"Feature vector of length {length} exceeds the {self.dim}-amplitude "
                    f"space of {self.num_qubits} qubits. Reduce it first with "
                    "backend.ml.reduction.DimensionalityReducer (fitted on training "
                    "data), or configure more qubits. Truncating here would discard "
                    "features silently; pass oversize_policy='truncate' to accept "
                    "that loss explicitly."
                )
            self.last_truncated_features = length - self.dim
            self.last_padded_slots = 0
            return vector[: self.dim]
        self.last_padded_slots = self.dim - length
        if length == self.dim:
            return vector
        padded = np.zeros(self.dim, dtype=np.float64)
        padded[:length] = vector
        return padded

    # ---------------------------------------------------------- normalisation
    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """L2-normalise so that :math:`\\sum_i |c_i|^2 = 1`.

        Raises:
            EncodingError: on non-finite input.
            ZeroVectorError: on an all-zero vector, unless ``allow_zero_vector``.
        """
        vector = np.asarray(vector, dtype=np.float64)
        if not np.all(np.isfinite(vector)):
            n_bad = int(np.sum(~np.isfinite(vector)))
            raise EncodingError(
                f"Feature vector contains {n_bad} non-finite value(s); it cannot be "
                "normalised into a quantum state."
            )
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            if not self.allow_zero_vector:
                raise ZeroVectorError(
                    "Cannot amplitude-encode an all-zero feature vector: it has no "
                    "direction and therefore no normalised state. Pass "
                    "allow_zero_vector=True to substitute the uniform superposition, "
                    "but note that the resulting prediction carries no information "
                    "about the input."
                )
            return np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=np.float64)
        return vector / norm

    def validate_state(self, state: np.ndarray) -> None:
        """Assert that ``state`` is a legal amplitude encoding.

        Checked before circuit construction so a malformed vector produces a clear
        error here rather than a Qiskit-internal one three frames deeper.
        """
        state = np.asarray(state)
        if state.ndim != 1:
            raise EncodingError(f"State must be 1-D, got shape {state.shape}.")
        if state.shape[0] != self.dim:
            raise EncodingError(
                f"State has {state.shape[0]} amplitudes but {self.num_qubits} qubits "
                f"require exactly {self.dim}."
            )
        if not np.all(np.isfinite(state)):
            raise EncodingError("State contains non-finite amplitudes.")
        total = float(np.sum(np.abs(state) ** 2))
        if abs(total - 1.0) > NORM_TOLERANCE:
            raise EncodingError(
                f"State is not normalised: sum(|c_i|^2) = {total:.12f}, expected 1.0 "
                f"within {NORM_TOLERANCE:g}."
            )

    # ------------------------------------------------------------- transform
    def transform(self, features: ArrayLike) -> np.ndarray:
        """Feature vector -> validated ``2**n``-amplitude state."""
        x = np.asarray(features, dtype=np.float64).ravel()
        if x.size == 0:
            raise EncodingError("Cannot encode an empty feature vector.")

        if self.pca is not None and x.shape[0] == getattr(self.pca, "n_features_in_", -1):
            x = np.asarray(self.pca.transform(x.reshape(1, -1))).ravel()

        state = self.normalize(self.pad(x))
        self.validate_state(state)
        return state

    def transform_batch(self, matrix: np.ndarray) -> np.ndarray:
        """Encode a whole partition. Returns ``(n_samples, 2**n)``.

        Vectorised because training evaluates every sample on every optimiser
        iteration; encoding row by row would dominate the runtime.
        """
        matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
        if matrix.shape[0] == 0:
            return np.empty((0, self.dim), dtype=np.float64)
        return np.vstack([self.transform(row) for row in matrix])

    # --------------------------------------------------------------- circuit
    def prepare_circuit(self, features: ArrayLike) -> QuantumCircuit:
        """Circuit that prepares the amplitude-encoded state.

        ``QuantumCircuit.initialize`` performs the real state-preparation
        synthesis: Qiskit decomposes the target state into a concrete gate
        sequence, so the transpiled depth reflects the genuine cost of loading a
        ``2**n``-amplitude vector. That cost is substantial, and reporting it
        honestly matters -- state preparation, not the ansatz, dominates the depth
        of an amplitude-encoded circuit.
        """
        return self.circuit_from_state(self.transform(features))

    def circuit_from_state(self, state: np.ndarray) -> QuantumCircuit:
        """Circuit preparing an already-validated state, skipping re-encoding."""
        self.validate_state(state)
        circuit = QuantumCircuit(self.num_qubits, name="AmplitudeEncoding")
        circuit.initialize(state, list(range(self.num_qubits)))
        return circuit

    # --------------------------------------------------------------- summary
    def describe(self) -> dict:
        return {
            "num_qubits": self.num_qubits,
            "dimension": self.dim,
            "is_reference_configuration": self.num_qubits in REFERENCE_QUBIT_COUNTS,
            "allow_zero_vector": self.allow_zero_vector,
            "oversize_policy": self.oversize_policy,
            "norm_tolerance": NORM_TOLERANCE,
            "internal_pca_fitted": self.pca is not None,
        }
