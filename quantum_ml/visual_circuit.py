"""E2 fixed 8-qubit angle-encoded quantum visual feature map (DEC-035, Candidate B).

This is the quantum stage of the E2 *demonstrator* -- an experimental secondary
signal, never the clinical decision-maker (DEC-035). It is deliberately the smallest
honest quantum visual computation the SIH fallback clause authorises, and it is *not*
an advantage model:

* **8 qubits**, one per PCA component of the frozen-localizer ROI's MobileNet
  descriptor (the compression + angle mapping live in
  :mod:`backend.ml.quantum_visual`, fitted on TRAIN only).
* **Ry angle encoding** with a **nearest-neighbour CZ entangling ring**
  ``(0,1),(1,2),...,(6,7),(7,0)``, repeated over ``<=2`` data re-uploading blocks.
* **0 trainable gates.** The circuit is a fixed, deterministic function of the input
  angles -- the eight ``Ry`` parameters are *data*, not weights. This is what makes the
  demonstrator ``sqrt(T/N)``-safe by construction (Caro et al. 2022): with no trainable
  quantum parameters there is no generalisation blow-up and no barren plateau to hide.
* **16 pre-registered local Z observables**, frozen by *structure* and never searched:
  8 single-qubit ``<Z_i>`` plus 8 nearest-neighbour ``<Z_i Z_{(i+1) mod 8}>`` matched to
  the CZ ring. Against 103 train positives this 16-feature readout is a defensible
  ratio; the deliberate refusal to use all C(8,2)=28 pairs is the DEC-032 multiplicity
  lesson applied.

Why this can be exact and cheap
-------------------------------
Every one of the 16 observables is a Pauli-Z string, hence *diagonal* in the
computational basis. For a state with basis probabilities ``p`` the expectation is
``<Z_S> = sum_x p(x) (-1)^popcount(x & S)`` -- it depends only on ``|amplitude|**2``.
So the whole 16-vector is a single GEMM ``|psi|**2 @ diagonals^T`` once the exact
statevector is in hand (:func:`quantum_ml.readout.observable_expectations`, reused
unchanged). The statevector itself comes from the existing exact **Aer statevector**
simulator. That fast contraction is asserted equal to Aer's own
``save_expectation_value`` to ``<=1e-10`` in :func:`validate_against_aer`, exactly as
``quantum_ml.readout`` gates its numbers -- nothing here reports a value those checks
do not cover.

Genuinely-quantum handle (telemetry only)
-----------------------------------------
:meth:`QuantumVisualFeatureMap.connected_correlations` exposes the connected
correlations ``C_ij = <Z_i Z_j> - <Z_i><Z_j>`` for the 8 nearest-neighbour pairs. On a
product (separable) state these are identically 0; nonzero values are direct evidence
the CZ ring produced correlations a separable angle encoding could not. This is shown
as an *entanglement witness*, **not** as an advantage metric -- the two are kept
strictly separate throughout E2.

This module reuses :mod:`quantum_ml.readout`'s encoding-agnostic observable layer
(``z_string_diagonal``, ``ObservableSet``, ``observable_expectations``). It deliberately
does **not** reuse ``readout.evolve_states`` / ``cascade_permutation``: those are
specific to the 16-qubit *amplitude*-encoding ansatz and do not describe this circuit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from quantum_ml.readout import ObservableSet, observable_expectations, z_string_diagonal

#: Bump when the circuit *topology* or observable set changes -- persisted artifacts and
#: payloads record it so a stored feature vector can never be silently reinterpreted.
VISUAL_CIRCUIT_VERSION = "v1-e2-visual-circuit-1"

#: One qubit per PCA component (DEC-035 / architecture Section 5.2).
E2_N_QUBITS = 8

#: Data re-uploading blocks. Spec caps this at 2 (Schuld et al. 2021: encoding fixes the
#: accessible function class; a second block widens it without any trainable parameter).
E2_N_REUPLOADING_BLOCKS = 2

#: Fixed, recorded seed. The circuit itself has **no** stochastic gate, so this changes
#: no gate value; it is recorded for provenance and handed to the Aer simulator so a run
#: is byte-reproducible. Determinism is asserted in the test suite regardless.
E2_SEED = 7

#: 8 single-qubit + 8 nearest-neighbour two-body observables.
E2_N_FEATURES = 16

_BACKEND_NAME = "aer_simulator_statevector"


class VisualCircuitError(RuntimeError):
    """Raised when the E2 circuit, observable set, or an input batch is malformed."""


# --------------------------------------------------------------------- topology
def nn_ring_pairs(n_qubits: int = E2_N_QUBITS) -> Tuple[Tuple[int, int], ...]:
    """Nearest-neighbour CZ ring ``(0,1),(1,2),...,(n-2,n-1),(n-1,0)``.

    The wraparound pair ``(n-1, 0)`` closes the ring; it is what makes the last
    observable ``<Z_{n-1} Z_0>`` a genuine nearest-neighbour term rather than a
    long-range one.
    """
    if n_qubits < 2:
        raise VisualCircuitError("The CZ ring needs at least 2 qubits.")
    return tuple((i, (i + 1) % n_qubits) for i in range(n_qubits))


def visual_observable_masks(n_qubits: int = E2_N_QUBITS) -> Tuple[int, ...]:
    """The frozen 16 masks: 8 singles ``1<<i`` then 8 NN pairs ``(1<<i)|(1<<(i+1)%n)``.

    Chosen entirely by *structure* (one per qubit, one per ring edge) and never by any
    score. The order is fixed: singles ascending by qubit, then ring edges ascending by
    their lower qubit, so ``labels``/``masks``/feature columns line up positionally.
    """
    singles = tuple(1 << i for i in range(n_qubits))
    pairs = tuple((1 << i) | (1 << ((i + 1) % n_qubits)) for i in range(n_qubits))
    return singles + pairs


def build_visual_observable_set(n_qubits: int = E2_N_QUBITS) -> ObservableSet:
    """The pre-registered E2 observable set as a frozen :class:`ObservableSet`.

    Labels are ``Z0..Z7`` then ``Z0Z1,Z1Z2,...,Z6Z7,Z7Z0`` -- the ring edges including
    the ``Z7Z0`` wraparound. Reuses ``readout.ObservableSet`` so it inherits the
    equal-length and unique-mask validation, and so the E2 readout is a named constant
    in code rather than something assembled per call (DEC-032 discipline).
    """
    masks = visual_observable_masks(n_qubits)
    single_labels = [f"Z{i}" for i in range(n_qubits)]
    pair_labels = [f"Z{i}Z{(i + 1) % n_qubits}" for i in range(n_qubits)]
    single_notes = [
        f"single-qubit magnetization <Z_{i}> on PCA component {i}"
        for i in range(n_qubits)
    ]
    pair_notes = [
        f"nearest-neighbour two-body correlation <Z_{i} Z_{(i + 1) % n_qubits}> "
        f"on CZ ring edge ({i},{(i + 1) % n_qubits})"
        for i in range(n_qubits)
    ]
    return ObservableSet(
        name="e2-visual-16",
        masks=masks,
        labels=tuple(single_labels + pair_labels),
        spatial_notes=tuple(single_notes + pair_notes),
    )


def build_encoding_circuit(
    n_qubits: int = E2_N_QUBITS, n_blocks: int = E2_N_REUPLOADING_BLOCKS
) -> Tuple[QuantumCircuit, ParameterVector]:
    """The fixed angle-encoding circuit and its data ``ParameterVector``.

    Structure per re-uploading block ``b``: ``Ry(theta_i)`` on every qubit ``i`` (the
    *same* eight data angles each block -- that is what "data re-uploading" means), then
    the nearest-neighbour CZ ring. No parameterised gate is ever trainable: ``theta`` is
    bound to the input angles at transform time and to nothing else.

    Returns the unbound circuit and its ``ParameterVector`` so callers bind by explicit
    ``{theta[i]: angle_i}`` mapping (never by positional order, which lexicographic
    parameter sorting could scramble beyond 10 qubits).
    """
    if not (1 <= n_blocks <= 2):
        raise VisualCircuitError(
            f"n_blocks must be 1 or 2 (spec caps re-uploading at 2); got {n_blocks}."
        )
    theta = ParameterVector("theta", n_qubits)
    circuit = QuantumCircuit(n_qubits, name="e2_visual_reservoir")
    ring = nn_ring_pairs(n_qubits)
    for _ in range(n_blocks):
        for i in range(n_qubits):
            circuit.ry(theta[i], i)
        for a, b in ring:
            circuit.cz(a, b)
    return circuit, theta


def _mask_to_pauli_label(mask: int, n_qubits: int) -> str:
    """Aer/Qiskit Pauli label (most-significant-qubit first) for a Z-string ``mask``.

    Qubit 0 is the *last* character, matching ``backends.expectation_batch_simulated``:
    ``Z_0`` on 8 qubits is ``"IIIIIIIZ"``.
    """
    chars = ["I"] * n_qubits
    for k in range(n_qubits):
        if (mask >> k) & 1:
            chars[n_qubits - 1 - k] = "Z"
    return "".join(chars)


@dataclass
class QuantumVisualFeatureMap:
    """Executes the fixed E2 circuit to an exact Aer statevector and reads 16 observables.

    The heavy objects (circuit, stacked observable diagonals, Aer simulator) are built
    once in ``__post_init__`` and reused, so repeated single-image inference stays warm.
    Nothing here is trainable; the only learned component in E2 is the downstream
    classical head.
    """

    n_qubits: int = E2_N_QUBITS
    n_blocks: int = E2_N_REUPLOADING_BLOCKS
    seed: int = E2_SEED

    _circuit: QuantumCircuit = field(init=False, repr=False)
    _theta: ParameterVector = field(init=False, repr=False)
    _observables: ObservableSet = field(init=False, repr=False)
    _diagonals: np.ndarray = field(init=False, repr=False)
    _simulator: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._circuit, self._theta = build_encoding_circuit(self.n_qubits, self.n_blocks)
        self._observables = build_visual_observable_set(self.n_qubits)
        self._diagonals = self._observables.diagonals(self.n_qubits)

    # ---------------------------------------------------------------- accessors
    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @property
    def observable_set(self) -> ObservableSet:
        return self._observables

    @property
    def diagonals(self) -> np.ndarray:
        return self._diagonals

    def _simulator_instance(self):
        if self._simulator is None:
            from qiskit_aer import AerSimulator

            self._simulator = AerSimulator(
                method="statevector", seed_simulator=self.seed
            )
        return self._simulator

    # ----------------------------------------------------------------- binding
    def _bound(self, angles_row: Sequence[float]) -> QuantumCircuit:
        mapping = {self._theta[i]: float(angles_row[i]) for i in range(self.n_qubits)}
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _prepare_angles(self, angles: np.ndarray) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(angles, dtype=np.float64))
        if arr.shape[1] != self.n_qubits:
            raise VisualCircuitError(
                f"Angle batch has {arr.shape[1]} columns but the circuit spans "
                f"{self.n_qubits} qubits. Refusing to pad or truncate."
            )
        if not np.all(np.isfinite(arr)):
            raise VisualCircuitError("Angle batch contains non-finite values.")
        return arr

    # ---------------------------------------------------------------- execution
    def statevectors(self, angles: np.ndarray) -> np.ndarray:
        """``(n, 2**n_qubits)`` exact Aer statevectors for a batch of angle rows."""
        arr = self._prepare_angles(angles)
        simulator = self._simulator_instance()
        circuits = []
        for row in arr:
            bound = self._bound(row)
            bound.save_statevector()
            circuits.append(bound)
        result = simulator.run(circuits).result()
        dim = 1 << self.n_qubits
        out = np.empty((arr.shape[0], dim), dtype=np.complex128)
        for k in range(arr.shape[0]):
            state = result.data(k)["statevector"]
            out[k] = np.asarray(getattr(state, "data", state), dtype=np.complex128)
        return out

    def transform(self, angles: np.ndarray) -> np.ndarray:
        """``(n, 16)`` quantum feature vector: exact Aer statevector then ``|psi|^2 @ diag^T``.

        The single GEMM is :func:`quantum_ml.readout.observable_expectations`, reused
        unchanged. Every returned value is validatable against Aer's own
        ``save_expectation_value`` via :func:`validate_against_aer`.
        """
        states = self.statevectors(angles)
        probabilities = np.abs(states) ** 2
        return observable_expectations(probabilities, self._diagonals)

    def connected_correlations(self, features: np.ndarray) -> np.ndarray:
        """``(n, n_qubits)`` connected correlations ``C_i = <Z_iZ_{i+1}> - <Z_i><Z_{i+1}>``.

        Entanglement-witness telemetry only. ``features`` is the 16-vector from
        :meth:`transform`; columns ``0..n-1`` are the singles and ``n..2n-1`` the ring
        pairs, so ``C_i`` pairs single ``i`` with single ``(i+1) mod n``.
        """
        arr = np.atleast_2d(np.asarray(features, dtype=np.float64))
        if arr.shape[1] != 2 * self.n_qubits:
            raise VisualCircuitError(
                f"Expected {2 * self.n_qubits} features, got {arr.shape[1]}."
            )
        singles = arr[:, : self.n_qubits]
        pairs = arr[:, self.n_qubits :]
        out = np.empty((arr.shape[0], self.n_qubits), dtype=np.float64)
        for i in range(self.n_qubits):
            out[:, i] = pairs[:, i] - singles[:, i] * singles[:, (i + 1) % self.n_qubits]
        return out

    # ---------------------------------------------------------------- telemetry
    def telemetry(self) -> Dict[str, Any]:
        """Honest, measured circuit provenance for :class:`QuantumVisualSummary`.

        ``n_trainable_parameters`` is 0 by construction; the eight ``Ry`` parameters are
        reported separately as ``n_encoding_parameters`` (data angles) so the distinction
        is never blurred. ``circuit_depth`` and ``two_qubit_gate_count`` are read back
        from the built circuit, not asserted.
        """
        two_qubit = sum(
            1 for instruction in self._circuit.data if instruction.operation.num_qubits == 2
        )
        return {
            "circuit_version": VISUAL_CIRCUIT_VERSION,
            "n_qubits": self.n_qubits,
            "n_reuploading_blocks": self.n_blocks,
            "n_encoding_parameters": self.n_qubits,
            "n_trainable_parameters": 0,
            "two_qubit_gate_count": int(two_qubit),
            "circuit_depth": int(self._circuit.depth()),
            "state_dimension": 1 << self.n_qubits,
            "n_quantum_features": len(self._observables),
            "observable_labels": list(self._observables.labels),
            "seed": self.seed,
            "backend_name": _BACKEND_NAME,
        }


def aer_expectations(
    feature_map: QuantumVisualFeatureMap, angles: np.ndarray
) -> np.ndarray:
    """``(n, 16)`` expectations computed independently by Aer ``save_expectation_value``.

    Each observable is submitted as a :class:`~qiskit.quantum_info.SparsePauliOp` and
    contracted by Aer in C++, so this shares no arithmetic with the fast
    ``|psi|^2 @ diag^T`` path -- which is precisely what makes their agreement in
    :func:`validate_against_aer` meaningful.
    """
    from qiskit.quantum_info import SparsePauliOp

    arr = feature_map._prepare_angles(angles)
    simulator = feature_map._simulator_instance()
    masks = feature_map.observable_set.masks
    labels = [_mask_to_pauli_label(m, feature_map.n_qubits) for m in masks]
    qubit_indices = list(range(feature_map.n_qubits))

    circuits = []
    for row in arr:
        bound = feature_map._bound(row)
        for obs_index, pauli_label in enumerate(labels):
            bound.save_expectation_value(
                SparsePauliOp(pauli_label), qubit_indices, label=f"exp_{obs_index}"
            )
        circuits.append(bound)
    result = simulator.run(circuits).result()

    out = np.empty((arr.shape[0], len(masks)), dtype=np.float64)
    for k in range(arr.shape[0]):
        data = result.data(k)
        for obs_index in range(len(masks)):
            out[k, obs_index] = float(np.real(data[f"exp_{obs_index}"]))
    return out


def validate_against_aer(
    feature_map: QuantumVisualFeatureMap,
    angles: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> Dict[str, Any]:
    """Assert the fast contraction equals Aer ``save_expectation_value`` to ``tolerance``.

    Returns the measured maximum absolute deviation and the tolerance so a payload can
    record the check rather than merely claim it. Raises :class:`VisualCircuitError` if
    the deviation exceeds ``tolerance`` -- no E2 number is reported unless this passes.
    """
    fast = feature_map.transform(angles)
    exact = aer_expectations(feature_map, angles)
    max_abs_deviation = float(np.max(np.abs(fast - exact))) if fast.size else 0.0
    if max_abs_deviation > tolerance:
        raise VisualCircuitError(
            f"Fast |psi|^2 @ diag^T disagrees with Aer save_expectation_value by "
            f"{max_abs_deviation:.3e} > {tolerance:.1e}. Refusing to report E2 features."
        )
    return {
        "max_abs_deviation": max_abs_deviation,
        "tolerance": tolerance,
        "passed": True,
        "n_samples_checked": int(np.atleast_2d(angles).shape[0]),
        "n_observables": len(feature_map.observable_set),
    }
