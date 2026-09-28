"""Multi-observable Z readout for the 16-qubit amplitude VQC.

Phase D read a single observable, :math:`Z_0`, and measured essentially nothing:
:math:`\\langle Z_0 \\rangle \\in [-0.0094, +0.0091]` with sd 0.0028, which squeezed the
:math:`(1 - \\langle Z_0 \\rangle)/2` score map into ``[0.4953, 0.5047]`` and left the BCE
objective pinned within 0.0025 of :math:`\\ln 2`. This module exists to establish
whether that was a property of the *state* or of the *observable*, which is the
question ``docs/PHASE_E0_DIAGNOSTIC_SPEC.md`` calls E0.1.

What :math:`\\langle Z_S \\rangle` actually measures here
-------------------------------------------------------
For a Pauli-Z string over qubit subset ``S`` (as a bitmask), the expectation on any
state with computational-basis probabilities ``p`` is

.. code-block:: text

    <Z_S> = sum_x p(x) * (-1)**popcount(x & S)

which is the Walsh-Hadamard transform of ``p`` evaluated at ``S``. On the *bare*
encoded state ``p(x) = pixel(x)**2 / ||pixel||**2``, and because
:mod:`backend.ml.pixel_pipeline` flattens in C order (``index = row*256 + col``) while
Qiskit indexes basis states little-endian (qubit ``k`` is bit ``k``), qubits 0-7 are the
**column** bits and 8-15 the **row** bits. So the single-qubit observables are exactly
the dyadic Walsh parities of the squared-intensity image:

===========  ==================================================================
Observable   Spatial meaning on the 256x256 ROI
===========  ==================================================================
``Z_0``      even-column minus odd-column energy (finest horizontal parity)
``Z_7``      left-half minus right-half energy
``Z_8``      even-row minus odd-row energy
``Z_15``     top-half minus bottom-half energy
``Z_i Z_j``  for ``j = i + 8``: diagonal quadrant contrast at matched x/y scale
===========  ==================================================================

That makes the Phase D result mechanical rather than mysterious: ``Z_0`` is the single
*highest-frequency* horizontal Walsh coefficient of the energy image, which is ~0 for
any natural photograph. It is a statement about the chosen projection, not about the
information content of the state -- and separating those two is the whole point of E0.1.

Why this module can afford to be exact
--------------------------------------
The 16-qubit ansatz unitary is 68 GB, so :mod:`quantum_ml.backends` never materialises
it and simulates each sample through Aer instead. But the L-layer hardware-aware ansatz
has a special structure: each layer is a *tensor product* of single-qubit rotations
(``ry`` then ``rz`` per qubit) followed by a CNOT cascade, and **a CNOT cascade is a
permutation of basis states**. So the exact evolved statevector is
16 rank-2 contractions plus one index gather per layer -- about 4.2 Mflop per sample
instead of a 68 GB matrix product, with no approximation anywhere.

This is an implementation shortcut, not a new model, so it is worthless unless it is
provably the same arithmetic. :func:`evolve_states` is asserted against Aer's own
statevector simulation of the identical circuit to 1e-12, and
:func:`observable_expectations` against Aer ``save_expectation_value`` to 1e-10, in
``tests/test_readout.py``. Nothing in E0 reports a number from this module that those
tests do not gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Qubits 0-7 index the column, 8-15 the row. See the module docstring; this is a
#: consequence of C-order flattening plus Qiskit's little-endian basis ordering, and
#: it is verified in the test suite rather than assumed.
COLUMN_QUBITS: Tuple[int, ...] = tuple(range(8))
ROW_QUBITS: Tuple[int, ...] = tuple(range(8, 16))

#: Rows per chunk when evolving a batch. A 16-qubit complex128 statevector is 1 MiB,
#: so 128 rows is a 128 MiB working set -- bounded, and large enough that the per-chunk
#: Python overhead is negligible against the BLAS work.
DEFAULT_CHUNK = 128


class ReadoutError(RuntimeError):
    """Raised when an observable set or a state batch is malformed."""


# --------------------------------------------------------------------- observables
def z_string_diagonal(n_qubits: int, mask: int) -> np.ndarray:
    """``(2**n_qubits,)`` diagonal of the Pauli-Z string selected by ``mask``.

    Entry ``i`` is ``(-1)**popcount(i & mask)``. ``mask = 1`` reproduces
    :func:`quantum_ml.backends.z0_diagonal` exactly, which the test suite asserts.
    """
    if mask < 0 or mask >= (1 << n_qubits):
        raise ReadoutError(
            f"Z-string mask {mask} is outside the {n_qubits}-qubit register."
        )
    indices = np.arange(1 << n_qubits, dtype=np.int64)
    parity = np.zeros(indices.shape, dtype=np.int64)
    remaining = mask
    while remaining:
        bit = remaining & -remaining
        parity ^= (indices & bit) != 0
        remaining ^= bit
    return np.where(parity, -1.0, 1.0)


def fwht(values: np.ndarray) -> np.ndarray:
    """Unnormalised fast Walsh-Hadamard transform along the last axis.

    ``fwht(p)[mask]`` equals ``<Z_mask>`` for a probability vector ``p``, so one call
    yields **all** ``2**n`` Z-string expectations in ``n * 2**n`` operations. E0 does not
    use this for its reported numbers -- it uses the explicit ±1 contraction in
    :func:`observable_expectations`, because a pre-registered handful of observables is
    all the spec permits and BLAS does that faster than a Python-level butterfly. It is
    provided, and tested against that contraction, because §4.2 of the spec claims the
    full spectrum is one transform away and an untested claim is not worth making.
    """
    out = np.array(values, dtype=np.float64, copy=True)
    length = out.shape[-1]
    if length & (length - 1):
        raise ReadoutError(f"FWHT needs a power-of-two length, got {length}.")
    step = 1
    while step < length:
        reshaped = out.reshape(*out.shape[:-1], -1, 2, step)
        left = reshaped[..., 0, :].copy()
        right = reshaped[..., 1, :].copy()
        reshaped[..., 0, :] = left + right
        reshaped[..., 1, :] = left - right
        step *= 2
    return out


@dataclass(frozen=True)
class ObservableSet:
    """A fixed, pre-registered list of Pauli-Z strings with human-readable labels.

    Frozen and constructed only by the module-level builders below. E0.1 forbids
    choosing observables by how well they score, so the set a run uses has to be a
    named constant rather than something assembled at call time.
    """

    name: str
    masks: Tuple[int, ...]
    labels: Tuple[str, ...]
    spatial_notes: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not (len(self.masks) == len(self.labels) == len(self.spatial_notes)):
            raise ReadoutError("ObservableSet fields must be the same length.")
        if len(set(self.masks)) != len(self.masks):
            raise ReadoutError(f"ObservableSet {self.name!r} repeats a Z-string mask.")

    def __len__(self) -> int:
        return len(self.masks)

    def diagonals(self, n_qubits: int) -> np.ndarray:
        """``(n_observables, 2**n_qubits)`` stacked ±1 diagonals."""
        return np.stack([z_string_diagonal(n_qubits, m) for m in self.masks])

    def describe(self) -> List[Dict[str, object]]:
        return [
            {
                "label": label,
                "mask": int(mask),
                "qubits": [k for k in range(64) if mask >> k & 1],
                "weight": int(bin(mask).count("1")),
                "spatial_meaning": note,
            }
            for mask, label, note in zip(self.masks, self.labels, self.spatial_notes)
        ]


def _single_note(qubit: int) -> str:
    if qubit in COLUMN_QUBITS:
        period = 1 << (qubit + 1)
        return (
            f"horizontal Walsh parity of column bit {qubit}: energy difference between "
            f"column blocks of width {period // 2} px, alternating with period {period} px"
        )
    scale = qubit - 8
    period = 1 << (scale + 1)
    return (
        f"vertical Walsh parity of row bit {scale}: energy difference between row blocks "
        f"of height {period // 2} px, alternating with period {period} px"
    )


def _pair_note(low: int, high: int) -> str:
    same_axis = (low in COLUMN_QUBITS) == (high in COLUMN_QUBITS)
    if not same_axis and high - low == 8:
        block = 1 << low
        return (
            f"diagonal quadrant contrast at matched x/y dyadic scale {block} px: "
            "(top-left + bottom-right) minus (top-right + bottom-left) energy over "
            f"{block}x{block} blocks"
        )
    if not same_axis:
        return (
            f"mixed-scale 2D contrast: column bit {low} against row bit {high - 8} "
            "(different dyadic scales in x and y)"
        )
    axis = "horizontal" if low in COLUMN_QUBITS else "vertical"
    return f"{axis} two-scale parity product: bits {low} and {high} on the same axis"


def single_qubit_masks(n_qubits: int) -> Tuple[int, ...]:
    return tuple(1 << k for k in range(n_qubits))


def matched_scale_pair_masks(n_qubits: int = 16) -> Tuple[int, ...]:
    """``Z_i Z_{i+8}``: column bit ``i`` with row bit ``i``, the same dyadic scale.

    The spatially motivated pairing. Any other pairing couples a horizontal feature at
    one scale to a vertical feature at a different scale, which has no reason to align
    with a compact lesion.
    """
    half = n_qubits // 2
    return tuple((1 << i) | (1 << (i + half)) for i in range(half))


def all_pair_masks(n_qubits: int) -> Tuple[int, ...]:
    return tuple(
        (1 << i) | (1 << j) for i in range(n_qubits) for j in range(i + 1, n_qubits)
    )


def _build(name: str, masks: Sequence[int], n_qubits: int) -> ObservableSet:
    labels: List[str] = []
    notes: List[str] = []
    for mask in masks:
        qubits = [k for k in range(n_qubits) if mask >> k & 1]
        labels.append("Z" + "_".join(str(q) for q in qubits) if len(qubits) == 1
                      else "Z{}Z{}".format(*qubits))
        notes.append(_single_note(qubits[0]) if len(qubits) == 1 else _pair_note(*qubits))
    return ObservableSet(name=name, masks=tuple(masks), labels=tuple(labels),
                         spatial_notes=tuple(notes))


def observable_set(variant: str, n_qubits: int = 16) -> ObservableSet:
    """The pre-registered observable set for an E0.1 variant.

    ``A0`` and ``A1`` share the single ``Z_0`` observable and differ only in the readout
    applied to it -- ``A0`` is the frozen Phase D score map, ``A1`` fits a scalar
    logistic head on the standardised value. Keeping them as separate variants over the
    same observable is what makes ``B - A1`` attributable to observable *count* rather
    than to the readout merely becoming trainable.
    """
    if variant in ("A0", "A1"):
        return _build(variant, (1,), n_qubits)
    if variant == "B":
        return _build(variant, single_qubit_masks(n_qubits), n_qubits)
    if variant == "C":
        return _build(
            variant,
            single_qubit_masks(n_qubits) + matched_scale_pair_masks(n_qubits),
            n_qubits,
        )
    if variant == "D":
        return _build(
            variant, single_qubit_masks(n_qubits) + all_pair_masks(n_qubits), n_qubits
        )
    raise ReadoutError(
        f"Unknown E0.1 variant {variant!r}. The pre-registered set is A0, A1, B, C, D; "
        "defining a new one at call time would be the observable search the spec forbids."
    )


#: Expected observable counts, asserted rather than derived, so a refactor that
#: silently changes a set is a test failure instead of a differently-shaped result.
VARIANT_SIZES: Dict[str, int] = {"A0": 1, "A1": 1, "B": 16, "C": 24, "D": 136}


# ------------------------------------------------------------------- state evolution
def cascade_permutation(n_qubits: int, n_layers_worth: int = 1) -> np.ndarray:
    """Gather index ``src`` such that ``evolved = state[src]`` for the CNOT cascade.

    The cascade is ``cx(0,1), cx(1,2), ..., cx(n-2,n-1)`` then ``cx(n-1,0)`` (the
    circular closure, skipped at 2 qubits exactly as
    :meth:`~quantum_ml.vqc_classifier.PixelVQC._build_ansatz_template` skips it).

    A CNOT maps basis index ``i`` to ``i ^ (bit_c(i) << t)``, an involution. For gates
    applied in order ``g1..gm`` the composed unitary is ``Gm...G1``, and
    ``(Gm...G1 v)[j] = v[s1(s2(...sm(j)))]`` -- so the gather index is built by walking
    the gate list **backwards**. Getting that order wrong would silently produce a
    different (still unitary) circuit, which is why the result is asserted against Aer.
    """
    gates: List[Tuple[int, int]] = [(q, q + 1) for q in range(n_qubits - 1)]
    if n_qubits > 2:
        gates.append((n_qubits - 1, 0))
    gates = gates * int(n_layers_worth)

    src = np.arange(1 << n_qubits, dtype=np.int64)
    for control, target in reversed(gates):
        src = src ^ (((src >> control) & 1) << target)
    return src


def _rotation_matrix(phi: float, psi: float) -> np.ndarray:
    """``Rz(psi) @ Ry(phi)`` -- circuit order is ``ry`` then ``rz``, so ``rz`` is left."""
    half_phi, half_psi = phi / 2.0, psi / 2.0
    cos, sin = np.cos(half_phi), np.sin(half_phi)
    minus = np.exp(-1j * half_psi)
    plus = np.exp(1j * half_psi)
    return np.array(
        [[minus * cos, -minus * sin], [plus * sin, plus * cos]], dtype=np.complex128
    )


def _apply_single_qubit(
    states: np.ndarray, qubit: int, gate: np.ndarray, n_qubits: int
) -> np.ndarray:
    """Apply a 2x2 to ``qubit`` across a ``(batch, 2**n)`` block, in place.

    Little-endian indexing means ``index = high * 2**(k+1) + bit_k * 2**k + low``, so a
    C-order reshape to ``(batch, 2**(n-1-k), 2, 2**k)`` puts bit ``k`` on axis 2 with no
    copy. The contraction is then two complex AXPYs over contiguous slabs.
    """
    view = states.reshape(states.shape[0], 1 << (n_qubits - 1 - qubit), 2, 1 << qubit)
    lower = view[:, :, 0, :].copy()
    upper = view[:, :, 1, :].copy()
    view[:, :, 0, :] = gate[0, 0] * lower + gate[0, 1] * upper
    view[:, :, 1, :] = gate[1, 0] * lower + gate[1, 1] * upper
    return states


def evolve_states(
    states: np.ndarray,
    weights: np.ndarray,
    *,
    n_qubits: int = 16,
    n_layers: int = 1,
) -> np.ndarray:
    """Exact ``U(theta) |x>`` for a batch of amplitude-encoded states.

    Mathematically identical to composing
    :meth:`~quantum_ml.vqc_classifier.PixelVQC.bind_ansatz` onto each state and running
    Aer, and asserted so to 1e-12 in the test suite. Returns complex128 of the same
    shape as ``states``.
    """
    states = np.atleast_2d(np.asarray(states))
    expected = 1 << n_qubits
    if states.shape[1] != expected:
        raise ReadoutError(
            f"{n_qubits} qubits needs {expected} amplitudes per row; got "
            f"{states.shape[1]}. Refusing to pad or truncate."
        )
    theta = np.asarray(weights, dtype=np.float64).ravel()
    if theta.size != 2 * n_qubits * n_layers:
        raise ReadoutError(
            f"Expected {2 * n_qubits * n_layers} parameters for {n_qubits} qubits x "
            f"{n_layers} layers, got {theta.size}."
        )

    evolved = np.array(states, dtype=np.complex128, copy=True)
    permutation = cascade_permutation(n_qubits)
    index = 0
    for _ in range(n_layers):
        for qubit in range(n_qubits):
            gate = _rotation_matrix(theta[index], theta[index + 1])
            _apply_single_qubit(evolved, qubit, gate, n_qubits)
            index += 2
        evolved = evolved[:, permutation]
    return evolved


def probabilities_from_amplitudes(states: np.ndarray) -> np.ndarray:
    """``|amplitude|**2`` for the identity circuit -- the pre-ansatz (W-identity) probe.

    No circuit is executed. This is the measurement E0.1 needs to separate a
    representation failure from a readout failure: if no pre-registered observable
    separates the classes *here*, no choice of readout on top of this encoding can, and
    the limitation is the representation rather than ``Z_0``.
    """
    real = np.atleast_2d(np.asarray(states, dtype=np.float64))
    return real * real


def observable_expectations(
    probabilities: np.ndarray, diagonals: np.ndarray
) -> np.ndarray:
    """``(batch, n_observables)`` expectations from basis probabilities.

    One BLAS GEMM against the stacked ±1 diagonals. Equivalent to indexing
    :func:`fwht` at the observable masks, and asserted equal to it and to Aer's
    ``save_expectation_value`` in the test suite.
    """
    probabilities = np.atleast_2d(np.asarray(probabilities, dtype=np.float64))
    if probabilities.shape[1] != diagonals.shape[1]:
        raise ReadoutError(
            f"Probability rows are length {probabilities.shape[1]} but the observable "
            f"diagonals are length {diagonals.shape[1]}."
        )
    return probabilities @ diagonals.T


def measure_batch(
    states: np.ndarray,
    diagonals: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    n_qubits: int = 16,
    n_layers: int = 1,
    chunk: int = DEFAULT_CHUNK,
) -> np.ndarray:
    """``(n_samples, n_observables)`` expectations, chunked to bound peak memory.

    ``weights=None`` is the W-identity probe: no circuit, probabilities straight from
    the encoded amplitudes. Otherwise the batch is evolved through the bound ansatz
    first. Chunking changes nothing about the result -- samples are independent -- it
    only keeps the complex128 working set near :data:`DEFAULT_CHUNK` MiB instead of the
    1.8 GiB a full 1,692-row training partition would need.
    """
    states = np.atleast_2d(np.asarray(states))
    if states.shape[0] == 0:
        return np.empty((0, diagonals.shape[0]), dtype=np.float64)

    out = np.empty((states.shape[0], diagonals.shape[0]), dtype=np.float64)
    step = max(1, int(chunk))
    for start in range(0, states.shape[0], step):
        stop = min(start + step, states.shape[0])
        block = states[start:stop]
        if weights is None:
            probabilities = probabilities_from_amplitudes(block)
        else:
            evolved = evolve_states(
                block, weights, n_qubits=n_qubits, n_layers=n_layers
            )
            probabilities = np.abs(evolved) ** 2
        out[start:stop] = observable_expectations(probabilities, diagonals)
    return out
