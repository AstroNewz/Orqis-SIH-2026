"""Tests for the multi-observable Z readout.

Two things are being defended here.

The first is arithmetic. :mod:`quantum_ml.readout` evolves the 16-qubit state in numpy
by exploiting the ansatz's structure -- a tensor product of single-qubit rotations, then
a CNOT cascade that is a pure basis permutation -- instead of going through Aer. That is
a large speedup and it is worth exactly nothing if it computes something subtly
different, so it is asserted against Qiskit's own statevector evolution and against Aer's
``save_expectation_value`` on the real 16-qubit register.

The second is the *spatial interpretation*, which is load-bearing for the E0.1 design.
``docs/PHASE_E0_DIAGNOSTIC_SPEC.md`` claims qubits 0-7 are the column bits and 8-15 the
row bits, and derives from that both why ``Z_0`` measured nothing (it is the finest
horizontal parity) and why ``Z_i Z_{i+8}`` is the spatially motivated pairing (matched
dyadic scale in x and y). If that mapping is wrong, variant C is measuring an arbitrary
qubit pair and its justification evaporates. :class:`TestSpatialMeaning` builds images
with known geometry and checks the observables report it.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from backend.ml.pixel_pipeline import V1_PIXEL_COUNT, V1_QUBIT_COUNT
from quantum_ml.backends import z0_diagonal
from quantum_ml.readout import (
    COLUMN_QUBITS,
    ROW_QUBITS,
    VARIANT_SIZES,
    ReadoutError,
    all_pair_masks,
    cascade_permutation,
    evolve_states,
    fwht,
    matched_scale_pair_masks,
    measure_batch,
    observable_expectations,
    observable_set,
    probabilities_from_amplitudes,
    z_string_diagonal,
)
from quantum_ml.vqc_classifier import VariationalQuantumClassifier


def _random_states(rng, n, n_qubits):
    x = rng.normal(size=(n, 1 << n_qubits))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _pauli_string(mask: int, n_qubits: int) -> str:
    """Qiskit writes Pauli labels most-significant-qubit first, so qubit 0 is last."""
    return "".join("Z" if (mask >> k) & 1 else "I" for k in range(n_qubits))[::-1]


class TestZStringDiagonal:
    def test_z0_matches_the_frozen_backend_helper_exactly(self):
        # Not "to within a tolerance" -- both are +-1 integers in float64, so any
        # difference at all would mean the two paths disagree about qubit ordering.
        for n in (2, 4, 8, 16):
            assert np.array_equal(z_string_diagonal(n, 1), z0_diagonal(n))

    def test_parity_definition(self):
        n = 4
        diagonal = z_string_diagonal(n, 0b0101)
        for index in range(1 << n):
            expected = -1.0 if bin(index & 0b0101).count("1") % 2 else 1.0
            assert diagonal[index] == expected

    def test_identity_mask_is_all_plus_one(self):
        assert np.array_equal(z_string_diagonal(4, 0), np.ones(16))

    def test_mask_outside_the_register_is_refused(self):
        with pytest.raises(ReadoutError, match="outside"):
            z_string_diagonal(4, 1 << 4)


class TestFWHT:
    def test_matches_the_explicit_contraction_for_every_mask(self):
        rng = np.random.default_rng(3)
        n = 8
        p = rng.random(1 << n)
        p /= p.sum()
        direct = np.array([p @ z_string_diagonal(n, m) for m in range(1 << n)])
        assert np.abs(fwht(p) - direct).max() < 1e-12

    def test_rejects_non_power_of_two(self):
        with pytest.raises(ReadoutError, match="power-of-two"):
            fwht(np.ones(6))

    def test_does_not_mutate_its_input(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        before = values.copy()
        fwht(values)
        assert np.array_equal(values, before)


class TestEvolution:
    @pytest.mark.parametrize("n_qubits", [2, 3, 5, 8])
    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_matches_qiskit_statevector_evolution(self, n_qubits, n_layers):
        rng = np.random.default_rng(100 + n_qubits * 10 + n_layers)
        model = VariationalQuantumClassifier(
            num_qubits=n_qubits, num_layers=n_layers, seed=5
        )
        weights = rng.uniform(-np.pi, np.pi, 2 * n_qubits * n_layers)
        bound = model.bind_ansatz(weights)
        states = _random_states(rng, 3, n_qubits)

        mine = evolve_states(states, weights, n_qubits=n_qubits, n_layers=n_layers)
        reference = np.stack([
            np.asarray(Statevector(row.astype(complex)).evolve(bound).data)
            for row in states
        ])
        assert np.abs(mine - reference).max() < 1e-12

    def test_preserves_the_norm(self):
        rng = np.random.default_rng(11)
        states = _random_states(rng, 4, 6)
        evolved = evolve_states(states, rng.uniform(-np.pi, np.pi, 12), n_qubits=6)
        assert np.abs(np.linalg.norm(evolved, axis=1) - 1.0).max() < 1e-12

    def test_does_not_mutate_the_input_states(self):
        rng = np.random.default_rng(12)
        states = _random_states(rng, 2, 5)
        before = states.copy()
        evolve_states(states, rng.uniform(-np.pi, np.pi, 10), n_qubits=5)
        assert np.array_equal(states, before)

    def test_wrong_amplitude_count_is_refused_not_padded(self):
        with pytest.raises(ReadoutError, match="Refusing to pad or truncate"):
            evolve_states(np.ones((1, 100)), np.zeros(12), n_qubits=6)

    def test_wrong_parameter_count_is_refused(self):
        with pytest.raises(ReadoutError, match="Expected 12 parameters"):
            evolve_states(np.ones((1, 64)) / 8, np.zeros(11), n_qubits=6)

    def test_cascade_permutation_is_a_bijection(self):
        for n in (3, 6, 10):
            src = cascade_permutation(n)
            assert np.array_equal(np.sort(src), np.arange(1 << n))

    def test_two_qubit_cascade_skips_the_circular_closure(self):
        # Mirrors the frozen ansatz, which skips cx(1, 0) at exactly 2 qubits. If this
        # diverged, evolve_states would silently model a different circuit at n=2.
        model = VariationalQuantumClassifier(num_qubits=2, num_layers=1, seed=1)
        assert sum(1 for inst in model.build_ansatz()[0].data if inst.operation.name == "cx") == 1


class TestAgainstAer:
    """The equivalence the spec gates every B/C/D number on."""

    def test_measure_batch_matches_aer_expectation_on_the_16_qubit_register(self):
        rng = np.random.default_rng(42)
        n_qubits, n_layers = V1_QUBIT_COUNT, 1
        model = VariationalQuantumClassifier(
            num_qubits=n_qubits, num_layers=n_layers, seed=42
        )
        weights = rng.uniform(-np.pi, np.pi, 2 * n_qubits * n_layers)
        bound = model.bind_ansatz(weights)
        states = _random_states(rng, 2, n_qubits)

        oset = observable_set("C", n_qubits)
        mine = measure_batch(
            states, oset.diagonals(n_qubits), weights=weights,
            n_qubits=n_qubits, n_layers=n_layers,
        )

        simulator = model.backend._simulator
        for column, mask in enumerate(oset.masks):
            observable = SparsePauliOp(_pauli_string(mask, n_qubits))
            circuits = []
            for row in states:
                circuit = QuantumCircuit(n_qubits)
                circuit.set_statevector(Statevector(row.astype(complex)))
                circuit.compose(bound, inplace=True)
                circuit.save_expectation_value(observable, list(range(n_qubits)))
                circuits.append(circuit)
            result = simulator.run(circuits).result()
            aer = np.array([
                float(np.real(result.data(i)["expectation_value"]))
                for i in range(len(states))
            ])
            assert np.abs(mine[:, column] - aer).max() < 1e-10, oset.labels[column]

    def test_z0_column_reproduces_the_frozen_expectation_path(self):
        rng = np.random.default_rng(7)
        n_qubits = V1_QUBIT_COUNT
        model = VariationalQuantumClassifier(num_qubits=n_qubits, num_layers=1, seed=42)
        weights = rng.uniform(-np.pi, np.pi, 2 * n_qubits)
        states = _random_states(rng, 3, n_qubits)
        mine = measure_batch(
            states, z_string_diagonal(n_qubits, 1)[None, :], weights=weights,
            n_qubits=n_qubits,
        ).ravel()
        assert np.abs(mine - model.expectation_states(states, weights)).max() < 1e-10

    def test_fwht_indexing_agrees_with_the_contraction_used_in_production(self):
        rng = np.random.default_rng(19)
        n = 8
        states = _random_states(rng, 2, n)
        weights = rng.uniform(-np.pi, np.pi, 2 * n)
        evolved = evolve_states(states, weights, n_qubits=n)
        probabilities = np.abs(evolved) ** 2
        spectrum = fwht(probabilities)
        oset = observable_set("D", n)
        contracted = observable_expectations(probabilities, oset.diagonals(n))
        for column, mask in enumerate(oset.masks):
            assert np.abs(spectrum[:, mask] - contracted[:, column]).max() < 1e-12


class TestSpatialMeaning:
    """The mapping the E0.1 observable design is derived from.

    ``index = row * 256 + col`` (C order) plus Qiskit's little-endian basis ordering puts
    the column in bits 0-7 and the row in bits 8-15. Everything the spec says about what
    each observable *means* follows from that, so it is checked against images whose
    geometry is known rather than taken on faith.
    """

    EDGE = 256

    def _amplitudes(self, image: np.ndarray) -> np.ndarray:
        from backend.ml.pixel_pipeline import amplitudes_from_grayscale_batch

        # ``reshape(1, -1)`` is C order, which is the flattening the whole mapping
        # depends on: index = row * 256 + col, exactly as grayscale_roi produces.
        flat = image.astype(np.uint8).reshape(1, -1)
        assert flat.shape == (1, V1_PIXEL_COUNT)
        return amplitudes_from_grayscale_batch(flat)

    def _expectation(self, image: np.ndarray, mask: int) -> float:
        amplitudes = self._amplitudes(image)
        probabilities = probabilities_from_amplitudes(amplitudes)
        return float(
            observable_expectations(
                probabilities, z_string_diagonal(V1_QUBIT_COUNT, mask)[None, :]
            )[0, 0]
        )

    def test_column_and_row_qubit_split(self):
        assert COLUMN_QUBITS == tuple(range(8))
        assert ROW_QUBITS == tuple(range(8, 16))

    def test_z7_reads_left_versus_right_half(self):
        image = np.zeros((self.EDGE, self.EDGE), dtype=np.uint8)
        image[:, : self.EDGE // 2] = 255  # left half bright
        assert self._expectation(image, 1 << 7) == pytest.approx(1.0, abs=1e-9)
        flipped = np.zeros_like(image)
        flipped[:, self.EDGE // 2:] = 255  # right half bright
        assert self._expectation(flipped, 1 << 7) == pytest.approx(-1.0, abs=1e-9)

    def test_z15_reads_top_versus_bottom_half(self):
        image = np.zeros((self.EDGE, self.EDGE), dtype=np.uint8)
        image[: self.EDGE // 2, :] = 255  # top half bright
        assert self._expectation(image, 1 << 15) == pytest.approx(1.0, abs=1e-9)

    def test_z0_reads_even_versus_odd_columns(self):
        image = np.zeros((self.EDGE, self.EDGE), dtype=np.uint8)
        image[:, ::2] = 255
        assert self._expectation(image, 1) == pytest.approx(1.0, abs=1e-9)

    def test_z8_reads_even_versus_odd_rows(self):
        image = np.zeros((self.EDGE, self.EDGE), dtype=np.uint8)
        image[::2, :] = 255
        assert self._expectation(image, 1 << 8) == pytest.approx(1.0, abs=1e-9)

    def test_matched_pair_z7z15_reads_the_diagonal_quadrant_contrast(self):
        half = self.EDGE // 2
        image = np.zeros((self.EDGE, self.EDGE), dtype=np.uint8)
        image[:half, :half] = 255   # top-left
        image[half:, half:] = 255   # bottom-right
        mask = (1 << 7) | (1 << 15)
        assert mask in matched_scale_pair_masks(V1_QUBIT_COUNT)
        assert self._expectation(image, mask) == pytest.approx(1.0, abs=1e-9)

        anti = np.zeros_like(image)
        anti[:half, half:] = 255    # top-right
        anti[half:, :half] = 255    # bottom-left
        assert self._expectation(anti, mask) == pytest.approx(-1.0, abs=1e-9)

    def test_z0_is_near_zero_on_a_smooth_image(self):
        """The mechanism behind the Phase D flat loss, reproduced deliberately.

        ``Z_0`` is the finest horizontal parity, so on anything smooth it reports the
        difference between adjacent columns -- which is why the measured spread over the
        real dataset was +-0.009 and the BCE could not move off ln 2.
        """
        gradient = np.tile(
            np.linspace(20, 235, self.EDGE, dtype=np.uint8), (self.EDGE, 1)
        )
        assert abs(self._expectation(gradient, 1)) < 1e-2
        # ...while the coarsest horizontal parity on the same image is large.
        assert abs(self._expectation(gradient, 1 << 7)) > 0.2


class TestObservableSets:
    def test_variant_sizes_are_exactly_as_pre_registered(self):
        for variant, size in VARIANT_SIZES.items():
            assert len(observable_set(variant, V1_QUBIT_COUNT)) == size
        assert VARIANT_SIZES == {"A0": 1, "A1": 1, "B": 16, "C": 24, "D": 136}

    def test_a0_and_a1_share_the_single_z0_observable(self):
        assert observable_set("A0").masks == (1,)
        assert observable_set("A1").masks == (1,)

    def test_each_variant_is_a_subset_of_d(self):
        # run_condition measures D once and slices; if a variant escaped D the slice
        # would raise a KeyError, but asserting it here names the reason.
        superset = set(observable_set("D", V1_QUBIT_COUNT).masks)
        for variant in ("A0", "A1", "B", "C"):
            assert set(observable_set(variant, V1_QUBIT_COUNT).masks) <= superset

    def test_matched_pairs_couple_the_same_dyadic_scale(self):
        for mask in matched_scale_pair_masks(V1_QUBIT_COUNT):
            qubits = [k for k in range(V1_QUBIT_COUNT) if mask >> k & 1]
            assert len(qubits) == 2
            low, high = qubits
            assert low in COLUMN_QUBITS and high in ROW_QUBITS
            assert high - low == 8

    def test_all_pairs_is_the_full_two_local_set(self):
        assert len(all_pair_masks(V1_QUBIT_COUNT)) == 16 * 15 // 2

    def test_no_duplicate_masks_in_any_variant(self):
        for variant in VARIANT_SIZES:
            masks = observable_set(variant, V1_QUBIT_COUNT).masks
            assert len(set(masks)) == len(masks)

    def test_unknown_variant_is_refused_with_the_reason(self):
        with pytest.raises(ReadoutError, match="observable search the spec forbids"):
            observable_set("E")

    def test_describe_reports_spatial_meaning_for_every_observable(self):
        for entry in observable_set("C", V1_QUBIT_COUNT).describe():
            assert entry["spatial_meaning"]
            assert entry["weight"] in (1, 2)


class TestIdentityProbe:
    def test_probabilities_from_amplitudes_needs_no_circuit(self):
        rng = np.random.default_rng(21)
        states = _random_states(rng, 3, V1_QUBIT_COUNT)
        probabilities = probabilities_from_amplitudes(states)
        assert probabilities.shape == (3, V1_PIXEL_COUNT)
        assert np.abs(probabilities.sum(axis=1) - 1.0).max() < 1e-12

    def test_measure_batch_with_no_weights_is_the_identity_circuit(self):
        rng = np.random.default_rng(22)
        n = 6
        states = _random_states(rng, 2, n)
        diagonals = observable_set("B", n).diagonals(n)
        identity = measure_batch(states, diagonals, weights=None, n_qubits=n)
        direct = observable_expectations(states**2, diagonals)
        assert np.abs(identity - direct).max() < 1e-12

    def test_chunking_does_not_change_the_result(self):
        rng = np.random.default_rng(23)
        n = 6
        states = _random_states(rng, 9, n)
        weights = rng.uniform(-np.pi, np.pi, 2 * n)
        diagonals = observable_set("C", n).diagonals(n)
        one = measure_batch(states, diagonals, weights=weights, n_qubits=n, chunk=100)
        many = measure_batch(states, diagonals, weights=weights, n_qubits=n, chunk=2)
        assert np.abs(one - many).max() < 1e-13

    def test_empty_batch_returns_an_empty_matrix(self):
        diagonals = observable_set("B", 4).diagonals(4)
        assert diagonals.shape == (4, 16)
        assert measure_batch(np.empty((0, 16)), diagonals, n_qubits=4).shape == (0, 4)

    def test_mismatched_diagonal_length_is_refused(self):
        with pytest.raises(ReadoutError, match="diagonals are length"):
            observable_expectations(np.ones((2, 16)) / 16, np.ones((1, 8)))
