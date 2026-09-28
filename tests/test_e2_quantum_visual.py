"""Tests for the E2 fixed 8-qubit angle-encoded quantum visual demonstrator (DEC-035/036).

E2 is the SIH *fallback* quantum visual demonstrator: an experimental **secondary**
signal, never the clinical decision-maker (DEC-033/DEC-034). The dangers these tests
must pin down are the ones that would let the demonstrator lie or leak:

* the test partition is never read and no fitted quantity ever sees a validation row --
  in the driver *and* in the TRAIN-only angle-range experiment;
* the circuit is a fixed function of the data angles: 0 trainable gates, exactly the
  pre-registered 16 local Z observables, never searched, and the fast
  ``|psi|^2 @ diag^T`` contraction agrees with an *independent* Aer
  ``save_expectation_value`` to a tight tolerance -- no E2 number is reported otherwise;
* the "quantum content" is real: with two re-uploading blocks the CZ ring produces
  nonzero connected correlations a separable encoding could not, and with one block it
  is provably zero (the ring commutes through every Z string) -- which is exactly why
  the spec requires two blocks;
* everything fittable (PCA, the angle scaler to ``[0, pi/2]``, the RFF control, the
  logistic head) is TRAIN-only and re-applies as pure arithmetic, and the portable head
  reproduces the sklearn readout to floating point;
* the matched controls (RFF-16, PCA-8) are built at equal dimension on the same head --
  the quantum result is never reported alone.

The real numbers come from the driver against the frozen caches; these tests are what
make those numbers worth reading. Nothing here reads a cache or runs a driver -- the
circuit tests execute the real Aer statevector on small synthetic batches, and the
payload tests skip when no run is on disk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

import backend.evaluation.e2_angle_experiment as angle_exp
import backend.evaluation.e2_quantum_visual as e2
import backend.ml.quantum_visual as qv
import quantum_ml.visual_circuit as vc
from backend.evaluation.e0_readout import _null_summary, _rank_in_null, fit_readout
from backend.evaluation.e2_quantum_visual import (
    CONDITIONS,
    PRIMARY,
    E2Inputs,
    build_representations,
)
from backend.ml.quantum_visual import (
    ANGLE_MAX,
    MOBILENET_DIM,
    PCA_COMPONENTS,
    AngleScaler,
    LogisticHead,
    QuantumVisualError,
    QuantumVisualPreprocessor,
    RandomFourierControl,
)
from backend.training.pixel_data import CONDITION_ORACLE_ALL, CONDITION_ORACLE_LESION
from quantum_ml.readout import ObservableSet, observable_expectations, z_string_diagonal
from quantum_ml.visual_circuit import (
    E2_N_FEATURES,
    E2_N_QUBITS,
    E2_N_REUPLOADING_BLOCKS,
    E2_SEED,
    QuantumVisualFeatureMap,
    VisualCircuitError,
    _mask_to_pauli_label,
    build_encoding_circuit,
    build_visual_observable_set,
    nn_ring_pairs,
    validate_against_aer,
    visual_observable_masks,
)

# --------------------------------------------------------------------------- sources
#: The two modules whose firewall is asserted by inspection: the driver, which loads the
#: caches, and the angle-range experiment, which selects the encoding on TRAIN only.
DRIVER_SOURCE = Path(e2.__file__).read_text(encoding="utf-8")
ANGLE_SOURCE = Path(angle_exp.__file__).read_text(encoding="utf-8")

#: ``DRIVER_SOURCE`` with adjacent implicitly-concatenated string literals joined, so a
#: source-inspection assertion tests the *statement* and not where the author wrapped it.
FLAT_DRIVER_SOURCE = re.sub(r'"\s*\n\s*"', "", DRIVER_SOURCE)

#: The driver's payload. Present only after a full run, so payload assertions skip.
REPORT_PATH = Path("reports_e2_quantum_visual.json")
ANGLE_REPORT_PATH = Path("reports_e2_angle_experiment.json")


@pytest.fixture
def payload() -> dict:
    """The driver's emitted payload, or a skip when no run has produced one yet."""
    if not REPORT_PATH.exists():
        pytest.skip("reports_e2_quantum_visual.json not present; run the driver first")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def angle_payload() -> dict:
    """The angle-experiment payload, or a skip when it has not been run."""
    if not ANGLE_REPORT_PATH.exists():
        pytest.skip("reports_e2_angle_experiment.json not present; run the experiment first")
    return json.loads(ANGLE_REPORT_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------- synthetic data
def _grouped_classification(
    n_patients: int = 20, per_patient: int = 2, n_features: int = 16, seed: int = 0
):
    """A patient-grouped, linearly-separable-ish batch shaped for :func:`fit_readout`.

    Half the patients are positive, each contributes ``per_patient`` images with a
    constant label, and the features carry a modest signal so the logistic head is a
    real (non-degenerate) fit rather than a coin flip.
    """
    rng = np.random.default_rng(seed)
    x, y, patients = [], [], []
    for p in range(n_patients):
        label = p % 2
        for _ in range(per_patient):
            row = rng.normal(size=n_features) + (0.8 * label)
            x.append(row)
            y.append(label)
            patients.append(f"P{p:03d}")
    return np.asarray(x), np.asarray(y, dtype=int), patients


def _mobilenet_inputs(n_train: int = 30, n_val: int = 12, seed: int = 1) -> E2Inputs:
    """A synthetic :class:`E2Inputs` with MobileNet-width descriptors and no cache read.

    Patients are disjoint between train and validation, mirroring the firewall the real
    ``build_inputs`` enforces, so ``build_representations`` runs the real circuit on data
    that never touched disk.
    """
    rng = np.random.default_rng(seed)
    return E2Inputs(
        condition=PRIMARY,
        mobilenet_train=rng.normal(size=(n_train, MOBILENET_DIM)),
        mobilenet_validation=rng.normal(size=(n_val, MOBILENET_DIM)),
        y_train=(rng.random(n_train) < 0.4).astype(int),
        y_validation=(rng.random(n_val) < 0.4).astype(int),
        patients_train=[f"TR{i:03d}" for i in range(n_train)],
        patients_validation=[f"VA{i:03d}" for i in range(n_val)],
        ids_train=[f"tr{i}" for i in range(n_train)],
        ids_validation=[f"va{i}" for i in range(n_val)],
    )


# ----------------------------------------------------------- the test partition is sacred
class TestDriverNeverReadsTheTestPartition:
    """The one claim the whole phase rests on: TRAIN + VALIDATION only, in the driver."""

    def test_the_module_never_selects_the_test_partition(self):
        assert not re.search(r"\.test\b", DRIVER_SOURCE), (
            "the E2 driver must not touch the .test partition attribute"
        )
        assert not re.search(r"partitions\s*\[\s*[\"']test[\"']\s*\]", DRIVER_SOURCE)
        assert not re.search(r"roi_mode\s*=\s*[\"']test[\"']", DRIVER_SOURCE)

    def test_the_only_partitions_iterated_are_train_and_validation(self):
        # build_inputs loops explicitly over ("train", "validation"); no third partition
        # name is ever introduced into that loop.
        assert re.search(r'for partition_name in \("train", "validation"\)', DRIVER_SOURCE)
        assert 'partitions["test"]' not in DRIVER_SOURCE

    def test_the_payload_declares_the_test_partition_unused(self):
        assert '"test_partition_used": False' in DRIVER_SOURCE
        assert '"partitions_read": ["train", "validation"]' in DRIVER_SOURCE
        assert "test_partition_note" in DRIVER_SOURCE

    def test_conditions_are_oracle_only_no_predicted_or_test_condition(self):
        # The two oracle conditions, primary first. A predicted or test condition would
        # be a different experiment; there is deliberately no way to name one here.
        assert PRIMARY == CONDITION_ORACLE_LESION
        assert CONDITIONS == (CONDITION_ORACLE_LESION, CONDITION_ORACLE_ALL)

    def test_no_advantage_claim_is_made(self):
        # DEC-033/034: E2 is the fallback demonstrator, not an advantage model.
        assert "No quantum-advantage claim" in DRIVER_SOURCE


class TestAngleExperimentIsTrainOnly:
    """The angle-range selection reads TRAIN fields only -- never validation or test."""

    def test_no_test_partition_reference(self):
        assert not re.search(r"\.test\b", ANGLE_SOURCE)
        assert not re.search(r"partitions\s*\[\s*[\"']test[\"']\s*\]", ANGLE_SOURCE)

    def test_no_validation_field_is_ever_read(self):
        # build_inputs loads validation, but this module references only the *_train
        # fields of E2Inputs. A single validation-field access would break selection
        # honesty (the winner is chosen on TRAIN-internal CV only).
        for attr in (
            "mobilenet_validation",
            "y_validation",
            "patients_validation",
            "ids_validation",
        ):
            assert attr not in ANGLE_SOURCE, f"angle experiment must not read .{attr}"

    def test_payload_declares_train_only_selection(self):
        assert '"test_partition_used": False' in ANGLE_SOURCE
        assert '"validation_used_for_selection": False' in ANGLE_SOURCE
        assert '"partition_used": "train"' in ANGLE_SOURCE

    def test_only_the_primary_oracle_condition_is_swept(self):
        # The sweep runs build_inputs(conditions=(PRIMARY,)) -- selection is a property of
        # the primary condition, never the secondary or a predicted one.
        assert "conditions=(PRIMARY,)" in ANGLE_SOURCE


class TestFittedQuantitiesSeeTrainOnly:
    def test_no_fit_call_in_the_driver_receives_a_validation_argument(self):
        # Every .fit(...) in the driver is fed a *_train matrix or a shuffled TRAIN copy;
        # validation appears only inside transform / scores / pca_scores calls.
        for call in re.findall(r"\.fit\(([^)]*)\)", FLAT_DRIVER_SOURCE):
            assert "validation" not in call, (
                f".fit() must never receive a validation argument: {call!r}"
            )


# ------------------------------------------------------------------------- frozen spec
class TestFrozenConstants:
    def test_the_circuit_is_eight_qubits_two_blocks_sixteen_features(self):
        assert E2_N_QUBITS == 8
        assert E2_N_REUPLOADING_BLOCKS == 2
        assert E2_N_FEATURES == 16
        assert E2_SEED == 7  # fixed, recorded seed

    def test_the_preprocessor_shape_is_frozen(self):
        assert PCA_COMPONENTS == 8  # one component per qubit
        assert MOBILENET_DIM == 576

    def test_the_angle_ceiling_is_pi_over_two(self):
        # DEC-036: [0, pi/2] keeps the two-block cos(2 theta) response monotonic.
        assert ANGLE_MAX == pytest.approx(np.pi / 2.0)


# --------------------------------------------------------------------------- topology
class TestCircuitTopology:
    def test_nn_ring_is_the_closed_nearest_neighbour_cycle(self):
        assert nn_ring_pairs(8) == (
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),
        )

    def test_the_ring_closes_with_a_wraparound_edge(self):
        pairs = nn_ring_pairs(8)
        assert pairs[-1] == (7, 0), "the ring must close 7->0, not dangle"

    def test_a_ring_needs_at_least_two_qubits(self):
        with pytest.raises(VisualCircuitError):
            nn_ring_pairs(1)

    def test_the_sixteen_masks_are_eight_singles_then_eight_ring_pairs(self):
        masks = visual_observable_masks(8)
        assert len(masks) == E2_N_FEATURES
        assert masks[:8] == tuple(1 << i for i in range(8))
        # ring pairs: (1<<i) | (1<<(i+1)%8); the last wraps 7->0 = 128|1 = 129.
        assert masks[8] == 0b11  # Z0Z1
        assert masks[15] == (1 << 7) | 1  # Z7Z0
        assert len(set(masks)) == len(masks), "masks must be unique"

    def test_the_observable_set_is_the_frozen_sixteen(self):
        obs = build_visual_observable_set(8)
        assert isinstance(obs, ObservableSet)
        assert obs.name == "e2-visual-16"
        assert len(obs) == E2_N_FEATURES
        assert list(obs.labels[:8]) == [f"Z{i}" for i in range(8)]
        assert obs.labels[8] == "Z0Z1"
        assert obs.labels[15] == "Z7Z0"

    def test_observables_are_never_searched_only_structural(self):
        # Every mask is weight 1 (a single) or weight 2 (a ring edge); no higher-order or
        # off-ring pair can appear -- the set is chosen by structure, not by score.
        for mask in visual_observable_masks(8):
            assert bin(mask).count("1") in (1, 2)


class TestEncodingCircuit:
    def test_two_block_circuit_has_no_free_weights_beyond_the_data_angles(self):
        circuit, theta = build_encoding_circuit(8, 2)
        # The eight Ry parameters are *data* angles (bound at transform time); the circuit
        # carries exactly them and nothing learnable.
        assert len(theta) == 8
        assert set(circuit.parameters) == set(theta)

    def test_gate_counts_match_the_two_block_ring_spec(self):
        circuit, _ = build_encoding_circuit(8, 2)
        ops = circuit.count_ops()
        assert ops.get("ry", 0) == 8 * 2  # eight angles re-uploaded per block
        assert ops.get("cz", 0) == 8 * 2  # a full ring per block
        two_qubit = sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)
        assert two_qubit == 16

    def test_one_block_has_a_single_ring(self):
        circuit, _ = build_encoding_circuit(8, 1)
        assert circuit.count_ops().get("cz", 0) == 8
        assert circuit.count_ops().get("ry", 0) == 8

    def test_block_count_is_capped_at_two_and_floored_at_one(self):
        with pytest.raises(VisualCircuitError):
            build_encoding_circuit(8, 0)
        with pytest.raises(VisualCircuitError):
            build_encoding_circuit(8, 3)


class TestPauliLabel:
    def test_qubit_zero_is_the_last_character(self):
        # Little-endian: Z_0 on 8 qubits is "IIIIIIIZ".
        assert _mask_to_pauli_label(1, 8) == "IIIIIIIZ"

    def test_top_qubit_is_the_first_character(self):
        assert _mask_to_pauli_label(1 << 7, 8) == "ZIIIIIII"

    def test_a_ring_pair_places_two_zs(self):
        assert _mask_to_pauli_label(0b11, 8) == "IIIIIIZZ"  # Z0Z1


class TestZStringContract:
    def test_diagonal_entry_is_parity_sign(self):
        # z_string_diagonal(n, mask)[i] = (-1)**popcount(i & mask).
        diag = z_string_diagonal(1, 1)
        assert diag.tolist() == [1.0, -1.0]

    def test_observable_expectation_of_a_known_distribution(self):
        # Uniform 2-qubit state: <Z_0> = 0; delta at |00>: <Z_0> = +1.
        diag_z0 = z_string_diagonal(2, 1).reshape(1, -1)
        uniform = np.full((1, 4), 0.25)
        delta00 = np.array([[1.0, 0.0, 0.0, 0.0]])
        assert observable_expectations(uniform, diag_z0)[0, 0] == pytest.approx(0.0)
        assert observable_expectations(delta00, diag_z0)[0, 0] == pytest.approx(1.0)


# ------------------------------------------------------------------- feature map (Aer)
class TestFeatureMapExecution:
    def test_transform_shape_and_bounds(self):
        fm = QuantumVisualFeatureMap()
        rng = np.random.default_rng(0)
        angles = rng.uniform(0.0, ANGLE_MAX, size=(4, 8))
        features = fm.transform(angles)
        assert features.shape == (4, E2_N_FEATURES)
        # Every observable is a Pauli-Z expectation, so each entry is in [-1, 1].
        assert np.all(features <= 1.0 + 1e-9)
        assert np.all(features >= -1.0 - 1e-9)

    def test_transform_is_deterministic(self):
        fm = QuantumVisualFeatureMap()
        rng = np.random.default_rng(1)
        angles = rng.uniform(0.0, ANGLE_MAX, size=(3, 8))
        assert np.allclose(fm.transform(angles), fm.transform(angles), atol=0.0)

    def test_the_fast_contraction_matches_independent_aer_to_tight_tolerance(self):
        # THE gate: |psi|^2 @ diag^T must equal Aer save_expectation_value. No E2 number
        # is reported unless this passes.
        fm = QuantumVisualFeatureMap()
        rng = np.random.default_rng(2)
        angles = rng.uniform(0.0, ANGLE_MAX, size=(6, 8))
        result = validate_against_aer(fm, angles, tolerance=1e-10)
        assert result["passed"] is True
        assert result["max_abs_deviation"] <= 1e-10
        assert result["n_observables"] == E2_N_FEATURES

    def test_telemetry_reports_zero_trainable_parameters(self):
        tele = QuantumVisualFeatureMap().telemetry()
        assert tele["n_trainable_parameters"] == 0
        assert tele["n_encoding_parameters"] == 8  # the eight data angles
        assert tele["n_qubits"] == 8
        assert tele["n_reuploading_blocks"] == 2
        assert tele["two_qubit_gate_count"] == 16
        assert tele["n_quantum_features"] == E2_N_FEATURES
        assert tele["seed"] == E2_SEED
        assert tele["backend_name"] == "aer_simulator_statevector"

    def test_wrong_angle_width_is_refused_not_padded(self):
        fm = QuantumVisualFeatureMap()
        with pytest.raises(VisualCircuitError):
            fm.transform(np.zeros((1, 7)))

    def test_non_finite_angles_are_refused(self):
        fm = QuantumVisualFeatureMap()
        bad = np.zeros((1, 8))
        bad[0, 0] = np.nan
        with pytest.raises(VisualCircuitError):
            fm.transform(bad)


class TestGenuineQuantumContent:
    """Two blocks entangle; one block cannot. This is why the spec requires two."""

    #: Generic angles well inside (0, pi/2), chosen to avoid product-state special points.
    ANGLES = np.array(
        [
            [0.3, 0.9, 0.5, 1.1, 0.7, 1.3, 0.4, 1.0],
            [1.2, 0.4, 0.8, 0.2, 1.0, 0.6, 1.4, 0.5],
            [0.6, 1.1, 0.3, 1.3, 0.9, 0.4, 0.7, 1.2],
        ]
    )

    def test_two_blocks_produce_nonzero_connected_correlations(self):
        fm = QuantumVisualFeatureMap(n_blocks=2)
        features = fm.transform(self.ANGLES)
        corr = fm.connected_correlations(features)
        # The CZ ring, sandwiched between two Ry layers, produces correlations a separable
        # angle encoding could not. (The driver measures max |C| ~ 0.67 on real TRAIN.)
        assert np.abs(corr).max() > 1e-2

    def test_one_block_is_provably_a_product_state_for_z_observables(self):
        # With a single block the CZ ring (diagonal) commutes through every Z-string
        # observable, so <Z_i Z_j> = <Z_i><Z_j> exactly and every connected correlation
        # is zero to machine precision -- it is not a quantum demonstrator.
        fm = QuantumVisualFeatureMap(n_blocks=1)
        features = fm.transform(self.ANGLES)
        corr = fm.connected_correlations(features)
        assert np.abs(corr).max() < 1e-9

    def test_connected_correlations_reject_wrong_width(self):
        fm = QuantumVisualFeatureMap()
        with pytest.raises(VisualCircuitError):
            fm.connected_correlations(np.zeros((2, 15)))


# --------------------------------------------------------------------------- preprocessor
class TestPreprocessor:
    def test_to_angles_lands_in_zero_to_pi_over_two(self):
        rng = np.random.default_rng(3)
        train = rng.normal(size=(40, MOBILENET_DIM))
        pre = QuantumVisualPreprocessor.fit(train, seed=7)
        angles = pre.to_angles(train)
        assert angles.shape == (40, PCA_COMPONENTS)
        assert np.all(angles >= 0.0)
        assert np.all(angles <= ANGLE_MAX + 1e-12)

    def test_pca_maps_to_eight_components(self):
        rng = np.random.default_rng(4)
        train = rng.normal(size=(40, MOBILENET_DIM))
        pre = QuantumVisualPreprocessor.fit(train, seed=7)
        assert pre.pca.n_components == PCA_COMPONENTS
        assert pre.pca_scores(train).shape == (40, PCA_COMPONENTS)

    def test_validation_outliers_clamp_never_extend_the_train_range(self):
        # A validation value far outside the train range must clip to the ceiling, not
        # refit a wider scale. Train-only scaling is a structural property here.
        rng = np.random.default_rng(5)
        train = rng.normal(size=(40, MOBILENET_DIM))
        pre = QuantumVisualPreprocessor.fit(train, seed=7)
        outlier = np.full((1, MOBILENET_DIM), 1e6)
        angles = pre.to_angles(outlier)
        assert np.all(angles >= -1e-12)
        assert np.all(angles <= ANGLE_MAX + 1e-12)

    def test_round_trips_through_dict(self):
        rng = np.random.default_rng(6)
        train = rng.normal(size=(40, MOBILENET_DIM))
        pre = QuantumVisualPreprocessor.fit(train, seed=7)
        restored = QuantumVisualPreprocessor.from_dict(pre.to_dict())
        assert np.allclose(pre.to_angles(train), restored.to_angles(train))
        assert restored.artifact_hash == pre.artifact_hash

    def test_pca_transform_matches_sklearn_orientation(self):
        # (x - mean) @ components^T reproduces sklearn PCA.transform exactly.
        from sklearn.decomposition import PCA

        rng = np.random.default_rng(7)
        train = rng.normal(size=(40, MOBILENET_DIM))
        pre = QuantumVisualPreprocessor.fit(train, seed=7)
        sk = PCA(n_components=PCA_COMPONENTS, svd_solver="full", random_state=7).fit(train)
        assert np.allclose(pre.pca_scores(train), sk.transform(train))


class TestAngleScaler:
    def test_maps_linearly_into_zero_to_max_angle_with_clipping(self):
        scaler = AngleScaler.fit(np.linspace(0.0, 1.0, 101).reshape(-1, 1))
        # lo=2nd pct=0.02, hi=98th pct=0.98; the midpoint lands at half the ceiling.
        assert scaler.transform([[0.5]])[0, 0] == pytest.approx(0.5 * ANGLE_MAX, abs=1e-6)
        assert scaler.transform([[-5.0]])[0, 0] == pytest.approx(0.0)  # clipped below
        assert scaler.transform([[5.0]])[0, 0] == pytest.approx(ANGLE_MAX)  # clipped above

    def test_constant_train_column_pins_to_angle_zero(self):
        scaler = AngleScaler.fit(np.array([[5.0], [5.0], [5.0]]))
        out = scaler.transform([[5.0], [8.0]])
        assert np.allclose(out, 0.0)
        assert np.all(np.isfinite(out))

    def test_max_angle_defaults_to_the_selected_ceiling_and_is_persisted(self):
        scaler = AngleScaler.fit(np.linspace(0.0, 1.0, 50).reshape(-1, 1))
        assert scaler.max_angle == pytest.approx(ANGLE_MAX)
        assert AngleScaler.from_dict(scaler.to_dict()).max_angle == pytest.approx(ANGLE_MAX)


class TestRandomFourierControl:
    def test_transform_is_bounded_deterministic_and_equal_dimension(self):
        rng = np.random.default_rng(8)
        pca_scores = rng.normal(size=(30, PCA_COMPONENTS))
        rff = RandomFourierControl.fit(pca_scores, n_features=E2_N_FEATURES, seed=7)
        z = rff.transform(pca_scores)
        assert z.shape == (30, E2_N_FEATURES)  # equal to the quantum vector width
        # z = sqrt(2/D) cos(...), so |z| <= sqrt(2/D).
        assert np.all(np.abs(z) <= np.sqrt(2.0 / E2_N_FEATURES) + 1e-12)
        assert np.allclose(z, rff.transform(pca_scores), atol=0.0)

    def test_round_trips_through_dict(self):
        rng = np.random.default_rng(9)
        pca_scores = rng.normal(size=(30, PCA_COMPONENTS))
        rff = RandomFourierControl.fit(pca_scores, n_features=E2_N_FEATURES, seed=7)
        restored = RandomFourierControl.from_dict(rff.to_dict())
        assert np.allclose(rff.transform(pca_scores), restored.transform(pca_scores))


class TestLogisticHeadReproducesReadout:
    def test_portable_head_matches_sklearn_readout_to_floating_point(self):
        x, y, patients = _grouped_classification(seed=10)
        labels = [f"f{i}" for i in range(x.shape[1])]
        readout = fit_readout(x, y, patients, variant="e2_test", labels=labels, seed=7)
        head = LogisticHead.from_readout_dict(readout.to_dict())
        # The whole point of the portable head: sklearn-free inference that reproduces the
        # fitted readout exactly, so the persisted object is validated not trusted.
        assert np.allclose(head.scores(x), readout.scores(x), atol=1e-12)

    def test_a_dropped_degenerate_column_is_reproduced_exactly(self):
        # A constant column is dropped by the readout's scaler (std ~ 0). The portable
        # head must rebuild it as scale=1, coef=0 so it contributes exactly nothing --
        # never inf * 0 = nan.
        x, y, patients = _grouped_classification(seed=11)
        x = np.column_stack([x, np.full(x.shape[0], 3.14)])  # a constant 17th column
        labels = [f"f{i}" for i in range(x.shape[1])]
        readout = fit_readout(x, y, patients, variant="e2_test", labels=labels, seed=7)
        head = LogisticHead.from_readout_dict(readout.to_dict())
        scores = head.scores(x)
        assert np.all(np.isfinite(scores))
        assert np.allclose(scores, readout.scores(x), atol=1e-12)


# --------------------------------------------------------------------------- controls
class TestBuildRepresentations:
    def test_exactly_the_quantum_and_two_matched_controls_at_their_dimensions(self):
        # The quantum result is never reported alone: RFF-16 and PCA-8 are built on the
        # same rows for the same head. Nothing is selected here.
        inputs = _mobilenet_inputs()
        pre = QuantumVisualPreprocessor.fit(inputs.mobilenet_train, seed=7)
        fm = QuantumVisualFeatureMap()
        rff = RandomFourierControl.fit(
            pre.pca_scores(inputs.mobilenet_train), n_features=E2_N_FEATURES, seed=7
        )
        reps = build_representations(inputs, pre, fm, rff)

        assert set(reps) == {"quantum", "rff", "pca"}
        assert reps["quantum"]["train"].shape == (30, E2_N_FEATURES)
        assert reps["quantum"]["validation"].shape == (12, E2_N_FEATURES)
        assert reps["rff"]["train"].shape == (30, E2_N_FEATURES)
        assert reps["pca"]["train"].shape == (30, PCA_COMPONENTS)
        assert reps["quantum"]["is_quantum"] is True
        assert reps["rff"]["is_quantum"] is False
        assert reps["pca"]["is_quantum"] is False


# --------------------------------------------------------------------------- null helpers
class TestGatingNullHelpers:
    def test_null_summary_reports_the_expected_quantiles(self):
        summary = _null_summary([0.1, 0.2, 0.3, 0.4, 0.5])
        assert set(summary) == {"n", "mean", "sd", "p50", "p95", "p99", "max"}
        assert summary["n"] == 5
        assert summary["p50"] == pytest.approx(0.3)

    def test_null_summary_of_nothing_is_none(self):
        assert _null_summary([]) is None

    def test_rank_flags_an_observed_above_the_null_p95(self):
        null = [0.1 + 0.001 * i for i in range(100)]  # ~[0.1, 0.199]
        rank = _rank_in_null(0.9, null)
        assert rank["exceeds_null_p95"] is True
        assert rank["empirical_p"] <= 0.05

    def test_rank_does_not_flag_an_observed_inside_the_null(self):
        null = [0.9 for _ in range(50)]
        rank = _rank_in_null(0.5, null)
        assert rank["exceeds_null_p95"] is False

    def test_rank_of_a_missing_observed_is_none(self):
        assert _rank_in_null(None, [0.1, 0.2, 0.3]) is None


# --------------------------------------------------- the emitted payload, when it exists
class TestEmittedPayload:
    """Assertions against a real run's payload. Skipped when no report is on disk."""

    def test_payload_declares_the_test_partition_unused(self, payload: dict):
        assert payload["test_partition_used"] is False
        assert payload["partitions_read"] == ["train", "validation"]

    def test_primary_condition_is_the_oracle_lesion(self, payload: dict):
        assert payload["primary_condition"] == CONDITION_ORACLE_LESION
        assert payload["primary_keep_kill_decision"] in ("KEEP", "KILL")

    def test_no_patient_appears_in_both_train_and_validation(self, payload: dict):
        for condition in payload["conditions"].values():
            assert condition["rows"]["patient_overlap_train_validation"] == 0

    def test_the_fast_contraction_was_validated_against_aer(self, payload: dict):
        for condition in payload["conditions"].values():
            aer = condition["aer_validation"]
            assert aer["passed"] is True
            assert aer["max_abs_deviation"] <= aer["tolerance"]

    def test_each_condition_carries_quantum_and_both_matched_controls(self, payload: dict):
        for condition in payload["conditions"].values():
            reps = condition["representations"]
            assert {"quantum", "rff", "pca"} <= set(reps)
            controls = condition["matched_controls_vs_quantum"]["controls"]
            assert {"rff", "pca"} <= set(controls)

    def test_the_portable_head_reproduces_the_sklearn_readout(self, payload: dict):
        for condition in payload["conditions"].values():
            for rep in condition["representations"].values():
                assert rep["portable_head_max_deviation_vs_sklearn"] < 1e-9

    def test_the_circuit_carries_no_trainable_quantum_parameters(self, payload: dict):
        for condition in payload["conditions"].values():
            quantum = condition["representations"]["quantum"]
            # The only trainable object is the classical head; the circuit adds none.
            assert quantum["is_quantum"] is True

    def test_the_entanglement_witness_is_present_and_nonzero(self, payload: dict):
        for condition in payload["conditions"].values():
            cct = condition["connected_correlation_telemetry"]
            assert cct["max_abs_connected_correlation"] > 0.0

    def test_the_label_null_is_reported_for_quantum_only(self, payload: dict):
        for condition in payload["conditions"].values():
            reps = condition["representations"]
            assert "label_permutation_null" in reps["quantum"]
            assert "label_permutation_null" not in reps["rff"]
            assert "label_permutation_null" not in reps["pca"]

    def test_keep_kill_records_all_three_gates(self, payload: dict):
        for condition in payload["conditions"].values():
            keep_kill = condition["keep_kill"]
            assert keep_kill["decision"] in ("KEEP", "KILL")
            assert "K1_latency" in keep_kill
            assert "K2_non_degenerate" in keep_kill
            assert "K3_exceeds_own_null_p95" in keep_kill

    def test_no_advantage_claim_in_the_role(self, payload: dict):
        assert "No quantum-advantage claim" in payload["role"]


class TestAngleExperimentPayload:
    """The TRAIN-only angle-range selection payload. Skipped when not on disk."""

    def test_selection_never_touched_validation_or_test(self, angle_payload: dict):
        assert angle_payload["test_partition_used"] is False
        assert angle_payload["validation_used_for_selection"] is False
        assert angle_payload["partition_used"] == "train"

    def test_the_selected_config_is_a_genuinely_entangled_two_block_map(
        self, angle_payload: dict
    ):
        selected = angle_payload["selected"]
        assert selected is not None
        assert selected["n_blocks"] >= 2  # one block is disqualified as a demonstrator
        assert selected["mean_abs_connected_correlation"] > 0.0
