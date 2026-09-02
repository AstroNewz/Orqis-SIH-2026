import numpy as np
import pytest
from quantum_ml.quantum_encoder import (
    EncodingError,
    QuantumEncoder,
    ZeroVectorError,
)
from quantum_ml.vqc_classifier import VariationalQuantumClassifier
from quantum_ml.zne_mitigation import ZNEMitigation
from quantum_ml.calibration import ProbabilityCalibrator


def test_quantum_encoder():
    encoder = QuantumEncoder(num_qubits=4)  # 2^4 = 16 dimensions
    assert encoder.dim == 16

    # Test short vector padding and normalization
    raw = [1.0, 2.0, 3.0]
    encoded = encoder.transform(raw)
    assert len(encoded) == 16
    assert np.isclose(np.linalg.norm(encoded), 1.0)
    assert encoder.last_padded_slots == 13

    # An oversized vector is rejected by default: dimensionality reduction is a
    # learned transform that must be fitted on training data, so it cannot be
    # improvised here (see DEC-023).
    with pytest.raises(EncodingError):
        encoder.transform(np.ones(50))

    # Truncation remains available as an explicit, recorded opt-in.
    lossy = QuantumEncoder(num_qubits=4, oversize_policy="truncate")
    encoded_long = lossy.transform(np.ones(50))
    assert len(encoded_long) == 16
    assert np.isclose(np.linalg.norm(encoded_long), 1.0)
    assert lossy.last_truncated_features == 34

    # Test circuit preparation
    qc = encoder.prepare_circuit(raw)
    assert qc.num_qubits == 4


def test_quantum_encoder_rejects_invalid_vectors():
    """PART 9: invalid vectors, zero vectors, and dimension mismatches."""
    encoder = QuantumEncoder(num_qubits=3)

    with pytest.raises(EncodingError):
        encoder.transform(np.array([]))
    with pytest.raises(EncodingError):
        encoder.transform(np.array([1.0, np.nan, 2.0]))
    with pytest.raises(EncodingError):
        encoder.transform(np.array([1.0, np.inf, 2.0]))

    # An all-zero vector has no direction, so no normalised state exists.
    with pytest.raises(ZeroVectorError):
        encoder.transform(np.zeros(8))

    # Opting in yields the uniform superposition, the only basis-independent choice.
    permissive = QuantumEncoder(num_qubits=3, allow_zero_vector=True)
    uniform = permissive.transform(np.zeros(8))
    assert np.allclose(uniform, 1.0 / np.sqrt(8))

    # validate_state guards the circuit boundary.
    with pytest.raises(EncodingError):
        encoder.validate_state(np.ones(4) / 2.0)  # wrong dimension
    with pytest.raises(EncodingError):
        encoder.validate_state(np.ones(8))  # not normalised
    encoder.validate_state(np.ones(8) / np.sqrt(8))


def test_quantum_encoder_reference_configurations():
    """The reference qubit counts map to the documented state dimensions."""
    assert [QuantumEncoder(num_qubits=n).dim for n in (8, 10, 12, 16)] == [
        256,
        1024,
        4096,
        65536,
    ]
    assert QuantumEncoder(num_qubits=8).describe()["is_reference_configuration"]
    assert not QuantumEncoder(num_qubits=7).describe()["is_reference_configuration"]


def test_quantum_encoder_pca():
    encoder = QuantumEncoder(num_qubits=3)  # D = 8
    X_train = np.random.randn(20, 64)  # 64 features -> 8 dimensions
    encoder.fit_pca(X_train)
    assert encoder.pca is not None

    test_sample = np.random.randn(64)
    encoded = encoder.transform(test_sample)
    assert len(encoded) == 8
    assert np.isclose(np.linalg.norm(encoded), 1.0)


def test_vqc_classifier_simulation():
    # Use 3 qubits for fast unit testing
    vqc = VariationalQuantumClassifier(num_qubits=3, num_layers=1, shots=512)
    features = np.random.randn(8)

    # Expectation value test
    exp_val, meta = vqc.compute_expectation_value(features)
    assert -1.0 <= exp_val <= 1.0
    assert meta["backend"] == "aer_simulator"
    assert meta["num_qubits"] == 3

    # Probability prediction test
    prob, prob_meta = vqc.predict_probability(features)
    assert 0.0 <= prob <= 1.0
    assert np.isclose(prob, (1.0 - prob_meta["expectation_value"]) / 2.0)


def test_vqc_spsa_training_step():
    vqc = VariationalQuantumClassifier(num_qubits=3, num_layers=1, shots=256)
    x_batch = np.random.randn(4, 8)
    y_batch = np.array([0, 1, 0, 1])

    initial_loss = vqc._batch_loss(x_batch, y_batch, vqc.weights)
    new_loss = vqc.train_step_spsa(x_batch, y_batch, lr=0.1, c=0.1)
    assert isinstance(new_loss, float)


def test_zne_mitigation():
    vqc = VariationalQuantumClassifier(num_qubits=3, num_layers=1, shots=256)
    zne = ZNEMitigation(vqc, scale_factors=[1.0, 1.5, 2.0])
    features = np.random.randn(8)

    mitigated_prob, meta = zne.mitigate_expectation(features)
    assert 0.0 <= mitigated_prob <= 1.0
    assert "mitigated_expectation" in meta
    assert "noise_reduction_delta" in meta


def test_probability_calibrator():
    calibrator = ProbabilityCalibrator(threshold=0.50, quantum_weight=0.60)

    # Multimodal fusion
    fused = calibrator.fuse_probabilities(quantum_prob=0.80, classical_prob=0.60)
    # 0.60 * 0.80 + 0.40 * 0.60 = 0.48 + 0.24 = 0.72
    assert np.isclose(fused, 0.72)

    # Brier score
    brier = calibrator.evaluate_brier_score(y_true=[0, 1], y_prob=[0.1, 0.9])
    # ((0.1 - 0)^2 + (0.9 - 1)^2) / 2 = (0.01 + 0.01)/2 = 0.01
    assert np.isclose(brier, 0.01)

    # Risk categorization mock mode
    risk, classification, details = calibrator.categorize_risk(0.85, is_mock=True)
    assert risk == "MOCK HIGH RISK"
    assert classification == "screening_positive"
    assert "TEST DATA" in details

    # Risk categorization real mode
    risk_real, cls_real, _ = calibrator.categorize_risk(0.20, is_mock=False)
    assert risk_real == "LOW RISK"
    assert cls_real == "screening_negative"
