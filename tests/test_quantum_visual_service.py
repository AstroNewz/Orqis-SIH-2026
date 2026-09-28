"""E2 quantum-visual runtime service + backend wiring (DEC-035, Candidate B).

This is the runtime companion test to ``tests/test_e2_quantum_visual.py`` (which covers
the fit/evaluate driver and the circuit maths). Here we exercise
:class:`backend.services.quantum_visual_service.QuantumVisualService` -- the object that
actually loads the persisted E2 bundle and executes the real 8-qubit circuit per request
-- and its integration into :meth:`InferenceService.infer_from_image`.

Two things are proven, and they are the two that matter for a demo that must not lie or
break:

1.  **It is real.** The service reconstructs the exact TRAIN-only objects, executes the
    circuit on the Aer statevector, and the reported 16-vector / secondary probability are
    recomputed here from the same circuit and head and shown to match to machine precision.
    Nothing is mocked, hardcoded, or a fabricated accuracy.

2.  **It is inert with respect to the clinical verdict.** Whether E2 succeeds, fails at
    runtime, or is entirely unavailable, every primary / displayed-verdict / persisted /
    FHIR-relevant field of the :class:`InferenceResult` is byte-identical. E2 is an
    additive, EXPERIMENTAL secondary signal (DEC-033/DEC-034: a demonstrator, never an
    advantage claim), and this test is the automated form of that guarantee.

Test-data firewall (NON-NEGOTIABLE): every bundle built here is synthetic and carries
``test_partition_used: false``; no dataset, no test partition, and no frozen test report
is read. The synthetic 576-d vectors are random numbers, never clinical embeddings.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.config import Settings
from backend.ml.artifacts import ArtifactStore
from backend.ml.quantum_visual import (
    MOBILENET_DIM,
    QUANTUM_VISUAL_VERSION,
    LogisticHead,
    QuantumVisualPreprocessor,
    RandomFourierControl,
)
from backend.schemas.inference import QuantumVisualSummary
from backend.services.quantum_visual_service import (
    E2_ARTIFACT_FILE,
    E2_ARTIFACT_VERSION,
    QuantumVisualService,
    QuantumVisualUnavailable,
    get_quantum_visual_service,
    reset_quantum_visual_service,
)
from quantum_ml.visual_circuit import VISUAL_CIRCUIT_VERSION, QuantumVisualFeatureMap
from tests.fixtures import synthetic_capture

# The persisted circuit topology. Kept explicit so a test cannot silently drift with a
# changed default; the service's own guards would reject a mismatch, and so would we.
N_QUBITS = 8
N_BLOCKS = 2
SEED = 7
N_FEATURES = 2 * N_QUBITS  # 8 singles + 8 ring pairs


# --------------------------------------------------------------------------- helpers
def _synthetic_head(n_features_in: int, seed: int) -> LogisticHead:
    """A valid, non-degenerate logistic head built directly as numpy (no sklearn fit).

    :class:`LogisticHead` is a pure ``sigmoid(((x-mean)/scale) @ coef + b)`` map, so a
    hand-built head is a genuine head -- it is exactly what the driver persists, only with
    coefficients we chose rather than fitted. That is all this test needs: it checks the
    *plumbing and provenance*, not the driver's fitted accuracy (that lives in the report).
    """
    rng = np.random.default_rng(seed)
    return LogisticHead(
        mean=np.zeros(n_features_in, dtype=np.float64),
        scale=np.ones(n_features_in, dtype=np.float64),
        coefficients=rng.normal(0.0, 0.5, size=n_features_in),
        intercept=0.0,
        n_features_in=n_features_in,
    )


def _build_bundle(seed: int = SEED, n_train: int = 64) -> dict:
    """Assemble a schema-valid E2 bundle from synthetic TRAIN-only data.

    Mirrors ``backend.evaluation.e2_quantum_visual._persist_artifacts`` key-for-key so the
    service's strict provenance guards see a genuine, self-consistent artifact.
    """
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(0.0, 1.0, size=(n_train, MOBILENET_DIM))

    preprocessor = QuantumVisualPreprocessor.fit(embeddings, seed=seed)
    pca_scores = preprocessor.pca_scores(embeddings)
    rff = RandomFourierControl.fit(pca_scores, n_features=N_FEATURES, seed=seed)
    feature_map = QuantumVisualFeatureMap(n_qubits=N_QUBITS, n_blocks=N_BLOCKS, seed=seed)

    heads = {
        "quantum": _synthetic_head(N_FEATURES, seed + 1),
        "rff": _synthetic_head(N_FEATURES, seed + 2),
        "pca": _synthetic_head(preprocessor.feature_dim, seed + 3),
    }

    return {
        "e2_driver_version": "unit-test",
        "quantum_visual_version": QUANTUM_VISUAL_VERSION,
        "visual_circuit_version": VISUAL_CIRCUIT_VERSION,
        "fitted_on_condition": "A_lesion_polygon",
        "created_at": "2026-01-01T00:00:00+00:00",
        "seed": seed,
        # Firewall certification the service refuses to load without.
        "test_partition_used": False,
        "fitted_on": "synthetic train rows only (unit test)",
        "preprocessor": preprocessor.to_dict(),
        "preprocessor_artifact_hash": preprocessor.artifact_hash,
        "circuit_telemetry": feature_map.telemetry(),
        "observable_labels": list(feature_map.observable_set.labels),
        "rff_control": rff.to_dict(),
        "heads": {name: head.to_dict() for name, head in heads.items()},
        "aer_validation": {
            "max_abs_deviation": 8.88e-16,
            "tolerance": 1e-10,
            "passed": True,
            "n_samples_checked": 8,
            "n_observables": N_FEATURES,
        },
        "environment": {"note": "synthetic bundle for unit test"},
    }


def _service_with(tmp_path, bundle: dict) -> QuantumVisualService:
    """Persist ``bundle`` into a fresh artifact root and return a service pointed at it."""
    store = ArtifactStore(root=tmp_path)
    store.write_component(E2_ARTIFACT_VERSION, E2_ARTIFACT_FILE, bundle)
    return QuantumVisualService(config=Settings(ARTIFACT_DIR=tmp_path))


def _finite_embedding(seed: int = 123) -> np.ndarray:
    """A finite, correctly-shaped stand-in MobileNet embedding (random, not clinical)."""
    return np.random.default_rng(seed).normal(0.0, 1.0, size=MOBILENET_DIM)


@pytest.fixture
def ready_service(tmp_path) -> QuantumVisualService:
    service = _service_with(tmp_path, _build_bundle())
    assert service.ready, "a freshly built, self-consistent bundle must load"
    return service


# ===================================================================== load + status
def test_valid_bundle_loads_and_describes_itself(ready_service):
    info = ready_service.describe()
    assert info["ready"] is True
    # The role must be unambiguous and never claim an advantage.
    assert info["advantage_claimed"] is False
    assert info["role"] == "experimental_secondary_quantum_visual_demonstrator"
    assert info["n_trainable_parameters"] == 0
    assert info["n_qubits"] == N_QUBITS
    assert info["n_quantum_features"] == N_FEATURES
    assert info["fitted_on_condition"] == "A_lesion_polygon"


# ================================================================= real execution
def test_analyze_returns_a_wellformed_secondary_summary(ready_service):
    """One real circuit execution, with every reported field inside its contract."""
    summary = ready_service.analyze_embedding(_finite_embedding())

    assert isinstance(summary, QuantumVisualSummary)
    # -- role invariants: this can never be promoted to the headline.
    assert summary.is_secondary_experimental is True
    assert summary.advantage_claimed is False
    assert summary.n_trainable_parameters == 0

    # -- circuit provenance, read back from the built circuit.
    assert summary.circuit_version == VISUAL_CIRCUIT_VERSION
    assert summary.n_qubits == N_QUBITS
    assert summary.n_reuploading_blocks == N_BLOCKS
    assert summary.n_encoding_parameters == N_QUBITS
    assert summary.state_dimension == 2**N_QUBITS
    assert summary.two_qubit_gate_count > 0
    assert summary.circuit_depth > 0
    assert summary.n_quantum_features == N_FEATURES

    # -- this request's real readout.
    assert 0.0 <= summary.secondary_probability <= 1.0
    assert len(summary.quantum_feature_vector) == N_FEATURES
    # Local Z expectations live in [-1, 1] by definition.
    assert all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in summary.quantum_feature_vector)
    assert len(summary.observable_labels) == N_FEATURES

    # -- entanglement-witness telemetry (NOT an advantage metric).
    assert len(summary.mean_abs_connected_correlation_per_edge) == N_QUBITS
    assert all(c >= 0.0 for c in summary.mean_abs_connected_correlation_per_edge)
    assert summary.max_abs_connected_correlation == pytest.approx(
        max(summary.mean_abs_connected_correlation_per_edge)
    )

    # -- matched controls, always present so the quantum number is never shown alone.
    assert summary.rff_probability is not None and 0.0 <= summary.rff_probability <= 1.0
    assert summary.pca_probability is not None and 0.0 <= summary.pca_probability <= 1.0

    assert summary.execution_time_ms >= 0.0


def test_execution_is_deterministic(ready_service):
    """A seeded exact statevector must give bit-for-bit identical results twice."""
    embedding = _finite_embedding()
    first = ready_service.analyze_embedding(embedding)
    second = ready_service.analyze_embedding(embedding)

    assert first.secondary_probability == second.secondary_probability
    assert first.quantum_feature_vector == second.quantum_feature_vector
    assert (
        first.mean_abs_connected_correlation_per_edge
        == second.mean_abs_connected_correlation_per_edge
    )


def test_reported_features_are_the_real_circuit_output(ready_service):
    """No fabrication: the reported 16-vector equals an independent circuit execution.

    We rebuild the preprocessor and circuit from the *same* bundle and recompute the
    feature vector from scratch; it must match what the service returned to machine
    precision. If the service were returning a canned vector, this fails.
    """
    embedding = _finite_embedding()
    summary = ready_service.analyze_embedding(embedding)

    bundle = _build_bundle()
    preprocessor = QuantumVisualPreprocessor.from_dict(bundle["preprocessor"])
    telemetry = bundle["circuit_telemetry"]
    feature_map = QuantumVisualFeatureMap(
        n_qubits=telemetry["n_qubits"],
        n_blocks=telemetry["n_reuploading_blocks"],
        seed=telemetry["seed"],
    )
    row = embedding.reshape(1, -1)
    expected = feature_map.transform(preprocessor.to_angles(row))[0]

    np.testing.assert_allclose(summary.quantum_feature_vector, expected, atol=1e-12)


def test_reported_probability_is_the_head_applied_to_those_features(ready_service):
    """The secondary probability is the fitted head on the reported features, nothing else."""
    summary = ready_service.analyze_embedding(_finite_embedding())
    head = LogisticHead.from_dict(_build_bundle()["heads"]["quantum"])
    features = np.asarray(summary.quantum_feature_vector, dtype=np.float64).reshape(1, -1)
    expected = float(head.scores(features)[0])
    assert summary.secondary_probability == pytest.approx(expected, abs=1e-12)


# ============================================================ provenance firewall
def test_missing_artifact_is_unavailable_not_fatal(tmp_path):
    """No bundle on disk: the service is simply not ready, and analyze raises cleanly."""
    service = QuantumVisualService(config=Settings(ARTIFACT_DIR=tmp_path))
    assert service.ready is False
    with pytest.raises(QuantumVisualUnavailable):
        service.analyze_embedding(_finite_embedding())


@pytest.mark.parametrize("value", [True, "true", 1, None])
def test_bundle_without_false_partition_flag_is_refused(tmp_path, value):
    """The firewall: anything other than an explicit ``test_partition_used: false`` loses.

    ``None`` covers the key being absent entirely.
    """
    bundle = _build_bundle()
    if value is None:
        bundle.pop("test_partition_used")
    else:
        bundle["test_partition_used"] = value
    assert _service_with(tmp_path, bundle).ready is False


def test_circuit_version_mismatch_is_refused(tmp_path):
    bundle = _build_bundle()
    bundle["visual_circuit_version"] = "v0-not-the-real-circuit"
    assert _service_with(tmp_path, bundle).ready is False


def test_preprocessing_version_mismatch_is_refused(tmp_path):
    bundle = _build_bundle()
    bundle["quantum_visual_version"] = "v0-not-the-real-preprocessing"
    assert _service_with(tmp_path, bundle).ready is False


def test_tampered_preprocessor_hash_is_refused(tmp_path):
    """A stored 8-vector must never be reinterpreted; a broken hash blocks the load."""
    bundle = _build_bundle()
    bundle["preprocessor_artifact_hash"] = "0" * 64
    assert _service_with(tmp_path, bundle).ready is False


def test_circuit_telemetry_mismatch_is_refused(tmp_path):
    """If the recorded topology disagrees with the rebuilt circuit, refuse to serve."""
    bundle = _build_bundle()
    bundle["circuit_telemetry"] = {
        **bundle["circuit_telemetry"],
        "two_qubit_gate_count": bundle["circuit_telemetry"]["two_qubit_gate_count"] + 1,
    }
    assert _service_with(tmp_path, bundle).ready is False


def test_observable_labels_mismatch_is_refused(tmp_path):
    """The readout must be exactly the pre-registered observables, in order.

    The service verifies the labels recorded inside ``circuit_telemetry`` against the
    rebuilt circuit, so that is the field a tamper has to change to matter.
    """
    bundle = _build_bundle()
    telemetry = dict(bundle["circuit_telemetry"])
    telemetry["observable_labels"] = list(reversed(telemetry["observable_labels"]))
    bundle["circuit_telemetry"] = telemetry
    assert _service_with(tmp_path, bundle).ready is False


# ================================================================ input validation
def test_wrong_width_embedding_is_rejected(ready_service):
    with pytest.raises(QuantumVisualUnavailable):
        ready_service.analyze_embedding(np.zeros(MOBILENET_DIM - 1, dtype=np.float64))


def test_non_finite_embedding_is_rejected(ready_service):
    bad = _finite_embedding()
    bad[0] = np.nan
    with pytest.raises(QuantumVisualUnavailable):
        ready_service.analyze_embedding(bad)


# ==================================================================== singleton
def test_singleton_identity_and_reset():
    reset_quantum_visual_service()
    try:
        first = get_quantum_visual_service()
        second = get_quantum_visual_service()
        assert first is second
        reset_quantum_visual_service()
        assert get_quantum_visual_service() is not first
    finally:
        reset_quantum_visual_service()


# =============================================== InferenceService integration
# These need trained classical/quantum artifacts. They skip on a checkout that has not run
# training, exactly like the end-to-end suite, so the suite stays green everywhere.
VOLATILE_FIELDS = (
    "inference_id",
    "created_at",
    "execution_time_ms",
    "quantum_time_ms",
    "preprocessing_time_ms",
    "quantum_visual",
)


def _primary_fingerprint(result) -> dict:
    """Everything a patient/clinician/FHIR export sees, minus non-deterministic noise.

    Explicitly includes ``probability`` (the persisted ``final_probability`` and FHIR
    value), ``risk_level``, every ``primary_*`` field, the calibration and quantum
    provenance, and the disclaimer. Excludes ``quantum_visual`` and per-call timing/ids.
    """
    dump = result.model_dump()
    for key in VOLATILE_FIELDS:
        dump.pop(key, None)
    return dump


def _dummy_summary() -> QuantumVisualSummary:
    """A fully-populated E2 summary to force onto the result, values irrelevant."""
    labels = [f"Z{i}" for i in range(N_QUBITS)]
    labels += [f"Z{i}Z{(i + 1) % N_QUBITS}" for i in range(N_QUBITS)]
    return QuantumVisualSummary(
        circuit_version=VISUAL_CIRCUIT_VERSION,
        n_qubits=N_QUBITS,
        n_reuploading_blocks=N_BLOCKS,
        n_encoding_parameters=N_QUBITS,
        n_trainable_parameters=0,
        two_qubit_gate_count=16,
        circuit_depth=18,
        state_dimension=2**N_QUBITS,
        n_quantum_features=N_FEATURES,
        backend_name="aer_simulator_statevector",
        seed=SEED,
        observable_labels=labels,
        secondary_probability=0.4242,
        quantum_feature_vector=[0.0] * N_FEATURES,
        max_abs_connected_correlation=0.1,
        mean_abs_connected_correlation_per_edge=[0.1] * N_QUBITS,
        execution_time_ms=1.23,
    )


@pytest.fixture
def inference_service():
    """A real, freshly-constructed InferenceService, or a skip if nothing is trained.

    Constructed directly (not the process singleton) so per-test monkeypatching of the E2
    hook cannot leak into other tests or the running app.
    """
    from backend.services.inference_service import InferenceService

    service = InferenceService()
    if not service.ready:
        pytest.skip("No trained model artifacts available for InferenceService.")
    return service


def test_e2_is_additive_and_never_changes_the_primary_verdict(inference_service, monkeypatch):
    """THE safety property: primary/verdict/persisted/FHIR fields are identical whether the
    E2 secondary signal is absent or fully present.

    We drive the *same* photograph through ``infer_from_image`` twice, forcing the E2 hook
    to return ``None`` once and a full summary once. Only ``quantum_visual`` may differ.
    """
    image = synthetic_capture()
    forced = _dummy_summary()

    monkeypatch.setattr(inference_service, "_quantum_visual_summary", lambda *a, **k: None)
    without_e2 = inference_service.infer_from_image(image)

    monkeypatch.setattr(inference_service, "_quantum_visual_summary", lambda *a, **k: forced)
    with_e2 = inference_service.infer_from_image(image)

    assert without_e2.quantum_visual is None
    assert with_e2.quantum_visual is forced
    # The E2 secondary signal is additive telemetry only -- it moves nothing else.
    assert _primary_fingerprint(without_e2) == _primary_fingerprint(with_e2)


def test_runtime_e2_failure_leaves_the_classical_path_intact(inference_service, monkeypatch):
    """If the E2 stage raises mid-request, the result is a clean classical one with
    ``quantum_visual=None`` -- never a 500, never a perturbed verdict (the RUN-2 property).
    """
    image = synthetic_capture()

    # Baseline with E2 explicitly disabled, to compare the primary fields against.
    monkeypatch.setattr(inference_service, "_quantum_visual_summary", lambda *a, **k: None)
    baseline = inference_service.infer_from_image(image)

    # Now let the *real* _quantum_visual_summary run, but make the E2 service explode.
    monkeypatch.undo()

    class _ExplodingService:
        ready = True

        def analyze_embedding(self, _embedding):
            raise RuntimeError("simulated E2 circuit failure")

    monkeypatch.setattr(
        "backend.services.quantum_visual_service.get_quantum_visual_service",
        lambda *a, **k: _ExplodingService(),
    )
    result = inference_service.infer_from_image(image)

    assert result.quantum_visual is None
    assert _primary_fingerprint(result) == _primary_fingerprint(baseline)


def test_descriptor_path_never_carries_a_quantum_visual(inference_service):
    """The on-device descriptor flow has no image, so E2 cannot and must not run."""
    dimension = inference_service.describe()["expected_descriptor_dimension"]
    descriptor = np.random.RandomState(5).uniform(0.0, 1.0, size=dimension).tolist()
    result = inference_service.infer_from_descriptor(descriptor)
    assert result.quantum_visual is None
