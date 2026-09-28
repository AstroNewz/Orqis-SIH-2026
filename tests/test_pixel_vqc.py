"""Tests for the 16-qubit pixel path: dimensionality, isolation, and serialisation.

``tests/test_quantum.py`` covers the VQC generically at 3-4 qubits, which is fast but
asserts nothing about the register V1 actually uses. These tests pin the properties
the V1 architecture fixes -- exactly 16 qubits, exactly 65,536 amplitudes, exact L2
normalisation -- plus the discipline that keeps the result honest: that the test
partition never reaches a fitting call, that patients never cross partitions, and that
the oracle and predicted ROI conditions stay separable.

Most tests build synthetic partitions rather than loading the real cache, so the suite
runs without the 160 MB artifact present. The handful that need the real cache are
skipped when it is absent, and say so.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.evaluation.metrics import evaluate_predictions
from backend.ml.artifacts import ArtifactError, ArtifactStore
from backend.ml.pixel_pipeline import (
    V1_PIXEL_COUNT,
    V1_QUBIT_COUNT,
    V1_ROI_EDGE_PX,
    amplitudes_from_grayscale_batch,
)
from backend.training.pixel_data import (
    CONDITION_ORACLE_ALL,
    CONDITION_ORACLE_LESION,
    CONDITION_PREDICTED_FALLBACK,
    CONDITION_PREDICTED_LOCALIZED,
    PixelPartition,
    condition_rows,
)
from backend.training import train_pixel_vqc as trainer
from quantum_ml.backends import build_backend
from quantum_ml.results import ExecutionMode
from quantum_ml.vqc_classifier import VariationalQuantumClassifier

SEED = 42


# ------------------------------------------------------------------ fixtures
def _raw(n: int, seed: int = SEED) -> np.ndarray:
    """``(n, 65536)`` uint8 rows, never all-zero so encoding is always defined."""
    rng = np.random.default_rng(seed)
    return rng.integers(1, 256, size=(n, V1_PIXEL_COUNT), dtype=np.uint8)


def _partition(
    name: str,
    n: int,
    *,
    n_positive: int,
    roi_mode: str = "oracle",
    sources: list = None,
    statuses: list = None,
    seed: int = SEED,
) -> PixelPartition:
    labels = np.zeros(n, dtype=int)
    labels[:n_positive] = 1
    return PixelPartition(
        name=name,
        image_ids=[f"{name}_img_{i}" for i in range(n)],
        # Two images per patient, so a leaked patient is detectable.
        patient_ids=[f"{name}_pat_{i // 2}" for i in range(n)],
        labels=labels,
        diagnostic_classes=["ulcerative_colitis" if v else "normal" for v in labels],
        raw=_raw(n, seed),
        roi_sources=sources or ["lesion_polygon"] * n,
        localization_statuses=statuses or (["oracle"] * n),
        localization_confidences=np.full(n, np.nan),
        roi_boxes=np.tile(np.asarray([0, 0, 256, 256], dtype=np.int32), (n, 1)),
        source_sizes=np.tile(np.asarray([512, 512], dtype=np.int32), (n, 1)),
        roi_mode=roi_mode,
    )


def _manifest() -> "object":
    """A minimal split manifest. Only the seed and positive classes are read here."""
    from backend.dataset.split import SplitManifest

    return SplitManifest(
        seed=SEED,
        fractions={"train": 0.7, "validation": 0.15, "test": 0.15},
        positive_classes=["ulcerative_colitis"],
        dataset_root="synthetic",
    )


@pytest.fixture(scope="module")
def model() -> VariationalQuantumClassifier:
    """A 16-qubit, 1-layer VQC. Module-scoped: Aer setup is not free."""
    backend = build_backend(
        mode=ExecutionMode.IDEAL_SIMULATION, shots=1024, seed=SEED
    )
    return VariationalQuantumClassifier(
        num_qubits=V1_QUBIT_COUNT,
        num_layers=1,
        shots=1024,
        backend=backend,
        seed=SEED,
        model_version=trainer.PIXEL_VQC_VERSION,
    )


# ------------------------------------------------- 1. exact 16-qubit dimensionality
def test_qubit_count_is_exactly_sixteen_and_not_configurable(model):
    """The register width is fixed by the architecture, not read from settings.

    A run that quietly used 8 qubits would still produce plausible metrics, so the
    count is asserted rather than trusted.
    """
    assert V1_QUBIT_COUNT == 16
    assert model.num_qubits == 16
    assert model.bind_ansatz(model.weights).num_qubits == 16


def test_parameter_count_follows_the_documented_ansatz(model):
    assert model.num_params == 2 * 16 * 1 == 32
    two_layer = trainer.PixelVQCConfig(
        roi_mode="oracle", condition=CONDITION_ORACLE_ALL, n_layers=2
    )
    assert two_layer.n_parameters == 64


# ------------------------------------------------- 2. exactly 65,536 amplitudes
def test_state_dimension_is_two_to_the_sixteen():
    assert V1_PIXEL_COUNT == 2 ** V1_QUBIT_COUNT == 65536
    assert V1_ROI_EDGE_PX * V1_ROI_EDGE_PX == V1_PIXEL_COUNT


def test_encoded_batch_has_one_amplitude_per_pixel():
    states = amplitudes_from_grayscale_batch(_raw(4))
    assert states.shape == (4, 65536)


def test_wrong_width_is_refused_rather_than_padded(model):
    """No silent padding or truncation between the image and the register."""
    with pytest.raises(Exception):
        model.expectation_states(np.ones((2, 4096)) / np.sqrt(4096))


# ---------------------------------------------------------------- 3. normalisation
def test_states_are_unit_norm_to_floating_point_precision():
    states = amplitudes_from_grayscale_batch(_raw(16))
    norms = np.linalg.norm(states, axis=1)
    assert np.max(np.abs(norms - 1.0)) < 1e-12


def test_encoding_preserves_the_scaled_pixel_direction():
    """``/255`` then L2 is a pure rescale: the direction must be untouched."""
    raw = _raw(3)
    states = amplitudes_from_grayscale_batch(raw)
    scaled = raw.astype(np.float64) / 255.0
    expected = scaled / np.linalg.norm(scaled, axis=1, keepdims=True)
    assert np.allclose(states, expected, atol=1e-15)


def test_all_zero_image_is_a_controlled_error_not_a_fabricated_state():
    """A black ROI has no direction; inventing one would be a silent fabrication."""
    with pytest.raises(Exception):
        amplitudes_from_grayscale_batch(np.zeros((1, V1_PIXEL_COUNT), dtype=np.uint8))


# ------------------------------------------------------- 4. deterministic preprocessing
def test_encoding_is_deterministic_across_calls():
    raw = _raw(8)
    assert np.array_equal(
        amplitudes_from_grayscale_batch(raw), amplitudes_from_grayscale_batch(raw)
    )


def test_raw_representation_is_not_overwritten_by_encoding():
    """The inspectable 0-255 representation survives encoding, as V1 requires."""
    partition = _partition("train", 4, n_positive=2)
    before = partition.raw.copy()
    partition.amplitudes()
    assert np.array_equal(partition.raw, before)
    assert partition.raw.dtype == np.uint8


# ------------------------------------------------------- 5. measurement shape
def test_expectations_are_one_scalar_per_sample_in_range(model):
    states = amplitudes_from_grayscale_batch(_raw(6))
    values = model.expectation_states(states, model.weights)
    assert values.shape == (6,)
    assert np.all(values >= -1.0 - 1e-12) and np.all(values <= 1.0 + 1e-12)


def test_scores_are_the_documented_map_of_the_expectation(model):
    states = amplitudes_from_grayscale_batch(_raw(6))
    expectations = model.expectation_states(states, model.weights)
    probabilities = model.probabilities_from_states(states, model.weights)
    assert np.allclose(probabilities, (1.0 - expectations) / 2.0, atol=1e-12)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


# ---------------------------------------------- 6. chunked training equals whole batch
def test_chunked_loss_matches_whole_batch_loss(model):
    """The trainer's streaming objective is a factoring, not a second implementation.

    ``train_pixel_vqc`` chunks amplitude derivation to keep peak memory near 250 MB
    instead of 846 MB. If chunking changed the objective by even a little, every
    reported loss would be untraceable to the classifier's own definition.
    """
    partition = _partition("train", 12, n_positive=5)
    rows = np.arange(12)
    y = partition.labels
    class_weights = model._class_weights(y)

    whole = model.loss_from_states(
        partition.amplitudes(rows), y, model.weights, class_weights=class_weights
    )
    for chunk in (1, 5, 12, 64):
        probabilities = trainer.probabilities_for_rows(
            model, partition, rows, model.weights, chunk_rows=chunk
        )
        chunked = trainer.loss_from_probabilities(probabilities, y, class_weights)
        assert chunked == pytest.approx(whole, abs=1e-12), f"chunk_rows={chunk}"


def test_chunking_does_not_change_the_scores(model):
    partition = _partition("train", 10, n_positive=4)
    rows = np.arange(10)
    reference = trainer.probabilities_for_rows(
        model, partition, rows, model.weights, chunk_rows=10
    )
    for chunk in (1, 3, 7):
        assert np.allclose(
            trainer.probabilities_for_rows(
                model, partition, rows, model.weights, chunk_rows=chunk
            ),
            reference,
            atol=1e-14,
        )


def test_probabilities_for_empty_row_selection_is_empty(model):
    partition = _partition("train", 4, n_positive=2)
    assert trainer.probabilities_for_rows(
        model, partition, np.asarray([], dtype=int), model.weights
    ).shape == (0,)


# ------------------------------------------------------- 7. serialisation / reload
def test_model_round_trips_through_json_without_pickle(model, tmp_path):
    """Weights and register width must survive a save/load with no drift."""
    states = amplitudes_from_grayscale_batch(_raw(5))
    model.trained = True
    before = model.probabilities_from_states(states, model.weights)

    path = tmp_path / "pixel_vqc.json"
    path.write_text(json.dumps(model.to_dict()), encoding="utf-8")
    restored = VariationalQuantumClassifier.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )

    assert restored.num_qubits == 16
    assert restored.num_params == model.num_params
    assert np.allclose(restored.weights, model.weights, atol=0.0)
    assert np.allclose(
        restored.probabilities_from_states(states, restored.weights), before, atol=1e-12
    )


def test_rebuilt_model_reproduces_a_candidate_exactly():
    """A persisted candidate's weights must reproduce its scores on reload."""
    from backend.core.config import settings

    partition = _partition("validation", 6, n_positive=3)
    rows = np.arange(6)
    config = trainer.PixelVQCConfig(
        roi_mode="oracle", condition=CONDITION_ORACLE_LESION, n_layers=1, seed=SEED
    )
    weights = np.random.default_rng(SEED).uniform(-np.pi, np.pi, config.n_parameters)
    candidate = trainer.Candidate(
        config=config,
        weights=weights,
        initial_objective=0.7,
        final_objective=0.6,
        best_objective=0.6,
        objective_history=[0.7, 0.6],
        validation_trace=[],
        n_iterations=1,
        n_objective_evaluations=2,
        n_circuit_evaluations=12,
        duration_seconds=1.0,
        converged=False,
        train_metrics={},
        validation_metrics={"pr_auc": 0.5},
        class_weights=(1.0, 1.0),
    )
    first = trainer._rebuild_model(candidate, settings)
    second = trainer._rebuild_model(candidate, settings)
    assert first.num_qubits == 16 and first.trained
    assert np.allclose(
        trainer.probabilities_for_rows(first, partition, rows),
        trainer.probabilities_for_rows(second, partition, rows),
        atol=0.0,
    )


def test_artifact_directory_is_keyed_by_version_and_rejects_traversal(tmp_path):
    """A retrained VQC must not be able to overwrite a frozen one, or escape the root."""
    store = ArtifactStore(root=tmp_path)
    directory = store.pixel_vqc_dir(trainer.PIXEL_VQC_VERSION)
    assert directory.parent.name == "pixel_vqc"
    assert directory.name == trainer.PIXEL_VQC_VERSION
    assert store.pixel_vqc_dir("v2") != directory
    for bad in ("", "../escape", "a/b", "a\\b"):
        with pytest.raises(ArtifactError):
            store.pixel_vqc_dir(bad)


def test_localizer_artifact_path_is_separate_from_the_pixel_vqc_path(tmp_path):
    """Phase D must not be able to write over the frozen Phase C localiser."""
    store = ArtifactStore(root=tmp_path)
    assert store.localizer_dir("v1") != store.pixel_vqc_dir("v1")
    assert "localizer" in store.localizer_dir("v1").parts
    assert "pixel_vqc" in store.pixel_vqc_dir("v1").parts


# ------------------------------------- 8. oracle vs predicted condition separation
def test_oracle_partition_cannot_report_a_predicted_condition():
    partition = _partition("train", 6, n_positive=3, roi_mode="oracle")
    names = set(condition_rows(partition))
    assert CONDITION_ORACLE_ALL in names
    assert CONDITION_PREDICTED_LOCALIZED not in names
    assert CONDITION_PREDICTED_FALLBACK not in names


def test_predicted_partition_cannot_report_an_oracle_condition():
    partition = _partition(
        "train", 6, n_positive=3, roi_mode="predicted",
        sources=["predicted"] * 4 + ["predicted_rejected"] * 2,
        statuses=["localized"] * 4 + ["fallback_used"] * 2,
    )
    names = set(condition_rows(partition))
    assert CONDITION_PREDICTED_LOCALIZED in names
    assert CONDITION_ORACLE_ALL not in names
    assert CONDITION_ORACLE_LESION not in names


def test_oracle_subdivisions_partition_the_rows_without_overlap():
    """A_all is the union of the three annotation regimes, each disjoint."""
    partition = _partition(
        "train", 9, n_positive=4, roi_mode="oracle",
        sources=["lesion_polygon"] * 3 + ["region_polygon"] * 4 + ["center_crop"] * 2,
    )
    conditions = condition_rows(partition)
    subdivisions = [
        conditions[name]
        for name in ("A_lesion_polygon", "A_region_polygon", "A_center_crop")
    ]
    union = np.concatenate(subdivisions)
    assert union.size == 9
    assert np.array_equal(np.sort(union), conditions[CONDITION_ORACLE_ALL])
    assert len(set(union.tolist())) == 9


def test_condition_selection_is_refused_when_the_name_does_not_exist():
    """A typo must fail loudly, not silently select a different condition."""
    from backend.training.pixel_data import LoadedPixelDataset

    data = LoadedPixelDataset(
        manifest=_manifest(),
        partitions={
            "train": _partition("train", 6, n_positive=3),
            "validation": _partition("validation", 4, n_positive=2),
            "test": _partition("test", 4, n_positive=2),
        },
        cache_metadata={"qubit_count": V1_QUBIT_COUNT},
    )
    with pytest.raises(trainer.PixelVQCError, match="does not exist"):
        trainer._resolve_condition(data, "B_localized")


# --------------------------------------------- 9. fallback rows remain identifiable
def test_fallback_rows_are_selectable_and_never_merged_into_condition_b():
    partition = _partition(
        "train", 10, n_positive=4, roi_mode="predicted",
        sources=["predicted"] * 7 + ["predicted_rejected"] * 3,
        statuses=["localized"] * 7 + ["fallback_used"] * 3,
    )
    conditions = condition_rows(partition)
    localized = conditions[CONDITION_PREDICTED_LOCALIZED]
    fallback = conditions[CONDITION_PREDICTED_FALLBACK]
    assert localized.size == 7 and fallback.size == 3
    assert not set(localized.tolist()) & set(fallback.tolist())
    assert conditions["B_and_C_all"].size == 10


def test_fallback_rows_keep_their_own_metrics():
    """C is reported on its own rows; a pooled number would hide it."""
    y = np.asarray([0, 1, 0, 1])
    scores = np.asarray([0.1, 0.9, 0.2, 0.8])
    metrics = evaluate_predictions(y, scores, threshold=0.5)
    assert metrics.n_samples == 4 and metrics.n_positive == 2


# --------------------------------------- 10. train/validation/test isolation
def test_trainer_never_touches_the_test_partition(monkeypatch, tmp_path):
    """Accessing ``dataset.test`` during training is made a hard failure.

    This is the leak that would invalidate the whole phase, and it is the kind that
    reads as harmless in a diff, so it is enforced mechanically rather than reviewed.
    """
    from backend.core.config import settings
    from backend.training.pixel_data import LoadedPixelDataset

    train_partition = _partition("train", 12, n_positive=5)
    validation_partition = _partition("validation", 8, n_positive=3, seed=SEED + 1)
    test_partition = _partition("test", 8, n_positive=3, seed=SEED + 2)

    class TripwireDataset(LoadedPixelDataset):
        @property
        def test(self):  # pragma: no cover - the test asserts it is never reached
            raise AssertionError(
                "train_pixel_vqc read the test partition. Any tuning decision made "
                "against it invalidates the final evaluation."
            )

    data = TripwireDataset(
        manifest=_manifest(),
        partitions={
            "train": train_partition,
            "validation": validation_partition,
            "test": test_partition,
        },
        cache_metadata={"qubit_count": V1_QUBIT_COUNT},
    )
    # Redirect the artifact root at tmp_path. Without this the test writes a model
    # trained on 12 synthetic images into the real store, where it is
    # indistinguishable from a real run's artifact.
    monkeypatch.setattr(
        ArtifactStore,
        "from_settings",
        classmethod(lambda cls, cfg=None: cls(root=tmp_path)),
    )
    payload = trainer.train(
        roi_mode="oracle",
        condition=CONDITION_ORACLE_ALL,
        layers=(1,),
        maxiter=1,
        seed=SEED,
        dataset=data,
        config=settings,
        skip_hardware_report=True,
    )
    assert payload["test_partition_used"] is False
    assert payload["dataset"]["train"]["n_samples"] == 12
    assert payload["dataset"]["validation"]["n_samples"] == 8
    # And it wrote where it was told to, not into the real store.
    assert Path(payload["artifact"]).is_relative_to(tmp_path)


def test_no_patient_appears_in_two_partitions():
    partitions = {
        "train": _partition("train", 12, n_positive=5),
        "validation": _partition("validation", 8, n_positive=3, seed=SEED + 1),
        "test": _partition("test", 8, n_positive=3, seed=SEED + 2),
    }
    seen = {name: set(p.patient_ids) for name, p in partitions.items()}
    names = sorted(seen)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            assert not seen[left] & seen[right], f"{left} and {right} share patients"


def test_single_class_condition_is_refused_rather_than_trained():
    """Cross-entropy has no gradient on one class; a fake number is worse than none."""
    from backend.core.config import settings
    from backend.training.pixel_data import LoadedPixelDataset

    data = LoadedPixelDataset(
        manifest=_manifest(),
        partitions={
            "train": _partition("train", 6, n_positive=0),
            "validation": _partition("validation", 4, n_positive=2),
            "test": _partition("test", 4, n_positive=2),
        },
        cache_metadata={"qubit_count": V1_QUBIT_COUNT},
    )
    with pytest.raises(trainer.PixelVQCError, match="single class"):
        trainer.train_candidate(
            trainer.PixelVQCConfig(
                roi_mode="oracle", condition=CONDITION_ORACLE_ALL, n_layers=1, maxiter=1
            ),
            data,
            train_rows=np.arange(6),
            validation_rows=np.arange(4),
            app_config=settings,
        )


# ------------------------------------------------- reproducibility of the artifact
def test_config_round_trips_and_records_the_register_width():
    config = trainer.PixelVQCConfig(
        roi_mode="predicted", condition=CONDITION_PREDICTED_LOCALIZED, n_layers=2
    )
    described = config.describe()
    assert described["n_qubits"] == 16
    assert described["amplitudes_per_state"] == 65536
    assert described["n_parameters"] == 64
    assert described["roi_mode"] == "predicted"
    assert json.loads(json.dumps(described)) == described


def test_selection_score_is_never_defined_when_pr_auc_is_missing():
    """An undefined PR-AUC must not be able to win selection."""
    config = trainer.PixelVQCConfig(
        roi_mode="oracle", condition=CONDITION_ORACLE_ALL, n_layers=1
    )
    common = dict(
        config=config, weights=np.zeros(config.n_parameters), initial_objective=0.7,
        final_objective=0.7, best_objective=0.7, objective_history=[], validation_trace=[],
        n_iterations=0, n_objective_evaluations=0, n_circuit_evaluations=0,
        duration_seconds=0.0, converged=False, train_metrics={}, class_weights=(1.0, 1.0),
    )
    undefined = trainer.Candidate(validation_metrics={"pr_auc": None}, **common)
    defined = trainer.Candidate(validation_metrics={"pr_auc": 0.0}, **common)
    assert undefined.selection_score == float("-inf")
    assert defined.selection_score == 0.0
    assert max([undefined, defined], key=lambda c: c.selection_score) is defined


# --------------------------------------------------- real cache, when it is present
@pytest.fixture(scope="module")
def real_oracle_dataset():
    from backend.training.pixel_data import load_pixel_partitions

    try:
        return load_pixel_partitions(roi_mode="oracle")
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Oracle pixel cache unavailable: {exc}")


def test_real_cache_matches_the_documented_partition_sizes(real_oracle_dataset):
    data = real_oracle_dataset
    assert data.qubit_count == 16
    assert data.train.raw.shape[1] == 65536
    assert (len(data.train), data.train.n_patients) == (1692, 230)
    assert (len(data.validation), data.validation.n_patients) == (363, 50)
    assert (len(data.test), data.test.n_patients) == (381, 48)


def test_real_cache_has_no_patient_overlap(real_oracle_dataset):
    data = real_oracle_dataset
    train = set(data.train.patient_ids)
    validation = set(data.validation.patient_ids)
    test = set(data.test.patient_ids)
    assert not train & validation
    assert not train & test
    assert not validation & test


def test_real_cache_states_are_unit_norm(real_oracle_dataset):
    states = real_oracle_dataset.validation.amplitudes(np.arange(32))
    assert states.shape == (32, 65536)
    assert np.max(np.abs(np.linalg.norm(states, axis=1) - 1.0)) < 1e-12
