"""The V1 pixel cache, and its partitioning along the patient-level split.

Two things are checked, and they are separable:

* the cache round-trips and refuses to be read at the wrong width -- tested on a
  cache built in-process, so these always run;
* the real cache partitions cleanly and agrees with the descriptor path about which
  patients and images are usable -- skipped without the clinical dataset.

The second group is what makes the V1 quantum result comparable to the classical
baseline: both paths must train on the same patients. If they diverged, the
comparison the evaluation rests on would be between two different experiments.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.dataset.smartom import discover_dataset_root
from backend.ml.artifacts import ArtifactStore
from backend.ml.pixel_pipeline import (
    V1_PIXEL_COUNT,
    V1_QUBIT_COUNT,
    V1_ROI_EDGE_PX,
)
from backend.training.prepare_pixels import (
    PIXEL_CACHE_VERSION,
    load_pixel_cache,
    pixel_cache_path,
    save_pixel_cache,
)

PARTITIONS = ("train", "validation", "test")


def _payload(n: int = 3, pixels: int = V1_PIXEL_COUNT, qubits: int = V1_QUBIT_COUNT):
    """A minimal cache payload, built through the real save path."""
    rng = np.random.default_rng(5)
    return {
        "cache_version": PIXEL_CACHE_VERSION,
        "preprocessing_version": "carescan-v1-pixel-1",
        "roi_mode": "oracle",
        "qubit_count": qubits,
        "roi_edge_px": V1_ROI_EDGE_PX,
        "resample_filter": "lanczos",
        "raw": rng.integers(1, 256, size=(n, pixels), dtype=np.uint8),
        "image_ids": [f"class/site/img{i}" for i in range(n)],
        "roi_boxes": np.tile(np.array([[0, 0, 100, 100]], dtype=np.int32), (n, 1)),
        "roi_sources": ["region_polygon"] * n,
        "source_sizes": np.tile(np.array([[640, 480]], dtype=np.int32), (n, 1)),
        # Deliberately omitted: localisation columns. An oracle payload has no
        # localisation to report, and the save/load path must default them rather
        # than requiring every caller to know about the localiser.
        "rejections": {},
        "failures": {},
        "n_candidates": n,
        "duration_seconds": 1.0,
    }


# ------------------------------------------------------------------ cache contract
def test_cache_round_trips_without_loss(tmp_path):
    """uint8 pixels must survive the archive byte for byte.

    Compression is lossless, but asserting it means a future switch to a lossy
    intermediate format cannot pass unnoticed -- that would silently change the
    quantum input.
    """
    payload = _payload()
    path = save_pixel_cache(payload, tmp_path / "pixels.npz")
    loaded = load_pixel_cache(path)

    assert np.array_equal(loaded["raw"], payload["raw"])
    assert loaded["image_ids"] == payload["image_ids"]
    assert loaded["roi_sources"] == payload["roi_sources"]
    assert np.array_equal(loaded["roi_boxes"], payload["roi_boxes"])
    assert loaded["qubit_count"] == V1_QUBIT_COUNT
    assert loaded["cache_version"] == PIXEL_CACHE_VERSION
    assert loaded["raw"].dtype == np.uint8


def test_cache_at_the_wrong_width_is_refused(tmp_path):
    """A cache built at another resolution must not load.

    This is the concrete path by which a run could claim 16 qubits while feeding
    the circuit something else, so it fails at load rather than at encode.
    """
    path = save_pixel_cache(_payload(pixels=16384), tmp_path / "small.npz")
    with pytest.raises(ValueError) as excinfo:
        load_pixel_cache(path)
    message = str(excinfo.value)
    assert str(V1_PIXEL_COUNT) in message and "16384" in message
    assert "rebuild" in message.lower()


def test_cache_declaring_the_wrong_qubit_count_is_refused(tmp_path):
    """Width and declared qubit count are checked independently.

    A cache could carry 65,536 correct pixels and still be labelled 12 qubits; the
    label is what provenance reports, so it has to be verified too.
    """
    path = save_pixel_cache(_payload(qubits=12), tmp_path / "mislabelled.npz")
    with pytest.raises(ValueError, match="fixed at 16"):
        load_pixel_cache(path)


def test_missing_cache_names_the_command_that_builds_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_pixels"):
        load_pixel_cache(tmp_path / "absent.npz")


def test_an_oracle_cache_has_no_localisation_columns_to_report(tmp_path):
    """Absent localisation defaults to ``"oracle"``, not to a fabricated status.

    A cache written before the localiser existed is still a correct oracle cache, so
    it loads rather than demanding a rebuild -- but it must not come back claiming
    its crops were localised.
    """
    path = save_pixel_cache(_payload(n=3), tmp_path / "oracle.npz")
    loaded = load_pixel_cache(path)
    assert loaded["localization_statuses"] == ["oracle"] * 3
    assert np.all(np.isnan(loaded["localization_confidences"]))


def test_a_predicted_cache_must_name_the_localiser_that_produced_it(tmp_path):
    """Condition B is attributable or it is not reportable.

    A predicted cache without a localiser version could not be traced to a model, so
    a result computed from it could not be attributed to one either.
    """
    payload = _payload(n=2)
    payload["roi_mode"] = "predicted"
    payload["localization_statuses"] = ["localized", "localized"]
    payload["localization_confidences"] = np.array([0.8, 0.7])
    path = save_pixel_cache(payload, tmp_path / "anonymous.npz")
    with pytest.raises(ValueError, match="localiser"):
        load_pixel_cache(path)


def test_misaligned_localisation_columns_are_refused_at_save(tmp_path):
    """A status list shorter than the pixel matrix would mislabel every row after it."""
    payload = _payload(n=3)
    payload["localization_statuses"] = ["localized", "localized"]
    with pytest.raises(ValueError, match="misaligned"):
        save_pixel_cache(payload, tmp_path / "short.npz")


def test_localisation_columns_round_trip(tmp_path):
    payload = _payload(n=3)
    payload["roi_mode"] = "predicted"
    payload["localizer_version"] = "carescan-localizer-1"
    payload["roi_sources"] = ["predicted", "predicted_rejected", "predicted"]
    payload["localization_statuses"] = ["localized", "fallback_used", "localized"]
    payload["localization_confidences"] = np.array([0.91, 0.12, 0.55])
    loaded = load_pixel_cache(save_pixel_cache(payload, tmp_path / "predicted.npz"))
    assert loaded["localization_statuses"] == payload["localization_statuses"]
    assert np.allclose(
        loaded["localization_confidences"], payload["localization_confidences"]
    )
    assert loaded["localizer_version"] == "carescan-localizer-1"


# ============================= THE REAL CACHE ================================
requires_pixel_cache = pytest.mark.skipif(
    discover_dataset_root() is None
    or not pixel_cache_path(ArtifactStore.from_settings()).exists(),
    reason="V1 pixel cache not built (run `python -m backend.training.prepare_pixels`).",
)


@pytest.fixture(scope="module")
def pixel_dataset():
    from backend.training.pixel_data import load_pixel_partitions

    return load_pixel_partitions()


@requires_pixel_cache
def test_the_loaded_dataset_is_sixteen_qubits(pixel_dataset):
    """MANDATORY: what training will consume is 16 qubits wide, verified on the data."""
    assert pixel_dataset.qubit_count == 16
    for partition in pixel_dataset.partitions.values():
        assert partition.raw.shape[1] == V1_PIXEL_COUNT
        assert partition.raw.dtype == np.uint8


@requires_pixel_cache
def test_no_patient_crosses_a_pixel_partition_boundary(pixel_dataset):
    """MANDATORY, on the pixel path: the leakage guarantee is not descriptor-specific.

    The pixel path is a second consumer of the same manifest, so it needs its own
    assertion -- a correct manifest can still be joined incorrectly.
    """
    seen = {}
    for name, partition in pixel_dataset.partitions.items():
        for patient_id in set(partition.patient_ids):
            assert patient_id not in seen, (
                f"LEAKAGE: patient {patient_id} in both {seen[patient_id]} and {name}"
            )
            seen[patient_id] = name
    assert len(seen) > 100


@requires_pixel_cache
def test_pixel_partitions_share_no_images(pixel_dataset):
    seen = {}
    for name, partition in pixel_dataset.partitions.items():
        for image_id in partition.image_ids:
            assert image_id not in seen, (
                f"Image {image_id} appears in both {seen[image_id]} and {name}"
            )
            seen[image_id] = name


@requires_pixel_cache
def test_rows_align_across_every_parallel_list(pixel_dataset):
    """A one-row offset here would train on shifted labels and be invisible in metrics."""
    for name, partition in pixel_dataset.partitions.items():
        n = len(partition.image_ids)
        assert partition.raw.shape[0] == n, name
        assert len(partition.patient_ids) == n
        assert len(partition.labels) == n
        assert len(partition.diagnostic_classes) == n
        assert len(partition.roi_sources) == n
        assert set(np.unique(partition.labels)) <= {0, 1}
        assert partition.n_positive > 0, f"{name} has no positives"


@requires_pixel_cache
def test_the_pixel_path_matches_the_descriptor_path_patient_for_patient(pixel_dataset):
    """Both paths must cover the same patients, or the comparison is not a comparison.

    Image counts are also compared: the two caches apply the same quality control,
    so a divergence means one path is quietly training on more data than the other.
    """
    from backend.training.data import load_partitions

    descriptor = load_partitions()
    for name in PARTITIONS:
        pixel_patients = set(pixel_dataset.partitions[name].patient_ids)
        descriptor_patients = set(descriptor.partitions[name].patient_ids)
        assert pixel_patients == descriptor_patients, (
            f"{name}: the pixel and descriptor paths disagree about which patients "
            f"it contains (symmetric difference "
            f"{len(pixel_patients ^ descriptor_patients)})"
        )
        assert len(pixel_dataset.partitions[name]) == len(
            descriptor.partitions[name].image_ids
        )


@requires_pixel_cache
def test_amplitudes_derived_from_the_real_cache_are_legal_states(pixel_dataset):
    """The states the circuit will receive must be unit-norm and non-negative."""
    partition = pixel_dataset.test
    states = partition.amplitudes(range(min(24, len(partition))))
    assert states.shape[1] == V1_PIXEL_COUNT
    norms = np.linalg.norm(states, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)
    assert np.all(states >= 0.0)
    assert np.all(np.isfinite(states))


@requires_pixel_cache
def test_amplitudes_match_recomputing_from_the_source_image(pixel_dataset):
    """The cache must be a shortcut, not a different pipeline.

    Recomputing from the original JPEG has to reproduce the cached pixels exactly,
    or a model trained on the cache is trained on something inference will never
    see.
    """
    from backend.dataset.smartom import cached_index
    from backend.ml.pixel_pipeline import preprocess_v1_pixels_for_record

    records = {record.image_id: record for record in cached_index().records}
    partition = pixel_dataset.test
    for position in (0, len(partition) // 2, len(partition) - 1):
        image_id = partition.image_ids[position]
        recomputed = preprocess_v1_pixels_for_record(records[image_id])
        assert np.array_equal(
            recomputed.raw_grayscale.ravel(order="C"), partition.raw[position]
        ), f"cache and live preprocessing disagree for {image_id}"
        assert np.array_equal(
            recomputed.amplitudes, partition.amplitudes([position])[0]
        )


@requires_pixel_cache
def test_the_oracle_cache_reports_no_localisation(pixel_dataset):
    """Condition A must not be able to masquerade as condition B.

    The oracle cache is annotation-derived; if it ever came back with localisation
    statuses, an oracle number could be reported as a predicted one.
    """
    assert pixel_dataset.cache_metadata.get("roi_mode") == "oracle"
    assert not pixel_dataset.cache_metadata.get("localizer_version")
    for partition in pixel_dataset.partitions.values():
        assert set(partition.localization_statuses) == {"oracle"}
        assert partition.localized_rows.size == 0
        assert partition.fallback_rows.size == 0


# ========================= THE PREDICTED-ROI CACHE ===========================
requires_predicted_cache = pytest.mark.skipif(
    discover_dataset_root() is None
    or not pixel_cache_path(ArtifactStore.from_settings(), "predicted").exists(),
    reason="Predicted-ROI cache not built "
    "(run `python -m backend.training.prepare_pixels --roi predicted`).",
)


@pytest.fixture(scope="module")
def predicted_dataset():
    from backend.training.pixel_data import load_pixel_partitions

    return load_pixel_partitions(roi_mode="predicted")


@requires_predicted_cache
def test_the_predicted_cache_is_also_sixteen_qubits(predicted_dataset):
    assert predicted_dataset.qubit_count == 16
    for partition in predicted_dataset.partitions.values():
        assert partition.raw.shape[1] == V1_PIXEL_COUNT
        assert partition.raw.dtype == np.uint8


@requires_predicted_cache
def test_the_predicted_cache_names_its_localiser(predicted_dataset):
    """Requirement 6, at the point the pixels are consumed rather than produced."""
    assert predicted_dataset.cache_metadata["roi_mode"] == "predicted"
    assert predicted_dataset.cache_metadata["localizer_version"]


@requires_predicted_cache
def test_no_patient_crosses_a_predicted_partition_boundary(predicted_dataset):
    """MANDATORY: the leakage guarantee holds on the predicted path too.

    A third consumer of the same manifest needs its own assertion; the ROI changed,
    so the join is a different one even though the manifest is not.
    """
    seen = {}
    for name, partition in predicted_dataset.partitions.items():
        for patient_id in set(partition.patient_ids):
            assert patient_id not in seen, (
                f"LEAKAGE: patient {patient_id} in both {seen[patient_id]} and {name}"
            )
            seen[patient_id] = name
    assert len(seen) > 100


@requires_predicted_cache
def test_the_predicted_and_oracle_caches_cover_the_same_images(
    predicted_dataset, pixel_dataset
):
    """Conditions A and B must differ only in the ROI, not in the population.

    If one cache held more images than the other, the A-versus-B comparison would be
    between two different experiments and the ROI would not be the only variable.
    """
    for name in PARTITIONS:
        assert set(predicted_dataset.partitions[name].image_ids) == set(
            pixel_dataset.partitions[name].image_ids
        ), name


@requires_predicted_cache
def test_every_predicted_row_is_localised_or_an_explicit_fallback(predicted_dataset):
    """MANDATORY (requirement 9): no cached row has an unaccounted-for origin.

    ``localized`` and ``fallback_used`` must partition the rows, and the ROI source
    must agree with the status on every one -- a row labelled ``predicted`` whose
    status was ``fallback_used`` would be exactly the silent conversion of a failure
    into a successful crop that the requirement forbids.
    """
    for name, partition in predicted_dataset.partitions.items():
        statuses = partition.localization_statuses
        assert set(statuses) <= {"localized", "fallback_used"}, name
        assert (
            partition.localized_rows.size + partition.fallback_rows.size == len(partition)
        ), name
        for status, source in zip(statuses, partition.roi_sources):
            expected = "predicted" if status == "localized" else "predicted_rejected"
            assert source == expected, (name, status, source)


@requires_predicted_cache
def test_the_fallback_subset_is_selectable_as_condition_c(predicted_dataset):
    """Condition C has to be addressable from a loaded partition, not reconstructed.

    Also asserts the fallback rows are not simply everything: a localiser that failed
    on every image would satisfy the partitioning check above while being useless.
    """
    total_fallback = sum(
        p.fallback_rows.size for p in predicted_dataset.partitions.values()
    )
    total_images = sum(len(p) for p in predicted_dataset.partitions.values())
    assert 0 < total_fallback < total_images * 0.5, (
        f"{total_fallback}/{total_images} rows are localisation fallbacks; condition B "
        f"would mostly be measuring the centre crop."
    )
    partition = predicted_dataset.test
    rows = partition.fallback_rows
    if rows.size:
        states = partition.amplitudes(rows)
        assert states.shape == (rows.size, V1_PIXEL_COUNT)


@requires_predicted_cache
def test_predicted_confidences_are_recorded_and_in_range(predicted_dataset):
    for name, partition in predicted_dataset.partitions.items():
        confidences = partition.localization_confidences
        assert confidences.shape[0] == len(partition), name
        assert np.all(np.isfinite(confidences)), name
        assert np.all((confidences >= 0.0) & (confidences <= 1.0)), name


@requires_predicted_cache
def test_predicted_pixels_differ_from_oracle_pixels(predicted_dataset, pixel_dataset):
    """The two conditions must actually be different inputs.

    If they matched, condition B would be condition A under another name and the
    comparison would be vacuous. Some rows legitimately coincide -- an unannotated
    image whose predicted box was rejected falls back to a centre crop close to the
    oracle centre crop -- so this asserts most rows differ, not all.
    """
    oracle = pixel_dataset.test
    predicted = predicted_dataset.test
    by_id = {image_id: i for i, image_id in enumerate(oracle.image_ids)}
    compared = differing = 0
    for position, image_id in enumerate(predicted.image_ids):
        if image_id not in by_id:
            continue
        compared += 1
        if not np.array_equal(predicted.raw[position], oracle.raw[by_id[image_id]]):
            differing += 1
    assert compared > 100
    assert differing / compared > 0.9, (
        f"Only {differing}/{compared} test images differ between the oracle and "
        f"predicted ROIs; the two conditions are not distinct inputs."
    )


@requires_predicted_cache
def test_amplitudes_from_the_predicted_cache_are_legal_states(predicted_dataset):
    partition = predicted_dataset.test
    states = partition.amplitudes(range(min(24, len(partition))))
    norms = np.linalg.norm(states, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)
    assert np.all(states >= 0.0) and np.all(np.isfinite(states))


@requires_pixel_cache
def test_roi_sources_are_recorded_and_mostly_annotated(pixel_dataset):
    """Provenance: which rule chose each region, and how often the fallback fired.

    The centre-crop share matters for interpreting the oracle-ROI condition -- those
    images have no annotation, so for them "oracle" and "predicted" are not
    meaningfully different and the upper bound is not really an upper bound.
    """
    counts = {}
    for partition in pixel_dataset.partitions.values():
        for source in partition.roi_sources:
            counts[source] = counts.get(source, 0) + 1
    assert set(counts) <= {"lesion_polygon", "region_polygon", "center_crop"}
    total = sum(counts.values())
    annotated = counts.get("lesion_polygon", 0) + counts.get("region_polygon", 0)
    assert annotated / total > 0.5, (
        f"Only {annotated}/{total} images have an annotated ROI; the oracle-ROI "
        f"condition would mostly be measuring the centre-crop fallback."
    )
