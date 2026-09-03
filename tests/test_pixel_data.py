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
