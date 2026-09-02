"""Dataset loading, patient grouping, and patient-level split integrity.

The centrepiece is :func:`test_no_patient_occurs_in_multiple_partitions`, which
PART 3 and PART 26 both call mandatory. Patient-level leakage is the failure mode that
makes an oral-cancer screening model look excellent and be worthless: the same lesion
photographed twice, once in train and once in test, turns memorisation into apparent
generalisation. So it is asserted here three ways -- against the persisted manifest,
against the records the manifest was built from, and against the loaded feature
partitions the model actually trains on.

Tests that need the clinical dataset skip when it is absent (it is not committed --
PART 25). Tests of the splitting *logic* build synthetic records and always run, so
the mandatory guarantee is covered on every checkout.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pytest

from backend.dataset.smartom import ImageRecord, discover_dataset_root
from backend.dataset.split import (
    DEFAULT_POSITIVE_CLASSES,
    SplitError,
    binary_label,
    build_patient_level_split,
    compute_partition_stats,
    fingerprint_records,
    patient_stratum,
)

PARTITIONS = ("train", "validation", "test")

# The real class vocabulary, in ascending severity. Using the actual strings matters:
# ``patient_stratum`` and ``binary_label`` both key off them, so a synthetic cohort
# built from invented names ("oscc") would exercise only the unrecognised-class path
# and would not test severity ordering at all.
NEGATIVE_CLASS = "normal"
POSITIVE_CLASS = "oral_cancer"
MILD_POSITIVE_CLASS = "opmd"


# --------------------------------------------------------------- synthetic records
def _record(image_id: str, patient_id: str, diagnostic_class: str) -> ImageRecord:
    """A minimal record carrying only what splitting reads.

    Constructed through the real dataclass rather than a stub, so a field rename
    breaks this test instead of silently bypassing it.
    """
    filename = f"{image_id}.jpg"
    return ImageRecord(
        image_id=image_id,
        path=Path(f"/synthetic/{diagnostic_class}/{filename}"),
        filename=filename,
        diagnostic_class=diagnostic_class,
        site="dorsal_tongue",
        patient_id=patient_id,
        patient_id_source="synthetic",
    )


def _synthetic_cohort(
    n_patients: int = 60,
    images_per_patient: int = 3,
    positive_every: int = 4,
) -> List[ImageRecord]:
    """A cohort with multiple images per patient and an imbalanced positive rate.

    Multiple images per patient is the whole point: with one image each, leakage is
    impossible and the test would prove nothing.
    """
    records = []
    for patient_index in range(n_patients):
        patient = f"P{patient_index:03d}"
        diagnostic_class = (
            POSITIVE_CLASS if patient_index % positive_every == 0 else NEGATIVE_CLASS
        )
        for image_index in range(images_per_patient):
            records.append(
                _record(f"{patient}-img{image_index}", patient, diagnostic_class)
            )
    return records


@pytest.fixture(scope="module")
def synthetic_records() -> List[ImageRecord]:
    return _synthetic_cohort()


@pytest.fixture(scope="module")
def synthetic_manifest(synthetic_records):
    return build_patient_level_split(synthetic_records, seed=1234)


# ============================ THE MANDATORY TEST =============================
def test_no_patient_occurs_in_multiple_partitions(synthetic_manifest, synthetic_records):
    """MANDATORY (PART 3, PART 26): a patient belongs to exactly one partition.

    Checked directly on the patient lists rather than through a helper, so this test
    cannot pass because the helper it calls is broken.
    """
    seen_in = {}
    for partition in PARTITIONS:
        for patient_id in synthetic_manifest.patients_in(partition):
            assert patient_id not in seen_in, (
                f"LEAKAGE: patient {patient_id} is in both "
                f"{seen_in[patient_id]!r} and {partition!r}"
            )
            seen_in[patient_id] = partition

    # Every patient in the cohort is placed exactly once -- no silent drops.
    cohort = {record.patient_id for record in synthetic_records}
    assert set(seen_in) == cohort

    # And the same must hold at image level: an image's partition must agree with
    # its patient's partition, which is what leakage-by-accident looks like.
    for record in synthetic_records:
        expected = seen_in[record.patient_id]
        actual = [p for p in PARTITIONS if record.image_id in set(synthetic_manifest.images_in(p))]
        assert actual == [expected], (
            f"LEAKAGE: image {record.image_id} of patient {record.patient_id} "
            f"(partition {expected}) appears in {actual}"
        )


def test_manifest_self_check_agrees(synthetic_manifest, synthetic_records):
    """The manifest's own guard must accept a clean split..."""
    synthetic_manifest.assert_no_patient_leakage(synthetic_records)


def test_manifest_self_check_catches_injected_leakage(synthetic_records):
    """...and must reject a poisoned one.

    Without this, ``assert_no_patient_leakage`` could be a no-op and every other
    leakage assertion in the suite would pass vacuously.

    The leak is injected into ``image_to_partition``, not ``patient_to_partition``,
    because that is what the guard reads: it re-derives patient placement from the
    image assignments precisely so that a tidy-looking ``patient_to_partition`` cannot
    vouch for a corrupted split. Poisoning the summary dict instead would prove
    nothing -- the guard is designed to ignore it.

    A private manifest is built here rather than mutating the module-scoped fixture,
    so a failure part-way through cannot leave a poisoned manifest behind for the
    tests that run after it.
    """
    manifest = build_patient_level_split(synthetic_records, seed=1234)
    train_patient = manifest.patients_in("train")[0]
    moved = [r.image_id for r in synthetic_records if r.patient_id == train_patient]
    assert len(moved) > 1, (
        "This test needs a patient with several images; with one image each, a "
        "patient cannot straddle a boundary and there is nothing to detect."
    )

    manifest.assert_no_patient_leakage(synthetic_records)  # clean to begin with

    # One image of a training patient reassigned to the test partition: the patient
    # now spans two partitions, which is the exact condition the guard must catch.
    manifest.image_to_partition[moved[0]] = "test"
    with pytest.raises(SplitError) as excinfo:
        manifest.assert_no_patient_leakage(synthetic_records)
    # The message must name the offending patient, or debugging a real leak means
    # re-deriving it by hand.
    assert train_patient in str(excinfo.value)


def test_split_is_deterministic_for_a_fixed_seed(synthetic_records):
    """Same records, same seed, same split -- byte for byte."""
    first = build_patient_level_split(synthetic_records, seed=99)
    second = build_patient_level_split(synthetic_records, seed=99)
    for partition in PARTITIONS:
        assert first.patients_in(partition) == second.patients_in(partition)
        assert first.images_in(partition) == second.images_in(partition)


def test_split_changes_with_the_seed(synthetic_records):
    """A different seed must actually reshuffle, or 'deterministic' means 'constant'."""
    first = build_patient_level_split(synthetic_records, seed=1)
    second = build_patient_level_split(synthetic_records, seed=2)
    assert first.patients_in("test") != second.patients_in("test")


def test_split_does_not_depend_on_input_order(synthetic_records):
    """Reordering the records must not move a patient.

    Directory iteration order varies across filesystems; if it changed the split, two
    machines would disagree about what "the test set" is.
    """
    forward = build_patient_level_split(synthetic_records, seed=7)
    reversed_records = list(reversed(synthetic_records))
    backward = build_patient_level_split(reversed_records, seed=7)
    assert forward.patient_to_partition == backward.patient_to_partition


def test_every_partition_receives_both_classes(synthetic_manifest, synthetic_records):
    """Stratification must survive patient-level grouping.

    A test set with no positives cannot produce a sensitivity, and a validation set
    with no positives cannot calibrate.
    """
    label_of = {
        record.patient_id: binary_label(record.diagnostic_class, DEFAULT_POSITIVE_CLASSES)
        for record in synthetic_records
    }
    for partition in PARTITIONS:
        labels = Counter(label_of[p] for p in synthetic_manifest.patients_in(partition))
        assert labels[0] > 0, f"{partition} has no negative patients"
        assert labels[1] > 0, f"{partition} has no positive patients"


def test_partition_stats_are_computed_not_assumed(synthetic_manifest, synthetic_records):
    """Reported counts must come from the records, and must add up."""
    stats = compute_partition_stats(
        synthetic_records, synthetic_manifest, DEFAULT_POSITIVE_CLASSES
    )
    total_images = sum(stat.n_images for stat in stats.values())
    total_patients = sum(stat.n_patients for stat in stats.values())
    assert total_images == len(synthetic_records)
    assert total_patients == len({r.patient_id for r in synthetic_records})
    for name, stat in stats.items():
        assert stat.n_positive_images + stat.n_negative_images == stat.n_images, name
        assert 0.0 <= stat.positive_rate <= 1.0
        assert stat.positive_rate == pytest.approx(
            stat.n_positive_images / stat.n_images
        )


def test_fingerprint_is_order_independent_and_content_sensitive(synthetic_records):
    """The manifest fingerprint must detect a changed cohort, not a reshuffled one."""
    baseline = fingerprint_records(synthetic_records)
    assert fingerprint_records(list(reversed(synthetic_records))) == baseline
    mutated = list(synthetic_records) + [_record("extra", "P999", NEGATIVE_CLASS)]
    assert fingerprint_records(mutated) != baseline


def test_patient_stratum_is_driven_by_the_severest_class():
    """A patient with any malignant image must stratify as malignant.

    Otherwise a patient with one oral_cancer image and four normal ones could be
    stratified as normal, and the positive rate per partition would drift.
    """
    # Order-independent: the stratum is a property of the set, not of iteration order.
    assert patient_stratum([NEGATIVE_CLASS, POSITIVE_CLASS]) == patient_stratum(
        [POSITIVE_CLASS, NEGATIVE_CLASS]
    )
    # And severity actually escalates, across every adjacent pair in the ordering.
    assert patient_stratum([NEGATIVE_CLASS]) != patient_stratum(
        [NEGATIVE_CLASS, POSITIVE_CLASS]
    )
    assert patient_stratum([NEGATIVE_CLASS, MILD_POSITIVE_CLASS]) != patient_stratum(
        [NEGATIVE_CLASS, MILD_POSITIVE_CLASS, POSITIVE_CLASS]
    )
    # The severest class wins outright, whatever else the patient has.
    assert patient_stratum(
        [NEGATIVE_CLASS, MILD_POSITIVE_CLASS, POSITIVE_CLASS]
    ) == patient_stratum([POSITIVE_CLASS])


def test_single_image_patients_still_split_cleanly():
    """The degenerate cohort must not crash the splitter."""
    records = [
        _record(
            f"P{i:03d}-img0",
            f"P{i:03d}",
            POSITIVE_CLASS if i % 3 == 0 else NEGATIVE_CLASS,
        )
        for i in range(30)
    ]
    manifest = build_patient_level_split(records, seed=5)
    manifest.assert_no_patient_leakage(records)
    assert sum(len(manifest.patients_in(p)) for p in PARTITIONS) == 30


def test_empty_cohort_is_an_error_not_an_empty_split():
    """Silently returning three empty partitions would let training 'succeed' on nothing."""
    with pytest.raises(SplitError):
        build_patient_level_split([], seed=1)


def test_partition_of_patient_round_trips(synthetic_manifest):
    for partition in PARTITIONS:
        for patient_id in synthetic_manifest.patients_in(partition):
            assert synthetic_manifest.partition_of_patient(patient_id) == partition
    assert synthetic_manifest.partition_of_patient("no-such-patient") is None


def test_manifest_survives_a_json_round_trip(synthetic_manifest, synthetic_records, tmp_path):
    """The persisted manifest is the reproducibility artifact; it must reload exactly."""
    from backend.dataset.split import SplitManifest

    path = synthetic_manifest.save(tmp_path / "split.json")
    reloaded = SplitManifest.load(path)
    assert reloaded.patient_to_partition == synthetic_manifest.patient_to_partition
    assert reloaded.seed == synthetic_manifest.seed
    for partition in PARTITIONS:
        assert reloaded.images_in(partition) == synthetic_manifest.images_in(partition)
    # A reloaded manifest is still leakage-free -- serialisation cannot introduce a leak.
    reloaded.assert_no_patient_leakage(synthetic_records)


# ============================= THE REAL DATASET ==============================
requires_dataset = pytest.mark.skipif(
    discover_dataset_root() is None,
    reason="Clinical dataset not present (it is not committed -- see PART 25).",
)


@pytest.fixture(scope="module")
def real_index():
    from backend.dataset.smartom import cached_index

    return cached_index()


@pytest.fixture(scope="module")
def real_dataset():
    """The loaded, transformed partitions the model is actually trained on."""
    from backend.training.data import load_partitions

    return load_partitions()


@requires_dataset
def test_real_dataset_has_no_patient_leakage(real_index, real_dataset):
    """MANDATORY, on the real cohort: no patient crosses a partition boundary.

    ``load_partitions`` re-verifies this on every call, but asserting it here means a
    regression shows up as a named failing test rather than as an exception inside a
    fixture somebody might mark xfail.
    """
    real_dataset.manifest.assert_no_patient_leakage(real_index.records)

    seen_in = {}
    for name, partition in real_dataset.partitions.items():
        for patient_id in set(partition.patient_ids):
            assert patient_id not in seen_in, (
                f"LEAKAGE: patient {patient_id} in both {seen_in[patient_id]} and {name}"
            )
            seen_in[patient_id] = name


@requires_dataset
def test_real_partitions_share_no_images(real_dataset):
    """Image identity is disjoint too -- the duplicate-image failure mode."""
    seen = {}
    for name, partition in real_dataset.partitions.items():
        for image_id in partition.image_ids:
            assert image_id not in seen, (
                f"Image {image_id} appears in both {seen[image_id]} and {name}"
            )
            seen[image_id] = name


@requires_dataset
def test_real_partitions_are_non_empty_and_labelled(real_dataset):
    for name, partition in real_dataset.partitions.items():
        assert len(partition.image_ids) > 0, f"{name} is empty"
        assert len(partition.labels) == len(partition.image_ids)
        assert set(np.unique(partition.labels)) <= {0, 1}
        assert partition.labels.sum() > 0, (
            f"{name} contains no positive cases; sensitivity and calibration are "
            f"undefined for it"
        )


@requires_dataset
def test_real_features_align_with_labels(real_dataset):
    """Row alignment: feature matrix, labels, image IDs and patient IDs must agree.

    A one-row offset here silently trains on shifted labels and is invisible in the
    metrics until the model is deployed.
    """
    for name, partition in real_dataset.partitions.items():
        n = len(partition.image_ids)
        assert partition.image_features.shape[0] == n
        assert len(partition.patient_ids) == n
        assert len(partition.labels) == n
        assert len(partition.diagnostic_classes) == n
        if partition.features is not None:
            assert partition.features.shape[0] == n
        assert np.all(np.isfinite(partition.image_features)), (
            f"{name} contains non-finite image features"
        )


@requires_dataset
def test_real_split_matches_the_persisted_manifest(real_dataset):
    """The loaded partitions must be the ones the manifest describes.

    If loading drifted from the manifest, the committed manifest would document a
    split nobody is using -- and the evaluation report would be unreproducible.
    """
    manifest = real_dataset.manifest
    for name, partition in real_dataset.partitions.items():
        for patient_id in set(partition.patient_ids):
            assert manifest.partition_of_patient(patient_id) == name
