"""The PTB-XL cohort driver: fold policy, label honesty, cache integrity (E4).

The feature extractor itself is fold-honest by construction -- every value is a
function of exactly one record -- so the risks live in the driver stacked on top
of it, and those are what this module pins down:

* **which folds may be touched.** Fold 10 is the frozen test partition. The
  default cohort is folds 1-9 and asking for 10 must raise rather than quietly
  extract it (``docs/PHASE_E4_DATA_EXPANSION.md`` sections 7.2/7.4).
* **how an absent label is represented.** A record with no diagnostic superclass
  is ``-1``/unknown, never a healthy negative.
* **whether a failed record can vanish.** An unreadable signal must be named in
  ``failures``, not silently dropped from the denominator.
* **whether a stale cache can be reused.** A matrix built by a different
  extractor version, or with a different feature layout, must be refused on load
  rather than reported as current numbers.

Records are written as genuine WFDB pairs and genuine PTB-XL-shaped CSVs rather
than mocked, so these tests run the same code path the 19,601-record corpus
does. The CSV writers are imported from :mod:`tests.test_ptbxl_dataset` to keep a
single definition of PTB-XL's metadata shape; the signal writer is local because
these records need real 12-lead names, which the reader tests' writer does not
emit and the extractor's axis features read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pytest

from backend.dataset.ptbxl import (
    SUPERCLASS_ORDER,
    TEST_FOLDS,
    TRAIN_FOLDS,
    VALIDATION_FOLDS,
    PtbxlError,
    build_index,
)
from backend.ml.features_ecg import (
    FEATURE_SET_VERSION,
    STANDARD_LEADS,
    feature_names,
)
from backend.training.prepare_ecg_features import (
    DEV_FOLDS,
    ECG_CACHE_VERSION,
    LABEL_UNKNOWN,
    cache_path,
    extract_cohort,
    load_cohort,
    parse_folds,
    save_cohort,
)
from tests.test_ptbxl_dataset import _write_database, _write_scp_statements

_FS = 100.0
_GAIN = 1000.0
_DURATION_S = 10.0

# Distinct per-lead gains, including negatives (AVR is inverted in a real ECG),
# so the axis features see a plausible spatial pattern rather than 12 copies.
_LEAD_SCALE = (1.0, 1.2, 0.4, -1.1, 0.6, 0.8, -0.5, -0.3, 0.7, 1.0, 1.1, 0.9)

# ecg_id 4 is deliberately flat: no beats, hence NaN features and a quality flag.
_FLATLINE_ECG_ID = 4


def _beat_trace(bpm: float = 60.0, amplitude_mv: float = 1.5) -> np.ndarray:
    """A crude but detectable QRS+T train: sharp R spikes with broad T waves."""
    n_samples = int(_DURATION_S * _FS)
    t = np.arange(n_samples) / _FS
    trace = np.zeros(n_samples, dtype=np.float64)
    period = 60.0 / bpm
    for beat in range(int(_DURATION_S / period) + 1):
        r_time = 0.4 + beat * period
        trace += amplitude_mv * np.exp(-0.5 * ((t - r_time) / 0.015) ** 2)
        trace += 0.25 * amplitude_mv * np.exp(-0.5 * ((t - r_time - 0.28) / 0.045) ** 2)
    return trace


def _raw_12_lead(trace: np.ndarray) -> np.ndarray:
    columns = [np.round(trace * scale * _GAIN) for scale in _LEAD_SCALE]
    return np.clip(np.column_stack(columns), -32768, 32767).astype(np.int16)


def _write_wfdb(
    directory: Path,
    name: str,
    raw: np.ndarray,
    lead_names: Sequence[str] = STANDARD_LEADS,
    fs: float = _FS,
    gain: float = _GAIN,
) -> Path:
    """Write a minimal single-file WFDB record with real lead names."""
    directory.mkdir(parents=True, exist_ok=True)
    n_samples, n_signals = raw.shape
    lines = [f"{name} {n_signals} {fs:g} {n_samples}"]
    for index in range(n_signals):
        checksum = int(np.sum(raw[:, index].astype(np.int64)) & 0xFFFF)
        lines.append(
            f"{name}.dat 16 {gain:g}(0)/mV 16 0 "
            f"{int(raw[0, index])} {checksum} 0 {lead_names[index]}"
        )
    (directory / f"{name}.hea").write_text("\n".join(lines) + "\n", encoding="utf-8")
    raw.astype("<i2").tofile(directory / f"{name}.dat")
    return directory / name


_ROWS: Sequence[Dict[str, object]] = (
    # One patient with two studies, both in train: the fold file is patient-level.
    {"ecg_id": 1, "patient_id": "100.0", "age": "56.0", "sex": "1",
     "scp_codes": "{'NORM': 100.0, 'SR': 0.0}", "strat_fold": "1"},
    {"ecg_id": 2, "patient_id": "100.0", "age": "57.0", "sex": "1",
     "scp_codes": "{'NORM': 100.0}", "strat_fold": "1"},
    # Multi-label abnormal (MI + STTC).
    {"ecg_id": 3, "patient_id": "200.0", "age": "62.0", "sex": "0",
     "scp_codes": "{'IMI': 100.0, 'NDT': 50.0}", "strat_fold": "5"},
    # Only a non-diagnostic rhythm code -> label unknown, not negative.
    {"ecg_id": _FLATLINE_ECG_ID, "patient_id": "300.0", "age": "", "sex": "",
     "scp_codes": "{'SR': 0.0}", "strat_fold": "9"},
    # The frozen test fold, present in metadata precisely so its exclusion is visible.
    {"ecg_id": 5, "patient_id": "400.0", "age": "70.0", "sex": "1",
     "scp_codes": "{'NORM': 100.0}", "strat_fold": "10"},
    {"ecg_id": 6, "patient_id": "500.0", "age": "44.0", "sex": "0",
     "scp_codes": "{'IMI': 100.0}", "strat_fold": "2"},
)


@pytest.fixture
def cohort_root(tmp_path: Path) -> Path:
    """A tiny PTB-XL: real metadata CSVs plus a real WFDB pair per record."""
    _write_scp_statements(tmp_path)
    _write_database(tmp_path, _ROWS)
    beats = _raw_12_lead(_beat_trace())
    flat = np.zeros_like(beats)
    for row in _ROWS:
        ecg_id = row["ecg_id"]
        raw = flat if ecg_id == _FLATLINE_ECG_ID else beats
        _write_wfdb(tmp_path / "records100", f"{ecg_id}_lr", raw)
    return tmp_path


# ------------------------------------------------------------------- fold parsing


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("train", set(TRAIN_FOLDS)),
        ("validation", set(VALIDATION_FOLDS)),
        ("val", set(VALIDATION_FOLDS)),
        ("test", set(TEST_FOLDS)),
        ("dev", set(DEV_FOLDS)),
        ("all", set(range(1, 11))),
        ("3", {3}),
        ("1-3", {1, 2, 3}),
        ("1-8,9", set(range(1, 10))),
        (" TRAIN , 9 ", set(TRAIN_FOLDS) | {9}),
        ("2,2,2", {2}),
    ],
)
def test_parse_folds_accepts_aliases_ranges_and_lists(spec: str, expected: set):
    assert parse_folds(spec) == expected


def test_dev_is_exactly_train_plus_validation_and_excludes_the_test_fold():
    assert DEV_FOLDS == set(TRAIN_FOLDS) | set(VALIDATION_FOLDS)
    assert not DEV_FOLDS & set(TEST_FOLDS)
    assert parse_folds("dev") == set(DEV_FOLDS)


@pytest.mark.parametrize(
    "spec",
    [
        "8-1",      # inverted range
        "0",        # PTB-XL folds start at 1
        "11",       # ...and stop at 10
        "0-11",
        "nine",
        "a-b",
        "",
        ",",
    ],
)
def test_parse_folds_rejects_typos_rather_than_selecting_a_different_cohort(spec: str):
    """A silently-misparsed spec would change the cohort a result was computed on."""
    with pytest.raises(ValueError):
        parse_folds(spec)


# -------------------------------------------------------------- frozen test fold


def test_requesting_the_test_fold_raises_without_an_explicit_override(cohort_root: Path):
    index = build_index(cohort_root)

    with pytest.raises(PtbxlError, match="frozen test partition"):
        extract_cohort(index, folds={10})

    with pytest.raises(PtbxlError, match="frozen test partition"):
        extract_cohort(index, folds={1, 10})  # smuggled in beside a dev fold


def test_default_cohort_silently_contains_no_test_fold_record(cohort_root: Path):
    payload = extract_cohort(build_index(cohort_root))

    assert 5 not in set(payload["ecg_ids"].tolist()), "fold-10 record leaked into dev"
    assert set(payload["strat_folds"].tolist()) <= set(DEV_FOLDS)
    assert payload["folds"] == sorted(DEV_FOLDS)


def test_the_test_fold_is_extractable_only_with_allow_test_fold(cohort_root: Path):
    payload = extract_cohort(build_index(cohort_root), folds={10}, allow_test_fold=True)

    assert payload["ecg_ids"].tolist() == [5]


# ----------------------------------------------------------------- extraction run


def test_extraction_emits_the_canonical_feature_layout(cohort_root: Path):
    payload = extract_cohort(build_index(cohort_root))
    names = feature_names()

    assert payload["feature_names"] == list(names)
    assert payload["features"].shape == (5, len(names))
    assert payload["features"].dtype == np.float64
    assert payload["feature_set_version"] == FEATURE_SET_VERSION
    assert payload["cache_version"] == ECG_CACHE_VERSION


def test_identifiers_stay_aligned_with_the_metadata(cohort_root: Path):
    payload = extract_cohort(build_index(cohort_root))

    # Sorted by ecg_id; fold 10 absent.
    assert payload["ecg_ids"].tolist() == [1, 2, 3, 4, 6]
    assert payload["patient_ids"].tolist() == [100, 100, 200, 300, 500]
    assert payload["strat_folds"].tolist() == [1, 1, 5, 9, 2]
    assert payload["n_candidates"] == 5
    assert payload["failures"] == {}


def test_labels_follow_the_pre_registered_rule(cohort_root: Path):
    payload = extract_cohort(build_index(cohort_root))
    labels = dict(zip(payload["ecg_ids"].tolist(), payload["labels"].tolist()))

    assert labels[1] == 0 and labels[2] == 0            # NORM only
    assert labels[3] == 1 and labels[6] == 1            # MI / MI+STTC
    assert labels[_FLATLINE_ECG_ID] == LABEL_UNKNOWN    # no diagnostic superclass


def test_an_unknown_label_is_never_counted_as_a_negative(cohort_root: Path):
    payload = extract_cohort(build_index(cohort_root))
    labels = payload["labels"]

    assert LABEL_UNKNOWN == -1
    assert int(np.count_nonzero(labels == 0)) == 2, "unknown must not inflate 'normal'"
    assert int(np.count_nonzero(labels == 1)) == 2
    assert int(np.count_nonzero(labels == LABEL_UNKNOWN)) == 1
    # label_known is the machine-readable form of the same statement.
    known = payload["label_known"].astype(bool)
    assert known.tolist() == [True, True, True, False, True]
    assert set(labels[known].tolist()) == {0, 1}


def test_superclass_matrix_is_one_hot_in_the_fixed_column_order(cohort_root: Path):
    payload = extract_cohort(build_index(cohort_root))
    rows = dict(zip(payload["ecg_ids"].tolist(), payload["superclasses"].tolist()))

    assert payload["superclass_order"] == list(SUPERCLASS_ORDER)
    order = list(SUPERCLASS_ORDER)
    assert rows[1][order.index("NORM")] == 1
    assert rows[3][order.index("MI")] == 1
    assert rows[3][order.index("STTC")] == 1
    assert sum(rows[3]) == 2
    assert sum(rows[_FLATLINE_ECG_ID]) == 0  # unlabeled: no superclass asserted


def test_unmeasurable_features_are_nan_rather_than_zero(cohort_root: Path):
    """A flatline has no beats. Zero-filling here would fabricate a measurement."""
    payload = extract_cohort(build_index(cohort_root))
    position = payload["ecg_ids"].tolist().index(_FLATLINE_ECG_ID)
    row = payload["features"][position]

    assert np.isnan(row).any(), "a beatless record cannot have every feature measured"
    assert payload["n_beats"][position] == 0
    assert payload["template_beats"][position] == 0
    assert payload["quality_flags"][position], "a flat record must carry a quality flag"


def test_a_measurable_record_yields_beats_and_finite_features(cohort_root: Path):
    payload = extract_cohort(build_index(cohort_root))
    position = payload["ecg_ids"].tolist().index(1)

    assert payload["n_beats"][position] >= 3
    assert payload["template_beats"][position] >= 3
    assert np.isfinite(payload["features"][position]).any()


def test_missing_age_and_sex_survive_as_sentinels(cohort_root: Path):
    payload = extract_cohort(build_index(cohort_root))
    position = payload["ecg_ids"].tolist().index(_FLATLINE_ECG_ID)

    assert np.isnan(payload["ages"][position])
    assert payload["sexes"][position] == -1


def test_limit_takes_the_first_records_by_id(cohort_root: Path):
    payload = extract_cohort(build_index(cohort_root), limit=2)

    assert payload["ecg_ids"].tolist() == [1, 2]
    assert payload["n_candidates"] == 2


# -------------------------------------------------------------- failure accounting


def test_an_unreadable_record_is_named_not_dropped(cohort_root: Path):
    (cohort_root / "records100" / "6_lr.dat").unlink()

    payload = extract_cohort(build_index(cohort_root))

    assert 6 not in payload["ecg_ids"].tolist()
    assert "6" in payload["failures"], "a failure must be reported, not vanish"
    assert payload["features"].shape[0] == 4
    # The denominator still knows about it: rows + failures == candidates.
    assert payload["features"].shape[0] + len(payload["failures"]) == payload["n_candidates"]


def test_a_corrupted_signal_fails_that_record_only(cohort_root: Path):
    """A truncated .dat must not zero-fill into features, nor abort the run."""
    target = cohort_root / "records100" / "3_lr.dat"
    target.write_bytes(target.read_bytes()[:64])

    payload = extract_cohort(build_index(cohort_root))

    assert "3" in payload["failures"]
    assert sorted(payload["ecg_ids"].tolist()) == [1, 2, 4, 6]


def test_extraction_of_an_empty_fold_selection_is_an_empty_matrix_not_a_crash(
    cohort_root: Path,
):
    payload = extract_cohort(build_index(cohort_root), folds={7})

    assert payload["features"].shape == (0, len(feature_names()))
    assert payload["n_candidates"] == 0
    assert payload["labels"].size == 0


# --------------------------------------------------------------------- cache I/O


def test_cache_path_is_keyed_by_extractor_version(tmp_path: Path):
    default = cache_path(tmp_path)
    other = cache_path(tmp_path, feature_set="ecg-v2")

    assert default.parent == tmp_path
    assert FEATURE_SET_VERSION in default.name
    assert default != other, "a v2 cache must not overwrite the v1 matrix"


def test_save_load_round_trip_preserves_every_column(cohort_root: Path, tmp_path: Path):
    payload = extract_cohort(build_index(cohort_root))
    out = save_cohort(payload, tmp_path / "cache" / "features.npz")

    cohort = load_cohort(out)

    assert cohort.n_records == 5
    assert cohort.n_patients == 4
    assert cohort.dimension == len(feature_names())
    assert cohort.feature_names == feature_names()
    # assert_array_equal treats NaN as equal, which is the point: NaNs must survive.
    np.testing.assert_array_equal(cohort.features, payload["features"])
    np.testing.assert_array_equal(cohort.labels, payload["labels"])
    np.testing.assert_array_equal(cohort.patient_ids, payload["patient_ids"])
    np.testing.assert_array_equal(cohort.superclasses, payload["superclasses"])
    np.testing.assert_array_equal(cohort.ages, payload["ages"])
    assert cohort.quality_flags == tuple(payload["quality_flags"])
    assert cohort.superclass_order == SUPERCLASS_ORDER
    assert cohort.metadata["folds"] == sorted(DEV_FOLDS)


def test_cohort_masks_select_folds_and_labeled_rows(cohort_root: Path, tmp_path: Path):
    payload = extract_cohort(build_index(cohort_root))
    cohort = load_cohort(save_cohort(payload, tmp_path / "features.npz"))

    assert cohort.fold_mask({1}).tolist() == [True, True, False, False, False]
    assert cohort.fold_mask({1, 2}).sum() == 3
    assert cohort.fold_mask({10}).sum() == 0
    assert cohort.labeled_mask().sum() == 4
    assert cohort.labels[cohort.labeled_mask()].min() >= 0


def test_save_rejects_a_misaligned_column(cohort_root: Path, tmp_path: Path):
    payload = extract_cohort(build_index(cohort_root))
    payload["labels"] = payload["labels"][:-1]  # one label short of the matrix

    with pytest.raises(ValueError, match="misaligned|rows"):
        save_cohort(payload, tmp_path / "bad.npz")


def test_save_rejects_misaligned_quality_flags(cohort_root: Path, tmp_path: Path):
    payload = extract_cohort(build_index(cohort_root))
    payload["quality_flags"] = payload["quality_flags"][:-1]

    with pytest.raises(ValueError, match="quality_flags"):
        save_cohort(payload, tmp_path / "bad.npz")


def test_load_rejects_a_cache_built_by_another_extractor_version(
    cohort_root: Path, tmp_path: Path
):
    payload = extract_cohort(build_index(cohort_root))
    payload["feature_set_version"] = "ecg-v0"
    out = save_cohort(payload, tmp_path / "stale.npz")

    with pytest.raises(ValueError, match="Rebuild the cache"):
        load_cohort(out)


def test_load_rejects_a_cache_with_a_different_feature_layout(
    cohort_root: Path, tmp_path: Path
):
    payload = extract_cohort(build_index(cohort_root))
    payload["feature_names"] = payload["feature_names"][:-1]
    out = save_cohort(payload, tmp_path / "relabelled.npz")

    with pytest.raises(ValueError, match="feature layout"):
        load_cohort(out)


def test_load_reports_a_missing_cache_with_the_command_that_builds_one(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="prepare_ecg_features"):
        load_cohort(tmp_path / "absent.npz")


def test_metadata_travels_inside_the_archive(cohort_root: Path, tmp_path: Path):
    """No sidecar JSON: a cache cannot be separated from its provenance."""
    payload = extract_cohort(build_index(cohort_root))
    out = save_cohort(payload, tmp_path / "features.npz")

    with np.load(out, allow_pickle=True) as archive:
        metadata = json.loads(str(archive["metadata"]))

    assert metadata["cache_version"] == ECG_CACHE_VERSION
    assert metadata["feature_set_version"] == FEATURE_SET_VERSION
    assert metadata["n_candidates"] == 5
    assert list(tmp_path.glob("*.json")) == []
