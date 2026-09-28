"""The PTB-XL raw waveform cache: fold policy, channel semantics, honest accounting.

``prepare_ecg_features`` caches a 97-number summary per record; this module caches the
19,601 x 12 x 1000 array those numbers were computed *from*, and the deep rung trains on
it directly. That makes its failure modes different in kind from the feature cache's, and
quieter:

* **the channel axis.** Every downstream convolution assumes channel *c* is the same lead
  in every row. A record whose header lists the leads in a different order, or lists a
  lead set that is not the 12 standard ones, would poison that assumption in a way no
  metric could reveal -- the model would simply learn a worse function and nobody would
  know why. So the permutation is asserted, and a non-conforming record is a *failure*
  rather than something to pad or reorder approximately.
* **alignment with the labels.** The deep arm reads waveforms from this cache and labels
  from the feature cache. The two are joined on ``ecg_id``, so the ordering contract
  between them is load-bearing, and a lookup miss must raise rather than shift every
  subsequent pair by one.
* **fold 10.** The frozen test partition must be refused by default here exactly as it is
  in the cohort builder (``docs/PHASE_E4_DATA_EXPANSION.md`` section 7.4).
* **nothing fitted.** The cache stores decoded physical millivolts. If any statistic
  pooled across records ever entered it, every future run would leak validation
  information and no downstream guard could detect it -- so the property is tested
  directly, by building the same record in two different corpora and requiring the rows
  to be bit-identical.

Records are written as genuine WFDB pairs and genuine PTB-XL-shaped CSVs, so these tests
run the same code path the real corpus does. The metadata writers come from
:mod:`tests.test_ptbxl_dataset` and the 12-lead signal writer from
:mod:`tests.test_prepare_ecg_features`, to keep one definition of each fixture shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pytest

from backend.dataset.ptbxl import TEST_FOLDS, PtbxlError, build_index
from backend.ml.features_ecg import STANDARD_LEADS
from backend.training.prepare_ecg_features import DEV_FOLDS, extract_cohort
from backend.training.prepare_ecg_signals import (
    EXPECTED_SAMPLES,
    SIGNAL_CACHE_VERSION,
    SignalCache,
    SignalCacheError,
    _lead_permutation,
    build_signal_cache,
    load_signal_cache,
    main,
    signal_array_path,
    signal_index_path,
)
from tests.test_prepare_ecg_features import (
    _FS,
    _GAIN,
    _beat_trace,
    _raw_12_lead,
    _write_wfdb,
)
from tests.test_ptbxl_dataset import _write_database, _write_scp_statements

# ecg_id 2 carries the standard leads in a non-standard header order: same waveform as
# ecg_id 1, so a correct permutation makes the two cached rows identical.
_SHUFFLED_ECG_ID = 2
_SHUFFLED_LEADS = ("V1", "V2", "V3", "V4", "V5", "V6", "I", "II", "III", "AVR", "AVL", "AVF")

# ecg_id 4 is flat. It is not a defect as far as *this* cache is concerned -- judging a
# record unusable is the feature extractor's job, not the raw store's.
_FLATLINE_ECG_ID = 4

# Three deliberate defects, one per rejection path in build_signal_cache.
_ELEVEN_LEAD_ECG_ID = 7
_SHORT_ECG_ID = 8
_UNREADABLE_ECG_ID = 9

_ROWS: Sequence[Dict[str, object]] = (
    {"ecg_id": 1, "patient_id": "100.0", "age": "56.0", "sex": "1",
     "scp_codes": "{'NORM': 100.0}", "strat_fold": "1"},
    {"ecg_id": _SHUFFLED_ECG_ID, "patient_id": "100.0", "age": "57.0", "sex": "1",
     "scp_codes": "{'NORM': 100.0}", "strat_fold": "1"},
    {"ecg_id": 3, "patient_id": "200.0", "age": "62.0", "sex": "0",
     "scp_codes": "{'IMI': 100.0}", "strat_fold": "5"},
    {"ecg_id": _FLATLINE_ECG_ID, "patient_id": "300.0", "age": "", "sex": "",
     "scp_codes": "{'SR': 0.0}", "strat_fold": "9"},
    # In the metadata precisely so its exclusion from the cache is observable.
    {"ecg_id": 5, "patient_id": "400.0", "age": "70.0", "sex": "1",
     "scp_codes": "{'NORM': 100.0}", "strat_fold": "10"},
    {"ecg_id": 6, "patient_id": "500.0", "age": "44.0", "sex": "0",
     "scp_codes": "{'IMI': 100.0}", "strat_fold": "2"},
    {"ecg_id": _ELEVEN_LEAD_ECG_ID, "patient_id": "600.0", "age": "50.0", "sex": "1",
     "scp_codes": "{'NORM': 100.0}", "strat_fold": "3"},
    {"ecg_id": _SHORT_ECG_ID, "patient_id": "700.0", "age": "51.0", "sex": "0",
     "scp_codes": "{'NORM': 100.0}", "strat_fold": "4"},
    {"ecg_id": _UNREADABLE_ECG_ID, "patient_id": "800.0", "age": "52.0", "sex": "1",
     "scp_codes": "{'NORM': 100.0}", "strat_fold": "6"},
)

_CLEAN_FOLDS = {1, 2, 5}
"""Folds holding only well-formed records: ecg_ids 1, 2, 3, 6."""


@pytest.fixture
def cohort_root(tmp_path: Path) -> Path:
    """A tiny PTB-XL with one well-formed record per good fold and three defects."""
    _write_scp_statements(tmp_path)
    _write_database(tmp_path, _ROWS)

    beats = _raw_12_lead(_beat_trace())
    records100 = tmp_path / "records100"

    # The shuffled record's columns are the standard ones, re-dealt into header order.
    order = [STANDARD_LEADS.index(name) for name in _SHUFFLED_LEADS]
    shuffled = beats[:, order]

    for row in _ROWS:
        ecg_id = int(row["ecg_id"])  # type: ignore[arg-type]
        name = f"{ecg_id}_lr"
        if ecg_id == _SHUFFLED_ECG_ID:
            _write_wfdb(records100, name, shuffled, lead_names=_SHUFFLED_LEADS)
        elif ecg_id == _FLATLINE_ECG_ID:
            _write_wfdb(records100, name, np.zeros_like(beats))
        elif ecg_id == _ELEVEN_LEAD_ECG_ID:
            _write_wfdb(records100, name, beats[:, :11], lead_names=STANDARD_LEADS[:11])
        elif ecg_id == _SHORT_ECG_ID:
            _write_wfdb(records100, name, beats[:500])
        elif ecg_id == _UNREADABLE_ECG_ID:
            path = _write_wfdb(records100, name, beats)
            path.with_suffix(".dat").unlink()  # header survives, samples do not
        else:
            _write_wfdb(records100, name, beats)
    return tmp_path


def _build(root: Path, out: Path, **kwargs) -> Dict[str, object]:
    return build_signal_cache(root=root, out_dir=out, **kwargs)


# ------------------------------------------------------------------ lead permutation


def test_standard_order_permutes_to_itself():
    assert _lead_permutation(STANDARD_LEADS) == list(range(len(STANDARD_LEADS)))


def test_a_reordered_header_yields_the_permutation_that_fixes_it():
    permutation = _lead_permutation(_SHUFFLED_LEADS)
    assert permutation is not None
    assert [_SHUFFLED_LEADS[index] for index in permutation] == list(STANDARD_LEADS)


def test_lead_names_are_matched_case_and_whitespace_insensitively():
    noisy = tuple(f"  {name.lower()} " for name in STANDARD_LEADS)
    assert _lead_permutation(noisy) == list(range(12))


@pytest.mark.parametrize(
    "lead_names",
    [
        STANDARD_LEADS[:11],                      # a lead is missing
        STANDARD_LEADS + ("VX",),                 # an extra channel
        STANDARD_LEADS[:11] + ("II",),            # V6 replaced by a duplicate of II
        ("I",) * 12,                              # all duplicates
    ],
    ids=["missing", "extra", "duplicate", "all-duplicates"],
)
def test_a_non_standard_lead_set_has_no_permutation(lead_names):
    """No padding, no best effort: the channel axis has to mean one thing."""
    assert _lead_permutation(lead_names) is None


# ------------------------------------------------------------------- the frozen fold


def test_fold_10_is_refused_and_the_message_names_the_policy(cohort_root, tmp_path):
    with pytest.raises(PtbxlError, match=r"frozen test partition"):
        _build(cohort_root, tmp_path / "out", folds={1, 10})


def test_the_refusal_cites_the_document_that_froze_it(cohort_root, tmp_path):
    with pytest.raises(PtbxlError) as excinfo:
        _build(cohort_root, tmp_path / "out", folds=set(TEST_FOLDS))
    assert "allow_test_fold=True" in str(excinfo.value)
    assert "7.4" in str(excinfo.value)


def test_the_default_cohort_is_folds_one_to_nine_and_excludes_the_test_fold(
    cohort_root, tmp_path
):
    report = _build(cohort_root, tmp_path / "out")
    assert report["folds"] == sorted(DEV_FOLDS)
    assert report["test_fold_included"] is False

    cache = load_signal_cache(tmp_path / "out")
    assert 5 not in set(cache.ecg_ids.tolist())
    assert not set(cache.strat_folds.tolist()) & set(TEST_FOLDS)


def test_the_override_exists_and_is_recorded_when_used(cohort_root, tmp_path):
    """The door can be opened, and opening it leaves a mark in the metadata."""
    report = _build(
        cohort_root, tmp_path / "out", folds={1, 10}, allow_test_fold=True
    )
    assert report["test_fold_included"] is True
    assert 5 in set(load_signal_cache(tmp_path / "out").ecg_ids.tolist())


# ------------------------------------------------------------------------- the array


def test_the_cache_has_the_declared_shape_and_dtype(cohort_root, tmp_path):
    _build(cohort_root, tmp_path / "out", folds=_CLEAN_FOLDS)
    cache = load_signal_cache(tmp_path / "out")

    assert cache.signals.shape == (4, 12, EXPECTED_SAMPLES)
    assert cache.signals.dtype == np.float32
    assert (cache.n_records, cache.n_leads, cache.n_samples) == (4, 12, EXPECTED_SAMPLES)


def test_records_are_ordered_by_ecg_id(cohort_root, tmp_path):
    _build(cohort_root, tmp_path / "out", folds=_CLEAN_FOLDS)
    ecg_ids = load_signal_cache(tmp_path / "out").ecg_ids
    assert ecg_ids.tolist() == [1, 2, 3, 6]


def test_the_signal_and_feature_caches_agree_on_ordering(cohort_root, tmp_path):
    """The deep arm reads waveforms here and labels there, joined on ecg_id. If the two
    orderings ever diverged, every metric would be computed on mismatched pairs."""
    _build(cohort_root, tmp_path / "out", folds=_CLEAN_FOLDS)
    cache = load_signal_cache(tmp_path / "out")
    cohort = extract_cohort(root=cohort_root, folds=_CLEAN_FOLDS, progress_every=0)

    assert cache.ecg_ids.tolist() == cohort["ecg_ids"].tolist()
    assert cache.patient_ids.tolist() == cohort["patient_ids"].tolist()
    assert cache.strat_folds.tolist() == cohort["strat_folds"].tolist()


def test_a_reordered_header_is_rotated_into_standard_order(cohort_root, tmp_path):
    """ecg_id 2 holds the same waveform as ecg_id 1 with the leads dealt out in a
    different header order. Correct handling makes the two rows identical; letting the
    header order through would make them a permutation of one another, which nothing
    downstream could notice."""
    _build(cohort_root, tmp_path / "out", folds={1})
    cache = load_signal_cache(tmp_path / "out")

    first = cache.signals[cache.rows_for([1])[0]]
    shuffled = cache.signals[cache.rows_for([_SHUFFLED_ECG_ID])[0]]
    assert np.array_equal(first, shuffled)
    # And it is genuinely reordered, not accidentally symmetric.
    assert not np.array_equal(first[0], first[6])


def test_samples_are_physical_millivolts_not_raw_counts(cohort_root, tmp_path):
    """The header declares 1000 ADU/mV; the cache must have divided by it."""
    _build(cohort_root, tmp_path / "out", folds={2})  # ecg_id 6
    cache = load_signal_cache(tmp_path / "out")

    raw = _raw_12_lead(_beat_trace())
    expected = raw.astype(np.float64) / _GAIN
    np.testing.assert_allclose(
        cache.signals[0], expected.T.astype(np.float32), rtol=1e-6, atol=1e-6
    )
    assert float(np.abs(cache.signals[0]).max()) < 10.0  # mV, not thousands of counts


def test_a_row_does_not_depend_on_which_other_records_were_cached(cohort_root, tmp_path):
    """The load-bearing no-leak property, tested as a property rather than asserted in a
    docstring: if any statistic pooled across the corpus entered a row, enlarging the
    corpus would change that row."""
    _build(cohort_root, tmp_path / "alone", folds={2})
    _build(cohort_root, tmp_path / "with_others", folds=_CLEAN_FOLDS)

    alone = load_signal_cache(tmp_path / "alone")
    together = load_signal_cache(tmp_path / "with_others")
    row = together.signals[together.rows_for([6])[0]]
    assert np.array_equal(alone.signals[0], row)  # bit-identical, not merely close


def test_the_metadata_states_that_nothing_is_fitted(cohort_root, tmp_path):
    report = _build(cohort_root, tmp_path / "out", folds=_CLEAN_FOLDS)
    assert "none" in str(report["fitted_statistics"]).lower()
    assert report["units"] == "mV"
    assert report["lead_names"] == list(STANDARD_LEADS)
    assert report["sampling_frequency"] == _FS


def test_a_flat_record_is_cached_rather_than_judged(cohort_root, tmp_path):
    """A flatline is finite and correctly shaped. Deciding it is unusable belongs to the
    feature extractor and the model's quality gate, not to the raw store -- a cache that
    silently drops physiologically odd records changes the denominator of every count
    computed from it."""
    _build(cohort_root, tmp_path / "out", folds={9})
    cache = load_signal_cache(tmp_path / "out")

    assert cache.ecg_ids.tolist() == [_FLATLINE_ECG_ID]
    assert not cache.failures
    assert float(np.abs(cache.signals[0]).max()) == 0.0


# --------------------------------------------------------------------- failure policy


def test_every_candidate_is_either_cached_or_named(cohort_root, tmp_path):
    report = _build(cohort_root, tmp_path / "out")
    assert report["n_candidates"] == 8  # folds 1-9, excluding the fold-10 record
    assert report["n_rows"] + report["n_failures"] == report["n_candidates"]
    assert report["n_rows"] == 5


@pytest.mark.parametrize(
    "ecg_id, pattern",
    [
        (_ELEVEN_LEAD_ECG_ID, "not the 12 standard leads"),
        (_SHORT_ECG_ID, "samples, expected 1000"),
        (_UNREADABLE_ECG_ID, "Error"),
    ],
    ids=["lead-set", "length", "unreadable"],
)
def test_a_defective_record_is_recorded_with_its_reason(
    cohort_root, tmp_path, ecg_id, pattern
):
    """Named in ``failures``, not dropped from the tally: a record that vanishes silently
    is indistinguishable from a record that was never in the corpus."""
    report = _build(cohort_root, tmp_path / "out")
    failures = report["failures"]
    assert str(ecg_id) in failures
    assert pattern in failures[str(ecg_id)]
    assert ecg_id not in set(load_signal_cache(tmp_path / "out").ecg_ids.tolist())


def test_the_array_is_trimmed_to_the_rows_actually_written(cohort_root, tmp_path):
    """The buffer is allocated for every candidate and filled in place. A failure to trim
    would leave zero rows on the end carrying real ecg_ids' positions."""
    report = _build(cohort_root, tmp_path / "out")
    assert report["n_failures"] == 3

    stored = np.load(signal_array_path(tmp_path / "out"), mmap_mode="r")
    assert stored.shape[0] == report["n_rows"] == 5
    assert np.isfinite(stored).all()


def test_an_empty_fold_selection_raises_rather_than_writing_nothing(cohort_root, tmp_path):
    with pytest.raises(SignalCacheError, match="No records in folds"):
        _build(cohort_root, tmp_path / "out", folds={8})


def test_a_corpus_of_only_defects_refuses_to_write_a_cache(cohort_root, tmp_path):
    """Fold 3 holds one 11-lead record and nothing else. An empty cache written without
    complaint would read downstream as a successful build."""
    with pytest.raises(SignalCacheError, match="No record decoded successfully"):
        _build(cohort_root, tmp_path / "out", folds={3})


def test_limit_applies_after_sorting_so_it_is_reproducible(cohort_root, tmp_path):
    _build(cohort_root, tmp_path / "out", folds=_CLEAN_FOLDS, limit=2)
    assert load_signal_cache(tmp_path / "out").ecg_ids.tolist() == [1, 2]


# ----------------------------------------------------------------- SignalCache guards


def _identifiers(n: int):
    return (
        np.arange(n, dtype=np.int64),
        np.arange(n, dtype=np.int64),
        np.ones(n, dtype=np.int16),
    )


def test_a_cache_whose_rows_and_ids_disagree_is_refused():
    ecg_ids, patient_ids, folds = _identifiers(3)
    with pytest.raises(SignalCacheError, match="waveform rows"):
        SignalCache(np.zeros((2, 12, 10), np.float32), ecg_ids, patient_ids, folds, {}, {})


def test_identifier_arrays_must_be_the_same_length():
    with pytest.raises(SignalCacheError, match="same length"):
        SignalCache(
            np.zeros((3, 12, 10), np.float32),
            np.arange(3, dtype=np.int64),
            np.arange(2, dtype=np.int64),
            np.ones(3, dtype=np.int16),
            {},
            {},
        )


def test_a_duplicate_ecg_id_is_refused():
    """The row lookup is a dict; a duplicate would make one of the two rows unreachable
    and the other one wrong for half its callers."""
    with pytest.raises(SignalCacheError, match="Duplicate ecg_id"):
        SignalCache(
            np.zeros((3, 12, 10), np.float32),
            np.asarray([1, 2, 2], dtype=np.int64),
            np.arange(3, dtype=np.int64),
            np.ones(3, dtype=np.int16),
            {},
            {},
        )


def test_rows_for_preserves_the_requested_order(cohort_root, tmp_path):
    _build(cohort_root, tmp_path / "out", folds=_CLEAN_FOLDS)
    cache = load_signal_cache(tmp_path / "out")
    assert cache.rows_for([6, 1, 3]).tolist() == [3, 0, 2]


def test_rows_for_raises_on_a_miss_rather_than_dropping_it(cohort_root, tmp_path):
    """Silently returning three rows for four ecg_ids would misalign waveforms against
    labels drawn from the feature cache, and every metric after that point would be
    computed on mismatched pairs."""
    _build(cohort_root, tmp_path / "out", folds=_CLEAN_FOLDS)
    cache = load_signal_cache(tmp_path / "out")
    with pytest.raises(SignalCacheError, match="not in the signal cache"):
        cache.rows_for([1, 2, 999])


# ------------------------------------------------------------------------ the loader


def test_a_missing_cache_names_the_command_that_builds_it(tmp_path):
    with pytest.raises(SignalCacheError, match="prepare_ecg_signals"):
        load_signal_cache(tmp_path)


def test_a_version_mismatch_is_refused_rather_than_loaded(cohort_root, tmp_path):
    """The filename carries the version, so a mismatch here means the *contents* were
    written by a different extractor -- exactly the case where stale numbers would be
    reported as current ones."""
    out = tmp_path / "out"
    _build(cohort_root, out, folds=_CLEAN_FOLDS)

    index_path = signal_index_path(out)
    with np.load(index_path, allow_pickle=True) as archive:
        payload = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(payload["metadata"].item()))
    metadata["signal_cache_version"] = "ptbxl-signal-0"
    payload["metadata"] = np.asarray(json.dumps(metadata), dtype=object)
    np.savez_compressed(index_path, **payload)

    with pytest.raises(SignalCacheError, match="ptbxl-signal-0"):
        load_signal_cache(out)


def test_the_cache_is_memory_mapped_by_default(cohort_root, tmp_path):
    """~900 MiB at the real corpus size. It should not have to be resident to train."""
    _build(cohort_root, tmp_path / "out", folds=_CLEAN_FOLDS)
    assert isinstance(load_signal_cache(tmp_path / "out").signals, np.memmap)
    assert not isinstance(
        load_signal_cache(tmp_path / "out", mmap=False).signals, np.memmap
    )


def test_the_loader_carries_the_failure_list_with_the_data(cohort_root, tmp_path):
    """A consumer should be able to see what is missing without re-reading the build log."""
    _build(cohort_root, tmp_path / "out")
    cache = load_signal_cache(tmp_path / "out")
    assert set(cache.failures) == {
        str(_ELEVEN_LEAD_ECG_ID),
        str(_SHORT_ECG_ID),
        str(_UNREADABLE_ECG_ID),
    }
    assert cache.metadata["n_candidates"] == 8


# --------------------------------------------------------------------------- the CLI


def test_main_builds_a_cache_for_an_explicit_fold_list(cohort_root, tmp_path):
    out = tmp_path / "out"
    assert main(["--root", str(cohort_root), "--out", str(out), "--folds", "1,2"]) == 0
    assert load_signal_cache(out).ecg_ids.tolist() == [1, 2, 6]


def test_main_expands_a_fold_range(cohort_root, tmp_path):
    out = tmp_path / "out"
    assert main(["--root", str(cohort_root), "--out", str(out), "--folds", "1-5"]) == 0
    assert load_signal_cache(out).metadata["folds"] == [1, 2, 3, 4, 5]


def test_main_refuses_the_test_fold_without_the_override(cohort_root, tmp_path):
    with pytest.raises(PtbxlError, match="frozen test partition"):
        main(["--root", str(cohort_root), "--out", str(tmp_path / "out"), "--folds", "1-10"])


def test_the_default_cache_version_is_the_one_on_record():
    """Bumping this silently would let two incompatible caches share a filename."""
    assert SIGNAL_CACHE_VERSION == "ptbxl-signal-1"
    assert signal_array_path(Path("x")).name == "signals_ptbxl-signal-1.npy"
    assert signal_index_path(Path("x")).name == "signals_ptbxl-signal-1_index.npz"
