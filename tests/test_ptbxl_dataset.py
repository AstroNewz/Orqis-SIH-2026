"""PTB-XL metadata parsing, patient-level fold integrity, and audit.

The centrepiece is :func:`test_no_patient_spans_strat_fold_partitions`, the PTB-XL
analogue of the SMART-OM leakage guard: PTB-XL ships a per-patient ``strat_fold``,
and the whole point of adopting it (rather than re-splitting) is that folds 1-8 /
9 / 10 are already patient-disjoint. If that ever stopped being true, the frozen
test fold would share patients with train and every E4 metric would be inflated,
so it is asserted directly here.

Tests of the parsing/audit *logic* build synthetic PTB-XL-shaped CSVs and always
run. One test exercises the real downloaded metadata and skips when it is absent
(the CSVs are gitignored, not committed).

A second group covers :func:`verify_signal_files`, the integrity half of the bulk
signal download. Its contract is the one the rest of E4 leans on: a downloaded
``.hea``/``.dat`` pair is only trusted once its SHA-256 matches PhysioNet's own
manifest, a file with no published digest counts as *unknown* rather than as a
pass, and only genuine mismatches may be deleted for refetching.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pytest

from backend.dataset.ptbxl import (
    ABNORMAL_SUPERCLASSES,
    DATABASE_CSV,
    SCP_STATEMENTS_CSV,
    SUPERCLASSES,
    SUPERCLASS_ORDER,
    CHECKSUMS_FILE,
    PTBXL_ROOT_ENV,
    EcgRecord,
    PtbxlError,
    assert_no_patient_leakage,
    audit,
    build_index,
    load_scp_superclass_map,
    read_stem_list,
    verify_checksums,
    verify_signal_files,
    _parse_scp_codes,
)

# --------------------------------------------------------------- synthetic metadata

# code -> diagnostic_class, plus one non-diagnostic code (SR) with no class, which
# must be skipped so it never contributes a superclass label.
_SCP_ROWS = [
    ("NORM", "normal ECG", "NORM"),
    ("IMI", "inferior myocardial infarction", "MI"),
    ("NDT", "non-diagnostic T abnormalities", "STTC"),
    ("SR", "sinus rhythm", ""),  # rhythm statement, no diagnostic_class
]


def _write_scp_statements(root: Path) -> None:
    header = [
        "",
        "description",
        "diagnostic",
        "form",
        "rhythm",
        "diagnostic_class",
        "diagnostic_subclass",
    ]
    with (root / SCP_STATEMENTS_CSV).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for code, description, diagnostic_class in _SCP_ROWS:
            writer.writerow([code, description, "1.0", "", "", diagnostic_class, code])


def _write_database(root: Path, rows: Sequence[Dict[str, object]]) -> None:
    header = [
        "ecg_id",
        "patient_id",
        "age",
        "sex",
        "scp_codes",
        "strat_fold",
        "filename_lr",
        "filename_hr",
    ]
    with (root / DATABASE_CSV).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [
                    row["ecg_id"],
                    row["patient_id"],
                    row.get("age", ""),
                    row.get("sex", ""),
                    row["scp_codes"],  # csv.writer quotes the embedded commas
                    row["strat_fold"],
                    f"records100/{row['ecg_id']}_lr",
                    f"records500/{row['ecg_id']}_hr",
                ]
            )


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    """A tiny but structurally faithful PTB-XL metadata directory."""
    _write_scp_statements(tmp_path)
    rows = [
        # patient 15709 has two studies, both in fold 1 (train) -> no leakage
        {"ecg_id": 1, "patient_id": "15709.0", "age": "56.0", "sex": "1",
         "scp_codes": "{'NORM': 100.0, 'SR': 0.0}", "strat_fold": "1"},
        {"ecg_id": 2, "patient_id": "15709.0", "age": "57.0", "sex": "1",
         "scp_codes": "{'NORM': 100.0}", "strat_fold": "1"},
        # multi-label: MI + STTC
        {"ecg_id": 3, "patient_id": "13243.0", "age": "19.0", "sex": "0",
         "scp_codes": "{'IMI': 100.0, 'NDT': 50.0}", "strat_fold": "9"},
        # unlabeled: only a non-diagnostic rhythm code
        {"ecg_id": 4, "patient_id": "20372.0", "age": "37.0", "sex": "0",
         "scp_codes": "{'SR': 0.0}", "strat_fold": "10"},
        # a clean test-fold NORM
        {"ecg_id": 5, "patient_id": "999.0", "age": "70.0", "sex": "1",
         "scp_codes": "{'NORM': 100.0}", "strat_fold": "10"},
    ]
    _write_database(tmp_path, rows)
    return tmp_path


# ------------------------------------------------------------------ parsing units


def test_parse_scp_codes_reads_stringified_dict():
    assert _parse_scp_codes("{'NORM': 100.0, 'SR': 0.0}") == {"NORM": 100.0, "SR": 0.0}


def test_parse_scp_codes_tolerates_garbage():
    assert _parse_scp_codes("") == {}
    assert _parse_scp_codes("not a dict") == {}
    assert _parse_scp_codes("['NORM']") == {}  # list, not dict


def test_scp_map_skips_non_diagnostic_codes(synthetic_root: Path):
    mapping = load_scp_superclass_map(synthetic_root)
    assert mapping == {"NORM": "NORM", "IMI": "MI", "NDT": "STTC"}
    assert "SR" not in mapping  # no diagnostic_class -> excluded


# ------------------------------------------------------------------ index building


def test_build_index_parses_records_and_superclasses(synthetic_root: Path):
    index = build_index(synthetic_root)
    assert len(index.records) == 5

    by_id = {r.ecg_id: r for r in index.records}
    # patient_id float string -> int
    assert by_id[1].patient_id == 15709
    # SR contributes no superclass; NORM does
    assert by_id[1].superclasses == frozenset({"NORM"})
    # multi-label aggregation
    assert by_id[3].superclasses == frozenset({"MI", "STTC"})
    # only a non-diagnostic code -> unlabeled
    assert by_id[4].superclasses == frozenset()
    assert by_id[4].has_diagnostic_label is False


def test_partition_property_follows_fold_convention(synthetic_root: Path):
    by_id = {r.ecg_id: r for r in build_index(synthetic_root).records}
    assert by_id[1].partition == "train"       # fold 1
    assert by_id[3].partition == "validation"  # fold 9
    assert by_id[5].partition == "test"        # fold 10


def test_grouping_helpers(synthetic_root: Path):
    index = build_index(synthetic_root)
    by_patient = index.by_patient()
    assert set(by_patient) == {15709, 13243, 20372, 999}
    assert len(by_patient[15709]) == 2  # two studies for one patient
    by_super = index.by_superclass()
    assert {r.ecg_id for r in by_super["MI"]} == {3}
    assert {r.ecg_id for r in by_super["NORM"]} == {1, 2, 5}


# ------------------------------------------------------- patient-level leakage guard


def test_no_patient_spans_strat_fold_partitions(synthetic_root: Path):
    # The synthetic cohort is disjoint by construction; the guard must accept it.
    records = build_index(synthetic_root).records
    assert_no_patient_leakage(records)  # must not raise


def test_leakage_guard_rejects_patient_in_two_partitions():
    def rec(ecg_id: int, patient_id: int, fold: int) -> EcgRecord:
        return EcgRecord(
            ecg_id=ecg_id,
            patient_id=patient_id,
            strat_fold=fold,
            scp_codes={"NORM": 100.0},
            superclasses=frozenset({"NORM"}),
            age=50.0,
            sex=0,
            filename_lr="",
            filename_hr="",
        )

    # patient 7 appears in fold 1 (train) and fold 10 (test) -> leakage
    leaky = [rec(1, 7, 1), rec(2, 7, 10), rec(3, 8, 1)]
    with pytest.raises(PtbxlError, match="leakage"):
        assert_no_patient_leakage(leaky)


# --------------------------------------------------------------------- checksums


def test_verify_checksums_ok_mismatch_and_unknown(tmp_path: Path):
    target = tmp_path / DATABASE_CSV
    target.write_text("hello ptbxl", encoding="utf-8")
    real = hashlib.sha256(b"hello ptbxl").hexdigest()

    other = tmp_path / "other.csv"
    other.write_text("x", encoding="utf-8")

    # SHA256SUMS lists a correct digest for the db file and a wrong one for other.csv
    (tmp_path / CHECKSUMS_FILE).write_text(
        f"{real}  {DATABASE_CSV}\n" f"{'0' * 64}  other.csv\n", encoding="utf-8"
    )

    report = verify_checksums(tmp_path, [DATABASE_CSV, "other.csv", "absent.csv"])
    assert report[DATABASE_CSV]["ok"] is True
    assert report["other.csv"]["ok"] is False
    assert report["absent.csv"]["ok"] is None  # not on disk / no entry -> unknown


# ------------------------------------------------------------------------- audit


def test_audit_counts_on_synthetic_metadata(synthetic_root: Path):
    report = audit(build_index(synthetic_root))
    assert report.n_records == 5
    assert report.n_patients == 4
    assert report.n_records_per_patient_max == 2
    assert report.n_unlabeled_records == 1
    assert report.superclass_record_counts == {"MI": 1, "NORM": 3, "STTC": 1}
    assert report.partition_record_counts == {"test": 2, "train": 2, "validation": 1}
    assert report.partition_patient_counts == {"test": 2, "train": 1, "validation": 1}
    assert report.patient_leakage_free is True
    # round-trips to JSON
    assert report.to_dict()["n_records"] == 5


def test_build_index_raises_when_metadata_absent(tmp_path: Path):
    with pytest.raises(PtbxlError):
        build_index(tmp_path)  # empty dir, no CSVs


# ------------------------------------------------- pre-registered primary label


def _record(superclasses: Sequence[str], fold: int = 1) -> EcgRecord:
    return EcgRecord(
        ecg_id=1,
        patient_id=1,
        strat_fold=fold,
        scp_codes={},
        superclasses=frozenset(superclasses),
        age=50.0,
        sex=0,
        filename_lr="records100/00000/00001_lr",
        filename_hr="records500/00000/00001_hr",
    )


@pytest.mark.parametrize("abnormal", sorted(ABNORMAL_SUPERCLASSES))
def test_any_abnormal_superclass_is_positive(abnormal: str):
    assert _record([abnormal]).abnormal_label == 1


def test_norm_alone_is_negative():
    assert _record(["NORM"]).abnormal_label == 0


def test_norm_co_occurring_with_a_diagnosis_is_positive():
    """Screening-conservative, and fixed in docs §7.4 before any model existed.

    A record carrying NORM *and* an abnormal superclass had a diagnostic
    statement made about it. Reading it as negative would be the more flattering
    choice for a screening metric, which is exactly why the direction is pinned
    down in code rather than left to the caller.
    """
    assert _record(["NORM", "MI"]).abnormal_label == 1


def test_record_without_any_diagnostic_superclass_is_unknown_not_negative():
    record = _record([])

    assert record.abnormal_label is None, "unknown must never be read as healthy"
    assert record.has_diagnostic_label is False


def test_abnormal_superclasses_partition_the_superclass_set():
    """The label rule must cover every superclass exactly once."""
    assert ABNORMAL_SUPERCLASSES | {"NORM"} == set(SUPERCLASSES)
    assert "NORM" not in ABNORMAL_SUPERCLASSES
    assert set(SUPERCLASS_ORDER) == set(SUPERCLASSES)
    assert len(SUPERCLASS_ORDER) == len(SUPERCLASSES), "order must not repeat a class"
    assert SUPERCLASS_ORDER[0] == "NORM"


# ------------------------------------------------------- signal-corpus integrity


def _write_signal_pair(
    root: Path, stem: str, header: bytes = b"hea", samples: bytes = b"dat"
) -> Dict[str, bytes]:
    """Write a fake ``<stem>.hea``/``<stem>.dat`` pair; return their bytes by name."""
    path = root / stem
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".hea").write_bytes(header)
    path.with_suffix(".dat").write_bytes(samples)
    return {f"{stem}.hea": header, f"{stem}.dat": samples}


def _write_manifest(root: Path, contents: Dict[str, bytes]) -> None:
    lines = [f"{hashlib.sha256(data).hexdigest()}  {name}" for name, data in contents.items()]
    (root / CHECKSUMS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_stem_list_strips_crlf_blanks_and_comments(tmp_path: Path):
    """CRLF in this list once caused 398 silent ``curl: (3)`` failures."""
    listing = tmp_path / "stems.txt"
    listing.write_bytes(
        b"records100/00000/00001_lr\r\n"
        b"\r\n"
        b"# a comment\r\n"
        b"  records100/00000/00002_lr  \r\n"
        b"records100\\00000\\00003_lr\n"
    )

    stems = read_stem_list(listing)

    assert stems == [
        "records100/00000/00001_lr",
        "records100/00000/00002_lr",
        "records100/00000/00003_lr",  # backslashes normalised to the manifest's form
    ]


def test_verify_signal_files_reports_ok_mismatch_unknown_and_incomplete(tmp_path: Path):
    good = _write_signal_pair(tmp_path, "records100/00000/00001_lr", b"h1", b"d1")
    bad = _write_signal_pair(tmp_path, "records100/00000/00002_lr", b"h2", b"d2")
    unlisted = _write_signal_pair(tmp_path, "records100/00000/00003_lr", b"h3", b"d3")
    # Only the .hea of 00004 exists: an incomplete record, not a corrupt one.
    (tmp_path / "records100/00000/00004_lr.hea").write_bytes(b"h4")

    manifest = dict(good)
    manifest.update({name: b"tampered" for name in bad})  # digests will not match
    _write_manifest(tmp_path, manifest)  # 00003 deliberately absent from the manifest

    report = verify_signal_files(
        tmp_path,
        [
            "records100/00000/00001_lr",
            "records100/00000/00002_lr",
            "records100/00000/00003_lr",
            "records100/00000/00004_lr",
        ],
    )

    assert report.n_stems_requested == 4
    assert report.n_stems_complete == 3
    assert report.incomplete_stems == ["records100/00000/00004_lr"]
    assert report.n_files_checked == 6  # the lone .hea of 00004 is never hashed
    assert report.n_ok == 2
    assert report.n_mismatch == 2
    assert sorted(report.corrupt_files) == sorted(bad)
    assert sorted(report.unknown_files) == sorted(unlisted)
    assert report.n_unknown == 2
    assert report.total_bytes == 12  # six 2-byte files
    assert report.deleted_files == []
    assert report.fully_verified is False


def test_verify_signal_files_passes_a_clean_corpus(tmp_path: Path):
    contents = {}
    stems = []
    for index in (1, 2, 3):
        stem = f"records100/00000/{index:05d}_lr"
        stems.append(stem)
        contents.update(_write_signal_pair(tmp_path, stem, f"h{index}".encode(), f"d{index}".encode()))
    _write_manifest(tmp_path, contents)

    report = verify_signal_files(tmp_path, stems)

    assert report.n_ok == 6
    assert report.n_mismatch == 0
    assert report.n_unknown == 0
    assert report.n_stems_incomplete == 0
    assert report.fully_verified is True


def test_verify_signal_files_deletes_only_the_corrupt_file(tmp_path: Path):
    """Deletion is the self-healing step: re-running the fetch script repairs it."""
    stem = "records100/00000/00001_lr"
    contents = _write_signal_pair(tmp_path, stem, b"header", b"samples")
    manifest = dict(contents)
    manifest[f"{stem}.dat"] = b"a different recording"  # only the .dat disagrees
    _write_manifest(tmp_path, manifest)

    report = verify_signal_files(tmp_path, [stem], delete_corrupt=True)

    assert report.deleted_files == [f"{stem}.dat"]
    assert not (tmp_path / f"{stem}.dat").exists()
    assert (tmp_path / f"{stem}.hea").exists(), "a matching file must never be deleted"


def test_verify_signal_files_never_deletes_a_file_with_no_published_checksum(tmp_path: Path):
    stem = "records100/00000/00001_lr"
    _write_signal_pair(tmp_path, stem)
    _write_manifest(tmp_path, {})  # empty manifest: both files are 'unknown'

    report = verify_signal_files(tmp_path, [stem], delete_corrupt=True)

    assert report.n_unknown == 2
    assert report.deleted_files == []
    assert (tmp_path / f"{stem}.dat").exists()
    # Unknown is not a pass -- honesty over a false green.
    assert report.fully_verified is False


def test_verify_signal_files_report_round_trips_to_json(tmp_path: Path):
    stem = "records100/00000/00001_lr"
    contents = _write_signal_pair(tmp_path, stem)
    _write_manifest(tmp_path, contents)

    report = verify_signal_files(tmp_path, [stem])
    out = tmp_path / "signal_verification_report.json"
    report.save(out)

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["n_ok"] == 2
    assert written["fully_verified"] is True
    assert written["root"] == str(tmp_path)


def test_verify_signal_files_on_an_empty_request_is_not_a_pass(tmp_path: Path):
    _write_manifest(tmp_path, {})

    report = verify_signal_files(tmp_path, [])

    assert report.n_files_checked == 0
    assert report.fully_verified is False, "verifying nothing must not report success"


def test_verify_signal_files_raises_when_no_root_can_be_found(tmp_path, monkeypatch):
    monkeypatch.setenv(PTBXL_ROOT_ENV, str(tmp_path / "nowhere"))

    with pytest.raises(PtbxlError):
        verify_signal_files(None, ["records100/00000/00001_lr"])


# --------------------------------------------------- real downloaded metadata (opt)

_REAL_ROOT = Path(__file__).resolve().parents[1] / "backend" / "artifacts" / "dataset" / "ptbxl"
_HAVE_REAL = (_REAL_ROOT / DATABASE_CSV).is_file()


@pytest.mark.skipif(not _HAVE_REAL, reason="PTB-XL metadata not downloaded (gitignored)")
def test_real_ptbxl_audit_matches_published_figures():
    """Lock the audited headline figures against PhysioNet's published counts.

    Only runs where the gitignored metadata is present; guarantees that when it is,
    the numbers this project reports (21,799 records / 18,869 patients, checksum OK,
    patient-disjoint folds) are the real ones and stay that way.
    """
    report = audit(root=_REAL_ROOT)
    assert report.n_records == 21799
    assert report.n_patients == 18869
    assert report.patient_leakage_free is True
    assert report.checksums[DATABASE_CSV]["ok"] is True
    assert report.superclass_record_counts["NORM"] == 9514
