"""PTB-XL dataset acquisition + audit (Phase E4 data-expansion arena).

PTB-XL is PhysioNet's large open 12-lead ECG corpus: 21,799 records from
18,869 patients, released CC BY 4.0 with no credentialing. It is the recommended
arena for the E4 quantum-advantage push because — unlike SMART-OM's megapixel
images, which E3 had to crush to 8 amplitudes — ECG yields naturally compact,
clinically-meaningful feature vectors that sit in the very dimensionality a
near-term quantum kernel/feature-map actually operates in. See
``docs/PHASE_E4_DATA_EXPANSION.md``.

The audit path of this module never touches the multi-GB signal files. It works
entirely from the two small metadata CSVs so the dataset can be *audited* —
record/patient counts,
class distribution, and above all the patient-level fold disjointness that every
E-series experiment depends on — before any signal is downloaded or any model is
trained. It deliberately mirrors ``backend/dataset/smartom.py`` (indexer +
dataclass records + ``by_patient`` grouping) and reuses the leakage-guard idea
from ``backend/dataset/split.py`` (re-derive patient->partition from the record
assignments and refuse any patient that spans partitions).

:func:`verify_signal_files` is the one function that does read the signal files,
and only as opaque bytes: it hashes each downloaded ``.hea``/``.dat`` against
PhysioNet's manifest so a truncated or corrupted fetch cannot reach the feature
extractor. It never decodes samples and never looks at a label.

Provenance (recorded so an audit is reproducible):

* Source .......... https://physionet.org/content/ptb-xl/1.0.3/
* Files root ....... https://physionet.org/files/ptb-xl/1.0.3/
* Version .......... 1.0.3
* Licence ......... Creative Commons Attribution 4.0 International (CC BY 4.0)
* Metadata ........ ptbxl_database.csv (one row per ECG) + scp_statements.csv
                    (SCP code -> diagnostic superclass dictionary)

The published ``ptbxl_database.csv`` SHA-256 is verified against PhysioNet's own
``SHA256SUMS.txt`` by :func:`verify_checksums`; no figure this module reports is
trusted until that check passes.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Provenance constants
# --------------------------------------------------------------------------

PTBXL_VERSION = "1.0.3"
PTBXL_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
PTBXL_CONTENT_URL = "https://physionet.org/content/ptb-xl/1.0.3/"
PTBXL_FILES_URL = "https://physionet.org/files/ptb-xl/1.0.3/"

DATABASE_CSV = "ptbxl_database.csv"
SCP_STATEMENTS_CSV = "scp_statements.csv"
CHECKSUMS_FILE = "SHA256SUMS.txt"
VERSION_FILE = "VERSION.txt"

# The five diagnostic superclasses PTB-XL aggregates SCP codes into.
SUPERCLASSES: FrozenSet[str] = frozenset({"NORM", "MI", "STTC", "CD", "HYP"})

SUPERCLASS_ORDER: Tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")
"""Fixed column order for any one-hot/multi-label encoding of the superclasses."""

ABNORMAL_SUPERCLASSES: FrozenSet[str] = frozenset({"MI", "STTC", "CD", "HYP"})
"""The positive side of the pre-registered primary task (docs §7.4)."""

# Environment override, matching the CARESCAN_DATASET_ROOT convention in smartom.py.
PTBXL_ROOT_ENV = "CARESCAN_PTBXL_ROOT"

# Default location: gitignored artifacts dataset dir (see .gitignore).
_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "dataset" / "ptbxl"

# Fold convention for the frozen partitions. PTB-XL ships ``strat_fold`` 1..10,
# assigned per patient; folds 9 and 10 are the authors' recommended validation
# and test folds. The test fold is frozen: it is read once, after an architecture
# is chosen, and never used for tuning.
TRAIN_FOLDS: FrozenSet[int] = frozenset(range(1, 9))  # 1..8
VALIDATION_FOLDS: FrozenSet[int] = frozenset({9})
TEST_FOLDS: FrozenSet[int] = frozenset({10})


class PtbxlError(RuntimeError):
    """Raised when the PTB-XL metadata is missing, malformed, or leaks patients."""


# --------------------------------------------------------------------------
# Records and index
# --------------------------------------------------------------------------


@dataclass
class EcgRecord:
    """One ECG study from ``ptbxl_database.csv`` (metadata only)."""

    ecg_id: int
    patient_id: int
    strat_fold: int
    scp_codes: Dict[str, float]
    superclasses: FrozenSet[str]
    age: Optional[float]
    sex: Optional[int]
    filename_lr: str
    filename_hr: str

    @property
    def partition(self) -> str:
        """Frozen partition implied by ``strat_fold``."""
        if self.strat_fold in TEST_FOLDS:
            return "test"
        if self.strat_fold in VALIDATION_FOLDS:
            return "validation"
        if self.strat_fold in TRAIN_FOLDS:
            return "train"
        return "unassigned"

    @property
    def has_diagnostic_label(self) -> bool:
        """True when at least one SCP code maps to a diagnostic superclass."""
        return len(self.superclasses) > 0

    @property
    def abnormal_label(self) -> Optional[int]:
        """The pre-registered primary label: 1 abnormal, 0 normal, None unknown.

        Fixed in ``docs/PHASE_E4_DATA_EXPANSION.md`` §7.4 *before any model was
        fitted*, and implemented here so no caller can quietly re-define it:

        * 1 when any of ``MI``/``STTC``/``CD``/``HYP`` is present -- including
          when ``NORM`` is present too, because a diagnostic statement was made
          and the screening-conservative reading is the positive one;
        * 0 when ``NORM`` is present and none of the four are;
        * ``None`` when the record carries no diagnostic superclass at all. An
          unknown label is *not* a healthy one, so such records are excluded
          rather than counted as negatives.
        """
        if self.superclasses & ABNORMAL_SUPERCLASSES:
            return 1
        if "NORM" in self.superclasses:
            return 0
        return None


@dataclass
class PtbxlIndex:
    """Fully parsed PTB-XL metadata index (no signal files required)."""

    root: Path
    records: List[EcgRecord] = field(default_factory=list)
    scp_to_superclass: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def by_patient(self) -> Dict[int, List[EcgRecord]]:
        out: Dict[int, List[EcgRecord]] = defaultdict(list)
        for record in self.records:
            out[record.patient_id].append(record)
        return dict(out)

    def by_superclass(self) -> Dict[str, List[EcgRecord]]:
        """Records grouped by superclass (a multi-label record appears in each)."""
        out: Dict[str, List[EcgRecord]] = defaultdict(list)
        for record in self.records:
            for superclass in record.superclasses:
                out[superclass].append(record)
        return dict(out)

    def by_partition(self) -> Dict[str, List[EcgRecord]]:
        out: Dict[str, List[EcgRecord]] = defaultdict(list)
        for record in self.records:
            out[record.partition].append(record)
        return dict(out)


# --------------------------------------------------------------------------
# Root discovery
# --------------------------------------------------------------------------


def discover_ptbxl_root(start: Optional[Path] = None) -> Optional[Path]:
    """Locate the PTB-XL metadata directory.

    Honours ``CARESCAN_PTBXL_ROOT`` first, then the supplied ``start``, then the
    default gitignored ``backend/artifacts/dataset/ptbxl``. A directory qualifies
    only if it actually contains ``ptbxl_database.csv``.
    """
    env = os.environ.get(PTBXL_ROOT_ENV)
    if env:
        candidate = Path(env).expanduser()
        return candidate if (candidate / DATABASE_CSV).is_file() else None

    for candidate in (start, _DEFAULT_ROOT):
        if candidate is not None and (Path(candidate) / DATABASE_CSV).is_file():
            return Path(candidate)
    return None


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _parse_scp_codes(raw: str) -> Dict[str, float]:
    """Parse the stringified ``{'NORM': 100.0, ...}`` dict from the CSV.

    Malformed cells return ``{}`` rather than raising, so one bad row cannot abort
    an audit; the audit counts such rows as unlabeled.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: Dict[str, float] = {}
    for key, value in parsed.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            out[str(key)] = 0.0
    return out


def _to_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: str) -> Optional[float]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_scp_superclass_map(root: Path) -> Dict[str, str]:
    """Map each SCP code to its diagnostic superclass from ``scp_statements.csv``.

    Only codes with a non-empty ``diagnostic_class`` are diagnostic; form/rhythm
    statements without one are intentionally skipped, so the returned mapping
    contains exactly the codes that contribute to a superclass label.
    """
    path = root / SCP_STATEMENTS_CSV
    if not path.is_file():
        raise PtbxlError(f"Missing {SCP_STATEMENTS_CSV} in {root}")

    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise PtbxlError(f"{SCP_STATEMENTS_CSV} is empty")
        try:
            dc_idx = header.index("diagnostic_class")
        except ValueError as exc:  # pragma: no cover - schema guard
            raise PtbxlError("scp_statements.csv has no diagnostic_class column") from exc
        for row in reader:
            if not row:
                continue
            code = row[0].strip()
            superclass = row[dc_idx].strip() if dc_idx < len(row) else ""
            if code and superclass:
                mapping[code] = superclass
    return mapping


def build_index(root: Optional[Path] = None) -> PtbxlIndex:
    """Parse PTB-XL metadata into an index (no signal files touched).

    Raises:
        PtbxlError: if the metadata directory or CSVs cannot be found.
    """
    resolved = Path(root) if root is not None else discover_ptbxl_root()
    if resolved is None or not (resolved / DATABASE_CSV).is_file():
        raise PtbxlError(
            f"PTB-XL metadata not found. Set {PTBXL_ROOT_ENV} or pass root= to a "
            f"directory containing {DATABASE_CSV}."
        )

    scp_map = load_scp_superclass_map(resolved)
    index = PtbxlIndex(root=resolved, scp_to_superclass=dict(scp_map))

    db_path = resolved / DATABASE_CSV
    with db_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ecg_id = _to_int(row.get("ecg_id", ""))
            patient_id = _to_int(row.get("patient_id", ""))
            strat_fold = _to_int(row.get("strat_fold", ""))
            if ecg_id is None or patient_id is None or strat_fold is None:
                index.warnings.append(
                    f"Skipped row with missing ecg_id/patient_id/strat_fold: "
                    f"ecg_id={row.get('ecg_id')!r}"
                )
                continue
            scp_codes = _parse_scp_codes(row.get("scp_codes", ""))
            superclasses = frozenset(
                scp_map[code] for code in scp_codes if code in scp_map
            )
            index.records.append(
                EcgRecord(
                    ecg_id=ecg_id,
                    patient_id=patient_id,
                    strat_fold=strat_fold,
                    scp_codes=scp_codes,
                    superclasses=superclasses,
                    age=_to_float(row.get("age", "")),
                    sex=_to_int(row.get("sex", "")),
                    filename_lr=(row.get("filename_lr", "") or "").strip(),
                    filename_hr=(row.get("filename_hr", "") or "").strip(),
                )
            )
    return index


# --------------------------------------------------------------------------
# Patient-level leakage guard (mirrors split.SplitManifest.assert_no_patient_leakage)
# --------------------------------------------------------------------------


def assert_no_patient_leakage(records: Sequence[EcgRecord]) -> None:
    """Re-derive patient->partition from the fold assignments; refuse conflicts.

    This is the same guarantee ``split.SplitManifest.assert_no_patient_leakage``
    enforces for images: a patient's every study must land in exactly one
    partition, so the frozen test fold shares no patient with train or validation.
    """
    patient_partitions: Dict[int, str] = {}
    conflicts: Dict[int, set] = defaultdict(set)
    for record in records:
        partition = record.partition
        seen = patient_partitions.setdefault(record.patient_id, partition)
        if seen != partition:
            conflicts[record.patient_id].update({seen, partition})
    if conflicts:
        sample = list(conflicts.items())[:5]
        detail = "; ".join(f"patient {pid} in {sorted(parts)}" for pid, parts in sample)
        raise PtbxlError(
            f"Patient-level leakage across strat_fold partitions: "
            f"{len(conflicts)} patient(s) span partitions ({detail})."
        )


# --------------------------------------------------------------------------
# Checksums
# --------------------------------------------------------------------------


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_expected_checksums(root: Path) -> Dict[str, str]:
    """Parse PhysioNet's ``SHA256SUMS.txt`` (``<hex>  <filename>`` lines)."""
    path = root / CHECKSUMS_FILE
    expected: Dict[str, str] = {}
    if not path.is_file():
        return expected
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            digest, name = parts
            expected[name.strip().lstrip("*")] = digest.strip()
    return expected


def verify_checksums(
    root: Path, filenames: Iterable[str] = (DATABASE_CSV,)
) -> Dict[str, Dict[str, Optional[str]]]:
    """Verify downloaded metadata files against PhysioNet's SHA256SUMS.

    Returns a per-file report ``{name: {expected, actual, ok}}``. A file with no
    published checksum is reported ``ok=None`` (unknown) rather than passed —
    honesty over a false green.
    """
    expected = _load_expected_checksums(root)
    report: Dict[str, Dict[str, Optional[str]]] = {}
    for name in filenames:
        file_path = root / name
        actual = _sha256(file_path) if file_path.is_file() else None
        want = expected.get(name)
        ok: Optional[bool]
        if actual is None or want is None:
            ok = None
        else:
            ok = actual == want
        report[name] = {"expected": want, "actual": actual, "ok": ok}
    return report


# --------------------------------------------------------------------------
# Signal-corpus integrity
# --------------------------------------------------------------------------


@dataclass
class SignalVerificationReport:
    """Integrity report for a downloaded subset of the WFDB signal corpus.

    A *stem* is a record path without extension, exactly as PTB-XL's
    ``filename_lr`` column spells it (``records100/00000/00001_lr``). Each stem
    must be present on disk as a ``.hea`` header **and** a ``.dat`` sample file,
    and each of those two files is hashed against PhysioNet's published manifest.
    """

    root: str
    generated_at: str
    n_stems_requested: int
    n_stems_complete: int
    n_stems_incomplete: int
    n_files_checked: int
    n_ok: int
    n_mismatch: int
    n_unknown: int
    total_bytes: int
    incomplete_stems: List[str]
    corrupt_files: List[str]
    unknown_files: List[str]
    deleted_files: List[str]

    @property
    def fully_verified(self) -> bool:
        """True only when every requested stem is present and every hash matched.

        ``n_unknown`` counts files PhysioNet publishes no checksum for; those are
        *not* treated as passes, so an unknown keeps this False.
        """
        return (
            self.n_stems_requested > 0
            and self.n_stems_incomplete == 0
            and self.n_mismatch == 0
            and self.n_unknown == 0
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "root": self.root,
            "generated_at": self.generated_at,
            "n_stems_requested": self.n_stems_requested,
            "n_stems_complete": self.n_stems_complete,
            "n_stems_incomplete": self.n_stems_incomplete,
            "n_files_checked": self.n_files_checked,
            "n_ok": self.n_ok,
            "n_mismatch": self.n_mismatch,
            "n_unknown": self.n_unknown,
            "total_bytes": self.total_bytes,
            "fully_verified": self.fully_verified,
            "incomplete_stems": self.incomplete_stems,
            "corrupt_files": self.corrupt_files,
            "unknown_files": self.unknown_files,
            "deleted_files": self.deleted_files,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def read_stem_list(path: Path) -> List[str]:
    """Read a newline-separated list of record stems, tolerating CRLF.

    The download script consumes the same file. CRLF once produced 398 silent
    ``curl: (3) URL rejected`` failures, so both readers strip it explicitly.
    """
    stems: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stem = line.strip().replace("\\", "/")
            if stem and not stem.startswith("#"):
                stems.append(stem)
    return stems


def verify_signal_files(
    root: Optional[Path] = None,
    stems: Optional[Iterable[str]] = None,
    *,
    delete_corrupt: bool = False,
    progress_every: int = 0,
) -> SignalVerificationReport:
    """Hash every ``.hea``/``.dat`` of ``stems`` against PhysioNet's manifest.

    This is the self-healing half of the bulk download: with
    ``delete_corrupt=True`` a file whose SHA-256 does not match the published
    digest is removed, so re-running the idempotent fetch script repairs the
    corpus rather than leaving a silently-truncated record in place. A file with
    no published checksum is counted as *unknown* and never deleted — the same
    "honesty over a false green" rule :func:`verify_checksums` follows.

    The manifest is parsed once, not per file: it holds ~65k entries.
    """
    resolved = Path(root) if root is not None else discover_ptbxl_root()
    if resolved is None:
        raise PtbxlError(
            f"PTB-XL root not found. Set {PTBXL_ROOT_ENV} or pass root= to the "
            f"directory holding {CHECKSUMS_FILE} and records100/."
        )
    expected = _load_expected_checksums(resolved)
    stem_list = [str(stem).strip().replace("\\", "/") for stem in (stems or [])]
    stem_list = [stem for stem in stem_list if stem]

    incomplete: List[str] = []
    corrupt: List[str] = []
    unknown: List[str] = []
    deleted: List[str] = []
    n_files = n_ok = n_mismatch = 0
    total_bytes = 0
    complete = 0

    for position, stem in enumerate(stem_list, start=1):
        paths = [(f"{stem}{suffix}", resolved / f"{stem}{suffix}") for suffix in (".hea", ".dat")]
        if not all(path.is_file() for _, path in paths):
            incomplete.append(stem)
            continue
        complete += 1
        for name, path in paths:
            n_files += 1
            total_bytes += path.stat().st_size
            want = expected.get(name)
            if want is None:
                unknown.append(name)
                continue
            if _sha256(path) == want:
                n_ok += 1
            else:
                n_mismatch += 1
                corrupt.append(name)
                if delete_corrupt:
                    path.unlink(missing_ok=True)
                    deleted.append(name)
        if progress_every and position % progress_every == 0:
            print(
                f"  verified {position}/{len(stem_list)} stems "
                f"(ok={n_ok} mismatch={n_mismatch} unknown={len(unknown)})",
                flush=True,
            )

    return SignalVerificationReport(
        root=str(resolved),
        generated_at=datetime.now(timezone.utc).isoformat(),
        n_stems_requested=len(stem_list),
        n_stems_complete=complete,
        n_stems_incomplete=len(incomplete),
        n_files_checked=n_files,
        n_ok=n_ok,
        n_mismatch=n_mismatch,
        n_unknown=len(unknown),
        total_bytes=total_bytes,
        incomplete_stems=incomplete,
        corrupt_files=corrupt,
        unknown_files=unknown,
        deleted_files=deleted,
    )


# --------------------------------------------------------------------------
# Audit report
# --------------------------------------------------------------------------


@dataclass
class PtbxlAuditReport:
    """Formal, serialisable audit of the PTB-XL metadata."""

    root: str
    version: Optional[str]
    generated_at: str
    n_records: int
    n_patients: int
    n_records_per_patient_max: int
    superclass_record_counts: Dict[str, int]
    partition_record_counts: Dict[str, int]
    partition_patient_counts: Dict[str, int]
    n_unlabeled_records: int
    n_missing_age: int
    n_missing_sex: int
    sex_counts: Dict[str, int]
    fold_counts: Dict[str, int]
    patient_leakage_free: bool
    checksums: Dict[str, Dict[str, Optional[str]]]
    warnings: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "root": self.root,
            "version": self.version,
            "generated_at": self.generated_at,
            "n_records": self.n_records,
            "n_patients": self.n_patients,
            "n_records_per_patient_max": self.n_records_per_patient_max,
            "superclass_record_counts": self.superclass_record_counts,
            "partition_record_counts": self.partition_record_counts,
            "partition_patient_counts": self.partition_patient_counts,
            "n_unlabeled_records": self.n_unlabeled_records,
            "n_missing_age": self.n_missing_age,
            "n_missing_sex": self.n_missing_sex,
            "sex_counts": self.sex_counts,
            "fold_counts": self.fold_counts,
            "patient_leakage_free": self.patient_leakage_free,
            "checksums": self.checksums,
            "warnings": self.warnings,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _read_version(root: Path) -> Optional[str]:
    path = root / VERSION_FILE
    if path.is_file():
        return path.read_text(encoding="utf-8").strip() or None
    return None


def audit(index: Optional[PtbxlIndex] = None, root: Optional[Path] = None) -> PtbxlAuditReport:
    """Produce a formal audit report from the PTB-XL metadata.

    Verifies the checksum of ``ptbxl_database.csv``, counts records/patients/
    classes, characterises missing labels and demographics, and — crucially —
    checks that no patient spans the strat_fold partitions (the frozen-test
    guarantee). ``patient_leakage_free`` is the go/no-go for trusting the folds.
    """
    if index is None:
        index = build_index(root)
    records = index.records
    resolved_root = index.root

    by_patient = index.by_patient()
    by_partition = index.by_partition()

    superclass_counts = Counter()
    for record in records:
        for superclass in record.superclasses:
            superclass_counts[superclass] += 1

    partition_record_counts = {p: len(recs) for p, recs in sorted(by_partition.items())}
    partition_patient_counts = {
        p: len({r.patient_id for r in recs}) for p, recs in sorted(by_partition.items())
    }

    sex_counter = Counter()
    for record in records:
        sex_counter[str(record.sex) if record.sex is not None else "missing"] += 1

    fold_counts = Counter(str(r.strat_fold) for r in records)

    try:
        assert_no_patient_leakage(records)
        leakage_free = True
    except PtbxlError:
        leakage_free = False

    report = PtbxlAuditReport(
        root=str(resolved_root),
        version=_read_version(resolved_root),
        generated_at=datetime.now(timezone.utc).isoformat(),
        n_records=len(records),
        n_patients=len(by_patient),
        n_records_per_patient_max=max((len(v) for v in by_patient.values()), default=0),
        superclass_record_counts=dict(sorted(superclass_counts.items())),
        partition_record_counts=partition_record_counts,
        partition_patient_counts=partition_patient_counts,
        n_unlabeled_records=sum(1 for r in records if not r.has_diagnostic_label),
        n_missing_age=sum(1 for r in records if r.age is None),
        n_missing_sex=sum(1 for r in records if r.sex is None),
        sex_counts=dict(sorted(sex_counter.items())),
        fold_counts=dict(sorted(fold_counts.items(), key=lambda kv: int(kv[0]))),
        patient_leakage_free=leakage_free,
        checksums=verify_checksums(resolved_root),
        warnings=list(index.warnings),
    )
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _format_report(report: PtbxlAuditReport) -> str:
    lines = [
        "PTB-XL metadata audit",
        "=====================",
        f"root ................ {report.root}",
        f"version ............. {report.version}",
        f"records ............. {report.n_records}",
        f"patients ............ {report.n_patients}",
        f"max studies/patient . {report.n_records_per_patient_max}",
        f"unlabeled records ... {report.n_unlabeled_records}",
        f"missing age ......... {report.n_missing_age}",
        f"missing sex ......... {report.n_missing_sex}",
        f"sex counts .......... {report.sex_counts}",
        f"patient leakage-free  {report.patient_leakage_free}",
        "",
        "superclass record counts (multi-label; sums may exceed record count):",
    ]
    for name, count in report.superclass_record_counts.items():
        lines.append(f"  {name:<6} {count}")
    lines.append("")
    lines.append("partition (from strat_fold: 1-8 train, 9 validation, 10 test):")
    for part in ("train", "validation", "test", "unassigned"):
        recs = report.partition_record_counts.get(part)
        pats = report.partition_patient_counts.get(part)
        if recs is not None:
            lines.append(f"  {part:<11} records={recs:<6} patients={pats}")
    lines.append("")
    lines.append("checksums (vs PhysioNet SHA256SUMS.txt):")
    for name, info in report.checksums.items():
        status = {True: "OK", False: "MISMATCH", None: "unknown"}[info["ok"]]
        lines.append(f"  {name:<22} {status}")
    if report.warnings:
        lines.append("")
        lines.append(f"warnings: {len(report.warnings)} (first 3 shown)")
        for warning in report.warnings[:3]:
            lines.append(f"  - {warning}")
    return "\n".join(lines)


def _format_signal_report(report: SignalVerificationReport) -> str:
    mib = report.total_bytes / (1024.0 * 1024.0)
    lines = [
        "PTB-XL signal-corpus integrity",
        "==============================",
        f"root ................ {report.root}",
        f"stems requested ..... {report.n_stems_requested}",
        f"stems complete ...... {report.n_stems_complete}",
        f"stems incomplete .... {report.n_stems_incomplete}",
        f"files hashed ........ {report.n_files_checked}",
        f"  SHA-256 OK ....... {report.n_ok}",
        f"  MISMATCH ......... {report.n_mismatch}",
        f"  unknown (no sum) . {report.n_unknown}",
        f"bytes on disk ....... {report.total_bytes} ({mib:.1f} MiB)",
        f"fully verified ...... {report.fully_verified}",
    ]
    if report.deleted_files:
        lines.append("")
        lines.append(
            f"deleted {len(report.deleted_files)} corrupt file(s); re-run the fetch "
            "script to repair"
        )
    for label, items in (
        ("incomplete stems", report.incomplete_stems),
        ("corrupt files", report.corrupt_files),
        ("files with no published checksum", report.unknown_files),
    ):
        if items:
            lines.append("")
            lines.append(f"{label}: {len(items)} (first 5 shown)")
            lines.extend(f"  - {item}" for item in items[:5])
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="PTB-XL metadata audit")
    parser.add_argument("--root", type=Path, default=None, help="PTB-XL metadata directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON audit report here (default: <root>/audit_report.json)",
    )
    parser.add_argument(
        "--verify-signals",
        type=Path,
        default=None,
        metavar="STEM_LIST",
        help=(
            "Instead of the metadata audit, hash the .hea/.dat of every record stem "
            "listed in this file against PhysioNet's SHA256SUMS.txt"
        ),
    )
    parser.add_argument(
        "--delete-corrupt",
        action="store_true",
        help="With --verify-signals: delete files whose SHA-256 does not match, so "
        "re-running the fetch script repairs them",
    )
    args = parser.parse_args(argv)

    if args.verify_signals is not None:
        stems = read_stem_list(args.verify_signals)
        report = verify_signal_files(
            args.root,
            stems,
            delete_corrupt=args.delete_corrupt,
            progress_every=2000,
        )
        out = args.out or (Path(report.root) / "signal_verification_report.json")
        report.save(out)
        print(_format_signal_report(report))
        print(f"\nJSON report written to {out}")
        return 0 if report.fully_verified else 1

    index = build_index(args.root)
    report = audit(index)
    out = args.out or (Path(report.root) / "audit_report.json")
    report.save(out)
    print(_format_report(report))
    print(f"\nJSON audit report written to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
