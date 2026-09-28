"""Raw 12-lead waveform cache for the E4 deep rung (``ptbxl-signal-1``).

The tabular rungs of the classical failure map (``e4_classical_baseline``) consume the
97-dimensional ``ecg-v1`` vector. The deep rung consumes what that vector was extracted
*from*: the raw 100 Hz, 12-lead, 10-second waveform. This module builds that cache once,
so training does not re-decode 19,601 WFDB records on every epoch.

Three properties are load-bearing, and each is enforced rather than assumed.

**Fold 10 is refused.** Like the cohort builder, this driver will not touch the frozen test
partition without an explicit override -- and on this machine the fold-10 signals were never
downloaded at all, so the refusal is a second lock on a door that has no key.

**Nothing is fitted here.** The cache stores physical millivolts exactly as the header
declares them. No mean, no scale, no filter, no resampling -- not one statistic pooled
across records. Per-record standardisation happens inside the model's input layer, where it
is a *record-local* operation and therefore fold-honest by construction, the same principle
that makes ``backend/ml/features_ecg.py`` structurally safe. A scaler fitted across the
corpus and baked in here would silently leak validation statistics into every future run,
and no downstream guard could detect it.

**The layout is fixed and validated.** Every row is ``(12, 1000)`` with leads in
``STANDARD_LEADS`` order, so channel *c* means the same lead in every record. A record that
decodes to a different length, a different lead set, or a non-finite sample is recorded in
``failures`` with its reason and **omitted** -- never padded, never zero-filled, never
quietly reordered. ``n_rows + n_failures == n_candidates`` is asserted before the cache is
written.

The array is stored as a plain ``.npy`` so it can be memory-mapped: at float32 the folds-1-9
corpus is ~900 MiB, which should not have to be resident to train on it.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from backend.dataset.ptbxl import (
    TEST_FOLDS,
    PtbxlError,
    PtbxlIndex,
    build_index,
)
from backend.dataset.wfdb_reader import WfdbError, read_record
from backend.ml.features_ecg import STANDARD_LEADS
from backend.training.prepare_ecg_features import DEV_FOLDS

logger = logging.getLogger(__name__)

SIGNAL_CACHE_VERSION = "ptbxl-signal-1"
SIGNAL_ARRAY_FILENAME = "signals_{version}.npy"
SIGNAL_INDEX_FILENAME = "signals_{version}_index.npz"

EXPECTED_SAMPLES = 1000
"""10 seconds at 100 Hz. PTB-XL's ``records100`` tree is uniform; a record that is not is a
defect in that record, and it is reported rather than reshaped into conformity."""

EXPECTED_FREQUENCY = 100.0


class SignalCacheError(RuntimeError):
    """Raised when the cache cannot be built or trusted."""


def signal_array_path(root: Path, version: str = SIGNAL_CACHE_VERSION) -> Path:
    return Path(root) / SIGNAL_ARRAY_FILENAME.format(version=version)


def signal_index_path(root: Path, version: str = SIGNAL_CACHE_VERSION) -> Path:
    return Path(root) / SIGNAL_INDEX_FILENAME.format(version=version)


def _lead_permutation(lead_names: Sequence[str]) -> Optional[List[int]]:
    """Column indices that put ``lead_names`` into ``STANDARD_LEADS`` order.

    Returns ``None`` when the record does not carry exactly the 12 standard leads. A
    missing or extra lead changes what the channel axis means, so such a record is a
    failure rather than something to pad.
    """
    if len(lead_names) != len(STANDARD_LEADS):
        # A 13-channel record could still be indexed by name, dropping the extra column,
        # but PTB-XL's records100 tree is uniformly 12-lead: such a record is an anomaly
        # worth reporting, not a column to discard quietly.
        return None
    available = {name.strip().upper(): index for index, name in enumerate(lead_names)}
    if len(available) != len(lead_names):
        return None
    try:
        return [available[lead] for lead in STANDARD_LEADS]
    except KeyError:
        return None


def build_signal_cache(
    index: Optional[PtbxlIndex] = None,
    *,
    root: Optional[Path] = None,
    out_dir: Path,
    folds: Optional[Iterable[int]] = None,
    limit: Optional[int] = None,
    allow_test_fold: bool = False,
    verify_checksums: bool = True,
    version: str = SIGNAL_CACHE_VERSION,
    progress_every: int = 1000,
) -> Dict[str, Any]:
    """Decode every record in ``folds`` into one ``(n, 12, 1000)`` float32 array.

    Records are ordered by ``ecg_id``, identically to
    ``backend.training.prepare_ecg_features.extract_cohort``, so the two caches can be
    aligned on ``ecg_id`` without either one having to trust the other's ordering.
    """
    if index is None:
        index = build_index(root)
    wanted = set(folds) if folds is not None else set(DEV_FOLDS)
    if wanted & set(TEST_FOLDS) and not allow_test_fold:
        raise PtbxlError(
            "Fold 10 is the frozen test partition. Caching its waveforms requires "
            "allow_test_fold=True (docs/PHASE_E4_DATA_EXPANSION.md §7.4)."
        )

    signal_root = Path(index.root)
    records = sorted(
        (record for record in index.records if record.strat_fold in wanted),
        key=lambda record: record.ecg_id,
    )
    if limit is not None:
        records = records[:limit]
    if not records:
        raise SignalCacheError(f"No records in folds {sorted(wanted)}.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    array_path = signal_array_path(out_dir, version)

    # Allocate the full array on disk and fill it in place: holding 19,601 x 12 x 1000
    # float32 in a Python list of arrays would peak at roughly twice the final size.
    buffer = np.lib.format.open_memmap(
        array_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(records), len(STANDARD_LEADS), EXPECTED_SAMPLES),
    )

    ecg_ids: List[int] = []
    patient_ids: List[int] = []
    strat_folds: List[int] = []
    failures: Dict[str, str] = {}
    checksum_unverified = 0
    written = 0

    started = time.perf_counter()
    for position, record in enumerate(records, start=1):
        if progress_every and position % progress_every == 0:
            elapsed = time.perf_counter() - started
            logger.info(
                "  %d/%d  (%.0f rec/s, %d cached, %d failed)",
                position,
                len(records),
                position / max(elapsed, 1e-9),
                written,
                len(failures),
            )
        stem = signal_root / record.filename_lr
        try:
            signal = read_record(stem, verify_checksums=verify_checksums)
        except (WfdbError, OSError) as error:
            failures[str(record.ecg_id)] = f"{type(error).__name__}: {error}"
            continue

        permutation = _lead_permutation(signal.lead_names)
        if permutation is None:
            failures[str(record.ecg_id)] = f"lead set {signal.lead_names} is not the 12 standard leads"
            continue
        if signal.n_samples != EXPECTED_SAMPLES:
            failures[str(record.ecg_id)] = (
                f"{signal.n_samples} samples, expected {EXPECTED_SAMPLES}"
            )
            continue
        if abs(float(signal.sampling_frequency) - EXPECTED_FREQUENCY) > 1e-6:
            failures[str(record.ecg_id)] = (
                f"{signal.sampling_frequency} Hz, expected {EXPECTED_FREQUENCY}"
            )
            continue

        samples = np.asarray(signal.samples, dtype=np.float32)[:, permutation].T
        if not np.isfinite(samples).all():
            failures[str(record.ecg_id)] = "non-finite samples in decoded waveform"
            continue
        if signal.checksums_ok is None:
            checksum_unverified += 1

        buffer[written] = samples
        ecg_ids.append(int(record.ecg_id))
        patient_ids.append(int(record.patient_id))
        strat_folds.append(int(record.strat_fold))
        written += 1

    elapsed = time.perf_counter() - started
    if written + len(failures) != len(records):
        raise SignalCacheError(
            f"Accounting error: {written} cached + {len(failures)} failed "
            f"!= {len(records)} candidates."
        )
    if written == 0:
        raise SignalCacheError("No record decoded successfully; refusing to write a cache.")

    # Truncate to the rows actually written. Reopening in r+ and copying is cheaper in
    # peak memory than np.save of a full in-memory copy.
    del buffer
    if written != len(records):
        full = np.load(array_path, mmap_mode="r")
        trimmed = np.lib.format.open_memmap(
            array_path.with_suffix(".trim.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(written, len(STANDARD_LEADS), EXPECTED_SAMPLES),
        )
        trimmed[:] = full[:written]
        del trimmed, full
        array_path.unlink()
        array_path.with_suffix(".trim.npy").rename(array_path)

    metadata = {
        "signal_cache_version": version,
        "n_candidates": len(records),
        "n_rows": written,
        "n_failures": len(failures),
        "folds": sorted(wanted),
        "test_fold_included": bool(wanted & set(TEST_FOLDS)),
        "lead_names": list(STANDARD_LEADS),
        "n_samples": EXPECTED_SAMPLES,
        "sampling_frequency": EXPECTED_FREQUENCY,
        "units": "mV",
        "checksums_verified": bool(verify_checksums),
        "records_without_header_checksum": checksum_unverified,
        "fitted_statistics": "none -- this cache stores decoded physical units only",
        "seconds": round(elapsed, 1),
        "bytes": int(array_path.stat().st_size),
    }
    index_path = signal_index_path(out_dir, version)
    np.savez_compressed(
        index_path,
        ecg_ids=np.asarray(ecg_ids, dtype=np.int64),
        patient_ids=np.asarray(patient_ids, dtype=np.int64),
        strat_folds=np.asarray(strat_folds, dtype=np.int16),
        metadata=np.asarray(json.dumps(metadata), dtype=object),
        failures=np.asarray(json.dumps(failures), dtype=object),
    )
    metadata["array_path"] = str(array_path)
    metadata["index_path"] = str(index_path)
    metadata["failures"] = failures
    return metadata


class SignalCache:
    """A memory-mapped waveform cache plus the identifiers needed to align it."""

    def __init__(
        self,
        signals: np.ndarray,
        ecg_ids: np.ndarray,
        patient_ids: np.ndarray,
        strat_folds: np.ndarray,
        metadata: Dict[str, Any],
        failures: Dict[str, str],
    ) -> None:
        if signals.shape[0] != ecg_ids.size:
            raise SignalCacheError(
                f"{signals.shape[0]} waveform rows but {ecg_ids.size} ecg ids."
            )
        if ecg_ids.size != patient_ids.size or ecg_ids.size != strat_folds.size:
            raise SignalCacheError("Identifier arrays are not the same length.")
        if np.unique(ecg_ids).size != ecg_ids.size:
            raise SignalCacheError("Duplicate ecg_id in the signal cache.")
        self.signals = signals
        self.ecg_ids = ecg_ids
        self.patient_ids = patient_ids
        self.strat_folds = strat_folds
        self.metadata = metadata
        self.failures = failures
        self._row_of: Dict[int, int] = {
            int(value): index for index, value in enumerate(ecg_ids)
        }

    @property
    def n_records(self) -> int:
        return int(self.ecg_ids.size)

    @property
    def n_leads(self) -> int:
        return int(self.signals.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.signals.shape[2])

    def rows_for(self, ecg_ids: Sequence[int]) -> np.ndarray:
        """Row indices for ``ecg_ids``, raising rather than silently dropping a miss.

        The deep arm is scored against labels that came from a *different* cache. If an
        ecg_id were missing here and quietly skipped, the two would fall out of alignment
        and every metric downstream would be computed on mismatched pairs.
        """
        missing = [int(value) for value in ecg_ids if int(value) not in self._row_of]
        if missing:
            raise SignalCacheError(
                f"{len(missing)} ecg_id(s) are not in the signal cache, "
                f"first few: {missing[:5]}."
            )
        return np.asarray([self._row_of[int(value)] for value in ecg_ids], dtype=np.int64)


def load_signal_cache(
    root: Path, version: str = SIGNAL_CACHE_VERSION, *, mmap: bool = True
) -> SignalCache:
    """Load the cache, memory-mapped by default."""
    array_path = signal_array_path(root, version)
    index_path = signal_index_path(root, version)
    for path in (array_path, index_path):
        if not path.exists():
            raise SignalCacheError(
                f"{path} not found. Build it with "
                f"`python -m backend.training.prepare_ecg_signals`."
            )
    signals = np.load(array_path, mmap_mode="r" if mmap else None)
    with np.load(index_path, allow_pickle=True) as archive:
        ecg_ids = np.asarray(archive["ecg_ids"], dtype=np.int64)
        patient_ids = np.asarray(archive["patient_ids"], dtype=np.int64)
        strat_folds = np.asarray(archive["strat_folds"], dtype=np.int16)
        metadata = json.loads(str(archive["metadata"].item()))
        failures = json.loads(str(archive["failures"].item()))
    if metadata.get("signal_cache_version") != version:
        raise SignalCacheError(
            f"Cache declares version {metadata.get('signal_cache_version')!r}, "
            f"expected {version!r}."
        )
    return SignalCache(signals, ecg_ids, patient_ids, strat_folds, metadata, failures)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the PTB-XL raw waveform cache.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("backend/artifacts/dataset"),
        help="Directory for the cache (gitignored).",
    )
    parser.add_argument("--folds", default="1-9", help="e.g. '1-9' or '1,2,3'.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-verify-checksums", action="store_true")
    parser.add_argument(
        "--allow-test-fold",
        action="store_true",
        help="Required to include fold 10. Do not use during development.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    folds: List[int] = []
    for chunk in str(args.folds).split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            low, high = chunk.split("-", 1)
            folds.extend(range(int(low), int(high) + 1))
        elif chunk:
            folds.append(int(chunk))

    report = build_signal_cache(
        root=args.root,
        out_dir=args.out,
        folds=folds,
        limit=args.limit,
        allow_test_fold=args.allow_test_fold,
        verify_checksums=not args.no_verify_checksums,
    )
    logger.info("")
    logger.info("Signal cache %s", report["signal_cache_version"])
    logger.info(
        "  %d/%d records cached, %d failed, %.1f MiB, %.1f s",
        report["n_rows"],
        report["n_candidates"],
        report["n_failures"],
        report["bytes"] / (1024 * 1024),
        report["seconds"],
    )
    if report["failures"]:
        for ecg_id, reason in list(report["failures"].items())[:10]:
            logger.info("    ecg %s: %s", ecg_id, reason)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
