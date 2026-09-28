"""Build the PTB-XL feature matrix: one ``ecg-v1`` vector per record (Phase E4).

Reads each record's WFDB pair from the downloaded 100 Hz corpus, runs the
fold-honest extractor in :mod:`backend.ml.features_ecg`, and caches the result as
a single ``.npz`` alongside the identifiers every honest evaluation needs:
``ecg_id``, ``patient_id``, ``strat_fold``, the pre-registered binary label, and
the five-way superclass matrix. Mirrors
:mod:`backend.training.prepare_pixels` -- metadata travels inside the archive,
failures are recorded per record rather than dropped, and a stale cache is
rejected on load rather than silently reused.

Three properties are deliberate and load-bearing:

1. **Nothing is fitted here.** The matrix holds raw per-record features with
   ``NaN`` where a feature was unmeasurable. Imputation, scaling and any
   projection are TRAIN-only steps in a separate module, because a median
   computed over this whole matrix would leak validation statistics into train.
2. **The frozen test fold is refused by default.** ``--folds`` defaults to
   train+validation, and asking for fold 10 requires ``--allow-test-fold``. The
   fold-10 signals are not even downloaded during development (see
   ``docs/PHASE_E4_DATA_EXPANSION.md`` §7.2), so this is a second lock on a door
   that is already bricked up -- but the code should state the rule too.
3. **Missing labels stay missing.** ``label`` is ``-1`` for the records carrying
   no diagnostic superclass. They are *not* negatives, and ``label_known`` makes
   that explicit so no downstream caller can quietly count them as healthy.

Usage::

    python -m backend.training.prepare_ecg_features
    python -m backend.training.prepare_ecg_features --folds 1-8 --limit 500
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from backend.dataset.ptbxl import (
    SUPERCLASS_ORDER,
    TEST_FOLDS,
    TRAIN_FOLDS,
    VALIDATION_FOLDS,
    EcgRecord,
    PtbxlError,
    PtbxlIndex,
    build_index,
)
from backend.dataset.wfdb_reader import WfdbError, read_record
from backend.ml.features_ecg import (
    FEATURE_SET_VERSION,
    EcgFeatureConfig,
    extract_features,
    feature_names,
)

logger = logging.getLogger(__name__)

ECG_CACHE_VERSION = "ptbxl-ecg-1"
ECG_CACHE_FILENAME = "features_{feature_set}.npz"

LABEL_UNKNOWN = -1
"""Sentinel for a record with no diagnostic superclass. Not a negative."""

DEV_FOLDS: FrozenSet[int] = frozenset(TRAIN_FOLDS | VALIDATION_FOLDS)
"""Folds 1-9. The default, and the only folds present on a development machine."""


def cache_path(root: Path, feature_set: str = FEATURE_SET_VERSION) -> Path:
    """Where the feature cache for a given extractor version lives.

    Keyed by extractor version so a future ``ecg-v2`` cannot silently overwrite
    the matrix an existing result was computed from.
    """
    return Path(root) / ECG_CACHE_FILENAME.format(feature_set=feature_set)


def parse_folds(spec: str) -> Set[int]:
    """Parse ``"1-8"``, ``"9"``, ``"1-8,9"``, ``"train"``, ``"dev"`` etc.

    Raises:
        ValueError: on an unparseable spec or a fold outside 1-10, so a typo
            cannot quietly select a different cohort than the caller meant.
    """
    aliases = {
        "train": set(TRAIN_FOLDS),
        "validation": set(VALIDATION_FOLDS),
        "val": set(VALIDATION_FOLDS),
        "test": set(TEST_FOLDS),
        "dev": set(DEV_FOLDS),
        "all": set(range(1, 11)),
    }
    folds: Set[int] = set()
    for chunk in str(spec).split(","):
        token = chunk.strip().lower()
        if not token:
            continue
        if token in aliases:
            folds |= aliases[token]
            continue
        if "-" in token:
            low, _, high = token.partition("-")
            try:
                start, stop = int(low), int(high)
            except ValueError as exc:
                raise ValueError(f"Unparseable fold range: {chunk!r}") from exc
            if start > stop:
                raise ValueError(f"Inverted fold range: {chunk!r}")
            folds |= set(range(start, stop + 1))
            continue
        try:
            folds.add(int(token))
        except ValueError as exc:
            raise ValueError(f"Unparseable fold: {chunk!r}") from exc
    if not folds:
        raise ValueError(f"No folds selected by {spec!r}")
    outside = sorted(f for f in folds if not 1 <= f <= 10)
    if outside:
        raise ValueError(f"Folds outside PTB-XL's 1-10: {outside}")
    return folds


def _superclass_row(record: EcgRecord) -> np.ndarray:
    return np.asarray(
        [1 if name in record.superclasses else 0 for name in SUPERCLASS_ORDER],
        dtype=np.uint8,
    )


def extract_cohort(
    index: Optional[PtbxlIndex] = None,
    *,
    root: Optional[Path] = None,
    folds: Optional[Iterable[int]] = None,
    limit: Optional[int] = None,
    config: Optional[EcgFeatureConfig] = None,
    progress_every: int = 500,
    allow_test_fold: bool = False,
    verify_checksums: bool = True,
) -> Dict[str, Any]:
    """Extract ``ecg-v1`` features for every record in the selected folds.

    Args:
        folds: PTB-XL ``strat_fold`` values to include. Defaults to 1-9.
        allow_test_fold: required to include fold 10. Without it, asking for the
            frozen test partition raises rather than quietly extracting it.
        verify_checksums: passed to the WFDB reader; leave True so a corrupted
            download surfaces as a per-record failure instead of as features.

    A record whose signal files are missing or unreadable is recorded in
    ``failures`` with its reason and omitted from the matrix -- never zero-filled,
    and never silently absent from the tally.
    """
    if index is None:
        index = build_index(root)
    wanted = set(folds) if folds is not None else set(DEV_FOLDS)
    if wanted & set(TEST_FOLDS) and not allow_test_fold:
        raise PtbxlError(
            "Fold 10 is the frozen test partition. Extracting it requires "
            "allow_test_fold=True, and it may only be evaluated once, at the end "
            "of the phase (docs/PHASE_E4_DATA_EXPANSION.md §7.4)."
        )

    cfg = config or EcgFeatureConfig()
    names = feature_names(cfg)
    dimension = len(names)
    signal_root = Path(index.root)

    records = [record for record in index.records if record.strat_fold in wanted]
    records.sort(key=lambda record: record.ecg_id)
    if limit is not None:
        records = records[:limit]

    rows: List[np.ndarray] = []
    ecg_ids: List[int] = []
    patient_ids: List[int] = []
    strat_folds: List[int] = []
    labels: List[int] = []
    superclass_rows: List[np.ndarray] = []
    ages: List[float] = []
    sexes: List[int] = []
    n_beats: List[int] = []
    template_beats: List[int] = []
    quality_flags: List[str] = []
    failures: Dict[str, str] = {}

    started = time.perf_counter()
    for position, record in enumerate(records, start=1):
        if progress_every and position % progress_every == 0:
            elapsed = time.perf_counter() - started
            logger.info(
                "  %d/%d  (%.0f rec/s, %d extracted, %d failed)",
                position,
                len(records),
                position / max(elapsed, 1e-9),
                len(rows),
                len(failures),
            )

        stem = signal_root / record.filename_lr
        try:
            signal = read_record(stem, verify_checksums=verify_checksums)
            features = extract_features(signal, config=cfg)
        except (WfdbError, OSError, ValueError) as exc:
            # One unreadable record must not abort a 20,000-record run, but it
            # must be named rather than vanish from the count.
            failures[str(record.ecg_id)] = f"{type(exc).__name__}: {exc}"
            continue

        if features.dimension != dimension:
            raise PtbxlError(
                f"Extractor returned {features.dimension} features for ecg_id "
                f"{record.ecg_id}, expected {dimension}. The cache layout would "
                "be corrupt."
            )

        rows.append(features.vector)
        ecg_ids.append(int(record.ecg_id))
        patient_ids.append(int(record.patient_id))
        strat_folds.append(int(record.strat_fold))
        label = record.abnormal_label
        labels.append(LABEL_UNKNOWN if label is None else int(label))
        superclass_rows.append(_superclass_row(record))
        ages.append(float("nan") if record.age is None else float(record.age))
        sexes.append(-1 if record.sex is None else int(record.sex))
        n_beats.append(int(features.n_beats))
        template_beats.append(int(features.template_beats))
        quality_flags.append(";".join(features.quality_flags))

    matrix = (
        np.vstack(rows) if rows else np.empty((0, dimension), dtype=np.float64)
    ).astype(np.float64)
    duration = time.perf_counter() - started

    labels_array = np.asarray(labels, dtype=np.int8)
    return {
        "cache_version": ECG_CACHE_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "folds": sorted(wanted),
        "features": matrix,
        "feature_names": list(names),
        "ecg_ids": np.asarray(ecg_ids, dtype=np.int32),
        "patient_ids": np.asarray(patient_ids, dtype=np.int64),
        "strat_folds": np.asarray(strat_folds, dtype=np.int8),
        "labels": labels_array,
        "label_known": (labels_array != LABEL_UNKNOWN).astype(np.uint8),
        "superclasses": (
            np.vstack(superclass_rows)
            if superclass_rows
            else np.empty((0, len(SUPERCLASS_ORDER)), dtype=np.uint8)
        ),
        "superclass_order": list(SUPERCLASS_ORDER),
        "ages": np.asarray(ages, dtype=np.float64),
        "sexes": np.asarray(sexes, dtype=np.int8),
        "n_beats": np.asarray(n_beats, dtype=np.int32),
        "template_beats": np.asarray(template_beats, dtype=np.int32),
        "quality_flags": quality_flags,
        "failures": failures,
        "n_candidates": len(records),
        "duration_seconds": duration,
    }


_METADATA_KEYS = (
    "cache_version",
    "feature_set_version",
    "folds",
    "feature_names",
    "superclass_order",
    "failures",
    "n_candidates",
    "duration_seconds",
)

_ARRAY_KEYS = (
    "features",
    "ecg_ids",
    "patient_ids",
    "strat_folds",
    "labels",
    "label_known",
    "superclasses",
    "ages",
    "sexes",
    "n_beats",
    "template_beats",
)


def save_cohort(payload: Dict[str, Any], path: Path) -> Path:
    """Persist the feature matrix. Metadata travels inside the archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = int(payload["features"].shape[0])
    for key in _ARRAY_KEYS:
        array = np.asarray(payload[key])
        if array.shape[0] != n:
            raise ValueError(
                f"Column {key!r} has {array.shape[0]} rows for {n} records; a "
                "misaligned cache would silently mismatch features to labels."
            )
    if len(payload["quality_flags"]) != n:
        raise ValueError("quality_flags is misaligned with the feature matrix.")

    metadata = {key: payload[key] for key in _METADATA_KEYS}
    np.savez_compressed(
        path,
        quality_flags=np.asarray(payload["quality_flags"], dtype=object),
        metadata=json.dumps(metadata),
        **{key: np.asarray(payload[key]) for key in _ARRAY_KEYS},
    )
    return path


@dataclass(frozen=True)
class EcgCohort:
    """A loaded feature matrix with the identifiers evaluation needs."""

    features: np.ndarray
    feature_names: Tuple[str, ...]
    ecg_ids: np.ndarray
    patient_ids: np.ndarray
    strat_folds: np.ndarray
    labels: np.ndarray
    label_known: np.ndarray
    superclasses: np.ndarray
    superclass_order: Tuple[str, ...]
    ages: np.ndarray
    sexes: np.ndarray
    n_beats: np.ndarray
    template_beats: np.ndarray
    quality_flags: Tuple[str, ...]
    metadata: Dict[str, Any]

    @property
    def n_records(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_patients(self) -> int:
        return int(np.unique(self.patient_ids).size)

    @property
    def dimension(self) -> int:
        return int(self.features.shape[1])

    def fold_mask(self, folds: Iterable[int]) -> np.ndarray:
        return np.isin(self.strat_folds, np.asarray(sorted(set(folds)), dtype=np.int8))

    def labeled_mask(self) -> np.ndarray:
        """Rows usable by the pre-registered binary task."""
        return self.label_known.astype(bool)


def load_cohort(path: Path) -> EcgCohort:
    """Read a feature cache and check it describes the extractor in use now.

    The version and width checks live here rather than only in a trainer because
    a stale cache is exactly how a run would end up reporting ``ecg-v1`` numbers
    computed by a different extractor.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No ECG feature cache at {path}. Build one with "
            "`python -m backend.training.prepare_ecg_features`."
        )
    with np.load(path, allow_pickle=True) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("feature_set_version") != FEATURE_SET_VERSION:
            raise ValueError(
                f"{path} was built by feature set "
                f"{metadata.get('feature_set_version')!r}, but this code is "
                f"{FEATURE_SET_VERSION!r}. Rebuild the cache."
            )
        names = tuple(metadata["feature_names"])
        expected = feature_names()
        if names != expected:
            raise ValueError(
                f"{path} carries a different feature layout ({len(names)} names) "
                f"than the extractor emits now ({len(expected)}). Rebuild the cache."
            )
        features = np.asarray(archive["features"], dtype=np.float64)
        if features.shape[1] != len(expected):
            raise ValueError(
                f"{path} holds {features.shape[1]}-dimensional features, expected "
                f"{len(expected)}."
            )
        return EcgCohort(
            features=features,
            feature_names=names,
            ecg_ids=np.asarray(archive["ecg_ids"]),
            patient_ids=np.asarray(archive["patient_ids"]),
            strat_folds=np.asarray(archive["strat_folds"]),
            labels=np.asarray(archive["labels"]),
            label_known=np.asarray(archive["label_known"]),
            superclasses=np.asarray(archive["superclasses"]),
            superclass_order=tuple(metadata["superclass_order"]),
            ages=np.asarray(archive["ages"]),
            sexes=np.asarray(archive["sexes"]),
            n_beats=np.asarray(archive["n_beats"]),
            template_beats=np.asarray(archive["template_beats"]),
            quality_flags=tuple(str(flag) for flag in archive["quality_flags"]),
            metadata=metadata,
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _summarise(payload: Dict[str, Any]) -> str:
    features = payload["features"]
    labels = payload["labels"]
    folds = np.asarray(payload["strat_folds"])
    patients = np.unique(payload["patient_ids"])
    non_finite = int(np.count_nonzero(~np.isfinite(features)))
    flagged = sum(1 for flag in payload["quality_flags"] if flag)

    lines = [
        "PTB-XL ecg-v1 feature extraction",
        "================================",
        f"folds ............... {payload['folds']}",
        f"candidates .......... {payload['n_candidates']}",
        f"extracted ........... {features.shape[0]} records, {patients.size} patients",
        f"failed .............. {len(payload['failures'])}",
        f"dimension ........... {features.shape[1]}",
        f"non-finite cells .... {non_finite} of {features.size}",
        f"quality-flagged ..... {flagged}",
        f"duration ............ {payload['duration_seconds']:.1f}s",
        "",
        "pre-registered binary label (docs section 7.4):",
        f"  abnormal (1) ...... {int(np.count_nonzero(labels == 1))}",
        f"  normal (0) ........ {int(np.count_nonzero(labels == 0))}",
        f"  unknown (-1) ...... {int(np.count_nonzero(labels == LABEL_UNKNOWN))}",
        "",
        "records per fold:",
    ]
    for fold in sorted(set(folds.tolist())):
        mask = folds == fold
        n_pat = np.unique(np.asarray(payload["patient_ids"])[mask]).size
        lines.append(f"  fold {fold:<2} records={int(mask.sum()):<6} patients={n_pat}")
    if payload["failures"]:
        lines.append("")
        lines.append(f"failures: {len(payload['failures'])} (first 5 shown)")
        for ecg_id, reason in list(payload["failures"].items())[:5]:
            lines.append(f"  - ecg_id {ecg_id}: {reason}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the PTB-XL ecg-v1 feature cache")
    parser.add_argument("--root", type=Path, default=None, help="PTB-XL dataset root")
    parser.add_argument(
        "--folds",
        default="dev",
        help="strat_folds to extract: 'dev' (1-9, default), 'train', 'validation', "
        "'1-8', '1-8,9', ...",
    )
    parser.add_argument("--limit", type=int, default=None, help="First N records only")
    parser.add_argument("--out", type=Path, default=None, help="Output .npz path")
    parser.add_argument(
        "--allow-test-fold",
        action="store_true",
        help="Permit fold 10, the frozen test partition. Only for the single "
        "final evaluation of the phase.",
    )
    parser.add_argument(
        "--no-verify-checksums",
        action="store_true",
        help="Skip the WFDB per-signal checksum check (not recommended)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    folds = parse_folds(args.folds)
    if folds & set(TEST_FOLDS) and args.allow_test_fold:
        print("!" * 72)
        print("FROZEN TEST FOLD REQUESTED. This may happen once per phase, at the end.")
        print("!" * 72)

    index = build_index(args.root)
    payload = extract_cohort(
        index,
        folds=folds,
        limit=args.limit,
        allow_test_fold=args.allow_test_fold,
        verify_checksums=not args.no_verify_checksums,
    )
    out = args.out or cache_path(index.root)
    save_cohort(payload, out)
    print(_summarise(payload))
    print(f"\nFeature cache written to {out}")
    return 0 if payload["features"].shape[0] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
