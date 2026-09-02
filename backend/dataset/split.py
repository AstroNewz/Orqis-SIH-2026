"""Deterministic patient-level dataset splitting.

Medical-image leakage rule: every image belonging to one patient lands in
exactly one partition. Random image-level splitting is never used, because
SMART-OM contains up to 8 anatomical sites per subject captured in two lighting
modes -- an image-level split would put near-identical views of the same mouth
on both sides of the train/test boundary and inflate every metric.

The split is stratified on a per-patient label profile rather than a per-image
label, because 140 of the 304 SMITA subjects carry images in more than one
diagnostic class. A patient's stratum is its most severe class, so the scarce
positive subjects are spread across partitions instead of clustering.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from backend.dataset.smartom import ImageRecord

# Ascending clinical severity. A patient's stratum is its highest-severity class.
SEVERITY_ORDER: Tuple[str, ...] = (
    "normal",
    "variation_from_normal",
    "opmd",
    "oral_cancer",
)

# Default binary screening target: which classes count as "refer".
# OPMD (oral potentially malignant disorder) and oral cancer both warrant
# specialist referral, so both are positive for a screening triage model.
DEFAULT_POSITIVE_CLASSES: Tuple[str, ...] = ("opmd", "oral_cancer")


class SplitError(ValueError):
    """Raised when a split cannot satisfy the patient-level invariant."""


@dataclass
class PartitionStats:
    """Observed composition of one partition."""

    name: str
    n_patients: int
    n_images: int
    images_per_class: Dict[str, int] = field(default_factory=dict)
    patients_per_stratum: Dict[str, int] = field(default_factory=dict)
    n_positive_images: int = 0
    n_negative_images: int = 0

    @property
    def positive_rate(self) -> float:
        total = self.n_positive_images + self.n_negative_images
        return self.n_positive_images / total if total else 0.0


@dataclass
class SplitManifest:
    """Reproducible, persistable record of a patient-level split."""

    seed: int
    fractions: Dict[str, float]
    positive_classes: List[str]
    dataset_root: str
    patient_to_partition: Dict[str, str] = field(default_factory=dict)
    image_to_partition: Dict[str, str] = field(default_factory=dict)
    stats: Dict[str, PartitionStats] = field(default_factory=dict)
    excluded_images: Dict[str, str] = field(default_factory=dict)
    """``image_id -> exclusion reason``, retained so exclusions stay auditable."""

    n_source_images: int = 0
    index_fingerprint: str = ""

    # -- accessors ---------------------------------------------------------
    def partition_of_patient(self, patient_id: str) -> Optional[str]:
        return self.patient_to_partition.get(patient_id)

    def images_in(self, partition: str) -> List[str]:
        return [k for k, v in self.image_to_partition.items() if v == partition]

    def patients_in(self, partition: str) -> List[str]:
        return [k for k, v in self.patient_to_partition.items() if v == partition]

    # -- invariant ---------------------------------------------------------
    def assert_no_patient_leakage(self, records: Iterable[ImageRecord]) -> None:
        """Raise if any patient's images span more than one partition.

        This is the guard the mandatory test exercises. It re-derives the
        patient->partition mapping from the *image* assignments so a corrupted
        manifest cannot pass by virtue of a clean ``patient_to_partition`` dict.
        """
        observed: Dict[str, str] = {}
        conflicts: Dict[str, set] = {}
        for record in records:
            partition = self.image_to_partition.get(record.image_id)
            if partition is None:
                continue
            previous = observed.setdefault(record.patient_id, partition)
            if previous != partition:
                conflicts.setdefault(record.patient_id, {previous}).add(partition)
        if conflicts:
            detail = ", ".join(
                f"{pid}: {sorted(parts)}" for pid, parts in sorted(conflicts.items())
            )
            raise SplitError(f"Patient-level leakage detected across partitions -> {detail}")

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "fractions": self.fractions,
            "positive_classes": self.positive_classes,
            "dataset_root": self.dataset_root,
            "n_source_images": self.n_source_images,
            "index_fingerprint": self.index_fingerprint,
            "patient_to_partition": self.patient_to_partition,
            "image_to_partition": self.image_to_partition,
            "excluded_images": self.excluded_images,
            "stats": {k: asdict(v) for k, v in self.stats.items()},
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "SplitManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = cls(
            seed=int(payload["seed"]),
            fractions={k: float(v) for k, v in payload["fractions"].items()},
            positive_classes=list(payload["positive_classes"]),
            dataset_root=str(payload.get("dataset_root", "")),
            patient_to_partition=dict(payload["patient_to_partition"]),
            image_to_partition=dict(payload["image_to_partition"]),
            excluded_images=dict(payload.get("excluded_images", {})),
            n_source_images=int(payload.get("n_source_images", 0)),
            index_fingerprint=str(payload.get("index_fingerprint", "")),
        )
        manifest.stats = {
            name: PartitionStats(**values) for name, values in payload.get("stats", {}).items()
        }
        return manifest


def binary_label(diagnostic_class: str, positive_classes: Sequence[str]) -> int:
    """Map a diagnostic class onto the binary screening target."""
    return 1 if diagnostic_class in positive_classes else 0


def patient_stratum(classes: Iterable[str]) -> str:
    """Most severe diagnostic class observed for a patient."""
    ranks = [SEVERITY_ORDER.index(c) for c in classes if c in SEVERITY_ORDER]
    return SEVERITY_ORDER[max(ranks)] if ranks else "unknown"


def _stable_patient_order(patient_ids: Iterable[str], seed: int) -> List[str]:
    """Order patients by a seeded hash.

    Hash-based ordering (rather than ``random.shuffle``) makes the split depend
    only on ``(patient_id, seed)``. Adding or removing an unrelated patient does
    not reshuffle everyone else, so a re-run after a dataset top-up keeps prior
    assignments stable.
    """

    def key(pid: str) -> str:
        return hashlib.sha256(f"{seed}:{pid}".encode("utf-8")).hexdigest()

    return sorted(patient_ids, key=key)


def build_patient_level_split(
    records: Sequence[ImageRecord],
    *,
    seed: int = 42,
    fractions: Optional[Dict[str, float]] = None,
    positive_classes: Sequence[str] = DEFAULT_POSITIVE_CLASSES,
    dataset_root: str = "",
    excluded_images: Optional[Dict[str, str]] = None,
) -> SplitManifest:
    """Build a deterministic, stratified, patient-level train/val/test split.

    Args:
        records: canonical image records to partition.
        seed: fixed seed; identical inputs and seed always produce the identical
            manifest.
        fractions: target *patient* fractions per partition. Defaults to
            ``{"train": 0.7, "validation": 0.15, "test": 0.15}``.
        positive_classes: diagnostic classes mapped to the positive label.
        dataset_root: recorded in the manifest for provenance.
        excluded_images: ``image_id -> reason`` for samples deliberately held
            out (e.g. quality-control rejects). Recorded, never silently dropped.

    Raises:
        SplitError: if there are no usable records, or if the resulting split
            violates the patient-level invariant.
    """
    fractions = dict(fractions or {"train": 0.70, "validation": 0.15, "test": 0.15})
    total_fraction = sum(fractions.values())
    if not 0.999 <= total_fraction <= 1.001:
        raise SplitError(f"Split fractions must sum to 1.0, got {total_fraction:.4f}")

    excluded = dict(excluded_images or {})
    usable = [r for r in records if r.image_id not in excluded]
    if not usable:
        raise SplitError("No usable records to split.")

    patients: Dict[str, List[ImageRecord]] = {}
    for record in usable:
        patients.setdefault(record.patient_id, []).append(record)

    strata: Dict[str, List[str]] = {}
    for patient_id, group in patients.items():
        stratum = patient_stratum(r.diagnostic_class for r in group)
        strata.setdefault(stratum, []).append(patient_id)

    partition_names = list(fractions)
    assignment: Dict[str, str] = {}

    # Allocate each stratum independently so scarce positive subjects are spread
    # rather than concentrated. Largest-remainder allocation guarantees every
    # partition gets a whole number of patients that sums to the stratum size.
    for stratum in sorted(strata):
        ordered = _stable_patient_order(strata[stratum], seed)
        n = len(ordered)
        exact = {name: n * fractions[name] for name in partition_names}
        counts = {name: int(exact[name]) for name in partition_names}
        remainder = n - sum(counts.values())
        # Give leftovers to the partitions with the largest fractional parts;
        # ties break on the configured order so the result stays deterministic.
        by_remainder = sorted(
            partition_names,
            key=lambda name: (-(exact[name] - int(exact[name])), partition_names.index(name)),
        )
        for i in range(remainder):
            counts[by_remainder[i % len(by_remainder)]] += 1

        cursor = 0
        for name in partition_names:
            for patient_id in ordered[cursor: cursor + counts[name]]:
                assignment[patient_id] = name
            cursor += counts[name]

    manifest = SplitManifest(
        seed=seed,
        fractions=fractions,
        positive_classes=list(positive_classes),
        dataset_root=str(dataset_root),
        patient_to_partition=dict(sorted(assignment.items())),
        image_to_partition={
            r.image_id: assignment[r.patient_id]
            for r in sorted(usable, key=lambda x: x.image_id)
        },
        excluded_images=excluded,
        n_source_images=len(records),
        index_fingerprint=fingerprint_records(records),
    )
    manifest.stats = compute_partition_stats(usable, manifest, positive_classes)
    manifest.assert_no_patient_leakage(usable)
    return manifest


def compute_partition_stats(
    records: Sequence[ImageRecord],
    manifest: SplitManifest,
    positive_classes: Sequence[str],
) -> Dict[str, PartitionStats]:
    """Summarise patient/image/class composition for every partition."""
    by_partition: Dict[str, List[ImageRecord]] = {name: [] for name in manifest.fractions}
    for record in records:
        partition = manifest.image_to_partition.get(record.image_id)
        if partition is not None:
            by_partition.setdefault(partition, []).append(record)

    stats: Dict[str, PartitionStats] = {}
    for name, group in by_partition.items():
        images_per_class: Dict[str, int] = {}
        patients_per_stratum: Dict[str, int] = {}
        patient_classes: Dict[str, List[str]] = {}
        positives = negatives = 0
        for record in group:
            images_per_class[record.diagnostic_class] = (
                images_per_class.get(record.diagnostic_class, 0) + 1
            )
            patient_classes.setdefault(record.patient_id, []).append(record.diagnostic_class)
            if binary_label(record.diagnostic_class, positive_classes):
                positives += 1
            else:
                negatives += 1
        for classes in patient_classes.values():
            stratum = patient_stratum(classes)
            patients_per_stratum[stratum] = patients_per_stratum.get(stratum, 0) + 1

        stats[name] = PartitionStats(
            name=name,
            n_patients=len(patient_classes),
            n_images=len(group),
            images_per_class=dict(sorted(images_per_class.items())),
            patients_per_stratum=dict(sorted(patients_per_stratum.items())),
            n_positive_images=positives,
            n_negative_images=negatives,
        )
    return stats


def fingerprint_records(records: Sequence[ImageRecord]) -> str:
    """Content-independent fingerprint of the record set, for provenance."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.image_id):
        digest.update(record.image_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(record.patient_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(record.diagnostic_class.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:16]
