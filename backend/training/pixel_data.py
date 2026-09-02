"""Partition the V1 pixel cache along the patient-level split.

The classical counterpart is :mod:`backend.training.data`, which loads the 163-value
descriptor cache. This module is its pixel-path twin and deliberately does not share
code with it: the two consume different caches with different dtypes and different
validity conditions, and a shared loader would need a mode flag on every line.
What they *do* share is the split manifest, which is the point -- both paths train
and evaluate on exactly the same patients, so their numbers are comparable.

Amplitudes are derived on demand rather than stored. The training partition is 1,692
images, which is 111 MB as cached uint8 and 887 MB as float64 amplitudes; deriving
per minibatch keeps the working set small and, more importantly, means there is only
one normalisation implementation in the codebase -- the one in
:mod:`backend.ml.pixel_pipeline` that inference also calls.

Patient-level leakage is re-verified on every load, not only when the manifest is
written. Everything downstream trusts the partitions this returns, so the check
belongs where the trust is placed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.dataset.smartom import ImageRecord, SmartOmIndex, build_index
from backend.dataset.split import SplitManifest, binary_label
from backend.ml.artifacts import ArtifactStore
from backend.ml.pixel_pipeline import (
    V1_PIXEL_COUNT,
    V1_QUBIT_COUNT,
    amplitudes_from_grayscale_batch,
)
from backend.training.data import PARTITIONS
from backend.training.prepare_pixels import load_pixel_cache, pixel_cache_path


class PixelPartitionError(RuntimeError):
    """Raised when the pixel cache and the split manifest disagree."""


@dataclass
class PixelPartition:
    """One partition of the V1 pixel path.

    ``raw`` is the cached uint8 representation; :meth:`amplitudes` derives the
    quantum states. Both are exposed because the raw one is what a human can
    inspect and the derived one is what the circuit consumes, and the architecture
    requires that neither be substituted for the other.
    """

    name: str
    image_ids: List[str]
    patient_ids: List[str]
    labels: np.ndarray
    diagnostic_classes: List[str]
    raw: np.ndarray
    roi_sources: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.image_ids)

    @property
    def n_positive(self) -> int:
        return int(self.labels.sum())

    @property
    def n_negative(self) -> int:
        return int(len(self.labels) - self.labels.sum())

    @property
    def positive_rate(self) -> float:
        return float(self.labels.mean()) if len(self.labels) else 0.0

    @property
    def n_patients(self) -> int:
        return len(set(self.patient_ids))

    def amplitudes(self, rows: Optional[Sequence[int]] = None) -> np.ndarray:
        """``(n, 65536)`` unit-norm float64 states.

        Args:
            rows: row indices to encode. ``None`` encodes the whole partition,
                which for the training split allocates 887 MB -- pass a minibatch
                unless the full matrix is genuinely needed.
        """
        block = self.raw if rows is None else self.raw[np.asarray(rows, dtype=int)]
        return amplitudes_from_grayscale_batch(block)

    def summary(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "n_images": len(self),
            "n_patients": self.n_patients,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "positive_rate": round(self.positive_rate, 4),
            "roi_source_counts": {
                source: self.roi_sources.count(source)
                for source in sorted(set(self.roi_sources))
            },
        }


@dataclass
class LoadedPixelDataset:
    """The three V1 pixel partitions plus the provenance needed to reproduce them."""

    manifest: SplitManifest
    partitions: Dict[str, PixelPartition]
    cache_metadata: Dict[str, object]

    @property
    def qubit_count(self) -> int:
        """Always 16 for V1. Read from the cache, so a stale cache cannot lie."""
        return int(self.cache_metadata["qubit_count"])

    @property
    def train(self) -> PixelPartition:
        return self.partitions["train"]

    @property
    def validation(self) -> PixelPartition:
        return self.partitions["validation"]

    @property
    def test(self) -> PixelPartition:
        return self.partitions["test"]

    def summary(self) -> Dict[str, object]:
        return {
            "qubit_count": self.qubit_count,
            "pixels_per_image": V1_PIXEL_COUNT,
            "roi_mode": self.cache_metadata.get("roi_mode"),
            "preprocessing_version": self.cache_metadata.get("preprocessing_version"),
            "cache_version": self.cache_metadata.get("cache_version"),
            "seed": self.manifest.seed,
            "partitions": {
                name: partition.summary() for name, partition in self.partitions.items()
            },
        }


def load_pixel_partitions(
    *,
    config: Optional[Settings] = None,
    roi_mode: str = "oracle",
    index: Optional[SmartOmIndex] = None,
) -> LoadedPixelDataset:
    """Load the pixel cache and split it by the persisted patient-level manifest.

    Args:
        config: settings override.
        roi_mode: ``"oracle"`` (annotation-derived, evaluation condition A) or
            ``"predicted"`` (localiser output, condition B). They are separate
            caches so a result cannot be attributed to the wrong one.
        index: a prebuilt dataset index, to avoid re-walking the tree.

    Raises:
        PixelPartitionError: the cache and manifest were built from different
            dataset states, or a partition is unusable.
    """
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)

    cache = load_pixel_cache(pixel_cache_path(store, roi_mode))
    manifest = SplitManifest.load(store.split_manifest_path)
    dataset_index = index if index is not None else build_index(cfg.DATASET_ROOT)
    records: Dict[str, ImageRecord] = {
        record.image_id: record for record in dataset_index.records
    }

    manifest.assert_no_patient_leakage(dataset_index.records)

    image_ids: List[str] = list(cache["image_ids"])
    raw = np.asarray(cache["raw"], dtype=np.uint8)
    roi_sources: List[str] = list(cache["roi_sources"])

    missing_from_index = [i for i in image_ids if i not in records]
    if missing_from_index:
        raise PixelPartitionError(
            f"{len(missing_from_index)} cached images are absent from the dataset "
            f"index (first: {missing_from_index[0]}). The pixel cache is stale -- "
            "rerun `python -m backend.training.prepare_pixels`."
        )
    missing_from_manifest = [
        i for i in image_ids if i not in manifest.image_to_partition
    ]
    if missing_from_manifest:
        raise PixelPartitionError(
            f"{len(missing_from_manifest)} cached images are absent from the split "
            f"manifest (first: {missing_from_manifest[0]}). The pixel cache and the "
            "manifest were generated from different dataset states; rebuild both."
        )

    grouped: Dict[str, List[int]] = {name: [] for name in PARTITIONS}
    for position, image_id in enumerate(image_ids):
        partition_name = manifest.image_to_partition[image_id]
        if partition_name in grouped:
            grouped[partition_name].append(position)

    partitions: Dict[str, PixelPartition] = {}
    for name, positions in grouped.items():
        if not positions:
            raise PixelPartitionError(
                f"Partition {name!r} is empty. Regenerate the split manifest."
            )
        rows = np.asarray(positions, dtype=int)
        ids = [image_ids[position] for position in rows]
        labels = np.asarray(
            [
                binary_label(records[i].diagnostic_class, manifest.positive_classes)
                for i in ids
            ],
            dtype=int,
        )
        if labels.sum() == 0:
            raise PixelPartitionError(
                f"Partition {name!r} contains no positive cases; sensitivity and "
                "calibration are undefined for it."
            )
        partitions[name] = PixelPartition(
            name=name,
            image_ids=ids,
            patient_ids=[records[i].patient_id for i in ids],
            labels=labels,
            diagnostic_classes=[records[i].diagnostic_class for i in ids],
            raw=raw[rows],
            roi_sources=[roi_sources[position] for position in rows],
        )

    metadata = {
        key: cache[key]
        for key in (
            "cache_version",
            "preprocessing_version",
            "roi_mode",
            "qubit_count",
            "roi_edge_px",
            "resample_filter",
        )
        if key in cache
    }
    if int(metadata.get("qubit_count", -1)) != V1_QUBIT_COUNT:
        raise PixelPartitionError(
            f"Cache reports {metadata.get('qubit_count')} qubits; V1 is fixed at "
            f"{V1_QUBIT_COUNT}."
        )
    return LoadedPixelDataset(
        manifest=manifest, partitions=partitions, cache_metadata=metadata
    )
