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


# ------------------------------------------------------------- ROI conditions
# The evaluation conditions the V1 experiment design requires, as explicit named
# row selections rather than as a convention each caller reimplements. They are
# kept apart on purpose: an average over A and B together would describe neither
# the annotation-given upper bound nor the deployable path.
#
# A is subdivided because the oracle cache's ROI is whichever annotation exists,
# and the three annotation situations are not one condition. Measured over the
# whole cache: 316 lesion-polygon crops (mean 44% of the frame), 1,510
# region-polygon crops (mean 97%), 610 centre crops (no annotation at all). The
# first group is almost entirely pathological and the second almost entirely
# normal, so ``A_all`` carries the annotator's own label decision in the crop
# geometry. Reporting the subdivisions is what makes that checkable.
CONDITION_ORACLE_ALL = "A_all"
CONDITION_ORACLE_LESION = "A_lesion_polygon"
CONDITION_ORACLE_REGION = "A_region_polygon"
CONDITION_ORACLE_CENTER = "A_center_crop"
CONDITION_PREDICTED_LOCALIZED = "B_localized"
CONDITION_PREDICTED_FALLBACK = "C_fallback"
CONDITION_PREDICTED_ALL = "B_and_C_all"

CONDITION_DESCRIPTIONS: Dict[str, str] = {
    CONDITION_ORACLE_ALL: (
        "Condition A: every row of the oracle cache, whichever annotation supplied "
        "the ROI. Confounded by annotation practice -- see the subdivisions."
    ),
    CONDITION_ORACLE_LESION: (
        "Condition A restricted to annotator lesion boxes: the tightest oracle ROI "
        "and the only one that isolates the lesion."
    ),
    CONDITION_ORACLE_REGION: (
        "Condition A restricted to annotator region boxes, which cover ~97% of the "
        "frame and are almost entirely normal mucosa."
    ),
    CONDITION_ORACLE_CENTER: (
        "Condition A rows with no annotation at all, cropped by the documented "
        "centre-crop fallback. Not an oracle ROI in any meaningful sense."
    ),
    CONDITION_PREDICTED_LOCALIZED: (
        "Condition B: the frozen localiser produced a box that passed its "
        "acceptance checks. The deployable path."
    ),
    CONDITION_PREDICTED_FALLBACK: (
        "Condition C: localisation ran and was rejected, so the documented "
        "fallback crop was used. Kept identifiable, never merged into B."
    ),
    CONDITION_PREDICTED_ALL: (
        "Conditions B and C together: what an end-to-end run actually sees. "
        "Reported alongside B and C, never instead of them."
    ),
}


def condition_rows(partition: PixelPartition) -> Dict[str, np.ndarray]:
    """Named ROI-condition row selections valid for ``partition``.

    Which conditions exist depends on which cache the partition came from, so an
    oracle partition can never accidentally report a "condition B" number and a
    predicted partition can never report a "condition A" one.
    """
    if partition.roi_mode == "predicted":
        conditions = {
            CONDITION_PREDICTED_LOCALIZED: partition.localized_rows,
            CONDITION_PREDICTED_FALLBACK: partition.fallback_rows,
            CONDITION_PREDICTED_ALL: np.arange(len(partition), dtype=int),
        }
    else:
        conditions = {
            CONDITION_ORACLE_ALL: np.arange(len(partition), dtype=int),
            CONDITION_ORACLE_LESION: partition.rows_for_roi_source("lesion_polygon"),
            CONDITION_ORACLE_REGION: partition.rows_for_roi_source("region_polygon"),
            CONDITION_ORACLE_CENTER: partition.rows_for_roi_source("center_crop"),
        }
    return {name: rows for name, rows in conditions.items() if rows.size}


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
    localization_statuses: List[str] = field(default_factory=list)
    """Per-image localisation outcome, ``"oracle"`` in an oracle cache.

    Carried at partition level so evaluation condition C -- the images the localiser
    could not localise -- can be selected from a loaded partition without going back
    to the cache. Without it, a condition B number would silently average accepted
    crops together with fallbacks.
    """

    localization_confidences: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    """Localiser confidence per image; NaN throughout an oracle cache."""

    roi_boxes: np.ndarray = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.int32)
    )
    """``(n, 4)`` ``(x0, y0, x1, y1)`` of the crop each row was taken from."""

    source_sizes: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int32)
    )
    """``(n, 2)`` ``(width, height)`` of the image each ROI was cropped from.

    Carried with the box so :meth:`roi_area_fraction` can be computed. Both are
    needed to report the condition-A confound that DEC-024 records: in the oracle
    cache the *choice* of annotation drives how much of the frame the crop covers,
    and that choice correlates with the label.
    """

    roi_mode: str = "oracle"
    """Which cache this partition came from: ``"oracle"`` or ``"predicted"``.

    Recorded on the partition, not just on the dataset, because
    :func:`condition_rows` needs it to decide which evaluation conditions the rows
    can legitimately be labelled with.
    """

    def __len__(self) -> int:
        return len(self.image_ids)

    @property
    def localized_rows(self) -> np.ndarray:
        """Row indices whose ROI came from an accepted localiser box (condition B)."""
        return np.flatnonzero(
            np.asarray([s == "localized" for s in self.localization_statuses])
        )

    @property
    def fallback_rows(self) -> np.ndarray:
        """Row indices where localisation ran and failed (condition C).

        Distinct from a never-localised centre crop, which has ``roi_source ==
        "center_crop"`` and appears only in the oracle cache.
        """
        return np.flatnonzero(
            np.asarray([s == "fallback_used" for s in self.localization_statuses])
        )

    def rows_for_roi_source(self, *sources: str) -> np.ndarray:
        """Row indices whose ROI came from any of ``sources``."""
        wanted = set(sources)
        return np.flatnonzero(
            np.asarray([source in wanted for source in self.roi_sources])
        )

    @property
    def roi_area_fraction(self) -> np.ndarray:
        """Crop area as a fraction of the source frame, per row.

        The number that makes the condition-A confound visible. Every ROI is
        resized to 256x256 regardless of how much of the frame it covered, so this
        fraction is not carried in the pixels as a size -- but it is carried as a
        *scale*: a 44%-of-frame lesion box upsampled to 256x256 has different
        texture granularity than a 97%-of-frame region box downsampled to the same
        grid. Reported so a condition-A result can be checked against it rather
        than assumed to be about tissue.
        """
        if self.roi_boxes.size == 0 or self.source_sizes.size == 0:
            return np.full(len(self), np.nan)
        boxes = np.asarray(self.roi_boxes, dtype=np.float64)
        sizes = np.asarray(self.source_sizes, dtype=np.float64)
        crop = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
            boxes[:, 3] - boxes[:, 1], 0.0
        )
        frame = np.maximum(sizes[:, 0] * sizes[:, 1], 1.0)
        return crop / frame

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
        area = self.roi_area_fraction
        finite_area = area[np.isfinite(area)]
        payload: Dict[str, object] = {
            "name": self.name,
            "roi_mode": self.roi_mode,
            "n_images": len(self),
            "n_patients": self.n_patients,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "positive_rate": round(self.positive_rate, 4),
            "roi_source_counts": {
                source: self.roi_sources.count(source)
                for source in sorted(set(self.roi_sources))
            },
            "localization_status_counts": {
                status: self.localization_statuses.count(status)
                for status in sorted(set(self.localization_statuses))
            },
            "condition_counts": {
                name: {
                    "n_images": int(rows.size),
                    "n_positive": int(self.labels[rows].sum()),
                    "n_negative": int(rows.size - self.labels[rows].sum()),
                    "n_patients": len({self.patient_ids[int(r)] for r in rows}),
                }
                for name, rows in condition_rows(self).items()
            },
        }
        if finite_area.size:
            payload["roi_area_fraction_mean"] = round(float(finite_area.mean()), 4)
        return payload


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
            "localizer_version": self.cache_metadata.get("localizer_version"),
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
    statuses: List[str] = list(cache["localization_statuses"])
    confidences = np.asarray(cache["localization_confidences"], dtype=np.float64)
    boxes = np.asarray(cache["roi_boxes"], dtype=np.int32).reshape(-1, 4)
    sizes = np.asarray(cache["source_sizes"], dtype=np.int32).reshape(-1, 2)

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
            localization_statuses=[statuses[position] for position in rows],
            localization_confidences=confidences[rows],
            roi_boxes=boxes[rows],
            source_sizes=sizes[rows],
            roi_mode=str(cache.get("roi_mode", roi_mode)),
        )

    metadata = {
        key: cache[key]
        for key in (
            "cache_version",
            "preprocessing_version",
            "roi_mode",
            "localizer_version",
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
