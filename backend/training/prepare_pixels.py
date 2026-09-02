"""Cache the V1 quantum input: one 256x256 uint8 grayscale ROI per image.

The V1 quantum path needs 65,536 pixels per image on every optimiser iteration.
Decoding 2,436 JPEGs and re-cropping them each time would dominate a training run
that now takes about ten minutes, so the localised grayscale ROIs are cached once.

**Raw pixels are cached, not amplitudes.** Three reasons, and they are the same
reasons the architecture keeps both representations:

1. The 0-255 representation is the traceable one. A cache of amplitudes could not
   be inspected as an image.
2. uint8 is eight times smaller than float64: 160 MB against 1.3 GB.
3. Normalisation is exact and costs microseconds, so deriving it at load time
   cannot drift from deriving it at inference time -- there is only one
   implementation, in :mod:`backend.ml.pixel_pipeline`.

Quality control and ROI selection are taken from the existing pipeline rather than
reimplemented, so this cache and the 163-value descriptor cache agree about which
images are usable and where the region of interest is. Two paths that disagreed
about that would not be comparable, and comparing them is the point of the
evaluation.

The ROI here is the **oracle** ROI -- derived from the annotator's polygons where
they exist, falling back to the documented centre crop. That is evaluation
condition A, the upper bound. Condition B needs the *predicted* ROI from the
trained localiser and therefore a separate cache, keyed by the localiser version:
see ``--roi predicted``. Mixing the two would report an oracle number as a
deployable one.

Usage::

    python -m backend.training.prepare_pixels
    python -m backend.training.prepare_pixels --limit 200
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from backend.core.config import Settings, settings as default_settings
from backend.dataset.smartom import ImageRecord, SmartOmIndex, build_index
from backend.ml.artifacts import ArtifactStore
from backend.ml.image_quality import assess_image_quality
from backend.ml.pixel_pipeline import (
    V1_PIXEL_COUNT,
    V1_PIXEL_PREPROCESSING_VERSION,
    V1_QUBIT_COUNT,
    V1_RESAMPLE_NAME,
    V1_ROI_EDGE_PX,
    PixelPreprocessingError,
    grayscale_roi,
)
from backend.ml.roi import extract_roi
from backend.ml.types import QualityVerdict

logger = logging.getLogger(__name__)

PIXEL_CACHE_VERSION = "v1-pixels-1"
PIXEL_CACHE_FILENAME = "v1_pixels_{roi}.npz"

RoiMode = str  # "oracle" | "predicted"


def pixel_cache_path(store: ArtifactStore, roi_mode: RoiMode = "oracle") -> Path:
    """Where the cache for a given ROI mode lives.

    Keyed by ROI mode because oracle and predicted ROIs produce different pixels
    for the same image, and a single filename would let one silently overwrite the
    other.
    """
    return store.dataset_dir / PIXEL_CACHE_FILENAME.format(roi=roi_mode)


def extract_pixels(
    index: SmartOmIndex,
    *,
    config: Optional[Settings] = None,
    limit: Optional[int] = None,
    classes: Optional[Sequence[str]] = None,
    progress_every: int = 200,
) -> Dict[str, Any]:
    """Localise, grayscale and resize every record to a 256x256 uint8 ROI.

    Rejections and failures are recorded per image with their reasons rather than
    dropped, so the count in the summary always accounts for every candidate.
    """
    cfg = config or default_settings

    records: List[ImageRecord] = list(index.records)
    if classes:
        wanted = set(classes)
        records = [record for record in records if record.diagnostic_class in wanted]
    if limit is not None:
        records = records[:limit]

    image_ids: List[str] = []
    rows: List[np.ndarray] = []
    roi_boxes: List[List[int]] = []
    roi_sources: List[str] = []
    source_sizes: List[List[int]] = []
    rejections: Dict[str, List[str]] = {}
    failures: Dict[str, str] = {}

    started = time.perf_counter()
    for position, record in enumerate(records, start=1):
        if progress_every and position % progress_every == 0:
            elapsed = time.perf_counter() - started
            logger.info(
                "  %d/%d  (%.1f img/s, %d accepted, %d rejected, %d failed)",
                position,
                len(records),
                position / max(elapsed, 1e-9),
                len(rows),
                len(rejections),
                len(failures),
            )

        path = Path(record.path)
        try:
            file_size = path.stat().st_size
            with Image.open(path) as handle:
                handle.load()
                width, height = handle.size
                roi = extract_roi(
                    width,
                    height,
                    lesion_polygons=record.lesion_polygons,
                    region_polygons=record.region_polygons,
                )
                quality = assess_image_quality(
                    handle,
                    file_size_bytes=file_size,
                    roi_box=roi.box,
                    roi_available=bool(record.lesion_polygons or record.region_polygons),
                    config=cfg,
                )
                if quality.verdict is QualityVerdict.REJECT:
                    rejections[record.image_id] = [
                        str(getattr(issue, "value", issue)) for issue in quality.issues
                    ]
                    continue
                raw = grayscale_roi(handle, roi)
        except (OSError, ValueError, PixelPreprocessingError) as exc:
            # One unreadable file must not abort a 40-minute run, but it must be
            # named in the summary rather than silently missing from the cache.
            failures[record.image_id] = f"{type(exc).__name__}: {exc}"
            continue

        rows.append(raw.ravel(order="C"))
        image_ids.append(record.image_id)
        roi_boxes.append(list(roi.box))
        roi_sources.append(str(getattr(roi.source, "value", roi.source)))
        source_sizes.append([width, height])

    matrix = (
        np.vstack(rows)
        if rows
        else np.empty((0, V1_PIXEL_COUNT), dtype=np.uint8)
    ).astype(np.uint8)
    duration = time.perf_counter() - started

    return {
        "cache_version": PIXEL_CACHE_VERSION,
        "preprocessing_version": V1_PIXEL_PREPROCESSING_VERSION,
        "roi_mode": "oracle",
        "qubit_count": V1_QUBIT_COUNT,
        "roi_edge_px": V1_ROI_EDGE_PX,
        "resample_filter": V1_RESAMPLE_NAME,
        "raw": matrix,
        "image_ids": image_ids,
        "roi_boxes": np.asarray(roi_boxes, dtype=np.int32).reshape(-1, 4),
        "roi_sources": roi_sources,
        "source_sizes": np.asarray(source_sizes, dtype=np.int32).reshape(-1, 2),
        "rejections": rejections,
        "failures": failures,
        "n_candidates": len(records),
        "duration_seconds": duration,
    }


def save_pixel_cache(payload: Dict[str, Any], path: Path) -> Path:
    """Persist the cache. Metadata travels inside the archive, not beside it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        key: payload[key]
        for key in (
            "cache_version",
            "preprocessing_version",
            "roi_mode",
            "qubit_count",
            "roi_edge_px",
            "resample_filter",
            "rejections",
            "failures",
            "n_candidates",
            "duration_seconds",
        )
    }
    np.savez_compressed(
        path,
        raw=payload["raw"],
        image_ids=np.asarray(payload["image_ids"], dtype=object),
        roi_boxes=payload["roi_boxes"],
        roi_sources=np.asarray(payload["roi_sources"], dtype=object),
        source_sizes=payload["source_sizes"],
        metadata=json.dumps(metadata),
    )
    return path


def load_pixel_cache(path: Path) -> Dict[str, Any]:
    """Read a pixel cache and check it describes the register we are about to use.

    The width check is here rather than only in the trainer because a stale cache
    is exactly how a run would end up claiming 16 qubits while feeding a circuit
    something else.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No V1 pixel cache at {path}. Build it with "
            "`python -m backend.training.prepare_pixels`."
        )
    with np.load(path, allow_pickle=True) as handle:
        metadata = json.loads(str(handle["metadata"]))
        raw = np.asarray(handle["raw"], dtype=np.uint8)
        payload = {
            "raw": raw,
            "image_ids": [str(v) for v in handle["image_ids"]],
            "roi_boxes": np.asarray(handle["roi_boxes"], dtype=np.int32),
            "roi_sources": [str(v) for v in handle["roi_sources"]],
            "source_sizes": np.asarray(handle["source_sizes"], dtype=np.int32),
        }
    payload.update(metadata)

    if raw.shape[0] != len(payload["image_ids"]):
        raise ValueError(
            f"Pixel cache is inconsistent: {raw.shape[0]} rows for "
            f"{len(payload['image_ids'])} image ids."
        )
    if raw.size and raw.shape[1] != V1_PIXEL_COUNT:
        raise ValueError(
            f"Pixel cache holds {raw.shape[1]} pixels per image but "
            f"{V1_QUBIT_COUNT} qubits require exactly {V1_PIXEL_COUNT}. The cache "
            f"was built at a different resolution -- rebuild it rather than "
            f"padding or truncating."
        )
    if payload.get("qubit_count") != V1_QUBIT_COUNT:
        raise ValueError(
            f"Pixel cache was built for {payload.get('qubit_count')} qubits, but "
            f"V1 is fixed at {V1_QUBIT_COUNT}. Rebuild the cache."
        )
    return payload


def prepare(
    *,
    config: Optional[Settings] = None,
    limit: Optional[int] = None,
    classes: Optional[Sequence[str]] = None,
    roi_mode: RoiMode = "oracle",
) -> Dict[str, Any]:
    """Index the dataset and cache one 256x256 grayscale ROI per usable image."""
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)

    if roi_mode != "oracle":
        raise NotImplementedError(
            "Only the oracle ROI cache is available until the MobileNet localiser is "
            "trained. Build it with `--roi oracle`; the predicted-ROI cache is "
            "produced by the localiser, keyed by its model version, so that an "
            "oracle result can never be reported as a predicted one."
        )

    logger.info("Indexing dataset...")
    index = build_index(cfg.DATASET_ROOT)
    logger.info("  %d records", len(index.records))

    logger.info("Caching %dx%d grayscale ROIs...", V1_ROI_EDGE_PX, V1_ROI_EDGE_PX)
    payload = extract_pixels(index, config=cfg, limit=limit, classes=classes)
    path = pixel_cache_path(store, roi_mode)
    save_pixel_cache(payload, path)

    raw = payload["raw"]
    logger.info(
        "  %d/%d cached in %.1fs (%d QC-rejected, %d failed) -> %s",
        raw.shape[0],
        payload["n_candidates"],
        payload["duration_seconds"],
        len(payload["rejections"]),
        len(payload["failures"]),
        path,
    )

    summary: Dict[str, Any] = {
        "pixel_cache": str(path),
        "cache_version": PIXEL_CACHE_VERSION,
        "roi_mode": roi_mode,
        "qubit_count": V1_QUBIT_COUNT,
        "pixels_per_image": V1_PIXEL_COUNT,
        "n_images": int(raw.shape[0]),
        "n_candidates": payload["n_candidates"],
        "n_quality_rejected": len(payload["rejections"]),
        "n_failed": len(payload["failures"]),
        "duration_seconds": round(float(payload["duration_seconds"]), 1),
        "cache_bytes": path.stat().st_size,
        "roi_source_counts": {
            source: payload["roi_sources"].count(source)
            for source in sorted(set(payload["roi_sources"]))
        },
    }
    if raw.size:
        summary["raw_grayscale"] = {
            "min": int(raw.min()),
            "max": int(raw.max()),
            "mean": round(float(raw.mean()), 3),
        }
        # An all-black ROI would raise at encoding time; naming it here means the
        # cache reports the problem instead of the trainer discovering it later.
        all_zero = [
            payload["image_ids"][i]
            for i in np.flatnonzero(raw.max(axis=1) == 0).tolist()
        ]
        summary["all_zero_images"] = all_zero
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cache 256x256 grayscale ROIs for the 16-qubit V1 quantum path."
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the image count.")
    parser.add_argument(
        "--classes", nargs="*", default=None, help="Restrict to these classes."
    )
    parser.add_argument(
        "--roi",
        default="oracle",
        choices=("oracle", "predicted"),
        help="oracle = annotation-derived (evaluation condition A); "
        "predicted = from the trained localiser (condition B).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(json.dumps(prepare(limit=args.limit, classes=args.classes, roi_mode=args.roi), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
