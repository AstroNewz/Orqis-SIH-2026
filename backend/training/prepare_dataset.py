"""Extract and cache per-image descriptors for the whole dataset.

Feature extraction is the expensive part of the pipeline: decoding a JPEG,
selecting the ROI, normalising, and running the descriptor takes on the order of
0.15 s per image, and every training run, every ablation, and every baseline needs
the same numbers. Doing it once and caching to a ``.npz`` turns a several-minute
step into a sub-second load.

What is cached is the **image descriptor only** -- the stage before fusion and
reduction. Those two are *fitted* transforms that must be fitted on the training
partition alone, so caching their output would bake a leak into the cache.

Quality control runs here in non-strict mode: a rejected image is recorded with
its reason codes and carried into the split manifest as an explicit exclusion,
rather than quietly going missing. Same for a decode or extraction failure.

Usage::

    python -m backend.training.prepare_dataset
    python -m backend.training.prepare_dataset --extractor mobilenet_v3_small
    python -m backend.training.prepare_dataset --limit 200 --classes oral_cancer opmd
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
from backend.dataset.smartom import ImageRecord, SmartOmIndex, build_index, descriptor_key
from backend.dataset.split import build_patient_level_split
from backend.ml.artifacts import ArtifactStore
from backend.ml.pipeline import build_pipeline

logger = logging.getLogger(__name__)

CACHE_VERSION = "features-1"


def clinical_row_for(index: SmartOmIndex, record: ImageRecord) -> Optional[Dict[str, Any]]:
    """The patient's risk-factor answers, merged from the metadata sources.

    Returns ``None`` when the patient has no demographic row at all. The clinical
    encoder maps that to its all-unknown encoding, which is the honest
    representation -- distinct from "asked and answered no".
    """
    demographics = index.patient_metadata.get(record.patient_id)
    findings = index.clinical_findings.get(record.patient_id)
    if not demographics and not findings:
        return None
    row: Dict[str, Any] = {}
    if demographics:
        row.update(demographics)
    if findings:
        row.update(findings)
    return row


def extract_features(
    index: SmartOmIndex,
    *,
    extractor_name: str,
    config: Optional[Settings] = None,
    limit: Optional[int] = None,
    classes: Optional[Sequence[str]] = None,
    progress_every: int = 200,
) -> Dict[str, Any]:
    """Run ROI selection, quality control, normalisation and extraction per image.

    Uses :meth:`~backend.ml.pipeline.ClassicalPipeline.image_descriptor`, which is
    the same code path inference uses -- so a cached feature vector is identical
    to one computed live for the same image. Reimplementing the stages here would
    let the two drift.
    """
    cfg = config or default_settings
    # An unfitted pipeline is fine: image_descriptor covers only the unfitted
    # stages. Fusion and reduction are fitted later, on the training split alone.
    pipeline = build_pipeline(config=cfg, extractor_name=extractor_name)

    records: List[ImageRecord] = list(index.records)
    if classes:
        wanted = set(classes)
        records = [record for record in records if record.diagnostic_class in wanted]
    if limit is not None:
        records = records[:limit]

    image_ids: List[str] = []
    rows: List[np.ndarray] = []
    roi_fractions: List[float] = []
    roi_sources: List[str] = []
    rejections: Dict[str, List[str]] = {}
    failures: Dict[str, str] = {}
    feature_names: List[str] = list(pipeline.extractor.feature_names)

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
                image = handle.convert("RGB")
                quality, roi, descriptor = pipeline.image_descriptor(
                    image,
                    file_size_bytes=file_size,
                    lesion_polygons=record.lesion_polygons,
                    region_polygons=record.region_polygons,
                    config=cfg,
                )
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            failures[record.image_id] = f"{type(exc).__name__}: {exc}"
            continue

        if descriptor is None:
            rejections[record.image_id] = [
                str(getattr(issue, "value", issue)) for issue in quality.issues
            ]
            continue

        rows.append(np.asarray(descriptor.values, dtype=np.float64))
        image_ids.append(record.image_id)
        roi_fractions.append(float(roi.area_fraction))
        roi_sources.append(str(getattr(roi.source, "value", roi.source)))

    matrix = (
        np.vstack(rows) if rows else np.empty((0, len(feature_names)), dtype=np.float64)
    )
    duration = time.perf_counter() - started

    return {
        "cache_version": CACHE_VERSION,
        "extractor": extractor_name,
        "extractor_info": pipeline.extractor.info.model_dump(mode="json")
        if hasattr(pipeline.extractor.info, "model_dump")
        else str(pipeline.extractor.info),
        "preprocessing_version": pipeline.preprocessing_version,
        "matrix": matrix,
        "image_ids": image_ids,
        "feature_names": feature_names,
        "roi_fractions": roi_fractions,
        "roi_sources": roi_sources,
        "rejections": rejections,
        "failures": failures,
        "n_candidates": len(records),
        "duration_seconds": duration,
    }


_ARRAY_KEYS = ("matrix", "image_ids", "feature_names", "roi_fractions", "roi_sources")


def save_feature_cache(payload: Dict[str, Any], path: Path) -> Path:
    """Write the cache as a compressed ``.npz``.

    ``.npz`` rather than pickle: it stores raw arrays with no executable payload,
    so loading a cache copied between machines cannot run code. The scalar and
    dict fields ride along as a JSON string in a 0-d array.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = {key: value for key, value in payload.items() if key not in _ARRAY_KEYS}
    np.savez_compressed(
        path,
        matrix=np.asarray(payload["matrix"], dtype=np.float64),
        image_ids=np.asarray(payload["image_ids"], dtype=object),
        feature_names=np.asarray(payload["feature_names"], dtype=object),
        roi_fractions=np.asarray(payload["roi_fractions"], dtype=np.float64),
        roi_sources=np.asarray(payload["roi_sources"], dtype=object),
        meta=np.asarray(json.dumps(sidecar, default=str)),
    )
    return path


def load_feature_cache(path: Path) -> Dict[str, Any]:
    """Read a cache written by :func:`save_feature_cache`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No feature cache at {path}. Run "
            "`python -m backend.training.prepare_dataset` first."
        )
    with np.load(path, allow_pickle=True) as data:
        payload: Dict[str, Any] = json.loads(str(data["meta"].item()))
        payload["matrix"] = np.asarray(data["matrix"], dtype=np.float64)
        payload["image_ids"] = [str(value) for value in data["image_ids"]]
        payload["feature_names"] = [str(value) for value in data["feature_names"]]
        payload["roi_fractions"] = np.asarray(data["roi_fractions"], dtype=np.float64)
        payload["roi_sources"] = [str(value) for value in data["roi_sources"]]
    return payload


def prepare(
    *,
    extractor_name: str = "handcrafted_lab",
    config: Optional[Settings] = None,
    limit: Optional[int] = None,
    classes: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
    write_split: bool = True,
) -> Dict[str, Any]:
    """Index the dataset, cache image descriptors, and write the split manifest."""
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)

    logger.info("Indexing dataset...")
    index = build_index(cfg.DATASET_ROOT)
    logger.info(
        "  %d records, %d patients, %d warnings",
        len(index.records),
        len({record.patient_id for record in index.records}),
        len(index.warnings),
    )

    logger.info("Extracting %s descriptors...", extractor_name)
    payload = extract_features(
        index, extractor_name=extractor_name, config=cfg, limit=limit, classes=classes
    )
    cache_path = store.feature_cache_path(extractor_name)
    save_feature_cache(payload, cache_path)
    logger.info(
        "  %d/%d cached in %.1fs (%d QC-rejected, %d failed) -> %s",
        len(payload["image_ids"]),
        payload["n_candidates"],
        payload["duration_seconds"],
        len(payload["rejections"]),
        len(payload["failures"]),
        cache_path,
    )

    summary: Dict[str, Any] = {
        "feature_cache": str(cache_path),
        "extractor": extractor_name,
        "n_features": int(np.asarray(payload["matrix"]).shape[1]),
        "n_images": len(payload["image_ids"]),
        "n_candidates": payload["n_candidates"],
        "n_quality_rejected": len(payload["rejections"]),
        "n_failed": len(payload["failures"]),
        "extraction_seconds": round(float(payload["duration_seconds"]), 1),
    }

    if write_split:
        # QC rejections and decode failures become explicit manifest exclusions so
        # the split accounts for them instead of them just being absent.
        excluded: Dict[str, str] = {
            image_id: "quality_control:" + ",".join(reasons)
            for image_id, reasons in payload["rejections"].items()
        }
        excluded.update(
            {
                image_id: f"extraction_failed:{reason}"
                for image_id, reason in payload["failures"].items()
            }
        )
        manifest = build_patient_level_split(
            index.records,
            seed=seed if seed is not None else cfg.RANDOM_SEED,
            dataset_root=str(index.root),
            excluded_images=excluded,
        )
        # Checked here, not only in the test suite: a manifest that leaks patients
        # must never reach disk, because everything downstream trusts it.
        manifest.assert_no_patient_leakage(index.records)
        split_path = store.split_manifest_path
        manifest.save(split_path)
        logger.info("  split manifest -> %s", split_path)
        summary["split_manifest"] = str(split_path)
        summary["split_stats"] = {
            name: {
                "patients": stats.n_patients,
                "images": stats.n_images,
                "positive": stats.n_positive_images,
                "negative": stats.n_negative_images,
                "positive_rate": round(stats.positive_rate, 4),
            }
            for name, stats in manifest.stats.items()
        }
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cache dataset image descriptors.")
    parser.add_argument(
        "--extractor",
        default="handcrafted_lab",
        help="handcrafted_lab | mobilenet_v3_small",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the image count.")
    parser.add_argument("--classes", nargs="*", default=None, help="Restrict to these classes.")
    parser.add_argument("--seed", type=int, default=None, help="Split seed.")
    parser.add_argument(
        "--no-split", action="store_true", help="Skip writing the split manifest."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = prepare(
        extractor_name=args.extractor,
        limit=args.limit,
        classes=args.classes,
        seed=args.seed,
        write_split=not args.no_split,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
