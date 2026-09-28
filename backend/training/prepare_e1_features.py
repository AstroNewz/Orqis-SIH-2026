"""One dataset pass producing every E1.1 candidate feature family.

Phase E1.1 (``docs/PHASE_E1_RESEARCH_SPEC.md`` §2) asks whether the 181-feature
descriptor is missing anything that *measurably* matters. Answering that needs
representations the existing caches do not hold:

``hog``          gradient **orientation** histograms -- gap 2/4
``texture``      LBP at two radii + GLCM co-occurrence per CIELAB channel -- gap 2/5
``multiscale``   an 8x8 grid plus centre-versus-surround contrast -- gap 4/5/8
``mobilenet``    frozen MobileNetV3-Small embedding -- the learned-representation ceiling

The E1 spec claimed no new dataset pass would be needed. That was wrong for the
colour-channel and CNN families: `v1_pixels_*.npz` holds 256x256 **grayscale**, so
LBP/GLCM per CIELAB channel and a three-channel CNN cannot be derived from it. One
pass is therefore run, and it is one pass for all four families rather than four.

Everything upstream is reused exactly as inference uses it -- ``extract_roi``,
``crop_to_roi``, ``normalise_crop`` from ``backend.ml`` -- so a candidate's ROI is
byte-identical to the ROI the 181-feature path already saw. That is what makes the
comparison in §8 a comparison of representations and not of preprocessing. Row order
matches ``features_handcrafted_lab.npz`` and is asserted downstream.

The test partition is *not* excluded here: this is a per-image feature cache keyed by
image id, exactly like the existing ones, and the split is applied by the reader. No
label, no fitted statistic and no selection decision occurs in this module.

Usage::

    python -m backend.training.prepare_e1_features
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.dataset.smartom import ImageRecord, SmartOmIndex, build_index
from backend.ml.artifacts import ArtifactStore
from backend.ml.features_image import _cell_slices, _gradient_magnitude
from backend.ml.image_quality import assess_image_quality
from backend.ml.preprocessing import NormalisedImage, normalise_crop
from backend.ml.roi import crop_to_roi, extract_roi
from backend.ml.types import QualityVerdict

logger = logging.getLogger(__name__)

E1_CACHE_VERSION = "v1-e1-features-1"

#: HOG geometry. The spec's "8x8 cell" means an 8x8 *grid* of cells over the 224px
#: crop, i.e. 28px cells -- which lands the descriptor on exactly the 1764 dims the
#: spec quotes: 7x7 blocks x 2x2 cells x 9 orientations.
HOG_ORIENTATIONS = 9
HOG_CELL_PX = (28, 28)
HOG_CELLS_PER_BLOCK = (2, 2)

#: LBP: uniform method gives ``P + 2`` bins, so 10 per radius per channel.
LBP_POINTS = 8
LBP_RADII: Tuple[int, ...] = (1, 3)

#: GLCM quantisation. 32 levels keeps the 32x32 co-occurrence matrix populated at
#: 224x224; 256 levels would be mostly empty and the properties would be noise.
GLCM_LEVELS = 32
GLCM_DISTANCES: Tuple[int, ...] = (1, 3)
GLCM_ANGLES: Tuple[float, ...] = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)
GLCM_PROPERTIES: Tuple[str, ...] = ("contrast", "homogeneity", "energy", "correlation")

#: Multiscale grid. 8x8 halves the existing 4x4 cell edge, which is the cheapest
#: possible probe of "is the 56px cell too coarse".
FINE_GRID = 8

#: Centre box for the lesion-versus-surround contrast term, as a fraction of the crop.
CENTRE_FRACTION = 0.5

#: CIELAB channel ranges, matching ``features_image._LAB_RANGES``.
_LAB_RANGES: Tuple[Tuple[float, float], ...] = ((0.0, 100.0), (-100.0, 100.0), (-100.0, 100.0))
_CHANNELS = ("L", "a", "b")


class E1FeatureError(RuntimeError):
    """Raised when the E1 feature pass cannot produce a usable cache."""


# --------------------------------------------------------------------------- families
def hog_features(normalised: NormalisedImage) -> np.ndarray:
    """Gradient-orientation histograms on the grayscale crop.

    This is the family the 181-feature descriptor most clearly lacks: it keeps
    gradient *magnitude* means per 4x4 cell and discards direction entirely, so two
    crops whose edges run at different angles with equal energy are indistinguishable
    to it. HOG is the standard minimal fix and therefore the fair probe.
    """
    from skimage.feature import hog

    gray = np.asarray(normalised.lab[..., 0], dtype=np.float64) / 100.0
    return np.asarray(
        hog(
            gray,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_CELL_PX,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm="L2-Hys",
            feature_vector=True,
        ),
        dtype=np.float64,
    )


def texture_features(normalised: NormalisedImage) -> np.ndarray:
    """Uniform LBP at two radii plus GLCM properties, per CIELAB channel.

    LBP answers "what local pattern", GLCM answers "which grey levels co-occur at
    which offset" -- both are *joint* statistics of neighbouring pixels, and the
    existing descriptor holds only per-cell marginals. Two radii give the cheapest
    honest multiscale probe available without a full pyramid.
    """
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

    lab = np.asarray(normalised.lab, dtype=np.float64)
    values: List[float] = []
    for channel, (low, high) in enumerate(_LAB_RANGES):
        plane = lab[..., channel]
        # Both LBP and GLCM run on an integer plane. LBP thresholds neighbours against
        # the centre, so on floats a difference of 1e-12 -- pure CIELAB round-off --
        # flips a bit and the code becomes noise; skimage warns about exactly this.
        # Quantising once serves both, and keeps the two texture views commensurate.
        scaled = np.clip((plane - low) / (high - low), 0.0, 1.0)
        quantised = np.asarray(np.round(scaled * (GLCM_LEVELS - 1)), dtype=np.uint8)

        for radius in LBP_RADII:
            codes = local_binary_pattern(quantised, LBP_POINTS, radius, method="uniform")
            hist = np.bincount(
                np.asarray(codes, dtype=np.int64).ravel(), minlength=LBP_POINTS + 2
            ).astype(np.float64)[: LBP_POINTS + 2]
            total = hist.sum()
            values.extend((hist / total if total > 0 else hist).tolist())

        matrix = graycomatrix(
            quantised,
            distances=list(GLCM_DISTANCES),
            angles=list(GLCM_ANGLES),
            levels=GLCM_LEVELS,
            symmetric=True,
            normed=True,
        )
        for prop in GLCM_PROPERTIES:
            # correlation is undefined for a constant patch; skimage emits nan there.
            block = np.asarray(graycoprops(matrix, prop), dtype=np.float64)
            values.extend(np.nan_to_num(block, nan=0.0).ravel().tolist())
    return np.asarray(values, dtype=np.float64)


def multiscale_features(normalised: NormalisedImage) -> np.ndarray:
    """An 8x8 grid of moments, 8x8 gradient means, and centre-versus-surround contrast.

    Deliberately built inside the *existing* feature philosophy -- same CIELAB planes,
    same moment/gradient summaries -- changing only spatial resolution and adding an
    explicit lesion-interior-minus-surround term. That makes it the control that can
    kill the quantum motivation classically: if halving the cell edge and naming the
    surround closes the gap, the bottleneck was grid coarseness, not interaction
    structure.
    """
    lab = np.asarray(normalised.lab, dtype=np.float64)
    height, width = lab.shape[:2]
    rows = _cell_slices(height, FINE_GRID)
    cols = _cell_slices(width, FINE_GRID)

    values: List[float] = []
    for channel in range(3):
        plane = lab[..., channel]
        for row in rows:
            for col in cols:
                cell = plane[row, col]
                values.append(float(np.mean(cell)))
                values.append(float(np.std(cell)))

    magnitude = _gradient_magnitude(lab[..., 0])
    for row in rows:
        for col in cols:
            values.append(float(np.mean(magnitude[row, col])))

    # Centre-versus-surround: the descriptor's only background channel today is the
    # fixed 12% polygon margin, which is never contrasted against the interior.
    half = CENTRE_FRACTION / 2.0
    r0, r1 = int(height * (0.5 - half)), int(height * (0.5 + half))
    c0, c1 = int(width * (0.5 - half)), int(width * (0.5 + half))
    mask = np.zeros((height, width), dtype=bool)
    mask[r0:r1, c0:c1] = True
    for channel in range(3):
        plane = lab[..., channel]
        centre, surround = plane[mask], plane[~mask]
        values.append(float(np.mean(centre) - np.mean(surround)))
        values.append(float(np.std(centre) - np.std(surround)))
    centre_grad, surround_grad = magnitude[mask], magnitude[~mask]
    values.append(float(np.mean(centre_grad) - np.mean(surround_grad)))
    values.append(float(np.mean(centre_grad) / max(float(np.mean(surround_grad)), 1e-9)))
    return np.asarray(values, dtype=np.float64)


def _family_names() -> Dict[str, List[str]]:
    """Human-readable names per family, so a payload can be read without the code."""
    hog_names = [f"hog_{i:04d}" for i in range(1764)]

    texture: List[str] = []
    for channel in _CHANNELS:
        for radius in LBP_RADII:
            texture.extend(
                f"lbp_{channel}_r{radius}_b{b:02d}" for b in range(LBP_POINTS + 2)
            )
        for prop in GLCM_PROPERTIES:
            for distance in GLCM_DISTANCES:
                for angle_index in range(len(GLCM_ANGLES)):
                    texture.append(f"glcm_{channel}_{prop}_d{distance}_a{angle_index}")

    multiscale: List[str] = []
    for channel in _CHANNELS:
        for cell in range(FINE_GRID * FINE_GRID):
            multiscale.append(f"g8_{channel}_c{cell:02d}_mean")
            multiscale.append(f"g8_{channel}_c{cell:02d}_std")
    multiscale.extend(f"g8_grad_c{cell:02d}_mean" for cell in range(FINE_GRID * FINE_GRID))
    for channel in _CHANNELS:
        multiscale.append(f"centre_minus_surround_{channel}_mean")
        multiscale.append(f"centre_minus_surround_{channel}_std")
    multiscale.append("centre_minus_surround_grad_mean")
    multiscale.append("centre_over_surround_grad_ratio")

    return {
        "hog": hog_names,
        "texture": texture,
        "multiscale": multiscale,
        "mobilenet": [f"mnv3_{i:03d}" for i in range(576)],
    }


# ------------------------------------------------------------------------------ driver
def extract(
    index: SmartOmIndex,
    *,
    config: Optional[Settings] = None,
    limit: Optional[int] = None,
    progress_every: int = 200,
) -> Dict[str, Any]:
    """Compute all four E1 families for every readable, QC-passing image.

    Quality control runs with the same settings ``prepare_dataset`` uses and a
    rejected image is skipped, not recorded as a feature row. That is what keeps this
    cache row-comparable to ``features_handcrafted_lab.npz``: both accept exactly the
    images that pass QC, in index order.
    """
    from PIL import Image

    from backend.ml.features_image import extract_mobilenet
    from backend.ml.pipeline import build_pipeline

    cfg = config or default_settings
    # An unfitted pipeline only supplies the two unfitted geometry/normalisation
    # constants, so the ROI and crop here cannot drift from the descriptor cache's.
    reference = build_pipeline(config=cfg, extractor_name="handcrafted_lab")
    records: List[ImageRecord] = list(index.records)
    if limit is not None:
        records = records[:limit]

    image_ids: List[str] = []
    per_family: Dict[str, List[np.ndarray]] = {
        "hog": [],
        "texture": [],
        "multiscale": [],
        "mobilenet": [],
    }
    roi_sources: List[str] = []
    failures: Dict[str, str] = {}
    rejections: Dict[str, List[str]] = {}
    timings: Dict[str, float] = {name: 0.0 for name in per_family}

    started = time.perf_counter()
    for position, record in enumerate(records, start=1):
        if progress_every and position % progress_every == 0:
            elapsed = time.perf_counter() - started
            logger.info(
                "  %d/%d (%.1f img/s, %d ok, %d failed)",
                position,
                len(records),
                position / max(elapsed, 1e-9),
                len(image_ids),
                len(failures),
            )
        path = Path(record.path)
        try:
            file_size = path.stat().st_size
            with Image.open(path) as handle:
                image = handle.convert("RGB")
                width, height = image.size
                roi = extract_roi(
                    width,
                    height,
                    lesion_polygons=record.lesion_polygons,
                    region_polygons=record.region_polygons,
                    center_crop_fraction=reference.center_crop_fraction,
                )
                quality = assess_image_quality(
                    image,
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
                normalised = normalise_crop(
                    crop_to_roi(image, roi), target_size=reference.target_size
                )
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the pass
            failures[record.image_id] = f"{type(exc).__name__}: {exc}"
            continue

        try:
            row: Dict[str, np.ndarray] = {}
            for name, function in (
                ("hog", hog_features),
                ("texture", texture_features),
                ("multiscale", multiscale_features),
                ("mobilenet", extract_mobilenet),
            ):
                mark = time.perf_counter()
                row[name] = np.asarray(function(normalised), dtype=np.float64)
                timings[name] += time.perf_counter() - mark
        except Exception as exc:  # noqa: BLE001
            failures[record.image_id] = f"{type(exc).__name__}: {exc}"
            continue

        for name, values in row.items():
            per_family[name].append(values)
        image_ids.append(record.image_id)
        roi_sources.append(str(getattr(roi.source, "value", roi.source)))

    duration = time.perf_counter() - started
    matrices = {
        name: (np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float64))
        for name, rows in per_family.items()
    }
    names = _family_names()
    for name, matrix in matrices.items():
        expected = len(names[name])
        if matrix.size and matrix.shape[1] != expected:
            # Names are generated from the same constants the extractor uses, so a
            # mismatch means one of them drifted. Fail rather than write a cache whose
            # columns cannot be identified.
            raise E1FeatureError(
                f"Family {name!r} produced {matrix.shape[1]} columns but "
                f"{expected} names were generated. Fix the constants, do not rename."
            )

    return {
        "cache_version": E1_CACHE_VERSION,
        "image_ids": np.asarray(image_ids, dtype=object),
        "roi_sources": np.asarray(roi_sources, dtype=object),
        **{f"matrix_{name}": matrix for name, matrix in matrices.items()},
        **{
            f"names_{name}": np.asarray(values, dtype=object)
            for name, values in names.items()
        },
        "metadata": json.dumps(
            {
                "cache_version": E1_CACHE_VERSION,
                "n_candidates": len(records),
                "n_accepted": len(image_ids),
                "n_quality_rejected": len(rejections),
                "n_failed": len(failures),
                "rejections": rejections,
                "failures": failures,
                "duration_seconds": round(duration, 3),
                "seconds_per_family": {k: round(v, 3) for k, v in timings.items()},
                "dimensions": {
                    name: (0 if not matrix.size else int(matrix.shape[1]))
                    for name, matrix in matrices.items()
                },
                "hog": {
                    "orientations": HOG_ORIENTATIONS,
                    "pixels_per_cell": list(HOG_CELL_PX),
                    "cells_per_block": list(HOG_CELLS_PER_BLOCK),
                    "block_norm": "L2-Hys",
                },
                "lbp": {
                    "points": LBP_POINTS,
                    "radii": list(LBP_RADII),
                    "method": "uniform",
                    "input": f"{GLCM_LEVELS}-level quantised CIELAB plane (uint8)",
                },
                "glcm": {
                    "levels": GLCM_LEVELS,
                    "distances": list(GLCM_DISTANCES),
                    "n_angles": len(GLCM_ANGLES),
                    "properties": list(GLCM_PROPERTIES),
                },
                "multiscale": {
                    "grid": FINE_GRID,
                    "centre_fraction": CENTRE_FRACTION,
                },
                "mobilenet": {
                    "architecture": "mobilenet_v3_small",
                    "weights": "IMAGENET1K_V1",
                    "trainable": False,
                },
                "roi_note": (
                    "ROI selected by backend.ml.roi.extract_roi and cropped with "
                    "crop_to_roi, then normalise_crop -- the same code path inference "
                    "and the 181-feature cache use. No label was read."
                ),
            },
            indent=1,
        ),
    }


def e1_cache_path(store: ArtifactStore) -> Path:
    return Path(store.dataset_dir) / "features_e1.npz"


def prepare(
    *,
    config: Optional[Settings] = None,
    limit: Optional[int] = None,
    index: Optional[SmartOmIndex] = None,
) -> Path:
    cfg = config or default_settings
    store = ArtifactStore.from_settings(cfg)
    dataset_index = index if index is not None else build_index(cfg.DATASET_ROOT)
    logger.info("Extracting E1 candidate families over %d images...", len(dataset_index.records))
    payload = extract(dataset_index, config=cfg, limit=limit)
    path = e1_cache_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    logger.info(
        "Wrote %s (%d rows, dims %s)",
        path,
        len(payload["image_ids"]),
        json.loads(payload["metadata"])["dimensions"],
    )
    return path


def load_e1_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise E1FeatureError(
            f"E1 feature cache is missing at {path}. Run "
            "`python -m backend.training.prepare_e1_features` first."
        )
    with np.load(path, allow_pickle=True) as handle:
        payload = {key: handle[key] for key in handle.files}
    payload["metadata"] = json.loads(str(payload["metadata"]))
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Cap the image count.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    prepare(limit=args.limit)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
