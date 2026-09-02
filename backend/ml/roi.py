"""Region-of-interest extraction.

Priority order, most to least specific:

1. **Lesion polygon** from the VIA lesion annotation -- the annotator's own
   delineation of the pathology. Available for 318 dataset images.
2. **Region polygon** from the VIA region annotation -- the annotated mucosal
   region. Available for a further 1536 images.
3. **Centre crop** -- for the 615 images with no polygon, and for every live
   capture from the phone, which has no annotation at all. Intra-oral
   photographs are framed on the site of interest, so the central region is the
   principled default.

The fallback chain matters for honesty: an unannotated live capture is processed
by exactly the same code path as an unannotated dataset image, so training-time
and inference-time ROI semantics agree. :attr:`RoiResult.source` records which
rule fired, and that value travels all the way into the inference response.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from backend.dataset.smartom import ImageRecord, Polygon
from backend.ml.types import RoiResult, RoiSource

# Fraction of the shorter edge kept by the centre-crop fallback. 0.8 keeps the
# framed subject while discarding the lip/instrument border that typically
# occupies the outer fifth of an intra-oral photograph.
CENTER_CROP_FRACTION = 0.80

# Proportional margin added around an annotated polygon. A lesion boundary drawn
# tight to the pathology omits the peri-lesional mucosa that provides the visual
# contrast the descriptor depends on.
POLYGON_MARGIN_FRACTION = 0.12

# An ROI smaller than this on either edge cannot support a 224x224 descriptor
# without inventing detail, so it is widened toward the frame instead.
MIN_ROI_EDGE_PX = 64


def _clamp_box(
    box: Tuple[int, int, int, int], width: int, height: int
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(int(x0), width))
    y0 = max(0, min(int(y0), height))
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    if x1 <= x0:
        x0, x1 = 0, width
    if y1 <= y0:
        y0, y1 = 0, height
    return (x0, y0, x1, y1)


def _expand_box(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    margin_fraction: float = POLYGON_MARGIN_FRACTION,
    min_edge: int = MIN_ROI_EDGE_PX,
) -> Tuple[int, int, int, int]:
    """Add a proportional margin, then widen until both edges clear ``min_edge``."""
    x0, y0, x1, y1 = box
    margin_x = (x1 - x0) * margin_fraction
    margin_y = (y1 - y0) * margin_fraction
    x0, y0, x1, y1 = x0 - margin_x, y0 - margin_y, x1 + margin_x, y1 + margin_y

    # Grow symmetrically about the centre until the minimum edge is met.
    for lo, hi, limit, axis in ((x0, x1, width, "x"), (y0, y1, height, "y")):
        if hi - lo >= min_edge:
            continue
        centre = (lo + hi) / 2.0
        half = min(min_edge, limit) / 2.0
        lo, hi = centre - half, centre + half
        # Slide back inside the frame rather than clipping, so the edge survives.
        if lo < 0:
            lo, hi = 0.0, min(float(limit), hi - lo)
        if hi > limit:
            lo, hi = max(0.0, lo - (hi - limit)), float(limit)
        if axis == "x":
            x0, x1 = lo, hi
        else:
            y0, y1 = lo, hi

    return _clamp_box((round(x0), round(y0), round(x1), round(y1)), width, height)


def polygons_bounding_box(
    polygons: Sequence[Polygon],
) -> Optional[Tuple[int, int, int, int]]:
    """Union bounding box over a set of polygons, or ``None`` if there are none.

    Degenerate outlines (no vertices) are skipped rather than raising: a VIA
    project can contain an empty region, and one bad shape must not lose the ROI
    for the whole image.
    """
    boxes = [p.bounding_box() for p in polygons if p.xs and p.ys]
    if not boxes:
        return None
    return (
        int(min(b[0] for b in boxes)),
        int(min(b[1] for b in boxes)),
        int(max(b[2] for b in boxes)),
        int(max(b[3] for b in boxes)),
    )


def center_crop_box(
    width: int, height: int, *, fraction: float = CENTER_CROP_FRACTION
) -> Tuple[int, int, int, int]:
    """Centred box covering ``fraction`` of each edge."""
    fraction = max(0.05, min(1.0, fraction))
    crop_w = max(1, int(round(width * fraction)))
    crop_h = max(1, int(round(height * fraction)))
    x0 = (width - crop_w) // 2
    y0 = (height - crop_h) // 2
    return (x0, y0, x0 + crop_w, y0 + crop_h)


def extract_roi(
    width: int,
    height: int,
    *,
    lesion_polygons: Sequence[Polygon] = (),
    region_polygons: Sequence[Polygon] = (),
    center_crop_fraction: float = CENTER_CROP_FRACTION,
) -> RoiResult:
    """Select the ROI for one image using the documented priority chain."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    box = polygons_bounding_box(lesion_polygons)
    if box is not None:
        source = RoiSource.LESION_POLYGON
    else:
        box = polygons_bounding_box(region_polygons)
        source = RoiSource.REGION_POLYGON if box is not None else RoiSource.CENTER_CROP

    if box is None:
        box = center_crop_box(width, height, fraction=center_crop_fraction)
        box = _clamp_box(box, width, height)
    else:
        box = _expand_box(box, width, height)

    x0, y0, x1, y1 = box
    return RoiResult(
        source=source,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        source_width=width,
        source_height=height,
    )


def extract_roi_for_record(
    record: ImageRecord, width: int, height: int, **kwargs
) -> RoiResult:
    """ROI for an indexed dataset record, using its annotations when present."""
    return extract_roi(
        width,
        height,
        lesion_polygons=record.lesion_polygons,
        region_polygons=record.region_polygons,
        **kwargs,
    )


def crop_to_roi(image, roi: RoiResult):
    """Crop a PIL image to an :class:`RoiResult`."""
    return image.crop(roi.box)
