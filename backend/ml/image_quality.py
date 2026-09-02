"""Image acquisition quality control.

Rejecting an unusable capture is a clinical safety requirement: inferring on a
dark, blurred, or cropped photograph produces a confident-looking number with no
basis. Every threshold is a named, documented setting in
:mod:`backend.core.config` -- none is an inline magic number.

Threshold provenance
--------------------
These are **engineering acceptability limits derived from this dataset**, not
clinically validated cut-offs. They were chosen by measuring the distribution of
each metric across the SMART-OM captures and placing the limit below the
observed range of usable images, so that ordinary clinical photographs pass and
degraded ones do not. They are deliberately permissive: quality control exists
to catch acquisition failures, not to filter difficult cases.

- ``QC_MIN_MEAN_LUMINANCE`` / ``QC_MAX_MEAN_LUMINANCE`` -- mean of the ITU-R
  BT.601 luma channel on 0-255. Bounds sit outside the observed range for the
  dataset, so they trip only on genuinely under/over-exposed frames.
- ``QC_MIN_LAPLACIAN_VARIANCE`` -- variance of the discrete Laplacian, the
  standard no-reference focus measure (Pech-Pacheco et al., ICPR 2000). Higher
  means sharper. The limit flags visibly defocused frames.
- ``QC_MAX_CLIPPED_FRACTION`` -- fraction of pixels at 0 or 255 in any channel.
  Catches flash blow-out and crushed shadows that destroy mucosal texture.
- ``QC_MIN_SHORT_EDGE_PX`` -- the feature extractor's native input edge (224).
  Below this the ROI must be upsampled, which invents detail.
- ``QC_MIN_ROI_FRACTION`` -- the lesion/region ROI must occupy at least this
  fraction of the frame, otherwise the lesion is too small or partly outside it.
- ``QC_MIN_BYTES_PER_PIXEL`` -- proxy for JPEG compression severity. Very low
  values indicate quantisation artefacts masquerading as texture.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.ml.types import (
    ImageQualityReport,
    QualityIssue,
    QualityVerdict,
)

# ITU-R BT.601 luma weights, matching PIL's "L" conversion.
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)

# 4-neighbour discrete Laplacian.
_LAPLACIAN_KERNEL = np.array(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)

# The focus measure is computed after downsampling to this short edge, so one
# threshold applies regardless of capture resolution. See :func:`focus_measure`.
FOCUS_REFERENCE_SHORT_EDGE = 512

# The lesion-truncation check only applies to ROIs smaller than this fraction of
# the frame. A region annotation covering the whole mucosal surface reaches the
# frame edge by construction -- 64.5% of dataset ROIs exceed 0.9 of the frame --
# so flagging those as "cut off" would be wrong. Truncation is only diagnosable
# when the ROI is a localised patch that happens to run into the border.
TRUNCATION_CHECK_MAX_ROI_FRACTION = 0.60

# Issues severe enough that inference must not proceed.
REJECTING_ISSUES = frozenset(
    {
        QualityIssue.UNREADABLE_IMAGE,
        QualityIssue.INSUFFICIENT_ILLUMINATION,
        QualityIssue.EXCESSIVE_ILLUMINATION,
        QualityIssue.EXCESSIVE_BLUR,
        QualityIssue.INSUFFICIENT_RESOLUTION,
    }
)

_GUIDANCE = {
    QualityIssue.INSUFFICIENT_ILLUMINATION: (
        "The photograph is too dark. Move to brighter light or switch on the flash, "
        "then retake it."
    ),
    QualityIssue.EXCESSIVE_ILLUMINATION: (
        "The photograph is washed out by glare. Reduce the light or move the flash "
        "further away, then retake it."
    ),
    QualityIssue.EXCESSIVE_BLUR: (
        "The photograph is out of focus. Hold the camera steady, wait for focus to "
        "lock, then retake it."
    ),
    QualityIssue.INSUFFICIENT_RESOLUTION: (
        "The photograph is too small to analyse. Retake it at the camera's full "
        "resolution."
    ),
    QualityIssue.INADEQUATE_ROI_SIZE: (
        "The area of interest fills too little of the frame. Move closer so the "
        "lesion is clearly visible."
    ),
    QualityIssue.INCOMPLETE_LESION_VISIBILITY: (
        "The area of interest appears cut off at the edge of the frame. Reframe so "
        "the whole area is visible."
    ),
    QualityIssue.EXCESSIVE_COMPRESSION: (
        "The photograph shows heavy compression artefacts. Retake it and send the "
        "original rather than a forwarded copy."
    ),
}


def _to_rgb_array(image) -> np.ndarray:
    """Convert a PIL image to a float64 RGB array on 0-255."""
    return np.asarray(image.convert("RGB"), dtype=np.float64)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma channel of an RGB array on 0-255."""
    return rgb @ _LUMA_WEIGHTS


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the discrete Laplacian: the standard no-reference focus measure.

    Implemented with array slicing rather than a convolution dependency so the
    quality gate has no SciPy/OpenCV requirement.
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    centre = gray[1:-1, 1:-1]
    response = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * centre
    )
    return float(np.var(response))


def _resample_to_short_edge(gray: np.ndarray, short_edge: int) -> np.ndarray:
    """Nearest-neighbour resample so the shorter edge equals ``short_edge``.

    Nearest-neighbour is deliberate: an interpolating filter smooths the image and
    would itself depress the gradient energy the focus measure is about to
    quantify. Strided sampling preserves the local contrast of the original
    pixels. Implemented with index arithmetic to keep the gate PIL-free.
    """
    height, width = gray.shape
    current = min(height, width)
    if current <= short_edge:
        return gray
    scale = short_edge / current
    rows = np.minimum((np.arange(max(1, round(height * scale))) / scale).astype(int), height - 1)
    cols = np.minimum((np.arange(max(1, round(width * scale))) / scale).astype(int), width - 1)
    return gray[np.ix_(rows, cols)]


def focus_measure(gray: np.ndarray, *, short_edge: int = FOCUS_REFERENCE_SHORT_EDGE) -> float:
    """Resolution-normalised focus measure.

    Raw Laplacian variance is not comparable across image sizes: the same scene
    photographed at 1771 px and at 280 px yields very different values, because
    per-pixel gradients shrink as resolution grows. Measured across a 400-image
    dataset sample the raw statistic spanned 3.2 to 611 with a median of 24.5,
    which makes any single raw threshold meaningless.

    Downsampling to a common short edge before measuring substantially reduces
    that dependence -- on the same sample the median rises to 125.5 and the
    distribution spreads out, making one threshold usable across capture sizes.
    It does not eliminate it: the correlation between the normalised measure and
    the source short edge is still 0.48, partly because larger-sensor captures
    genuinely are sharper. The threshold is therefore set well below the observed
    range (see :data:`~backend.core.config.Settings.QC_MIN_LAPLACIAN_VARIANCE`)
    so residual scale sensitivity cannot by itself reject a usable image.
    """
    return laplacian_variance(_resample_to_short_edge(gray, short_edge))


def clipped_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels saturated at 0 or 255 in at least one channel."""
    saturated = np.any((rgb <= 0.5) | (rgb >= 254.5), axis=2)
    return float(np.mean(saturated))


def roi_touches_border(
    box: Sequence[int], width: int, height: int, *, margin_px: int = 2
) -> bool:
    """True when the ROI runs into the frame edge, i.e. the lesion may be cut off."""
    x0, y0, x1, y1 = box
    return (
        x0 <= margin_px
        or y0 <= margin_px
        or x1 >= width - margin_px
        or y1 >= height - margin_px
    )


def assess_image_quality(
    image,
    *,
    file_size_bytes: Optional[int] = None,
    roi_box: Optional[Sequence[int]] = None,
    roi_available: bool = True,
    config: Optional[Settings] = None,
) -> ImageQualityReport:
    """Run every quality check and return a single verdict.

    Args:
        image: a PIL ``Image``. Already-open images are accepted so callers can
            avoid a second decode.
        file_size_bytes: encoded size, used for the compression proxy. When
            ``None`` the compression check is skipped rather than guessed.
        roi_box: ``(x0, y0, x1, y1)`` of the region that will be analysed. When
            ``None`` the ROI checks are skipped.
        roi_available: ``False`` when ROI detection produced nothing, which is
            itself an incomplete-visibility signal.
        config: settings override, for tests and experiments.
    """
    cfg = config or default_settings
    issues: List[QualityIssue] = []

    try:
        rgb = _to_rgb_array(image)
    except Exception:  # noqa: BLE001 - decode failure is a quality verdict, not a crash
        return ImageQualityReport(
            verdict=QualityVerdict.REJECT,
            issues=[QualityIssue.UNREADABLE_IMAGE],
            mean_luminance=0.0,
            laplacian_variance=0.0,
            clipped_fraction=0.0,
            width=0,
            height=0,
            bytes_per_pixel=0.0,
            roi_fraction=0.0,
            messages=[
                "The photograph could not be read. Please retake it."
            ],
        )

    height, width = rgb.shape[0], rgb.shape[1]
    gray = luminance(rgb)
    mean_luma = float(np.mean(gray))
    focus = focus_measure(gray)
    clipped = clipped_fraction(rgb)

    n_pixels = width * height
    bytes_per_pixel = (
        float(file_size_bytes) / n_pixels if file_size_bytes and n_pixels else 0.0
    )

    roi_fraction = 0.0
    if roi_box is not None and n_pixels:
        x0, y0, x1, y1 = roi_box
        roi_fraction = max(0.0, (x1 - x0) * (y1 - y0)) / n_pixels

    # --- illumination ---
    if mean_luma < cfg.QC_MIN_MEAN_LUMINANCE:
        issues.append(QualityIssue.INSUFFICIENT_ILLUMINATION)
    elif mean_luma > cfg.QC_MAX_MEAN_LUMINANCE:
        issues.append(QualityIssue.EXCESSIVE_ILLUMINATION)
    if clipped > cfg.QC_MAX_CLIPPED_FRACTION:
        # Blow-out and crushed shadows both destroy mucosal texture; report the
        # direction the mean points to so the guidance is actionable.
        issue = (
            QualityIssue.EXCESSIVE_ILLUMINATION
            if mean_luma >= 128.0
            else QualityIssue.INSUFFICIENT_ILLUMINATION
        )
        if issue not in issues:
            issues.append(issue)

    # --- focus ---
    if focus < cfg.QC_MIN_LAPLACIAN_VARIANCE:
        issues.append(QualityIssue.EXCESSIVE_BLUR)

    # --- resolution ---
    if min(width, height) < cfg.QC_MIN_SHORT_EDGE_PX:
        issues.append(QualityIssue.INSUFFICIENT_RESOLUTION)

    # --- compression ---
    if file_size_bytes is not None and bytes_per_pixel < cfg.QC_MIN_BYTES_PER_PIXEL:
        issues.append(QualityIssue.EXCESSIVE_COMPRESSION)

    # --- ROI adequacy and completeness ---
    if roi_box is not None:
        if roi_fraction < cfg.QC_MIN_ROI_FRACTION:
            issues.append(QualityIssue.INADEQUATE_ROI_SIZE)
        # Only a localised ROI can be diagnosed as truncated; see
        # TRUNCATION_CHECK_MAX_ROI_FRACTION.
        if roi_fraction <= TRUNCATION_CHECK_MAX_ROI_FRACTION and roi_touches_border(
            roi_box, width, height
        ):
            issues.append(QualityIssue.INCOMPLETE_LESION_VISIBILITY)
    elif not roi_available:
        issues.append(QualityIssue.INCOMPLETE_LESION_VISIBILITY)

    if not issues:
        verdict = QualityVerdict.ACCEPT
    elif any(issue in REJECTING_ISSUES for issue in issues):
        verdict = QualityVerdict.REJECT
    else:
        # Recoverable concerns: proceed but mark the result as lower confidence.
        verdict = QualityVerdict.FLAG

    return ImageQualityReport(
        verdict=verdict,
        issues=issues,
        mean_luminance=round(mean_luma, 3),
        laplacian_variance=round(focus, 3),
        clipped_fraction=round(clipped, 5),
        width=width,
        height=height,
        bytes_per_pixel=round(bytes_per_pixel, 5),
        roi_fraction=round(roi_fraction, 5),
        messages=[_GUIDANCE[i] for i in issues if i in _GUIDANCE],
    )
