"""V1 pixel path: localised ROI -> 65,536 grayscale pixels -> 16-qubit amplitudes.

This is the input path for the V1 quantum classifier, and it is deliberately
separate from :mod:`backend.ml.features_image`. That module produces a 163-value
handcrafted/MobileNet *descriptor* which the classical baselines and the existing
multimodal path consume; this module produces raw localised pixels. They are not
two views of one thing and must not be conflated in reporting: the descriptor path
compresses an image into engineered summary statistics, whereas the V1 quantum path
loads pixel intensities directly into amplitudes. Both remain in the repository --
the descriptor path as the classical baseline to beat, this one as V1.

The stage order is the one given in the V1 pipeline specification::

    ROI CROP -> GRAYSCALE -> 256x256 RESIZE -> PIXEL VECTOR -> /255 -> L2 NORMALISE

The specification's acceptance-test listing gives resize before grayscale instead.
The two orders are equivalent up to 8-bit rounding, because PIL's RGB->L conversion
is a fixed linear combination of the channels (ITU-R 601-2) and Lanczos resampling
is also linear, so they commute exactly in real arithmetic and differ only where
the intermediate is quantised back to uint8. That equivalence is asserted, not
assumed: see ``test_grayscale_and_resize_commute`` in
``tests/test_pixel_pipeline.py``, which measures the discrepancy on real dataset
images. Grayscale-first is implemented because it is the order the primary pipeline
states, and because resampling one channel instead of three is cheaper.

Two representations are returned and both are kept:

* :attr:`V1PixelVector.raw_grayscale` -- the 256x256 uint8 array, values 0-255,
  exactly as the resize produced them. Required for traceability: it is the last
  point in the pipeline a human can look at the input and recognise it as an image.
* :attr:`V1PixelVector.amplitudes` -- the 65,536 float64 amplitudes, L2-normalised.

The normalised vector is *derived*, never written back over the raw one. A
downstream stage that wants pixels gets pixels; a stage that wants a quantum state
gets the state.

An all-black ROI has no direction and therefore no normalised state. It raises
:class:`ZeroImageError` rather than being given a fabricated substitute -- a
uniform superposition would encode nothing about the patient while still producing
a confident-looking probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from backend.dataset.smartom import ImageRecord, Polygon
from backend.ml.roi import crop_to_roi, extract_roi
from backend.ml.types import RoiResult

# ---------------------------------------------------------------- fixed geometry
# FIXED for V1, by architectural decision, and deliberately module-level constants
# rather than settings fields: the qubit count is not a tunable. Making it
# configurable would permit exactly the failure the decision forbids -- a run that
# reports 16 qubits while executing a smaller circuit. A different width is a
# different experiment and needs a different, explicitly named configuration.
V1_QUBIT_COUNT = 16
V1_PIXEL_COUNT = 1 << V1_QUBIT_COUNT  # 65,536 amplitudes
V1_ROI_EDGE_PX = 256  # 256 * 256 == 65,536

# Lanczos, matching the 224x224 descriptor path in backend.ml.preprocessing, so the
# two paths do not disagree about what "resized" means. Lanczos rings slightly at
# hard edges and PIL clamps the overshoot back into 0-255 for an 8-bit image; that
# clamp is why the two stage orders differ at all.
V1_RESAMPLE = Image.Resampling.LANCZOS
V1_RESAMPLE_NAME = "lanczos"

V1_PIXEL_PREPROCESSING_VERSION = "carescan-v1-pixel-1"

# Tolerance on ||v||_2 == 1 for the returned amplitudes. Summing 65,536 float64
# squares accumulates round-off on the order of 1e-13; 1e-9 sits far above that and
# far below any genuine normalisation bug. Matches quantum_ml.quantum_encoder so
# a vector accepted here cannot be rejected one stage later.
NORM_TOLERANCE = 1.0e-9

if V1_ROI_EDGE_PX * V1_ROI_EDGE_PX != V1_PIXEL_COUNT:  # pragma: no cover - import guard
    raise ImportError(
        f"V1 geometry is inconsistent: a {V1_ROI_EDGE_PX}x{V1_ROI_EDGE_PX} ROI is "
        f"{V1_ROI_EDGE_PX ** 2} pixels but {V1_QUBIT_COUNT} qubits address "
        f"{V1_PIXEL_COUNT} amplitudes. Amplitude encoding requires them to be equal."
    )


class PixelPreprocessingError(ValueError):
    """Raised when an image cannot be turned into a V1 amplitude vector."""


class ZeroImageError(PixelPreprocessingError):
    """Raised when the localised ROI contains no signal at all.

    Distinct from :class:`quantum_ml.quantum_encoder.ZeroVectorError`, which guards
    the same condition one stage later. Both exist on purpose: this one names the
    *image* as the problem, which is what an operator can act on ("the capture is
    black, retake it"), and it fires before any quantum machinery is involved.
    """


@dataclass(frozen=True)
class V1PixelVector:
    """The V1 quantum input, with every intermediate representation retained.

    Frozen because these arrays are the provenance record of one inference. The
    arrays themselves are additionally marked read-only, so a downstream stage that
    tries to normalise in place fails loudly instead of corrupting the raw pixels
    that the response claims to describe.
    """

    raw_grayscale: np.ndarray
    """``(256, 256)`` uint8, values 0-255. The preserved raw representation."""

    scaled: np.ndarray
    """``(65536,)`` float64, ``raw / 255.0``. Still proportional to intensity."""

    amplitudes: np.ndarray
    """``(65536,)`` float64, ``scaled / ||scaled||_2``. The quantum state."""

    scaled_l2_norm: float
    """``||scaled||_2`` before normalisation.

    Retained because it is the one number normalisation destroys: two ROIs
    differing only in overall brightness produce identical amplitudes, so without
    this the response could not distinguish them.
    """

    roi: RoiResult
    source_size: Tuple[int, int]
    version: str = V1_PIXEL_PREPROCESSING_VERSION
    resample_filter: str = V1_RESAMPLE_NAME

    @property
    def qubit_count(self) -> int:
        return V1_QUBIT_COUNT

    @property
    def n_pixels(self) -> int:
        return int(self.amplitudes.shape[0])

    @property
    def roi_aspect_ratio(self) -> float:
        """Width/height of the ROI *before* the square resize.

        A non-square ROI is stretched to 256x256, which distorts shape. That is the
        documented behaviour of a fixed-size resize, but the distortion has to be
        visible in the provenance rather than inferred, because it is a real
        difference between two captures of the same lesion.
        """
        height = max(1, self.roi.y1 - self.roi.y0)
        return float(self.roi.x1 - self.roi.x0) / float(height)

    def describe(self) -> dict:
        """Provenance for the inference response. Values, not restated constants."""
        return {
            "preprocessing_version": self.version,
            "qubit_count": self.qubit_count,
            "n_pixels": self.n_pixels,
            "roi_edge_px": V1_ROI_EDGE_PX,
            "resample_filter": self.resample_filter,
            "roi_source": self.roi.source.value,
            "roi_box": list(self.roi.box),
            "roi_aspect_ratio": round(self.roi_aspect_ratio, 6),
            "source_width": self.source_size[0],
            "source_height": self.source_size[1],
            "raw_grayscale_min": int(self.raw_grayscale.min()),
            "raw_grayscale_max": int(self.raw_grayscale.max()),
            "raw_grayscale_mean": round(float(self.raw_grayscale.mean()), 6),
            "scaled_l2_norm": round(self.scaled_l2_norm, 9),
            "amplitude_l2_norm": round(float(np.linalg.norm(self.amplitudes)), 12),
        }


def _as_readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def grayscale_roi(
    image: Image.Image,
    roi: RoiResult,
    *,
    edge: int = V1_ROI_EDGE_PX,
) -> np.ndarray:
    """Crop to ``roi``, convert to grayscale, resize to ``edge`` x ``edge``.

    Returns a uint8 array. The conversion goes through RGB first so that every
    input mode -- palette, RGBA, CMYK, 16-bit -- reaches ``L`` by one documented
    route, instead of PIL choosing a different one per mode.
    """
    cropped = crop_to_roi(image, roi)
    if cropped.width <= 0 or cropped.height <= 0:
        raise PixelPreprocessingError(
            f"ROI {roi.box} produced an empty crop from a "
            f"{image.width}x{image.height} image."
        )
    gray = cropped.convert("RGB").convert("L")
    resized = gray.resize((edge, edge), V1_RESAMPLE)
    raw = np.asarray(resized, dtype=np.uint8)
    if raw.shape != (edge, edge):  # pragma: no cover - PIL contract
        raise PixelPreprocessingError(
            f"Resize returned shape {raw.shape}, expected {(edge, edge)}."
        )
    return raw


def amplitudes_from_grayscale(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """``(scaled, amplitudes, l2_norm)`` from a 256x256 uint8 array.

    Flattened in C order, so pixel ``(row, col)`` lands at index
    ``row * 256 + col``. The choice is arbitrary but it must be *fixed*: the same
    ordering has to be used at training and at inference or the trained weights
    address different pixels than they were fitted on.

    Raises:
        PixelPreprocessingError: wrong shape, or non-finite values.
        ZeroImageError: every pixel is zero.
    """
    raw = np.asarray(raw)
    if raw.size != V1_PIXEL_COUNT:
        raise PixelPreprocessingError(
            f"Expected {V1_PIXEL_COUNT} pixels for {V1_QUBIT_COUNT} qubits, got "
            f"{raw.size}. Refusing to pad or truncate: either would silently change "
            f"which pixel each amplitude carries."
        )

    scaled = raw.astype(np.float64).ravel(order="C") / 255.0
    if not np.all(np.isfinite(scaled)):  # pragma: no cover - uint8 cannot be NaN
        raise PixelPreprocessingError("Grayscale ROI contains non-finite values.")

    norm = float(np.linalg.norm(scaled))
    if norm <= 0.0:
        raise ZeroImageError(
            "The localised region is uniformly zero (every one of "
            f"{V1_PIXEL_COUNT} pixels is black), so it has no direction and no "
            "normalised quantum state. This is a capture or localisation failure, "
            "not something to substitute a value for: re-acquire the image."
        )

    amplitudes = scaled / norm
    residual = abs(float(np.linalg.norm(amplitudes)) - 1.0)
    if residual > NORM_TOLERANCE:  # pragma: no cover - float64 guard
        raise PixelPreprocessingError(
            f"Normalisation failed: |||v|| - 1| = {residual:.3e}, tolerance "
            f"{NORM_TOLERANCE:g}."
        )
    return scaled, amplitudes, norm


def amplitudes_from_grayscale_batch(raw: np.ndarray) -> np.ndarray:
    """Vectorised :func:`amplitudes_from_grayscale` for a whole partition.

    Returns ``(n_samples, 65536)`` float64. Training evaluates every sample on
    every optimiser iteration, so this exists to avoid a Python loop over
    thousands of 65,536-element rows -- not to be a second implementation. It
    produces bit-identical output to the per-image function, which
    ``test_batch_and_single_amplitudes_agree_exactly`` asserts; the single-image
    path remains the reference.

    Raises:
        PixelPreprocessingError: wrong row width.
        ZeroImageError: naming every all-zero row, so the caller can act on the
            specific images rather than on "a batch failed".
    """
    raw = np.atleast_2d(np.asarray(raw))
    if raw.shape[0] == 0:
        return np.empty((0, V1_PIXEL_COUNT), dtype=np.float64)
    if raw.shape[1] != V1_PIXEL_COUNT:
        raise PixelPreprocessingError(
            f"Expected {V1_PIXEL_COUNT} pixels per row for {V1_QUBIT_COUNT} qubits, "
            f"got {raw.shape[1]}. Refusing to pad or truncate."
        )

    scaled = raw.astype(np.float64) / 255.0
    # Norms computed one row at a time, deliberately. ``np.linalg.norm(X, axis=1)``
    # reduces with ``add.reduce`` while the 1-D call dispatches to a BLAS dot
    # product, and over 65,536 terms the two disagree in the last bit. That is
    # harmless numerically but it would mean training and inference operate on
    # subtly different vectors, so the per-row call is used in both places. The
    # cost is one BLAS call per image against a 65,536-element division.
    norms = np.fromiter(
        (np.linalg.norm(row) for row in scaled), dtype=np.float64, count=scaled.shape[0]
    )
    zero_rows = np.flatnonzero(norms <= 0.0)
    if zero_rows.size:
        raise ZeroImageError(
            f"{zero_rows.size} of {raw.shape[0]} rows are uniformly zero (first at "
            f"index {int(zero_rows[0])}); an all-black region has no normalised "
            f"quantum state. Fix or exclude those captures explicitly."
        )
    return scaled / norms[:, None]


def preprocess_v1_pixels(
    image: Image.Image,
    *,
    roi: Optional[RoiResult] = None,
    lesion_polygons: Sequence[Polygon] = (),
    region_polygons: Sequence[Polygon] = (),
) -> V1PixelVector:
    """Run the V1 pixel path on an open image.

    Args:
        image: the capture, any PIL mode.
        roi: a localisation result to use as-is -- from the MobileNet localiser at
            inference time, or from an annotation for the oracle-ROI evaluation
            path. When ``None`` the ROI is derived from the polygons given, falling
            back to the documented centre crop, so a live capture with no
            annotation follows the same code path as an unannotated dataset image.
        lesion_polygons: annotator lesion outlines, when available.
        region_polygons: annotator region outlines, when available.

    Raises:
        ZeroImageError: the localised region is entirely black.
        PixelPreprocessingError: any other shape or value failure.
    """
    if roi is None:
        roi = extract_roi(
            image.width,
            image.height,
            lesion_polygons=lesion_polygons,
            region_polygons=region_polygons,
        )

    raw = grayscale_roi(image, roi)
    scaled, amplitudes, norm = amplitudes_from_grayscale(raw)
    return V1PixelVector(
        raw_grayscale=_as_readonly(raw),
        scaled=_as_readonly(scaled),
        amplitudes=_as_readonly(amplitudes),
        scaled_l2_norm=norm,
        roi=roi,
        source_size=(int(image.width), int(image.height)),
    )


def preprocess_v1_pixels_from_path(
    path: Path, *, roi: Optional[RoiResult] = None, **kwargs
) -> V1PixelVector:
    """Open a file and run the V1 pixel path over it."""
    path = Path(path)
    try:
        with Image.open(path) as image:
            image.load()
            return preprocess_v1_pixels(image, roi=roi, **kwargs)
    except (OSError, ValueError) as exc:
        if isinstance(exc, PixelPreprocessingError):
            raise
        raise PixelPreprocessingError(f"Could not read image {path.name!r}: {exc}") from exc


def preprocess_v1_pixels_for_record(
    record: ImageRecord, *, roi: Optional[RoiResult] = None
) -> V1PixelVector:
    """Oracle-ROI path for an indexed dataset record.

    Uses the annotator's own outlines, which is the *upper bound* evaluation path
    (condition A). It is not what inference does -- inference has no annotation and
    must use the predicted ROI (condition B). Keeping the two behind different
    function names is what stops an oracle number being reported as a predicted
    one.
    """
    return preprocess_v1_pixels_from_path(
        record.path,
        roi=roi,
        lesion_polygons=record.lesion_polygons,
        region_polygons=record.region_polygons,
    )
