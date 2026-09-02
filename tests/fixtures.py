"""Shared test fixtures.

The synthetic capture below exists so the end-to-end test can exercise the real
image path -- quality control, ROI extraction, normalisation, feature extraction --
without depending on the clinical dataset being present. The dataset is not committed
(PART 25), so a test that required it would be skipped on most checkouts, and PART 36
asks for an actual integration test "using the supplied dataset **or a controlled test
fixture**".

It is a *controlled fixture*, not a fake result: every number the pipeline reports for
it is computed from these pixels. It is deliberately not presented as a clinical image
and carries no label.
"""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image

# Chosen to clear the configured quality gates with margin, so the test fails on a
# pipeline regression rather than on a borderline fixture:
#   short edge 512 >= QC_MIN_SHORT_EDGE_PX (224)
#   mean luminance ~129 within [40, 235]
#   focus measure ~1.8e3 >> QC_MIN_LAPLACIAN_VARIANCE (12)
#   clipped fraction ~4e-5 << QC_MAX_CLIPPED_FRACTION (0.25)
#   bytes/pixel ~0.46 >> QC_MIN_BYTES_PER_PIXEL (0.05)
FIXTURE_SIZE: Tuple[int, int] = (640, 512)


def synthetic_capture(size: Tuple[int, int] = FIXTURE_SIZE, seed: int = 7) -> Image.Image:
    """A deterministic, QC-passing stand-in for an intra-oral photograph.

    Warm mucosa-like base with low-frequency shading, high-frequency texture (which is
    what gives it a genuine focus measure and a realistic JPEG size), and a lighter
    patch in the centre so ROI logic has a region to settle on.
    """
    width, height = size
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:height, 0:width]

    base = np.stack(
        [
            170 + 25 * np.sin(xx / 90.0),
            105 + 18 * np.cos(yy / 70.0),
            105 + 15 * np.sin((xx + yy) / 110.0),
        ],
        axis=-1,
    )
    base += rng.normal(0, 14, size=(height, width, 3))

    cy, cx = height // 2, width // 2
    radius = min(height, width) // 5
    patch = ((yy - cy) ** 2 + (xx - cx) ** 2) < radius * radius
    base[patch] += np.array([35.0, 40.0, 30.0])

    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")


def synthetic_capture_jpeg(
    size: Tuple[int, int] = FIXTURE_SIZE,
    seed: int = 7,
    quality: int = 92,
) -> bytes:
    """The same capture encoded as JPEG, for multipart upload tests."""
    buffer = io.BytesIO()
    synthetic_capture(size=size, seed=seed).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def unusable_capture() -> bytes:
    """A capture the quality gate must reject: tiny, dark, and flat.

    Used to prove rejection is a 422 with retake guidance rather than a 500 or a
    confident probability computed from an unusable photograph.
    """
    array = np.full((64, 64, 3), 8, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, "RGB").save(buffer, format="JPEG", quality=60)
    return buffer.getvalue()
