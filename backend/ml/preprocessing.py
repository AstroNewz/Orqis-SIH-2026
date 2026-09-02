"""Image normalisation: Iraw -> IROI -> Inorm.

Colour is preserved throughout. The reference methodology explicitly avoids
flattened grayscale as the primary representation, and for good reason: the
clinically informative signal in oral mucosal lesions is largely chromatic --
erythroplakia is *red*, leukoplakia is *white*, healthy mucosa is pink. A
grayscale projection maps a red patch and a white patch of equal luminance onto
the same value, destroying the distinction the model most needs.

Two normalised representations are produced from the same ROI crop:

- **sRGB**, resized to the extractor's native input size, scaled to [0, 1] and
  standardised with the extractor's own channel statistics. This is what the deep
  feature extractor consumes; using anything else would mismatch its pretraining.
- **CIELAB**, from which the handcrafted descriptors are computed. CIELAB is
  chosen over HSV because it is perceptually near-uniform: a fixed Euclidean
  distance corresponds to a roughly constant perceived colour difference, so the
  mean and spread of a* (green-red) and b* (blue-yellow) are directly
  interpretable as "how red" and "how yellow" a mucosal region is. HSV's hue is
  circular and its saturation is unstable at low value, which makes plain
  summary statistics misleading on the dark, specular-highlighted regions that
  intra-oral photography produces.

The conversion is implemented here, against the D65 white point, rather than
pulled from scikit-image, so the pipeline keeps one fewer heavyweight dependency
and the colour maths stays auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

PREPROCESSING_VERSION = "carescan-preproc-1"

# ImageNet channel statistics. Required for any torchvision-pretrained backbone:
# the network was trained on inputs standardised this way, so deviating shifts
# every activation.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)

# D65 reference white in XYZ, the standard illuminant for sRGB.
_D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

# Linear sRGB -> CIE XYZ (IEC 61966-2-1).
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

# CIE L*a*b* companding constants.
_LAB_EPSILON = 216.0 / 24389.0
_LAB_KAPPA = 24389.0 / 27.0


@dataclass(frozen=True)
class NormalisedImage:
    """The normalised representations of one ROI crop."""

    rgb01: np.ndarray
    """``(H, W, 3)`` float64 sRGB in [0, 1], resized to the model input size."""

    standardised: np.ndarray
    """``(H, W, 3)`` float64, ImageNet-standardised. Feeds the deep extractor."""

    lab: np.ndarray
    """``(H, W, 3)`` float64 CIELAB. L* in [0, 100], a*/b* roughly [-128, 127]."""

    preprocessing_version: str = PREPROCESSING_VERSION

    @property
    def size(self) -> Tuple[int, int]:
        return (self.rgb01.shape[1], self.rgb01.shape[0])


def srgb_to_linear(rgb01: np.ndarray) -> np.ndarray:
    """Undo the sRGB transfer function (gamma expansion)."""
    return np.where(rgb01 <= 0.04045, rgb01 / 12.92, ((rgb01 + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    """Convert sRGB in [0, 1] to CIELAB under the D65 illuminant.

    Args:
        rgb01: ``(..., 3)`` array of sRGB values in [0, 1].

    Returns:
        ``(..., 3)`` array of ``(L*, a*, b*)``.
    """
    linear = srgb_to_linear(np.clip(np.asarray(rgb01, dtype=np.float64), 0.0, 1.0))
    xyz = linear @ _SRGB_TO_XYZ.T
    scaled = xyz / _D65_WHITE

    # Nonlinear companding: cube root above the linear-segment threshold.
    f = np.where(
        scaled > _LAB_EPSILON,
        np.cbrt(scaled),
        (_LAB_KAPPA * scaled + 16.0) / 116.0,
    )
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack(
        [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1
    )


def resize_rgb(image, size: Tuple[int, int]):
    """Resize a PIL image with a high-quality resampling filter.

    Lanczos is used for downscaling (the common case: multi-megapixel captures
    reduced to 224px) because nearest/bilinear alias the fine mucosal texture the
    descriptors measure.
    """
    from PIL import Image

    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def normalise_crop(
    crop,
    *,
    target_size: Tuple[int, int] = (224, 224),
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> NormalisedImage:
    """Produce every normalised representation of one ROI crop.

    Args:
        crop: a PIL image, already cropped to the ROI.
        target_size: ``(width, height)`` expected by the feature extractor.
        mean, std: per-channel standardisation statistics. Default to ImageNet's.
    """
    mean = IMAGENET_MEAN if mean is None else np.asarray(mean, dtype=np.float64)
    std = IMAGENET_STD if std is None else np.asarray(std, dtype=np.float64)

    resized = resize_rgb(crop, target_size)
    rgb01 = np.asarray(resized, dtype=np.float64) / 255.0
    standardised = (rgb01 - mean) / std
    lab = rgb_to_lab(rgb01)

    return NormalisedImage(rgb01=rgb01, standardised=standardised, lab=lab)


def to_chw(standardised: np.ndarray) -> np.ndarray:
    """Reorder ``(H, W, C)`` to the ``(C, H, W)`` layout torch models expect."""
    return np.transpose(standardised, (2, 0, 1)).astype(np.float32)
