"""Classical image feature extraction: Inorm -> xclassical.

Two interchangeable extractors, selected by name and recorded in the artifact
metadata so every inference is traceable to the exact representation it used.

``handcrafted_lab`` (default)
    Colour and texture descriptors computed from the CIELAB representation. No
    torch dependency, no weight download, bit-identical on every machine. This
    is the default precisely because it makes the whole pipeline runnable and
    testable offline -- a hard requirement for the test suite and for CI.

``mobilenet_v3_small``
    torchvision's MobileNetV3-Small, ImageNet-pretrained, fully frozen, used as
    a fixed 576-dimensional descriptor. Selected via ``IMAGE_EXTRACTOR`` /
    the ``extractor`` argument.

Pretrained-model due diligence (PART 6)
---------------------------------------
Checked before adopting MobileNetV3-Small, not assumed:

- **Dependency**: needs ``torch`` + ``torchvision``. Verified installed as
  torchvision 0.28.0+cpu. Both are optional imports -- the module loads and the
  handcrafted extractor works without them.
- **Availability**: ``MobileNet_V3_Small_Weights.IMAGENET1K_V1`` downloads on
  first use (9.83 MB) and caches under ``~/.cache/torch/hub/checkpoints``.
  Verified downloading and loading successfully.
- **Input dimensions**: 224x224x3, ImageNet-standardised. ``normalise_crop``
  applies exactly that; deviating would mismatch the pretraining distribution.
- **Licensing**: torchvision is BSD-3-Clause and its ImageNet weights are
  released under the same licence, so redistribution inside this project is
  permitted. The weights are *not* committed -- they are fetched on demand.
- **Reproducibility**: frozen, ``eval()`` mode, ``torch.no_grad()``, CPU,
  single-sample forward pass. No dropout, no batch-norm update, no
  nondeterministic kernel. Repeated calls on the same input are bit-identical.
- **Offline behaviour**: the first call needs network access. If the download
  fails, :class:`ImageFeatureExtractor` raises a clear error rather than
  silently substituting random weights, and the caller can fall back to
  ``handcrafted_lab``.
- **Training/inference compatibility**: the same frozen extractor runs in
  training and at inference, so there is no train/serve skew.

The extractor is **frozen** in both cases. No fine-tuning is performed: with 20
oral-cancer images in the entire dataset, unfreezing a backbone would memorise
them rather than learn anything transferable. This is recorded in DECISIONS.md.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Sequence, Tuple

import numpy as np

from backend.ml.preprocessing import (
    PREPROCESSING_VERSION,
    NormalisedImage,
    to_chw,
)
from backend.ml.types import FeatureExtractorInfo

HANDCRAFTED_LAB = "handcrafted_lab"
MOBILENET_V3_SMALL = "mobilenet_v3_small"
AVAILABLE_EXTRACTORS: Tuple[str, ...] = (HANDCRAFTED_LAB, MOBILENET_V3_SMALL)

INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 3)

# Handcrafted descriptor layout. Changing any of these changes the feature
# dimension, so the version string below must be bumped alongside them.
_GRID = 4  # 4x4 spatial cells
_HIST_BINS = 12  # per CIELAB channel
HANDCRAFTED_VERSION = "1"

# CIELAB channel ranges used to bin the histograms. L* is defined on [0, 100];
# a*/b* are unbounded in principle but lie within +/-100 for any real sRGB colour,
# so fixed edges keep the descriptor comparable across images.
_LAB_RANGES = ((0.0, 100.0), (-100.0, 100.0), (-100.0, 100.0))

_MOBILENET_FEATURE_DIM = 576  # channels after the final block + global pool
_MOBILENET_VERSION = "torchvision:IMAGENET1K_V1"


class FeatureExtractionError(RuntimeError):
    """Raised when a feature extractor cannot produce a descriptor."""


# --------------------------------------------------------------------------
# Handcrafted CIELAB descriptor
# --------------------------------------------------------------------------


def _cell_slices(length: int, n: int) -> List[slice]:
    """Split ``length`` into ``n`` contiguous, near-equal slices."""
    edges = [round(i * length / n) for i in range(n + 1)]
    return [slice(edges[i], max(edges[i] + 1, edges[i + 1])) for i in range(n)]


def _gradient_magnitude(channel: np.ndarray) -> np.ndarray:
    """Central-difference gradient magnitude, as a texture-energy proxy.

    Implemented by slicing rather than convolution so no SciPy dependency is
    needed. Edges are replicated so the output keeps the input shape.
    """
    padded = np.pad(channel, 1, mode="edge")
    gx = (padded[1:-1, 2:] - padded[1:-1, :-2]) * 0.5
    gy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) * 0.5
    return np.sqrt(gx * gx + gy * gy)


def handcrafted_feature_names() -> List[str]:
    """Names of every handcrafted feature, in output order."""
    names: List[str] = []
    channels = ("L", "a", "b")
    for row in range(_GRID):
        for col in range(_GRID):
            for channel in channels:
                names.append(f"cell{row}{col}_{channel}_mean")
                names.append(f"cell{row}{col}_{channel}_std")
    for channel, (lo, hi) in zip(channels, _LAB_RANGES):
        for b in range(_HIST_BINS):
            names.append(f"hist_{channel}_{b}")
    for row in range(_GRID):
        for col in range(_GRID):
            names.append(f"cell{row}{col}_grad_mean")
    names.extend(
        [
            "grad_mean",
            "grad_std",
            "grad_p90",
            "edge_density",
            "chroma_mean",
            "chroma_std",
            "redness_mean",
            "redness_std",
            "yellowness_mean",
            "yellowness_std",
            "lightness_mean",
            "lightness_std",
            "lightness_p10",
            "lightness_p90",
            "specular_fraction",
        ]
    )
    return names


HANDCRAFTED_FEATURE_NAMES = handcrafted_feature_names()
HANDCRAFTED_FEATURE_DIM = len(HANDCRAFTED_FEATURE_NAMES)


def extract_handcrafted(normalised: NormalisedImage) -> np.ndarray:
    """Compute the deterministic CIELAB colour/texture descriptor.

    Returns a float64 vector of length :data:`HANDCRAFTED_FEATURE_DIM`.
    """
    lab = normalised.lab
    lightness, a_star, b_star = lab[..., 0], lab[..., 1], lab[..., 2]

    values: List[float] = []

    # --- per-cell CIELAB statistics: where in the ROI is it red / white / dark ---
    rows = _cell_slices(lab.shape[0], _GRID)
    cols = _cell_slices(lab.shape[1], _GRID)
    for row in rows:
        for col in cols:
            for channel in (lightness, a_star, b_star):
                patch = channel[row, col]
                values.append(float(np.mean(patch)))
                values.append(float(np.std(patch)))

    # --- global CIELAB histograms: the ROI's overall colour composition ---
    for channel, (lo, hi) in zip((lightness, a_star, b_star), _LAB_RANGES):
        counts, _ = np.histogram(channel, bins=_HIST_BINS, range=(lo, hi))
        total = counts.sum()
        values.extend((counts / total if total else counts).astype(np.float64).tolist())

    # --- texture: gradient energy on L*, globally and per cell ---
    gradient = _gradient_magnitude(lightness)
    for row in rows:
        for col in cols:
            values.append(float(np.mean(gradient[row, col])))

    values.append(float(np.mean(gradient)))
    values.append(float(np.std(gradient)))
    values.append(float(np.percentile(gradient, 90)))
    # Edge density relative to the image's own gradient scale, so it measures
    # texture *structure* rather than exposure.
    threshold = float(np.mean(gradient)) + float(np.std(gradient))
    values.append(float(np.mean(gradient > threshold)))

    # --- chromatic summaries: the clinically salient axes ---
    chroma = np.sqrt(a_star * a_star + b_star * b_star)
    values.append(float(np.mean(chroma)))
    values.append(float(np.std(chroma)))
    values.append(float(np.mean(a_star)))  # +a* = red, the erythroplakia axis
    values.append(float(np.std(a_star)))
    values.append(float(np.mean(b_star)))  # +b* = yellow, keratin/slough
    values.append(float(np.std(b_star)))
    values.append(float(np.mean(lightness)))
    values.append(float(np.std(lightness)))
    values.append(float(np.percentile(lightness, 10)))
    values.append(float(np.percentile(lightness, 90)))
    # Specular highlights: bright and near-achromatic. Intra-oral flash produces
    # these, and their extent tells the model how much of the ROI is unreadable.
    values.append(float(np.mean((lightness > 90.0) & (chroma < 12.0))))

    vector = np.asarray(values, dtype=np.float64)
    if vector.shape[0] != HANDCRAFTED_FEATURE_DIM:  # pragma: no cover - guard
        raise FeatureExtractionError(
            f"Handcrafted descriptor produced {vector.shape[0]} values, "
            f"expected {HANDCRAFTED_FEATURE_DIM}."
        )
    if not np.all(np.isfinite(vector)):
        raise FeatureExtractionError("Handcrafted descriptor contains non-finite values.")
    return vector


# --------------------------------------------------------------------------
# MobileNetV3-Small descriptor
# --------------------------------------------------------------------------

_mobilenet_lock = threading.Lock()
_mobilenet_model = None  # cached across calls; loading costs ~1s


def _load_mobilenet():
    """Load and freeze MobileNetV3-Small, caching the instance.

    Raises:
        FeatureExtractionError: if torch/torchvision are absent or the weights
            cannot be obtained. Never substitutes random weights -- a silently
            untrained backbone would produce meaningless features that look fine.
    """
    global _mobilenet_model
    with _mobilenet_lock:
        if _mobilenet_model is not None:
            return _mobilenet_model
        try:
            import torch
            from torchvision.models import (
                MobileNet_V3_Small_Weights,
                mobilenet_v3_small,
            )
        except ImportError as exc:
            raise FeatureExtractionError(
                "The 'mobilenet_v3_small' extractor requires torch and torchvision. "
                f"Install them or use extractor='{HANDCRAFTED_LAB}'. ({exc})"
            ) from exc

        try:
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        except Exception as exc:  # noqa: BLE001 - network/cache failures vary
            raise FeatureExtractionError(
                "Could not obtain MobileNetV3-Small ImageNet weights (first use "
                "requires network access to download ~9.8 MB). Use "
                f"extractor='{HANDCRAFTED_LAB}' for offline operation. ({exc})"
            ) from exc

        model.eval()
        # Explicit freeze: no parameter of this backbone is ever trained here.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        torch.set_grad_enabled(False)
        _mobilenet_model = model
        return model


def extract_mobilenet(normalised: NormalisedImage) -> np.ndarray:
    """Compute the frozen MobileNetV3-Small pooled descriptor (576-d)."""
    import torch

    model = _load_mobilenet()
    batch = torch.from_numpy(to_chw(normalised.standardised)).unsqueeze(0)
    with torch.no_grad():
        # Take the pooled convolutional embedding, before the ImageNet classifier
        # head: the 1000-way logits encode ImageNet categories, not mucosal
        # appearance, whereas the pooled features are a general visual descriptor.
        maps = model.features(batch)
        pooled = torch.nn.functional.adaptive_avg_pool2d(maps, 1).flatten(1)
    vector = pooled.squeeze(0).to(torch.float64).cpu().numpy()
    if vector.shape[0] != _MOBILENET_FEATURE_DIM:  # pragma: no cover - guard
        raise FeatureExtractionError(
            f"MobileNetV3-Small produced {vector.shape[0]} features, "
            f"expected {_MOBILENET_FEATURE_DIM}."
        )
    return vector


# --------------------------------------------------------------------------
# Public extractor
# --------------------------------------------------------------------------


class ImageFeatureExtractor:
    """Frozen image descriptor with recorded provenance."""

    def __init__(self, name: str = HANDCRAFTED_LAB) -> None:
        if name not in AVAILABLE_EXTRACTORS:
            raise ValueError(
                f"Unknown image extractor {name!r}. Available: {AVAILABLE_EXTRACTORS}"
            )
        self.name = name

    @property
    def feature_dimension(self) -> int:
        return (
            HANDCRAFTED_FEATURE_DIM
            if self.name == HANDCRAFTED_LAB
            else _MOBILENET_FEATURE_DIM
        )

    @property
    def info(self) -> FeatureExtractorInfo:
        if self.name == HANDCRAFTED_LAB:
            return FeatureExtractorInfo(
                name=HANDCRAFTED_LAB,
                version=HANDCRAFTED_VERSION,
                input_shape=INPUT_SHAPE,
                feature_dimension=HANDCRAFTED_FEATURE_DIM,
                preprocessing_version=PREPROCESSING_VERSION,
                weights_source="none (closed-form descriptor)",
                deterministic=True,
                notes=(
                    f"{_GRID}x{_GRID} CIELAB cell statistics, {_HIST_BINS}-bin per-channel "
                    "histograms, gradient-energy texture, chromatic and specular summaries."
                ),
            )
        return FeatureExtractorInfo(
            name=MOBILENET_V3_SMALL,
            version=_MOBILENET_VERSION,
            input_shape=INPUT_SHAPE,
            feature_dimension=_MOBILENET_FEATURE_DIM,
            preprocessing_version=PREPROCESSING_VERSION,
            weights_source="torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1",
            deterministic=True,
            notes="Frozen backbone, eval mode, no_grad, CPU. Pooled features before the classifier.",
        )

    @property
    def feature_names(self) -> List[str]:
        if self.name == HANDCRAFTED_LAB:
            return list(HANDCRAFTED_FEATURE_NAMES)
        return [f"mnv3_{i}" for i in range(_MOBILENET_FEATURE_DIM)]

    def extract(self, normalised: NormalisedImage) -> np.ndarray:
        if self.name == HANDCRAFTED_LAB:
            return extract_handcrafted(normalised)
        return extract_mobilenet(normalised)

    def extract_batch(self, batch: Sequence[NormalisedImage]) -> np.ndarray:
        """Descriptors for several images as an ``(n, feature_dimension)`` matrix."""
        if not batch:
            return np.empty((0, self.feature_dimension), dtype=np.float64)
        return np.vstack([self.extract(item) for item in batch])


def resolve_extractor(name: Optional[str]) -> ImageFeatureExtractor:
    """Build an extractor by name, defaulting to the offline handcrafted one."""
    return ImageFeatureExtractor(name or HANDCRAFTED_LAB)
