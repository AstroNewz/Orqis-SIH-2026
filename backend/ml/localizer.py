"""MobileNet lesion localiser: full image -> predicted ROI box + confidence.

This is MobileNet's role in V1 -- *localisation*, not classification. It answers
"where is the lesion" so the 16-qubit pixel path receives a region rather than a
whole photograph. The classification decision belongs to the VQC downstream.

Target definition (DEC-020): the localiser regresses the **lesion** bounding box
and is trained only on the 318 lesion-annotated images. It is deliberately *not*
trained on "lesion box where available, region box otherwise", because in this
dataset that target leaks the label. Lesion annotations exist for 318 images of
which none are normal; region annotations exist for 1,536 of which 1,531 are
normal; and lesion boxes cover about 44% of the frame against 97% for region
boxes. Box area alone separates the two annotation types at AUC 0.983, so a
localiser trained on the union would emit a small box for pathology and a
full-frame box for a normal mucosa -- handing the downstream classifier the label
through the *size* of its own input. Restricting the target to lesion boxes costs
training data and leaves most normals without a predicted ROI, which is reported
as evaluation condition C rather than hidden.

Architecture: torchvision ``mobilenet_v3_small`` with the classifier head replaced
by a five-output regression head -- four box coordinates plus one confidence logit.
The confidence is *learned*, not derived from the box: it is supervised against the
IoU the predicted box actually achieves, so it estimates "how well did I localise
this" rather than "does this image contain a lesion". That distinction matters,
because a confidence trained on lesion-presence would be a classifier and would
leak in exactly the way the box target was chosen to avoid.

Boxes are predicted in **normalised** coordinates (fractions of width and height),
so the head is resolution-independent and a 4000x3000 capture and a 640x480 one are
treated alike.

Every prediction is one of three explicit outcomes -- see :class:`LocalizationStatus`.
A failure never silently becomes a successful crop; the caller receives
``FALLBACK_USED`` or ``REJECTED`` and the accompanying reason, and the ROI carries
:attr:`~backend.ml.types.RoiSource.PREDICTED_REJECTED` so it stays distinguishable
downstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.ml.roi import center_crop_box
from backend.ml.types import RoiResult, RoiSource

# ---------------------------------------------------------------- configuration
LOCALIZER_VERSION = "carescan-localizer-1"
"""Bumped whenever the architecture, target definition or preprocessing changes.

Travels in the inference response and in the predicted-ROI pixel cache filename,
so a cached crop can always be traced to the localiser that produced it.
"""

LOCALIZER_INPUT_SIZE = (224, 224)
"""``(width, height)`` fed to the network. The MobileNet pretraining resolution."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MIN_CONFIDENCE = 0.35
"""Below this the prediction is not accepted.

Chosen on the **validation** partition (see ``train_localizer.py --sweep``), never
on test. It is an operating point, not a validated constant: raising it converts
weak localisations into explicit fallbacks, which is the safer direction for a
screening tool but leaves fewer images with a real ROI.
"""

MIN_BOX_EDGE_FRACTION = 0.05
"""A predicted edge below 5% of the frame is treated as a degenerate box.

A collapsed box would crop a sliver and then be upsampled to 256x256, inventing
detail that was never photographed.
"""

MAX_BOX_AREA_FRACTION = 0.98
"""A box covering essentially the whole frame is not a localisation.

Accepting it would let the localiser score as "successful" while doing nothing,
which is precisely the statistic condition C exists to separate.
"""


class LocalizationStatus(str, Enum):
    """What actually happened, recorded per image."""

    LOCALIZED = "localized"
    """The network produced a box that passed every acceptance check."""

    FALLBACK_USED = "fallback_used"
    """The network ran but its output was unusable, so the centre crop was used.

    The ROI is still returned -- the pipeline must not stall -- but it is labelled
    :attr:`RoiSource.PREDICTED_REJECTED` and counted in condition C.
    """

    REJECTED = "rejected"
    """No ROI at all: the image could not be read or the model is unavailable.

    Distinct from ``FALLBACK_USED`` because there is no usable region to hand on.
    """


@dataclass(frozen=True)
class LocalizationResult:
    """One localisation attempt, with its outcome made explicit."""

    status: LocalizationStatus
    roi: Optional[RoiResult]
    confidence: float
    predicted_box_normalised: Optional[Tuple[float, float, float, float]]
    reasons: Tuple[str, ...] = ()
    localizer_version: str = LOCALIZER_VERSION

    @property
    def is_localized(self) -> bool:
        return self.status is LocalizationStatus.LOCALIZED

    @property
    def used_fallback(self) -> bool:
        return self.status is LocalizationStatus.FALLBACK_USED

    def describe(self) -> Dict[str, Any]:
        """Provenance for the inference response."""
        return {
            "localizer_version": self.localizer_version,
            "localization_status": self.status.value,
            "localization_confidence": round(float(self.confidence), 6),
            "predicted_box_normalised": (
                [round(v, 6) for v in self.predicted_box_normalised]
                if self.predicted_box_normalised is not None
                else None
            ),
            "roi_source": self.roi.source.value if self.roi is not None else None,
            "roi_box": list(self.roi.box) if self.roi is not None else None,
            "localization_reasons": list(self.reasons),
        }


# --------------------------------------------------------------------- geometry
def normalised_box_from_pixels(
    box: Sequence[float], width: int, height: int
) -> Tuple[float, float, float, float]:
    """Pixel ``(x0, y0, x1, y1)`` -> fractions of the frame."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    x0, y0, x1, y1 = (float(v) for v in box)
    return (x0 / width, y0 / height, x1 / width, y1 / height)


def pixel_box_from_normalised(
    box: Sequence[float], width: int, height: int
) -> Tuple[int, int, int, int]:
    """Fractions -> pixel ``(x0, y0, x1, y1)``, clamped and ordered.

    Coordinates are sorted rather than assumed ordered: an unconstrained regression
    head can predict ``x1 < x0``, and silently keeping that would produce a negative
    width that only fails several stages later.
    """
    x0, x1 = sorted((float(box[0]) * width, float(box[2]) * width))
    y0, y1 = sorted((float(box[1]) * height, float(box[3]) * height))
    x0 = int(max(0, min(round(x0), width)))
    x1 = int(max(0, min(round(x1), width)))
    y0 = int(max(0, min(round(y0), height)))
    y1 = int(max(0, min(round(y1), height)))
    return (x0, y0, x1, y1)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two boxes in any consistent coordinate system."""
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def check_predicted_box(
    box_normalised: Sequence[float],
    confidence: float,
    *,
    min_confidence: float = MIN_CONFIDENCE,
    min_edge_fraction: float = MIN_BOX_EDGE_FRACTION,
    max_area_fraction: float = MAX_BOX_AREA_FRACTION,
) -> List[str]:
    """Reasons the prediction is unusable. Empty list means accepted.

    Returned as a list rather than a bool so the failure is reportable: "which
    check failed, on how many images" is a metric the localiser has to publish.
    """
    reasons: List[str] = []
    x0, y0, x1, y1 = (float(v) for v in box_normalised)
    if not all(np.isfinite([x0, y0, x1, y1, confidence])):
        return ["non_finite_prediction"]
    if confidence < min_confidence:
        reasons.append(f"low_confidence:{confidence:.4f}<{min_confidence}")

    width = abs(x1 - x0)
    height = abs(y1 - y0)
    if width < min_edge_fraction or height < min_edge_fraction:
        reasons.append(f"degenerate_box:{width:.4f}x{height:.4f}")
    if width * height > max_area_fraction:
        reasons.append(f"box_covers_whole_frame:{width * height:.4f}")
    # A box entirely outside the frame is a different failure from a small one.
    if x1 <= 0.0 or y1 <= 0.0 or x0 >= 1.0 or y0 >= 1.0:
        reasons.append("box_outside_frame")
    return reasons


# ------------------------------------------------------------------ the network
def build_localizer_network(pretrained: bool = True):
    """``mobilenet_v3_small`` with a 5-output head: 4 box coords + 1 confidence logit.

    ImageNet initialisation is used by default: 215 training images is far too few
    to learn early visual features from scratch, and the pretrained weights are a
    documented, versioned starting point rather than a fitted-on-our-data one, so
    they introduce no leakage.
    """
    import torch.nn as nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    network = mobilenet_v3_small(weights=weights)
    in_features = network.classifier[0].in_features
    network.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, 5),
    )
    return network


def preprocess_for_localizer(image) -> np.ndarray:
    """Full image -> ``(3, 224, 224)`` float32, ImageNet-standardised.

    The **whole frame** is used, never a crop: the localiser's job is to find the
    region, so pre-cropping would beg the question. Resized without preserving
    aspect ratio, matching the normalised-coordinate target -- both the input and
    the label live in the same stretched frame, so the mapping is consistent.
    """
    from PIL import Image

    resized = image.convert("RGB").resize(LOCALIZER_INPUT_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        IMAGENET_STD, dtype=np.float32
    )
    return np.ascontiguousarray(array.transpose(2, 0, 1))


class LesionLocalizer:
    """Trained localiser, loaded for inference.

    Deterministic by construction: evaluation mode disables dropout, inference runs
    under ``no_grad``, and no augmentation or sampling happens at predict time. The
    same image therefore yields the same box on every call, which
    ``test_localizer.py`` asserts rather than assumes.
    """

    def __init__(self, network=None, *, metadata: Optional[Dict[str, Any]] = None):
        self.network = network
        self.metadata: Dict[str, Any] = metadata or {}
        self.version = str(self.metadata.get("localizer_version", LOCALIZER_VERSION))
        self.min_confidence = float(self.metadata.get("min_confidence", MIN_CONFIDENCE))
        if network is not None:
            network.eval()

    # ------------------------------------------------------------ persistence
    @classmethod
    def load(cls, directory: Path) -> "LesionLocalizer":
        """Load weights and the recorded training configuration.

        Raises:
            FileNotFoundError: no artifact, naming the command that trains one.
        """
        import torch

        directory = Path(directory)
        weights_path = directory / "localizer.pt"
        metadata_path = directory / "localizer.json"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"No localiser weights at {weights_path}. Train one with "
                "`python -m backend.training.train_localizer`."
            )
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        network = build_localizer_network(pretrained=False)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        network.load_state_dict(state)
        network.eval()
        return cls(network, metadata=metadata)

    @property
    def is_available(self) -> bool:
        return self.network is not None

    # -------------------------------------------------------------- inference
    def predict_normalised(self, images: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        """``(boxes, confidences)`` for a batch of PIL images.

        Boxes are ``(n, 4)`` normalised coordinates, confidences ``(n,)`` in [0, 1].
        Both come straight from the network: no clamping, no acceptance logic. The
        raw prediction is what gets recorded, and the checks are applied separately
        so a rejected box is still reportable.
        """
        import torch

        if not self.is_available:
            raise RuntimeError("Localiser has no network loaded.")
        if not len(images):
            return np.empty((0, 4), dtype=np.float64), np.empty(0, dtype=np.float64)

        batch = np.stack([preprocess_for_localizer(image) for image in images])
        with torch.no_grad():
            output = self.network(torch.from_numpy(batch))
            # Sigmoid on all five outputs: box coordinates are fractions of the
            # frame and the confidence is a probability, so both are bounded in
            # [0, 1] by construction rather than by a later clamp.
            output = torch.sigmoid(output)
        values = output.numpy().astype(np.float64)
        return values[:, :4], values[:, 4]

    def localize(
        self, image, *, center_crop_fraction: float = 0.80
    ) -> LocalizationResult:
        """Localise one image, returning an explicit outcome.

        A model-unavailable or unreadable-image case returns ``REJECTED`` with no
        ROI. A prediction that fails the acceptance checks returns
        ``FALLBACK_USED`` with the centre crop, labelled
        :attr:`RoiSource.PREDICTED_REJECTED` so it never counts as a localisation.
        """
        width, height = int(image.width), int(image.height)
        if not self.is_available:
            return LocalizationResult(
                status=LocalizationStatus.REJECTED,
                roi=None,
                confidence=0.0,
                predicted_box_normalised=None,
                reasons=("localizer_unavailable",),
                localizer_version=self.version,
            )

        boxes, confidences = self.predict_normalised([image])
        box = tuple(float(v) for v in boxes[0])
        confidence = float(confidences[0])
        reasons = check_predicted_box(
            box, confidence, min_confidence=self.min_confidence
        )

        if reasons:
            fallback = center_crop_box(width, height, fraction=center_crop_fraction)
            return LocalizationResult(
                status=LocalizationStatus.FALLBACK_USED,
                roi=RoiResult(
                    source=RoiSource.PREDICTED_REJECTED,
                    x0=fallback[0],
                    y0=fallback[1],
                    x1=fallback[2],
                    y1=fallback[3],
                    source_width=width,
                    source_height=height,
                ),
                confidence=confidence,
                predicted_box_normalised=box,
                reasons=tuple(reasons),
                localizer_version=self.version,
            )

        x0, y0, x1, y1 = pixel_box_from_normalised(box, width, height)
        # The normalised box passed its checks, but rounding to pixels on a small
        # image can still collapse an edge. Caught here rather than trusted.
        if x1 - x0 < 1 or y1 - y0 < 1:
            fallback = center_crop_box(width, height, fraction=center_crop_fraction)
            return LocalizationResult(
                status=LocalizationStatus.FALLBACK_USED,
                roi=RoiResult(
                    source=RoiSource.PREDICTED_REJECTED,
                    x0=fallback[0],
                    y0=fallback[1],
                    x1=fallback[2],
                    y1=fallback[3],
                    source_width=width,
                    source_height=height,
                ),
                confidence=confidence,
                predicted_box_normalised=box,
                reasons=("degenerate_after_pixel_rounding",),
                localizer_version=self.version,
            )

        return LocalizationResult(
            status=LocalizationStatus.LOCALIZED,
            roi=RoiResult(
                source=RoiSource.PREDICTED,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                source_width=width,
                source_height=height,
            ),
            confidence=confidence,
            predicted_box_normalised=box,
            localizer_version=self.version,
        )


@dataclass
class LocalizationTally:
    """Counts of every outcome, for the rates the localiser must publish.

    Kept as a class rather than assembled ad hoc at each call site so that the
    three statuses cannot drift apart in different reports, and so
    ``localized + fallback + rejected == total`` is checkable in one place.
    """

    localized: int = 0
    fallback_used: int = 0
    rejected: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)
    confidences: List[float] = field(default_factory=list)
    ious: List[float] = field(default_factory=list)

    def record(self, result: LocalizationResult, iou: Optional[float] = None) -> None:
        if result.status is LocalizationStatus.LOCALIZED:
            self.localized += 1
        elif result.status is LocalizationStatus.FALLBACK_USED:
            self.fallback_used += 1
        else:
            self.rejected += 1
        for reason in result.reasons:
            # Keyed by the reason *kind*, not the value, so
            # "low_confidence:0.21<0.35" and "low_confidence:0.30<0.35" aggregate.
            key = reason.split(":", 1)[0]
            self.reasons[key] = self.reasons.get(key, 0) + 1
        if result.status is not LocalizationStatus.REJECTED:
            self.confidences.append(float(result.confidence))
        if iou is not None:
            self.ious.append(float(iou))

    @property
    def total(self) -> int:
        return self.localized + self.fallback_used + self.rejected

    def summary(self) -> Dict[str, Any]:
        total = max(1, self.total)
        report: Dict[str, Any] = {
            "n_images": self.total,
            "localized": self.localized,
            "fallback_used": self.fallback_used,
            "rejected": self.rejected,
            "localization_rate": round(self.localized / total, 4),
            "fallback_rate": round(self.fallback_used / total, 4),
            "rejection_rate": round(self.rejected / total, 4),
            "failure_reasons": dict(sorted(self.reasons.items())),
        }
        if self.confidences:
            values = np.asarray(self.confidences, dtype=np.float64)
            report["confidence"] = {
                "mean": round(float(values.mean()), 4),
                "min": round(float(values.min()), 4),
                "max": round(float(values.max()), 4),
                "median": round(float(np.median(values)), 4),
            }
        if self.ious:
            values = np.asarray(self.ious, dtype=np.float64)
            report["iou"] = {
                "mean": round(float(values.mean()), 4),
                "median": round(float(np.median(values)), 4),
                # The share above 0.5 is the standard detection-quality summary;
                # 0.25 is reported too because a coarse-but-useful crop still
                # helps the downstream pipeline even when it would not count as a
                # detection.
                "at_least_0.25": round(float((values >= 0.25).mean()), 4),
                "at_least_0.5": round(float((values >= 0.5).mean()), 4),
                "at_least_0.75": round(float((values >= 0.75).mean()), 4),
            }
        return report
