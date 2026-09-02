"""Strongly typed models for the classical ML pipeline.

Pydantic models, not ``Map<String, dynamic>``-style dicts, so every stage
boundary is validated and every artifact is self-describing.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class QualityIssue(str, Enum):
    """Reasons an image can fail acquisition quality control."""

    INSUFFICIENT_ILLUMINATION = "insufficient_illumination"
    EXCESSIVE_ILLUMINATION = "excessive_illumination"
    EXCESSIVE_BLUR = "excessive_blur"
    INCOMPLETE_LESION_VISIBILITY = "incomplete_lesion_visibility"
    EXCESSIVE_COMPRESSION = "excessive_compression"
    INADEQUATE_ROI_SIZE = "inadequate_roi_size"
    INSUFFICIENT_RESOLUTION = "insufficient_resolution"
    UNREADABLE_IMAGE = "unreadable_image"


class QualityVerdict(str, Enum):
    ACCEPT = "accept"
    FLAG = "flag"
    REJECT = "reject"


class ImageQualityReport(BaseModel):
    """Outcome of the image quality-control stage."""

    model_config = ConfigDict(frozen=True)

    verdict: QualityVerdict
    issues: List[QualityIssue] = Field(default_factory=list)
    mean_luminance: float
    laplacian_variance: float
    clipped_fraction: float
    width: int
    height: int
    bytes_per_pixel: float
    roi_fraction: float
    messages: List[str] = Field(default_factory=list)

    @property
    def acceptable(self) -> bool:
        return self.verdict is not QualityVerdict.REJECT

    @property
    def patient_message(self) -> str:
        """Human-readable guidance, free of technical detail."""
        if self.verdict is QualityVerdict.ACCEPT:
            return "Image quality is sufficient for analysis."
        if not self.messages:
            return "Please retake the photograph in better conditions."
        return " ".join(self.messages)


class RoiSource(str, Enum):
    LESION_POLYGON = "lesion_polygon"
    REGION_POLYGON = "region_polygon"
    CENTER_CROP = "center_crop"
    FULL_IMAGE = "full_image"


class RoiResult(BaseModel):
    """Region of interest selected for downstream feature extraction."""

    model_config = ConfigDict(frozen=True)

    source: RoiSource
    x0: int
    y0: int
    x1: int
    y1: int
    source_width: int
    source_height: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area_fraction(self) -> float:
        total = self.source_width * self.source_height
        return (self.width * self.height) / total if total else 0.0

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


class FeatureExtractorInfo(BaseModel):
    """Provenance of the image feature extractor, persisted with the artifacts."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    input_shape: Tuple[int, int, int]
    feature_dimension: int
    preprocessing_version: str
    weights_source: str
    deterministic: bool = True
    notes: str = ""


class ImageFeatures(BaseModel):
    """Numerical image descriptor plus the provenance needed to reproduce it."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    values: List[float]
    extractor: FeatureExtractorInfo
    roi: RoiResult

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)


class ClinicalFeatures(BaseModel):
    """Encoded clinical risk factors."""

    model_config = ConfigDict(frozen=True)

    values: List[float]
    names: List[str]
    raw: Dict[str, Optional[str]] = Field(default_factory=dict)
    n_missing: int = 0

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)


class FusedFeatures(BaseModel):
    """Image + clinical features after fusion and dimensionality reduction."""

    model_config = ConfigDict(frozen=True)

    values: List[float]
    feature_mode: str
    n_image_features: int
    n_clinical_features: int
    n_reduced: int
    reduction_method: str
    explained_variance_ratio: Optional[float] = None

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)


class PreprocessingOutcome(BaseModel):
    """Everything the preprocessing pipeline produces for one image."""

    model_config = ConfigDict(frozen=True)

    quality: ImageQualityReport
    roi: Optional[RoiResult] = None
    image_features: Optional[ImageFeatures] = None
    clinical_features: Optional[ClinicalFeatures] = None
    fused: Optional[FusedFeatures] = None

    @property
    def usable(self) -> bool:
        return self.fused is not None and self.quality.acceptable


class PreprocessingError(RuntimeError):
    """Raised when preprocessing cannot complete for a technical reason."""


class ImageQualityRejected(ValueError):
    """Raised when an image fails quality control and must be reacquired."""

    def __init__(self, report: ImageQualityReport):
        super().__init__(report.patient_message)
        self.report = report
