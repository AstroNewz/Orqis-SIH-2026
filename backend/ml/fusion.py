"""Feature fusion: xclassical + xclinical -> xfused.

Scalers are fitted on **training data only** and persisted. Fitting on the full
dataset would let validation and test statistics bleed into the transform, which
is a subtle but real form of leakage that inflates every downstream metric.

Three feature modes are supported so the contribution of each modality can be
measured rather than assumed (PART 7):

``image_only``     image descriptor alone
``clinical_only``  clinical risk factors alone
``multimodal``     both, standardised then concatenated

Standardisation before concatenation is not cosmetic. The handcrafted image
descriptor mixes CIELAB means (order 10-100), normalised histogram bins (order
0.01-0.1) and gradient energies (order 1-100), while the clinical block is
already roughly unit-scale. Concatenating raw would let a single L* mean dominate
the L2 norm of the amplitude-encoded state, drowning out the clinical signal
entirely -- amplitude encoding normalises the whole vector, so relative scale
directly determines relative influence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np

FeatureMode = Literal["image_only", "clinical_only", "multimodal"]
FEATURE_MODES: tuple = ("image_only", "clinical_only", "multimodal")

FUSION_VERSION = "fusion-1"


class FusionError(ValueError):
    """Raised when fusion inputs are inconsistent with the fitted configuration."""


@dataclass
class StandardScaler:
    """Minimal, persistable z-score scaler.

    Written out rather than pulled from scikit-learn so the fitted state is a
    plain JSON-serialisable dict, keeping model artifacts readable and free of
    pickle-version coupling. Behaviour matches
    ``sklearn.preprocessing.StandardScaler`` for finite input.
    """

    mean: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None
    n_features: int = 0
    n_samples_seen: int = 0

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.scale is not None

    def fit(self, matrix: np.ndarray) -> "StandardScaler":
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise FusionError(f"StandardScaler.fit expects a non-empty 2-D matrix, got {matrix.shape}")
        self.mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        # A constant column carries no information; scaling by 1.0 maps it to a
        # constant 0 rather than producing inf/nan.
        self.scale = np.where(std > 1e-12, std, 1.0)
        self.n_features = matrix.shape[1]
        self.n_samples_seen = matrix.shape[0]
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise FusionError("StandardScaler used before fit().")
        matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
        if matrix.shape[1] != self.n_features:
            raise FusionError(
                f"StandardScaler expects {self.n_features} features, got {matrix.shape[1]}."
            )
        return (matrix - self.mean) / self.scale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": None if self.mean is None else self.mean.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
            "n_features": self.n_features,
            "n_samples_seen": self.n_samples_seen,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StandardScaler":
        scaler = cls(n_features=int(payload.get("n_features", 0)))
        scaler.n_samples_seen = int(payload.get("n_samples_seen", 0))
        if payload.get("mean") is not None:
            scaler.mean = np.asarray(payload["mean"], dtype=np.float64)
        if payload.get("scale") is not None:
            scaler.scale = np.asarray(payload["scale"], dtype=np.float64)
        return scaler


@dataclass
class FeatureFusion:
    """Standardises and concatenates the image and clinical feature blocks."""

    mode: FeatureMode = "multimodal"
    version: str = FUSION_VERSION
    image_scaler: StandardScaler = field(default_factory=StandardScaler)
    clinical_scaler: StandardScaler = field(default_factory=StandardScaler)
    image_feature_names: List[str] = field(default_factory=list)
    clinical_feature_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in FEATURE_MODES:
            raise FusionError(f"Unknown feature mode {self.mode!r}. Expected one of {FEATURE_MODES}.")

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        image_features: Optional[np.ndarray] = None,
        clinical_features: Optional[np.ndarray] = None,
        *,
        image_feature_names: Sequence[str] = (),
        clinical_feature_names: Sequence[str] = (),
    ) -> "FeatureFusion":
        """Fit the scalers on training-partition features only."""
        if self.uses_image:
            if image_features is None:
                raise FusionError(f"mode={self.mode!r} requires image features to fit.")
            self.image_scaler.fit(image_features)
            self.image_feature_names = list(image_feature_names)
        if self.uses_clinical:
            if clinical_features is None:
                raise FusionError(f"mode={self.mode!r} requires clinical features to fit.")
            self.clinical_scaler.fit(clinical_features)
            self.clinical_feature_names = list(clinical_feature_names)
        return self

    # ----------------------------------------------------------- properties
    @property
    def uses_image(self) -> bool:
        return self.mode in ("image_only", "multimodal")

    @property
    def uses_clinical(self) -> bool:
        return self.mode in ("clinical_only", "multimodal")

    @property
    def fitted(self) -> bool:
        return (not self.uses_image or self.image_scaler.fitted) and (
            not self.uses_clinical or self.clinical_scaler.fitted
        )

    @property
    def n_image_features(self) -> int:
        return self.image_scaler.n_features if self.uses_image else 0

    @property
    def n_clinical_features(self) -> int:
        return self.clinical_scaler.n_features if self.uses_clinical else 0

    @property
    def n_features(self) -> int:
        return self.n_image_features + self.n_clinical_features

    @property
    def feature_names(self) -> List[str]:
        names: List[str] = []
        if self.uses_image:
            names.extend(
                self.image_feature_names
                or [f"img_{i}" for i in range(self.n_image_features)]
            )
        if self.uses_clinical:
            names.extend(
                self.clinical_feature_names
                or [f"clin_{i}" for i in range(self.n_clinical_features)]
            )
        return names

    # ------------------------------------------------------------ transform
    def transform(
        self,
        image_features: Optional[np.ndarray] = None,
        clinical_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Standardise and concatenate. Returns an ``(n_samples, n_features)`` matrix."""
        if not self.fitted:
            raise FusionError("FeatureFusion used before fit().")

        blocks: List[np.ndarray] = []
        if self.uses_image:
            if image_features is None:
                raise FusionError(f"mode={self.mode!r} requires image features.")
            blocks.append(self.image_scaler.transform(image_features))
        if self.uses_clinical:
            if clinical_features is None:
                raise FusionError(f"mode={self.mode!r} requires clinical features.")
            blocks.append(self.clinical_scaler.transform(clinical_features))

        if len({block.shape[0] for block in blocks}) > 1:
            raise FusionError(
                "Image and clinical blocks have different sample counts: "
                f"{[block.shape for block in blocks]}"
            )
        return np.hstack(blocks)

    def transform_one(
        self,
        image_features: Optional[np.ndarray] = None,
        clinical_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fuse a single sample into a 1-D vector."""
        image = None if image_features is None else np.atleast_2d(image_features)
        clinical = None if clinical_features is None else np.atleast_2d(clinical_features)
        return self.transform(image, clinical)[0]

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "image_scaler": self.image_scaler.to_dict(),
            "clinical_scaler": self.clinical_scaler.to_dict(),
            "image_feature_names": self.image_feature_names,
            "clinical_feature_names": self.clinical_feature_names,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FeatureFusion":
        fusion = cls(
            mode=payload.get("mode", "multimodal"),
            version=str(payload.get("version", FUSION_VERSION)),
        )
        fusion.image_scaler = StandardScaler.from_dict(payload.get("image_scaler", {}))
        fusion.clinical_scaler = StandardScaler.from_dict(payload.get("clinical_scaler", {}))
        fusion.image_feature_names = list(payload.get("image_feature_names", []))
        fusion.clinical_feature_names = list(payload.get("clinical_feature_names", []))
        return fusion
