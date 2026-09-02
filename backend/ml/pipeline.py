"""The classical pipeline: raw image -> reduced feature vector.

This module wires together the stages PART 4 prescribes, in order, and nothing
else. It contains no quantum code and imports no quantum library -- its only
output is the reduced real-valued vector the quantum subsystem encodes.

    RAW IMAGE
      -> ROI GEOMETRY            (roi.extract_roi)
      -> IMAGE QUALITY CONTROL   (image_quality.assess_image_quality)
      -> IMAGE NORMALISATION     (preprocessing.normalise_crop)
      -> CLASSICAL FEATURES      (features_image.ImageFeatureExtractor)
      -> CLINICAL FEATURES       (features_clinical.ClinicalFeatureEncoder)
      -> FEATURE FUSION          (fusion.FeatureFusion)
      -> DIMENSIONALITY REDUCTION(reduction.DimensionalityReducer)
      -> xreduced                (consumed by quantum_ml)

Why ROI geometry precedes quality control
-----------------------------------------
Two of the mandated quality checks -- inadequate ROI size and incomplete lesion
visibility -- are statements *about the ROI*, so the ROI box must exist before
they can be evaluated. Only the box is computed first, which is pure integer
arithmetic. No pixel is normalised and no descriptor is extracted until quality
control has returned a verdict, so a rejected capture costs nothing beyond a
decode.

Fitting discipline
------------------
:meth:`ClassicalPipeline.fit` accepts **training-partition features only**. The
clinical encoder, both fusion scalers, and the PCA basis are fitted there and
nowhere else. Validation and test features are only ever passed through
:meth:`transform`. This is the single most important invariant in the module:
fitting on all partitions would inflate every metric downstream, silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.ml.features_clinical import ClinicalFeatureEncoder
from backend.ml.features_image import (
    HANDCRAFTED_LAB,
    ImageFeatureExtractor,
    resolve_extractor,
)
from backend.ml.fusion import FeatureFusion, FeatureMode
from backend.ml.image_quality import assess_image_quality
from backend.ml.preprocessing import PREPROCESSING_VERSION, normalise_crop
from backend.ml.reduction import (
    DimensionalityReducer,
    build_reducer,
    target_dimension_for_qubits,
)
from backend.ml.roi import crop_to_roi, extract_roi
from backend.ml.types import (
    ClinicalFeatures,
    FusedFeatures,
    ImageFeatures,
    ImageQualityRejected,
    ImageQualityReport,
    PreprocessingError,
    PreprocessingOutcome,
    QualityVerdict,
    RoiResult,
)

PIPELINE_VERSION = "carescan-pipeline-1"


@dataclass
class ClassicalPipeline:
    """Fitted classical pipeline, from pixels to the quantum layer's input.

    Construct with :func:`build_pipeline` (from settings) or directly, then
    either :meth:`fit` on training features or load a persisted state via
    :mod:`backend.ml.artifacts`.
    """

    extractor: ImageFeatureExtractor
    clinical_encoder: ClinicalFeatureEncoder
    fusion: FeatureFusion
    reducer: DimensionalityReducer
    n_qubits: int
    version: str = PIPELINE_VERSION
    preprocessing_version: str = PREPROCESSING_VERSION
    target_size: Tuple[int, int] = (224, 224)
    center_crop_fraction: float = 0.80
    training_metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ properties
    @property
    def feature_mode(self) -> FeatureMode:
        return self.fusion.mode

    @property
    def quantum_dimension(self) -> int:
        """Amplitude-space dimension implied by the qubit count."""
        return target_dimension_for_qubits(self.n_qubits)

    @property
    def fitted(self) -> bool:
        clinical_ok = not self.fusion.uses_clinical or self.clinical_encoder.fitted
        return clinical_ok and self.fusion.fitted and self.reducer.fitted

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise PreprocessingError(
                "ClassicalPipeline is not fitted. Run backend.training.train_classical "
                "or load a persisted artifact before inference."
            )

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        *,
        image_features: Optional[np.ndarray] = None,
        clinical_rows: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
        fit_clinical_encoder: bool = True,
    ) -> "ClassicalPipeline":
        """Fit every learned transform on the **training partition only**.

        Args:
            image_features: ``(n_train, d_image)`` descriptors for training images.
                Required unless the feature mode is ``clinical_only``.
            clinical_rows: metadata rows aligned with ``image_features``, one per
                training image. ``None`` entries are patients without a metadata
                row, which the encoder handles explicitly. Required unless the
                feature mode is ``image_only``.
            fit_clinical_encoder: set ``False`` when the encoder was already
                fitted on the training patients elsewhere.

        Returns:
            self, fitted.
        """
        clinical_matrix: Optional[np.ndarray] = None

        if self.fusion.uses_clinical:
            if clinical_rows is None:
                raise PreprocessingError(
                    f"feature_mode={self.feature_mode!r} requires clinical_rows to fit."
                )
            if fit_clinical_encoder:
                # Only rows that exist can inform the scaler; rows for patients
                # without metadata contribute the all-unknown encoding at
                # transform time and must not skew the fitted statistics.
                self.clinical_encoder.fit([row for row in clinical_rows if row])
            clinical_matrix = self.clinical_encoder.transform_many(list(clinical_rows))

        if self.fusion.uses_image and image_features is None:
            raise PreprocessingError(
                f"feature_mode={self.feature_mode!r} requires image_features to fit."
            )

        image_matrix = None
        if image_features is not None:
            image_matrix = np.asarray(image_features, dtype=np.float64)

        self.fusion.fit(
            image_matrix,
            clinical_matrix,
            image_feature_names=self.extractor.feature_names,
            clinical_feature_names=(
                self.clinical_encoder.feature_names if self.fusion.uses_clinical else []
            ),
        )
        fused = self.fusion.transform(image_matrix, clinical_matrix)
        self.reducer.fit(fused, input_feature_names=self.fusion.feature_names)
        return self

    # ------------------------------------------------------ batch transform
    def transform_matrix(
        self,
        *,
        image_features: Optional[np.ndarray] = None,
        clinical_rows: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
    ) -> np.ndarray:
        """Fuse and reduce a whole partition. ``(n_samples, n_reduced)``."""
        self._require_fitted()
        clinical_matrix = None
        if self.fusion.uses_clinical:
            if clinical_rows is None:
                raise PreprocessingError(
                    f"feature_mode={self.feature_mode!r} requires clinical_rows."
                )
            clinical_matrix = self.clinical_encoder.transform_many(list(clinical_rows))
        image_matrix = (
            None if image_features is None else np.asarray(image_features, dtype=np.float64)
        )
        fused = self.fusion.transform(image_matrix, clinical_matrix)
        return self.reducer.transform(fused)

    # ---------------------------------------------------- per-image stages
    def image_descriptor(
        self,
        image,
        *,
        file_size_bytes: Optional[int] = None,
        lesion_polygons: Sequence = (),
        region_polygons: Sequence = (),
        config: Optional[Settings] = None,
    ) -> Tuple[ImageQualityReport, RoiResult, Optional[ImageFeatures]]:
        """Run ROI selection, quality control, normalisation and extraction.

        The image descriptor is ``None`` when quality control rejected the
        capture: extracting features from an unusable frame would produce a
        number with no basis.
        """
        width, height = image.size
        roi = extract_roi(
            width,
            height,
            lesion_polygons=lesion_polygons,
            region_polygons=region_polygons,
            center_crop_fraction=self.center_crop_fraction,
        )
        quality = assess_image_quality(
            image,
            file_size_bytes=file_size_bytes,
            roi_box=roi.box,
            roi_available=bool(lesion_polygons or region_polygons),
            config=config,
        )
        if quality.verdict is QualityVerdict.REJECT:
            return quality, roi, None

        try:
            normalised = normalise_crop(crop_to_roi(image, roi), target_size=self.target_size)
            values = self.extractor.extract(normalised)
        except Exception as exc:  # noqa: BLE001 - surfaced as a typed pipeline error
            raise PreprocessingError(
                f"Feature extraction failed for a {width}x{height} capture: {exc}"
            ) from exc

        descriptor = ImageFeatures(
            values=values.tolist(), extractor=self.extractor.info, roi=roi
        )
        return quality, roi, descriptor

    # ------------------------------------------------------------ inference
    def run(
        self,
        image,
        *,
        clinical_row: Optional[Mapping[str, Any]] = None,
        file_size_bytes: Optional[int] = None,
        lesion_polygons: Sequence = (),
        region_polygons: Sequence = (),
        strict_quality: bool = True,
        config: Optional[Settings] = None,
    ) -> PreprocessingOutcome:
        """Full single-image path, returning everything the backend records.

        Args:
            image: an open PIL image.
            clinical_row: the patient's risk-factor answers, or ``None``.
            file_size_bytes: encoded size, for the compression check.
            lesion_polygons / region_polygons: annotations when available. Absent
                for app captures, in which case a centre crop is used.
            strict_quality: raise :class:`ImageQualityRejected` on a REJECT
                verdict. ``False`` is used for dataset processing, where samples
                must be recorded rather than silently discarded.
            config: settings override.

        Raises:
            ImageQualityRejected: quality control rejected the capture and
                ``strict_quality`` is set.
            PreprocessingError: a technical failure in a downstream stage.
        """
        self._require_fitted()

        quality, roi, descriptor = self.image_descriptor(
            image,
            file_size_bytes=file_size_bytes,
            lesion_polygons=lesion_polygons,
            region_polygons=region_polygons,
            config=config,
        )
        if descriptor is None:
            if strict_quality:
                raise ImageQualityRejected(quality)
            return PreprocessingOutcome(quality=quality, roi=roi)

        clinical: Optional[ClinicalFeatures] = None
        clinical_vector: Optional[np.ndarray] = None
        if self.fusion.uses_clinical:
            clinical = self.clinical_encoder.transform(clinical_row)
            clinical_vector = clinical.vector

        image_vector = descriptor.vector if self.fusion.uses_image else None
        fused_vector = self.fusion.transform_one(image_vector, clinical_vector)
        reduced = self.reducer.transform_one(fused_vector)

        fused = FusedFeatures(
            values=reduced.tolist(),
            feature_mode=self.feature_mode,
            n_image_features=self.fusion.n_image_features,
            n_clinical_features=self.fusion.n_clinical_features,
            n_reduced=int(reduced.shape[0]),
            reduction_method=self.reducer.method,
            explained_variance_ratio=round(self.reducer.total_explained_variance_ratio, 6),
        )
        return PreprocessingOutcome(
            quality=quality,
            roi=roi,
            image_features=descriptor if self.fusion.uses_image else None,
            clinical_features=clinical,
            fused=fused,
        )

    # -------------------------------------------------------------- summary
    def describe(self) -> Dict[str, Any]:
        """Provenance of every stage, for ``/model/info`` and artifact metadata."""
        return {
            "pipeline_version": self.version,
            "preprocessing_version": self.preprocessing_version,
            "feature_mode": self.feature_mode,
            "target_size": list(self.target_size),
            "center_crop_fraction": self.center_crop_fraction,
            "image_extractor": self.extractor.info.model_dump(),
            "clinical_encoder": {
                "version": self.clinical_encoder.version,
                "fitted": self.clinical_encoder.fitted,
                "n_features": self.clinical_encoder.n_features
                if self.clinical_encoder.fitted
                else 0,
                "n_fit_rows": self.clinical_encoder.n_fit_rows,
            },
            "fusion": {
                "version": self.fusion.version,
                "mode": self.fusion.mode,
                "n_image_features": self.fusion.n_image_features,
                "n_clinical_features": self.fusion.n_clinical_features,
                "n_features": self.fusion.n_features,
            },
            "reduction": self.reducer.summary(),
            "n_qubits": self.n_qubits,
            "quantum_dimension": self.quantum_dimension,
            "training_metadata": dict(self.training_metadata),
        }

    @property
    def reduced_dimension(self) -> int:
        """Length of the vector handed to the quantum encoder, before padding."""
        return self.reducer.output_dimension


def build_pipeline(
    *,
    config: Optional[Settings] = None,
    feature_mode: Optional[FeatureMode] = None,
    extractor_name: Optional[str] = None,
    n_qubits: Optional[int] = None,
) -> ClassicalPipeline:
    """Construct an unfitted pipeline from settings, with explicit overrides.

    Overrides exist for the ablation experiments (PART 16), which sweep feature
    mode, extractor and qubit count without editing configuration.
    """
    cfg = config or default_settings
    qubits = n_qubits if n_qubits is not None else cfg.QUANTUM_QUBITS
    mode: FeatureMode = feature_mode or cfg.FEATURE_MODE
    return ClassicalPipeline(
        extractor=resolve_extractor(extractor_name or HANDCRAFTED_LAB),
        clinical_encoder=ClinicalFeatureEncoder(),
        fusion=FeatureFusion(mode=mode),
        reducer=build_reducer(n_qubits=qubits),
        n_qubits=qubits,
    )
