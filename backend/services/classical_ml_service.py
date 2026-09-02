"""Removed: this module used to fabricate image features.

``ClassicalMLService.extract_features`` derived a 512-dimensional vector from
``sha256(image_path)`` seeded into ``np.random.RandomState`` and passed it off as a
MobileNetV3 embedding. It never opened the image. Two identical photographs stored
under different filenames scored differently; the same photograph re-uploaded scored
the same only because the *path* was the same. The "baseline probability" was a
sigmoid over three of those random values.

That is exactly the fabrication the FINAL EXECUTION RULE prohibits, so it is gone
rather than deprecated -- a shim that still returned numbers would leave the failure
mode reachable.

Real feature extraction now lives in:

* :mod:`backend.ml.features_image` -- handcrafted CIELAB/texture descriptors computed
  from actual pixels, after quality control and ROI extraction.
* :mod:`backend.ml.features_clinical` -- deterministic encoding of the risk factors,
  with fitted scalers loaded from the model artifacts.
* :class:`backend.services.inference_service.InferenceService` -- orchestrates both,
  plus fusion, reduction, the quantum circuit and calibration.

For the classical *comparison* models (logistic regression, random forest, gradient
boosting) see :mod:`backend.ml.baselines`; their scores surface on
``InferenceResult.classical_probability``.

This file is kept as a tombstone so the import error names its replacement.
"""

from __future__ import annotations

__all__: list[str] = []


def __getattr__(name: str) -> object:
    raise AttributeError(
        f"backend.services.classical_ml_service.{name} was removed because it "
        "fabricated image features without reading the image. Use "
        "backend.services.inference_service.InferenceService (real pipeline) or "
        "backend.ml.baselines (classical comparison models) instead."
    )
