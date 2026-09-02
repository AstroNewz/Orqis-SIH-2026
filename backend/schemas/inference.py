"""Typed inference results returned by the screening pipeline.

PART 13 requires a structured result rather than a bare primitive, and PART 33
forbids ``Map<String, dynamic>`` / untyped dicts as the primary domain model. Every
field a caller needs to interpret or audit a screening is declared here.

The three probabilities are deliberately all present. ``raw_score`` is the
quantum measurement, ``probability_uncalibrated`` is that score read naively as a
probability, and ``probability`` is the calibrated output. Reporting only the last
one hides whether calibration did anything; reporting only the first would be
clinically wrong. Keeping all three makes the transformation auditable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER = (
    "AI-assisted oral-cancer screening risk estimate. This is not a diagnosis. "
    "The result does not replace professional clinical assessment or "
    "histopathological confirmation, and the operating thresholds used are "
    "experimental engineering choices rather than clinically validated cut-offs."
)


class QualitySummary(BaseModel):
    """Image quality control outcome, patient-safe."""

    verdict: str
    passed: bool
    reasons: List[str] = Field(default_factory=list)
    """Human-readable failure reasons, e.g. "image is too blurred". Safe to show:
    they describe the capture, not the model internals."""
    metrics: Dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class CalibrationSummary(BaseModel):
    """What the calibration stage did, and on what basis."""

    method: str
    is_calibrated: bool
    fitted_on: str = ""
    screening_threshold: float
    high_risk_threshold: float
    bands_source: str
    """``calibration_artifact`` when the bands came from this model version's own
    out-of-fold derivation, ``configured_fallback`` when they came from settings.
    A result carrying the fallback should be read with more caution."""
    thresholds_are_clinically_validated: bool = False


class QuantumSummary(BaseModel):
    """Execution provenance for the quantum stage.

    Present so a result is reproducible and so the Flutter client can display *how*
    inference ran -- but the client never has to branch on it. PART 29 requires the
    frontend to be identical across execution modes.
    """

    n_qubits: int
    circuit_depth: int
    n_ansatz_layers: int
    n_parameters: int
    amplitude_dimension: int
    execution_mode: str
    backend_name: str
    shots: Optional[int] = None
    requested_mode: Optional[str] = None
    fell_back: bool = False
    fallback_reason: Optional[str] = None
    is_exact: bool = True
    optimizer: Optional[str] = None


class FeatureSummary(BaseModel):
    """Which feature space the inference ran in."""

    feature_mode: str
    extractor: str
    n_image_features: int = 0
    n_clinical_features: int = 0
    n_reduced_features: int = 0
    reduction_method: str = ""
    preprocessing_version: str = ""
    pipeline_version: str = ""
    clinical_features_supplied: bool = False


class InferenceResult(BaseModel):
    """One screening inference, fully traceable to a model version.

    ``model_config`` disables Pydantic's ``model_`` namespace protection so
    ``model_version`` can keep its name: it is the term the rest of the repository,
    the artifact store and the API contract all use.
    """

    inference_id: str
    model_version: str
    created_at: datetime

    probability: float = Field(ge=0.0, le=1.0)
    """The calibrated probability. This is the only value that should be shown as a
    risk estimate."""
    probability_uncalibrated: float = Field(ge=0.0, le=1.0)
    raw_score: float = Field(ge=0.0, le=1.0)
    expectation_value: float = Field(ge=-1.0, le=1.0)

    risk_level: str
    classification: str
    details: str

    classical_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    """The persisted logistic-regression baseline's probability for the *same* reduced
    feature vector, when that baseline is available.

    Reported for comparison only -- PART 10 requires the quantum contribution to be
    evaluated against classical baselines rather than assumed superior, and on this
    dataset the baselines win (ISS-008). It is never blended into
    :attr:`probability`: no fusion weight has been validated, and a blended number
    would not correspond to any model in the evaluation report.

    **It is not calibrated.** The baselines are fitted with ``class_weight="balanced"``
    to cope with a 4.7% positive rate, which deliberately inflates their probabilities
    away from the observed base rate. This value is comparable to other baseline
    scores by *ranking*; it is not comparable in magnitude to :attr:`probability`, and
    must not be shown to a patient as a risk percentage."""
    classical_model: Optional[str] = None

    calibration: CalibrationSummary
    quantum: QuantumSummary
    features: FeatureSummary
    quality: Optional[QualitySummary] = None

    execution_time_ms: float = Field(ge=0.0)
    quantum_time_ms: float = Field(ge=0.0)
    preprocessing_time_ms: float = Field(ge=0.0)

    is_mock: bool = False
    """True only for the explicitly-labelled development path that runs without
    trained artifacts. A mock result is never presented as a real inference."""
    disclaimer: str = DISCLAIMER

    model_config = ConfigDict(protected_namespaces=())
