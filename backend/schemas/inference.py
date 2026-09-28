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


class QuantumVisualSummary(BaseModel):
    """EXPERIMENTAL secondary quantum *visual* signal (E2 / DEC-035, Candidate B).

    A real, executed 8-qubit angle-encoded quantum feature map applied to the MobileNet
    ROI embedding, read out through 16 fixed local Z observables and a small classical
    logistic head. This is a SECONDARY, experimental readout only. It is additive and
    strictly non-clinical:

    * it never headlines the verdict (that is :attr:`InferenceResult.primary_risk_level`);
    * it never feeds :attr:`InferenceResult.probability`, the persisted
      ``final_probability`` or any FHIR value -- those stay quantum-VQC-calibrated;
    * it makes no quantum-advantage claim (DEC-033/DEC-034 authorise a *demonstrator*
      only), which is why its equal-dimension RFF control and raw PCA-8 control are
      carried beside it and it is never reported alone;
    * if the E2 stage fails or is unavailable at runtime this whole object is simply
      ``None`` and the classical primary path is unaffected.

    Everything here is either read back from the built circuit / persisted artifact
    (provenance) or produced by executing the circuit on *this* request's embedding; no
    value is mocked, hardcoded, or a fabricated accuracy.
    """

    # -- role, kept unambiguous so a client can never promote this to the headline
    role: str = (
        "EXPERIMENTAL secondary quantum visual demonstrator (E2/DEC-035). Not the "
        "clinical decision-maker; the classical primary verdict is unaffected by it."
    )
    is_secondary_experimental: bool = True
    advantage_claimed: bool = False

    # -- circuit execution provenance (read back from the built circuit, not asserted)
    circuit_version: str
    n_qubits: int
    n_reuploading_blocks: int
    n_encoding_parameters: int
    """The 8 ``Ry`` data angles. These are *data*, not weights."""
    n_trainable_parameters: int = 0
    """Zero by construction: the circuit is a fixed function of the input angles."""
    two_qubit_gate_count: int
    circuit_depth: int
    state_dimension: int
    n_quantum_features: int
    backend_name: str
    seed: int
    observable_labels: List[str] = Field(default_factory=list)

    # -- this request's real execution
    secondary_probability: float = Field(ge=0.0, le=1.0)
    """The experimental quantum-visual secondary probability for this image. Reported
    for interest only; not calibrated to the clinical base rate and never shown as the
    patient-facing risk percentage."""
    quantum_feature_vector: List[float] = Field(default_factory=list)
    """The 16 local-observable expectations, each in ``[-1, 1]``."""
    max_abs_connected_correlation: float
    """Entanglement witness: the largest ``|<Z_iZ_j> - <Z_i><Z_j>|`` over the ring
    edges for this image. Nonzero is direct evidence the CZ ring produced correlations a
    separable angle encoding could not. NOT an advantage metric."""
    mean_abs_connected_correlation_per_edge: List[float] = Field(default_factory=list)
    execution_time_ms: float = Field(ge=0.0)

    # -- exactness provenance recorded at fit time (fast |psi|^2 @ diag^T vs Aer)
    aer_max_abs_deviation: Optional[float] = None
    aer_tolerance: Optional[float] = None
    aer_validation_passed: Optional[bool] = None

    # -- matched controls on THIS image, so the quantum number is never shown alone
    rff_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    """Equal-dimension (16) random-Fourier control on the same PCA-8 input and the same
    head. DEC-033's honest "what would any fixed nonlinear map get?" analogue."""
    pca_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    """Raw PCA-8 handed straight to the head: "does the quantum map add anything over the
    numbers it was built from?" """

    # -- static evaluation verdict, populated when the TRAIN+VALIDATION E2 report is
    #    discoverable (the held-out test partition is never involved); None otherwise.
    keep_kill_decision: Optional[str] = None
    quantum_validation_pr_auc: Optional[float] = None
    rff_validation_pr_auc: Optional[float] = None
    pca_validation_pr_auc: Optional[float] = None
    fitted_on_condition: Optional[str] = None
    preprocessor_artifact_hash: Optional[str] = None


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

    # ------------------------------------------- primary (display headline) verdict
    # The frontend headlines the strongest *validated* model's risk band. On this
    # dataset that is the classical baseline: it ranks above the quantum VQC, which is
    # last of four (ISS-008 / DEC-034). Only the *band* is promoted -- the classical
    # score is an uncalibrated ranking value (see :attr:`classical_probability`) and is
    # never shown as a percentage. The reported calibrated probability
    # (:attr:`probability`, and the stored ``final_probability`` / FHIR value) therefore
    # stays the quantum-calibrated one, which is the only calibrated number available;
    # the quantum band remains on :attr:`risk_level` as the honest secondary readout.
    primary_model: str = "quantum_vqc_calibrated"
    """Which model's band the client should headline. ``logistic_regression`` when the
    validated classical baseline and its validation-selected operating point are both
    available, otherwise the quantum VQC (also the mock-path fallback)."""
    primary_risk_level: Optional[str] = None
    """The headline risk band. For the classical model, a two-band verdict
    (``LOW RISK`` / ``MODERATE RISK``) from :attr:`primary_probability` against
    :attr:`primary_threshold`: there is no validation-safe classical high-risk cut-off,
    so no ``HIGH RISK`` band is claimed. ``None`` falls back to :attr:`risk_level`."""
    primary_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    """The score the headline band was derived from. When
    ``primary_model == "logistic_regression"`` this equals :attr:`classical_probability`
    -- comparable by *ranking* only, NOT a calibrated risk percentage
    (:attr:`primary_calibrated` is then ``False``). ``None`` falls back to
    :attr:`probability`. It is *not* the persisted ``final_probability``; that stays the
    calibrated quantum probability."""
    primary_threshold: Optional[float] = None
    """Screening operating point used to band :attr:`primary_probability`. For the
    classical model this is the validation-selected threshold (``chosen_on`` =
    ``"validation"`` in the evaluation artifact); ``None`` falls back to the quantum
    calibration threshold."""
    primary_calibrated: bool = True
    """``False`` when :attr:`primary_probability` is an uncalibrated ranking score (the
    classical baseline). A client must not render it as a percentage in that case; the
    band is the patient-facing verdict, and the calibrated percentage shown elsewhere is
    the quantum :attr:`probability`."""

    calibration: CalibrationSummary
    quantum: QuantumSummary
    features: FeatureSummary
    quality: Optional[QualitySummary] = None

    quantum_visual: Optional[QuantumVisualSummary] = None
    """EXPERIMENTAL secondary E2 quantum-visual demonstrator (DEC-035, Candidate B).

    ``None`` whenever the E2 stage is unavailable or fails at runtime -- the classical
    primary path is unaffected either way. When present it is additive telemetry only:
    it never headlines the verdict (:attr:`primary_risk_level` does) and never feeds
    :attr:`probability`, the persisted ``final_probability`` or any FHIR value. See
    :class:`QuantumVisualSummary`."""

    execution_time_ms: float = Field(ge=0.0)
    quantum_time_ms: float = Field(ge=0.0)
    preprocessing_time_ms: float = Field(ge=0.0)

    is_mock: bool = False
    """True only for the explicitly-labelled development path that runs without
    trained artifacts. A mock result is never presented as a real inference."""
    disclaimer: str = DISCLAIMER

    model_config = ConfigDict(protected_namespaces=())
