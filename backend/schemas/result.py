from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, AliasChoices, computed_field

from backend.schemas.inference import QuantumVisualSummary

SCREENING_DISCLAIMER = (
    "AI-assisted oral-cancer screening risk estimate. This is not a diagnosis and "
    "does not replace professional clinical assessment or histopathological "
    "confirmation. Consult a qualified clinician about this result."
)


class AssessmentResultResponse(BaseModel):
    """Assessment result schema compatible with Group 1 Flutter AssessmentResult model.

    The original Group 1 field set is unchanged -- the existing Flutter client keeps
    deserialising it. Everything under "Provenance" is additive and optional, so an
    older client ignores it while a newer one can show which model produced the number
    and how it was calibrated (PART 18).
    """

    id: str
    assessmentId: str = Field(
        validation_alias=AliasChoices("screening_id", "assessment_id", "assessmentId"),
        serialization_alias="assessmentId",
    )
    riskLevel: str = Field(
        validation_alias=AliasChoices("risk_level", "riskLevel"),
        serialization_alias="riskLevel",
    )
    details: str

    # Extended metadata fields
    classicalProbability: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("classical_probability", "classicalProbability"),
        serialization_alias="classicalProbability",
    )
    quantumProbability: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("quantum_probability", "quantumProbability"),
        serialization_alias="quantumProbability",
    )
    finalProbability: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("final_probability", "finalProbability"),
        serialization_alias="finalProbability",
    )
    threshold: Optional[float] = 0.50
    classification: Optional[str] = None
    modelVersion: Optional[str] = Field(
        default="v1.0.0-qml",
        validation_alias=AliasChoices("model_version", "modelVersion"),
        serialization_alias="modelVersion",
    )
    quantumQubits: Optional[int] = Field(
        default=8,
        validation_alias=AliasChoices("quantum_qubits", "quantumQubits"),
        serialization_alias="quantumQubits",
    )
    quantumShots: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("quantum_shots", "quantumShots"),
        serialization_alias="quantumShots",
        description="Null under exact statevector simulation, where nothing is sampled.",
    )
    executionTimeMs: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("execution_time_ms", "executionTimeMs"),
        serialization_alias="executionTimeMs",
    )
    isMock: bool = Field(
        default=False,
        validation_alias=AliasChoices("is_mock", "isMock"),
        serialization_alias="isMock",
    )
    createdAt: Optional[datetime] = Field(
        default=None,
        validation_alias=AliasChoices("created_at", "createdAt"),
        serialization_alias="createdAt",
    )

    # ----------------------------------------------------- Provenance (PART 18)
    inferenceId: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("inference_id", "inferenceId"),
        serialization_alias="inferenceId",
    )
    rawScore: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("raw_score", "rawScore"),
        serialization_alias="rawScore",
        description="Quantum measurement before calibration. Not a probability.",
    )
    probabilityUncalibrated: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("probability_uncalibrated", "probabilityUncalibrated"),
        serialization_alias="probabilityUncalibrated",
    )
    highRiskThreshold: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("high_risk_threshold", "highRiskThreshold"),
        serialization_alias="highRiskThreshold",
    )
    calibrationMethod: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("calibration_method", "calibrationMethod"),
        serialization_alias="calibrationMethod",
    )
    bandsSource: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("bands_source", "bandsSource"),
        serialization_alias="bandsSource",
    )
    executionMode: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("execution_mode", "executionMode"),
        serialization_alias="executionMode",
        description="ideal_simulation | noisy_simulation | ibm_hardware | mock",
    )
    backendName: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("backend_name", "backendName"),
        serialization_alias="backendName",
    )
    circuitDepth: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("circuit_depth", "circuitDepth"),
        serialization_alias="circuitDepth",
    )
    featureMode: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("feature_mode", "featureMode"),
        serialization_alias="featureMode",
        description="image_only | clinical_only | multimodal",
    )
    quantumTimeMs: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("quantum_time_ms", "quantumTimeMs"),
        serialization_alias="quantumTimeMs",
    )

    # ----------------------------------- Displayed (headline) verdict (DEC-034)
    # The band the client shows as the verdict. When ``primaryModel`` is the classical
    # baseline it is the strongest validated model on this dataset; ``primaryCalibrated``
    # is then False and ``primaryProbability`` is a ranking score, never a percentage.
    # ``finalProbability`` above remains the calibrated quantum probability.
    primaryModel: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("primary_model", "primaryModel"),
        serialization_alias="primaryModel",
    )
    primaryRiskLevel: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("primary_risk_level", "primaryRiskLevel"),
        serialization_alias="primaryRiskLevel",
    )
    primaryProbability: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("primary_probability", "primaryProbability"),
        serialization_alias="primaryProbability",
        description="Ranking score behind the headline band. Not a percentage when primaryCalibrated is false.",
    )
    primaryThreshold: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("primary_threshold", "primaryThreshold"),
        serialization_alias="primaryThreshold",
    )
    primaryCalibrated: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices("primary_calibrated", "primaryCalibrated"),
        serialization_alias="primaryCalibrated",
    )

    # -------------------------- EXPERIMENTAL secondary quantum-visual (E2/DEC-035) --
    # A real, executed 8-qubit quantum-visual demonstrator (Candidate B). It is purely
    # additive telemetry: it never headlines the verdict (``primaryRiskLevel`` does) and
    # never feeds ``finalProbability``, ``threshold`` or any FHIR value -- see
    # :class:`~backend.schemas.inference.QuantumVisualSummary` and DEC-033/DEC-034.
    #
    # It is attached to a *fresh* ``/screening/analyze`` response from the live inference
    # (see ``ScreeningService.analyze_screening``); it is deliberately NOT persisted, so a
    # result re-read from ``/results/{id}`` or ``/patients/{id}/history`` carries ``None``.
    # Persisting it would require a schema migration on an existing DB for a non-clinical,
    # experimental signal; keeping it live-only is both safer and honest about its role.
    #
    # The inner object keeps its native ``snake_case`` field names (the reused
    # ``QuantumVisualSummary`` contract), unlike this envelope's camelCase -- a single
    # source of truth for those fields rather than a duplicated projection that could drift.
    quantumVisual: Optional[QuantumVisualSummary] = Field(
        default=None,
        validation_alias=AliasChoices("quantum_visual", "quantumVisual"),
        serialization_alias="quantumVisual",
        description=(
            "EXPERIMENTAL secondary E2 quantum-visual demonstrator (DEC-035). Additive "
            "telemetry only; never the headline verdict and never fed into any "
            "probability/threshold/FHIR value. Present only on a fresh analyze response; "
            "null when a result is re-read from history."
        ),
    )

    @computed_field(  # type: ignore[prop-decorator]
        alias="disclaimer",
        description="Required patient-facing wording. Sent with every result.",
    )
    @property
    def disclaimer(self) -> str:
        """Attached server-side rather than left to the client to remember.

        PART 35 requires the output to be described as an AI-assisted screening risk
        estimate; making it a computed field means no response can omit it, including
        the ones read back from ``/api/results/{id}``.
        """
        return SCREENING_DISCLAIMER

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )
