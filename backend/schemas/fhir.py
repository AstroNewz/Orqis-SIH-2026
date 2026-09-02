from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FHIRCoding(BaseModel):
    system: str = "http://snomed.info/sct"
    code: str
    display: str


class FHIRCodeableConcept(BaseModel):
    coding: List[FHIRCoding]
    text: str


class FHIRObservation(BaseModel):
    """HL7 FHIR R4 Observation Resource for model-derived screening probability."""
    resourceType: str = "Observation"
    id: str
    status: str = "final"
    code: FHIRCodeableConcept = Field(
        default_factory=lambda: FHIRCodeableConcept(
            coding=[
                FHIRCoding(
                    code="371569005",
                    display="Oral cavity cancer screening assessment",
                )
            ],
            text="Oral Cavity Cancer Screening Probability",
        )
    )
    subject: Dict[str, str]
    effectiveDateTime: str
    valueQuantity: Dict[str, Any]
    interpretation: Optional[List[FHIRCodeableConcept]] = None
    note: Optional[List[Dict[str, str]]] = None


class FHIRRiskAssessment(BaseModel):
    """HL7 FHIR R4 RiskAssessment Resource for risk-oriented clinical interpretation."""
    resourceType: str = "RiskAssessment"
    id: str
    status: str = "final"
    subject: Dict[str, str]
    occurrenceDateTime: str
    basis: List[Dict[str, str]]
    prediction: List[Dict[str, Any]]
    note: Optional[List[Dict[str, str]]] = None
