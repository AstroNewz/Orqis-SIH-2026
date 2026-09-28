"""Schemas for the clinic/hospital web portal.

Additive: nothing here is referenced by the Flutter client or by any pre-existing
route, so the mobile contract in :mod:`backend.schemas.screening` and
:mod:`backend.schemas.result` is untouched.

Field names are camelCase on the wire, matching the existing result contract that the
portal's TypeScript types already mirror.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class LoginRequest(BaseModel):
    """Portal sign-in payload."""

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)
    """Bounded so an oversized body is rejected by validation rather than handed to
    bcrypt, which truncates at 72 bytes anyway."""


class ClinicUserResponse(BaseModel):
    """The signed-in account. Never carries the password hash."""

    id: str
    email: str
    clinicId: str = Field(serialization_alias="clinicId")
    fullName: Optional[str] = Field(default=None, serialization_alias="fullName")
    role: str

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class LoginResponse(BaseModel):
    """Issued token plus the identity the portal needs to render its header.

    The clinic id is returned so the client never has to guess or ask for it; the
    server still reads the authoritative value from the token on every request, so a
    client that tampers with this copy gains nothing.
    """

    accessToken: str = Field(serialization_alias="accessToken")
    tokenType: str = Field(default="bearer", serialization_alias="tokenType")
    expiresInMinutes: int = Field(serialization_alias="expiresInMinutes")
    user: ClinicUserResponse

    model_config = ConfigDict(populate_by_name=True)


class WorklistItemResponse(BaseModel):
    """One row of the clinic worklist.

    ``riskLevel`` is the *displayed* band (DEC-034): the primary model's band when the
    result has one, else the quantum band. ``primaryCalibrated`` travels with it so the
    table can suppress the percentage for an uncalibrated ranking score exactly as the
    result screen does -- a worklist that printed a ranking as a percentage would be
    the single most misleading thing in the portal.
    """

    screeningId: str = Field(serialization_alias="screeningId")
    patientId: str = Field(serialization_alias="patientId")
    scanType: str = Field(serialization_alias="scanType")
    status: str
    createdAt: datetime = Field(serialization_alias="createdAt")

    riskLevel: Optional[str] = Field(default=None, serialization_alias="riskLevel")
    primaryModel: Optional[str] = Field(default=None, serialization_alias="primaryModel")
    primaryCalibrated: Optional[bool] = Field(
        default=None, serialization_alias="primaryCalibrated"
    )
    finalProbability: Optional[float] = Field(
        default=None,
        serialization_alias="finalProbability",
        description="Calibrated quantum probability. Secondary readout, not the verdict.",
    )
    threshold: Optional[float] = None
    modelVersion: Optional[str] = Field(default=None, serialization_alias="modelVersion")
    isMock: bool = Field(default=False, serialization_alias="isMock")
    hasResult: bool = Field(default=False, serialization_alias="hasResult")

    model_config = ConfigDict(populate_by_name=True)


class WorklistResponse(BaseModel):
    """A page of the worklist with the totals needed to page through it."""

    items: List[WorklistItemResponse]
    total: int
    limit: int
    offset: int

    model_config = ConfigDict(populate_by_name=True)


class ClinicPatientSummary(BaseModel):
    """One row of the patient roster.

    Carries no name, age, or sex: patients are pseudonymous by design (PART 19, and
    ``backend/models/screening.py:32-34`` on why age and sex are never stored), so the
    roster identifies them by UUID and nothing else.
    """

    patientId: str = Field(serialization_alias="patientId")
    screeningCount: int = Field(serialization_alias="screeningCount")
    lastScreeningAt: Optional[datetime] = Field(
        default=None, serialization_alias="lastScreeningAt"
    )
    latestRiskLevel: Optional[str] = Field(
        default=None, serialization_alias="latestRiskLevel"
    )
    createdAt: datetime = Field(serialization_alias="createdAt")
    isActive: bool = Field(default=True, serialization_alias="isActive")

    model_config = ConfigDict(populate_by_name=True)


class ClinicPatientsResponse(BaseModel):
    """A page of the patient roster."""

    items: List[ClinicPatientSummary]
    total: int
    limit: int
    offset: int

    model_config = ConfigDict(populate_by_name=True)


class ClinicStatsResponse(BaseModel):
    """Dashboard counters for one clinic.

    ``bandCounts`` is keyed by the band string as stored (``"HIGH RISK"`` and friends)
    rather than by a normalised enum, so the portal looks a band up with the same value
    it renders. ``pendingScreenings`` is the difference between screenings and results:
    a capture that never produced a verdict is visible instead of silently absent.
    """

    clinicId: str = Field(serialization_alias="clinicId")
    totalScreenings: int = Field(serialization_alias="totalScreenings")
    totalPatients: int = Field(serialization_alias="totalPatients")
    completedScreenings: int = Field(serialization_alias="completedScreenings")
    pendingScreenings: int = Field(serialization_alias="pendingScreenings")
    mockResults: int = Field(
        serialization_alias="mockResults",
        description="Results flagged is_mock. Development stubs, not clinical throughput.",
    )
    bandCounts: Dict[str, int] = Field(serialization_alias="bandCounts")

    model_config = ConfigDict(populate_by_name=True)
