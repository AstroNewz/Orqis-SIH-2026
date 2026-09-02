from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class PatientBase(BaseModel):
    clinic_id: str = Field(default="default_clinic", description="Clinical unit identifier")


class PatientCreate(PatientBase):
    id: str | None = Field(default=None, description="Optional custom UUID v4")


class PatientResponse(PatientBase):
    id: str
    created_at: datetime
    is_active: bool

    model_config = ConfigDict(from_attributes=True)
