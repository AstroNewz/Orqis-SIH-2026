from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field, AliasChoices
from backend.schemas.result import AssessmentResultResponse


class ScreeningCreate(BaseModel):
    """Payload for submitting a screening image or features for analysis.

    Two ways to supply the image, in order of preference:

    1. ``features`` -- the image descriptor computed on the device. PART 19 prefers
       this: the photograph never leaves the phone. The vector must have exactly the
       length the served model expects (``GET /api/model/info`` reports it as
       ``expected_descriptor_dimension``). A wrong length is rejected with 422 rather
       than padded or truncated, because either would silently change the features
       the model was trained on.
    2. ``image_path`` -- a path the backend can read, from ``POST
       /api/screening/upload``. The backend then runs quality control, ROI extraction
       and feature extraction itself.
    """

    patient_id: Optional[str] = Field(default=None, description="Optional UUID v4 for the patient")
    image_path: str = Field(description="Local or remote path to the image")
    scan_type: str = Field(default="Intra-oral Scan", description="Type of scan")

    # Clinical risk factors. Tri-state on purpose: False means the patient was asked
    # and said no, None means it was not collected. The encoder keeps those apart.
    smoking_history: Optional[bool] = Field(default=None)
    alcohol_consumption: Optional[bool] = Field(default=None)
    betel_quid: Optional[bool] = Field(default=None)
    age: Optional[float] = Field(default=None, ge=0, le=120, description="Years, if collected")
    sex: Optional[str] = Field(default=None, description="'male' / 'female', if collected")

    features: Optional[List[float]] = Field(
        default=None,
        description=(
            "Image descriptor from the on-device extractor. Length must equal the "
            "served model's expected_descriptor_dimension."
        ),
    )

    # Mode flags
    is_mock: bool = Field(
        default=False,
        description=(
            "Request the labelled development stub instead of a real inference. "
            "Results are marked MOCK and carry no clinical meaning."
        ),
    )


class AssessmentResponse(BaseModel):
    """Assessment schema matching Group 1 Flutter Assessment model."""
    id: str
    imagePath: str = Field(
        validation_alias=AliasChoices("image_path", "imagePath"),
        serialization_alias="imagePath",
    )
    timestamp: datetime = Field(
        validation_alias=AliasChoices("created_at", "timestamp"),
        serialization_alias="timestamp",
    )
    type: str = Field(
        validation_alias=AliasChoices("scan_type", "type"),
        serialization_alias="type",
    )

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )


class HistoryEntryResponse(BaseModel):
    """History entry schema matching Group 1 Flutter HistoryEntry model."""
    assessment: AssessmentResponse
    result: Optional[AssessmentResultResponse] = None

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )
