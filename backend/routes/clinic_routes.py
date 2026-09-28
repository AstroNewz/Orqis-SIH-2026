"""Read-only clinic surface for the web portal.

Every route here is a GET, requires a portal account, and is scoped to the caller's own
``clinic_id`` via :func:`~backend.auth.dependencies.require_clinic_access`. The clinic id
in the path is checked against the token's account rather than trusted, so the path
parameter is a readable URL, not an authorisation decision.

Nothing in this module writes to the database, touches the ML pipeline, or reads the
frozen evaluation partition. It only projects rows the screening pipeline already wrote
into the shapes a clinician's dashboard needs.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.auth.dependencies import require_clinic_access, require_clinic_user
from backend.db import crud
from backend.db.session import get_db
from backend.models.clinic_user import ClinicUser
from backend.schemas.clinic import (
    ClinicPatientSummary,
    ClinicPatientsResponse,
    ClinicStatsResponse,
    WorklistItemResponse,
    WorklistResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/clinics", tags=["Clinic portal"])

# Page sizes are capped server-side. A client asking for 10 000 rows gets 200; the
# portal's paging is then a real constraint rather than a suggestion.
MAX_PAGE_SIZE = 200


@router.get("/{clinic_id}/screenings", response_model=WorklistResponse)
def list_screenings(
    clinic_id: str,
    limit: int = Query(default=50, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    risk: Optional[str] = Query(
        default=None,
        max_length=32,
        description="Filter on the displayed band, e.g. 'HIGH' or 'MODERATE RISK'.",
    ),
    db: Session = Depends(get_db),
    user: ClinicUser = Depends(require_clinic_user),
) -> WorklistResponse:
    """The clinic worklist: screenings newest first, with their verdict when there is one.

    A screening with no result is included and reports ``hasResult: false`` -- a capture
    that never produced a verdict is exactly what a clinician needs to see, and dropping
    it would make the worklist quietly disagree with the screening count on the dashboard.
    """
    require_clinic_access(clinic_id, user)

    rows = crud.list_clinic_screenings(
        db, clinic_id=clinic_id, limit=limit, offset=offset, risk=risk
    )

    items = []
    for screening, result in rows:
        # The displayed band is the primary model's when the row has one (DEC-034);
        # older rows predate the primary columns and fall back to the quantum band.
        displayed_band = None
        if result is not None:
            displayed_band = result.primary_risk_level or result.risk_level

        items.append(
            WorklistItemResponse(
                screeningId=screening.id,
                patientId=screening.patient_id,
                scanType=screening.scan_type,
                status=screening.status,
                createdAt=screening.created_at,
                riskLevel=displayed_band,
                primaryModel=result.primary_model if result else None,
                primaryCalibrated=result.primary_calibrated if result else None,
                # The calibrated quantum probability. Kept a secondary readout here for
                # the same reason as on the result screen: it is the only calibrated
                # number, and it is not the headline verdict.
                finalProbability=result.final_probability if result else None,
                threshold=result.threshold if result else None,
                modelVersion=result.model_version if result else None,
                isMock=bool(result.is_mock) if result else False,
                hasResult=result is not None,
            )
        )

    return WorklistResponse(
        items=items,
        total=crud.count_clinic_screenings(db, clinic_id=clinic_id, risk=risk),
        limit=limit,
        offset=offset,
    )


@router.get("/{clinic_id}/patients", response_model=ClinicPatientsResponse)
def list_patients(
    clinic_id: str,
    limit: int = Query(default=50, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: ClinicUser = Depends(require_clinic_user),
) -> ClinicPatientsResponse:
    """The clinic's patient roster with screening counts and each patient's latest band.

    Patients are pseudonymous: the roster carries a UUID, counts and timestamps, and no
    demographic field whatsoever. Age and sex are accepted for inference but never
    stored (``backend/models/screening.py:32-34``), so there is nothing else to show.
    """
    require_clinic_access(clinic_id, user)

    roster = crud.list_clinic_patients(db, clinic_id=clinic_id, limit=limit, offset=offset)
    items = [
        ClinicPatientSummary(
            patientId=str(row["patient_id"]),
            screeningCount=int(row["screening_count"]),
            lastScreeningAt=row["last_screening_at"],
            latestRiskLevel=row["latest_risk_level"],
            createdAt=row["created_at"],
            isActive=bool(row["is_active"]),
        )
        for row in roster
    ]

    return ClinicPatientsResponse(
        items=items,
        total=crud.count_clinic_patients(db, clinic_id=clinic_id),
        limit=limit,
        offset=offset,
    )


@router.get("/{clinic_id}/stats", response_model=ClinicStatsResponse)
def clinic_stats(
    clinic_id: str,
    db: Session = Depends(get_db),
    user: ClinicUser = Depends(require_clinic_user),
) -> ClinicStatsResponse:
    """Dashboard counters: screenings, patients, band distribution, and mock count.

    These are throughput counters, not performance metrics. They describe what this
    clinic has captured; they say nothing about model accuracy, which lives in the
    frozen evaluation artifacts and is not computed or re-derived here.
    """
    require_clinic_access(clinic_id, user)

    band_counts = crud.clinic_band_counts(db, clinic_id=clinic_id)
    total_screenings = crud.count_clinic_screenings(db, clinic_id=clinic_id)
    completed = sum(band_counts.values())

    return ClinicStatsResponse(
        clinicId=clinic_id,
        totalScreenings=total_screenings,
        totalPatients=crud.count_clinic_patients(db, clinic_id=clinic_id),
        completedScreenings=completed,
        # Never negative even if a result somehow outlives its screening row.
        pendingScreenings=max(0, total_screenings - completed),
        mockResults=crud.count_clinic_mock_results(db, clinic_id=clinic_id),
        bandCounts=band_counts,
    )
