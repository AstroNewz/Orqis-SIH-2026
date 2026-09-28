import uuid
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, select

from backend.db.base import Base
from backend.db.session import engine
from backend.models.patient import Patient
from backend.models.screening import Screening
from backend.models.result import ScreeningResult
from backend.models.audit_log import AuditLog
from backend.models.clinic_user import ClinicUser
from backend.schemas.inference import InferenceResult
from backend.schemas.screening import ScreeningCreate

# The band a clinician sees is the strongest validated model's band (DEC-034):
# ``primary_risk_level`` when the row has one, else the quantum ``risk_level`` for
# rows written before the primary columns existed. Every clinic query below ranks,
# filters, and counts on this expression so the portal never disagrees with the
# result screen about which band a screening is in.
DISPLAYED_BAND = func.coalesce(
    ScreeningResult.primary_risk_level,
    ScreeningResult.risk_level,
)


def init_db() -> None:
    """Initialize all database tables defined in Base metadata."""
    Base.metadata.create_all(bind=engine)


def get_or_create_patient(db: Session, patient_id: Optional[str] = None, clinic_id: str = "default_clinic") -> Patient:
    """Get an existing patient or create a new pseudonymous patient record."""
    if patient_id:
        stmt = select(Patient).where(Patient.id == patient_id)
        patient = db.scalar(stmt)
        if patient:
            return patient

    new_id = patient_id or str(uuid.uuid4())
    patient = Patient(id=new_id, clinic_id=clinic_id, is_active=True)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def create_screening(db: Session, screening_in: ScreeningCreate, patient: Patient) -> Screening:
    """Create a new screening record."""
    screening = Screening(
        id=str(uuid.uuid4()),
        patient_id=patient.id,
        image_path=screening_in.image_path,
        scan_type=screening_in.scan_type,
        status="PENDING",
        smoking_history=screening_in.smoking_history,
        alcohol_consumption=screening_in.alcohol_consumption,
        betel_quid=screening_in.betel_quid,
    )
    db.add(screening)
    db.commit()
    db.refresh(screening)
    return screening


def get_screening(db: Session, screening_id: str) -> Optional[Screening]:
    """Retrieve a screening record by ID."""
    stmt = select(Screening).where(Screening.id == screening_id)
    return db.scalar(stmt)


def get_patient_screenings(db: Session, patient_id: str) -> List[Screening]:
    """Retrieve all screenings for a specific patient in chronological order."""
    stmt = select(Screening).where(Screening.patient_id == patient_id).order_by(Screening.created_at.desc())
    return list(db.scalars(stmt).all())


# --------------------------------------------------------- Clinic portal (read-only)
# Everything below powers the clinic/hospital web portal. All of it is read-only and
# every query joins through ``Patient.clinic_id``, so a caller can only ever see the
# clinic on their own token -- there is no code path that takes a clinic id from a
# query string without checking it against the caller first (see
# ``backend.routes.clinic_routes``).


def list_clinic_screenings(
    db: Session,
    clinic_id: str,
    limit: int = 50,
    offset: int = 0,
    risk: Optional[str] = None,
) -> List[Tuple[Screening, Optional[ScreeningResult]]]:
    """Return a clinic's screenings, newest first, paired with their result.

    The result is LEFT joined: a screening that is still ``PENDING`` -- or whose
    analysis failed -- must still appear on the worklist, otherwise a clinician has no
    way to see that a capture never produced a verdict.

    ``risk`` filters on the *displayed* band (:data:`DISPLAYED_BAND`) and matches a
    substring, so ``"HIGH"`` and ``"HIGH RISK"`` both work. It is parameterised, never
    interpolated into SQL.
    """
    stmt = (
        select(Screening, ScreeningResult)
        .join(Patient, Patient.id == Screening.patient_id)
        .outerjoin(ScreeningResult, ScreeningResult.screening_id == Screening.id)
        .where(Patient.clinic_id == clinic_id)
        .order_by(Screening.created_at.desc())
    )

    if risk:
        stmt = stmt.where(func.upper(DISPLAYED_BAND).like(f"%{risk.strip().upper()}%"))

    stmt = stmt.limit(max(1, limit)).offset(max(0, offset))
    return [(row[0], row[1]) for row in db.execute(stmt).all()]


def count_clinic_screenings(
    db: Session,
    clinic_id: str,
    risk: Optional[str] = None,
) -> int:
    """Total screenings matching the same filter as :func:`list_clinic_screenings`.

    Returned alongside a page so the portal can render "showing 20 of 137" and decide
    whether a "load more" control belongs on screen at all.
    """
    stmt = (
        select(func.count(Screening.id))
        .join(Patient, Patient.id == Screening.patient_id)
        .outerjoin(ScreeningResult, ScreeningResult.screening_id == Screening.id)
        .where(Patient.clinic_id == clinic_id)
    )
    if risk:
        stmt = stmt.where(func.upper(DISPLAYED_BAND).like(f"%{risk.strip().upper()}%"))
    return int(db.scalar(stmt) or 0)


def list_clinic_patients(
    db: Session,
    clinic_id: str,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, object]]:
    """Return a clinic's patient roster with per-patient screening summaries.

    Three bounded queries rather than one per patient: the page of patients, one
    grouped aggregate for counts and last-seen timestamps, and one pass over that
    page's screenings to pick each patient's most recent band. Patients with no
    screening yet are included with a zero count and a null band -- they exist in the
    roster the moment a record is opened, and hiding them would make the roster
    disagree with the patient count on the dashboard.
    """
    patients = list(
        db.scalars(
            select(Patient)
            .where(Patient.clinic_id == clinic_id)
            .order_by(Patient.created_at.desc())
            .limit(max(1, limit))
            .offset(max(0, offset))
        ).all()
    )
    if not patients:
        return []

    patient_ids = [p.id for p in patients]

    counts = {
        row[0]: (int(row[1]), row[2])
        for row in db.execute(
            select(
                Screening.patient_id,
                func.count(Screening.id),
                func.max(Screening.created_at),
            )
            .where(Screening.patient_id.in_(patient_ids))
            .group_by(Screening.patient_id)
        ).all()
    }

    # Newest first, so the first row seen for a patient is their latest band.
    latest_band: Dict[str, Optional[str]] = {}
    for patient_id, band in db.execute(
        select(Screening.patient_id, DISPLAYED_BAND)
        .join(ScreeningResult, ScreeningResult.screening_id == Screening.id)
        .where(Screening.patient_id.in_(patient_ids))
        .order_by(Screening.created_at.desc())
    ).all():
        latest_band.setdefault(patient_id, band)

    roster: List[Dict[str, object]] = []
    for patient in patients:
        count, last_seen = counts.get(patient.id, (0, None))
        roster.append(
            {
                "patient_id": patient.id,
                "clinic_id": patient.clinic_id,
                "screening_count": count,
                "last_screening_at": last_seen,
                "latest_risk_level": latest_band.get(patient.id),
                "created_at": patient.created_at,
                "is_active": patient.is_active,
            }
        )
    return roster


def count_clinic_patients(db: Session, clinic_id: str) -> int:
    """Total patients registered to a clinic."""
    stmt = select(func.count(Patient.id)).where(Patient.clinic_id == clinic_id)
    return int(db.scalar(stmt) or 0)


def clinic_band_counts(db: Session, clinic_id: str) -> Dict[str, int]:
    """Count a clinic's completed screenings per displayed risk band.

    Keyed by the band string exactly as stored (``"HIGH RISK"``, ``"MODERATE RISK"``,
    ``"LOW RISK"`` -- see ``quantum_ml.calibration.RiskLevel``), so the portal can look
    the band up with the same value it renders. Screenings with no result contribute to
    neither band; :func:`count_clinic_screenings` is the denominator.
    """
    rows = db.execute(
        select(DISPLAYED_BAND, func.count(ScreeningResult.id))
        .join(Screening, Screening.id == ScreeningResult.screening_id)
        .join(Patient, Patient.id == Screening.patient_id)
        .where(Patient.clinic_id == clinic_id)
        .group_by(DISPLAYED_BAND)
    ).all()
    return {str(band): int(count) for band, count in rows if band is not None}


def count_clinic_mock_results(db: Session, clinic_id: str) -> int:
    """Count a clinic's results flagged ``is_mock``.

    Surfaced on the dashboard so a demo database is never mistaken for clinical
    throughput -- the same reason the result screen shows a TEST DATA banner.
    """
    stmt = (
        select(func.count(ScreeningResult.id))
        .join(Screening, Screening.id == ScreeningResult.screening_id)
        .join(Patient, Patient.id == Screening.patient_id)
        .where(Patient.clinic_id == clinic_id, ScreeningResult.is_mock.is_(True))
    )
    return int(db.scalar(stmt) or 0)


def get_patient(db: Session, patient_id: str) -> Optional[Patient]:
    """Retrieve a patient record by ID without creating one."""
    return db.scalar(select(Patient).where(Patient.id == patient_id))


# ------------------------------------------------------------------- Clinic accounts


def get_clinic_user_by_email(db: Session, email: str) -> Optional[ClinicUser]:
    """Look up a portal account by email, case-insensitively."""
    stmt = select(ClinicUser).where(ClinicUser.email == email.strip().lower())
    return db.scalar(stmt)


def create_clinic_user(
    db: Session,
    email: str,
    hashed_password: str,
    clinic_id: str = "default_clinic",
    full_name: Optional[str] = None,
    role: str = "clinician",
) -> ClinicUser:
    """Create a portal account from an already-hashed password.

    Takes the hash rather than the plaintext so no plaintext password can reach this
    layer, where it would be one stray ``logger.debug`` away from the log file.
    """
    user = ClinicUser(
        id=str(uuid.uuid4()),
        email=email.strip().lower(),
        hashed_password=hashed_password,
        clinic_id=clinic_id,
        full_name=full_name,
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user



def create_screening_result(
    db: Session,
    screening: Screening,
    risk_level: str,
    details: str,
    classical_prob: Optional[float],
    quantum_prob: Optional[float],
    final_prob: float,
    threshold: float = 0.50,
    classification: str = "screening_negative",
    model_version: str = "v1.0.0-qml",
    quantum_qubits: int = 8,
    quantum_shots: int = 1024,
    execution_time_ms: Optional[float] = None,
    is_mock: bool = False,
) -> ScreeningResult:
    """Record an inference and risk classification result."""
    result = ScreeningResult(
        id=str(uuid.uuid4()),
        screening_id=screening.id,
        risk_level=risk_level,
        details=details,
        classical_probability=classical_prob,
        quantum_probability=quantum_prob,
        final_probability=final_prob,
        threshold=threshold,
        classification=classification,
        model_version=model_version,
        quantum_qubits=quantum_qubits,
        quantum_shots=quantum_shots,
        execution_time_ms=execution_time_ms,
        is_mock=is_mock,
    )
    screening.status = "COMPLETED"
    db.add(result)
    db.add(screening)
    db.commit()
    db.refresh(result)
    return result


def record_inference(
    db: Session,
    screening: Screening,
    inference: InferenceResult,
) -> ScreeningResult:
    """Persist a typed inference result with its full provenance.

    Preferred over :func:`create_screening_result` for real inferences: it takes the
    structured result straight from the inference service, so no caller has to
    re-derive which threshold or which backend produced a stored row.

    What is deliberately *not* written: the feature vector and the image. PART 19/20
    ask for data minimisation and for raw clinical images not to be retained, and a
    stored 163-value descriptor of a patient's lesion is patient data. ``raw_score``
    is enough to replay a recalibration, which is the one thing the features would
    otherwise be needed for.
    """
    result = ScreeningResult(
        id=str(uuid.uuid4()),
        screening_id=screening.id,
        risk_level=inference.risk_level,
        details=inference.details,
        classical_probability=inference.classical_probability,
        quantum_probability=inference.probability,
        # The persisted headline number stays the calibrated quantum probability: it is
        # the only calibrated value, it is what the FHIR export reports, and it is
        # self-consistent with the stored ``risk_level``/``threshold`` band. The
        # displayed verdict headlines the strongest validated model's *band* via the
        # ``primary_*`` columns below, without letting the uncalibrated classical score
        # masquerade as a probability here.
        final_probability=inference.probability,
        threshold=inference.calibration.screening_threshold,
        classification=inference.classification,
        # Displayed (headline) verdict: the strongest validated model's band, with its
        # ranking score and operating point. Falls back to the quantum band for the mock
        # path and for any model served without a classical baseline.
        primary_model=inference.primary_model,
        primary_risk_level=inference.primary_risk_level or inference.risk_level,
        primary_probability=(
            inference.primary_probability
            if inference.primary_probability is not None
            else inference.probability
        ),
        primary_threshold=(
            inference.primary_threshold
            if inference.primary_threshold is not None
            else inference.calibration.screening_threshold
        ),
        primary_calibrated=inference.primary_calibrated,
        model_version=inference.model_version,
        inference_id=inference.inference_id,
        raw_score=inference.raw_score,
        probability_uncalibrated=inference.probability_uncalibrated,
        high_risk_threshold=inference.calibration.high_risk_threshold,
        calibration_method=inference.calibration.method,
        bands_source=inference.calibration.bands_source,
        execution_mode=inference.quantum.execution_mode,
        backend_name=inference.quantum.backend_name,
        circuit_depth=inference.quantum.circuit_depth,
        feature_mode=inference.features.feature_mode,
        quantum_time_ms=inference.quantum_time_ms,
        quantum_qubits=inference.quantum.n_qubits,
        quantum_shots=inference.quantum.shots,
        execution_time_ms=inference.execution_time_ms,
        is_mock=inference.is_mock,
    )
    screening.status = "COMPLETED"
    db.add(result)
    db.add(screening)
    db.commit()
    db.refresh(result)
    return result


def get_screening_result(db: Session, screening_id: str) -> Optional[ScreeningResult]:
    """Retrieve a screening result by screening ID."""
    stmt = select(ScreeningResult).where(ScreeningResult.screening_id == screening_id)
    return db.scalar(stmt)

def create_audit_log(
    db: Session,
    action: str,
    entity_type: str,
    entity_id: str,
    actor_id: str = "system",
    details: Optional[str] = None,
) -> AuditLog:
    """Record an immutable audit event."""
    log = AuditLog(
        id=str(uuid.uuid4()),
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        actor_id=actor_id,
        details=details,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log
