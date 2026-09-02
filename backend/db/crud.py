import uuid
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from backend.db.base import Base
from backend.db.session import engine
from backend.models.patient import Patient
from backend.models.screening import Screening
from backend.models.result import ScreeningResult
from backend.models.audit_log import AuditLog
from backend.schemas.inference import InferenceResult
from backend.schemas.screening import ScreeningCreate


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
        final_probability=inference.probability,
        threshold=inference.calibration.screening_threshold,
        classification=inference.classification,
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
