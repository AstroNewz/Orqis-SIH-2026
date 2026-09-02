import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.db.base import Base
from backend.models import Patient, Screening, ScreeningResult, AuditLog
from backend.schemas.screening import ScreeningCreate
from backend.db import crud


@pytest.fixture
def db_session():
    """In-memory SQLite database session for testing."""
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_patient_crud(db_session):
    patient = crud.get_or_create_patient(db_session, clinic_id="clinic_01")
    assert patient.id is not None
    assert patient.clinic_id == "clinic_01"
    assert patient.is_active is True

    # Retrieve existing
    existing = crud.get_or_create_patient(db_session, patient_id=patient.id)
    assert existing.id == patient.id


def test_screening_and_result_flow(db_session):
    patient = crud.get_or_create_patient(db_session)
    
    screening_in = ScreeningCreate(
        image_path="/mock/path/test.jpg",
        scan_type="Intra-oral Scan",
        smoking_history=True,
        alcohol_consumption=False,
        betel_quid=True,
        is_mock=True,
    )
    
    screening = crud.create_screening(db_session, screening_in, patient)
    assert screening.id is not None
    assert screening.status == "PENDING"
    assert screening.smoking_history is True
    assert screening.betel_quid is True

    # Create Result
    result = crud.create_screening_result(
        db_session,
        screening=screening,
        risk_level="MOCK LOW RISK",
        details="TEST DATA: Normal oral mucosa, no high-risk lesion detected.",
        classical_prob=0.15,
        quantum_prob=0.12,
        final_prob=0.135,
        threshold=0.50,
        classification="screening_negative",
        model_version="v1.0.0-qml",
        is_mock=True,
    )

    assert result.id is not None
    assert result.screening_id == screening.id
    assert screening.status == "COMPLETED"
    assert result.final_probability == 0.135
    assert result.is_mock is True

    # Retrieve result
    fetched_result = crud.get_screening_result(db_session, screening.id)
    assert fetched_result is not None
    assert fetched_result.risk_level == "MOCK LOW RISK"

    # Retrieve patient screenings
    history = crud.get_patient_screenings(db_session, patient.id)
    assert len(history) == 1
    assert history[0].id == screening.id


def test_audit_log(db_session):
    log = crud.create_audit_log(
        db_session,
        action="SCREENING_ANALYZE",
        entity_type="screening",
        entity_id="test-uuid",
        actor_id="tester",
        details="Test audit entry",
    )
    assert log.id is not None
    assert log.action == "SCREENING_ANALYZE"
