import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.db.base import Base
from backend.db.session import get_db

# Create test in-memory SQLite database with StaticPool
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
Base.metadata.create_all(bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "quantum_backend" in data
    assert "quantum_qubits" in data


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "docs_url" in response.json()


def test_analyze_screening_mock_mode():
    payload = {
        "patient_id": "test-patient-123",
        "image_path": "/test/images/lesion_01.jpg",
        "scan_type": "Intra-oral Scan",
        "smoking_history": True,
        "alcohol_consumption": False,
        "betel_quid": True,
        "is_mock": True,
    }
    response = client.post("/api/screening/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Verify Group 1 Flutter contract compatibility
    assert "id" in data
    assert "assessmentId" in data
    assert "riskLevel" in data
    assert "details" in data
    assert "finalProbability" in data
    assert "quantumProbability" in data
    assert "classicalProbability" in data

    # Verify mock labeling compliance
    assert "MOCK" in data["riskLevel"]
    assert "TEST DATA" in data["details"]
    assert data["isMock"] is True

    screening_id = data["assessmentId"]

    # Test GET /api/screening/{id}
    s_resp = client.get(f"/api/screening/{screening_id}")
    assert s_resp.status_code == 200
    s_data = s_resp.json()
    assert s_data["id"] == screening_id
    assert s_data["imagePath"] == "/test/images/lesion_01.jpg"

    # Test GET /api/results/{id}
    r_resp = client.get(f"/api/results/{screening_id}")
    assert r_resp.status_code == 200
    r_data = r_resp.json()
    assert r_data["assessmentId"] == screening_id
    assert r_data["riskLevel"] == data["riskLevel"]

    # Test GET /api/patients/{id}/history
    h_resp = client.get("/api/patients/test-patient-123/history")
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert len(h_data) >= 1
    assert "assessment" in h_data[0]
    assert "result" in h_data[0]

    # Test GET /api/screening/{id}/fhir
    f_resp = client.get(f"/api/screening/{screening_id}/fhir")
    assert f_resp.status_code == 200
    f_data = f_resp.json()
    assert "observation" in f_data
    assert "risk_assessment" in f_data
    assert f_data["observation"]["resourceType"] == "Observation"
    assert f_data["risk_assessment"]["resourceType"] == "RiskAssessment"
