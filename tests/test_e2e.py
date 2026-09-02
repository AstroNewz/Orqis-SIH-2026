"""End-to-end integration test (PART 36).

Walks the whole documented path against a live app instance: upload -> quality
control -> ROI -> normalisation -> feature extraction -> clinical encoding -> fusion
-> dimensionality reduction -> amplitude encoding -> VQC -> raw score -> probability
-> calibration -> risk band -> API response -> persistence -> history -> FHIR.

Nothing here is stubbed. There is no mock inference service, no patched quantum
backend and no hardcoded probability: the assertions are on *properties* of the
result (in range, self-consistent, correctly provenanced) rather than on specific
values, because specific values belong in the evaluation report, not in a test.

The real-inference tests skip when no trained model is present, so a fresh checkout
that has not run training still has a green suite. The mock and error-path tests
always run -- they do not need artifacts.
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.db.base import Base
from backend.db.session import get_db
from tests.fixtures import synthetic_capture_jpeg, unusable_capture

# In-memory shared test database
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

RISK_LEVELS = {"LOW RISK", "MODERATE RISK", "HIGH RISK"}
EXECUTION_MODES = {"ideal_simulation", "noisy_simulation", "ibm_hardware"}


@pytest.fixture(scope="module")
def model_info():
    """The served model's contract, or a skip if nothing is trained here."""
    response = client.get("/api/model/info")
    assert response.status_code == 200
    info = response.json()
    if not info.get("ready"):
        pytest.skip(f"No trained model artifacts available: {info.get('reason')}")
    return info


def _assert_provenanced(body: dict) -> None:
    """Every real result must be traceable to a model version (PART 24)."""
    assert body["inferenceId"]
    assert body["modelVersion"] and body["modelVersion"] != "mock-development"
    assert body["isMock"] is False
    assert body["executionMode"] in EXECUTION_MODES
    assert body["backendName"]
    assert body["circuitDepth"] > 0
    assert body["quantumQubits"] > 0
    assert body["featureMode"] in {"image_only", "clinical_only", "multimodal"}
    assert body["calibrationMethod"] in {"platt", "isotonic", "identity"}
    assert body["bandsSource"] in {"calibration_artifact", "configured_fallback", "explicit"}
    assert body["quantumTimeMs"] >= 0.0
    assert body["executionTimeMs"] >= body["quantumTimeMs"]


def _assert_coherent_probability(body: dict) -> None:
    """The reported numbers must agree with each other and with the bands."""
    assert 0.0 <= body["rawScore"] <= 1.0
    assert 0.0 <= body["probabilityUncalibrated"] <= 1.0
    assert 0.0 <= body["finalProbability"] <= 1.0
    # finalProbability is the calibrated quantum probability; nothing is blended in.
    assert body["quantumProbability"] == pytest.approx(body["finalProbability"])
    assert 0.0 < body["threshold"] < 1.0
    assert body["threshold"] <= body["highRiskThreshold"]

    probability, screening, high = (
        body["finalProbability"],
        body["threshold"],
        body["highRiskThreshold"],
    )
    if probability >= high:
        expected = "HIGH RISK"
    elif probability >= screening:
        expected = "MODERATE RISK"
    else:
        expected = "LOW RISK"
    assert body["riskLevel"] == expected, (
        f"p={probability} against bands {screening}/{high} should be {expected}"
    )
    assert body["classification"] == (
        "screening_positive" if probability >= screening else "screening_negative"
    )


def _assert_disclaimed(body: dict) -> None:
    """PART 35: the result must not read as a diagnosis."""
    assert "AI-assisted" in body["disclaimer"]
    assert "not a diagnosis" in body["disclaimer"]
    assert "histopathological" in body["disclaimer"]
    assert "Notice: This is an AI-assisted screening estimate" in body["details"]
    for forbidden in ("diagnosed", "clinically proven", "100% accurate", "guarantees"):
        assert forbidden not in body["details"].lower()


# --------------------------------------------------------------- the image path
def test_e2e_upload_and_analyze_real_image(model_info):
    """The full path from a JPEG upload to a persisted, provenanced result."""
    upload = client.post(
        "/api/screening/upload",
        files={"file": ("capture.jpg", synthetic_capture_jpeg(), "image/jpeg")},
        data={"patient_id": "patient-e2e-image"},
    )
    assert upload.status_code == 201, upload.text
    assert int(upload.json()["size_bytes"]) > 0
    image_path = upload.json()["image_path"]

    response = client.post(
        "/api/screening/analyze",
        json={
            "patient_id": "patient-e2e-image",
            "image_path": image_path,
            "scan_type": "Intra-oral Scan",
            "smoking_history": True,
            "alcohol_consumption": False,
            "betel_quid": True,
            "age": 54,
            "sex": "male",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    _assert_provenanced(body)
    _assert_coherent_probability(body)
    _assert_disclaimed(body)
    assert body["modelVersion"] == model_info["model_version"]
    # The image was read: extraction dominates a sub-10ms circuit evaluation.
    assert body["executionTimeMs"] > body["quantumTimeMs"]


def test_e2e_image_path_is_deterministic(model_info):
    """The same photograph must produce the same probability twice.

    Reproducibility is a requirement, not an accident: a seeded backend and a frozen
    extractor mean two submissions of identical pixels must agree exactly.
    """
    blob = synthetic_capture_jpeg()
    probabilities = []
    for index in range(2):
        upload = client.post(
            "/api/screening/upload",
            files={"file": (f"repeat-{index}.jpg", blob, "image/jpeg")},
        )
        response = client.post(
            "/api/screening/analyze",
            json={
                "patient_id": "patient-e2e-determinism",
                "image_path": upload.json()["image_path"],
            },
        )
        assert response.status_code == 200, response.text
        probabilities.append(response.json()["finalProbability"])

    assert probabilities[0] == pytest.approx(probabilities[1], abs=1e-12)


# ---------------------------------------------------------- the descriptor path
def test_e2e_descriptor_path_matches_contract(model_info):
    """The privacy-preserving route: features computed on-device, no image sent."""
    dimension = model_info["expected_descriptor_dimension"]
    assert dimension > 0

    rng = np.random.RandomState(11)
    descriptor = rng.uniform(0.0, 1.0, size=dimension).tolist()

    response = client.post(
        "/api/screening/analyze",
        json={
            "patient_id": "patient-e2e-descriptor",
            "image_path": "on-device",
            "features": descriptor,
            "smoking_history": False,
            "alcohol_consumption": False,
            "betel_quid": False,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    _assert_provenanced(body)
    _assert_coherent_probability(body)
    _assert_disclaimed(body)


def test_e2e_clinical_flags_change_the_result(model_info):
    """Multimodal means the clinical answers must actually reach the model.

    Skipped for an image-only model, where they are correctly ignored.
    """
    if not model_info.get("clinical_features_expected"):
        pytest.skip("This model is image-only; clinical flags are not part of its input.")

    dimension = model_info["expected_descriptor_dimension"]
    descriptor = np.random.RandomState(23).uniform(0.0, 1.0, size=dimension).tolist()

    def probability_for(**flags):
        response = client.post(
            "/api/screening/analyze",
            json={
                "patient_id": "patient-e2e-clinical",
                "image_path": "on-device",
                "features": descriptor,
                **flags,
            },
        )
        assert response.status_code == 200, response.text
        return response.json()["finalProbability"]

    no_factors = probability_for(
        smoking_history=False, alcohol_consumption=False, betel_quid=False
    )
    all_factors = probability_for(
        smoking_history=True, alcohol_consumption=True, betel_quid=True
    )
    not_collected = probability_for()

    assert no_factors != all_factors, "Risk factors are being discarded before the model."
    # Tri-state: "not asked" must not be silently equal to "asked, answered no".
    assert not_collected != no_factors


# --------------------------------------------------------------- history + FHIR
def test_e2e_history_and_fhir(model_info):
    """Two screenings for one patient, then history and FHIR export."""
    patient = "patient-sih-001"
    dimension = model_info["expected_descriptor_dimension"]
    rng = np.random.RandomState(3)

    first = client.post(
        "/api/screening/analyze",
        json={
            "patient_id": patient,
            "image_path": "on-device-1",
            "scan_type": "Intra-oral Scan",
            "features": rng.uniform(0.0, 1.0, size=dimension).tolist(),
            "smoking_history": True,
            "alcohol_consumption": True,
            "betel_quid": True,
        },
    )
    assert first.status_code == 200, first.text
    first_body = first.json()

    second = client.post(
        "/api/screening/analyze",
        json={
            "patient_id": patient,
            "image_path": "on-device-2",
            "scan_type": "Intra-oral Follow-up",
            "features": rng.uniform(0.0, 1.0, size=dimension).tolist(),
            "smoking_history": False,
            "alcohol_consumption": False,
            "betel_quid": False,
        },
    )
    assert second.status_code == 200, second.text

    history_response = client.get(f"/api/patients/{patient}/history")
    assert history_response.status_code == 200
    history = history_response.json()
    assert len(history) == 2
    assert {entry["assessment"]["type"] for entry in history} == {
        "Intra-oral Scan",
        "Intra-oral Follow-up",
    }
    for entry in history:
        assert entry["result"] is not None
        assert entry["result"]["riskLevel"] in RISK_LEVELS
        # Provenance survives the round trip through the database.
        assert entry["result"]["inferenceId"]
        assert entry["result"]["executionMode"] in EXECUTION_MODES

    screening_id = first_body["assessmentId"]
    stored = client.get(f"/api/results/{screening_id}")
    assert stored.status_code == 200
    assert stored.json()["finalProbability"] == pytest.approx(
        first_body["finalProbability"]
    )
    _assert_disclaimed(stored.json())

    fhir_response = client.get(f"/api/screening/{screening_id}/fhir")
    assert fhir_response.status_code == 200
    fhir = fhir_response.json()
    assert fhir["observation"]["resourceType"] == "Observation"
    assert fhir["observation"]["subject"]["reference"] == f"Patient/{patient}"
    assert fhir["risk_assessment"]["resourceType"] == "RiskAssessment"
    # PART 21: a risk estimate, never a pathological diagnosis.
    note = fhir["risk_assessment"]["note"][0]["text"]
    assert "screening risk estimate" in note
    assert "Histopathological confirmation required" in note
    assert fhir["risk_assessment"]["prediction"][0]["probabilityDecimal"] == pytest.approx(
        first_body["finalProbability"], abs=1e-4
    )


# ---------------------------------------------------------------- failure paths
def test_e2e_unusable_image_is_rejected_for_reacquisition():
    """PART 5: a poor capture is sent back to be retaken, not scored."""
    upload = client.post(
        "/api/screening/upload",
        files={"file": ("dark.jpg", unusable_capture(), "image/jpeg")},
    )
    assert upload.status_code == 201

    response = client.post(
        "/api/screening/analyze",
        json={
            "patient_id": "patient-e2e-bad-image",
            "image_path": upload.json()["image_path"],
        },
    )
    if response.status_code == 503:
        pytest.skip("No trained model artifacts available.")
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail
    # Actionable guidance, and no internals.
    assert "Traceback" not in detail
    assert "backend." not in detail


def test_e2e_wrong_descriptor_length_is_rejected(model_info):
    """A length mismatch is a 422 that names the expectation, never a pad or truncate."""
    dimension = model_info["expected_descriptor_dimension"]
    response = client.post(
        "/api/screening/analyze",
        json={
            "patient_id": "patient-e2e-wrong-length",
            "image_path": "on-device",
            "features": [0.1] * (dimension + 7),
        },
    )
    assert response.status_code == 422
    assert str(dimension) in response.json()["detail"]


def test_e2e_missing_image_is_rejected():
    """No descriptor and no readable file means no inference -- not a synthesised one."""
    response = client.post(
        "/api/screening/analyze",
        json={
            "patient_id": "patient-e2e-missing",
            "image_path": "/nonexistent/scan.jpg",
        },
    )
    assert response.status_code == 422
    assert "/nonexistent/scan.jpg" not in response.json()["detail"]


def test_e2e_non_finite_features_are_rejected(model_info):
    """NaN in, 422 out.

    Sent as a raw body because ``NaN`` is not strict JSON and httpx refuses to encode
    it -- but Python's ``json.dumps`` emits it by default, so a client built that way
    really can put it on the wire. Pydantic then happily accepts it as a float, which
    is why the service checks finiteness itself.
    """
    dimension = model_info["expected_descriptor_dimension"]
    descriptor = [0.5] * dimension
    descriptor[0] = float("nan")
    response = client.post(
        "/api/screening/analyze",
        content=json.dumps(
            {
                "patient_id": "patient-e2e-nan",
                "image_path": "on-device",
                "features": descriptor,
            }
        ),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_e2e_upload_rejects_non_image():
    response = client.post(
        "/api/screening/upload",
        files={"file": ("payload.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_e2e_failed_screening_is_still_recorded():
    """A rejected capture leaves an auditable trail rather than vanishing."""
    patient = "patient-e2e-failure-trail"
    response = client.post(
        "/api/screening/analyze",
        json={"patient_id": patient, "image_path": "/nonexistent/scan.jpg"},
    )
    assert response.status_code == 422

    history = client.get(f"/api/patients/{patient}/history")
    assert history.status_code == 200
    entries = history.json()
    assert len(entries) == 1
    # The attempt is visible; it just has no result attached.
    assert entries[0]["result"] is None


# ------------------------------------------------------------------- mock path
def test_e2e_mock_is_labelled_everywhere():
    """The development stub must be impossible to mistake for a real result."""
    response = client.post(
        "/api/screening/analyze",
        json={
            "patient_id": "patient-e2e-mock",
            "image_path": "/tmp/anything.jpg",
            "is_mock": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["isMock"] is True
    assert body["riskLevel"].startswith("MOCK ")
    assert body["details"].startswith("TEST DATA:")
    assert body["modelVersion"] == "mock-development"
    assert body["executionMode"] == "mock"
    assert body["bandsSource"] == "mock"
    # No circuit ran, so nothing claims one did.
    assert body["quantumQubits"] == 0
    assert body["circuitDepth"] == 0


def test_e2e_mock_is_deterministic():
    payload = {
        "patient_id": "patient-e2e-mock-stable",
        "image_path": "/tmp/stable.jpg",
        "is_mock": True,
    }
    first = client.post("/api/screening/analyze", json=payload).json()
    second = client.post("/api/screening/analyze", json=payload).json()
    assert first["finalProbability"] == second["finalProbability"]
    assert first["riskLevel"] == second["riskLevel"]
    # ...but each attempt is a distinct record.
    assert first["inferenceId"] != second["inferenceId"]
