"""The concrete tracks and the HTTP surface that serves them.

Two classes of failure are worth tests here, and they are different in kind.

The first is ordinary: a malformed ECG must be refused, an unknown track must 404, a
missing artifact must read as 503 rather than 500. The second is the one that would not
announce itself -- a lead-major/time-major mix-up that scores happily on transposed input,
an uncalibrated ranking score rendered as a risk percentage, a development number served
as though a frozen test partition had been evaluated. Those are silent, and they are the
reason the contract carries provenance at all, so they are tested directly.

Nothing here loads a trained artifact, so the suite runs on a clean checkout.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.ml.track import (
    Modality,
    RiskBand,
    TrackInputError,
    TrackNotReadyError,
    TrackRegistry,
)
from backend.ml.tracks import install_default_tracks
from backend.ml.tracks.ecg import EXPECTED_SHAPE, STANDARD_LEAD_ORDER, EcgTrack
from backend.ml.tracks.oral_lesion import OralLesionTrack, _band


# --------------------------------------------------------------------- registration
def test_install_registers_both_tracks():
    reg = TrackRegistry()
    install_default_tracks(reg)
    assert reg.ids() == ["ecg_12lead", "oral_lesion"]


def test_install_is_idempotent():
    """Startup and a test fixture may both call it; the second call must not raise."""
    reg = TrackRegistry()
    install_default_tracks(reg)
    install_default_tracks(reg)
    assert len(reg) == 2


def test_constructing_a_track_touches_no_artifact(tmp_path):
    """__init__ must stay cheap, or one missing model becomes a startup crash."""
    track = EcgTrack(classifier_path=tmp_path / "absent.joblib")
    assert track.describe().ready is False


# ------------------------------------------------------------------------ ECG track
def _flat_ecg() -> np.ndarray:
    return np.zeros(EXPECTED_SHAPE, dtype=np.float32)


def test_ecg_publishes_its_input_contract():
    spec = EcgTrack().describe().input_spec
    assert spec.modality is Modality.SIGNAL
    assert spec.shape == [12, 1000]
    assert spec.sampling_frequency_hz == 100.0
    assert spec.units == "mV"
    assert spec.channel_names == STANDARD_LEAD_ORDER


def test_ecg_is_not_claimable_and_serves_no_quantum_component():
    """Both statements are about what exists today, and both must stay honest."""
    descriptor = EcgTrack().describe()
    assert descriptor.validation.frozen_test_evaluated is False
    assert descriptor.validation.is_clinically_claimable is False
    assert descriptor.uses_quantum is False
    assert descriptor.quantum_role is None
    assert "fold 10" in descriptor.validation.notes


def test_ecg_names_the_missing_artifact_rather_than_failing_vaguely(tmp_path):
    track = EcgTrack(classifier_path=tmp_path / "absent.joblib")
    with pytest.raises(TrackNotReadyError, match="absent.joblib"):
        track.analyze({"signal": _flat_ecg().tolist()})


@pytest.mark.parametrize(
    "payload, pattern",
    [
        (None, "required"),
        ({}, "signal"),
        (np.zeros((12, 500)), r"12 x 1000"),
        (np.zeros((8, 1000)), r"12 x 1000"),
        ("not a signal", "numeric"),
    ],
)
def test_ecg_rejects_malformed_input_before_loading_anything(payload, pattern, tmp_path):
    """Validation precedes loading, so a bad request reads as a bad request (422) and not
    as a missing model (503)."""
    track = EcgTrack(classifier_path=tmp_path / "absent.joblib")
    with pytest.raises(TrackInputError, match=pattern):
        track.analyze(payload)


def test_ecg_rejects_non_finite_samples(tmp_path):
    signal = _flat_ecg()
    signal[3, 17] = np.nan
    track = EcgTrack(classifier_path=tmp_path / "absent.joblib")
    with pytest.raises(TrackInputError, match="NaN"):
        track.analyze(signal)


def test_ecg_accepts_time_major_by_transposing_it_explicitly(tmp_path):
    """A (1000, 12) array is a common convention and an unambiguous one. Accepting it is
    fine; what is not fine is letting it reach the model unrotated, which no downstream
    check could catch."""
    track = EcgTrack(classifier_path=tmp_path / "absent.joblib")
    assert track._as_array(np.zeros((1000, 12))).shape == EXPECTED_SHAPE
    # Validation passed, so the failure is now about the model, not the payload.
    with pytest.raises(TrackNotReadyError):
        track.analyze(np.zeros((1000, 12), dtype=np.float32))


def test_ecg_bundle_without_a_threshold_is_refused(tmp_path):
    """An estimator without its operating point is not servable: the pairing is the part
    that would otherwise fail silently."""
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "partial.joblib"
    joblib.dump({"estimator": object(), "feature_set_version": "ecg-v1"}, path)
    with pytest.raises(TrackNotReadyError, match="threshold"):
        EcgTrack(classifier_path=path).analyze(_flat_ecg())


def test_ecg_reports_a_broken_bundle_as_unready_not_as_a_crash(tmp_path):
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(b"not a joblib file")
    descriptor = EcgTrack(classifier_path=path).describe()
    assert descriptor.ready is False
    assert descriptor.unready_reason


# --------------------------------------------------------------- oral-lesion mapping
class _FakeResult:
    """A constructed InferenceResult stand-in, so the mapping is testable without 200 MB
    of artifacts. This is where a silent semantic error would live."""

    def __init__(self, **overrides):
        self.inference_id = "abc"
        self.model_version = "v1-handcrafted"
        self.created_at = datetime.now(timezone.utc)
        self.probability = 0.31
        self.probability_uncalibrated = 0.44
        self.raw_score = 0.44
        self.expectation_value = 0.12
        self.risk_level = "MODERATE RISK"
        self.classification = "suspicious"
        self.details = "d"
        self.classical_probability = 0.82
        self.classical_model = "logistic_regression"
        self.primary_model = "logistic_regression"
        self.primary_risk_level = "LOW RISK"
        self.primary_probability = 0.82
        self.primary_threshold = 0.91
        self.primary_calibrated = False
        self.calibration = None
        self.quantum = None
        self.features = None
        self.quality = None
        self.quantum_visual = None
        self.execution_time_ms = 12.0
        self.quantum_time_ms = 3.0
        self.preprocessing_time_ms = 1.0
        self.is_mock = False
        self.disclaimer = "Research use only."
        self.__dict__.update(overrides)


def test_the_classical_band_headlines_and_is_flagged_uncalibrated():
    """DEC-034 in one assertion: the classical baseline leads the verdict, and its score
    travels marked as a ranking value so no client renders 0.82 as "82% risk"."""
    assessment = OralLesionTrack.to_assessment(_FakeResult())
    assert assessment.risk_band is RiskBand.LOW
    assert assessment.primary_model == "logistic_regression"
    assert assessment.probability == 0.82
    assert assessment.probability_is_calibrated is False


def test_the_calibrated_quantum_probability_is_preserved_separately():
    """It is the number persisted as final_probability and exported to FHIR; losing it in
    translation would desynchronise the API from the database."""
    assessment = OralLesionTrack.to_assessment(_FakeResult())
    assert assessment.quantum_probability == 0.31
    assert assessment.classical_probability == 0.82
    assert assessment.detail["quantum_risk_level"] == "MODERATE RISK"


def test_the_two_opinions_are_never_blended():
    assessment = OralLesionTrack.to_assessment(_FakeResult())
    assert assessment.probability in (0.82, 0.31)
    assert abs(assessment.probability - 0.565) > 1e-9  # the average of the two


def test_when_no_classical_primary_exists_the_quantum_band_leads_and_is_calibrated():
    result = _FakeResult(
        primary_model="quantum_vqc_calibrated",
        primary_risk_level=None,
        primary_probability=None,
        primary_threshold=None,
        primary_calibrated=True,
    )
    assessment = OralLesionTrack.to_assessment(result)
    assert assessment.risk_band is RiskBand.MODERATE
    assert assessment.probability == 0.31
    assert assessment.probability_is_calibrated is True


def test_an_unknown_band_becomes_indeterminate_not_low():
    """A lookup miss must not resolve to the reassuring answer."""
    assert _band("SOMETHING NEW") is RiskBand.INDETERMINATE
    assert _band(None) is RiskBand.INDETERMINATE
    assert _band("") is RiskBand.INDETERMINATE


@pytest.mark.parametrize(
    "label, expected",
    [
        ("LOW RISK", RiskBand.LOW),
        ("moderate risk", RiskBand.MODERATE),
        ("HIGH RISK", RiskBand.HIGH),
        ("High", RiskBand.HIGH),
        ("MEDIUM", RiskBand.MODERATE),
    ],
)
def test_band_mapping(label, expected):
    assert _band(label) is expected


def test_a_mock_result_stays_marked_as_one():
    assessment = OralLesionTrack.to_assessment(_FakeResult(is_mock=True))
    assert assessment.is_mock is True


# ------------------------------------------------------------------------ the routes
@pytest.fixture(scope="module")
def client():
    from backend.main import app

    return TestClient(app)


def test_listing_includes_unready_tracks_with_a_reason(client):
    """Hiding a broken track makes the platform look healthier than it is."""
    response = client.get("/api/tracks")
    assert response.status_code == 200
    tracks = {t["track_id"]: t for t in response.json()}
    assert set(tracks) == {"ecg_12lead", "oral_lesion"}
    ecg = tracks["ecg_12lead"]
    assert ecg["ready"] is False
    assert ecg["unready_reason"]


def test_every_listed_track_publishes_its_test_set_status(client):
    for track in client.get("/api/tracks").json():
        assert "frozen_test_evaluated" in track["validation"]
        assert "partition_scored" in track["validation"]


def test_describe_one_track(client):
    response = client.get("/api/tracks/oral_lesion")
    assert response.status_code == 200
    assert response.json()["modality"] == "image"


def test_unknown_track_is_404_and_names_what_exists(client):
    response = client.get("/api/tracks/diabetes")
    assert response.status_code == 404
    assert "ecg_12lead" in response.json()["detail"]


def test_analyzing_an_unknown_track_is_404(client):
    assert client.post("/api/tracks/diabetes/analyze", json={}).status_code == 404


def test_a_missing_model_is_503_not_500(client):
    response = client.post(
        "/api/tracks/ecg_12lead/analyze", json={"signal": _flat_ecg().tolist()}
    )
    assert response.status_code == 503


def test_a_malformed_payload_is_422_and_outranks_the_missing_model(client):
    """The caller should be told what they got wrong, not what the server is missing."""
    response = client.post("/api/tracks/ecg_12lead/analyze", json={"signal": [[0.0] * 5] * 12})
    assert response.status_code == 422
    assert "12 x 1000" in response.json()["detail"]


def test_an_empty_upload_is_refused(client):
    response = client.post(
        "/api/tracks/oral_lesion/analyze-image", files={"file": ("x.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 422


def test_an_oversized_upload_is_refused_without_being_analyzed(client):
    from backend.routes.track_routes import MAX_UPLOAD_BYTES

    payload = b"\x00" * (MAX_UPLOAD_BYTES + 1)
    response = client.post(
        "/api/tracks/oral_lesion/analyze-image",
        files={"file": ("big.jpg", payload, "image/jpeg")},
    )
    assert response.status_code == 413


def test_the_existing_screening_surface_is_untouched(client):
    """The platform routes are additive. If this ever fails, they stopped being additive."""
    assert client.get("/health").status_code == 200
    assert client.get("/api/model/info").status_code == 200
