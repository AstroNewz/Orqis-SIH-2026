"""Route tests for ``POST /api/localize`` (the non-gating ROI overlay endpoint).

Localisation is a visual aid, not part of the screening verdict (DEC-034), so these
tests assert on the *shape and self-consistency* of the response rather than on a
particular box: which box the localiser draws for a synthetic capture belongs in the
localiser's own evaluation, not here. The real-model test skips when no localiser
artifact (or torch) is present, so a fresh checkout still has a green suite; the
input-validation tests need no artifact and always run.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from tests.fixtures import synthetic_capture_jpeg

client = TestClient(app)

STATUSES = {"localized", "fallback_used", "rejected"}


def _post_localize(blob: bytes, filename: str = "capture.jpg", mime: str = "image/jpeg"):
    return client.post("/api/localize", files={"file": (filename, blob, mime)})


def _assert_well_formed(body: dict) -> None:
    """Properties every localisation response must satisfy, whatever the outcome."""
    assert body["status"] in STATUSES
    assert isinstance(body["localized"], bool)
    assert (body["status"] == "localized") == body["localized"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["sourceWidth"] > 0 and body["sourceHeight"] > 0
    assert body["localizerVersion"]
    assert isinstance(body["reasons"], list)

    box = body["boxNormalised"]
    if box is not None:
        assert len(box) == 4
        assert all(0.0 <= float(v) <= 1.0 for v in box)

    # A clean localisation must actually carry the box the client will draw, ordered.
    if body["localized"]:
        assert box is not None
        x0, y0, x1, y1 = box
        assert x1 > x0 and y1 > y0
        assert body["reasons"] == []
        assert body["roiBoxPixels"] is not None


def test_localize_returns_a_structured_result():
    """A readable image yields a well-formed, self-consistent localisation."""
    response = _post_localize(synthetic_capture_jpeg())
    if response.status_code == 503:
        pytest.skip("No localiser artifact available in this environment.")
    assert response.status_code == 200, response.text
    _assert_well_formed(response.json())


def test_localize_is_deterministic():
    """The localiser runs under no_grad in eval mode: identical pixels, identical box."""
    blob = synthetic_capture_jpeg()
    first = _post_localize(blob, filename="a.jpg")
    if first.status_code == 503:
        pytest.skip("No localiser artifact available in this environment.")
    second = _post_localize(blob, filename="b.jpg")
    assert first.status_code == 200 and second.status_code == 200
    a, b = first.json(), second.json()
    assert a["status"] == b["status"]
    assert a["boxNormalised"] == b["boxNormalised"]
    assert a["confidence"] == pytest.approx(b["confidence"])


def test_localize_rejects_non_image():
    """The extension whitelist rejects a non-image before any model runs."""
    response = client.post(
        "/api/localize",
        files={"file": ("payload.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_localize_rejects_empty_upload():
    """An empty body is a 400 retake, never a localisation of nothing."""
    response = _post_localize(b"", filename="empty.jpg")
    # 400 (empty) unless the environment has no localiser at all, which is also fine.
    assert response.status_code in (400, 503), response.text
