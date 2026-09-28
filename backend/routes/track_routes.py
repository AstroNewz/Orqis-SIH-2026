"""HTTP surface for the multi-condition platform.

Additive by design. ``/api/screening/analyze`` and every other existing path keep their
exact behaviour and their exact response schemas; these routes sit beside them and expose
the same oral-lesion model through the track contract, plus whatever other conditions are
registered. A client that never calls ``/api/tracks`` cannot tell this module exists.

Two things here are contract rather than convenience:

``GET /api/tracks`` lists **unready** tracks too, with a reason. A platform that hides a
broken track looks healthier than it is, and a client that only learns about a condition
when it works cannot tell the user why it is missing.

Every response carries the track's :class:`~backend.ml.track.ValidationSummary`, including
``frozen_test_evaluated``. That flag is the difference between a research result and a
clinical claim, and putting it in the payload means a client cannot render a number as
validated by forgetting to ask.

Error mapping follows the existing screening router: a bad payload reads as 422, a missing
or unloadable model as 503, an unknown track as 404, and anything unexpected is logged
server-side and returned as a bare 500 with no detail.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Body, File, HTTPException, UploadFile, status

from backend.ml.track import (
    TrackAssessment,
    TrackDescriptor,
    TrackInputError,
    TrackNotFoundError,
    TrackNotReadyError,
    registry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tracks", tags=["Platform"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _resolve(track_id: str):
    try:
        return registry.get(track_id)
    except TrackNotFoundError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(error)) from error


def _run(track, payload: Any) -> TrackAssessment:
    """Invoke a track and translate its failure vocabulary into HTTP."""
    try:
        return track.analyze(payload)
    except TrackInputError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
    except TrackNotReadyError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("Unhandled failure in track %r", getattr(track, "track_id", "?"))
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error."
        ) from None


@router.get("", response_model=List[TrackDescriptor], summary="List every screening track")
def list_tracks() -> List[TrackDescriptor]:
    """Every registered condition, ready or not.

    A track whose artifacts are missing appears with ``ready=false`` and an
    ``unready_reason`` rather than being omitted, so a client can grey it out and say why.
    """
    return registry.describe_all()


@router.get("/{track_id}", response_model=TrackDescriptor, summary="Describe one track")
def describe_track(track_id: str) -> TrackDescriptor:
    track = _resolve(track_id)
    try:
        return track.describe()
    except Exception as error:  # noqa: BLE001
        logger.exception("describe() raised for track %r", track_id)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"{type(error).__name__}: {error}"
        ) from error


@router.post(
    "/{track_id}/analyze",
    response_model=TrackAssessment,
    summary="Run one assessment (JSON payload)",
)
def analyze_track(track_id: str, payload: Dict[str, Any] = Body(...)) -> TrackAssessment:
    """Screen one record on a non-image track.

    The body is passed to the track as-is; each track publishes what it expects in its
    :class:`~backend.ml.track.InputSpec`, so the transport stays modality-agnostic while
    the validation stays specific.
    """
    return _run(_resolve(track_id), payload)


@router.post(
    "/{track_id}/analyze-image",
    response_model=TrackAssessment,
    summary="Run one assessment (image upload)",
)
async def analyze_track_image(
    track_id: str, file: UploadFile = File(...)
) -> TrackAssessment:
    """Screen one image on an image track.

    Kept separate from the JSON route rather than made polymorphic: multipart and JSON
    have different size limits and different failure modes, and one endpoint that guesses
    which it received would get that wrong under exactly the conditions that matter.
    """
    track = _resolve(track_id)
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail="The upload is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    return _run(track, data)
