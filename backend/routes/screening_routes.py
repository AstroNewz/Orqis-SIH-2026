"""HTTP surface for screening.

Path convention
---------------
PART 18 suggests ``/api/v1/inference``. This repository already ships a Flutter
client against ``/api/screening/analyze``, ``/api/results/{id}`` and
``/api/patients/{id}/history`` (see ARCHITECTURE.md and
``lib/data/datasources/assessment_api_datasource.dart``), and PART 18 also says not to
adopt the suggested paths blindly where the existing architecture specifies another
contract. The existing paths are kept and ``/api/model/info`` is added alongside them;
DEC-020 records the reconciliation.

Error mapping
-------------
Inference failures carry their own ``status_code`` and a message written to be shown
to a patient. They are translated here rather than collapsed into a 500, so a poor
photograph reads as "retake it" (422) and an unloaded model reads as "try shortly"
(503). Anything *not* an :class:`InferenceError` is an unexpected fault: it is logged
with a traceback server-side and returned as a bare 500 with no detail, because
PART 34 forbids exposing stack traces to the client.
"""

import io
import logging
import os
import shutil
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.db import crud
from backend.db.session import get_db
from backend.ml.artifacts import ArtifactStore
from backend.ml.localizer import LOCALIZER_VERSION, LesionLocalizer
from backend.schemas.localization import LocalizationResponse
from backend.schemas.result import AssessmentResultResponse
from backend.schemas.screening import (
    AssessmentResponse,
    HistoryEntryResponse,
    ScreeningCreate,
)
from backend.services.inference_service import InferenceError, get_inference_service
from backend.services.screening_service import ScreeningService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Screening"])
screening_service = ScreeningService()

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Extensions the image pipeline can actually decode. Whitelisted rather than
# blacklisted, and the *extension the server chooses* is derived from this list --
# never from the client-supplied filename, which is attacker-controlled.
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

# Content types for serving a stored capture back. Derived from the extension the
# *server* chose at upload time, never from anything a client sent.
IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


@router.post("/screening/upload", status_code=status.HTTP_201_CREATED)
async def upload_screening_image(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
) -> Dict[str, str]:
    """Accept an intra-oral photograph and return a server-side path for analysis.

    Streamed in bounded chunks and cut off at :data:`MAX_UPLOAD_BYTES`, so a large
    body cannot exhaust memory or disk. The stored filename is a fresh UUID -- the
    client's filename is used only to pick an extension, and only after the extension
    has been checked against :data:`ALLOWED_IMAGE_EXTENSIONS`.
    """
    original_ext = os.path.splitext(file.filename or "")[1].lower()
    if original_ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Unsupported image format. Supported: "
                + ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
            ),
        )

    safe_filename = f"{uuid.uuid4()}{original_ext}"
    dest_path = os.path.join(UPLOAD_DIR, safe_filename)

    written = 0
    try:
        with open(dest_path, "wb") as buffer:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            "Image exceeds the "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit. "
                            "Capture at a lower resolution."
                        ),
                    )
                buffer.write(chunk)
    except HTTPException:
        # Do not leave a truncated file behind for the analyze call to trip over.
        _discard(dest_path)
        raise
    except OSError as exc:
        _discard(dest_path)
        logger.exception("Could not store upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The image could not be stored. Please try again.",
        ) from exc
    finally:
        await file.close()

    if written == 0:
        _discard(dest_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image was empty. Please capture it again.",
        )

    logger.info("Stored upload %s (%d bytes)", safe_filename, written)
    return {
        "status": "uploaded",
        "image_path": dest_path,
        "patient_id": patient_id or str(uuid.uuid4()),
        "size_bytes": str(written),
    }


def _discard(path: str) -> None:
    """Remove a partially written upload, ignoring a missing file."""
    try:
        os.unlink(path)
    except OSError:
        pass


@lru_cache(maxsize=1)
def _get_localizer() -> Optional[LesionLocalizer]:
    """Load the MobileNet ROI localiser once, or ``None`` if it is unavailable.

    Cached for the process: the artifact does not change at runtime and the torch load
    is not free. Returning ``None`` rather than raising keeps localisation strictly
    non-gating -- the route turns it into a clean 503 and the client simply omits the
    overlay -- so a missing torch install or absent artifact never breaks the demo.
    """
    try:
        directory = ArtifactStore.from_settings().localizer_dir(LOCALIZER_VERSION)
        return LesionLocalizer.load(directory)
    except Exception as exc:  # noqa: BLE001 -- any load failure degrades to "no overlay"
        logger.warning("Lesion localiser is unavailable: %s", exc)
        return None


@router.post("/screening/analyze", response_model=AssessmentResultResponse)
def analyze_screening(
    payload: ScreeningCreate,
    db: Session = Depends(get_db),
) -> AssessmentResultResponse:
    """Run the end-to-end screening pipeline: quality control, feature extraction,
    amplitude encoding, VQC inference, calibration, risk classification, persistence.

    Returns 422 when the submission itself cannot be scored (unusable photograph,
    wrong descriptor length) and 503 when the model or quantum backend is not
    available. Both carry a message intended for the patient.
    """
    try:
        return screening_service.analyze_screening(db=db, screening_in=payload)
    except InferenceError as exc:
        # str(exc) is patient-safe by construction -- every subclass of InferenceError
        # is documented to carry a message that can be shown to a client.
        logger.info("Screening rejected (%d): %s", exc.status_code, exc)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected failure during screening analysis: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Screening analysis failed unexpectedly. Please try again.",
        ) from exc


@router.post("/localize", response_model=LocalizationResponse)
async def localize_image(file: UploadFile = File(...)) -> LocalizationResponse:
    """Locate the lesion ROI in an intra-oral photograph for a *visual overlay*.

    Independent of ``/screening/analyze`` by design (DEC-034): it runs only the
    MobileNet localiser and never the screening pipeline, so the frozen centre-crop
    calibration is untouched. The response is advisory -- the client draws a box when
    ``localized`` is true and shows nothing otherwise -- so a localiser that is
    unavailable or unsure never blocks a screening.

    Returns 415 for an unsupported extension, 413 past the size cap, 422 for an
    unreadable image, and 503 when no localiser artifact is loaded.
    """
    original_ext = os.path.splitext(file.filename or "")[1].lower()
    if original_ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Unsupported image format. Supported: "
                + ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
            ),
        )

    # Read into memory under the same cap as upload; the bytes are never persisted --
    # localisation needs only to look at the pixels, not store them.
    buffer = io.BytesIO()
    written = 0
    try:
        while chunk := await file.read(UPLOAD_CHUNK_BYTES):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        "Image exceeds the "
                        f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit. "
                        "Capture at a lower resolution."
                    ),
                )
            buffer.write(chunk)
    finally:
        await file.close()

    if written == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image was empty. Please capture it again.",
        )

    localizer = _get_localizer()
    if localizer is None or not localizer.is_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lesion localisation is not available right now.",
        )

    try:
        from PIL import Image

        image = Image.open(io.BytesIO(buffer.getvalue()))
        image.load()  # force decode now so a truncated file fails here, not mid-predict
        image = image.convert("RGB")
    except Exception as exc:  # noqa: BLE001 -- any decode failure is a client retake
        logger.info("Localisation image could not be read: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The image could not be read. Please capture it again.",
        ) from exc

    result = localizer.localize(image)
    roi = result.roi
    return LocalizationResponse(
        status=result.status.value,
        localized=result.is_localized,
        confidence=float(result.confidence),
        box_normalised=(
            [round(float(v), 6) for v in result.predicted_box_normalised]
            if result.predicted_box_normalised is not None
            else None
        ),
        roi_box_pixels=list(roi.box) if roi is not None else None,
        source_width=int(image.width),
        source_height=int(image.height),
        roi_source=roi.source.value if roi is not None else None,
        reasons=list(result.reasons),
        localizer_version=result.localizer_version,
    )


@router.get("/model/info")
def model_info() -> Dict[str, Any]:
    """Describe the served model: version, qubits, depth, calibration, feature contract.

    The client needs ``expected_descriptor_dimension`` from here before it can send an
    on-device descriptor. Reports ``ready: false`` rather than failing when no trained
    artifacts are present, so a fresh checkout can still start the backend.
    """
    return get_inference_service().describe()


@router.get("/screening/{screening_id}", response_model=AssessmentResponse)
def get_screening(
    screening_id: str,
    db: Session = Depends(get_db),
) -> AssessmentResponse:
    """Retrieve screening session metadata by ID."""
    screening = crud.get_screening(db, screening_id)
    if not screening:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Screening session '{screening_id}' not found",
        )
    return AssessmentResponse.model_validate(screening)


@router.get("/screening/{screening_id}/image")
def get_screening_image(
    screening_id: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    """Serve the stored capture for a screening so a past result can render its photo.

    The path is read from the screening record and never from the request: the caller
    supplies only an opaque screening id. Even so the resolved path is checked to be
    inside :data:`UPLOAD_DIR` before anything is opened, because ``image_path`` was
    accepted from a client at analyse time and a record predating the upload route
    could point anywhere on disk. A path outside the upload directory is reported as
    404 rather than 403 -- whether a file exists elsewhere on the server is not
    something this endpoint should confirm.
    """
    screening = crud.get_screening(db, screening_id)
    if not screening:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Screening session '{screening_id}' not found",
        )

    try:
        resolved = os.path.realpath(screening.image_path)
        upload_root = os.path.realpath(UPLOAD_DIR)
        inside_uploads = os.path.commonpath([resolved, upload_root]) == upload_root
    except (OSError, ValueError):
        # commonpath raises on different drives on Windows; treat as "not ours".
        inside_uploads = False

    if not inside_uploads or not os.path.isfile(resolved):
        logger.info("No servable capture for screening %s.", screening_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No stored image is available for this screening.",
        )

    extension = os.path.splitext(resolved)[1].lower()
    return FileResponse(
        resolved,
        media_type=IMAGE_MEDIA_TYPES.get(extension, "application/octet-stream"),
    )


@router.get("/results/{screening_id}", response_model=AssessmentResultResponse)
def get_screening_result(
    screening_id: str,
    db: Session = Depends(get_db),
) -> AssessmentResultResponse:
    """Retrieve screening assessment results by screening ID."""
    result = crud.get_screening_result(db, screening_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result for screening '{screening_id}' not found",
        )
    return AssessmentResultResponse.model_validate(result)


@router.get("/patients/{patient_id}/history", response_model=List[HistoryEntryResponse])
def get_patient_history(
    patient_id: str,
    db: Session = Depends(get_db),
) -> List[HistoryEntryResponse]:
    """Retrieve chronological screening history for a specific patient."""
    return screening_service.get_patient_history(db, patient_id)


@router.get("/screening/{screening_id}/fhir")
def get_fhir_screening_report(
    screening_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Export screening result as interoperable HL7 FHIR R4 resources."""
    fhir_data = screening_service.build_fhir_resources(db, screening_id)
    if not fhir_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"FHIR resource for screening '{screening_id}' not found",
        )
    return fhir_data
