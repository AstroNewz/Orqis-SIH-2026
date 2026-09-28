import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import settings
from backend.db.crud import init_db
from backend.ml.tracks import install_default_tracks
from backend.routes.auth_routes import router as auth_router, seed_clinic_admin
from backend.routes.clinic_routes import router as clinic_router
from backend.routes.screening_routes import router as screening_router
from backend.routes.track_routes import router as track_router
from backend.services.inference_service import get_inference_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Initialize database tables on startup
    init_db()
    # Create the portal account from env vars if one is configured. No-op otherwise,
    # and never overwrites an existing account -- see seed_clinic_admin.
    seed_clinic_admin()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Unified Hybrid Quantum-Classical Machine Learning (QML) Backend for "
        "Oral Cavity Cancer Screening (Team BraKet 3.1.0 — SIH 2026)."
    ),
    lifespan=lifespan,
)

# CORS middleware for Flutter mobile app and React web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def audit_logging_middleware(request: Request, call_next):
    """
    Log request execution duration and status code.
    Ensures zero sensitive patient data or image payloads are logged.
    """
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000.0
    
    # Safe metadata logging header
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    return response


# Mount API routers
app.include_router(screening_router)
app.include_router(auth_router)
app.include_router(clinic_router)

# The multi-condition platform surface, mounted alongside the single-condition routes
# above rather than in place of them. Registration is additive and idempotent; a track
# whose artifacts are missing registers anyway and reports itself as unready, so a
# partial deployment starts and serves whatever it does have.
app.include_router(track_router)
install_default_tracks()


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Health and readiness.

    ``status`` is about the process; ``model_ready`` is about whether trained
    artifacts are loaded. They are reported separately on purpose: a backend with no
    trained model is still healthy enough to serve history and mock requests, and a
    load-balancer probe should not take it out of rotation for that. Reading the
    model state must never make this endpoint fail, so artifact errors are swallowed
    into ``model_ready: false``.
    """
    try:
        service = get_inference_service()
        model_ready = service.ready
        model_version = service.model_version if model_ready else None
        execution_mode = service.execution_mode if model_ready else None
    except Exception:  # pragma: no cover - health must not raise
        logger.exception("Could not determine model readiness.")
        model_ready, model_version, execution_mode = False, None, None

    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "model_ready": model_ready,
        "model_version": model_version,
        # The *effective* execution mode once the model is loaded, which can differ
        # from the configured one: PART 12 requires a fallback to simulation when IBM
        # hardware is unreachable, and this is where an operator sees that happened.
        "quantum_execution_mode": execution_mode or settings.QUANTUM_EXECUTION_MODE,
        "quantum_execution_mode_configured": settings.QUANTUM_EXECUTION_MODE,
        "quantum_backend": settings.QUANTUM_BACKEND,
        "quantum_qubits": settings.QUANTUM_QUBITS,
    }


@app.get("/", tags=["System"])
def root_info() -> Dict[str, str]:
    """
    Root API discovery endpoint.
    """
    return {
        "message": "Welcome to CareScan / QuOra Hybrid QML Backend",
        "docs_url": "/docs",
        "health_url": "/health",
    }
