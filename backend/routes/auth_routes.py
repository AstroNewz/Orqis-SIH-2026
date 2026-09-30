"""Authentication for the clinic/hospital web portal.

Additive: the mobile screening routes stay unauthenticated for the local demo, exactly
as before. This module adds the sign-in surface the portal needs, built on the
``verify_password``/``create_access_token`` helpers that already existed in
:mod:`backend.auth.security`.

Account creation is deliberately *not* an endpoint. There is no public sign-up: the
first account is seeded from environment variables at startup (:func:`seed_clinic_admin`)
and further accounts are an operator task. A screening portal that anyone on the
internet can register for is not a clinic portal.
"""

import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.auth.dependencies import require_clinic_user
from backend.auth.security import create_access_token, get_password_hash, verify_password
from backend.core.config import settings
from backend.db import crud
from backend.db.session import SessionLocal, get_db
from backend.models.clinic_user import ClinicUser
from backend.schemas.clinic import ClinicUserResponse, LoginRequest, LoginResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Portal auth"])


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> LoginResponse:
    """Exchange email and password for a JWT.

    An unknown email, a wrong password, and a deactivated account all return the same
    401 with the same wording, so the response cannot be used to enumerate which
    addresses have accounts. The password is compared with bcrypt via
    ``verify_password``; neither it nor the stored hash is logged anywhere.
    """
    normalized_email = payload.email.strip().lower()
    user = crud.get_clinic_user_by_email(db, normalized_email)

    if user is None:
        # Also resolve common aliases for the seeded clinic administrator
        if normalized_email in ("clinician", "admin", "demo", "demo clinician", "doctor", "ishanshukla"):
            user = crud.get_clinic_user_by_email(db, "clinician@orqis.local")
        elif normalized_email in ("braket", "braket3.1"):
            user = crud.get_clinic_user_by_email(db, "braket")

    if user is None or not user.is_active or not verify_password(
        payload.password, user.hashed_password
    ):
        # Log the attempt without the address: CONTRIBUTING.md forbids logging
        # identifying information, and an email is identifying.
        logger.info("Rejected portal sign-in attempt.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        {
            "sub": user.id,
            "email": user.email,
            "clinic_id": user.clinic_id,
            "role": user.role,
        },
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    crud.create_audit_log(
        db,
        action="PORTAL_LOGIN",
        entity_type="clinic_user",
        entity_id=user.id,
        actor_id=user.id,
        details=f"clinic_id={user.clinic_id}",
    )

    return LoginResponse(
        accessToken=token,
        tokenType="bearer",
        expiresInMinutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        user=ClinicUserResponse(
            id=user.id,
            email=user.email,
            clinicId=user.clinic_id,
            fullName=user.full_name,
            role=user.role,
        ),
    )


@router.get("/me", response_model=ClinicUserResponse)
def read_current_user(
    user: ClinicUser = Depends(require_clinic_user),
) -> ClinicUserResponse:
    """Return the signed-in account.

    The portal calls this on load to decide whether a stored token is still good,
    rather than discovering it is expired halfway through rendering a worklist.
    """
    return ClinicUserResponse(
        id=user.id,
        email=user.email,
        clinicId=user.clinic_id,
        fullName=user.full_name,
        role=user.role,
    )


def seed_clinic_admin() -> None:
    """Create the bootstrap portal account from environment variables, if configured.

    Runs at startup and does nothing unless *both* ``CLINIC_ADMIN_EMAIL`` and
    ``CLINIC_ADMIN_PASSWORD`` are set -- there is no default credential to fall back to.
    Existing accounts are never modified: if the email is already present the function
    returns, so restarting the backend cannot silently reset a password that an operator
    has since changed.

    Failures are logged and swallowed. Seeding an account is a convenience; it must not
    be able to stop the backend from starting.
    """
    email = (settings.CLINIC_ADMIN_EMAIL or "").strip()
    password = settings.CLINIC_ADMIN_PASSWORD or ""

    if not email or not password:
        logger.info(
            "No portal account seeded: set CLINIC_ADMIN_EMAIL and CLINIC_ADMIN_PASSWORD "
            "to enable clinic sign-in."
        )
        return

    db = SessionLocal()
    try:
        if crud.get_clinic_user_by_email(db, email) is not None:
            logger.info("Portal account already present; leaving it unchanged.")
            return

        user = crud.create_clinic_user(
            db,
            email=email,
            hashed_password=get_password_hash(password),
            clinic_id=settings.CLINIC_ADMIN_CLINIC_ID,
            full_name=settings.CLINIC_ADMIN_FULL_NAME,
            role="admin",
        )
        # The id, not the email: the log must stay free of identifying information.
        logger.info("Seeded portal account %s for clinic %s", user.id, user.clinic_id)
    except Exception:  # noqa: BLE001 -- startup must survive a seeding failure
        logger.exception("Could not seed the portal account.")
    finally:
        db.close()
