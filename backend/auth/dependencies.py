"""FastAPI dependencies for portal authentication.

:func:`backend.auth.security.verify_jwt_token` is deliberately *optional* -- it returns
``None`` when no header is present, so the existing screening routes keep working
unauthenticated for the local demo. The clinic portal needs the opposite: a dependency
that refuses the request outright. Rather than change the shared helper (and with it the
behaviour of every existing route), the strict variant lives here.

Separating them also keeps one rule easy to audit: a route either requires a clinic user
or it does not, and the answer is visible in its signature.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import Depends, Header, HTTPException, status
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.db import crud
from backend.db.session import get_db
from backend.models.clinic_user import ClinicUser

logger = logging.getLogger(__name__)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated. Sign in to continue.",
    headers={"WWW-Authenticate": "Bearer"},
)


def decode_bearer_token(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Decode a required ``Authorization: Bearer <jwt>`` header.

    Every failure -- missing header, wrong scheme, malformed or expired token -- returns
    the same 401 with the same message. Distinguishing them would tell an unauthenticated
    caller which half of a credential it got right.
    """
    if not authorization:
        raise _UNAUTHORIZED

    try:
        scheme, token = authorization.split()
    except ValueError:
        raise _UNAUTHORIZED from None

    if scheme.lower() != "bearer":
        raise _UNAUTHORIZED

    try:
        return jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            options={"verify_aud": False},
        )
    except JWTError:
        raise _UNAUTHORIZED from None


def require_clinic_user(
    payload: Dict[str, Any] = Depends(decode_bearer_token),
    db: Session = Depends(get_db),
) -> ClinicUser:
    """Resolve the signed-in portal account, or fail with 401.

    The account is re-read from the database on every request rather than trusted from
    the token body: a token stays valid until it expires, so a deactivated account must
    stop working immediately instead of an hour later. The ``clinic_id`` used for
    tenancy scoping therefore comes from the stored row, never from a claim a client
    could have edited.
    """
    subject = payload.get("sub")
    if not subject:
        raise _UNAUTHORIZED

    user = db.get(ClinicUser, str(subject))
    if user is None:
        # Fall back to the email claim: a database rebuilt from a re-seed keeps the same
        # email but issues a new row id, and forcing a sign-in loop for that is noise.
        email = payload.get("email")
        user = crud.get_clinic_user_by_email(db, str(email)) if email else None

    if user is None or not user.is_active:
        raise _UNAUTHORIZED

    return user


def require_clinic_access(clinic_id: str, user: ClinicUser) -> None:
    """Assert the signed-in user belongs to ``clinic_id``.

    404 rather than 403 on a mismatch: confirming that another clinic's id exists is
    itself a disclosure, and a clinician has no legitimate way to learn one.
    """
    if clinic_id != user.clinic_id:
        logger.info("Cross-clinic access refused for user %s", user.id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clinic '{clinic_id}' not found",
        )
