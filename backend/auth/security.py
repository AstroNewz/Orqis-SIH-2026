import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import HTTPException, status, Header

from backend.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def generate_pseudonymous_id() -> str:
    """Generate a RFC 4122 compliant UUID v4 pseudonymous identifier."""
    return str(uuid.uuid4())


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password with bcrypt."""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def verify_jwt_token(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """
    Verify optional or required JWT Authorization Bearer token (supports Supabase/Custom JWT).
    Allows unauthenticated access for local screening demos while validating if present.
    """
    if not authorization:
        return None

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme",
            )
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            options={"verify_aud": False},
        )
        return payload
    except (JWTError, ValueError):
        # In development, fail gracefully with 401 if a bad token is sent
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
