import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from backend.db.base import Base


class ClinicUser(Base):
    """A clinician or clinic administrator who signs in to the web portal.

    Distinct from :class:`~backend.models.patient.Patient` on purpose: a patient row is
    pseudonymous by design (PART 19), whereas a portal account needs a real login
    identity. Keeping them in separate tables means the clinician's email can never
    leak into a patient record or a screening export.

    ``clinic_id`` is the tenancy key and matches ``Patient.clinic_id``; every clinic
    route scopes its query to the value on the caller's token, so one clinic cannot
    read another's worklist.
    """
    __tablename__ = "clinic_users"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    """Stored lower-cased by the callers in ``backend.db.crud`` so lookup is
    case-insensitive without relying on database collation."""
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)
    """bcrypt hash from ``backend.auth.security.get_password_hash``. No plaintext
    password is ever stored, logged, or returned by any route."""
    clinic_id: Mapped[str] = mapped_column(
        String(64),
        default="default_clinic",
        index=True,
        nullable=False,
    )
    full_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    role: Mapped[str] = mapped_column(String(32), default="clinician", nullable=False)
    """``clinician`` or ``admin``. Read-only portal routes do not branch on it yet; it
    is stored so a later write surface has something to authorise against."""
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    """Cleared instead of deleting the row, so audit entries keep resolving to an actor."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
