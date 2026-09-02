import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.db.base import Base


class Patient(Base):
    """Pseudonymous patient record identified by UUID v4."""
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    clinic_id: Mapped[str] = mapped_column(String(64), default="default_clinic", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    screenings: Mapped[list["Screening"]] = relationship(
        "Screening",
        back_populates="patient",
        cascade="all, delete-orphan",
    )
