import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.db.base import Base


class Screening(Base):
    """Screening session record for a patient."""
    __tablename__ = "screenings"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    patient_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
    )
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    scan_type: Mapped[str] = mapped_column(String(64), default="Intra-oral Scan", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False)
    
    # Clinical metadata (University of Peradeniya risk factors).
    # Nullable on purpose: NULL means "not collected", False means "asked, answered
    # no". The clinical encoder carries a separate unknown channel for each factor,
    # so collapsing the two here would discard information the model uses.
    #
    # Age and sex are accepted by the API for inference but deliberately NOT stored:
    # they are quasi-identifiers, and PART 19 asks for data minimisation. They affect
    # the probability and nothing else.
    smoking_history: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True, default=None)
    alcohol_consumption: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True, default=None)
    betel_quid: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True, default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    patient: Mapped["Patient"] = relationship("Patient", back_populates="screenings")
    result: Mapped[Optional["ScreeningResult"]] = relationship(
        "ScreeningResult",
        back_populates="screening",
        uselist=False,
        cascade="all, delete-orphan",
    )
