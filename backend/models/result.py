import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Boolean, DateTime, Float, Integer, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.db.base import Base


class ScreeningResult(Base):
    """Inference and risk classification result for a screening."""
    __tablename__ = "screening_results"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    screening_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("screenings.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    risk_level: Mapped[str] = mapped_column(String(64), nullable=False)
    details: Mapped[str] = mapped_column(Text, nullable=False)
    
    classical_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quantum_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, default=0.50, nullable=False)
    classification: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), default="v1.0.0-qml", nullable=False)

    # --- Provenance (PART 20: calibration metadata, execution metadata, audit) ---
    # All nullable so rows written before this existed still load. Every field here
    # answers a question an auditor can legitimately ask about a stored result:
    # which score the circuit produced, what turned it into a probability, where the
    # band boundaries came from, and on which backend it ran.
    inference_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    raw_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    """The quantum measurement before calibration. Kept so a recalibration can be
    replayed against stored results without re-running any circuits."""
    probability_uncalibrated: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    high_risk_threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    calibration_method: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    bands_source: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    """``calibration_artifact``, ``configured_fallback``, or ``mock``. Distinguishes a
    result whose bands were derived for this model from one that fell back to config."""
    execution_mode: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    backend_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    circuit_depth: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    feature_mode: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    quantum_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    quantum_qubits: Mapped[int] = mapped_column(Integer, default=8)
    quantum_shots: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    """Sampling shots, or NULL under exact statevector simulation where no sampling
    occurs. Storing 1024 there would claim shot noise the result does not have.

    No column ``default``: SQLAlchemy applies a Python-side default whenever the
    attribute is None at INSERT, so a default would silently overwrite the NULL that
    exact simulation is supposed to record."""
    execution_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_mock: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    screening: Mapped["Screening"] = relationship("Screening", back_populates="result")
