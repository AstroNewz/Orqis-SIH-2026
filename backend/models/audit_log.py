import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from backend.db.base import Base


class AuditLog(Base):
    """Audit log tracking system events and data access for compliance."""
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(64), default="system", nullable=False)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
