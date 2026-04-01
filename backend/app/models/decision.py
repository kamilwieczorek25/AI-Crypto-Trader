"""ORM model for Claude trading decisions."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ClaudeDecision(Base):
    __tablename__ = "claude_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    # Raw Claude I/O (stored before execution)
    raw_prompt: Mapped[str] = mapped_column(Text)
    raw_response: Mapped[str] = mapped_column(Text)

    # Parsed decision fields
    action: Mapped[str] = mapped_column(String(4))              # BUY | SELL | HOLD
    symbol: Mapped[str] = mapped_column(String(20))
    timeframe: Mapped[str] = mapped_column(String(4))
    quantity_pct: Mapped[float] = mapped_column(Float)
    stop_loss_pct: Mapped[float] = mapped_column(Float)
    take_profit_pct: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    primary_signals: Mapped[str] = mapped_column(Text)          # JSON list
    risk_factors: Mapped[str] = mapped_column(Text)             # JSON list
    reasoning: Mapped[str] = mapped_column(Text)

    executed: Mapped[bool] = mapped_column(Boolean, default=False)
    risk_profile: Mapped[str | None] = mapped_column(String(20), nullable=True)
    model_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
