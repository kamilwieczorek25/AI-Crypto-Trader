"""ORM model for periodic portfolio snapshots."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    total_value_usdt: Mapped[float] = mapped_column(Float)
    cash_usdt: Mapped[float] = mapped_column(Float)
    positions_value_usdt: Mapped[float] = mapped_column(Float)
    total_pnl_usdt: Mapped[float] = mapped_column(Float)
    total_pnl_pct: Mapped[float] = mapped_column(Float)
    num_open_positions: Mapped[int] = mapped_column(Integer, default=0)
    positions_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON snapshot
