"""ORM model for executed trades."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    symbol: Mapped[str] = mapped_column(String(20))
    direction: Mapped[str] = mapped_column(String(4))          # BUY | SELL
    mode: Mapped[str] = mapped_column(String(8))               # demo | real
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    quantity_pct: Mapped[float] = mapped_column(Float)
    stop_loss_pct: Mapped[float] = mapped_column(Float, default=0.0)
    take_profit_pct: Mapped[float] = mapped_column(Float, default=0.0)
    order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # P&L (filled in when position is closed)
    pnl_usdt: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    fee_usdt: Mapped[float] = mapped_column(Float, default=0.0)

    decision_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
