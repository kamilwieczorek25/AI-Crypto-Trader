"""ORM model for open positions."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    quantity: Mapped[float] = mapped_column(Float)
    avg_entry_price: Mapped[float] = mapped_column(Float)
    current_price: Mapped[float] = mapped_column(Float, default=0.0)

    stop_loss_price: Mapped[float] = mapped_column(Float, default=0.0)
    take_profit_price: Mapped[float] = mapped_column(Float, default=0.0)
    # Trailing stop: track highest price seen since entry
    highest_price: Mapped[float] = mapped_column(Float, default=0.0)
    # Original SL distance in % (used to compute trailing SL)
    trailing_stop_pct: Mapped[float] = mapped_column(Float, default=0.0)
    # Source: "bot" (opened by the bot) or "external" (pre-existing on exchange)
    source: Mapped[str] = mapped_column(String(10), default="bot")

    @property
    def value_usdt(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl_usdt(self) -> float:
        return self.quantity * (self.current_price - self.avg_entry_price)

    @property
    def pnl_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price * 100
