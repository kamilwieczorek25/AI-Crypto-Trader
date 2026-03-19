"""Pydantic schemas for portfolio state."""

from datetime import datetime

from pydantic import BaseModel


class PositionOut(BaseModel):
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    value_usdt: float
    pnl_usdt: float
    pnl_pct: float
    stop_loss_price: float
    take_profit_price: float
    opened_at: datetime

    model_config = {"from_attributes": True}


class PortfolioState(BaseModel):
    total_value_usdt: float
    cash_usdt: float
    positions_value_usdt: float
    total_pnl_usdt: float
    total_pnl_pct: float
    num_open_positions: int
    positions: list[PositionOut] = []
