"""Pydantic schemas for WebSocket event payloads."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class WsEvent(BaseModel):
    type: str
    data: Any


class BotStatusData(BaseModel):
    running: bool
    mode: str
    next_cycle_in_seconds: int | None = None
    cycle_count: int = 0
    last_cycle_at: datetime | None = None


class TradeHistoryItem(BaseModel):
    id: int
    created_at: datetime
    symbol: str
    direction: str
    mode: str
    quantity: float
    price: float
    quantity_pct: float
    pnl_usdt: float | None
    pnl_pct: float | None

    model_config = {"from_attributes": True}


class DecisionOut(BaseModel):
    id: int
    created_at: datetime
    action: str
    symbol: str
    timeframe: str
    quantity_pct: float
    confidence: float
    primary_signals: list[str]
    risk_factors: list[str]
    reasoning: str
    executed: bool

    model_config = {"from_attributes": True}
