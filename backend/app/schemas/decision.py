"""Pydantic schema for Claude's trading decision output."""

from typing import Annotated

from pydantic import BaseModel, Field


class TradeDecision(BaseModel):
    action: str = Field(..., pattern="^(BUY|SELL|HOLD)$")
    symbol: str
    timeframe: str
    quantity_pct: Annotated[float, Field(ge=0.0, le=100.0)]
    stop_loss_pct: Annotated[float, Field(ge=0.0, le=20.0)]
    take_profit_pct: Annotated[float, Field(ge=0.0, le=50.0)]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    sell_pct: Annotated[float, Field(ge=0.0, le=100.0)] = 100.0  # partial sell: 0-100%
    primary_signals: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    reasoning: str = ""
