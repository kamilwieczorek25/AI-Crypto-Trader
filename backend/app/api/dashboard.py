"""Dashboard REST endpoints — portfolio, positions, trades, decisions, credits."""

import json
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.decision import ClaudeDecision
from app.models.snapshot import PortfolioSnapshot
from app.models.trade import Trade
from app.schemas.dashboard import DecisionOut, TradeHistoryItem
from app.services.portfolio import portfolio_service

router = APIRouter(prefix="/api")


@router.get("/portfolio")
async def get_portfolio() -> dict:
    return portfolio_service.get_state().model_dump()


@router.get("/positions")
async def get_positions() -> list[dict]:
    state = portfolio_service.get_state()
    return [p.model_dump() for p in state.positions]


@router.get("/trades")
async def get_trades(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> dict:
    offset = (page - 1) * page_size
    result = await db.execute(
        select(Trade).order_by(desc(Trade.created_at)).offset(offset).limit(page_size)
    )
    trades = result.scalars().all()
    items = [TradeHistoryItem.model_validate(t).model_dump() for t in trades]
    return {"page": page, "page_size": page_size, "items": items}


@router.get("/decisions")
async def get_decisions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> dict:
    offset = (page - 1) * page_size
    result = await db.execute(
        select(ClaudeDecision)
        .order_by(desc(ClaudeDecision.created_at))
        .offset(offset)
        .limit(page_size)
    )
    decisions = result.scalars().all()

    items = []
    for d in decisions:
        items.append(
            DecisionOut(
                id=d.id,
                created_at=d.created_at,
                action=d.action,
                symbol=d.symbol,
                timeframe=d.timeframe,
                quantity_pct=d.quantity_pct,
                confidence=d.confidence,
                primary_signals=json.loads(d.primary_signals or "[]"),
                risk_factors=json.loads(d.risk_factors or "[]"),
                reasoning=d.reasoning,
                executed=d.executed,
            ).model_dump()
        )
    return {"page": page, "page_size": page_size, "items": items}


@router.get("/snapshots")
async def get_snapshots(
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    result = await db.execute(
        select(PortfolioSnapshot)
        .order_by(PortfolioSnapshot.created_at)
        .limit(limit)
    )
    snaps = result.scalars().all()
    return [
        {
            "created_at": s.created_at.isoformat(),
            "total_value_usdt": s.total_value_usdt,
            "cash_usdt": s.cash_usdt,
            "positions_value_usdt": s.positions_value_usdt,
            "total_pnl_usdt": s.total_pnl_usdt,
            "total_pnl_pct": s.total_pnl_pct,
        }
        for s in snaps
    ]


@router.get("/analytics")
async def get_analytics(db: AsyncSession = Depends(get_db)) -> dict:
    """Return aggregate trade statistics for the analytics panel."""
    from sqlalchemy import func

    # Use SQL aggregation where possible
    result = await db.execute(
        select(Trade)
        .where(Trade.direction == "SELL", Trade.pnl_usdt.isnot(None))
        .order_by(Trade.created_at)
    )
    sell_trades = result.scalars().all()

    total_result = await db.execute(select(func.count()).select_from(Trade))
    total_trades = total_result.scalar() or 0

    wins = [t for t in sell_trades if t.pnl_usdt > 0]
    losses = [t for t in sell_trades if t.pnl_usdt <= 0]
    pnls = [t.pnl_usdt for t in sell_trades]
    fees = sum(t.fee_usdt for t in sell_trades if t.fee_usdt)

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    current_streak = 0
    for t in sell_trades:
        if t.pnl_usdt > 0:
            current_streak = max(current_streak + 1, 1) if current_streak >= 0 else 1
            max_consec_wins = max(max_consec_wins, current_streak)
        else:
            current_streak = min(current_streak - 1, -1) if current_streak <= 0 else -1
            max_consec_losses = max(max_consec_losses, abs(current_streak))

    # Max drawdown from portfolio snapshots
    snap_result = await db.execute(
        select(PortfolioSnapshot.total_value_usdt)
        .order_by(PortfolioSnapshot.created_at)
    )
    values = [v for (v,) in snap_result.all()]
    max_drawdown_pct = 0.0
    if values:
        peak = values[0]
        for v in values:
            peak = max(peak, v)
            drawdown = (peak - v) / peak * 100 if peak > 0 else 0
            max_drawdown_pct = max(max_drawdown_pct, drawdown)

    # Simplified Sharpe ratio (annualized, assuming 5min cycles)
    sharpe_ratio = None
    if len(pnls) >= 2:
        import numpy as np
        returns = np.array(pnls)
        avg_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        if std_ret > 0:
            # ~288 cycles/day * 365 = ~105,120 / year
            sharpe_ratio = round(float(avg_ret / std_ret * (288 * 365) ** 0.5), 2)

    return {
        "total_trades": total_trades,
        "closed_trades": len(sell_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(sell_trades) if sell_trades else None,
        "avg_pnl_usdt": sum(pnls) / len(pnls) if pnls else 0.0,
        "best_pnl_usdt": max(pnls, default=0.0),
        "worst_pnl_usdt": min(pnls, default=0.0),
        "total_realized_pnl": sum(pnls),
        "total_fees_usdt": round(fees, 4),
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "sharpe_ratio": sharpe_ratio,
    }


@router.get("/usage")
async def get_usage() -> dict:
    """Return session Claude API usage and estimated cost."""
    from app.services.claude_engine import get_usage_stats
    return get_usage_stats()


@router.get("/ohlcv/{symbol}/{timeframe}")
async def get_ohlcv(symbol: str, timeframe: str) -> list[dict]:
    """Return OHLCV data for the chart (symbol uses _ instead of /)."""
    from app.services.market_data import market_data_service

    sym = symbol.replace("_", "/")
    ohlcv = await market_data_service.get_ohlcv(sym, timeframe, limit=200)
    return [
        {
            "t": int(row[0]),
            "o": row[1],
            "h": row[2],
            "l": row[3],
            "c": row[4],
            "v": row[5],
        }
        for row in ohlcv
    ]
