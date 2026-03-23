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


@router.get("/market-info")
async def get_market_info() -> dict:
    """Return current market regime, BTC anchor data, and risk profile."""
    from app.services.bot_runner import bot_runner
    from app.services.claude_engine import get_profile_info

    regime = getattr(bot_runner, "_last_regime", {})
    return {
        "market_regime": regime,
        "risk_profile": get_profile_info(),
        "auto_risk_profile": settings.AUTO_RISK_PROFILE,
        "circuit_breaker_tripped": getattr(bot_runner, "_circuit_breaker_tripped", False),
        "max_drawdown_limit_pct": settings.MAX_DRAWDOWN_PCT,
    }


@router.get("/symbols")
async def get_symbols() -> list[str]:
    """Return the current symbol universe + position symbols for chart dropdown."""
    from app.services.market_data import market_data_service
    from app.services.fast_scanner import fast_scanner

    symbols: set[str] = set()
    # Add position symbols
    state = portfolio_service.get_state()
    for p in state.positions:
        symbols.add(p.symbol)
    # Add hot scanner symbols
    for s in fast_scanner.hot_symbols:
        symbols.add(s)
    # Add cached OHLCV symbols (already fetched in previous cycles)
    for key in market_data_service._ohlcv_cache:
        sym = key.split(":")[0]
        symbols.add(sym)
    # Sort alphabetically
    return sorted(symbols)


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


@router.get("/exchange-account")
async def get_exchange_account() -> dict:
    """Return full Binance account state — all balances, fiat, crypto, dust."""
    from app.services.market_data import market_data_service

    if settings.is_demo:
        return {"error": "Exchange account only available in real mode", "assets": []}

    if not settings.BINANCE_API_KEY:
        return {"error": "No Binance API key configured", "assets": []}

    try:
        balances = await market_data_service.fetch_spot_balances()
    except Exception as exc:
        return {"error": str(exc), "assets": []}

    if not balances:
        return {"error": "No balances returned (check API permissions)", "assets": []}

    # Fetch all tickers in one call for price resolution
    exchange = await market_data_service._get_exchange()
    try:
        tickers = await exchange.fetch_tickers()
    except Exception:
        tickers = {}

    stablecoins = {"USDT", "USDC", "BUSD", "TUSD", "FDUSD", "DAI"}
    fiats = {"USD", "EUR", "GBP", "PLN", "TRY", "BRL", "ARS", "UAH", "RUB",
             "NGN", "AUD", "JPY", "KRW", "INR", "ZAR", "CAD", "CHF",
             "CZK", "SEK", "NOK", "DKK", "HUF", "RON", "BGN", "HRK"}

    assets = []
    total_value = 0.0

    for asset, info in sorted(balances.items()):
        total = info["total"]
        free = info["free"]
        locked = round(total - free, 8)

        # Determine type and value
        if asset in stablecoins:
            asset_type = "stablecoin"
            value_usdt = total
            price = 1.0
            pair = ""
        elif asset in fiats:
            asset_type = "fiat"
            # Try to get fiat rate
            pair = f"{asset}/{settings.QUOTE_CURRENCY}"
            t = tickers.get(pair)
            if not t or not t.get("last"):
                pair = f"{asset}/USDT"
                t = tickers.get(pair)
            price = float(t["last"]) if t and t.get("last") else 0.0
            value_usdt = total * price if price > 0 else 0.0
        else:
            asset_type = "crypto"
            pair = f"{asset}/{settings.QUOTE_CURRENCY}"
            t = tickers.get(pair)
            price = float(t["last"]) if t and t.get("last") else 0.0
            if price <= 0:
                # Fallback to the other stablecoin
                fallback = "USDT" if settings.QUOTE_CURRENCY == "USDC" else "USDC"
                pair = f"{asset}/{fallback}"
                t = tickers.get(pair)
                price = float(t["last"]) if t and t.get("last") else 0.0
            if price <= 0:
                pair = ""
            value_usdt = total * price if price > 0 else 0.0

        # Check if bot manages this position
        pos = portfolio_service.get_position(f"{asset}/{settings.QUOTE_CURRENCY}") or portfolio_service.get_position(f"{asset}/USDT") or portfolio_service.get_position(f"{asset}/USDC")
        source = getattr(pos, "source", None) if pos else None

        total_value += value_usdt
        assets.append({
            "asset": asset,
            "type": asset_type,
            "total": total,
            "free": free,
            "locked": locked,
            "pair": pair,
            "price": round(price, 6),
            "value_usdt": round(value_usdt, 2),
            "managed_by": source,  # "bot", "external", or null
        })

    # Sort: stablecoins first, then by value descending
    type_order = {"stablecoin": 0, "crypto": 1, "fiat": 2}
    assets.sort(key=lambda a: (type_order.get(a["type"], 9), -a["value_usdt"]))

    return {
        "total_value_usdt": round(total_value, 2),
        "num_assets": len(assets),
        "assets": assets,
    }
