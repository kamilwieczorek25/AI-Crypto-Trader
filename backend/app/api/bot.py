"""Bot control endpoints — start, stop, mode, risk profile."""

import logging
import os
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db, save_bot_state
from app.services.bot_runner import bot_runner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/bot")

# Generate a random admin token at startup
_ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN") or secrets.token_urlsafe(32)
if not os.environ.get("ADMIN_TOKEN"):
    # Only print to stdout (not logger) so it doesn't leak into log aggregation
    print(f"\n{'='*60}")
    print(f"  Admin API token: {_ADMIN_TOKEN}")
    print(f"  (set ADMIN_TOKEN env var to use a fixed token)")
    print(f"{'='*60}\n")


def _require_admin(x_admin_token: str = Header(default="")) -> None:
    """Protect destructive endpoints with the session admin token.

    The token is auto-generated at startup and printed in the logs.
    Pass it via the X-Admin-Token header.
    """
    if not x_admin_token or not secrets.compare_digest(x_admin_token, _ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid or missing admin token")


class ModeRequest(BaseModel):
    mode: str  # "demo" | "real"


@router.post("/reset-demo")
async def reset_demo(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Wipe all demo history and reset portfolio to initial balance."""
    from sqlalchemy import delete
    from app.models.decision import ClaudeDecision
    from app.models.position import Position
    from app.models.snapshot import PortfolioSnapshot
    from app.models.trade import Trade
    from app.services.portfolio import portfolio_service

    if bot_runner.is_running:
        await bot_runner.stop()

    # Wrap all deletes in a single transaction
    try:
        await db.execute(delete(Trade))
        await db.execute(delete(ClaudeDecision))
        await db.execute(delete(Position))
        await db.execute(delete(PortfolioSnapshot))
        await db.commit()
    except Exception:
        await db.rollback()
        raise

    # Reset in-memory portfolio state
    portfolio_service._cash_usdt = settings.DEMO_INITIAL_BALANCE
    portfolio_service._initial_value = settings.DEMO_INITIAL_BALANCE
    portfolio_service._positions.clear()

    # Clear the persisted cash so load_from_db won't restore stale value on next restart
    await save_bot_state("cash_usdt", str(settings.DEMO_INITIAL_BALANCE))

    logger.info("Demo history wiped — portfolio reset to $%.2f", settings.DEMO_INITIAL_BALANCE)
    return {"status": "reset", "cash_usdt": settings.DEMO_INITIAL_BALANCE}


@router.post("/start")
async def start_bot() -> dict:
    if bot_runner.is_running:
        return {"status": "already_running"}
    await bot_runner.start()
    return {"status": "started", "mode": settings.MODE}


@router.post("/stop")
async def stop_bot() -> dict:
    await bot_runner.stop()
    return {"status": "stopped"}


@router.get("/status")
async def bot_status() -> dict:
    from datetime import datetime, timezone
    from app.services.claude_engine import get_profile_info

    next_in: int | None = None
    if bot_runner._next_cycle_at and bot_runner._running:
        now = datetime.now(timezone.utc)
        elapsed = (now - bot_runner._next_cycle_at).total_seconds()
        interval = bot_runner._effective_cycle_interval()
        next_in = max(0, int(interval - elapsed))

    return {
        "running": bot_runner.is_running,
        "mode": settings.MODE,
        "risk_profile": get_profile_info(),
        "cycle_count": bot_runner._cycle_count,
        "last_cycle_at": bot_runner._last_cycle_at.isoformat() if bot_runner._last_cycle_at else None,
        "next_cycle_in_seconds": next_in,
        "market_regime": bot_runner._last_regime,
        "circuit_breaker_tripped": bot_runner._circuit_breaker_tripped,
        "less_fear": settings.LESS_FEAR,
    }


class RiskProfileRequest(BaseModel):
    profile: str  # conservative | balanced | aggressive | fast_profit


@router.get("/risk-profile")
async def get_risk_profile() -> dict:
    from app.services.claude_engine import get_profile_info, PROFILE_KEYS
    return {"current": get_profile_info(), "available": PROFILE_KEYS}


@router.post("/risk-profile")
async def set_risk_profile(req: RiskProfileRequest) -> dict:
    from app.services.claude_engine import PROFILE_KEYS, get_profile_info
    if req.profile not in PROFILE_KEYS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid profile. Choose from: {PROFILE_KEYS}",
        )
    settings.RISK_PROFILE = req.profile
    await save_bot_state("risk_profile", req.profile)
    logger.info("Risk profile switched to %s", req.profile)
    info = get_profile_info()
    await bot_runner._broadcast("BOT_STATUS", {
        "running": bot_runner.is_running,
        "mode": settings.MODE,
        "risk_profile": info,
        "cycle_count": bot_runner._cycle_count,
    })
    return {"risk_profile": info}


@router.post("/mode")
async def set_mode(
    req: ModeRequest,
    _auth: None = Depends(_require_admin),
) -> dict:
    if req.mode not in ("demo", "real"):
        raise HTTPException(status_code=400, detail="mode must be 'demo' or 'real'")
    if bot_runner.is_running:
        raise HTTPException(
            status_code=409,
            detail="Cannot switch mode while bot is running — stop it first",
        )
    if req.mode == "real" and not settings.REAL_TRADING:
        raise HTTPException(
            status_code=403,
            detail="Real trading requires REAL_TRADING=true in environment",
        )
    settings.MODE = req.mode
    await save_bot_state("mode", req.mode)
    logger.info("Mode switched to %s", req.mode)
    return {"mode": settings.MODE}


class LessFearRequest(BaseModel):
    enabled: bool


@router.get("/less-fear")
async def get_less_fear() -> dict:
    return {"enabled": settings.LESS_FEAR}


@router.post("/less-fear")
async def set_less_fear(req: LessFearRequest) -> dict:
    settings.LESS_FEAR = req.enabled
    await save_bot_state("less_fear", str(req.enabled).lower())
    logger.info("Less-fear mode %s", "ENABLED" if req.enabled else "DISABLED")
    # Broadcast updated status so frontend syncs immediately
    from app.services.claude_engine import get_profile_info
    await bot_runner._broadcast("BOT_STATUS", {
        "running": bot_runner.is_running,
        "mode": settings.MODE,
        "risk_profile": get_profile_info(),
        "cycle_count": bot_runner._cycle_count,
        "less_fear": settings.LESS_FEAR,
    })
    return {"enabled": settings.LESS_FEAR}


class AdoptRequest(BaseModel):
    symbol: str | None = None  # None = adopt all external positions
    sl_pct: float | None = None
    tp_pct: float | None = None


@router.post("/adopt-positions")
async def adopt_positions(
    req: AdoptRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Convert external positions to bot-managed with SL/TP."""
    from app.services.portfolio import portfolio_service

    adopted = []

    if req.symbol:
        pos = await portfolio_service.adopt_position(db, req.symbol, req.sl_pct, req.tp_pct)
        if pos:
            adopted.append(pos.symbol)
    else:
        for pos in portfolio_service.all_positions():
            if pos.source == "external":
                result = await portfolio_service.adopt_position(
                    db, pos.symbol, req.sl_pct, req.tp_pct,
                )
                if result:
                    adopted.append(result.symbol)

    # Broadcast updated portfolio so dashboard refreshes
    await bot_runner._broadcast("PORTFOLIO_UPDATE", portfolio_service.get_state().model_dump())

    return {"adopted": adopted, "count": len(adopted)}
