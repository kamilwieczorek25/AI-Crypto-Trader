"""Bot control endpoints — start, stop, mode, risk profile."""

import logging
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db, save_bot_state
from app.services.bot_runner import bot_runner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/bot")

# Generate a random admin token at startup (printed to logs)
_ADMIN_TOKEN = secrets.token_urlsafe(32)
logger.info("Admin API token for this session: %s", _ADMIN_TOKEN)


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
    _auth: None = Depends(_require_admin),
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
    from app.services.claude_engine import get_profile_info
    return {
        "running": bot_runner.is_running,
        "mode": settings.MODE,
        "risk_profile": get_profile_info(),
        "cycle_count": bot_runner._cycle_count,
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
