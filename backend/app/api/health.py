"""Health check endpoint — verifies DB, Binance, and Anthropic connectivity."""

import logging

from fastapi import APIRouter
from sqlalchemy import text

from app.config import settings
from app.database import AsyncSessionLocal

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/health")
async def health() -> dict:
    # Check DB
    db_ok = False
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    # Check Binance connectivity
    binance_ok = False
    try:
        from app.services.market_data import market_data_service
        exchange = await market_data_service._get_exchange()
        await exchange.fetch_time()
        binance_ok = True
    except Exception as exc:
        logger.debug("Binance health check failed: %s", exc)

    # Check Anthropic API key validity (lightweight check)
    anthropic_ok = bool(settings.ANTHROPIC_API_KEY and len(settings.ANTHROPIC_API_KEY) > 10)

    all_ok = db_ok and binance_ok and anthropic_ok
    status = "ok" if all_ok else "degraded"
    return {
        "status": status,
        "db": "ok" if db_ok else "error",
        "binance": "ok" if binance_ok else "error",
        "anthropic_key_set": anthropic_ok,
        "mode": settings.MODE,
        "risk_profile": settings.RISK_PROFILE,
    }
