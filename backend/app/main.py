"""FastAPI application — lifespan wiring."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import bot, dashboard, health, ws
from app.config import settings
from app.database import AsyncSessionLocal, create_tables, load_bot_state
from app.services.bot_runner import bot_runner
from app.services.portfolio import portfolio_service
from app.utils.logging import configure_logging

# Models must be imported so Base.metadata picks them up
import app.models  # noqa: F401

_log = logging.getLogger(__name__)


async def _pretrain_ml() -> None:
    """Fetch historical candles and warm-start LSTM + RL agent in the background."""
    try:
        from app.services.lstm_model import lstm_predictor
        from app.services.market_data import market_data_service
        from app.services.rl_agent import rl_agent

        _log.info("ML pre-training: fetching top symbols…")
        symbols = await market_data_service.get_top_symbols()
        train_syms = symbols[:6]  # limit to avoid rate-limit pressure at startup

        all_candles: dict = {}
        for sym in train_syms:
            candles = await market_data_service.get_ohlcv(sym, "1h", limit=500)
            if candles:
                all_candles[sym] = candles

        if not all_candles:
            _log.warning("ML pre-training: no candle data — skipping")
            return

        _log.info("ML pre-training: %d symbols, training LSTM…", len(all_candles))
        await lstm_predictor.train_async(all_candles)

        _log.info("ML pre-training: pre-training RL agent…")
        await rl_agent.pretrain_async(all_candles)

        _log.info("ML pre-training complete.")
    except Exception as exc:
        _log.warning("ML pre-training failed (non-fatal): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    await create_tables()

    # Restore portfolio from DB
    async with AsyncSessionLocal() as db:
        await portfolio_service.load_from_db(db)

    # Restore persisted bot settings (risk profile, mode)
    state = await load_bot_state()
    if "risk_profile" in state:
        settings.RISK_PROFILE = state["risk_profile"]
    if "mode" in state and state["mode"] in ("demo", "real"):
        settings.MODE = state["mode"]

    # Warn early if Anthropic key is missing
    if not settings.ANTHROPIC_API_KEY:
        _log.warning(
            "ANTHROPIC_API_KEY is not set — bot will fail on first cycle. "
            "Add it to your .env file."
        )

    # Wire WebSocket hub into bot runner
    bot_runner.set_ws_hub(ws.ws_hub)

    # Kick off ML pre-training in the background (non-blocking)
    asyncio.create_task(_pretrain_ml(), name="ml_pretrain")

    yield

    # Cleanup
    await bot_runner.stop()
    from app.services.market_data import market_data_service
    await market_data_service.close()


app = FastAPI(title="AI Crypto Trader", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(dashboard.router)
app.include_router(bot.router)
app.include_router(ws.router)
