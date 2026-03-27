"""FastAPI application — lifespan wiring."""

import asyncio
import logging
import os
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

        # Train GPU Multi-Timeframe Fusion model (15m/1h/4h/1d simultaneously)
        from app.services import gpu_client
        if gpu_client.is_enabled():
            _log.info("ML pre-training: fetching multi-TF candles for MTF model…")
            all_tf_candles: dict[str, dict[str, list]] = {}
            for sym in train_syms:
                sym_tfs: dict[str, list] = {}
                for tf in ("15m", "1h", "4h", "1d"):
                    c = await market_data_service.get_ohlcv(sym, tf, limit=300)
                    if c:
                        sym_tfs[tf] = c
                if sym_tfs:
                    all_tf_candles[sym] = sym_tfs
            if all_tf_candles:
                result = await gpu_client.train_mtf(all_tf_candles)
                _log.info("GPU MTF model trained: %s", result)
            else:
                _log.warning("ML pre-training: no multi-TF data available — MTF skipped")

        _log.info("ML pre-training complete.")
    except Exception as exc:
        _log.warning("ML pre-training failed (non-fatal): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    try:
        await create_tables()
    except Exception as exc:
        _log.critical(
            "DATABASE SETUP FAILED: %s\n"
            "If using PostgreSQL, run:\n"
            "  sudo -u postgres psql -d ai_trader -c \"GRANT ALL ON SCHEMA public TO ai_trader;\"\n"
            "Then restart the container.",
            exc,
        )
        raise

    # Restore portfolio from DB
    async with AsyncSessionLocal() as db:
        await portfolio_service.load_from_db(db)

        # Sync with Binance exchange in real mode (read actual balances + holdings)
        if not settings.is_demo and settings.SYNC_EXCHANGE_ON_STARTUP:
            if settings.BINANCE_API_KEY and settings.BINANCE_SECRET:
                _log.info("Real mode: syncing portfolio from Binance exchange...")
                try:
                    sync_result = await portfolio_service.sync_from_exchange(db)
                    n_imp = len(sync_result.get("imported", []))
                    n_upd = len(sync_result.get("updated", []))
                    _log.info(
                        "Exchange sync done: %d imported, %d updated, USDT=$%.2f",
                        n_imp, n_upd, portfolio_service.cash_usdt,
                    )
                except Exception as exc:
                    _log.warning("Exchange sync failed (non-fatal): %s", exc)
            else:
                _log.warning(
                    "Real mode but no BINANCE_API_KEY/SECRET — "
                    "cannot sync exchange state. Using DB-only portfolio."
                )

    # Restore persisted bot settings (risk profile, mode, less_fear)
    state = await load_bot_state()
    if "risk_profile" in state:
        settings.RISK_PROFILE = state["risk_profile"]
    if "mode" in state and state["mode"] in ("demo", "real"):
        settings.MODE = state["mode"]
    if "less_fear" in state:
        db_val = state["less_fear"].lower() == "true"
        # .env wins when it explicitly enables Less Fear — the DB can turn it on
        # (user clicked the button) but cannot silently override an env-level true.
        settings.LESS_FEAR = db_val or settings.LESS_FEAR
        _log.info(
            "Less-fear mode: %s (env=%s db=%s)",
            "ENABLED" if settings.LESS_FEAR else "disabled",
            settings.LESS_FEAR, db_val,
        )

    # Warn early if Anthropic key is missing
    if not settings.ANTHROPIC_API_KEY:
        _log.warning(
            "ANTHROPIC_API_KEY is not set — bot will fail on first cycle. "
            "Add it to your .env file."
        )

    # Wire WebSocket hub into bot runner
    bot_runner.set_ws_hub(ws.ws_hub)

    # Check GPU server connectivity (if configured)
    from app.services import gpu_client
    if gpu_client.is_enabled():
        info = await gpu_client.health()
        if info:
            _log.info("GPU server connected: %s (%s, VRAM %.1fGB)",
                      info.get("gpu"), info.get("device"), info.get("vram_gb", 0))
        else:
            _log.warning("GPU_SERVER_URL is set but server is unreachable — ML will run on local CPU")

    # Start fast local scanner (background, zero API cost)
    from app.services.fast_scanner import fast_scanner
    asyncio.create_task(fast_scanner.start(), name="fast_scanner_start")

    # Start whale trade detector (Binance WebSocket, zero API cost)
    from app.services.whale_detector import whale_detector
    asyncio.create_task(whale_detector.start(), name="whale_detector_start")

    # Kick off ML pre-training in the background (non-blocking)
    asyncio.create_task(_pretrain_ml(), name="ml_pretrain")

    # Auto-start the bot so it runs immediately on server startup
    await bot_runner.start()

    yield

    # Cleanup
    await bot_runner.stop()
    from app.services.fast_scanner import fast_scanner as _fs
    await _fs.stop()
    from app.services.whale_detector import whale_detector as _wd
    await _wd.stop()
    from app.services.market_data import market_data_service
    await market_data_service.close()


app = FastAPI(title="AI Crypto Trader", lifespan=lifespan)

_ALLOWED_ORIGINS = [
    "http://localhost:9080",
    "http://127.0.0.1:9080",
    f"http://localhost:{os.environ.get('FRONTEND_PORT', '9080')}",
]
# Auto-allow the machine's own hostname/IP on the frontend port
import socket as _sock
try:
    _hostname = _sock.gethostname()
    _ip = _sock.gethostbyname(_hostname)
    _ALLOWED_ORIGINS.append(f"http://{_hostname}:9080")
    _ALLOWED_ORIGINS.append(f"http://{_ip}:9080")
except Exception:
    pass
# Allow user to add extra origins via env/.env (comma-separated)
# For Docker/local LAN: set CORS_ORIGINS=* to allow all origins
_extra = settings.CORS_ORIGINS or os.environ.get("CORS_ORIGINS", "")
if _extra == "*":
    _ALLOWED_ORIGINS = ["*"]
elif _extra:
    _ALLOWED_ORIGINS.extend(o.strip() for o in _extra.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=_ALLOWED_ORIGINS != ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["content-type", "x-admin-token"],
)

app.include_router(health.router)
app.include_router(dashboard.router)
app.include_router(bot.router)
app.include_router(ws.router)
