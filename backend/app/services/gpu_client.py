"""HTTP client for the remote GPU inference server.

When settings.GPU_SERVER_URL is set, this module proxies LSTM and RL
training/prediction to the remote GPU machine.  When the remote server
is unreachable, every call returns None so the caller can fall back to
local CPU inference.
"""

import logging
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient | None:
    global _client
    url = settings.GPU_SERVER_URL.rstrip("/") if settings.GPU_SERVER_URL else ""
    if not url:
        return None
    if _client is None or _client.is_closed:
        headers = {}
        if settings.GPU_SERVER_TOKEN:
            headers["Authorization"] = f"Bearer {settings.GPU_SERVER_TOKEN}"
        _client = httpx.AsyncClient(
            base_url=url,
            timeout=httpx.Timeout(settings.GPU_SERVER_TIMEOUT, connect=10),
            headers=headers,
        )
    return _client


def is_enabled() -> bool:
    return bool(settings.GPU_SERVER_URL)


async def health() -> dict | None:
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.get("/health")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("GPU server health check failed: %s", e)
        return None


async def train_lstm(candles: dict[str, list]) -> dict | None:
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.post("/train/lstm", json={"candles": candles})
        r.raise_for_status()
        data = r.json()
        logger.info("GPU LSTM train: %s", data)
        return data
    except Exception as e:
        logger.warning("GPU LSTM train failed, falling back to CPU: %s", e)
        return None


async def predict_lstm(candles: list) -> dict | None:
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.post("/predict/lstm", json={"candles": candles})
        r.raise_for_status()
        data = r.json()
        return data if data.get("status") == "ok" else None
    except Exception as e:
        logger.warning("GPU LSTM predict failed, falling back to CPU: %s", e)
        return None


async def train_rl(candles: dict[str, list]) -> dict | None:
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.post("/train/rl", json={"candles": candles})
        r.raise_for_status()
        data = r.json()
        logger.info("GPU RL train: %s", data)
        return data
    except Exception as e:
        logger.warning("GPU RL train failed, falling back to CPU: %s", e)
        return None


async def predict_rl(state: list[float]) -> dict | None:
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.post("/predict/rl", json={"state": state})
        r.raise_for_status()
        data = r.json()
        return data if data.get("trained") else None
    except Exception as e:
        logger.warning("GPU RL predict failed, falling back to CPU: %s", e)
        return None


async def sentiment(texts: list[str]) -> dict | None:
    """Score news headlines using GPU sentence-transformer embeddings."""
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.post("/sentiment", json={"texts": texts})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("GPU sentiment failed, falling back to keywords: %s", e)
        return None


async def predict_ensemble(
    candles: list,
    state: list[float],
    headlines: list[str] | None = None,
) -> dict | None:
    """Full ensemble prediction: Transformer + LSTM + RL + Sentiment."""
    c = _get_client()
    if c is None:
        return None
    try:
        payload = {"candles": candles, "state": state, "headlines": headlines or []}
        r = await c.post("/predict/ensemble", json=payload)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("GPU ensemble predict failed: %s", e)
        return None


async def monte_carlo(
    candles: list,
    entry_price: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    hours_ahead: int = 24,
    simulations: int = 10000,
) -> dict | None:
    """GPU Monte Carlo simulation for SL/TP probability estimation."""
    c = _get_client()
    if c is None:
        return None
    try:
        payload = {
            "candles": candles,
            "entry_price": entry_price,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "hours_ahead": hours_ahead,
            "simulations": simulations,
        }
        r = await c.post("/simulate/montecarlo", json=payload)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("GPU Monte Carlo failed: %s", e)
        return None


# ── New GPU model endpoints ──────────────────────────────────────────────────


async def train_mtf(candles: dict[str, dict[str, list]]) -> dict | None:
    """Train Multi-Timeframe Fusion model on cross-TF data."""
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.post("/train/mtf", json={"candles": candles})
        r.raise_for_status()
        data = r.json()
        logger.info("GPU MTF train: %s", data)
        return data
    except Exception as e:
        logger.warning("GPU MTF train failed: %s", e)
        return None


async def predict_mtf(candles: dict[str, list]) -> dict | None:
    """Predict using Multi-Timeframe Fusion (all TFs at once)."""
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.post("/predict/mtf", json={"candles": candles})
        r.raise_for_status()
        data = r.json()
        return data if data.get("status") == "ok" else None
    except Exception as e:
        logger.warning("GPU MTF predict failed: %s", e)
        return None


async def predict_volatility(candles: list) -> dict | None:
    """Predict future volatility for better SL/TP & Monte Carlo σ."""
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.post("/predict/volatility", json={"candles": candles})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("GPU volatility predict failed: %s", e)
        return None


async def detect_anomaly(candles: list) -> dict | None:
    """Detect anomalous price/volume patterns (pumps, flash crashes)."""
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.post("/detect/anomaly", json={"candles": candles})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("GPU anomaly detection failed: %s", e)
        return None


async def predict_exit(state: list[float]) -> dict | None:
    """Get optimal exit action for an open position."""
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.post("/predict/exit", json={"state": state})
        r.raise_for_status()
        data = r.json()
        return data if data.get("trained") else None
    except Exception as e:
        logger.warning("GPU exit predict failed: %s", e)
        return None


async def train_exit(experiences: list[dict]) -> dict | None:
    """Train Exit RL from position outcome experiences."""
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.post("/train/exit", json={"experiences": experiences})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("GPU exit train failed: %s", e)
        return None


async def explain_attention(candles: list) -> dict | None:
    """Extract attention weights showing which candles/features drove prediction."""
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.post("/explain/attention", json={"candles": candles})
        r.raise_for_status()
        data = r.json()
        return data if data.get("status") == "ok" else None
    except Exception as e:
        logger.warning("GPU attention explain failed: %s", e)
        return None


async def compute_correlations(candles: dict[str, list]) -> dict | None:
    """GPU-accelerated cross-symbol correlation matrix."""
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.post("/correlations", json={"candles": candles})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("GPU correlations failed: %s", e)
        return None


async def rank_momentum(candles: dict[str, list]) -> dict | None:
    """Cross-sectional momentum ranking across all symbols.

    Accepts {symbol: [ohlcv_candles]} for all symbols in the universe.
    Returns {symbol: percentile_0_to_1} — e.g. 0.95 = top 5% by risk-adj momentum.

    Uses multi-horizon returns (1h, 4h, 24h) normalised by realised volatility.
    """
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.post("/rank/momentum", json={"candles": candles})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("GPU momentum ranking failed: %s", e)
        return None


async def cluster_rotation(candles: dict[str, list]) -> dict | None:
    """Sector rotation clustering via GPU spectral analysis.

    Groups all symbols into 6–8 sectors by price correlation, then scores
    each sector's recent momentum.  Returns:
    {
      "sector_heat":  {symbol: heat_-1_to_1},   # per-symbol sector score
      "hot_sectors":  [{label, symbols, heat}],  # ranked sector list
      "cold_sectors": [{label, symbols, heat}],
    }
    """
    c = _get_client()
    if c is None:
        return None
    try:
        r = await c.post("/cluster/rotation", json={"candles": candles})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("GPU sector rotation failed: %s", e)
        return None
