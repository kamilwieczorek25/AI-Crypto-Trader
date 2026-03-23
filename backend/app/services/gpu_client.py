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
