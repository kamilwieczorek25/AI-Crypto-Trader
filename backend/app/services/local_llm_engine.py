"""Local LLM engine — Ollama-backed drop-in replacement for claude_engine.

Public interface mirrors claude_engine:
  call_local_llm(prompt, validation_mode) -> (TradeDecision, raw_json)
  is_available() -> bool
  check_and_pull_model() — async, call once at startup (non-blocking task)
"""

import asyncio
import json
import logging
import time
from typing import Any

import aiohttp

from app.config import settings
from app.schemas.decision import TradeDecision

# Re-use prompt builders and profile helpers from claude_engine — no duplication
from app.services.claude_engine import (
    _build_system_prompt,
    _get_profile,
)

logger = logging.getLogger(__name__)


def _get_ollama_url() -> str | None:
    """Resolve the Ollama base URL.

    Priority:
    1. settings.LOCAL_LLM_URL if explicitly set
    2. Derived from settings.GPU_SERVER_URL — same host, port 11434
       e.g. http://192.168.1.50:9090  →  http://192.168.1.50:11434
    3. None (disabled)
    """
    if settings.LOCAL_LLM_URL:
        return settings.LOCAL_LLM_URL.rstrip("/")
    if settings.GPU_SERVER_URL:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(settings.GPU_SERVER_URL)
        derived = urlunparse(parsed._replace(netloc=f"{parsed.hostname}:11434"))
        return derived.rstrip("/")
    return None


# ── Custom exception ─────────────────────────────────────────────────────────

class LLMUnavailableError(RuntimeError):
    """Raised when Ollama is unreachable — triggers Claude fallback."""


# ── Availability cache ───────────────────────────────────────────────────────

_available: bool | None = None          # None = never checked
_available_checked_at: float = 0.0      # epoch seconds
_AVAILABLE_CACHE_TTL: float = 30.0      # re-check interval


def is_available() -> bool:
    """Return whether Ollama is reachable (cached 30 s, non-blocking)."""
    global _available, _available_checked_at
    now = time.monotonic()
    if _available is None or (now - _available_checked_at) > _AVAILABLE_CACHE_TTL:
        # Schedule a background refresh without blocking the caller.
        # On the very first call we return False until the first check completes.
        asyncio.ensure_future(_refresh_availability())
        if _available is None:
            return False
    return _available


async def _refresh_availability() -> None:
    global _available, _available_checked_at
    url = _get_ollama_url()
    if not url:
        _available = False
        _available_checked_at = time.monotonic()
        return
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                _available = resp.status == 200
    except Exception:
        _available = False
    _available_checked_at = time.monotonic()
    logger.debug("Ollama availability: %s", _available)


# ── JSON schema for constrained generation ───────────────────────────────────

_TRADE_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action":          {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "symbol":          {"type": "string"},
        "timeframe":       {"type": "string"},
        "quantity_pct":    {"type": "number"},
        "stop_loss_pct":   {"type": "number"},
        "take_profit_pct": {"type": "number"},
        "confidence":      {"type": "number"},
        "sell_pct":        {"type": "number"},
        "primary_signals": {"type": "array", "items": {"type": "string"}},
        "risk_factors":    {"type": "array", "items": {"type": "string"}},
        "reasoning":       {"type": "string"},
    },
    "required": [
        "action", "symbol", "timeframe", "quantity_pct",
        "stop_loss_pct", "take_profit_pct", "confidence",
        "primary_signals", "risk_factors", "reasoning",
    ],
}

# ── Usage tracking (in-memory, resets on restart) ────────────────────────────

_usage_stats: dict[str, Any] = {
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_calls": 0,
    "calls_succeeded": 0,
    "calls_failed": 0,
}


def get_usage_stats() -> dict:
    return dict(_usage_stats)


def _record_usage(prompt_tokens: int, completion_tokens: int, *, success: bool) -> None:
    _usage_stats["total_prompt_tokens"]     += prompt_tokens
    _usage_stats["total_completion_tokens"] += completion_tokens
    _usage_stats["total_calls"]             += 1
    if success:
        _usage_stats["calls_succeeded"] += 1
    else:
        _usage_stats["calls_failed"] += 1
    logger.info(
        "LocalLLM [%s]: prompt=%d completion=%d | total_calls=%d",
        settings.LOCAL_LLM_MODEL, prompt_tokens, completion_tokens,
        _usage_stats["total_calls"],
    )


# ── Core inference call ───────────────────────────────────────────────────────

async def call_local_llm(
    prompt: str,
    validation_mode: bool = False,
) -> tuple[TradeDecision, str]:
    """Call Ollama with JSON-schema constrained generation.

    Args:
        prompt: The user prompt (market data).
        validation_mode: If True, use validator system prompt (quant-first flow).

    Returns (TradeDecision, raw_json_str).
    One retry on parse/validation failure.
    Raises LLMUnavailableError if Ollama is unreachable.
    """
    system_prompt = _build_system_prompt(validation_mode=validation_mode)
    profile = _get_profile()

    payload = {
        "model": settings.LOCAL_LLM_MODEL,
        "stream": False,
        "options": {"temperature": 0.1},
        "format": _TRADE_DECISION_SCHEMA,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
    }

    ollama_url = _get_ollama_url()
    if not ollama_url:
        raise LLMUnavailableError("No Ollama URL configured (set LOCAL_LLM_URL or GPU_SERVER_URL)")

    async def _attempt() -> tuple[TradeDecision, str]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{ollama_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=settings.LOCAL_LLM_TIMEOUT),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise LLMUnavailableError(
                            f"Ollama returned HTTP {resp.status}: {body[:200]}"
                        )
                    data = await resp.json()
        except aiohttp.ClientConnectorError as exc:
            raise LLMUnavailableError(f"Ollama unreachable: {exc}") from exc

        # Token usage
        prompt_tokens     = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        raw_content: str = data["message"]["content"]
        parsed = json.loads(raw_content)
        decision = TradeDecision(**parsed)

        # Enforce profile position-size cap (mirrors claude_engine behaviour)
        decision.quantity_pct = min(decision.quantity_pct, profile.max_position_pct)

        if decision.stop_loss_pct <= 0 and decision.action in ("BUY", "SELL"):
            decision.stop_loss_pct = 3.0
        if decision.take_profit_pct <= 0 and decision.action in ("BUY", "SELL"):
            decision.take_profit_pct = 10.0

        if (
            not validation_mode
            and decision.confidence < profile.min_confidence
            and decision.action != "HOLD"
        ):
            logger.info(
                "LocalLLM: confidence %.2f below profile min %.2f — overriding to HOLD",
                decision.confidence, profile.min_confidence,
            )
            decision.action = "HOLD"
            decision.quantity_pct = 0.0
            decision.stop_loss_pct = 0.0
            decision.take_profit_pct = 0.0

        _record_usage(prompt_tokens, completion_tokens, success=True)
        return decision, raw_content

    try:
        return await _attempt()
    except LLMUnavailableError:
        _record_usage(0, 0, success=False)
        raise
    except Exception as first_err:
        logger.warning("LocalLLM first attempt failed (%s) — retrying", first_err)
        try:
            return await _attempt()
        except LLMUnavailableError:
            _record_usage(0, 0, success=False)
            raise
        except Exception as second_err:
            _record_usage(0, 0, success=False)
            raise RuntimeError(
                f"LocalLLM failed after retry: {second_err}"
            ) from second_err


# ── Startup model pull ────────────────────────────────────────────────────────

async def check_and_pull_model() -> None:
    """Check whether the configured model is already pulled; pull it if not.

    Designed to run as a non-blocking asyncio task at startup.
    The download can take 5–30 min; the bot falls back to Claude during that time.
    """
    model = settings.LOCAL_LLM_MODEL
    url   = _get_ollama_url()

    if not url:
        logger.info("Ollama: no URL configured (LOCAL_LLM_URL and GPU_SERVER_URL both unset) — skipping")
        return

    # Brief pause to let the GPU VM's Ollama process finish booting
    await asyncio.sleep(5)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning("Ollama /api/tags returned %d — skipping model pull", resp.status)
                    return
                tags_data = await resp.json()

        pulled_names: list[str] = [m["name"] for m in tags_data.get("models", [])]
        # Ollama stores names as "qwen2.5:14b" — match by prefix in case of digest suffix
        already_pulled = any(
            name == model or name.startswith(model + ":") or name.startswith(model.split(":")[0])
            for name in pulled_names
        )

        if already_pulled:
            logger.info("Ollama: model '%s' already present — no pull needed", model)
            # Mark as available immediately
            global _available, _available_checked_at
            _available = True
            _available_checked_at = time.monotonic()
            return

        logger.info(
            "Ollama: model '%s' not found — starting pull (~8 GB, may take several minutes)…",
            model,
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{url}/api/pull",
                json={"name": model, "stream": False},
                timeout=aiohttp.ClientTimeout(total=3600),  # up to 1 h for large models
            ) as resp:
                pull_data = await resp.json()
                status = pull_data.get("status", "unknown")
                logger.info("Ollama pull '%s' completed — status: %s", model, status)

        # Mark as available after successful pull
        _available = True
        _available_checked_at = time.monotonic()

    except Exception as exc:
        logger.warning("Ollama model pull failed (non-fatal, Claude fallback active): %s", exc)
