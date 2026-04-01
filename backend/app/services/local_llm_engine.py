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

# ── Pull deduplication guard ──────────────────────────────────────────────────
_pull_in_progress: bool = False         # prevents concurrent model pulls


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
    """Check Ollama reachability AND that the configured model is present."""
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
                if resp.status != 200:
                    _available = False
                else:
                    data = await resp.json()
                    pulled = [m["name"] for m in data.get("models", [])]
                    model = settings.LOCAL_LLM_MODEL
                    # Exact match or "<model>:<digest>" suffix — do NOT match on
                    # just the base name (e.g. "qwen2.5") so qwen2.5:7b does NOT
                    # satisfy a requirement for qwen2.5:14b.
                    _available = any(
                        name == model or name.startswith(model + ":")
                        for name in pulled
                    )
                    if not _available:
                        logger.warning(
                            "Ollama: model '%s' not in tag list (available: %s) — "
                            "scheduling pull",
                            model, pulled or "none",
                        )
                        asyncio.ensure_future(check_and_pull_model())
    except Exception:
        _available = False
    _available_checked_at = time.monotonic()
    logger.debug("Ollama availability (model=%s): %s", settings.LOCAL_LLM_MODEL, _available)


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
                        # 404 = model not loaded — mark unavailable immediately so
                        # subsequent calls skip Ollama and fall back to Claude while
                        # the pull runs in the background.
                        if resp.status == 404:
                            global _available, _available_checked_at
                            _available = False
                            _available_checked_at = time.monotonic()
                            asyncio.ensure_future(_refresh_availability())
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
        # Sanitise numeric fields before Pydantic validation — the LLM sometimes
        # returns negative percentages (e.g. stop_loss_pct=-10.0) which would
        # fail the ge=0 constraint before the post-construction fixups run.
        for _pct_field in ("stop_loss_pct", "take_profit_pct", "quantity_pct", "sell_pct"):
            if _pct_field in parsed and isinstance(parsed[_pct_field], (int, float)):
                parsed[_pct_field] = abs(float(parsed[_pct_field]))
        if "confidence" in parsed and isinstance(parsed["confidence"], (int, float)):
            parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
        # Confidence discount: local models tend to be overconfident compared to
        # Claude.  Apply a 0.7× multiplier so that marginal BUY decisions are
        # more likely to fall below the profile's min_confidence threshold and
        # get downgraded to HOLD.
        if "confidence" in parsed:
            _raw_conf = parsed["confidence"]
            parsed["confidence"] = round(parsed["confidence"] * 0.7, 4)
            logger.info(
                "LocalLLM: confidence discount 0.7× applied: %.2f → %.2f",
                _raw_conf, parsed["confidence"],
            )
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

    Designed to run as a non-blocking asyncio task at startup or when a 404 is
    detected.  A global flag prevents concurrent pulls from racing.
    The download can take 5–30 min; the bot falls back to Claude during that time.
    """
    global _pull_in_progress
    if _pull_in_progress:
        logger.debug("Ollama: pull already in progress — skipping duplicate request")
        return
    _pull_in_progress = True

    model = settings.LOCAL_LLM_MODEL
    url   = _get_ollama_url()

    if not url:
        logger.info("Ollama: no URL configured (LOCAL_LLM_URL and GPU_SERVER_URL both unset) — skipping")
        _pull_in_progress = False
        return

    # Brief pause to let the GPU VM's Ollama process finish booting
    await asyncio.sleep(5)

    logger.info("Ollama: checking %s/api/tags for model '%s'", url, model)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning("Ollama /api/tags returned HTTP %d — Claude fallback active", resp.status)
                    return
                tags_data = await resp.json()

    except aiohttp.ClientConnectorError as exc:
        logger.warning(
            "Ollama unreachable at %s — not installed or not yet started "
            "(Claude fallback active). Detail: %r", url, exc,
        )
        return
    except asyncio.TimeoutError:
        logger.warning("Ollama /api/tags timed out at %s — Claude fallback active", url)
        return
    except Exception as exc:
        logger.warning("Ollama /api/tags unexpected error (%s: %r) — Claude fallback active",
                       type(exc).__name__, exc)
        return

    pulled_names: list[str] = [m["name"] for m in tags_data.get("models", [])]
    # Exact match OR "<model>:<digest>" suffix only — do NOT match on just the
    # base name so "qwen2.5:7b" does NOT satisfy "qwen2.5:14b".
    already_pulled = any(
        name == model or name.startswith(model + ":")
        for name in pulled_names
    )

    if already_pulled:
        logger.info("Ollama: model '%s' already present — no pull needed", model)
        global _available, _available_checked_at
        _available = True
        _available_checked_at = time.monotonic()
        return

    logger.info(
        "Ollama: model '%s' not found (available: %s) — starting pull (~8 GB)…",
        model, pulled_names or "none",
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{url}/api/pull",
                json={"name": model, "stream": False},
                timeout=aiohttp.ClientTimeout(total=3600),
            ) as resp:
                pull_data = await resp.json()
                status = pull_data.get("status", "unknown")
                logger.info("Ollama pull '%s' completed — status: %s", model, status)

        _available = True
        _available_checked_at = time.monotonic()

    except Exception as exc:
        logger.warning("Ollama pull '%s' failed (%s: %r) — Claude fallback active",
                       model, type(exc).__name__, exc)
    finally:
        _pull_in_progress = False
