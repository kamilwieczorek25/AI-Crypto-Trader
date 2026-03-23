"""Additional market intelligence sources — all free, no API keys required.

Provides:
1. Fear & Greed Index  (Alternative.me)          — macro sentiment
2. CoinGecko trending  (CoinGecko API v3)        — social buzz / hype detection
3. Binance funding rates (Binance public API)     — derivatives positioning
4. Binance long/short ratio (Binance public API)  — crowd positioning
5. Open interest       (Binance Futures API)      — OI trend for conviction
6. BTC dominance       (CoinGecko global API)    — altseason detection
7. Short squeeze setup (funding + OI + price)    — per-symbol squeeze potential
"""

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(10.0, connect=5.0)

# ── In-memory caches (avoid rate limits) ─────────────────────────────────────
_cache: dict[str, tuple[float, Any]] = {}  # key -> (expires_ts, data)


def _get_cached(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    expires, data = entry
    if datetime.now(timezone.utc).timestamp() > expires:
        del _cache[key]
        return None
    return data


def _set_cached(key: str, data: Any, ttl_seconds: int = 300) -> None:
    expires = datetime.now(timezone.utc).timestamp() + ttl_seconds
    _cache[key] = (expires, data)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Fear & Greed Index  (Alternative.me — free, no key)
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_fear_greed() -> dict[str, Any]:
    """Fetch the Crypto Fear & Greed Index.

    Returns:
        {
            "value": 72,                   # 0-100 (0=extreme fear, 100=extreme greed)
            "label": "Greed",              # text classification
            "trend": "rising",             # vs yesterday
            "yesterday_value": 68,
            "signal": "caution_greed"      # actionable signal for the scorer
        }
    """
    cached = _get_cached("fear_greed")
    if cached is not None:
        return cached

    default = {"value": 50, "label": "Neutral", "trend": "flat",
               "yesterday_value": 50, "signal": "neutral"}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                "https://api.alternative.me/fng/",
                params={"limit": 2, "format": "json"},
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])

        if not data:
            return default

        today = int(data[0].get("value", 50))
        label = data[0].get("value_classification", "Neutral")
        yesterday = int(data[1].get("value", 50)) if len(data) > 1 else today

        trend = "rising" if today > yesterday + 3 else (
            "falling" if today < yesterday - 3 else "flat"
        )

        # Actionable signal
        if today >= 80:
            signal = "extreme_greed"     # contrarian: market may top
        elif today >= 65:
            signal = "caution_greed"     # momentum but be careful
        elif today <= 20:
            signal = "extreme_fear"      # contrarian: potential bottom
        elif today <= 35:
            signal = "fear"              # cautious buying territory
        else:
            signal = "neutral"

        result = {
            "value": today,
            "label": label,
            "trend": trend,
            "yesterday_value": yesterday,
            "signal": signal,
        }
        _set_cached("fear_greed", result, ttl_seconds=600)  # 10 min cache
        return result

    except Exception as exc:
        logger.warning("Fear & Greed fetch failed: %s", exc)
        return default


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CoinGecko Trending Coins  (free, no key, 10-30 req/min)
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_trending_coins() -> list[dict[str, Any]]:
    """Fetch CoinGecko's trending coins (by search interest).

    Returns list of:
        {"symbol": "PEPE", "name": "Pepe", "market_cap_rank": 25, "score": 0}
    """
    cached = _get_cached("trending")
    if cached is not None:
        return cached

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get("https://api.coingecko.com/api/v3/search/trending")
            resp.raise_for_status()
            coins = resp.json().get("coins", [])

        result = []
        for entry in coins[:10]:
            item = entry.get("item", {})
            result.append({
                "symbol": item.get("symbol", "").upper(),
                "name": item.get("name", ""),
                "market_cap_rank": item.get("market_cap_rank"),
                "score": item.get("score", 0),
            })

        _set_cached("trending", result, ttl_seconds=900)  # 15 min cache
        return result

    except Exception as exc:
        logger.warning("CoinGecko trending fetch failed: %s", exc)
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Binance Funding Rates  (public, no key)
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_funding_rates(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch current funding rates from Binance Futures.

    Positive rate = longs pay shorts (bullish crowd)
    Negative rate = shorts pay longs (bearish crowd)
    Extreme rates (>0.1% or <-0.1%) often precede reversals.

    Returns {symbol: {"rate": 0.0005, "signal": "bullish_crowd", "extreme": False}}
    """
    cached = _get_cached("funding")
    if cached is not None:
        return cached

    result: dict[str, dict] = {}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                "https://fapi.binance.com/fapi/v1/premiumIndex",
            )
            resp.raise_for_status()
            data = resp.json()

        # Build lookup: ETHUSDT -> ETH/USDT
        sym_map = {}
        for sym in symbols:
            base = sym.replace("/", "")
            sym_map[base] = sym

        for entry in data:
            binance_sym = entry.get("symbol", "")
            if binance_sym not in sym_map:
                continue

            rate = float(entry.get("lastFundingRate", 0))
            our_sym = sym_map[binance_sym]

            extreme = abs(rate) > 0.001  # 0.1%
            if rate > 0.0005:
                signal = "extreme_bullish_crowd" if extreme else "bullish_crowd"
            elif rate < -0.0005:
                signal = "extreme_bearish_crowd" if extreme else "bearish_crowd"
            else:
                signal = "neutral"

            result[our_sym] = {
                "rate": round(rate, 6),
                "rate_pct": round(rate * 100, 4),
                "signal": signal,
                "extreme": extreme,
            }

        _set_cached("funding", result, ttl_seconds=300)  # 5 min cache
        return result

    except Exception as exc:
        logger.warning("Funding rate fetch failed: %s", exc)
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Binance Long/Short Ratio  (public, no key)
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_long_short_ratios(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch Binance top trader long/short account ratio.

    Ratio > 1.5 = crowd is heavily long (contrarian bearish)
    Ratio < 0.67 = crowd is heavily short (contrarian bullish)

    Returns {symbol: {"ratio": 1.2, "signal": "crowd_long"}}
    """
    cached = _get_cached("ls_ratio")
    if cached is not None:
        return cached

    result: dict[str, dict] = {}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for sym in symbols[:15]:  # limit to avoid rate limits
                binance_sym = sym.replace("/", "")
                try:
                    resp = await client.get(
                        "https://fapi.binance.com/futures/data/topLongShortAccountRatio",
                        params={"symbol": binance_sym, "period": "1h", "limit": 1},
                    )
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    if not data:
                        continue

                    ratio = float(data[0].get("longShortRatio", 1.0))
                    long_pct = float(data[0].get("longAccount", 0.5)) * 100
                    short_pct = float(data[0].get("shortAccount", 0.5)) * 100

                    if ratio > 2.0:
                        signal = "extreme_crowd_long"   # contrarian bearish
                    elif ratio > 1.3:
                        signal = "crowd_long"
                    elif ratio < 0.5:
                        signal = "extreme_crowd_short"  # contrarian bullish
                    elif ratio < 0.77:
                        signal = "crowd_short"
                    else:
                        signal = "neutral"

                    result[sym] = {
                        "ratio": round(ratio, 3),
                        "long_pct": round(long_pct, 1),
                        "short_pct": round(short_pct, 1),
                        "signal": signal,
                    }
                except Exception:
                    continue

        _set_cached("ls_ratio", result, ttl_seconds=300)
        return result

    except Exception as exc:
        logger.warning("Long/short ratio fetch failed: %s", exc)
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Open Interest  (Binance Futures public API — no key)
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_open_interest(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch open interest for symbols and compute recent change %.

    Rising OI + rising price  = real buying (bullish continuation)
    Rising OI + falling price = short build-up (bearish)
    Falling OI                = positions closing (trend weakening)

    Returns {symbol: {"oi_usdt": float, "change_pct": float, "price_trend": int}}
    """
    cached = _get_cached("open_interest")
    if cached is not None:
        return cached

    result: dict[str, dict] = {}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for sym in symbols[:15]:
                binance_sym = sym.replace("/", "")
                try:
                    # Current OI
                    resp = await client.get(
                        "https://fapi.binance.com/fapi/v1/openInterest",
                        params={"symbol": binance_sym},
                    )
                    if resp.status_code != 200:
                        continue
                    current_oi = float(resp.json().get("openInterest", 0))

                    # Historical OI (last 2 data points at 1h interval)
                    resp_hist = await client.get(
                        "https://fapi.binance.com/futures/data/openInterestHist",
                        params={"symbol": binance_sym, "period": "1h", "limit": 2},
                    )
                    prev_oi = current_oi
                    price_trend = 0
                    if resp_hist.status_code == 200:
                        hist = resp_hist.json()
                        if hist and len(hist) >= 2:
                            prev_oi = float(hist[-2].get("sumOpenInterest", current_oi))
                            # Use sumOpenInterestValue (USDT) for price reference
                            val_now = float(hist[-1].get("sumOpenInterestValue", 0))
                            val_prev = float(hist[-2].get("sumOpenInterestValue", 0))
                            # If OI is flat but value changed → price moved
                            if val_now > val_prev * 1.005:
                                price_trend = 1   # price rising
                            elif val_now < val_prev * 0.995:
                                price_trend = -1  # price falling

                    change_pct = (
                        (current_oi - prev_oi) / prev_oi * 100 if prev_oi > 0 else 0
                    )

                    result[sym] = {
                        "oi_contracts": current_oi,
                        "change_pct": round(change_pct, 2),
                        "price_trend": price_trend,
                    }
                except Exception:
                    continue

        _set_cached("open_interest", result, ttl_seconds=300)
        return result

    except Exception as exc:
        logger.warning("Open interest fetch failed: %s", exc)
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BTC Dominance / Altseason signal  (CoinGecko /global — free, no key)
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_btc_dominance() -> dict[str, Any]:
    """Fetch BTC market dominance from CoinGecko global endpoint.

    BTC dominance falling = capital rotating into alts (altseason)
    BTC dominance rising  = capital fleeing into BTC (risk-off)

    Returns:
        {
            "btc_dominance":  42.5,       # percentage 0-100
            "trend":          "falling",  # "rising" | "falling" | "flat"
            "altseason":      True,       # dominance < 45% and falling
            "signal":         "altseason" # "altseason" | "btc_dominance" | "neutral"
        }
    """
    cached = _get_cached("btc_dominance")
    if cached is not None:
        return cached

    default = {
        "btc_dominance": 50.0,
        "trend": "flat",
        "altseason": False,
        "signal": "neutral",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get("https://api.coingecko.com/api/v3/global")
            resp.raise_for_status()
            data = resp.json().get("data", {})

        market_cap_pct = data.get("market_cap_percentage", {})
        btc_dom = float(market_cap_pct.get("btc", 50.0))

        # Simple trend: compare against cached prior reading
        prior = _get_cached("btc_dominance_prior")
        if prior is not None:
            delta = btc_dom - prior.get("btc_dominance", btc_dom)
            trend = "rising" if delta > 0.5 else ("falling" if delta < -0.5 else "flat")
        else:
            trend = "flat"
        _set_cached("btc_dominance_prior", {"btc_dominance": btc_dom}, ttl_seconds=3600)

        altseason = btc_dom < 45.0 and trend in ("falling", "flat")

        if btc_dom < 40.0:
            signal = "altseason"      # strong altseason conditions
        elif btc_dom < 45.0 and trend == "falling":
            signal = "altseason"      # developing altseason
        elif btc_dom > 58.0:
            signal = "btc_dominance"  # BTC dominance — alts may lag
        elif btc_dom > 52.0 and trend == "rising":
            signal = "btc_dominance"  # BTC taking market share
        else:
            signal = "neutral"

        result = {
            "btc_dominance": round(btc_dom, 2),
            "eth_dominance": round(float(market_cap_pct.get("eth", 0)), 2),
            "trend": trend,
            "altseason": altseason,
            "signal": signal,
        }
        _set_cached("btc_dominance", result, ttl_seconds=900)  # 15 min cache
        return result

    except Exception as exc:
        logger.warning("BTC dominance fetch failed: %s", exc)
        return default


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Short Squeeze Detector  (derived from existing funding + OI + price data)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_squeeze_setups(
    funding: dict[str, dict],
    open_interest: dict[str, dict],
    symbols_data: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Detect potential short squeeze setups per symbol.

    A short squeeze requires:
    1. Negative funding rate (shorts paying longs → heavy short crowd)
    2. Rising OI (short positions being added, not closed)
    3. Rising price (pressure building against shorts)

    When all three align, the squeeze potential is high — forced short
    covering can produce explosive upside moves.

    Returns {symbol: {"squeeze_score": 0-1, "signal": str, "reasons": [str]}}
    """
    results: dict[str, dict] = {}

    for sym in set(list(funding.keys()) + list(open_interest.keys())):
        f = funding.get(sym, {})
        oi = open_interest.get(sym, {})

        reasons: list[str] = []
        score = 0.0

        # Component 1: Negative funding (shorts paying longs)
        rate = f.get("rate", 0)
        if rate < -0.0005:   # < -0.05%
            score += 0.35
            reasons.append(f"neg_funding({rate*100:.4f}%)")
        elif rate < -0.0001:
            score += 0.15
            reasons.append(f"mild_neg_funding({rate*100:.4f}%)")

        # Component 2: Rising OI (shorts accumulating, not fleeing)
        oi_change = oi.get("change_pct", 0)
        if oi_change > 5:
            score += 0.35
            reasons.append(f"oi_rising(+{oi_change:.1f}%)")
        elif oi_change > 2:
            score += 0.15
            reasons.append(f"oi_mild_rising(+{oi_change:.1f}%)")

        # Component 3: Price rising against shorts (price_trend from OI endpoint)
        price_trend = oi.get("price_trend", 0)
        if price_trend > 0:
            score += 0.30
            reasons.append("price_rising_vs_shorts")
        elif price_trend == 0 and oi_change > 3:
            # Flat price with rising OI = coiling spring
            score += 0.15
            reasons.append("coiling_spring")

        if score >= 0.65:
            signal = "high_squeeze_potential"
        elif score >= 0.40:
            signal = "moderate_squeeze_potential"
        elif score > 0:
            signal = "low_squeeze_potential"
        else:
            signal = "no_squeeze"

        results[sym] = {
            "squeeze_score": round(score, 3),
            "signal": signal,
            "reasons": reasons,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Aggregated market intelligence (single call)
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_market_intelligence(
    symbols: list[str],
) -> dict[str, Any]:
    """Fetch all additional data sources in parallel.

    Returns:
    {
        "fear_greed":    {...},
        "trending":      [...],
        "funding":       {symbol: {...}},
        "long_short":    {symbol: {...}},
        "open_interest": {symbol: {...}},
        "btc_dominance": {...},
        "squeeze":       {symbol: {...}},   # derived, no extra API call
    }
    """
    import asyncio

    # Fire all requests concurrently (including new BTC dominance call)
    fg_task       = asyncio.create_task(fetch_fear_greed())
    trending_task = asyncio.create_task(fetch_trending_coins())
    funding_task  = asyncio.create_task(fetch_funding_rates(symbols))
    ls_task       = asyncio.create_task(fetch_long_short_ratios(symbols))
    oi_task       = asyncio.create_task(fetch_open_interest(symbols))
    btcd_task     = asyncio.create_task(fetch_btc_dominance())

    # Wait for all (each has its own error handling)
    fear_greed, trending, funding, long_short, open_interest, btc_dominance = (
        await asyncio.gather(fg_task, trending_task, funding_task, ls_task, oi_task, btcd_task)
    )

    # Derive squeeze setups from already-fetched funding + OI data (no extra API call)
    squeeze = detect_squeeze_setups(funding, open_interest)

    # Log summary
    trending_syms   = [t["symbol"] for t in trending[:5]]
    extreme_funding = [s for s, d in funding.items() if d.get("extreme")]
    oi_rising       = [s for s, d in open_interest.items() if d.get("change_pct", 0) > 5]
    squeeze_hot     = [s for s, d in squeeze.items() if d.get("signal") == "high_squeeze_potential"]
    logger.info(
        "Market intel: FG=%d(%s) BTC.D=%.1f%%(%s) trending=%s extreme_funding=%s "
        "oi_rising=%s squeeze=%s",
        fear_greed.get("value", 0), fear_greed.get("label", "?"),
        btc_dominance.get("btc_dominance", 0), btc_dominance.get("signal", "?"),
        ",".join(trending_syms) or "none",
        ",".join(extreme_funding) or "none",
        ",".join(oi_rising) or "none",
        ",".join(squeeze_hot) or "none",
    )

    return {
        "fear_greed":    fear_greed,
        "trending":      trending,
        "funding":       funding,
        "long_short":    long_short,
        "open_interest": open_interest,
        "btc_dominance": btc_dominance,
        "squeeze":       squeeze,
    }
