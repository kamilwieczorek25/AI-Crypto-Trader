"""News sentiment service — CryptoCompare free API + keyword scoring."""

import logging
import re
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_BASE_URL = "https://min-api.cryptocompare.com/data/v2/news/"

# Positive / negative keyword lists for fast sentiment scoring
_POSITIVE = {
    "bullish", "surge", "rally", "soar", "skyrocket", "breakout", "adopt",
    "partnership", "upgrade", "launch", "approval", "etf", "institutional",
    "accumulate", "growth", "profit", "gains", "strong", "milestone",
    "record", "integration", "mainnet",
}
_NEGATIVE = {
    "bearish", "crash", "dump", "plunge", "hack", "exploit", "ban",
    "regulation", "lawsuit", "sec", "fraud", "fear", "panic",
    "liquidation", "bankrupt", "scam", "ponzi", "rug", "plummet",
    "vulnerability", "breach", "suspend",
}

# Tickers that are common English words → require prefix/suffix matching
_AMBIGUOUS_TICKERS = {
    "ADA", "BAT", "LINK", "ONE", "SAND", "NEAR", "ROSE", "ATOM", "FLOW",
    "RAY", "SUN", "FUN", "WIN", "FOR", "HOT", "KEY", "DOCK", "REEF",
    "MASK", "SPELL", "PEOPLE", "CHESS", "LOOM",
}


def _score_text(text: str) -> float:
    """Return sentiment in [-1, 1]. 0 = neutral."""
    words = set(re.sub(r"[^\w\s]", " ", text.lower()).split())
    pos = sum(1 for w in _POSITIVE if w in words)
    neg = sum(1 for w in _NEGATIVE if w in words)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


def _matches_symbol(coin: str, categories: str, title: str) -> bool:
    """Check if news article matches a coin ticker, avoiding false positives."""
    coin_upper = coin.upper()

    # For ambiguous tickers, require crypto-specific context
    if coin_upper in _AMBIGUOUS_TICKERS:
        # Must appear in CryptoCompare categories (reliable) or as "$TICKER" / "TICKER/"
        if coin_upper in categories:
            return True
        # Match patterns like "$ADA", "ADA/", or "ADA " preceded by a non-alpha
        pattern = rf'(?:^|[^a-zA-Z])(?:\$)?{re.escape(coin_upper)}(?:/|[^a-zA-Z]|$)'
        if re.search(pattern, title.upper()):
            return True
        return False

    # Normal tickers: check categories first (most reliable), then title
    if coin_upper in categories:
        return True
    if coin_upper in title.upper():
        return True
    return False


# In-memory sentiment history for time-series tracking
_sentiment_history: dict[str, list[tuple[float, float]]] = {}  # symbol -> [(timestamp, sentiment)]
_MAX_HISTORY = 100  # per symbol


async def fetch_news_sentiment(symbols: list[str]) -> dict[str, Any]:
    """Return per-symbol news sentiment dict with time-series data.

    Result shape:
    {
      "ETH/USDT": {
        "article_count": 3,
        "avg_sentiment": 0.25,
        "min_sentiment": -0.5,
        "max_sentiment": 0.8,
        "sentiment_trend": 0.1,  # current vs 7-cycle moving average
        "headlines": ["...", "..."]
      }, ...
    }
    """
    import time

    headers: dict[str, str] = {}
    if settings.CRYPTOCOMPARE_API_KEY:
        headers["authorization"] = f"Apikey {settings.CRYPTOCOMPARE_API_KEY}"

    # Extract base coins e.g. "ETH/USDT" → "ETH"
    coin_map: dict[str, str] = {sym.split("/")[0]: sym for sym in symbols}
    result: dict[str, Any] = {
        sym: {
            "article_count": 0, "avg_sentiment": 0.0,
            "min_sentiment": 0.0, "max_sentiment": 0.0,
            "sentiment_trend": 0.0, "headlines": [],
        }
        for sym in symbols
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Fetch more articles for better coverage across symbols
            resp = await client.get(
                _BASE_URL,
                params={"lang": "EN", "sortOrder": "latest", "limit": 100},
                headers=headers,
            )
            resp.raise_for_status()
            articles: list[dict] = resp.json().get("Data", [])
    except Exception as exc:
        logger.warning("News fetch failed: %s", exc)
        return result

    # Per-symbol sentiment collectors for min/max/std tracking
    sym_sentiments: dict[str, list[float]] = {sym: [] for sym in symbols}

    # Try GPU semantic sentiment (batch all article texts at once)
    gpu_scores: dict[int, float] = {}
    from app.services import gpu_client
    if gpu_client.is_enabled() and articles:
        texts = [f"{a.get('title', '')} {a.get('body', '')}" for a in articles]
        gpu_result = await gpu_client.sentiment(texts)
        if gpu_result and gpu_result.get("scores"):
            for i, score in enumerate(gpu_result["scores"]):
                gpu_scores[i] = score
            logger.info("GPU sentiment scored %d articles (model=%s)",
                        len(gpu_scores), gpu_result.get("model"))

    for idx, article in enumerate(articles):
        title = article.get("title", "")
        body = article.get("body", "")
        categories = article.get("categories", "").upper()
        # Use GPU semantic score if available, else keyword fallback
        if idx in gpu_scores:
            sentiment = gpu_scores[idx]
        else:
            combined = f"{title} {body}"
            sentiment = _score_text(combined)

        # Match article to symbols with improved ticker matching
        for coin, sym in coin_map.items():
            if _matches_symbol(coin, categories, title):
                entry = result[sym]
                entry["article_count"] += 1
                sym_sentiments[sym].append(sentiment)
                if len(entry["headlines"]) < 3:
                    entry["headlines"].append(title[:120])

    # Compute aggregates
    now = time.time()
    for sym in symbols:
        sents = sym_sentiments[sym]
        if sents:
            result[sym]["avg_sentiment"] = round(sum(sents) / len(sents), 3)
            result[sym]["min_sentiment"] = round(min(sents), 3)
            result[sym]["max_sentiment"] = round(max(sents), 3)

            # Update time-series history
            if sym not in _sentiment_history:
                _sentiment_history[sym] = []
            _sentiment_history[sym].append((now, result[sym]["avg_sentiment"]))
            if len(_sentiment_history[sym]) > _MAX_HISTORY:
                _sentiment_history[sym] = _sentiment_history[sym][-_MAX_HISTORY:]

        # Compute sentiment trend (current vs recent moving average)
        history = _sentiment_history.get(sym, [])
        if len(history) >= 2:
            recent = [s for _, s in history[-7:]]
            ma = sum(recent) / len(recent)
            current = history[-1][1]
            result[sym]["sentiment_trend"] = round(current - ma, 3)

    return result
