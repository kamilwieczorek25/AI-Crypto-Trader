"""Market data service — ccxt Binance: OHLCV, order book, tickers, full symbol universe."""

import asyncio
import logging
from typing import Any

import ccxt.async_support as ccxt

from app.config import settings

logger = logging.getLogger(__name__)

TIMEFRAMES = ["15m", "1h", "4h", "1d"]
OHLCV_LIMIT = 100  # candles per timeframe


class MarketDataService:
    def __init__(self) -> None:
        self._exchange: ccxt.Exchange | None = None
        # OHLCV cache: "symbol:timeframe" -> list of candles (kept at OHLCV_LIMIT length)
        self._ohlcv_cache: dict[str, list[list[float]]] = {}

    async def _get_exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            self._exchange = ccxt.binance(
                {
                    "apiKey": settings.BINANCE_API_KEY or None,
                    "secret": settings.BINANCE_SECRET or None,
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                }
            )
        return self._exchange

    async def close(self) -> None:
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None

    # ------------------------------------------------------------------
    # USDT altcoin universe — filtered by volume, sorted by 24h volume
    # n=0 means return all pairs that pass the volume filter
    # ------------------------------------------------------------------
    async def get_top_symbols(self, n: int | None = None) -> list[str]:
        if n is None:
            n = settings.TOP_N_SYMBOLS
        exchange = await self._get_exchange()
        tickers = await exchange.fetch_tickers()
        usdt_pairs = [
            (sym, t)
            for sym, t in tickers.items()
            if sym.endswith("/USDT")
            and not sym.startswith("BTC")
            and (t.get("quoteVolume") or 0) >= settings.MIN_VOLUME_USDT
        ]
        usdt_pairs.sort(key=lambda x: x[1].get("quoteVolume") or 0, reverse=True)
        symbols = [sym for sym, _ in (usdt_pairs[:n] if n > 0 else usdt_pairs)]
        logger.info(
            "Symbol universe: %d pairs (min_vol=$%s, cap=%s)",
            len(symbols),
            f"{settings.MIN_VOLUME_USDT:,.0f}",
            n if n > 0 else "none",
        )
        return symbols

    # ------------------------------------------------------------------
    # Concurrent multi-symbol OHLCV fetch
    # ------------------------------------------------------------------
    async def get_multi_symbol_ohlcv(
        self, symbols: list[str]
    ) -> dict[str, dict[str, list[list[float]]]]:
        """Fetch all timeframes for all symbols concurrently (rate-limit safe)."""
        sem = asyncio.Semaphore(settings.FETCH_CONCURRENCY)

        async def _fetch_one(sym: str) -> tuple[str, dict[str, list[list[float]]]]:
            async with sem:
                return sym, await self.get_multi_timeframe_ohlcv(sym)

        results = await asyncio.gather(*[_fetch_one(s) for s in symbols], return_exceptions=True)
        out: dict[str, dict[str, list[list[float]]]] = {}
        for r in results:
            if isinstance(r, Exception):
                logger.warning("OHLCV batch error: %s", r)
            else:
                sym, data = r
                out[sym] = data
        return out

    # ------------------------------------------------------------------
    # Concurrent multi-symbol orderbook fetch
    # ------------------------------------------------------------------
    async def get_multi_symbol_orderbooks(
        self, symbols: list[str]
    ) -> dict[str, dict[str, Any]]:
        sem = asyncio.Semaphore(settings.FETCH_CONCURRENCY)

        async def _fetch_one(sym: str) -> tuple[str, dict[str, Any]]:
            async with sem:
                return sym, await self.get_orderbook(sym)

        results = await asyncio.gather(*[_fetch_one(s) for s in symbols], return_exceptions=True)
        out: dict[str, dict[str, Any]] = {}
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Orderbook batch error: %s", r)
            else:
                sym, data = r
                out[sym] = data
        return out

    # ------------------------------------------------------------------
    # Multi-timeframe OHLCV
    # ------------------------------------------------------------------
    async def get_ohlcv(
        self, symbol: str, timeframe: str, limit: int = OHLCV_LIMIT
    ) -> list[list[float]]:
        """Return list of [timestamp, open, high, low, close, volume].

        On first call fetches `limit` candles in full.
        On subsequent calls only fetches candles newer than the cached tip
        (incremental update — saves ~95% of API calls after warmup).
        """
        cache_key = f"{symbol}:{timeframe}"
        exchange = await self._get_exchange()
        cached = self._ohlcv_cache.get(cache_key)

        try:
            if cached:
                # Only fetch candles since last cached timestamp
                since_ms = int(cached[-1][0]) + 1
                new_candles = await exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ms, limit=10
                )
                if new_candles:
                    # Drop overlap — use strict > to avoid duplicating boundary candle
                    merged = cached[:-1] + [
                        c for c in new_candles if c[0] > cached[-2][0]
                    ] if len(cached) > 1 else new_candles
                    self._ohlcv_cache[cache_key] = merged[-limit:]
            else:
                # Full fetch on first call
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if ohlcv:
                    self._ohlcv_cache[cache_key] = ohlcv
        except Exception as exc:
            logger.warning("OHLCV fetch failed %s %s: %s", symbol, timeframe, exc)

        return self._ohlcv_cache.get(cache_key, [])

    async def get_multi_timeframe_ohlcv(
        self, symbol: str
    ) -> dict[str, list[list[float]]]:
        """Fetch all 4 timeframes for a symbol."""
        result: dict[str, list[list[float]]] = {}
        for tf in TIMEFRAMES:
            result[tf] = await self.get_ohlcv(symbol, tf)
        return result

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------
    async def get_orderbook(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        exchange = await self._get_exchange()
        try:
            ob = await exchange.fetch_order_book(symbol, limit)
            bids = ob.get("bids", [])  # [[price, qty], ...]
            asks = ob.get("asks", [])

            if not bids or not asks:
                return {"spread_pct": 0, "bid_wall": 0, "ask_wall": 0, "pressure_ratio": 1.0}

            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread_pct = (best_ask - best_bid) / best_bid * 100

            # Largest single level within top-10 as "wall"
            bid_wall = max(qty for _, qty in bids[:10]) if bids else 0
            ask_wall = max(qty for _, qty in asks[:10]) if asks else 0

            total_bid_vol = sum(qty for _, qty in bids)
            total_ask_vol = sum(qty for _, qty in asks)
            pressure_ratio = total_bid_vol / total_ask_vol if total_ask_vol else 1.0

            return {
                "spread_pct": round(spread_pct, 4),
                "bid_wall": round(bid_wall, 4),
                "ask_wall": round(ask_wall, 4),
                "pressure_ratio": round(pressure_ratio, 4),
            }
        except Exception as exc:
            logger.warning("Orderbook fetch failed %s: %s", symbol, exc)
            return {"spread_pct": 0, "bid_wall": 0, "ask_wall": 0, "pressure_ratio": 1.0}

    # ------------------------------------------------------------------
    # Current price
    # ------------------------------------------------------------------
    async def get_price(self, symbol: str) -> float:
        exchange = await self._get_exchange()
        try:
            ticker = await exchange.fetch_ticker(symbol)
            return float(ticker.get("last") or ticker.get("close") or 0)
        except Exception as exc:
            logger.warning("Price fetch failed %s: %s", symbol, exc)
            return 0.0

    async def get_prices(self, symbols: list[str]) -> dict[str, float]:
        """Fetch latest prices concurrently."""
        import asyncio
        sem = asyncio.Semaphore(settings.FETCH_CONCURRENCY)

        async def _fetch(sym: str) -> tuple[str, float]:
            async with sem:
                return sym, await self.get_price(sym)

        results = await asyncio.gather(
            *[_fetch(s) for s in symbols], return_exceptions=True
        )
        prices: dict[str, float] = {}
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Price fetch error: %s", r)
            else:
                sym, p = r
                if p > 0:
                    prices[sym] = p
        return prices


market_data_service = MarketDataService()
