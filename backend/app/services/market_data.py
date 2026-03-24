"""Market data service — ccxt Binance: OHLCV, order book, tickers, full symbol universe."""

import asyncio
import logging
from datetime import datetime, timezone
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
        # Known symbols: tracks all USDT pairs seen so far for new-listing detection
        self._known_symbols: set[str] = set()
        self._first_scan_done: bool = False
        # New listings detected this session (symbol -> detection time)
        self._new_listings: dict[str, datetime] = {}
        # Last gainers from most recent ticker scan
        self._last_gainers: list[dict] = []

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
    # Altcoin universe — filtered by volume, boosted by gainers
    # and new listings. n=0 means return all passing pairs.
    # ------------------------------------------------------------------
    async def get_top_symbols(self, n: int | None = None) -> list[str]:
        if n is None:
            n = settings.TOP_N_SYMBOLS
        exchange = await self._get_exchange()
        tickers = await exchange.fetch_tickers()

        quote = settings.QUOTE_CURRENCY  # e.g. "USDC" or "USDT"
        pair_suffix = f"/{quote}"

        # Relaxed volume floor for gainers — uses its own configurable setting so
        # low-liquidity USDC pairs (e.g. ONT/USDC) are not silently excluded
        gainer_vol_floor = settings.GAINER_MIN_VOLUME_USDT
        # Very relaxed floor for new listings (they start with near-zero history)
        new_listing_vol_floor = settings.GAINER_MIN_VOLUME_USDT * 0.3

        # Collect all current quote-currency pairs for new-listing detection
        all_quote_syms: set[str] = set()
        quote_pairs: list[tuple[str, dict]] = []
        top_gainers: list[tuple[str, dict, float]] = []

        for sym, t in tickers.items():
            if not sym.endswith(pair_suffix) or sym.startswith("BTC"):
                continue
            all_quote_syms.add(sym)
            vol = t.get("quoteVolume") or 0
            pct = t.get("percentage") or 0  # 24h price change %

            # Regular universe: high volume
            if vol >= settings.MIN_VOLUME_USDT:
                quote_pairs.append((sym, t))

            # Top gainers: significant move + minimum liquidity
            if abs(pct) >= settings.GAINER_MIN_PCT and vol >= gainer_vol_floor:
                top_gainers.append((sym, t, pct))

        # ── New listing detection ──
        # First scan: just record the baseline. After that, diff to find new pairs.
        newly_listed: list[tuple[str, dict]] = []
        now = datetime.now(timezone.utc)
        if not self._first_scan_done:
            self._known_symbols = all_quote_syms.copy()
            self._first_scan_done = True
            logger.info("New-listing baseline: %d %s pairs recorded", len(self._known_symbols), quote)
        else:
            brand_new = all_quote_syms - self._known_symbols
            for sym in brand_new:
                t = tickers.get(sym, {})
                vol = t.get("quoteVolume") or 0
                # Only inject if it has some minimum liquidity
                if vol >= new_listing_vol_floor:
                    self._new_listings[sym] = now
                    newly_listed.append((sym, t))
                    logger.info(
                        "🆕 NEW LISTING DETECTED: %s (vol=$%.0f, pct=%+.1f%%)",
                        sym, vol, t.get("percentage", 0),
                    )
            self._known_symbols = all_quote_syms.copy()

        # Expire old new-listing flags (keep for NEW_LISTING_WATCH_HOURS)
        cutoff = now.timestamp() - settings.NEW_LISTING_WATCH_HOURS * 3600
        expired = [s for s, dt in self._new_listings.items() if dt.timestamp() < cutoff]
        for s in expired:
            del self._new_listings[s]

        # Sort main universe by volume
        quote_pairs.sort(key=lambda x: x[1].get("quoteVolume") or 0, reverse=True)
        symbols_set = {sym for sym, _ in quote_pairs}
        symbols = [sym for sym, _ in (quote_pairs[:n] if n > 0 else quote_pairs)]

        # Inject top gainers that aren't already in the universe
        top_gainers.sort(key=lambda x: abs(x[2]), reverse=True)
        injected_gainers: list[str] = []
        for sym, t, pct in top_gainers[:settings.GAINER_INJECT_COUNT]:
            if sym not in symbols_set:
                symbols.append(sym)
                symbols_set.add(sym)
                injected_gainers.append(f"{sym}({pct:+.1f}%)")

        # Inject new listings (all currently tracked, within watch window)
        injected_new: list[str] = []
        for sym in list(self._new_listings.keys()):
            if sym not in symbols_set and sym in all_quote_syms:
                t = tickers.get(sym, {})
                vol = t.get("quoteVolume") or 0
                if vol >= new_listing_vol_floor:
                    symbols.append(sym)
                    symbols_set.add(sym)
                    age_h = (now - self._new_listings[sym]).total_seconds() / 3600
                    injected_new.append(f"{sym}({age_h:.1f}h)")

        # Store last gainers for prompt enrichment
        self._last_gainers = [
            {"symbol": sym, "pct_24h": round(pct, 2),
             "volume": t.get("quoteVolume", 0)}
            for sym, t, pct in top_gainers[:10]
        ]

        if injected_gainers:
            logger.info("Top gainers injected: %s", ", ".join(injected_gainers))
        if injected_new:
            logger.info("New listings injected: %s", ", ".join(injected_new))

        logger.info(
            "Symbol universe: %d pairs (%d base + %d gainers + %d new listings, min_vol=$%s, cap=%s)",
            len(symbols), len(quote_pairs),
            len(injected_gainers), len(injected_new),
            f"{settings.MIN_VOLUME_USDT:,.0f}",
            n if n > 0 else "none",
        )
        return symbols

    @property
    def last_gainers(self) -> list[dict]:
        """Return the top gainers from the last fetch_tickers call."""
        return self._last_gainers

    @property
    def new_listings(self) -> dict[str, datetime]:
        """Return currently tracked new listings (symbol -> first-seen time)."""
        return dict(self._new_listings)

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
                return {"spread_pct": 0, "bid_wall": 0, "ask_wall": 0,
                        "pressure_ratio": 1.0, "depth_imbalance": 0.0,
                        "bid_depth_2pct": 0, "ask_depth_2pct": 0}

            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            spread_pct = (best_ask - best_bid) / best_bid * 100

            # Largest single level within top-10 as "wall"
            bid_wall = max(qty for _, qty in bids[:10]) if bids else 0
            ask_wall = max(qty for _, qty in asks[:10]) if asks else 0

            total_bid_vol = sum(qty for _, qty in bids)
            total_ask_vol = sum(qty for _, qty in asks)
            pressure_ratio = total_bid_vol / total_ask_vol if total_ask_vol else 1.0

            # Depth imbalance within 2% of mid price (USDT-weighted)
            depth_floor = mid_price * 0.98
            depth_ceil = mid_price * 1.02
            bid_depth_2pct = sum(
                p * q for p, q in bids if p >= depth_floor
            )
            ask_depth_2pct = sum(
                p * q for p, q in asks if p <= depth_ceil
            )
            total_depth = bid_depth_2pct + ask_depth_2pct
            # Imbalance: +1 = all bids, -1 = all asks, 0 = balanced
            depth_imbalance = (
                (bid_depth_2pct - ask_depth_2pct) / total_depth
                if total_depth > 0 else 0.0
            )

            return {
                "spread_pct": round(spread_pct, 4),
                "bid_wall": round(bid_wall, 4),
                "ask_wall": round(ask_wall, 4),
                "pressure_ratio": round(pressure_ratio, 4),
                "depth_imbalance": round(depth_imbalance, 4),
                "bid_depth_2pct": round(bid_depth_2pct, 2),
                "ask_depth_2pct": round(ask_depth_2pct, 2),
            }
        except Exception as exc:
            logger.warning("Orderbook fetch failed %s: %s", symbol, exc)
            return {"spread_pct": 0, "bid_wall": 0, "ask_wall": 0,
                    "pressure_ratio": 1.0, "depth_imbalance": 0.0,
                    "bid_depth_2pct": 0, "ask_depth_2pct": 0}

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


    # ------------------------------------------------------------------
    # Exchange account: real balances (requires API key)
    # ------------------------------------------------------------------
    async def fetch_spot_balances(self) -> dict[str, dict[str, float]]:
        """Fetch actual spot balances from Binance.

        Returns {symbol_base: {"free": float, "total": float}}
        e.g. {"BTC": {"free": 0.5, "total": 0.5}, "USDT": {"free": 1200, "total": 1200}}

        Requires valid BINANCE_API_KEY and BINANCE_SECRET.
        """
        exchange = await self._get_exchange()
        try:
            balance = await exchange.fetch_balance()
        except Exception as exc:
            logger.error("Failed to fetch Binance balance: %s", exc)
            return {}

        result: dict[str, dict[str, float]] = {}
        for asset, info in balance.get("total", {}).items():
            total = float(info) if isinstance(info, (int, float, str)) else 0.0
            if total <= 0:
                continue
            free = float(balance.get("free", {}).get(asset, 0))
            result[asset] = {"free": free, "total": total}
        return result

    async def fetch_usdt_balance(self) -> float:
        """Return available USDT balance on Binance spot account."""
        balances = await self.fetch_spot_balances()
        return balances.get("USDT", {}).get("free", 0.0)


market_data_service = MarketDataService()
