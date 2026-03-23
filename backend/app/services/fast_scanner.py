"""Fast local altcoin scanner — runs every 60s, zero API cost (Binance only).

Scans a wide universe of USDT pairs for unusual momentum, volume spikes,
and RSI extremes.  Maintains a "hot list" that the main 5-min cycle injects
into its quant-scoring pipeline so more altcoins get considered.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class HotCandidate:
    """A symbol flagged by the fast scanner as interesting."""
    symbol: str
    score: float  # 0-100 composite
    reasons: list[str] = field(default_factory=list)
    price_change_5m: float = 0.0
    price_change_15m: float = 0.0
    price_change_1h: float = 0.0
    volume_ratio: float = 1.0  # current vs 20-period avg
    rsi_14: float = 50.0
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FastScanner:
    """Background scanner that checks a wide altcoin universe every ~60s."""

    def __init__(self) -> None:
        self._hot_list: list[HotCandidate] = []
        self._running = False
        self._task: asyncio.Task | None = None
        # Cache of previous ticker snapshots for momentum calculation
        self._prev_tickers: dict[str, float] = {}  # symbol -> price
        self._prev_ts: datetime | None = None

    @property
    def hot_symbols(self) -> list[str]:
        """Return current hot-list symbols (for injection into main cycle)."""
        return [c.symbol for c in self._hot_list]

    @property
    def hot_candidates(self) -> list[HotCandidate]:
        """Return full hot-list with details."""
        return list(self._hot_list)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="fast_scanner")
        logger.info(
            "Fast scanner started (interval=%ds, min_vol=$%s, hot_size=%d)",
            settings.SCANNER_INTERVAL_SECONDS,
            f"{settings.SCANNER_MIN_VOLUME_USDT:,.0f}",
            settings.SCANNER_HOT_LIST_SIZE,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Fast scanner stopped")

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._scan()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Fast scanner error (non-fatal): %s", exc)
            await asyncio.sleep(settings.SCANNER_INTERVAL_SECONDS)

    async def _scan(self) -> None:
        from app.services.market_data import market_data_service
        from app.services.whale_detector import whale_detector

        exchange = await market_data_service._get_exchange()
        tickers = await exchange.fetch_tickers()
        now = datetime.now(timezone.utc)

        # Get whale activity map for score boosting
        whale_data = whale_detector.get_all_whale_data()

        # Build current price map
        current_prices: dict[str, float] = {}
        candidates: list[HotCandidate] = []

        for sym, t in tickers.items():
            if not sym.endswith(f"/{settings.QUOTE_CURRENCY}") or sym.startswith("BTC"):
                continue
            vol = t.get("quoteVolume") or 0
            if vol < settings.SCANNER_MIN_VOLUME_USDT:
                continue

            price = t.get("last") or t.get("close") or 0
            if price <= 0:
                continue
            current_prices[sym] = price

            pct_24h = t.get("percentage") or 0
            pct_1h = t.get("change") or 0  # not always available

            # -- Compute signals --
            score = 0.0
            reasons: list[str] = []

            # 1. Short-term momentum from our own ticker diff
            pct_short = 0.0
            if self._prev_tickers and sym in self._prev_tickers:
                prev_p = self._prev_tickers[sym]
                if prev_p > 0:
                    pct_short = (price - prev_p) / prev_p * 100

            # Strong short-term move (since last scan ~60s ago)
            if abs(pct_short) >= 1.0:
                score += min(abs(pct_short) * 10, 30)
                direction = "up" if pct_short > 0 else "down"
                reasons.append(f"short_move_{direction}_{abs(pct_short):.1f}%")

            # 2. 24h momentum — big movers
            if abs(pct_24h) >= 5:
                score += min(abs(pct_24h) * 1.5, 25)
                reasons.append(f"24h_{pct_24h:+.1f}%")

            # 3. Volume analysis — compare to typical (use bid/ask volume proxy)
            avg_vol = t.get("baseVolume") or 0
            bid_vol = t.get("bidVolume") or 0
            ask_vol = t.get("askVolume") or 0
            vol_ratio = 1.0
            # High 24h volume relative to the scanner floor = interesting
            if vol > 0:
                vol_ratio = vol / max(settings.SCANNER_MIN_VOLUME_USDT, 1)
                if vol_ratio > 3.0:
                    score += min(vol_ratio * 3, 20)
                    reasons.append(f"high_vol_{vol_ratio:.1f}x")

            # 4. Bid/ask imbalance (if available) — proxy for order flow
            if bid_vol > 0 and ask_vol > 0:
                imbalance = bid_vol / (bid_vol + ask_vol)
                if imbalance > 0.65:
                    score += 10
                    reasons.append(f"bid_pressure_{imbalance:.0%}")
                elif imbalance < 0.35:
                    score += 5
                    reasons.append(f"ask_pressure_{imbalance:.0%}")

            # 5. 24h high/low proximity — near breakout
            high24 = t.get("high") or 0
            low24 = t.get("low") or 0
            if high24 > low24 > 0:
                range_pct = (high24 - low24) / low24 * 100
                near_high_pct = (high24 - price) / high24 * 100 if high24 > 0 else 99
                near_low_pct = (price - low24) / low24 * 100 if low24 > 0 else 99

                if near_high_pct < 1.0 and range_pct > 3:
                    score += 15
                    reasons.append(f"near_24h_high({near_high_pct:.1f}%)")
                elif near_low_pct < 1.0 and range_pct > 3:
                    score += 10
                    reasons.append(f"near_24h_low({near_low_pct:.1f}%)")

            # 6. Whale activity boost — large trades detected via WebSocket
            wd = whale_data.get(sym)
            if wd:
                whale_vol = wd.get("whale_total_volume", 0)
                net_flow = wd.get("whale_net_flow", 0)
                if whale_vol > 0:
                    # Any whale activity = interesting
                    score += 15
                    flow_dir = "buy" if net_flow > 0 else "sell"
                    reasons.append(f"whale_{flow_dir}_${whale_vol:,.0f}")
                    # Strong directional whale flow = extra boost
                    if abs(net_flow) > whale_vol * 0.3:
                        score += 10
                        reasons.append(f"whale_bias_{flow_dir}")

            # Only keep if score is meaningful
            if score >= settings.SCANNER_MIN_SCORE:
                candidates.append(HotCandidate(
                    symbol=sym,
                    score=min(score, 100),
                    reasons=reasons,
                    price_change_5m=pct_short,
                    price_change_1h=pct_1h,
                    volume_ratio=vol_ratio,
                    detected_at=now,
                ))

        # Update ticker cache for next momentum diff
        self._prev_tickers = current_prices
        self._prev_ts = now

        # Sort by score, keep top N
        candidates.sort(key=lambda c: c.score, reverse=True)
        self._hot_list = candidates[:settings.SCANNER_HOT_LIST_SIZE]

        if self._hot_list:
            top3 = " | ".join(
                f"{c.symbol} score={c.score:.0f} [{','.join(c.reasons[:2])}]"
                for c in self._hot_list[:3]
            )
            logger.info(
                "Fast scanner: %d scanned, %d hot — top: %s",
                len(current_prices), len(self._hot_list), top3,
            )
        else:
            logger.debug("Fast scanner: %d scanned, 0 hot", len(current_prices))


# Module-level singleton
fast_scanner = FastScanner()
