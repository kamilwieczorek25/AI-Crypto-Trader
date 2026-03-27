"""Whale trade detector — monitors Binance WebSocket for large trades.

Connects to Binance's aggregated trade stream for all USDT pairs (!aggTrade).
When a single trade exceeds the configurable threshold ($50K+), it flags the
symbol with timestamp and direction, making it available to the fast scanner
and quant scorer.

Uses the public WebSocket endpoint — no API key required.
"""

import asyncio
import json
import logging
import ssl
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiohttp

from app.config import settings

logger = logging.getLogger(__name__)

BINANCE_WS = "wss://stream.binance.com:9443/ws"


@dataclass
class WhaleEvent:
    """A single large trade detected on Binance."""
    symbol: str         # e.g. "OPN/USDT"
    side: str           # "BUY" or "SELL"
    usdt_value: float   # total USDT value of the trade
    price: float
    quantity: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WhaleDetector:
    """Background WebSocket listener for large trades on Binance."""

    def __init__(self) -> None:
        self._running = False
        self._task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None
        self._ws_verify_ssl = settings.WHALE_WS_VERIFY_SSL
        self._ws_ssl_fallback_used = False
        # Recent whale events per symbol (last WHALE_MEMORY_MINUTES minutes)
        self._events: dict[str, list[WhaleEvent]] = defaultdict(list)
        # Tracked symbols — refreshed periodically from the symbol universe
        self._tracked_symbols: set[str] = set()

    @property
    def whale_symbols(self) -> list[str]:
        """Symbols with recent whale activity."""
        self._expire_old_events()
        return list(self._events.keys())

    def get_whale_data(self, symbol: str) -> dict:
        """Get whale activity summary for a symbol (for quant scorer)."""
        self._expire_old_events()
        events = self._events.get(symbol, [])
        if not events:
            return {}

        total_buy_vol = sum(e.usdt_value for e in events if e.side == "BUY")
        total_sell_vol = sum(e.usdt_value for e in events if e.side == "SELL")
        buy_count = sum(1 for e in events if e.side == "BUY")
        sell_count = sum(1 for e in events if e.side == "SELL")

        return {
            "whale_buy_volume": total_buy_vol,
            "whale_sell_volume": total_sell_vol,
            "whale_buy_count": buy_count,
            "whale_sell_count": sell_count,
            "whale_net_flow": total_buy_vol - total_sell_vol,
            "whale_total_volume": total_buy_vol + total_sell_vol,
            "latest_event_age_s": (
                datetime.now(timezone.utc) - events[-1].timestamp
            ).total_seconds(),
        }

    def get_all_whale_data(self) -> dict[str, dict]:
        """Get whale data for all active symbols."""
        self._expire_old_events()
        return {sym: self.get_whale_data(sym) for sym in self._events}

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="whale_detector")
        logger.info(
            "Whale detector started (threshold=$%s, memory=%dm)",
            f"{settings.WHALE_MIN_USDT:,.0f}",
            settings.WHALE_MEMORY_MINUTES,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
        logger.info("Whale detector stopped")

    def _expire_old_events(self) -> None:
        """Remove whale events older than WHALE_MEMORY_MINUTES."""
        cutoff = datetime.now(timezone.utc).timestamp() - (
            settings.WHALE_MEMORY_MINUTES * 60
        )
        expired_syms = []
        for sym, events in self._events.items():
            self._events[sym] = [
                e for e in events if e.timestamp.timestamp() > cutoff
            ]
            if not self._events[sym]:
                expired_syms.append(sym)
        for sym in expired_syms:
            del self._events[sym]

    async def _run(self) -> None:
        """Main loop — connects to Binance WS, reconnects on failure."""
        while self._running:
            try:
                await self._listen()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._maybe_enable_insecure_fallback(exc):
                    await asyncio.sleep(1)
                    continue
                logger.warning("Whale WS error: %s — reconnecting in 5s", exc)
                await asyncio.sleep(5)

    def _maybe_enable_insecure_fallback(self, exc: Exception) -> bool:
        """Switch to insecure TLS on cert errors when explicitly allowed."""
        cert_error = isinstance(exc, ssl.SSLCertVerificationError)

        if not cert_error and isinstance(exc, aiohttp.ClientConnectorCertificateError):
            cert_error = True

        if not cert_error and "certificate verify failed" in str(exc).lower():
            cert_error = True

        if not cert_error:
            return False

        if not settings.WHALE_WS_INSECURE_FALLBACK:
            logger.warning(
                "Whale WS TLS verification failed. Set WHALE_WS_INSECURE_FALLBACK=true "
                "or fix your CA trust chain."
            )
            return False

        if not self._ws_verify_ssl or self._ws_ssl_fallback_used:
            return False

        self._ws_verify_ssl = False
        self._ws_ssl_fallback_used = True
        logger.warning(
            "Whale WS TLS verification failed; falling back to ssl=False. "
            "For secure operation, install the intercepting CA and keep WHALE_WS_VERIFY_SSL=true."
        )
        return True

    async def _listen(self) -> None:
        """Subscribe to Binance aggTrade stream for top symbols."""
        if not self._session:
            self._session = aiohttp.ClientSession()

        # Subscribe to top volume symbols' trade streams
        # We use individual streams combined — max 200 streams per connection
        from app.services.market_data import market_data_service

        symbols = await market_data_service.get_top_symbols()
        # Convert to Binance WS format: "btcusdt@aggTrade"
        streams = []
        quote_lower = settings.QUOTE_CURRENCY.lower()  # e.g. "usdc"
        for sym in symbols[:100]:  # limit to top 100 to stay within WS limits
            pair_suffix = f"/{settings.QUOTE_CURRENCY}"
            base = sym.replace(pair_suffix, "").lower() + quote_lower
            streams.append(f"{base}@aggTrade")
            self._tracked_symbols.add(sym)

        if not streams:
            logger.warning("Whale detector: no symbols to track, retrying in 30s")
            await asyncio.sleep(30)
            return

        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        logger.info("Whale detector: connecting to %d trade streams", len(streams))

        async with self._session.ws_connect(
            url,
            heartbeat=20,
            ssl=self._ws_verify_ssl,
        ) as ws:
            logger.info("Whale detector: WebSocket connected")
            async for msg in ws:
                if not self._running:
                    break
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        trade = data.get("data", data)
                        await self._process_trade(trade)
                    except (json.JSONDecodeError, KeyError):
                        continue
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break

    async def _process_trade(self, trade: dict) -> None:
        """Check if a single trade qualifies as a whale trade."""
        # aggTrade fields: s=symbol, p=price, q=quantity, m=isBuyerMaker
        raw_sym = trade.get("s", "")  # e.g. "OPNUSDT"
        price = float(trade.get("p", 0))
        qty = float(trade.get("q", 0))
        is_buyer_maker = trade.get("m", False)

        usdt_value = price * qty
        if usdt_value < settings.WHALE_MIN_USDT:
            return

        # Convert "OPNUSDC" -> "OPN/USDC" (or USDT)
        quote = settings.QUOTE_CURRENCY  # e.g. "USDC"
        if raw_sym.endswith(quote):
            symbol = raw_sym[:-len(quote)] + f"/{quote}"
        elif raw_sym.endswith("USDT"):
            symbol = raw_sym[:-4] + "/USDT"
        else:
            return

        # Buyer maker = seller is taker = SELL pressure
        # Not buyer maker = buyer is taker = BUY pressure
        side = "SELL" if is_buyer_maker else "BUY"

        event = WhaleEvent(
            symbol=symbol,
            side=side,
            usdt_value=usdt_value,
            price=price,
            quantity=qty,
        )

        self._events[symbol].append(event)

        logger.info(
            "🐋 WHALE %s %s: $%s (%.4f @ $%.4f)",
            side, symbol, f"{usdt_value:,.0f}", qty, price,
        )


# Module-level singleton
whale_detector = WhaleDetector()
