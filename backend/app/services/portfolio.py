"""Portfolio service — balance, positions, P&L tracking."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.position import Position
from app.models.snapshot import PortfolioSnapshot
from app.schemas.portfolio import PortfolioState, PositionOut

logger = logging.getLogger(__name__)


class PortfolioService:
    """Tracks cash balance and open positions in-memory + DB."""

    def __init__(self) -> None:
        self._cash_usdt: float = settings.DEMO_INITIAL_BALANCE
        self._initial_value: float = settings.DEMO_INITIAL_BALANCE
        # in-memory position cache {symbol: Position}
        self._positions: dict[str, Position] = {}

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    async def load_from_db(self, db: AsyncSession) -> None:
        """Restore in-memory state from DB at startup."""
        from app.models.bot_state import BotState

        # Restore open positions
        rows = await db.execute(select(Position))
        for pos in rows.scalars().all():
            self._positions[pos.symbol] = pos

        # 1st priority: bot_state.cash_usdt (written immediately after every trade — crash-safe)
        cash_state = await db.get(BotState, "cash_usdt")
        if cash_state is not None:
            self._cash_usdt = float(cash_state.value)
            # In real mode, initial value is set by sync_from_exchange later
            if settings.is_demo:
                self._initial_value = settings.DEMO_INITIAL_BALANCE
            else:
                self._initial_value = self._cash_usdt + self.positions_value
            logger.info(
                "Portfolio loaded: %d positions, cash=$%.2f (from bot_state — crash-safe)",
                len(self._positions),
                self._cash_usdt,
            )
            return

        # 2nd priority: latest portfolio snapshot
        snap_row = await db.execute(
            select(PortfolioSnapshot)
            .order_by(PortfolioSnapshot.created_at.desc())
            .limit(1)
        )
        snap = snap_row.scalar_one_or_none()
        if snap is not None:
            self._cash_usdt = snap.cash_usdt
            if settings.is_demo:
                self._initial_value = settings.DEMO_INITIAL_BALANCE
            else:
                self._initial_value = self._cash_usdt + self.positions_value
            logger.info(
                "Portfolio loaded: %d positions, cash=$%.2f (from snapshot %s)",
                len(self._positions),
                self._cash_usdt,
                snap.created_at.isoformat(),
            )
        else:
            logger.info(
                "Portfolio loaded: %d positions, cash=$%.2f (no snapshot — using initial balance)",
                len(self._positions),
                self._cash_usdt,
            )

    # ------------------------------------------------------------------ #
    # Exchange sync (real mode)
    # ------------------------------------------------------------------ #
    async def sync_from_exchange(self, db: AsyncSession) -> dict[str, Any]:
        """Read actual Binance spot balances and reconcile with internal state.

        - Imports USDT balance as cash
        - Imports non-USDT holdings as external positions
        - Preserves bot-opened positions (source='bot')
        - Returns summary of what changed

        Only call in real mode with valid API keys.
        """
        from app.services.market_data import market_data_service

        changes: dict[str, Any] = {"imported": [], "updated": [], "usdt_synced": False}

        # 1. Fetch real balances
        try:
            balances = await market_data_service.fetch_spot_balances()
        except Exception as exc:
            logger.error("Exchange sync failed: %s", exc)
            return changes

        if not balances:
            logger.warning("Exchange sync: no balances returned (check API key permissions)")
            return changes

        # 2. Sync cash balance (USDT + USDC + other stablecoins → treated as cash)
        _STABLECOINS = ("USDT", "USDC", "BUSD", "TUSD", "FDUSD", "DAI")
        real_cash = 0.0
        for stable in _STABLECOINS:
            info = balances.pop(stable, {})
            real_cash += info.get("free", 0.0)

        if real_cash > 0:
            old_cash = self._cash_usdt
            self._cash_usdt = real_cash
            self._initial_value = self._cash_usdt + self.positions_value
            changes["usdt_synced"] = True
            logger.info(
                "Exchange sync: cash $%.2f -> $%.2f (from Binance stablecoins)",
                old_cash, real_cash,
            )
            await self._persist_cash()

        # 3. Sync spot holdings as positions
        for asset, info in balances.items():
            total = info.get("total", 0.0)
            if total <= 0:
                continue

            # Skip stablecoins (already consumed above), fiat currencies, and non-tradeable
            _SKIP = ("USD", "LDUSDT", "LDUSDC",
                     "EUR", "GBP", "PLN", "TRY", "BRL", "ARS", "UAH", "RUB",
                     "NGN", "AUD", "JPY", "KRW", "INR", "ZAR", "CAD", "CHF",
                     "CZK", "SEK", "NOK", "DKK", "HUF", "RON", "BGN", "HRK")
            if asset in _STABLECOINS or asset in _SKIP:
                continue

            # Try quote currency pair first, then fallback
            symbol = f"{asset}/{settings.QUOTE_CURRENCY}"
            try:
                price = await market_data_service.get_price(symbol)
            except Exception:
                price = 0.0

            if price <= 0:
                # Fallback: try the other stablecoin pair
                fallback_quote = "USDT" if settings.QUOTE_CURRENCY == "USDC" else "USDC"
                symbol_fallback = f"{asset}/{fallback_quote}"
                try:
                    price = await market_data_service.get_price(symbol_fallback)
                    if price > 0:
                        symbol = symbol_fallback
                except Exception:
                    price = 0.0

            if price <= 0:
                continue

            value_usdt = total * price
            if value_usdt < settings.SYNC_MIN_VALUE_USDT:
                continue  # skip dust

            # Deduplicate: if this asset already exists under a different quote pair, merge
            alt_quote = "USDT" if settings.QUOTE_CURRENCY == "USDC" else "USDC"
            alt_symbol = f"{asset}/{alt_quote}"
            if symbol != alt_symbol and alt_symbol in self._positions:
                old_pos = self._positions.pop(alt_symbol)
                logger.info(
                    "Exchange sync: migrating %s -> %s (quote currency change)",
                    alt_symbol, symbol,
                )
                # Delete old DB record
                from app.models.position import Position as PosModel
                from sqlalchemy import delete
                await db.execute(
                    delete(PosModel).where(PosModel.symbol == alt_symbol)
                )
                changes["updated"].append({
                    "symbol": alt_symbol, "reason": "migrated_to", "new_symbol": symbol,
                })

            existing = self._positions.get(symbol)
            if existing and existing.source == "bot":
                # Bot-managed position: don't overwrite, but verify quantity matches
                if abs(existing.quantity - total) / max(total, 0.0001) > 0.01:
                    logger.warning(
                        "Exchange sync: %s quantity mismatch — bot=%.6f, exchange=%.6f",
                        symbol, existing.quantity, total,
                    )
                    changes["updated"].append({
                        "symbol": symbol, "reason": "quantity_mismatch",
                        "bot_qty": existing.quantity, "exchange_qty": total,
                    })
                continue

            if existing and existing.source == "external":
                # Update external position with latest exchange data
                existing.quantity = total
                existing.current_price = price
                existing.avg_entry_price = price  # best estimate for externals
                existing.updated_at = datetime.now(timezone.utc)
                db.add(existing)
                changes["updated"].append({"symbol": symbol, "quantity": total, "value_usdt": value_usdt})
            else:
                # New external holding — import it
                pos = Position(
                    symbol=symbol,
                    quantity=total,
                    avg_entry_price=price,  # we don't know real entry, use current
                    current_price=price,
                    stop_loss_price=0.0,    # no SL for external positions
                    take_profit_price=0.0,  # no TP for external positions
                    highest_price=price,
                    trailing_stop_pct=0.0,
                    source="external",
                )
                db.add(pos)
                await db.flush()
                self._positions[symbol] = pos
                changes["imported"].append({"symbol": symbol, "quantity": total, "value_usdt": value_usdt})
                logger.info(
                    "Exchange sync: imported %s — %.6f units ($%.2f)",
                    symbol, total, value_usdt,
                )

        await db.commit()

        # Update initial value to reflect full portfolio
        self._initial_value = self.total_value

        n_imp = len(changes["imported"])
        n_upd = len(changes["updated"])
        logger.info(
            "Exchange sync complete: USDT=$%.2f, %d imported, %d updated, %d total positions",
            self._cash_usdt, n_imp, n_upd, len(self._positions),
        )
        return changes

    # ------------------------------------------------------------------ #
    # Balance helpers
    # ------------------------------------------------------------------ #
    @property
    def cash_usdt(self) -> float:
        return self._cash_usdt

    def set_cash(self, amount: float) -> None:
        self._cash_usdt = max(0.0, amount)

    @property
    def positions_value(self) -> float:
        return sum(p.quantity * p.current_price for p in self._positions.values())

    @property
    def total_value(self) -> float:
        return self._cash_usdt + self.positions_value

    @property
    def total_pnl_usdt(self) -> float:
        return self.total_value - self._initial_value

    @property
    def total_pnl_pct(self) -> float:
        if self._initial_value == 0:
            return 0.0
        return self.total_pnl_usdt / self._initial_value * 100

    @property
    def exposure_pct(self) -> float:
        """Total altcoin exposure as % of portfolio."""
        if self.total_value == 0:
            return 0.0
        return self.positions_value / self.total_value * 100

    # ------------------------------------------------------------------ #
    # Position management
    # ------------------------------------------------------------------ #
    def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def all_positions(self) -> list[Position]:
        return list(self._positions.values())

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current_price on all tracked positions (skip zero/negative).

        Also updates trailing stop: if price makes a new high, ratchet SL up.
        """
        for sym, pos in self._positions.items():
            if sym in prices and prices[sym] > 0:
                pos.current_price = prices[sym]
                # Trailing stop-loss: if new high, ratchet SL upward
                if prices[sym] > pos.highest_price and pos.highest_price > 0:
                    pos.highest_price = prices[sym]
                    if pos.trailing_stop_pct > 0:
                        new_sl = pos.highest_price * (1 - pos.trailing_stop_pct / 100)
                        if new_sl > pos.stop_loss_price:
                            pos.stop_loss_price = new_sl

    def check_sl_tp_triggers(self) -> list[tuple[str, str, float]]:
        """Return (symbol, reason, price) for positions that hit SL or TP.

        Trailing TP: when price first hits TP, don't sell — activate trailing
        mode and track the peak. Sell only when price pulls back from peak
        by TRAILING_TP_PULLBACK_PCT, or drops back below original TP.

        reason is 'stop_loss' or 'take_profit'.
        """
        from app.config import settings

        pullback_pct = settings.TRAILING_TP_PULLBACK_PCT
        floor_enabled = settings.TRAILING_TP_FLOOR
        profit_lock_activate = settings.PROFIT_LOCK_ACTIVATE_PCT
        profit_lock_floor_min = settings.PROFIT_LOCK_FLOOR_PCT
        profit_lock_keep = settings.PROFIT_LOCK_KEEP_PCT / 100.0  # 50 → 0.50
        triggers: list[tuple[str, str, float]] = []

        for sym, pos in list(self._positions.items()):
            price = pos.current_price
            if price <= 0:
                continue

            # Stop-loss always fires immediately
            if pos.stop_loss_price > 0 and price <= pos.stop_loss_price:
                triggers.append((sym, "stop_loss", price))
                continue

            # ── Profit lock: protect unrealised gains before TP ──────────
            # Sliding floor = max(FLOOR_MIN, peak_pnl × KEEP_PCT).
            # E.g. peaked +10%, keep=50% → floor = +5%.
            # Skip if trailing TP is already active — that system takes over.
            if profit_lock_activate > 0 and pos.avg_entry_price > 0 and not pos.tp_activated:
                peak_pnl_pct = (
                    (pos.highest_price - pos.avg_entry_price)
                    / pos.avg_entry_price * 100
                ) if pos.highest_price > 0 else 0.0
                if peak_pnl_pct >= profit_lock_activate:
                    # Sliding floor: keep a fraction of peak gains, but at least FLOOR_MIN
                    dynamic_floor = max(profit_lock_floor_min, peak_pnl_pct * profit_lock_keep)
                    current_pnl_pct = (price - pos.avg_entry_price) / pos.avg_entry_price * 100
                    if current_pnl_pct <= dynamic_floor:
                        logger.info(
                            "Profit lock: %s peaked at +%.1f%%, now +%.1f%% "
                            "(floor +%.1f%%) — selling to protect gains at $%.6f",
                            sym, peak_pnl_pct, current_pnl_pct,
                            dynamic_floor, price,
                        )
                        triggers.append((sym, "profit_lock", price))
                        continue

            if pos.take_profit_price <= 0:
                continue

            if not pos.tp_activated:
                # TP not yet hit — check if price just reached it
                if price >= pos.take_profit_price:
                    pos.tp_activated = True
                    pos.tp_peak_price = price
                    logger.info(
                        "Trailing TP activated: %s hit TP $%.6f (now $%.6f) — letting profits run",
                        sym, pos.take_profit_price, price,
                    )
            else:
                # TP already activated — track peak and check pullback
                if price > pos.tp_peak_price:
                    pos.tp_peak_price = price

                # Safety floor: price fell back below original TP → sell now
                if floor_enabled and price < pos.take_profit_price:
                    logger.info(
                        "Trailing TP floor: %s dropped back below TP $%.6f → selling at $%.6f",
                        sym, pos.take_profit_price, price,
                    )
                    triggers.append((sym, "take_profit", price))
                    continue

                # Pullback from peak → sell
                if pos.tp_peak_price > 0:
                    drop_from_peak = (pos.tp_peak_price - price) / pos.tp_peak_price * 100
                    if drop_from_peak >= pullback_pct:
                        logger.info(
                            "Trailing TP triggered: %s pulled back %.1f%% from peak $%.6f → selling at $%.6f",
                            sym, drop_from_peak, pos.tp_peak_price, price,
                        )
                        triggers.append((sym, "take_profit", price))

        return triggers

    async def adopt_position(
        self, db: AsyncSession, symbol: str,
        sl_pct: float | None = None, tp_pct: float | None = None,
    ) -> Position | None:
        """Convert an external position to bot-managed with SL/TP.

        If sl_pct / tp_pct are not provided, uses config defaults.
        Returns the updated position or None if not found / already bot-managed.
        """
        pos = self._positions.get(symbol)
        if pos is None or pos.source == "bot":
            return None

        if sl_pct is None:
            sl_pct = settings.MIN_SL_PCT
        if tp_pct is None:
            tp_pct = sl_pct * settings.MIN_REWARD_RISK_RATIO

        price = pos.current_price if pos.current_price > 0 else pos.avg_entry_price
        pos.source = "bot"
        pos.stop_loss_price = round(price * (1 - sl_pct / 100), 6)
        pos.take_profit_price = round(price * (1 + tp_pct / 100), 6)
        pos.highest_price = max(pos.highest_price, price)
        pos.trailing_stop_pct = sl_pct
        pos.updated_at = datetime.now(timezone.utc)
        db.add(pos)
        await db.commit()
        logger.info(
            "Adopted %s: SL=$%.6f (%.1f%%), TP=$%.6f (%.1f%%)",
            symbol, pos.stop_loss_price, sl_pct, pos.take_profit_price, tp_pct,
        )
        return pos

    async def _persist_cash(self) -> None:
        """Write current cash balance to bot_state table immediately (crash recovery)."""
        from app.database import save_bot_state
        await save_bot_state("cash_usdt", str(round(self._cash_usdt, 8)))

    async def open_position(
        self,
        db: AsyncSession,
        symbol: str,
        quantity: float,
        price: float,
        stop_loss_price: float,
        take_profit_price: float,
        fee_usdt: float = 0.0,
    ) -> Position:
        if quantity <= 0 or price <= 0:
            logger.error("Invalid open_position: qty=%.6f price=%.6f", quantity, price)
            raise ValueError("quantity and price must be positive")

        existing = self._positions.get(symbol)
        if existing:
            # Average into existing position
            total_qty = existing.quantity + quantity
            new_avg = (
                existing.avg_entry_price * existing.quantity + price * quantity
            ) / total_qty
            existing.avg_entry_price = new_avg
            existing.quantity = total_qty
            existing.current_price = price
            # Recalculate SL/TP based on new averaged entry price
            # Preserve the same percentage distances as the new trade's SL/TP
            if price > 0:
                sl_pct = (price - stop_loss_price) / price
                tp_pct = (take_profit_price - price) / price
                existing.stop_loss_price = new_avg * (1 - sl_pct)
                existing.take_profit_price = new_avg * (1 + tp_pct)
                existing.trailing_stop_pct = sl_pct * 100
            existing.highest_price = max(existing.highest_price, price)
            existing.updated_at = datetime.now(timezone.utc)
            db.add(existing)
            await db.commit()
            # Deduct cash for averaging into position (plus fee)
            self._cash_usdt -= quantity * price + fee_usdt
            await self._persist_cash()
            return existing

        pos = Position(
            symbol=symbol,
            quantity=quantity,
            avg_entry_price=price,
            current_price=price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            highest_price=price,
            trailing_stop_pct=(price - stop_loss_price) / price * 100 if price > 0 else 0.0,
        )
        db.add(pos)
        await db.commit()
        await db.refresh(pos)
        self._positions[symbol] = pos
        # Deduct cash and persist immediately for crash recovery
        self._cash_usdt -= quantity * price + fee_usdt
        await self._persist_cash()
        return pos

    async def close_position(
        self, db: AsyncSession, symbol: str, close_price: float,
        fee_usdt: float = 0.0, sell_pct: float = 100.0,
    ) -> tuple[float, float]:
        """Close position (fully or partially), return (pnl_usdt, pnl_pct).

        sell_pct: 1-100. If < 100, only sell that percentage of the position.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return 0.0, 0.0

        sell_pct = max(1.0, min(100.0, sell_pct))
        sell_ratio = sell_pct / 100.0
        sell_qty = pos.quantity * sell_ratio

        entry_value = sell_qty * pos.avg_entry_price
        if entry_value < 0.01:
            entry_value = 0.01  # avoid div-by-zero on microcap dust
        pnl_usdt = sell_qty * (close_price - pos.avg_entry_price) - fee_usdt
        pnl_pct = pnl_usdt / entry_value * 100
        proceeds = sell_qty * close_price - fee_usdt
        self._cash_usdt += proceeds

        if sell_pct >= 99.5:
            # Full close
            self._positions.pop(symbol, None)
            await db.execute(delete(Position).where(Position.symbol == symbol))
        else:
            # Partial close — reduce quantity, keep position
            pos.quantity -= sell_qty
            pos.updated_at = datetime.now(timezone.utc)
            db.add(pos)

        await db.commit()
        await self._persist_cash()
        return round(pnl_usdt, 4), round(pnl_pct, 4)

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #
    async def take_snapshot(self, db: AsyncSession) -> PortfolioSnapshot:
        positions_data = [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "pnl_usdt": p.pnl_usdt,
            }
            for p in self._positions.values()
        ]
        snap = PortfolioSnapshot(
            total_value_usdt=round(self.total_value, 2),
            cash_usdt=round(self._cash_usdt, 2),
            positions_value_usdt=round(self.positions_value, 2),
            total_pnl_usdt=round(self.total_pnl_usdt, 2),
            total_pnl_pct=round(self.total_pnl_pct, 4),
            num_open_positions=len(self._positions),
            positions_json=json.dumps(positions_data),
        )
        db.add(snap)
        await db.commit()
        await db.refresh(snap)
        return snap

    # ------------------------------------------------------------------ #
    # State dict for API / WS
    # ------------------------------------------------------------------ #
    def get_state(self) -> PortfolioState:
        positions_out = [
            PositionOut(
                symbol=p.symbol,
                quantity=p.quantity,
                avg_entry_price=p.avg_entry_price,
                current_price=p.current_price,
                value_usdt=p.value_usdt,
                pnl_usdt=p.pnl_usdt,
                pnl_pct=p.pnl_pct,
                stop_loss_price=p.stop_loss_price,
                take_profit_price=p.take_profit_price,
                opened_at=p.opened_at,
                source=getattr(p, "source", "bot"),
            )
            for p in self._positions.values()
        ]
        return PortfolioState(
            total_value_usdt=round(self.total_value, 2),
            cash_usdt=round(self._cash_usdt, 2),
            positions_value_usdt=round(self.positions_value, 2),
            total_pnl_usdt=round(self.total_pnl_usdt, 2),
            total_pnl_pct=round(self.total_pnl_pct, 4),
            num_open_positions=len(self._positions),
            positions=positions_out,
        )


portfolio_service = PortfolioService()
