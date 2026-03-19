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
            self._initial_value = settings.DEMO_INITIAL_BALANCE
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
            self._initial_value = settings.DEMO_INITIAL_BALANCE  # P&L always vs original balance
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

        reason is 'stop_loss' or 'take_profit'.
        """
        triggers: list[tuple[str, str, float]] = []
        for sym, pos in list(self._positions.items()):
            price = pos.current_price
            if price <= 0:
                continue
            if pos.stop_loss_price > 0 and price <= pos.stop_loss_price:
                triggers.append((sym, "stop_loss", price))
            elif pos.take_profit_price > 0 and price >= pos.take_profit_price:
                triggers.append((sym, "take_profit", price))
        return triggers

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
