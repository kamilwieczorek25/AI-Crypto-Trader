"""Trade executor — demo fills + real orders + safety gates."""

import logging
import random

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.trade import Trade
from app.schemas.decision import TradeDecision
from app.services.market_data import MarketDataService
from app.services.portfolio import PortfolioService

logger = logging.getLogger(__name__)

# Slippage range for demo simulation (realistic market impact)
_SLIPPAGE_MIN = 0.0003   # 0.03%
_SLIPPAGE_MAX = 0.0012   # 0.12%

# Binance maker/taker fee (standard tier)
_FEE_RATE = 0.001  # 0.1%


def _apply_slippage(price: float, direction: str) -> float:
    """Simulate market slippage — BUY fills slightly above, SELL slightly below."""
    pct = random.uniform(_SLIPPAGE_MIN, _SLIPPAGE_MAX)
    return price * (1 + pct) if direction == "BUY" else price * (1 - pct)


class TradeExecutor:
    def __init__(
        self,
        portfolio: PortfolioService,
        market: MarketDataService,
    ) -> None:
        self._portfolio = portfolio
        self._market = market

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    async def execute(
        self, db: AsyncSession, decision: TradeDecision, decision_id: int
    ) -> Trade | None:
        """Execute a Claude decision. Returns Trade row or None (HOLD/skipped)."""
        if decision.action == "HOLD":
            logger.info("HOLD decision — no trade")
            return None

        if decision.action == "BUY":
            # Check if adding this position would exceed exposure limit
            proposed_usdt = self._portfolio.total_value * (decision.quantity_pct / 100)
            new_exposure = self._portfolio.positions_value + proposed_usdt
            new_exposure_pct = new_exposure / max(self._portfolio.total_value, 1) * 100
            if new_exposure_pct >= settings.MAX_TOTAL_EXPOSURE_PCT:
                logger.warning(
                    "Projected exposure %.1f%% >= limit %.1f%% — skipping BUY",
                    new_exposure_pct,
                    settings.MAX_TOTAL_EXPOSURE_PCT,
                )
                return None
            if decision.quantity_pct > settings.MAX_POSITION_PCT:
                logger.warning(
                    "quantity_pct %.1f%% > MAX %.1f%% — capping",
                    decision.quantity_pct,
                    settings.MAX_POSITION_PCT,
                )
                decision.quantity_pct = settings.MAX_POSITION_PCT

        if settings.is_demo:
            return await self._execute_demo(db, decision, decision_id)
        else:
            return await self._execute_real(db, decision, decision_id)

    # ------------------------------------------------------------------ #
    # Demo execution — simulated fill with slippage
    # ------------------------------------------------------------------ #
    async def _execute_demo(
        self, db: AsyncSession, decision: TradeDecision, decision_id: int
    ) -> Trade | None:
        raw_price = await self._market.get_price(decision.symbol)
        if raw_price <= 0:
            logger.error("Cannot get price for %s — demo trade skipped", decision.symbol)
            return None

        # Apply realistic slippage to simulated fill
        price = _apply_slippage(raw_price, decision.action)
        total_value = self._portfolio.total_value
        fee_usdt = 0.0

        if decision.action == "BUY":
            usdt_amount = total_value * (decision.quantity_pct / 100)
            if usdt_amount > self._portfolio.cash_usdt:
                original_pct = decision.quantity_pct
                usdt_amount = self._portfolio.cash_usdt * 0.99
                decision.quantity_pct = usdt_amount / total_value * 100
                logger.warning(
                    "Cash limited: reduced position from %.1f%% to %.1f%% ($%.2f available)",
                    original_pct, decision.quantity_pct, self._portfolio.cash_usdt,
                )
            if usdt_amount <= 0:
                logger.warning("Not enough cash for BUY demo trade")
                return None
            # Apply trading fee
            fee_usdt = usdt_amount * _FEE_RATE
            usdt_after_fee = usdt_amount - fee_usdt
            quantity = usdt_after_fee / price
            stop_loss_price = price * (1 - decision.stop_loss_pct / 100)
            take_profit_price = price * (1 + decision.take_profit_pct / 100)
            await self._portfolio.open_position(
                db, decision.symbol, quantity, price, stop_loss_price, take_profit_price,
                fee_usdt=fee_usdt,
            )
            pnl_usdt, pnl_pct = None, None

        elif decision.action == "SELL":
            pos = self._portfolio.get_position(decision.symbol)
            if pos is None:
                logger.warning("SELL on %s but no open position", decision.symbol)
                return None
            sell_pct = getattr(decision, "sell_pct", 100.0)
            quantity = pos.quantity * (min(100.0, max(1.0, sell_pct)) / 100.0)
            sell_proceeds = quantity * price
            fee_usdt = sell_proceeds * _FEE_RATE
            pnl_usdt, pnl_pct = await self._portfolio.close_position(
                db, decision.symbol, price, fee_usdt=fee_usdt, sell_pct=sell_pct,
            )
        else:
            return None

        trade = Trade(
            symbol=decision.symbol,
            direction=decision.action,
            mode="demo",
            quantity=quantity,
            price=price,
            quantity_pct=decision.quantity_pct,
            stop_loss_pct=decision.stop_loss_pct,
            take_profit_pct=decision.take_profit_pct,
            pnl_usdt=pnl_usdt,
            pnl_pct=pnl_pct,
            fee_usdt=fee_usdt,
            decision_id=decision_id,
            notes=f"slippage fill (raw=${raw_price:.6f}), fee=${fee_usdt:.4f}",
        )
        db.add(trade)
        await db.commit()
        await db.refresh(trade)
        logger.info(
            "[DEMO] %s %s qty=%.6f @ $%.6f (raw $%.6f, slip %.3f%%, fee $%.4f)",
            decision.action, decision.symbol, quantity, price, raw_price,
            abs(price - raw_price) / raw_price * 100, fee_usdt,
        )
        return trade

    # ------------------------------------------------------------------ #
    # Real execution — triple-gated
    # ------------------------------------------------------------------ #
    async def _execute_real(
        self, db: AsyncSession, decision: TradeDecision, decision_id: int
    ) -> Trade | None:
        if not settings.real_trading_allowed:
            logger.error(
                "Real trading attempted but guards not satisfied "
                "(MODE=real AND REAL_TRADING=true required)"
            )
            return None

        from app.services.market_data import market_data_service

        exchange = await market_data_service._get_exchange()
        total_value = self._portfolio.total_value

        if decision.action == "BUY":
            usdt_amount = total_value * (decision.quantity_pct / 100)
            if usdt_amount > self._portfolio.cash_usdt:
                original_pct = decision.quantity_pct
                usdt_amount = self._portfolio.cash_usdt * 0.99
                decision.quantity_pct = usdt_amount / total_value * 100
                logger.warning(
                    "Cash limited: reduced position from %.1f%% to %.1f%%",
                    original_pct, decision.quantity_pct,
                )
            price = await self._market.get_price(decision.symbol)
            if price <= 0:
                logger.error("Cannot get price for %s — real trade skipped", decision.symbol)
                return None
            quantity = usdt_amount / price

            order = await exchange.create_market_buy_order(decision.symbol, quantity)
            fill_price = float(order.get("average") or order.get("price") or price)
            fee_usdt = float(order.get("cost", usdt_amount)) * _FEE_RATE
            stop_loss_price = fill_price * (1 - decision.stop_loss_pct / 100)
            take_profit_price = fill_price * (1 + decision.take_profit_pct / 100)
            await self._portfolio.open_position(
                db, decision.symbol, quantity, fill_price,
                stop_loss_price, take_profit_price, fee_usdt=fee_usdt,
            )
            pnl_usdt, pnl_pct = None, None
            order_id = str(order.get("id", ""))

        elif decision.action == "SELL":
            pos = self._portfolio.get_position(decision.symbol)
            if pos is None:
                logger.warning("SELL on %s but no open position (real)", decision.symbol)
                return None
            sell_pct = getattr(decision, "sell_pct", 100.0)
            quantity = pos.quantity * (min(100.0, max(1.0, sell_pct)) / 100.0)
            order = await exchange.create_market_sell_order(decision.symbol, quantity)
            fill_price = float(order.get("average") or order.get("price") or 0)
            if fill_price <= 0:
                logger.error("Real SELL fill price is 0 for %s — aborting", decision.symbol)
                return None
            fee_usdt = float(order.get("cost", quantity * fill_price)) * _FEE_RATE
            pnl_usdt, pnl_pct = await self._portfolio.close_position(
                db, decision.symbol, fill_price, fee_usdt=fee_usdt, sell_pct=sell_pct,
            )
            order_id = str(order.get("id", ""))
        else:
            return None

        trade = Trade(
            symbol=decision.symbol,
            direction=decision.action,
            mode="real",
            quantity=quantity,
            price=fill_price,
            quantity_pct=decision.quantity_pct,
            stop_loss_pct=decision.stop_loss_pct,
            take_profit_pct=decision.take_profit_pct,
            pnl_usdt=pnl_usdt,
            pnl_pct=pnl_pct,
            fee_usdt=fee_usdt,
            order_id=order_id,
            decision_id=decision_id,
        )
        db.add(trade)
        await db.commit()
        await db.refresh(trade)
        logger.info(
            "[REAL] %s %s qty=%.6f @ $%.4f order_id=%s fee=$%.4f",
            decision.action, decision.symbol, quantity, fill_price, order_id, fee_usdt,
        )
        return trade
