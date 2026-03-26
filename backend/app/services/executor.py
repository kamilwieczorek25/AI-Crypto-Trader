"""Trade executor — demo fills + real orders + safety gates + DCA."""

import logging
import random
from datetime import datetime, timezone

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


def _extract_fee(order: dict, fallback_cost: float) -> float:
    """Extract fee in quote currency from ccxt order response."""
    fee_info = order.get("fee")
    if fee_info and fee_info.get("cost"):
        return float(fee_info["cost"])
    # Fallback: estimate from order cost
    cost = float(order.get("cost", 0)) or fallback_cost
    return cost * _FEE_RATE


def _apply_slippage(price: float, direction: str) -> float:
    """Simulate market slippage — BUY fills slightly above, SELL slightly below."""
    pct = random.uniform(_SLIPPAGE_MIN, _SLIPPAGE_MAX)
    return price * (1 + pct) if direction == "BUY" else price * (1 - pct)


class DCAOrder:
    """Pending DCA (Dollar-Cost Average) limit order."""
    __slots__ = ("symbol", "usdt_amount", "target_price", "stop_loss_price",
                 "take_profit_price", "created_at", "decision_id")

    def __init__(self, symbol: str, usdt_amount: float, target_price: float,
                 stop_loss_price: float, take_profit_price: float, decision_id: int):
        self.symbol = symbol
        self.usdt_amount = usdt_amount
        self.target_price = target_price
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price
        self.created_at = datetime.now(timezone.utc)
        self.decision_id = decision_id


class TradeExecutor:
    def __init__(
        self,
        portfolio: PortfolioService,
        market: MarketDataService,
    ) -> None:
        self._portfolio = portfolio
        self._market = market
        # Pending DCA orders: symbol -> DCAOrder
        self._pending_dca: dict[str, DCAOrder] = {}

    # ------------------------------------------------------------------ #
    # DCA order management
    # ------------------------------------------------------------------ #
    async def check_dca_fills(self, db: AsyncSession) -> list[Trade]:
        """Check pending DCA orders — fill if price has dipped to target."""
        filled: list[Trade] = []
        expired: list[str] = []
        now = datetime.now(timezone.utc)

        for sym, order in list(self._pending_dca.items()):
            current_price = await self._market.get_price(sym)
            if current_price <= 0:
                continue

            # Expire DCA orders older than 24h
            age_hours = (now - order.created_at).total_seconds() / 3600
            if age_hours > 24:
                expired.append(sym)
                continue

            # Fill if price dipped to target
            if current_price <= order.target_price:
                price = _apply_slippage(current_price, "BUY")
                fee_usdt = order.usdt_amount * _FEE_RATE
                usdt_after_fee = order.usdt_amount - fee_usdt
                quantity = usdt_after_fee / price

                if order.usdt_amount > self._portfolio.cash_usdt:
                    logger.warning("DCA fill skipped: insufficient cash for %s", sym)
                    expired.append(sym)
                    continue

                await self._portfolio.open_position(
                    db, sym, quantity, price,
                    order.stop_loss_price, order.take_profit_price,
                    fee_usdt=fee_usdt,
                )
                trade = Trade(
                    symbol=sym,
                    direction="BUY",
                    mode="demo" if settings.is_demo else "real",
                    quantity=quantity,
                    price=price,
                    quantity_pct=0,
                    stop_loss_pct=0,
                    take_profit_pct=0,
                    pnl_usdt=None,
                    pnl_pct=None,
                    fee_usdt=fee_usdt,
                    decision_id=order.decision_id,
                    notes=f"DCA fill at ${price:.6f} (target ${order.target_price:.6f})",
                )
                db.add(trade)
                await db.commit()
                await db.refresh(trade)
                filled.append(trade)
                expired.append(sym)
                logger.info(
                    "[DCA FILL] %s qty=%.6f @ $%.6f (target $%.6f)",
                    sym, quantity, price, order.target_price,
                )

        for sym in expired:
            self._pending_dca.pop(sym, None)

        return filled

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
            total_value = self._portfolio.total_value
            # Auto-bump tiny percentages up to meet Binance $5.50 minimum notional
            min_notional = 6.0  # slightly above Binance $5 minimum for safety
            min_pct = min_notional / max(total_value, 1) * 100
            if decision.quantity_pct < min_pct:
                if min_pct <= settings.MAX_POSITION_PCT and min_notional <= self._portfolio.cash_usdt:
                    logger.info(
                        "Bumping quantity_pct from %.1f%% to %.1f%% to meet $%.0f minimum notional",
                        decision.quantity_pct, min_pct, min_notional,
                    )
                    decision.quantity_pct = min_pct
                else:
                    logger.warning(
                        "Order too small (%.1f%% = $%.2f) and cannot bump to minimum — skipping BUY %s",
                        decision.quantity_pct, total_value * decision.quantity_pct / 100, decision.symbol,
                    )
                    return None

            # Check max open positions limit
            n_open = len(self._portfolio.all_positions())
            if settings.MAX_OPEN_POSITIONS > 0 and n_open >= settings.MAX_OPEN_POSITIONS:
                logger.warning(
                    "Already %d/%d open positions — skipping BUY %s",
                    n_open, settings.MAX_OPEN_POSITIONS, decision.symbol,
                )
                return None

            # Check if adding this position would exceed exposure limit
            proposed_usdt = total_value * (decision.quantity_pct / 100)
            new_exposure = self._portfolio.positions_value + proposed_usdt
            new_exposure_pct = new_exposure / max(total_value, 1) * 100
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

            # DCA split: place first tranche now, second as pending limit order
            dca_enabled = settings.DCA_ENABLED
            dca_split = settings.DCA_SPLIT_PCT / 100.0  # e.g. 0.6 = 60% now
            dca_dip = settings.DCA_DIP_PCT / 100.0      # e.g. 0.02 = 2% dip

            if dca_enabled and dca_split < 1.0:
                first_usdt = usdt_amount * dca_split
                second_usdt = usdt_amount - first_usdt
            else:
                first_usdt = usdt_amount
                second_usdt = 0.0

            # Apply trading fee
            fee_usdt = first_usdt * _FEE_RATE
            usdt_after_fee = first_usdt - fee_usdt
            quantity = usdt_after_fee / price
            stop_loss_price = price * (1 - decision.stop_loss_pct / 100)
            take_profit_price = price * (1 + decision.take_profit_pct / 100)
            await self._portfolio.open_position(
                db, decision.symbol, quantity, price, stop_loss_price, take_profit_price,
                fee_usdt=fee_usdt,
            )

            # Register DCA pending order for second tranche
            if second_usdt > 1.0:
                dca_target_price = price * (1 - dca_dip)
                self._pending_dca[decision.symbol] = DCAOrder(
                    symbol=decision.symbol,
                    usdt_amount=second_usdt,
                    target_price=dca_target_price,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    decision_id=decision_id,
                )
                logger.info(
                    "[DCA] %s: tranche1=$%.2f now, tranche2=$%.2f pending @ $%.6f (-%s%%)",
                    decision.symbol, first_usdt, second_usdt, dca_target_price,
                    settings.DCA_DIP_PCT,
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
            # Binance minimum notional is ~$5 — reject if too small
            if usdt_amount < 5.5:
                logger.warning(
                    "Order too small ($%.2f < $5.50 minimum) — skipping BUY %s",
                    usdt_amount, decision.symbol,
                )
                return None
            price = await self._market.get_price(decision.symbol)
            if price <= 0:
                logger.error("Cannot get price for %s — real trade skipped", decision.symbol)
                return None
            quantity = usdt_amount / price

            order = await exchange.create_market_buy_order(decision.symbol, quantity)
            fill_price = float(order.get("average") or order.get("price") or price)
            fee_usdt = _extract_fee(order, usdt_amount)
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

            try:
                order = await exchange.create_market_sell_order(decision.symbol, quantity)
            except Exception as sell_err:
                err_msg = str(sell_err).lower()
                if "insufficient" in err_msg or "balance" in err_msg:
                    # Actual exchange balance may differ — try fetching real balance
                    logger.warning(
                        "Insufficient balance selling %.6f %s — checking actual exchange balance",
                        quantity, decision.symbol,
                    )
                    base_asset = decision.symbol.split("/")[0]
                    try:
                        bal = await exchange.fetch_balance()
                        actual_free = float(bal.get(base_asset, {}).get("free", 0))
                    except Exception:
                        actual_free = 0.0
                    if actual_free > 0:
                        logger.info("Retrying sell with actual balance: %.6f %s", actual_free, base_asset)
                        quantity = actual_free
                        order = await exchange.create_market_sell_order(decision.symbol, quantity)
                    else:
                        # Nothing on exchange — just close the DB position
                        logger.warning(
                            "No %s balance on exchange — closing phantom DB position at last price",
                            base_asset,
                        )
                        last_price = await self._market.get_price(decision.symbol)
                        fill_price = last_price if last_price > 0 else pos.entry_price
                        pnl_usdt, pnl_pct = await self._portfolio.close_position(
                            db, decision.symbol, fill_price, fee_usdt=0.0, sell_pct=100.0,
                        )
                        return None
                else:
                    raise

            fill_price = float(order.get("average") or order.get("price") or 0)
            if fill_price <= 0:
                logger.error("Real SELL fill price is 0 for %s — aborting", decision.symbol)
                return None
            fee_usdt = _extract_fee(order, quantity * fill_price)
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
