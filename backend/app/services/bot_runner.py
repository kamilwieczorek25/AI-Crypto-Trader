"""Main async bot cycle loop — wires all services together."""

import asyncio
import json
import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.decision import ClaudeDecision
from app.schemas.dashboard import BotStatusData
from app.services import claude_engine, news
from app.services.executor import TradeExecutor
from app.services.lstm_model import lstm_predictor
from app.services.market_data import MarketDataService, market_data_service
from app.services.portfolio import PortfolioService, portfolio_service
from app.services.rag_engine import rag_engine
from app.services.rl_agent import rl_agent
from app.services.technical import compute_indicators, detect_market_regime, compute_correlation_matrix, detect_support_resistance

logger = logging.getLogger(__name__)


class BotRunner:
    def __init__(self) -> None:
        self._running = False
        self._task: asyncio.Task | None = None
        self._cycle_count = 0
        self._last_cycle_at: datetime | None = None
        self._next_cycle_at: datetime | None = None
        self._market = market_data_service
        self._portfolio = portfolio_service
        self._executor = TradeExecutor(self._portfolio, self._market)
        self._ws_hub: "WebSocketHub | None" = None  # injected after import
        # ML: per-cycle RL state cache (symbol -> state vector)
        self._rl_states: dict = {}
        # Dynamic sizing: recent trade streak tracking
        self._recent_results: list[bool] = []  # True=win, False=loss (last 10)
        # Cooldown: symbols that recently hit stop-loss
        self._sl_cooldown: dict[str, datetime] = {}  # symbol -> cooldown_until

    # ------------------------------------------------------------------ #
    # Control
    # ------------------------------------------------------------------ #
    def set_ws_hub(self, hub: object) -> None:
        self._ws_hub = hub  # type: ignore[assignment]

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="bot_loop")
        logger.info("Bot started in %s mode", settings.MODE)
        await self._broadcast_status()

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Bot stopped")
        await self._broadcast_status()

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    async def _loop(self) -> None:
        while self._running:
            try:
                await self._cycle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Cycle error: %s", exc)
                await self._broadcast_error(str(exc))

            self._next_cycle_at = datetime.now(timezone.utc).replace(
                microsecond=0
            )
            await self._broadcast_status()

            # Wait for next cycle with countdown broadcasts every 30s
            remaining = settings.CYCLE_INTERVAL_SECONDS
            while remaining > 0 and self._running:
                sleep_chunk = min(30, remaining)
                await asyncio.sleep(sleep_chunk)
                remaining -= sleep_chunk
                if self._running:
                    await self._broadcast_status()
                    await self._broadcast_price_tick()

    # ------------------------------------------------------------------ #
    # Single cycle
    # ------------------------------------------------------------------ #
    async def _cycle(self) -> None:
        logger.info("=== Cycle %d starting ===", self._cycle_count + 1)

        async with AsyncSessionLocal() as db:
            # 1. Get symbol universe (all passing volume filter, or capped by TOP_N_SYMBOLS)
            symbols = await self._market.get_top_symbols()

            if not symbols:
                logger.warning("No symbols passed volume filter — skipping cycle")
                await self._broadcast_error("No symbols available (exchange may be down)")
                self._cycle_count += 1
                self._last_cycle_at = datetime.now(timezone.utc)
                return

            # 2. Fetch OHLCV + orderbooks concurrently for all symbols
            #    Also fetch BTC/USDT as market anchor (even though it's excluded from trading)
            logger.info("Fetching data for %d symbols + BTC anchor...", len(symbols))
            all_ohlcv, all_orderbooks, btc_ohlcv = await asyncio.gather(
                self._market.get_multi_symbol_ohlcv(symbols),
                self._market.get_multi_symbol_orderbooks(symbols),
                self._market.get_multi_timeframe_ohlcv("BTC/USDT"),
            )

            symbols_data: dict = {}
            for sym in symbols:
                mtf = all_ohlcv.get(sym, {})
                indicators = {tf: compute_indicators(ohlcv) for tf, ohlcv in mtf.items()}
                price = indicators.get("1h", {}).get("close") or await self._market.get_price(sym)
                # Support/resistance from 4h candles (more meaningful levels)
                sr_candles = mtf.get("4h", mtf.get("1h", []))
                sr_levels = detect_support_resistance(sr_candles)
                symbols_data[sym] = {
                    "price": price,
                    "indicators": indicators,
                    "orderbook": all_orderbooks.get(sym, {}),
                    "support_resistance": sr_levels,
                }

            # 3. Update position prices
            prices = {sym: d["price"] for sym, d in symbols_data.items()}
            self._portfolio.update_prices(prices)

            # Broadcast market data
            await self._broadcast(
                "MARKET_DATA_UPDATE",
                {
                    sym: {
                        "price": d["price"],
                        "indicators_1h": d["indicators"].get("1h", {}),
                    }
                    for sym, d in symbols_data.items()
                },
            )

            # 4. Fetch news sentiment
            news_data = await news.fetch_news_sentiment(symbols)

            # 4a. Detect market regime and correlations
            market_regime = detect_market_regime(all_ohlcv)
            correlation_info = compute_correlation_matrix(all_ohlcv, top_n=15)

            # 4b. Compute BTC anchor data (trend, RSI, momentum for the whole market)
            btc_anchor: dict = {}
            for tf in ("1h", "4h", "1d"):
                btc_candles = btc_ohlcv.get(tf, [])
                if btc_candles:
                    btc_anchor[tf] = compute_indicators(btc_candles)

            logger.info(
                "Market regime: %s (vol=%s, ret=%.2f%%)",
                market_regime.get("regime", "?"),
                market_regime.get("volatility", "?"),
                market_regime.get("avg_return_pct", 0),
            )

            # 4b. Compute ML signals (LSTM + RL + RAG)
            portfolio_state = self._portfolio.get_state()
            portfolio_dict  = portfolio_state.model_dump()
            open_syms = {p.symbol for p in self._portfolio.all_positions()}

            ml_signals: dict = {}
            rl_states:  dict = {}
            for sym, data in symbols_data.items():
                candles_1h = all_ohlcv.get(sym, {}).get("1h", [])
                ind_1h     = data["indicators"].get("1h", {})
                lstm_pred  = lstm_predictor.predict(candles_1h)
                state      = rl_agent.build_state(
                    ind_1h, portfolio_dict, lstm_pred, sym in open_syms
                )
                rl_states[sym] = state
                ml_signals[sym] = {
                    "lstm": lstm_pred,
                    "rl":   rl_agent.recommend(state),
                }

            # RL: observe start of cycle → record portfolio value for reward baseline
            first_sym = next(iter(rl_states)) if rl_states else None
            if first_sym is not None:
                rl_agent.set_cycle_baseline(portfolio_state.total_value_usdt)

            # RAG: query relevant historical context
            rag_query   = " ".join(list(symbols_data.keys())[:4])
            rag_context = rag_engine.query(f"trading signals {rag_query}", k=5)

            # 5. Build prompt (with ML ensemble, RAG context, regime, correlations, BTC anchor)
            prompt = claude_engine.build_prompt(
                portfolio=portfolio_dict,
                symbols_data=symbols_data,
                news=news_data,
                ml_signals=ml_signals,
                rag_context=rag_context,
                market_regime=market_regime,
                correlation_info=correlation_info,
                btc_anchor=btc_anchor,
            )

            # 6. Call Claude
            logger.info("Calling Claude...")
            try:
                decision, raw_response = await claude_engine.call_claude(prompt)
            except Exception as exc:
                logger.error("Claude call failed: %s", exc)
                raise

            # 6b. Log top candidates based on ML signals for diagnostics
            if ml_signals:
                candidates = []
                for sym, sig in ml_signals.items():
                    lstm = sig.get("lstm", {})
                    rl = sig.get("rl", {})
                    score = lstm.get("confidence", 0) * 0.5
                    # Boost if LSTM+RL agree on BUY
                    if lstm.get("signal") == "BUY" and rl.get("action") == "BUY":
                        score += 0.3
                    candidates.append((sym, lstm.get("signal", "?"), score))
                candidates.sort(key=lambda x: x[2], reverse=True)
                top3 = candidates[:3]
                logger.info(
                    "Top ML candidates: %s",
                    " | ".join(f"{s}={sig}({sc:.2f})" for s, sig, sc in top3),
                )
                logger.info(
                    "Claude chose: %s %s (conf=%.2f)",
                    decision.action, decision.symbol, decision.confidence,
                )

            # 6c. Dynamic position sizing — scale down after consecutive losses
            if decision.action == "BUY":
                size_mult = self._position_size_multiplier()
                if size_mult < 1.0:
                    original_pct = decision.quantity_pct
                    decision.quantity_pct = round(decision.quantity_pct * size_mult, 2)
                    logger.info(
                        "Dynamic sizing: %.1f%% -> %.1f%% (mult=%.2f, streak=%s)",
                        original_pct, decision.quantity_pct, size_mult,
                        self._streak_summary(),
                    )

            # 6d. Cooldown after stop-loss — reject BUY on symbols recently stopped out
            now = datetime.now(timezone.utc)
            if decision.action == "BUY" and decision.symbol in self._sl_cooldown:
                cool_until = self._sl_cooldown[decision.symbol]
                if now < cool_until:
                    remaining_min = (cool_until - now).total_seconds() / 60
                    logger.info(
                        "Cooldown active on %s (%.0f min left) — overriding to HOLD",
                        decision.symbol, remaining_min,
                    )
                    decision.action = "HOLD"
                    decision.quantity_pct = 0.0
                else:
                    del self._sl_cooldown[decision.symbol]

            # 7. Log decision to DB (executed=False)
            db_decision = ClaudeDecision(
                raw_prompt=prompt[:50_000],  # safety truncation
                raw_response=raw_response,
                action=decision.action,
                symbol=decision.symbol,
                timeframe=decision.timeframe,
                quantity_pct=decision.quantity_pct,
                stop_loss_pct=decision.stop_loss_pct,
                take_profit_pct=decision.take_profit_pct,
                confidence=decision.confidence,
                primary_signals=json.dumps(decision.primary_signals),
                risk_factors=json.dumps(decision.risk_factors),
                reasoning=decision.reasoning,
                risk_profile=settings.RISK_PROFILE,
                executed=False,
            )
            db.add(db_decision)
            await db.commit()
            await db.refresh(db_decision)

            # 8. Broadcast decision (before execution)
            await self._broadcast(
                "CLAUDE_DECISION",
                {
                    "id": db_decision.id,
                    "action": decision.action,
                    "symbol": decision.symbol,
                    "timeframe": decision.timeframe,
                    "quantity_pct": decision.quantity_pct,
                    "confidence": decision.confidence,
                    "primary_signals": decision.primary_signals,
                    "risk_factors": decision.risk_factors,
                    "reasoning": decision.reasoning,
                },
            )

            # 9. Execute trade
            trade = await self._executor.execute(db, decision, db_decision.id)

            # 10. Mark decision executed
            db_decision.executed = trade is not None
            db.add(db_decision)
            await db.commit()

            # 11. Broadcast trade result
            if trade is not None:
                await self._broadcast(
                    "TRADE_EXECUTED",
                    {
                        "id": trade.id,
                        "symbol": trade.symbol,
                        "direction": trade.direction,
                        "mode": trade.mode,
                        "quantity": trade.quantity,
                        "price": trade.price,
                        "pnl_usdt": trade.pnl_usdt,
                        "pnl_pct": trade.pnl_pct,
                    },
                )
                # 11a. Track win/loss for dynamic sizing
                if trade.direction == "SELL" and trade.pnl_usdt is not None:
                    self._recent_results.append(trade.pnl_usdt > 0)
                    self._recent_results = self._recent_results[-10:]  # keep last 10

            # 11b. RL: record action & compute reward NOW (after execution)
            #      so the reward is attributed to the correct action
            if first_sym is not None:
                decided_state = rl_states.get(decision.symbol, rl_states[first_sym])
                rl_agent.record_action(decided_state, decision.action)
                # Compute reward immediately using post-trade portfolio value
                rl_agent.observe_cycle_end(
                    decided_state, self._portfolio.total_value
                )
                self._rl_states["_decided"] = decided_state

            # 11c. RAG: index news and this decision into long-term memory
            for sym, ns in news_data.items():
                hdl = ns.get("headlines", [])
                if hdl:
                    rag_engine.add_news(sym, hdl, ns.get("avg_sentiment", 0.0))

            rag_engine.add_decision_outcome(
                symbol=decision.symbol,
                action=decision.action,
                reasoning=decision.reasoning,
                pnl_pct=trade.pnl_pct if trade is not None else None,
                signals=decision.primary_signals,
            )

            # 12. Take portfolio snapshot + broadcast
            await self._portfolio.take_snapshot(db)
            await self._broadcast("PORTFOLIO_UPDATE", self._portfolio.get_state().model_dump())

        self._cycle_count += 1
        self._last_cycle_at = datetime.now(timezone.utc)
        logger.info("=== Cycle %d complete ===", self._cycle_count)

    # ------------------------------------------------------------------ #
    # Dynamic sizing & cooldown helpers
    # ------------------------------------------------------------------ #
    def _position_size_multiplier(self) -> float:
        """Scale position size based on recent win/loss streak.

        3+ consecutive losses → 50% size, 2 losses → 70%, otherwise 100%.
        3+ consecutive wins → 120% (capped by profile max).
        """
        if len(self._recent_results) < 2:
            return 1.0
        # Count consecutive streak from end
        last = self._recent_results[-1]
        streak = 0
        for r in reversed(self._recent_results):
            if r == last:
                streak += 1
            else:
                break
        if not last:  # losing streak
            if streak >= 3:
                return 0.5
            if streak >= 2:
                return 0.7
        else:  # winning streak
            if streak >= 3:
                return 1.2
        return 1.0

    def _streak_summary(self) -> str:
        if not self._recent_results:
            return "none"
        recent = self._recent_results[-5:]
        return "".join("W" if r else "L" for r in recent)

    def _add_sl_cooldown(self, symbol: str, minutes: int = 60) -> None:
        """Prevent re-buying a symbol for N minutes after stop-loss hit."""
        from datetime import timedelta
        self._sl_cooldown[symbol] = datetime.now(timezone.utc) + timedelta(minutes=minutes)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    async def _broadcast(self, event_type: str, data: object) -> None:
        if self._ws_hub:
            await self._ws_hub.broadcast({"type": event_type, "data": data})

    async def _broadcast_status(self) -> None:
        now = datetime.now(timezone.utc)
        next_in: int | None = None
        if self._next_cycle_at and self._running:
            # _next_cycle_at is set to `now` when the wait period starts
            elapsed = (now - self._next_cycle_at).total_seconds()
            next_in = max(0, int(settings.CYCLE_INTERVAL_SECONDS - elapsed))

        from app.services.claude_engine import get_profile_info
        await self._broadcast(
            "BOT_STATUS",
            {
                **BotStatusData(
                    running=self._running,
                    mode=settings.MODE,
                    next_cycle_in_seconds=next_in,
                    cycle_count=self._cycle_count,
                    last_cycle_at=self._last_cycle_at,
                ).model_dump(),
                "risk_profile": get_profile_info(),
            },
        )

    async def _broadcast_price_tick(self) -> None:
        if not self._portfolio.all_positions():
            return
        symbols = [p.symbol for p in self._portfolio.all_positions()]
        prices = await self._market.get_prices(symbols)
        self._portfolio.update_prices(prices)
        await self._broadcast(
            "PRICE_TICK",
            {sym: {"price": price} for sym, price in prices.items()},
        )
        await self._check_sl_tp()

    async def _check_sl_tp(self) -> None:
        """Auto-close positions that have hit their stop-loss or take-profit price."""
        triggers = self._portfolio.check_sl_tp_triggers()
        if not triggers:
            return

        for symbol, reason, price in triggers:
            logger.info("SL/TP triggered: %s on %s @ $%.6f", reason, symbol, price)
            # Add cooldown after stop-loss to prevent revenge trading
            if reason == "stop_loss":
                self._add_sl_cooldown(symbol, minutes=60)
                self._recent_results.append(False)
                self._recent_results = self._recent_results[-10:]
            elif reason == "take_profit":
                self._recent_results.append(True)
                self._recent_results = self._recent_results[-10:]
            async with AsyncSessionLocal() as db:
                # Log a synthetic SELL decision so the audit trail is complete
                from app.models.decision import ClaudeDecision
                label = "Stop-loss" if reason == "stop_loss" else "Take-profit"
                db_decision = ClaudeDecision(
                    raw_prompt=f"[AUTO {reason.upper()}]",
                    raw_response=f"Auto-triggered at ${price:.6f}",
                    action="SELL",
                    symbol=symbol,
                    timeframe="auto",
                    quantity_pct=0.0,
                    stop_loss_pct=0.0,
                    take_profit_pct=0.0,
                    confidence=1.0,
                    primary_signals=f'["{label} triggered at ${price:.6f}"]',
                    risk_factors="[]",
                    reasoning=f"Automatic {label.lower()} execution triggered at ${price:.6f}.",
                    executed=False,
                )
                db.add(db_decision)
                await db.commit()
                await db.refresh(db_decision)

                from app.schemas.decision import TradeDecision
                synthetic = TradeDecision(
                    action="SELL",
                    symbol=symbol,
                    timeframe="auto",
                    quantity_pct=0.0,
                    stop_loss_pct=0.0,
                    take_profit_pct=0.0,
                    confidence=1.0,
                    primary_signals=[f"{label} triggered at ${price:.6f}"],
                    risk_factors=[],
                    reasoning=f"Automatic {label.lower()} execution.",
                )
                trade = await self._executor.execute(db, synthetic, db_decision.id)

                if trade:
                    db_decision.executed = True
                    db.add(db_decision)
                    await db.commit()
                    await self._broadcast("TRADE_EXECUTED", {
                        "id": trade.id,
                        "symbol": trade.symbol,
                        "direction": trade.direction,
                        "mode": trade.mode,
                        "quantity": trade.quantity,
                        "price": trade.price,
                        "pnl_usdt": trade.pnl_usdt,
                        "pnl_pct": trade.pnl_pct,
                        "trigger": reason,
                    })
                    await self._broadcast(
                        "PORTFOLIO_UPDATE", self._portfolio.get_state().model_dump()
                    )

    async def _broadcast_error(self, message: str) -> None:
        await self._broadcast("ERROR", {"message": message})


bot_runner = BotRunner()
