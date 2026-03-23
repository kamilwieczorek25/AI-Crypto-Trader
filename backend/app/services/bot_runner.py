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
from app.schemas.decision import TradeDecision
from app.services import claude_engine, news
from app.services.discord import send_trade_notification, send_alert
from app.services.fast_scanner import fast_scanner
from app.services.quant_scorer import rank_symbols
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
        # Exit RL: last-known exit state per open position (symbol -> 18-dim state)
        self._exit_states: dict[str, list[float]] = {}
        # Cache of last cycle's ml_signals for use in _build_exit_state()
        self._ml_signals_cache: dict = {}
        # GPU cross-sectional momentum ranks: {symbol: percentile_0_to_1}
        self._momentum_ranks: dict[str, float] = {}
        # GPU sector rotation heat: {symbol: heat_-1_to_1}
        self._sector_heat: dict[str, float] = {}
        # Dynamic sizing: recent trade streak tracking
        self._recent_results: list[bool] = []  # True=win, False=loss (last 10)
        # Kelly criterion: cached fraction from backtest results
        self._kelly_fraction: float = 1.0  # 1.0 = no adjustment (full quant sizing)
        # Cooldown: symbols that recently hit stop-loss
        self._sl_cooldown: dict[str, datetime] = {}  # symbol -> cooldown_until
        # Circuit breaker: track peak portfolio value (set properly on start())
        self._peak_value: float = 0.0
        self._circuit_breaker_tripped = False
        # Market regime for dashboard broadcast
        self._last_regime: dict = {}
        # Symbols banned by exchange (not permitted for this account)
        self._banned_symbols: set[str] = set()
        # Low-funds notification: only send Discord alert once
        self._low_funds_notified: bool = False

    # ------------------------------------------------------------------ #
    # Control
    # ------------------------------------------------------------------ #
    def set_ws_hub(self, hub: object) -> None:
        self._ws_hub = hub  # type: ignore[assignment]

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        # Initialize peak value from actual portfolio (after exchange sync)
        actual_value = self._portfolio.total_value
        if actual_value > 0:
            self._peak_value = actual_value
        else:
            self._peak_value = settings.DEMO_INITIAL_BALANCE
        self._circuit_breaker_tripped = False
        logger.info(
            "Circuit breaker baseline: peak=$%.2f (actual portfolio)",
            self._peak_value,
        )

        # Auto-backtest on startup (non-blocking — runs in background)
        if settings.AUTO_BACKTEST:
            asyncio.create_task(self._run_startup_backtest(), name="startup_backtest")

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

            # Dynamic interval: conservative/balanced use longer cycles,
            # aggressive/fast_profit keep the configured interval
            interval = self._effective_cycle_interval()
            remaining = interval
            while remaining > 0 and self._running:
                sleep_chunk = min(30, remaining)
                await asyncio.sleep(sleep_chunk)
                remaining -= sleep_chunk
                if self._running:
                    try:
                        await self._broadcast_status()
                        await self._broadcast_price_tick()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning("Price tick error (non-fatal): %s", exc)

    # ------------------------------------------------------------------ #
    # Single cycle
    # ------------------------------------------------------------------ #
    async def _cycle(self) -> None:
        logger.info("=== Cycle %d starting ===", self._cycle_count + 1)

        # ── Exchange balance sync (real mode, every cycle) ──
        if not settings.is_demo and settings.BINANCE_API_KEY:
            try:
                async with AsyncSessionLocal() as sync_db:
                    await self._portfolio.sync_from_exchange(sync_db)
            except Exception as exc:
                logger.warning("Cycle exchange sync failed (non-fatal): %s", exc)

        # ── Circuit breaker check ──
        current_value = self._portfolio.total_value
        self._peak_value = max(self._peak_value, current_value)
        if self._peak_value > 0:
            drawdown_pct = (self._peak_value - current_value) / self._peak_value * 100
            if drawdown_pct >= settings.MAX_DRAWDOWN_PCT:
                msg = (
                    f"Circuit breaker: drawdown {drawdown_pct:.1f}% >= limit "
                    f"{settings.MAX_DRAWDOWN_PCT}% — pausing bot"
                )
                logger.warning(msg)
                self._circuit_breaker_tripped = True
                await self._broadcast_error(msg)
                await send_alert("Circuit Breaker Triggered", msg)
                await self.stop()
                return

        async with AsyncSessionLocal() as db:
            # 1. Get symbol universe
            symbols = await self._market.get_top_symbols()

            # 1a. Inject hot candidates from fast scanner (wider altcoin coverage)
            hot_syms = fast_scanner.hot_symbols
            if hot_syms:
                existing = set(symbols)
                injected = [s for s in hot_syms if s not in existing]
                if injected:
                    symbols.extend(injected)
                    logger.info(
                        "Fast scanner injected %d hot symbols: %s",
                        len(injected), ", ".join(injected[:5]),
                    )

            if not symbols:
                logger.warning("No symbols passed volume filter — skipping cycle")
                await self._broadcast_error("No symbols available (exchange may be down)")
                self._cycle_count += 1
                self._last_cycle_at = datetime.now(timezone.utc)
                return

            # 2. Fetch OHLCV + orderbooks + BTC anchor (resilient — BTC failure is non-fatal)
            logger.info("Fetching data for %d symbols + BTC anchor...", len(symbols))
            try:
                all_ohlcv, all_orderbooks = await asyncio.gather(
                    self._market.get_multi_symbol_ohlcv(symbols),
                    self._market.get_multi_symbol_orderbooks(symbols),
                )
            except Exception as exc:
                logger.error("Data fetch failed: %s — skipping cycle", exc)
                await self._broadcast_error(f"Market data unavailable: {exc}")
                self._cycle_count += 1
                self._last_cycle_at = datetime.now(timezone.utc)
                return

            btc_ohlcv: dict = {}
            try:
                btc_ohlcv = await self._market.get_multi_timeframe_ohlcv(f"BTC/{settings.QUOTE_CURRENCY}")
            except Exception:
                # Fallback: try BTC/USDT if quote currency pair doesn't exist
                try:
                    btc_ohlcv = await self._market.get_multi_timeframe_ohlcv("BTC/USDT")
                except Exception as exc:
                    logger.warning("BTC anchor fetch failed (non-fatal): %s", exc)

            symbols_data: dict = {}
            for sym in symbols:
                mtf = all_ohlcv.get(sym, {})
                indicators = {tf: compute_indicators(ohlcv) for tf, ohlcv in mtf.items()}
                price = indicators.get("1h", {}).get("close") or await self._market.get_price(sym)
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

            # 4. Fetch news sentiment (resilient — empty on failure)
            try:
                news_data = await news.fetch_news_sentiment(symbols)
            except Exception as exc:
                logger.warning("News fetch failed (non-fatal): %s", exc)
                news_data = {}

            # 4a. Detect market regime and correlations
            market_regime = detect_market_regime(all_ohlcv)
            correlation_info = compute_correlation_matrix(all_ohlcv, top_n=15)
            self._last_regime = market_regime

            # 4a2. Auto-adjust risk profile based on regime
            new_profile = claude_engine.auto_adjust_risk_profile(
                market_regime.get("regime", "unknown")
            )
            if new_profile:
                await self._broadcast_status()  # push new profile to dashboard

            # 4b. Compute BTC anchor data
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

            # 4c. Compute ML signals (LSTM + RL + RAG) — resilient
            portfolio_state = self._portfolio.get_state()
            portfolio_dict  = portfolio_state.model_dump()
            open_syms = {p.symbol for p in self._portfolio.all_positions()}

            ml_signals: dict = {}
            rl_states:  dict = {}
            try:
                from app.services import gpu_client

                async def _ml_for_symbol(sym: str, data: dict) -> tuple:
                    """Compute LSTM + RL + ensemble + MTF + anomaly + vol + attention signals.

                    All symbols are independent — gathered in parallel so GPU
                    round-trips overlap instead of queuing sequentially.
                    """
                    candles_1h = all_ohlcv.get(sym, {}).get("1h", [])
                    ind_1h     = data["indicators"].get("1h", {})

                    # LSTM: try remote GPU first, fall back to local CPU
                    lstm_pred = await lstm_predictor.predict_remote(candles_1h)
                    if lstm_pred is None:
                        lstm_pred = lstm_predictor.predict(candles_1h)

                    state = rl_agent.build_state(
                        ind_1h, portfolio_dict, lstm_pred, sym in open_syms
                    )

                    # RL: try remote GPU first, fall back to local CPU
                    rl_rec = await rl_agent.recommend_remote(state)
                    if rl_rec is None:
                        rl_rec = rl_agent.recommend(state)

                    sig: dict = {"lstm": lstm_pred, "rl": rl_rec}

                    # GPU ensemble: Transformer + LSTM + RL + semantic sentiment
                    if gpu_client.is_enabled():
                        headlines = news_data.get(sym, {}).get("headlines", [])
                        ensemble = await gpu_client.predict_ensemble(
                            candles_1h, state.tolist(), headlines,
                        )
                        if ensemble:
                            sig["ensemble"] = {
                                "signal":     ensemble.get("ensemble_signal", "HOLD"),
                                "confidence": ensemble.get("ensemble_confidence", 0),
                                "agreement":  ensemble.get("agreement_score", 0),
                            }

                        # Multi-Timeframe Fusion prediction
                        mtf_candles = {
                            tf: all_ohlcv.get(sym, {}).get(tf, [])
                            for tf in ("15m", "1h", "4h", "1d")
                        }
                        mtf_candles = {tf: c for tf, c in mtf_candles.items() if c}
                        if len(mtf_candles) >= 2:
                            mtf_pred = await gpu_client.predict_mtf(mtf_candles)
                            if mtf_pred:
                                sig["mtf"] = {
                                    "signal": mtf_pred.get("signal", "HOLD"),
                                    "confidence": mtf_pred.get("confidence", 0),
                                    "timeframes": mtf_pred.get("timeframes_used", []),
                                }

                        # Anomaly detection
                        anomaly = await gpu_client.detect_anomaly(candles_1h)
                        if anomaly:
                            sig["anomaly"] = {
                                "is_anomaly": anomaly.get("is_anomaly", False),
                                "anomaly_score": anomaly.get("anomaly_score", 0),
                            }

                        # Volatility forecast (for MC refinement later)
                        vol_pred = await gpu_client.predict_volatility(candles_1h)
                        if vol_pred:
                            sig["vol_forecast"] = {
                                "predicted_vol": vol_pred.get("predicted_vol", 0),
                                "source": vol_pred.get("source", "historical"),
                            }

                        # Attention explainability (only for top symbols by volume)
                        attn = await gpu_client.explain_attention(candles_1h)
                        if attn:
                            sig["attention"] = {
                                "top_features": list(attn.get("feature_importance", {}).items())[:3],
                                "top_candles": attn.get("top_candle_positions", [])[:3],
                                "entropy": attn.get("attention_entropy", 0),
                            }

                    return sym, state, sig

                # Run all symbols concurrently — GPU round-trips overlap instead
                # of serialising (e.g. 8 symbols × 6 calls = parallel)
                results = await asyncio.gather(
                    *[_ml_for_symbol(sym, data) for sym, data in symbols_data.items()],
                    return_exceptions=True,
                )
                for res in results:
                    if isinstance(res, Exception):
                        logger.warning("ML signal failed for one symbol (non-fatal): %s", res)
                        continue
                    sym, state, sig = res
                    rl_states[sym]  = state
                    ml_signals[sym] = sig

                # GPU cross-symbol correlation tracker
                if gpu_client.is_enabled():
                    corr_candles = {
                        sym: all_ohlcv.get(sym, {}).get("1h", [])
                        for sym in list(symbols_data.keys())[:30]
                    }
                    corr_candles = {s: c for s, c in corr_candles.items() if c}
                    if len(corr_candles) >= 3:
                        gpu_corr = await gpu_client.compute_correlations(corr_candles)
                        if gpu_corr:
                            for div_sig in gpu_corr.get("divergence_signals", []):
                                laggard = div_sig.get("laggard", "")
                                if laggard in ml_signals:
                                    ml_signals[laggard]["corr_divergence"] = div_sig
                            if gpu_corr.get("high_corr_pairs"):
                                correlation_info["gpu_high_corr_pairs"] = gpu_corr["high_corr_pairs"]

                # Cache ml_signals for next-cycle exit state building
                self._ml_signals_cache = dict(ml_signals)

                # GPU cross-sectional momentum ranking + sector rotation
                # Both calls are cheap (pure matrix ops) and run in parallel
                if gpu_client.is_enabled() and len(symbols_data) >= 4:
                    all_1h_candles = {
                        sym: all_ohlcv.get(sym, {}).get("1h", [])
                        for sym in symbols_data
                    }
                    all_1h_candles = {s: c for s, c in all_1h_candles.items() if c}
                    if len(all_1h_candles) >= 4:
                        _mom_task = asyncio.ensure_future(
                            gpu_client.rank_momentum(all_1h_candles)
                        )
                        _rot_task = asyncio.ensure_future(
                            gpu_client.cluster_rotation(all_1h_candles)
                        )
                        _mom_result, _rot_result = await asyncio.gather(
                            _mom_task, _rot_task, return_exceptions=True
                        )
                        self._momentum_ranks = (
                            _mom_result if isinstance(_mom_result, dict) else {}
                        )
                        _rot_data = _rot_result if isinstance(_rot_result, dict) else {}
                        self._sector_heat = _rot_data.get("sector_heat", {})
                        if _rot_data.get("hot_sectors"):
                            logger.info(
                                "GPU sectors: hot=%s cold=%s",
                                [s["label"] for s in _rot_data["hot_sectors"][:2]],
                                [s["label"] for s in _rot_data.get("cold_sectors", [])[:2]],
                            )

                # Exit RL: for each open position, predict optimal exit action
                if gpu_client.is_enabled():
                    for pos in self._portfolio.all_positions():
                        sym = pos.symbol
                        ind_1h = symbols_data.get(sym, {}).get("indicators", {}).get("1h", {})
                        exit_state = self._build_exit_state(pos, ind_1h, btc_anchor)
                        self._exit_states[sym] = exit_state
                        try:
                            exit_pred = await gpu_client.predict_exit(exit_state)
                        except Exception:
                            exit_pred = None
                        if exit_pred:
                            if sym not in ml_signals:
                                ml_signals[sym] = {}
                            ml_signals[sym]["exit_rl"] = {
                                "action":   exit_pred.get("action", "HOLD_POS"),
                                "q_values": exit_pred.get("q_values", []),
                            }

            except Exception as exc:
                logger.warning("ML signal computation failed (non-fatal): %s", exc)

            # RL: set cycle baseline
            first_sym = next(iter(rl_states)) if rl_states else None
            if first_sym is not None:
                rl_agent.set_cycle_baseline(portfolio_state.total_value_usdt)

            # RAG: query relevant historical context (resilient)
            rag_context: list[str] = []
            try:
                rag_query   = " ".join(list(symbols_data.keys())[:4])
                rag_context = rag_engine.query(f"trading signals {rag_query}", k=5)
            except Exception as exc:
                logger.warning("RAG query failed (non-fatal): %s", exc)

            # 4d. Fetch additional market intelligence (funding, L/S, fear/greed)
            market_intel: dict = {}
            try:
                from app.services.market_intel import fetch_market_intelligence
                market_intel = await fetch_market_intelligence(symbols)
            except Exception as exc:
                logger.warning("Market intel fetch failed (non-fatal): %s", exc)

            # ── Low-funds guard: skip trading if cash below minimum trade size ──
            min_trade_usdt = 6.0  # Binance minimum notional
            available_cash = self._portfolio.cash_usdt

            if available_cash < min_trade_usdt:
                if not self._low_funds_notified:
                    msg = (
                        f"Low funds: ${available_cash:.2f} {settings.QUOTE_CURRENCY} available "
                        f"(need ${min_trade_usdt:.2f} minimum to trade). "
                        f"Monitoring {len(self._portfolio.all_positions())} open position(s). "
                        f"Deposit {settings.QUOTE_CURRENCY} to resume trading."
                    )
                    logger.warning(msg)
                    await send_alert("Low Funds — Trading Paused", msg)
                    await self._broadcast_error(msg)
                    self._low_funds_notified = True

                # Still check SL/TP for existing positions
                await self._broadcast_price_tick()
                await self._portfolio.take_snapshot(db)
                await self._broadcast("PORTFOLIO_UPDATE", self._portfolio.get_state().model_dump())
                self._cycle_count += 1
                self._last_cycle_at = datetime.now(timezone.utc)
                return
            else:
                # Funds available again — reset notification flag
                if self._low_funds_notified:
                    logger.info("Funds restored: $%.2f %s available — resuming trading", available_cash, settings.QUOTE_CURRENCY)
                    self._low_funds_notified = False

            # 5. Quant scoring — signal-first architecture
            #    Only consider bot-managed positions as "held" for SELL signals
            #    (external positions are user-managed — bot won't sell them)
            bot_held = {
                p.symbol for p in self._portfolio.all_positions()
                if getattr(p, "source", "bot") == "bot"
            }
            quant_candidates = rank_symbols(
                symbols_data, news_data,
                ml_signals=ml_signals,
                btc_anchor=btc_anchor,
                held_symbols=bot_held,
                market_regime=market_regime,
                market_intel=market_intel,
                momentum_ranks=self._momentum_ranks or None,
                sector_heat=self._sector_heat or None,
            )

            # 4e. Refine BUY candidates with GPU Monte Carlo simulation
            try:
                from app.services import gpu_client as _gpu
                _gpu_available = _gpu.is_enabled()
            except Exception:
                _gpu_available = False
            if quant_candidates and _gpu_available:
                from app.services.quant_scorer import refine_with_monte_carlo

                async def _mc_refine(cand):
                    if cand.action != "BUY":
                        return cand
                    candles_1h = all_ohlcv.get(cand.symbol, {}).get("1h", [])
                    levels = {
                        "sl_pct": cand.stop_loss_pct,
                        "tp_pct": cand.take_profit_pct,
                        "rr_ratio": cand.reward_risk_ratio,
                        "quantity_pct": cand.quantity_pct,
                        "atr_pct": 0,
                    }
                    refined = await refine_with_monte_carlo(levels, candles_1h, cand.entry_price)
                    cand.stop_loss_pct = refined["sl_pct"]
                    cand.take_profit_pct = refined["tp_pct"]
                    cand.reward_risk_ratio = refined["rr_ratio"]
                    cand.stop_loss_price = round(cand.entry_price * (1 - refined["sl_pct"] / 100), 6)
                    cand.take_profit_price = round(cand.entry_price * (1 + refined["tp_pct"] / 100), 6)
                    if "mc_edge" in refined:
                        cand.signals["mc_edge"] = refined["mc_edge"]
                        cand.signals["mc_tp_prob"] = refined.get("mc_tp_prob", 0)
                        cand.signals["mc_sl_prob"] = refined.get("mc_sl_prob", 0)
                    return cand

                mc_results = await asyncio.gather(
                    *[_mc_refine(c) for c in quant_candidates],
                    return_exceptions=True,
                )
                quant_candidates = [r for r in mc_results if not isinstance(r, Exception)]

            # 4f. Anomaly detection — reject BUY candidates flagged as anomalous
            if quant_candidates and _gpu_available:
                async def _anomaly_check(cand):
                    if cand.action != "BUY":
                        return cand
                    sig = ml_signals.get(cand.symbol, {})
                    anom = sig.get("anomaly", {})
                    if anom.get("is_anomaly"):
                        logger.warning(
                            "Anomaly rejection: %s (score=%.2f) — skipping BUY",
                            cand.symbol, anom.get("anomaly_score", 0),
                        )
                        return None  # reject
                    return cand

                anom_results = await asyncio.gather(
                    *[_anomaly_check(c) for c in quant_candidates],
                    return_exceptions=True,
                )
                quant_candidates = [
                    r for r in anom_results
                    if r is not None and not isinstance(r, Exception)
                ]

            # 4g. Exit RL — check if open positions should be closed/reduced
            if _gpu_available and self._portfolio.all_positions():
                try:
                    from app.services import gpu_client as _gpu
                    for pos in self._portfolio.all_positions():
                        if getattr(pos, "source", "bot") != "bot":
                            continue
                        sig = ml_signals.get(pos.symbol, {})
                        ind_1h = symbols_data.get(pos.symbol, {}).get("indicators", {}).get("1h", {})
                        hold_duration = (datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 3600 if hasattr(pos, 'opened_at') and pos.opened_at else 0
                        pnl_pct = ((pos.current_price - pos.avg_entry_price) / pos.avg_entry_price * 100) if pos.avg_entry_price > 0 else 0
                        highest = getattr(pos, 'highest_price', pos.current_price) or pos.current_price
                        drawdown_from_high = ((highest - pos.current_price) / highest * 100) if highest > 0 else 0
                        exit_state = [
                            pnl_pct / 100,
                            min(hold_duration / 48, 1.0),
                            drawdown_from_high / 100,
                            ind_1h.get("rsi14", 50) / 100,
                            ind_1h.get("macd_hist", 0) * 50,
                            ind_1h.get("bb_pct_b", 0.5),
                            ind_1h.get("volume_ratio", 1.0) / 5,
                            ind_1h.get("trend", 0),
                            ind_1h.get("obv_trend", 0),
                            ind_1h.get("price_vs_vwap", 0) / 5,
                            sig.get("ensemble", {}).get("confidence", 0.5),
                            1.0 if sig.get("ensemble", {}).get("signal") == "SELL" else 0.0,
                            sig.get("anomaly", {}).get("anomaly_score", 0) / 5,
                            sig.get("vol_forecast", {}).get("predicted_vol", 0.02) * 10,
                            pos.stop_loss_pct / 100 if hasattr(pos, 'stop_loss_pct') and pos.stop_loss_pct else 0.05,
                            pos.take_profit_pct / 100 if hasattr(pos, 'take_profit_pct') and pos.take_profit_pct else 0.10,
                            min(pos.quantity * pos.current_price / max(self._portfolio.total_value, 1), 1.0),
                            sig.get("mtf", {}).get("confidence", 0.5),
                        ]
                        exit_rec = await _gpu.predict_exit(exit_state)
                        if exit_rec:
                            ml_signals.setdefault(pos.symbol, {})["exit_rl"] = exit_rec
                except Exception as exc:
                    logger.warning("Exit RL check failed (non-fatal): %s", exc)

            # 4h. Train MTF model with multi-TF candle data
            if _gpu_available:
                try:
                    from app.services import gpu_client as _gpu
                    mtf_train_data = {}
                    for sym in list(symbols_data.keys())[:15]:
                        sym_ohlcv = all_ohlcv.get(sym, {})
                        if len(sym_ohlcv) >= 2:
                            mtf_train_data[sym] = sym_ohlcv
                    if mtf_train_data:
                        asyncio.create_task(_gpu.train_mtf(mtf_train_data))
                except Exception:
                    pass

            # 5a. No candidates → HOLD (no Claude call, saves API cost)
            if not quant_candidates:
                logger.info("Quant scorer: 0 candidates — HOLD cycle (no Claude call)")
                claude_engine.record_skipped_cycle()
                await self._broadcast("CYCLE_SKIPPED", {
                    "reason": "Quant scorer: no candidates above threshold",
                })
                self._cycle_count += 1
                self._last_cycle_at = datetime.now(timezone.utc)
                await self._portfolio.take_snapshot(db)
                await self._broadcast("PORTFOLIO_UPDATE", self._portfolio.get_state().model_dump())
                return

            # 5a2. Correlation rejection — skip BUY candidates highly correlated
            #      with existing positions (r > 0.8) to avoid concentration risk
            if correlation_info and self._portfolio.all_positions():
                held_syms = {p.symbol for p in self._portfolio.all_positions()}
                corr_matrix = correlation_info.get("matrix", {})
                filtered: list = []
                for cand in quant_candidates:
                    if cand.action != "BUY":
                        filtered.append(cand)
                        continue
                    correlated = False
                    for held_sym in held_syms:
                        key = tuple(sorted([cand.symbol, held_sym]))
                        r = corr_matrix.get(key, corr_matrix.get(
                            f"{key[0]}_{key[1]}", 0
                        ))
                        if isinstance(r, (int, float)) and abs(r) > 0.8:
                            logger.info(
                                "Correlation rejection: %s correlated with held %s (r=%.2f)",
                                cand.symbol, held_sym, r,
                            )
                            correlated = True
                            break
                    if not correlated:
                        filtered.append(cand)
                if len(filtered) < len(quant_candidates):
                    logger.info(
                        "Correlation filter: %d -> %d candidates",
                        len(quant_candidates), len(filtered),
                    )
                    quant_candidates = filtered

            if not quant_candidates:
                logger.info("All candidates rejected by correlation filter — HOLD")
                claude_engine.record_skipped_cycle()
                self._cycle_count += 1
                self._last_cycle_at = datetime.now(timezone.utc)
                await self._portfolio.take_snapshot(db)
                await self._broadcast("PORTFOLIO_UPDATE", self._portfolio.get_state().model_dump())
                return

            # 5b. Filter out banned symbols (not permitted by exchange)
            if self._banned_symbols:
                before = len(quant_candidates)
                quant_candidates = [
                    c for c in quant_candidates
                    if c.symbol not in self._banned_symbols
                ]
                if len(quant_candidates) < before:
                    logger.info(
                        "Banned symbols filter: %d -> %d candidates",
                        before, len(quant_candidates),
                    )

            if not quant_candidates:
                logger.info("All candidates banned or rejected — HOLD")
                claude_engine.record_skipped_cycle()
                self._cycle_count += 1
                self._last_cycle_at = datetime.now(timezone.utc)
                await self._portfolio.take_snapshot(db)
                await self._broadcast("PORTFOLIO_UPDATE", self._portfolio.get_state().model_dump())
                return

            # 5c. Skip Claude if top candidate score is too weak to ever pass
            top_candidates = quant_candidates[:2]
            best_score = top_candidates[0].score if top_candidates else 0
            if best_score < 55 and not any(c.action == "SELL" for c in top_candidates):
                logger.info(
                    "Quant: top score %.0f < 55 — skipping Claude call (would HOLD anyway)",
                    best_score,
                )
                claude_engine.record_skipped_cycle()
                self._cycle_count += 1
                self._last_cycle_at = datetime.now(timezone.utc)
                await self._portfolio.take_snapshot(db)
                await self._broadcast("PORTFOLIO_UPDATE", self._portfolio.get_state().model_dump())
                return

            logger.info(
                "Quant: %d candidates, top %d → Claude validation: %s",
                len(quant_candidates), len(top_candidates),
                " | ".join(f"{c.symbol} {c.action} score={c.score:.0f}" for c in top_candidates),
            )

            # 6. Build validation prompt (compact — only candidates + context)
            prompt = claude_engine.build_validation_prompt(
                candidates=top_candidates,
                portfolio=portfolio_dict,
                news=news_data,
                market_regime=market_regime,
                btc_anchor=btc_anchor,
                correlation_info=correlation_info,
                market_intel=market_intel,
            )

            # 6a. Model tier: Haiku for BUY-only validation, Sonnet for SELL decisions
            use_haiku = (
                settings.USE_HAIKU_FOR_HOLD
                and not any(c.action == "SELL" for c in top_candidates)
            )

            # 7. Call Claude (validation mode)
            tier_label = "haiku" if use_haiku else "sonnet"
            logger.info("Calling Claude [%s] for validation...", tier_label)
            try:
                decision, raw_response = await claude_engine.call_claude(
                    prompt, use_haiku=use_haiku, validation_mode=True,
                )
            except Exception as exc:
                logger.error("Claude validation failed: %s", exc)
                raise

            # 7a. Override SL/TP with quant model values (Claude validates, not sets levels)
            if decision.action in ("BUY", "SELL"):
                matched = next(
                    (c for c in top_candidates
                     if c.symbol == decision.symbol and c.action == decision.action),
                    None,
                )
                if matched:
                    decision.stop_loss_pct = matched.stop_loss_pct
                    decision.take_profit_pct = matched.take_profit_pct
                    decision.confidence = matched.score / 100.0
                    # Don't let Claude increase position size beyond quant suggestion
                    decision.quantity_pct = min(decision.quantity_pct, matched.quantity_pct)
                    logger.info(
                        "Claude approved: %s %s (quant_score=%.0f, R:R=%.1f, SL=%.1f%%, TP=%.1f%%)",
                        decision.action, decision.symbol, matched.score,
                        matched.reward_risk_ratio, matched.stop_loss_pct, matched.take_profit_pct,
                    )
                else:
                    # Claude chose a symbol/action not in candidates — treat as HOLD
                    logger.warning(
                        "Claude chose %s %s not in candidates — overriding to HOLD",
                        decision.action, decision.symbol,
                    )
                    decision.action = "HOLD"
                    decision.quantity_pct = 0.0
            else:
                logger.info("Claude rejected all candidates — HOLD")

            # 7a2. Less-fear override: if Claude says HOLD but top candidate scores well,
            #      auto-approve the highest-scoring BUY candidate
            if decision.action == "HOLD" and settings.LESS_FEAR:
                best_buy = next(
                    (c for c in top_candidates if c.action == "BUY" and c.score >= 55),
                    None,
                )
                if best_buy:
                    logger.info(
                        "Less-fear override: Claude said HOLD but %s has score=%.0f — auto-approving BUY",
                        best_buy.symbol, best_buy.score,
                    )
                    decision.action = "BUY"
                    decision.symbol = best_buy.symbol
                    decision.timeframe = best_buy.timeframe
                    decision.quantity_pct = best_buy.quantity_pct
                    decision.stop_loss_pct = best_buy.stop_loss_pct
                    decision.take_profit_pct = best_buy.take_profit_pct
                    decision.confidence = best_buy.score / 100.0
                    decision.reasoning = (
                        f"[LESS FEAR OVERRIDE] Claude rejected but quant score "
                        f"{best_buy.score:.0f} exceeds threshold. "
                        f"Original: {decision.reasoning}"
                    )

            # 7b. Dynamic position sizing — scale down after consecutive losses
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

            # 7b2. Kelly criterion sizing — scale position based on backtest edge
            if decision.action == "BUY" and settings.KELLY_SIZING:
                kelly_mult = min(self._kelly_fraction, settings.KELLY_FRACTION_CAP)
                if kelly_mult < 1.0:
                    original_pct = decision.quantity_pct
                    decision.quantity_pct = round(decision.quantity_pct * kelly_mult, 2)
                    logger.info(
                        "Kelly sizing: %.1f%% -> %.1f%% (kelly=%.2f)",
                        original_pct, decision.quantity_pct, kelly_mult,
                    )

            # 7c. Cooldown after stop-loss — reject BUY on symbols recently stopped out
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

            # 8. Log decision to DB (executed=False)
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

            # 9. Broadcast decision (before execution)
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

            # 10. Execute trade (no fallback to unvalidated symbols)
            trade = None
            tried_symbols: set[str] = set()

            tried_symbols.add(decision.symbol)
            try:
                trade = await self._executor.execute(db, decision, db_decision.id)
            except Exception as exec_err:
                err_msg = str(exec_err).lower()
                if "not permitted" in err_msg or "not allowed" in err_msg:
                    self._banned_symbols.add(decision.symbol)
                    logger.warning(
                        "Symbol %s banned (not permitted). Banned: %s",
                        decision.symbol, self._banned_symbols,
                    )
                else:
                    raise

            # 11. Mark decision executed
            db_decision.executed = trade is not None
            db.add(db_decision)
            await db.commit()

            # 12. Broadcast trade result + Discord notification
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
                # Discord notification
                await send_trade_notification(
                    action=trade.direction,
                    symbol=trade.symbol,
                    price=trade.price,
                    quantity=trade.quantity,
                    pnl_usdt=trade.pnl_usdt,
                    pnl_pct=trade.pnl_pct,
                    confidence=decision.confidence,
                    reasoning=decision.reasoning,
                )
                # 12a. Track win/loss for dynamic sizing
                if trade.direction == "SELL" and trade.pnl_usdt is not None:
                    self._recent_results.append(trade.pnl_usdt > 0)
                    self._recent_results = self._recent_results[-10:]
                    # Exit RL: submit close experience so the model learns from this outcome
                    if trade.pnl_pct is not None:
                        await self._submit_exit_experience(
                            trade.symbol, trade.pnl_pct, action=3
                        )  # action 3 = CLOSE

            # 12b. RL: record action & compute reward NOW (after execution)
            #      so the reward is attributed to the correct action
            if first_sym is not None:
                decided_state = rl_states.get(decision.symbol, rl_states[first_sym])
                rl_agent.record_action(decided_state, decision.action)
                # Compute reward immediately using post-trade portfolio value
                rl_agent.observe_cycle_end(
                    decided_state, self._portfolio.total_value
                )
                self._rl_states["_decided"] = decided_state

            # 12c. RAG: index news and this decision into long-term memory
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

            # 13. Take portfolio snapshot + broadcast
            await self._portfolio.take_snapshot(db)
            await self._broadcast("PORTFOLIO_UPDATE", self._portfolio.get_state().model_dump())

            # 13a. Check pending DCA orders
            dca_fills = await self._executor.check_dca_fills(db)
            for dca_trade in dca_fills:
                await self._broadcast("TRADE_EXECUTED", {
                    "id": dca_trade.id,
                    "symbol": dca_trade.symbol,
                    "direction": "BUY",
                    "mode": dca_trade.mode,
                    "quantity": dca_trade.quantity,
                    "price": dca_trade.price,
                    "trigger": "dca",
                })
                await send_trade_notification(
                    action="BUY",
                    symbol=dca_trade.symbol,
                    price=dca_trade.price,
                    quantity=dca_trade.quantity,
                    pnl_usdt=None,
                    pnl_pct=None,
                    confidence=0.8,
                    reasoning="DCA tranche 2 filled on dip.",
                    trigger="DCA",
                )

            # 13b. Pyramiding — add to profitable positions with high quant score
            if settings.PYRAMID_ENABLED:
                await self._check_pyramiding(db, symbols_data, news_data,
                                             ml_signals, btc_anchor, market_intel)

        self._cycle_count += 1
        self._last_cycle_at = datetime.now(timezone.utc)
        logger.info("=== Cycle %d complete ===", self._cycle_count)

    # ------------------------------------------------------------------ #
    # Dynamic sizing, cooldown & cycle interval helpers
    # ------------------------------------------------------------------ #
    def _effective_cycle_interval(self) -> int:
        """Compute cycle interval based on risk profile."""
        base = settings.CYCLE_INTERVAL_SECONDS
        profile = settings.RISK_PROFILE
        multipliers = {
            "conservative": 1.5,   # 5m -> 7.5m
            "balanced":     1.0,   # 5m -> 5m
            "aggressive":   1.0,   # 5m -> 5m
            "fast_profit":  1.0,   # 5m -> 5m
        }
        mult = multipliers.get(profile, 1.0)
        interval = int(base * mult)
        return interval

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

    def _add_sl_cooldown(self, symbol: str, minutes: int = 20) -> None:
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
                "market_regime": self._last_regime,
                "circuit_breaker_tripped": self._circuit_breaker_tripped,
                "less_fear": settings.LESS_FEAR,
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
        """Auto-close positions that have hit their stop-loss, take-profit, or time limit.

        Only manages bot-opened positions — external (pre-existing) positions are skipped.
        """
        triggers = self._portfolio.check_sl_tp_triggers()

        # Filter out external positions — bot should not auto-close those
        triggers = [
            (sym, reason, price) for sym, reason, price in triggers
            if getattr(self._portfolio.get_position(sym), "source", "bot") == "bot"
        ]

        # Time-based exit: close stagnant BOT positions after MAX_HOLD_HOURS
        max_hold = settings.MAX_HOLD_HOURS
        if max_hold > 0:
            now = datetime.now(timezone.utc)
            for pos in self._portfolio.all_positions():
                # Skip external positions — user manages those
                if getattr(pos, "source", "bot") != "bot":
                    continue
                opened = pos.opened_at
                if opened is None:
                    continue
                # Ensure timezone-aware comparison
                if opened.tzinfo is None:
                    opened = opened.replace(tzinfo=timezone.utc)
                age_hours = (now - opened).total_seconds() / 3600
                if age_hours < max_hold:
                    continue
                # Only time-exit if P&L is stagnant (-1% to +1%)
                if -1.0 <= pos.pnl_pct <= 1.0:
                    triggers.append((pos.symbol, "time_exit", pos.current_price))
                    logger.info(
                        "Time-based exit: %s held %.0fh (P&L=%.2f%%) — closing stagnant position",
                        pos.symbol, age_hours, pos.pnl_pct,
                    )

        if not triggers:
            return

        for symbol, reason, price in triggers:
            logger.info("SL/TP triggered: %s on %s @ $%.6f", reason, symbol, price)
            # Add cooldown after stop-loss to prevent revenge trading
            if reason == "stop_loss":
                self._add_sl_cooldown(symbol, minutes=20)
                self._recent_results.append(False)
                self._recent_results = self._recent_results[-10:]
            elif reason == "take_profit":
                self._recent_results.append(True)
                self._recent_results = self._recent_results[-10:]
            elif reason == "time_exit":
                # Time exit: count as neutral (neither win nor loss streak)
                pass
            async with AsyncSessionLocal() as db:
                # Log a synthetic SELL decision so the audit trail is complete
                from app.models.decision import ClaudeDecision

                # Build descriptive label
                if reason == "stop_loss":
                    label = "Stop-loss"
                elif reason == "take_profit":
                    pos = self._portfolio.get_position(symbol)
                    if pos and pos.tp_peak_price > 0 and pos.tp_peak_price > pos.take_profit_price:
                        peak = pos.tp_peak_price
                        extra_pct = (peak - pos.take_profit_price) / pos.take_profit_price * 100
                        label = f"Trailing take-profit (ran +{extra_pct:.1f}% past TP, peak ${peak:.6f})"
                    else:
                        label = "Take-profit"
                else:
                    label = "Time exit"
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
                    # Discord notification for SL/TP
                    await send_trade_notification(
                        action="SELL",
                        symbol=trade.symbol,
                        price=trade.price,
                        quantity=trade.quantity,
                        pnl_usdt=trade.pnl_usdt,
                        pnl_pct=trade.pnl_pct,
                        confidence=1.0,
                        reasoning=f"Automatic {label.lower()} execution.",
                        trigger="SL" if reason == "stop_loss" else "TP",
                    )
                    await self._broadcast(
                        "PORTFOLIO_UPDATE", self._portfolio.get_state().model_dump()
                    )
                    # Exit RL: submit SL/TP close experience for offline learning
                    if trade.pnl_pct is not None:
                        await self._submit_exit_experience(
                            trade.symbol, trade.pnl_pct, action=3
                        )  # action 3 = CLOSE

    async def _broadcast_error(self, message: str) -> None:
        await self._broadcast("ERROR", {"message": message})

    # ------------------------------------------------------------------ #
    # Exit RL helpers
    # ------------------------------------------------------------------ #

    def _build_exit_state(self, pos, ind: dict, btc_anchor: dict) -> list[float]:
        """Build an 18-dimensional state vector for the GPU Exit RL model.

        Dimensions:
          0  pnl_pct        — normalised to [-1, 1]  (maps ±50% → ±1)
          1  hold_hours     — normalised 0–1 (48 h = full)
          2  sl_dist        — price distance to stop-loss as fraction of entry (0–1)
          3  tp_dist        — price distance to take-profit as fraction of entry (0–1)
          4  rsi_14         — normalised 0–1
          5  macd_hist      — soft-clamped via tanh
          6  bb_squeeze     — 0 or 1
          7  volume_ratio   — log-ratio, clamped 0–1
          8  obv_trend      — normalised 0–1
          9  vwap_dist      — soft-clamped via tanh (positive = price above VWAP)
          10 btc_trend      — BTC 1h close return, tanh-normalised
          11 market_regime  — 1=bull, 0.5=sideways, 0=bear
          12 cash_ratio     — cash / portfolio total (0–1)
          13 exposure_frac  — total altcoin exposure fraction (0–1)
          14 n_positions    — open positions / 10, clamped 0–1
          15 win_streak     — recent win-rate from last 10 trades (0–1)
          16 ensemble_conf  — ML ensemble confidence cached on pos (0–1)
          17 anomaly_score  — anomaly score cached on pos (0–1)
        """
        import math
        import numpy as np

        def _safe(v, default: float = 0.0) -> float:
            try:
                f = float(v)
                return f if math.isfinite(f) else default
            except Exception:
                return default

        # Position metrics
        pnl_norm   = float(np.clip(_safe(getattr(pos, "pnl_pct", 0)) / 50.0, -1.0, 1.0))
        hold_h     = 0.0
        opened     = getattr(pos, "opened_at", None)
        if opened is not None:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            hold_h = float(np.clip((now - opened).total_seconds() / 3600.0 / 48.0, 0.0, 1.0))

        ep   = _safe(getattr(pos, "entry_price", 0))
        cur  = _safe(getattr(pos, "current_price", ep))
        sl   = _safe(getattr(pos, "stop_loss", 0))
        tp   = _safe(getattr(pos, "take_profit", 0))
        sl_d = float(np.clip(abs(cur - sl) / max(ep, 1e-9), 0.0, 1.0)) if sl else 0.0
        tp_d = float(np.clip(abs(tp - cur) / max(ep, 1e-9), 0.0, 1.0)) if tp else 0.0

        # Technical indicators for this symbol
        rsi  = float(np.clip(_safe(ind.get("rsi_14", 50)) / 100.0, 0.0, 1.0))
        macd = float(np.tanh(_safe(ind.get("macd_hist", 0)) * 5.0))
        bbsq = float(np.clip(_safe(ind.get("bb_squeeze", 0)), 0.0, 1.0))
        vol  = float(np.clip(math.log1p(max(_safe(ind.get("volume_ratio", 1.0)) - 1.0, 0.0)), 0.0, 1.0))
        obv  = float(np.clip(_safe(ind.get("obv_trend", 0)) * 0.5 + 0.5, 0.0, 1.0))
        vwap = float(np.tanh(_safe(ind.get("vwap_dist_pct", 0)) / 3.0))

        # BTC anchor context
        btc_1h  = btc_anchor.get("1h", {}) if btc_anchor else {}
        btc_ret = float(np.tanh(_safe(btc_1h.get("close_change_pct", 0)) / 3.0))

        # Market regime (from last known regime dict)
        regime_map = {"bull": 1.0, "sideways": 0.5, "bear": 0.0}
        regime_str = self._last_regime.get("regime", "sideways")
        regime_f   = regime_map.get(regime_str, 0.5)

        # Portfolio context
        total = max(_safe(self._portfolio.total_value), 1.0)
        cash  = float(np.clip(_safe(self._portfolio.cash_usdt) / total, 0.0, 1.0))
        exp   = float(np.clip(_safe(getattr(self._portfolio, "total_exposure_pct", 0)) / 100.0, 0.0, 1.0))
        n_pos = float(np.clip(len(self._portfolio.all_positions()) / 10.0, 0.0, 1.0))
        win_r = (sum(self._recent_results) / len(self._recent_results)
                 if self._recent_results else 0.5)

        # Cached ML signals (may be stale from prior cycle — acceptable)
        prior = self._ml_signals_cache.get(pos.symbol, {}) if hasattr(self, "_ml_signals_cache") else {}
        ens_conf  = float(np.clip(_safe(prior.get("ensemble", {}).get("confidence", 0.5)), 0.0, 1.0))
        anom_s    = float(np.clip(_safe(prior.get("anomaly", {}).get("anomaly_score", 0.0)), 0.0, 1.0))

        return [
            pnl_norm, hold_h, sl_d, tp_d,
            rsi, macd, bbsq, vol, obv, vwap,
            btc_ret, regime_f,
            cash, exp, n_pos, win_r,
            ens_conf, anom_s,
        ]

    async def _submit_exit_experience(
        self, symbol: str, pnl_pct: float, action: int
    ) -> None:
        """Send a completed position's outcome to the GPU Exit RL for offline learning.

        action mapping (must match gpu-server ExitDQN):
          0 = HOLD_POS  (not used here — only called on close)
          1 = PARTIAL_25
          2 = PARTIAL_50
          3 = CLOSE
        """
        try:
            from app.services import gpu_client
            if not gpu_client.is_enabled():
                return

            entry_state = self._exit_states.get(symbol)
            if entry_state is None:
                return  # no exit state recorded for this position

            # Terminal next-state: zero vector (position is closed)
            next_state = [0.0] * 18

            # Reward: scaled P&L (the model sees raw return signal)
            reward = float(pnl_pct / 100.0)

            experience = {
                "state":      entry_state,
                "action":     action,
                "reward":     reward,
                "next_state": next_state,
                "done":       True,
            }
            result = await gpu_client.train_exit([experience])
            if result:
                logger.debug("Exit RL experience submitted for %s (pnl=%.2f%%): %s",
                             symbol, pnl_pct, result)

            # Clean up stale exit state
            self._exit_states.pop(symbol, None)

        except Exception as exc:
            logger.debug("Exit RL experience submission failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------ #
    # Pyramiding — add to winning positions
    # ------------------------------------------------------------------ #
    async def _check_pyramiding(
        self, db: AsyncSession, symbols_data: dict, news_data: dict,
        ml_signals: dict | None, btc_anchor: dict | None, market_intel: dict | None,
    ) -> None:
        """Add to profitable positions if quant score remains high."""
        from app.services.quant_scorer import score_symbol

        min_pnl = settings.PYRAMID_MIN_PNL_PCT
        min_score = settings.PYRAMID_MIN_SCORE
        add_pct = settings.PYRAMID_ADD_PCT / 100.0

        intel = market_intel or {}
        funding_map = intel.get("funding", {})
        ls_map = intel.get("long_short", {})
        oi_map = intel.get("open_interest", {})

        for pos in self._portfolio.all_positions():
            if pos.pnl_pct < min_pnl:
                continue  # only pyramid into winners
            sym = pos.symbol
            if sym not in symbols_data:
                continue

            result = score_symbol(
                sym, symbols_data[sym],
                news_data.get(sym, {}),
                ml_signal=(ml_signals or {}).get(sym),
                btc_anchor=btc_anchor,
                is_held=True,
                funding_data=funding_map.get(sym),
                ls_data=ls_map.get(sym),
                oi_data=oi_map.get(sym),
            )

            if result["score"] < min_score or result["direction"] <= 0:
                continue

            # Pyramid: add add_pct of original position value
            add_usdt = pos.quantity * pos.avg_entry_price * add_pct
            if add_usdt < 5:
                continue
            if add_usdt > self._portfolio.cash_usdt * 0.5:
                continue  # don't use more than 50% of cash for pyramid

            # Check exposure limit
            new_exposure_pct = (
                (self._portfolio.positions_value + add_usdt) /
                max(self._portfolio.total_value, 1) * 100
            )
            if new_exposure_pct >= settings.MAX_TOTAL_EXPOSURE_PCT:
                continue

            price = pos.current_price
            if price <= 0:
                continue

            from app.services.executor import _apply_slippage, _FEE_RATE
            fill_price = _apply_slippage(price, "BUY")
            fee = add_usdt * _FEE_RATE
            qty = (add_usdt - fee) / fill_price

            await self._portfolio.open_position(
                db, sym, qty, fill_price,
                pos.stop_loss_price, pos.take_profit_price,
                fee_usdt=fee,
            )
            logger.info(
                "[PYRAMID] %s: added $%.2f (%.6f qty) @ $%.6f (P&L=+%.1f%%, score=%.0f)",
                sym, add_usdt, qty, fill_price, pos.pnl_pct, result["score"],
            )
            await self._broadcast("TRADE_EXECUTED", {
                "symbol": sym,
                "direction": "BUY",
                "mode": "demo" if settings.is_demo else "real",
                "quantity": qty,
                "price": fill_price,
                "trigger": "pyramid",
            })

    # ------------------------------------------------------------------ #
    # Kelly criterion — compute from backtest results
    # ------------------------------------------------------------------ #
    def update_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> None:
        """Compute Kelly fraction from backtest results.

        f* = (win_rate * avg_win - (1 - win_rate) * |avg_loss|) / avg_win
        Use half-Kelly for safety (capped by KELLY_FRACTION_CAP).
        """
        if avg_win <= 0 or win_rate <= 0:
            self._kelly_fraction = 1.0
            return

        abs_loss = abs(avg_loss) if avg_loss != 0 else avg_win
        kelly = (win_rate * avg_win - (1 - win_rate) * abs_loss) / avg_win

        # Half-Kelly for safety
        kelly *= 0.5

        # Clamp to [0.1, 1.0] — never go below 10% or above 100%
        self._kelly_fraction = max(0.1, min(1.0, kelly))
        logger.info(
            "Kelly fraction updated: %.2f (win_rate=%.1f%%, avg_win=%.1f%%, avg_loss=%.1f%%)",
            self._kelly_fraction, win_rate * 100, avg_win * 100, avg_loss * 100,
        )

    # ------------------------------------------------------------------ #
    # Auto-backtest integration
    # ------------------------------------------------------------------ #
    async def _run_startup_backtest(self) -> None:
        """Run backtest on startup, then schedule periodic re-runs."""
        from app.services.auto_tuner import run_auto_backtest, backtest_scheduler

        # Small delay to let other services initialize
        await asyncio.sleep(5)

        logger.info("Running startup auto-backtest...")
        result, applied = await run_auto_backtest(
            days=settings.BACKTEST_DAYS,
            n_symbols=settings.BACKTEST_SYMBOLS,
        )

        if result:
            await self._broadcast("BACKTEST_COMPLETE", {
                "total_trades": result.total_trades,
                "win_rate_pct": result.win_rate_pct,
                "total_return_pct": result.total_return_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "tuning_applied": applied,
            })

        # Schedule periodic re-runs
        if settings.BACKTEST_INTERVAL_HOURS > 0:
            logger.info(
                "Scheduling auto-backtest every %.0fh",
                settings.BACKTEST_INTERVAL_HOURS,
            )
            # Wait for the interval before first scheduled run (startup run just completed)
            await asyncio.sleep(settings.BACKTEST_INTERVAL_HOURS * 3600)
            await backtest_scheduler(settings.BACKTEST_INTERVAL_HOURS)


bot_runner = BotRunner()
