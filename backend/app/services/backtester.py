"""Backtesting framework for the quant scorer.

Fetches historical OHLCV from Binance via CCXT, replays candles through
the quant scoring engine, simulates trades with ATR-based SL/TP, and
reports win rate, P&L, max drawdown, Sharpe ratio, and per-factor analysis.

Usage (from project root):
    python -m backend.app.services.backtester --days 90 --symbols 10
    # or via the CLI wrapper:
    python backtest.py --days 90 --symbols 10

Requires BINANCE_API_KEY / BINANCE_SECRET in .env (read-only access is fine).
"""

import asyncio
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# Add backend/ to path so we can import app.*
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import ccxt.async_support as ccxt

from app.config import settings
from app.services.technical import compute_indicators, detect_support_resistance
from app.services.quant_scorer import (
    score_symbol,
    compute_trade_levels,
    TradeCandidate,
)

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

FEE_PCT = 0.10  # Binance spot taker fee (0.1%)
SLIPPAGE_PCT = 0.05  # Simulated slippage per trade
MIN_CANDLES_REQUIRED = 50  # Need at least this many 1h candles before first trade
BTC_SYMBOL = f"BTC/{settings.QUOTE_CURRENCY}"

# Candle index constants (CCXT format)
TS, OPEN, HIGH, LOW, CLOSE, VOL = 0, 1, 2, 3, 4, 5


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """Record of a simulated trade."""
    symbol: str
    action: str             # BUY or SELL (close)
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    stop_loss_price: float
    take_profit_price: float
    stop_loss_pct: float
    take_profit_pct: float
    quantity_pct: float
    quant_score: float
    reward_risk_ratio: float
    exit_reason: str        # "tp_hit", "sl_hit", "signal_exit", "end_of_data"
    pnl_pct: float
    pnl_usdt: float
    signals: list[str] = field(default_factory=list)
    factor_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class OpenPosition:
    """A position being held during the backtest."""
    symbol: str
    entry_price: float
    entry_time: datetime
    stop_loss_price: float
    take_profit_price: float
    stop_loss_pct: float
    take_profit_pct: float
    quantity_pct: float
    quantity_usdt: float
    quant_score: float
    reward_risk_ratio: float
    signals: list[str]
    factor_scores: dict[str, float]


@dataclass
class BacktestResult:
    """Aggregate results of a backtest run."""
    # Run params
    symbols: list[str]
    period_days: int
    start_date: str
    end_date: str
    initial_balance: float

    # Performance
    final_balance: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float

    # Risk metrics
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float   # gross wins / gross losses
    avg_win_pct: float
    avg_loss_pct: float
    avg_rr_achieved: float  # actual R:R on closed trades

    # Score analysis
    avg_winning_score: float
    avg_losing_score: float
    score_buckets: dict[str, dict]  # "60-70", "70-80", "80-90", "90-100" → stats

    # Factor analysis
    factor_win_avg: dict[str, float]  # avg factor value for winning trades
    factor_loss_avg: dict[str, float]  # avg factor value for losing trades

    # Time analysis
    avg_hold_hours: float
    trades: list[BacktestTrade]
    equity_curve: list[tuple[str, float]]  # (ISO date, balance)

    # Skipped cycles
    total_cycles: int
    cycles_with_candidates: int
    cycles_skipped: int


# ── Historical data fetcher ──────────────────────────────────────────────────

async def fetch_historical_ohlcv(
    symbol: str,
    timeframe: str,
    days: int,
    exchange: ccxt.Exchange,
) -> list[list]:
    """Fetch historical OHLCV from Binance in batches (max 1000 per request)."""
    tf_ms = {
        "15m": 15 * 60 * 1000,
        "1h":  60 * 60 * 1000,
        "4h":  4 * 60 * 60 * 1000,
        "1d":  24 * 60 * 60 * 1000,
    }
    interval_ms = tf_ms.get(timeframe, 3600_000)
    total_candles = (days * 24 * 3600 * 1000) // interval_ms
    batch_size = 1000

    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_candles: list[list] = []

    while len(all_candles) < total_candles:
        try:
            batch = await exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=batch_size,
            )
        except Exception as exc:
            logger.warning("Failed to fetch %s %s: %s", symbol, timeframe, exc)
            break

        if not batch:
            break

        all_candles.extend(batch)
        since = batch[-1][TS] + interval_ms

        # Rate limit courtesy
        await asyncio.sleep(0.1)

        if len(batch) < batch_size:
            break  # no more data available

    return all_candles


async def fetch_all_data(
    symbols: list[str],
    days: int,
    timeframes: list[str] = None,
) -> dict[str, dict[str, list[list]]]:
    """Fetch multi-timeframe OHLCV for all symbols + BTC.

    Returns {symbol: {timeframe: [candles]}}
    """
    if timeframes is None:
        timeframes = ["1h", "4h", "1d"]  # skip 15m for speed in backtest

    exchange = ccxt.binance({
        "apiKey": settings.BINANCE_API_KEY or None,
        "secret": settings.BINANCE_SECRET or None,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    all_symbols = list(set(symbols + [BTC_SYMBOL]))
    result: dict[str, dict[str, list]] = {}

    try:
        total = len(all_symbols) * len(timeframes)
        done = 0
        for sym in all_symbols:
            result[sym] = {}
            for tf in timeframes:
                candles = await fetch_historical_ohlcv(sym, tf, days, exchange)
                result[sym][tf] = candles
                done += 1
                logger.info(
                    "Fetched %s %s: %d candles (%d/%d)",
                    sym, tf, len(candles), done, total,
                )
        return result
    finally:
        await exchange.close()


async def get_top_symbols_for_backtest(n: int = 10) -> list[str]:
    """Get top N Binance USDT pairs by volume for backtesting."""
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    try:
        tickers = await exchange.fetch_tickers()
        usdt_pairs = []
        for sym, t in tickers.items():
            if not sym.endswith(f"/{settings.QUOTE_CURRENCY}"):
                continue
            if sym == BTC_SYMBOL:
                continue  # BTC is used as anchor, not traded
            base = sym.split("/")[0]
            if base in ("USDC", "BUSD", "TUSD", "FDUSD", "DAI", "EUR"):
                continue  # skip stablecoins
            vol = t.get("quoteVolume") or 0
            if vol >= settings.MIN_VOLUME_USDT:
                usdt_pairs.append((sym, vol))
        usdt_pairs.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in usdt_pairs[:n]]
    finally:
        await exchange.close()


# ── Synthetic orderbook from candles ─────────────────────────────────────────

def _synthetic_orderbook(candles: list[list], idx: int) -> dict:
    """Create a synthetic orderbook from candle data.

    We don't have real orderbook history, so approximate from
    candle structure (high-low spread, volume).
    """
    if idx < 1:
        return {"spread_pct": 0.1, "pressure_ratio": 1.0}

    c = candles[idx]
    price = c[CLOSE]
    hl_spread = (c[HIGH] - c[LOW]) / max(price, 1e-10) * 100
    # If close is in upper half of range → buyer pressure
    mid = (c[HIGH] + c[LOW]) / 2
    pressure = 1.0 + (c[CLOSE] - mid) / max(c[HIGH] - c[LOW], 1e-10) * 0.5

    return {
        "spread_pct": max(0.01, hl_spread * 0.1),
        "pressure_ratio": max(0.3, min(3.0, pressure)),
    }


# ── Backtester engine ───────────────────────────────────────────────────────

def _build_symbol_data_at(
    all_candles: dict[str, list[list]],
    tf_indices: dict[str, int],
    lookback: int = 100,
) -> dict[str, Any] | None:
    """Build symbol_data dict for a single symbol at a point in time.

    Uses candle data up to (but not including future) the current index.
    """
    candles_1h = all_candles.get("1h", [])
    idx_1h = tf_indices.get("1h", 0)
    if idx_1h < MIN_CANDLES_REQUIRED:
        return None

    indicators: dict[str, dict] = {}
    for tf, candles in all_candles.items():
        idx = tf_indices.get(tf, 0)
        if idx < 20:
            continue
        window = candles[max(0, idx - lookback):idx + 1]
        if len(window) >= 20:
            indicators[tf] = compute_indicators(window)

    if "1h" not in indicators:
        return None

    price = candles_1h[idx_1h][CLOSE]

    # S/R from 4h candles
    sr_candles = all_candles.get("4h", all_candles.get("1h", []))
    sr_idx = tf_indices.get("4h", tf_indices.get("1h", 0))
    sr_window = sr_candles[max(0, sr_idx - lookback):sr_idx + 1]
    sr_levels = detect_support_resistance(sr_window) if len(sr_window) >= 20 else {}

    ob = _synthetic_orderbook(candles_1h, idx_1h)

    return {
        "price": price,
        "indicators": indicators,
        "orderbook": ob,
        "support_resistance": sr_levels,
    }


def _check_sl_tp_intracandle(
    pos: OpenPosition,
    candle: list,
) -> tuple[str, float] | None:
    """Check if SL or TP was hit within a candle's high/low range.

    Returns (reason, exit_price) or None.
    For BUY positions: SL if low <= SL price, TP if high >= TP price.
    If both hit in same candle, SL takes priority (conservative).
    """
    low = candle[LOW]
    high = candle[HIGH]

    sl_hit = low <= pos.stop_loss_price
    tp_hit = high >= pos.take_profit_price

    if sl_hit and tp_hit:
        # Both hit — assume SL hit first (conservative)
        return "sl_hit", pos.stop_loss_price
    if sl_hit:
        return "sl_hit", pos.stop_loss_price
    if tp_hit:
        return "tp_hit", pos.take_profit_price
    return None


async def run_backtest(
    symbols: list[str],
    days: int = 90,
    initial_balance: float = 10_000.0,
    min_score: float | None = None,
    min_rr: float | None = None,
    max_positions: int = 5,
    timeframes: list[str] | None = None,
) -> BacktestResult:
    """Run a full backtest of the quant scorer strategy.

    Replays 1h candles chronologically. At each candle:
    1. Check SL/TP on open positions
    2. Score all symbols
    3. Enter new positions if candidates exist
    """
    if min_score is not None:
        settings.MIN_QUANT_SCORE = min_score
    if min_rr is not None:
        settings.MIN_REWARD_RISK_RATIO = min_rr

    print(f"\n{'='*60}")
    print(f"  QUANT SCORER BACKTEST")
    print(f"  Symbols: {len(symbols)} | Period: {days}d | Balance: ${initial_balance:,.0f}")
    print(f"  MIN_SCORE={settings.MIN_QUANT_SCORE} MIN_R:R={settings.MIN_REWARD_RISK_RATIO}")
    print(f"  SL_ATR_MULT={settings.SL_ATR_MULTIPLIER} SL_RANGE=[{settings.MIN_SL_PCT}%, {settings.MAX_SL_PCT}%]")
    print(f"{'='*60}\n")

    # 1. Fetch data
    print("Fetching historical data...")
    t0 = time.time()
    all_data = await fetch_all_data(symbols, days, timeframes)
    elapsed = time.time() - t0
    print(f"Data fetched in {elapsed:.1f}s\n")

    # Verify we got data
    for sym in list(symbols):
        candles_1h = all_data.get(sym, {}).get("1h", [])
        if len(candles_1h) < MIN_CANDLES_REQUIRED:
            print(f"  WARNING: {sym} only has {len(candles_1h)} 1h candles (need {MIN_CANDLES_REQUIRED}), removing")
            symbols.remove(sym)
    if not symbols:
        raise RuntimeError("No symbols have enough data for backtesting")

    # BTC anchor data
    btc_candles = all_data.get(BTC_SYMBOL, {})

    # 2. Build aligned 1h timeline
    # Find the common start/end across all symbols
    start_ts = 0
    end_ts = float("inf")
    for sym in symbols:
        c1h = all_data[sym]["1h"]
        if c1h:
            start_ts = max(start_ts, c1h[MIN_CANDLES_REQUIRED][TS])
            end_ts = min(end_ts, c1h[-1][TS])

    if start_ts >= end_ts:
        raise RuntimeError("No overlapping time range across symbols")

    total_1h_candles = all_data[symbols[0]]["1h"]
    timeline_indices = [
        i for i, c in enumerate(total_1h_candles)
        if start_ts <= c[TS] <= end_ts
    ]

    print(f"Simulation: {len(timeline_indices)} hourly candles")
    start_dt = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc)
    print(f"  From: {start_dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"  To:   {end_dt.strftime('%Y-%m-%d %H:%M')}\n")

    # 3. Simulate
    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0.0
    positions: dict[str, OpenPosition] = {}  # symbol -> position
    trades: list[BacktestTrade] = []
    equity_curve: list[tuple[str, float]] = []
    daily_returns: list[float] = []
    prev_equity = initial_balance
    total_cycles = 0
    cycles_with_candidates = 0
    cycles_skipped = 0

    # Build per-symbol timestamp indices for O(1) lookup
    # Map each symbol's candles into a dict of {tf: {ts: index}}
    sym_ts_map: dict[str, dict[str, dict[int, int]]] = {}
    for sym in symbols + [BTC_SYMBOL]:
        sym_ts_map[sym] = {}
        for tf, candles in all_data.get(sym, {}).items():
            sym_ts_map[sym][tf] = {c[TS]: i for i, c in enumerate(candles)}

    # Helper: find the latest index in a timeframe at or before a given 1h timestamp
    def _tf_index_at(sym: str, tf: str, ts_1h: int) -> int:
        ts_map = sym_ts_map.get(sym, {}).get(tf, {})
        candles = all_data.get(sym, {}).get(tf, [])
        if not candles:
            return -1
        # Find largest candle ts <= ts_1h
        best_idx = -1
        for c_ts, idx in ts_map.items():
            if c_ts <= ts_1h:
                if idx > best_idx:
                    best_idx = idx
        return best_idx

    sim_start = time.time()
    for step, hourly_idx in enumerate(timeline_indices):
        candle_1h = all_data[symbols[0]]["1h"][hourly_idx]
        current_ts = candle_1h[TS]
        current_dt = datetime.fromtimestamp(current_ts / 1000, tz=timezone.utc)

        # Progress indicator
        if step % 200 == 0:
            pct = step / max(len(timeline_indices), 1) * 100
            pos_value = sum(
                p.quantity_usdt * (all_data[p.symbol]["1h"][
                    min(hourly_idx, len(all_data[p.symbol]["1h"]) - 1)
                ][CLOSE] / p.entry_price)
                for p in positions.values()
                if hourly_idx < len(all_data[p.symbol]["1h"])
            )
            total_val = balance + pos_value
            print(f"  [{pct:5.1f}%] {current_dt.strftime('%Y-%m-%d %H:%M')} | "
                  f"balance=${balance:,.2f} pos={len(positions)} total=${total_val:,.2f}")

        # 3a. Check SL/TP on open positions
        symbols_to_close = []
        for sym, pos in positions.items():
            sym_candles = all_data.get(sym, {}).get("1h", [])
            if hourly_idx >= len(sym_candles):
                continue
            candle = sym_candles[hourly_idx]
            result = _check_sl_tp_intracandle(pos, candle)
            if result:
                reason, exit_price = result
                # Apply fee + slippage
                exit_price_after = exit_price * (1 - (FEE_PCT + SLIPPAGE_PCT) / 100)
                pnl_pct = (exit_price_after - pos.entry_price) / pos.entry_price * 100
                pnl_usdt = pos.quantity_usdt * pnl_pct / 100

                trades.append(BacktestTrade(
                    symbol=sym,
                    action="SELL",
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    entry_time=pos.entry_time,
                    exit_time=current_dt,
                    stop_loss_price=pos.stop_loss_price,
                    take_profit_price=pos.take_profit_price,
                    stop_loss_pct=pos.stop_loss_pct,
                    take_profit_pct=pos.take_profit_pct,
                    quantity_pct=pos.quantity_pct,
                    quant_score=pos.quant_score,
                    reward_risk_ratio=pos.reward_risk_ratio,
                    exit_reason=reason,
                    pnl_pct=round(pnl_pct, 2),
                    pnl_usdt=round(pnl_usdt, 2),
                    signals=pos.signals,
                    factor_scores=pos.factor_scores,
                ))
                balance += pos.quantity_usdt + pnl_usdt
                symbols_to_close.append(sym)

        for sym in symbols_to_close:
            del positions[sym]

        # 3b. Equity tracking
        pos_value = sum(
            p.quantity_usdt * (all_data[p.symbol]["1h"][
                min(hourly_idx, len(all_data[p.symbol]["1h"]) - 1)
            ][CLOSE] / p.entry_price)
            for p in positions.values()
            if hourly_idx < len(all_data[p.symbol]["1h"])
        )
        total_equity = balance + pos_value
        peak_balance = max(peak_balance, total_equity)
        dd = (peak_balance - total_equity) / peak_balance * 100 if peak_balance > 0 else 0
        max_drawdown = max(max_drawdown, dd)

        # Daily equity curve (every 24 candles)
        if step % 24 == 0:
            equity_curve.append((current_dt.isoformat(), round(total_equity, 2)))
            daily_returns.append((total_equity - prev_equity) / max(prev_equity, 1))
            prev_equity = total_equity

        # 3c. Score symbols (every 4 candles to save compute — 4h frequency)
        if step % 4 != 0:
            continue

        total_cycles += 1

        if len(positions) >= max_positions:
            cycles_skipped += 1
            continue

        # Build BTC anchor
        btc_anchor: dict = {}
        for tf in ("1h", "4h", "1d"):
            btc_c = btc_candles.get(tf, [])
            btc_idx = _tf_index_at(BTC_SYMBOL, tf, current_ts)
            if btc_idx >= 20:
                window = btc_c[max(0, btc_idx - 100):btc_idx + 1]
                if len(window) >= 20:
                    btc_anchor[tf] = compute_indicators(window)

        # Score each symbol
        candidates: list[TradeCandidate] = []
        for sym in symbols:
            if sym in positions:
                continue  # already holding

            # Get current indices for all timeframes
            tf_indices: dict[str, int] = {}
            for tf in all_data.get(sym, {}):
                idx = _tf_index_at(sym, tf, current_ts)
                if idx >= 0:
                    tf_indices[tf] = idx

            sym_data = _build_symbol_data_at(all_data[sym], tf_indices)
            if sym_data is None:
                continue

            result = score_symbol(
                sym, sym_data, {},  # No news in backtest
                ml_signal=None,     # No ML in backtest
                btc_anchor=btc_anchor,
            )

            score = result["score"]
            direction = result["direction"]

            if direction > 0 and score >= settings.MIN_QUANT_SCORE:
                levels = compute_trade_levels(sym, sym_data, "BUY", score)
                if levels:
                    price = sym_data["price"]
                    # Apply entry fee + slippage
                    entry_price = price * (1 + (FEE_PCT + SLIPPAGE_PCT) / 100)
                    candidates.append(TradeCandidate(
                        symbol=sym,
                        action="BUY",
                        score=score,
                        timeframe="1h",
                        entry_price=entry_price,
                        stop_loss_price=round(entry_price * (1 - levels["sl_pct"] / 100), 6),
                        take_profit_price=round(entry_price * (1 + levels["tp_pct"] / 100), 6),
                        stop_loss_pct=levels["sl_pct"],
                        take_profit_pct=levels["tp_pct"],
                        reward_risk_ratio=levels["rr_ratio"],
                        quantity_pct=levels["quantity_pct"],
                        signals=result["signals"],
                        factor_scores=result["factors"],
                    ))

        if candidates:
            cycles_with_candidates += 1
            # Take the best candidate
            candidates.sort(key=lambda c: c.score, reverse=True)
            best = candidates[0]

            # Size the position
            alloc_usdt = balance * best.quantity_pct / 100
            if alloc_usdt < 10:  # min trade size
                cycles_skipped += 1
                continue

            positions[best.symbol] = OpenPosition(
                symbol=best.symbol,
                entry_price=best.entry_price,
                entry_time=current_dt,
                stop_loss_price=best.stop_loss_price,
                take_profit_price=best.take_profit_price,
                stop_loss_pct=best.stop_loss_pct,
                take_profit_pct=best.take_profit_pct,
                quantity_pct=best.quantity_pct,
                quantity_usdt=alloc_usdt,
                quant_score=best.score,
                reward_risk_ratio=best.reward_risk_ratio,
                signals=best.signals,
                factor_scores=best.factor_scores,
            )
            balance -= alloc_usdt
        else:
            cycles_skipped += 1

    # 4. Close any remaining positions at last price
    for sym, pos in list(positions.items()):
        last_candle = all_data[sym]["1h"][-1]
        exit_price = last_candle[CLOSE] * (1 - (FEE_PCT + SLIPPAGE_PCT) / 100)
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        pnl_usdt = pos.quantity_usdt * pnl_pct / 100
        trades.append(BacktestTrade(
            symbol=sym, action="SELL",
            entry_price=pos.entry_price, exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=datetime.fromtimestamp(last_candle[TS] / 1000, tz=timezone.utc),
            stop_loss_price=pos.stop_loss_price, take_profit_price=pos.take_profit_price,
            stop_loss_pct=pos.stop_loss_pct, take_profit_pct=pos.take_profit_pct,
            quantity_pct=pos.quantity_pct, quant_score=pos.quant_score,
            reward_risk_ratio=pos.reward_risk_ratio, exit_reason="end_of_data",
            pnl_pct=round(pnl_pct, 2), pnl_usdt=round(pnl_usdt, 2),
            signals=pos.signals, factor_scores=pos.factor_scores,
        ))
        balance += pos.quantity_usdt + pnl_usdt
    positions.clear()

    sim_elapsed = time.time() - sim_start
    print(f"\nSimulation done in {sim_elapsed:.1f}s")

    # 5. Compute statistics
    return _compute_results(
        symbols=symbols,
        days=days,
        start_date=start_dt.isoformat(),
        end_date=end_dt.isoformat(),
        initial_balance=initial_balance,
        final_balance=balance,
        trades=trades,
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        max_drawdown=max_drawdown,
        total_cycles=total_cycles,
        cycles_with_candidates=cycles_with_candidates,
        cycles_skipped=cycles_skipped,
    )


# ── Results computation ──────────────────────────────────────────────────────

def _compute_results(
    symbols: list[str],
    days: int,
    start_date: str,
    end_date: str,
    initial_balance: float,
    final_balance: float,
    trades: list[BacktestTrade],
    equity_curve: list[tuple[str, float]],
    daily_returns: list[float],
    max_drawdown: float,
    total_cycles: int,
    cycles_with_candidates: int,
    cycles_skipped: int,
) -> BacktestResult:
    """Compute all statistics from raw trade data."""
    total_return = (final_balance - initial_balance) / initial_balance * 100

    winners = [t for t in trades if t.pnl_pct > 0]
    losers = [t for t in trades if t.pnl_pct <= 0]
    total = len(trades)
    win_rate = len(winners) / max(total, 1) * 100

    avg_win = sum(t.pnl_pct for t in winners) / max(len(winners), 1)
    avg_loss = sum(t.pnl_pct for t in losers) / max(len(losers), 1)

    gross_wins = sum(t.pnl_usdt for t in winners)
    gross_losses = abs(sum(t.pnl_usdt for t in losers))
    profit_factor = gross_wins / max(gross_losses, 0.01)

    # Sharpe ratio (annualized, from daily returns)
    if len(daily_returns) > 1:
        mean_r = sum(daily_returns) / len(daily_returns)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1))
        sharpe = (mean_r / max(std_r, 1e-10)) * math.sqrt(365) if std_r > 0 else 0
    else:
        sharpe = 0.0

    # Average actual R:R
    actual_rrs = []
    for t in trades:
        if t.exit_reason == "tp_hit" and t.stop_loss_pct > 0:
            actual_rrs.append(t.pnl_pct / t.stop_loss_pct)
        elif t.exit_reason == "sl_hit" and t.stop_loss_pct > 0:
            actual_rrs.append(-1.0)
    avg_rr = sum(actual_rrs) / max(len(actual_rrs), 1)

    # Score buckets
    buckets = {"60-70": [], "70-80": [], "80-90": [], "90-100": []}
    for t in trades:
        s = t.quant_score
        if s < 70:
            buckets["60-70"].append(t)
        elif s < 80:
            buckets["70-80"].append(t)
        elif s < 90:
            buckets["80-90"].append(t)
        else:
            buckets["90-100"].append(t)

    score_bucket_stats = {}
    for label, bucket_trades in buckets.items():
        if bucket_trades:
            bw = [t for t in bucket_trades if t.pnl_pct > 0]
            score_bucket_stats[label] = {
                "trades": len(bucket_trades),
                "win_rate": len(bw) / len(bucket_trades) * 100,
                "avg_pnl": sum(t.pnl_pct for t in bucket_trades) / len(bucket_trades),
                "total_pnl_usdt": sum(t.pnl_usdt for t in bucket_trades),
            }
        else:
            score_bucket_stats[label] = {
                "trades": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl_usdt": 0,
            }

    # Factor analysis — all 26 signals tracked
    factor_keys = [
        "rsi_signal", "macd_signal", "macd_div_signal", "bb_signal",
        "bb_squeeze_signal", "volume_signal", "vol_zscore_signal",
        "obv_signal", "vwap_signal", "trend_signal", "sr_signal",
        "breakout_signal", "momentum_accel_signal", "btc_signal",
        "ml_signal", "orderbook_signal", "depth_signal", "whale_signal",
        "funding_signal", "ls_ratio_signal", "oi_signal",
        "gpu_momentum_signal", "sector_rotation_signal",
        "squeeze_signal", "beta_signal", "news_burst_signal",
    ]
    factor_win: dict[str, list[float]] = {k: [] for k in factor_keys}
    factor_loss: dict[str, list[float]] = {k: [] for k in factor_keys}
    for t in trades:
        target = factor_win if t.pnl_pct > 0 else factor_loss
        for k in factor_keys:
            if k in t.factor_scores:
                target[k].append(t.factor_scores[k])

    factor_win_avg = {
        k: sum(v) / max(len(v), 1) for k, v in factor_win.items()
    }
    factor_loss_avg = {
        k: sum(v) / max(len(v), 1) for k, v in factor_loss.items()
    }

    # Average hold time
    hold_hours = []
    for t in trades:
        diff = (t.exit_time - t.entry_time).total_seconds() / 3600
        hold_hours.append(diff)
    avg_hold = sum(hold_hours) / max(len(hold_hours), 1)

    # Avg winning/losing score
    avg_winning_score = sum(t.quant_score for t in winners) / max(len(winners), 1)
    avg_losing_score = sum(t.quant_score for t in losers) / max(len(losers), 1)

    return BacktestResult(
        symbols=symbols,
        period_days=days,
        start_date=start_date,
        end_date=end_date,
        initial_balance=initial_balance,
        final_balance=round(final_balance, 2),
        total_return_pct=round(total_return, 2),
        total_trades=total,
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate_pct=round(win_rate, 1),
        max_drawdown_pct=round(max_drawdown, 2),
        sharpe_ratio=round(sharpe, 2),
        profit_factor=round(profit_factor, 2),
        avg_win_pct=round(avg_win, 2),
        avg_loss_pct=round(avg_loss, 2),
        avg_rr_achieved=round(avg_rr, 2),
        avg_winning_score=round(avg_winning_score, 1),
        avg_losing_score=round(avg_losing_score, 1),
        score_buckets=score_bucket_stats,
        factor_win_avg={k: round(v, 3) for k, v in factor_win_avg.items()},
        factor_loss_avg={k: round(v, 3) for k, v in factor_loss_avg.items()},
        avg_hold_hours=round(avg_hold, 1),
        trades=trades,
        equity_curve=equity_curve,
        total_cycles=total_cycles,
        cycles_with_candidates=cycles_with_candidates,
        cycles_skipped=cycles_skipped,
    )


# ── Factor predictiveness analysis ──────────────────────────────────────────

def analyse_factor_predictiveness(result: "BacktestResult") -> dict:
    """Analyse which factors best predict trade profitability.

    Computes, for every factor tracked in BacktestTrade.factor_scores:
    - win_avg    : mean value in winning trades
    - loss_avg   : mean value in losing trades
    - predictive_delta : win_avg - loss_avg (positive = bullish factor works)
    - recommended_nudge: ±nudge_step to apply to quant_scorer weights

    Also calls `quant_scorer.nudge_weights_from_backtest()` to update the
    live weights so subsequent cycles immediately benefit.

    Returns a summary dict with factor-level analysis.
    """
    from app.services.quant_scorer import nudge_weights_from_backtest

    if not result.trades:
        return {"status": "no_trades", "deltas": {}}

    # Collect ALL factor keys present in any trade
    all_keys: set[str] = set()
    for t in result.trades:
        all_keys.update(t.factor_scores.keys())

    winners = [t for t in result.trades if t.pnl_pct > 0]
    losers  = [t for t in result.trades if t.pnl_pct <= 0]

    def _avg(trades: list, key: str) -> float:
        vals = [t.factor_scores[key] for t in trades if key in t.factor_scores]
        return sum(vals) / len(vals) if vals else 0.0

    factor_win_avg  = {k: _avg(winners, k) for k in all_keys}
    factor_loss_avg = {k: _avg(losers,  k) for k in all_keys}

    predictive_deltas = {
        k: round(factor_win_avg[k] - factor_loss_avg[k], 4)
        for k in all_keys
    }

    # Sort by absolute predictive power
    ranked = sorted(predictive_deltas.items(), key=lambda x: abs(x[1]), reverse=True)

    # Apply live weight nudging
    nudge_weights_from_backtest(factor_win_avg, factor_loss_avg, nudge_step=0.005)

    logger.info(
        "Factor predictiveness (top 5): %s",
        ", ".join(f"{k}={v:+.4f}" for k, v in ranked[:5]),
    )

    return {
        "status":             "ok",
        "total_trades":       len(result.trades),
        "winning_trades":     len(winners),
        "losing_trades":      len(losers),
        "factor_win_avg":     {k: round(v, 4) for k, v in factor_win_avg.items()},
        "factor_loss_avg":    {k: round(v, 4) for k, v in factor_loss_avg.items()},
        "predictive_deltas":  predictive_deltas,
        "top_predictors":     [{"factor": k, "delta": v} for k, v in ranked[:10]],
        "weights_nudged":     True,
    }


# ── Report printer ───────────────────────────────────────────────────────────

def print_report(r: BacktestResult) -> None:
    """Print a comprehensive backtest report to stdout."""
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Period: {r.start_date[:10]} to {r.end_date[:10]} ({r.period_days}d)")
    print(f"  Symbols: {', '.join(r.symbols[:5])}{'...' if len(r.symbols) > 5 else ''}")
    print(f"  Config: MIN_SCORE={settings.MIN_QUANT_SCORE} MIN_R:R={settings.MIN_REWARD_RISK_RATIO}")

    print(f"\n{'─'*60}")
    print(f"  PERFORMANCE")
    print(f"{'─'*60}")
    ret_emoji = "✅" if r.total_return_pct > 0 else "❌"
    print(f"  {ret_emoji} Total Return: {r.total_return_pct:+.2f}%")
    print(f"     ${r.initial_balance:,.2f} → ${r.final_balance:,.2f}")
    print(f"  📊 Total Trades: {r.total_trades}")
    print(f"     Winners: {r.winning_trades} | Losers: {r.losing_trades}")
    wr_emoji = "✅" if r.win_rate_pct >= 35 else "⚠️"
    print(f"  {wr_emoji} Win Rate: {r.win_rate_pct:.1f}%")
    print(f"     (need ≥33% with 2:1 R:R to profit)")

    print(f"\n{'─'*60}")
    print(f"  RISK METRICS")
    print(f"{'─'*60}")
    dd_emoji = "✅" if r.max_drawdown_pct < 15 else "⚠️"
    print(f"  {dd_emoji} Max Drawdown: {r.max_drawdown_pct:.2f}%")
    sr_emoji = "✅" if r.sharpe_ratio > 0.5 else "⚠️"
    print(f"  {sr_emoji} Sharpe Ratio: {r.sharpe_ratio:.2f}")
    pf_emoji = "✅" if r.profit_factor > 1 else "❌"
    print(f"  {pf_emoji} Profit Factor: {r.profit_factor:.2f}")
    print(f"     (>1 = profitable, >2 = strong)")
    print(f"  📈 Avg Win:  {r.avg_win_pct:+.2f}%")
    print(f"  📉 Avg Loss: {r.avg_loss_pct:+.2f}%")
    print(f"  ⚖️  Avg R:R Achieved: {r.avg_rr_achieved:.2f}")
    print(f"  🕐 Avg Hold Time: {r.avg_hold_hours:.1f}h")

    print(f"\n{'─'*60}")
    print(f"  CYCLE EFFICIENCY")
    print(f"{'─'*60}")
    print(f"  Total scoring cycles: {r.total_cycles}")
    print(f"  Cycles with candidates: {r.cycles_with_candidates} ({r.cycles_with_candidates/max(r.total_cycles,1)*100:.1f}%)")
    print(f"  Cycles skipped (no signal): {r.cycles_skipped} ({r.cycles_skipped/max(r.total_cycles,1)*100:.1f}%)")

    print(f"\n{'─'*60}")
    print(f"  SCORE BUCKET ANALYSIS")
    print(f"{'─'*60}")
    print(f"  {'Bucket':<10} {'Trades':>7} {'Win%':>7} {'AvgPnL':>8} {'TotalPnL':>10}")
    for label, stats in r.score_buckets.items():
        if stats["trades"] > 0:
            print(
                f"  {label:<10} {stats['trades']:>7} "
                f"{stats['win_rate']:>6.1f}% "
                f"{stats['avg_pnl']:>+7.2f}% "
                f"${stats['total_pnl_usdt']:>+9.2f}"
            )

    print(f"\n{'─'*60}")
    print(f"  FACTOR ANALYSIS (avg values)")
    print(f"{'─'*60}")
    print(f"  {'Factor':<20} {'Winners':>10} {'Losers':>10} {'Delta':>10}")
    for k in sorted(r.factor_win_avg.keys()):
        w = r.factor_win_avg.get(k, 0)
        l_ = r.factor_loss_avg.get(k, 0)
        delta = w - l_
        marker = " ★" if abs(delta) > 0.1 else ""
        print(f"  {k:<20} {w:>+10.3f} {l_:>+10.3f} {delta:>+10.3f}{marker}")
    print(f"  (★ = significant factor — consider increasing its weight)\n")

    # Exit reason breakdown
    tp_count = sum(1 for t in r.trades if t.exit_reason == "tp_hit")
    sl_count = sum(1 for t in r.trades if t.exit_reason == "sl_hit")
    eod_count = sum(1 for t in r.trades if t.exit_reason == "end_of_data")
    print(f"  Exit Reasons: TP={tp_count} SL={sl_count} EndOfData={eod_count}")

    # Top 5 best and worst trades
    sorted_trades = sorted(r.trades, key=lambda t: t.pnl_pct, reverse=True)
    if sorted_trades:
        print(f"\n  {'Top 5 Winners':<40}")
        for t in sorted_trades[:5]:
            print(f"    {t.symbol:<12} pnl={t.pnl_pct:+.2f}% score={t.quant_score:.0f} R:R={t.reward_risk_ratio:.1f} exit={t.exit_reason}")
        print(f"\n  {'Top 5 Losers':<40}")
        for t in sorted_trades[-5:]:
            print(f"    {t.symbol:<12} pnl={t.pnl_pct:+.2f}% score={t.quant_score:.0f} R:R={t.reward_risk_ratio:.1f} exit={t.exit_reason}")

    print(f"\n{'='*60}\n")


def save_report(r: BacktestResult, path: str = "backtest_results.json") -> None:
    """Save backtest results to JSON for further analysis."""
    data = {
        "symbols": r.symbols,
        "period_days": r.period_days,
        "start_date": r.start_date,
        "end_date": r.end_date,
        "initial_balance": r.initial_balance,
        "final_balance": r.final_balance,
        "total_return_pct": r.total_return_pct,
        "total_trades": r.total_trades,
        "winning_trades": r.winning_trades,
        "losing_trades": r.losing_trades,
        "win_rate_pct": r.win_rate_pct,
        "max_drawdown_pct": r.max_drawdown_pct,
        "sharpe_ratio": r.sharpe_ratio,
        "profit_factor": r.profit_factor,
        "avg_win_pct": r.avg_win_pct,
        "avg_loss_pct": r.avg_loss_pct,
        "avg_rr_achieved": r.avg_rr_achieved,
        "avg_winning_score": r.avg_winning_score,
        "avg_losing_score": r.avg_losing_score,
        "score_buckets": r.score_buckets,
        "factor_win_avg": r.factor_win_avg,
        "factor_loss_avg": r.factor_loss_avg,
        "avg_hold_hours": r.avg_hold_hours,
        "equity_curve": r.equity_curve,
        "total_cycles": r.total_cycles,
        "cycles_with_candidates": r.cycles_with_candidates,
        "cycles_skipped": r.cycles_skipped,
        "config": {
            "MIN_QUANT_SCORE": settings.MIN_QUANT_SCORE,
            "MIN_REWARD_RISK_RATIO": settings.MIN_REWARD_RISK_RATIO,
            "SL_ATR_MULTIPLIER": settings.SL_ATR_MULTIPLIER,
            "MIN_SL_PCT": settings.MIN_SL_PCT,
            "MAX_SL_PCT": settings.MAX_SL_PCT,
            "MAX_POSITION_PCT": settings.MAX_POSITION_PCT,
            "FEE_PCT": FEE_PCT,
            "SLIPPAGE_PCT": SLIPPAGE_PCT,
        },
        "trades": [
            {
                "symbol": t.symbol,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "sl_pct": t.stop_loss_pct,
                "tp_pct": t.take_profit_pct,
                "quant_score": t.quant_score,
                "rr_ratio": t.reward_risk_ratio,
                "exit_reason": t.exit_reason,
                "pnl_pct": t.pnl_pct,
                "pnl_usdt": t.pnl_usdt,
                "signals": t.signals,
            }
            for t in r.trades
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")


# ── CLI entry point ──────────────────────────────────────────────────────────

async def main(
    days: int = 90,
    n_symbols: int = 10,
    balance: float = 10_000.0,
    min_score: float | None = None,
    min_rr: float | None = None,
    max_positions: int = 5,
    symbols_list: str | None = None,
    save_path: str | None = None,
) -> None:
    """Main entry point for the backtester."""
    # Load .env
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

    if symbols_list:
        symbols = [s.strip().upper() for s in symbols_list.split(",")]
        # Ensure they end with /USDT
        symbols = [s if "/" in s else f"{s}/{settings.QUOTE_CURRENCY}" for s in symbols]
    else:
        print(f"Fetching top {n_symbols} symbols by volume...")
        symbols = await get_top_symbols_for_backtest(n_symbols)
        print(f"Selected: {', '.join(symbols)}\n")

    result = await run_backtest(
        symbols=symbols,
        days=days,
        initial_balance=balance,
        min_score=min_score,
        min_rr=min_rr,
        max_positions=max_positions,
    )

    print_report(result)

    out_path = save_path or "backtest_results.json"
    save_report(result, out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest the quant scorer strategy")
    parser.add_argument("--days", type=int, default=90, help="How many days of history (default: 90)")
    parser.add_argument("--symbols", type=int, default=10, help="Number of top symbols (default: 10)")
    parser.add_argument("--balance", type=float, default=10_000.0, help="Starting balance (default: 10000)")
    parser.add_argument("--min-score", type=float, default=None, help="Override MIN_QUANT_SCORE")
    parser.add_argument("--min-rr", type=float, default=None, help="Override MIN_REWARD_RISK_RATIO")
    parser.add_argument("--max-positions", type=int, default=5, help="Max concurrent positions (default: 5)")
    parser.add_argument("--symbols-list", type=str, default=None, help="Comma-separated symbols (e.g. ETH,SOL,XRP)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: backtest_results.json)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(main(
        days=args.days,
        n_symbols=args.symbols,
        balance=args.balance,
        min_score=args.min_score,
        min_rr=args.min_rr,
        max_positions=args.max_positions,
        symbols_list=args.symbols_list,
        save_path=args.output,
    ))
