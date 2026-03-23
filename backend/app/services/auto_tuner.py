"""Auto-tuner — runs backtest periodically and adjusts quant scorer settings.

Runs on bot startup and then every BACKTEST_INTERVAL_HOURS.
Adjusts MIN_QUANT_SCORE and scorer weights based on backtest results.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.backtester import (
    run_backtest,
    get_top_symbols_for_backtest,
    print_report,
    save_report,
    BacktestResult,
)
from app.services import quant_scorer

logger = logging.getLogger(__name__)

# ── Tuning bounds (safety clamps) ───────────────────────────────────────────
_MIN_SCORE_RANGE = (45.0, 80.0)      # never go below 45 or above 80
_MIN_RR_RANGE = (1.5, 4.0)           # reward:risk ratio bounds
_SL_ATR_RANGE = (1.0, 3.0)           # ATR multiplier bounds
_WEIGHT_RANGE = (0.02, 0.25)         # per-factor weight bounds


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def tune_from_results(result: BacktestResult) -> dict[str, Any]:
    """Analyse backtest results and compute tuning adjustments.

    Returns a dict of {setting_name: new_value} for settings that should change.
    Only changes settings when there's clear statistical evidence.
    """
    changes: dict[str, Any] = {}

    if result.total_trades < 10:
        logger.info("Auto-tune: only %d trades — insufficient data, skipping", result.total_trades)
        return changes

    # ── 1. Tune MIN_QUANT_SCORE based on score bucket analysis ──────────
    # If high-score trades (80+) win much more than low-score (60-70):
    #   → current threshold is fine or can be raised
    # If all buckets perform similarly:
    #   → score isn't predictive, lower threshold to get more trades
    # If low-score trades actually win more:
    #   → lower threshold (the filter is too tight)

    bucket_stats = result.score_buckets
    low_bucket = bucket_stats.get("60-70", {})
    mid_bucket = bucket_stats.get("70-80", {})
    high_bucket = bucket_stats.get("80-90", {})

    low_wr = low_bucket.get("win_rate", 0) if low_bucket.get("trades", 0) >= 3 else None
    mid_wr = mid_bucket.get("win_rate", 0) if mid_bucket.get("trades", 0) >= 3 else None
    high_wr = high_bucket.get("win_rate", 0) if high_bucket.get("trades", 0) >= 3 else None

    current_min_score = settings.MIN_QUANT_SCORE

    if high_wr is not None and low_wr is not None:
        if high_wr > low_wr + 10:
            # Higher scores predict better → tighten threshold slightly
            new_score = _clamp(current_min_score + 3, *_MIN_SCORE_RANGE)
            if new_score != current_min_score:
                changes["MIN_QUANT_SCORE"] = round(new_score, 1)
                logger.info("Auto-tune: raising MIN_QUANT_SCORE %.1f → %.1f (high scores win more)",
                            current_min_score, new_score)
        elif low_wr > high_wr + 5:
            # Low scores win more (odd) → lower threshold
            new_score = _clamp(current_min_score - 5, *_MIN_SCORE_RANGE)
            if new_score != current_min_score:
                changes["MIN_QUANT_SCORE"] = round(new_score, 1)
                logger.info("Auto-tune: lowering MIN_QUANT_SCORE %.1f → %.1f (low scores win more)",
                            current_min_score, new_score)

    # If win rate is very low overall, lower score threshold to get more trades
    # (might be filtering out good opportunities)
    if result.win_rate_pct < 25 and result.total_trades >= 15:
        new_score = _clamp(current_min_score - 5, *_MIN_SCORE_RANGE)
        if new_score != current_min_score:
            changes["MIN_QUANT_SCORE"] = round(new_score, 1)
            logger.info("Auto-tune: lowering MIN_QUANT_SCORE (win_rate=%.1f%% too low)", result.win_rate_pct)

    # If win rate is very high, we can be more selective
    if result.win_rate_pct > 55 and result.total_trades >= 15:
        new_score = _clamp(current_min_score + 3, *_MIN_SCORE_RANGE)
        if new_score != current_min_score:
            changes["MIN_QUANT_SCORE"] = round(new_score, 1)
            logger.info("Auto-tune: raising MIN_QUANT_SCORE (win_rate=%.1f%% strong)", result.win_rate_pct)

    # ── 2. Tune SL_ATR_MULTIPLIER based on exit reasons ─────────────────
    # Too many SL hits → widen SL (increase multiplier)
    # Too few SL hits (lots of TP) → tighten SL (decrease multiplier)
    sl_hits = sum(1 for t in result.trades if t.exit_reason == "sl_hit")
    tp_hits = sum(1 for t in result.trades if t.exit_reason == "tp_hit")
    total_exits = sl_hits + tp_hits

    if total_exits >= 10:
        sl_ratio = sl_hits / total_exits
        current_sl_mult = settings.SL_ATR_MULTIPLIER

        if sl_ratio > 0.70:
            # Too many stops → widen SL
            new_mult = _clamp(current_sl_mult + 0.25, *_SL_ATR_RANGE)
            if new_mult != current_sl_mult:
                changes["SL_ATR_MULTIPLIER"] = round(new_mult, 2)
                logger.info("Auto-tune: widening SL_ATR_MULT %.2f → %.2f (SL hit rate=%.0f%%)",
                            current_sl_mult, new_mult, sl_ratio * 100)
        elif sl_ratio < 0.35:
            # Very few stops → can tighten SL for better R:R
            new_mult = _clamp(current_sl_mult - 0.15, *_SL_ATR_RANGE)
            if new_mult != current_sl_mult:
                changes["SL_ATR_MULTIPLIER"] = round(new_mult, 2)
                logger.info("Auto-tune: tightening SL_ATR_MULT %.2f → %.2f (SL hit rate=%.0f%%)",
                            current_sl_mult, new_mult, sl_ratio * 100)

    # ── 3. Tune factor weights based on winner/loser analysis ────────────
    # Factors where winners score significantly higher than losers → increase weight
    # Factors where there's no difference → decrease weight
    if result.total_trades >= 20:
        weight_changes = _tune_weights(result.factor_win_avg, result.factor_loss_avg)
        if weight_changes:
            changes["_weight_adjustments"] = weight_changes

    return changes


def _tune_weights(
    factor_win_avg: dict[str, float],
    factor_loss_avg: dict[str, float],
) -> dict[str, float] | None:
    """Adjust quant scorer weights based on factor predictiveness.

    Returns new weights dict if changes were made, None otherwise.
    """
    current_weights = dict(quant_scorer._WEIGHTS)
    new_weights = dict(current_weights)
    changed = False

    for factor_key in current_weights:
        w_avg = factor_win_avg.get(factor_key, 0)
        l_avg = factor_loss_avg.get(factor_key, 0)
        delta = w_avg - l_avg

        current_w = current_weights[factor_key]

        if delta > 0.15:
            # Strong predictor — increase weight by 20%
            new_w = _clamp(current_w * 1.20, *_WEIGHT_RANGE)
            if abs(new_w - current_w) > 0.005:
                new_weights[factor_key] = round(new_w, 3)
                changed = True
                logger.info("Auto-tune weight: %s %.3f → %.3f (delta=+%.3f, predictive)",
                            factor_key, current_w, new_w, delta)

        elif delta < -0.05:
            # Counter-predictive (winners score lower) — decrease weight by 20%
            new_w = _clamp(current_w * 0.80, *_WEIGHT_RANGE)
            if abs(new_w - current_w) > 0.005:
                new_weights[factor_key] = round(new_w, 3)
                changed = True
                logger.info("Auto-tune weight: %s %.3f → %.3f (delta=%.3f, counter-predictive)",
                            factor_key, current_w, new_w, delta)

        elif abs(delta) < 0.02 and current_w > 0.06:
            # Not predictive — shrink slightly
            new_w = _clamp(current_w * 0.90, *_WEIGHT_RANGE)
            if abs(new_w - current_w) > 0.005:
                new_weights[factor_key] = round(new_w, 3)
                changed = True
                logger.info("Auto-tune weight: %s %.3f → %.3f (delta~0, not predictive)",
                            factor_key, current_w, new_w)

    if not changed:
        return None

    # Re-normalize to sum=1.0
    total = sum(new_weights.values())
    if total > 0:
        new_weights = {k: round(v / total, 3) for k, v in new_weights.items()}
        # Fix rounding: add remainder to largest weight
        remainder = 1.0 - sum(new_weights.values())
        if abs(remainder) > 0.0001:
            max_key = max(new_weights, key=new_weights.get)
            new_weights[max_key] = round(new_weights[max_key] + remainder, 3)

    return new_weights


def apply_tuning(changes: dict[str, Any]) -> list[str]:
    """Apply tuning changes to live settings. Returns list of change descriptions."""
    applied: list[str] = []

    if "MIN_QUANT_SCORE" in changes:
        old = settings.MIN_QUANT_SCORE
        settings.MIN_QUANT_SCORE = changes["MIN_QUANT_SCORE"]
        applied.append(f"MIN_QUANT_SCORE: {old} → {changes['MIN_QUANT_SCORE']}")

    if "SL_ATR_MULTIPLIER" in changes:
        old = settings.SL_ATR_MULTIPLIER
        settings.SL_ATR_MULTIPLIER = changes["SL_ATR_MULTIPLIER"]
        applied.append(f"SL_ATR_MULTIPLIER: {old} → {changes['SL_ATR_MULTIPLIER']}")

    if "MIN_REWARD_RISK_RATIO" in changes:
        old = settings.MIN_REWARD_RISK_RATIO
        settings.MIN_REWARD_RISK_RATIO = changes["MIN_REWARD_RISK_RATIO"]
        applied.append(f"MIN_REWARD_RISK_RATIO: {old} → {changes['MIN_REWARD_RISK_RATIO']}")

    if "_weight_adjustments" in changes:
        new_weights = changes["_weight_adjustments"]
        old_weights = dict(quant_scorer._WEIGHTS)
        quant_scorer._WEIGHTS.update(new_weights)
        # Log changed weights
        for k, v in new_weights.items():
            if old_weights.get(k) != v:
                applied.append(f"weight/{k}: {old_weights.get(k, 0):.3f} → {v:.3f}")

    return applied


async def run_auto_backtest(
    days: int = 30,
    n_symbols: int = 8,
) -> tuple[BacktestResult | None, list[str]]:
    """Run a backtest and return results + applied tuning changes.

    Uses fewer symbols and shorter period than manual backtest for speed.
    """
    logger.info("Auto-backtest starting (%dd, %d symbols)...", days, n_symbols)
    t0 = time.time()

    try:
        symbols = await get_top_symbols_for_backtest(n_symbols)
        if not symbols:
            logger.warning("Auto-backtest: no symbols available")
            return None, []

        result = await run_backtest(
            symbols=symbols,
            days=days,
            initial_balance=settings.DEMO_INITIAL_BALANCE,
            max_positions=5,
        )

        elapsed = time.time() - t0
        logger.info(
            "Auto-backtest done in %.0fs: %d trades, win_rate=%.1f%%, return=%+.2f%%",
            elapsed, result.total_trades, result.win_rate_pct, result.total_return_pct,
        )

        # Compute and apply tuning
        changes = tune_from_results(result)
        applied = apply_tuning(changes) if changes else []

        if applied:
            logger.info("Auto-tune applied %d changes: %s", len(applied), " | ".join(applied))
        else:
            logger.info("Auto-tune: no changes needed")

        # Update Kelly fraction in bot_runner (position sizing from backtest edge)
        if settings.KELLY_SIZING and result.total_trades >= 10:
            try:
                from app.services.bot_runner import bot_runner
                bot_runner.update_kelly_fraction(
                    win_rate=result.win_rate_pct / 100,
                    avg_win=result.avg_win_pct / 100,
                    avg_loss=result.avg_loss_pct / 100,
                )
            except Exception as exc:
                logger.warning("Kelly update failed (non-fatal): %s", exc)

        # Save results
        out_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "last_backtest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_report(result, str(out_path))

        return result, applied

    except Exception as exc:
        logger.error("Auto-backtest failed (non-fatal): %s", exc)
        return None, []


async def backtest_scheduler(interval_hours: float = 24.0) -> None:
    """Background task: run backtest every N hours.

    First run happens immediately, then repeats on schedule.
    """
    while True:
        try:
            result, applied = await run_auto_backtest(
                days=settings.BACKTEST_DAYS,
                n_symbols=settings.BACKTEST_SYMBOLS,
            )
            if result and applied:
                # Notify via Discord
                from app.services.discord import send_alert
                changes_text = "\n".join(f"• {c}" for c in applied)
                await send_alert(
                    "Auto-Tune Update",
                    f"Backtest ({result.period_days}d, {result.total_trades} trades, "
                    f"win={result.win_rate_pct:.0f}%):\n{changes_text}",
                )
        except Exception as exc:
            logger.error("Backtest scheduler error: %s", exc)

        await asyncio.sleep(interval_hours * 3600)
