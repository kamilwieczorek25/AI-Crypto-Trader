"""Smart Exit Analyzer — local signal-based exit engine.

Runs every cycle for each open position.  Uses technical indicators +
GPU Exit RL (when available) to decide whether to hold, reduce, or close.

Decision priority:
  1. GPU Exit RL (Dueling DQN) — if trained and confident → obey
  2. Local reversal detector — multi-signal confirmation of a trend reversal
  3. Profit lock (sliding floor) — safety net based on peak PnL

Returns one of:
  HOLD      — keep the position
  PARTIAL   — reduce by ~50%
  CLOSE     — full exit
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── Result dataclass ─────────────────────────────────────────────────
@dataclass
class ExitSignal:
    action: str       # "HOLD" | "PARTIAL" | "CLOSE"
    reason: str       # human-readable reason
    confidence: float # 0.0 – 1.0
    source: str       # "exit_rl" | "local" | "profit_lock"


# ── Thresholds (tuned for crypto 1h TF) ─────────────────────────────
# How many local reversal signals must fire before we act
_MIN_REVERSAL_SIGNALS = 3   # out of ~8 checks → strong confirmation
_MIN_PARTIAL_SIGNALS  = 2   # weaker → partial exit


def analyze_exit(
    *,
    pos,                     # Position ORM object
    ind_1h: dict,            # 1h indicators (rsi14, macd_hist, etc.)
    btc_anchor: dict,        # BTC indicators dict
    exit_rl: dict | None,    # GPU Exit RL prediction (or None)
    ensemble: dict | None,   # GPU ensemble signal (or None)
    anomaly: dict | None,    # anomaly detection result (or None)
    vol_forecast: dict | None,  # volatility forecast (or None)
    profit_lock_activate: float,
    profit_lock_floor_min: float,
    profit_lock_keep: float, # 0.0-1.0 (already divided by 100)
) -> ExitSignal:
    """Analyze whether an open position should be exited.

    Called once per cycle per open bot-managed position.
    """
    entry = pos.avg_entry_price
    price = pos.current_price
    if entry <= 0 or price <= 0:
        return ExitSignal("HOLD", "no price data", 0.0, "local")

    pnl_pct = (price - entry) / entry * 100
    highest = getattr(pos, "highest_price", price) or price
    peak_pnl_pct = (highest - entry) / entry * 100 if highest > 0 else 0.0
    drawdown_from_peak = (highest - price) / highest * 100 if highest > 0 else 0.0

    # ── 1. GPU Exit RL (highest priority when trained) ───────────────
    if exit_rl and exit_rl.get("trained", False):
        rl_action = exit_rl.get("action", "HOLD_POS")
        q_values = exit_rl.get("q_values", {})

        # Only act if the model is confident (Q-spread between best and HOLD)
        q_hold = q_values.get("HOLD_POS", 0) if isinstance(q_values, dict) else 0
        q_best = max(q_values.values()) if isinstance(q_values, dict) and q_values else 0
        q_spread = q_best - q_hold

        if rl_action == "CLOSE" and q_spread > 0.05:
            return ExitSignal("CLOSE", f"Exit RL: CLOSE (Q-spread={q_spread:.3f})", min(q_spread * 5, 1.0), "exit_rl")
        if rl_action in ("PARTIAL_25", "PARTIAL_50") and q_spread > 0.03:
            return ExitSignal("PARTIAL", f"Exit RL: {rl_action} (Q-spread={q_spread:.3f})", min(q_spread * 5, 1.0), "exit_rl")

    # ── 2. Local reversal detector (multi-signal confirmation) ───────
    signals = _count_reversal_signals(
        pnl_pct=pnl_pct,
        peak_pnl_pct=peak_pnl_pct,
        drawdown_from_peak=drawdown_from_peak,
        ind_1h=ind_1h,
        btc_anchor=btc_anchor,
        ensemble=ensemble,
        anomaly=anomaly,
        vol_forecast=vol_forecast,
    )
    n_signals = len(signals)

    if n_signals >= _MIN_REVERSAL_SIGNALS:
        reasons = ", ".join(signals[:4])
        conf = min(n_signals / 6.0, 1.0)
        # Full close if strongly negative or large drawdown
        if pnl_pct < -1.0 or drawdown_from_peak > 5.0:
            return ExitSignal("CLOSE", f"Reversal ({n_signals} signals: {reasons})", conf, "local")
        else:
            return ExitSignal("PARTIAL", f"Reversal ({n_signals} signals: {reasons})", conf, "local")

    if n_signals >= _MIN_PARTIAL_SIGNALS and pnl_pct > 1.0:
        # We're in profit and seeing early reversal signs → partial exit to lock some
        reasons = ", ".join(signals[:3])
        return ExitSignal("PARTIAL", f"Early reversal ({n_signals} signals: {reasons})", 0.5, "local")

    # ── 3. Profit lock safety net (sliding floor) ────────────────────
    if (
        profit_lock_activate > 0
        and peak_pnl_pct >= profit_lock_activate
        and not getattr(pos, "tp_activated", False)
    ):
        dynamic_floor = max(profit_lock_floor_min, peak_pnl_pct * profit_lock_keep)
        if pnl_pct <= dynamic_floor:
            return ExitSignal(
                "CLOSE",
                f"Profit lock: peaked +{peak_pnl_pct:.1f}%, floor +{dynamic_floor:.1f}%",
                0.8,
                "profit_lock",
            )

    return ExitSignal("HOLD", "no exit signals", 0.0, "local")


def _count_reversal_signals(
    *,
    pnl_pct: float,
    peak_pnl_pct: float,
    drawdown_from_peak: float,
    ind_1h: dict,
    btc_anchor: dict,
    ensemble: dict | None,
    anomaly: dict | None,
    vol_forecast: dict | None,
) -> list[str]:
    """Count how many independent reversal signals are firing.

    Each signal is a lightweight check on a single indicator or
    combination — no heavy computation, runs in < 1ms.
    Returns list of human-readable signal labels that fired.
    """
    fired: list[str] = []

    # 1. RSI overbought reversal — was high, now dropping
    rsi = ind_1h.get("rsi14", 50)
    if rsi > 70:
        fired.append(f"RSI overbought ({rsi:.0f})")
    elif rsi < 30 and pnl_pct > 0:
        # RSI crashed while we're still in profit — momentum gone
        fired.append(f"RSI crashed ({rsi:.0f})")

    # 2. MACD histogram flipping negative (bearish momentum shift)
    macd_hist = ind_1h.get("macd_hist", 0)
    if macd_hist < -0.001:  # small threshold to avoid noise
        fired.append(f"MACD bearish ({macd_hist:.4f})")

    # 3. Price below VWAP — selling pressure dominates
    price_vs_vwap = ind_1h.get("price_vs_vwap", 0)
    if price_vs_vwap < -1.0:
        fired.append(f"Below VWAP ({price_vs_vwap:.1f}%)")

    # 4. Bollinger Bands %B below 0.2 — at lower band
    bb_pct_b = ind_1h.get("bb_pct_b", 0.5)
    if bb_pct_b < 0.2:
        fired.append(f"BB lower ({bb_pct_b:.2f})")

    # 5. Significant drawdown from peak (lost > 40% of gains)
    if peak_pnl_pct > 2 and drawdown_from_peak > 3:
        fired.append(f"Drawdown {drawdown_from_peak:.1f}% from peak")

    # 6. OBV trend bearish (selling volume > buying volume)
    obv_trend = ind_1h.get("obv_trend", 0)
    if obv_trend < 0:
        fired.append("OBV bearish")

    # 7. BTC dragging market down
    btc_1h = btc_anchor.get("1h", {}) if isinstance(btc_anchor, dict) else {}
    btc_macd = btc_1h.get("macd_hist", 0)
    btc_trend = btc_1h.get("trend", 0)
    if btc_macd < -0.001 and btc_trend < 0:
        fired.append("BTC bearish")

    # 8. GPU ensemble says SELL
    if ensemble:
        ens_signal = ensemble.get("signal", "")
        ens_conf = ensemble.get("confidence", 0)
        if ens_signal == "SELL" and ens_conf > 0.5:
            fired.append(f"Ensemble SELL ({ens_conf:.0%})")

    # 9. Anomaly detected (unusual market behaviour)
    if anomaly:
        a_score = anomaly.get("anomaly_score", 0)
        is_anomaly = anomaly.get("is_anomaly", False)
        if is_anomaly and a_score > 2.0:
            fired.append(f"Anomaly (score={a_score:.1f})")

    # 10. Volatility spike forecast — turbulence ahead
    if vol_forecast:
        pred_vol = vol_forecast.get("predicted_vol", 0)
        if pred_vol > 0.05:  # >5% expected hourly volatility
            fired.append(f"Vol spike ({pred_vol:.1%})")

    return fired
