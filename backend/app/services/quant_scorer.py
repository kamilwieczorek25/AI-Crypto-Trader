"""Quantitative scoring engine — signal-first, backtestable symbol ranking.

Scores each symbol on a 0–100 scale using weighted technical factors.
Computes ATR-based stop-loss/take-profit levels with enforced minimum
reward-to-risk ratio. Produces structured trade candidates for Claude
to validate (not originate).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# ── Scoring weights (sum to 1.0) ────────────────────────────────────────────
# These are tuned for crypto momentum/mean-reversion blend.
# Each factor produces a raw signal in [-1, +1] which gets multiplied by weight.
_WEIGHTS: dict[str, float] = {
    "rsi_signal":       0.08,   # RSI oversold/overbought
    "macd_signal":      0.07,   # MACD histogram direction + magnitude
    "bb_signal":        0.06,   # Bollinger Band position
    "bb_squeeze_signal":0.05,   # Bollinger squeeze (breakout detector)
    "volume_signal":    0.09,   # Volume ratio (activity)
    "obv_signal":       0.04,   # On-Balance Volume trend
    "vwap_signal":      0.05,   # Price vs VWAP
    "trend_signal":     0.08,   # Multi-timeframe trend alignment
    "sr_signal":        0.05,   # Support/resistance proximity
    "btc_signal":       0.07,   # BTC anchor trend (critical for alts)
    "ml_signal":        0.04,   # LSTM + RL consensus
    "orderbook_signal": 0.04,   # Bid/ask pressure ratio
    "depth_signal":     0.05,   # Orderbook depth imbalance within 2%
    "whale_signal":     0.07,   # Whale trade flow (WebSocket real-time)
    "funding_signal":   0.05,   # Binance funding rate (contrarian)
    "ls_ratio_signal":  0.05,   # Long/short ratio (contrarian)
    "oi_signal":        0.06,   # Open interest trend
}


@dataclass
class TradeCandidate:
    """A pre-screened trade opportunity with quantitative backing."""
    symbol: str
    action: str                 # "BUY" or "SELL"
    score: float                # 0–100 composite score
    timeframe: str              # primary signal timeframe
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    stop_loss_pct: float        # distance below/above entry
    take_profit_pct: float      # distance above/below entry
    reward_risk_ratio: float    # TP distance / SL distance
    quantity_pct: float         # suggested position size (% of portfolio)
    signals: list[str] = field(default_factory=list)
    factor_scores: dict[str, float] = field(default_factory=dict)


# ── Individual factor scoring functions ──────────────────────────────────────
# Each returns a value in [-1.0, +1.0]
# Positive = bullish / buy signal, Negative = bearish / sell signal

def _score_rsi(ind: dict) -> float:
    """RSI mean-reversion signal: oversold → buy, overbought → sell."""
    rsi = ind.get("rsi14", 50)
    if rsi <= 20:
        return 1.0          # extremely oversold
    elif rsi <= 30:
        return 0.7
    elif rsi <= 35:
        return 0.3
    elif rsi >= 80:
        return -1.0         # extremely overbought
    elif rsi >= 70:
        return -0.7
    elif rsi >= 65:
        return -0.3
    return 0.0               # neutral zone


def _score_macd(ind: dict) -> float:
    """MACD histogram direction and magnitude."""
    macd_hist = ind.get("macd_hist", 0)
    if macd_hist == 0:
        return 0.0
    # Normalize: typical MACD hist in crypto is ±0.001 to ±0.05
    magnitude = min(abs(macd_hist) * 50, 1.0)
    return magnitude if macd_hist > 0 else -magnitude


def _score_bb(ind: dict) -> float:
    """Bollinger Band %B — below 0.2 = buy zone, above 0.8 = sell zone."""
    bb = ind.get("bb_pct_b", 0.5)
    if bb <= 0.05:
        return 1.0
    elif bb <= 0.15:
        return 0.7
    elif bb <= 0.25:
        return 0.3
    elif bb >= 0.95:
        return -1.0
    elif bb >= 0.85:
        return -0.7
    elif bb >= 0.75:
        return -0.3
    return 0.0


def _score_volume(ind: dict) -> float:
    """Volume ratio — high volume confirms moves. Score direction by trend."""
    vol_ratio = ind.get("volume_ratio", 1.0)
    vol_trend = ind.get("volume_trend", 1.0)
    trend_dir = ind.get("trend", 0)

    if vol_ratio < 0.5:
        return 0.0  # dead volume, no signal
    if vol_ratio > 2.0:
        # Big volume — direction depends on trend
        return 0.8 * (1.0 if trend_dir > 0 else -0.5)
    elif vol_ratio > 1.5:
        return 0.5 * (1.0 if trend_dir > 0 else -0.3)
    elif vol_ratio > 1.2:
        return 0.2 * (1.0 if trend_dir > 0 else 0.0)
    return 0.0


def _score_obv(ind: dict) -> float:
    """OBV trend: confirms or diverges from price trend."""
    obv = ind.get("obv_trend", 0)
    trend = ind.get("trend", 0)
    if obv > 0 and trend > 0:
        return 0.6    # OBV confirms uptrend
    elif obv > 0 and trend <= 0:
        return 0.4    # bullish divergence (OBV up, price down)
    elif obv < 0 and trend < 0:
        return -0.6   # OBV confirms downtrend
    elif obv < 0 and trend >= 0:
        return -0.4   # bearish divergence (OBV down, price up)
    return 0.0


def _score_vwap(ind: dict) -> float:
    """Price vs VWAP: below VWAP = potential buy, above = potential sell."""
    pv = ind.get("price_vs_vwap", 0)
    if pv < -2.0:
        return 0.7     # well below VWAP
    elif pv < -0.5:
        return 0.3
    elif pv > 2.0:
        return -0.7    # well above VWAP
    elif pv > 0.5:
        return -0.3
    return 0.0


def _score_trend(indicators_all_tf: dict[str, dict]) -> float:
    """Multi-timeframe trend alignment. More TFs aligned = stronger signal."""
    if not indicators_all_tf:
        return 0.0

    bullish_count = 0
    total = 0
    # Weight higher timeframes more
    tf_weights = {"15m": 0.5, "1h": 1.0, "4h": 1.5, "1d": 2.0}
    weighted_sum = 0.0
    weight_total = 0.0

    for tf, ind in indicators_all_tf.items():
        w = tf_weights.get(tf, 1.0)
        trend = ind.get("trend", 0)
        weighted_sum += trend * w
        weight_total += w
        total += 1
        if trend > 0:
            bullish_count += 1

    if weight_total == 0:
        return 0.0

    # Normalize to [-1, 1]
    return max(-1.0, min(1.0, weighted_sum / weight_total))


def _score_sr(sr: dict, price: float) -> float:
    """Support/resistance proximity: near support = buy, near resistance = sell."""
    if not sr or not price or price <= 0:
        return 0.0

    support = sr.get("nearest_support")
    resistance = sr.get("nearest_resistance")

    score = 0.0
    if support and support > 0:
        dist_to_support = (price - support) / price * 100
        if 0 < dist_to_support < 1.5:
            score += 0.8  # very near support — buy zone
        elif 0 < dist_to_support < 3.0:
            score += 0.4

    if resistance and resistance > 0:
        dist_to_resistance = (resistance - price) / price * 100
        if 0 < dist_to_resistance < 1.5:
            score -= 0.8  # very near resistance — sell zone
        elif 0 < dist_to_resistance < 3.0:
            score -= 0.4

    return max(-1.0, min(1.0, score))


def _score_btc(btc_anchor: dict | None) -> float:
    """BTC trend: if BTC is bearish, penalize all altcoin buys."""
    if not btc_anchor:
        return 0.0

    # Combine 1h and 4h BTC signals
    signals = []
    for tf in ("1h", "4h"):
        ind = btc_anchor.get(tf, {})
        if not ind:
            continue
        trend = ind.get("trend", 0)
        rsi = ind.get("rsi14", 50)
        # BTC bullish: trend up + RSI not overbought
        if trend > 0 and rsi < 70:
            signals.append(0.5)
        elif trend > 0:
            signals.append(0.2)
        # BTC bearish: trend down + RSI not oversold
        elif trend < 0 and rsi > 30:
            signals.append(-0.7)
        elif trend < 0:
            signals.append(-0.3)
        else:
            signals.append(0.0)

    return sum(signals) / len(signals) if signals else 0.0


def _score_ml(ml_signal: dict | None) -> float:
    """LSTM + RL consensus. Only counts when they agree."""
    if not ml_signal:
        return 0.0

    lstm = ml_signal.get("lstm", {})
    rl = ml_signal.get("rl", {})
    lstm_sig = lstm.get("signal", "HOLD")
    lstm_conf = lstm.get("confidence", 0)
    rl_sig = rl.get("action", "HOLD")

    if lstm_sig == rl_sig:
        if lstm_sig == "BUY":
            return min(lstm_conf, 1.0) * 0.8
        elif lstm_sig == "SELL":
            return -min(lstm_conf, 1.0) * 0.8
    # Disagreement: slight signal in LSTM direction if it's confident
    if lstm_sig == "BUY" and lstm_conf > 0.6:
        return 0.2
    if lstm_sig == "SELL" and lstm_conf > 0.6:
        return -0.2
    return 0.0


def _score_orderbook(ob: dict) -> float:
    """Order book pressure ratio: high buy pressure = bullish."""
    pressure = ob.get("pressure_ratio", 1.0)
    if pressure > 2.0:
        return 0.7
    elif pressure > 1.5:
        return 0.4
    elif pressure > 1.2:
        return 0.2
    elif pressure < 0.5:
        return -0.7
    elif pressure < 0.67:
        return -0.4
    elif pressure < 0.83:
        return -0.2
    return 0.0


def _score_depth(ob: dict) -> float:
    """Orderbook depth imbalance within 2% of price.

    Strong bid depth = buyers waiting = bullish support.
    Strong ask depth = sellers stacked = resistance.
    """
    imbalance = ob.get("depth_imbalance", 0.0)
    if imbalance > 0.5:
        return 0.7    # 3:1 bid vs ask ratio within 2%
    elif imbalance > 0.3:
        return 0.4
    elif imbalance > 0.15:
        return 0.2
    elif imbalance < -0.5:
        return -0.7
    elif imbalance < -0.3:
        return -0.4
    elif imbalance < -0.15:
        return -0.2
    return 0.0


def _score_whale(whale_data: dict | None) -> float:
    """Whale trade activity — large trades signal institutional interest.

    Net buy flow = bullish (smart money accumulating).
    Net sell flow = bearish (smart money distributing).
    """
    if not whale_data:
        return 0.0
    net_flow = whale_data.get("whale_net_flow", 0)
    total_vol = whale_data.get("whale_total_volume", 0)
    if total_vol <= 0:
        return 0.0

    # Normalize: what fraction of total whale volume is directional?
    bias = net_flow / total_vol  # [-1, +1]

    if bias > 0.5:
        return 0.7    # strong whale buying
    elif bias > 0.2:
        return 0.4
    elif bias > 0:
        return 0.2
    elif bias < -0.5:
        return -0.7   # strong whale selling
    elif bias < -0.2:
        return -0.4
    elif bias < 0:
        return -0.2
    return 0.0


def _score_bb_squeeze(ind: dict) -> float:
    """Bollinger squeeze — low volatility compression precedes breakout.

    Squeeze + bullish trend = strong buy signal.
    Squeeze + bearish trend = strong sell signal.
    No squeeze = neutral (handled by bb_signal).
    """
    squeeze = ind.get("bb_squeeze", 0)
    if not squeeze:
        return 0.0
    # Squeeze detected — direction from trend
    trend = ind.get("trend", 0)
    if trend > 0:
        return 0.7    # squeeze + uptrend = breakout buy
    elif trend < 0:
        return -0.7   # squeeze + downtrend = breakdown sell
    return 0.3        # squeeze + neutral = slight bullish bias (breakout up is more common)


def _score_oi(oi_data: dict | None) -> float:
    """Open Interest — rising OI + rising price = real trend.

    Rising OI + rising price = strong continuation signal.
    Rising OI + falling price = bearish (shorts entering).
    Falling OI = positions closing, trend weakening.
    """
    if not oi_data:
        return 0.0
    oi_change = oi_data.get("change_pct", 0)
    price_trend = oi_data.get("price_trend", 0)

    if oi_change > 5:
        # OI rising significantly
        if price_trend > 0:
            return 0.6    # rising OI + rising price = real buying
        else:
            return -0.5   # rising OI + falling price = shorts entering
    elif oi_change < -5:
        # OI falling significantly
        if price_trend > 0:
            return 0.2    # short squeeze (OI falling + price rising)
        else:
            return -0.2   # long liquidation
    return 0.0

def _score_funding(funding_data: dict | None) -> float:
    """Funding rate — contrarian signal.

    High positive rate = crowd is long (longs pay shorts) → contrarian bearish
    High negative rate = crowd is short → contrarian bullish
    Extreme rates often precede reversals.
    """
    if not funding_data:
        return 0.0
    rate = funding_data.get("rate", 0)
    if rate > 0.001:       # extreme positive → contrarian sell
        return -0.7
    elif rate > 0.0005:
        return -0.3
    elif rate < -0.001:    # extreme negative → contrarian buy
        return 0.7
    elif rate < -0.0005:
        return 0.3
    return 0.0


def _score_ls_ratio(ls_data: dict | None) -> float:
    """Long/short ratio — contrarian signal.

    Crowd heavily long (ratio > 1.5) = contrarian bearish
    Crowd heavily short (ratio < 0.67) = contrarian bullish
    """
    if not ls_data:
        return 0.0
    ratio = ls_data.get("ratio", 1.0)
    if ratio > 2.0:
        return -0.7      # extreme crowd long → contrarian sell
    elif ratio > 1.5:
        return -0.4
    elif ratio > 1.2:
        return -0.15
    elif ratio < 0.5:
        return 0.7        # extreme crowd short → contrarian buy
    elif ratio < 0.67:
        return 0.4
    elif ratio < 0.83:
        return 0.15
    return 0.0

# ── Composite scoring ────────────────────────────────────────────────────────

def score_symbol(
    symbol: str,
    symbol_data: dict[str, Any],
    news_data: dict,
    ml_signal: dict | None = None,
    btc_anchor: dict | None = None,
    is_held: bool = False,
    funding_data: dict | None = None,
    ls_data: dict | None = None,
    oi_data: dict | None = None,
    whale_data: dict | None = None,
) -> dict[str, Any]:
    """Score a symbol and return detailed factor breakdown.

    Returns dict with 'score' (0–100), 'direction' (+1=buy, -1=sell),
    'factors', and 'signals' list.
    """
    indicators = symbol_data.get("indicators", {})
    ind_1h = indicators.get("1h", {})
    ob = symbol_data.get("orderbook", {})
    sr = symbol_data.get("support_resistance", {})
    price = symbol_data.get("price", 0)

    # Compute each factor
    factors: dict[str, float] = {
        "rsi_signal":       _score_rsi(ind_1h),
        "macd_signal":      _score_macd(ind_1h),
        "bb_signal":        _score_bb(ind_1h),
        "bb_squeeze_signal":_score_bb_squeeze(ind_1h),
        "volume_signal":    _score_volume(ind_1h),
        "obv_signal":       _score_obv(ind_1h),
        "vwap_signal":      _score_vwap(ind_1h),
        "trend_signal":     _score_trend(indicators),
        "sr_signal":        _score_sr(sr, price),
        "btc_signal":       _score_btc(btc_anchor),
        "ml_signal":        _score_ml(ml_signal),
        "orderbook_signal": _score_orderbook(ob),
        "depth_signal":     _score_depth(ob),
        "whale_signal":     _score_whale(whale_data),
        "funding_signal":   _score_funding(funding_data),
        "ls_ratio_signal":  _score_ls_ratio(ls_data),
        "oi_signal":        _score_oi(oi_data),
    }

    # RSI divergence adds bonus
    rsi_div = ind_1h.get("rsi_divergence", 0)
    if rsi_div > 0:
        factors["rsi_signal"] = min(1.0, factors["rsi_signal"] + 0.3)
    elif rsi_div < 0:
        factors["rsi_signal"] = max(-1.0, factors["rsi_signal"] - 0.3)

    # Weighted composite: sum(factor * weight) → range [-1, +1]
    composite = sum(factors[k] * _WEIGHTS[k] for k in _WEIGHTS)

    # Normalize to 0–100 scale (50 = neutral)
    score = max(0, min(100, 50 + composite * 50))

    # Direction: positive composite = buy, negative = sell
    direction = 1 if composite > 0 else (-1 if composite < 0 else 0)

    # Build signal descriptions for the top contributing factors
    signals: list[str] = []
    sorted_factors = sorted(factors.items(), key=lambda x: abs(x[1]), reverse=True)
    for fname, fval in sorted_factors[:4]:
        if abs(fval) < 0.1:
            continue
        label = fname.replace("_signal", "").upper()
        direction_str = "bullish" if fval > 0 else "bearish"
        signals.append(f"{label}={fval:+.2f}({direction_str})")

    return {
        "score": round(score, 1),
        "composite": round(composite, 4),
        "direction": direction,
        "factors": {k: round(v, 3) for k, v in factors.items()},
        "signals": signals,
    }


# ── ATR-based SL/TP computation ─────────────────────────────────────────────

def compute_trade_levels(
    symbol: str,
    symbol_data: dict[str, Any],
    action: str,
    score: float,
) -> dict[str, float] | None:
    """Compute stop-loss and take-profit using ATR, enforcing min R:R ratio.

    Returns dict with sl_pct, tp_pct, rr_ratio, quantity_pct or None if
    the setup doesn't meet minimum R:R requirements.
    """
    indicators = symbol_data.get("indicators", {})
    price = symbol_data.get("price", 0)
    if not price or price <= 0:
        return None

    # Use 1h ATR as base, 4h ATR as confirmation
    atr_1h = indicators.get("1h", {}).get("atr", 0)
    atr_4h = indicators.get("4h", {}).get("atr", 0)

    # If no ATR available, use a conservative default
    if atr_1h <= 0:
        atr_1h = price * 0.02  # 2% default

    atr_pct = (atr_1h / price) * 100  # ATR as % of price

    # Stop-loss: 1.5x ATR (gives room for normal volatility)
    sl_multiplier = settings.SL_ATR_MULTIPLIER
    sl_pct = round(atr_pct * sl_multiplier, 2)

    # Clamp SL to reasonable range
    sl_pct = max(settings.MIN_SL_PCT, min(settings.MAX_SL_PCT, sl_pct))

    # Take-profit: enforce minimum R:R ratio
    min_rr = settings.MIN_REWARD_RISK_RATIO
    tp_pct = round(sl_pct * min_rr, 2)

    # Boost TP for high-score setups (score 80+ gets extra upside)
    if score >= 80:
        tp_pct = round(tp_pct * 1.5, 2)
    elif score >= 70:
        tp_pct = round(tp_pct * 1.2, 2)

    # Clamp TP
    tp_pct = max(tp_pct, sl_pct * min_rr)  # always enforce min R:R
    tp_pct = min(tp_pct, 30.0)  # cap at 30%

    rr_ratio = round(tp_pct / sl_pct, 2) if sl_pct > 0 else 0

    # Reject if R:R is below minimum
    if rr_ratio < min_rr:
        return None

    # Position size: scale with score (higher score → bigger position)
    base_pct = settings.MAX_POSITION_PCT
    if score >= 80:
        qty_pct = base_pct * 0.9
    elif score >= 70:
        qty_pct = base_pct * 0.7
    elif score >= 60:
        qty_pct = base_pct * 0.5
    else:
        qty_pct = base_pct * 0.3

    qty_pct = round(min(qty_pct, base_pct), 2)

    return {
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "rr_ratio": rr_ratio,
        "quantity_pct": qty_pct,
        "atr_pct": round(atr_pct, 2),
    }


async def refine_with_monte_carlo(
    sl_tp: dict,
    candles_1h: list,
    price: float,
) -> dict:
    """Refine SL/TP levels using GPU Monte Carlo simulation.

    Returns the original dict enriched with MC probability estimates,
    or the original dict unchanged if GPU is unavailable.
    """
    from app.services import gpu_client

    if not gpu_client.is_enabled() or not candles_1h or price <= 0:
        return sl_tp

    mc = await gpu_client.monte_carlo(
        candles=candles_1h,
        entry_price=price,
        stop_loss_pct=sl_tp["sl_pct"],
        take_profit_pct=sl_tp["tp_pct"],
        hours_ahead=48,
        simulations=10000,
    )
    if not mc or "error" in mc:
        return sl_tp

    sl_tp["mc_tp_prob"] = mc.get("tp_probability", 0)
    sl_tp["mc_sl_prob"] = mc.get("sl_probability", 0)
    sl_tp["mc_edge"] = mc.get("edge", 0)
    sl_tp["mc_rr"] = mc.get("reward_risk_ratio", 0)

    # If MC shows negative edge, tighten TP or widen SL
    edge = mc.get("edge", 0)
    if edge < -0.5 and sl_tp["tp_pct"] > sl_tp["sl_pct"] * 1.5:
        # Reduce TP to improve hit probability
        sl_tp["tp_pct"] = round(sl_tp["sl_pct"] * settings.MIN_REWARD_RISK_RATIO, 2)
        sl_tp["rr_ratio"] = round(sl_tp["tp_pct"] / sl_tp["sl_pct"], 2)
        sl_tp["mc_adjusted"] = True

    return sl_tp


# ── Main entry point: rank & build candidates ───────────────────────────────

def rank_symbols(
    symbols_data: dict[str, Any],
    news: dict[str, Any],
    ml_signals: dict[str, dict] | None = None,
    btc_anchor: dict | None = None,
    held_symbols: set[str] | None = None,
    market_regime: dict | None = None,
    market_intel: dict | None = None,
) -> list[TradeCandidate]:
    """Score all symbols and return ranked trade candidates.

    Only returns candidates that:
    1. Score >= MIN_QUANT_SCORE (configurable)
    2. Have valid ATR-based SL/TP
    3. Meet minimum R:R ratio
    4. Pass regime-aware filtering

    Also includes SELL candidates for held positions with bearish scores.
    """
    held = held_symbols or set()
    regime = (market_regime or {}).get("regime", "unknown")
    min_score = settings.MIN_QUANT_SCORE

    # Adjust threshold based on regime
    regime_adjustments = {
        "strong_uptrend":   -8,   # lower threshold = more trades
        "uptrend":          -4,
        "ranging":           0,
        "downtrend":        +5,   # higher threshold = fewer trades
        "strong_downtrend": +10,
        "choppy":           +8,
    }
    adjusted_min = min_score + regime_adjustments.get(regime, 0)

    # Fear & Greed adjustment: extreme greed → tighten, extreme fear → loosen
    intel = market_intel or {}
    fg = intel.get("fear_greed", {})
    fg_value = fg.get("value", 50)
    if fg_value >= 80:
        adjusted_min += 5    # extreme greed → be more selective
    elif fg_value <= 20:
        adjusted_min -= 5    # extreme fear → allow more trades (contrarian)

    # Less-fear mode: significantly lower threshold
    if settings.LESS_FEAR:
        adjusted_min = min(adjusted_min, 45)
        logger.info("Less-fear mode: threshold capped at %.0f", adjusted_min)

    # Extract per-symbol derivatives data
    funding_map = intel.get("funding", {})
    ls_map = intel.get("long_short", {})
    oi_map = intel.get("open_interest", {})

    # Whale data from real-time WebSocket detector
    from app.services.whale_detector import whale_detector
    whale_map = whale_detector.get_all_whale_data()

    candidates: list[TradeCandidate] = []

    for sym, data in symbols_data.items():
        result = score_symbol(
            sym, data,
            news.get(sym, {}),
            ml_signal=(ml_signals or {}).get(sym),
            btc_anchor=btc_anchor,
            is_held=sym in held,
            funding_data=funding_map.get(sym),
            ls_data=ls_map.get(sym),
            oi_data=oi_map.get(sym),
            whale_data=whale_map.get(sym),
        )

        score = result["score"]
        direction = result["direction"]
        signals = result["signals"]

        # BUY candidates: score must be above threshold
        if direction > 0 and score >= adjusted_min and sym not in held:
            levels = compute_trade_levels(sym, data, "BUY", score)
            if levels:
                price = data.get("price", 0)
                candidates.append(TradeCandidate(
                    symbol=sym,
                    action="BUY",
                    score=score,
                    timeframe="1h",
                    entry_price=price,
                    stop_loss_price=round(price * (1 - levels["sl_pct"] / 100), 6),
                    take_profit_price=round(price * (1 + levels["tp_pct"] / 100), 6),
                    stop_loss_pct=levels["sl_pct"],
                    take_profit_pct=levels["tp_pct"],
                    reward_risk_ratio=levels["rr_ratio"],
                    quantity_pct=levels["quantity_pct"],
                    signals=signals,
                    factor_scores=result["factors"],
                ))

        # SELL candidates: held positions with bearish score
        elif sym in held and score < 45:
            # For sells, we don't need SL/TP (position is being closed)
            candidates.append(TradeCandidate(
                symbol=sym,
                action="SELL",
                score=100 - score,  # invert: lower score → better sell signal
                timeframe="1h",
                entry_price=data.get("price", 0),
                stop_loss_price=0,
                take_profit_price=0,
                stop_loss_pct=0,
                take_profit_pct=0,
                reward_risk_ratio=0,
                quantity_pct=100,  # full close by default (Claude can partial)
                signals=signals,
                factor_scores=result["factors"],
            ))

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)

    if candidates:
        logger.info(
            "Quant scorer: %d candidates (min_score=%.0f, regime=%s) — top: %s",
            len(candidates), adjusted_min, regime,
            " | ".join(f"{c.symbol} {c.action} score={c.score:.0f} R:R={c.reward_risk_ratio:.1f}"
                       for c in candidates[:3]),
        )
    else:
        logger.info(
            "Quant scorer: 0 candidates (min_score=%.0f, regime=%s) — HOLD cycle",
            adjusted_min, regime,
        )

    return candidates
