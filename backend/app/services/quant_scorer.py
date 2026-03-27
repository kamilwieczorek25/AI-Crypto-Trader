"""Quantitative scoring engine — signal-first, backtestable symbol ranking.

Scores each symbol on a 0–100 scale using weighted technical factors.
Computes ATR-based stop-loss/take-profit levels with enforced minimum
reward-to-risk ratio. Produces structured trade candidates for Claude
to validate (not originate).

Factor set (27 total):
- breakout_signal       : price breaking above N-period high (strong momentum)
- vol_zscore_signal     : statistically unusual volume for THIS symbol
- macd_div_signal       : swing-based MACD divergence
- momentum_accel_signal : 2nd derivative of price (is momentum accelerating?)
- gpu_momentum_signal   : cross-sectional rank across 100+ symbols (GPU)
- sector_rotation_signal: sector-hot bonus from GPU clustering
- squeeze_signal        : short squeeze potential (neg funding + rising OI + rising price)
- beta_signal           : high-beta alts preferred when altseason detected
- news_burst_signal     : CryptoPanic catalyst velocity (N articles in 30min)

Runtime features:
- Regime-adaptive weight overrides (trending/ranging/choppy)
- Backtest-driven live weight nudging (±0.005 per factor per run)
- Session-aware threshold adjustment (Asian/US/dead-zone)
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# ── Base scoring weights (sum to 1.0, 27 factors) ────────────────────────────
# Existing factors trimmed slightly to make room for 3 new ones:
#   squeeze_signal (+0.03), beta_signal (+0.02), news_burst_signal (+0.03)
# Reductions: obv -0.01, vwap -0.01, depth -0.01, orderbook -0.01,
#             oi -0.01, ls_ratio -0.01, momentum_accel -0.01, sr -0.01
# pct_24h_signal (+0.04): 24h price change momentum — gainers score higher.
# Reductions for pct_24h: obv -0.01, depth -0.01, sr -0.01, momentum_accel -0.01
_WEIGHTS: dict[str, float] = {
    "rsi_signal":            0.06,   # RSI oversold/overbought
    "macd_signal":           0.05,   # MACD histogram direction + magnitude
    "macd_div_signal":       0.04,   # Swing-based MACD divergence
    "bb_signal":             0.04,   # Bollinger Band position
    "bb_squeeze_signal":     0.04,   # Bollinger squeeze (volatility contraction)
    "volume_signal":         0.05,   # Volume ratio (activity vs average)
    "vol_zscore_signal":     0.05,   # Volume Z-score — statistically unusual
    "obv_signal":            0.01,   # On-Balance Volume trend
    "vwap_signal":           0.01,   # Price vs VWAP
    "trend_signal":          0.05,   # Multi-timeframe trend alignment
    "sr_signal":             0.01,   # Support/resistance proximity
    "breakout_signal":       0.06,   # N-period high/low breakout
    "momentum_accel_signal": 0.02,   # Momentum acceleration 2nd derivative
    "btc_signal":            0.05,   # BTC anchor trend (critical for alts)
    "ml_signal":             0.08,   # LSTM + RL + GPU ensemble + MTF + anomaly
    "orderbook_signal":      0.01,   # Bid/ask pressure ratio
    "depth_signal":          0.01,   # Orderbook depth imbalance within 2%
    "whale_signal":          0.05,   # Whale trade flow (WebSocket real-time)
    "funding_signal":        0.04,   # Binance funding rate (contrarian)
    "ls_ratio_signal":       0.03,   # Long/short ratio (contrarian)
    "oi_signal":             0.03,   # Open interest trend
    "gpu_momentum_signal":   0.04,   # Cross-sectional GPU momentum rank
    "sector_rotation_signal":0.05,   # Sector rotation hot-bonus (GPU clusters)
    "squeeze_signal":        0.03,   # Short squeeze potential (new)
    "beta_signal":           0.02,   # Altcoin beta vs BTC (new)
    "news_burst_signal":     0.03,   # CryptoPanic catalyst velocity (new)
    "pct_24h_signal":        0.04,   # 24h price change momentum (gainers prioritised)
}
# Verify: sum(_WEIGHTS.values()) == 1.00  →  checked via test below

# ── Live weights (start as copy of _WEIGHTS; nudged by backtest analysis) ─────
# These are what `score_symbol` actually uses at runtime.
_live_weights: dict[str, float] = dict(_WEIGHTS)

# ── Regime-adaptive weight multipliers ───────────────────────────────────────
# Applied on top of _live_weights when a specific market regime is active.
# Values are multipliers (1.0 = unchanged).  Re-normalised to sum=1 at use time.
_REGIME_WEIGHT_OVERRIDES: dict[str, dict[str, float]] = {
    "strong_uptrend": {
        # Trending up: favour momentum / breakout / GPU ranking
        "breakout_signal":       2.0,
        "momentum_accel_signal": 1.8,
        "gpu_momentum_signal":   1.7,
        "trend_signal":          1.5,
        "sector_rotation_signal":1.5,
        "macd_signal":           1.4,
        # Dampen mean-reversion signals
        "bb_signal":             0.5,
        "rsi_signal":            0.6,
        "vwap_signal":           0.5,
        "sr_signal":             0.6,
    },
    "uptrend": {
        "breakout_signal":       1.5,
        "momentum_accel_signal": 1.4,
        "gpu_momentum_signal":   1.3,
        "trend_signal":          1.3,
    },
    "ranging": {
        # Range-bound: favour mean-reversion and support/resistance
        "rsi_signal":            1.8,
        "bb_signal":             1.8,
        "sr_signal":             1.6,
        "vwap_signal":           1.5,
        "macd_div_signal":       1.4,
        # Dampen momentum signals
        "breakout_signal":       0.5,
        "momentum_accel_signal": 0.5,
        "gpu_momentum_signal":   0.7,
    },
    "choppy": {
        # High volatility: trust volume/whale/squeeze above everything else
        "vol_zscore_signal":     2.0,
        "volume_signal":         1.7,
        "whale_signal":          1.8,
        "funding_signal":        1.5,
        "squeeze_signal":        1.8,
        # Penalise directional signals
        "breakout_signal":       0.3,
        "trend_signal":          0.4,
        "momentum_accel_signal": 0.4,
    },
    "downtrend": {
        # Falling market: only take very high-conviction setups
        "squeeze_signal":        1.8,   # squeezes still fire in downtrends
        "news_burst_signal":     1.4,   # catalyst breaks beat the trend
        "ml_signal":             1.3,
        # Dampen weak momentum signals
        "breakout_signal":       0.6,
        "trend_signal":          0.5,
    },
    "strong_downtrend": {
        "squeeze_signal":        2.0,
        "ml_signal":             1.5,
        "whale_signal":          1.4,
        # Strongly dampen momentum
        "breakout_signal":       0.2,
        "trend_signal":          0.2,
        "gpu_momentum_signal":   0.4,
    },
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

def _score_rsi(ind: dict) -> float:
    rsi = ind.get("rsi14", 50)
    if rsi <= 20:   return 1.0
    elif rsi <= 30: return 0.7
    elif rsi <= 35: return 0.3
    elif rsi >= 80: return -1.0
    elif rsi >= 70: return -0.7
    elif rsi >= 65: return -0.3
    return 0.0


def _score_macd(ind: dict) -> float:
    macd_hist = ind.get("macd_hist", 0)
    if macd_hist == 0:
        return 0.0
    magnitude = min(abs(macd_hist) * 50, 1.0)
    return magnitude if macd_hist > 0 else -magnitude


def _score_macd_divergence(ind: dict) -> float:
    """Swing-based MACD divergence — stronger signal than slope comparison."""
    return float(ind.get("macd_divergence", 0.0))


def _score_bb(ind: dict) -> float:
    bb = ind.get("bb_pct_b", 0.5)
    if bb <= 0.05:   return 1.0
    elif bb <= 0.15: return 0.7
    elif bb <= 0.25: return 0.3
    elif bb >= 0.95: return -1.0
    elif bb >= 0.85: return -0.7
    elif bb >= 0.75: return -0.3
    return 0.0


def _score_volume(ind: dict) -> float:
    vol_ratio = ind.get("volume_ratio", 1.0)
    trend_dir = ind.get("trend", 0)
    if vol_ratio < 0.5:
        return 0.0
    if vol_ratio > 2.0:
        return 0.8 * (1.0 if trend_dir > 0 else -0.5)
    elif vol_ratio > 1.5:
        return 0.5 * (1.0 if trend_dir > 0 else -0.3)
    elif vol_ratio > 1.2:
        return 0.2 * (1.0 if trend_dir > 0 else 0.0)
    return 0.0


def _score_vol_zscore(ind: dict) -> float:
    """Volume Z-score — statistically unusual volume for THIS symbol.

    A Z-score of 2.5 means volume is 2.5 standard deviations above its
    own recent mean, regardless of absolute size.  Much more reliable
    than comparing to a fixed floor.
    """
    z = ind.get("volume_zscore", 0.0)
    trend_dir = ind.get("trend", 0)
    if z >= 3.0:
        # Extreme — very likely driven by a catalyst
        return 0.9 * (1.0 if trend_dir >= 0 else -0.6)
    elif z >= 2.5:
        return 0.7 * (1.0 if trend_dir >= 0 else -0.4)
    elif z >= 1.5:
        return 0.4 * (1.0 if trend_dir >= 0 else -0.2)
    elif z <= -1.5:
        return -0.2  # volume drying up = be cautious
    return 0.0


def _score_obv(ind: dict) -> float:
    obv = ind.get("obv_trend", 0)
    trend = ind.get("trend", 0)
    if obv > 0 and trend > 0:   return 0.6
    elif obv > 0 and trend <= 0: return 0.4
    elif obv < 0 and trend < 0:  return -0.6
    elif obv < 0 and trend >= 0: return -0.4
    return 0.0


def _score_vwap(ind: dict) -> float:
    pv = ind.get("price_vs_vwap", 0)
    if pv < -2.0:   return 0.7
    elif pv < -0.5: return 0.3
    elif pv > 2.0:  return -0.7
    elif pv > 0.5:  return -0.3
    return 0.0


def _score_trend(indicators_all_tf: dict[str, dict]) -> float:
    if not indicators_all_tf:
        return 0.0
    tf_weights = {"15m": 0.5, "1h": 1.0, "4h": 1.5, "1d": 2.0}
    weighted_sum = 0.0
    weight_total = 0.0
    for tf, ind in indicators_all_tf.items():
        w = tf_weights.get(tf, 1.0)
        trend = ind.get("trend", 0)
        weighted_sum += trend * w
        weight_total += w
    if weight_total == 0:
        return 0.0
    return max(-1.0, min(1.0, weighted_sum / weight_total))


def _score_breakout(ind: dict) -> float:
    """Price breaking above N-period high = strong momentum signal.

    breakout_48h = +1 (above 48-bar high), -1 (below 48-bar low), 0 (inside).
    Combine with momentum acceleration for a stronger confirmation.
    """
    bo = ind.get("breakout_48h", 0.0)
    accel = ind.get("momentum_accel", 0.0)
    if bo > 0:
        # Breakout above prior range + accelerating momentum = very bullish
        return min(1.0, 0.7 + accel * 0.3)
    elif bo < 0:
        return max(-1.0, -0.7 + accel * 0.3)
    return 0.0


def _score_momentum_accel(ind: dict) -> float:
    """2nd derivative of price — is momentum speeding up or slowing down?

    Used to distinguish genuine breakouts from exhausted moves.
    """
    accel = ind.get("momentum_accel", 0.0)
    if accel > 0.5:   return 0.7
    elif accel > 0.2: return 0.4
    elif accel > 0.0: return 0.1
    elif accel < -0.5: return -0.5   # slight asymmetry: decelerating bear < accel bull
    elif accel < -0.2: return -0.3
    return 0.0


def _score_sr(sr: dict, price: float) -> float:
    if not sr or not price or price <= 0:
        return 0.0
    support = sr.get("nearest_support")
    resistance = sr.get("nearest_resistance")
    score = 0.0
    if support and support > 0:
        dist = (price - support) / price * 100
        if 0 < dist < 1.5:  score += 0.8
        elif 0 < dist < 3.0: score += 0.4
    if resistance and resistance > 0:
        dist = (resistance - price) / price * 100
        if 0 < dist < 1.5:  score -= 0.8
        elif 0 < dist < 3.0: score -= 0.4
    return max(-1.0, min(1.0, score))


def _score_btc(btc_anchor: dict | None) -> float:
    if not btc_anchor:
        return 0.0
    signals = []
    for tf in ("1h", "4h"):
        ind = btc_anchor.get(tf, {})
        if not ind:
            continue
        trend = ind.get("trend", 0)
        rsi = ind.get("rsi14", 50)
        if trend > 0 and rsi < 70:   signals.append(0.5)
        elif trend > 0:               signals.append(0.2)
        elif trend < 0 and rsi > 30:  signals.append(-0.7)
        elif trend < 0:               signals.append(-0.3)
        else:                         signals.append(0.0)
    return sum(signals) / len(signals) if signals else 0.0


def _score_ml(ml_signal: dict | None) -> float:
    if not ml_signal:
        return 0.0
    lstm = ml_signal.get("lstm", {})
    rl = ml_signal.get("rl", {})
    lstm_sig = lstm.get("signal", "HOLD")
    lstm_conf = lstm.get("confidence", 0)
    rl_sig = rl.get("action", "HOLD")
    score = 0.0
    if lstm_sig == rl_sig:
        if lstm_sig == "BUY":    score += min(lstm_conf, 1.0) * 0.4
        elif lstm_sig == "SELL": score -= min(lstm_conf, 1.0) * 0.4
    elif lstm_sig == "BUY" and lstm_conf > 0.6:  score += 0.1
    elif lstm_sig == "SELL" and lstm_conf > 0.6: score -= 0.1

    ens = ml_signal.get("ensemble", {})
    ens_sig, ens_conf, ens_agree = ens.get("signal", "HOLD"), ens.get("confidence", 0), ens.get("agreement", 0)
    if ens_sig == "BUY" and ens_agree >= 0.75:    score += min(ens_conf, 1.0) * 0.3
    elif ens_sig == "SELL" and ens_agree >= 0.75: score -= min(ens_conf, 1.0) * 0.3

    mtf = ml_signal.get("mtf", {})
    mtf_sig, mtf_conf = mtf.get("signal", "HOLD"), mtf.get("confidence", 0)
    if mtf_sig == "BUY" and mtf_conf >= 0.6:    score += min(mtf_conf, 1.0) * 0.2
    elif mtf_sig == "SELL" and mtf_conf >= 0.6: score -= min(mtf_conf, 1.0) * 0.2

    if ml_signal.get("anomaly", {}).get("is_anomaly"):
        score -= 0.5

    if ml_signal.get("corr_divergence"):
        score += 0.15

    return max(-1.0, min(1.0, score))


def _score_orderbook(ob: dict) -> float:
    pressure = ob.get("pressure_ratio", 1.0)
    if pressure > 2.0:   return 0.7
    elif pressure > 1.5: return 0.4
    elif pressure > 1.2: return 0.2
    elif pressure < 0.5:  return -0.7
    elif pressure < 0.67: return -0.4
    elif pressure < 0.83: return -0.2
    return 0.0


def _score_depth(ob: dict) -> float:
    imbalance = ob.get("depth_imbalance", 0.0)
    if imbalance > 0.5:   return 0.7
    elif imbalance > 0.3: return 0.4
    elif imbalance > 0.15: return 0.2
    elif imbalance < -0.5:  return -0.7
    elif imbalance < -0.3:  return -0.4
    elif imbalance < -0.15: return -0.2
    return 0.0


def _score_whale(whale_data: dict | None) -> float:
    if not whale_data:
        return 0.0
    net_flow = whale_data.get("whale_net_flow", 0)
    total_vol = whale_data.get("whale_total_volume", 0)
    if total_vol <= 0:
        return 0.0
    bias = net_flow / total_vol
    if bias > 0.5:    return 0.7
    elif bias > 0.2:  return 0.4
    elif bias > 0:    return 0.2
    elif bias < -0.5: return -0.7
    elif bias < -0.2: return -0.4
    elif bias < 0:    return -0.2
    return 0.0


def _score_bb_squeeze(ind: dict) -> float:
    squeeze = ind.get("bb_squeeze", 0)
    if not squeeze:
        return 0.0
    trend = ind.get("trend", 0)
    if trend > 0:   return 0.7
    elif trend < 0: return -0.7
    return 0.3


def _score_oi(oi_data: dict | None) -> float:
    if not oi_data:
        return 0.0
    oi_change = oi_data.get("change_pct", 0)
    price_trend = oi_data.get("price_trend", 0)
    if oi_change > 5:
        return 0.6 if price_trend > 0 else -0.5
    elif oi_change < -5:
        return 0.2 if price_trend > 0 else -0.2
    return 0.0


def _score_funding(funding_data: dict | None) -> float:
    if not funding_data:
        return 0.0
    rate = funding_data.get("rate", 0)
    if rate > 0.001:       return -0.7
    elif rate > 0.0005:    return -0.3
    elif rate < -0.001:    return 0.7
    elif rate < -0.0005:   return 0.3
    return 0.0


def _score_ls_ratio(ls_data: dict | None) -> float:
    if not ls_data:
        return 0.0
    ratio = ls_data.get("ratio", 1.0)
    if ratio > 2.0:    return -0.7
    elif ratio > 1.5:  return -0.4
    elif ratio > 1.2:  return -0.15
    elif ratio < 0.5:  return 0.7
    elif ratio < 0.67: return 0.4
    elif ratio < 0.83: return 0.15
    return 0.0


def _score_gpu_momentum(symbol: str, momentum_ranks: dict | None) -> float:
    """Cross-sectional GPU momentum rank.

    momentum_ranks is a dict {symbol: percentile_0_to_1} computed by the
    /rank/momentum endpoint on the GPU server.  Top decile = +1, bottom = -1.
    """
    if not momentum_ranks:
        return 0.0
    pct = momentum_ranks.get(symbol, 0.5)
    if pct >= 0.90:   return 0.9    # top 10% of all symbols
    elif pct >= 0.80: return 0.6
    elif pct >= 0.70: return 0.3
    elif pct <= 0.10: return -0.9   # bottom 10%
    elif pct <= 0.20: return -0.5
    elif pct <= 0.30: return -0.2
    return 0.0


def _score_sector_rotation(symbol: str, sector_heat: dict | None) -> float:
    """Sector rotation bonus — is this coin's sector currently hot?

    sector_heat: {symbol: heat_score_-1_to_1} from /cluster/rotation.
    Heat = weighted average momentum of all symbols in the same GPU cluster.
    """
    if not sector_heat:
        return 0.0
    heat = sector_heat.get(symbol, 0.0)
    if heat >= 0.6:   return 0.8
    elif heat >= 0.4: return 0.5
    elif heat >= 0.2: return 0.2
    elif heat <= -0.6: return -0.6
    elif heat <= -0.4: return -0.3
    return 0.0


def _score_squeeze(symbol: str, squeeze_data: dict | None) -> float:
    """Short squeeze potential from combined funding + OI + price trend.

    High squeeze potential = crowd is heavily short AND price is rising AND
    open interest is building → forced covering can produce rapid spikes.
    """
    if not squeeze_data:
        return 0.0
    sq = squeeze_data.get(symbol, {})
    score = sq.get("squeeze_score", 0.0)
    # Map 0-1 squeeze score to -1..+1 signal (pure bullish — squeezes are upside)
    if score >= 0.65:   return 0.9
    elif score >= 0.40: return 0.6
    elif score >= 0.20: return 0.3
    return 0.0


def _score_beta(symbol: str, beta_data: dict | None, btc_dominance: dict | None) -> float:
    """Beta vs BTC signal — context-dependent.

    During altseason (BTC.D falling): prefer HIGH beta alts (amplified upside)
    During BTC dominance (BTC.D rising): prefer LOW beta alts (defensive)
    In neutral conditions: slight preference for moderate beta (1.0–1.8)
    """
    if not beta_data:
        return 0.0

    beta_info = beta_data.get(symbol, {})
    beta = beta_info.get("beta", 1.0)
    corr  = beta_info.get("correlation", 0.5)

    # Low correlation = idiosyncratic mover (could be good or bad — neutral)
    if abs(corr) < 0.3:
        return 0.1  # slight bonus for decorrelated assets

    dom = btc_dominance or {}
    dom_signal = dom.get("signal", "neutral")

    if dom_signal == "altseason":
        # Want high-beta alts for maximum alt upside
        if beta >= 2.0:   return 0.8
        elif beta >= 1.5: return 0.6
        elif beta >= 1.0: return 0.3
        elif beta < 0.5:  return -0.3
    elif dom_signal == "btc_dominance":
        # BTC taking market share — prefer low-beta (defensive) or short high-beta
        if beta >= 2.0:   return -0.6
        elif beta >= 1.5: return -0.3
        elif beta <= 0.7: return 0.4  # defensive, low corr
        elif beta <= 1.0: return 0.2
    else:
        # Neutral: moderate beta is fine
        if 0.8 <= beta <= 1.8:   return 0.2
        elif beta > 2.5:         return -0.2  # too volatile
    return 0.0


def _score_news_burst(symbol: str, news_data: dict | None) -> float:
    """CryptoPanic news burst signal — catalyst velocity detector.

    A sudden spike in article count (N articles in 30 min) often precedes
    a significant price move as retail and bots react to the news.

    Uses news_data already augmented by fetch_cryptopanic_news().
    """
    if not news_data:
        return 0.0
    nd = news_data.get(symbol, {})
    burst = nd.get("news_burst", False)
    count = nd.get("news_count_30m", 0)
    # Also factor in existing average sentiment for direction
    avg_sent = nd.get("avg_sentiment", 0.0)

    if burst and count >= 6:
        # Very high velocity — something significant is happening
        direction = 1.0 if avg_sent >= 0 else -0.5
        return 0.9 * direction
    elif burst and count >= 3:
        direction = 1.0 if avg_sent >= 0 else -0.4
        return 0.6 * direction
    elif count == 2:
        # Building — not quite a burst yet
        return 0.2 if avg_sent > 0 else 0.0
    return 0.0


def _score_pct_24h(pct_24h: float) -> float:
    """24-hour price change momentum signal.

    Strong gainers (+10%+) are more likely to be in a trending move.
    Strong losers (-10%+) are bearish — only enter on squeeze/reversal setups.
    Moderate positive moves (+5-10%) get a mild boost; flat = neutral.
    """
    if pct_24h >= 20:    return 0.9   # explosive gainer — very bullish
    elif pct_24h >= 10:  return 0.7   # strong uptrend
    elif pct_24h >= 5:   return 0.4   # moderate gainer
    elif pct_24h >= 2:   return 0.2   # mild positive drift
    elif pct_24h <= -20: return -0.8  # severe dump
    elif pct_24h <= -10: return -0.6  # strong downtrend
    elif pct_24h <= -5:  return -0.3  # moderate decline
    elif pct_24h <= -2:  return -0.1  # mild negative drift
    return 0.0


def _get_regime_weights(regime: str) -> dict[str, float]:
    """Return live weights adjusted for current market regime.

    Applies _REGIME_WEIGHT_OVERRIDES multipliers on top of _live_weights,
    then re-normalises so the result still sums to 1.0.
    """
    overrides = _REGIME_WEIGHT_OVERRIDES.get(regime, {})
    if not overrides:
        return dict(_live_weights)

    adjusted = {
        k: v * overrides.get(k, 1.0)
        for k, v in _live_weights.items()
    }
    total = sum(adjusted.values())
    if total <= 0:
        return dict(_live_weights)
    return {k: v / total for k, v in adjusted.items()}


def nudge_weights_from_backtest(factor_win_avg: dict[str, float],
                                factor_loss_avg: dict[str, float],
                                nudge_step: float = 0.005) -> None:
    """Adjust _live_weights based on backtest factor predictiveness.

    For each factor, computes (win_avg - loss_avg).  If a factor is
    consistently higher in winning trades, increase its weight.  If it's
    higher in losing trades, decrease it.  Changes are capped at ±nudge_step
    per call and the result is re-normalised so weights still sum to 1.0.

    Called by `_run_startup_backtest` in bot_runner.py after each backtest.
    """
    if not factor_win_avg or not factor_loss_avg:
        return

    deltas: dict[str, float] = {}
    for k in _live_weights:
        win_val  = factor_win_avg.get(k, 0.0)
        loss_val = factor_loss_avg.get(k, 0.0)
        diff = win_val - loss_val
        # Positive diff = factor was higher in wins → increase weight
        # Use tanh to smooth large outliers
        import math
        delta = math.tanh(diff * 5) * nudge_step
        deltas[k] = delta

    # Apply deltas and clamp each weight to [0.005, 0.20]
    for k in _live_weights:
        _live_weights[k] = max(0.005, min(0.20, _live_weights[k] + deltas.get(k, 0)))

    # Re-normalise
    total = sum(_live_weights.values())
    if total > 0:
        for k in _live_weights:
            _live_weights[k] = round(_live_weights[k] / total, 5)

    logger.info(
        "Weights nudged from backtest: top movers=%s",
        sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:4],
    )


# ── Composite scoring ─────────────────────────────────────────────────────────

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
    momentum_ranks: dict | None = None,
    sector_heat: dict | None = None,
    # New parameters for workstream-3 signals:
    squeeze_data: dict | None = None,     # {symbol: {squeeze_score, signal, reasons}}
    beta_data: dict | None = None,        # {symbol: {beta, correlation, r_squared}}
    btc_dominance: dict | None = None,    # {"btc_dominance": float, "signal": str, ...}
    market_regime: str = "unknown",       # regime string for weight selection
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
    pct_24h = symbol_data.get("pct_24h", 0.0)

    # Select regime-adjusted weights (re-normalised to sum=1 inside _get_regime_weights)
    weights = _get_regime_weights(market_regime)

    # Compute each factor
    factors: dict[str, float] = {
        "rsi_signal":            _score_rsi(ind_1h),
        "macd_signal":           _score_macd(ind_1h),
        "macd_div_signal":       _score_macd_divergence(ind_1h),
        "bb_signal":             _score_bb(ind_1h),
        "bb_squeeze_signal":     _score_bb_squeeze(ind_1h),
        "volume_signal":         _score_volume(ind_1h),
        "vol_zscore_signal":     _score_vol_zscore(ind_1h),
        "obv_signal":            _score_obv(ind_1h),
        "vwap_signal":           _score_vwap(ind_1h),
        "trend_signal":          _score_trend(indicators),
        "sr_signal":             _score_sr(sr, price),
        "breakout_signal":       _score_breakout(ind_1h),
        "momentum_accel_signal": _score_momentum_accel(ind_1h),
        "btc_signal":            _score_btc(btc_anchor),
        "ml_signal":             _score_ml(ml_signal),
        "orderbook_signal":      _score_orderbook(ob),
        "depth_signal":          _score_depth(ob),
        "whale_signal":          _score_whale(whale_data),
        "funding_signal":        _score_funding(funding_data),
        "ls_ratio_signal":       _score_ls_ratio(ls_data),
        "oi_signal":             _score_oi(oi_data),
        "gpu_momentum_signal":   _score_gpu_momentum(symbol, momentum_ranks),
        "sector_rotation_signal":_score_sector_rotation(symbol, sector_heat),
        # New signals (workstream 3)
        "squeeze_signal":        _score_squeeze(symbol, squeeze_data),
        "beta_signal":           _score_beta(symbol, beta_data, btc_dominance),
        "news_burst_signal":     _score_news_burst(symbol, news_data),
        # 24h price change momentum
        "pct_24h_signal":        _score_pct_24h(pct_24h),
    }

    # RSI divergence adds bonus to RSI score
    rsi_div = ind_1h.get("rsi_divergence", 0)
    if rsi_div > 0:
        factors["rsi_signal"] = min(1.0, factors["rsi_signal"] + 0.3)
    elif rsi_div < 0:
        factors["rsi_signal"] = max(-1.0, factors["rsi_signal"] - 0.3)

    # Weighted composite using regime-adjusted weights: sum(factor * weight) → [-1, +1]
    composite = sum(factors[k] * weights.get(k, 0.0) for k in factors)

    # ── Range-bound / low-conviction filter ──────────────────────────
    # Count how many factors have a meaningful opinion (|value| >= 0.2)
    # If too few factors are active, the signal is just noise.
    active_bullish = sum(1 for v in factors.values() if v >= 0.2)
    active_bearish = sum(1 for v in factors.values() if v <= -0.2)
    active_total = active_bullish + active_bearish

    # Require at least 3 active factors for any directional signal;
    # otherwise force neutral — prevents buying flat/range-bound coins
    if active_total < 3 and abs(composite) < 0.15:
        composite = 0.0

    # Require at least 2 bullish factors for a BUY signal — one factor alone
    # (e.g., single MACD divergence) is not enough conviction to buy
    if composite > 0 and active_bullish < 2:
        composite = 0.0

    # Normalize to 0–100 scale (50 = neutral)
    score = max(0, min(100, 50 + composite * 50))

    direction = 1 if composite > 0 else (-1 if composite < 0 else 0)

    # Build signal descriptions for the top contributing factors
    signals: list[str] = []
    sorted_factors = sorted(factors.items(), key=lambda x: abs(x[1]), reverse=True)
    for fname, fval in sorted_factors[:7]:  # show top 7 for 27-factor model
        if abs(fval) < 0.1:
            continue
        label = fname.replace("_signal", "").upper()
        direction_str = "bullish" if fval > 0 else "bearish"
        signals.append(f"{label}={fval:+.2f}({direction_str})")

    return {
        "score":     round(score, 1),
        "composite": round(composite, 4),
        "direction": direction,
        "factors":   {k: round(v, 3) for k, v in factors.items()},
        "signals":   signals,
        "regime_weights_used": market_regime,
        "conviction": {"bullish": active_bullish, "bearish": active_bearish, "total": active_total},
    }


# ── ATR-based SL/TP computation ──────────────────────────────────────────────

def compute_trade_levels(
    symbol: str,
    symbol_data: dict[str, Any],
    action: str,
    score: float,
) -> dict[str, float] | None:
    """Compute stop-loss and take-profit using ATR, enforcing min R:R ratio."""
    indicators = symbol_data.get("indicators", {})
    price = symbol_data.get("price", 0)
    if not price or price <= 0:
        return None

    atr_1h = indicators.get("1h", {}).get("atr", 0)
    atr_4h = indicators.get("4h", {}).get("atr", 0)

    if atr_1h <= 0:
        atr_1h = price * 0.02

    atr_pct = (atr_1h / price) * 100

    # Reject coins with ATR < 0.5% — too flat, SL/TP will whipsaw
    if atr_pct < 0.5 and action == "BUY":
        logger.debug("Rejecting %s: ATR %.2f%% too low (range-bound)", symbol, atr_pct)
        return None

    sl_multiplier = settings.SL_ATR_MULTIPLIER
    sl_pct = round(atr_pct * sl_multiplier, 2)
    sl_pct = max(settings.MIN_SL_PCT, min(settings.MAX_SL_PCT, sl_pct))

    min_rr = settings.MIN_REWARD_RISK_RATIO
    tp_pct = round(sl_pct * min_rr, 2)

    if score >= 80:
        tp_pct = round(tp_pct * 1.5, 2)
    elif score >= 70:
        tp_pct = round(tp_pct * 1.2, 2)

    tp_pct = max(tp_pct, sl_pct * min_rr)
    tp_pct = min(tp_pct, 30.0)

    rr_ratio = round(tp_pct / sl_pct, 2) if sl_pct > 0 else 0

    if rr_ratio < min_rr:
        return None

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
    """Refine SL/TP levels using GPU Monte Carlo simulation."""
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

    edge = mc.get("edge", 0)
    if edge < -0.5 and sl_tp["tp_pct"] > sl_tp["sl_pct"] * 1.5:
        sl_tp["tp_pct"] = round(sl_tp["sl_pct"] * settings.MIN_REWARD_RISK_RATIO, 2)
        sl_tp["rr_ratio"] = round(sl_tp["tp_pct"] / sl_tp["sl_pct"], 2)
        sl_tp["mc_adjusted"] = True

    return sl_tp


# ── Session-aware threshold helpers ──────────────────────────────────────────

def _session_adjustment() -> float:
    """Return a score threshold delta based on current UTC hour.

    Altcoin behaviour varies significantly by session:
    - Asian session (00:00–08:00 UTC): alts decouple from BTC, more pump activity
      → lower threshold (more opportunities)
    - London/EU open (07:00–10:00 UTC): overlap creates volatility → neutral
    - US session (13:00–22:00 UTC): narrative-driven, high liquidity → neutral
    - Dead zone (22:00–00:00 UTC): thin liquidity, false signals more common
      → higher threshold (be more selective)
    - Weekends: thin markets, higher chance of fake breakouts → +3 adjustment
    """
    now = datetime.now(timezone.utc)
    hour = now.hour
    weekday = now.weekday()  # 0=Mon, 6=Sun

    adjustment = 0.0

    # Weekend premium: thin liquidity = more false signals
    if weekday >= 5:
        adjustment += 3.0

    # Dead-zone penalty: very thin market
    if 22 <= hour or hour < 1:
        adjustment += 2.0
    # Asian session bonus: alts are more active
    elif 1 <= hour < 8:
        adjustment -= 3.0
    # US open overlap with EU close (high volume)
    elif 13 <= hour < 17:
        adjustment -= 1.0

    return adjustment


# ── Main entry point: rank & build candidates ────────────────────────────────

def rank_symbols(
    symbols_data: dict[str, Any],
    news: dict[str, Any],
    ml_signals: dict[str, dict] | None = None,
    btc_anchor: dict | None = None,
    held_symbols: set[str] | None = None,
    market_regime: dict | None = None,
    market_intel: dict | None = None,
    momentum_ranks: dict | None = None,
    sector_heat: dict | None = None,
    beta_data: dict | None = None,
) -> list[TradeCandidate]:
    """Score all symbols and return ranked trade candidates.

    Only returns candidates that:
    1. Score >= MIN_QUANT_SCORE (configurable, regime + session adjusted)
    2. Have valid ATR-based SL/TP
    3. Meet minimum R:R ratio
    4. Pass regime-aware filtering

    Also includes SELL candidates for held positions with bearish scores.
    """
    held = held_symbols or set()
    regime_dict = market_regime or {}
    regime = regime_dict.get("regime", "unknown")
    min_score = settings.MIN_QUANT_SCORE

    # Regime-aware threshold adjustment
    regime_adjustments = {
        "strong_uptrend":   -8,
        "uptrend":          -4,
        "ranging":           0,
        "downtrend":        +5,
        "strong_downtrend": +10,
        "choppy":           +8,
    }
    adjusted_min = min_score + regime_adjustments.get(regime, 0)

    # Fear & Greed adjustment
    intel = market_intel or {}
    fg = intel.get("fear_greed", {})
    fg_value = fg.get("value", 50)
    if fg_value >= 80:
        adjusted_min += 5
    elif fg_value <= 20:
        adjusted_min -= 5

    # Session-aware adjustment (new)
    session_delta = _session_adjustment()
    adjusted_min += session_delta
    if session_delta != 0:
        logger.debug("Session adjustment: %+.0f pts (threshold now %.0f)", session_delta, adjusted_min)

    # Less-fear mode override
    if settings.LESS_FEAR:
        adjusted_min = min(adjusted_min, 45)
        logger.info("Less-fear mode: threshold capped at %.0f", adjusted_min)

    # Extract per-symbol derivatives data (including new squeeze + dominance)
    funding_map   = intel.get("funding", {})
    ls_map        = intel.get("long_short", {})
    oi_map        = intel.get("open_interest", {})
    squeeze_map   = intel.get("squeeze", {})
    btc_dominance = intel.get("btc_dominance", {})

    # BTC dominance adjustment to threshold:
    # Altseason = lower threshold (more alt opportunities worth chasing)
    # BTC dominance = raise threshold (alts underperform, be selective)
    dom_signal = btc_dominance.get("signal", "neutral")
    if dom_signal == "altseason":
        adjusted_min -= 2
        logger.debug("Altseason detected (BTC.D=%.1f%%) — threshold -2", btc_dominance.get("btc_dominance", 0))
    elif dom_signal == "btc_dominance":
        adjusted_min += 3
        logger.debug("BTC dominance (BTC.D=%.1f%%) — threshold +3", btc_dominance.get("btc_dominance", 0))

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
            momentum_ranks=momentum_ranks,
            sector_heat=sector_heat,
            squeeze_data=squeeze_map or None,
            beta_data=beta_data,
            btc_dominance=btc_dominance,
            market_regime=regime,
        )

        score = result["score"]
        direction = result["direction"]
        signals = result["signals"]

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

        elif sym in held and score < 45:
            candidates.append(TradeCandidate(
                symbol=sym,
                action="SELL",
                score=100 - score,
                timeframe="1h",
                entry_price=data.get("price", 0),
                stop_loss_price=0,
                take_profit_price=0,
                stop_loss_pct=0,
                take_profit_pct=0,
                reward_risk_ratio=0,
                quantity_pct=100,
                signals=signals,
                factor_scores=result["factors"],
            ))

    candidates.sort(key=lambda c: c.score, reverse=True)

    if candidates:
        logger.info(
            "Quant scorer: %d candidates (min_score=%.0f, regime=%s, session_adj=%+.0f) — top: %s",
            len(candidates), adjusted_min, regime, session_delta,
            " | ".join(f"{c.symbol} {c.action} score={c.score:.0f} R:R={c.reward_risk_ratio:.1f}"
                       for c in candidates[:3]),
        )
    else:
        logger.info(
            "Quant scorer: 0 candidates (min_score=%.0f, regime=%s, session_adj=%+.0f) — HOLD cycle",
            adjusted_min, regime, session_delta,
        )

    return candidates
