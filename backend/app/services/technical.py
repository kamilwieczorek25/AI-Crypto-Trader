"""Technical analysis service — pandas-ta indicators."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import pandas_ta as ta  # type: ignore[import]
    _HAS_PANDAS_TA = True
except ImportError:
    _HAS_PANDAS_TA = False
    logger.warning("pandas_ta not installed — technical indicators will be zeroed")


def _ohlcv_to_df(ohlcv: list[list[float]]) -> pd.DataFrame:
    if not ohlcv:
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def _safe_float(val: object) -> float:
    try:
        f = float(val)  # type: ignore[arg-type]
        return f if not np.isnan(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def detect_market_regime(ohlcv_data: dict[str, dict[str, list]]) -> dict:
    """Detect overall market regime from aggregated symbol data.

    Returns {'regime': str, 'volatility': str, 'description': str}
    """
    if not ohlcv_data:
        return {"regime": "unknown", "volatility": "unknown", "description": "No data"}

    returns = []
    volatilities = []
    for sym, tf_data in ohlcv_data.items():
        candles = tf_data.get("1h", [])
        if len(candles) < 20:
            continue
        closes = np.array([c[4] for c in candles[-20:]])
        ret = (closes[-1] - closes[0]) / (closes[0] + 1e-10)
        returns.append(ret)
        daily_rets = np.diff(closes) / (closes[:-1] + 1e-10)
        volatilities.append(np.std(daily_rets))

    if not returns:
        return {"regime": "unknown", "volatility": "unknown", "description": "Insufficient data"}

    avg_return = np.mean(returns)
    avg_vol = np.mean(volatilities)

    # Classify regime
    if avg_return > 0.02 and avg_vol < 0.03:
        regime = "strong_uptrend"
        desc = "Strong uptrend with low volatility — favor momentum BUY strategies"
    elif avg_return > 0.01:
        regime = "uptrend"
        desc = "Moderate uptrend — balanced approach, lean toward BUY"
    elif avg_return < -0.02 and avg_vol < 0.03:
        regime = "strong_downtrend"
        desc = "Strong downtrend — prioritize SELL open positions and capital preservation"
    elif avg_return < -0.01:
        regime = "downtrend"
        desc = "Moderate downtrend — be cautious, tighter stops"
    elif avg_vol > 0.04:
        regime = "choppy"
        desc = "High volatility / choppy — avoid trading unless very high confidence"
    else:
        regime = "ranging"
        desc = "Range-bound market — look for RSI extremes and mean reversion"

    # Classify volatility
    if avg_vol > 0.04:
        vol_label = "high"
    elif avg_vol > 0.02:
        vol_label = "medium"
    else:
        vol_label = "low"

    return {
        "regime": regime,
        "volatility": vol_label,
        "avg_return_pct": round(avg_return * 100, 2),
        "avg_volatility_pct": round(avg_vol * 100, 2),
        "description": desc,
    }


def compute_correlation_matrix(ohlcv_data: dict[str, dict[str, list]], top_n: int = 10) -> dict:
    """Compute simple correlation info among top symbols.

    Returns info about highly correlated clusters.
    """
    closes_map: dict[str, np.ndarray] = {}
    for sym, tf_data in ohlcv_data.items():
        candles = tf_data.get("1h", [])
        if len(candles) >= 20:
            closes_map[sym] = np.array([c[4] for c in candles[-20:]])

    symbols = list(closes_map.keys())[:top_n]
    if len(symbols) < 2:
        return {"clusters": [], "high_correlation_pairs": []}

    # Compute returns
    returns_matrix = []
    for sym in symbols:
        c = closes_map[sym]
        rets = np.diff(c) / (c[:-1] + 1e-10)
        returns_matrix.append(rets)

    # Find highly correlated pairs (> 0.8)
    n = len(symbols)
    high_corr_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            min_len = min(len(returns_matrix[i]), len(returns_matrix[j]))
            if min_len < 5:
                continue
            corr = np.corrcoef(returns_matrix[i][:min_len], returns_matrix[j][:min_len])[0, 1]
            if abs(corr) > 0.8:
                high_corr_pairs.append({
                    "pair": [symbols[i], symbols[j]],
                    "correlation": round(float(corr), 3),
                })

    return {
        "high_correlation_pairs": high_corr_pairs[:10],
        "num_symbols_analyzed": len(symbols),
    }


def compute_indicators(ohlcv: list[list[float]]) -> dict[str, float]:
    """Return a dict of TA indicators for a single timeframe's OHLCV data."""
    df = _ohlcv_to_df(ohlcv)
    if df.empty or len(df) < 30:
        return _empty_indicators(status="insufficient_data")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    indicators: dict[str, float] = {}

    if _HAS_PANDAS_TA:
        # RSI 14
        rsi = ta.rsi(close, length=14)
        indicators["rsi14"] = _safe_float(rsi.iloc[-1] if rsi is not None else None)

        # MACD histogram (12, 26, 9)
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            hist_col = [c for c in macd_df.columns if "h" in c.lower()]
            indicators["macd_hist"] = _safe_float(
                macd_df[hist_col[0]].iloc[-1] if hist_col else None
            )
        else:
            indicators["macd_hist"] = 0.0

        # Bollinger Bands %B + Squeeze detection
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None and not bb.empty:
            pct_col = [c for c in bb.columns if "p" in c.lower()]
            indicators["bb_pct_b"] = _safe_float(
                bb[pct_col[0]].iloc[-1] if pct_col else None
            )
            # BB Squeeze: bandwidth = (upper - lower) / middle
            upper_col = [c for c in bb.columns if "u" in c.lower()]
            lower_col = [c for c in bb.columns if "l" in c.lower() and "p" not in c.lower()]
            mid_col = [c for c in bb.columns if "m" in c.lower()]
            if upper_col and lower_col and mid_col:
                bbu = bb[upper_col[0]]
                bbl = bb[lower_col[0]]
                bbm = bb[mid_col[0]]
                bw = ((bbu - bbl) / bbm.replace(0, np.nan)).dropna()
                if len(bw) >= 20:
                    current_bw = _safe_float(bw.iloc[-1])
                    avg_bw = _safe_float(bw.rolling(20).mean().iloc[-1])
                    # Squeeze = bandwidth is <60% of its 20-period average
                    indicators["bb_squeeze"] = 1.0 if (avg_bw > 0 and current_bw < avg_bw * 0.6) else 0.0
                    indicators["bb_bandwidth"] = current_bw
                else:
                    indicators["bb_squeeze"] = 0.0
                    indicators["bb_bandwidth"] = 0.0
            else:
                indicators["bb_squeeze"] = 0.0
                indicators["bb_bandwidth"] = 0.0
        else:
            indicators["bb_pct_b"] = 0.5
            indicators["bb_squeeze"] = 0.0
            indicators["bb_bandwidth"] = 0.0

        # EMA 20
        ema20 = ta.ema(close, length=20)
        indicators["ema20"] = _safe_float(ema20.iloc[-1] if ema20 is not None else None)

        # ATR 14
        atr = ta.atr(high, low, close, length=14)
        indicators["atr"] = _safe_float(atr.iloc[-1] if atr is not None else None)
    else:
        # Fallback: basic RSI via pure pandas
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_val = 100 - (100 / (1 + rs))
        indicators["rsi14"] = _safe_float(rsi_val.iloc[-1])
        indicators["macd_hist"] = 0.0
        indicators["bb_pct_b"] = 0.5
        indicators["bb_squeeze"] = 0.0
        indicators["bb_bandwidth"] = 0.0
        indicators["ema20"] = _safe_float(close.ewm(span=20).mean().iloc[-1])
        indicators["atr"] = _safe_float(
            (high - low).rolling(14).mean().iloc[-1]
        )

    # Volume ratio (current vs 20-bar average)
    vol_avg = volume.rolling(20).mean().iloc[-1]
    indicators["volume_ratio"] = _safe_float(
        volume.iloc[-1] / vol_avg if vol_avg and vol_avg != 0 else 1.0
    )

    # Volume trend (is volume increasing over last 5 bars?)
    if len(volume) >= 10:
        vol_recent = volume.iloc[-5:].mean()
        vol_prior = volume.iloc[-10:-5].mean()
        indicators["volume_trend"] = _safe_float(
            vol_recent / vol_prior if vol_prior > 0 else 1.0
        )
    else:
        indicators["volume_trend"] = 1.0

    # OBV (On-Balance Volume) — normalised: direction of cumulative volume pressure
    obv = (np.sign(close.diff()) * volume).cumsum()
    if len(obv) >= 20:
        obv_ema = obv.ewm(span=20).mean()
        indicators["obv_trend"] = 1.0 if obv.iloc[-1] > obv_ema.iloc[-1] else -1.0
    else:
        indicators["obv_trend"] = 0.0

    # VWAP (Volume Weighted Average Price) — intraday anchor
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    indicators["vwap"] = _safe_float(vwap.iloc[-1])
    # Price vs VWAP: >0 means above VWAP (bullish), <0 means below (bearish)
    current_close = _safe_float(close.iloc[-1])
    if indicators["vwap"] > 0 and current_close:
        indicators["price_vs_vwap"] = round(
            (current_close - indicators["vwap"]) / indicators["vwap"] * 100, 3
        )
    else:
        indicators["price_vs_vwap"] = 0.0

    # Current price and trend label (multi-candle: 3 of last 5 closes above EMA20)
    indicators["close"] = current_close
    ema20_val = indicators.get("ema20", 0.0)

    if len(close) >= 5 and ema20_val > 0:
        above_count = sum(1 for c in close.iloc[-5:] if c > ema20_val)
        indicators["trend"] = 1.0 if above_count >= 3 else -1.0
    else:
        indicators["trend"] = 1.0 if current_close > ema20_val else -1.0

    # RSI divergence detection (price vs RSI direction over last 10 bars)
    if _HAS_PANDAS_TA and len(close) >= 14:
        rsi_series = ta.rsi(close, length=14)
        if rsi_series is not None and len(rsi_series) >= 10:
            price_slope = close.iloc[-1] - close.iloc[-10]
            rsi_slope = rsi_series.iloc[-1] - rsi_series.iloc[-10]
            # Bullish divergence: price down but RSI up
            if price_slope < 0 and rsi_slope > 0:
                indicators["rsi_divergence"] = 1.0  # bullish
            # Bearish divergence: price up but RSI down
            elif price_slope > 0 and rsi_slope < 0:
                indicators["rsi_divergence"] = -1.0  # bearish
            else:
                indicators["rsi_divergence"] = 0.0
        else:
            indicators["rsi_divergence"] = 0.0
    else:
        indicators["rsi_divergence"] = 0.0

    indicators["data_status"] = 1.0  # signals valid data
    return indicators


def _empty_indicators(status: str = "ok") -> dict[str, float]:
    return {
        "rsi14": 50.0,
        "macd_hist": 0.0,
        "bb_pct_b": 0.5,
        "bb_squeeze": 0.0,
        "bb_bandwidth": 0.0,
        "ema20": 0.0,
        "atr": 0.0,
        "volume_ratio": 1.0,
        "volume_trend": 1.0,
        "obv_trend": 0.0,
        "vwap": 0.0,
        "price_vs_vwap": 0.0,
        "close": 0.0,
        "trend": 0.0,
        "rsi_divergence": 0.0,
        "data_status": 0.0 if status == "insufficient_data" else 1.0,
    }


def detect_support_resistance(ohlcv: list[list[float]], num_levels: int = 3) -> dict:
    """Detect key support and resistance levels using swing highs/lows.

    Returns {'support': [prices], 'resistance': [prices], 'nearest_support': float,
             'nearest_resistance': float, 'price_vs_sr': str}
    """
    if not ohlcv or len(ohlcv) < 20:
        return {"support": [], "resistance": [], "nearest_support": 0.0,
                "nearest_resistance": 0.0, "price_vs_sr": "unknown"}

    highs = np.array([c[2] for c in ohlcv])
    lows = np.array([c[3] for c in ohlcv])
    closes = np.array([c[4] for c in ohlcv])
    current = closes[-1]

    # Find swing highs and lows (pivot points with lookback=5)
    swing_highs = []
    swing_lows = []
    lookback = 5
    for i in range(lookback, len(highs) - lookback):
        if highs[i] == max(highs[i - lookback:i + lookback + 1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i - lookback:i + lookback + 1]):
            swing_lows.append(lows[i])

    # Cluster nearby levels (within 1% of each other)
    def _cluster(levels: list[float], pct: float = 0.01) -> list[float]:
        if not levels:
            return []
        sorted_levels = sorted(levels)
        clusters: list[list[float]] = [[sorted_levels[0]]]
        for lvl in sorted_levels[1:]:
            if abs(lvl - clusters[-1][-1]) / (clusters[-1][-1] + 1e-10) < pct:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        # Return average of each cluster, sorted by frequency (most-tested first)
        result = sorted(clusters, key=len, reverse=True)
        return [round(sum(c) / len(c), 6) for c in result[:num_levels]]

    support = [s for s in _cluster(swing_lows) if s < current]
    resistance = [r for r in _cluster(swing_highs) if r > current]

    nearest_sup = max(support) if support else 0.0
    nearest_res = min(resistance) if resistance else 0.0

    # Classify position relative to S/R
    if nearest_sup > 0 and nearest_res > 0:
        range_size = nearest_res - nearest_sup
        pos_in_range = (current - nearest_sup) / range_size if range_size > 0 else 0.5
        if pos_in_range < 0.25:
            sr_label = "near_support"  # close to support → potential bounce
        elif pos_in_range > 0.75:
            sr_label = "near_resistance"  # close to resistance → potential rejection
        else:
            sr_label = "mid_range"
    elif nearest_sup > 0:
        sr_label = "above_support"
    elif nearest_res > 0:
        sr_label = "below_resistance"
    else:
        sr_label = "unknown"

    return {
        "support": support[:num_levels],
        "resistance": resistance[:num_levels],
        "nearest_support": nearest_sup,
        "nearest_resistance": nearest_res,
        "price_vs_sr": sr_label,
    }
