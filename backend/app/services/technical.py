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


# ── Divergence detection helpers ─────────────────────────────────────────────

def _find_swing_highs(arr: np.ndarray, lookback: int = 3) -> list[int]:
    """Return indices of swing high points (local maxima)."""
    highs = []
    for i in range(lookback, len(arr) - lookback):
        window = arr[i - lookback: i + lookback + 1]
        if arr[i] == window.max():
            highs.append(i)
    return highs


def _find_swing_lows(arr: np.ndarray, lookback: int = 3) -> list[int]:
    """Return indices of swing low points (local minima)."""
    lows = []
    for i in range(lookback, len(arr) - lookback):
        window = arr[i - lookback: i + lookback + 1]
        if arr[i] == window.min():
            lows.append(i)
    return lows


def detect_divergences(close: np.ndarray, rsi: np.ndarray, macd_hist: np.ndarray) -> dict:
    """Detect RSI and MACD divergences using swing-point comparison.

    Bullish RSI divergence  : price makes lower low, RSI makes higher low  → buy signal
    Bearish RSI divergence  : price makes higher high, RSI makes lower high → sell signal
    Bullish MACD divergence : price makes lower low, MACD hist makes higher low
    Bearish MACD divergence : price makes higher high, MACD hist makes lower high

    Returns dict with 'rsi_divergence' and 'macd_divergence', each in [-1, 0, 1].
    """
    results = {"rsi_divergence": 0.0, "macd_divergence": 0.0}

    if len(close) < 20:
        return results

    close_arr = np.array(close)
    rsi_arr   = np.array(rsi)
    macd_arr  = np.array(macd_hist)

    # Use last 40 bars for swing detection
    window = min(40, len(close_arr))
    c   = close_arr[-window:]
    r   = rsi_arr[-window:]
    m   = macd_arr[-window:]

    price_lows  = _find_swing_lows(c)
    price_highs = _find_swing_highs(c)

    # ── RSI divergence ────────────────────────────────────────────────
    if len(price_lows) >= 2 and len(r) == len(c):
        i1, i2 = price_lows[-2], price_lows[-1]
        if c[i2] < c[i1] and r[i2] > r[i1]:
            # Bullish: price lower low, RSI higher low
            results["rsi_divergence"] = 1.0
        elif c[i2] < c[i1] and r[i2] < r[i1] - 5:
            # Confirmed downtrend: price AND RSI both lower
            pass

    if len(price_highs) >= 2 and len(r) == len(c):
        i1, i2 = price_highs[-2], price_highs[-1]
        if c[i2] > c[i1] and r[i2] < r[i1]:
            # Bearish: price higher high, RSI lower high
            results["rsi_divergence"] = -1.0

    # ── MACD divergence ───────────────────────────────────────────────
    if len(price_lows) >= 2 and len(m) == len(c):
        i1, i2 = price_lows[-2], price_lows[-1]
        if c[i2] < c[i1] and m[i2] > m[i1]:
            results["macd_divergence"] = 1.0   # bullish

    if len(price_highs) >= 2 and len(m) == len(c):
        i1, i2 = price_highs[-2], price_highs[-1]
        if c[i2] > c[i1] and m[i2] < m[i1]:
            results["macd_divergence"] = -1.0  # bearish

    return results


def compute_indicators(ohlcv: list[list[float]]) -> dict[str, float]:
    """Return a dict of TA indicators for a single timeframe's OHLCV data.

    New fields added:
    - volume_zscore      : Z-score of current volume vs its own 20-bar mean
    - breakout_48h       : +1 if price broke above 48-bar high, -1 if broke below low, 0 otherwise
    - momentum_accel     : 2nd derivative of close price (normalised)
    - macd_divergence    : swing-based MACD divergence (-1/0/+1)
    - rsi_divergence     : swing-based RSI divergence (-1/0/+1) [improved from slope-based]
    """
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
            # Full MACD histogram series for divergence calculation
            _macd_hist_series = macd_df[hist_col[0]].fillna(0).values if hist_col else np.zeros(len(close))
        else:
            indicators["macd_hist"] = 0.0
            _macd_hist_series = np.zeros(len(close))

        # Bollinger Bands %B + Squeeze detection
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None and not bb.empty:
            pct_col = [c for c in bb.columns if "p" in c.lower()]
            indicators["bb_pct_b"] = _safe_float(
                bb[pct_col[0]].iloc[-1] if pct_col else None
            )
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

        # EMA 50 / EMA 200 — trend filter (golden/death cross)
        if len(close) >= 50:
            ema50 = ta.ema(close, length=50)
            indicators["ema50"] = _safe_float(ema50.iloc[-1] if ema50 is not None else None)
        else:
            indicators["ema50"] = 0.0
        if len(close) >= 200:
            ema200 = ta.ema(close, length=200)
            indicators["ema200"] = _safe_float(ema200.iloc[-1] if ema200 is not None else None)
        else:
            indicators["ema200"] = 0.0
        # Trend-up flag: EMA50 > EMA200 — used as a hard BUY filter.
        # When EMA200 is unavailable (insufficient history) we set 1.0 so
        # the filter does not block trading on freshly-listed coins.
        if indicators["ema50"] > 0 and indicators["ema200"] > 0:
            indicators["trend_up"] = 1.0 if indicators["ema50"] > indicators["ema200"] else 0.0
        else:
            indicators["trend_up"] = 1.0

        # ADX 14 — trend strength (>20 = trending, <20 = chop / no edge for breakouts)
        try:
            adx_df = ta.adx(high, low, close, length=14)
            if adx_df is not None and not adx_df.empty:
                adx_col = [c for c in adx_df.columns if "adx" in c.lower()]
                indicators["adx"] = _safe_float(adx_df[adx_col[0]].iloc[-1] if adx_col else None)
            else:
                indicators["adx"] = 0.0
        except Exception:
            indicators["adx"] = 0.0

        # Choppiness Index 14 — 100*log10(sum(TR)/(maxH-minL)) / log10(N)
        # >= 61.8 = strong chop (whipsaw zone), <= 38.2 = strong trend
        try:
            n_chop = 14
            if len(close) > n_chop:
                tr = pd.concat([
                    (high - low),
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs(),
                ], axis=1).max(axis=1)
                sum_tr  = tr.rolling(n_chop).sum()
                rng_hi  = high.rolling(n_chop).max()
                rng_lo  = low.rolling(n_chop).min()
                ci = 100.0 * np.log10(
                    (sum_tr / (rng_hi - rng_lo).replace(0, np.nan))
                ) / np.log10(n_chop)
                indicators["choppiness"] = _safe_float(ci.iloc[-1])
            else:
                indicators["choppiness"] = 50.0
        except Exception:
            indicators["choppiness"] = 50.0

        # ATR 14
        atr = ta.atr(high, low, close, length=14)
        indicators["atr"] = _safe_float(atr.iloc[-1] if atr is not None else None)

        # ── Swing-based RSI + MACD divergence ────────────────────────
        rsi_full = ta.rsi(close, length=14)
        if rsi_full is not None and len(rsi_full.dropna()) >= 20:
            rsi_vals  = rsi_full.fillna(50).values
            div = detect_divergences(
                close.values, rsi_vals, _macd_hist_series
            )
            indicators["rsi_divergence"]  = div["rsi_divergence"]
            indicators["macd_divergence"] = div["macd_divergence"]
        else:
            indicators["rsi_divergence"]  = 0.0
            indicators["macd_divergence"] = 0.0

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
        indicators["ema50"] = _safe_float(close.ewm(span=50).mean().iloc[-1]) if len(close) >= 50 else 0.0
        indicators["ema200"] = _safe_float(close.ewm(span=200).mean().iloc[-1]) if len(close) >= 200 else 0.0
        indicators["trend_up"] = 1.0 if (indicators["ema50"] == 0 or indicators["ema200"] == 0 or indicators["ema50"] > indicators["ema200"]) else 0.0
        indicators["adx"] = 0.0
        indicators["choppiness"] = 50.0
        indicators["atr"] = _safe_float(
            (high - low).rolling(14).mean().iloc[-1]
        )
        indicators["rsi_divergence"]  = 0.0
        indicators["macd_divergence"] = 0.0

    # ── Volume ratio (current vs 20-bar average) ──────────────────────
    vol_avg = volume.rolling(20).mean().iloc[-1]
    current_vol = volume.iloc[-1]
    indicators["volume_ratio"] = _safe_float(
        current_vol / vol_avg if vol_avg and vol_avg != 0 else 1.0
    )

    # ── Volume Z-score — statistically unusual for THIS symbol ────────
    if len(volume) >= 10:
        vol_window = volume.iloc[-20:] if len(volume) >= 20 else volume
        v_mean = float(vol_window.mean())
        v_std  = float(vol_window.std())
        if v_std > 0:
            indicators["volume_zscore"] = _safe_float(
                (current_vol - v_mean) / v_std
            )
        else:
            indicators["volume_zscore"] = 0.0
    else:
        indicators["volume_zscore"] = 0.0

    # ── Volume trend (is volume increasing over last 5 bars?) ─────────
    if len(volume) >= 10:
        vol_recent = volume.iloc[-5:].mean()
        vol_prior = volume.iloc[-10:-5].mean()
        indicators["volume_trend"] = _safe_float(
            vol_recent / vol_prior if vol_prior > 0 else 1.0
        )
    else:
        indicators["volume_trend"] = 1.0

    # ── OBV trend ─────────────────────────────────────────────────────
    obv = (np.sign(close.diff()) * volume).cumsum()
    if len(obv) >= 20:
        obv_ema = obv.ewm(span=20).mean()
        indicators["obv_trend"] = 1.0 if obv.iloc[-1] > obv_ema.iloc[-1] else -1.0
    else:
        indicators["obv_trend"] = 0.0

    # ── VWAP ──────────────────────────────────────────────────────────
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    indicators["vwap"] = _safe_float(vwap.iloc[-1])
    current_close = _safe_float(close.iloc[-1])
    if indicators["vwap"] > 0 and current_close:
        indicators["price_vs_vwap"] = round(
            (current_close - indicators["vwap"]) / indicators["vwap"] * 100, 3
        )
    else:
        indicators["price_vs_vwap"] = 0.0

    # ── Trend label ───────────────────────────────────────────────────
    indicators["close"] = current_close
    ema20_val = indicators.get("ema20", 0.0)

    if len(close) >= 5 and ema20_val > 0:
        above_count = sum(1 for c in close.iloc[-5:] if c > ema20_val)
        indicators["trend"] = 1.0 if above_count >= 3 else -1.0
    else:
        indicators["trend"] = 1.0 if current_close > ema20_val else -1.0

    # ── Breakout from 48-bar high/low ────────────────────────────────
    # Compares current close to the max high and min low of the prior N bars.
    # Breakout above N-bar high = strong momentum continuation signal.
    breakout_window = min(48, len(close) - 1)
    if breakout_window >= 10:
        prior_highs = high.iloc[-breakout_window - 1:-1]
        prior_lows  = low.iloc[-breakout_window - 1:-1]
        period_high = float(prior_highs.max())
        period_low  = float(prior_lows.min())
        if period_high > 0 and current_close > period_high:
            indicators["breakout_48h"] = 1.0   # price broke above prior high
        elif period_low > 0 and current_close < period_low:
            indicators["breakout_48h"] = -1.0  # price broke below prior low
        else:
            indicators["breakout_48h"] = 0.0
    else:
        indicators["breakout_48h"] = 0.0

    # ── Momentum acceleration (2nd derivative of close) ───────────────
    # Measures whether momentum is speeding up or slowing down.
    # Uses 3-bar returns to reduce noise: (last 3 - prior 3) vs (prior 3 - before that)
    if len(close) >= 9:
        r_recent = float((close.iloc[-1] - close.iloc[-4]) / (close.iloc[-4] + 1e-10))
        r_prior  = float((close.iloc[-4] - close.iloc[-7]) / (close.iloc[-7] + 1e-10))
        accel = r_recent - r_prior
        # Normalise: ±5% change in 3-bar return = full signal
        indicators["momentum_accel"] = float(np.clip(accel / 0.05, -1.0, 1.0))
    else:
        indicators["momentum_accel"] = 0.0

    indicators["data_status"] = 1.0
    return indicators


def _empty_indicators(status: str = "ok") -> dict[str, float]:
    return {
        "rsi14": 50.0,
        "macd_hist": 0.0,
        "bb_pct_b": 0.5,
        "bb_squeeze": 0.0,
        "bb_bandwidth": 0.0,
        "ema20": 0.0,
        "ema50": 0.0,
        "ema200": 0.0,
        "trend_up": 1.0,
        "adx": 0.0,
        "choppiness": 50.0,
        "atr": 0.0,
        "volume_ratio": 1.0,
        "volume_zscore": 0.0,
        "volume_trend": 1.0,
        "obv_trend": 0.0,
        "vwap": 0.0,
        "price_vs_vwap": 0.0,
        "close": 0.0,
        "trend": 0.0,
        "breakout_48h": 0.0,
        "momentum_accel": 0.0,
        "rsi_divergence": 0.0,
        "macd_divergence": 0.0,
        "data_status": 0.0 if status == "insufficient_data" else 1.0,
    }


def compute_beta_vs_btc(
    altcoin_ohlcv: list[list[float]],
    btc_ohlcv: list[list[float]],
    lookback: int = 48,
) -> dict[str, float]:
    """Compute the altcoin's beta relative to BTC using 1h OHLCV data.

    Beta measures how much the altcoin moves per 1% BTC move.
    - beta > 1.5  : high-beta alt — amplifies BTC moves (volatile, high upside)
    - beta ~1.0   : tracks BTC closely
    - beta < 0.5  : low-beta or decorrelated
    - beta < 0    : negative correlation (rare, unusual)

    Returns:
        {
            "beta":        1.35,    # regression coefficient vs BTC
            "correlation": 0.82,    # Pearson r with BTC returns
            "r_squared":   0.67,    # R² of the OLS fit
        }
    """
    default = {"beta": 1.0, "correlation": 0.5, "r_squared": 0.25}
    if not altcoin_ohlcv or not btc_ohlcv or len(altcoin_ohlcv) < 10 or len(btc_ohlcv) < 10:
        return default

    try:
        alt_closes = np.array([c[4] for c in altcoin_ohlcv[-lookback:]], dtype=float)
        btc_closes  = np.array([c[4] for c in btc_ohlcv[-lookback:]], dtype=float)

        min_len = min(len(alt_closes), len(btc_closes))
        if min_len < 5:
            return default
        alt_closes = alt_closes[-min_len:]
        btc_closes  = btc_closes[-min_len:]

        # Log returns
        alt_rets = np.diff(np.log(alt_closes + 1e-12))
        btc_rets = np.diff(np.log(btc_closes + 1e-12))

        if len(alt_rets) < 4:
            return default

        # OLS beta: cov(alt, btc) / var(btc)
        btc_var = float(np.var(btc_rets, ddof=1))
        if btc_var <= 0:
            return default

        cov  = float(np.cov(alt_rets, btc_rets, ddof=1)[0, 1])
        beta = cov / btc_var

        # Pearson correlation + R²
        corr_mat = np.corrcoef(alt_rets, btc_rets)
        corr = float(corr_mat[0, 1]) if not np.isnan(corr_mat[0, 1]) else 0.0
        r_sq = corr ** 2

        beta = float(np.clip(beta, -2.0, 4.0))

        return {
            "beta":        round(beta, 3),
            "correlation": round(corr, 3),
            "r_squared":   round(r_sq, 3),
        }

    except Exception as exc:
        logger.debug("compute_beta_vs_btc failed: %s", exc)
        return default


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

    swing_highs = []
    swing_lows = []
    lookback = 5
    for i in range(lookback, len(highs) - lookback):
        if highs[i] == max(highs[i - lookback:i + lookback + 1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i - lookback:i + lookback + 1]):
            swing_lows.append(lows[i])

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
        result = sorted(clusters, key=len, reverse=True)
        return [round(sum(c) / len(c), 6) for c in result[:num_levels]]

    support = [s for s in _cluster(swing_lows) if s < current]
    resistance = [r for r in _cluster(swing_highs) if r > current]

    nearest_sup = max(support) if support else 0.0
    nearest_res = min(resistance) if resistance else 0.0

    if nearest_sup > 0 and nearest_res > 0:
        range_size = nearest_res - nearest_sup
        pos_in_range = (current - nearest_sup) / range_size if range_size > 0 else 0.5
        if pos_in_range < 0.25:
            sr_label = "near_support"
        elif pos_in_range > 0.75:
            sr_label = "near_resistance"
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
