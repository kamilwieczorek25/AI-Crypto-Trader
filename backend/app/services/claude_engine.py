"""Claude AI engine — prompt builder + tool_use caller with cost optimisation."""

import json
import logging
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic

from app.config import settings
from app.schemas.decision import TradeDecision

logger = logging.getLogger(__name__)

# ── Model tiers ──────────────────────────────────────────────────────────────
_MODEL_SONNET = "claude-sonnet-4-6"     # $3/M input, $15/M output
_MODEL_HAIKU  = "claude-haiku-4-5-20251001"        # $0.80/M input, $4/M output


# ── Risk profiles ────────────────────────────────────────────────────────────

@dataclass
class RiskProfile:
    label: str
    min_confidence: float
    max_position_pct: float
    system_addendum: str   # appended to the base system prompt


_PROFILES: dict[str, RiskProfile] = {
    "conservative": RiskProfile(
        label="Conservative",
        min_confidence=0.75,
        max_position_pct=2.0,
        system_addendum="""\

RISK PROFILE: CONSERVATIVE
- Only trade when confidence ≥ 0.75. Prefer HOLD on any ambiguity.
- Maximum position size: 2% of portfolio.
- Focus on 4h and 1d timeframes. Ignore 15m noise.
- Prioritise capital preservation over profit.
- Only enter on textbook setups: RSI oversold bounce, clear MACD crossover, strong BB support.
- Wide stop-losses (6–10%) are acceptable to avoid whipsaws.
""",
    ),
    "balanced": RiskProfile(
        label="Balanced",
        min_confidence=0.55,
        max_position_pct=5.0,
        system_addendum="""\

RISK PROFILE: BALANCED (default)
- Trade when confidence ≥ 0.55.
- Maximum position size: 5% of portfolio.
- Use all timeframes equally.
- Balance reward vs risk; normal stop-losses (4–8%).
""",
    ),
    "aggressive": RiskProfile(
        label="Aggressive",
        min_confidence=0.45,
        max_position_pct=5.0,
        system_addendum="""\

RISK PROFILE: AGGRESSIVE
- Trade when confidence ≥ 0.45. Lean toward action over inaction.
- Maximum position size: 5% of portfolio.
- Favour 15m and 1h signals. React quickly to momentum shifts.
- Tighter stop-losses (3–5%) to cut losses fast and redeploy capital.
- Look for breakouts, volume spikes, and RSI recovery plays.
- It is acceptable to be wrong more often if winners are larger.
""",
    ),
    "fast_profit": RiskProfile(
        label="Fast Profit",
        min_confidence=0.45,
        max_position_pct=5.0,
        system_addendum="""\

RISK PROFILE: FAST PROFIT (high aggression, but disciplined)
- Trade when confidence ≥ 0.45.  HOLD is a perfectly valid answer when no
  setup clears the bar — missed trades cost nothing, bad trades cost money.
- Maximum position size: 5% of portfolio.
- PRIMARY timeframes: 15m and 1h.  4h/1d are trend filters — do NOT BUY
  when the 4h trend is bearish unless an explicit reversal signal is present.
- Momentum entries (RSI rising + MACD positive + volume > 1.5×) are good
  IF the higher-timeframe trend agrees and the coin is not already +20%/24h.
- Use stop-losses 3-5% and take-profits 6-15% (R:R ≥ 2).
- Sell quickly when momentum stalls — do not hold losing positions.
- In ranging or downtrending markets, default to HOLD/CASH.
""",
    ),
}

_BASE_SYSTEM_PROMPT = """\
You are an expert cryptocurrency trading AI. Your job is to analyse multi-timeframe \
market data and make a single, well-reasoned trading decision each cycle.

Hard rules that ALWAYS apply regardless of risk profile:
- Maximum total altcoin exposure: 70% of total portfolio.
- SELL is ONLY valid for symbols you currently hold in open positions. Never choose SELL for a symbol not in your portfolio — it will be rejected.
- If you have no cash available AND no open positions, return HOLD.
- If you have no cash but DO have open positions, only consider SELL.
- Always call the make_trading_decision tool with your decision.
- NEVER buy a symbol that is highly correlated (r>0.8) with a symbol you already hold — diversify.
- Adapt your aggression to the detected market regime (e.g., be more cautious in downtrends/choppy markets).
- Pay attention to RSI divergences and OBV trend — they often signal reversals before price does.
- Indicators marked [INSUFFICIENT DATA] should be treated as unreliable — do not trade based on them.

ADVANCED FEATURES:
- BTC ANCHOR: BTC/USDT data is provided as market context. When BTC is bearish, most altcoins drop. Factor BTC's trend/RSI into every decision.
- SUPPORT/RESISTANCE: S/R levels are provided per symbol. Prefer BUYing near support and SELLing near resistance. S/R breakouts with volume confirmation are strong signals.
- PARTIAL SELLS: Use sell_pct (1-100) to take partial profits. Example: sell 50% at +10% profit, keep 50% running. Use partial sells when a position is profitable but trend is still intact.
- TIME OF DAY: Trading session info is provided. During low-liquidity sessions (Asia), prefer wider stops. During high-volume sessions (US), tighter stops are safer.
- VWAP: Volume Weighted Average Price is shown per symbol. Price above VWAP = bullish (institutional buyers present). Price below VWAP = bearish. Prefer BUYing below VWAP for better entries.
- NEW LISTINGS & TOP GAINERS: Newly listed coins and top 24h gainers are highlighted. New listings often pump hard in the first 24-48h but are extremely volatile. Use tight stop-losses (5-8%) on new listings. Top gainers may have momentum but beware of pump-and-dump — confirm with volume and orderbook pressure before entry.

You receive:
- Time of day and active trading session
- Current portfolio state (cash, positions, P&L)
- BTC/USDT anchor data (trend, RSI, momentum)
- Market regime detection (uptrend/downtrend/ranging/choppy)
- Correlation warnings for highly-correlated pairs
- USDT altcoins with full indicator data + support/resistance levels
- Order book pressure for each symbol
- Recent news sentiment per symbol (with min/max/trend — may use GPU semantic analysis)
- AI Signal Ensemble: LSTM neural-net direction probabilities + RL agent Q-value recommendations
- GPU Ensemble (when available): Transformer + LSTM + Dueling DQN + semantic sentiment combined signal with agreement score
- Multi-Timeframe Fusion (MTF): GPU model seeing 15m+1h+4h+1d simultaneously — the strongest directional signal when available
- Volatility Forecast: GPU-predicted future σ — more accurate than backward-looking ATR
- Anomaly Detection: GPU autoencoder flags pump-and-dumps, flash crashes — ⚠ANOMALY means DO NOT BUY
- Exit RL: GPU reinforcement learning agent specialized in exit timing — HOLD_POS/PARTIAL_25/PARTIAL_50/CLOSE
- Attention Explainability: which candles and features the Transformer focused on — validate with your own analysis
- Correlation Divergence: when correlated pairs diverge, the laggard often catches up — mean-reversion opportunity
- RAG context: relevant snippets from past trades and market insights
- New listings and top 24h gainers with price change and volume data

When the LSTM and RL agent agree on a direction, treat that as a strong corroborating signal.
When they disagree, exercise extra caution — prefer HOLD unless technical indicators are compelling.
When a GPU ensemble signal is present with agreement >= 75%, treat it as the strongest ML signal available.
When MTF model has confidence >= 70%, treat it as strong cross-timeframe confirmation.
When Exit RL recommends CLOSE or PARTIAL, strongly consider following it for held positions.
When anomaly flag is raised, NEVER BUY — the autoencoder detected abnormal price/volume patterns.
Always explain in your reasoning how you weighted the AI ensemble signals.
"""

_VALIDATOR_SYSTEM_PROMPT = """\
You are a RISK VALIDATOR for an automated cryptocurrency trading system.

A quantitative scoring model has pre-screened trade candidates using RSI, MACD, \
Bollinger Bands, VWAP, OBV, volume profile, multi-timeframe trend alignment, \
support/resistance, BTC correlation, and ML ensemble signals. Each candidate has a \
composite score (0-100), ATR-based stop-loss/take-profit levels, and an enforced \
minimum reward:risk ratio.

YOUR ROLE: Final safety filter. The quant model handles the "what" — you handle \
the "why not."

WHAT TO CHECK:
- News/sentiment risk the quant model cannot see (hacks, delistings, regulatory action)
- Portfolio concentration risk (too many correlated positions)
- Timing risk (upcoming events, low-liquidity sessions for large positions)
- Pump-and-dump patterns (massive gains on thin volume, no fundamental driver)
- Conflicting signals the model might have averaged away

DECISION RULES:
1. To APPROVE a candidate: return it via make_trading_decision (keep symbol, action, SL, TP).
2. To REJECT all candidates: return action=HOLD with reasoning explaining the risk.
3. You may reduce quantity_pct (never increase it beyond the quant suggestion).
4. You MUST use the stop_loss_pct and take_profit_pct provided by the quant model.
5. Default bias: APPROVE. Heavy filtering already happened. Only reject with clear, specific reasoning.
6. SELL is ONLY valid for symbols currently held. Approving a SELL candidate is usually correct.
7. Always call the make_trading_decision tool with your decision.
"""

_LESS_FEAR_ADDENDUM = """

NOTE: LESS-FEAR MODE ACTIVE
- The operator has enabled less-fear mode — lean toward APPROVE on borderline
  candidates that have strong quant scores AND a constructive market regime.
- This does NOT override risk discipline:
    * In confirmed downtrend / strong_downtrend / choppy regimes, the default
      action is still HOLD.  Approve only on clear bullish reversal evidence.
    * REJECT any candidate already up >20% / 24h unless news/flow justifies it.
    * REJECT candidates whose 4h or 1d trend is bearish.
- Concrete catastrophic risks (hack, delisting, rug-pull, anomaly) always REJECT.
"""


def _get_profile() -> RiskProfile:
    return _PROFILES.get(settings.RISK_PROFILE, _PROFILES["balanced"])


def _build_system_prompt(validation_mode: bool = False) -> str:
    profile = _get_profile()
    base = _VALIDATOR_SYSTEM_PROMPT if validation_mode else _BASE_SYSTEM_PROMPT
    prompt = base + profile.system_addendum
    if settings.LESS_FEAR:
        prompt += _LESS_FEAR_ADDENDUM
    return prompt


_TOOL_DEFINITION = {
    "name": "make_trading_decision",
    "description": "Submit a structured trading decision for the bot to execute.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["BUY", "SELL", "HOLD"],
                "description": "Trading action",
            },
            "symbol": {
                "type": "string",
                "description": "Trading pair e.g. ETH/USDT",
            },
            "timeframe": {
                "type": "string",
                "description": "Primary signal timeframe e.g. 1h",
            },
            "quantity_pct": {
                "type": "number",
                "description": "Percentage of total portfolio to trade (0–5)",
            },
            "stop_loss_pct": {
                "type": "number",
                "description": "Stop-loss distance below entry in %",
            },
            "take_profit_pct": {
                "type": "number",
                "description": "Take-profit distance above entry in %",
            },
            "confidence": {
                "type": "number",
                "description": "Decision confidence 0.0–1.0",
            },
            "sell_pct": {
                "type": "number",
                "description": "For SELL: percentage of position to sell (1-100). Use 100 for full close, 25-75 for partial profit taking. Default 100.",
            },
            "primary_signals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key signals driving this decision",
            },
            "risk_factors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Known risks or concerns",
            },
            "reasoning": {
                "type": "string",
                "description": "One-paragraph reasoning for the decision",
            },
        },
        "required": [
            "action", "symbol", "timeframe", "quantity_pct",
            "stop_loss_pct", "take_profit_pct", "confidence",
            "primary_signals", "risk_factors", "reasoning",
        ],
    },
}


def build_prompt(
    portfolio: dict[str, Any],
    symbols_data: dict[str, Any],
    news: dict[str, Any],
    ml_signals: "dict[str, dict] | None" = None,
    rag_context: "list[str] | None" = None,
    market_regime: "dict | None" = None,
    correlation_info: "dict | None" = None,
    btc_anchor: "dict | None" = None,
    top_gainers: "list[dict] | None" = None,
    new_listings: "dict[str, Any] | None" = None,
) -> str:
    """Build a compact, token-efficient prompt for Claude.

    Uses tabular CSV-like format for indicators instead of verbose English.
    """
    from datetime import datetime, timezone
    profile = _get_profile()
    lines: list[str] = []

    held_symbols: set[str] = {
        pos["symbol"] for pos in portfolio.get("positions", [])
    }

    # Time context
    now = datetime.now(timezone.utc)
    hour = now.hour
    if 0 <= hour < 8:
        session = "Asia"
    elif 8 <= hour < 14:
        session = "Europe"
    elif 14 <= hour < 21:
        session = "US"
    else:
        session = "Late-US"

    lines.append(f"PROFILE={profile.label.upper()} min_conf={profile.min_confidence} max_pos={profile.max_position_pct}% symbols={len(symbols_data)}")
    lines.append(f"TIME={now.strftime('%Y-%m-%d %H:%M')}UTC {now.strftime('%A')} SESSION={session}")

    # Portfolio (compact)
    pv = portfolio['total_value_usdt']
    cash = portfolio['cash_usdt']
    lines.append(f"\nPORTFOLIO: total=${pv:,.2f} cash=${cash:,.2f}({cash/max(pv,1)*100:.0f}%) pnl={portfolio['total_pnl_pct']:+.2f}% positions={portfolio['num_open_positions']}")

    if portfolio.get("positions"):
        lines.append("HELD:")
        for pos in portfolio["positions"]:
            lines.append(f"  {pos['symbol']} qty={pos['quantity']:.4f} entry=${pos['avg_entry_price']:.4f} now=${pos['current_price']:.4f} pnl={pos['pnl_pct']:+.2f}%")
    else:
        lines.append("HELD: NONE (SELL not valid)")

    # Market regime (one line)
    if market_regime and market_regime.get("regime") != "unknown":
        mr = market_regime
        lines.append(f"\nREGIME={mr['regime'].upper()} vol={mr.get('volatility','?')} ret={mr.get('avg_return_pct',0):+.2f}% | {mr.get('description','')}")

    # Correlation warnings (compact)
    if correlation_info and correlation_info.get("high_correlation_pairs"):
        pairs = " ".join(f"{p['pair'][0]}↔{p['pair'][1]}(r={p['correlation']})" for p in correlation_info["high_correlation_pairs"])
        lines.append(f"CORR_WARN: {pairs}")

    # BTC anchor (compact)
    if btc_anchor:
        btc_parts = []
        for tf, ind in btc_anchor.items():
            t = "↑" if ind.get("trend", 0) > 0 else "↓"
            btc_parts.append(f"[{tf}]${ind.get('close',0):.0f} RSI={ind.get('rsi14',0):.0f} MACD={ind.get('macd_hist',0):.2f}{t}")
        lines.append(f"\nBTC: {' '.join(btc_parts)}")

    # Top gainers (24h biggest movers — momentum context)
    if top_gainers:
        gainer_strs = [f"{g['symbol']}({g['pct_24h']:+.1f}% vol=${g['volume']/1e6:.1f}M)" for g in top_gainers[:5]]
        lines.append(f"\n🔥 TOP GAINERS 24h: {' '.join(gainer_strs)}")

    # New listings (recently appeared on Binance — high opportunity + high risk)
    if new_listings:
        from datetime import datetime as _dt, timezone as _tz
        now_ts = _dt.now(_tz.utc)
        new_parts = []
        for sym, first_seen in new_listings.items():
            age_h = (now_ts - first_seen).total_seconds() / 3600
            new_parts.append(f"{sym}(age={age_h:.1f}h)")
        lines.append(f"🆕 NEW LISTINGS: {' '.join(new_parts[:5])} — high volatility, use tight SL")

    # Symbol data — compact tabular format
    lines.append("\n=== SYMBOLS ===")
    lines.append("SYM|TF|RSI|MACD|BB%B|VWAP%|VolR|OBV|ATR|Trend|RsiDiv")
    for symbol, data in symbols_data.items():
        held_tag = " ★HELD" if symbol in held_symbols else ""
        lines.append(f"\n{symbol}{held_tag} price=${data.get('price', 0):.6f}")

        for tf, ind in data.get("indicators", {}).items():
            t = "↑" if ind.get("trend", 0) > 0 else "↓"
            flag = "!" if ind.get("data_status", 1) <= 0 else ""
            rd = ind.get("rsi_divergence", 0)
            div = "B" if rd > 0 else ("S" if rd < 0 else "-")
            obv = "↑" if ind.get("obv_trend", 0) > 0 else ("↓" if ind.get("obv_trend", 0) < 0 else "→")
            lines.append(
                f"{flag}{symbol}|{tf}|{ind.get('rsi14',0):.1f}|{ind.get('macd_hist',0):.4f}|"
                f"{ind.get('bb_pct_b',0.5):.2f}|{ind.get('price_vs_vwap',0):+.1f}%|"
                f"{ind.get('volume_ratio',1):.2f}|{obv}|{ind.get('atr',0):.4f}|{t}|{div}"
            )

        ob = data.get("orderbook", {})
        lines.append(f"  OB: spread={ob.get('spread_pct',0):.4f}% pressure={ob.get('pressure_ratio',1):.3f}")

        sr = data.get("support_resistance", {})
        if sr.get("nearest_support") or sr.get("nearest_resistance"):
            sr_str = f"  S/R: pos={sr.get('price_vs_sr','?')}"
            if sr.get("nearest_support"):
                sr_str += f" S=${sr['nearest_support']:.6f}"
            if sr.get("nearest_resistance"):
                sr_str += f" R=${sr['nearest_resistance']:.6f}"
            lines.append(sr_str)

        ns = news.get(symbol, {})
        if ns.get("article_count", 0) > 0:
            lines.append(f"  News: n={ns['article_count']} sent={ns.get('avg_sentiment',0):+.3f} trend={ns.get('sentiment_trend',0):+.3f}")
            for headline in ns.get("headlines", [])[:1]:
                lines.append(f"    {headline}")

    # ML signals (compact)
    if ml_signals:
        lines.append("\n=== ML ENSEMBLE ===")
        agree_syms: list[str] = []
        for sym, sig in ml_signals.items():
            lstm = sig.get("lstm", {})
            rl   = sig.get("rl",   {})
            ls = lstm.get("signal", "?")
            lc = lstm.get("confidence", 0.0)
            ra = rl.get("action", "?")
            q  = rl.get("q_values", {})
            q_str = " ".join(f"{a}={v:.2f}" for a, v in q.items())
            sell_warn = ""
            if (ls == "SELL" or ra == "SELL") and sym not in held_symbols:
                sell_warn = " ⚠no-pos"
            parts = [f"  {sym}: LSTM={ls}({lc:.0%}) RL={ra} Q[{q_str}]{sell_warn}"]
            # GPU ensemble signal (when available)
            ens = sig.get("ensemble")
            if ens:
                parts.append(f" GPU={ens['signal']}({ens['confidence']:.0%} agree={ens['agreement']:.0%})")
            # Multi-Timeframe Fusion signal
            mtf = sig.get("mtf")
            if mtf:
                parts.append(f" MTF={mtf['signal']}({mtf['confidence']:.0%} tf={','.join(mtf.get('timeframes', []))})")
            # Anomaly flag
            anom = sig.get("anomaly")
            if anom and anom.get("is_anomaly"):
                parts.append(f" ⚠ANOMALY(z={anom['anomaly_score']:.1f})")
            # Volatility forecast
            vol = sig.get("vol_forecast")
            if vol:
                parts.append(f" σ={vol['predicted_vol']:.4f}({vol['source']})")
            lines.append("".join(parts))
            # Attention explainability (what the model focused on)
            attn = sig.get("attention")
            if attn and attn.get("top_features"):
                feat_str = " ".join(f"{k}={v:.3f}" for k, v in attn["top_features"][:3])
                lines.append(f"    Attn: {feat_str} candles={attn.get('top_candles', [])}")
            # Exit RL recommendation for held positions
            exit_rl = sig.get("exit_rl")
            if exit_rl and sym in held_symbols:
                lines.append(f"    ExitRL: {exit_rl['action']} Q={exit_rl.get('q_values', {})}")
            # Correlation divergence signal
            corr_div = sig.get("corr_divergence")
            if corr_div:
                lines.append(f"    CorrDiv: {corr_div['signal']} (gap={corr_div['return_gap_pct']:.1f}% r={corr_div['correlation']:.2f})")
            if ls == ra and ls != "HOLD":
                if not (ls == "SELL" and sym not in held_symbols):
                    agree_syms.append(f"{sym}→{ls}")
            # High-confidence GPU ensemble agreement overrides
            elif ens and ens.get("agreement", 0) >= 0.75 and ens.get("signal") != "HOLD":
                if not (ens["signal"] == "SELL" and sym not in held_symbols):
                    agree_syms.append(f"{sym}→{ens['signal']}(GPU)")
            # MTF high-confidence signal
            elif mtf and mtf.get("confidence", 0) >= 0.7 and mtf.get("signal") != "HOLD":
                if not (mtf["signal"] == "SELL" and sym not in held_symbols):
                    agree_syms.append(f"{sym}→{mtf['signal']}(MTF)")
        if agree_syms:
            lines.append(f"  CONSENSUS: {', '.join(agree_syms)}")

    # RAG context (compact)
    if rag_context:
        lines.append("\nHIST:")
        for snippet in rag_context[:3]:
            lines.append(f"  {snippet[:200]}")

    lines.append(f"\nDecide: best of {len(symbols_data)} symbols. Profile={profile.label}. Use sell_pct<100 for partial profit taking.")
    return "\n".join(lines)


def build_validation_prompt(
    candidates: list,
    portfolio: dict[str, Any],
    news: dict[str, Any],
    market_regime: "dict | None" = None,
    btc_anchor: "dict | None" = None,
    correlation_info: "dict | None" = None,
    market_intel: "dict | None" = None,
    ml_signals: "dict | None" = None,
) -> str:
    """Build a compact validation prompt for Claude to approve/reject pre-scored candidates.

    Much smaller than the full prompt — only candidates + portfolio context.
    """
    from datetime import datetime, timezone

    profile = _get_profile()
    lines: list[str] = []

    # Time context
    now = datetime.now(timezone.utc)
    hour = now.hour
    if 0 <= hour < 8:
        session = "Asia"
    elif 8 <= hour < 14:
        session = "Europe"
    elif 14 <= hour < 21:
        session = "US"
    else:
        session = "Late-US"

    lines.append(f"TIME={now.strftime('%Y-%m-%d %H:%M')}UTC SESSION={session}")
    lines.append(f"PROFILE={profile.label.upper()} max_pos={profile.max_position_pct}%")

    # Portfolio (compact)
    pv = portfolio['total_value_usdt']
    cash = portfolio['cash_usdt']
    lines.append(
        f"\nPORTFOLIO: total=${pv:,.2f} cash=${cash:,.2f}"
        f"({cash / max(pv, 1) * 100:.0f}%) pnl={portfolio['total_pnl_pct']:+.2f}%"
        f" positions={portfolio['num_open_positions']}"
    )

    if portfolio.get("positions"):
        lines.append("HELD:")
        for pos in portfolio["positions"]:
            lines.append(
                f"  {pos['symbol']} entry=${pos['avg_entry_price']:.4f}"
                f" now=${pos['current_price']:.4f} pnl={pos['pnl_pct']:+.2f}%"
            )

    # Market regime
    if market_regime and market_regime.get("regime") != "unknown":
        mr = market_regime
        lines.append(
            f"\nREGIME={mr['regime'].upper()} vol={mr.get('volatility', '?')}"
            f" ret={mr.get('avg_return_pct', 0):+.2f}%"
        )

    # BTC anchor
    if btc_anchor:
        btc_parts = []
        for tf, ind in btc_anchor.items():
            t = "\u2191" if ind.get("trend", 0) > 0 else "\u2193"
            btc_parts.append(
                f"[{tf}]RSI={ind.get('rsi14', 0):.0f}"
                f" MACD={ind.get('macd_hist', 0):.2f}{t}"
            )
        lines.append(f"BTC: {' '.join(btc_parts)}")

    # Correlation warnings
    if correlation_info and correlation_info.get("high_correlation_pairs"):
        pairs = " ".join(
            f"{p['pair'][0]}\u2194{p['pair'][1]}(r={p['correlation']})"
            for p in correlation_info["high_correlation_pairs"]
        )
        lines.append(f"CORR: {pairs}")

    # Market intelligence (fear/greed, funding rates)
    intel = market_intel or {}
    fg = intel.get("fear_greed", {})
    if fg.get("value"):
        lines.append(
            f"FEAR_GREED: {fg['value']}/100 ({fg.get('label','?')}) "
            f"trend={fg.get('trend','?')} signal={fg.get('signal','?')}"
        )
    funding_map = intel.get("funding", {})
    ls_map = intel.get("long_short", {})

    # Candidates
    lines.append(f"\n=== {len(candidates)} CANDIDATE(S) FROM QUANT MODEL ===")
    for i, c in enumerate(candidates, 1):
        lines.append(f"\nCANDIDATE {i}: {c.action} {c.symbol}")
        lines.append(f"  Score: {c.score:.0f}/100 | R:R={c.reward_risk_ratio:.1f}")
        lines.append(f"  Entry: ${c.entry_price:.6f}")
        if c.action == "BUY":
            lines.append(f"  SL: ${c.stop_loss_price:.6f} (-{c.stop_loss_pct:.1f}%)")
            lines.append(f"  TP: ${c.take_profit_price:.6f} (+{c.take_profit_pct:.1f}%)")
            lines.append(f"  Size: {c.quantity_pct:.1f}% of portfolio")
            # Monte Carlo risk simulation (when available)
            if "mc_edge" in c.signals:
                lines.append(
                    f"  MC Sim: TP prob={c.signals['mc_tp_prob']:.1f}% "
                    f"SL prob={c.signals['mc_sl_prob']:.1f}% "
                    f"edge={c.signals['mc_edge']:.3f}"
                )
        lines.append(f"  Signals: {', '.join(k for k in c.signals if not k.startswith('mc_'))}")

        # Top factor scores
        top_factors = sorted(
            c.factor_scores.items(), key=lambda x: abs(x[1]), reverse=True
        )[:5]
        lines.append(
            f"  Factors: {' '.join(f'{k}={v:+.2f}' for k, v in top_factors)}"
        )

        # News context for this symbol
        ns = news.get(c.symbol, {})
        if ns.get("article_count", 0) > 0:
            lines.append(
                f"  News: n={ns['article_count']}"
                f" sent={ns.get('avg_sentiment', 0):+.3f}"
            )
            for h in ns.get("headlines", [])[:2]:
                lines.append(f"    {h}")

        # Derivatives data for this symbol
        fd = funding_map.get(c.symbol)
        if fd:
            lines.append(f"  Funding: rate={fd['rate_pct']:+.4f}% ({fd['signal']})")
        ls = ls_map.get(c.symbol)
        if ls:
            lines.append(f"  L/S: ratio={ls['ratio']:.2f} long={ls['long_pct']}% ({ls['signal']})")

        # ML ensemble signals for this symbol (GPU models)
        _ml = (ml_signals or {}).get(c.symbol, {})
        if _ml:
            _ens = _ml.get("ensemble", {})
            _mtf = _ml.get("mtf", {})
            _anom = _ml.get("anomaly", {})
            _parts = []
            if _ens:
                _parts.append(
                    f"ensemble={_ens.get('signal', '?')}"
                    f"(conf={_ens.get('confidence', 0):.2f}"
                    f" agree={_ens.get('agreement', 0):.2f})"
                )
            if _mtf:
                _parts.append(f"MTF={_mtf.get('signal', '?')}")
            if _anom.get("is_anomaly"):
                _parts.append(f"ANOMALY(score={_anom.get('anomaly_score', 0):.2f})")
            if _parts:
                lines.append(f"  ML: {' '.join(_parts)}")

    lines.append(
        f"\nApprove the best candidate, or HOLD if all have disqualifying risks."
        f" Use the provided SL/TP values."
    )
    return "\n".join(lines)


# ── Token usage / cost tracker (in-memory, resets on restart) ────────────────
# Sonnet pricing: $3/M input, $15/M output  |  Haiku: $0.80/M input, $4/M output
# Prompt caching: 90% discount on cache hits (write costs 25% premium)
_COST_TABLE = {
    _MODEL_SONNET: {"input": 3.0,  "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
    _MODEL_HAIKU:  {"input": 0.80, "output": 4.0,  "cache_write": 1.00, "cache_read": 0.08},
}

_usage_stats: dict = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_cache_read_tokens": 0,
    "total_cache_write_tokens": 0,
    "total_calls": 0,
    "total_cost_usd": 0.0,
    "calls_sonnet": 0,
    "calls_haiku": 0,
    "calls_skipped": 0,
    "symbols_filtered": 0,
}


def get_usage_stats() -> dict:
    return dict(_usage_stats)


def _record_usage(model: str, input_tokens: int, output_tokens: int,
                  cache_read: int = 0, cache_write: int = 0) -> None:
    rates = _COST_TABLE.get(model, _COST_TABLE[_MODEL_SONNET])
    # Standard input tokens = total input - cached tokens
    standard_input = max(0, input_tokens - cache_read - cache_write)
    cost = (
        standard_input / 1_000_000 * rates["input"]
        + output_tokens / 1_000_000 * rates["output"]
        + cache_read    / 1_000_000 * rates["cache_read"]
        + cache_write   / 1_000_000 * rates["cache_write"]
    )
    _usage_stats["total_input_tokens"]      += input_tokens
    _usage_stats["total_output_tokens"]     += output_tokens
    _usage_stats["total_cache_read_tokens"] += cache_read
    _usage_stats["total_cache_write_tokens"]+= cache_write
    _usage_stats["total_calls"]             += 1
    _usage_stats["total_cost_usd"]          += cost
    if model == _MODEL_HAIKU:
        _usage_stats["calls_haiku"] += 1
    else:
        _usage_stats["calls_sonnet"] += 1
    cache_pct = (cache_read / max(input_tokens, 1)) * 100
    logger.info(
        "Claude [%s]: in=%d(cache_hit=%d/%d%% write=%d) out=%d cost=$%.4f | total=$%.4f",
        "haiku" if model == _MODEL_HAIKU else "sonnet",
        input_tokens, cache_read, cache_pct, cache_write,
        output_tokens, cost, _usage_stats["total_cost_usd"],
    )


def record_skipped_cycle() -> None:
    """Record a cycle that was skipped (no Claude call)."""
    _usage_stats["calls_skipped"] += 1


async def call_claude(prompt: str, use_haiku: bool = False, validation_mode: bool = False) -> tuple[TradeDecision, str]:
    """Call Claude with tool_use and prompt caching.

    Args:
        prompt: The user prompt (market data).
        use_haiku: If True, use cheaper Haiku model instead of Sonnet.
        validation_mode: If True, use validator system prompt (quant-first flow).

    Returns (decision, raw_response_json).
    One retry on failure. Raises on second failure.
    """
    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    profile = _get_profile()
    model = _MODEL_HAIKU if use_haiku else _MODEL_SONNET
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY, timeout=120.0)
    system_prompt = _build_system_prompt(validation_mode=validation_mode)

    # System prompt with cache_control — cached across calls (saves ~90% on hits)
    system_blocks = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    async def _attempt() -> tuple[TradeDecision, str]:
        response = await client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_blocks,
            tools=[_TOOL_DEFINITION],
            tool_choice={"type": "tool", "name": "make_trading_decision"},
            messages=[{"role": "user", "content": prompt}],
        )
        # Record token usage with cache metrics
        if response.usage:
            cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            _record_usage(
                model, response.usage.input_tokens, response.usage.output_tokens,
                cache_read=cache_read, cache_write=cache_write,
            )

        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_block is None:
            raise ValueError("Claude did not return a tool_use block")
        raw_json = json.dumps(tool_block.input)
        decision = TradeDecision(**tool_block.input)

        # Enforce profile limits on Claude's output
        decision.quantity_pct = min(decision.quantity_pct, profile.max_position_pct)

        # Validate stop_loss < entry < take_profit
        if decision.stop_loss_pct <= 0 and decision.action in ("BUY", "SELL"):
            decision.stop_loss_pct = 3.0  # default safety net
        if decision.take_profit_pct <= 0 and decision.action in ("BUY", "SELL"):
            decision.take_profit_pct = 10.0

        # In validation mode, skip confidence override — quant score is applied externally
        if not validation_mode and decision.confidence < profile.min_confidence and decision.action != "HOLD":
            logger.info(
                "Confidence %.2f below profile min %.2f — overriding to HOLD",
                decision.confidence,
                profile.min_confidence,
            )
            decision.action = "HOLD"
            decision.quantity_pct = 0.0
            decision.stop_loss_pct = 0.0
            decision.take_profit_pct = 0.0

        return decision, raw_json

    try:
        return await _attempt()
    except Exception as first_err:
        logger.warning("Claude first attempt failed (%s): %s — retrying", model, first_err)
        try:
            return await _attempt()
        except Exception as second_err:
            raise RuntimeError(f"Claude API failed after retry: {second_err}") from second_err


def get_profile_info() -> dict:
    p = _get_profile()
    return {
        "key": settings.RISK_PROFILE,
        "label": p.label,
        "min_confidence": p.min_confidence,
        "max_position_pct": p.max_position_pct,
        "less_fear": settings.LESS_FEAR,
    }


def auto_adjust_risk_profile(regime: str) -> str | None:
    """Map market regime to an appropriate risk profile.

    Returns the new profile key if changed, or None if no change needed.
    Only acts when AUTO_RISK_PROFILE is enabled.
    """
    if not settings.AUTO_RISK_PROFILE:
        return None

    # User has pinned the profile — no auto-switch
    if settings.LOCK_RISK_PROFILE:
        return None

    # Less-fear mode: prevent auto-downgrade to conservative
    if settings.LESS_FEAR:
        return None

    mapping = {
        "strong_uptrend": "aggressive",
        "uptrend": "balanced",
        "ranging": "balanced",
        "downtrend": "conservative",
        "strong_downtrend": "conservative",
        "choppy": "conservative",
    }
    target = mapping.get(regime)
    if target and target != settings.RISK_PROFILE:
        old = settings.RISK_PROFILE
        settings.RISK_PROFILE = target
        logger.info("Auto risk-profile: %s -> %s (regime=%s)", old, target, regime)
        return target
    return None


PROFILE_KEYS = list(_PROFILES.keys())


# ── Cost optimisation helpers ────────────────────────────────────────────────

def score_symbol(
    symbol: str,
    indicators: dict[str, dict],
    orderbook: dict,
    news_data: dict,
    ml_signal: "dict | None" = None,
    is_held: bool = False,
    is_new_listing: bool = False,
) -> float:
    """Score a symbol's 'interestingness' for pre-filtering.

    Higher score = more likely to be actionable.
    Held positions always get a bonus to ensure SELL eligibility.
    """
    score = 0.0

    # Held positions get a big bonus — we always want Claude to evaluate them
    if is_held:
        score += 50.0

    # New listings get a significant bonus — high-opportunity window
    if is_new_listing:
        score += 30.0

    ind_1h = indicators.get("1h", {})

    # RSI extremes (oversold/overbought = interesting)
    rsi = ind_1h.get("rsi14", 50)
    if rsi < 30 or rsi > 70:
        score += 15.0
    elif rsi < 35 or rsi > 65:
        score += 5.0

    # MACD magnitude (strong momentum)
    macd = abs(ind_1h.get("macd_hist", 0))
    if macd > 0:
        score += min(macd * 100, 10.0)

    # Volume ratio (high volume = something happening)
    vol_ratio = ind_1h.get("volume_ratio", 1.0)
    if vol_ratio > 2.0:
        score += 15.0
    elif vol_ratio > 1.5:
        score += 8.0
    elif vol_ratio > 1.2:
        score += 3.0

    # RSI divergence (reversal signal)
    if ind_1h.get("rsi_divergence", 0) != 0:
        score += 10.0

    # OBV trend (confirms or diverges from price)
    if ind_1h.get("obv_trend", 0) != 0:
        score += 3.0

    # Bollinger Band extremes
    bb = ind_1h.get("bb_pct_b", 0.5)
    if bb < 0.1 or bb > 0.9:
        score += 8.0
    elif bb < 0.2 or bb > 0.8:
        score += 3.0

    # VWAP deviation
    vwap_dev = abs(ind_1h.get("price_vs_vwap", 0))
    if vwap_dev > 2.0:
        score += 5.0

    # Order book pressure imbalance
    pressure = orderbook.get("pressure_ratio", 1.0)
    if pressure > 1.5 or pressure < 0.67:
        score += 5.0

    # News sentiment extremes
    sentiment = abs(news_data.get("avg_sentiment", 0))
    if sentiment > 0.3:
        score += 8.0
    elif sentiment > 0.15:
        score += 3.0

    # ML consensus (LSTM + RL agree on non-HOLD)
    if ml_signal:
        lstm_sig = ml_signal.get("lstm", {}).get("signal", "HOLD")
        rl_sig = ml_signal.get("rl", {}).get("action", "HOLD")
        lstm_conf = ml_signal.get("lstm", {}).get("confidence", 0)
        if lstm_sig == rl_sig and lstm_sig != "HOLD":
            score += 20.0
        elif lstm_sig != "HOLD" and lstm_conf > 0.6:
            score += 10.0
        elif lstm_sig != "HOLD":
            score += 5.0

    return round(score, 1)


def prefilter_symbols(
    symbols_data: dict[str, Any],
    news: dict[str, Any],
    ml_signals: "dict[str, dict] | None" = None,
    held_symbols: "set[str] | None" = None,
    max_symbols: int = 0,
    new_listings: "set[str] | None" = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Filter symbols_data to the top N most interesting.

    Always includes held positions and new listings so Claude can evaluate them.
    Returns (filtered_symbols_data, scores_dict).
    """
    new_set = new_listings or set()
    if max_symbols <= 0 or len(symbols_data) <= max_symbols:
        scores = {
            sym: score_symbol(sym, d.get("indicators", {}), d.get("orderbook", {}),
                              news.get(sym, {}), (ml_signals or {}).get(sym),
                              sym in (held_symbols or set()), sym in new_set)
            for sym, d in symbols_data.items()
        }
        return symbols_data, scores

    held = held_symbols or set()
    scores: dict[str, float] = {}
    for sym, d in symbols_data.items():
        scores[sym] = score_symbol(
            sym, d.get("indicators", {}), d.get("orderbook", {}),
            news.get(sym, {}), (ml_signals or {}).get(sym),
            sym in held, sym in new_set,
        )

    # Sort by score, always include held symbols and new listings
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    must_include = held | (new_set & symbols_data.keys())
    selected: set[str] = set(must_include)
    for sym, _ in ranked:
        if len(selected) >= max_symbols:
            break
        selected.add(sym)

    filtered = {sym: symbols_data[sym] for sym in symbols_data if sym in selected}
    n_filtered = len(symbols_data) - len(filtered)
    if n_filtered > 0:
        _usage_stats["symbols_filtered"] += n_filtered
        logger.info(
            "Pre-filter: %d -> %d symbols (dropped %d low-score)",
            len(symbols_data), len(filtered), n_filtered,
        )

    return filtered, scores


def should_skip_cycle(
    symbols_data: dict[str, Any],
    ml_signals: "dict[str, dict] | None" = None,
    held_symbols: "set[str] | None" = None,
    market_regime: "dict | None" = None,
    new_listings: "set[str] | None" = None,
) -> tuple[bool, str]:
    """Decide whether to skip calling Claude this cycle.

    Skip when:
    - No open positions AND market is flat (all RSI 40-60, low volume, no ML consensus)
    - No held positions to evaluate for SELL
    - No new listings to evaluate

    Returns (should_skip, reason).
    """
    if not settings.SKIP_FLAT_CYCLES:
        return False, ""

    held = held_symbols or set()

    # Never skip if we have open positions — always evaluate SELL opportunities
    if held:
        return False, ""

    # Never skip if new listings are in the universe — high-opportunity window
    if new_listings and new_listings & symbols_data.keys():
        return False, ""

    # Check regime — skip in choppy/flat markets
    regime = (market_regime or {}).get("regime", "unknown")
    if regime in ("strong_uptrend", "strong_downtrend", "uptrend", "downtrend"):
        return False, ""  # trending → opportunities exist

    # Check if any symbol has actionable signals
    has_rsi_extreme = False
    has_volume_spike = False
    has_ml_consensus = False

    for sym, data in symbols_data.items():
        ind_1h = data.get("indicators", {}).get("1h", {})
        rsi = ind_1h.get("rsi14", 50)
        if rsi < 30 or rsi > 70:
            has_rsi_extreme = True
        vol_ratio = ind_1h.get("volume_ratio", 1.0)
        if vol_ratio > 1.8:
            has_volume_spike = True

    if ml_signals:
        for sym, sig in ml_signals.items():
            lstm_sig = sig.get("lstm", {}).get("signal", "HOLD")
            rl_sig = sig.get("rl", {}).get("action", "HOLD")
            if lstm_sig == rl_sig and lstm_sig != "HOLD":
                has_ml_consensus = True
                break

    if not has_rsi_extreme and not has_volume_spike and not has_ml_consensus:
        reason = f"Flat market (regime={regime}), no RSI extremes, no volume spikes, no ML consensus"
        logger.info("Skip-cycle: %s", reason)
        record_skipped_cycle()
        return True, reason

    return False, ""


def choose_model_tier(
    symbols_data: dict[str, Any],
    ml_signals: "dict[str, dict] | None" = None,
    held_symbols: "set[str] | None" = None,
    market_regime: "dict | None" = None,
) -> bool:
    """Decide whether to use Haiku (cheap) or Sonnet (powerful).

    Returns True if Haiku should be used.
    Uses Sonnet for:
    - Open positions (SELL decisions are high-stakes)
    - Strong ML consensus on BUY
    - Trending markets with volume
    """
    if not settings.USE_HAIKU_FOR_HOLD:
        return False  # always Sonnet

    held = held_symbols or set()
    if held:
        return False  # Sonnet for SELL decisions

    # Strong ML consensus → Sonnet
    if ml_signals:
        for sym, sig in ml_signals.items():
            lstm_sig = sig.get("lstm", {}).get("signal", "HOLD")
            rl_sig = sig.get("rl", {}).get("action", "HOLD")
            lstm_conf = sig.get("lstm", {}).get("confidence", 0)
            if lstm_sig == rl_sig and lstm_sig == "BUY" and lstm_conf > 0.6:
                return False  # Sonnet for high-conviction

    # Strong trend → Sonnet (opportunities likely)
    regime = (market_regime or {}).get("regime", "unknown")
    if regime in ("strong_uptrend", "strong_downtrend"):
        return False

    # Check for any extreme signals
    for sym, data in symbols_data.items():
        ind_1h = data.get("indicators", {}).get("1h", {})
        rsi = ind_1h.get("rsi14", 50)
        vol_ratio = ind_1h.get("volume_ratio", 1.0)
        if (rsi < 25 or rsi > 75) and vol_ratio > 1.5:
            return False  # Sonnet for extreme setups

    # Default: Haiku for quiet cycles
    return True
