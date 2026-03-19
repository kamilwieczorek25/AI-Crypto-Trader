"""Claude AI engine — prompt builder + tool_use caller."""

import json
import logging
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic

from app.config import settings
from app.schemas.decision import TradeDecision

logger = logging.getLogger(__name__)


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
        min_confidence=0.38,
        max_position_pct=5.0,
        system_addendum="""\

RISK PROFILE: FAST PROFIT (maximum aggression)
- Trade when confidence ≥ 0.38. Bias strongly toward BUY/SELL over HOLD.
- Maximum position size: 5% of portfolio.
- PRIMARY timeframes: 15m and 1h only. Use 4h/1d only as trend filter.
- Chase momentum: if RSI is rising + MACD histogram is positive + volume ratio > 1.5, BUY.
- Use tight stop-losses (2–4%) and ambitious take-profits (15–30%).
- Sell quickly when momentum stalls — do not hold losing positions.
- Look for coins with the highest volume ratio and strongest short-term trend.
- Speed is the priority: capture short-term swings, not long-term holds.
- HOLD is a last resort. Always look for the best available opportunity.
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

You receive:
- Time of day and active trading session
- Current portfolio state (cash, positions, P&L)
- BTC/USDT anchor data (trend, RSI, momentum)
- Market regime detection (uptrend/downtrend/ranging/choppy)
- Correlation warnings for highly-correlated pairs
- USDT altcoins with full indicator data + support/resistance levels
- Order book pressure for each symbol
- Recent news sentiment per symbol (with min/max/trend)
- AI Signal Ensemble: LSTM neural-net direction probabilities + RL agent Q-value recommendations
- RAG context: relevant snippets from past trades and market insights

When the LSTM and RL agent agree on a direction, treat that as a strong corroborating signal.
When they disagree, exercise extra caution — prefer HOLD unless technical indicators are compelling.
Always explain in your reasoning how you weighted the AI ensemble signals.
"""


def _get_profile() -> RiskProfile:
    return _PROFILES.get(settings.RISK_PROFILE, _PROFILES["balanced"])


def _build_system_prompt() -> str:
    profile = _get_profile()
    return _BASE_SYSTEM_PROMPT + profile.system_addendum


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
) -> str:
    from datetime import datetime, timezone
    profile = _get_profile()
    lines: list[str] = []

    # Derive the set of held symbols for validation hints throughout the prompt
    held_symbols: set[str] = {
        pos["symbol"] for pos in portfolio.get("positions", [])
    }

    # Time context — trading sessions affect crypto volatility
    now = datetime.now(timezone.utc)
    hour = now.hour
    weekday = now.strftime("%A")
    if 0 <= hour < 8:
        session = "Asia (low altcoin liquidity, BTC-dominated)"
    elif 8 <= hour < 14:
        session = "Europe (increasing volume, altcoin activity picks up)"
    elif 14 <= hour < 21:
        session = "US (peak volume, highest volatility, most institutional flow)"
    else:
        session = "Late US / pre-Asia (declining volume, spreads may widen)"

    lines.append(f"=== ACTIVE RISK PROFILE: {profile.label.upper()} ===")
    lines.append(
        f"Min confidence: {profile.min_confidence} | "
        f"Max position: {profile.max_position_pct}% | "
        f"Total symbols to evaluate: {len(symbols_data)}"
    )
    lines.append(f"Time: {now.strftime('%Y-%m-%d %H:%M')} UTC ({weekday}) | Session: {session}")
    lines.append(
        f"Min confidence: {profile.min_confidence} | "
        f"Max position: {profile.max_position_pct}% | "
        f"Total symbols to evaluate: {len(symbols_data)}"
    )

    # Portfolio summary
    lines.append("\n=== PORTFOLIO STATE ===")
    lines.append(f"Total value: ${portfolio['total_value_usdt']:,.2f} USDT")
    lines.append(
        f"Cash: ${portfolio['cash_usdt']:,.2f} "
        f"({portfolio['cash_usdt'] / max(portfolio['total_value_usdt'], 1) * 100:.1f}%)"
    )
    lines.append(f"Positions value: ${portfolio['positions_value_usdt']:,.2f}")
    lines.append(
        f"Total P&L: ${portfolio['total_pnl_usdt']:+,.2f} ({portfolio['total_pnl_pct']:+.2f}%)"
    )
    lines.append(f"Open positions: {portfolio['num_open_positions']}")

    if portfolio.get("positions"):
        lines.append("\nOpen positions (SELL-eligible symbols):")
        for pos in portfolio["positions"]:
            lines.append(
                f"  {pos['symbol']}: {pos['quantity']:.4f} units @ ${pos['avg_entry_price']:.4f}"
                f"  current=${pos['current_price']:.4f}  P&L={pos['pnl_pct']:+.2f}%"
            )
    else:
        lines.append("\nSELL-eligible symbols: NONE (no open positions — do not choose SELL)")

    lines.append("\n=== MARKET ANALYSIS ===")

    # Market regime context (helps adjust aggression)
    if market_regime and market_regime.get("regime") != "unknown":
        lines.append(f"\n=== MARKET REGIME ===")
        lines.append(f"  Regime: {market_regime['regime'].upper()}")
        lines.append(f"  Volatility: {market_regime.get('volatility', '?')}")
        lines.append(f"  Avg 20-bar return: {market_regime.get('avg_return_pct', 0):+.2f}%")
        lines.append(f"  Avg volatility: {market_regime.get('avg_volatility_pct', 0):.2f}%")
        lines.append(f"  Guidance: {market_regime.get('description', '')}")

    # Correlation warnings (avoid concentration risk)
    if correlation_info and correlation_info.get("high_correlation_pairs"):
        lines.append(f"\n=== CORRELATION WARNINGS ===")
        lines.append("  These pairs are highly correlated (>0.8) — avoid holding both:")
        for pair in correlation_info["high_correlation_pairs"]:
            lines.append(
                f"  ⚠ {pair['pair'][0]} ↔ {pair['pair'][1]}: r={pair['correlation']}"
            )

    # BTC market anchor (BTC drives altcoin market; its trend is critical context)
    if btc_anchor:
        lines.append("\n=== BTC MARKET ANCHOR (BTC/USDT — not tradeable, context only) ===")
        for tf, ind in btc_anchor.items():
            trend_label = "BULLISH" if ind.get("trend", 0) > 0 else "BEARISH"
            lines.append(
                f"  [{tf}] Price=${ind.get('close', 0):.2f}"
                f"  RSI={ind.get('rsi14', 0):.1f}"
                f"  MACD_hist={ind.get('macd_hist', 0):.2f}"
                f"  Trend={trend_label}"
                f"  VolRatio={ind.get('volume_ratio', 1):.2f}"
                f"  OBV={'↑' if ind.get('obv_trend', 0) > 0 else '↓' if ind.get('obv_trend', 0) < 0 else '→'}"
            )
        lines.append("  ⚡ When BTC is bearish, most altcoins fall. Exercise extra caution buying alts.")

    for symbol, data in symbols_data.items():
        held_tag = "  ★ HELD — SELL eligible" if symbol in held_symbols else ""
        lines.append(f"\n── {symbol}{held_tag} ──")
        lines.append(f"  Current price: ${data.get('price', 0):.6f}")

        for tf, ind in data.get("indicators", {}).items():
            trend_label = "BULLISH" if ind.get("trend", 0) > 0 else "BEARISH"
            data_flag = "" if ind.get("data_status", 1) > 0 else " [INSUFFICIENT DATA]"
            rsi_div = ind.get("rsi_divergence", 0)
            div_label = ""
            if rsi_div > 0:
                div_label = " RSI_DIV=BULLISH"
            elif rsi_div < 0:
                div_label = " RSI_DIV=BEARISH"
            lines.append(
                f"  [{tf}]{data_flag} RSI={ind.get('rsi14', 0):.1f}"
                f"  MACD_hist={ind.get('macd_hist', 0):.4f}"
                f"  BB%B={ind.get('bb_pct_b', 0.5):.2f}"
                f"  EMA20=${ind.get('ema20', 0):.4f}"
                f"  VolRatio={ind.get('volume_ratio', 1):.2f}"
                f"  VolTrend={ind.get('volume_trend', 1):.2f}"
                f"  OBV={'↑' if ind.get('obv_trend', 0) > 0 else '↓' if ind.get('obv_trend', 0) < 0 else '→'}"
                f"  ATR={ind.get('atr', 0):.4f}"
                f"  Trend={trend_label}{div_label}"
            )

        ob = data.get("orderbook", {})
        lines.append(
            f"  OrderBook: spread={ob.get('spread_pct', 0):.4f}%"
            f"  bid_wall={ob.get('bid_wall', 0):.2f}"
            f"  ask_wall={ob.get('ask_wall', 0):.2f}"
            f"  buy/sell_pressure={ob.get('pressure_ratio', 1):.3f}"
        )

        # Support / Resistance levels
        sr = data.get("support_resistance", {})
        if sr.get("nearest_support") or sr.get("nearest_resistance"):
            sr_parts = [f"  S/R: position={sr.get('price_vs_sr', '?')}"]
            if sr.get("nearest_support"):
                sr_parts.append(f"nearest_support=${sr['nearest_support']:.6f}")
            if sr.get("nearest_resistance"):
                sr_parts.append(f"nearest_resistance=${sr['nearest_resistance']:.6f}")
            lines.append("  ".join(sr_parts))

        ns = news.get(symbol, {})
        lines.append(
            f"  News: {ns.get('article_count', 0)} articles"
            f"  sentiment={ns.get('avg_sentiment', 0):+.3f}"
            f"  range=[{ns.get('min_sentiment', 0):+.2f},{ns.get('max_sentiment', 0):+.2f}]"
            f"  trend={ns.get('sentiment_trend', 0):+.3f}"
        )
        for headline in ns.get("headlines", [])[:2]:
            lines.append(f"    • {headline}")

    # ── AI Signal Ensemble (LSTM + RL) ──────────────────────────────────────
    if ml_signals:
        lines.append("\n=== AI SIGNAL ENSEMBLE ===")
        lines.append(
            "(LSTM neural-net direction probabilities + RL agent Q-value recommendations)"
        )
        agree_syms: list[str] = []
        for sym, sig in ml_signals.items():
            lstm  = sig.get("lstm", {})
            rl    = sig.get("rl",   {})
            lstm_signal = lstm.get("signal", "?")
            lstm_conf   = lstm.get("confidence", 0.0)
            lstm_status = lstm.get("status", "?")
            rl_action   = rl.get("action", "?")
            rl_trained  = rl.get("trained", False)
            rl_epsilon  = rl.get("epsilon", 1.0)
            rl_steps    = rl.get("steps", 0)
            q           = rl.get("q_values", {})
            q_str = "  ".join(f"{a}={v:.3f}" for a, v in q.items())
            lines.append(
                f"  {sym}:"
                f"  LSTM={lstm_signal}({lstm_conf:.0%}, {lstm_status})"
                f"  probs: BUY={lstm.get('BUY',0):.0%} HOLD={lstm.get('HOLD',0):.0%} SELL={lstm.get('SELL',0):.0%}"
            )
            # Warn when ML says SELL but we have no position to sell
            sell_note = ""
            if lstm_signal == "SELL" or rl_action == "SELL":
                if sym not in held_symbols:
                    sell_note = "  ⚠ SELL INVALID — not held, treat as HOLD"
            lines.append(
                f"         RL={rl_action}"
                f"  trained={rl_trained} ε={rl_epsilon:.2f} steps={rl_steps}"
                f"  Q[{q_str}]{sell_note}"
            )
            # Only count consensus when the action is actually executable
            if lstm_signal == rl_action and lstm_signal != "HOLD":
                if lstm_signal == "SELL" and sym not in held_symbols:
                    pass  # skip invalid SELL consensus
                else:
                    agree_syms.append(f"{sym}→{lstm_signal}")
        if agree_syms:
            lines.append(
                f"  ** CONSENSUS SIGNALS (LSTM+RL agree): {', '.join(agree_syms)} **"
            )
        else:
            lines.append("  (No strong LSTM+RL consensus — exercise caution)")

    # ── RAG context ──────────────────────────────────────────────────────
    if rag_context:
        lines.append("\n=== HISTORICAL CONTEXT (from past trades & market memory) ===")
        for snippet in rag_context:
            lines.append(f"  • {snippet}")

    lines.append(
        f"\nSelect the single best opportunity from the {len(symbols_data)} symbols above. "
        f"Remember your profile: {profile.label}. "
        f"For SELL decisions, consider using sell_pct < 100 for partial profit taking when the trend is still intact. "
        f"Make your trading decision now."
    )
    return "\n".join(lines)


# ── Token usage / cost tracker (in-memory, resets on restart) ────────────────
# claude-sonnet-4-6 pricing: $3/M input tokens, $15/M output tokens
_INPUT_COST_PER_M  = 3.0
_OUTPUT_COST_PER_M = 15.0

_usage_stats: dict = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_calls": 0,
    "total_cost_usd": 0.0,
}


def get_usage_stats() -> dict:
    return dict(_usage_stats)


def _record_usage(input_tokens: int, output_tokens: int) -> None:
    cost = (input_tokens / 1_000_000 * _INPUT_COST_PER_M
            + output_tokens / 1_000_000 * _OUTPUT_COST_PER_M)
    _usage_stats["total_input_tokens"]  += input_tokens
    _usage_stats["total_output_tokens"] += output_tokens
    _usage_stats["total_calls"]         += 1
    _usage_stats["total_cost_usd"]      += cost
    logger.info(
        "Claude usage: in=%d out=%d cost=$%.4f | session total=$%.4f",
        input_tokens, output_tokens, cost, _usage_stats["total_cost_usd"],
    )


async def call_claude(prompt: str) -> tuple[TradeDecision, str]:
    """Call Claude with tool_use. Returns (decision, raw_response_json).
    One retry on failure. Raises on second failure.
    """
    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    profile = _get_profile()
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    system_prompt = _build_system_prompt()

    async def _attempt() -> tuple[TradeDecision, str]:
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system_prompt,
            tools=[_TOOL_DEFINITION],
            tool_choice={"type": "tool", "name": "make_trading_decision"},
            messages=[{"role": "user", "content": prompt}],
        )
        # Record token usage
        if response.usage:
            _record_usage(response.usage.input_tokens, response.usage.output_tokens)

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

        if decision.confidence < profile.min_confidence and decision.action != "HOLD":
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
        logger.warning("Claude first attempt failed: %s — retrying", first_err)
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
    }


PROFILE_KEYS = list(_PROFILES.keys())
