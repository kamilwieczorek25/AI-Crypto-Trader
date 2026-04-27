"""Application configuration — loaded from environment / .env file."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Walk up from backend/app/ → backend/ → project root to find .env
_HERE = Path(__file__).parent          # backend/app/
_ENV_FILE = next(
    (p / ".env" for p in [_HERE.parent.parent, _HERE.parent, _HERE] if (p / ".env").exists()),
    _HERE.parent.parent / ".env",      # fallback: project root
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Anthropic
    ANTHROPIC_API_KEY: str = ""
    # Admin API key (sk-ant-admin01-...) — needed for credit balance only (optional)
    ANTHROPIC_ADMIN_KEY: str = ""

    # Binance
    BINANCE_API_KEY: str = ""
    BINANCE_SECRET: str = ""

    # Quote currency — determines which trading pairs the bot uses
    # "USDT" for /USDT pairs, "USDC" for /USDC pairs
    QUOTE_CURRENCY: str = "USDC"

    # Trading mode
    MODE: str = "demo"           # "demo" | "real"
    REAL_TRADING: bool = False   # second safety gate

    # Risk profile — controls Claude's aggressiveness
    # conservative | balanced | aggressive | fast_profit
    RISK_PROFILE: str = "balanced"

    # Auto-adjust risk profile based on market regime (overrides RISK_PROFILE each cycle)
    AUTO_RISK_PROFILE: bool = True

    # Less-fear mode: override Claude's conservative HOLD bias
    # When enabled: lowers quant threshold, disables auto-risk downgrade, forces Claude to approve high-score candidates
    LESS_FEAR: bool = False

    # Lock risk profile: prevent AUTO_RISK_PROFILE from auto-switching the profile.
    # The manually selected profile is kept regardless of market regime.
    LOCK_RISK_PROFILE: bool = False

    # Max drawdown circuit breaker (% of initial balance — pauses bot if exceeded)
    MAX_DRAWDOWN_PCT: float = 15.0

    # Discord webhook URL for trade notifications (blank = disabled)
    DISCORD_WEBHOOK_URL: str = ""

    # ── Cost optimisation ─────────────────────────────────────────────────
    # Max symbols to send to Claude (pre-filtered by score). 0 = all.
    MAX_PROMPT_SYMBOLS: int = 8
    # Skip Claude call when market is flat (saves ~20-40% of calls)
    SKIP_FLAT_CYCLES: bool = True
    # Use cheaper Haiku model for routine/hold cycles, Sonnet for SELL decisions
    USE_HAIKU_FOR_HOLD: bool = True
    # Express lane: skip Claude validation entirely when LESS_FEAR=True.
    # With LESS_FEAR on we override Claude's HOLD anyway, so the call is pure waste.
    EXPRESS_SKIP_CLAUDE_WHEN_LESS_FEAR: bool = True
    # Hard cap: max Claude calls the express lane may make per 60-second window.
    EXPRESS_MAX_CLAUDE_PER_MINUTE: int = 1
    # Hard cap: max Claude calls the express lane may make per hour. 0 = unlimited.
    EXPRESS_MAX_CLAUDE_PER_HOUR: int = 4
    # Per-symbol express Claude cooldown (minutes) — don't re-call Claude for the
    # same symbol within this window even if it stays hot.  0 = disabled.
    EXPRESS_CLAUDE_SYMBOL_COOLDOWN_MIN: int = 15

    # Main-cycle Claude bypass — let GPU model take the decision directly.
    # Tier 1: LESS_FEAR bypass — if LESS_FEAR=True and we'd override Claude's HOLD
    #   anyway, skip the call entirely (mirrors what express lane already does).
    MAIN_CYCLE_SKIP_CLAUDE_WHEN_LESS_FEAR: bool = True
    # Tier 2: High-conviction GPU bypass — skip Claude when quant score AND GPU
    #   ensemble are both confident. Claude would agree in >90% of these cases.
    #   Set to 0.0 to disable. Recommended: 75.0
    SKIP_CLAUDE_ABOVE_SCORE: float = 75.0
    # Minimum GPU ensemble confidence required for the high-conviction bypass.
    SKIP_CLAUDE_GPU_MIN_CONFIDENCE: float = 0.65
    # Tier 3: Hard hourly cap on main-cycle Claude calls (0 = unlimited).
    #   Even if tiers 1+2 miss, this prevents a call storm in active markets.
    MAIN_CYCLE_MAX_CLAUDE_PER_HOUR: int = 2

    # ── Quant scorer ────────────────────────────────────────────────────────
    # Minimum composite score (0-100) to consider a trade candidate
    MIN_QUANT_SCORE: float = 60.0
    # ATR multiplier for stop-loss (2.5 = 2.5x ATR — gives room for noise)
    SL_ATR_MULTIPLIER: float = 2.5
    # Minimum reward-to-risk ratio (2.5 = TP must be 2.5x SL distance — ensures
    # winners outpace losers; raised from 2.0 to fix asymmetric R:R bleed.)
    MIN_REWARD_RISK_RATIO: float = 2.5
    # Min/max stop-loss % (clamp ATR-based SL to sane range)
    MIN_SL_PCT: float = 3.0
    MAX_SL_PCT: float = 8.0
    # SELL trigger for held positions: emit SELL when score drops this many
    # points below the regime-adjusted BUY threshold (was hard-coded to 45).
    # Smaller value = more eager to sell weak positions in flat markets.
    QUANT_SELL_THRESHOLD_DELTA: float = 10.0

    # ── Trend / regime filters (stop trading in chop & downtrends) ────────
    # When True, no new BUY candidates are produced when market regime is
    # downtrend / strong_downtrend / choppy.  Existing positions are still
    # managed (SELL/exit logic stays active).
    REGIME_BLOCK_BUYS_IN_DOWNTREND: bool = True
    # When True, require EMA50 > EMA200 on the 1h timeframe before approving a
    # BUY.  Prevents knife-catching in confirmed downtrends.
    EMA_TREND_FILTER_ENABLED: bool = True
    # Minimum ADX value for momentum/breakout BUYs (filters whipsaws in chop).
    # 0 = disabled.  Recommended: 18-22.
    MIN_ADX_FOR_BREAKOUT: float = 20.0
    # Pause new BUYs after N losing trades inside the rolling 5-trade window.
    # Resets when a winner appears.  0 = disabled.  Recommended: 3.
    LOSS_STREAK_PAUSE_THRESHOLD: int = 3
    # How long to pause BUYs after the loss-streak threshold fires (hours).
    LOSS_STREAK_PAUSE_HOURS: float = 6.0

    # ── Hard risk limits (always win over %-based limits) ─────────────────
    # Hard $ ceiling on loss per trade (0 = disabled, use SL_PCT only).
    # When set, position size is also capped so that hitting the SL would
    # not lose more than this many USDC.
    MAX_LOSS_PER_TRADE_USD: float = 0.0
    # Daily loss circuit breaker: pause new BUYs for the rest of the UTC
    # day if cumulative realised + unrealised PnL drops below this %.
    # Resets at 00:00 UTC.  0 = disabled.  Recommended: -3 to -5.
    DAILY_LOSS_LIMIT_PCT: float = -4.0
    # Per-symbol cooldown after a profit-take exit (minutes).  Stops the
    # bot from repeatedly buying back into a coin that just gave us TP.
    POST_TP_SYMBOL_COOLDOWN_MIN: int = 240
    # Block new BUYs when a held position correlates >= this with the
    # candidate over the last 20 1-h candles.  0 = disabled.
    MAX_HELD_CORRELATION: float = 0.75
    # Tighten MIN_QUANT_SCORE by this amount on Saturday & Sunday UTC.
    # Crypto liquidity halves on weekends and whipsaws are common.
    WEEKEND_SCORE_BUMP: float = 5.0
    # Halt all trading when |USDC/USDT spread| > this fraction (depeg
    # detection).  0 = disabled.  Default: 0.005 = 0.5%.
    STABLECOIN_DEPEG_THRESHOLD: float = 0.005

    # ── Trading fees ──────────────────────────────────────────────────────
    # Set to True if you've enabled "Pay fees with BNB" on Binance and hold
    # BNB \u2014 fees drop from 0.10% to 0.075% per side (25% discount).
    BNB_FEE_DISCOUNT: bool = False

    # ── Auto-backtest & tuning ─────────────────────────────────────────────
    # Run backtest on bot startup and auto-tune scorer settings
    AUTO_BACKTEST: bool = True
    # Re-run backtest every N hours (0 = startup only, no repeat)
    BACKTEST_INTERVAL_HOURS: float = 24.0
    # How many days of history per backtest run (shorter = faster)
    BACKTEST_DAYS: int = 30
    # How many symbols to include in backtest
    BACKTEST_SYMBOLS: int = 8

    # ── DCA (Dollar-Cost Averaging) ──────────────────────────────────────
    # Split BUY entries into two tranches for better average price
    DCA_ENABLED: bool = True
    # First tranche as % of total position (rest is pending DCA order)
    DCA_SPLIT_PCT: float = 60.0
    # Dip % below entry for second tranche to fill
    DCA_DIP_PCT: float = 2.0

    # ── Kelly criterion sizing ───────────────────────────────────────────
    # Use Kelly formula (from backtest results) to scale position sizes
    KELLY_SIZING: bool = True
    # Cap on Kelly fraction (0.5 = half-Kelly, safer than full Kelly)
    KELLY_FRACTION_CAP: float = 0.5

    # ── Trailing take-profit ──────────────────────────────────────────────
    # When price hits TP, don't sell immediately — let it run.
    # Sell only when price pulls back this % from the peak above TP.
    TRAILING_TP_PULLBACK_PCT: float = 4.0
    # Safety floor: if price drops back below original TP, sell immediately.
    TRAILING_TP_FLOOR: bool = True
    # Minutes to block re-entry after a take-profit exit.
    # Prevents churning in/out of the same trending coin repeatedly.
    TP_COOLDOWN_MINUTES: int = 45

    # ── Smart exit engine ────────────────────────────────────────────────
    # Runs every cycle for open positions.  Combines GPU Exit RL (DQN),
    # local reversal detector (10 technical signals), and profit lock.
    # When disabled, only SL/TP/time-exit triggers remain active.
    SMART_EXIT_ENABLED: bool = True

    # ── Profit lock (protect unrealised gains before TP) ────────────────
    # When a position reaches PROFIT_LOCK_ACTIVATE_PCT unrealised gain,
    # a sliding floor is placed.  The floor = max(FLOOR_PCT, peak × KEEP_PCT).
    # If the gain drops to that floor, the position is sold — locking in
    # profit instead of riding all the way back to stop-loss.
    # Examples with current defaults (activate=6, floor=2, keep=75%):
    #   peaked +6%  → floor = max(2%, 6%×0.75)  = +4.5% → sell at +4.5%
    #   peaked +10% → floor = max(2%, 10%×0.75) = +7.5% → sell at +7.5%
    #   peaked +20% → floor = max(2%, 20%×0.75) = +15%  → sell at +15%
    # Tightened from old 3/1/50 (which clipped winners to net <1% after fees,
    # producing negative EV in flat markets).  Set ACTIVATE = 0 to disable.
    PROFIT_LOCK_ACTIVATE_PCT: float = 6.0   # activate when PnL >= +6%
    PROFIT_LOCK_FLOOR_PCT: float = 2.0      # minimum floor (absolute %)
    PROFIT_LOCK_KEEP_PCT: float = 75.0      # keep this % of peak gains

    # ── Time-based exit ──────────────────────────────────────────────────
    # Max hours to hold a stagnant position (0 = disabled)
    MAX_HOLD_HOURS: float = 48.0

    # ── Pyramiding (add to winners) ──────────────────────────────────────
    # Add to profitable positions when quant score stays high
    PYRAMID_ENABLED: bool = True
    # Minimum P&L % before pyramiding is considered
    PYRAMID_MIN_PNL_PCT: float = 3.0
    # Minimum quant score for the symbol to pyramid into
    PYRAMID_MIN_SCORE: float = 70.0
    # How much to add as % of original position value
    PYRAMID_ADD_PCT: float = 50.0

    # ── Whale trade detector ───────────────────────────────────────────
    # Minimum trade size (USDT) to qualify as a "whale" trade
    WHALE_MIN_USDT: float = 50_000.0
    # How many minutes to remember whale events (rolling window)
    WHALE_MEMORY_MINUTES: int = 30
    # Verify TLS certificates for whale detector Binance WebSocket.
    # Set to false only in controlled environments where cert interception
    # prevents validation.
    WHALE_WS_VERIFY_SSL: bool = True
    # If certificate validation fails and WHALE_WS_VERIFY_SSL is true,
    # allow a one-time fallback to insecure TLS (ssl=False).
    WHALE_WS_INSECURE_FALLBACK: bool = True

    # ── Fast local scanner ──────────────────────────────────────────────
    # Background scanner interval (seconds) — checks wide altcoin universe
    SCANNER_INTERVAL_SECONDS: int = 60
    # Minimum 24h volume to include in fast scan (lower than main universe)
    SCANNER_MIN_VOLUME_USDT: float = 300_000.0
    # Max hot-list candidates to inject into main cycle
    SCANNER_HOT_LIST_SIZE: int = 10
    # Minimum scanner score (0-100) to qualify as "hot"
    SCANNER_MIN_SCORE: float = 15.0
    # Maximum 24h price change (%) to consider a coin. Tokens already up >X%
    # are likely near the top of a pump — buying them is extremely risky.
    # 0 = disabled.  Recommended: 30-60 (tightened from 80 to reduce FOMO).
    SCANNER_MAX_24H_PCT: float = 50.0
    # Require positive short-term momentum for the scanner to flag a coin.
    # When enabled, coins with falling price get score halved.
    SCANNER_PREFER_UPTREND: bool = True

    # ── Gainer injection ─────────────────────────────────────────────────
    # Minimum 24h volume for a gainer to be injected into the main cycle.
    # Independent of MIN_VOLUME_USDT so USDC pairs with lower liquidity still get caught.
    GAINER_MIN_VOLUME_USDT: float = 50_000.0

    # ── Local LLM (Ollama) ────────────────────────────────────────────
    # When USE_LOCAL_LLM=True the bot prefers Ollama on the GPU VM for
    # trade validation and falls back to Claude if Ollama is unavailable.
    # LOCAL_LLM_URL: leave blank to auto-derive from GPU_SERVER_URL
    #   (same host, port 11434).  Set explicitly to override.
    USE_LOCAL_LLM: bool = True
    LOCAL_LLM_URL: str = ""              # blank = derive from GPU_SERVER_URL
    LOCAL_LLM_MODEL: str = "qwen2.5:14b"
    # GPU inference: ~5–15 s.  CPU-only (7B): ~60–120 s.  CPU-only (14B): ~180 s.
    LOCAL_LLM_TIMEOUT: int = 120         # seconds per inference call

    # ── Remote GPU server (optional) ─────────────────────────────────
    # URL of the GPU inference server (e.g. http://192.168.1.50:9090)
    # When set, LSTM/RL training and prediction are offloaded to that machine.
    # When blank, ML runs locally on CPU (default behaviour).
    GPU_SERVER_URL: str = ""
    # Timeout in seconds for GPU server HTTP calls
    GPU_SERVER_TIMEOUT: int = 120
    # Shared auth token for GPU server (must match GPU_SERVER_TOKEN on the GPU machine)
    GPU_SERVER_TOKEN: str = ""

    # ── Exchange sync ─────────────────────────────────────────────────
    # Sync portfolio state from Binance on startup (real mode only)
    SYNC_EXCHANGE_ON_STARTUP: bool = True
    # Minimum holding value (USDT) to import as position (filters dust)
    SYNC_MIN_VALUE_USDT: float = 5.0

    # Startup warmup: block new BUY trades for this many seconds after start/restart
    # so OHLCV caches and ML signals can warm up before any entry decisions are made.
    # 0 = disabled (trade immediately).  Default: 90 s (one sleep-loop pass + margin).
    STARTUP_WARMUP_SECONDS: int = 90

    # Bot behaviour
    CYCLE_INTERVAL_SECONDS: int = 300
    DEMO_INITIAL_BALANCE: float = 10_000.0
    MAX_POSITION_PCT: float = 25.0
    MAX_TOTAL_EXPOSURE_PCT: float = 70.0
    # Maximum open positions at any time.  On small balances (< $500),
    # fewer concentrated positions beat many tiny ones — each position
    # must be large enough to absorb round-trip fees (~0.2%) and still
    # produce meaningful gains.  3 positions × ~$40–50 each on $150.
    MAX_OPEN_POSITIONS: int = 3

    # Symbol universe
    # TOP_N_SYMBOLS: 0 = all pairs passing the volume filter, >0 = hard cap
    TOP_N_SYMBOLS: int = 0
    # Minimum 24h quote volume (USDT) to be included in the universe
    MIN_VOLUME_USDT: float = 5_000_000.0
    # Separate floor for USDC quote pairs — USDC pairs trade at 10-20× lower
    # volume than USDT equivalents, so the $5M bar makes the universe nearly empty.
    # $300K still filters dust while covering the full mid-cap USDC tier.
    MIN_VOLUME_USDT_USDC: float = 300_000.0
    # Max concurrent OHLCV/orderbook fetches (avoid rate-limit bans)
    FETCH_CONCURRENCY: int = 10
    # Top gainers: min 24h price change % to qualify as a "gainer"
    GAINER_MIN_PCT: float = 5.0
    # How many top gainers to inject into the symbol universe each cycle.
    # Lowered from 10 → 4 because gainer injection was the prime FOMO source
    # (chasing already-pumped coins gave the bot a strong buy-side bias).
    GAINER_INJECT_COUNT: int = 4
    # Skip gainers whose 24h move is already above this %, to avoid buying tops.
    # 0 = disabled.
    GAINER_MAX_PCT: float = 30.0
    # How many hours to keep watching a newly listed coin (auto-inject into universe)
    NEW_LISTING_WATCH_HOURS: float = 48.0

    # News
    CRYPTOCOMPARE_API_KEY: str = ""

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://ai_trader:ai_trader@localhost:5432/ai_trader"

    # Logging
    LOG_LEVEL: str = "INFO"

    # CORS allowlist for frontend origins (comma-separated) or "*"
    CORS_ORIGINS: str = "*"

    @field_validator("CYCLE_INTERVAL_SECONDS")
    @classmethod
    def _cycle_interval_positive(cls, v: int) -> int:
        if v < 10:
            raise ValueError("CYCLE_INTERVAL_SECONDS must be >= 10")
        return v

    @field_validator("DEMO_INITIAL_BALANCE")
    @classmethod
    def _initial_balance_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("DEMO_INITIAL_BALANCE must be > 0")
        return v

    @field_validator("MIN_VOLUME_USDT")
    @classmethod
    def _min_volume_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("MIN_VOLUME_USDT must be >= 0")
        return v

    @property
    def is_demo(self) -> bool:
        return self.MODE.lower() != "real"

    @property
    def real_trading_allowed(self) -> bool:
        return self.MODE.lower() == "real" and self.REAL_TRADING


settings = Settings()
