# AI Crypto Trader

A production-ready autonomous trading bot for Binance that uses **Claude (claude-sonnet-4-6)** as its decision engine. The bot combines a 27-factor quantitative signal stack, LSTM + reinforcement learning models, real-time whale detection, market intelligence, and news sentiment analysis — then routes the top candidates to Claude for structured BUY / SELL / HOLD decisions via forced `tool_use`.

Runs in **demo mode** (paper trading) by default. Live trading requires explicit opt-in.

---

## What It Can Do

### Signal Intelligence
- **27-factor quant scorer** (0–100 scale): RSI, MACD, MACD divergence, Bollinger Bands + squeeze detection, Bollinger squeeze setup, volume ratio, volume Z-score, OBV trend, VWAP position, multi-timeframe trend alignment, support/resistance proximity, BTC correlation anchor, BTC beta, orderbook bid/ask pressure, depth imbalance, funding rate (contrarian), long/short ratio, open interest trend, whale flow, ML consensus, GPU momentum ranking, sector rotation heat, breakout signal, momentum acceleration, news burst detection, and **24h price change momentum**
- **LSTM price predictor** — 2-layer LSTM trained on 500-candle windows across top symbols; predicts SELL / HOLD / BUY with probability distribution
- **DQN reinforcement learning agent** — learns from live paper-trading outcomes via experience replay; improves every cycle
- **Real-time whale detector** — monitors Binance WebSocket trade stream for large trades (default ≥ $50k USDT) and injects whale flow signal into the scorer
- **Background fast scanner** — continuously scans a wide altcoin universe every 60 seconds (independent of the main cycle) to surface breakout candidates early
- **Express lane** — when the fast scanner detects a high-scoring candidate between main cycles, an immediate lightweight focused analysis fires (OHLCV fetch → re-score → Haiku validation → execute) within seconds, without waiting for the full 5-min cycle. Includes a 5-minute cooldown per symbol to prevent spam.
- **Top gainers injection** — automatically adds top 24h performers above a configurable % threshold into the universe each cycle
- **New listing tracking** — newly listed coins are auto-injected for 48 hours regardless of volume
- **Market intelligence module** — aggregated data layer pulling Fear & Greed index, CoinGecko trending coins, Binance funding rates, long/short ratios, open interest changes, BTC dominance (altseason detection), and per-symbol short squeeze detection — all cached with TTL
- **Session-aware thresholds** — score thresholds auto-adjust by UTC trading session (lower during Asian pump hours, higher during thin-liquidity dead zones, weekend penalty)

### AI Decision Layer
- **Claude as validator, not originator** — only pre-scored, high-conviction candidates are sent to Claude, keeping API costs low and focusing reasoning where it matters
- **Four risk profiles**: `conservative` (confidence ≥ 0.75, max 2% position), `balanced` (≥ 0.55, max 5%), `aggressive` (≥ 0.45), `fast_profit` (≥ 0.40, short holds)
- **Auto risk profile** — detects market regime each cycle (bull / bear / sideways) and adjusts the profile automatically
- **Less-fear mode** — overrides Claude's conservative bias: lowers quant threshold (55 → 45), blocks auto-downgrade to conservative profile, forces Claude to approve high-scoring candidates it would otherwise HOLD. Toggleable via dashboard button or `LESS_FEAR=true` in `.env`
- **Three-tier Claude gating** — Claude is only called when all local models agree; default is auto-HOLD with no API call. Tier 1: LESS_FEAR bypass (skip Claude when override is active). Tier 2: GPU high-conviction bypass (skip when quant score + GPU ensemble both confident). Tier 3: hourly hard cap (default 4 calls/hour). Uses Haiku for quiet cycles, Sonnet for actionable signals
- **RAG memory** — retrieves relevant historical trade outcomes and news context before each Claude call; profitable past trades surface higher in results (outcome-weighted + recency-decayed, SQLite-persisted across restarts)

### Trade Execution
- **DCA (Dollar-Cost Averaging)** — splits BUY entries into two tranches (default 60% / 40%), with the second tranche filling on a configurable dip % for a better average entry
- **Kelly criterion sizing** — scales position sizes using edge estimates from backtest results (half-Kelly cap for safety)
- **Pyramiding** — adds to profitable open positions when quant score remains high and P&L exceeds threshold
- **ATR-based stop-loss / take-profit** — dynamic levels calibrated to each symbol's volatility; minimum reward-to-risk ratio enforced
- **Monte Carlo SL/TP refinement** — GPU runs 10K simulated price paths to validate and tighten SL/TP; if simulated edge is negative, TP is reduced automatically
- **Smart exit engine** — three-layer intelligent exit system that runs every cycle for open positions:
  - **GPU Exit RL** (Dueling DQN) — when trained, its CLOSE/PARTIAL predictions are now executed automatically based on Q-value confidence spread
  - **Local reversal detector** — 10 independent technical signals (RSI overbought/crash, MACD bearish flip, price below VWAP, BB lower band, drawdown from peak, OBV bearish, BTC dragging down, GPU ensemble SELL, anomaly detected, volatility spike) — needs 3+ confirmations to close, 2+ for partial exit
  - **Profit lock sliding floor** — when a position peaks at +X%, a dynamic floor at max(1%, X×50%) protects gains (e.g. peaked +10% → sells at +5% if gains erode)
- **Trailing take-profit** — when price hits TP, instead of selling immediately, activates trailing mode: tracks the peak and sells only when price pulls back a configurable % from that peak (or drops below original TP as safety floor)
- **Partial sells** — Claude, Exit RL, and the smart exit engine can all direct partial position exits (25%, 50%, etc.) instead of all-or-nothing closes
- **Time-based exits** — stagnant positions automatically closed after a configurable hold duration
- **Exchange sync** — in real mode, imports actual Binance balances and positions into the portfolio state every cycle. Pre-existing holdings imported as "external" (read-only)
- **Position adoption** — external positions can be converted to bot-managed via dashboard button or `POST /api/bot/adopt-positions`. Sets SL/TP based on current price and config defaults, enabling full bot management (SL/TP monitoring, trailing stops, Exit RL, sell signals)

### Risk Management
- **Demo / real mode gate** — two independent flags (`MODE=real` AND `REAL_TRADING=true`) required for live orders
- **Max drawdown circuit breaker** — pauses the bot if portfolio drops a configurable % from its peak
- **Position size caps** — max % per trade, max total altcoin exposure, and max concurrent open positions (default 3) enforced at both the prompt and executor level
- **Banned symbol auto-detection** — when the exchange rejects a symbol (not permitted, market closed, invalid symbol, bad request), the bot auto-bans it for the session; temporary errors get a 15-minute cooldown instead
- **Discord notifications** — trade alerts, errors, and daily summaries sent to a webhook

### Auto-Tuning
- **Auto-backtest** — runs on startup and every 24 hours, testing scorer parameters against recent history
- **Auto-tuner** — adjusts `MIN_QUANT_SCORE`, `SL_ATR_MULTIPLIER`, and Kelly fraction based on backtest results

### Optional GPU Server
Run `gpu-server/server.py` on any machine (GPU optional — falls back to CPU) to unlock:
- **Transformer model** with multi-head self-attention (replaces LSTM, longer 30-candle window, 10 features)
- **Dueling Double DQN** with prioritized experience replay (replaces vanilla DQN)
- **Semantic news sentiment** via `sentence-transformers/all-MiniLM-L6-v2` (replaces keyword counting)
- **Full ensemble endpoint** — Transformer (45%) + LSTM (25%) + RL (15%) + Sentiment (15%) combined signal
- **Multi-Timeframe Fusion** — single Transformer sees 15m+1h+4h+1d simultaneously, learns cross-TF patterns (e.g. "15m reversal while 4h trends up")
- **Volatility Forecasting** — LSTM predicts future σ for better SL/TP placement and Monte Carlo accuracy
- **Anomaly Detection Autoencoder** — flags pump-and-dumps, flash crashes, whale manipulation; blocks entry automatically
- **Optimal Exit RL** — dedicated Dueling DQN trained only on exit timing: HOLD / PARTIAL_25% / PARTIAL_50% / CLOSE. Predictions are now executed by the smart exit engine when Q-value confidence is sufficient
- **Attention Explainability** — extracts which candles and features the Transformer focused on, shown to Claude
- **Cross-Symbol Correlation Tracker** — GPU-parallel Pearson correlation matrix; detects divergence = mean-reversion signals, used to reject BUY candidates correlated (r > 0.8) with existing positions

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Dashboard (Nginx)                  │  :9080
│         Chart.js · WebSocket live updates           │
└────────────────────┬────────────────────────────────┘
                     │ REST + WebSocket
┌────────────────────▼────────────────────────────────┐
│                 FastAPI Backend                     │  :9000
│                                                     │
│  bot_runner ──► quant_scorer ──► claude_engine      │
│       │              │                │             │
│  fast_scanner   27 signals      tool_use API        │
│  whale_detector  market_data    RAG context         │
│  backtester      technical      risk profiles       │
│  auto_tuner      news/intel     3-tier gating       │
│  exit_analyzer   market_intel   cost optimiser      │
│       │                                             │
│  executor ──► Binance (ccxt) ── demo / real mode    │
│       │                                             │
│  PostgreSQL (portfolio · trades · decisions · RAG)  │
└────────────────────┬────────────────────────────────┘
                     │ HTTP (optional)
┌────────────────────▼────────────────────────────────┐
│            GPU Inference Server (10 models)         │  :9090
│                                                     │
│  ┌─ Price Direction ───────────────────────────┐    │
│  │  Transformer · LSTM · Multi-TF Fusion       │    │
│  └─────────────────────────────────────────────┘    │
│  ┌─ Risk Management ──────────────────────────┐    │
│  │  Dueling DQN · Exit RL · Anomaly Detector   │    │
│  └─────────────────────────────────────────────┘    │
│  ┌─ Market Intelligence ──────────────────────┐    │
│  │  Volatility Forecast · Correlation Tracker  │    │
│  │  Semantic Sentiment · Attention Explainer   │    │
│  └─────────────────────────────────────────────┘    │
│  ┌─ Simulation ───────────────────────────────┐    │
│  │  Monte Carlo (10K GPU-parallel paths)       │    │
│  └─────────────────────────────────────────────┘    │
│  Background training loop (60s) · Data augmentation │
│           (any machine, GPU optional)               │
└─────────────────────────────────────────────────────┘
```

**Stack:** Python 3.11+ · FastAPI · SQLAlchemy (PostgreSQL + SQLite) · ccxt · pandas-ta · PyTorch · Anthropic SDK · Nginx

---

## Quick Start — Docker (recommended)

```bash
git clone https://github.com/kamilwieczorek25/AI-Crypto-Trader.git
cd AI-Crypto-Trader

cp .env.example .env
```

Edit `.env` — at minimum set your Anthropic API key:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

Then start:

```bash
docker compose up --build
```

| URL | What |
|-----|------|
| `http://localhost:9080` | Dashboard |
| `http://localhost:9000/docs` | API (Swagger) |
| `http://localhost:9000/api/health` | Health check |

The bot auto-starts on server startup. The dashboard shows live portfolio stats, open positions, Claude's last reasoning, and trade history.

---

## Quick Start — Local (no Docker)

Requires **Python 3.10+**.

```bash
git clone https://github.com/kamilwieczorek25/AI-Crypto-Trader.git
cd AI-Crypto-Trader

cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY at minimum

bash run-local.sh
```

The script:
1. Creates a Python virtualenv and installs all dependencies (including PyTorch CPU-only)
2. Starts the frontend on `http://localhost:9080`
3. Starts the FastAPI backend on `http://localhost:9000`
4. On macOS, activates `caffeinate` to keep the machine awake

Re-running `bash run-local.sh` kills any previous instance and restarts cleanly.

---

## Deploy on Ubuntu (headless server)

Requires Docker already installed.

```bash
git clone https://github.com/kamilwieczorek25/AI-Crypto-Trader.git ~/AI-Crypto-Trader
cd ~/AI-Crypto-Trader

cp .env.example .env
nano .env   # add ANTHROPIC_API_KEY at minimum

bash deploy-ubuntu.sh
```

The script:
1. Installs PostgreSQL if not present, creates the `ai_trader` database and user
2. Configures `pg_hba.conf` to allow Docker container connections
3. Builds Docker images and starts the services via `docker compose`
4. Opens firewall ports (9000, 9080) via `ufw` if active
5. Waits for the health check to pass

The dashboard is then accessible at `http://<server-ip>:9080` from your local network.

> **Database:** Ubuntu deployment uses PostgreSQL by default (via asyncpg). Local development uses SQLite. The backend auto-detects the driver from `DATABASE_URL`.

---

## Optional: GPU Inference Server

Run on any machine (GPU or CPU). Unlocks the Transformer model, Dueling DQN, and semantic sentiment.

```bash
cd gpu-server
pip install -r requirements.txt
# For CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124
python server.py
```

Then add to `.env`:

```env
GPU_SERVER_URL=http://<machine-ip>:9090
```

Running it on the **same machine** as the bot (`http://localhost:9090`) works fine and still unlocks all the better models — no dedicated GPU machine required.

---

## Trading Modes

| `MODE` | `REAL_TRADING` | Behaviour |
|--------|---------------|-----------|
| `demo` | `false` | Paper trading with simulated fills (default) |
| `demo` | `true` | Paper trading (second flag ignored) |
| `real` | `false` | Blocked — will error on start |
| `real` | `true` | **Live Binance orders** |

Demo mode simulates realistic slippage (0.03–0.12%) and Binance taker fees (0.1%) on every fill.

---

## Key Configuration

All settings live in `.env`. See `.env.example` for the full list with comments.

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | **Required** |
| `BINANCE_API_KEY` / `BINANCE_SECRET` | — | Required for real mode; demo works without |
| `MODE` | `demo` | `demo` or `real` |
| `REAL_TRADING` | `false` | Second safety gate for live orders |
| `QUOTE_CURRENCY` | `USDC` | Quote asset — `USDC` or `USDT` |
| `RISK_PROFILE` | `balanced` | `conservative` · `balanced` · `aggressive` · `fast_profit` |
| `AUTO_RISK_PROFILE` | `true` | Auto-adjust profile based on market regime |
| `MAX_POSITION_PCT` | `25.0` | Max % of portfolio per single trade |
| `MAX_TOTAL_EXPOSURE_PCT` | `70.0` | Max % of portfolio in altcoins at once |
| `MAX_OPEN_POSITIONS` | `3` | Max concurrent open positions (0 = unlimited) |
| `MAX_DRAWDOWN_PCT` | `15.0` | Circuit breaker — pauses bot if exceeded |
| `CYCLE_INTERVAL_SECONDS` | `300` | How often a full analysis cycle runs |
| `MIN_QUANT_SCORE` | `60.0` | Minimum score (0–100) to consider a trade |
| `SL_ATR_MULTIPLIER` | `2.5` | Stop-loss = ATR × this multiplier |
| `MIN_REWARD_RISK_RATIO` | `2.0` | TP distance must be ≥ 2× SL distance |
| `DCA_ENABLED` | `true` | Split BUY entries into two tranches |
| `KELLY_SIZING` | `true` | Scale position size from backtest edge |
| `PYRAMID_ENABLED` | `true` | Add to profitable open positions |
| `LESS_FEAR` | `false` | Override conservative bias — lower thresholds, force buys |
| `TRAILING_TP_PULLBACK_PCT` | `6.0` | Sell only after this % pullback from peak above TP |
| `TRAILING_TP_FLOOR` | `true` | Safety: sell immediately if price drops back below TP |
| `SMART_EXIT_ENABLED` | `true` | Enable smart exit engine (Exit RL + reversal detector + profit lock) |
| `PROFIT_LOCK_ACTIVATE_PCT` | `3.0` | Arm profit lock when position reaches this % gain |
| `PROFIT_LOCK_FLOOR_PCT` | `1.0` | Minimum floor % (absolute) for profit lock |
| `PROFIT_LOCK_KEEP_PCT` | `50.0` | Keep this % of peak gains (sliding floor) |
| `MAIN_CYCLE_MAX_CLAUDE_PER_HOUR` | `4` | Hard hourly cap on main-cycle Claude API calls |
| `SKIP_CLAUDE_GPU_MIN_CONFIDENCE` | `0.65` | Min GPU ensemble confidence to bypass Claude |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL (default) or `sqlite+aiosqlite:///...` |
| `DISCORD_WEBHOOK_URL` | — | Trade alerts and errors (optional) |
| `GPU_SERVER_URL` | — | GPU inference server URL (optional) |
| `GPU_SERVER_TOKEN` | — | Shared auth token for GPU server (optional) |

---

## Dashboard Features

- **Portfolio summary** — total value, cash balance, unrealised P&L, win rate
- **Open positions table** — live P&L per position with entry price, SL/TP levels, hold duration, and source badge (bot / external). External positions have an **Adopt** button to convert them to bot-managed.
- **Adopt All External** — bulk-adopt button appears when external positions exist
- **Mode / risk / less-fear toggles** — dashboard buttons with rich hover tooltips explaining what each mode changes (confidence thresholds, position sizes, SL ranges, auto-select conditions)
- **Claude's last decision** — full reasoning, signals used, confidence score, and model tier (Sonnet / Haiku)
- **Price chart** — candlestick with RSI and MACD panels (Chart.js)
- **Trade history** — paginated log of all executed trades with P&L
- **Live updates** — WebSocket push for portfolio and trade events; no manual refresh needed

---

## Safety Notes

- Never commit `.env` — it contains live API keys. The repo's `.gitignore` excludes it by default.
- Start in `MODE=demo` and run for at least a full day before switching to real.
- Keep `MAX_POSITION_PCT` reasonable for your balance — for small balances (<$500), 25–40% with `MAX_OPEN_POSITIONS=3` concentrates capital into meaningful positions.
- Set `MAX_DRAWDOWN_PCT` before going live — it is your last automatic safety net.
- The smart exit engine (`SMART_EXIT_ENABLED=true`) runs independently of Claude and will close positions when technical indicators confirm a reversal — no API cost for exit decisions.
