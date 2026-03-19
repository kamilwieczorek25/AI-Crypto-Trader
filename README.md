# AI Crypto Trader

A production-ready Python AI trading bot on Binance that uses Claude as the decision engine.

## Quick Start

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY (required)
# Optionally fill BINANCE_API_KEY + BINANCE_SECRET for real market data
# Leave MODE=demo (default) for paper trading

docker-compose up --build
```

Open `http://localhost` in your browser. Click **START** to run the first cycle.

## Architecture

- **Backend**: FastAPI + SQLAlchemy (SQLite) + ccxt + pandas-ta + Anthropic Claude
- **Frontend**: Nginx serving vanilla JS + Chart.js
- **AI Engine**: Claude `claude-sonnet-4-6` with forced `tool_use` for structured decisions

## Safety

- `MODE=demo` (default) — paper trading only, no real orders ever placed
- `MODE=real` + `REAL_TRADING=true` — required for live orders (triple-gated)
- Max 5% portfolio per trade, max 70% total altcoin exposure (enforced at prompt + executor level)

## Modes

| MODE | REAL_TRADING | Behaviour |
|------|-------------|-----------|
| demo | false | Paper trading (default) |
| demo | true  | Paper trading |
| real | false | Blocked — will error |
| real | true  | Live Binance orders |

## Environment Variables

See `.env.example` for all options.

## Dashboard

`http://<host-ip>` — accessible on your home network when running in Docker.

Features:
- Live portfolio stats (total value, cash, P&L)
- Open positions table with live P&L
- Claude's last decision with reasoning
- Price chart with RSI + MACD panels
- Paginated trade history
