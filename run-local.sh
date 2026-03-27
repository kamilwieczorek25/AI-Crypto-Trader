#!/usr/bin/env bash
# ─── run-local.sh ────────────────────────────────────────────────────────────
# Runs the AI Trader locally (macOS / Linux) WITHOUT Docker.
# Requirements: Python 3.11+
# Usage: bash run-local.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
VENV_DIR="$PROJECT_DIR/.venv"

# ── 0. Find Python 3.10+ ─────────────────────────────────────────────────────
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10; do
  if command -v "$candidate" &>/dev/null; then
    PYTHON="$candidate"
    break
  fi
done
if [ -z "$PYTHON" ]; then
  # Fallback: check if plain python3 is >= 3.10
  if command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c "import sys; print(sys.version_info >= (3,10))")
    if [ "$PY_VER" = "True" ]; then
      PYTHON="python3"
    fi
  fi
fi
if [ -z "$PYTHON" ]; then
  echo "✗ Python 3.10+ is required. Install it with:"
  echo "   brew install python@3.12"
  exit 1
fi
echo "✔ Using $PYTHON ($(${PYTHON} --version))"

# ── 1. Copy .env if missing ──────────────────────────────────────────────────
if [ ! -f "$PROJECT_DIR/.env" ]; then
  echo "⚠  .env not found — copying from .env.example"
  cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
  echo "   Edit $PROJECT_DIR/.env and add your ANTHROPIC_API_KEY, then re-run."
  exit 1
fi

# ── 2. Create virtualenv ─────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "▶ Creating Python virtualenv..."
  "$PYTHON" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ── 3. Install dependencies ──────────────────────────────────────────────────
echo "▶ Installing Python dependencies..."
pip install --quiet --upgrade pip
# Install standard PyTorch build.
# On Apple Silicon this enables Metal (MPS) acceleration when available.
pip install --quiet torch
pip install --quiet -r "$BACKEND_DIR/requirements.txt"

# ── 3b. Kill any previous instances ──────────────────────────────────────────
echo "▶ Stopping previous instances..."
pkill -f "uvicorn app.main" 2>/dev/null || true
pkill -f "python.*http.server 9080" 2>/dev/null || true
pkill -f "caffeinate" 2>/dev/null || true
for port in 9000 9080; do
  lsof -ti :"$port" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
done
sleep 1

# ── 4. Create SQLite data dir ────────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/data"

# Override paths for local (relative to project data dir)
export DATABASE_URL="sqlite+aiosqlite:///$PROJECT_DIR/data/trader.db"
export DATA_DIR="$PROJECT_DIR/data"

# ── 5. Serve frontend in background with Python ──────────────────────────────
cleanup() {
  kill "$FRONTEND_PID" 2>/dev/null || true
  kill "$CAFFEINATE_PID" 2>/dev/null || true
  for port in 9000 9080; do
    lsof -ti :"$port" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

# ── 5a. Keep macOS awake while script is running ─────────────────────────────
CAFFEINATE_PID=""
if command -v caffeinate &>/dev/null; then
  caffeinate -dims &
  CAFFEINATE_PID=$!
  echo "✔ caffeinate active (Mac will stay awake)"
fi

echo "▶ Starting frontend on http://localhost:9080"
cd "$PROJECT_DIR/frontend"
"$PYTHON" -m http.server 9080 &> /tmp/ai_trader_frontend.log &
FRONTEND_PID=$!

# ── 6. Start FastAPI backend ─────────────────────────────────────────────────
echo "▶ Starting backend on http://localhost:9000"
echo "   Dashboard: http://localhost:9080"
echo "   API docs:  http://localhost:9000/docs"
echo ""
cd "$BACKEND_DIR"
uvicorn app.main:app --host 0.0.0.0 --port 9000 --reload

# Cleanup handled by trap above
