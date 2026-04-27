#!/usr/bin/env bash
# ─── deploy-ubuntu.sh ────────────────────────────────────────────────────────
# Deploys AI Trader on Ubuntu. Assumes Docker is already installed.
# Assumes repo is already cloned to ~/AI_Trader-1  (or wherever this script is).
# Run as a regular user with sudo privileges.
# Usage: bash ~/AI_Trader-1/deploy-ubuntu.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

DB_NAME="ai_trader"
DB_USER="ai_trader"
DB_PASS="ai_trader"

echo "═══════════════════════════════════════════════"
echo " AI Crypto Trader — Ubuntu deployment"
echo " Project dir: $PROJECT_DIR"
echo "═══════════════════════════════════════════════"

# ── 1. Verify Docker is available ───────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "✗ Docker not found. Please install Docker first."
  exit 1
fi
echo "✔ Docker $(docker --version | grep -oP '\d+\.\d+\.\d+')"

if ! docker compose version &>/dev/null; then
  echo "✗ 'docker compose' plugin not found. Please install docker-compose-plugin."
  exit 1
fi
echo "✔ Docker Compose $(docker compose version --short)"

# ── 2. Install & configure PostgreSQL ────────────────────────────────────────
echo ""
echo "▶ Setting up PostgreSQL..."

if ! command -v psql &>/dev/null; then
  echo "  Installing PostgreSQL..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq postgresql postgresql-contrib > /dev/null
fi

# Ensure PostgreSQL is running
sudo systemctl start postgresql
sudo systemctl enable postgresql
echo "✔ PostgreSQL $(psql --version | grep -oP '\d+\.\d+')"

# Create database user and database (idempotent)
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" \
  | grep -q 1 || sudo -u postgres psql -c "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';"

sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" \
  | grep -q 1 || sudo -u postgres psql -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};" 2>/dev/null || true
sudo -u postgres psql -d "${DB_NAME}" -c "GRANT ALL ON SCHEMA public TO ${DB_USER};" 2>/dev/null || true
sudo -u postgres psql -d "${DB_NAME}" -c "ALTER SCHEMA public OWNER TO ${DB_USER};" 2>/dev/null || true
echo "✔ Database '${DB_NAME}' ready (user: ${DB_USER})"

# Allow Docker containers to connect via host.docker.internal (172.17.0.0/16)
PG_VERSION=$(ls /etc/postgresql/ | sort -V | tail -1)
PG_HBA="/etc/postgresql/${PG_VERSION}/main/pg_hba.conf"
PG_CONF="/etc/postgresql/${PG_VERSION}/main/postgresql.conf"

# Listen on all interfaces (needed for Docker containers to connect)
if ! grep -q "^listen_addresses = '\*'" "$PG_CONF" 2>/dev/null; then
  echo "  Configuring PostgreSQL to listen on all interfaces..."
  sudo sed -i "s/^#\?listen_addresses\s*=.*/listen_addresses = '*'/" "$PG_CONF"
fi

# Allow password auth from Docker networks (bridge + compose networks)
if ! grep -q "172.16.0.0/12" "$PG_HBA" 2>/dev/null; then
  echo "  Adding Docker networks to pg_hba.conf..."
  # Remove old narrow rule if present
  sudo sed -i '/172\.17\.0\.0\/16.*ai_trader/d' "$PG_HBA" 2>/dev/null || true
  echo "host    ${DB_NAME}    ${DB_USER}    172.16.0.0/12    md5" | sudo tee -a "$PG_HBA" > /dev/null
fi

sudo systemctl restart postgresql
echo "✔ PostgreSQL configured for Docker access"

# ── 3. Copy .env if missing ──────────────────────────────────────────────────
if [ ! -f "$PROJECT_DIR/.env" ]; then
  echo ""
  echo "▶ Creating .env from template..."
  cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
  # Set the DATABASE_URL to point to host PostgreSQL via Docker gateway
  sed -i "s|^DATABASE_URL=.*|DATABASE_URL=postgresql+asyncpg://${DB_USER}:${DB_PASS}@host.docker.internal:5432/${DB_NAME}|" "$PROJECT_DIR/.env"
  echo ""
  echo "╔══════════════════════════════════════════════════╗"
  echo "║  ACTION REQUIRED: Edit .env before continuing   ║"
  echo "║                                                  ║"
  echo "║  nano $PROJECT_DIR/.env                         ║"
  echo "║                                                  ║"
  echo "║  Required: ANTHROPIC_API_KEY                     ║"
  echo "║  Optional: BINANCE_API_KEY, BINANCE_SECRET       ║"
  echo "╚══════════════════════════════════════════════════╝"
  echo ""
  echo "After editing, re-run this script."
  exit 1
else
  # Ensure existing .env uses host.docker.internal for PostgreSQL
  if grep -q "^DATABASE_URL=postgresql" "$PROJECT_DIR/.env" && ! grep -q "host.docker.internal" "$PROJECT_DIR/.env"; then
    echo "  Updating DATABASE_URL to use host.docker.internal..."
    sed -i "s|^DATABASE_URL=postgresql+asyncpg://\([^@]*\)@localhost:|DATABASE_URL=postgresql+asyncpg://\1@host.docker.internal:|" "$PROJECT_DIR/.env"
  fi
fi

# ── 3b. Sync missing settings from .env.example → .env ──────────────────────
# For every key present in .env.example but absent in .env, append it with
# its default value (plus the preceding comment block for context).
# Secret / personal keys are skipped — the user must set those manually.
if [ -f "$PROJECT_DIR/.env.example" ]; then
  echo ""
  echo "▶ Checking .env for missing settings..."

  # Keys that require personal credentials — never auto-add with placeholder values
  _SECRET_KEYS="ANTHROPIC_API_KEY BINANCE_API_KEY BINANCE_SECRET DISCORD_WEBHOOK_URL CRYPTOCOMPARE_API_KEY GPU_SERVER_URL"

  ADDED=0
  pending_lines=()   # comment/blank lines buffered until we see a real key=value

  while IFS= read -r line; do
    # Accumulate comment and blank lines — they belong to the next real key
    if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "${line// }" ]]; then
      pending_lines+=("$line")
      continue
    fi

    # Real key=value line
    key="${line%%=*}"
    value="${line#*=}"

    # Skip secrets and placeholder values (user must fill these in manually)
    _skip=0
    for _sk in $_SECRET_KEYS; do
      [[ "$key" == "$_sk" ]] && { _skip=1; break; }
    done
    # Also skip if value itself looks like a placeholder
    [[ "$value" == *"sk-ant-"* || "$value" == *"your-"* || "$value" == *"<"*">"* ]] && _skip=1

    if [[ $_skip -eq 1 ]]; then
      pending_lines=()
      continue
    fi

    if [[ -n "$key" ]] && ! grep -q "^${key}=" "$PROJECT_DIR/.env"; then
      # Append a blank separator, the buffered comment block, then the key=value
      {
        echo ""
        printf '%s\n' "${pending_lines[@]}"
        echo "$line"
      } >> "$PROJECT_DIR/.env"
      echo "  + $key=$value"
      ADDED=$((ADDED + 1))
    fi

    pending_lines=()   # reset after every real key, added or not
  done < "$PROJECT_DIR/.env.example"

  if [ "$ADDED" -gt 0 ]; then
    echo "✔ Added $ADDED missing setting(s) to .env"
  else
    echo "✔ .env is up to date — no missing settings"
  fi
fi

# ── 3c. Apply recommended trading-strategy values ─────────────────────────
# These knobs control the bot's risk/reward profile. The historical defaults
# produced a bot that earned in strong uptrends but bled in flat/down markets.
# We FORCE the recommended values on every deploy so existing servers get the
# fix automatically.  Set FORCE_TRADING_PARAMS=0 in your shell to opt out.
FORCE_TRADING_PARAMS=${FORCE_TRADING_PARAMS:-1}
if [ "$FORCE_TRADING_PARAMS" = "1" ] && [ -f "$PROJECT_DIR/.env" ]; then
  echo ""
  echo "▶ Applying recommended trading-strategy parameters..."

  # key=value pairs to enforce. Anything missing is appended; anything
  # present is updated in place. Lines that match exactly are left alone.
  declare -A RECOMMENDED=(
    # Asymmetric exit fix \u2014 the #1 leak in flat markets
    [PROFIT_LOCK_ACTIVATE_PCT]="6.0"
    [PROFIT_LOCK_FLOOR_PCT]="2.0"
    [PROFIT_LOCK_KEEP_PCT]="75.0"
    [TRAILING_TP_PULLBACK_PCT]="4.0"
    # Risk:reward + stop range
    [MIN_REWARD_RISK_RATIO]="2.5"
    [MIN_SL_PCT]="3.0"
    [MAX_SL_PCT]="8.0"
    [MIN_QUANT_SCORE]="60"
    [QUANT_SELL_THRESHOLD_DELTA]="10"
    # Cash regime / trend filter / chop guard
    [REGIME_BLOCK_BUYS_IN_DOWNTREND]="true"
    [EMA_TREND_FILTER_ENABLED]="true"
    [MIN_ADX_FOR_BREAKOUT]="20"
    # Loss-streak circuit breaker
    [LOSS_STREAK_PAUSE_THRESHOLD]="3"
    [LOSS_STREAK_PAUSE_HOURS]="6"
    # De-bias the buy side
    [LESS_FEAR]="false"
    [RISK_PROFILE]="balanced"
    [GAINER_INJECT_COUNT]="4"
    [GAINER_MAX_PCT]="30.0"
    [SCANNER_MAX_24H_PCT]="50.0"
    # Per-trade concentration cap — vol-adjusted sizing keeps it lower
    # in volatile markets but caps at this for low-ATR setups.
    [MAX_POSITION_PCT]="25.0"
  )

  CHANGED=0
  for key in "${!RECOMMENDED[@]}"; do
    val="${RECOMMENDED[$key]}"
    # Skip if exact line already present (no change needed)
    if grep -qE "^${key}=${val}$" "$PROJECT_DIR/.env"; then
      continue
    fi
    if grep -qE "^${key}=" "$PROJECT_DIR/.env"; then
      # Replace existing value (escape & and / in val for sed)
      esc_val=$(printf '%s' "$val" | sed -e 's/[\\/&|]/\\&/g')
      sed -i -E "s|^${key}=.*|${key}=${esc_val}|" "$PROJECT_DIR/.env"
    else
      # Append at end of file
      printf '\n%s=%s\n' "$key" "$val" >> "$PROJECT_DIR/.env"
    fi
    echo "  ~ ${key}=${val}"
    CHANGED=$((CHANGED + 1))
  done
  if [ "$CHANGED" -gt 0 ]; then
    echo "✔ Applied $CHANGED recommended trading parameter(s)"
  else
    echo "✔ All recommended trading parameters already match"
  fi
fi

# ── 3d. Check ANTHROPIC_API_KEY is set ───────────────────────────────────────
ANTHROPIC_KEY=$(grep -E '^ANTHROPIC_API_KEY=' "$PROJECT_DIR/.env" | cut -d'=' -f2 | tr -d ' ')
if [ -z "$ANTHROPIC_KEY" ] || [ "$ANTHROPIC_KEY" = "sk-ant-..." ]; then
  echo ""
  echo "✗ ANTHROPIC_API_KEY is not set in .env"
  echo "  Edit: nano $PROJECT_DIR/.env"
  exit 1
fi

# ── 4. Open firewall ports (if ufw is active) ───────────────────────────────
if command -v ufw &>/dev/null && sudo ufw status | grep -q "Status: active"; then
  echo "▶ Opening firewall ports 9000 and 9080..."
  sudo ufw allow 9000/tcp
  sudo ufw allow 9080/tcp
fi

# ── 5. Build and start ───────────────────────────────────────────────────────
echo ""
echo "▶ Building Docker images (this may take a few minutes on first run)..."
cd "$PROJECT_DIR"
docker compose -f "$COMPOSE_FILE" pull --ignore-buildable 2>/dev/null || true
docker compose -f "$COMPOSE_FILE" build --pull

echo ""
echo "▶ Starting services..."
docker compose -f "$COMPOSE_FILE" up -d

echo ""
echo "▶ Waiting for backend to become healthy..."
for i in {1..20}; do
  if curl -sf http://localhost:9000/api/health &>/dev/null; then
    echo "✔ Backend is up!"
    break
  fi
  sleep 3
  echo "  Waiting... ($((i*3))s)"
done

# ── 5b. Optional: run a 30-day backtest as a sanity check ────────────────────
# Set RUN_BACKTEST=0 to skip (e.g. on a slow network or low-power host).
RUN_BACKTEST=${RUN_BACKTEST:-1}
BACKTEST_DAYS=${BACKTEST_DAYS:-30}
BACKTEST_SYMBOLS=${BACKTEST_SYMBOLS:-10}
if [ "$RUN_BACKTEST" = "1" ]; then
  echo ""
  echo "▶ Running ${BACKTEST_DAYS}-day backtest (top ${BACKTEST_SYMBOLS} symbols)..."
  echo "  (set RUN_BACKTEST=0 to skip, BACKTEST_DAYS=N to change window)"
  # Copy backtest.py into the running container (it isn't part of the image
  # because the Dockerfile only ships the backend/ tree).
  if docker compose -f "$COMPOSE_FILE" cp "$PROJECT_DIR/backtest.py" trader:/app/backtest.py; then
    # Run inside the container so all Python deps are available.  Output goes
    # to /data/last_backtest.json which is bind-mounted to the trader-data
    # volume, so it persists across restarts and is visible from the dashboard.
    if docker compose -f "$COMPOSE_FILE" exec -T trader \
        python /app/backtest.py \
          --days "$BACKTEST_DAYS" \
          --symbols "$BACKTEST_SYMBOLS" \
          --output /data/last_backtest.json; then
      echo "✔ Backtest complete \u2014 results saved to /data/last_backtest.json"
      echo "  View summary in the dashboard's 'Backtest' tab, or:"
      echo "    docker compose -f $COMPOSE_FILE exec trader cat /data/last_backtest.json | head -50"
    else
      echo "⚠ Backtest exited non-zero \u2014 deployment will continue."
      echo "  You can re-run later with:"
      echo "    docker compose -f $COMPOSE_FILE exec trader python /app/backtest.py --days 30"
    fi
  else
    echo "⚠ Could not copy backtest.py into container \u2014 skipping backtest."
  fi
fi

# ── 6. Optional: install systemd service for auto-start on reboot ───────────
SERVICE_FILE="/etc/systemd/system/ai-trader.service"
if [ ! -f "$SERVICE_FILE" ]; then
  echo ""
  read -rp "▶ Install systemd service for auto-start on reboot? [y/N] " ANSWER
  if [[ "$ANSWER" =~ ^[Yy]$ ]]; then
    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=AI Crypto Trader
After=docker.service postgresql.service
Requires=docker.service
Wants=postgresql.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$PROJECT_DIR
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable ai-trader
    echo "✔ Systemd service installed and enabled."
  fi
fi

# ── 7. Print summary ─────────────────────────────────────────────────────────
HOST_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "═══════════════════════════════════════════════"
echo " Deployment complete!"
echo ""
echo " Dashboard:  http://$HOST_IP:9080"
echo " API:        http://$HOST_IP:9000"
echo " API docs:   http://$HOST_IP:9000/docs"
echo ""
echo " Useful commands:"
echo "   View logs:    docker compose -f $COMPOSE_FILE logs -f"
echo "   Stop:         docker compose -f $COMPOSE_FILE down"
echo "   Restart:      docker compose -f $COMPOSE_FILE restart"
echo "   Update:       git -C $PROJECT_DIR pull && docker compose -f $COMPOSE_FILE up --build -d"
echo "═══════════════════════════════════════════════"
