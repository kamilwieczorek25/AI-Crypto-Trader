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

# ── 2. Copy .env if missing ──────────────────────────────────────────────────
if [ ! -f "$PROJECT_DIR/.env" ]; then
  echo ""
  echo "▶ Creating .env from template..."
  cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
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
fi

# ── 3. Check ANTHROPIC_API_KEY is set ────────────────────────────────────────
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

# ── 6. Optional: install systemd service for auto-start on reboot ───────────
SERVICE_FILE="/etc/systemd/system/ai-trader.service"
if [ ! -f "$SERVICE_FILE" ]; then
  echo ""
  read -rp "▶ Install systemd service for auto-start on reboot? [y/N] " ANSWER
  if [[ "$ANSWER" =~ ^[Yy]$ ]]; then
    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=AI Crypto Trader
After=docker.service
Requires=docker.service

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
echo "   View logs:    docker compose -C $PROJECT_DIR logs -f"
echo "   Stop:         docker compose -C $PROJECT_DIR down"
echo "   Restart:      docker compose -C $PROJECT_DIR restart"
echo "   Update:       git -C $PROJECT_DIR pull && docker compose -C $PROJECT_DIR up --build -d"
echo "═══════════════════════════════════════════════"
