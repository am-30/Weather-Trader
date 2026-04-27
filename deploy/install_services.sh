#!/bin/bash
# =============================================================================
# Kalshi Weather Trader — Install systemd Services
# Run this after setup_vm.sh and after filling in your .env file.
#
# Usage (run as root on the VM):
#   bash /home/trader/kalshi-weather-trader/deploy/install_services.sh
# =============================================================================
set -euo pipefail

APP_DIR="/home/trader/kalshi-weather-trader"
ENV_FILE="$APP_DIR/.env"

# Check .env has been filled in
if grep -q "your-key-id-here" "$ENV_FILE" 2>/dev/null; then
    echo "ERROR: $ENV_FILE still contains placeholder values."
    echo "       Please fill in KALSHI_ACCESS_KEY and KALSHI_PRIVATE_KEY first."
    exit 1
fi

echo "Installing systemd services..."

# Make wrapper scripts executable
chmod +x "$APP_DIR/deploy/run_orchestrator.sh"
chmod +x "$APP_DIR/deploy/run_dashboard.sh"

# Copy unit files
cp "$APP_DIR/deploy/orchestrator.service" /etc/systemd/system/kalshi-orchestrator.service
cp "$APP_DIR/deploy/dashboard.service"    /etc/systemd/system/kalshi-dashboard.service

systemctl daemon-reload

systemctl enable kalshi-orchestrator
systemctl enable kalshi-dashboard

systemctl restart kalshi-orchestrator
systemctl restart kalshi-dashboard

echo ""
echo "Services started. Status:"
echo ""
systemctl status kalshi-orchestrator --no-pager -l | head -20
echo ""
systemctl status kalshi-dashboard --no-pager -l | head -20

cat <<DONE

============================================================
 Services are running!
============================================================

Useful commands:

  View trading engine logs:
    journalctl -u kalshi-orchestrator -f

  View dashboard logs:
    journalctl -u kalshi-dashboard -f

  Restart a service:
    systemctl restart kalshi-orchestrator
    systemctl restart kalshi-dashboard

  Stop everything:
    systemctl stop kalshi-orchestrator kalshi-dashboard

Dashboard URL:
  http://$(curl -s ifconfig.me 2>/dev/null || echo "<your-droplet-ip>"):5000

DONE
