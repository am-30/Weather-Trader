#!/bin/bash
# =============================================================================
# Kalshi Weather Trader — Droplet Migration Helper
# Run this on the NEW droplet after setup_vm.sh, to restore data from the old droplet.
#
# Before running, copy these two files from the old droplet to /tmp on the new one:
#
#   From your Mac:
#     scp trader@<old-ip>:/var/backups/kalshi/<latest>.sql.gz /tmp/kalshi-migration.sql.gz
#     scp trader@<old-ip>:/home/trader/kalshi-weather-trader/.env /tmp/kalshi-migration.env
#   Then:
#     scp /tmp/kalshi-migration.sql.gz root@<new-ip>:/tmp/
#     scp /tmp/kalshi-migration.env    root@<new-ip>:/tmp/
#
# Usage (run as root on the new VM):
#   bash /home/trader/kalshi-weather-trader/deploy/migrate.sh
# =============================================================================
set -euo pipefail

APP_USER="trader"
APP_DIR="/home/trader/kalshi-weather-trader"
DB_NAME="kalshi_trader"
DB_USER="trader"
ENV_SRC="/tmp/kalshi-migration.env"
DB_SRC="/tmp/kalshi-migration.sql.gz"

echo ""
echo "============================================================"
echo " Kalshi Weather Trader — Migration"
echo "============================================================"
echo ""

# --- Check prerequisites -----------------------------------------------------
if [ ! -f "$ENV_SRC" ]; then
    echo "ERROR: $ENV_SRC not found."
    echo ""
    echo "Copy it from your Mac after pulling from the old droplet:"
    echo "  scp trader@<old-ip>:/home/trader/kalshi-weather-trader/.env /tmp/kalshi-migration.env"
    echo "  scp /tmp/kalshi-migration.env root@<new-ip>:/tmp/"
    exit 1
fi

if [ ! -f "$DB_SRC" ]; then
    echo "ERROR: $DB_SRC not found."
    echo ""
    echo "Copy it from your Mac after pulling from the old droplet:"
    echo "  scp trader@<old-ip>:/var/backups/kalshi/<latest>.sql.gz /tmp/kalshi-migration.sql.gz"
    echo "  scp /tmp/kalshi-migration.sql.gz root@<new-ip>:/tmp/"
    exit 1
fi

# --- Restore .env -------------------------------------------------------------
echo "[1/3] Restoring .env..."
cp "$ENV_SRC" "$APP_DIR/.env"
chown "$APP_USER:$APP_USER" "$APP_DIR/.env"
chmod 600 "$APP_DIR/.env"
echo "      Done."

# --- Restore database ---------------------------------------------------------
echo "[2/3] Restoring database from backup..."
# Drop and recreate to ensure a clean slate — pg_dump output has no DROP statements
# so restoring into an existing database with any objects causes conflicts.
sudo -u postgres psql -c "DROP DATABASE IF EXISTS $DB_NAME;"
sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
gunzip < "$DB_SRC" | sudo -u postgres psql "$DB_NAME"
echo "      Done."

# --- Start services -----------------------------------------------------------
echo "[3/3] Starting services..."
bash "$APP_DIR/deploy/install_services.sh"

# --- Cleanup ------------------------------------------------------------------
rm -f "$ENV_SRC" "$DB_SRC"
echo "      Removed migration files from /tmp."

cat <<DONE

============================================================
 Migration complete!
============================================================

Verify everything looks correct:
  Dashboard:            http://$(curl -s ifconfig.me 2>/dev/null || echo "<your-droplet-ip>"):5000
  Trading engine logs:  journalctl -u kalshi-orchestrator -f
  Dashboard logs:       journalctl -u kalshi-dashboard -f

Once confirmed working:
  1. Destroy the old droplet from DigitalOcean
  2. Update your Mac rsync backup command with the new IP:
       rsync -avz --progress trader@<new-ip>:/var/backups/kalshi/ ~/kalshi-backups/

DONE
