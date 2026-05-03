#!/bin/bash
# =============================================================================
# Kalshi Weather Trader — VM Setup Script
# Run this once as root on a fresh Ubuntu 22.04 or 24.04 DigitalOcean droplet.
#
# Usage:
#   bash <(curl -fsSL https://raw.githubusercontent.com/am-30/Weather-Trader/main/deploy/setup_vm.sh) https://github.com/am-30/Weather-Trader
#   -- or --
#   bash setup_vm.sh https://github.com/am-30/Weather-Trader
# =============================================================================
set -euo pipefail

# --- Config ------------------------------------------------------------------
APP_USER="trader"
APP_DIR="/home/trader/kalshi-weather-trader"
VENV_DIR="/home/trader/venv"
DB_NAME="kalshi_trader"
DB_USER="trader"
DB_PASS="$(openssl rand -hex 20)"   # random password, saved to .env

GITHUB_REPO="${1:-}"
if [ -z "$GITHUB_REPO" ]; then
    echo ""
    read -rp "GitHub repo URL (e.g. https://github.com/yourname/repo): " GITHUB_REPO
fi

echo ""
echo "============================================================"
echo " Kalshi Weather Trader — VM Setup"
echo " Repo: $GITHUB_REPO"
echo "============================================================"
echo ""

# --- 1. System packages ------------------------------------------------------
echo "[1/9] Installing system packages..."
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    git \
    postgresql \
    postgresql-contrib \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    pkg-config \
    libssl-dev \
    libpq-dev \
    ufw

echo "      Done."

# --- 2. Swap file (before pip install — scipy/numpy need the headroom) --------
echo "[2/9] Setting up 2GB swap file..."
if swapon --show | grep -q /swapfile; then
    echo "      Swap already active, skipping."
else
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "      Done."
fi

# --- 3. Create app user ------------------------------------------------------
echo "[3/9] Creating app user '$APP_USER'..."
if id "$APP_USER" &>/dev/null; then
    echo "      User already exists, skipping."
else
    useradd -m -s /bin/bash "$APP_USER"
    echo "      Done."
fi

# --- 4. PostgreSQL setup -----------------------------------------------------
echo "[4/9] Setting up PostgreSQL database..."
systemctl enable postgresql
systemctl start postgresql

# Wait for PostgreSQL to be ready to accept connections
echo "      Waiting for PostgreSQL to be ready..."
for i in {1..15}; do
    sudo -u postgres psql -c "SELECT 1" >/dev/null 2>&1 && break
    sleep 2
done
sudo -u postgres psql -c "SELECT 1" >/dev/null 2>&1 || { echo "ERROR: PostgreSQL did not start in time."; exit 1; }
echo "      PostgreSQL is ready."

# Create role and database (idempotent)
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" \
    | grep -q 1 || sudo -u postgres psql -c \
    "CREATE ROLE $DB_USER WITH LOGIN PASSWORD '$DB_PASS';"

sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" \
    | grep -q 1 || sudo -u postgres psql -c \
    "CREATE DATABASE $DB_NAME OWNER $DB_USER;"

# Allow local connections without peer auth for the trader user
PG_HBA="$(sudo -u postgres psql -t -c "SHOW hba_file;" | xargs)"
if ! grep -q "^local.*$DB_NAME.*$DB_USER" "$PG_HBA" 2>/dev/null; then
    # Prepend rule before the default local rules
    sed -i "/^local.*all.*all/i local   $DB_NAME   $DB_USER   md5" "$PG_HBA"
    systemctl reload postgresql
fi

echo "      Done. Database: $DB_NAME, User: $DB_USER"

# --- 5. Clone repository -----------------------------------------------------
echo "[5/9] Cloning repository..."
if [ -d "$APP_DIR/.git" ]; then
    echo "      Repo already exists, pulling latest..."
    sudo -u "$APP_USER" git -C "$APP_DIR" pull
else
    sudo -u "$APP_USER" git clone "$GITHUB_REPO" "$APP_DIR"
fi
echo "      Done."

# --- 6. Python virtualenv + dependencies -------------------------------------
echo "[6/9] Setting up Python virtualenv and installing dependencies..."
sudo -u "$APP_USER" python3 -m venv "$VENV_DIR"
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"
echo "      Done."

# --- 7. Create .env from template --------------------------------------------
echo "[7/9] Creating .env file..."
ENV_FILE="$APP_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    echo "      .env already exists, skipping (delete it to regenerate)."
else
    DB_URL="postgresql://$DB_USER:$DB_PASS@localhost:5432/$DB_NAME"
    sed "s|postgresql://user:password@host:5432/dbname|$DB_URL|g" \
        "$APP_DIR/.env.example" > "$ENV_FILE"
    chown "$APP_USER:$APP_USER" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "      Created at $ENV_FILE"
    echo "      DATABASE_URL has been pre-filled: $DB_URL"
fi

# --- 8. Firewall -------------------------------------------------------------
echo "[8/9] Configuring firewall..."
ufw --force enable
ufw allow ssh
ufw allow 5000/tcp   # Streamlit dashboard
echo "      Done. Port 5000 open for the dashboard."

# --- 9. Database backup cron -------------------------------------------------
echo "[9/9] Installing database backup cron job..."
mkdir -p /var/backups/kalshi
chmod 755 /var/backups/kalshi

tee /usr/local/bin/backup-kalshi-db.sh > /dev/null << 'BACKUP_SCRIPT'
#!/bin/bash
BACKUP_DIR=/var/backups/kalshi
BACKUP_FILE="$BACKUP_DIR/kalshi_trader_$(date +%Y%m%d_%H%M%S).sql.gz"
sudo -u postgres pg_dump kalshi_trader | gzip > "$BACKUP_FILE"
chmod 644 "$BACKUP_FILE"
ls -t "$BACKUP_DIR"/kalshi_trader_*.sql.gz | tail -n +29 | xargs -r rm
BACKUP_SCRIPT

chmod +x /usr/local/bin/backup-kalshi-db.sh
echo '0 */6 * * * root /usr/local/bin/backup-kalshi-db.sh' > /etc/cron.d/kalshi-backup
echo "      Done. Backups run every 6 hours to /var/backups/kalshi/"

# --- Done --------------------------------------------------------------------
cat <<DONE

============================================================
 Setup complete!
============================================================

NEXT STEPS:

1. Fill in your secrets in:
     $ENV_FILE

   Required fields to fill in:
     KALSHI_ACCESS_KEY   — your key ID from kalshi.com
     KALSHI_PRIVATE_KEY  — your RSA private key (paste the full PEM block)
     KALSHI_ENV          — "demo" or "prod"

   DATABASE_URL is already filled in (Postgres is running locally).

2. If migrating from another droplet, run:
     bash $APP_DIR/deploy/migrate.sh

3. Otherwise, install and start the services:
     bash $APP_DIR/deploy/install_services.sh

4. Visit your dashboard at:
     http://$(curl -s ifconfig.me 2>/dev/null || echo "<your-droplet-ip>"):5000

DONE
