#!/bin/bash
# =============================================================================
# Kalshi Weather Trader — VM Setup Script
# Run this once as root on a fresh Ubuntu 22.04 or 24.04 DigitalOcean droplet.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/YOUR/REPO/main/deploy/setup_vm.sh | bash
#   -- or --
#   bash setup_vm.sh https://github.com/YOUR/REPO
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
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    git \
    postgresql \
    postgresql-contrib \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    pkg-config \
    libssl-dev \
    libpq-dev \
    ufw

echo "      Done."

# --- 2. Create app user ------------------------------------------------------
echo "[2/7] Creating app user '$APP_USER'..."
if id "$APP_USER" &>/dev/null; then
    echo "      User already exists, skipping."
else
    useradd -m -s /bin/bash "$APP_USER"
    echo "      Done."
fi

# --- 3. PostgreSQL setup -----------------------------------------------------
echo "[3/7] Setting up PostgreSQL database..."
systemctl enable postgresql
systemctl start postgresql

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

# --- 4. Clone repository -----------------------------------------------------
echo "[4/7] Cloning repository..."
if [ -d "$APP_DIR/.git" ]; then
    echo "      Repo already exists, pulling latest..."
    sudo -u "$APP_USER" git -C "$APP_DIR" pull
else
    sudo -u "$APP_USER" git clone "$GITHUB_REPO" "$APP_DIR"
fi
echo "      Done."

# --- 5. Python virtualenv + dependencies -------------------------------------
echo "[5/7] Setting up Python virtualenv and installing dependencies..."
sudo -u "$APP_USER" python3.11 -m venv "$VENV_DIR"
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install --quiet --upgrade pip
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install --quiet -e "$APP_DIR"
echo "      Done."

# --- 6. Create .env from template --------------------------------------------
echo "[6/7] Creating .env file..."
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

# --- 7. Firewall -------------------------------------------------------------
echo "[7/7] Configuring firewall..."
ufw --force enable
ufw allow ssh
ufw allow 5000/tcp   # Streamlit dashboard
echo "      Done. Port 5000 open for the dashboard."

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

2. If you have existing data to import from Replit:
     sudo -u $APP_USER psql $DB_NAME < /tmp/replit_db_export.sql

3. Install and start the services:
     bash $APP_DIR/deploy/install_services.sh

4. Visit your dashboard at:
     http://$(curl -s ifconfig.me 2>/dev/null || echo "<your-droplet-ip>"):5000

DONE
