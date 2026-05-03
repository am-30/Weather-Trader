# Deploying to DigitalOcean

## What you need
- A DigitalOcean account
- Your Kalshi API key and RSA private key
- This repo pushed to GitHub (github.com/am-30/Weather-Trader)

---

## Fresh setup on a new droplet

### Step 1 — Create a DigitalOcean Droplet

1. Log in to DigitalOcean → Create → Droplets
2. Choose: **Ubuntu 22.04**, **Basic**, **Regular**, **$6/mo (1 vCPU / 1GB RAM)** with a 2GB swap file (handled automatically by setup script)
3. Choose a datacenter region close to you
4. Add your SSH key
5. Click Create

### Step 2 — Run the setup script

SSH into your new droplet as root, then run:
```bash
bash <(curl -fsSL https://raw.githubusercontent.com/am-30/Weather-Trader/main/deploy/setup_vm.sh) https://github.com/am-30/Weather-Trader
```

This installs all system packages, PostgreSQL, Python, the app, UFW firewall, a 2GB swap file, and a database backup cron job.

### Step 3 — Fill in your secrets

```bash
nano /home/trader/kalshi-weather-trader/.env
```

Fill in:
- `KALSHI_ACCESS_KEY` — your key ID from kalshi.com
- `KALSHI_PRIVATE_KEY` — your RSA private key (paste the full PEM block, replacing newlines with `\n`)
- `KALSHI_ENV` — `demo` for paper trading, `prod` for real money

`DATABASE_URL` is already filled in by the setup script.

### Step 4 — Start the services

```bash
bash /home/trader/kalshi-weather-trader/deploy/install_services.sh
```

### Step 5 — Open the dashboard

Visit `http://<your-droplet-ip>:5000` in your browser.

---

## Migrating from an existing droplet

Use this when moving to a new droplet (e.g. resizing to a smaller plan).

### Step 1 — On the old droplet, run a fresh backup

```bash
sudo /usr/local/bin/backup-kalshi-db.sh
ls /var/backups/kalshi/   # note the latest filename
```

### Step 2 — On your Mac, pull the backup and .env

```bash
scp trader@<old-ip>:/var/backups/kalshi/<latest>.sql.gz /tmp/kalshi-migration.sql.gz
scp trader@<old-ip>:/home/trader/kalshi-weather-trader/.env /tmp/kalshi-migration.env
```

### Step 3 — Create a new droplet and run setup_vm.sh

Follow Steps 1–2 from the fresh setup section above.

### Step 4 — Push migration files to the new droplet

```bash
scp /tmp/kalshi-migration.sql.gz root@<new-ip>:/tmp/
scp /tmp/kalshi-migration.env    root@<new-ip>:/tmp/
```

### Step 5 — Run the migration script

SSH into the new droplet as root:
```bash
bash /home/trader/kalshi-weather-trader/deploy/migrate.sh
```

This restores the `.env`, imports the database, and starts both services.

### Step 6 — Verify and cut over

1. Visit `http://<new-ip>:5000` — confirm the dashboard loads with your data
2. Check logs: `journalctl -u kalshi-orchestrator -f`
3. Destroy the old droplet from DigitalOcean
4. Update your Mac rsync backup command with the new IP

---

## Day-to-day operations

| Task | Command |
|------|---------|
| View trading engine logs | `journalctl -u kalshi-orchestrator -f` |
| View dashboard logs | `journalctl -u kalshi-dashboard -f` |
| Restart trading engine | `systemctl restart kalshi-orchestrator` |
| Restart dashboard | `systemctl restart kalshi-dashboard` |
| Stop everything | `systemctl stop kalshi-orchestrator kalshi-dashboard` |
| Pull latest code | `cd /home/trader/kalshi-weather-trader && git pull && systemctl restart kalshi-orchestrator kalshi-dashboard` |
| Check service status | `systemctl status kalshi-orchestrator kalshi-dashboard` |
| Run a manual DB backup | `sudo /usr/local/bin/backup-kalshi-db.sh` |
| View backup files | `ls -lh /var/backups/kalshi/` |

## Pulling backups to your Mac

```bash
rsync -avz --progress trader@<droplet-ip>:/var/backups/kalshi/ ~/kalshi-backups/
```

Backups run automatically every 6 hours. Run the rsync manually whenever you want a local copy.

## Updating the app

```bash
ssh root@<your-droplet-ip>
cd /home/trader/kalshi-weather-trader
git pull
sudo -u trader /home/trader/venv/bin/pip install -r requirements.txt
systemctl restart kalshi-orchestrator kalshi-dashboard
```
