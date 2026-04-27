# Deploying to DigitalOcean

## What you need
- A GitHub account with this repo pushed to it
- A DigitalOcean account
- Your Kalshi API key and RSA private key

## Step 1 — Export your Replit data (do this first, before cancelling Replit)

In the Replit shell:
```
bash scripts/export_db.sh
```
Download the generated `replit_db_export.sql` file from the Replit file browser.

## Step 2 — Create a DigitalOcean Droplet

1. Log in to DigitalOcean → Create → Droplets
2. Choose: **Ubuntu 22.04**, **Basic**, **Regular**, **2 GB RAM / 1 vCPU** ($12/mo)
3. Choose a datacenter region close to you
4. Add your SSH key (or choose password auth)
5. Click Create

## Step 3 — Run the setup script on the VM

SSH into your new droplet as root, then run:
```
bash <(curl -fsSL https://raw.githubusercontent.com/YOUR/REPO/main/deploy/setup_vm.sh)
```
When prompted, paste your GitHub repo URL. The script installs everything and prints next steps.

## Step 4 — Fill in your secrets

On the VM, edit the .env file:
```
nano /home/trader/kalshi-weather-trader/.env
```
Fill in:
- `KALSHI_ACCESS_KEY` — your key ID from kalshi.com
- `KALSHI_PRIVATE_KEY` — your RSA private key (paste the full PEM block, replacing newlines with `\n`)
- `KALSHI_ENV` — `demo` or `prod`

DATABASE_URL is already filled in by the setup script.

## Step 5 — Import your Replit data

Upload the SQL dump and import it:
```
scp replit_db_export.sql root@<your-droplet-ip>:/tmp/
ssh root@<your-droplet-ip>
sudo -u trader psql kalshi_trader < /tmp/replit_db_export.sql
```

## Step 6 — Start the services

```
bash /home/trader/kalshi-weather-trader/deploy/install_services.sh
```

## Step 7 — Open the dashboard

Visit `http://<your-droplet-ip>:5000` in your browser.

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

## Updating the app

```
ssh root@<your-droplet-ip>
cd /home/trader/kalshi-weather-trader
git pull
sudo -u trader /home/trader/venv/bin/pip install -e .
systemctl restart kalshi-orchestrator kalshi-dashboard
```
