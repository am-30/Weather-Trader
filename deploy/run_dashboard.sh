#!/bin/bash
# Wrapper for the Streamlit dashboard — loads .env then execs streamlit.
# Called by dashboard.service. Do not run directly in production.
set -a
source /home/trader/kalshi-weather-trader/.env
set +a
exec /home/trader/venv/bin/streamlit run \
    kalshi_weather_trader/ui/app.py \
    --server.port 5000 \
    --server.address 0.0.0.0 \
    --server.headless true
