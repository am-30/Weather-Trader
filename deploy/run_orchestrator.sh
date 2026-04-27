#!/bin/bash
# Wrapper for the trading engine — loads .env then execs the orchestrator.
# Called by orchestrator.service. Do not run directly in production.
set -a
source /home/trader/kalshi-weather-trader/.env
set +a
exec /home/trader/venv/bin/python -m scheduler.orchestrator
