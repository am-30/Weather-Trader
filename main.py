"""
Entry point hint — this file is not used in production.

To run the trading engine:
    cd kalshi_weather_trader && python -m scheduler.orchestrator

To run the dashboard:
    streamlit run kalshi_weather_trader/ui/app.py --server.port 5000

For VM deployment, see deploy/README.md.
"""


def main():
    print(__doc__)


if __name__ == "__main__":
    main()
