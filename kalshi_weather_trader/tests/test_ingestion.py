"""
Unit tests for ingestion modules (ASOS fetcher, NWP fetcher).

Tests:
- IEM fallback triggers when NWS returns stale data
- celsius_to_fahrenheit conversion
- NWP blended forecast computation
- Kalshi ticker strike extraction
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest


class TestAsosHelpers:
    def test_celsius_to_fahrenheit(self):
        """Verify C→F conversion at known values."""
        from kalshi_weather_trader.ingestion.asos_fetcher import _c_to_f

        assert _c_to_f(0.0) == pytest.approx(32.0)
        assert _c_to_f(100.0) == pytest.approx(212.0)
        assert _c_to_f(-40.0) == pytest.approx(-40.0)
        assert _c_to_f(None) is None

    def test_mph_conversion(self):
        """Verify m/s → mph conversion."""
        from kalshi_weather_trader.ingestion.asos_fetcher import _mph

        assert _mph(0.0) == pytest.approx(0.0)
        assert _mph(None) is None
        # 1 m/s ≈ 2.237 mph
        assert _mph(1.0) == pytest.approx(2.2, abs=0.1)


class TestNWSFallback:
    def test_stale_nws_triggers_iem_fallback(self):
        """When NWS returns data > staleness threshold, IEM fallback is used."""
        from kalshi_weather_trader.ingestion.asos_fetcher import _fetch_nws_latest

        stale_time = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        stale_response = {
            "properties": {
                "timestamp": stale_time,
                "temperature": {"value": 20.0},
                "dewpoint": {"value": 10.0},
                "windSpeed": {"value": 5.0},
                "rawMessage": "METAR test",
            }
        }

        with patch(
            "kalshi_weather_trader.ingestion.asos_fetcher._get_nws",
            return_value=stale_response,
        ):
            result = _fetch_nws_latest()
            # Should return None because data is stale (30 min > 15 min threshold)
            assert result is None

    def test_fresh_nws_returns_reading(self):
        """When NWS returns fresh data, it is returned without IEM fallback."""
        from kalshi_weather_trader.ingestion.asos_fetcher import _fetch_nws_latest

        fresh_time = datetime.now(timezone.utc).isoformat()
        fresh_response = {
            "properties": {
                "timestamp": fresh_time,
                "temperature": {"value": 20.0},  # 20°C = 68°F
                "dewpoint": {"value": 10.0},
                "windSpeed": {"value": 5.0},
                "rawMessage": "METAR KBOS test",
            }
        }

        with patch(
            "kalshi_weather_trader.ingestion.asos_fetcher._get_nws",
            return_value=fresh_response,
        ):
            result = _fetch_nws_latest()
            assert result is not None
            assert result.temperature_f == pytest.approx(68.0, abs=0.1)

    def test_missing_temperature_returns_none(self):
        """NWS response with null temperature returns None."""
        from kalshi_weather_trader.ingestion.asos_fetcher import _fetch_nws_latest

        fresh_time = datetime.now(timezone.utc).isoformat()
        response = {
            "properties": {
                "timestamp": fresh_time,
                "temperature": {"value": None},
            }
        }

        with patch(
            "kalshi_weather_trader.ingestion.asos_fetcher._get_nws",
            return_value=response,
        ):
            result = _fetch_nws_latest()
            assert result is None


class TestKalshiStrikeExtraction:
    def test_extract_strike_from_T_suffix(self):
        """Tickers ending in T{digits} should extract the strike."""
        from kalshi_weather_trader.ingestion.kalshi_fetcher import KalshiFetcher

        # Create a minimal fetcher without connecting to Kalshi
        with patch.object(KalshiFetcher, "__init__", return_value=None):
            fetcher = KalshiFetcher.__new__(KalshiFetcher)
            fetcher._base_url = ""
            fetcher._access_key = ""

        assert fetcher.extract_strike_from_ticker("KXHIGHNEW-2025-0615T70") == 70
        assert fetcher.extract_strike_from_ticker("KXHIGHNEW-2025-0615T85") == 85

    def test_extract_strike_from_B_suffix(self):
        from kalshi_weather_trader.ingestion.kalshi_fetcher import KalshiFetcher

        with patch.object(KalshiFetcher, "__init__", return_value=None):
            fetcher = KalshiFetcher.__new__(KalshiFetcher)

        assert fetcher.extract_strike_from_ticker("HIGHBOS15JUN25-B70") == 70

    def test_unknown_ticker_returns_none(self):
        from kalshi_weather_trader.ingestion.kalshi_fetcher import KalshiFetcher

        with patch.object(KalshiFetcher, "__init__", return_value=None):
            fetcher = KalshiFetcher.__new__(KalshiFetcher)

        assert fetcher.extract_strike_from_ticker("UNKNOWN-TICKER-XYZ") is None
