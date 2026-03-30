"""
Tests for Phase 3: Ensemble spread and cloud cover regime awareness.

All tests use deterministic seeding (seed=42) and small n_paths
for fast execution.
"""
from __future__ import annotations

import math
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kalshi_weather_trader.quant.monte_carlo import MCParams, run_simulation, price_full_distribution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NWP_FLAT = [50.0] * 26  # flat NWP curve (constant attractor)


def _make_params(**overrides) -> MCParams:
    """Build MCParams with safe defaults for regime tests."""
    defaults = dict(
        T0=50.0,
        hard_floor=40.0,
        nwp_curve=_NWP_FLAT,
        bias=0.0,
        theta=0.5,
        sigma=0.5,
        hour_offset=10,
        n_paths=5_000,
        day_fraction_remaining=0.5,
        persistence_filter_offset=0.0,
        ou_max_stationary_std=5.0,  # high cap so regime_factor is the only variable
        ensemble_spread=0.0,
        mean_cloudcover_10_16=50.0,
    )
    defaults.update(overrides)
    return MCParams(**defaults)


# ---------------------------------------------------------------------------
# Item 3.1 — Ensemble spread tests
# ---------------------------------------------------------------------------


class TestEnsembleSpread:
    def test_default_ensemble_spread_zero(self):
        """ensemble_spread defaults to 0.0 and mean_cloudcover to 50.0."""
        p = MCParams(T0=50.0, hard_floor=40.0, nwp_curve=_NWP_FLAT, hour_offset=10, day_fraction_remaining=0.5)
        assert p.ensemble_spread == 0.0
        assert p.mean_cloudcover_10_16 == 50.0

    def test_high_spread_widens_distribution(self):
        """ensemble_spread > threshold inflates sigma → wider paths_max distribution."""
        base = _make_params(ensemble_spread=0.0)
        high = _make_params(ensemble_spread=5.0)  # above 3.0 threshold

        _, base_max = run_simulation(base, seed=42)
        _, high_max = run_simulation(high, seed=42)

        assert high_max.std() > base_max.std(), (
            f"High spread should widen distribution: {high_max.std():.3f} vs {base_max.std():.3f}"
        )

    def test_low_spread_unchanged(self):
        """ensemble_spread < threshold (1.5°F) → regime_factor = 1.0, same as no spread."""
        base = _make_params(ensemble_spread=0.0)
        low = _make_params(ensemble_spread=1.5)  # below 3.0 threshold

        _, base_max = run_simulation(base, seed=42)
        _, low_max = run_simulation(low, seed=42)

        # Should be identical (same seed, same effective sigma)
        np.testing.assert_array_almost_equal(base_max, low_max, decimal=6,
            err_msg="Below-threshold spread should not change the distribution")


# ---------------------------------------------------------------------------
# Item 3.2 — Cloud cover tests
# ---------------------------------------------------------------------------


class TestCloudCover:
    def test_overcast_narrows_distribution(self):
        """cloudcover > 80% reduces sigma → narrower paths_max distribution."""
        neutral = _make_params(mean_cloudcover_10_16=50.0)
        overcast = _make_params(mean_cloudcover_10_16=90.0)

        _, neutral_max = run_simulation(neutral, seed=42)
        _, overcast_max = run_simulation(overcast, seed=42)

        assert overcast_max.std() < neutral_max.std(), (
            f"Overcast should narrow distribution: {overcast_max.std():.3f} vs {neutral_max.std():.3f}"
        )

    def test_clear_widens_distribution(self):
        """cloudcover < 20% increases sigma → wider paths_max distribution."""
        neutral = _make_params(mean_cloudcover_10_16=50.0)
        clear = _make_params(mean_cloudcover_10_16=10.0)

        _, neutral_max = run_simulation(neutral, seed=42)
        _, clear_max = run_simulation(clear, seed=42)

        assert clear_max.std() > neutral_max.std(), (
            f"Clear sky should widen distribution: {clear_max.std():.3f} vs {neutral_max.std():.3f}"
        )

    def test_neutral_cloudcover_unchanged(self):
        """cloudcover between 20-80% → regime_factor = 1.0, no change."""
        cc50 = _make_params(mean_cloudcover_10_16=50.0)
        cc60 = _make_params(mean_cloudcover_10_16=60.0)

        _, max50 = run_simulation(cc50, seed=42)
        _, max60 = run_simulation(cc60, seed=42)

        np.testing.assert_array_almost_equal(max50, max60, decimal=6,
            err_msg="Neutral cloudcover range should not change the distribution")


# ---------------------------------------------------------------------------
# Item 3.3 — Climatological baseline tests
# ---------------------------------------------------------------------------


class TestClimatologicalBaseline:
    def test_climatological_prob_within_window(self):
        """P(max >= 50) ≈ 0.16 for a N(45, 5) distribution (1 std above mean)."""
        from kalshi_weather_trader.backtesting.climatology import climatological_prob

        target = date(2026, 3, 15)

        # Mock DB returning synthetic 10-year history: 300 dates, N(45, 5)
        rng = np.random.default_rng(0)
        highs = rng.normal(45.0, 5.0, 300)
        # Create (date, high_f) tuples spanning March ±15 days range
        records = []
        for i, h in enumerate(highs):
            # Spread across a 10-year window of March dates
            obs = date(2016 + i // 30, 3, 1 + (i % 28))
            records.append((obs, float(h)))

        with patch("kalshi_weather_trader.backtesting.climatology.db_manager") as mock_db:
            mock_db.get_historical_daily_highs.return_value = records
            prob = climatological_prob("KBOS", strike=50.0, target_date=target, window_days=15)

        assert prob is not None
        # P(N(45,5) >= 49.5) ≈ 0.185, allow generous tolerance for random sample
        assert 0.10 <= prob <= 0.30, f"Expected P ≈ 0.16-0.18, got {prob}"

    def test_climatological_prob_insufficient_data(self):
        """Returns None when fewer than 10 historical records are in the window."""
        from kalshi_weather_trader.backtesting.climatology import climatological_prob

        target = date(2026, 3, 15)
        # Only 5 records
        records = [(date(2020, 3, 15 + i), 45.0) for i in range(5)]

        with patch("kalshi_weather_trader.backtesting.climatology.db_manager") as mock_db:
            mock_db.get_historical_daily_highs.return_value = records
            prob = climatological_prob("KBOS", strike=50.0, target_date=target, window_days=15)

        assert prob is None
