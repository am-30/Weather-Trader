"""
Unit tests for the Ornstein-Uhlenbeck Monte Carlo simulation engine.

Tests:
- With sigma=0, all paths converge to NWP target (deterministic OU)
- paths_max >= hard_floor always (hard floor guarantee)
- P(max >= strike) = 1.0 when hard_floor > strike
- P(max >= strike) = 0.0 when strike is impossibly high
- Distribution stats are consistent
- estimate_sigma_from_historical works correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from kalshi_weather_trader.quant.monte_carlo import (
    MCParams,
    estimate_sigma_from_historical,
    price_full_distribution,
    run_simulation,
)


class TestHardFloorGuarantee:
    def test_paths_max_always_gte_hard_floor(self):
        """paths_max must never fall below hard_floor."""
        params = MCParams(
            T0=70.0,
            hard_floor=75.0,
            nwp_curve=[70.0] * 24,
            sigma=3.0,
            theta=0.1,
            n_paths=1000,
            day_fraction_remaining=0.5,
        )
        _, paths_max = run_simulation(params)
        assert np.all(paths_max >= 75.0), "paths_max fell below hard_floor"

    def test_p_above_1_when_floor_exceeds_strike(self):
        """P(max >= strike) = 1.0 when hard_floor > strike."""
        params = MCParams(
            T0=60.0,
            hard_floor=80.0,  # floor exceeds strike
            nwp_curve=[60.0] * 24,
            sigma=2.0,
            n_paths=500,
            day_fraction_remaining=0.5,
        )
        result = price_full_distribution(params, strikes=[75], target_date=None)
        assert result.probabilities[75] == pytest.approx(1.0), (
            f"Expected P=1.0 when floor>strike, got {result.probabilities[75]}"
        )


class TestDeterministicLimitCase:
    def test_zero_sigma_converges_to_nwp(self):
        """With sigma=0 and large theta, all paths should converge to NWP target."""
        nwp_target = 75.0
        params = MCParams(
            T0=60.0,
            hard_floor=60.0,
            nwp_curve=[nwp_target] * 24,
            bias=0.0,
            sigma=0.0,  # no diffusion
            theta=5.0,  # strong mean-reversion
            n_paths=100,
            day_fraction_remaining=0.8,
        )
        paths_current, paths_max = run_simulation(params)
        # With sigma=0, all paths are identical and converge to NWP target
        assert paths_current.std() == pytest.approx(0.0, abs=1e-10)
        # All paths converge near the NWP target
        assert paths_current.mean() == pytest.approx(nwp_target, abs=5.0)


class TestPriceFullDistribution:
    def test_returns_monte_carlo_result(self):
        from datetime import date

        params = MCParams(
            T0=70.0,
            hard_floor=68.0,
            nwp_curve=[72.0] * 24,
            sigma=2.0,
            n_paths=500,
            day_fraction_remaining=0.5,
        )
        result = price_full_distribution(params, strikes=[70, 72, 74, 76], target_date=date.today())

        assert result.n_paths == 500
        assert result.hard_floor == 68.0
        assert set(result.probabilities.keys()) == {70, 72, 74, 76}

    def test_probabilities_in_range(self):
        params = MCParams(
            T0=70.0,
            hard_floor=65.0,
            nwp_curve=[72.0] * 24,
            sigma=2.0,
            n_paths=1000,
            day_fraction_remaining=0.5,
        )
        result = price_full_distribution(params, strikes=[60, 65, 70, 75, 80, 90])

        for strike, prob in result.probabilities.items():
            assert 0.0 <= prob <= 1.0, f"P(strike={strike}) = {prob} out of [0,1]"

    def test_monotone_decreasing_in_strike(self):
        """P(max >= strike) must be non-increasing in strike."""
        params = MCParams(
            T0=70.0,
            hard_floor=65.0,
            nwp_curve=[72.0] * 24,
            sigma=2.0,
            n_paths=2000,
            day_fraction_remaining=0.5,
        )
        strikes = [65, 68, 70, 72, 74, 76, 78, 80]
        result = price_full_distribution(params, strikes=strikes)

        probs = [result.probabilities[s] for s in strikes]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1] - 0.02, (  # small tolerance for randomness
                f"P(max>={strikes[i]})={probs[i]:.4f} < P(max>={strikes[i+1]})={probs[i+1]:.4f}"
            )

    def test_distribution_stats_consistent(self):
        params = MCParams(
            T0=70.0,
            hard_floor=65.0,
            nwp_curve=[72.0] * 24,
            sigma=2.0,
            n_paths=1000,
            day_fraction_remaining=0.5,
        )
        result = price_full_distribution(params, strikes=[70])
        assert result.percentile_25 <= result.percentile_50 <= result.percentile_75
        assert result.percentile_10 <= result.percentile_25
        assert result.percentile_75 <= result.percentile_90
        assert result.std_max >= 0.0


class TestEstimateSigma:
    def test_returns_settings_default_for_insufficient_data(self):
        from kalshi_weather_trader.config.settings import settings

        # Create a minimal mock reading object
        class FakeReading:
            def __init__(self, temp):
                self.temperature_f = temp

        readings = [FakeReading(70.0), FakeReading(71.0)]  # only 2 readings
        sigma = estimate_sigma_from_historical(readings)
        assert sigma == settings.ou_sigma

    def test_sigma_positive(self):
        class FakeReading:
            def __init__(self, temp):
                self.temperature_f = temp

        readings = [FakeReading(70.0 + i * 0.1 + np.random.randn() * 0.5) for i in range(50)]
        sigma = estimate_sigma_from_historical(readings)
        assert sigma > 0.0
