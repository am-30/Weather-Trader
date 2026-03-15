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
    _interpolate_cdf,
    compute_normalized_market_probs,
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


class TestInterpolateCDF:
    """Tests for the _interpolate_cdf helper."""

    def _make_probs(self) -> dict[float, float]:
        """Synthetic CDF: P(max >= T) decreasing from 1.0 at 30 to 0.0 at 80."""
        return {30.0: 1.0, 40.0: 0.8, 50.0: 0.5, 60.0: 0.2, 80.0: 0.0}

    def test_exact_key_returns_value(self):
        probs = self._make_probs()
        assert _interpolate_cdf(probs, 50.0) == pytest.approx(0.5)
        assert _interpolate_cdf(probs, 40.0) == pytest.approx(0.8)
        assert _interpolate_cdf(probs, 80.0) == pytest.approx(0.0)

    def test_below_all_keys_returns_1(self):
        probs = self._make_probs()
        assert _interpolate_cdf(probs, 20.0) == pytest.approx(1.0)
        assert _interpolate_cdf(probs, 30.0) == pytest.approx(1.0)  # exact min key

    def test_above_all_keys_returns_0(self):
        probs = self._make_probs()
        assert _interpolate_cdf(probs, 90.0) == pytest.approx(0.0)

    def test_interpolates_between_keys(self):
        probs = self._make_probs()
        # Between 40 (0.8) and 50 (0.5) at midpoint 45 → expect 0.65
        result = _interpolate_cdf(probs, 45.0)
        assert result == pytest.approx(0.65, abs=1e-9)

    def test_empty_probs_returns_half(self):
        assert _interpolate_cdf({}, 50.0) == pytest.approx(0.5)


class TestComputeNormalizedMarketProbs:
    """Tests for the full partition normalization flow."""

    def _make_complete_partition(self) -> tuple[list[dict], dict[float, float]]:
        """Build a 4-bucket complete partition with known probabilities.

        Kalshi B-ticker caps are INCLUSIVE: B38 (floor=38, cap=39) covers {38°F, 39°F}.
        NWS rounds to nearest integer, so the continuous settlement boundaries are
        at half-integers: bottom cap-0.5=37.5, middle [floor-0.5, cap+0.5), top floor-0.5.

        Buckets:  <38  |  {38,39}  |  {40,41}  |  >=42
        CDF keys at half-integer boundaries:
          37.5 → 1.0 (below all keys → CDF=1.0)
          39.5 → interp between 39.0 and 40.0 — use exact value 0.55 in probs
          41.5 → interp between 41.0 and 42.0 — use exact value 0.25 in probs
        Expected raw probs:
          T38:  1 - CDF(37.5)         = 1 - 1.0 = 0.3   (CDF(37.5)=0.7 given below)
          B38:  CDF(37.5) - CDF(39.5) = 0.7 - 0.4 = 0.3
          B40:  CDF(39.5) - CDF(41.5) = 0.4 - 0.1 = 0.3
          T42:  CDF(41.5)             = 0.1
          sum = 1.0
        """
        markets = [
            {"ticker": "T38",  "floor_strike": None, "cap_strike": 38.0},
            {"ticker": "B38",  "floor_strike": 38.0, "cap_strike": 39.0},  # inclusive cap
            {"ticker": "B40",  "floor_strike": 40.0, "cap_strike": 41.0},  # inclusive cap
            {"ticker": "T42",  "floor_strike": 42.0, "cap_strike": None},
        ]
        # Provide CDF at the half-integer boundaries directly so no interpolation needed
        cumulative_probs = {37.5: 0.7, 39.5: 0.4, 41.5: 0.1}
        return markets, cumulative_probs

    def test_complete_partition_sums_to_1(self):
        markets, cumulative_probs = self._make_complete_partition()
        normalized, sum_raw, gaps = compute_normalized_market_probs(markets, cumulative_probs)
        assert sum(normalized.values()) == pytest.approx(1.0, abs=1e-6)
        assert gaps == []

    def test_complete_partition_sum_raw_near_1(self):
        markets, cumulative_probs = self._make_complete_partition()
        _, sum_raw, _ = compute_normalized_market_probs(markets, cumulative_probs)
        assert sum_raw == pytest.approx(1.0, abs=1e-6)

    def test_gap_partition_detects_gap(self):
        """B38 has cap=39 (inclusive → exclusive bound 40), next market has floor=41 → gap [40, 41]."""
        markets = [
            {"ticker": "T38",  "floor_strike": None, "cap_strike": 38.0},
            {"ticker": "B38",  "floor_strike": 38.0, "cap_strike": 39.0},
            # floor=41 instead of 40 creates a gap
            {"ticker": "B41",  "floor_strike": 41.0, "cap_strike": None},
        ]
        cumulative_probs = {38.0: 0.8, 40.0: 0.5, 41.0: 0.3}
        _, _, gaps = compute_normalized_market_probs(markets, cumulative_probs)
        # B38 has cap=39 (inclusive) → exclusive upper bound = 40.
        # Next market has floor=41 → gap is [40, 41].
        assert len(gaps) == 1
        assert gaps[0][0] == pytest.approx(40.0)
        assert gaps[0][1] == pytest.approx(41.0)

    def test_normalization_scales_raw_probs(self):
        """If raw sum is 0.9, normalized values should each be raw/0.9."""
        markets = [
            {"ticker": "T38",  "floor_strike": None, "cap_strike": 38.0},
            {"ticker": "T42",  "floor_strike": 42.0, "cap_strike": None},
        ]
        # Bottom T38: boundary at cap-0.5=37.5 → P = 1 - CDF(37.5) = 1 - 0.5 = 0.5
        # Top T42: boundary at floor-0.5=41.5 → P = CDF(41.5) = 0.4
        # sum = 0.9 → normalized = 0.5/0.9, 0.4/0.9
        cumulative_probs = {37.5: 0.5, 41.5: 0.4}
        normalized, sum_raw, _ = compute_normalized_market_probs(markets, cumulative_probs)
        assert sum_raw == pytest.approx(0.9, abs=1e-6)
        assert normalized["T38"] == pytest.approx(0.5 / 0.9, abs=1e-6)
        assert normalized["T42"] == pytest.approx(0.4 / 0.9, abs=1e-6)
        assert sum(normalized.values()) == pytest.approx(1.0, abs=1e-6)

    def test_returns_correct_ticker_keys(self):
        markets, cumulative_probs = self._make_complete_partition()
        normalized, _, _ = compute_normalized_market_probs(markets, cumulative_probs)
        assert set(normalized.keys()) == {"T38", "B38", "B40", "T42"}
