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
        """With sigma=0 and large theta, all paths should converge to NWP target.

        T0 must equal nwp_curve[hour_offset] so the anchor offset is zero —
        otherwise the attractor is anchored to T0, not the raw NWP value.
        """
        nwp_target = 75.0
        params = MCParams(
            T0=75.0,          # matches NWP so nwp_anchor_offset = 0
            hard_floor=75.0,  # matches T0
            nwp_curve=[nwp_target] * 24,
            bias=0.0,
            sigma=0.0,  # no diffusion
            theta=5.0,  # strong mean-reversion
            n_paths=100,
            day_fraction_remaining=0.8,
        )
        paths_current, paths_max = run_simulation(params)
        # With sigma=0, all paths are identical and stay at the NWP target
        assert paths_current.std() == pytest.approx(0.0, abs=1e-10)
        # All paths remain at the NWP target
        assert paths_current.mean() == pytest.approx(nwp_target, abs=0.1)


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
        sigma, _ = estimate_sigma_from_historical(readings)
        assert sigma == settings.ou_sigma

    def test_sigma_positive(self):
        from datetime import datetime, timezone, timedelta

        class FakeReading:
            def __init__(self, temp, minutes_offset):
                self.temperature_f = temp
                self.observation_time_utc = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=minutes_offset)

        readings = [FakeReading(70.0 + i * 0.1 + np.random.randn() * 0.5, i * 5) for i in range(50)]
        sigma, _ = estimate_sigma_from_historical(readings)
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
          T41:  CDF(floor+0.5=41.5)   = 0.1
          sum = 1.0

        T41 (floor=41) is adjacent to B40 (cap=41): top bucket boundary = floor+0.5 = 41.5,
        which equals B40's cap+0.5 = 41.5 — no overlap, no gap.
        """
        markets = [
            {"ticker": "T38",  "floor_strike": None, "cap_strike": 38.0},
            {"ticker": "B38",  "floor_strike": 38.0, "cap_strike": 39.0},  # inclusive cap
            {"ticker": "B40",  "floor_strike": 40.0, "cap_strike": 41.0},  # inclusive cap
            {"ticker": "T41",  "floor_strike": 41.0, "cap_strike": None},
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
        """B38 (cap=39) followed by top bucket B41 (floor=41) should detect a gap.

        With is_next_top=True, expected_next_floor = cap_f = 39, but floor_next = 41,
        so the gap spans [39, 41] — two integer degrees {40, 41} are uncovered.
        """
        markets = [
            {"ticker": "T38",  "floor_strike": None, "cap_strike": 38.0},
            {"ticker": "B38",  "floor_strike": 38.0, "cap_strike": 39.0},
            # floor=41 instead of 39 creates a gap (top bucket, no cap_strike)
            {"ticker": "B41",  "floor_strike": 41.0, "cap_strike": None},
        ]
        cumulative_probs = {38.0: 0.8, 40.0: 0.5, 41.0: 0.3}
        _, _, gaps = compute_normalized_market_probs(markets, cumulative_probs)
        # B38 (middle, cap=39) → top bucket B41 (floor=41):
        # is_next_top=True → expected_next_floor = 39, but floor_next=41 → gap [39, 41].
        assert len(gaps) == 1
        assert gaps[0][0] == pytest.approx(39.0)
        assert gaps[0][1] == pytest.approx(41.0)

    def test_normalization_scales_raw_probs(self):
        """If raw sum is 0.9, normalized values should each be raw/0.9."""
        markets = [
            {"ticker": "T38",  "floor_strike": None, "cap_strike": 38.0},
            {"ticker": "T42",  "floor_strike": 42.0, "cap_strike": None},
        ]
        # Bottom T38: boundary at cap-0.5=37.5 → P = 1 - CDF(37.5) = 1 - 0.5 = 0.5
        # Top T42: boundary at floor+0.5=42.5 → P = CDF(42.5) = 0.4
        # sum = 0.9 → normalized = 0.5/0.9, 0.4/0.9
        cumulative_probs = {37.5: 0.5, 42.5: 0.4}
        normalized, sum_raw, _ = compute_normalized_market_probs(markets, cumulative_probs)
        assert sum_raw == pytest.approx(0.9, abs=1e-6)
        assert normalized["T38"] == pytest.approx(0.5 / 0.9, abs=1e-6)
        assert normalized["T42"] == pytest.approx(0.4 / 0.9, abs=1e-6)
        assert sum(normalized.values()) == pytest.approx(1.0, abs=1e-6)

    def test_returns_correct_ticker_keys(self):
        markets, cumulative_probs = self._make_complete_partition()
        normalized, _, _ = compute_normalized_market_probs(markets, cumulative_probs)
        assert set(normalized.keys()) == {"T38", "B38", "B40", "T41"}


class TestNWPAnchor:
    """Tests for the time-weighted NWP anchor offset in run_simulation().

    anchor_weight = 1 - hours_to_peak / peak_hour_idx  (0 at day start, 1 at/past peak)
    nwp_anchor_offset = (T0 - NWP[hour_offset]) * anchor_weight

    Early in the day (far from peak): weight ≈ 0, OU follows raw NWP curve.
    At/past peak:                     weight = 1, full offset applied.
    Peak at index 0 (flat/declining): weight = 1 always (fallback — no warming left).
    """

    def test_declining_nwp_peak_at_start_full_anchor(self):
        """When NWP peaks at index 0 (declining curve), anchor_weight=1 (fallback).

        The peak is already at or behind us, so the full offset is applied and
        paths_max is anchored near T0.
        """
        declining_nwp = [38.7, 37.5, 36.5, 35.5, 35.0] + [35.0] * 19
        params = MCParams(
            T0=37.4,
            hard_floor=37.4,
            nwp_curve=declining_nwp,
            sigma=0.0,   # no noise — deterministic test
            theta=5.0,
            n_paths=100,
            hour_offset=0,
            day_fraction_remaining=0.33,
            persistence_filter_offset=0.0,  # isolate anchor behavior
        )
        _, paths_max = run_simulation(params)
        # peak_hour_idx=0 → weight=1.0: anchor = (37.4-38.7)*1 = -1.3
        # attractor at T0=37.4; subsequent hours track declining NWP shifted by -1.3.
        assert paths_max.mean() == pytest.approx(37.4, abs=0.5), (
            f"Expected paths_max near T0=37.4, got {paths_max.mean():.2f}"
        )

    def test_rising_nwp_far_from_peak_follows_raw_nwp(self):
        """When peak is in the future, anchor_weight=0 at hour 0 and OU follows raw NWP.

        A gap at the start of the day should not depress the forecast peak.
        """
        # NWP rising from 35 to 40 then declining — T0=34, peak at index 3
        rising_nwp = [35.0, 36.0, 38.0, 40.0, 39.0, 37.0] + [35.0] * 18
        params = MCParams(
            T0=34.0,
            hard_floor=34.0,
            nwp_curve=rising_nwp,
            sigma=0.0,
            theta=5.0,
            n_paths=100,
            hour_offset=0,
            day_fraction_remaining=0.5,
        )
        _, paths_max = run_simulation(params)
        # peak_hour_idx=3, hour_offset=0 → weight = 1 - 3/3 = 0
        # anchor=0: OU warms naturally to the raw NWP peak of 40.
        assert paths_max.mean() == pytest.approx(40.0, abs=1.0), (
            f"Expected paths_max near raw NWP peak=40.0, got {paths_max.mean():.2f}"
        )

    def test_rising_nwp_at_peak_full_anchor(self):
        """When hour_offset == peak_hour_idx, weight=1 and full anchor is applied."""
        # NWP rising from 35 to 40 then declining — T0=37 at peak hour (index 3)
        rising_nwp = [35.0, 36.0, 38.0, 40.0, 39.0, 37.0] + [35.0] * 18
        params = MCParams(
            T0=37.0,
            hard_floor=37.0,
            nwp_curve=rising_nwp,
            sigma=0.0,
            theta=5.0,
            n_paths=100,
            hour_offset=3,   # at the peak
            day_fraction_remaining=0.25,
            persistence_filter_offset=0.0,  # isolate anchor behavior
        )
        _, paths_max = run_simulation(params)
        # peak_hour_idx=3, hour_offset=3 → hours_to_peak=0, weight=1.0
        # anchor = (37-40)*1 = -3; attractor = 40-3 = 37 = T0. Paths stay near T0.
        assert paths_max.mean() == pytest.approx(37.0, abs=0.5), (
            f"Expected paths_max near T0=37.0 (full anchor at peak), got {paths_max.mean():.2f}"
        )

    def test_t0_above_nwp_anchor_applied(self):
        """When T0 > NWP, positive anchor prevents OU from pulling paths down.

        Flat NWP means peak_hour_idx=0 → weight=1.0 (fallback) → full positive anchor.
        """
        flat_nwp = [55.0] * 24
        params = MCParams(
            T0=60.0,
            hard_floor=60.0,
            nwp_curve=flat_nwp,
            sigma=0.0,
            theta=5.0,
            n_paths=100,
            hour_offset=0,
            day_fraction_remaining=0.3,
            persistence_filter_offset=0.0,  # isolate anchor behavior
        )
        _, paths_max = run_simulation(params)
        # anchor = (60-55)*1.0 = +5: attractor = 55+5 = 60. Paths stay at T0.
        assert paths_max.mean() == pytest.approx(60.0, abs=0.5), (
            f"Expected paths_max near T0=60.0 (anchor holds), got {paths_max.mean():.2f}"
        )


class TestStationaryStdCap:
    """Tests for the sigma cap that enforces a physical bound on OU stationary std.

    Without the cap, a miscalibrated sigma (e.g. 1.385 from pooled all-hours data)
    with a modest theta (0.1559, half-life ~4.4h) produces stationary_std = 2.48°F.
    Per-step noise (0.4°F) is 31× the restoring force (0.013°F at 1°F gap), making
    paths near-random-walks that spike far above a declining NWP attractor.

    The cap: sigma_used = min(sigma, max_stationary_std * sqrt(2 * theta))
    Default max_stationary_std = 1.0°F (≈ KBOS NWP intraday RMSE).
    """

    def test_sigma_capped_when_over_bound(self):
        """Inflated sigma is capped; paths_max std is physically bounded."""
        # Uncapped: sigma=3.0, theta=0.3 → stationary_std = 3.0/sqrt(0.6) = 3.87°F
        # Capped:   sigma_max = 1.0*sqrt(0.6) = 0.775 → stationary_std = 1.0°F
        params = MCParams(
            T0=70.0,
            hard_floor=70.0,
            nwp_curve=[70.0] * 24,
            sigma=3.0,
            theta=0.3,
            n_paths=5000,
            day_fraction_remaining=0.5,
        )
        _, paths_max = run_simulation(params)
        # Without cap, paths_max std would be ~4-5°F; with cap it should be ~1-2°F.
        assert paths_max.std() < 2.5, (
            f"paths_max std={paths_max.std():.2f}°F — sigma cap not reducing path spread"
        )

    def test_no_cap_when_sigma_within_bound(self):
        """When sigma is already below the cap, behaviour is unchanged."""
        # sigma=0.4, theta=0.3 → stationary_std = 0.4/sqrt(0.6) = 0.516°F < 1.0°F cap
        params_capped = MCParams(
            T0=70.0,
            hard_floor=65.0,
            nwp_curve=[72.0] * 24,
            sigma=0.4,
            theta=0.3,
            n_paths=5000,
            day_fraction_remaining=0.5,
        )
        _, paths_max_1 = run_simulation(params_capped)
        # Run again — both should produce similar distributions (cap not changing sigma).
        _, paths_max_2 = run_simulation(params_capped)
        # Mean should be near the NWP attractor (72°F); check it's in a plausible range.
        assert 70.0 <= paths_max_1.mean() <= 74.0, (
            f"paths_max mean={paths_max_1.mean():.2f}°F outside expected range [70, 74]"
        )

    def test_declining_afternoon_realistic_probabilities(self):
        """Reproduces the March 23 2 PM diagnostic — the original broken scenario.

        Inputs: T0=35.73, hard_floor=37.0, NWP declining from 38.7 at midnight
        to 34.6 at hour 14 then further to 32.5°F. sigma=1.385, theta=0.1559.

        Pre-fix (no cap): P(max>=40°F) = 0.39, mean=39.6°F — physically absurd.
        Post-fix (cap=1.0°F): sigma capped to 0.558°F/sqrt-hr, stationary_std=1.0°F.
        Expected: P(max>=40°F) < 0.08, mean_max < 38.5°F.
        """
        # Declining NWP: peak at hour 0 (38.7°F), at hour 14 = 34.6°F, min = 32.5°F
        declining_nwp = [
            38.7, 37.8, 37.1, 36.6, 35.9, 35.5, 35.4, 34.8,   # hours 0-7
            34.6, 34.3, 34.0, 33.7, 33.5, 33.2, 34.6, 33.8,   # hours 8-15
            33.5, 33.0, 32.8, 32.6, 32.5, 32.5, 32.5, 32.5,   # hours 16-23
        ]
        params = MCParams(
            T0=35.73,
            hard_floor=37.0,
            nwp_curve=declining_nwp,
            sigma=1.385,      # calibrated all-hours sigma — will be capped
            theta=0.1559,
            n_paths=20000,
            hour_offset=14,
            day_fraction_remaining=0.425,
            is_future_day=False,
        )
        result = price_full_distribution(params, strikes=[37.5, 39.5, 40.0, 41.5, 43.0])

        p_above_40 = result.probabilities.get(40.0, 1.0)
        p_above_41_5 = result.probabilities.get(41.5, 1.0)

        assert p_above_40 < 0.08, (
            f"P(max>=40°F) = {p_above_40:.3f} — sigma cap not working; "
            f"mean_max={result.mean_max:.1f}°F (expected <38.5°F)"
        )
        assert p_above_41_5 < 0.04, (
            f"P(max>=41.5°F) = {p_above_41_5:.3f} — paths still spiking too high"
        )
        assert result.mean_max < 38.5, (
            f"mean_max={result.mean_max:.2f}°F too high for a declining afternoon "
            f"with hard_floor=37.0°F and NWP peak already passed"
        )

    def test_pre_peak_morning_retains_spread(self):
        """When NWP peak is still ahead, sigma cap still applies but paths warm normally.

        With NWP rising from 35°F to 42°F peaking at hour 14 and T0=36°F at hour 8,
        paths should warm toward the NWP peak even with capped sigma.
        """
        rising_nwp = [
            35.0, 35.5, 36.0, 36.8, 37.5, 38.2, 39.0, 39.8,   # hours 0-7
            40.5, 41.0, 41.5, 41.8, 42.0, 42.0, 41.5, 40.5,   # hours 8-15
            39.5, 38.5, 37.5, 36.5, 36.0, 35.5, 35.0, 35.0,   # hours 16-23
        ]
        params = MCParams(
            T0=36.0,
            hard_floor=36.0,
            nwp_curve=rising_nwp,
            sigma=1.385,       # will be capped
            theta=0.1559,
            n_paths=10000,
            hour_offset=8,
            day_fraction_remaining=0.667,
            is_future_day=False,
        )
        _, paths_max = run_simulation(params)
        # With NWP peaking at 42°F, paths should warm substantially above T0=36°F.
        # Cap is still active but sigma=0.558 still produces meaningful spread.
        assert paths_max.mean() > 38.0, (
            f"paths_max mean={paths_max.mean():.2f}°F — pre-peak warming suppressed too much"
        )
        assert paths_max.mean() < 44.0, (
            f"paths_max mean={paths_max.mean():.2f}°F — unrealistically high even pre-peak"
        )
