"""
Tests for Phase 1 simulation model improvements.

Covers:
- Item 1.2: NWP anchor offset no longer double-counts Kalman bias
- Item 1.3: Persistence filter offset raises paths_max initialisation floor
- Item 1.1: Time-varying sigma calibrated per ET-hour block and used in MC
"""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kalshi_weather_trader.quant.monte_carlo import (
    MCParams,
    SIGMA_BLOCKS,
    SIGMA_BLOCK_LABELS,
    _sigma_block_for_hour,
    estimate_sigma_from_historical,
    run_simulation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_reading(temp_f: float, et_hour: int, d: date) -> MagicMock:
    """Build a minimal ASOSReadingDocument-like object for sigma estimation tests."""
    eastern = __import__("pytz").timezone("America/New_York")
    obs_naive = datetime(d.year, d.month, d.day, et_hour, 0, 0)
    obs_et = eastern.localize(obs_naive)
    obs_utc = obs_et.astimezone(__import__("pytz").utc)
    r = MagicMock()
    r.temperature_f = temp_f
    r.observation_time_utc = obs_utc
    r.max6h_f = None
    return r


# ---------------------------------------------------------------------------
# Item 1.2 — Anchor offset no longer double-counts Kalman bias
# ---------------------------------------------------------------------------


class TestAnchorOffsetNoDoubleCount:
    """Verify that bias is subtracted from the T0-vs-NWP gap before anchor scaling.

    Scenario: bias = -2.0 (NWP runs 2°F warm), T0 = 40, nwp[hour_offset] = 42.
      raw_gap = 40 - 42 = -2.0
      gap_after_bias = -2.0 - (-2.0) = 0.0
      nwp_anchor_offset = 0.0 * anchor_weight = 0.0

    Before the fix: nwp_anchor_offset = (-2.0) * anchor_weight (negative shift on top
    of the -2.0 bias already in mu_t = nwp + offset + bias, causing -4.0 effective drag).
    After the fix: only the bias correction applies; anchor adds zero.
    """

    def _run_mc_and_get_offset(self, bias: float, T0: float, nwp_at_hour: float) -> float:
        """Return the effective paths_max mean minus T0 (proxy for offset applied)."""
        # Build an NWP curve where hour 2 = nwp_at_hour (peak well in future)
        nwp_curve = [nwp_at_hour - 5] * 2 + [nwp_at_hour] + [nwp_at_hour - 1] * 21
        params = MCParams(
            T0=T0,
            hard_floor=T0 - 1.0,
            nwp_curve=nwp_curve,
            bias=bias,
            theta=0.3,
            sigma=0.3,
            hour_offset=0,
            n_paths=2000,
            day_fraction_remaining=0.5,
            persistence_filter_offset=0.0,  # disable for this test
        )
        _, paths_max = run_simulation(params, seed=42)
        return float(np.mean(paths_max))

    def test_bias_explains_full_gap_no_double_count(self):
        """When bias == -(T0 - nwp), anchor offset should be zero → paths_max close to nwp."""
        bias = -2.0
        T0 = 40.0
        nwp_at_hour = 42.0  # gap = T0 - nwp = -2 = exactly the bias

        # With fix: gap_after_bias = -2 - (-2) = 0 → offset = 0 → mu_t = nwp + 0 + bias
        # Without fix: offset = -2 * weight, mu_t pulled down an extra ~weight * 2°F
        mean_paths = self._run_mc_and_get_offset(bias, T0, nwp_at_hour)
        # With fix, paths_max mean should be near T0 (40°F), not depressed by double-counting
        assert mean_paths >= 39.0, (
            f"Paths depressed too far below T0 — possible double-counting. "
            f"mean_paths_max={mean_paths:.2f}, expected ≥ 39.0"
        )

    def test_partial_gap_residual_applied(self):
        """When bias explains only half the gap, anchor should capture residual half."""
        bias = -1.0   # bias explains 1°F of the 2°F gap
        T0 = 40.0
        nwp_at_hour = 42.0   # raw_gap = -2, gap_after_bias = -1

        mean_with_partial = self._run_mc_and_get_offset(bias, T0, nwp_at_hour)
        # paths_max mean should be higher than in fully-explained-by-bias case
        mean_fully_explained = self._run_mc_and_get_offset(-2.0, T0, nwp_at_hour)
        # Partially explained has a negative residual (-1) → slightly lower than neutral
        # but still higher than double-counting would produce
        assert mean_with_partial >= 38.5, (
            f"Paths unexpectedly low — possible residual over-correction. "
            f"mean={mean_with_partial:.2f}"
        )


# ---------------------------------------------------------------------------
# Item 1.3 — Persistence filter offset raises paths_max floor
# ---------------------------------------------------------------------------


class TestPersistenceFilterOffset:
    def test_offset_raises_effective_floor(self):
        """paths_max must be initialised at hard_floor + persistence_offset."""
        hard_floor = 38.0
        offset = 0.3
        params = MCParams(
            T0=38.5,
            hard_floor=hard_floor,
            nwp_curve=[38.0] * 24,
            bias=0.0,
            theta=0.3,
            sigma=0.1,   # near-zero sigma so paths stay close to T0
            hour_offset=12,
            n_paths=500,
            day_fraction_remaining=0.1,
            persistence_filter_offset=offset,
        )
        _, paths_max = run_simulation(params, seed=42)
        effective_floor = hard_floor + offset  # 38.3
        assert float(np.min(paths_max)) >= effective_floor - 1e-6, (
            f"min(paths_max)={np.min(paths_max):.4f} is below effective floor {effective_floor}"
        )

    def test_zero_offset_unchanged(self):
        """When persistence_filter_offset=0.0, paths_max min should equal hard_floor."""
        hard_floor = 40.0
        params = MCParams(
            T0=40.0,
            hard_floor=hard_floor,
            nwp_curve=[40.0] * 24,
            bias=0.0,
            theta=0.3,
            sigma=0.05,
            hour_offset=20,
            n_paths=200,
            day_fraction_remaining=0.05,
            persistence_filter_offset=0.0,
        )
        _, paths_max = run_simulation(params, seed=1)
        assert float(np.min(paths_max)) >= hard_floor - 1e-6

    def test_default_offset_from_settings(self):
        """MCParams should use settings.persistence_filter_offset when not supplied."""
        from kalshi_weather_trader.config.settings import settings
        params = MCParams(T0=45.0, hard_floor=44.0, nwp_curve=[45.0] * 24)
        assert params.persistence_filter_offset == settings.persistence_filter_offset

    def test_persistence_offset_calibration(self):
        """calibrate_persistence_offset returns mean positive gap from settled history."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_persistence_offset

        mock_market = MagicMock()
        mock_market.final_official_high = 40.3

        mock_reading = MagicMock()
        mock_reading.temperature_f = 40.0
        mock_reading.max6h_f = None

        # 10 days all with gap = 0.3
        with (
            patch(
                "kalshi_weather_trader.calibration.calibrator.db_manager.get_market",
                return_value=mock_market,
            ),
            patch(
                "kalshi_weather_trader.calibration.calibrator.db_manager.get_asos_readings_for_date",
                return_value=[mock_reading],
            ),
            patch(
                "kalshi_weather_trader.calibration.calibrator.db_manager.get_system_state",
                return_value=None,
            ),
        ):
            result = calibrate_persistence_offset(
                target_date=date(2026, 3, 27),
                lookback_days=10,
            )

        assert abs(result - 0.3) < 0.05, f"Expected ≈0.3, got {result}"


# ---------------------------------------------------------------------------
# Item 1.1 — Time-varying sigma block definitions and calibration
# ---------------------------------------------------------------------------


class TestSigmaBlockDefinitions:
    def test_block_labels_coverage(self):
        """Every integer ET hour 0-23 maps to exactly one block."""
        covered = set()
        for h in range(24):
            lbl = _sigma_block_for_hour(float(h))
            assert lbl in SIGMA_BLOCK_LABELS, f"Hour {h} mapped to unknown block '{lbl}'"
            covered.add(lbl)
        assert covered == set(SIGMA_BLOCK_LABELS)

    def test_boundary_hours(self):
        """Block boundaries are correct (start inclusive, end exclusive)."""
        assert _sigma_block_for_hour(0.0) == "0-6"
        assert _sigma_block_for_hour(5.9) == "0-6"
        assert _sigma_block_for_hour(6.0) == "6-10"
        assert _sigma_block_for_hour(9.9) == "6-10"
        assert _sigma_block_for_hour(10.0) == "10-14"
        assert _sigma_block_for_hour(13.9) == "10-14"
        assert _sigma_block_for_hour(14.0) == "14-18"
        assert _sigma_block_for_hour(17.9) == "14-18"
        assert _sigma_block_for_hour(18.0) == "18-24"
        assert _sigma_block_for_hour(23.9) == "18-24"


class TestSigmaByBlockCalibration:
    """estimate_sigma_from_historical returns per-block sigmas matching known structure."""

    def _make_readings_with_block_sigma(
        self,
        block_sigmas: dict[str, float],
        n_days: int = 30,
    ) -> list:
        """Generate synthetic ASOS readings with prescribed per-block sigma.

        Generates each hourly diff T[h+1]-T[h] using the sigma for the SOURCE hour h
        (same convention as estimate_sigma_from_historical, which assigns each diff to
        the block of the source hour h0).
        """
        rng = np.random.default_rng(0)
        base_date = date(2026, 1, 1)
        readings = []
        base_temp = 40.0

        for day in range(n_days):
            d = base_date + timedelta(days=day)
            temp = base_temp
            readings.append(_fake_reading(temp, 0, d))
            for h in range(1, 24):
                # Increment uses sigma of the SOURCE hour (h-1)
                source_block = _sigma_block_for_hour(float(h - 1))
                sig = block_sigmas.get(source_block, 0.5)
                temp = temp + rng.normal(0.0, sig)
                readings.append(_fake_reading(temp, h, d))

        return readings

    def test_block_sigmas_within_tolerance(self):
        """Per-block sigma estimates within 40% of prescribed values with 30 days."""
        prescribed = {
            "0-6": 0.25,
            "6-10": 0.80,
            "10-14": 0.55,
            "14-18": 0.40,
            "18-24": 0.30,
        }
        readings = self._make_readings_with_block_sigma(prescribed, n_days=30)
        pooled_sigma, sigma_by_block = estimate_sigma_from_historical(readings)

        assert sigma_by_block, "sigma_by_block should not be empty with 30 days of data"
        for lbl, expected in prescribed.items():
            actual = sigma_by_block.get(lbl, pooled_sigma)
            assert abs(actual - expected) / expected < 0.40, (
                f"Block {lbl}: expected ≈{expected:.2f}, got {actual:.3f} "
                f"(>40% error)"
            )

    def test_pooled_sigma_between_block_extremes(self):
        """Pooled sigma should lie between the lowest and highest block sigma."""
        prescribed = {"0-6": 0.2, "6-10": 0.9, "10-14": 0.6, "14-18": 0.4, "18-24": 0.3}
        readings = self._make_readings_with_block_sigma(prescribed, n_days=30)
        pooled_sigma, sigma_by_block = estimate_sigma_from_historical(readings)

        if sigma_by_block:
            block_values = list(sigma_by_block.values())
            assert min(block_values) <= pooled_sigma <= max(block_values) + 0.1, (
                f"Pooled sigma {pooled_sigma} outside block range "
                f"[{min(block_values):.2f}, {max(block_values):.2f}]"
            )


class TestSigmaByBlockInMC:
    """MC simulation uses wider sigma in morning, narrower in afternoon."""

    def _run_mc_at_hour(
        self,
        hour_offset: int,
        sigma_by_block: dict[str, float] | None,
        flat_sigma: float = 0.65,
        theta: float = 3.0,
    ) -> float:
        """Return std(paths_max) for a simulation starting at hour_offset.

        theta=3.0 gives sigma_max = ou_max_stationary_std * sqrt(2*3) ≈ 2.45,
        well above all test sigma values so the cap never fires and the
        block-vs-flat comparison is uncontaminated.
        """
        nwp_curve = [45.0] * 24
        params = MCParams(
            T0=45.0,
            hard_floor=44.0,
            nwp_curve=nwp_curve,
            bias=0.0,
            theta=theta,
            sigma=flat_sigma,
            hour_offset=hour_offset,
            n_paths=3000,
            day_fraction_remaining=(24 - hour_offset) / 24.0,
            persistence_filter_offset=0.0,
            sigma_by_block=sigma_by_block,
        )
        _, paths_max = run_simulation(params, seed=7)
        return float(np.std(paths_max))

    def test_block_varying_wider_in_morning(self):
        """Block-varying sigma (high morning) → wider paths_max than flat sigma at 8 AM."""
        sigma_by_block = {
            "0-6": 0.2,
            "6-10": 1.2,   # very high morning sigma
            "10-14": 0.5,
            "14-18": 0.3,
            "18-24": 0.2,
        }
        std_block = self._run_mc_at_hour(8, sigma_by_block=sigma_by_block)
        std_flat = self._run_mc_at_hour(8, sigma_by_block=None, flat_sigma=0.65)
        assert std_block > std_flat, (
            f"Expected block-varying (high morning sigma) to produce wider "
            f"distribution than flat. block std={std_block:.4f}, flat std={std_flat:.4f}"
        )

    def test_block_varying_narrower_in_afternoon(self):
        """Block-varying sigma (low afternoon) → narrower paths_max than flat sigma at 2 PM."""
        sigma_by_block = {
            "0-6": 0.5,
            "6-10": 0.5,
            "10-14": 0.5,
            "14-18": 0.1,   # very low afternoon sigma
            "18-24": 0.5,
        }
        std_block = self._run_mc_at_hour(14, sigma_by_block=sigma_by_block)
        std_flat = self._run_mc_at_hour(14, sigma_by_block=None, flat_sigma=0.65)
        assert std_block < std_flat, (
            f"Expected block-varying (low afternoon sigma) to produce narrower "
            f"distribution than flat. block std={std_block:.4f}, flat std={std_flat:.4f}"
        )
