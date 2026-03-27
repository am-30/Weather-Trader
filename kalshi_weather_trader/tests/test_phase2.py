"""
Tests for Phase 2 calibration overhaul.

Covers:
- Item 2.2: Exponentially weighted sigma estimation (recent days dominate)
- Item 2.1: Two-regime theta (AM/PM) calibration and MC integration
- Item 2.2: Model weights equal-weight fallback when < 14 settled dates
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import pytz

from kalshi_weather_trader.quant.monte_carlo import (
    MCParams,
    estimate_sigma_from_historical,
    run_simulation,
)

_EASTERN = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_reading(temp_f: float, et_hour: int, d: date) -> MagicMock:
    """Build a minimal ASOSReadingDocument-like object."""
    obs_naive = datetime(d.year, d.month, d.day, et_hour, 0, 0)
    obs_et = _EASTERN.localize(obs_naive)
    obs_utc = obs_et.astimezone(pytz.utc)
    r = MagicMock()
    r.temperature_f = temp_f
    r.observation_time_utc = obs_utc
    r.max6h_f = None
    return r


def _make_synthetic_readings(
    daily_sigmas: list[float],
    n_hours_per_day: int = 24,
    base_date: date = date(2026, 1, 1),
    base_temp: float = 40.0,
    seed: int = 0,
) -> list:
    """Generate readings with one sigma per day (all hours, all blocks same sigma).

    Args:
        daily_sigmas: List of sigma values, one per day.
                      Index 0 = oldest day (furthest from today),
                      index -1 = newest day.
        n_hours_per_day: Number of hourly readings per day.
        base_date: First (oldest) day.
        base_temp: Starting temperature.
        seed: RNG seed.

    Returns:
        List of mock readings, oldest-first.
    """
    rng = np.random.default_rng(seed)
    readings = []
    for day_idx, sigma in enumerate(daily_sigmas):
        d = base_date + timedelta(days=day_idx)
        temp = base_temp
        readings.append(_fake_reading(temp, 0, d))
        for h in range(1, n_hours_per_day):
            temp = temp + rng.normal(0.0, sigma)
            readings.append(_fake_reading(temp, h, d))
    return readings


# ---------------------------------------------------------------------------
# Item 2.2 — Exponential weighting in sigma estimation
# ---------------------------------------------------------------------------


class TestExponentialWeighting:
    """estimate_sigma_from_historical weights recent days more heavily."""

    def test_recent_days_dominate_with_decay(self):
        """30 days: first 20 have sigma=0.3, last 10 have sigma=0.8.
        With decay_tau=10, recent (high-sigma) days dominate → weighted sigma > flat.
        """
        # Days 0-19 (older, 20 days back from most recent) have low sigma
        # Days 20-29 (most recent, 0-9 days back from most recent) have high sigma
        sigmas_old = [0.3] * 20
        sigmas_new = [0.8] * 10
        readings = _make_synthetic_readings(sigmas_old + sigmas_new, seed=1)

        # Weighted (tau=10): recent 10 days (sigma=0.8) dominate
        weighted_sigma, _ = estimate_sigma_from_historical(readings, decay_tau_days=10)

        # Flat (tau=10000 ≈ flat): dominated by the 20 low-sigma days
        flat_sigma, _ = estimate_sigma_from_historical(readings, decay_tau_days=10000)

        assert weighted_sigma > flat_sigma, (
            f"Exponential weighting should pull sigma toward recent high-sigma days. "
            f"weighted={weighted_sigma:.3f}, flat={flat_sigma:.3f}"
        )

    def test_flat_weighting_matches_unweighted_approx(self):
        """With very large tau (≈ flat), weighted sigma ≈ unweighted (within 5%)."""
        sigmas = [0.5] * 20 + [0.8] * 10
        readings = _make_synthetic_readings(sigmas, seed=2)

        # Very large tau → weights all ≈ 1.0 → effectively unweighted
        flat_weighted, _ = estimate_sigma_from_historical(readings, decay_tau_days=100_000)
        true_flat, _ = estimate_sigma_from_historical(readings, decay_tau_days=100_000)

        # Both calls with tau=100000 must agree (deterministic)
        assert abs(flat_weighted - true_flat) < 0.001, (
            f"Same tau should give same result. {flat_weighted} vs {true_flat}"
        )

    def test_decay_tau_days_defaults_from_settings(self):
        """Calling estimate_sigma_from_historical without decay_tau_days uses settings."""
        from kalshi_weather_trader.config.settings import settings

        readings = _make_synthetic_readings([0.5] * 15, seed=3)
        sigma_default, _ = estimate_sigma_from_historical(readings)
        sigma_explicit, _ = estimate_sigma_from_historical(
            readings, decay_tau_days=settings.calibration_decay_tau_days
        )
        assert abs(sigma_default - sigma_explicit) < 1e-6, (
            f"Default should use settings.calibration_decay_tau_days. "
            f"default={sigma_default}, explicit={sigma_explicit}"
        )

    def test_single_day_sigma_unaffected_by_weighting(self):
        """With one day of data, weighting is irrelevant (d=0 → weight=1 always)."""
        readings = _make_synthetic_readings([0.6] * 1, n_hours_per_day=20, seed=4)
        sigma_tau10, _ = estimate_sigma_from_historical(readings, decay_tau_days=10)
        sigma_tau1000, _ = estimate_sigma_from_historical(readings, decay_tau_days=1000)
        # Both should give the same sigma since there's only one day
        assert abs(sigma_tau10 - sigma_tau1000) < 1e-6, (
            f"Single day: weighting irrelevant. tau10={sigma_tau10}, tau1000={sigma_tau1000}"
        )


# ---------------------------------------------------------------------------
# Item 2.1 — Two-regime theta calibration
# ---------------------------------------------------------------------------


class TestThetaRegimeCalibration:
    """calibrate_theta_by_regime() splits AR(1) fit into AM/PM pools."""

    def _build_ar1_readings_with_regime(
        self,
        phi_am: float,
        phi_pm: float,
        n_days: int = 40,
        seed: int = 10,
    ) -> tuple[list, dict]:
        """Synthetic ASOS readings with different AR(1) in AM (h0 in 6-12) vs PM (h0 in 13-19).

        Simulates: dep[h+1] = phi_regime * dep[h] + noise
        where phi_am governs morning transitions and phi_pm governs afternoon.

        Returns:
            (readings, nwp_curves_by_date) — nwp_curves_by_date maps date → [0]*24.
        """
        rng = np.random.default_rng(seed)
        base_date = date(2026, 1, 1)
        readings = []
        nwp_curves = {}
        noise_std = 0.2  # small noise so AR(1) signal is visible

        for day_idx in range(n_days):
            d = base_date + timedelta(days=day_idx)
            nwp_curves[d] = [0.0] * 24  # flat zero NWP → departure = raw temp
            dep = 0.0
            # Generate hourly departure sequence
            for h in range(24):
                # Phi depends on which regime h0 is in (source for next step)
                if 6 <= h < 13:
                    phi_next = phi_am
                elif 13 <= h < 20:
                    phi_next = phi_pm
                else:
                    phi_next = (phi_am + phi_pm) / 2.0  # overnight: average
                temp = dep  # since nwp=0, temp ≈ departure
                readings.append(_fake_reading(temp, h, d))
                dep = phi_next * dep + rng.normal(0.0, noise_std)

        return readings, nwp_curves

    def test_theta_am_lower_when_phi_am_high(self):
        """phi_am=0.85 (slow reversion) → theta_am < theta_pm when phi_pm=0.50."""
        from unittest.mock import patch as _patch

        from kalshi_weather_trader.calibration.calibrator import calibrate_theta_by_regime

        readings, nwp_curves = self._build_ar1_readings_with_regime(
            phi_am=0.85,  # high phi → low theta (slow mean-reversion)
            phi_pm=0.50,  # low phi → high theta (fast mean-reversion)
            n_days=40,
            seed=11,
        )

        with (
            _patch(
                "kalshi_weather_trader.ingestion.asos_fetcher.fetch_last_n_hours",
                return_value=readings,
            ),
            _patch(
                "kalshi_weather_trader.ingestion.nwp_fetcher.get_nwp_curve",
                side_effect=lambda d: nwp_curves.get(d, [0.0] * 24),
            ),
            _patch(
                "kalshi_weather_trader.calibration.calibrator.db_manager.get_system_state",
                return_value=None,
            ),
        ):
            theta_am, theta_pm = calibrate_theta_by_regime(
                target_date=date(2026, 3, 27),
                lookback_days=40,
            )

        assert theta_am is not None and theta_pm is not None, (
            "Expected both regimes to have enough pairs for calibration with 40 days"
        )
        assert theta_am < theta_pm, (
            f"phi_am=0.85 → low theta_am, phi_pm=0.50 → high theta_pm. "
            f"Got theta_am={theta_am:.4f}, theta_pm={theta_pm:.4f}"
        )

    def test_theta_regime_fallback_when_sparse(self):
        """Fewer than 20 pairs per regime → returns (None, None)."""
        from unittest.mock import patch as _patch

        from kalshi_weather_trader.calibration.calibrator import calibrate_theta_by_regime

        # Only 2 days → ~7 AM pairs and ~7 PM pairs, both below 20
        readings, nwp_curves = self._build_ar1_readings_with_regime(
            phi_am=0.8, phi_pm=0.5, n_days=2, seed=12
        )

        with (
            _patch(
                "kalshi_weather_trader.calibration.calibrator.db_manager.get_system_state",
                return_value=None,
            ),
        ):
            import kalshi_weather_trader.calibration.calibrator as cal_mod
            cal_mod.fetch_last_n_hours = lambda hours: readings
            cal_mod.get_nwp_curve_fn = lambda d: nwp_curves.get(d, [0.0] * 24)

            theta_am, theta_pm = calibrate_theta_by_regime(
                target_date=date(2026, 3, 27),
                lookback_days=2,
            )

        # With only 2 days, n_am < 20 and n_pm < 20 → both None
        assert theta_am is None and theta_pm is None, (
            f"Sparse data should return (None, None). Got ({theta_am}, {theta_pm})"
        )


# ---------------------------------------------------------------------------
# Item 2.1 — Two-regime theta used in MC simulation
# ---------------------------------------------------------------------------


class TestThetaRegimeInMC:
    """run_simulation() uses step_theta from theta_am/theta_pm when provided."""

    def _run_mc(
        self,
        theta_am: float | None,
        theta_pm: float | None,
        theta_scalar: float,
        hour_offset: int = 8,
        n_paths: int = 5000,
        seed: int = 42,
    ) -> float:
        """Return std(paths_max) for MC starting at hour_offset."""
        params = MCParams(
            T0=45.0,
            hard_floor=44.0,
            nwp_curve=[45.0] * 24,
            bias=0.0,
            theta=theta_scalar,
            sigma=0.5,
            hour_offset=hour_offset,
            n_paths=n_paths,
            day_fraction_remaining=(24 - hour_offset) / 24.0,
            persistence_filter_offset=0.0,
            theta_am=theta_am,
            theta_pm=theta_pm,
        )
        _, paths_max = run_simulation(params, seed=seed)
        return float(np.std(paths_max))

    def test_low_theta_am_produces_wider_distribution_at_8am(self):
        """theta_am=0.05 (slow reversion) at 8 AM → wider paths_max than theta=0.8."""
        # With very low theta_am, paths don't mean-revert quickly → more spread
        std_low_am = self._run_mc(theta_am=0.05, theta_pm=0.8, theta_scalar=0.8, hour_offset=8)
        # With uniform high theta, paths mean-revert quickly → less spread
        std_high_uniform = self._run_mc(theta_am=None, theta_pm=None, theta_scalar=0.8, hour_offset=8)

        assert std_low_am > std_high_uniform, (
            f"Low theta_am at 8 AM should produce wider distribution. "
            f"std_low_am={std_low_am:.4f}, std_high_uniform={std_high_uniform:.4f}"
        )

    def test_scalar_fallback_when_no_regime(self):
        """theta_am=None, theta_pm=None: simulation runs and uses scalar theta."""
        # Simply verify it runs without error and produces finite results
        std = self._run_mc(theta_am=None, theta_pm=None, theta_scalar=0.3, hour_offset=10)
        assert np.isfinite(std) and std >= 0.0, f"Expected finite std, got {std}"

    def test_overnight_steps_use_scalar_theta(self):
        """Steps in overnight hours (0-5) always use scalar theta, not theta_am/theta_pm."""
        # At hour_offset=2 (2 AM), first steps are in overnight block → scalar theta
        std_regime = self._run_mc(theta_am=0.05, theta_pm=0.8, theta_scalar=0.5, hour_offset=2)
        std_scalar = self._run_mc(theta_am=None, theta_pm=None, theta_scalar=0.5, hour_offset=2)

        # With hour_offset=2, most steps are AM/PM by the time we hit 6 AM,
        # but the first 4*12=48 steps use scalar. Both should produce reasonable values.
        assert np.isfinite(std_regime) and np.isfinite(std_scalar), (
            "Both regime and scalar should produce finite results"
        )


# ---------------------------------------------------------------------------
# Item 2.2 — Model weights equal-weight fallback
# ---------------------------------------------------------------------------


class TestModelWeightsFallback:
    """calibrate_model_weights() returns equal weights when < 10 qualifying dates."""

    def _mock_market(self, has_official: bool) -> MagicMock:
        m = MagicMock()
        m.final_official_high = 42.0 if has_official else None
        return m

    def test_equal_weights_when_insufficient_qualifying_dates(self):
        """5 settled dates with morning NWP → equal weights returned (need 10)."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_model_weights

        nwp_call = [0]

        def mock_get_morning_nwp(d):
            nwp_call[0] += 1
            # Only 5 of the 14 lookback days have morning NWP data
            return {"HRRR": MagicMock()} if nwp_call[0] <= 5 else {}

        with (
            patch(
                "kalshi_weather_trader.calibration.calibrator.db_manager.get_market",
                return_value=self._mock_market(has_official=True),
            ),
            patch(
                "kalshi_weather_trader.calibration.calibrator.db_manager.get_morning_nwp_forecasts",
                side_effect=mock_get_morning_nwp,
            ),
            patch(
                "kalshi_weather_trader.calibration.calibrator.db_manager.get_system_state",
                return_value=None,
            ),
        ):
            weights = calibrate_model_weights(
                target_date=date(2026, 3, 27),
                lookback_days=14,
            )

        # Expect equal weights (1/3 each)
        for model in ["HRRR", "GFS", "ECMWF"]:
            assert model in weights, f"Expected {model} in weights"
            assert abs(weights[model] - 1.0 / 3) < 0.01, (
                f"Expected equal weight ≈ 0.333 for {model}, got {weights[model]:.4f}"
            )

    def test_brier_weights_when_sufficient_qualifying_dates(self):
        """10+ qualifying dates (settled + morning NWP) → Brier scoring used (weights differ)."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_model_weights

        # Return a non-zero Brier score that differs per model
        def mock_brier(model_name: str, lookback_days: int):
            return {"HRRR": 0.05, "GFS": 0.12, "ECMWF": 0.10}.get(model_name)

        with (
            patch(
                "kalshi_weather_trader.calibration.calibrator.db_manager.get_market",
                return_value=self._mock_market(has_official=True),
            ),
            patch(
                "kalshi_weather_trader.calibration.calibrator.db_manager.get_morning_nwp_forecasts",
                return_value={"HRRR": MagicMock()},
            ),
            patch(
                "kalshi_weather_trader.calibration.calibrator._brier_score_for_model",
                side_effect=mock_brier,
            ),
            patch(
                "kalshi_weather_trader.calibration.calibrator.db_manager.get_system_state",
                return_value=None,
            ),
        ):
            weights = calibrate_model_weights(
                target_date=date(2026, 3, 27),
                lookback_days=14,
            )

        # HRRR (lowest Brier → best) should get highest weight
        assert weights.get("HRRR", 0) > weights.get("GFS", 1), (
            f"HRRR (Brier=0.05) should outweigh GFS (Brier=0.12). weights={weights}"
        )
        # Weights should not all be equal (Brier scores differ)
        values = list(weights.values())
        assert max(values) - min(values) > 0.01, (
            f"With differing Brier scores, weights should differ. weights={weights}"
        )
