"""
Tests for Phase A critical fixes.

Covers:
- A.1: drift_adj removed from OU attractor (use_drift_in_attractor=False default)
- A.2: persistence_filter_offset clamp raised to [0.0, 1.5]; zeros included in mean
- A.3: ou_max_stationary_std default lowered 2.0 → 1.5; RMSE safety factor 1.5 → 1.0
- A.4: Kalman covariance cap (_apply_covariance_cap) + innovation gate in update()
- A.6: 100% probability resolved by A.1 (integration test)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kalshi_weather_trader.quant.monte_carlo import (
    MCParams,
    price_full_distribution,
    run_simulation,
)
from kalshi_weather_trader.quant.kalman_filter import KalmanFilter, load_or_initialize_filter


# ---------------------------------------------------------------------------
# A.1 — drift_adj removed from OU attractor by default
# ---------------------------------------------------------------------------


class TestAttractorDriftFlag:
    """Verify use_drift_in_attractor controls whether drift is included in mu_t.

    Setup strategy: set sigma=0 so paths converge deterministically to the attractor.
    Set T0 = nwp_curve[0] + bias so that nwp_anchor_offset = 0 (gap_after_bias = 0).
    This isolates drift as the only variable distinguishing the two cases.
    """

    _NWP = 50.0
    _BIAS = 2.0
    _DRIFT = 1.5
    # T0 = nwp + bias → raw_gap = T0 - nwp = bias; gap_after_bias = bias - bias = 0
    _T0 = _NWP + _BIAS

    def _base_params(self, use_drift: bool) -> MCParams:
        return MCParams(
            T0=self._T0,
            hard_floor=self._T0 - 5.0,
            nwp_curve=[self._NWP] * 24,
            bias=self._BIAS,
            sigma=0.0,         # deterministic: paths go straight to attractor
            theta=10.0,        # very fast reversion
            drift_adj=self._DRIFT,
            hour_offset=0,
            n_paths=500,
            day_fraction_remaining=0.1,
            ou_max_stationary_std=5.0,   # cap inactive with sigma=0
            use_drift_in_attractor=use_drift,
        )

    def test_attractor_no_drift(self) -> None:
        """With use_drift_in_attractor=False (default), drift is NOT in mu_t."""
        params = self._base_params(use_drift=False)
        paths_current, _ = run_simulation(params, seed=0)
        # Expected attractor: nwp + anchor(=0) + bias = 50 + 0 + 2 = 52
        assert paths_current.mean() == pytest.approx(self._NWP + self._BIAS, abs=0.2)
        # Specifically NOT 53.5 (nwp + bias + drift)
        assert paths_current.mean() < self._NWP + self._BIAS + self._DRIFT - 0.5

    def test_attractor_with_drift_flag(self) -> None:
        """With use_drift_in_attractor=True, drift IS in mu_t."""
        params = self._base_params(use_drift=True)
        paths_current, _ = run_simulation(params, seed=0)
        # Expected attractor: nwp + anchor(=0) + bias + drift = 50 + 0 + 2 + 1.5 = 53.5
        assert paths_current.mean() == pytest.approx(
            self._NWP + self._BIAS + self._DRIFT, abs=0.2
        )

    def test_drift_still_stored_in_params(self) -> None:
        """drift_adj is preserved in MCParams regardless of use_drift_in_attractor."""
        params = self._base_params(use_drift=False)
        assert params.drift_adj == self._DRIFT
        assert params.use_drift_in_attractor is False

    def test_default_is_false(self) -> None:
        """MCParams default for use_drift_in_attractor is False."""
        params = MCParams(T0=50.0, hard_floor=40.0, nwp_curve=[50.0] * 24)
        assert params.use_drift_in_attractor is False


# ---------------------------------------------------------------------------
# A.2 — persistence_filter_offset clamp raised; zeros included in mean
# ---------------------------------------------------------------------------


class TestPersistenceOffsetClampRaised:
    """Verify calibrate_persistence_offset() can return values above the old 0.5 cap."""

    def _make_mock_market(self, official_high: float) -> MagicMock:
        m = MagicMock()
        m.final_official_high = official_high
        return m

    def _make_mock_readings(self, asos_max: float) -> list:
        r = MagicMock()
        r.temperature_f = asos_max
        r.max6h_f = None
        return [r]

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_mean_includes_zeros(self, mock_db: MagicMock) -> None:
        """All gaps (including zeros) are included in the mean.

        8 dates: 6 with gap=+1.0°F, 2 with gap=0.0°F.
        Mean of all 8 = (6*1.0 + 2*0.0) / 8 = 0.75.
        Old positive-only mean = 6/6 * 1.0 = 1.0, then capped at 0.5.
        """
        from kalshi_weather_trader.calibration.calibrator import calibrate_persistence_offset

        n_dates = 8
        gaps = [1.0] * 6 + [0.0] * 2  # 6 positive, 2 zero

        def market_side_effect(d: date) -> MagicMock:
            # Return markets only for the 8 most-recent past dates
            today = date(2026, 3, 29)
            for i, g in enumerate(gaps, start=1):
                if d == today - timedelta(days=i):
                    m = MagicMock()
                    m.final_official_high = 50.0 + g
                    return m
            return None

        def asos_side_effect(d: date) -> list:
            today = date(2026, 3, 29)
            for i in range(1, n_dates + 1):
                if d == today - timedelta(days=i):
                    return self._make_mock_readings(50.0)
            return []

        mock_db.get_market.side_effect = market_side_effect
        mock_db.get_asos_readings_for_date.side_effect = asos_side_effect
        mock_db.get_system_state.return_value = None

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=date(2026, 3, 29),
        ):
            result = calibrate_persistence_offset()

        assert result == pytest.approx(0.75, abs=0.01), (
            f"Expected 0.75 (mean of all gaps including zeros), got {result}"
        )
        assert result > 0.5, "Result should exceed old 0.5 clamp"

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_new_clamp_allows_values_above_point5(self, mock_db: MagicMock) -> None:
        """Calibrated offset can reach 1.0 when all gaps are +1.0°F."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_persistence_offset

        def market_side_effect(d: date) -> MagicMock:
            today = date(2026, 3, 29)
            for i in range(1, 8):
                if d == today - timedelta(days=i):
                    m = MagicMock()
                    m.final_official_high = 51.0  # gap = 1.0
                    return m
            return None

        def asos_side_effect(d: date) -> list:
            today = date(2026, 3, 29)
            for i in range(1, 8):
                if d == today - timedelta(days=i):
                    return self._make_mock_readings(50.0)
            return []

        mock_db.get_market.side_effect = market_side_effect
        mock_db.get_asos_readings_for_date.side_effect = asos_side_effect
        mock_db.get_system_state.return_value = None

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=date(2026, 3, 29),
        ):
            result = calibrate_persistence_offset()

        assert result == pytest.approx(1.0, abs=0.01)
        assert result > 0.5, "New clamp [0.0, 1.5] allows values above 0.5"

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_negative_gaps_reduce_offset(self, mock_db: MagicMock) -> None:
        """Negative gaps (ASOS over-read) are included and reduce the mean, floored at 0."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_persistence_offset

        # 5 gaps: 3 × +1.0, 2 × -0.5 → mean = (3 - 1) / 5 = 0.4
        gaps_data = [1.0, 1.0, 1.0, -0.5, -0.5]

        def market_side_effect(d: date) -> MagicMock:
            today = date(2026, 3, 29)
            for i, g in enumerate(gaps_data, start=1):
                if d == today - timedelta(days=i):
                    m = MagicMock()
                    m.final_official_high = 50.0 + g
                    return m
            return None

        def asos_side_effect(d: date) -> list:
            today = date(2026, 3, 29)
            for i in range(1, len(gaps_data) + 1):
                if d == today - timedelta(days=i):
                    return self._make_mock_readings(50.0)
            return []

        mock_db.get_market.side_effect = market_side_effect
        mock_db.get_asos_readings_for_date.side_effect = asos_side_effect
        mock_db.get_system_state.return_value = None

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=date(2026, 3, 29),
        ):
            result = calibrate_persistence_offset()

        assert result == pytest.approx(0.4, abs=0.01)


# ---------------------------------------------------------------------------
# A.4 — Covariance cap and innovation gate
# ---------------------------------------------------------------------------


class TestCovarianceCap:
    """Verify _apply_covariance_cap() brings P within bounds."""

    def test_covariance_cap_after_update(self) -> None:
        """Cap fires after update() when P starts at pathological values."""
        kf = KalmanFilter(
            initial_dt=0.0,
            initial_bias=0.0,
            initial_covariance=[[100.0, -99.0], [-99.0, 100.0]],
            q_temp=0.1,
            q_bias=0.05,
            r_obs=0.4,
        )
        kf._nwp_current = 60.0
        kf.update(asos_temp=60.0, nwp_current_hour=60.0)
        assert kf.P[0, 0] <= 2.0, f"P[0,0]={kf.P[0,0]} should be ≤ 2.0"
        assert kf.P[1, 1] <= 2.0, f"P[1,1]={kf.P[1,1]} should be ≤ 2.0"
        # Off-diagonal should not exceed 0.8 * sqrt(P[0,0] * P[1,1])
        off_diag_limit = 0.8 * (kf.P[0, 0] * kf.P[1, 1]) ** 0.5
        assert abs(kf.P[0, 1]) <= off_diag_limit + 1e-9

    def test_cap_applied_directly(self) -> None:
        """_apply_covariance_cap() can be called directly to fix pathological P."""
        kf = KalmanFilter(
            initial_dt=0.0,
            initial_bias=0.0,
            initial_covariance=[[50.0, -49.0], [-49.0, 50.0]],
        )
        kf._apply_covariance_cap()
        assert kf.P[0, 0] <= 2.0
        assert kf.P[1, 1] <= 2.0

    def test_normal_covariance_not_capped(self) -> None:
        """Covariance within bounds is not modified."""
        kf = KalmanFilter(
            initial_dt=0.0,
            initial_bias=0.0,
            initial_covariance=[[0.3, -0.1], [-0.1, 0.2]],
        )
        p00_before = kf.P[0, 0]
        p11_before = kf.P[1, 1]
        kf._apply_covariance_cap()
        assert kf.P[0, 0] == pytest.approx(p00_before)
        assert kf.P[1, 1] == pytest.approx(p11_before)

    @patch("kalshi_weather_trader.db.db_manager.get_system_state")
    def test_covariance_cap_after_warmstart(self, mock_get_state: MagicMock) -> None:
        """Cap is applied after warm-start construction before gap inflation.

        db_manager is imported locally inside load_or_initialize_filter(), so we
        patch the underlying module function, not the kalman_filter module attribute.
        """
        yesterday_state = MagicMock()
        yesterday_state.kalman_temp_estimate = 60.0
        yesterday_state.kalman_bias_estimate = 2.0
        yesterday_state.kalman_covariance = [[50.0, -49.0], [-49.0, 50.0]]
        yesterday_state.last_updated_utc = None  # skip gap inflation

        # today's state = None → triggers warm-start from yesterday
        mock_get_state.side_effect = [None, yesterday_state]

        kf = load_or_initialize_filter(
            target_date=date(2026, 3, 29),
            current_asos_temp=60.0,
            nwp_at_load_time=58.0,
        )
        # Warm-start inflates by 1.2× → P[0,0] = 60 → cap brings it to 2.0
        assert kf.P[0, 0] <= 2.0, f"P[0,0]={kf.P[0,0]} should be capped to ≤ 2.0"
        assert kf.P[1, 1] <= 2.0

    @patch("kalshi_weather_trader.db.db_manager.get_system_state")
    def test_covariance_cap_on_restore(self, mock_get_state: MagicMock) -> None:
        """Cap fires when restoring today's existing state row with pathological P."""
        today_state = MagicMock()
        today_state.kalman_temp_estimate = 60.0
        today_state.kalman_bias_estimate = 2.0
        today_state.kalman_covariance = [[40.0, -39.0], [-39.0, 40.0]]
        today_state.last_updated_utc = None  # skip gap inflation

        mock_get_state.return_value = today_state

        kf = load_or_initialize_filter(
            target_date=date(2026, 3, 29),
            current_asos_temp=60.0,
            nwp_at_load_time=58.0,
        )
        assert kf.P[0, 0] <= 2.0
        assert kf.P[1, 1] <= 2.0


class TestInnovationGate:
    """Verify the Kalman innovation gate rejects outlier ASOS readings."""

    def _fresh_kf(self) -> KalmanFilter:
        """Small P so S is small and gate sensitivity is clear."""
        kf = KalmanFilter(
            initial_dt=0.0,
            initial_bias=0.0,
            initial_covariance=[[0.1, 0.0], [0.0, 0.1]],
            q_temp=0.1,
            q_bias=0.05,
            r_obs=0.4,
        )
        kf._nwp_current = 60.0
        return kf

    def test_small_innovation_passes_gate(self) -> None:
        """A 2°F innovation is within the 4σ gate and updates the state.

        With P=[[0.1,0],[0,0.1]], H=[[1,1]], R=0.4:
          S = H @ P @ H.T + R = 0.1+0.1+0+0 + 0.4 = 0.6; sqrt(S)=0.775
          gate = 4.0 * 0.775 = 3.1°F
          innovation = (62-60) - (0+0) = 2.0 < 3.1 → passes
        """
        kf = self._fresh_kf()
        x0 = float(kf.x[0, 0])
        kf.update(asos_temp=62.0, nwp_current_hour=60.0)
        assert kf.x[0, 0] != pytest.approx(x0, abs=1e-6), "State should have updated"

    def test_large_innovation_rejected(self) -> None:
        """A 5°F innovation exceeds the 4σ gate and state is unchanged.

        innovation = (65-60) - (0+0) = 5.0; mahal = 5.0/0.775 ≈ 6.45 > 4.0 → rejected
        """
        kf = self._fresh_kf()
        x_before = kf.x.copy()
        p_before = kf.P.copy()
        kf.update(asos_temp=65.0, nwp_current_hour=60.0)
        np.testing.assert_array_equal(kf.x, x_before), "State must not change when gate rejects"
        np.testing.assert_array_equal(kf.P, p_before), "P must not change when gate rejects"

    def test_gate_threshold_boundary(self) -> None:
        """Verify gate fires just above threshold (4σ) and passes just below.

        Using P=[[0.1,0],[0,0.1]], R=0.4 → S=0.6, sqrt(S)=0.7746.
        Gate at 4.0σ fires for |innovation| > 4.0 * 0.7746 = 3.098°F.
        """
        # Just below gate: innovation = 3.0 < 3.098 → should update
        kf_pass = self._fresh_kf()
        x_before = float(kf_pass.x[0, 0])
        kf_pass.update(asos_temp=63.0, nwp_current_hour=60.0)
        assert kf_pass.x[0, 0] != pytest.approx(x_before, abs=1e-6)

        # Just above gate: innovation = 4.0 > 3.098 → should be rejected
        kf_reject = self._fresh_kf()
        x_before2 = kf_reject.x.copy()
        kf_reject.update(asos_temp=64.0, nwp_current_hour=60.0)
        np.testing.assert_array_equal(kf_reject.x, x_before2)


# ---------------------------------------------------------------------------
# A.3 — ou_max_stationary_std reduced to 1.5
# ---------------------------------------------------------------------------


class TestOuMaxStationaryStdReduced:
    """Verify sigma cap fires at lower ou_max_stationary_std=1.5."""

    def test_cap_fires_with_new_default(self) -> None:
        """sigma=1.41, theta=0.29, ou_max_std=1.5 → cap fires; sigma_used=1.14.

        sigma_max = 1.5 * sqrt(2 * 0.29) = 1.5 * 0.7616 = 1.142
        With old cap 2.0: sigma_max = 2.0 * 0.7616 = 1.523 > 1.41 → no cap.
        """
        params_capped = MCParams(
            T0=50.0,
            hard_floor=45.0,
            nwp_curve=[50.0] * 24,
            sigma=1.41,
            theta=0.29,
            ou_max_stationary_std=1.5,
            n_paths=1000,
            day_fraction_remaining=0.5,
        )
        params_uncapped = MCParams(
            T0=50.0,
            hard_floor=45.0,
            nwp_curve=[50.0] * 24,
            sigma=1.41,
            theta=0.29,
            ou_max_stationary_std=2.0,
            n_paths=1000,
            day_fraction_remaining=0.5,
        )
        _, paths_max_capped = run_simulation(params_capped, seed=42)
        _, paths_max_uncapped = run_simulation(params_uncapped, seed=42)

        # Capped paths should have tighter spread
        assert paths_max_capped.std() < paths_max_uncapped.std(), (
            "Capped simulation should have narrower paths_max distribution"
        )

    def test_default_from_settings(self) -> None:
        """MCParams with no explicit ou_max_stationary_std reads from settings (now 1.5)."""
        from kalshi_weather_trader.config.settings import settings
        params = MCParams(T0=50.0, hard_floor=45.0, nwp_curve=[50.0] * 24)
        assert params.ou_max_stationary_std == settings.ou_max_stationary_std
        assert settings.ou_max_stationary_std == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# A.6 — No more 100% probability (integration test)
# ---------------------------------------------------------------------------


class TestNo100PercentProbability:
    """Integration test: Phase A fixes collectively eliminate P=100% snapshots.

    Mimics the March 29 11:52 AM conditions that produced P=100% on the 48.5°F strike.
    With drift removed from the attractor and ou_max_stationary_std=1.5, the
    distribution should be non-degenerate.
    """

    def test_no_100_percent_probability(self) -> None:
        """Phase A fixes produce P < 100% for a strike above the attractor peak.

        Scenario design:
          - hour_offset=0, T0=nwp[0]+bias (anchor_weight≈0 → anchor≈0)
          - Attractor_peak without drift = nwp_peak + 0 + bias = 49.1 + 2.0 = 51.1°F
          - Strike = 53.5°F (2.4°F above attractor peak, ~1.6 sigma)
          - drift_adj=5.0 (large — would push attractor to 56.1 if active)
          With drift removed, P(max>=53.5) should be meaningfully below 99%.
          With drift active, P(max>=53.5) would be much higher (attractor=56.1).
        """
        nwp_curve = [
            40.0, 40.5, 41.0, 41.5, 42.0, 43.0,  # hours 0-5
            44.0, 45.5, 47.0, 48.5, 49.0, 49.1,  # hours 6-11 (morning ramp)
            49.0, 48.5, 48.0, 47.0, 46.0, 45.0,  # hours 12-17 (afternoon)
            44.0, 43.0, 42.0, 41.0, 40.0, 39.0,  # hours 18-23 (evening)
        ]
        # T0 = nwp[0] + bias → raw_gap = bias, gap_after_bias = 0 → anchor ≈ 0
        # (anchor_weight at hour 0 = 1 - 11/11 = 0 exactly with peak at hour 11)
        T0 = nwp_curve[0] + 2.0  # = 40.0 + 2.0 = 42.0
        params = MCParams(
            T0=T0,
            hard_floor=T0 - 2.0,  # 40.0 — below T0
            nwp_curve=nwp_curve,
            bias=2.0,
            sigma=1.2,
            theta=0.29,
            drift_adj=5.0,           # large drift — must NOT be in attractor
            use_drift_in_attractor=False,
            ou_max_stationary_std=1.5,
            hour_offset=0,
            n_paths=10_000,
            day_fraction_remaining=1.0,
        )
        # Strike above attractor peak (51.1°F) — many paths won't reach it
        result = price_full_distribution(params, strikes=[53.5], seed=1)
        p_yes = result.probabilities.get(53.5, 0.0)
        assert p_yes < 0.99, (
            f"P(max>=53.5)={p_yes:.4f} should be < 0.99 after Phase A fixes; "
            "attractor peaks at ~51°F so not all paths reach 53.5°F"
        )
        # Sanity: strike is reachable (within ~2 sigma of attractor peak)
        # mean_max ≈ 51.3, std_max ≈ 1.25 → 53.5 is ~1.8 sigma above mean_max → P ≈ 3-6%
        assert p_yes > 0.02, (
            f"P(max>=53.5)={p_yes:.4f} should be > 0.02; "
            "53.5°F is ~2 sigma above the attractor peak, so some paths should reach it"
        )

    def test_with_drift_active_gives_higher_probability(self) -> None:
        """Confirm drift is the cause: use_drift_in_attractor=True raises P for above-peak strike."""
        nwp_curve = [
            40.0, 40.5, 41.0, 41.5, 42.0, 43.0,
            44.0, 45.5, 47.0, 48.5, 49.0, 49.1,
            49.0, 48.5, 48.0, 47.0, 46.0, 45.0,
            44.0, 43.0, 42.0, 41.0, 40.0, 39.0,
        ]
        T0 = nwp_curve[0] + 2.0  # anchor_weight = 0 at hour 0
        common = dict(
            T0=T0, hard_floor=T0 - 2.0, nwp_curve=nwp_curve,
            bias=2.0, sigma=1.2, theta=0.29, drift_adj=2.0,
            ou_max_stationary_std=1.5,
            hour_offset=0, n_paths=5_000, day_fraction_remaining=1.0,
        )
        params_no_drift = MCParams(**common, use_drift_in_attractor=False)
        params_with_drift = MCParams(**common, use_drift_in_attractor=True)
        # Strike above no-drift attractor peak (51.1°F) but below with-drift peak (53.1°F)
        r_no = price_full_distribution(params_no_drift, strikes=[52.5], seed=3)
        r_with = price_full_distribution(params_with_drift, strikes=[52.5], seed=3)
        assert r_with.probabilities[52.5] > r_no.probabilities[52.5], (
            "Adding drift to attractor should increase P(max>=strike above no-drift peak)"
        )
