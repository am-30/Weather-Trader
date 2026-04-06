"""
Unit tests for the 2D Kalman filter implementation.

Tests:
- P stays positive-definite after 100 update cycles
- Bias is observable and converges toward the true NWP offset
- Joseph form ensures P symmetry
- predict() inflates P and updates NWP reference; does not shift departure state
- update() pulls estimate toward observation
"""

from __future__ import annotations

import numpy as np
import pytest

from kalshi_weather_trader.quant.kalman_filter import KalmanFilter


class TestKalmanInit:
    def test_initial_state(self):
        kf = KalmanFilter(initial_dt=0.0, nwp_current_hour=72.0)
        assert kf.temperature == pytest.approx(72.0)
        assert kf.bias == pytest.approx(0.0)
        assert kf.P.shape == (2, 2)

    def test_custom_covariance(self):
        cov = [[2.0, 0.5], [0.5, 1.0]]
        kf = KalmanFilter(initial_dt=0.0, nwp_current_hour=70.0, initial_covariance=cov)
        assert kf.P[0, 0] == pytest.approx(2.0)

    def test_invalid_covariance_raises(self):
        with pytest.raises(ValueError, match="2×2"):
            KalmanFilter(initial_dt=0.0, initial_covariance=[[1.0, 0.0, 0.0]])


class TestKalmanPredict:
    def test_predict_updates_nwp_reference(self):
        """predict() with a new NWP value updates the absolute temperature estimate."""
        kf = KalmanFilter(initial_dt=0.0, nwp_current_hour=70.0)
        kf.predict(nwp_at_current_hour=72.0)
        # dT unchanged (0.0), NWP moved to 72 → T_abs = 72.0
        assert kf.temperature == pytest.approx(72.0)

    def test_predict_does_not_shift_departure_state(self):
        """Predict step must NOT move dT — departure is stable across NWP hours."""
        kf = KalmanFilter(initial_dt=1.5, nwp_current_hour=68.0)
        dt_before = float(kf.x[0, 0])
        kf.predict(nwp_at_current_hour=70.0)
        assert float(kf.x[0, 0]) == pytest.approx(dt_before)
        assert kf._nwp_current == pytest.approx(70.0)
        assert kf.temperature == pytest.approx(70.0 + dt_before)

    def test_predict_decays_bias_by_one_step(self):
        """predict(dt=1.0) decays B by kalman_bias_decay (default 0.95) per hour."""
        from kalshi_weather_trader.config.settings import settings
        kf = KalmanFilter(initial_dt=0.0, nwp_current_hour=70.0, initial_bias=1.5)
        kf.predict(dt=1.0)
        expected = 1.5 * settings.kalman_bias_decay
        assert kf.bias == pytest.approx(expected, abs=1e-6)

    def test_predict_increases_uncertainty(self):
        kf = KalmanFilter(initial_dt=0.0, nwp_current_hour=70.0, q_temp=1.0, q_bias=0.5)
        P_before = kf.P.copy()
        kf.predict()
        # Trace of P should increase after predict (more uncertainty)
        assert np.trace(kf.P) > np.trace(P_before)

    def test_predict_without_nwp_leaves_reference_unchanged(self):
        """Calling predict() without nwp_at_current_hour should leave _nwp_current alone."""
        kf = KalmanFilter(initial_dt=1.0, nwp_current_hour=65.0)
        kf.predict()  # no NWP arg
        assert kf._nwp_current == pytest.approx(65.0)
        assert kf.temperature == pytest.approx(66.0)  # 65 + 1


class TestKalmanUpdate:
    def test_update_pulls_toward_observation(self):
        kf = KalmanFilter(initial_dt=2.0, initial_bias=0.0, nwp_current_hour=68.0)
        # T_abs = 70.0; observe 75.0
        kf.update(asos_temp=75.0, nwp_current_hour=68.0)
        # Temperature should move toward 75
        assert kf.temperature > 70.0
        assert kf.temperature < 75.0

    def test_update_reduces_uncertainty(self):
        kf = KalmanFilter(initial_dt=0.0, nwp_current_hour=70.0)
        P_before = kf.P.copy()
        kf.update(asos_temp=70.0, nwp_current_hour=70.0)
        # Trace of P should decrease after update (less uncertainty)
        assert np.trace(kf.P) < np.trace(P_before)

    def test_p_stays_positive_definite_100_cycles(self):
        """P must remain positive-definite after 100 update + predict cycles."""
        kf = KalmanFilter(initial_dt=0.0, initial_bias=0.0, nwp_current_hour=72.0)
        rng = np.random.default_rng(42)
        temps = 72.0 + rng.standard_normal(100) * 2.0

        for i, temp in enumerate(temps):
            kf.update(temp, nwp_current_hour=72.0)
            kf.predict(nwp_at_current_hour=72.0)

            eigenvalues = np.linalg.eigvalsh(kf.P)
            assert np.all(eigenvalues > 0), (
                f"P not positive-definite at step {i}: eigenvalues={eigenvalues}"
            )

    def test_p_symmetry_maintained(self):
        """P must remain symmetric after repeated updates."""
        kf = KalmanFilter(initial_dt=0.0, nwp_current_hour=70.0)
        for temp in [70.0, 71.0, 73.0, 72.5, 74.0, 71.5]:
            kf.update(temp, nwp_current_hour=70.0)
            kf.predict()
        assert kf.P[0, 1] == pytest.approx(kf.P[1, 0], abs=1e-10)


class TestKalmanBiasObservability:
    """Verify that K[1] is nonzero and bias evolves with ASOS observations.

    With H = [[1, 1]], the observation z = asos_temp - nwp couples both dT and B
    to every ASOS tick. These tests confirm that bias is no longer frozen.
    """

    def test_bias_nonzero_after_first_update(self):
        """K[1] ≈ 0.42 on the very first update with default P=I, R=0.4."""
        kf = KalmanFilter(initial_dt=0.0, initial_bias=0.0, nwp_current_hour=60.0)
        kf.update(asos_temp=62.0, nwp_current_hour=60.0)
        # Bias should have moved from 0.0 — K[1] is ~0.42 on the first tick
        assert kf.bias != pytest.approx(0.0, abs=0.01)

    def test_bias_observable(self):
        """After 12 updates with NWP consistently 2°F cold, bias > 0.5°F."""
        kf = KalmanFilter(initial_dt=0.0, initial_bias=0.0, nwp_current_hour=60.0)
        for _ in range(12):
            kf.update(asos_temp=62.0, nwp_current_hour=60.0)
        assert kf.bias > 0.5
        assert kf.temperature > 60.0

    def test_temperature_property_is_absolute(self):
        """temperature = nwp_current + dT + B (absolute °F)."""
        kf = KalmanFilter(initial_dt=2.0, initial_bias=0.5, nwp_current_hour=68.0)
        assert kf.temperature == pytest.approx(70.5)   # 68 + 2 + 0.5 (nwp + dT + B)

    def test_bias_increases_monotonically_with_consistent_positive_residual(self):
        """When NWP is consistently too low, bias should increase toward the true offset."""
        kf = KalmanFilter(initial_dt=0.0, initial_bias=0.0, nwp_current_hour=60.0)
        biases = []
        for _ in range(20):
            kf.update(asos_temp=62.0, nwp_current_hour=60.0)
            biases.append(kf.bias)
        # Bias should be strictly increasing for the first several updates
        for i in range(1, 10):
            assert biases[i] >= biases[i - 1], (
                f"Bias decreased at step {i}: {biases[i-1]:.4f} → {biases[i]:.4f}"
            )

    def test_bias_decays_when_innovations_stop(self):
        """B decays back toward zero once ASOS matches NWP — transient dynamics don't persist.

        Phase 1: consistent 2°F NWP underestimate drives B above 0.5°F (genuine bias).
        Phase 2: ASOS now matches NWP exactly (zero innovation) — hourly predict steps
        should pull B back significantly via the state transition decay.
        """
        kf = KalmanFilter(initial_dt=0.0, initial_bias=0.0, nwp_current_hour=60.0)

        # Phase 1: push B up with consistent positive innovation
        for _ in range(12):
            kf.update(asos_temp=62.0, nwp_current_hour=60.0)
        b_peak = kf.bias
        assert b_peak > 0.5, f"Expected B > 0.5 after phase 1, got {b_peak:.4f}"

        # Phase 2: 24 hourly cycles of predict + zero-innovation update
        for _ in range(24):
            kf.predict(nwp_at_current_hour=60.0, dt=1.0)
            kf.update(asos_temp=60.0, nwp_current_hour=60.0)

        # After 24h of zero NWP error + decay, B should be well below the peak
        assert kf.bias < b_peak * 0.5, (
            f"Expected B < {b_peak * 0.5:.4f} after 24h decay, got {kf.bias:.4f}"
        )
        assert kf.bias > -0.5, f"B should not swing strongly negative, got {kf.bias:.4f}"


class TestKalmanBiasDecayOverride:
    """Verify that bias_decay is stored per-instance and used in predict()."""

    def test_bias_decay_stored_when_passed_explicitly(self):
        kf = KalmanFilter(initial_dt=0.0, nwp_current_hour=70.0, bias_decay=0.85)
        assert kf._bias_decay == pytest.approx(0.85)

    def test_bias_decay_defaults_to_settings_when_none(self):
        from kalshi_weather_trader.config.settings import settings
        kf = KalmanFilter(initial_dt=0.0, nwp_current_hour=70.0)
        assert kf._bias_decay == pytest.approx(settings.kalman_bias_decay)

    def test_predict_uses_instance_decay_not_settings(self):
        """Two filters with different bias_decay diverge after one predict step."""
        kf_fast = KalmanFilter(initial_dt=0.0, initial_bias=2.0, nwp_current_hour=70.0, bias_decay=0.85)
        kf_slow = KalmanFilter(initial_dt=0.0, initial_bias=2.0, nwp_current_hour=70.0, bias_decay=0.99)
        kf_fast.predict(dt=1.0)
        kf_slow.predict(dt=1.0)
        assert kf_fast.bias == pytest.approx(2.0 * 0.85, abs=1e-6)
        assert kf_slow.bias == pytest.approx(2.0 * 0.99, abs=1e-6)
        assert kf_fast.bias < kf_slow.bias

    def test_bias_decay_1_0_no_decay(self):
        """bias_decay=1.0 means random walk — B unchanged after 10 predict steps."""
        kf = KalmanFilter(initial_dt=0.0, initial_bias=3.0, nwp_current_hour=70.0, bias_decay=1.0)
        for _ in range(10):
            kf.predict(dt=1.0)
        assert kf.bias == pytest.approx(3.0, abs=1e-4)

    def test_dt_half_applies_half_power_decay(self):
        """predict(dt=0.5) applies decay^0.5, not full decay."""
        kf = KalmanFilter(initial_dt=0.0, initial_bias=4.0, nwp_current_hour=70.0, bias_decay=0.90)
        kf.predict(dt=0.5)
        assert kf.bias == pytest.approx(4.0 * (0.90 ** 0.5), abs=1e-6)


class TestKalmanBiasConvergence:
    def test_temperature_tracks_observations_with_bias(self):
        """With a nonzero bias, temperature estimate should converge to true temp."""
        kf = KalmanFilter(initial_dt=0.0, initial_bias=3.0, nwp_current_hour=72.0)
        true_temp = 75.0

        for _ in range(100):
            kf.update(true_temp, nwp_current_hour=72.0)

        # Temperature should converge toward 75 (the observed temperature)
        assert abs(kf.temperature - true_temp) < 1.0
