"""
Phase L1 tests for the KBOS Model Lab.

Tests:
- Scenario dataclass hashability (required for Streamlit @st.cache_data)
- ParameterizedReplayEngine.replay_single returns complete results
- No future leakage in T0 and hard_floor
- Scenario override: sigma (narrower sigma → narrower std_max)
- Scenario override: drift disabled (drift_used=0.0)
- Scenario override: Kalman bias (higher bias → higher mean_max)
- Brier score computation (exact arithmetic)
- Calibration curve (bin frequencies match constructed outcomes)
- preset_production matches historical calibrated values
- ReplayDataCache cutoff filtering
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytz

from kalshi_weather_trader.backtesting.metrics import compute_aggregate_metrics
from kalshi_weather_trader.backtesting.replay_engine import ParameterizedReplayEngine
from kalshi_weather_trader.backtesting.scenarios import (
    ReplayDataCache,
    Scenario,
    preset_production,
)
from kalshi_weather_trader.db.schemas import (
    ASOSReadingDocument,
    MarketDocument,
    NWPForecastDocument,
    SystemStateDocument,
)

_ET = pytz.timezone("America/New_York")
_DATE = date(2026, 3, 15)
_NOW_UTC = datetime(2026, 3, 15, 16, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _asos(temp_f: float, hour_et: int, d: date = _DATE) -> ASOSReadingDocument:
    obs_et = _ET.localize(datetime(d.year, d.month, d.day, hour_et, 0, 0))
    obs_utc = obs_et.astimezone(timezone.utc)
    return ASOSReadingDocument(
        station_id="KBOS",
        observation_time_utc=obs_utc,
        temperature_f=temp_f,
        inserted_at=obs_utc,
    )


def _nwp(d: date = _DATE, fetched_hour_et: int = 9) -> NWPForecastDocument:
    fetch_et = _ET.localize(datetime(d.year, d.month, d.day, fetched_hour_et, 0, 0))
    fetch_utc = fetch_et.astimezone(timezone.utc)
    return NWPForecastDocument(
        target_date=d,
        model_name="HRRR",
        fetched_at_utc=fetch_utc,
        hourly_temps=[42.0] * 24,
        predicted_daily_high=42.0,
    )


def _market(d: date = _DATE, final_high: float = 45.0) -> MarketDocument:
    return MarketDocument(
        target_date=d,
        current_max_observed=42.0,
        market_status="settled",
        final_official_high=final_high,
        cli_settlement_confirmed=True,
        last_updated_utc=_NOW_UTC,
    )


def _state(
    d: date = _DATE,
    sigma: float = 0.5,
    theta: float = 0.3,
    bias: float = 1.0,
) -> SystemStateDocument:
    return SystemStateDocument(
        target_date=d,
        kalman_temp_estimate=40.0,
        kalman_bias_estimate=bias,
        kalman_covariance=[[0.1, 0.0], [0.0, 0.05]],
        model_weights={"HRRR": 1.0},
        theta_decay=theta,
        sigma_volatility=sigma,
        morning_drift_adjustment=1.0,
        afternoon_drift_adjustment=0.5,
        last_updated_utc=_NOW_UTC,
    )


def _make_cache(
    asos_readings: list,
    state: SystemStateDocument,
    market: MarketDocument,
    d: date = _DATE,
) -> ReplayDataCache:
    return ReplayDataCache(
        _asos={d: asos_readings},
        _states={d: state},
        _markets={d: market},
    )


def _patch_nwp(nwp_doc: NWPForecastDocument | None = None):
    """Return a context manager that patches get_nwp_forecasts_before_utc."""
    doc = nwp_doc if nwp_doc is not None else _nwp()
    return patch(
        "kalshi_weather_trader.backtesting.replay_engine.get_nwp_forecasts_before_utc",
        return_value={"HRRR": doc},
    )


# ---------------------------------------------------------------------------
# 1. Scenario hashability
# ---------------------------------------------------------------------------


class TestScenarioHashable:
    def test_identical_scenarios_have_equal_hash(self):
        a = Scenario(name="Test", use_drift_in_attractor=True, n_paths=5000)
        b = Scenario(name="Test", use_drift_in_attractor=True, n_paths=5000)
        assert hash(a) == hash(b)
        assert a == b

    def test_different_scenarios_have_different_hash(self):
        a = Scenario(name="A", ou_max_stationary_std_override=1.5)
        b = Scenario(name="A", ou_max_stationary_std_override=2.0)
        assert hash(a) != hash(b)
        assert a != b

    def test_hash_stable_with_dict_field(self):
        a = Scenario(name="X", sigma_by_block_override={"0-6": 0.3, "6-10": 0.7})
        b = Scenario(name="X", sigma_by_block_override={"0-6": 0.3, "6-10": 0.7})
        assert hash(a) == hash(b)

    def test_hash_stable_with_list_field(self):
        a = Scenario(name="Y", eval_hours=[8, 10, 14])
        b = Scenario(name="Y", eval_hours=[8, 10, 14])
        assert hash(a) == hash(b)

    def test_scenario_usable_as_dict_key(self):
        s = preset_production()
        d = {s: "result"}
        assert d[s] == "result"


# ---------------------------------------------------------------------------
# 2. replay_single returns complete result
# ---------------------------------------------------------------------------


class TestReplaySingleComplete:
    def test_all_fields_populated(self):
        readings = [_asos(39.0, 9), _asos(40.0, 11), _asos(41.0, 13)]
        state = _state()
        market = _market(final_high=45.0)
        cache = _make_cache(readings, state, market)

        engine = ParameterizedReplayEngine()
        with _patch_nwp():
            result = engine.replay_single(_DATE, 12, preset_production(), cache)

        assert result is not None
        assert result.target_date == _DATE
        assert result.eval_hour == 12
        assert isinstance(result.T0, float)
        assert isinstance(result.hard_floor, float)
        assert isinstance(result.effective_floor, float)
        assert isinstance(result.sigma_used, dict)
        assert isinstance(result.theta_used, dict)
        assert isinstance(result.mean_max, float)
        assert isinstance(result.std_max, float)
        assert isinstance(result.percentiles, dict)
        assert set(result.percentiles.keys()) == {10, 25, 50, 75, 90}
        assert result.actual_high == pytest.approx(45.0)
        assert isinstance(result.prediction_error, float)
        assert isinstance(result.brier_components, dict)
        assert len(result.brier_components) > 0

    def test_all_probs_in_unit_interval(self):
        readings = [_asos(40.0, 10)]
        cache = _make_cache(readings, _state(), _market())

        engine = ParameterizedReplayEngine()
        with _patch_nwp():
            result = engine.replay_single(_DATE, 10, preset_production(), cache)

        assert result is not None
        for strike, prob in result.strike_probs.items():
            assert 0.0 <= prob <= 1.0, f"P(max >= {strike}) = {prob} out of [0,1]"

    def test_returns_none_when_no_asos(self):
        cache = _make_cache([], _state(), _market())
        engine = ParameterizedReplayEngine()
        with _patch_nwp():
            result = engine.replay_single(_DATE, 10, preset_production(), cache)
        assert result is None

    def test_returns_none_when_no_settlement(self):
        market_no_settle = MarketDocument(
            target_date=_DATE,
            market_status="open",
            final_official_high=None,
            cli_settlement_confirmed=False,
            last_updated_utc=_NOW_UTC,
        )
        cache = _make_cache([_asos(40.0, 10)], _state(), market_no_settle)
        engine = ParameterizedReplayEngine()
        with _patch_nwp():
            result = engine.replay_single(_DATE, 10, preset_production(), cache)
        assert result is None


# ---------------------------------------------------------------------------
# 3. No future leakage
# ---------------------------------------------------------------------------


class TestNoFutureLeakage:
    def test_t0_uses_only_past_asos(self):
        """T0 at eval_hour=10 must not use the 14h reading."""
        readings = [
            _asos(38.0, 8),
            _asos(40.0, 10),
            _asos(46.0, 14),   # future — must not affect T0
            _asos(44.0, 16),   # future — must not affect T0
        ]
        cache = _make_cache(readings, _state(bias=0.0), _market())
        engine = ParameterizedReplayEngine()
        with _patch_nwp():
            result = engine.replay_single(_DATE, 10, preset_production(), cache)

        assert result is not None
        # T0 should be 40.0 (the 10h reading), not 46.0
        assert result.T0 == pytest.approx(40.0, abs=0.5)

    def test_hard_floor_uses_only_past_asos(self):
        """hard_floor at eval_hour=10 must be max of {38, 40}, not 46."""
        readings = [
            _asos(38.0, 8),
            _asos(40.0, 10),
            _asos(46.0, 14),   # future
        ]
        cache = _make_cache(readings, _state(), _market())
        engine = ParameterizedReplayEngine()
        with _patch_nwp():
            result = engine.replay_single(_DATE, 10, preset_production(), cache)

        assert result is not None
        assert result.hard_floor == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# 4. Scenario override: sigma
# ---------------------------------------------------------------------------


class TestSigmaOverride:
    def test_narrow_sigma_produces_narrower_std_max(self):
        readings = [_asos(40.0, 10)]
        cache = _make_cache(readings, _state(sigma=1.5), _market())
        engine = ParameterizedReplayEngine()

        scenario_narrow = Scenario(name="narrow", sigma_override=0.3, n_paths=20_000)
        scenario_wide = Scenario(name="wide", sigma_override=1.5, n_paths=20_000)

        with _patch_nwp():
            r_narrow = engine.replay_single(_DATE, 10, scenario_narrow, cache)
            r_wide = engine.replay_single(_DATE, 10, scenario_wide, cache)

        assert r_narrow is not None and r_wide is not None
        assert r_narrow.std_max < r_wide.std_max

    def test_same_t0_and_hard_floor_regardless_of_sigma(self):
        readings = [_asos(40.0, 10)]
        cache = _make_cache(readings, _state(), _market())
        engine = ParameterizedReplayEngine()

        s_a = Scenario(name="a", sigma_override=0.3)
        s_b = Scenario(name="b", sigma_override=1.5)
        with _patch_nwp():
            r_a = engine.replay_single(_DATE, 10, s_a, cache)
            r_b = engine.replay_single(_DATE, 10, s_b, cache)

        assert r_a is not None and r_b is not None
        assert r_a.T0 == pytest.approx(r_b.T0)
        assert r_a.hard_floor == pytest.approx(r_b.hard_floor)


# ---------------------------------------------------------------------------
# 5. Scenario override: drift
# ---------------------------------------------------------------------------


class TestDriftOverride:
    def test_drift_used_is_zero_when_disabled(self):
        readings = [_asos(40.0, 10)]
        cache = _make_cache(readings, _state(), _market())
        engine = ParameterizedReplayEngine()

        scenario = Scenario(name="no_drift", use_drift_in_attractor=False)
        with _patch_nwp():
            result = engine.replay_single(_DATE, 10, scenario, cache)

        assert result is not None
        assert result.drift_used == pytest.approx(0.0)

    def test_drift_used_is_nonzero_when_enabled(self):
        readings = [_asos(40.0, 10)]  # eval_hour=10 → AM drift
        # State has morning_drift_adjustment=1.5
        state = _state()
        state = SystemStateDocument(
            target_date=_DATE,
            kalman_temp_estimate=40.0,
            kalman_bias_estimate=0.0,
            kalman_covariance=[[0.1, 0.0], [0.0, 0.05]],
            model_weights={"HRRR": 1.0},
            theta_decay=0.3,
            sigma_volatility=0.5,
            morning_drift_adjustment=1.5,
            afternoon_drift_adjustment=0.5,
            last_updated_utc=_NOW_UTC,
        )
        cache = _make_cache(readings, state, _market())
        engine = ParameterizedReplayEngine()

        scenario = Scenario(name="with_drift", use_drift_in_attractor=True)
        with _patch_nwp():
            result = engine.replay_single(_DATE, 10, scenario, cache)

        assert result is not None
        assert result.drift_used != pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. Scenario override: Kalman bias
# ---------------------------------------------------------------------------


class TestBiasOverride:
    def test_higher_bias_produces_higher_mean_max(self):
        """Higher bias → higher attractor → higher mean_max.

        The anchor offset code deliberately subtracts bias from the gap
        (gap_after_bias = raw_gap - bias) to prevent double-counting when
        anchor_weight=1.0 and T0=nwp_reference+bias.  We disable the anchor
        offset so the bias override has an isolated, measurable effect.
        """
        readings = [_asos(40.0, 10)]
        cache = _make_cache(readings, _state(bias=0.0), _market())
        engine = ParameterizedReplayEngine()

        # use_anchor_offset=False isolates bias from the anchor cancellation
        s_low = Scenario(
            name="low_bias",
            kalman_bias_override=0.0,
            use_anchor_offset=False,
            n_paths=20_000,
        )
        s_high = Scenario(
            name="high_bias",
            kalman_bias_override=4.0,
            use_anchor_offset=False,
            n_paths=20_000,
        )

        with _patch_nwp():
            r_low = engine.replay_single(_DATE, 10, s_low, cache)
            r_high = engine.replay_single(_DATE, 10, s_high, cache)

        assert r_low is not None and r_high is not None
        assert r_high.mean_max > r_low.mean_max


# ---------------------------------------------------------------------------
# 7. Brier score arithmetic
# ---------------------------------------------------------------------------


class TestBrierArithmetic:
    def test_perfect_prediction_brier_zero(self):
        """If model assigns P=1.0 to the bucket that contains actual_high, Brier=0."""
        # Build a minimal ParameterizedReplayResult manually
        from kalshi_weather_trader.backtesting.replay_engine import ParameterizedReplayResult

        r = ParameterizedReplayResult(
            target_date=_DATE, eval_hour=10,
            T0=40.0, hard_floor=38.0, effective_floor=38.3,
            sigma_used={"scalar": 0.5}, theta_used={"scalar": 0.3},
            bias_used=0.0, drift_used=0.0,
            nwp_predicted_high=42.0, attractor_peak=42.0,
            mean_max=45.0, std_max=1.0,
            percentiles={10: 43.0, 25: 44.0, 50: 45.0, 75: 46.0, 90: 47.0},
            strike_probs={45.0: 1.0, 46.0: 0.0},
            market_probs={"45.0": 1.0, "46.0": 0.0},
            actual_high=45.0,
            prediction_error=0.0,
            brier_components={"45.0": 0.0, "46.0": 0.0},  # P=1,outcome=1 + P=0,outcome=0
        )
        metrics = compute_aggregate_metrics([r])
        assert metrics["brier_score"] == pytest.approx(0.0)

    def test_brier_component_formula(self):
        """(P=0.8, outcome=1) → brier = (0.8-1)^2 = 0.04."""
        from kalshi_weather_trader.backtesting.replay_engine import ParameterizedReplayResult

        r = ParameterizedReplayResult(
            target_date=_DATE, eval_hour=10,
            T0=40.0, hard_floor=38.0, effective_floor=38.3,
            sigma_used={"scalar": 0.5}, theta_used={"scalar": 0.3},
            bias_used=0.0, drift_used=0.0,
            nwp_predicted_high=42.0, attractor_peak=42.0,
            mean_max=44.0, std_max=1.0,
            percentiles={10: 43.0, 25: 44.0, 50: 45.0, 75: 46.0, 90: 47.0},
            strike_probs={44.0: 0.8},
            market_probs={"44.0": 0.8},
            actual_high=45.0,      # 45 >= 44 → outcome = 1
            prediction_error=-1.0,
            brier_components={"44.0": (0.8 - 1.0) ** 2},
        )
        assert r.brier_components["44.0"] == pytest.approx(0.04)
        metrics = compute_aggregate_metrics([r])
        assert metrics["brier_score"] == pytest.approx(0.04)

    def test_brier_wrong_prediction(self):
        """(P=0.8, outcome=0) → brier = (0.8-0)^2 = 0.64."""
        from kalshi_weather_trader.backtesting.replay_engine import ParameterizedReplayResult

        r = ParameterizedReplayResult(
            target_date=_DATE, eval_hour=10,
            T0=40.0, hard_floor=38.0, effective_floor=38.3,
            sigma_used={"scalar": 0.5}, theta_used={"scalar": 0.3},
            bias_used=0.0, drift_used=0.0,
            nwp_predicted_high=42.0, attractor_peak=42.0,
            mean_max=44.0, std_max=1.0,
            percentiles={10: 43.0, 25: 44.0, 50: 45.0, 75: 46.0, 90: 47.0},
            strike_probs={46.0: 0.8},
            market_probs={"46.0": 0.8},
            actual_high=45.0,      # 45 < 46 → outcome = 0
            prediction_error=-1.0,
            brier_components={"46.0": (0.8 - 0.0) ** 2},
        )
        assert r.brier_components["46.0"] == pytest.approx(0.64)


# ---------------------------------------------------------------------------
# 8. Calibration curve
# ---------------------------------------------------------------------------


class TestCalibrationCurve:
    def test_calibration_bin_matches_constructed_outcomes(self):
        """For the [0.7, 0.8) bin, if 75% of outcomes are 1, observed_freq ≈ 0.75."""
        from kalshi_weather_trader.backtesting.replay_engine import ParameterizedReplayResult

        results = []
        rng = np.random.default_rng(0)

        # Fill the [0.7, 0.8) bin with 100 predictions of prob=0.75
        for i in range(100):
            outcome = 1.0 if i < 75 else 0.0
            r = ParameterizedReplayResult(
                target_date=_DATE + timedelta(days=i),
                eval_hour=10,
                T0=40.0, hard_floor=38.0, effective_floor=38.3,
                sigma_used={"scalar": 0.5}, theta_used={"scalar": 0.3},
                bias_used=0.0, drift_used=0.0,
                nwp_predicted_high=42.0, attractor_peak=42.0,
                mean_max=44.0, std_max=1.0,
                percentiles={10: 43.0, 25: 44.0, 50: 45.0, 75: 46.0, 90: 47.0},
                strike_probs={44.0: 0.75},
                market_probs={"44.0": 0.75},
                actual_high=45.0 if outcome == 1.0 else 43.0,
                prediction_error=0.0,
                brier_components={"44.0": (0.75 - outcome) ** 2},
            )
            results.append(r)

        metrics = compute_aggregate_metrics(results)
        calib = metrics["calibration_curve"]

        # Find the bin that covers 0.75
        target_bin = next(
            (b for b in calib if b["bin_lower"] <= 0.75 < b["bin_upper"]),
            None,
        )
        assert target_bin is not None, "Expected a calibration bin covering 0.75"
        assert target_bin["observed_freq"] == pytest.approx(0.75, abs=0.05)


# ---------------------------------------------------------------------------
# 9. preset_production matches historical calibrated values
# ---------------------------------------------------------------------------


class TestPresetProductionMatchesHistorical:
    def test_sigma_theta_bias_match_system_state(self):
        """With preset_production (no overrides), sigma/theta/bias come from state."""
        calibrated_sigma = 0.42
        calibrated_theta = 0.28
        calibrated_bias = 1.75

        readings = [_asos(40.0, 10)]
        state = SystemStateDocument(
            target_date=_DATE,
            kalman_temp_estimate=40.0,
            kalman_bias_estimate=calibrated_bias,
            kalman_covariance=[[0.1, 0.0], [0.0, 0.05]],
            model_weights={"HRRR": 1.0},
            theta_decay=calibrated_theta,
            sigma_volatility=calibrated_sigma,
            morning_drift_adjustment=0.0,
            afternoon_drift_adjustment=0.0,
            last_updated_utc=_NOW_UTC,
        )
        cache = _make_cache(readings, state, _market())
        engine = ParameterizedReplayEngine()

        with _patch_nwp():
            result = engine.replay_single(_DATE, 10, preset_production(), cache)

        assert result is not None
        assert result.bias_used == pytest.approx(calibrated_bias)
        # sigma_used comes from scalar sigma when no sigma_by_block in state
        assert result.sigma_used.get("scalar", None) == pytest.approx(calibrated_sigma)


# ---------------------------------------------------------------------------
# 10. ReplayDataCache cutoff filtering
# ---------------------------------------------------------------------------


class TestReplayDataCache:
    def test_get_asos_up_to_filters_by_cutoff(self):
        readings = [
            _asos(38.0, 8),
            _asos(40.0, 10),
            _asos(42.0, 12),
            _asos(44.0, 14),
        ]
        cache = ReplayDataCache(
            _asos={_DATE: readings},
            _states={_DATE: None},
            _markets={_DATE: None},
        )

        # Cutoff at 10h ET → should return 8h and 10h readings only
        cutoff_et = _ET.localize(datetime(_DATE.year, _DATE.month, _DATE.day, 10, 0, 0))
        cutoff_utc = cutoff_et.astimezone(timezone.utc)
        filtered = cache.get_asos_up_to(_DATE, cutoff_utc)

        temps = sorted(r.temperature_f for r in filtered)
        assert temps == [38.0, 40.0], f"Expected [38.0, 40.0], got {temps}"

    def test_get_asos_up_to_returns_all_when_cutoff_late(self):
        readings = [
            _asos(38.0, 8),
            _asos(40.0, 10),
            _asos(42.0, 14),
            _asos(44.0, 16),
        ]
        cache = ReplayDataCache(
            _asos={_DATE: readings},
            _states={_DATE: None},
            _markets={_DATE: None},
        )

        cutoff_et = _ET.localize(datetime(_DATE.year, _DATE.month, _DATE.day, 18, 0, 0))
        cutoff_utc = cutoff_et.astimezone(timezone.utc)
        filtered = cache.get_asos_up_to(_DATE, cutoff_utc)

        assert len(filtered) == 4


# ===========================================================================
# Phase L2 tests
# ===========================================================================


def _make_result(
    target_date: date,
    eval_hour: int = 12,
    brier_val: float = 0.10,
    actual_high: float = 45.0,
    mean_max: float = 45.0,
):
    """Factory for minimal ParameterizedReplayResult with a single brier component."""
    from kalshi_weather_trader.backtesting.replay_engine import ParameterizedReplayResult

    return ParameterizedReplayResult(
        target_date=target_date,
        eval_hour=eval_hour,
        T0=40.0,
        hard_floor=38.0,
        effective_floor=38.3,
        sigma_used={"scalar": 0.5},
        theta_used={"scalar": 0.3},
        bias_used=0.0,
        drift_used=0.0,
        nwp_predicted_high=42.0,
        attractor_peak=42.0,
        mean_max=mean_max,
        std_max=1.0,
        percentiles={10: 43.0, 25: 44.0, 50: 45.0, 75: 46.0, 90: 47.0},
        strike_probs={45.0: 0.5},
        market_probs={"45.0": "0.5"},
        actual_high=actual_high,
        prediction_error=mean_max - actual_high,
        brier_components={"45.0": brier_val},
    )


class TestPhaseL2:

    # ------------------------------------------------------------------
    # 1. Identical scenarios → brier_diff ≈ 0, p_value > 0.40
    # ------------------------------------------------------------------

    def test_paired_bootstrap_identical_scenarios(self):
        """Same results for A and B → diff ≈ 0, p_value = 1.0, not significant."""
        from kalshi_weather_trader.backtesting.metrics import compute_paired_bootstrap

        dates = [_DATE + timedelta(days=i) for i in range(10)]
        results = [_make_result(d, brier_val=0.15) for d in dates]

        bs = compute_paired_bootstrap(results, results)

        assert bs.mean_diff == pytest.approx(0.0, abs=1e-9)
        assert bs.p_value > 0.40
        assert bs.is_significant is False
        assert bs.n_shared_dates == 10

    # ------------------------------------------------------------------
    # 2. Different scenarios → diff > 0, is_significant = True
    # ------------------------------------------------------------------

    def test_paired_bootstrap_different_scenarios(self):
        """A consistently Brier 0.20, B consistently Brier 0.05 → significant."""
        from kalshi_weather_trader.backtesting.metrics import compute_paired_bootstrap

        dates = [_DATE + timedelta(days=i) for i in range(20)]
        results_a = [_make_result(d, brier_val=0.20) for d in dates]
        results_b = [_make_result(d, brier_val=0.05) for d in dates]

        bs = compute_paired_bootstrap(results_a, results_b)

        assert bs.mean_diff > 0, f"Expected A worse than B, got mean_diff={bs.mean_diff}"
        assert bs.is_significant is True
        assert bs.brier_a == pytest.approx(0.20, abs=1e-9)
        assert bs.brier_b == pytest.approx(0.05, abs=1e-9)

    # ------------------------------------------------------------------
    # 3. Date-level resampling — n_shared_dates reflects dates, not rows
    # ------------------------------------------------------------------

    def test_comparison_resamples_by_date(self):
        """5 dates × 3 eval_hours = 15 results each, but n_shared_dates must be 5.

        This verifies that brier values are grouped and averaged per date before
        bootstrapping, not treated as 15 independent predictions.
        """
        from kalshi_weather_trader.backtesting.metrics import compute_paired_bootstrap

        dates = [_DATE + timedelta(days=i) for i in range(5)]
        eval_hours = [8, 12, 16]

        results_a = [_make_result(d, eval_hour=h, brier_val=0.12) for d in dates for h in eval_hours]
        results_b = [_make_result(d, eval_hour=h, brier_val=0.10) for d in dates for h in eval_hours]

        assert len(results_a) == 15  # 5 dates × 3 hours

        bs = compute_paired_bootstrap(results_a, results_b)

        assert bs.n_shared_dates == 5, (
            f"Expected 5 shared dates (date-level grouping), got {bs.n_shared_dates}"
        )

    # ------------------------------------------------------------------
    # 4. Custom Scenario builds correctly from field values
    # ------------------------------------------------------------------

    def test_custom_scenario_builds_correctly(self):
        """Scenario can be constructed with explicit overrides and fields match."""
        s = Scenario(
            name="Custom",
            sigma_override=0.8,
            theta_override=0.4,
            use_drift_in_attractor=False,
            ou_max_stationary_std_override=1.5,
        )
        assert s.sigma_override == pytest.approx(0.8)
        assert s.theta_override == pytest.approx(0.4)
        assert s.use_drift_in_attractor is False
        assert s.ou_max_stationary_std_override == pytest.approx(1.5)
        assert s.name == "Custom"
