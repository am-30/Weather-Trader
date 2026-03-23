"""
Unit tests for the Phase 0 backtesting infrastructure.

Tests:
- ReplayEngine correctly reconstructs historical MCParams without future leakage
- Hard floor uses only ASOS readings up to eval_hour (no lookahead)
- T0 uses only ASOS readings up to eval_hour (no lookahead)
- Brier score == 0 for a perfect model
- Brier score == 0.25 for a random (0.5) model
- Calibration curve: observed_freq matches predicted_prob for perfectly calibrated bins
- compare_variants returns non-significant result when variants are nearly identical
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from kalshi_weather_trader.backtesting.comparison import compare_variants
from kalshi_weather_trader.backtesting.metrics import (
    compute_backtest_metrics,
    _calibration_curve,
    _extract_strike_rows,
)
from kalshi_weather_trader.backtesting.replay_engine import ReplayEngine, _closest_reading
from kalshi_weather_trader.db.schemas import (
    ASOSReadingDocument,
    MarketDocument,
    NWPForecastDocument,
    SystemStateDocument,
)

_NOW = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
_DATE = date(2026, 3, 15)


# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------


def _asos(temp_f: float, hour_et: int, date_: date = _DATE) -> ASOSReadingDocument:
    """Create a synthetic ASOSReadingDocument at a given ET hour."""
    import pytz
    et = pytz.timezone("America/New_York")
    obs_et = et.localize(datetime(date_.year, date_.month, date_.day, hour_et, 0, 0))
    obs_utc = obs_et.astimezone(timezone.utc)
    return ASOSReadingDocument(
        station_id="KBOS",
        observation_time_utc=obs_utc,
        temperature_f=temp_f,
        inserted_at=obs_utc,
    )


def _nwp(target_date: date = _DATE, fetched_hour_et: int = 9) -> NWPForecastDocument:
    """Create a synthetic NWPForecastDocument fetched at a given ET hour."""
    import pytz
    et = pytz.timezone("America/New_York")
    fetch_et = et.localize(datetime(target_date.year, target_date.month, target_date.day, fetched_hour_et, 0, 0))
    fetch_utc = fetch_et.astimezone(timezone.utc)
    curve = [40.0] * 24
    return NWPForecastDocument(
        target_date=target_date,
        model_name="HRRR",
        fetched_at_utc=fetch_utc,
        hourly_temps=curve,
        predicted_daily_high=40.0,
    )


def _market(
    target_date: date = _DATE,
    final_high: float = 41.0,
    current_max: float = 39.0,
) -> MarketDocument:
    """Create a synthetic settled MarketDocument."""
    return MarketDocument(
        target_date=target_date,
        current_max_observed=current_max,
        market_status="settled",
        final_official_high=final_high,
        last_updated_utc=_NOW,
    )


def _state(target_date: date = _DATE) -> SystemStateDocument:
    """Create a synthetic SystemStateDocument."""
    return SystemStateDocument(
        target_date=target_date,
        kalman_temp_estimate=39.5,
        kalman_bias_estimate=0.5,
        kalman_covariance=[[0.1, 0.0], [0.0, 0.05]],
        model_weights={"HRRR": 1.0},
        theta_decay=0.3,
        sigma_volatility=0.5,
        morning_drift_adjustment=0.0,
        afternoon_drift_adjustment=0.0,
        last_updated_utc=_NOW,
    )


def _make_replay_df(
    dates: list[date],
    eval_hours: list[int],
    strikes: list[float],
    prob_fn,
    actual_high_fn,
) -> pd.DataFrame:
    """Build a synthetic replay DataFrame for metrics/comparison testing.

    Args:
        dates:          List of target dates.
        eval_hours:     List of eval hours.
        strikes:        List of strike values.
        prob_fn:        Callable(date, hour, strike) -> float predicted prob.
        actual_high_fn: Callable(date) -> float actual high.

    Returns:
        DataFrame in replay_all() output format.
    """
    rows = []
    for d in dates:
        actual_high = actual_high_fn(d)
        for h in eval_hours:
            row: dict = {
                "target_date": d,
                "eval_hour": h,
                "T0": 39.0,
                "hard_floor": 38.0,
                "sigma": 0.5,
                "theta": 0.3,
                "bias": 0.0,
                "nwp_predicted_high": 41.0,
                "actual_high": actual_high,
                "mean_max": 40.0,
                "std_max": 1.0,
            }
            for s in strikes:
                prob = prob_fn(d, h, s)
                row[f"prob_{s:.1f}"] = prob
                outcome = 1.0 if actual_high >= s else 0.0
                row[f"brier_{s:.1f}"] = (prob - outcome) ** 2
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: replay_date() populates all required ReplayResult fields
# ---------------------------------------------------------------------------


class TestReplaySingleDate:
    def test_replay_single_date(self):
        """replay_date() returns a ReplayResult with all required fields populated."""
        asos_readings = [_asos(38.0, 9), _asos(39.5, 11), _asos(40.0, 13)]
        market = _market(final_high=41.0, current_max=40.0)
        state = _state()
        nwp_before = {"HRRR": _nwp(fetched_hour_et=9)}

        with (
            patch(
                "kalshi_weather_trader.backtesting.replay_engine.get_asos_readings_for_date",
                return_value=asos_readings,
            ),
            patch(
                "kalshi_weather_trader.backtesting.replay_engine.get_market",
                return_value=market,
            ),
            patch(
                "kalshi_weather_trader.backtesting.replay_engine.get_system_state",
                return_value=state,
            ),
            patch(
                "kalshi_weather_trader.backtesting.replay_engine.get_nwp_forecasts_before_utc",
                return_value=nwp_before,
            ),
        ):
            engine = ReplayEngine()
            results = engine.replay_date(_DATE, eval_hours=[12], seed=42)

        assert len(results) == 1, "Expected exactly one result for eval_hour=12"
        r = results[0]

        assert r.target_date == _DATE
        assert r.eval_hour == 12
        assert r.actual_high == pytest.approx(41.0)
        assert isinstance(r.T0, float)
        assert isinstance(r.hard_floor, float)
        assert isinstance(r.mean_max, float)
        assert isinstance(r.std_max, float)

        # All probabilities must be in [0, 1]
        for strike, prob in r.strike_probs.items():
            assert 0.0 <= prob <= 1.0, f"prob={prob} out of [0,1] for strike={strike}"

        # Brier scores must be non-negative
        for strike, bs in r.brier_scores.items():
            assert bs >= 0.0, f"Negative Brier score {bs} for strike={strike}"


# ---------------------------------------------------------------------------
# Test 2: No future leakage in T0
# ---------------------------------------------------------------------------


class TestNoFutureleakageT0:
    def test_replay_uses_historical_not_current(self):
        """T0 at eval_hour=10 must use hour-10 reading, not hour-14."""
        asos_readings = [
            _asos(35.0, 8),   # 8 AM — past at eval 10
            _asos(37.0, 10),  # 10 AM — the eval moment
            _asos(40.0, 12),  # 12 PM — FUTURE, must not be used
            _asos(42.0, 14),  # 2 PM — FUTURE, must not be used
        ]
        market = _market(final_high=41.5, current_max=42.0)
        state = _state()

        with (
            patch(
                "kalshi_weather_trader.backtesting.replay_engine.get_asos_readings_for_date",
                return_value=asos_readings,
            ),
            patch(
                "kalshi_weather_trader.backtesting.replay_engine.get_market",
                return_value=market,
            ),
            patch(
                "kalshi_weather_trader.backtesting.replay_engine.get_system_state",
                return_value=state,
            ),
            patch(
                "kalshi_weather_trader.backtesting.replay_engine.get_nwp_forecasts_before_utc",
                return_value={},
            ),
        ):
            engine = ReplayEngine()
            results = engine.replay_date(_DATE, eval_hours=[10], seed=42)

        assert len(results) == 1
        r = results[0]
        # T0 must be the 10 AM reading (37.0), NOT 40 or 42 from the future
        assert r.T0 == pytest.approx(37.0), (
            f"T0={r.T0} should be 37.0 (hour-10 reading), not a future reading"
        )


# ---------------------------------------------------------------------------
# Test 3: No future leakage in hard floor
# ---------------------------------------------------------------------------


class TestNoFutureLeakageHardFloor:
    def test_no_future_leakage_in_hard_floor(self):
        """Hard floor at eval_hour=10 must not include 2PM reading of 42°F."""
        asos_readings = [
            _asos(35.0, 8),   # 8 AM — past at eval 10
            _asos(36.0, 10),  # 10 AM — the eval moment
            _asos(42.0, 14),  # 2 PM — FUTURE, must not be included in hard floor
        ]
        market = _market(final_high=42.0, current_max=42.0)
        state = _state()

        with (
            patch(
                "kalshi_weather_trader.backtesting.replay_engine.get_asos_readings_for_date",
                return_value=asos_readings,
            ),
            patch(
                "kalshi_weather_trader.backtesting.replay_engine.get_market",
                return_value=market,
            ),
            patch(
                "kalshi_weather_trader.backtesting.replay_engine.get_system_state",
                return_value=state,
            ),
            patch(
                "kalshi_weather_trader.backtesting.replay_engine.get_nwp_forecasts_before_utc",
                return_value={},
            ),
        ):
            engine = ReplayEngine()
            results = engine.replay_date(_DATE, eval_hours=[10], seed=42)

        assert len(results) == 1
        r = results[0]
        # hard_floor must be max(35.0, 36.0) = 36.0, NOT 42.0
        assert r.hard_floor <= 36.0, (
            f"hard_floor={r.hard_floor} should be ≤36.0 (no 2PM leakage)"
        )


# ---------------------------------------------------------------------------
# Test 4: Brier score == 0 for a perfect model
# ---------------------------------------------------------------------------


class TestBrierScorePerfectModel:
    def test_brier_score_perfect_model(self):
        """A model that predicts p=outcome exactly should have Brier score == 0."""
        dates = [date(2026, 3, d) for d in range(10, 20)]
        strikes = [38.0, 39.0, 40.0, 41.0, 42.0]

        def actual_high_fn(d: date) -> float:
            return 40.0  # All days settle at 40°F

        def prob_fn(d: date, h: int, s: float) -> float:
            # Perfect model: knows the outcome exactly
            return 1.0 if 40.0 >= s else 0.0

        df = _make_replay_df(dates, [10, 14], strikes, prob_fn, actual_high_fn)
        metrics = compute_backtest_metrics(df, n_bootstrap=100)

        assert metrics["brier_score"]["value"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 5: Brier score == 0.25 for a random (0.5) model
# ---------------------------------------------------------------------------


class TestBrierScoreRandomModel:
    def test_brier_score_random_model(self):
        """A model that always predicts 0.5 should have Brier score == 0.25."""
        dates = [date(2026, 3, d) for d in range(10, 20)]
        strikes = [38.0, 39.0, 40.0, 41.0, 42.0]

        # Mix of outcomes: half above, half below each strike
        actual_highs = {
            date(2026, 3, 10): 38.0, date(2026, 3, 11): 42.0,
            date(2026, 3, 12): 38.0, date(2026, 3, 13): 42.0,
            date(2026, 3, 14): 38.0, date(2026, 3, 15): 42.0,
            date(2026, 3, 16): 38.0, date(2026, 3, 17): 42.0,
            date(2026, 3, 18): 38.0, date(2026, 3, 19): 42.0,
        }

        df = _make_replay_df(
            dates=dates,
            eval_hours=[10],
            strikes=strikes,
            prob_fn=lambda d, h, s: 0.5,
            actual_high_fn=lambda d: actual_highs[d],
        )
        metrics = compute_backtest_metrics(df, n_bootstrap=100)

        assert metrics["brier_score"]["value"] == pytest.approx(0.25, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 6: Calibration curve bins match observed frequency
# ---------------------------------------------------------------------------


class TestCalibrationCurvePerfect:
    def test_calibration_curve_perfect(self):
        """Events predicted at ~0.75 should have observed_freq ≈ 0.75."""
        rng = np.random.default_rng(0)
        n = 1000

        # All predictions in [0.7, 0.8] bucket; 75% of outcomes are 1
        probs = rng.uniform(0.70, 0.80, size=n)
        outcomes = rng.binomial(1, 0.75, size=n).astype(float)

        rows = []
        for i in range(n):
            d = date(2026, 3, 1) + timedelta(days=i % 20)
            rows.append({
                "target_date": d,
                "eval_hour": 10,
                "strike": 40.0,
                "prob": float(probs[i]),
                "outcome": float(outcomes[i]),
                "brier": (float(probs[i]) - float(outcomes[i])) ** 2,
            })
        long_df = pd.DataFrame(rows)

        curve = _calibration_curve(long_df, n_bins=10)
        # Find the [0.7, 0.8) bin
        target_bin = next(
            (b for b in curve if abs(b["bin_lower"] - 0.7) < 0.01),
            None,
        )
        assert target_bin is not None, "Expected bin starting at 0.7 in calibration curve"
        assert target_bin["observed_freq"] == pytest.approx(0.75, abs=0.05), (
            f"Expected observed_freq≈0.75, got {target_bin['observed_freq']:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 7: compare_variants is not significant when variants are nearly identical
# ---------------------------------------------------------------------------


class TestComparisonNoSignificance:
    def test_comparison_no_significance(self):
        """Comparing a model against itself yields p_value > 0.05 (not significant).

        When baseline == variant, every bootstrap resample gives difference = 0.0.
        p_value = fraction of samples where diff >= 0 = 1.0 >> 0.05.
        """
        dates = [date(2026, 3, d) for d in range(1, 21)]  # 20 dates
        strikes = [39.0, 40.0, 41.0]
        rng = np.random.default_rng(0)
        actual_highs = {date(2026, 3, d): float(rng.uniform(38.0, 42.0)) for d in range(1, 21)}

        df = _make_replay_df(
            dates=dates,
            eval_hours=[10, 14],
            strikes=strikes,
            prob_fn=lambda d, h, s: 0.5,
            actual_high_fn=lambda d: actual_highs[d],
        )

        # Pass the same df as both — difference is always exactly 0.0
        result = compare_variants(
            baseline_results=df,
            variant_results=df,
            n_bootstrap=200,
        )

        assert result, "compare_variants returned empty dict"
        assert result["p_value"] > 0.05, (
            f"Expected p_value > 0.05 for identical variants, got {result['p_value']:.3f}"
        )
        assert not result["significant"], (
            "Identical variants must NOT be flagged as significant"
        )
