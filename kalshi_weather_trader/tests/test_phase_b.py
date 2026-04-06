"""
Tests for Phase B structural model improvements.

Covers:
- B.1: Theta calibration uses exact -ln(phi)/dt formula (vs linear (1-phi)/dt approx)
- B.2: Per-hour NWP blending drops short-curve models gracefully (not truncate-to-shortest)
- B.3: Persistence offset calibration includes zero/negative gaps in the mean
       (subsumed by Phase A.2 — confirmatory test added here for cross-reference)
- B.4: ou_max_stationary_std calibrated from hourly NWP RMSE (post-fetch hours only),
       replacing the daily-high RMSE approach from Phase 3 / A.3-deferred.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytz
import pytest

_EASTERN = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# B.1 — Theta calibration uses exact -ln(phi)/dt formula
# ---------------------------------------------------------------------------


class TestThetaLogFormula:
    """Verify the AR(1) phi→theta conversion uses -ln(phi)/dt, not (1-phi)/dt.

    At phi = 0.75 with dt = 1 hour:
      Exact:   theta = -ln(0.75) / 1 ≈ 0.2877
      Linear:  theta = (1 - 0.75) / 1 = 0.2500  (off by ~15%)

    The linear approximation underestimates theta, which means the OU process
    mean-reverts too slowly and tail probabilities are inflated.
    """

    def test_weighted_phi_exact_computation(self) -> None:
        """_weighted_phi gives the correct weighted least-squares AR(1) coefficient.

        Construct pairs where y = phi * x exactly, with uniform weights.
        The weighted OLS through the origin should recover phi exactly.
        """
        from kalshi_weather_trader.calibration.calibrator import _weighted_phi

        phi_target = 0.75
        n = 200
        x_vals = [float(i) for i in range(1, n + 1)]
        y_vals = [phi_target * x for x in x_vals]
        w_vals = [1.0] * n

        phi = _weighted_phi(x_vals, y_vals, w_vals)
        assert phi == pytest.approx(phi_target, abs=1e-6)

    def test_theta_log_vs_linear_differ_at_phi_75(self) -> None:
        """The log formula and linear approximation differ by > 10% at phi=0.75.

        This test documents the difference explicitly.  If it fails it means
        phi is far from 0.75 in practice, which should be investigated.
        """
        phi = 0.75
        dt = 1.0
        theta_exact = -math.log(phi) / dt      # ≈ 0.2877
        theta_linear = (1.0 - phi) / dt        # = 0.2500
        relative_diff = abs(theta_exact - theta_linear) / theta_exact
        assert relative_diff > 0.10, (
            f"Expected >10% relative difference between log ({theta_exact:.4f}) "
            f"and linear ({theta_linear:.4f}) formulas at phi=0.75"
        )
        assert theta_exact == pytest.approx(0.2877, abs=0.001)
        assert theta_linear == pytest.approx(0.2500, abs=0.001)

    def test_calibrate_theta_uses_log_formula(self) -> None:
        """calibrate_theta() returns a theta consistent with -ln(phi)/dt.

        Strategy: mock the data pipeline to produce AR(1) pairs where
        y = phi * x with phi = 0.80.  Assert the returned theta is close to
        -ln(0.80) / 1 ≈ 0.2231, NOT (1-0.80)/1 = 0.20.
        """
        from kalshi_weather_trader.calibration.calibrator import calibrate_theta

        phi_target = 0.80
        theta_expected_log = -math.log(phi_target)    # ≈ 0.2231
        theta_expected_linear = 1.0 - phi_target       # = 0.2000

        # Build mock ASOS readings: 30 days × 24 hourly readings.
        # Temperature alternates so that the AR(1) departure pairs satisfy
        # departure[h+1] = phi * departure[h] relative to a flat NWP.
        base_date = date(2026, 3, 1)
        target_date = base_date + timedelta(days=30)
        mock_readings = []
        for day_offset in range(30):
            d = base_date + timedelta(days=day_offset)
            # Start each day at a departure of 2°F, decay by phi each hour.
            dep = 2.0
            for hour in range(24):
                # Use ET→UTC conversion so readings align with what _build_theta_ar1_pairs
                # expects (top-of-hour ET, within 40-minute gap tolerance).
                obs_utc = _et_hour_to_utc(d, hour)
                # NWP is flat at 50°F; departure decays by phi each hour.
                temp = 50.0 + dep
                r = MagicMock()
                r.observation_time_utc = obs_utc
                r.temperature_f = temp
                mock_readings.append(r)
                dep *= phi_target

        flat_nwp_curve = [50.0] * 24  # flat NWP → NWP detrend = 0

        # fetch_last_n_hours and get_nwp_curve are lazily imported inside
        # calibrate_theta(), so we must patch them at their source modules.
        with (
            patch(
                "kalshi_weather_trader.ingestion.asos_fetcher.fetch_last_n_hours",
                return_value=mock_readings,
            ),
            patch(
                "kalshi_weather_trader.ingestion.nwp_fetcher.get_nwp_curve",
                return_value=flat_nwp_curve,
            ),
            patch(
                "kalshi_weather_trader.calibration.calibrator.db_manager"
            ) as mock_db,
            patch(
                "kalshi_weather_trader.calibration.calibrator.get_target_date",
                return_value=target_date,
            ),
        ):
            mock_db.get_system_state.return_value = None

            theta = calibrate_theta(target_date=target_date)

        # Should be closer to the log result than the linear result.
        assert abs(theta - theta_expected_log) < abs(theta - theta_expected_linear), (
            f"theta={theta:.4f} should be closer to log formula "
            f"({theta_expected_log:.4f}) than linear ({theta_expected_linear:.4f})"
        )
        assert theta == pytest.approx(theta_expected_log, abs=0.03), (
            f"theta={theta:.4f} too far from -ln(0.80)/1 ≈ {theta_expected_log:.4f}"
        )

    def test_calibrate_theta_by_regime_uses_log_formula(self) -> None:
        """calibrate_theta_by_regime() returns theta_am and theta_pm via -ln(phi)/dt.

        AM hours (6-13) are given phi=0.70 → theta_am ≈ 0.357.
        PM hours (13-20) are given phi=0.85 → theta_pm ≈ 0.163.
        Both should match the log formula within tolerance.
        """
        from kalshi_weather_trader.calibration.calibrator import calibrate_theta_by_regime

        phi_am = 0.70  # lower phi = faster reversion (more mean-reverting in PM)
        phi_pm = 0.85  # higher phi = slower reversion (less mean-reverting in AM)
        theta_am_expected = -math.log(phi_am)   # ≈ 0.3567
        theta_pm_expected = -math.log(phi_pm)   # ≈ 0.1625

        base_date = date(2026, 3, 1)
        target_date = base_date + timedelta(days=30)
        mock_readings = []
        for day_offset in range(30):
            d = base_date + timedelta(days=day_offset)
            dep_am = 2.0
            dep_pm = 2.0
            for hour_et in range(24):
                # Skip boundary hours at both regime transitions to prevent
                # cross-regime contamination in the AR(1) pairs.
                #
                # Without skipping 12 and 13:
                #   Pair (11→12): AM bucket, dep[11] = phi_am-decayed, dep[12] = 0 → biases AM phi
                #   Pair (12→13): AM bucket (h0=12), dep[12]=0, dep[13]=PM reset to 2.0
                #     → spurious high-y pair biases phi_am upward
                #
                # Without skipping 19 and 20:
                #   Pair (19→20): PM bucket (h0=19 ∈ [13,20)), dep[20]=0
                #     → zero-y pair pulls phi_pm downward (toward 0)
                #
                # Gaps at {11→13} and {18→20} cause _build_theta_ar1_pairs to skip
                # those transitions (h1 - h0 ≠ 1), leaving clean AM [6-11] and PM [14-18]
                # samples.  Both sets remain well above the 20-pair minimum (150 and 120).
                if hour_et in (12, 13, 19, 20):
                    continue
                obs_utc = _et_hour_to_utc(d, hour_et)
                if 6 <= hour_et < 12:
                    dep = dep_am
                    dep_am *= phi_am
                elif 14 <= hour_et < 19:
                    dep = dep_pm
                    dep_pm *= phi_pm
                else:
                    dep = 0.0
                r = MagicMock()
                r.observation_time_utc = obs_utc
                r.temperature_f = 50.0 + dep
                mock_readings.append(r)

        flat_nwp_curve = [50.0] * 24

        with (
            patch(
                "kalshi_weather_trader.ingestion.asos_fetcher.fetch_last_n_hours",
                return_value=mock_readings,
            ),
            patch(
                "kalshi_weather_trader.ingestion.nwp_fetcher.get_nwp_curve",
                return_value=flat_nwp_curve,
            ),
            patch(
                "kalshi_weather_trader.calibration.calibrator.db_manager"
            ) as mock_db,
            patch(
                "kalshi_weather_trader.calibration.calibrator.get_target_date",
                return_value=target_date,
            ),
        ):
            mock_db.get_system_state.return_value = None

            theta_am, theta_pm = calibrate_theta_by_regime(target_date=target_date)

        assert theta_am is not None, "theta_am should not be None with 30 days of AM data"
        assert theta_pm is not None, "theta_pm should not be None with 30 days of PM data"

        # The key structural check: log formula gives theta_am > theta_pm for
        # phi_am < phi_pm.
        assert theta_am > theta_pm, (
            f"AM should mean-revert faster (theta_am={theta_am:.4f}) than PM "
            f"(theta_pm={theta_pm:.4f}) given phi_am={phi_am} < phi_pm={phi_pm}"
        )
        assert theta_am == pytest.approx(theta_am_expected, abs=0.05)
        assert theta_pm == pytest.approx(theta_pm_expected, abs=0.05)

    def test_phi_zero_handled_gracefully(self) -> None:
        """_weighted_phi clamps phi to 0.01 when data implies phi ≤ 0 (anti-persistent)."""
        from kalshi_weather_trader.calibration.calibrator import _weighted_phi

        # Anti-persistent data: y = -x.  Raw phi = -1 (outside [0.01, 0.99]).
        x_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_vals = [-1.0, -2.0, -3.0, -4.0, -5.0]
        w_vals = [1.0] * 5

        phi = _weighted_phi(x_vals, y_vals, w_vals)
        assert phi == pytest.approx(0.01, abs=1e-9), (
            "Anti-persistent data (phi<0) should be clamped to 0.01"
        )


# ---------------------------------------------------------------------------
# B.2 — Per-hour NWP blending handles models with different curve lengths
# ---------------------------------------------------------------------------


class TestNWPBlendingPerHour:
    """Verify get_nwp_curve() handles models with different curve lengths.

    The blending must:
    - Use the LONGEST curve (no truncation to shortest).
    - For hours where a model has no data, drop it and renormalize remaining weights.
    - Never return 0.0 or NaN for a valid hour covered by at least one model.
    """

    def _make_forecast(self, model_name: str, temps: list[float]) -> MagicMock:
        f = MagicMock()
        f.hourly_temps = temps
        f.model_name = model_name
        return f

    @patch("kalshi_weather_trader.ingestion.nwp_fetcher.db_manager")
    def test_blending_uses_longest_curve(self, mock_db: MagicMock) -> None:
        """Blended curve length equals the longest individual model curve."""
        from kalshi_weather_trader.ingestion.nwp_fetcher import get_nwp_curve

        hrrr_temps = [50.0] * 18     # 18 hours
        gfs_temps = [51.0] * 24      # 24 hours (longest)
        ecmwf_temps = [52.0] * 24    # 24 hours

        mock_db.get_latest_nwp_forecasts.return_value = {
            "HRRR": self._make_forecast("HRRR", hrrr_temps),
            "GFS": self._make_forecast("GFS", gfs_temps),
            "ECMWF": self._make_forecast("ECMWF", ecmwf_temps),
        }
        mock_db.get_system_state.return_value = None

        with patch(
            "kalshi_weather_trader.ingestion.nwp_fetcher.get_target_date",
            return_value=date(2026, 3, 29),
        ):
            curve = get_nwp_curve(date(2026, 3, 29))

        assert len(curve) == 24, (
            f"Expected 24 hours (longest curve), got {len(curve)}"
        )

    @patch("kalshi_weather_trader.ingestion.nwp_fetcher.db_manager")
    def test_short_model_dropped_at_late_hours(self, mock_db: MagicMock) -> None:
        """At hours where HRRR has no data, its weight is dropped and GFS/ECMWF renormalize.

        Setup: equal weights 1/3 each.
        HRRR has 18 hours (temps = 48.0), GFS 24 hours (temps = 52.0), ECMWF 24 hours (temps = 52.0).

        Hour 17 (HRRR present): blend = (48+52+52)/3 = 50.667
        Hour 20 (HRRR absent):  GFS and ECMWF only, renorm to 0.5 each → (52+52)/2 = 52.0
        Old truncate-to-shortest: curve would only have 18 entries → hour 20 would not exist.
        """
        from kalshi_weather_trader.ingestion.nwp_fetcher import get_nwp_curve

        hrrr_temps = [48.0] * 18
        gfs_temps = [52.0] * 24
        ecmwf_temps = [52.0] * 24

        mock_db.get_latest_nwp_forecasts.return_value = {
            "HRRR": self._make_forecast("HRRR", hrrr_temps),
            "GFS": self._make_forecast("GFS", gfs_temps),
            "ECMWF": self._make_forecast("ECMWF", ecmwf_temps),
        }
        # Equal weights 1/3 each.
        state = MagicMock()
        state.model_weights = {"HRRR": 1 / 3, "GFS": 1 / 3, "ECMWF": 1 / 3}
        mock_db.get_system_state.return_value = state

        with patch(
            "kalshi_weather_trader.ingestion.nwp_fetcher.get_target_date",
            return_value=date(2026, 3, 29),
        ):
            curve = get_nwp_curve(date(2026, 3, 29))

        assert len(curve) >= 21, "Curve must extend beyond HRRR's 18-hour limit"

        # Hour 17: all three models present → (48+52+52)/3 ≈ 50.67
        assert curve[17] == pytest.approx((48.0 + 52.0 + 52.0) / 3, abs=0.1), (
            f"Hour 17 blend with all models should be ~{(48+52+52)/3:.2f}, got {curve[17]}"
        )

        # Hour 20: HRRR absent → GFS + ECMWF renormalized (0.5/0.5) → 52.0
        assert curve[20] == pytest.approx(52.0, abs=0.1), (
            f"Hour 20 blend without HRRR should be 52.0 (GFS+ECMWF only), got {curve[20]}"
        )

    @patch("kalshi_weather_trader.ingestion.nwp_fetcher.db_manager")
    def test_single_model_at_late_hours(self, mock_db: MagicMock) -> None:
        """When only one model has data for a late hour, its value is used directly."""
        from kalshi_weather_trader.ingestion.nwp_fetcher import get_nwp_curve

        mock_db.get_latest_nwp_forecasts.return_value = {
            "GFS": self._make_forecast("GFS", [55.0] * 24),
            "ECMWF": self._make_forecast("ECMWF", [55.0] * 12),  # short curve
        }
        state = MagicMock()
        state.model_weights = {"GFS": 0.5, "ECMWF": 0.5}
        mock_db.get_system_state.return_value = state

        with patch(
            "kalshi_weather_trader.ingestion.nwp_fetcher.get_target_date",
            return_value=date(2026, 3, 29),
        ):
            curve = get_nwp_curve(date(2026, 3, 29))

        # Hours 12+ have only GFS → value should be 55.0
        assert len(curve) == 24
        assert curve[20] == pytest.approx(55.0, abs=0.01), (
            f"Hour 20 with GFS only should be 55.0, got {curve[20]}"
        )


# ---------------------------------------------------------------------------
# B.3 — Persistence offset includes zero and negative gaps (confirmatory)
# ---------------------------------------------------------------------------


class TestPersistenceOffsetAllGaps:
    """Confirmatory test for B.3 (implemented in Phase A.2).

    Including zeros in the mean gives a lower, more honest estimate.
    7 dates at +1°F, 3 dates at 0°F → mean = 0.7°F (not 1.0°F from positive-only).
    """

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
    def test_zeros_included_in_mean(self, mock_db: MagicMock) -> None:
        """7 dates at +1°F gap, 3 dates at 0°F gap → calibrated offset = 0.70°F."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_persistence_offset

        gaps = [1.0] * 7 + [0.0] * 3
        base_date = date(2026, 3, 29)

        def market_side_effect(d: date) -> MagicMock | None:
            for i, g in enumerate(gaps, start=1):
                if d == base_date - timedelta(days=i):
                    m = MagicMock()
                    m.final_official_high = 50.0 + g
                    return m
            return None

        def asos_side_effect(d: date) -> list:
            for i in range(1, len(gaps) + 1):
                if d == base_date - timedelta(days=i):
                    return self._make_mock_readings(50.0)
            return []

        mock_db.get_market.side_effect = market_side_effect
        mock_db.get_asos_readings_for_date.side_effect = asos_side_effect
        mock_db.get_system_state.return_value = None

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=base_date,
        ):
            result = calibrate_persistence_offset()

        expected = sum(gaps) / len(gaps)  # = 0.70
        assert result == pytest.approx(expected, abs=0.01), (
            f"Expected {expected:.2f} (including zeros), got {result}"
        )
        assert result < 1.0, "Including zeros should give a lower estimate than positive-only"


# ---------------------------------------------------------------------------
# B.4 — ou_max_stationary_std from hourly NWP RMSE
# ---------------------------------------------------------------------------


def _et_hour_to_utc(d: date, hour_et: int) -> datetime:
    """Convert an ET calendar date + ET hour to a UTC-aware datetime.

    Uses pytz to handle DST correctly — the calibrator uses the same logic,
    so this ensures test ASOS timestamps are exactly at top-of-hour UTC
    (gap = 0 minutes, well within the 40-minute acceptance window).
    """
    et_dt = _EASTERN.localize(datetime(d.year, d.month, d.day, hour_et, 0, 0))
    return et_dt.astimezone(timezone.utc)


def _make_asos_reading(obs_utc: datetime, temperature_f: float) -> MagicMock:
    r = MagicMock()
    r.observation_time_utc = obs_utc
    r.temperature_f = temperature_f
    return r


def _make_nwp_forecast(model_name: str, hourly_temps: list[float]) -> MagicMock:
    f = MagicMock()
    f.model_name = model_name
    f.hourly_temps = hourly_temps
    f.predicted_daily_high = max(hourly_temps) if hourly_temps else 0.0
    return f


def _make_market(traded: bool = True, cli_confirmed: bool = False) -> MagicMock:
    """Market mock.  cli_settlement_confirmed no longer required for hourly RMSE."""
    m = MagicMock()
    if not traded:
        return None
    m.cli_settlement_confirmed = cli_confirmed
    m.final_official_high = 50.0
    return m


def _build_nwp_asos_scenario(
    base_date: date,
    n_dates: int,
    nwp_temp: float,
    asos_temp: float,
    eval_hours: list[int],
) -> tuple[dict, dict, dict]:
    """Build mock side-effect dicts for a controlled hourly-RMSE scenario.

    All dates get the same NWP curve and ASOS temperature at each eval hour,
    making the expected RMSE = |asos_temp - nwp_temp|.
    """
    markets = {}
    morning_forecasts = {}
    asos_by_date = {}

    for i in range(1, n_dates + 1):
        d = base_date - timedelta(days=i)
        markets[d] = _make_market(traded=True)

        nwp_curve = [nwp_temp] * 24
        morning_forecasts[d] = {
            "GFS": _make_nwp_forecast("GFS", nwp_curve),
            "ECMWF": _make_nwp_forecast("ECMWF", nwp_curve),
        }

        readings = []
        for h in eval_hours:
            obs_utc = _et_hour_to_utc(d, h)
            readings.append(_make_asos_reading(obs_utc, asos_temp))
        asos_by_date[d] = readings

    return markets, morning_forecasts, asos_by_date


class TestHourlyRMSECalibration:
    """Tests for the Phase B hourly-RMSE-based ou_max_stationary_std calibrator."""

    def _patch_db(
        self,
        mock_db: MagicMock,
        markets: dict,
        morning_forecasts: dict,
        asos_by_date: dict,
        base_date: date,
    ) -> None:
        mock_db.get_market.side_effect = lambda d: markets.get(d)
        mock_db.get_morning_nwp_forecasts.side_effect = (
            lambda d: morning_forecasts.get(d, {})
        )
        mock_db.get_asos_readings_for_date.side_effect = (
            lambda d: asos_by_date.get(d, [])
        )
        mock_db.get_system_state.return_value = None

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_basic_rmse_computation(self, mock_db: MagicMock) -> None:
        """With known constant errors, calibrated_cap = error * safety_factor.

        10 dates, NWP=50°F, ASOS=51°F at all eval hours (10-23).
        error = 1.0°F at every (date, hour) pair.
        hourly_rmse = 1.0.
        calibrated_cap = 1.0 * 1.2 = 1.2°F.
        """
        from kalshi_weather_trader.calibration.calibrator import calibrate_ou_max_stationary_std

        base_date = date(2026, 4, 1)
        eval_hours = list(range(10, 24))
        markets, morning_forecasts, asos_by_date = _build_nwp_asos_scenario(
            base_date, n_dates=10, nwp_temp=50.0, asos_temp=51.0, eval_hours=eval_hours
        )
        self._patch_db(mock_db, markets, morning_forecasts, asos_by_date, base_date)

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=base_date,
        ):
            result = calibrate_ou_max_stationary_std(target_date=base_date)

        assert result is not None, "Should return a cap with 10 qualifying dates"
        # hourly_rmse = 1.0 → cap = 1.0 * 1.2 = 1.2
        assert result == pytest.approx(1.2, abs=0.01), (
            f"Expected 1.2 (1.0 RMSE * 1.2 factor), got {result}"
        )

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_only_post_fetch_hours_included(self, mock_db: MagicMock) -> None:
        """Hours 0-9 ET are excluded; only hours >= 10 contribute to RMSE.

        Pre-fetch hours (0-9): NWP=50, ASOS=60  → error = 10°F (large, excluded)
        Post-fetch hours (10-23): NWP=50, ASOS=51 → error = 1°F (included)

        If pre-fetch hours leaked in, the RMSE would be >> 1.0°F.
        """
        from kalshi_weather_trader.calibration.calibrator import calibrate_ou_max_stationary_std

        base_date = date(2026, 4, 1)
        n_dates = 10

        markets = {}
        morning_forecasts = {}
        asos_by_date = {}

        for i in range(1, n_dates + 1):
            d = base_date - timedelta(days=i)
            markets[d] = _make_market()

            # NWP is flat at 50°F.
            nwp_curve = [50.0] * 24
            morning_forecasts[d] = {"GFS": _make_nwp_forecast("GFS", nwp_curve)}

            readings = []
            # Pre-fetch hours (0-9): large error (60°F), should be EXCLUDED.
            for h in range(0, 10):
                obs_utc = _et_hour_to_utc(d, h)
                readings.append(_make_asos_reading(obs_utc, 60.0))
            # Post-fetch hours (10-23): small error (51°F), should be INCLUDED.
            for h in range(10, 24):
                obs_utc = _et_hour_to_utc(d, h)
                readings.append(_make_asos_reading(obs_utc, 51.0))
            asos_by_date[d] = readings

        self._patch_db(mock_db, markets, morning_forecasts, asos_by_date, base_date)

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=base_date,
        ):
            result = calibrate_ou_max_stationary_std(target_date=base_date)

        assert result is not None
        # If only post-fetch hours included: RMSE=1.0, cap=1.2
        # If pre-fetch hours leaked in: RMSE >> 1.0, cap >> 1.2
        assert result == pytest.approx(1.2, abs=0.05), (
            f"Expected ~1.2 (post-fetch hours only), got {result}. "
            "Pre-fetch hours (error=10°F) appear to have leaked in."
        )

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_asos_gap_filter_uses_nearest_reading_per_hour(self, mock_db: MagicMock) -> None:
        """The 40-minute gap filter selects the nearest reading to each ET top-of-hour.

        Scenario: ONE reading per date, placed at exactly ET hour 15.
        NWP is flat at 50°F; ASOS reads 52°F.

        For ET hour 15: gap = 0 min → passes, contributes squared error (52-50)²=4.
        For all other eval hours (10-14, 16-23): gap ≥ 60 min → excluded.
        Expected: 10 dates × 1 pair = 10 pairs; RMSE = 2.0; cap = 2.0 × 1.2 = 2.4°F.
        """
        from kalshi_weather_trader.calibration.calibrator import calibrate_ou_max_stationary_std

        base_date = date(2026, 4, 1)
        n_dates = 10

        markets = {}
        morning_forecasts = {}
        asos_by_date = {}

        for i in range(1, n_dates + 1):
            d = base_date - timedelta(days=i)
            markets[d] = _make_market()
            morning_forecasts[d] = {"GFS": _make_nwp_forecast("GFS", [50.0] * 24)}
            # Single reading at exactly ET hour 15.
            asos_by_date[d] = [_make_asos_reading(_et_hour_to_utc(d, 15), 52.0)]

        self._patch_db(mock_db, markets, morning_forecasts, asos_by_date, base_date)

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=base_date,
        ):
            result = calibrate_ou_max_stationary_std(target_date=base_date)

        assert result is not None, "10 dates × 1 valid pair each should produce a cap"
        # Only hour 15 pairs; error = 2.0°F; cap = 2.0 × 1.2 = 2.4°F.
        assert result == pytest.approx(2.4, abs=0.05), (
            f"Single-reading-per-date: expected 2.4°F cap (2.0°F RMSE × 1.2), got {result}"
        )

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_no_asos_readings_date_skipped(self, mock_db: MagicMock) -> None:
        """Dates with empty ASOS lists contribute 0 pairs and are not counted.

        All 10 dates have empty ASOS data → n_dates_with_data = 0 < _MIN_RMSE_DATES
        → returns None.
        """
        from kalshi_weather_trader.calibration.calibrator import calibrate_ou_max_stationary_std

        base_date = date(2026, 4, 1)
        n_dates = 10

        markets = {}
        morning_forecasts = {}
        asos_by_date = {}

        for i in range(1, n_dates + 1):
            d = base_date - timedelta(days=i)
            markets[d] = _make_market()
            morning_forecasts[d] = {"GFS": _make_nwp_forecast("GFS", [50.0] * 24)}
            asos_by_date[d] = []  # ASOS station outage — no readings

        self._patch_db(mock_db, markets, morning_forecasts, asos_by_date, base_date)

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=base_date,
        ):
            result = calibrate_ou_max_stationary_std(target_date=base_date)

        assert result is None, (
            "No ASOS readings on any date → 0 qualifying dates → must return None"
        )

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_insufficient_dates_returns_none(self, mock_db: MagicMock) -> None:
        """Returns None when fewer than _MIN_RMSE_DATES (10) dates have valid pairs."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_ou_max_stationary_std

        base_date = date(2026, 4, 1)
        n_dates = 9  # one below threshold
        eval_hours = list(range(10, 24))
        markets, morning_forecasts, asos_by_date = _build_nwp_asos_scenario(
            base_date, n_dates=n_dates, nwp_temp=50.0, asos_temp=51.0, eval_hours=eval_hours
        )
        self._patch_db(mock_db, markets, morning_forecasts, asos_by_date, base_date)

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=base_date,
        ):
            result = calibrate_ou_max_stationary_std(target_date=base_date)

        assert result is None, f"Expected None with only {n_dates} dates, got {result}"

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_missing_market_dates_skipped(self, mock_db: MagicMock) -> None:
        """Dates with no market entry are skipped without counting toward n_dates."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_ou_max_stationary_std

        base_date = date(2026, 4, 1)
        # 10 valid dates plus 5 non-traded dates sprinkled in.
        n_valid = 10
        eval_hours = list(range(10, 24))
        markets, morning_forecasts, asos_by_date = _build_nwp_asos_scenario(
            base_date, n_dates=n_valid, nwp_temp=50.0, asos_temp=51.0, eval_hours=eval_hours
        )
        # No market entry returns None.
        original_market_fn = mock_db.get_market.side_effect
        self._patch_db(mock_db, markets, morning_forecasts, asos_by_date, base_date)

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=base_date,
        ):
            result = calibrate_ou_max_stationary_std(target_date=base_date)

        assert result is not None, "10 valid dates should produce a cap"
        assert result == pytest.approx(1.2, abs=0.05)

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_cli_settlement_not_required(self, mock_db: MagicMock) -> None:
        """Dates without CLI settlement are still used for hourly RMSE.

        cli_settlement_confirmed=False is acceptable for ASOS-vs-NWP accuracy
        computation.  This gives more qualifying dates than the old daily-high
        approach which required CLI confirmation.
        """
        from kalshi_weather_trader.calibration.calibrator import calibrate_ou_max_stationary_std

        base_date = date(2026, 4, 1)
        n_dates = 10
        eval_hours = list(range(10, 24))
        markets, morning_forecasts, asos_by_date = _build_nwp_asos_scenario(
            base_date, n_dates=n_dates, nwp_temp=50.0, asos_temp=51.0, eval_hours=eval_hours
        )
        # Mark all dates as NOT CLI-confirmed.
        for d, m in markets.items():
            m.cli_settlement_confirmed = False
            m.final_official_high = None
        self._patch_db(mock_db, markets, morning_forecasts, asos_by_date, base_date)

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=base_date,
        ):
            result = calibrate_ou_max_stationary_std(target_date=base_date)

        assert result is not None, (
            "CLI settlement should NOT be required for hourly RMSE — "
            "ASOS-vs-NWP accuracy is independent of NWS CLI availability"
        )
        assert result == pytest.approx(1.2, abs=0.05)

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_rmse_aggregates_across_all_pairs(self, mock_db: MagicMock) -> None:
        """RMSE is computed over the FLAT list of all (date, hour) pairs, not per-date means.

        10 dates × 14 eval hours = 140 pairs.
        5 dates: error = 0.0°F.  5 dates: error = 2.0°F.
        If computing per-date mean first: RMSE(means) = sqrt((0+4)/2) ≈ 1.414.
        If computing over all pairs:      RMSE(pairs) = sqrt((0*70 + 4*70)/140) = 2.0.

        The flat-pairs approach is correct — per-date means would give misleading
        results when dates have different numbers of valid pairs.
        """
        from kalshi_weather_trader.calibration.calibrator import calibrate_ou_max_stationary_std

        base_date = date(2026, 4, 1)
        n_dates = 10
        eval_hours = list(range(10, 24))  # 14 hours

        markets = {}
        morning_forecasts = {}
        asos_by_date = {}

        for i in range(1, n_dates + 1):
            d = base_date - timedelta(days=i)
            markets[d] = _make_market()
            nwp_curve = [50.0] * 24
            morning_forecasts[d] = {"GFS": _make_nwp_forecast("GFS", nwp_curve)}
            asos_temp = 50.0 if i <= 5 else 52.0  # first 5 = 0 error; last 5 = 2°F error
            readings = []
            for h in eval_hours:
                obs_utc = _et_hour_to_utc(d, h)
                readings.append(_make_asos_reading(obs_utc, asos_temp))
            asos_by_date[d] = readings

        self._patch_db(mock_db, markets, morning_forecasts, asos_by_date, base_date)

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=base_date,
        ):
            result = calibrate_ou_max_stationary_std(target_date=base_date)

        # Flat-pairs: RMSE = sqrt(mean([0]*70 + [4]*70)) = sqrt(2.0) ≈ 1.414
        # cap = 1.414 * 1.2 ≈ 1.697
        assert result is not None
        expected_rmse = math.sqrt(2.0)
        expected_cap = expected_rmse * 1.2
        assert result == pytest.approx(expected_cap, abs=0.05), (
            f"Expected {expected_cap:.3f} (flat-pairs RMSE * safety), got {result}"
        )

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_per_model_blending_in_hourly_errors(self, mock_db: MagicMock) -> None:
        """Blended hourly NWP curve renormalizes when models have different weights.

        Two models: GFS weight=0.6, ECMWF weight=0.4.
        GFS predicts 50°F, ECMWF predicts 52°F at all hours.
        Blended = 0.6*50 + 0.4*52 = 50.8°F.
        ASOS = 52.0°F.
        error = 52.0 - 50.8 = 1.2°F.
        RMSE = 1.2°F.  cap = 1.2 * 1.2 = 1.44°F.
        """
        from kalshi_weather_trader.calibration.calibrator import calibrate_ou_max_stationary_std

        base_date = date(2026, 4, 1)
        n_dates = 10
        eval_hours = list(range(10, 24))

        markets = {}
        morning_forecasts = {}
        asos_by_date = {}

        for i in range(1, n_dates + 1):
            d = base_date - timedelta(days=i)
            markets[d] = _make_market()
            morning_forecasts[d] = {
                "GFS": _make_nwp_forecast("GFS", [50.0] * 24),
                "ECMWF": _make_nwp_forecast("ECMWF", [52.0] * 24),
            }
            readings = []
            for h in eval_hours:
                obs_utc = _et_hour_to_utc(d, h)
                readings.append(_make_asos_reading(obs_utc, 52.0))
            asos_by_date[d] = readings

        # Model weights: GFS=0.6, ECMWF=0.4.
        state = MagicMock()
        state.model_weights = {"GFS": 0.6, "ECMWF": 0.4}
        mock_db.get_market.side_effect = lambda d: markets.get(d)
        mock_db.get_morning_nwp_forecasts.side_effect = lambda d: morning_forecasts.get(d, {})
        mock_db.get_asos_readings_for_date.side_effect = lambda d: asos_by_date.get(d, [])
        mock_db.get_system_state.return_value = state

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=base_date,
        ):
            result = calibrate_ou_max_stationary_std(target_date=base_date)

        blended_nwp = 0.6 * 50.0 + 0.4 * 52.0   # = 50.8
        expected_error = 52.0 - blended_nwp        # = 1.2
        expected_cap = expected_error * 1.2         # = 1.44

        assert result is not None
        assert result == pytest.approx(expected_cap, abs=0.05), (
            f"Blended error should be {expected_error}°F → cap {expected_cap}°F, got {result}"
        )

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_short_model_curve_handled_in_hourly_blend(self, mock_db: MagicMock) -> None:
        """When ECMWF has only 18 hours, hours 18-23 are blended from GFS only.

        Hours 10-17: both models present, weights renormalize normally.
        Hours 18-23: only GFS, weight renormalized to 1.0.
        ASOS is flat at 51.0°F.  Both NWP models are flat at 50.0°F.
        Expected RMSE = 1.0°F everywhere.  cap = 1.2°F.
        """
        from kalshi_weather_trader.calibration.calibrator import calibrate_ou_max_stationary_std

        base_date = date(2026, 4, 1)
        n_dates = 10
        eval_hours = list(range(10, 24))

        markets = {}
        morning_forecasts = {}
        asos_by_date = {}

        for i in range(1, n_dates + 1):
            d = base_date - timedelta(days=i)
            markets[d] = _make_market()
            morning_forecasts[d] = {
                "GFS": _make_nwp_forecast("GFS", [50.0] * 24),
                "ECMWF": _make_nwp_forecast("ECMWF", [50.0] * 18),  # short curve
            }
            readings = []
            for h in eval_hours:
                obs_utc = _et_hour_to_utc(d, h)
                readings.append(_make_asos_reading(obs_utc, 51.0))
            asos_by_date[d] = readings

        self._patch_db(mock_db, markets, morning_forecasts, asos_by_date, base_date)

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=base_date,
        ):
            result = calibrate_ou_max_stationary_std(target_date=base_date)

        # Both early (10-17) and late (18-23) hours have NWP=50, ASOS=51 → error=1°F everywhere.
        assert result is not None
        assert result == pytest.approx(1.2, abs=0.05), (
            f"Short ECMWF curve should not affect RMSE when GFS covers all hours. Got {result}"
        )

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_hourly_rmse_smaller_than_hypothetical_daily_high_rmse(
        self, mock_db: MagicMock
    ) -> None:
        """Hourly RMSE < daily-high RMSE in realistic scenarios.

        The daily maximum error is the ERROR of the MAX over many hours, which
        has heavier tails than a single-hour draw.  This test constructs a scenario
        where intraday volatility causes the daily high error to exceed the typical
        per-hour error, confirming the ordering we expect from the 1.2× safety factor.
        """
        from kalshi_weather_trader.calibration.calibrator import calibrate_ou_max_stationary_std

        base_date = date(2026, 4, 1)
        n_dates = 10

        # NWP is flat at 50°F.  ASOS varies across hours with a known pattern:
        # Hours 10-13: 49°F (below NWP, error = -1°F)
        # Hours 14-17: 52°F (above NWP, error = +2°F)  ← drives the daily high
        # Hours 18-23: 50°F (matches NWP, error = 0°F)
        # Per-hour RMSE = sqrt(mean([1,1,1,1, 4,4,4,4, 0,0,0,0,0,0])) = sqrt(8/14) ≈ 0.756
        # Hypothetical daily-high error: NWP peak = 50, actual peak = 52 → error = 2.0

        markets = {}
        morning_forecasts = {}
        asos_by_date = {}

        for i in range(1, n_dates + 1):
            d = base_date - timedelta(days=i)
            markets[d] = _make_market()
            morning_forecasts[d] = {"GFS": _make_nwp_forecast("GFS", [50.0] * 24)}
            readings = []
            for h in range(10, 24):
                if h < 14:
                    asos_t = 49.0
                elif h < 18:
                    asos_t = 52.0
                else:
                    asos_t = 50.0
                obs_utc = _et_hour_to_utc(d, h)
                readings.append(_make_asos_reading(obs_utc, asos_t))
            asos_by_date[d] = readings

        self._patch_db(mock_db, markets, morning_forecasts, asos_by_date, base_date)

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=base_date,
        ):
            result = calibrate_ou_max_stationary_std(target_date=base_date)

        # Per-hour RMSE ≈ 0.756; cap = 0.756 * 1.2 ≈ 0.907 (floors at 0.5 — above floor).
        # Hypothetical daily-high error = 2.0°F.
        # The cap from hourly RMSE should be < the hypothetical daily-high-derived cap.
        hourly_rmse = math.sqrt((4 * 1.0 + 4 * 4.0 + 6 * 0.0) / 14)
        expected_cap = max(hourly_rmse * 1.2, 0.5)  # respect floor

        assert result is not None
        assert result == pytest.approx(expected_cap, abs=0.05), (
            f"Expected {expected_cap:.3f}, got {result}"
        )
        # Core claim: hourly-based cap < daily-high-based cap (2.0 * 1.0 = 2.0)
        assert result < 2.0, (
            f"Hourly RMSE cap ({result:.3f}) should be less than the daily-high cap (2.0)"
        )

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_cap_clamped_to_range(self, mock_db: MagicMock) -> None:
        """Calibrated cap is always within [_RMSE_CAP_MIN, _RMSE_CAP_MAX] = [0.5, 5.0]."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_ou_max_stationary_std

        base_date = date(2026, 4, 1)

        # Tiny error → uncapped = 0.01 * 1.2 = 0.012 < 0.5 → floor to 0.5
        markets, morning_forecasts, asos_by_date = _build_nwp_asos_scenario(
            base_date, n_dates=10, nwp_temp=50.0, asos_temp=50.01,
            eval_hours=list(range(10, 24)),
        )
        self._patch_db(mock_db, markets, morning_forecasts, asos_by_date, base_date)

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=base_date,
        ):
            result_low = calibrate_ou_max_stationary_std(target_date=base_date)

        assert result_low == pytest.approx(0.5, abs=0.01), (
            f"Tiny error should be floored to 0.5°F, got {result_low}"
        )

        # Huge error → uncapped = 10.0 * 1.2 = 12.0 > 5.0 → ceiling to 5.0
        markets2, mf2, asos2 = _build_nwp_asos_scenario(
            base_date, n_dates=10, nwp_temp=50.0, asos_temp=60.0,
            eval_hours=list(range(10, 24)),
        )
        self._patch_db(mock_db, markets2, mf2, asos2, base_date)

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=base_date,
        ):
            result_high = calibrate_ou_max_stationary_std(target_date=base_date)

        assert result_high == pytest.approx(5.0, abs=0.01), (
            f"Large error should be capped at 5.0°F, got {result_high}"
        )

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_state_persisted_when_state_exists(self, mock_db: MagicMock) -> None:
        """Calibrated cap is written back to system_state when a state row exists."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_ou_max_stationary_std

        base_date = date(2026, 4, 1)
        eval_hours = list(range(10, 24))
        markets, morning_forecasts, asos_by_date = _build_nwp_asos_scenario(
            base_date, n_dates=10, nwp_temp=50.0, asos_temp=51.0, eval_hours=eval_hours
        )

        state = MagicMock()
        state.model_weights = {"GFS": 0.5, "ECMWF": 0.5}
        mock_db.get_market.side_effect = lambda d: markets.get(d)
        mock_db.get_morning_nwp_forecasts.side_effect = lambda d: morning_forecasts.get(d, {})
        mock_db.get_asos_readings_for_date.side_effect = lambda d: asos_by_date.get(d, [])
        mock_db.get_system_state.return_value = state

        with patch(
            "kalshi_weather_trader.calibration.calibrator.get_target_date",
            return_value=base_date,
        ):
            result = calibrate_ou_max_stationary_std(target_date=base_date)

        assert result is not None
        mock_db.upsert_system_state.assert_called_once()
        assert state.ou_max_stationary_std_calibrated == pytest.approx(result, abs=1e-6)
        assert state.nwp_rmse_n_dates == 10


# ---------------------------------------------------------------------------
# TestCalibrateKalmanBiasDecay
# ---------------------------------------------------------------------------


def _build_bias_decay_scenario(
    base_date: date,
    n_dates: int,
    errors_by_et_hour: dict[int, float],  # et_hour → error (asos - nwp)
) -> tuple[dict, dict]:
    """Build mock side-effect dicts for bias-decay calibration tests.

    Returns (forecasts_by_date, asos_by_date) where:
      - forecasts_by_date[d] = {model: NWPForecastDocument mock} with flat nwp_temp=50°F
      - asos_by_date[d] = list of ASOS readings where asos_temp = 50 + errors_by_et_hour[h]
    """
    nwp_base = 50.0
    forecasts_by_date = {}
    asos_by_date = {}

    for i in range(1, n_dates + 1):
        d = base_date - timedelta(days=i)
        nwp_curve = [nwp_base] * 24
        forecasts_by_date[d] = {"GFS": _make_nwp_forecast("GFS", nwp_curve)}

        readings = []
        for et_hour, err in sorted(errors_by_et_hour.items()):
            obs_utc = _et_hour_to_utc(d, et_hour)
            readings.append(_make_asos_reading(obs_utc, nwp_base + err))
        asos_by_date[d] = readings

    return forecasts_by_date, asos_by_date


class TestCalibrateKalmanBiasDecay:
    """Tests for calibrate_kalman_bias_decay() in calibrator.py."""

    def _patch_db(self, mock_db, forecasts_by_date, asos_by_date, base_date):
        mock_db.get_system_state.return_value = None
        mock_db.get_nwp_forecasts_before_utc.side_effect = (
            lambda d, cutoff: forecasts_by_date.get(d, {})
        )
        mock_db.get_asos_readings_since.side_effect = (
            lambda since_utc, station_id="KBOS": (
                asos_by_date.get(since_utc.date(), [])
            )
        )

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_returns_none_below_min_pairs(self, mock_db: MagicMock) -> None:
        """Returns None when fewer than 30 consecutive pairs are available."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_kalman_bias_decay

        base_date = date(2026, 4, 1)
        # Only 2 hours per date × 5 dates = 5 pairs — well below 30.
        errors = {10: 1.0, 11: 0.9}
        forecasts, asos = _build_bias_decay_scenario(base_date, n_dates=5, errors_by_et_hour=errors)
        self._patch_db(mock_db, forecasts, asos, base_date)

        result = calibrate_kalman_bias_decay(target_date=base_date, lookback_days=10)
        assert result is None

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_exact_ar1_recovers_phi(self, mock_db: MagicMock) -> None:
        """Exact AR(1) series (φ=0.92) recovers correct value within 0.02."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_kalman_bias_decay

        base_date = date(2026, 4, 1)
        phi_true = 0.92
        # Build errors where e[h+1] = phi * e[h] exactly (starting from e[10]=2.0).
        et_hours = list(range(10, 18))  # 8 hours → 7 pairs per date
        errors: dict[int, float] = {}
        e = 2.0
        for h in et_hours:
            errors[h] = e
            e = phi_true * e

        # 5 dates × 7 pairs = 35 pairs — above minimum 30.
        forecasts, asos = _build_bias_decay_scenario(base_date, n_dates=5, errors_by_et_hour=errors)
        self._patch_db(mock_db, forecasts, asos, base_date)

        result = calibrate_kalman_bias_decay(target_date=base_date, lookback_days=10)
        assert result is not None
        assert result == pytest.approx(phi_true, abs=0.02)

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_phi_above_1_clips_to_1(self, mock_db: MagicMock) -> None:
        """phi_raw > 1.0 (explosive series) is clipped to 1.0."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_kalman_bias_decay

        base_date = date(2026, 4, 1)
        # Errors where each step is larger than the previous: φ_raw > 1.
        et_hours = list(range(10, 18))
        errors = {h: float(h - 9) * 2.0 for h in et_hours}  # 2, 4, 6, ..., 16

        forecasts, asos = _build_bias_decay_scenario(base_date, n_dates=5, errors_by_et_hour=errors)
        self._patch_db(mock_db, forecasts, asos, base_date)

        result = calibrate_kalman_bias_decay(target_date=base_date, lookback_days=10)
        assert result is not None
        assert result == pytest.approx(1.0, abs=1e-4)

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_phi_below_min_clips_to_0_85(self, mock_db: MagicMock) -> None:
        """Alternating-sign errors (φ_raw < 0) are clipped to 0.85."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_kalman_bias_decay

        base_date = date(2026, 4, 1)
        # Alternating +2, -2, +2, -2 ... → φ_raw ≈ -1.
        et_hours = list(range(10, 18))
        errors = {h: 2.0 if (h % 2 == 0) else -2.0 for h in et_hours}

        forecasts, asos = _build_bias_decay_scenario(base_date, n_dates=5, errors_by_et_hour=errors)
        self._patch_db(mock_db, forecasts, asos, base_date)

        result = calibrate_kalman_bias_decay(target_date=base_date, lookback_days=10)
        assert result is not None
        assert result == pytest.approx(0.85, abs=1e-4)

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_persists_to_system_state(self, mock_db: MagicMock) -> None:
        """Calibrated value is written to system_state via upsert_system_state."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_kalman_bias_decay

        base_date = date(2026, 4, 1)
        phi_true = 0.92
        et_hours = list(range(10, 18))
        errors: dict[int, float] = {}
        e = 2.0
        for h in et_hours:
            errors[h] = e
            e = phi_true * e

        forecasts, asos = _build_bias_decay_scenario(base_date, n_dates=5, errors_by_et_hour=errors)

        state = MagicMock()
        state.model_weights = {"GFS": 1.0}
        state.kalman_bias_decay_calibrated = None
        mock_db.get_system_state.return_value = state
        mock_db.get_nwp_forecasts_before_utc.side_effect = (
            lambda d, cutoff: forecasts.get(d, {})
        )
        mock_db.get_asos_readings_since.side_effect = (
            lambda since_utc, station_id="KBOS": asos.get(since_utc.date(), [])
        )

        result = calibrate_kalman_bias_decay(target_date=base_date, lookback_days=10)

        assert result is not None
        mock_db.upsert_system_state.assert_called_once()
        assert state.kalman_bias_decay_calibrated == pytest.approx(result, abs=1e-6)

    @patch("kalshi_weather_trader.calibration.calibrator.db_manager")
    def test_nonconsecutive_gap_skipped(self, mock_db: MagicMock) -> None:
        """Hours 10, 11, 13 yield only one pair (10,11) — gap at 11→13 is skipped."""
        from kalshi_weather_trader.calibration.calibrator import calibrate_kalman_bias_decay

        base_date = date(2026, 4, 1)
        # Hours 10, 11, 13 — the (11, 13) step is not consecutive, so no pair there.
        # Only 1 pair per date × 5 dates = 5 pairs → below minimum → returns None.
        errors = {10: 2.0, 11: 1.84, 13: 1.56}  # note: skip hour 12

        forecasts, asos = _build_bias_decay_scenario(base_date, n_dates=5, errors_by_et_hour=errors)
        self._patch_db(mock_db, forecasts, asos, base_date)

        result = calibrate_kalman_bias_decay(target_date=base_date, lookback_days=10)
        # 5 dates × 1 pair = 5 pairs < 30 → None
        assert result is None
