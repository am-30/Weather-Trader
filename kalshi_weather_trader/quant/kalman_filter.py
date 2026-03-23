"""
2D Kalman Filter for temperature tracking and NWP bias estimation.

State vector: x = [[T_t], [B_t]]
  - T_t: current true temperature estimate (°F)
  - B_t: NWP model bias estimate (°F)

Predict step (triggered by NWP hourly delta):
    x = F @ x + u
    P = F @ P @ F.T + Q

Update step (triggered by 5-minute ASOS reading):
    y = z - H @ x            (innovation)
    S = H @ P @ H.T + R      (innovation covariance)
    K = P @ H.T @ inv(S)     (Kalman gain)
    x = x + K @ y
    P = (I - K@H) @ P @ (I - K@H).T + K @ R @ K.T   ← Joseph form

The Joseph form is used for numerical stability; it keeps P positive-definite
even after hundreds of update cycles.

All covariance noise parameters are read from config/settings.py.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import structlog

from kalshi_weather_trader.config.settings import settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_F = np.eye(2, dtype=float)          # State transition matrix (identity)
_H = np.array([[1.0, 0.0]])          # Observation matrix (we observe T, not B)
_I = np.eye(2, dtype=float)          # Identity matrix


# ---------------------------------------------------------------------------
# KalmanFilter class
# ---------------------------------------------------------------------------


class KalmanFilter:
    """2D Kalman filter tracking temperature (T) and model bias (B).

    Attributes:
        x: State vector [[T_t], [B_t]] (2×1 numpy array).
        P: 2×2 covariance matrix (positive-definite).
        Q: 2×2 process noise matrix (diagonal).
        R: 1×1 observation noise matrix.
    """

    def __init__(
        self,
        initial_temp: float,
        initial_bias: float = 0.0,
        initial_covariance: Optional[list[list[float]]] = None,
        q_temp: Optional[float] = None,
        q_bias: Optional[float] = None,
        r_obs: Optional[float] = None,
    ) -> None:
        """Initialise the Kalman filter.

        Args:
            initial_temp:        Starting temperature estimate in °F.
            initial_bias:        Starting model bias estimate. Defaults to 0.
            initial_covariance:  2×2 covariance matrix as nested list.
                                 Defaults to identity.
            q_temp:              Process noise for temperature.  Reads from
                                 settings if not provided.
            q_bias:              Process noise for bias.  Reads from settings.
            r_obs:               Observation noise.  Reads from settings.

        Returns:
            None

        Raises:
            ValueError: If initial_covariance is not a 2×2 matrix.
        """
        self.x = np.array([[float(initial_temp)], [float(initial_bias)]])

        if initial_covariance is not None:
            P = np.array(initial_covariance, dtype=float)
            if P.shape != (2, 2):
                raise ValueError("initial_covariance must be 2×2")
            self.P = P
        else:
            self.P = np.eye(2, dtype=float)

        q_t = q_temp if q_temp is not None else settings.kalman_q_temp
        q_b = q_bias if q_bias is not None else settings.kalman_q_bias
        r = r_obs if r_obs is not None else settings.kalman_r_obs

        self.Q = np.array([[q_t, 0.0], [0.0, q_b]])
        self.R = np.array([[r]])

        logger.debug(
            "kalman.init",
            T0=initial_temp,
            B0=initial_bias,
            q_temp=q_t,
            q_bias=q_b,
            r_obs=r,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        """Current temperature estimate in °F.

        Returns:
            Float temperature estimate.
        """
        return float(self.x[0, 0])

    @property
    def bias(self) -> float:
        """Current model bias estimate in °F.

        Returns:
            Float bias estimate.
        """
        return float(self.x[1, 0])

    @property
    def covariance_as_list(self) -> list[list[float]]:
        """Return the covariance matrix as a JSON-serialisable nested list.

        Returns:
            2×2 nested list of floats.
        """
        return self.P.tolist()

    # ------------------------------------------------------------------
    # Core filter steps
    # ------------------------------------------------------------------

    def predict(self, nwp_delta: float, dt: float = 1.0) -> None:
        """Predict step: propagate state forward using NWP temperature delta.

        Called once per NWP model update (hourly).

        The control input u shifts the temperature estimate by the NWP-predicted
        hourly change:
            u = [[nwp_delta * dt], [0.0]]

        Args:
            nwp_delta: Expected temperature change from NWP model (°F/hour).
            dt:        Time step in hours. Defaults to 1.0.

        Returns:
            None

        Raises:
            Nothing.
        """
        # Clamp to guard against corrupt NWP responses (e.g. a 10°F/hr model spike
        # would otherwise shift the temperature estimate by the full amount unchecked).
        # 5°F/hr is a physically generous ceiling — typical Boston diurnal ramp is
        # 1–3°F/hr. On clean NWP data this clamp will never fire.
        max_delta = settings.kalman_max_nwp_delta
        if abs(nwp_delta) > max_delta:
            clamped = max(-max_delta, min(max_delta, nwp_delta))
            logger.warning(
                "kalman.predict.delta_clamped",
                raw_delta=round(nwp_delta, 3),
                clamped_to=clamped,
            )
            nwp_delta = clamped

        u = np.array([[nwp_delta * dt], [0.0]])
        self.x = _F @ self.x + u
        self.P = _F @ self.P @ _F.T + self.Q

        logger.debug(
            "kalman.predict",
            nwp_delta=nwp_delta,
            dt=dt,
            T_pred=self.temperature,
            B_pred=self.bias,
        )

    def update(self, asos_temp: float) -> None:
        """Update step: correct state with an ASOS temperature observation.

        Called every 5 minutes when a new ASOS reading is available.
        Uses the Joseph form for P update to maintain positive-definiteness.

        Args:
            asos_temp: Observed ASOS temperature in °F.

        Returns:
            None

        Raises:
            numpy.linalg.LinAlgError: If S is singular (should not occur with
                valid noise parameters — logged and state is left unchanged).
        """
        z = np.array([[float(asos_temp)]])

        # Innovation
        y = z - _H @ self.x

        # Innovation covariance
        S = _H @ self.P @ _H.T + self.R

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError as exc:
            logger.error(
                "kalman.update.singular_S",
                S=S.tolist(),
                error=str(exc),
            )
            return

        # Kalman gain
        K = self.P @ _H.T @ S_inv

        # State update
        self.x = self.x + K @ y

        # Joseph form covariance update (numerically stable)
        I_KH = _I - K @ _H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        logger.debug(
            "kalman.update",
            asos_temp=asos_temp,
            innovation=float(y[0, 0]),
            T_est=self.temperature,
            B_est=self.bias,
            K0=float(K[0, 0]),
            K1=float(K[1, 0]),
        )


# ---------------------------------------------------------------------------
# DB persistence helpers
# ---------------------------------------------------------------------------


def load_or_initialize_filter(
    target_date: date,
    current_asos_temp: float,
) -> KalmanFilter:
    """Load the Kalman filter from the database or initialise a fresh one.

    If a system_state row exists for ``target_date``, restores the filter state.
    Otherwise, creates a new filter starting at ``current_asos_temp`` with
    zero bias and identity covariance.

    Args:
        target_date:       The active trading date.
        current_asos_temp: Latest ASOS temperature used for cold-start init.

    Returns:
        Initialised ``KalmanFilter`` ready for predict/update cycles.

    Raises:
        Nothing — falls back to cold-start on any database error.
    """
    from kalshi_weather_trader.db import db_manager

    try:
        state = db_manager.get_system_state(target_date)
    except Exception as exc:
        logger.warning("kalman.load.db_read_failed", error=str(exc))
        state = None

    if state is None:
        yesterday = target_date - timedelta(days=1)
        yesterday_state = None
        try:
            yesterday_state = db_manager.get_system_state(yesterday)
        except Exception as exc:
            logger.warning("kalman.load.yesterday_read_failed", error=str(exc))

        if yesterday_state is not None:
            initial_bias = yesterday_state.kalman_bias_estimate
            inflated_cov = [
                [yesterday_state.kalman_covariance[i][j] * 1.2 for j in range(2)]
                for i in range(2)
            ]
            logger.info(
                "kalman.load.warm_start",
                date=str(target_date),
                yesterday=str(yesterday),
                T0=current_asos_temp,
                B0=initial_bias,
            )
            return KalmanFilter(
                initial_temp=current_asos_temp,
                initial_bias=initial_bias,
                initial_covariance=inflated_cov,
            )
        else:
            logger.info("kalman.load.cold_start", date=str(target_date), T0=current_asos_temp)
            return KalmanFilter(initial_temp=current_asos_temp, initial_bias=0.0)

    kf = KalmanFilter(
        initial_temp=state.kalman_temp_estimate,
        initial_bias=state.kalman_bias_estimate,
        initial_covariance=state.kalman_covariance,
    )

    # Gap inflation: if the saved state is stale (app was down or restarting),
    # inject accumulated process noise so the Kalman gain is large enough to
    # correct a large temperature error quickly.  Without this, P collapses to
    # ~0.01 after many ASOS updates and K becomes ~0.024, making a 9°F
    # innovation move T by only ~0.2°F per tick.
    #
    # Each predict(nwp_delta=0) call adds Q to P (P += Q, since F=I and u=0).
    # Capped at 12 hours to prevent runaway on very long outages.
    now_utc = datetime.now(timezone.utc)
    if state.last_updated_utc is not None:
        gap_hours = (now_utc - state.last_updated_utc).total_seconds() / 3600
        if gap_hours > 0.5:
            inflate_steps = min(int(gap_hours), 12)
            for _ in range(inflate_steps):
                kf.predict(nwp_delta=0.0, dt=1.0)
            logger.info(
                "kalman.load.gap_inflation",
                date=str(target_date),
                gap_hours=round(gap_hours, 2),
                inflate_steps=inflate_steps,
                P_00=round(kf.P[0, 0], 4),
            )

    logger.info(
        "kalman.load.restored",
        date=str(target_date),
        T=state.kalman_temp_estimate,
        B=state.kalman_bias_estimate,
    )
    return kf


def sync_filter_to_db(kf: KalmanFilter, target_date: date) -> None:
    """Persist the current Kalman filter state to the database.

    Merges only Kalman-specific fields into the system_state row.
    Does NOT overwrite model_weights, drift adjustments, or calibration params.

    Args:
        kf:          The Kalman filter whose state to persist.
        target_date: The active trading date.

    Returns:
        None

    Raises:
        Nothing — errors are logged.
    """
    from kalshi_weather_trader.config.settings import settings as cfg
    from kalshi_weather_trader.db import db_manager
    from kalshi_weather_trader.db.schemas import SystemStateDocument
    from datetime import datetime, timezone

    try:
        # Load existing state to preserve non-Kalman fields
        existing = db_manager.get_system_state(target_date)

        if existing is None:
            # Bootstrap a new system_state row
            doc = SystemStateDocument(
                target_date=target_date,
                kalman_temp_estimate=kf.temperature,
                kalman_bias_estimate=kf.bias,
                kalman_covariance=kf.covariance_as_list,
                model_weights={"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2},
                theta_decay=cfg.ou_theta,
                sigma_volatility=cfg.ou_sigma,
                last_updated_utc=datetime.now(timezone.utc),
            )
        else:
            # Merge Kalman state only
            doc = SystemStateDocument(
                target_date=target_date,
                kalman_temp_estimate=kf.temperature,
                kalman_bias_estimate=kf.bias,
                kalman_covariance=kf.covariance_as_list,
                model_weights=existing.model_weights,
                mu_drift=existing.mu_drift,
                theta_decay=existing.theta_decay,
                sigma_volatility=existing.sigma_volatility,
                morning_drift_adjustment=existing.morning_drift_adjustment,
                afternoon_drift_adjustment=existing.afternoon_drift_adjustment,
                last_calibrated_utc=existing.last_calibrated_utc,
                last_updated_utc=datetime.now(timezone.utc),
            )

        db_manager.upsert_system_state(doc)
        logger.debug(
            "kalman.sync_to_db.done",
            date=str(target_date),
            T=kf.temperature,
            B=kf.bias,
        )
    except Exception as exc:
        logger.error("kalman.sync_to_db.failed", date=str(target_date), error=str(exc))
