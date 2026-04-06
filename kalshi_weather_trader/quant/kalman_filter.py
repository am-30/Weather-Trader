"""
2D Kalman Filter for temperature tracking and NWP bias estimation.

State vector: x = [[dT_t], [B_t]]
  - dT_t: temperature departure from NWP forecast at the current hour (°F)
  - B_t:  persistent NWP model bias estimate (°F)

Absolute temperature estimate: T_abs = nwp_current_hour + dT_t

With H = [[1, 1]], the observation z = asos_temp - nwp_current_hour = dT + B,
so both state components are directly coupled to every ASOS measurement.
This makes the bias B_t immediately observable (K[1] ≈ 0.42 from the first tick)
rather than frozen at its initial value as it would be with H = [[1, 0]].

Predict step (triggered by NWP hourly update):
    x = F @ x         (no control input — departure is stable across NWP hours)
    P = F @ P @ F.T + Q

Update step (triggered by 2-minute ASOS reading):
    z = asos_temp - nwp_current_hour
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

_H = np.array([[1.0, 1.0]])          # Observation matrix: z = dT + B, making bias observable
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
        initial_dt: float = 0.0,
        initial_bias: float = 0.0,
        initial_covariance: Optional[list[list[float]]] = None,
        nwp_current_hour: Optional[float] = None,
        q_temp: Optional[float] = None,
        q_bias: Optional[float] = None,
        r_obs: Optional[float] = None,
    ) -> None:
        """Initialise the Kalman filter.

        The state vector x = [[dT], [B]] where:
          dT: temperature departure from the NWP forecast at the current hour (°F).
              Typically 0.0 on a new trading day; the filter learns it from ASOS.
          B:  persistent NWP model bias (°F). Carried across days via warm-start.

        With H = [[1, 1]], every ASOS observation z = asos_temp - nwp_current_hour
        updates both dT and B, making bias observable from the very first tick.
        The absolute temperature estimate is nwp_current_hour + dT.

        Args:
            initial_dt:          Starting temperature departure from NWP (°F).
                                 Pass 0.0 on cold/warm start (most common case).
            initial_bias:        Starting model bias estimate. Defaults to 0.
            initial_covariance:  2×2 covariance matrix as nested list.
                                 Defaults to identity.
            nwp_current_hour:    NWP blended forecast for the current ET hour (°F).
                                 Used to compute absolute temperature as nwp + dT.
                                 If None, temperature falls back to dT alone until
                                 the first predict() or update() call provides it.
            q_temp:              Process noise for temperature departure.  Reads
                                 from settings if not provided.
            q_bias:              Process noise for bias.  Reads from settings.
            r_obs:               Observation noise.  Reads from settings.

        Returns:
            None

        Raises:
            ValueError: If initial_covariance is not a 2×2 matrix.
        """
        self.x = np.array([[float(initial_dt)], [float(initial_bias)]])
        self._nwp_current: Optional[float] = nwp_current_hour

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
            dT0=initial_dt,
            B0=initial_bias,
            nwp_current_hour=nwp_current_hour,
            T_abs=self.temperature,
            q_temp=q_t,
            q_bias=q_b,
            r_obs=r,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        """Current absolute temperature estimate in °F (nwp_current_hour + dT + B).

        With H = [[1, 1]], the innovation converges to zero when dT + B equals the
        full NWP residual (asos - nwp).  The absolute temperature is therefore:
            T_abs = nwp + dT + B = nwp + H @ x

        Returns 0 + dT + B if NWP has not yet been set (only possible in the first
        seconds of a cold start before the NWP job fires — self-corrects quickly).

        Returns:
            Float absolute temperature estimate in °F.
        """
        nwp = self._nwp_current if self._nwp_current is not None else 0.0
        return float(nwp + self.x[0, 0] + self.x[1, 0])

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
    # Covariance stabilisation
    # ------------------------------------------------------------------

    def _apply_covariance_cap(self) -> None:
        """Cap the covariance matrix diagonals and clip the off-diagonal for stability.

        Enforces settings.kalman_p_max_diagonal on P[0,0] and P[1,1], then clips
        P[0,1] to ±0.8 * sqrt(P[0,0]*P[1,1]) to maintain a valid (non-degenerate)
        correlation coefficient and restores symmetry via P[1,0] = P[0,1].

        Called after every update() step and after warm-start construction + each
        gap-inflation predict step in load_or_initialize_filter(). Prevents the
        near-perfect anti-correlation (P[0,1] ≈ −P[0,0]) and large diagonal values
        (P[0,0]≈40) observed after compounding warm-start × 1.2 inflation on top of
        an already-pathological prior covariance.

        Returns:
            None

        Raises:
            Nothing.
        """
        p_max = settings.kalman_p_max_diagonal
        self.P[0, 0] = min(self.P[0, 0], p_max)
        self.P[1, 1] = min(self.P[1, 1], p_max)
        off_diag_limit = 0.8 * (self.P[0, 0] * self.P[1, 1]) ** 0.5
        self.P[0, 1] = float(np.clip(self.P[0, 1], -off_diag_limit, off_diag_limit))
        self.P[1, 0] = self.P[0, 1]

    # ------------------------------------------------------------------
    # Core filter steps
    # ------------------------------------------------------------------

    def predict(self, nwp_at_current_hour: Optional[float] = None, dt: float = 1.0) -> None:
        """Predict step: propagate uncertainty forward and update NWP reference.

        Called once per NWP model update (hourly) and for gap inflation.

        With the departure-based state vector x = [dT, B]:
          - dT is stable across NWP hours (NWP shift ≈ true temp shift, so
            the departure is unchanged in expectation). F[0,0] = 1.0.
          - B decays at settings.kalman_bias_decay per hour, scaled by dt so
            the decay is correct regardless of call cadence. F[1,1] = decay^dt.

        A genuine persistent NWP bias (same sign all day) is sustained by
        repeated innovations pushing B up; transient intraday warming that
        NWP predicted correctly produces near-zero innovations once temperature
        stabilises, allowing the decay to pull B back toward zero.

        The NWP job calls predict(dt=1.0) once per hour.
        The ASOS job calls predict(dt≈0.033) for intra-tick gap (2 min).
        Gap inflation calls predict(dt=1.0) up to 12 times on restart.
        All three are handled correctly by F[1,1] = decay**dt.

        Args:
            nwp_at_current_hour: Blended NWP absolute forecast for the current
                                 ET hour (°F). If provided, updates the stored
                                 NWP reference so temperature property is current.
                                 Omit for gap-inflation-only calls (P inflated,
                                 NWP reference left unchanged).
            dt:                  Time step in hours. Defaults to 1.0.

        Returns:
            None

        Raises:
            Nothing.
        """
        if nwp_at_current_hour is not None:
            self._nwp_current = nwp_at_current_hour

        # Build dt-scaled state transition matrix.
        # dT row: identity (departure is stable across NWP hours).
        # B row: exponential decay at kalman_bias_decay per hour.
        _decay = settings.kalman_bias_decay ** dt
        _F_dyn = np.array([[1.0, 0.0], [0.0, _decay]])
        self.x = _F_dyn @ self.x
        self.P = _F_dyn @ self.P @ _F_dyn.T + self.Q

        logger.debug(
            "kalman.predict",
            nwp_at_current_hour=nwp_at_current_hour,
            dt=dt,
            T_abs=self.temperature,
            dT_pred=float(self.x[0, 0]),
            B_pred=self.bias,
        )

    def update(self, asos_temp: float, nwp_current_hour: Optional[float] = None) -> None:
        """Update step: correct state with an ASOS temperature observation.

        Called every 2 minutes when a new ASOS reading is available.
        Uses the Joseph form for P update to maintain positive-definiteness.

        The observation is the departure from NWP: z = asos_temp - nwp_current_hour.
        With H = [[1, 1]], the innovation y = z - (dT + B) drives updates to both
        the temperature departure (dT) and the persistent NWP bias (B).

        Args:
            asos_temp:         Observed ASOS temperature in °F.
            nwp_current_hour:  Blended NWP forecast for the current ET hour (°F).
                               If provided, updates the stored NWP reference before
                               computing the innovation. If None, uses the last known
                               NWP value (set by the most recent predict() call).
                               Falls back to 0.0 if NWP has never been set, which
                               only occurs in the first minutes of a cold start.

        Returns:
            None

        Raises:
            numpy.linalg.LinAlgError: If S is singular (should not occur with
                valid noise parameters — logged and state is left unchanged).
        """
        if nwp_current_hour is not None:
            self._nwp_current = nwp_current_hour

        nwp = self._nwp_current if self._nwp_current is not None else 0.0
        z = np.array([[float(asos_temp) - nwp]])

        # Innovation: z - H @ x = (asos - nwp) - (dT + B)
        y = z - _H @ self.x

        # Innovation covariance
        S = _H @ self.P @ _H.T + self.R

        # Innovation gate: reject implausible ASOS observations (outlier / corrupt data).
        # Mahalanobis distance = |innovation| / sqrt(S[0,0]).
        # Threshold of 4.0σ (not the more common 3.0) accommodates the 0.9°F ASOS sensor
        # quantisation step: with P capped at 2.0, a 1.8°F sensor step gives mahal≈1.3σ.
        # Only genuine data corruption (>~6°F with capped P) is rejected.
        _gate_sigma = settings.kalman_innovation_gate_sigma
        _s_val = float(S[0, 0])
        if _s_val > 0:
            _mahal = abs(float(y[0, 0])) / (_s_val ** 0.5)
            if _mahal > _gate_sigma:
                logger.warning(
                    "kalman.update.innovation_gate_rejected",
                    asos_temp=asos_temp,
                    nwp_current=round(nwp, 2),
                    innovation=round(float(y[0, 0]), 3),
                    mahalanobis=round(_mahal, 3),
                    gate_sigma=_gate_sigma,
                )
                return

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

        # Cap covariance diagonals to prevent explosion from pathological warm-starts
        # or compounding gap inflation. See _apply_covariance_cap() for details.
        self._apply_covariance_cap()

        logger.debug(
            "kalman.update",
            asos_temp=asos_temp,
            nwp_current=round(nwp, 2),
            z_departure=round(float(z[0, 0]), 3),
            innovation=float(y[0, 0]),
            T_est_abs=round(self.temperature, 2),
            dT_est=round(float(self.x[0, 0]), 3),
            B_est=round(self.bias, 3),
            K0=float(K[0, 0]),
            K1=float(K[1, 0]),
        )


# ---------------------------------------------------------------------------
# DB persistence helpers
# ---------------------------------------------------------------------------


def load_or_initialize_filter(
    target_date: date,
    current_asos_temp: float,
    nwp_at_load_time: Optional[float] = None,
) -> KalmanFilter:
    """Load the Kalman filter from the database or initialise a fresh one.

    If a system_state row exists for ``target_date``, restores the filter state
    by converting the stored absolute temperature back to a departure:
        dT = kalman_temp_estimate - nwp_at_load_time

    If no row exists, attempts a warm start from yesterday's converged bias.
    Falls back to a cold start (dT=0, B=0) if no prior state is available.

    Args:
        target_date:       The active trading date.
        current_asos_temp: Latest ASOS temperature (unused in favour of DB state
                           when a row exists; used only for legacy compat).
        nwp_at_load_time:  Blended NWP forecast for the current ET hour (°F).
                           Used to reconstruct the departure state dT from the
                           stored absolute temperature. If None, dT defaults to
                           0.0 (conservative; the next ASOS update re-learns it).

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
                nwp_current_hour=nwp_at_load_time,
                B0=initial_bias,
            )
            _kf_warm = KalmanFilter(
                initial_dt=0.0,
                initial_bias=initial_bias,
                initial_covariance=inflated_cov,
                nwp_current_hour=nwp_at_load_time,
            )
            # Cap covariance immediately after warm-start: inflated_cov = yesterday_P × 1.2
            # can still be pathologically large if yesterday_P was already out-of-range.
            _kf_warm._apply_covariance_cap()
            return _kf_warm
        else:
            logger.info(
                "kalman.load.cold_start",
                date=str(target_date),
                nwp_current_hour=nwp_at_load_time,
            )
            return KalmanFilter(
                initial_dt=0.0,
                initial_bias=0.0,
                nwp_current_hour=nwp_at_load_time,
            )

    # Sanity-check the stored estimate before restoring it.
    # If the stored temperature is implausible (>15°F from current ASOS), the
    # state is corrupted — a bad NWP=None load or a bug stored a nonsensical
    # value. Restoring it causes an innovation-gate deadlock: every ASOS update
    # has a huge Mahalanobis distance and is permanently rejected.
    # In this case reinitialize from the current ASOS reading instead of the
    # stored value so the filter can converge from a sane starting point.
    _SANITY_THRESHOLD_F = 15.0
    _stored_temp = state.kalman_temp_estimate
    if abs(_stored_temp - current_asos_temp) > _SANITY_THRESHOLD_F:
        logger.warning(
            "kalman.load.corrupt_state_detected",
            stored_temp=round(_stored_temp, 2),
            current_asos_temp=round(current_asos_temp, 2),
            diff=round(abs(_stored_temp - current_asos_temp), 2),
            threshold=_SANITY_THRESHOLD_F,
            action="reinitializing_from_asos",
        )
        # Reinitialize from current ASOS rather than restoring the bad state.
        # Preserve the bias estimate (likely still valid even if temp was wrong)
        # and use an inflated covariance to converge quickly.
        _kf_reset = KalmanFilter(
            initial_dt=current_asos_temp - (nwp_at_load_time or 0.0) - state.kalman_bias_estimate,
            initial_bias=state.kalman_bias_estimate,
            initial_covariance=[[2.0, 0.0], [0.0, 2.0]],  # start uncertain, converge fast
            nwp_current_hour=nwp_at_load_time,
        )
        _kf_reset._apply_covariance_cap()
        return _kf_reset

    # Reconstruct the departure state dT from the stored absolute temperature.
    # T_abs = NWP + dT + B  →  dT = T_abs - NWP - B
    if nwp_at_load_time is not None:
        initial_dt = state.kalman_temp_estimate - nwp_at_load_time - state.kalman_bias_estimate
    else:
        # NWP unavailable: anchor dT so that T_abs = current_asos_temp, not = bias_only.
        # Without this, temperature = 0 + 0 + bias (nonsense) gets synced to DB,
        # triggering the innovation-gate deadlock on the next load.
        initial_dt = current_asos_temp - state.kalman_bias_estimate
        logger.warning(
            "kalman.load.nwp_none_asos_anchor",
            stored_temp=round(state.kalman_temp_estimate, 2),
            current_asos_temp=round(current_asos_temp, 2),
            initial_dt=round(initial_dt, 3),
        )

    kf = KalmanFilter(
        initial_dt=initial_dt,
        initial_bias=state.kalman_bias_estimate,
        initial_covariance=state.kalman_covariance,
        nwp_current_hour=nwp_at_load_time,
    )
    # Cap immediately on restore: the stored covariance may be pathological if the
    # app crashed mid-inflation or if an old row has extreme values from prior bugs.
    kf._apply_covariance_cap()

    # Gap inflation: if the saved state is stale (app was down or restarting),
    # inject accumulated process noise so the Kalman gain is large enough to
    # correct a large temperature error quickly.  Without this, P collapses to
    # ~0.01 after many ASOS updates and K becomes ~0.024, making a 9°F
    # innovation move T by only ~0.2°F per tick.
    #
    # Each predict() call adds Q to P (P += Q, since F=I).
    # Capped at 12 hours to prevent runaway on very long outages.
    now_utc = datetime.now(timezone.utc)
    if state.last_updated_utc is not None:
        gap_hours = (now_utc - state.last_updated_utc).total_seconds() / 3600
        if gap_hours > 0.5:
            inflate_steps = min(int(gap_hours), 12)
            for _ in range(inflate_steps):
                kf.predict(dt=1.0)  # no NWP arg — just inflate P
                kf._apply_covariance_cap()  # prevent compounding inflation
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
        T_abs=state.kalman_temp_estimate,
        dT=round(initial_dt, 3),
        B=state.kalman_bias_estimate,
        nwp_at_load_time=nwp_at_load_time,
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
            # Merge Kalman state only — preserve ALL non-Kalman calibration fields
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
                persistence_filter_offset=existing.persistence_filter_offset,
                sigma_by_block=existing.sigma_by_block,
                theta_am=existing.theta_am,
                theta_pm=existing.theta_pm,
                ou_max_stationary_std_calibrated=existing.ou_max_stationary_std_calibrated,
                nwp_rmse_n_dates=existing.nwp_rmse_n_dates,
                last_calibrated_utc=existing.last_calibrated_utc,
                last_updated_utc=datetime.now(timezone.utc),
            )

        db_manager.upsert_system_state(doc)
        logger.debug(
            "kalman.sync_to_db.done",
            date=str(target_date),
            T_abs=kf.temperature,
            dT=round(float(kf.x[0, 0]), 3),
            B=kf.bias,
            nwp_current=kf._nwp_current,
        )
    except Exception as exc:
        logger.error("kalman.sync_to_db.failed", date=str(target_date), error=str(exc))
