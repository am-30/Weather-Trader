"""
All Pydantic v2 data-transfer models for the Kalshi weather trading system.

This is the single source of truth for every validated data structure.
No other module defines Pydantic models — they import from here.

Collections / tables represented:
- MarketDocument        → markets table
- ASOSReadingDocument   → asos_readings table
- NWPForecastDocument   → nwp_forecasts table
- SystemStateDocument   → system_state table
- IntradaySnapshotDocument → intraday_snapshots table
- TradeLogDocument      → trade_logs table
- MonteCarloResult      → (in-memory, not persisted directly)
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Utility validators
# ---------------------------------------------------------------------------


def _round1(v: Optional[float]) -> Optional[float]:
    """Round a temperature to 1 decimal place, or return None."""
    if v is None:
        return None
    return round(float(v), 1)


def _ensure_utc(v: datetime | str) -> datetime:
    """Ensure a datetime is timezone-aware UTC."""
    if isinstance(v, str):
        v = datetime.fromisoformat(v.replace("Z", "+00:00"))
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v


# ---------------------------------------------------------------------------
# MarketDocument
# ---------------------------------------------------------------------------


class MarketDocument(BaseModel):
    """Represents a row in the ``markets`` table.

    Attributes:
        target_date:          The calendar date this market covers.
        current_max_observed: Hard floor — highest ASOS reading seen today.
        market_status:        'open', 'closed', 'settled'.
        auto_trade_enabled:   Kill switch — False halts all order submission.
        final_official_high:  Official NWS high temperature once settled.
        last_updated_utc:     UTC timestamp of last modification.
    """

    target_date: date
    current_max_observed: Optional[float] = Field(default=None)
    market_status: str = Field(default="open")
    auto_trade_enabled: bool = Field(default=True)
    final_official_high: Optional[float] = Field(default=None)
    cli_settlement_confirmed: bool = Field(default=False)
    last_updated_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("current_max_observed", "final_official_high", mode="before")
    @classmethod
    def round_temp(cls, v: Optional[float]) -> Optional[float]:
        """Round temperature to 1 d.p.

        Args:
            v: Raw float or None.

        Returns:
            Rounded float or None.

        Raises:
            Nothing.
        """
        return _round1(v)

    @field_validator("last_updated_utc", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime | str) -> datetime:
        """Ensure timestamp is UTC-aware.

        Args:
            v: Datetime or ISO string.

        Returns:
            UTC-aware datetime.

        Raises:
            ValueError: If string cannot be parsed.
        """
        return _ensure_utc(v)


# ---------------------------------------------------------------------------
# ASOSReadingDocument
# ---------------------------------------------------------------------------


class ASOSReadingDocument(BaseModel):
    """Represents a row in the ``asos_readings`` table.

    Attributes:
        station_id:           ICAO station code (always 'KBOS').
        observation_time_utc: UTC time of the 5-minute ASOS observation.
        temperature_f:        Air temperature in Fahrenheit.
        dew_point_f:          Dew-point temperature, if available.
        wind_speed_mph:       Wind speed in mph, if available.
        raw_metar:            Raw METAR string, if available.
        inserted_at:          UTC time this row was inserted.
    """

    station_id: str = Field(default="KBOS")
    observation_time_utc: datetime
    temperature_f: float
    dew_point_f: Optional[float] = Field(default=None)
    wind_speed_mph: Optional[float] = Field(default=None, ge=0.0)
    raw_metar: Optional[str] = Field(default=None)
    inserted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("temperature_f", "dew_point_f", mode="before")
    @classmethod
    def round_temp(cls, v: Optional[float]) -> Optional[float]:
        """Round temperature to 1 d.p.

        Args:
            v: Raw float or None.

        Returns:
            Rounded float or None.

        Raises:
            Nothing.
        """
        return _round1(v)

    @field_validator("observation_time_utc", "inserted_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime | str) -> datetime:
        """Ensure timestamps are UTC-aware.

        Args:
            v: Datetime or ISO string.

        Returns:
            UTC-aware datetime.

        Raises:
            ValueError: If parsing fails.
        """
        return _ensure_utc(v)


# ---------------------------------------------------------------------------
# NWPForecastDocument
# ---------------------------------------------------------------------------


class NWPForecastDocument(BaseModel):
    """Represents a row in the ``nwp_forecasts`` table.

    Attributes:
        target_date:          The date this forecast covers.
        model_name:           'HRRR', 'GFS', or 'ECMWF'.
        fetched_at_utc:       UTC time the forecast was retrieved.
        hourly_temps:         24-element list of hourly Fahrenheit forecasts.
        predicted_daily_high: Max of hourly_temps.
    """

    target_date: date
    model_name: str = Field(..., pattern=r"^(HRRR|GFS|ECMWF|GFS_ENS|ECMWF_ENS)$")
    fetched_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    hourly_temps: list[float] = Field(..., min_length=1, max_length=48)
    predicted_daily_high: float
    mean_cloudcover_10_16: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Mean cloud cover (%) for ET hours 10-16, from NWP forecast. Null for ensemble rows.",
    )
    ensemble_highs: Optional[list[float]] = Field(
        default=None,
        description="Per-member predicted daily highs (°F). Populated only for GFS_ENS/ECMWF_ENS rows.",
    )
    ensemble_spread: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Std of ensemble_highs (°F). Populated only for GFS_ENS/ECMWF_ENS rows.",
    )

    @field_validator("fetched_at_utc", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime | str) -> datetime:
        """Ensure timestamp is UTC-aware.

        Args:
            v: Datetime or ISO string.

        Returns:
            UTC-aware datetime.

        Raises:
            ValueError: If parsing fails.
        """
        return _ensure_utc(v)

    @field_validator("predicted_daily_high", mode="before")
    @classmethod
    def round_high(cls, v: float) -> float:
        """Round predicted high to 1 d.p.

        Args:
            v: Raw float.

        Returns:
            Rounded float.

        Raises:
            TypeError: If v is not numeric.
        """
        return round(float(v), 1)

    @model_validator(mode="after")
    def set_predicted_high_from_temps(self) -> "NWPForecastDocument":
        """Compute predicted_daily_high from hourly_temps if not explicitly set.

        Args:
            None (uses self).

        Returns:
            Self with predicted_daily_high confirmed.

        Raises:
            Nothing.
        """
        if self.hourly_temps:
            computed = round(max(self.hourly_temps), 1)
            # Only override if predicted_daily_high wasn't manually set higher
            if self.predicted_daily_high < computed:
                self.predicted_daily_high = computed
        return self


# ---------------------------------------------------------------------------
# SystemStateDocument
# ---------------------------------------------------------------------------


class SystemStateDocument(BaseModel):
    """Represents a row in the ``system_state`` table.

    Attributes:
        target_date:               Active trading date.
        kalman_temp_estimate:      Kalman filter temperature estimate (°F).
        kalman_bias_estimate:      Kalman filter model-bias estimate.
        kalman_covariance:         2×2 covariance matrix as nested list.
        model_weights:             Dict mapping model name → weight (sums to 1).
        mu_drift:                  Global daily drift correction.
        theta_decay:               OU mean-reversion speed.
        sigma_volatility:          OU diffusion coefficient.
        morning_drift_adjustment:  Calibrated AM correction.
        afternoon_drift_adjustment: Calibrated PM correction.
        last_calibrated_utc:       When calibration last ran.
        last_updated_utc:          When this row was last modified.
    """

    target_date: date
    kalman_temp_estimate: float
    kalman_bias_estimate: float = Field(default=0.0)
    kalman_covariance: list[list[float]] = Field(
        default_factory=lambda: [[1.0, 0.0], [0.0, 1.0]]
    )
    model_weights: dict[str, float] = Field(
        default_factory=lambda: {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}
    )
    mu_drift: float = Field(default=0.0)
    theta_decay: float = Field(default=0.3, gt=0.0)
    sigma_volatility: float = Field(default=2.0, gt=0.0)
    morning_drift_adjustment: float = Field(default=0.0)
    afternoon_drift_adjustment: float = Field(default=0.0)
    persistence_filter_offset: float = Field(
        default=0.75,
        description="Calibrated ASOS-to-NWS daily max gap (°F). Applied to paths_max init in MC.",
    )
    sigma_by_block: Optional[dict[str, float]] = Field(
        default=None,
        description=(
            "Per-time-block OU sigma estimates. Keys are ET hour ranges "
            "('0-6', '6-10', '10-14', '14-18', '18-24'). None until enough "
            "history is available for block-level calibration (≥10 samples/block)."
        ),
    )
    theta_am: Optional[float] = Field(
        default=None,
        gt=0.0,
        description=(
            "OU mean-reversion speed for the AM regime (ET hours 6–13). "
            "Physical basis: morning solar heating is an active forcing; "
            "departures from NWP persist because cloud/albedo errors compound. "
            "Expected lower than theta_pm. None until 30+ days are available. "
            "Falls back to scalar theta_decay in run_simulation when None."
        ),
    )
    theta_pm: Optional[float] = Field(
        default=None,
        gt=0.0,
        description=(
            "OU mean-reversion speed for the PM regime (ET hours 13–20). "
            "Physical basis: convective mixing near peak temperature acts as a "
            "thermostat, pulling T back toward large-scale NWP prediction. "
            "Expected higher than theta_am. None until 30+ days are available. "
            "Falls back to scalar theta_decay in run_simulation when None."
        ),
    )
    ou_max_stationary_std_calibrated: Optional[float] = Field(
        default=None,
        gt=0.0,
        description=(
            "Calibrated cap on the OU stationary std (°F), computed from the empirical "
            "RMSE of morning NWP daily-high predictions against NWS CLI confirmed "
            "final_official_high. Value = blended_RMSE × 1.5 (safety factor). "
            "None until ≥10 qualifying dates are available. When non-None, passed to "
            "MCParams and used in run_simulation() instead of settings.ou_max_stationary_std."
        ),
    )
    nwp_rmse_n_dates: Optional[int] = Field(
        default=None,
        description="Number of qualifying dates used in ou_max_stationary_std calibration.",
    )
    kalman_bias_decay_calibrated: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "AR(1) calibrated per-hour bias decay factor for the Kalman filter. "
            "Estimated via Yule-Walker from consecutive intraday NWP error pairs. "
            "Clipped to [0.85, 1.0]. None until ≥30 consecutive pairs available."
        ),
    )
    nwp_daily_max_bias: float = Field(
        default=0.0,
        description=(
            "EMA of (actual_high − blended_predicted_high) over settled dates. "
            "Positive = NWP systematically underestimates the daily peak. "
            "Added to mu_t in run_simulation() as a persistent upward shift on "
            "the OU attractor. Corrects for afternoon-heating underestimation "
            "that the Kalman filter cannot detect from current-temperature residuals."
        ),
    )
    last_calibrated_utc: Optional[datetime] = Field(default=None)
    last_updated_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("last_calibrated_utc", "last_updated_utc", mode="before")
    @classmethod
    def ensure_utc(cls, v: Optional[datetime | str]) -> Optional[datetime]:
        """Ensure timestamps are UTC-aware.

        Args:
            v: Datetime, ISO string, or None.

        Returns:
            UTC-aware datetime or None.

        Raises:
            ValueError: If string cannot be parsed.
        """
        if v is None:
            return None
        return _ensure_utc(v)

    @field_validator("kalman_covariance")
    @classmethod
    def validate_covariance_shape(cls, v: list[list[float]]) -> list[list[float]]:
        """Ensure covariance is a 2×2 matrix.

        Args:
            v: Nested list representing the covariance matrix.

        Returns:
            Validated 2×2 nested list.

        Raises:
            ValueError: If the matrix is not 2×2.
        """
        if len(v) != 2 or any(len(row) != 2 for row in v):
            raise ValueError("kalman_covariance must be a 2×2 matrix")
        return v

    @field_validator("model_weights")
    @classmethod
    def validate_weights_sum(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure model weights are non-negative and sum to approximately 1.

        Args:
            v: Dict of model → weight.

        Returns:
            Validated weights dict.

        Raises:
            ValueError: If weights do not sum to ~1.0 or are negative.
        """
        if any(w < 0 for w in v.values()):
            raise ValueError("All model weights must be non-negative")
        total = sum(v.values())
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"Model weights must sum to 1.0, got {total:.4f}")
        return v


# ---------------------------------------------------------------------------
# IntradaySnapshotDocument
# ---------------------------------------------------------------------------


class IntradaySnapshotDocument(BaseModel):
    """Represents a row in the ``intraday_snapshots`` table.

    Attributes:
        target_date:             Active trading date.
        snapshot_time_utc:       UTC time of snapshot.
        snapshot_time_eastern:   HH:MM Eastern representation (display only).
        current_asos_temp_f:     Most recent ASOS temperature reading.
        current_max_observed_f:  Hard floor value at snapshot time.
        hrrr_predicted_high:     HRRR model predicted daily high.
        gfs_predicted_high:      GFS model predicted daily high.
        ecmwf_predicted_high:    ECMWF model predicted daily high.
        blended_predicted_high:  Weighted blend of available model highs.
        kalman_temp_estimate:    Kalman filter temperature estimate.
        kalman_bias_estimate:    Kalman filter bias estimate.
        kalshi_implied_prob_yes: Kalshi market implied YES probability.
        kalshi_bid:              Kalshi YES bid price (0–1 scale).
        kalshi_ask:              Kalshi YES ask price (0–1 scale).
        kalshi_strike:           Strike temperature in °F.
        model_fair_value_prob:   Monte Carlo estimated probability.
        model_edge:              Fair value minus market price.
        is_forced:               True if snapshot was manually triggered.
    """

    target_date: date
    snapshot_time_utc: datetime
    snapshot_time_eastern: str = Field(..., max_length=5, pattern=r"^\d{2}:\d{2}$")
    current_asos_temp_f: float
    current_max_observed_f: float
    hrrr_predicted_high: Optional[float] = Field(default=None)
    gfs_predicted_high: Optional[float] = Field(default=None)
    ecmwf_predicted_high: Optional[float] = Field(default=None)
    blended_predicted_high: float
    kalman_temp_estimate: float
    kalman_bias_estimate: float
    kalshi_implied_prob_yes: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    kalshi_bid: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    kalshi_ask: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    kalshi_strike: Optional[float] = Field(default=None)
    model_fair_value_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_edge: Optional[float] = Field(default=None)
    is_forced: bool = Field(default=False)

    @field_validator(
        "current_asos_temp_f",
        "current_max_observed_f",
        "hrrr_predicted_high",
        "gfs_predicted_high",
        "ecmwf_predicted_high",
        "blended_predicted_high",
        mode="before",
    )
    @classmethod
    def round_temp(cls, v: Optional[float]) -> Optional[float]:
        """Round temperature to 1 d.p.

        Args:
            v: Raw float or None.

        Returns:
            Rounded float or None.

        Raises:
            Nothing.
        """
        return _round1(v)

    @field_validator("snapshot_time_utc", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime | str) -> datetime:
        """Ensure snapshot timestamp is UTC-aware.

        Args:
            v: Datetime or ISO string.

        Returns:
            UTC-aware datetime.

        Raises:
            ValueError: If parsing fails.
        """
        return _ensure_utc(v)


# ---------------------------------------------------------------------------
# TradeLogDocument
# ---------------------------------------------------------------------------


class TradeLogDocument(BaseModel):
    """Represents a row in the ``trade_logs`` table.

    Attributes:
        trade_id:            UUID primary key (auto-generated).
        target_date:         Trading date.
        executed_at_utc:     UTC time the trade decision was made.
        market_ticker:       Kalshi market ticker string.
        action:              'BUY_YES' or 'BUY_NO'.
        kalshi_strike:       Strike temperature integer.
        contracts:           Number of contracts traded.
        price_cents:         Fill price in cents (1–99).
        fair_value_prob:     Model's estimated probability.
        kalshi_implied_prob: Market's implied probability.
        edge_at_execution:   fair_value_prob minus market probability.
        kelly_fraction:      Computed Kelly sizing fraction.
        dry_run:             True if no real order was submitted.
        order_id:            Kalshi-assigned order ID (if not dry run).
        status:              'filled', 'pending', 'cancelled', 'dry_run'.
        notes:               Free-text log notes.
    """

    trade_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_date: date
    executed_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    market_ticker: str
    action: str = Field(..., pattern=r"^(BUY_YES|BUY_NO)$")
    kalshi_strike: float
    contracts: int = Field(..., ge=1)
    price_cents: int = Field(..., ge=1, le=99)
    fair_value_prob: float = Field(..., ge=0.0, le=1.0)
    kalshi_implied_prob: float = Field(..., ge=0.0, le=1.0)
    edge_at_execution: float
    kelly_fraction: Optional[float] = Field(default=None, ge=0.0)
    dry_run: bool = Field(default=True)
    order_id: Optional[str] = Field(default=None)
    status: str = Field(default="pending")
    notes: Optional[str] = Field(default=None)
    inserted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("executed_at_utc", "inserted_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime | str) -> datetime:
        """Ensure timestamps are UTC-aware.

        Args:
            v: Datetime or ISO string.

        Returns:
            UTC-aware datetime.

        Raises:
            ValueError: If parsing fails.
        """
        return _ensure_utc(v)


# ---------------------------------------------------------------------------
# MonteCarloResult
# ---------------------------------------------------------------------------


class MonteCarloResult(BaseModel):
    """In-memory result from one Monte Carlo pricing run.

    Not persisted directly — key fields are stored in intraday_snapshots.

    Attributes:
        target_date:           Date the simulation covers.
        computed_at_utc:       UTC time the simulation was run.
        n_paths:               Number of OU simulation paths.
        n_steps:               Number of time steps per path.
        hard_floor:            Minimum max temperature enforced.
        probabilities:         Dict mapping strike → P(max >= strike).
        percentile_10:         10th percentile of final max distribution.
        percentile_25:         25th percentile.
        percentile_50:         Median (50th percentile).
        percentile_75:         75th percentile.
        percentile_90:         90th percentile.
        mean_max:              Mean of simulated maximum temperatures.
        std_max:               Std dev of simulated maximum temperatures.
    """

    target_date: date
    computed_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    n_paths: int
    n_steps: int
    hard_floor: float
    probabilities: dict[float, float]  # strike (float °F) → probability
    percentile_10: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_90: float
    mean_max: float
    std_max: float

    @field_validator("computed_at_utc", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime | str) -> datetime:
        """Ensure timestamp is UTC-aware.

        Args:
            v: Datetime or ISO string.

        Returns:
            UTC-aware datetime.

        Raises:
            ValueError: If parsing fails.
        """
        return _ensure_utc(v)


# ---------------------------------------------------------------------------
# PaperTradeDocument
# ---------------------------------------------------------------------------


class PaperTradeDocument(BaseModel):
    """Represents a row in the ``paper_trade_positions`` table.

    Tracks simulated trades that mirror intended live trading rules:
    - Entry at 10 AM ET when ask < 50¢ and model has positive edge
    - Limit-sell exit at 75¢ (if market bid reaches 75¢)
    - Otherwise settles based on official NWS daily high

    Attributes:
        position_id:         UUID primary key (auto-generated).
        target_date:         Trading date.
        market_ticker:       Kalshi market ticker string.
        action:              'BUY_YES' or 'BUY_NO'.
        kalshi_strike:       Strike temperature (°F).
        entry_at_utc:        UTC time the simulated entry was recorded.
        entry_price_cents:   Market ask price at entry (1–99 cents).
        contracts:           Number of contracts.
        cost_usd:            Total cost in dollars (entry_price_cents/100 * contracts).
        fair_value_prob:     Model's estimated probability at entry.
        edge_at_entry:       Model prob minus market ask decimal.
        kelly_fraction:      Kelly fraction if computed (None for flat mode).
        budget_mode:         'kelly' or 'flat'.
        bankroll_at_entry:   Bankroll before this bet (Kelly mode only).
        status:              'open' | 'limit_sell_closed' | 'settled_win' | 'settled_loss'.
        exit_at_utc:         UTC time position was closed.
        exit_price_cents:    75 for limit sell; 100 (win) or 0 (loss) at settlement.
        pnl_cents:           Net profit/loss in total cents.
        pnl_usd:             Net profit/loss in dollars.
        official_high_f:     Official NWS daily high used for settlement.
        settlement_win:      True if settled as a win.
        inserted_at:         Row insertion timestamp.
    """

    position_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_date: date
    market_ticker: str
    action: str = Field(..., pattern=r"^(BUY_YES|BUY_NO)$")
    kalshi_strike: float

    entry_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    entry_price_cents: int = Field(..., ge=1, le=99)
    contracts: int = Field(..., ge=1)
    cost_usd: float = Field(..., ge=0.0)
    fair_value_prob: float = Field(..., ge=0.0, le=1.0)
    edge_at_entry: float
    kelly_fraction: Optional[float] = Field(default=None, ge=0.0)
    budget_mode: str = Field(..., pattern=r"^(kelly|flat)$")
    bankroll_at_entry: Optional[float] = Field(default=None)

    status: str = Field(default="open")
    exit_at_utc: Optional[datetime] = Field(default=None)
    exit_price_cents: Optional[int] = Field(default=None, ge=0, le=100)
    pnl_cents: Optional[float] = Field(default=None)
    pnl_usd: Optional[float] = Field(default=None)

    official_high_f: Optional[float] = Field(default=None)
    settlement_win: Optional[bool] = Field(default=None)
    inserted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("entry_at_utc", "inserted_at", mode="before")
    @classmethod
    def ensure_entry_utc(cls, v: datetime | str) -> datetime:
        """Ensure entry/insert timestamps are UTC-aware."""
        return _ensure_utc(v)

    @field_validator("exit_at_utc", mode="before")
    @classmethod
    def ensure_exit_utc(cls, v: Optional[datetime | str]) -> Optional[datetime]:
        """Ensure exit timestamp is UTC-aware if present."""
        if v is None:
            return None
        return _ensure_utc(v)
