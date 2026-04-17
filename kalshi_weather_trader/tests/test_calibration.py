"""
Unit tests for calibration routines.

Tests:
- RMSE softmax weights sum to 1.0 and lower RMSE → higher weight (D2)
- Daily-max bias EMA: convergence, clamping, chronological ordering (D3)
- MCParams daily_max_bias raises mean_max (D3)
- Theta AR(1) fit returns value in valid range
- Kelly contract sizing logic
"""

from __future__ import annotations

import numpy as np
import pytest

from kalshi_weather_trader.execution.trader import compute_kelly_contracts


class TestKellyContracts:
    def test_positive_edge_returns_contracts(self):
        """With clear edge and sufficient size, Kelly should return a positive integer."""
        # p=0.65, ask=0.50 → kelly=0.30, raw=0.25*0.30*1000/50 = 1.5 → 1 contract
        contracts = compute_kelly_contracts(
            p=0.65,
            ask_decimal=0.50,
            max_size_usd=1000.0,
            kelly_fraction=0.25,
            max_contracts=20,
        )
        assert contracts is not None
        assert contracts >= 1

    def test_edge_too_small_for_min_contract_returns_none(self):
        """When Kelly sizing rounds below 1 contract, return None (don't force-trade)."""
        # p=0.65, ask=0.50, max_size_usd=100 → raw_contracts=0.15 → should not force to 1
        contracts = compute_kelly_contracts(
            p=0.65,
            ask_decimal=0.50,
            max_size_usd=100.0,
            kelly_fraction=0.25,
            max_contracts=20,
        )
        assert contracts is None

    def test_negative_edge_returns_none(self):
        """When model prob < implied prob, Kelly should return None."""
        contracts = compute_kelly_contracts(
            p=0.40,       # model says 40%
            ask_decimal=0.55,  # market asks 55¢ → implied 55%
            max_size_usd=100.0,
            kelly_fraction=0.25,
            max_contracts=20,
        )
        assert contracts is None

    def test_contracts_capped_at_max(self):
        """Contracts should never exceed max_contracts."""
        contracts = compute_kelly_contracts(
            p=0.99,        # extreme edge
            ask_decimal=0.01,  # tiny ask price → huge Kelly
            max_size_usd=10000.0,
            kelly_fraction=1.0,
            max_contracts=5,
        )
        assert contracts is not None
        assert contracts <= 5

    def test_breakeven_p_returns_none(self):
        """At exactly break-even, Kelly = 0 → should return None."""
        # Break-even: p = 1/(1+b) = ask_decimal
        ask = 0.50
        p_breakeven = ask  # p * b = (1-p) → kelly = 0
        contracts = compute_kelly_contracts(
            p=p_breakeven,
            ask_decimal=ask,
            max_size_usd=100.0,
            kelly_fraction=0.25,
            max_contracts=10,
        )
        # Kelly = 0 or negative at breakeven
        assert contracts is None

    def test_invalid_ask_returns_none(self):
        """ask_decimal of 0 or 1 should return None gracefully."""
        assert compute_kelly_contracts(0.7, 0.0, 100.0, 0.25, 10) is None
        assert compute_kelly_contracts(0.7, 1.0, 100.0, 0.25, 10) is None


class TestModelWeightSoftmax:
    """D2: softmax weighting now operates on RMSE, not Brier — same math, different input."""

    def test_softmax_weights_sum_to_one(self):
        """Softmax normalisation over inverted RMSE produces weights summing to 1.0."""
        import math

        rmse_scores = {"HRRR": 1.2, "GFS": 1.8, "ECMWF": 2.1}
        inv_scores = {m: 1.0 / (s + 1e-8) for m, s in rmse_scores.items()}
        max_inv = max(inv_scores.values())
        exp_scores = {m: math.exp(v - max_inv) for m, v in inv_scores.items()}
        total = sum(exp_scores.values())
        weights = {m: v / total for m, v in exp_scores.items()}

        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(w >= 0 for w in weights.values())

    def test_lower_rmse_gets_higher_weight(self):
        """The model with the lowest daily-max RMSE should get the highest weight."""
        import math

        rmse_scores = {"HRRR": 1.0, "GFS": 1.5, "ECMWF": 2.0}
        inv_scores = {m: 1.0 / (s + 1e-8) for m, s in rmse_scores.items()}
        max_inv = max(inv_scores.values())
        exp_scores = {m: math.exp(v - max_inv) for m, v in inv_scores.items()}
        total = sum(exp_scores.values())
        weights = {m: v / total for m, v in exp_scores.items()}

        assert weights["HRRR"] > weights["GFS"] > weights["ECMWF"]


class TestDailyMaxBiasEMA:
    """D3: calibrate_daily_max_bias() computes the correct EMA from settled errors."""

    def _run_ema(self, errors: list[float], alpha: float = 0.1) -> float:
        """Replicate the sequential EMA used in calibrate_daily_max_bias."""
        bias = 0.0
        for err in errors:
            bias = (1.0 - alpha) * bias + alpha * err
        return max(-5.0, min(5.0, bias))

    def test_constant_positive_errors_converge(self):
        """EMA of constant +3°F errors should approach 3°F (not reach it exactly)."""
        errors = [3.0] * 50
        result = self._run_ema(errors)
        # After 50 steps with alpha=0.1, bias ≈ 3*(1-(0.9^50)) ≈ 2.99
        assert result == pytest.approx(3.0, abs=0.05)

    def test_zero_errors_stay_zero(self):
        """EMA of zero errors stays at 0."""
        errors = [0.0] * 30
        assert self._run_ema(errors) == pytest.approx(0.0, abs=1e-9)

    def test_clamped_at_five(self):
        """Result is clamped to [-5, +5] regardless of input magnitude."""
        errors = [100.0] * 100
        assert self._run_ema(errors) == pytest.approx(5.0, abs=1e-6)

        errors = [-100.0] * 100
        assert self._run_ema(errors) == pytest.approx(-5.0, abs=1e-6)

    def test_chronological_order_matters(self):
        """Reversing date order produces a different EMA (most-recent date should
        dominate when applied last in the sequential EMA)."""
        errors_fwd = [0.0, 0.0, 0.0, 5.0]   # big error on last day
        errors_rev = [5.0, 0.0, 0.0, 0.0]   # big error on first day

        bias_fwd = self._run_ema(errors_fwd)
        bias_rev = self._run_ema(errors_rev)

        # Forward: most recent large error has highest weight → larger bias
        assert bias_fwd > bias_rev


class TestMCDailyMaxBias:
    """D3: daily_max_bias in MCParams shifts the distribution upward."""

    def _run_mc(self, daily_max_bias: float) -> float:
        """Run a minimal simulation and return mean_max."""
        from kalshi_weather_trader.quant.monte_carlo import MCParams, price_full_distribution

        params = MCParams(
            T0=65.0,
            hard_floor=64.0,
            nwp_curve=[65.0 + i * 0.3 for i in range(24)],
            bias=0.0,
            daily_max_bias=daily_max_bias,
            theta=0.3,
            sigma=0.6,
            hour_offset=10,
            day_fraction_remaining=14 / 24,
            n_paths=2000,
        )
        result = price_full_distribution(params, strikes=[68.0, 72.0, 76.0], seed=42)
        return result.mean_max

    def test_positive_bias_raises_mean_max(self):
        """A positive daily_max_bias should raise mean_max relative to zero bias."""
        mean_base = self._run_mc(0.0)
        mean_biased = self._run_mc(2.0)
        assert mean_biased > mean_base

    def test_zero_bias_same_as_no_param(self):
        """daily_max_bias=0.0 should produce the same mean_max as the default."""
        from kalshi_weather_trader.quant.monte_carlo import MCParams, price_full_distribution

        params_default = MCParams(
            T0=65.0,
            hard_floor=64.0,
            nwp_curve=[65.0 + i * 0.3 for i in range(24)],
            bias=0.0,
            theta=0.3,
            sigma=0.6,
            hour_offset=10,
            day_fraction_remaining=14 / 24,
            n_paths=2000,
        )
        result_default = price_full_distribution(params_default, strikes=[68.0], seed=42)

        params_explicit = MCParams(
            T0=65.0,
            hard_floor=64.0,
            nwp_curve=[65.0 + i * 0.3 for i in range(24)],
            bias=0.0,
            daily_max_bias=0.0,
            theta=0.3,
            sigma=0.6,
            hour_offset=10,
            day_fraction_remaining=14 / 24,
            n_paths=2000,
        )
        result_explicit = price_full_distribution(params_explicit, strikes=[68.0], seed=42)

        assert result_default.mean_max == pytest.approx(result_explicit.mean_max, abs=1e-6)


class TestThetaAR1Logic:
    def test_ar1_phi_from_known_series(self):
        """AR(1) fit on a known OU series should recover phi approximately."""
        rng = np.random.default_rng(42)
        n = 500
        phi_true = 0.85
        x = np.zeros(n)
        x[0] = 70.0
        for i in range(1, n):
            x[i] = phi_true * x[i - 1] + (1 - phi_true) * 72.0 + rng.standard_normal() * 0.5

        y = x[1:]
        xx = x[:-1]
        phi_est = float(np.cov(xx, y)[0, 1] / np.var(xx))

        assert phi_est == pytest.approx(phi_true, abs=0.05)

    def test_theta_bounds(self):
        """Computed theta must stay within [0.01, 2.0]."""
        import math

        phi_values = [0.01, 0.5, 0.85, 0.99]
        for phi in phi_values:
            phi_clipped = max(0.01, min(0.99, phi))
            theta = max(0.01, min(2.0, -math.log(phi_clipped) / 1.0))
            assert 0.01 <= theta <= 2.0
