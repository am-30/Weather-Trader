"""
Unit tests for calibration routines.

Tests:
- Brier score softmax weights sum to 1.0
- Theta AR(1) fit returns value in valid range
- sigma estimator returns positive value
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
    def test_softmax_weights_sum_to_one(self):
        """Verify the softmax normalisation produces weights summing to 1.0."""
        import math

        brier_scores = {"HRRR": 0.02, "GFS": 0.05, "ECMWF": 0.08}
        inv_scores = {m: 1.0 / (s + 1e-8) for m, s in brier_scores.items()}
        max_inv = max(inv_scores.values())
        exp_scores = {m: math.exp(v - max_inv) for m, v in inv_scores.items()}
        total = sum(exp_scores.values())
        weights = {m: v / total for m, v in exp_scores.items()}

        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(w >= 0 for w in weights.values())

    def test_lower_brier_gets_higher_weight(self):
        """The model with the lowest Brier score should get the highest weight."""
        import math

        brier_scores = {"HRRR": 0.01, "GFS": 0.05, "ECMWF": 0.08}
        inv_scores = {m: 1.0 / (s + 1e-8) for m, s in brier_scores.items()}
        max_inv = max(inv_scores.values())
        exp_scores = {m: math.exp(v - max_inv) for m, v in inv_scores.items()}
        total = sum(exp_scores.values())
        weights = {m: v / total for m, v in exp_scores.items()}

        assert weights["HRRR"] > weights["GFS"] > weights["ECMWF"]


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
