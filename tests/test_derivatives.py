"""Tests for Black-Scholes-Merton and Heston derivatives engines."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from app.engines.black_scholes import bsm_price, bsm_greeks, bsm_full
from app.engines.heston import heston_full, heston_price_by_integration


# ── Standard benchmark parameters ───────────────────────────────────
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2


# ═══════════════════════════════════════════════════════════════════
# BSM Price Tests
# ═══════════════════════════════════════════════════════════════════

class TestBSMPrice:
    """Verify BSM analytical prices against known textbook values."""

    def test_bsm_call_price(self):
        result = bsm_price(S, K, T, r, sigma, "call")
        # Known ATM call ≈ 10.4506
        assert abs(result["price"] - 10.4506) < 0.01

    def test_bsm_put_price(self):
        result = bsm_price(S, K, T, r, sigma, "put")
        # Known ATM put ≈ 5.5735
        assert abs(result["price"] - 5.5735) < 0.01

    def test_bsm_put_call_parity(self):
        call = bsm_price(S, K, T, r, sigma, "call")["price"]
        put = bsm_price(S, K, T, r, sigma, "put")["price"]
        # C − P = S − K·exp(−rT)
        parity_rhs = S - K * np.exp(-r * T)
        assert abs((call - put) - parity_rhs) < 1e-10

    def test_bsm_deep_itm_call(self):
        # Deep ITM call (S >> K) should be ≈ S − K·exp(−rT)
        result = bsm_price(200.0, 50.0, 1.0, 0.05, 0.2, "call")
        intrinsic = 200.0 - 50.0 * np.exp(-0.05)
        assert abs(result["price"] - intrinsic) < 1.0

    def test_bsm_deep_otm_call(self):
        # Deep OTM call (S << K) should be ≈ 0
        result = bsm_price(50.0, 200.0, 1.0, 0.05, 0.2, "call")
        assert result["price"] < 0.01

    def test_bsm_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            bsm_price(S, K, T, r, -0.2, "call")


# ═══════════════════════════════════════════════════════════════════
# BSM Greeks Tests
# ═══════════════════════════════════════════════════════════════════

class TestBSMGreeks:
    """Verify BSM analytical Greeks against hand-computed values."""

    def test_delta_call(self):
        greeks = bsm_greeks(S, K, T, r, sigma, "call")
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        expected_delta = norm.cdf(d1)
        assert abs(greeks["delta"] - expected_delta) < 1e-10

    def test_delta_put(self):
        greeks = bsm_greeks(S, K, T, r, sigma, "put")
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        expected_delta = norm.cdf(d1) - 1.0
        assert abs(greeks["delta"] - expected_delta) < 1e-10

    def test_delta_call_range(self):
        greeks = bsm_greeks(S, K, T, r, sigma, "call")
        assert 0.0 < greeks["delta"] < 1.0

    def test_gamma_positive(self):
        greeks = bsm_greeks(S, K, T, r, sigma, "call")
        assert greeks["gamma"] > 0

    def test_gamma_same_for_call_put(self):
        call_gamma = bsm_greeks(S, K, T, r, sigma, "call")["gamma"]
        put_gamma = bsm_greeks(S, K, T, r, sigma, "put")["gamma"]
        assert abs(call_gamma - put_gamma) < 1e-12

    def test_vega_positive(self):
        greeks = bsm_greeks(S, K, T, r, sigma, "call")
        assert greeks["vega"] > 0

    def test_vega_same_for_call_put(self):
        call_vega = bsm_greeks(S, K, T, r, sigma, "call")["vega"]
        put_vega = bsm_greeks(S, K, T, r, sigma, "put")["vega"]
        assert abs(call_vega - put_vega) < 1e-12

    def test_rho_call_positive(self):
        greeks = bsm_greeks(S, K, T, r, sigma, "call")
        assert greeks["rho"] > 0

    def test_rho_put_negative(self):
        greeks = bsm_greeks(S, K, T, r, sigma, "put")
        assert greeks["rho"] < 0


# ═══════════════════════════════════════════════════════════════════
# BSM Full Integration Test
# ═══════════════════════════════════════════════════════════════════

class TestBSMFull:
    def test_full_returns_all_keys(self):
        result = bsm_full(S, K, T, r, sigma, "call")
        assert set(result.keys()) == {"price", "delta", "gamma", "vega", "theta", "rho"}

    def test_full_consistency(self):
        full = bsm_full(S, K, T, r, sigma, "call")
        price = bsm_price(S, K, T, r, sigma, "call")
        greeks = bsm_greeks(S, K, T, r, sigma, "call")
        assert abs(full["price"] - price["price"]) < 1e-12
        assert abs(full["delta"] - greeks["delta"]) < 1e-12


# ═══════════════════════════════════════════════════════════════════
# Heston Model Tests
# ═══════════════════════════════════════════════════════════════════

class TestHeston:
    """Verify Heston model against BSM limits and put-call parity."""

    def test_heston_call_converges_to_bsm(self):
        """When vol-of-vol ξ → 0 and kappa is large, Heston ≈ BSM."""
        bsm_call = bsm_price(S, K, T, r, sigma, "call")["price"]
        heston_call = heston_price_by_integration(
            S, K, T, r,
            v0=sigma**2,   # v0 = σ²
            kappa=5.0,
            theta=sigma**2,
            xi=0.001,       # near-zero vol-of-vol
            rho=0.0,
            option_type="call",
        )
        assert abs(heston_call - bsm_call) < 0.5

    def test_heston_put_call_parity(self):
        """Heston call − put = S − K·exp(−rT)."""
        params = dict(S=S, K=K, T=T, r=r, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        call = heston_price_by_integration(**params, option_type="call")
        put = heston_price_by_integration(**params, option_type="put")
        parity_rhs = S - K * np.exp(-r * T)
        assert abs((call - put) - parity_rhs) < 0.5

    def test_heston_positive_call_price(self):
        result = heston_full(
            S, K, T, r, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
        )
        assert result["price"] > 0

    def test_heston_implied_vol_roundtrip(self):
        """Heston price → implied vol → BSM reprice should match."""
        result = heston_full(
            S, K, T, r, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
        )
        iv = result["implied_volatility"]
        repriced = bsm_price(S, K, T, r, iv, "call")["price"]
        assert abs(repriced - result["price"]) < 0.01

    def test_heston_full_keys(self):
        result = heston_full(
            S, K, T, r, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
        )
        assert set(result.keys()) == {"price", "implied_volatility"}
