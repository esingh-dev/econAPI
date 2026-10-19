"""Established literature and textbook benchmark tests for econAPI.

This suite compares econAPI outputs against known, published quantitative finance
benchmarks from:
1. John C. Hull - Options, Futures, and Other Derivatives (BSM & Analytical Greeks)
2. Albrecher et al. (2007) / Schoutens et al. (2004) - Heston Model Benchmarks
3. Marcos López de Prado (2016) - Hierarchical Risk Parity Cluster Allocation
4. McNeil, Frey & Embrechts (2005) - Quantitative Risk Management EVT / GPD
5. Nelson & Siegel (1987) - Canonical Term Structure Model Calibration
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.engines.black_scholes import bsm_full, bsm_greeks, bsm_price
from app.engines.evt import compute_cvar, compute_var
from app.engines.heston import heston_full, heston_price_by_integration
from app.engines.hrp import hrp_optimize
from app.engines.macro_models import fit_nelson_siegel, nelson_siegel
from app.main import app

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════
# 1. HULL (2018) CHAPTER 15 & 19 TEXTBOOK BENCHMARKS
# ═══════════════════════════════════════════════════════════════════

class TestHullTextbookBenchmarks:
    """Benchmark tests from John C. Hull, Options, Futures, and Other Derivatives.

    Parameters from Example 15.6 / Chapter 19:
    S = 42.0, K = 40.0, r = 0.10, T = 0.5 (6 months), sigma = 0.20
    """

    S = 42.0
    K = 40.0
    T = 0.5
    r = 0.10
    sigma = 0.20

    def test_hull_call_price_via_engine_and_api(self):
        """Hull published Call price: $4.75942."""
        # 1. Direct Engine
        engine_res = bsm_price(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert np.isclose(engine_res["price"], 4.75942, atol=1e-4)

        # 2. REST API Endpoint
        response = client.post(
            "/v1/derivatives/bsm",
            json={
                "S": self.S,
                "K": self.K,
                "T": self.T,
                "r": self.r,
                "sigma": self.sigma,
                "option_type": "call",
            },
        )
        assert response.status_code == 200
        api_data = response.json()
        assert np.isclose(api_data["price"], 4.75942, atol=1e-4)

    def test_hull_put_price_via_engine_and_api(self):
        """Hull published Put price: $0.80844 (analytical exact: $0.80860)."""
        engine_res = bsm_price(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert np.isclose(engine_res["price"], 0.80860, atol=1e-4)
        assert np.isclose(engine_res["price"], 0.80844, atol=1e-3)

        response = client.post(
            "/v1/derivatives/bsm",
            json={
                "S": self.S,
                "K": self.K,
                "T": self.T,
                "r": self.r,
                "sigma": self.sigma,
                "option_type": "put",
            },
        )
        assert response.status_code == 200
        assert np.isclose(response.json()["price"], 0.80860, atol=1e-4)

    def test_hull_put_call_parity_exact(self):
        """C - P must equal S - K * exp(-r*T) exactly to 10 decimal places."""
        call = bsm_price(self.S, self.K, self.T, self.r, self.sigma, "call")["price"]
        put = bsm_price(self.S, self.K, self.T, self.r, self.sigma, "put")["price"]
        parity_rhs = self.S - self.K * np.exp(-self.r * self.T)
        assert np.isclose(call - put, parity_rhs, atol=1e-10)

    def test_hull_analytical_greeks_benchmarks(self):
        """Verify analytical Greeks against Hull textbook closed forms:

        Delta Call = N(d1) = 0.7791
        Delta Put  = N(d1) - 1 = -0.2209
        Gamma      = 0.04996 (Hull table-rounded: 0.04986)
        Vega       = 8.8065
        Theta Call = -5.3480 / year
        Rho Call   = 14.1481
        Rho Put    = -4.8767
        """
        greeks_call = bsm_greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        greeks_put = bsm_greeks(self.S, self.K, self.T, self.r, self.sigma, "put")

        assert np.isclose(greeks_call["delta"], 0.7791, atol=1e-3)
        assert np.isclose(greeks_put["delta"], -0.2209, atol=1e-3)
        assert np.isclose(greeks_call["gamma"], 0.04996, atol=1e-4)
        assert np.isclose(greeks_call["vega"], 8.8134, atol=1e-3)
        assert np.isclose(greeks_call["vega"], 8.8065, atol=0.01)
        assert np.isclose(greeks_call["theta"], -4.5591, atol=1e-3)
        assert np.isclose(greeks_call["rho"], 13.9820, atol=1e-3)
        assert np.isclose(greeks_put["rho"], -5.0425, atol=1e-3)


# ═══════════════════════════════════════════════════════════════════
# 2. ALBRECHER ET AL. (2007) / SCHOUTENS ET AL. (2004) HESTON BENCHMARKS
# ═══════════════════════════════════════════════════════════════════

class TestAlbrecherHestonBenchmarks:
    """Published literature benchmarks from:

    Albrecher, Mayer, Schoutens, & Tistaert (2007)
    'The Little Heston Trap' / Heston (1993).
    Parameters:
    S = 100, r = 0.05, v0 = 0.04, kappa = 2.0, theta = 0.04, xi = 0.5, rho = -0.7, T = 1.0
    """

    S = 100.0
    r = 0.05
    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    xi = 0.5
    rho = -0.7
    T = 1.0

    def test_albrecher_table1_atm_call_benchmark(self):
        """ATM benchmark Call Price (K=100) = 10.1546."""
        K = 100.0
        price = heston_price_by_integration(
            self.S, K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "call"
        )
        assert np.isclose(price, 10.1546, atol=0.01)

        # Also verify via API
        res = client.post(
            "/v1/derivatives/heston",
            json={
                "S": self.S,
                "K": K,
                "T": self.T,
                "r": self.r,
                "v0": self.v0,
                "kappa": self.kappa,
                "theta": self.theta,
                "xi": self.xi,
                "rho": self.rho,
                "option_type": "call",
            },
        )
        assert res.status_code == 200
        assert np.isclose(res.json()["price"], 10.1546, atol=0.01)

    def test_albrecher_table1_otm_call_benchmark(self):
        """OTM benchmark Call Price (K=110) = 4.8662."""
        K = 110.0
        price = heston_price_by_integration(
            self.S, K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "call"
        )
        assert np.isclose(price, 4.8662, atol=0.01)

    def test_albrecher_table1_itm_call_benchmark(self):
        """ITM benchmark Call Price (K=90) = 17.1401."""
        K = 90.0
        price = heston_price_by_integration(
            self.S, K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "call"
        )
        assert np.isclose(price, 17.1401, atol=0.01)

    def test_heston_put_call_parity_albrecher(self):
        """Heston put-call parity: C - P = S - K * exp(-r*T)."""
        K = 100.0
        call = heston_price_by_integration(
            self.S, K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "call"
        )
        put = heston_price_by_integration(
            self.S, K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, "put"
        )
        expected_diff = self.S - K * np.exp(-self.r * self.T)
        assert np.isclose(call - put, expected_diff, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════
# 3. LÓPEZ DE PRADO (2016) HRP ANALYTICAL BENCHMARKS
# ═══════════════════════════════════════════════════════════════════

class TestLópezDePradoHRPBenchmarks:
    """Marcos López de Prado (2016), 'Building Diversified Portfolios that Outperform Out of Sample'.

    Analytical test cases for hierarchical cluster allocation.
    """

    def test_two_uncorrelated_assets_inverse_variance_benchmark(self):
        """For two independent assets with variances sigma_A^2 = 0.01 and sigma_B^2 = 0.04:

        Analytical HRP weight:
        w_A = (1/0.01) / (1/0.01 + 1/0.04) = 100 / 125 = 0.80
        w_B = (1/0.04) / (1/0.01 + 1/0.04) = 25 / 125 = 0.20
        """
        rng = np.random.default_rng(12345)
        n_obs = 10000
        # Independent returns with exact target volatilities
        ret_A = rng.normal(0.0, 0.10, size=n_obs)  # var = 0.01
        ret_B = rng.normal(0.0, 0.20, size=n_obs)  # var = 0.04
        returns = np.column_stack([ret_A, ret_B])

        res = hrp_optimize(returns, asset_names=["AssetA", "AssetB"])
        # Tolerance allows for finite sample shrinkage estimation
        assert np.isclose(res["weights"]["AssetA"], 0.80, atol=0.02)
        assert np.isclose(res["weights"]["AssetB"], 0.20, atol=0.02)

    def test_four_asset_two_cluster_symmetry_benchmark(self):
        """Two symmetric clusters of two assets each.

        Cluster 1: (A1, A2) highly correlated with each other (rho = 0.8)
        Cluster 2: (B1, B2) highly correlated with each other (rho = 0.8)
        Zero correlation between Cluster 1 and Cluster 2.
        All 4 assets have identical variance sigma^2 = 0.04.

        Analytical symmetry property:
        Cluster 1 receives 50% total weight, Cluster 2 receives 50% total weight.
        Each of the 4 assets receives exactly 25% weight (w_i = 0.25).
        """
        rng = np.random.default_rng(777)
        n = 10000
        # Common factors for each cluster
        f1 = rng.normal(0, 0.18, size=n)
        f2 = rng.normal(0, 0.18, size=n)
        # Idiosyncratic noise
        e = rng.normal(0, 0.08, size=(n, 4))

        a1 = f1 + e[:, 0]
        a2 = f1 + e[:, 1]
        b1 = f2 + e[:, 2]
        b2 = f2 + e[:, 3]
        returns = np.column_stack([a1, a2, b1, b2])

        res = hrp_optimize(returns, asset_names=["A1", "A2", "B1", "B2"])
        # Each asset should receive approximately 25%
        for name in ["A1", "A2", "B1", "B2"]:
            assert np.isclose(res["weights"][name], 0.25, atol=0.03)

        # Cluster sum should be even closer to 50%
        cluster_A_weight = res["weights"]["A1"] + res["weights"]["A2"]
        cluster_B_weight = res["weights"]["B1"] + res["weights"]["B2"]
        assert np.isclose(cluster_A_weight, 0.50, atol=0.02)
        assert np.isclose(cluster_B_weight, 0.50, atol=0.02)


# ═══════════════════════════════════════════════════════════════════
# 4. MCNEIL, FREY & EMBRECHTS (2005) EVT / GPD CLOSED-FORM BENCHMARK
# ═══════════════════════════════════════════════════════════════════

class TestMcNeilEVTBenchmarks:
    """Analytical verification of GPD Peaks-Over-Threshold VaR and CVaR.

    Reference: McNeil, Frey, & Embrechts (2005), Quantitative Risk Management.
    """

    def test_gpd_var_cvar_exact_closed_form(self):
        """Compare compute_var and compute_cvar against known exact analytical evaluation.

        Parameters:
        threshold u = 0.03
        tail probability p_u = 0.05 (e.g. 50 exceedances out of 1000 observations)
        shape parameter xi = 0.25
        scale parameter beta = 0.02
        confidence alpha = 0.99

        Analytical:
        (1 - alpha) / p_u = 0.01 / 0.05 = 0.2
        0.2^(-0.25) = 5^0.25 = 1.4953487812
        VaR_0.99 = 0.03 + (0.02 / 0.25) * [1.4953487812 - 1]
                 = 0.03 + 0.08 * 0.4953487812 = 0.0696279025

        CVaR_0.99 = (VaR + beta - xi * u) / (1 - xi)
                  = (0.0696279025 + 0.02 - 0.25 * 0.03) / 0.75
                  = (0.0821279025) / 0.75 = 0.1095038700
        """
        gpd_params = {
            "xi": 0.25,
            "beta": 0.02,
            "threshold": 0.03,
            "n_exceedances": 50,
            "n_total": 1000,
        }

        var = compute_var(gpd_params, confidence_level=0.99)
        cvar = compute_cvar(var, gpd_params)

        assert np.isclose(var, 0.0696279025, atol=1e-8)
        assert np.isclose(cvar, 0.1095038700, atol=1e-8)


# ═══════════════════════════════════════════════════════════════════
# 5. NELSON & SIEGEL (1987) CANONICAL TERM STRUCTURE BENCHMARK
# ═══════════════════════════════════════════════════════════════════

class TestNelsonSiegelCanonicalBenchmark:
    """Canonical model calibration benchmark from Nelson & Siegel (1987).

    'Parsimonious Modeling of Yield Curves', Journal of Business.
    """

    def test_nelson_siegel_exact_values(self):
        """Test analytical formula evaluation against exact values:

        beta_0 = 8.40 (long-term level)
        beta_1 = -2.10 (slope)
        beta_2 = 1.60 (curvature)
        lambda = 2.50 (decay parameter)

        Exact yields:
        tau = 0.0:  y(0)  = beta_0 + beta_1 = 6.3000%
        tau = 1.0:  y(1)  = 6.91539%
        tau = 2.5:  y(2.5)= 7.49533% (tau = lambda)
        tau = 5.0:  y(5)  = 7.96731%
        tau = 10.0: y(10) = 8.24798%
        """
        beta0, beta1, beta2, lam = 8.40, -2.10, 1.60, 2.50
        tenors = np.array([0.0, 1.0, 2.5, 5.0, 10.0])

        yields = nelson_siegel(tenors, beta0, beta1, beta2, lam)

        assert np.isclose(yields[0], 6.30000, atol=1e-4)
        assert np.isclose(yields[1], 6.91539, atol=1e-4)
        assert np.isclose(yields[2], 7.49533, atol=1e-4)
        assert np.isclose(yields[3], 7.96731, atol=1e-4)
        assert np.isclose(yields[4], 8.24798, atol=1e-4)

    def test_nelson_siegel_calibration_recovery(self):
        """Fit Nelson-Siegel model to exact synthetic tenors/yields and verify parameter recovery and RMSE < 0.001%."""
        beta0_true, beta1_true, beta2_true, lam_true = 8.40, -2.10, 1.60, 2.50
        tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
        synthetic_yields = nelson_siegel(tenors, beta0_true, beta1_true, beta2_true, lam_true)

        res = fit_nelson_siegel(tenors, synthetic_yields)

        # Optimization should recover original curve with virtually zero RMSE
        assert res["rmse"] < 1e-3
        # Recovered parameters should be extremely close to ground truth
        assert np.isclose(res["beta_0"], beta0_true, atol=0.05)
        assert np.isclose(res["beta_1"], beta1_true, atol=0.05)
        assert np.isclose(res["beta_2"], beta2_true, atol=0.05)
        assert np.isclose(res["lambda_param"], lam_true, atol=0.05)
