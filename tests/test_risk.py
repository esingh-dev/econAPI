"""Tests for Extreme Value Theory (EVT) and Macro Econometric engines."""

from __future__ import annotations

import numpy as np
import pytest

from app.engines.evt import evt_analysis, fit_gpd_pot, compute_var, compute_cvar
from app.engines.macro_models import fit_nelson_siegel, nelson_siegel


# ═══════════════════════════════════════════════════════════════════
# EVT / GPD Tail Risk Tests
# ═══════════════════════════════════════════════════════════════════

class TestEVTTailRisk:
    """Verify Peaks-Over-Threshold and GPD risk metric calculations."""

    @pytest.fixture()
    def simulated_returns(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        # 1000 daily returns with slight positive drift and fat tails (t-distribution or normal)
        return rng.normal(0.0005, 0.02, size=1000)

    def test_evt_var_positive(self, simulated_returns: np.ndarray):
        """VaR should represent a positive loss amount."""
        res = evt_analysis(simulated_returns, confidence_level=0.99, threshold_quantile=0.95)
        assert res["var"] > 0

    def test_evt_cvar_greater_than_var(self, simulated_returns: np.ndarray):
        """Expected Shortfall (CVaR) must be greater than or equal to VaR."""
        res = evt_analysis(simulated_returns, confidence_level=0.99, threshold_quantile=0.95)
        assert res["cvar"] >= res["var"]

    def test_evt_gpd_params_finite(self, simulated_returns: np.ndarray):
        """GPD parameters xi and beta must be finite numbers and beta > 0."""
        res = evt_analysis(simulated_returns, confidence_level=0.99, threshold_quantile=0.95)
        gpd = res["gpd_params"]
        assert np.isfinite(gpd["xi"])
        assert np.isfinite(gpd["beta"])
        assert gpd["beta"] > 0
        assert gpd["n_exceedances"] > 0
        assert gpd["n_total"] == 1000

    def test_evt_confidence_levels_monotonicity(self, simulated_returns: np.ndarray):
        """Higher confidence level should produce higher VaR."""
        res_95 = evt_analysis(simulated_returns, confidence_level=0.95, threshold_quantile=0.90)
        res_99 = evt_analysis(simulated_returns, confidence_level=0.99, threshold_quantile=0.90)
        assert res_99["var"] > res_95["var"]
        assert res_99["cvar"] > res_95["cvar"]

    def test_insufficient_exceedances_raises(self):
        """If data length is too small or threshold is too extreme, raise ValueError."""
        small_returns = np.array([0.01, -0.01, 0.02, -0.02])
        with pytest.raises(ValueError):
            fit_gpd_pot(small_returns, threshold_quantile=0.99)


# ═══════════════════════════════════════════════════════════════════
# Nelson-Siegel Yield Curve Tests
# ═══════════════════════════════════════════════════════════════════

class TestNelsonSiegel:
    """Verify Nelson-Siegel parametric yield curve fitting."""

    @pytest.fixture()
    def treasury_curve(self) -> tuple[np.ndarray, np.ndarray]:
        # Realistic inverted US Treasury curve
        tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
        yields = np.array([5.25, 5.10, 4.85, 4.50, 4.35, 4.20, 4.15, 4.10, 4.30, 4.40])
        return tenors, yields

    def test_nelson_siegel_fit_rmse(self, treasury_curve):
        """Fitting error RMSE should be reasonably small (< 0.25%)."""
        tenors, yields = treasury_curve
        res = fit_nelson_siegel(tenors, yields)
        assert res["rmse"] < 0.25

    def test_nelson_siegel_parameters(self, treasury_curve):
        """Parameters should be economically sensible."""
        tenors, yields = treasury_curve
        res = fit_nelson_siegel(tenors, yields)
        # beta_0 is the long-term asymptotic rate
        assert 3.0 < res["beta_0"] < 7.0
        # lambda must be positive decay
        assert res["lambda_param"] > 0

    def test_nelson_siegel_roundtrip(self, treasury_curve):
        """Fitted yields should closely match observed market yields."""
        tenors, yields = treasury_curve
        res = fit_nelson_siegel(tenors, yields)
        fitted = np.array(res["fitted_yields"])
        max_abs_err = np.max(np.abs(yields - fitted))
        assert max_abs_err < 0.35

    def test_nelson_siegel_tau_zero_continuity(self):
        """At tau -> 0, y(0) should equal beta0 + beta1 without division by zero error."""
        tau = np.array([0.0, 1.0, 5.0])
        val = nelson_siegel(tau, beta0=4.0, beta1=1.0, beta2=-1.0, lam=2.0)
        assert np.isclose(val[0], 4.0 + 1.0)


# ═══════════════════════════════════════════════════════════════════
# VECM & Cointegration Tests
# ═══════════════════════════════════════════════════════════════════

class TestVECMCointegration:
    """Verify Johansen cointegration test and VECM fitting."""

    @pytest.fixture()
    def cointegrated_series(self) -> np.ndarray:
        # Generate two cointegrated series: y2 = 2 * y1 + stationary_noise
        rng = np.random.default_rng(42)
        n = 200
        e1 = rng.normal(0, 1, size=n)
        e2 = rng.normal(0, 0.5, size=n)
        # Random walk
        y1 = np.cumsum(e1)
        # Cointegrated with y1
        y2 = 2.0 * y1 + e2
        return np.column_stack([y1, y2])

    def test_johansen_cointegration(self, cointegrated_series: np.ndarray):
        from app.engines.macro_models import johansen_cointegration_test
        res = johansen_cointegration_test(cointegrated_series, det_order=0, k_ar_diff=1)
        assert len(res["trace_stat"]) == 2
        assert len(res["max_eig_stat"]) == 2
        # At least one cointegrating vector should be detected
        assert res["r_95"] >= 1

    def test_fit_vecm_model(self, cointegrated_series: np.ndarray):
        from app.engines.macro_models import fit_vecm_model
        res = fit_vecm_model(cointegrated_series, k_ar_diff=1, coint_rank=1, deterministic="co")
        assert len(res["alpha"]) == 2
        assert len(res["beta"]) == 2
        assert np.isfinite(res["llf"])
        assert len(res["sigma_u"]) == 2

