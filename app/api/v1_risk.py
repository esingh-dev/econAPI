"""API v1 — Risk analytics, EVT, GARCH, Factor Models, Stress Testing, and VaR Backtesting."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from app.engines import evt, macro_models, time_series_risk
from app.schemas.fixed_income import NelsonSiegelRequest, NelsonSiegelResponse
from app.schemas.risk import (
    EWMARequest,
    EWMAResponse,
    EVTRequest,
    EVTResponse,
    FactorModelRequest,
    FactorModelResponse,
    GARCHRequest,
    GARCHResponse,
    GPDParams,
    MultiModelVaRRequest,
    MultiModelVaRResponse,
    StressTestRequest,
    StressTestResponse,
    VaRBacktestRequest,
    VaRBacktestResponse,
)

router = APIRouter(tags=["Risk & Macro"])


# ═════════════════════════════════════════════════════════════════════
# 1. Extreme Value Theory (EVT) Peaks-Over-Threshold
# ═════════════════════════════════════════════════════════════════════

@router.post("/v1/risk/evt-cvar", response_model=EVTResponse)
def compute_evt_cvar(req: EVTRequest) -> EVTResponse:
    """Compute tail-risk VaR and Expected Shortfall via Extreme Value Theory (GPD/POT).

    Fits a Generalized Pareto Distribution to tail losses with Kolmogorov-Smirnov goodness of fit,
    QQ-plot coordinates, Mean Excess diagnostics, and 95% asymptotic confidence intervals.
    """
    returns_array = np.array(req.returns, dtype=float)
    result = evt.evt_analysis(
        returns_array,
        confidence_level=req.confidence_level,
        threshold_quantile=req.threshold_quantile,
        include_diagnostics=req.include_diagnostics,
    )
    return EVTResponse(
        gpd_params=GPDParams(**result["gpd_params"]),
        var=result["var"],
        cvar=result["cvar"],
        confidence_level=result["confidence_level"],
        var_ci_95_lower=result.get("var_ci_95_lower"),
        var_ci_95_upper=result.get("var_ci_95_upper"),
        threshold_diagnostics=result.get("threshold_diagnostics"),
    )


# ═════════════════════════════════════════════════════════════════════
# 2. Multi-Model VaR & Expected Shortfall (Normal, Student-t, Cornish-Fisher)
# ═════════════════════════════════════════════════════════════════════

@router.post("/v1/risk/var-suite", response_model=MultiModelVaRResponse)
def compute_var_suite_endpoint(req: MultiModelVaRRequest) -> MultiModelVaRResponse:
    """Compute VaR and CVaR across Historical, Parametric Normal, Student-t, and Cornish-Fisher models."""
    result = time_series_risk.compute_var_suite(
        req.returns, req.confidence_level, req.time_horizon_days
    )
    return MultiModelVaRResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 3. Volatility Models: GARCH(1,1) & EWMA
# ═════════════════════════════════════════════════════════════════════

@router.post("/v1/risk/garch", response_model=GARCHResponse)
def fit_garch_endpoint(req: GARCHRequest) -> GARCHResponse:
    """Fit GARCH(1,1) conditional volatility model via Quasi-Maximum Likelihood (QMLE)."""
    result = time_series_risk.fit_garch11(req.returns)
    return GARCHResponse(**result)


@router.post("/v1/risk/ewma", response_model=EWMAResponse)
def compute_ewma_endpoint(req: EWMARequest) -> EWMAResponse:
    """Compute RiskMetrics EWMA volatility path."""
    result = time_series_risk.compute_ewma_volatility(req.returns, req.decay)
    return EWMAResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 4. Factor Models & Risk Decomposition
# ═════════════════════════════════════════════════════════════════════

@router.post("/v1/risk/factor-model", response_model=FactorModelResponse)
def factor_model_endpoint(req: FactorModelRequest) -> FactorModelResponse:
    """Decompose asset returns into systematic factor exposures and idiosyncratic risk."""
    result = time_series_risk.factor_model_decomposition(
        req.asset_returns, req.factor_returns, req.factor_names
    )
    return FactorModelResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 5. Stress Testing & Scenarios
# ═════════════════════════════════════════════════════════════════════

@router.post("/v1/risk/stress-test", response_model=StressTestResponse)
def stress_test_endpoint(req: StressTestRequest) -> StressTestResponse:
    """Evaluate portfolio P&L under historical and macro stress shock scenarios."""
    result = time_series_risk.run_stress_test(
        req.portfolio_weights, req.factor_exposures, req.scenarios
    )
    return StressTestResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 6. VaR Backtesting (Kupiec, Christoffersen, Basel Traffic Light)
# ═════════════════════════════════════════════════════════════════════

@router.post("/v1/risk/var-backtest", response_model=VaRBacktestResponse)
def var_backtest_endpoint(req: VaRBacktestRequest) -> VaRBacktestResponse:
    """Backtest VaR predictions using Kupiec POF, Christoffersen Independence, and Basel Traffic Light rules."""
    result = time_series_risk.backtest_var(
        req.realized_returns, req.forecast_vars, req.confidence_level
    )
    return VaRBacktestResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 7. Legacy Nelson-Siegel route
# ═════════════════════════════════════════════════════════════════════

@router.post("/v1/macro/nelson-siegel", response_model=NelsonSiegelResponse)
def fit_yield_curve(req: NelsonSiegelRequest) -> NelsonSiegelResponse:
    """Fit the Nelson-Siegel parametric yield curve to observed market yields."""
    tenors_array = np.array(req.tenors, dtype=float)
    yields_array = np.array(req.yields, dtype=float)
    result = macro_models.fit_nelson_siegel(
        tenors_array, yields_array, req.unit_convention, req.robust_loss
    )
    return NelsonSiegelResponse(**result)
