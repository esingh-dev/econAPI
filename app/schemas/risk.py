"""Pydantic v2 schemas for Risk analytics, EVT, GARCH, VaR Backtesting, and Stress Testing."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ═════════════════════════════════════════════════════════════════════
# 1. Extreme Value Theory (EVT) & POT
# ═════════════════════════════════════════════════════════════════════

class EVTRequest(BaseModel):
    """Input for EVT Peaks-Over-Threshold analysis."""

    returns: list[float] = Field(..., min_length=50, description="Historical return series (>= 50 observations)")
    confidence_level: float = Field(default=0.99, gt=0.5, lt=1.0, description="VaR/CVaR confidence level")
    threshold_quantile: float = Field(default=0.95, gt=0.5, lt=1.0, description="POT threshold quantile")
    include_diagnostics: bool = Field(default=True, description="Include Mean Excess Plot and stability scan")


class GPDParams(BaseModel):
    xi: float = Field(description="Shape parameter (xi)")
    beta: float = Field(description="Scale parameter (beta)")
    threshold: float = Field(description="Threshold value u")
    n_exceedances: int
    n_total: int
    ks_statistic: Optional[float] = None
    ks_p_value: Optional[float] = None
    fit_accepted: Optional[bool] = None
    qq_plot: Optional[list[dict[str, float]]] = None


class EVTResponse(BaseModel):
    gpd_params: GPDParams
    var: float = Field(description="Value-at-Risk")
    cvar: float = Field(description="Conditional VaR / Expected Shortfall")
    confidence_level: float
    var_ci_95_lower: Optional[float] = None
    var_ci_95_upper: Optional[float] = None
    threshold_diagnostics: Optional[dict] = None


# ═════════════════════════════════════════════════════════════════════
# 2. Multi-Model VaR & Expected Shortfall (Normal, t, Cornish-Fisher)
# ═════════════════════════════════════════════════════════════════════

class MultiModelVaRRequest(BaseModel):
    returns: list[float] = Field(..., min_length=30)
    confidence_level: float = Field(default=0.99, gt=0.5, lt=1.0)
    time_horizon_days: int = Field(default=1, ge=1)


class MultiModelVaRResponse(BaseModel):
    confidence_level: float
    time_horizon_days: int
    sample_moments: dict[str, float]
    models: dict[str, dict]


# ═════════════════════════════════════════════════════════════════════
# 3. Volatility Models: GARCH(1,1) & EWMA
# ═════════════════════════════════════════════════════════════════════

class GARCHRequest(BaseModel):
    returns: list[float] = Field(..., min_length=50)


class GARCHResponse(BaseModel):
    omega: float
    alpha: float
    beta: float
    persistence: float
    is_stationary: bool
    current_annualized_vol: float
    long_run_annualized_vol: Optional[float] = None
    converged: bool


class EWMARequest(BaseModel):
    returns: list[float] = Field(..., min_length=20)
    decay: float = Field(default=0.94, gt=0.5, lt=1.0)


class EWMAResponse(BaseModel):
    decay_lambda: float
    current_annualized_vol: float
    volatility_path: list[float]


# ═════════════════════════════════════════════════════════════════════
# 4. Factor Models & Risk Decomposition
# ═════════════════════════════════════════════════════════════════════

class FactorModelRequest(BaseModel):
    asset_returns: list[float] = Field(..., min_length=20)
    factor_returns: list[list[float]] = Field(..., min_length=20)
    factor_names: Optional[list[str]] = None


class FactorModelResponse(BaseModel):
    alpha: float
    factor_betas: dict[str, float]
    r_squared: float
    systematic_variance_pct: float
    idiosyncratic_variance_pct: float
    total_annualized_vol: float
    idiosyncratic_annualized_vol: float


# ═════════════════════════════════════════════════════════════════════
# 5. Stress Testing & Scenarios
# ═════════════════════════════════════════════════════════════════════

class StressTestRequest(BaseModel):
    portfolio_weights: dict[str, float]
    factor_exposures: dict[str, dict[str, float]]
    scenarios: Optional[dict[str, dict[str, float]]] = None


class StressTestResponse(BaseModel):
    scenarios: dict[str, dict]


# ═════════════════════════════════════════════════════════════════════
# 6. VaR Backtesting (Kupiec, Christoffersen, Traffic Light)
# ═════════════════════════════════════════════════════════════════════

class VaRBacktestRequest(BaseModel):
    realized_returns: list[float] = Field(..., min_length=30)
    forecast_vars: list[float] = Field(..., min_length=30)
    confidence_level: float = Field(default=0.99, gt=0.5, lt=1.0)


class VaRBacktestResponse(BaseModel):
    n_observations: int
    n_exceptions: int
    expected_exceptions: float
    observed_failure_rate: float
    expected_failure_rate: float
    kupiec_lr_pof: float
    kupiec_p_value: float
    kupiec_passed: bool
    christoffersen_lr_ind: float
    christoffersen_p_value: float
    independence_passed: bool
    conditional_coverage_lr: float
    conditional_coverage_p_value: float
    basel_traffic_light_zone: str
