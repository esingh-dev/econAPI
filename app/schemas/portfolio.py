"""Pydantic v2 schemas for Portfolio Optimization, Construction, HRP, and Attribution."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ═════════════════════════════════════════════════════════════════════
# 1. Hierarchical Risk Parity (HRP)
# ═════════════════════════════════════════════════════════════════════

class HRPRequest(BaseModel):
    returns: list[list[float]] = Field(..., min_length=5, description="T x N return matrix")
    asset_names: Optional[list[str]] = Field(default=None, description="Names for the N assets")
    cov_estimator: Literal["ledoit_wolf", "sample", "ewma", "semi_covariance"] = Field(default="ledoit_wolf")
    missing_method: Literal["dropna", "forward_fill", "mean_impute"] = Field(default="dropna")
    min_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    max_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    prev_weights: Optional[dict[str, float]] = None
    tx_cost_bps: float = Field(default=0.0, ge=0.0)


class HRPResponse(BaseModel):
    weights: dict[str, float]
    portfolio_variance: float
    portfolio_volatility: Optional[float] = None
    sharpe_ratio: float
    annualized_return: Optional[float] = None
    turnover: Optional[float] = None
    transaction_cost: Optional[float] = None
    covariance_estimator: str
    asset_order: list[str]


class HRPBacktestRequest(BaseModel):
    returns: list[list[float]] = Field(..., min_length=60)
    train_window: int = Field(default=120, ge=30)
    rebalance_freq: int = Field(default=20, ge=1)
    asset_names: Optional[list[str]] = None
    cov_estimator: Literal["ledoit_wolf", "sample", "ewma"] = "ledoit_wolf"


class HRPBacktestResponse(BaseModel):
    out_of_sample_returns: list[float]
    metrics: dict
    n_periods: int


# ═════════════════════════════════════════════════════════════════════
# 2. Markowitz Mean-Variance Optimization
# ═════════════════════════════════════════════════════════════════════

class MeanVarianceRequest(BaseModel):
    expected_returns: list[float] = Field(..., min_length=2)
    cov_matrix: list[list[float]] = Field(..., min_length=2)
    objective: Literal["max_sharpe", "min_variance", "target_return", "target_risk"] = "max_sharpe"
    target_value: Optional[float] = None
    risk_free_rate: float = Field(default=0.02)
    min_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    max_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    allow_short: bool = False
    asset_names: Optional[list[str]] = None
    prev_weights: Optional[list[float]] = None
    max_turnover: Optional[float] = None


class MeanVarianceResponse(BaseModel):
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    converged: bool
    objective: str


# ═════════════════════════════════════════════════════════════════════
# 3. Risk Parity / Equal Risk Contribution
# ═════════════════════════════════════════════════════════════════════

class RiskParityRequest(BaseModel):
    cov_matrix: list[list[float]] = Field(..., min_length=2)
    asset_names: Optional[list[str]] = None
    budget: Optional[list[float]] = None


class RiskParityResponse(BaseModel):
    weights: dict[str, float]
    risk_contributions: dict[str, float]
    portfolio_volatility: float
    converged: bool


# ═════════════════════════════════════════════════════════════════════
# 4. Black-Litterman Model
# ═════════════════════════════════════════════════════════════════════

class BlackLittermanRequest(BaseModel):
    market_weights: list[float] = Field(..., min_length=2)
    cov_matrix: list[list[float]] = Field(..., min_length=2)
    risk_aversion: float = Field(default=2.5, gt=0)
    tau: float = Field(default=0.05, gt=0)
    pick_matrix_P: Optional[list[list[float]]] = None
    view_returns_Q: Optional[list[float]] = None
    omega_uncertainty: Optional[list[list[float]]] = None
    asset_names: Optional[list[str]] = None


class BlackLittermanResponse(BaseModel):
    prior_returns: dict[str, float]
    posterior_returns: dict[str, float]
    posterior_weights: dict[str, float]
    prior_weights: Optional[dict[str, float]] = None


# ═════════════════════════════════════════════════════════════════════
# 5. CVaR / Expected Shortfall Optimization
# ═════════════════════════════════════════════════════════════════════

class CVaROptimizationRequest(BaseModel):
    scenario_returns: list[list[float]] = Field(..., min_length=10, description="T scenarios x N assets")
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    min_weight: float = Field(default=0.0, ge=0.0)
    max_weight: float = Field(default=1.0, le=1.0)
    asset_names: Optional[list[str]] = None


class CVaROptimizationResponse(BaseModel):
    weights: dict[str, float]
    var: float
    cvar: float
    confidence_level: float
    converged: bool


# ═════════════════════════════════════════════════════════════════════
# 6. Performance Attribution & Risk Metrics
# ═════════════════════════════════════════════════════════════════════

class BrinsonAttributionRequest(BaseModel):
    portfolio_weights: list[float] = Field(..., min_length=2)
    benchmark_weights: list[float] = Field(..., min_length=2)
    portfolio_returns: list[float] = Field(..., min_length=2)
    benchmark_returns: list[float] = Field(..., min_length=2)
    sector_names: Optional[list[str]] = None


class SectorAttributionItem(BaseModel):
    sector: str
    portfolio_weight: float
    benchmark_weight: float
    portfolio_return: float
    benchmark_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_effect: float


class BrinsonAttributionResponse(BaseModel):
    portfolio_total_return: float
    benchmark_total_return: float
    total_excess_return: float
    total_allocation: float
    total_selection: float
    total_interaction: float
    sectors: list[SectorAttributionItem]


class PortfolioMetricsRequest(BaseModel):
    returns: list[float] = Field(..., min_length=5)
    benchmark_returns: Optional[list[float]] = None
    risk_free_rate: float = Field(default=0.02)
    annualization_factor: int = Field(default=252)


class PortfolioMetricsResponse(BaseModel):
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    max_drawdown: float
    ulcer_index: float
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None
