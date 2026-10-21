"""API v1 — Portfolio optimization, Construction, HRP, and Attribution."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from app.engines import hrp, portfolio_optimization
from app.schemas.portfolio import (
    BlackLittermanRequest,
    BlackLittermanResponse,
    BrinsonAttributionRequest,
    BrinsonAttributionResponse,
    CVaROptimizationRequest,
    CVaROptimizationResponse,
    HRPBacktestRequest,
    HRPBacktestResponse,
    HRPRequest,
    HRPResponse,
    MeanVarianceRequest,
    MeanVarianceResponse,
    PortfolioMetricsRequest,
    PortfolioMetricsResponse,
    RiskParityRequest,
    RiskParityResponse,
)

router = APIRouter(prefix="/v1/portfolio", tags=["Portfolio"])


# ═════════════════════════════════════════════════════════════════════
# 1. Hierarchical Risk Parity (HRP)
# ═════════════════════════════════════════════════════════════════════

@router.post("/hrp", response_model=HRPResponse)
def optimize_hrp(req: HRPRequest) -> HRPResponse:
    """Compute Hierarchical Risk Parity weights with covariance shrinkage, constraints, and turnover."""
    returns_arr = np.array(req.returns, dtype=float)
    result = hrp.hrp_optimize(
        returns_arr,
        asset_names=req.asset_names,
        cov_estimator=req.cov_estimator,
        missing_method=req.missing_method,
        min_weight=req.min_weight,
        max_weight=req.max_weight,
        prev_weights=req.prev_weights,
        tx_cost_bps=req.tx_cost_bps,
    )
    return HRPResponse(**result)


@router.post("/hrp/backtest", response_model=HRPBacktestResponse)
def backtest_hrp(req: HRPBacktestRequest) -> HRPBacktestResponse:
    """Walk-forward rolling out-of-sample backtest of HRP portfolio allocation."""
    returns_arr = np.array(req.returns, dtype=float)
    result = hrp.hrp_walk_forward_backtest(
        returns_arr,
        train_window=req.train_window,
        rebalance_freq=req.rebalance_freq,
        asset_names=req.asset_names,
        cov_estimator=req.cov_estimator,
    )
    return HRPBacktestResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 2. Markowitz Mean-Variance Optimization
# ═════════════════════════════════════════════════════════════════════

@router.post("/mean-variance", response_model=MeanVarianceResponse)
def optimize_mean_variance_endpoint(req: MeanVarianceRequest) -> MeanVarianceResponse:
    """Compute Mean-Variance portfolio weights (Max Sharpe, Min Variance, Target Return, Target Risk)."""
    result = portfolio_optimization.optimize_mean_variance(
        req.expected_returns,
        req.cov_matrix,
        objective=req.objective,
        target_value=req.target_value,
        risk_free_rate=req.risk_free_rate,
        min_weight=req.min_weight,
        max_weight=req.max_weight,
        allow_short=req.allow_short,
        asset_names=req.asset_names,
        prev_weights=req.prev_weights,
        max_turnover=req.max_turnover,
    )
    return MeanVarianceResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 3. Risk Parity / Equal Risk Contribution
# ═════════════════════════════════════════════════════════════════════

@router.post("/risk-parity", response_model=RiskParityResponse)
def optimize_risk_parity_endpoint(req: RiskParityRequest) -> RiskParityResponse:
    """Equal Risk Contribution (ERC) portfolio optimization equalizing marginal risk contributions."""
    result = portfolio_optimization.optimize_risk_parity(
        req.cov_matrix, asset_names=req.asset_names, budget=req.budget
    )
    return RiskParityResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 4. Black-Litterman Model
# ═════════════════════════════════════════════════════════════════════

@router.post("/black-litterman", response_model=BlackLittermanResponse)
def black_litterman_endpoint(req: BlackLittermanRequest) -> BlackLittermanResponse:
    """Compute Black-Litterman implied equilibrium returns, posterior returns, and posterior covariance."""
    result = portfolio_optimization.black_litterman(
        req.market_weights,
        req.cov_matrix,
        risk_aversion=req.risk_aversion,
        tau=req.tau,
        pick_matrix_P=req.pick_matrix_P,
        view_returns_Q=req.view_returns_Q,
        omega_uncertainty=req.omega_uncertainty,
        asset_names=req.asset_names,
    )
    return BlackLittermanResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 5. CVaR / Expected Shortfall Optimization
# ═════════════════════════════════════════════════════════════════════

@router.post("/cvar-optimization", response_model=CVaROptimizationResponse)
def optimize_cvar_endpoint(req: CVaROptimizationRequest) -> CVaROptimizationResponse:
    """Rockafellar-Uryasev Linear Programming CVaR / Expected Shortfall portfolio optimization."""
    result = portfolio_optimization.optimize_cvar(
        req.scenario_returns,
        confidence_level=req.confidence_level,
        min_weight=req.min_weight,
        max_weight=req.max_weight,
        asset_names=req.asset_names,
    )
    return CVaROptimizationResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 6. Performance Attribution & Risk Metrics
# ═════════════════════════════════════════════════════════════════════

@router.post("/attribution", response_model=BrinsonAttributionResponse)
def brinson_attribution_endpoint(req: BrinsonAttributionRequest) -> BrinsonAttributionResponse:
    """Brinson-Fachler performance attribution (allocation, selection, interaction effects)."""
    result = portfolio_optimization.brinson_performance_attribution(
        req.portfolio_weights,
        req.benchmark_weights,
        req.portfolio_returns,
        req.benchmark_returns,
        sector_names=req.sector_names,
    )
    return BrinsonAttributionResponse(**result)


@router.post("/metrics", response_model=PortfolioMetricsResponse)
def compute_metrics_endpoint(req: PortfolioMetricsRequest) -> PortfolioMetricsResponse:
    """Compute portfolio risk-adjusted metrics (Sharpe, Sortino, Calmar, Max Drawdown, Ulcer, Tracking Error)."""
    result = portfolio_optimization.compute_portfolio_metrics(
        np.array(req.returns),
        benchmark_returns=np.array(req.benchmark_returns) if req.benchmark_returns else None,
        risk_free_rate=req.risk_free_rate,
        annualization_factor=req.annualization_factor,
    )
    return PortfolioMetricsResponse(**result)
