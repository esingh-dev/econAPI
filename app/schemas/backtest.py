"""Schemas for walk-forward backtesting and strategy performance analytics."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class WalkForwardBacktestRequest(BaseModel):
    returns: list[list[float]] = Field(..., min_length=40, description="T x N historical returns matrix")
    asset_names: list[str] = Field(..., min_length=2, description="Names of the N assets")
    strategy: Literal["hrp", "equal_weight", "inverse_vol"] = Field(default="hrp")
    train_window: int = Field(default=60, ge=20, description="Rolling training lookback in days")
    rebalance_freq: int = Field(default=20, ge=1, description="Days between portfolio rebalancing")
    tx_cost_bps: float = Field(default=10.0, ge=0.0, description="Transaction fee in basis points")
    slippage_bps: float = Field(default=5.0, ge=0.0, description="Slippage friction in basis points")


class WalkForwardBacktestResponse(BaseModel):
    strategy: str
    n_periods: int
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    mean_turnover: float
    equity_curve: list[float]
    drawdowns: list[float]
    weights_snapshots: list[dict[str, float]]
