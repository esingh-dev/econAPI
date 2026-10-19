"""Pydantic v2 request/response schemas for derivatives endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


# ── Black-Scholes-Merton ────────────────────────────────────────────

class BSMRequest(BaseModel):
    """Input parameters for BSM analytical pricer."""

    S: float = Field(..., gt=0, description="Spot price of the underlying asset")
    K: float = Field(..., gt=0, description="Strike price")
    T: float = Field(..., gt=0, description="Time to maturity in years")
    r: float = Field(..., description="Continuously compounded risk-free rate")
    sigma: float = Field(..., gt=0, description="Annualized volatility")
    option_type: Literal["call", "put"] = Field(
        default="call", description="Option type"
    )


class BSMResponse(BaseModel):
    """BSM pricing result including price and all Greeks."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


# ── Heston Stochastic Volatility ────────────────────────────────────

class HestonRequest(BaseModel):
    """Input parameters for Heston model pricer."""

    S: float = Field(..., gt=0, description="Spot price")
    K: float = Field(..., gt=0, description="Strike price")
    T: float = Field(..., gt=0, description="Time to maturity in years")
    r: float = Field(..., description="Risk-free rate")
    v0: float = Field(..., gt=0, description="Initial instantaneous variance")
    kappa: float = Field(..., gt=0, description="Mean-reversion speed")
    theta: float = Field(..., gt=0, description="Long-run variance level")
    xi: float = Field(..., gt=0, description="Volatility of variance (vol-of-vol)")
    rho: float = Field(
        ..., ge=-1, le=1, description="Correlation between asset and variance processes"
    )
    option_type: Literal["call", "put"] = Field(
        default="call", description="Option type"
    )


class HestonResponse(BaseModel):
    """Heston pricing result."""

    price: float
    implied_volatility: float
