"""Schemas for high-throughput batch pricing of options and bond portfolios."""

from __future__ import annotations

from pydantic import BaseModel, Field
from app.schemas.derivatives import BSMRequest, BSMResponse
from app.schemas.fixed_income import BondPriceRequest, BondPriceResponse


class BatchBSMRequest(BaseModel):
    items: list[BSMRequest] = Field(..., min_length=1, max_length=500, description="List of options to price")


class BatchBSMItemResult(BaseModel):
    index: int
    success: bool
    result: BSMResponse | None = None
    error: str | None = None


class BatchBSMResponse(BaseModel):
    total_requested: int
    total_succeeded: int
    compute_time_ms: float
    results: list[BatchBSMItemResult]


class BondPortfolioItem(BaseModel):
    id: str = Field(..., description="Unique bond or CUSIP identifier")
    position_size: float = Field(default=1.0, gt=0, description="Number of bonds held or face value multiplier")
    params: BondPriceRequest


class BatchBondPortfolioRequest(BaseModel):
    bonds: list[BondPortfolioItem] = Field(..., min_length=1, max_length=500)


class BatchBondPortfolioResponse(BaseModel):
    total_bonds: int
    portfolio_market_value: float
    portfolio_modified_duration: float
    portfolio_convexity: float
    portfolio_total_dv01: float
    individual_bonds: list[dict]
