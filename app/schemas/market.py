"""Pydantic v2 schemas for market data integration and live model calibrations."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class QualityReportSchema(BaseModel):
    is_valid: bool
    total_records: int
    missing_records_filled: int
    stale_price_detected: bool
    anomalous_spikes: int
    warnings: list[str]


class HistoricalPricesResponse(BaseModel):
    symbol: str
    dates: list[str]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]
    adjusted_close: list[float]
    quality_report: QualityReportSchema


class OptionItemSchema(BaseModel):
    strike: float
    option_type: str
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    moneyness: float


class OptionChainResponse(BaseModel):
    symbol: str
    spot_price: float
    expiration_date: str
    available_expirations: list[str]
    calls: list[OptionItemSchema]
    puts: list[OptionItemSchema]
    quality_report: QualityReportSchema


class LiveSVIRequest(BaseModel):
    symbol: str = Field(default="SPY", description="Equity/ETF ticker symbol")
    expiration: Optional[str] = Field(default=None, description="Optional target expiration date YYYY-MM-DD")
    min_open_interest: int = Field(default=5, ge=0, description="Liquidity filter for options")


class LiveSVIResponse(BaseModel):
    symbol: str
    expiration_date: str
    spot_price: float
    forward_estimate: float
    strikes: list[float]
    market_vols: list[float]
    svi_params: dict[str, float]
    fitted_vols: list[float]
    rmse: float
    roger_lee_satisfied: bool
    butterfly_arbitrage_free: bool
    quality_report: QualityReportSchema


class LiveTreasuryCurveResponse(BaseModel):
    as_of_date: str
    source: str
    observed_tenors: list[float]
    observed_yields: list[float]
    nss_fitted: dict
    discount_factors: list[float]
    quality_report: QualityReportSchema
