"""Abstract base classes and schemas for financial data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional


@dataclass
class QualityReport:
    is_valid: bool = True
    total_records: int = 0
    missing_records_filled: int = 0
    stale_price_detected: bool = False
    anomalous_spikes: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class PriceSeries:
    symbol: str
    dates: list[str]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]
    adjusted_close: list[float]
    quality_report: QualityReport = field(default_factory=QualityReport)


@dataclass
class OptionQuoteItem:
    strike: float
    option_type: str  # "call" | "put"
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    moneyness: float  # S / K


@dataclass
class OptionChainData:
    symbol: str
    spot_price: float
    expiration_date: str
    available_expirations: list[str]
    calls: list[OptionQuoteItem]
    puts: list[OptionQuoteItem]
    quality_report: QualityReport = field(default_factory=QualityReport)


@dataclass
class TreasuryCurveData:
    as_of_date: str
    source: str
    tenors: list[float]          # in years (e.g. 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
    yields: list[float]          # annualized decimal (e.g. 0.045 for 4.5%)
    quality_report: QualityReport = field(default_factory=QualityReport)


class BaseDataProvider(ABC):
    """Unified abstract interface for market and macro financial data providers."""

    @abstractmethod
    def get_historical_prices(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> PriceSeries:
        """Fetch historical OHLCV and adjusted close series."""
        pass

    @abstractmethod
    def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None,
    ) -> OptionChainData:
        """Fetch option chain with strikes, bids, asks, IV, and open interest."""
        pass

    @abstractmethod
    def get_treasury_curve(self) -> TreasuryCurveData:
        """Fetch active sovereign government Treasury yield curve."""
        pass
