"""Free Yahoo Finance Market Data Adapter for Equities and Option Chains."""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np
import yfinance as yf

from app.data.base import (
    BaseDataProvider,
    OptionChainData,
    OptionQuoteItem,
    PriceSeries,
    QualityReport,
    TreasuryCurveData,
)
from app.data.cache import market_cache
from app.data.quality import inspect_and_clean_prices

logger = logging.getLogger(__name__)


class YFinanceProvider(BaseDataProvider):
    """Free market data provider using Yahoo Finance public endpoints."""

    def get_historical_prices(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> PriceSeries:
        cache_key = f"hist_{symbol}_{start_date}_{end_date}_{interval}"
        cached = market_cache.get(cache_key)
        if cached is not None:
            return cached

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
        )

        df, quality_report = inspect_and_clean_prices(df, symbol)

        dates = [d.strftime("%Y-%m-%d") for d in df.index]
        series = PriceSeries(
            symbol=symbol.upper(),
            dates=dates,
            open=df["Open"].tolist() if "Open" in df else [],
            high=df["High"].tolist() if "High" in df else [],
            low=df["Low"].tolist() if "Low" in df else [],
            close=df["Close"].tolist() if "Close" in df else [],
            volume=df["Volume"].tolist() if "Volume" in df else [],
            adjusted_close=df["Adj Close"].tolist() if "Adj Close" in df else df["Close"].tolist(),
            quality_report=quality_report,
        )

        market_cache.set(cache_key, series)
        return series

    def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None,
    ) -> OptionChainData:
        cache_key = f"opt_{symbol}_{expiration}"
        cached = market_cache.get(cache_key)
        if cached is not None:
            return cached

        ticker = yf.Ticker(symbol)
        expirations = ticker.options

        if not expirations:
            # Fallback for symbols without option chains
            return OptionChainData(
                symbol=symbol.upper(),
                spot_price=0.0,
                expiration_date="",
                available_expirations=[],
                calls=[],
                puts=[],
                quality_report=QualityReport(is_valid=False, warnings=[f"No listed options available for {symbol}"]),
            )

        target_exp = expiration if expiration in expirations else expirations[0]
        chain = ticker.option_chain(target_exp)

        # Get spot price
        fast_info = getattr(ticker, "fast_info", None)
        spot_price = float(fast_info.last_price) if fast_info and hasattr(fast_info, "last_price") and fast_info.last_price else 100.0

        def parse_items(df, opt_type: str) -> list[OptionQuoteItem]:
            items = []
            for _, row in df.iterrows():
                strike = float(row.get("strike", 0.0))
                moneyness = spot_price / strike if strike > 0 else 1.0
                items.append(
                    OptionQuoteItem(
                        strike=strike,
                        option_type=opt_type,
                        bid=float(row.get("bid", 0.0)),
                        ask=float(row.get("ask", 0.0)),
                        last_price=float(row.get("lastPrice", 0.0)),
                        volume=int(row.get("volume", 0)) if not np.isnan(row.get("volume", 0)) else 0,
                        open_interest=int(row.get("openInterest", 0)) if not np.isnan(row.get("openInterest", 0)) else 0,
                        implied_volatility=float(row.get("impliedVolatility", 0.0)),
                        moneyness=round(moneyness, 4),
                    )
                )
            return items

        calls = parse_items(chain.calls, "call")
        puts = parse_items(chain.puts, "put")

        result = OptionChainData(
            symbol=symbol.upper(),
            spot_price=spot_price,
            expiration_date=target_exp,
            available_expirations=list(expirations),
            calls=calls,
            puts=puts,
            quality_report=QualityReport(is_valid=True, total_records=len(calls) + len(puts)),
        )

        market_cache.set(cache_key, result)
        return result

    def get_treasury_curve(self) -> TreasuryCurveData:
        # Fallback to FRED provider
        from app.data.fred_provider import FREDProvider
        return FREDProvider().get_treasury_curve()
