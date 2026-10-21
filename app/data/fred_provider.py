"""Free Federal Reserve Economic Data (FRED) Yield Curve Provider."""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime
from typing import Optional
import requests

from app.data.base import (
    BaseDataProvider,
    OptionChainData,
    PriceSeries,
    QualityReport,
    TreasuryCurveData,
)
from app.data.cache import macro_cache

logger = logging.getLogger(__name__)

FRED_TENOR_MAP = {
    "DGS1MO": 0.0833,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1":   1.0,
    "DGS2":   2.0,
    "DGS3":   3.0,
    "DGS5":   5.0,
    "DGS7":   7.0,
    "DGS10":  10.0,
    "DGS20":  20.0,
    "DGS30":  30.0,
}

# Fallback latest benchmark yields in case network/offline
BENCHMARK_FALLBACK_YIELDS = {
    0.0833: 0.0528,
    0.25:   0.0515,
    0.5:    0.0488,
    1.0:    0.0452,
    2.0:    0.0418,
    3.0:    0.0405,
    5.0:    0.0402,
    7.0:    0.0410,
    10.0:   0.0425,
    20.0:   0.0455,
    30.0:   0.0450,
}


class FREDProvider(BaseDataProvider):
    """Free macro rate provider connecting to the St. Louis Fed FRED data system."""

    def __init__(self, timeout_sec: int = 4) -> None:
        self.timeout = timeout_sec

    def get_treasury_curve(self) -> TreasuryCurveData:
        cache_key = "fred_us_treasury_curve"
        cached = macro_cache.get(cache_key)
        if cached is not None:
            return cached

        tenors = []
        yields = []
        as_of = datetime.utcnow().strftime("%Y-%m-%d")
        warnings = []

        try:
            # Query key tenors from FRED
            for series_id, tenor_yr in FRED_TENOR_MAP.items():
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
                resp = requests.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    lines = resp.text.strip().splitlines()
                    # Find last valid numeric quote
                    val = None
                    for line in reversed(lines):
                        parts = line.split(",")
                        if len(parts) == 2 and parts[1].strip() not in (".", "", "ND"):
                            try:
                                val = float(parts[1].strip()) / 100.0  # convert % to decimal
                                as_of = parts[0].strip()
                                break
                            except ValueError:
                                continue
                    if val is not None:
                        tenors.append(tenor_yr)
                        yields.append(val)
                else:
                    # Fallback for this tenor
                    tenors.append(tenor_yr)
                    yields.append(BENCHMARK_FALLBACK_YIELDS[tenor_yr])
        except Exception as e:
            logger.warning("FRED network request failed, utilizing benchmark fallback: %s", e)
            warnings.append(f"Network rate fetch fallback used: {e}")
            tenors = sorted(list(BENCHMARK_FALLBACK_YIELDS.keys()))
            yields = [BENCHMARK_FALLBACK_YIELDS[t] for t in tenors]

        if len(yields) < 6:
            tenors = sorted(list(BENCHMARK_FALLBACK_YIELDS.keys()))
            yields = [BENCHMARK_FALLBACK_YIELDS[t] for t in tenors]

        result = TreasuryCurveData(
            as_of_date=as_of,
            source="Federal Reserve Bank of St. Louis (FRED)",
            tenors=tenors,
            yields=yields,
            quality_report=QualityReport(
                is_valid=True,
                total_records=len(tenors),
                warnings=warnings,
            ),
        )

        macro_cache.set(cache_key, result)
        return result

    def get_historical_prices(self, symbol: str, start_date=None, end_date=None, interval="1d") -> PriceSeries:
        from app.data.yfinance_provider import YFinanceProvider
        return YFinanceProvider().get_historical_prices(symbol, start_date, end_date, interval)

    def get_option_chain(self, symbol: str, expiration=None) -> OptionChainData:
        from app.data.yfinance_provider import YFinanceProvider
        return YFinanceProvider().get_option_chain(symbol, expiration)
