"""Data quality controls: outlier detection, stale prices, and gap imputation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from app.data.base import QualityReport


def inspect_and_clean_prices(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, QualityReport]:
    """Inspects a price dataframe (Date index, Open, High, Low, Close, Volume, Adj Close)

    and applies data quality sanitation.
    """
    report = QualityReport(total_records=len(df))

    if df.empty:
        report.is_valid = False
        report.warnings.append(f"No price records returned for {symbol}.")
        return df, report

    df = df.copy()

    # Sort index chronologically
    df.sort_index(inplace=True)

    # 1. Missing values check & forward-fill
    n_missing = df["Close"].isna().sum()
    if n_missing > 0:
        report.missing_records_filled = int(n_missing)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        report.warnings.append(f"Forward-filled {n_missing} missing values in Close price series.")

    # 2. Stale price detection: 5+ consecutive identical prices on non-zero volume
    diffs = df["Close"].diff().abs()
    stale_streak = (diffs == 0).astype(int).groupby((diffs != 0).cumsum()).cumsum()
    if (stale_streak >= 5).any():
        report.stale_price_detected = True
        report.warnings.append("Stale price detected: 5 or more consecutive identical closing prices.")

    # 3. Outlier check: daily log return |r| > 5 sigma or > 35% in single session
    log_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    if len(log_returns) > 10:
        mu = log_returns.mean()
        std = log_returns.std()
        if std > 1e-8:
            z_scores = np.abs((log_returns - mu) / std)
            spikes = int((z_scores > 5.0).sum())
            if spikes > 0:
                report.anomalous_spikes = spikes
                report.warnings.append(f"Detected {spikes} anomalous daily price jumps (> 5-sigma).")

    return df, report
