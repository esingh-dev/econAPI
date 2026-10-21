"""API router for live market data integration and automated engine calibrations."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from app.data import FREDProvider, YFinanceProvider
from app.engines.fixed_income import compute_curve_suite, fit_nss
from app.engines.volatility_surface import calibrate_svi_slice
from app.schemas.market import (
    HistoricalPricesResponse,
    LiveSVIRequest,
    LiveSVIResponse,
    LiveTreasuryCurveResponse,
    OptionChainResponse,
    QualityReportSchema,
)

router = APIRouter(prefix="/v1/market", tags=["Market Integration (v2.0)"])

yf_provider = YFinanceProvider()
fred_provider = FREDProvider()


@router.get("/prices", response_model=HistoricalPricesResponse)
def get_historical_prices(
    symbol: str = Query(..., description="Ticker symbol (e.g. SPY, AAPL, TLT)"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    interval: str = Query("1d", description="Bar interval: 1d, 1wk, 1mo"),
):
    """Retrieve historical prices with automated data quality checks (stale prices, outliers, missing values)."""
    try:
        series = yf_provider.get_historical_prices(symbol, start_date, end_date, interval)
        if not series.dates:
            raise HTTPException(status_code=404, detail=f"No price data found for symbol '{symbol}'.")

        qr = series.quality_report
        return HistoricalPricesResponse(
            symbol=series.symbol,
            dates=series.dates,
            open=series.open,
            high=series.high,
            low=series.low,
            close=series.close,
            volume=series.volume,
            adjusted_close=series.adjusted_close,
            quality_report=QualityReportSchema(
                is_valid=qr.is_valid,
                total_records=qr.total_records,
                missing_records_filled=qr.missing_records_filled,
                stale_price_detected=qr.stale_price_detected,
                anomalous_spikes=qr.anomalous_spikes,
                warnings=qr.warnings,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Market data fetch error: {str(e)}")


@router.get("/options-chain", response_model=OptionChainResponse)
def get_options_chain(
    symbol: str = Query(..., description="Underlying ticker (e.g. SPY, AAPL)"),
    expiration: Optional[str] = Query(None, description="Expiration date YYYY-MM-DD (defaults to nearest)"),
):
    """Retrieve full listed option chain with bids, asks, open interest, and implied volatility."""
    try:
        chain = yf_provider.get_option_chain(symbol, expiration)
        if not chain.calls and not chain.puts:
            raise HTTPException(status_code=404, detail=f"No options chain found for symbol '{symbol}'.")

        qr = chain.quality_report
        return OptionChainResponse(
            symbol=chain.symbol,
            spot_price=chain.spot_price,
            expiration_date=chain.expiration_date,
            available_expirations=chain.available_expirations,
            calls=[c.__dict__ for c in chain.calls],
            puts=[p.__dict__ for p in chain.puts],
            quality_report=QualityReportSchema(
                is_valid=qr.is_valid,
                total_records=qr.total_records,
                missing_records_filled=qr.missing_records_filled,
                stale_price_detected=qr.stale_price_detected,
                anomalous_spikes=qr.anomalous_spikes,
                warnings=qr.warnings,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Options chain fetch error: {str(e)}")


@router.post("/live-surface", response_model=LiveSVIResponse)
def calibrate_live_market_surface(req: LiveSVIRequest):
    """Calibrate Gatheral's SVI volatility surface model directly on live exchange option chains

    and verify Roger Lee wing bounds and Gatheral butterfly arbitrage conditions.
    """
    try:
        chain = yf_provider.get_option_chain(req.symbol, req.expiration)
        if not chain.calls:
            raise HTTPException(status_code=404, detail=f"Insufficient option quotes for '{req.symbol}'.")

        # Filter liquid calls with valid IV and positive bid
        liquid_quotes = [
            c for c in chain.calls
            if c.implied_volatility > 0.02 and c.implied_volatility < 2.0 and c.open_interest >= req.min_open_interest and c.bid > 0
        ]

        if len(liquid_quotes) < 5:
            # Fallback to all available calls
            liquid_quotes = [c for c in chain.calls if c.implied_volatility > 0.02][:15]

        if len(liquid_quotes) < 4:
            raise HTTPException(status_code=422, detail="Fewer than 4 liquid strikes available to fit SVI surface.")

        # Sort by strike
        liquid_quotes.sort(key=lambda x: x.strike)
        strikes = [q.strike for q in liquid_quotes]
        market_vols = [q.implied_volatility for q in liquid_quotes]

        # Calculate time to expiration
        now = datetime.utcnow()
        exp_dt = datetime.strptime(chain.expiration_date, "%Y-%m-%d")
        T = max(0.01, (exp_dt - now).days / 365.25)
        forward = chain.spot_price * (1.0 + 0.045 * T)

        # Fit SVI
        svi_fit = calibrate_svi_slice(strikes=strikes, implied_vols=market_vols, forward=forward, T=T)

        qr = chain.quality_report
        return LiveSVIResponse(
            symbol=chain.symbol,
            expiration_date=chain.expiration_date,
            spot_price=chain.spot_price,
            forward_estimate=round(forward, 2),
            strikes=strikes,
            market_vols=market_vols,
            svi_params={
                "a": svi_fit["a"],
                "b": svi_fit["b"],
                "rho": svi_fit["rho"],
                "m": svi_fit["m"],
                "sigma": svi_fit["sigma"],
            },
            fitted_vols=svi_fit["fitted_vols"],
            rmse=svi_fit["rmse"],
            roger_lee_satisfied=svi_fit["arbitrage_checks"].get("roger_lee_wings", {}).get("call_wing_ok", True),
            butterfly_arbitrage_free=svi_fit["arbitrage_checks"].get("butterfly_arbitrage_free", True),
            quality_report=QualityReportSchema(
                is_valid=qr.is_valid,
                total_records=len(strikes),
                missing_records_filled=qr.missing_records_filled,
                stale_price_detected=qr.stale_price_detected,
                anomalous_spikes=qr.anomalous_spikes,
                warnings=qr.warnings,
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Live SVI calibration failed: {str(e)}")


@router.get("/treasury-curve", response_model=LiveTreasuryCurveResponse)
def get_live_treasury_curve():
    """Retrieve active US Treasury yield curve from Federal Reserve (FRED)

    and calibrate Nelson-Siegel-Svensson term structure with discount factors.
    """
    try:
        curve = fred_provider.get_treasury_curve()
        nss_fit = fit_nss(tenors=curve.tenors, yields=curve.yields)
        curve_suite = compute_curve_suite(curve.tenors, nss_fit["fitted_yields"], rate_convention="continuous")
        discount_factors = curve_suite["discount_factors"]

        qr = curve.quality_report
        return LiveTreasuryCurveResponse(
            as_of_date=curve.as_of_date,
            source=curve.source,
            observed_tenors=curve.tenors,
            observed_yields=curve.yields,
            nss_fitted={
                "beta_0": nss_fit["beta_0"],
                "beta_1": nss_fit["beta_1"],
                "beta_2": nss_fit["beta_2"],
                "beta_3": nss_fit["beta_3"],
                "lambda_1": nss_fit["lambda_1"],
                "lambda_2": nss_fit["lambda_2"],
                "rmse": nss_fit["rmse"],
                "converged": nss_fit["converged"],
            },
            discount_factors=[round(df, 6) for df in discount_factors],
            quality_report=QualityReportSchema(
                is_valid=qr.is_valid,
                total_records=qr.total_records,
                missing_records_filled=qr.missing_records_filled,
                stale_price_detected=qr.stale_price_detected,
                anomalous_spikes=qr.anomalous_spikes,
                warnings=qr.warnings,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Treasury curve calculation failed: {str(e)}")
