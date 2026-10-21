"""API v1 — Fixed Income, Yield Curves, Bond Analytics, and Interest Rate Models."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from app.engines import fixed_income, macro_models
from app.schemas.fixed_income import (
    BondPriceRequest,
    BondPriceResponse,
    BootstrapRequest,
    BootstrapResponse,
    CIRRequest,
    CIRResponse,
    CurveShockRequest,
    CurveShockResponse,
    CurveSuiteRequest,
    CurveSuiteResponse,
    HullWhiteRequest,
    HullWhiteResponse,
    KRDRequest,
    KRDResponse,
    NelsonSiegelRequest,
    NelsonSiegelResponse,
    NSSRequest,
    NSSResponse,
    VasicekRequest,
    VasicekResponse,
)

router = APIRouter(prefix="/v1/fixed-income", tags=["Fixed Income & Yield Curves"])


# ═════════════════════════════════════════════════════════════════════
# 1. Curve Parametrization: Nelson-Siegel & Nelson-Siegel-Svensson
# ═════════════════════════════════════════════════════════════════════

@router.post("/nelson-siegel", response_model=NelsonSiegelResponse)
def fit_nelson_siegel_endpoint(req: NelsonSiegelRequest) -> NelsonSiegelResponse:
    """Fit 3-factor Nelson-Siegel term structure model with unit auto-detection and fit diagnostics."""
    result = macro_models.fit_nelson_siegel(
        np.array(req.tenors), np.array(req.yields), req.unit_convention, req.robust_loss
    )
    return NelsonSiegelResponse(**result)


@router.post("/nss", response_model=NSSResponse)
def fit_nss_endpoint(req: NSSRequest) -> NSSResponse:
    """Fit 4-factor / 6-parameter Nelson-Siegel-Svensson (NSS) parametric yield curve."""
    result = fixed_income.fit_nss(np.array(req.tenors), np.array(req.yields))
    return NSSResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 2. Curve Suite & Bootstrapping
# ═════════════════════════════════════════════════════════════════════

@router.post("/curve-suite", response_model=CurveSuiteResponse)
def compute_curve_suite_endpoint(req: CurveSuiteRequest) -> CurveSuiteResponse:
    """Generate Spot, Instantaneous Forward, Par yield curves, and Discount Factors."""
    result = fixed_income.compute_curve_suite(req.tenors, req.spot_rates, req.rate_convention)
    return CurveSuiteResponse(**result)


@router.post("/bootstrap", response_model=BootstrapResponse)
def bootstrap_zero_curve_endpoint(req: BootstrapRequest) -> BootstrapResponse:
    """Bootstrap a continuous zero-coupon yield curve from money market deposits, futures, and swaps."""
    inst_list = [inst.model_dump() for inst in req.instruments]
    result = fixed_income.bootstrap_zero_curve(inst_list)
    return BootstrapResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 3. Bond Pricing & Risk Analytics
# ═════════════════════════════════════════════════════════════════════

@router.post("/bond-price", response_model=BondPriceResponse)
def price_bond_endpoint(req: BondPriceRequest) -> BondPriceResponse:
    """Compute clean/dirty bond price, YTM (Brent root-finder), Macaulay/Modified duration, Convexity, and DV01."""
    result = fixed_income.price_bond(
        req.coupon_rate, req.maturity, req.ytm, req.clean_price,
        req.face_value, req.frequency, accrual_fraction=req.accrual_fraction
    )
    return BondPriceResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 4. Curve Scenarios & Key-Rate Duration
# ═════════════════════════════════════════════════════════════════════

@router.post("/curve-shock", response_model=CurveShockResponse)
def apply_curve_shock_endpoint(req: CurveShockRequest) -> CurveShockResponse:
    """Simulate macro yield curve shock scenarios: parallel, steepener, flattener, and butterfly twist."""
    result = fixed_income.apply_curve_shocks(
        req.tenors, req.yields, req.scenario, req.shift_bps, req.custom_shifts_bps
    )
    return CurveShockResponse(**result)


@router.post("/key-rate-duration", response_model=KRDResponse)
def compute_krd_endpoint(req: KRDRequest) -> KRDResponse:
    """Compute Key-Rate Duration (KRD) profile across benchmark curve tenors."""
    result = fixed_income.compute_key_rate_durations(
        req.coupon_rate, req.maturity, req.key_tenors, req.current_yield, req.face_value, req.frequency
    )
    return KRDResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 5. Interest Rate Term Structure Models
# ═════════════════════════════════════════════════════════════════════

@router.post("/models/vasicek", response_model=VasicekResponse)
def price_vasicek_endpoint(req: VasicekRequest) -> VasicekResponse:
    """Analytical zero-coupon bond pricing under the Vasicek short-rate model."""
    result = fixed_income.vasicek_bond_price(req.r0, req.T, req.a, req.b, req.sigma)
    return VasicekResponse(**result)


@router.post("/models/cir", response_model=CIRResponse)
def price_cir_endpoint(req: CIRRequest) -> CIRResponse:
    """Analytical zero-coupon bond pricing under the Cox-Ingersoll-Ross (CIR) square-root diffusion model."""
    result = fixed_income.cir_bond_price(req.r0, req.T, req.a, req.b, req.sigma)
    return CIRResponse(**result)


@router.post("/models/hull-white", response_model=HullWhiteResponse)
def price_hull_white_endpoint(req: HullWhiteRequest) -> HullWhiteResponse:
    """Analytical zero-coupon bond pricing under the 1-Factor Hull-White short-rate model."""
    result = fixed_income.hull_white_bond_price(
        req.r0, req.T, req.a, req.sigma, req.initial_forward_rate
    )
    return HullWhiteResponse(**result)
