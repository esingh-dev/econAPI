"""API v1 — Derivatives pricing endpoints (BSM & Heston)."""

from __future__ import annotations

from fastapi import APIRouter

from app.engines import black_scholes, heston
from app.schemas.derivatives import (
    BSMRequest,
    BSMResponse,
    HestonRequest,
    HestonResponse,
)

router = APIRouter(prefix="/v1/derivatives", tags=["Derivatives"])


@router.post("/bsm", response_model=BSMResponse)
def price_bsm(req: BSMRequest) -> BSMResponse:
    """Price a European option using the Black-Scholes-Merton model.

    Returns the analytical price along with Delta, Gamma, Vega, Theta, and Rho.
    """
    result = black_scholes.bsm_full(
        req.S, req.K, req.T, req.r, req.sigma, req.option_type
    )
    return BSMResponse(**result)


@router.post("/heston", response_model=HestonResponse)
def price_heston(req: HestonRequest) -> HestonResponse:
    """Price a European option under the Heston stochastic volatility model.

    Returns the price and the BSM implied volatility backed out via Brent's method.
    """
    result = heston.heston_full(
        req.S, req.K, req.T, req.r,
        req.v0, req.kappa, req.theta, req.xi, req.rho,
        req.option_type,
    )
    return HestonResponse(**result)
