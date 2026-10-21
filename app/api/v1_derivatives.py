"""API v1 — Derivatives pricing, Trees, Exotics, Monte Carlo, and Volatility Surfaces."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from app.engines import black_scholes, exotics, heston, monte_carlo, trees, volatility_surface
from app.schemas.derivatives import (
    AsianOptionRequest,
    AsianOptionResponse,
    BarrierOptionRequest,
    BarrierOptionResponse,
    BinaryOptionRequest,
    BinaryOptionResponse,
    BSMRequest,
    BSMResponse,
    CalendarArbitrageRequest,
    CalendarArbitrageResponse,
    HestonCalibrationRequest,
    HestonCalibrationResponse,
    HestonRequest,
    HestonResponse,
    MonteCarloRequest,
    MonteCarloResponse,
    SVICalibrationRequest,
    SVICalibrationResponse,
    TreeOptionRequest,
    TreeOptionResponse,
)

router = APIRouter(prefix="/v1/derivatives", tags=["Derivatives"])


# ═════════════════════════════════════════════════════════════════════
# 1. BSM & Greeks with Safeguards
# ═════════════════════════════════════════════════════════════════════

@router.post("/bsm", response_model=BSMResponse)
def price_bsm(req: BSMRequest) -> BSMResponse:
    """Price European options under generalized BSM with continuous dividend yield q.

    Returns analytical price, first-order Greeks (Delta, Gamma, Vega, Theta, Rho),
    higher-order Greeks (Vanna, Volga, Charm, Veta, Color, Speed, Ultima),
    no-arbitrage bound checks, and optional numerical finite-difference validation.
    """
    result = black_scholes.bsm_full(
        req.S, req.K, req.T, req.r, req.sigma, req.option_type, req.q, req.validate_greeks, extended=True
    )
    return BSMResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 2. Heston Stochastic Volatility & Calibration
# ═════════════════════════════════════════════════════════════════════

@router.post("/heston", response_model=HestonResponse)
def price_heston(req: HestonRequest) -> HestonResponse:
    """Price European options under the Heston stochastic volatility model.

    Uses Albrecher et al. (2007) continuous Little Trap characteristic function formulation
    with adaptive Gauss-Kronrod quadrature, Feller condition verification, and robust implied volatility.
    """
    result = heston.heston_full(
        req.S, req.K, req.T, req.r,
        req.v0, req.kappa, req.theta, req.xi, req.rho,
        req.option_type, req.q, extended=True,
    )
    return HestonResponse(**result)


@router.post("/heston/calibrate", response_model=HestonCalibrationResponse)
def calibrate_heston_endpoint(req: HestonCalibrationRequest) -> HestonCalibrationResponse:
    """Calibrate Heston model parameters (v0, kappa, theta, xi, rho) to market option quotes."""
    market_list = [q.model_dump() for q in req.market_data]
    result = heston.calibrate_heston(
        market_list, req.S, req.r, req.q, req.initial_guess
    )
    return HestonCalibrationResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 3. Binomial & Trinomial Trees
# ═════════════════════════════════════════════════════════════════════

@router.post("/tree", response_model=TreeOptionResponse)
def price_tree_option(req: TreeOptionRequest) -> TreeOptionResponse:
    """Price European or American options via Binomial (CRR / Jarrow-Rudd) or Trinomial Trees.

    Supports continuous dividend yield q and early exercise premium extraction.
    """
    if req.tree_type == "trinomial":
        result = trees.trinomial_tree_price(
            req.S, req.K, req.T, req.r, req.sigma, req.steps, req.option_type, req.style, req.q
        )
    else:
        result = trees.binomial_tree_price(
            req.S, req.K, req.T, req.r, req.sigma, req.steps, req.option_type, req.style, req.tree_type, req.q
        )
    return TreeOptionResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 4. Exotic Derivatives: Asian, Barrier, Binary
# ═════════════════════════════════════════════════════════════════════

@router.post("/asian", response_model=AsianOptionResponse)
def price_asian_option(req: AsianOptionRequest) -> AsianOptionResponse:
    """Price Asian options (Geometric exact Kemna-Vorst or Arithmetic Turnbull-Wakeman approximation)."""
    if req.averaging_type == "geometric":
        result = exotics.geometric_asian_price(
            req.S, req.K, req.T, req.r, req.sigma, req.option_type, req.q, req.n_obs
        )
    else:
        result = exotics.arithmetic_asian_price(
            req.S, req.K, req.T, req.r, req.sigma, req.option_type, req.q, req.n_obs
        )
    return AsianOptionResponse(**result)


@router.post("/barrier", response_model=BarrierOptionResponse)
def price_barrier_option(req: BarrierOptionRequest) -> BarrierOptionResponse:
    """Price single barrier options (Up/Down, In/Out) via Reiner-Rubinstein analytical formulas with rebate."""
    result = exotics.barrier_option_price(
        req.S, req.K, req.H, req.T, req.r, req.sigma, req.barrier_type, req.option_type, req.q, req.rebate
    )
    return BarrierOptionResponse(**result)


@router.post("/binary", response_model=BinaryOptionResponse)
def price_binary_option(req: BinaryOptionRequest) -> BinaryOptionResponse:
    """Price Cash-or-Nothing and Asset-or-Nothing digital/binary options analytically."""
    result = exotics.binary_option_price(
        req.S, req.K, req.T, req.r, req.sigma, req.payoff_type, req.option_type, req.cash_amount, req.q
    )
    return BinaryOptionResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 5. Monte Carlo Pricing Engine
# ═════════════════════════════════════════════════════════════════════

@router.post("/monte-carlo", response_model=MonteCarloResponse)
def price_monte_carlo(req: MonteCarloRequest) -> MonteCarloResponse:
    """Simulate option pricing via Monte Carlo with Antithetic and Control Variate variance reduction."""
    result = monte_carlo.monte_carlo_price(
        req.S, req.K, req.T, req.r, req.sigma, req.option_type, req.payoff_style,
        req.barrier_level, req.q, req.n_paths, req.n_steps,
        req.use_antithetic, req.use_control_variate, req.seed
    )
    return MonteCarloResponse(**result)


# ═════════════════════════════════════════════════════════════════════
# 6. Volatility Surfaces & SVI
# ═════════════════════════════════════════════════════════════════════

@router.post("/svi/calibrate", response_model=SVICalibrationResponse)
def calibrate_svi_endpoint(req: SVICalibrationRequest) -> SVICalibrationResponse:
    """Calibrate SVI raw slice parameters with Roger Lee wing bounds and butterfly arbitrage checks."""
    result = volatility_surface.calibrate_svi_slice(
        np.array(req.strikes), np.array(req.implied_vols), req.forward, req.T
    )
    return SVICalibrationResponse(**result)


@router.post("/svi/calendar-arbitrage", response_model=CalendarArbitrageResponse)
def check_calendar_arbitrage_endpoint(req: CalendarArbitrageRequest) -> CalendarArbitrageResponse:
    """Verify calendar spread arbitrage across term structure of SVI volatility slices."""
    result = volatility_surface.check_calendar_arbitrage(
        req.expiries, req.svi_slices
    )
    return CalendarArbitrageResponse(**result)
