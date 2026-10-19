"""API v1 — Risk analytics & macro-econometrics endpoints."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from app.engines import evt, macro_models
from app.schemas.risk import (
    EVTRequest,
    EVTResponse,
    GPDParams,
    NelsonSiegelRequest,
    NelsonSiegelResponse,
)

router = APIRouter(tags=["Risk & Macro"])


@router.post("/v1/risk/evt-cvar", response_model=EVTResponse)
def compute_evt_cvar(req: EVTRequest) -> EVTResponse:
    """Compute tail-risk VaR and CVaR via Extreme Value Theory (GPD/POT).

    Fits a Generalised Pareto Distribution to losses exceeding a high
    threshold and returns analytical VaR and Expected Shortfall.
    """
    returns_array = np.array(req.returns, dtype=float)
    result = evt.evt_analysis(
        returns_array, req.confidence_level, req.threshold_quantile
    )
    return EVTResponse(
        gpd_params=GPDParams(**result["gpd_params"]),
        var=result["var"],
        cvar=result["cvar"],
        confidence_level=result["confidence_level"],
    )


@router.post("/v1/macro/nelson-siegel", response_model=NelsonSiegelResponse)
def fit_yield_curve(req: NelsonSiegelRequest) -> NelsonSiegelResponse:
    """Fit the Nelson-Siegel parametric yield curve to observed market yields.

    Returns the four fitted parameters (β₀, β₁, β₂, λ), the fitted yields,
    and the root-mean-square fitting error.
    """
    tenors_array = np.array(req.tenors, dtype=float)
    yields_array = np.array(req.yields, dtype=float)
    result = macro_models.fit_nelson_siegel(tenors_array, yields_array)
    return NelsonSiegelResponse(**result)
