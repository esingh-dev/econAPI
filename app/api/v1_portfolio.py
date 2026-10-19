"""API v1 — Portfolio optimisation endpoints (HRP)."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from app.engines import hrp
from app.schemas.portfolio import HRPRequest, HRPResponse

router = APIRouter(prefix="/v1/portfolio", tags=["Portfolio"])


@router.post("/hrp", response_model=HRPResponse)
def optimize_hrp(req: HRPRequest) -> HRPResponse:
    """Compute Hierarchical Risk Parity portfolio weights.

    Accepts a 2-D return matrix and returns optimised weights,
    portfolio variance, and annualised Sharpe ratio.
    """
    returns_array = np.array(req.returns, dtype=float)
    result = hrp.hrp_optimize(returns_array, asset_names=req.asset_names)
    return HRPResponse(**result)
