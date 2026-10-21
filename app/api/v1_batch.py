"""API router for high-throughput batch option pricing and bond portfolio risk decomposition."""

from __future__ import annotations

import time
from fastapi import APIRouter
from app.engines.black_scholes import bsm_full
from app.engines.fixed_income import price_bond
from app.schemas.batch import (
    BatchBondPortfolioRequest,
    BatchBondPortfolioResponse,
    BatchBSMItemResult,
    BatchBSMRequest,
    BatchBSMResponse,
)
from app.schemas.derivatives import BSMResponse

router = APIRouter(prefix="/v1/batch", tags=["Batch Processing (v2.0)"])


@router.post("/bsm", response_model=BatchBSMResponse)
def batch_price_options(req: BatchBSMRequest):
    """Batch-price up to 500 options concurrently with complete Greeks."""
    t0 = time.perf_counter()
    results: list[BatchBSMItemResult] = []
    success_count = 0

    for idx, item in enumerate(req.items):
        try:
            res = bsm_full(
                S=item.S,
                K=item.K,
                T=item.T,
                r=item.r,
                sigma=item.sigma,
                option_type=item.option_type,
                q=item.q,
                validate_greeks=item.validate_greeks,
            )
            parsed = BSMResponse(**res)
            results.append(BatchBSMItemResult(index=idx, success=True, result=parsed))
            success_count += 1
        except Exception as e:
            results.append(BatchBSMItemResult(index=idx, success=False, error=str(e)))

    compute_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return BatchBSMResponse(
        total_requested=len(req.items),
        total_succeeded=success_count,
        compute_time_ms=compute_ms,
        results=results,
    )


@router.post("/bonds", response_model=BatchBondPortfolioResponse)
def batch_price_bond_portfolio(req: BatchBondPortfolioRequest):
    """Price an entire bond portfolio and compute aggregated market value, portfolio duration, and total DV01."""
    total_val = 0.0
    weighted_dur = 0.0
    weighted_cvx = 0.0
    total_dv01 = 0.0
    individual_out = []

    for item in req.bonds:
        p = item.params
        res = price_bond(
            coupon_rate=p.coupon_rate,
            maturity=p.maturity,
            ytm=p.ytm,
            clean_price=p.clean_price,
            face_value=p.face_value,
            frequency=p.frequency,
            accrual_fraction=p.accrual_fraction,
        )
        pos_val = res["dirty_price"] * item.position_size
        total_val += pos_val
        total_dv01 += res["dv01"] * item.position_size

        individual_out.append({
            "id": item.id,
            "position_size": item.position_size,
            "clean_price": res["clean_price"],
            "dirty_price": res["dirty_price"],
            "accrued_interest": res["accrued_interest"],
            "ytm": res["ytm"],
            "modified_duration": res["modified_duration"],
            "convexity": res["convexity"],
            "dv01": res["dv01"],
            "position_market_value": round(pos_val, 2),
        })

    if total_val > 0:
        for b in individual_out:
            w = b["position_market_value"] / total_val
            weighted_dur += w * b["modified_duration"]
            weighted_cvx += w * b["convexity"]

    return BatchBondPortfolioResponse(
        total_bonds=len(req.bonds),
        portfolio_market_value=round(total_val, 2),
        portfolio_modified_duration=round(weighted_dur, 4),
        portfolio_convexity=round(weighted_cvx, 4),
        portfolio_total_dv01=round(total_dv01, 4),
        individual_bonds=individual_out,
    )
