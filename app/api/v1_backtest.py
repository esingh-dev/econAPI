"""API router for walk-forward backtesting and strategy performance analytics."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from app.engines.backtester import run_walk_forward_backtest
from app.schemas.backtest import WalkForwardBacktestRequest, WalkForwardBacktestResponse

router = APIRouter(prefix="/v1/backtest", tags=["Backtesting & Research (v2.0)"])


@router.post("/walk-forward", response_model=WalkForwardBacktestResponse)
def execute_walk_forward_backtest(req: WalkForwardBacktestRequest):
    """Execute point-in-time walk-forward backtest avoiding look-ahead bias

    incorporating realistic slippage, transaction friction, and weight drift.
    """
    try:
        result = run_walk_forward_backtest(
            returns_matrix=req.returns,
            asset_names=req.asset_names,
            strategy=req.strategy,
            train_window=req.train_window,
            rebalance_freq=req.rebalance_freq,
            tx_cost_bps=req.tx_cost_bps,
            slippage_bps=req.slippage_bps,
        )
        return WalkForwardBacktestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Backtest execution error: {str(e)}")
