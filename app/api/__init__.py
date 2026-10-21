"""API router definitions for econAPI."""

from app.api.v1_derivatives import router as derivatives_router
from app.api.v1_fixed_income import router as fixed_income_router
from app.api.v1_portfolio import router as portfolio_router
from app.api.v1_risk import router as risk_router
from app.api.v1_market import router as market_router
from app.api.v1_backtest import router as backtest_router
from app.api.v1_batch import router as batch_router

__all__ = [
    "derivatives_router",
    "fixed_income_router",
    "portfolio_router",
    "risk_router",
    "market_router",
    "backtest_router",
    "batch_router",
]
