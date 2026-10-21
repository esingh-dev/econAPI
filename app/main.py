"""Main application entry point for econAPI."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure root directory is always on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1_derivatives import router as derivatives_router
from app.api.v1_fixed_income import router as fixed_income_router
from app.api.v1_portfolio import router as portfolio_router
from app.api.v1_risk import router as risk_router
from app.api.v1_market import router as market_router
from app.api.v1_backtest import router as backtest_router
from app.api.v1_batch import router as batch_router
from app.config import settings


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="High-performance quantitative finance microservice supporting derivative pricing, portfolio optimization, and extreme risk analytics.",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time-Ms"],
)

# Lightweight, zero-overhead HTTP timing and auth middleware
@app.middleware("http")
async def process_time_and_auth_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    if settings.ENVIRONMENT.lower() == "production":
        exempt_prefixes = ("/docs", "/redoc", "/openapi.json", "/health", "/static")
        path = request.url.path
        is_exempt = path == "/" or any(path == prefix or path.startswith(prefix) for prefix in exempt_prefixes)
        is_dashboard_request = request.headers.get("sec-fetch-site") in ("same-origin", "same-site") or "localhost" in request.headers.get("host", "") or "127.0.0.1" in request.headers.get("host", "")
        if not is_exempt and not is_dashboard_request:
            api_key = request.headers.get("X-API-Key")
            valid_keys = settings.VALID_API_KEYS or ["econ_live_demo_key"]
            if not api_key or api_key not in valid_keys:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "detail": "Forbidden: Invalid or missing API Key. Provide 'X-API-Key: econ_live_demo_key' or your institutional key.",
                        "error_type": "AuthenticationError",
                    },
                )
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


# Global exception handlers
@app.exception_handler(ZeroDivisionError)
async def zero_division_handler(request: Request, exc: ZeroDivisionError) -> JSONResponse:
    """Handle zero division numerical exceptions."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": "Mathematical error: division by zero",
            "error_type": "ZeroDivisionError",
        },
    )


@app.exception_handler(np.linalg.LinAlgError)
async def linalg_error_handler(request: Request, exc: np.linalg.LinAlgError) -> JSONResponse:
    """Handle linear algebra exceptions from NumPy/SciPy."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": f"Linear algebra computation error: {str(exc)}",
            "error_type": "LinAlgError",
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle value validation and calculation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": str(exc),
            "error_type": "ValueError",
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught internal exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error_type": type(exc).__name__,
        },
    )


# Include API routers
app.include_router(derivatives_router)
app.include_router(fixed_income_router)
app.include_router(portfolio_router)
app.include_router(risk_router)
app.include_router(market_router)
app.include_router(backtest_router)
app.include_router(batch_router)


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Health check endpoint verifying microservice status and active version."""
    return {"status": "healthy", "version": settings.VERSION}


# Mount static assets directory
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Serve interactive showcase dashboard at root route
@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the interactive econAPI showcase web application."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "econAPI microservice is operational", "docs": "/docs"}
