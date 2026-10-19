"""Main application entry point for econAPI."""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.api.v1_derivatives import router as derivatives_router
from app.api.v1_portfolio import router as portfolio_router
from app.api.v1_risk import router as risk_router
from app.config import settings


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing API key validation in production environments."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if settings.ENVIRONMENT.lower() == "production":
            exempt_prefixes = ("/docs", "/redoc", "/openapi.json", "/health")
            path = request.url.path
            if not any(path == prefix or path.startswith(prefix) for prefix in exempt_prefixes):
                api_key = request.headers.get("X-API-Key")
                if not api_key or api_key not in settings.VALID_API_KEYS:
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={
                            "detail": "Forbidden: Invalid or missing API Key",
                            "error_type": "AuthenticationError",
                        },
                    )
        return await call_next(request)


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
)

# API key authentication middleware
app.add_middleware(APIKeyAuthMiddleware)


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
app.include_router(portfolio_router)
app.include_router(risk_router)


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Health check endpoint verifying microservice status and active version."""
    return {"status": "healthy", "version": settings.VERSION}
