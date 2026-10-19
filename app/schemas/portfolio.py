"""Pydantic v2 request/response schemas for portfolio endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from typing import Optional


class HRPRequest(BaseModel):
    """Input for Hierarchical Risk Parity optimisation."""

    returns: list[list[float]] = Field(
        ...,
        description=(
            "2-D array of historical returns with shape (T, N). "
            "T observations (rows) across N assets (columns)."
        ),
    )
    asset_names: Optional[list[str]] = Field(
        default=None,
        description="Optional human-readable asset names (length must equal N).",
    )

    @model_validator(mode="after")
    def validate_returns(self) -> "HRPRequest":
        if len(self.returns) < 2:
            raise ValueError("Need at least 2 time periods (rows).")
        n_assets = len(self.returns[0])
        if n_assets < 2:
            raise ValueError("Need at least 2 assets (columns).")
        for i, row in enumerate(self.returns):
            if len(row) != n_assets:
                raise ValueError(
                    f"Row {i} has {len(row)} values but expected {n_assets}."
                )
        if self.asset_names is not None and len(self.asset_names) != n_assets:
            raise ValueError(
                f"asset_names length ({len(self.asset_names)}) "
                f"must match number of assets ({n_assets})."
            )
        return self


class HRPResponse(BaseModel):
    """HRP optimisation result."""

    weights: dict[str, float]
    portfolio_variance: float
    sharpe_ratio: float
