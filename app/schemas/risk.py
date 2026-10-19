"""Pydantic v2 request/response schemas for risk & macro endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


# ── EVT / GPD tail-risk ──────────────────────────────────────────────

class EVTRequest(BaseModel):
    """Input for EVT Peaks-Over-Threshold analysis."""

    returns: list[float] = Field(
        ..., min_length=50, description="Historical return series (≥ 50 observations)"
    )
    confidence_level: float = Field(
        default=0.99, gt=0.5, lt=1.0, description="VaR/CVaR confidence level"
    )
    threshold_quantile: float = Field(
        default=0.95, gt=0.5, lt=1.0, description="POT threshold quantile"
    )


class GPDParams(BaseModel):
    """Fitted GPD parameters."""

    xi: float = Field(description="Shape parameter (ξ)")
    beta: float = Field(description="Scale parameter (β)")
    threshold: float = Field(description="Threshold value u")
    n_exceedances: int
    n_total: int


class EVTResponse(BaseModel):
    """EVT tail-risk analysis result."""

    gpd_params: GPDParams
    var: float = Field(description="Value-at-Risk")
    cvar: float = Field(description="Conditional VaR / Expected Shortfall")
    confidence_level: float


# ── Nelson-Siegel yield curve ────────────────────────────────────────

class NelsonSiegelRequest(BaseModel):
    """Input for Nelson-Siegel term structure fitting."""

    tenors: list[float] = Field(
        ..., min_length=4, description="Maturity tenors in years"
    )
    yields: list[float] = Field(
        ..., min_length=4, description="Observed market yields"
    )

    @model_validator(mode="after")
    def validate_lengths(self) -> "NelsonSiegelRequest":
        if len(self.tenors) != len(self.yields):
            raise ValueError(
                f"tenors ({len(self.tenors)}) and yields ({len(self.yields)}) "
                f"must have the same length."
            )
        return self


class NelsonSiegelResponse(BaseModel):
    """Nelson-Siegel fitted result."""

    beta_0: float
    beta_1: float
    beta_2: float
    lambda_param: float
    fitted_yields: list[float]
    rmse: float
