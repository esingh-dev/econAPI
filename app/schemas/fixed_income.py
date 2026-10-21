"""Pydantic v2 schemas for Fixed Income, Yield Curves, Bond Analytics, and Interest Rate Models."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ═════════════════════════════════════════════════════════════════════
# 1. Nelson-Siegel & Nelson-Siegel-Svensson
# ═════════════════════════════════════════════════════════════════════

class NelsonSiegelRequest(BaseModel):
    tenors: list[float] = Field(..., min_length=4, description="Maturity tenors in years")
    yields: list[float] = Field(..., min_length=4, description="Observed market yields")
    unit_convention: Literal["auto", "percent", "decimal"] = Field(default="auto")
    robust_loss: Literal["linear", "huber", "soft_l1"] = Field(default="linear")

    @model_validator(mode="after")
    def check_lengths(self) -> "NelsonSiegelRequest":
        if len(self.tenors) != len(self.yields):
            raise ValueError(f"tenors ({len(self.tenors)}) and yields ({len(self.yields)}) must match in length.")
        return self


class NelsonSiegelResponse(BaseModel):
    beta_0: float
    beta_1: float
    beta_2: float
    lambda_param: float
    fitted_yields: list[float]
    residuals: list[float]
    rmse: float
    mae: Optional[float] = None
    max_error: Optional[float] = None
    r_squared: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    unit_convention: str
    warnings: list[str] = Field(default_factory=list)


class NSSRequest(BaseModel):
    tenors: list[float] = Field(..., min_length=6, description="Maturity tenors in years (min 6)")
    yields: list[float] = Field(..., min_length=6, description="Observed market yields (min 6)")

    @model_validator(mode="after")
    def check_lengths(self) -> "NSSRequest":
        if len(self.tenors) != len(self.yields):
            raise ValueError("tenors and yields must have the same length.")
        return self


class NSSResponse(BaseModel):
    beta_0: float
    beta_1: float
    beta_2: float
    beta_3: float
    lambda_1: float
    lambda_2: float
    fitted_yields: list[float]
    rmse: float
    converged: bool


# ═════════════════════════════════════════════════════════════════════
# 2. Yield Curve Suite & Bootstrapping
# ═════════════════════════════════════════════════════════════════════

class CurveSuiteRequest(BaseModel):
    tenors: list[float] = Field(..., min_length=2)
    spot_rates: list[float] = Field(..., min_length=2, description="Annualized spot rates as decimals (e.g. 0.045)")
    rate_convention: Literal["continuous", "annual_compounded", "semi_annual"] = "continuous"


class CurveSuiteResponse(BaseModel):
    tenors: list[float]
    spot_rates: list[float]
    discount_factors: list[float]
    forward_rates: list[float]
    par_yields: list[float]


class BootstrapInstrument(BaseModel):
    type: Literal["deposit", "money_market", "future", "fra", "swap"]
    tenor: float = Field(..., gt=0)
    rate: float = Field(..., description="Annualized quote as decimal")
    freq: Optional[int] = Field(default=1, description="Payment frequency per year for swaps")


class BootstrapRequest(BaseModel):
    instruments: list[BootstrapInstrument] = Field(..., min_length=2)


class BootstrapResponse(BaseModel):
    tenors: list[float]
    discount_factors: list[float]
    zero_rates: list[float]


# ═════════════════════════════════════════════════════════════════════
# 3. Bond Pricing & Analytics
# ═════════════════════════════════════════════════════════════════════

class BondPriceRequest(BaseModel):
    coupon_rate: float = Field(..., ge=0, description="Annual coupon rate as decimal (e.g. 0.05 for 5%)")
    maturity: float = Field(..., gt=0, description="Maturity in years")
    ytm: Optional[float] = Field(default=None, description="Annualized YTM as decimal")
    clean_price: Optional[float] = Field(default=None, description="Clean price (if solving for YTM)")
    face_value: float = Field(default=100.0, gt=0)
    frequency: int = Field(default=2, ge=1, le=12)
    accrual_fraction: float = Field(default=0.0, ge=0.0, le=1.0)


class BondPriceResponse(BaseModel):
    clean_price: float
    dirty_price: float
    accrued_interest: float
    ytm: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float
    pv01: float


# ═════════════════════════════════════════════════════════════════════
# 4. Curve Shocks & Key-Rate Duration
# ═════════════════════════════════════════════════════════════════════

class CurveShockRequest(BaseModel):
    tenors: list[float] = Field(..., min_length=2)
    yields: list[float] = Field(..., min_length=2)
    scenario: Literal["parallel", "steepener", "flattener", "twist", "custom"]
    shift_bps: float = Field(default=25.0)
    custom_shifts_bps: Optional[list[float]] = None


class CurveShockResponse(BaseModel):
    scenario: str
    original_yields: list[float]
    shifted_yields: list[float]
    delta_bps: list[float]


class KRDRequest(BaseModel):
    coupon_rate: float = Field(..., ge=0)
    maturity: float = Field(..., gt=0)
    current_yield: float = Field(default=0.045)
    key_tenors: Optional[list[float]] = None
    face_value: float = 100.0
    frequency: int = 2


class KRDResponse(BaseModel):
    key_tenors: list[float]
    key_rate_durations: list[float]
    total_krd: float
    modified_duration: float


# ═════════════════════════════════════════════════════════════════════
# 5. Interest Rate Term Structure Models
# ═════════════════════════════════════════════════════════════════════

class VasicekRequest(BaseModel):
    r0: float = Field(..., description="Current short rate as decimal (e.g. 0.04)")
    T: float = Field(..., gt=0, description="Maturity in years")
    a: float = Field(..., gt=0, description="Mean-reversion speed")
    b: float = Field(..., description="Long-run mean rate as decimal")
    sigma: float = Field(..., gt=0, description="Short rate volatility")


class VasicekResponse(BaseModel):
    price: float
    yield_val: float = Field(alias="yield")
    A: float
    B: float
    model: str = "vasicek"


class CIRRequest(BaseModel):
    r0: float = Field(..., gt=0)
    T: float = Field(..., gt=0)
    a: float = Field(..., gt=0)
    b: float = Field(..., gt=0)
    sigma: float = Field(..., gt=0)


class CIRResponse(BaseModel):
    price: float
    yield_val: float = Field(alias="yield")
    A: float
    B: float
    model: str = "cir"


class HullWhiteRequest(BaseModel):
    r0: float = Field(..., description="Current short rate")
    T: float = Field(..., gt=0)
    a: float = Field(..., gt=0, description="Mean-reversion speed")
    sigma: float = Field(..., gt=0)
    initial_forward_rate: float = Field(default=0.045)


class HullWhiteResponse(BaseModel):
    price: float
    yield_val: float = Field(alias="yield")
    B: float
    model: str = "hull_white_1f"
