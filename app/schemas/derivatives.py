"""Pydantic v2 schemas for Derivatives pricing, Greeks, Trees, Exotics, Monte Carlo, and Volatility Surfaces."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ═════════════════════════════════════════════════════════════════════
# 1. Black-Scholes-Merton & Greeks
# ═════════════════════════════════════════════════════════════════════

class BSMRequest(BaseModel):
    """Input parameters for generalized Black-Scholes-Merton option pricing."""

    S: float = Field(..., gt=0, description="Spot price of underlying asset (currency units, e.g. 100.0)")
    K: float = Field(..., gt=0, description="Strike price (currency units, e.g. 100.0)")
    T: float = Field(..., ge=0, description="Time to expiration in years (e.g. 1.0 for 1 year)")
    r: float = Field(..., description="Risk-free interest rate as annualized decimal (e.g. 0.05 for 5%)")
    sigma: float = Field(..., ge=0, description="Annualized volatility as decimal (e.g. 0.20 for 20%)")
    option_type: Literal["call", "put"] = Field(default="call", description="Option type")
    q: float = Field(default=0.0, ge=0, description="Continuous dividend yield as decimal (e.g. 0.02 for 2%)")
    validate_greeks: bool = Field(default=False, description="Validate analytical Greeks against finite differences")


class GreekValidationItem(BaseModel):
    analytical: float
    numerical: float
    rel_error: float


class GreeksValidationReport(BaseModel):
    all_passed: bool
    max_rel_error: float
    checks: dict[str, GreekValidationItem]


class NoArbitrageBounds(BaseModel):
    valid: bool
    price: float
    lower_bound: float
    upper_bound: float
    violation: Optional[str] = None


class BSMResponse(BaseModel):
    """Output for generalized Black-Scholes-Merton pricing with first- and higher-order Greeks."""

    price: float = Field(description="Analytical option price")
    delta: float = Field(description="Delta (dV/dS)")
    gamma: float = Field(description="Gamma (d2V/dS2)")
    vega: float = Field(description="Vega (dV/dSigma)")
    theta: float = Field(description="Theta (-dV/dT annualized)")
    rho: float = Field(description="Rho (dV/dr)")
    vanna: Optional[float] = Field(default=None, description="Vanna (dVega/dS)")
    volga: Optional[float] = Field(default=None, description="Volga / Vomma (dVega/dSigma)")
    charm: Optional[float] = Field(default=None, description="Charm / Delta decay (-dDelta/dT)")
    veta: Optional[float] = Field(default=None, description="Veta (-dVega/dT)")
    color: Optional[float] = Field(default=None, description="Color (dGamma/dT)")
    speed: Optional[float] = Field(default=None, description="Speed (dGamma/dS)")
    ultima: Optional[float] = Field(default=None, description="Ultima (dVolga/dSigma)")
    d1: Optional[float] = None
    d2: Optional[float] = None
    no_arbitrage_bounds: Optional[NoArbitrageBounds] = None
    greeks_validation: Optional[GreeksValidationReport] = None
    warnings: list[str] = Field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════
# 2. Heston Stochastic Volatility & Calibration
# ═════════════════════════════════════════════════════════════════════

class HestonRequest(BaseModel):
    """Input for Heston stochastic volatility pricing."""

    S: float = Field(..., gt=0, description="Spot price")
    K: float = Field(..., gt=0, description="Strike price")
    T: float = Field(..., gt=0, description="Time to expiration in years")
    r: float = Field(..., description="Annualized risk-free rate as decimal")
    v0: float = Field(..., gt=0, description="Initial instantaneous variance (e.g. 0.04)")
    kappa: float = Field(..., gt=0, description="Mean-reversion rate (e.g. 1.5)")
    theta: float = Field(..., gt=0, description="Long-run variance (e.g. 0.04)")
    xi: float = Field(..., gt=0, description="Volatility of variance (e.g. 0.3)")
    rho: float = Field(..., ge=-1.0, le=1.0, description="Spot-variance correlation in [-1, 1]")
    option_type: Literal["call", "put"] = Field(default="call")
    q: float = Field(default=0.0, ge=0, description="Continuous dividend yield")


class HestonResponse(BaseModel):
    """Heston model option price and convergence diagnostics."""

    price: float
    implied_volatility: float
    quadrature_error: Optional[float] = None
    feller_satisfied: bool
    converged: bool = True
    warnings: list[str] = Field(default_factory=list)


class MarketQuote(BaseModel):
    K: float = Field(..., gt=0)
    T: float = Field(..., gt=0)
    price: float = Field(..., ge=0)
    option_type: Literal["call", "put"] = "call"


class HestonCalibrationRequest(BaseModel):
    market_data: list[MarketQuote] = Field(..., min_length=5)
    S: float = Field(..., gt=0)
    r: float
    q: float = 0.0
    initial_guess: Optional[dict[str, float]] = None


class HestonCalibrationResponse(BaseModel):
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    rmse: float
    feller_satisfied: bool
    converged: bool
    iterations: int
    warnings: list[str] = Field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════
# 3. Binomial & Trinomial Trees
# ═════════════════════════════════════════════════════════════════════

class TreeOptionRequest(BaseModel):
    S: float = Field(..., gt=0)
    K: float = Field(..., gt=0)
    T: float = Field(..., gt=0)
    r: float
    sigma: float = Field(..., gt=0)
    steps: int = Field(default=200, ge=10, le=2000)
    option_type: Literal["call", "put"] = "call"
    style: Literal["european", "american"] = "american"
    tree_type: Literal["crr", "jarrow_rudd", "trinomial"] = "crr"
    q: float = 0.0


class TreeOptionResponse(BaseModel):
    price: float
    european_price: float
    early_exercise_premium: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    tree_type: str
    steps: int
    style: str


# ═════════════════════════════════════════════════════════════════════
# 4. Exotic Options: Asian, Barrier, Binary
# ═════════════════════════════════════════════════════════════════════

class AsianOptionRequest(BaseModel):
    S: float = Field(..., gt=0)
    K: float = Field(..., gt=0)
    T: float = Field(..., gt=0)
    r: float
    sigma: float = Field(..., gt=0)
    option_type: Literal["call", "put"] = "call"
    averaging_type: Literal["geometric", "arithmetic"] = "arithmetic"
    q: float = 0.0
    n_obs: int = Field(default=252, ge=2, le=1000)


class AsianOptionResponse(BaseModel):
    price: float
    averaging_type: str
    option_type: str
    sigma_adj: Optional[float] = None
    expected_average: Optional[float] = None
    implied_vol_avg: Optional[float] = None


class BarrierOptionRequest(BaseModel):
    S: float = Field(..., gt=0)
    K: float = Field(..., gt=0)
    H: float = Field(..., gt=0, description="Barrier level")
    T: float = Field(..., gt=0)
    r: float
    sigma: float = Field(..., gt=0)
    barrier_type: Literal["up_and_out", "up_and_in", "down_and_out", "down_and_in"]
    option_type: Literal["call", "put"] = "call"
    q: float = 0.0
    rebate: float = Field(default=0.0, ge=0)


class BarrierOptionResponse(BaseModel):
    price: float
    barrier_type: str
    option_type: str
    barrier_level: float
    rebate: float
    breached_at_start: bool


class BinaryOptionRequest(BaseModel):
    S: float = Field(..., gt=0)
    K: float = Field(..., gt=0)
    T: float = Field(..., ge=0)
    r: float
    sigma: float = Field(..., gt=0)
    payoff_type: Literal["cash_or_nothing", "asset_or_nothing"] = "cash_or_nothing"
    option_type: Literal["call", "put"] = "call"
    cash_amount: float = Field(default=1.0, gt=0)
    q: float = 0.0


class BinaryOptionResponse(BaseModel):
    price: float
    delta: float
    payoff_type: str
    option_type: str


# ═════════════════════════════════════════════════════════════════════
# 5. Monte Carlo Pricing
# ═════════════════════════════════════════════════════════════════════

class MonteCarloRequest(BaseModel):
    S: float = Field(..., gt=0)
    K: float = Field(..., gt=0)
    T: float = Field(..., gt=0)
    r: float
    sigma: float = Field(..., gt=0)
    option_type: Literal["call", "put"] = "call"
    payoff_style: Literal["european", "asian_arithmetic", "barrier_up_out", "barrier_down_out"] = "european"
    barrier_level: Optional[float] = None
    q: float = 0.0
    n_paths: int = Field(default=50_000, ge=1000, le=500_000)
    n_steps: int = Field(default=100, ge=10, le=1000)
    use_antithetic: bool = True
    use_control_variate: bool = True
    seed: Optional[int] = 42


class MonteCarloResponse(BaseModel):
    price: float
    standard_error: float
    ci_95_lower: float
    ci_95_upper: float
    n_paths: int
    n_steps: int
    antithetic_used: bool
    control_variate_used: bool
    payoff_style: str


# ═════════════════════════════════════════════════════════════════════
# 6. Volatility Surface & SVI
# ═════════════════════════════════════════════════════════════════════

class SVICalibrationRequest(BaseModel):
    strikes: list[float] = Field(..., min_length=4)
    implied_vols: list[float] = Field(..., min_length=4)
    forward: float = Field(..., gt=0)
    T: float = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_lengths(self) -> "SVICalibrationRequest":
        if len(self.strikes) != len(self.implied_vols):
            raise ValueError("strikes and implied_vols must have identical lengths.")
        return self


class SVICalibrationResponse(BaseModel):
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    rmse: float
    fitted_vols: list[float]
    converged: bool
    arbitrage_checks: dict


class CalendarArbitrageRequest(BaseModel):
    expiries: list[float] = Field(..., min_length=2)
    svi_slices: list[dict] = Field(..., min_length=2)


class CalendarArbitrageResponse(BaseModel):
    calendar_arbitrage_free: bool
    num_slice_violations: int
    max_variance_drop: float
    warnings: list[str] = Field(default_factory=list)
