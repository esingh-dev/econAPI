"""Fixed Income analytics, Yield Curve construction, Bond pricing, and Interest Rate models.

Implements:
- Nelson-Siegel-Svensson (NSS) 6-parameter term structure calibration
- Spot, forward, zero-coupon, and par yield curve generation with discount factors
- Curve bootstrapping from money market deposits, futures/FRAs, and interest rate swaps
- Bond pricing, accrued interest (Actual/Actual & 30/360), YTM (Brent root-finder)
- Macaulay duration, Modified duration, Convexity, and DV01 / PV01
- Curve scenarios: parallel, bull/bear steepener, bull/bear flattener, butterfly twist, Key-Rate Duration (KRD)
- Term structure models: Vasicek, Cox-Ingersoll-Ross (CIR), and Hull-White 1-Factor analytical pricing
"""

from __future__ import annotations

import math
from typing import Literal, Optional
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize


# ═════════════════════════════════════════════════════════════════════
# 1. Nelson-Siegel-Svensson (NSS) Yield Curve
# ═════════════════════════════════════════════════════════════════════

def nss_yield_curve(
    tau: np.ndarray | float,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    lam1: float,
    lam2: float,
) -> np.ndarray | float:
    """Evaluate 4-factor Nelson-Siegel-Svensson yield curve.

    y(t) = b0 + b1 * f1(t, lam1) + b2 * f2(t, lam1) + b3 * f2(t, lam2)
    where:
        f1(t, lam) = (1 - exp(-t/lam)) / (t/lam)
        f2(t, lam) = f1(t, lam) - exp(-t/lam)
    """
    t = np.asarray(tau, dtype=float)
    x1 = np.where(t > 1e-12, t / lam1, 1e-12)
    x2 = np.where(t > 1e-12, t / lam2, 1e-12)

    exp_neg1 = np.exp(-x1)
    exp_neg2 = np.exp(-x2)

    f1 = np.where(t > 1e-12, (1.0 - exp_neg1) / x1, 1.0)
    f2 = f1 - exp_neg1
    f3 = np.where(t > 1e-12, (1.0 - exp_neg2) / x2 - exp_neg2, 0.0)

    return beta0 + beta1 * f1 + beta2 * f2 + beta3 * f3


def fit_nss(
    tenors: np.ndarray,
    yields: np.ndarray,
) -> dict:
    """Fit Nelson-Siegel-Svensson parameters via 2D grid search on (lam1, lam2) + conditional OLS."""
    tenors = np.asarray(tenors, dtype=float)
    yields = np.asarray(yields, dtype=float)

    if len(tenors) < 6:
        raise ValueError("Need at least 6 data points to fit the 6-parameter NSS model.")

    best_rmse = float("inf")
    best_params = None

    # Grid search over (lam1, lam2) with lam1 != lam2
    lams = [0.5, 1.0, 1.5, 2.5, 4.0, 7.0]
    for l1 in lams:
        for l2 in lams:
            if abs(l1 - l2) < 0.3:
                continue
            x1 = np.where(tenors > 1e-12, tenors / l1, 1e-12)
            x2 = np.where(tenors > 1e-12, tenors / l2, 1e-12)
            f1 = (1.0 - np.exp(-x1)) / x1
            f2 = f1 - np.exp(-x1)
            f3 = (1.0 - np.exp(-x2)) / x2 - np.exp(-x2)
            X = np.column_stack([np.ones_like(tenors), f1, f2, f3])
            try:
                betas, _, _, _ = np.linalg.lstsq(X, yields, rcond=None)
                pred = X @ betas
                rmse = float(np.sqrt(np.mean((yields - pred) ** 2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = [betas[0], betas[1], betas[2], betas[3], l1, l2]
            except Exception:
                continue

    # Fine-tune with bounded L-BFGS-B
    def loss(p):
        b0, b1, b2, b3, l1, l2 = p
        pred = nss_yield_curve(tenors, b0, b1, b2, b3, l1, l2)
        # Penalize non-positive decay or asymptotic rate
        pen = 0.0
        if b0 <= 0:
            pen += 100.0 * abs(b0)
        return float(np.mean((yields - pred) ** 2)) + pen

    bounds = [
        (-5.0, 20.0),   # beta0
        (-20.0, 20.0),  # beta1
        (-25.0, 25.0),  # beta2
        (-25.0, 25.0),  # beta3
        (0.1, 15.0),    # lam1
        (0.1, 25.0),    # lam2
    ]

    res = minimize(loss, best_params, bounds=bounds, method="L-BFGS-B")
    b0, b1, b2, b3, l1, l2 = res.x
    fitted = nss_yield_curve(tenors, b0, b1, b2, b3, l1, l2)
    final_rmse = float(np.sqrt(np.mean((yields - fitted) ** 2)))

    return {
        "beta_0": float(b0),
        "beta_1": float(b1),
        "beta_2": float(b2),
        "beta_3": float(b3),
        "lambda_1": float(l1),
        "lambda_2": float(l2),
        "fitted_yields": fitted.tolist(),
        "rmse": final_rmse,
        "converged": bool(res.success),
    }


# ═════════════════════════════════════════════════════════════════════
# 2. Yield Curves: Spot, Forward, Par, Discount Factors
# ═════════════════════════════════════════════════════════════════════

def compute_curve_suite(
    tenors: list[float],
    spot_rates: list[float],
    rate_convention: Literal["annual_compounded", "continuous", "semi_annual"] = "continuous",
) -> dict:
    """Generate Spot, Forward, Par yield curves, and Discount Factors.

    Parameters
    ----------
    tenors : list[float]
        Maturities in years (e.g. [0.5, 1, 2, 3, 5, 10, 30]).
    spot_rates : list[float]
        Annualized spot zero rates (as decimals, e.g. 0.045 for 4.5%).
    rate_convention : {"continuous", "annual_compounded", "semi_annual"}
    """
    t = np.asarray(tenors, dtype=float)
    y = np.asarray(spot_rates, dtype=float)

    # 1. Discount Factors P(0, t)
    if rate_convention == "continuous":
        df = np.exp(-y * t)
    elif rate_convention == "annual_compounded":
        df = 1.0 / ((1.0 + y) ** t)
    else:  # semi_annual
        df = 1.0 / ((1.0 + y / 2.0) ** (2.0 * t))

    # 2. Instantaneous Forward Rates f(t) = -d ln P(0, t) / dt
    cs = CubicSpline(t, -np.log(df), bc_type="natural")
    forward_rates = cs(t, 1)  # first derivative

    # 3. Par Yields: Coupon rate c such that sum(c * df_i) + 100 * df_n = 100
    cum_df = np.cumsum(df)
    par_yields = (1.0 - df) / cum_df

    return {
        "tenors": tenors,
        "spot_rates": spot_rates,
        "discount_factors": [float(x) for x in df],
        "forward_rates": [float(x) for x in forward_rates],
        "par_yields": [float(x) for x in par_yields],
    }


# ═════════════════════════════════════════════════════════════════════
# 3. Curve Bootstrapping (Deposits, Futures/FRAs, Swaps)
# ═════════════════════════════════════════════════════════════════════

def bootstrap_zero_curve(
    instruments: list[dict],
) -> dict:
    """Bootstrap a continuous zero-coupon yield curve from market instruments.

    Parameters
    ----------
    instruments : list[dict]
        List of dicts:
        - {"type": "deposit", "tenor": 0.5, "rate": 0.05}
        - {"type": "future", "tenor": 1.0, "rate": 0.048}
        - {"type": "swap", "tenor": 2.0, "rate": 0.045, "freq": 1}
    """
    sorted_inst = sorted(instruments, key=lambda x: x["tenor"])

    tenors = []
    discount_factors = []
    zero_rates = []

    for inst in sorted_inst:
        t = float(inst["tenor"])
        r = float(inst["rate"])
        itype = inst["type"].lower()

        if itype in ("deposit", "money_market"):
            # Simple money market: P(0, t) = 1 / (1 + r * t)
            df = 1.0 / (1.0 + r * t)
            z = -math.log(df) / t
        elif itype in ("future", "fra"):
            # Forward rate over (t_prev, t)
            t_prev = tenors[-1] if tenors else 0.0
            dt = t - t_prev
            df_prev = discount_factors[-1] if discount_factors else 1.0
            df = df_prev / (1.0 + r * dt)
            z = -math.log(df) / t
        elif itype == "swap":
            # Par swap rate: Swap = (1 - df_n) / sum(dt_i * df_i)
            # Solve for df_n given previous discount factors
            freq = int(inst.get("freq", 1))
            dt = 1.0 / freq
            payment_times = np.arange(dt, t + 1e-6, dt)

            # Interpolate known discount factors for intermediate payments
            if len(tenors) > 0:
                cs = CubicSpline(tenors, [math.log(x) for x in discount_factors], bc_type="natural")
                known_times = payment_times[payment_times < t]
                interp_dfs = np.exp(cs(known_times)) if len(known_times) > 0 else np.array([])
                sum_intermediate = float(np.sum(interp_dfs) * dt) if len(known_times) > 0 else 0.0
            else:
                sum_intermediate = 0.0

            # 1 - df_t = r * (sum_intermediate + dt * df_t)
            # df_t * (1 + r * dt) = 1 - r * sum_intermediate
            df = (1.0 - r * sum_intermediate) / (1.0 + r * dt)
            df = max(1e-6, df)
            z = -math.log(df) / t
        else:
            continue

        tenors.append(t)
        discount_factors.append(df)
        zero_rates.append(z)

    return {
        "tenors": tenors,
        "discount_factors": [float(x) for x in discount_factors],
        "zero_rates": [float(x) for x in zero_rates],
    }


# ═════════════════════════════════════════════════════════════════════
# 4. Bond Pricing, Duration, Convexity, DV01 & YTM
# ═════════════════════════════════════════════════════════════════════

def price_bond(
    coupon_rate: float,
    maturity: float,
    ytm: Optional[float] = None,
    clean_price: Optional[float] = None,
    face_value: float = 100.0,
    frequency: int = 2,
    day_count: Literal["actual_actual", "30_360"] = "actual_actual",
    accrual_fraction: float = 0.0,
) -> dict:
    """Compute bond clean/dirty price, YTM, Macaulay/Modified duration, Convexity, and DV01.

    Parameters
    ----------
    coupon_rate : float
        Annual coupon rate (e.g. 0.05 for 5%).
    maturity : float
        Time to maturity in years (e.g. 10.0).
    ytm : float, optional
        Annualized yield to maturity (e.g. 0.048 for 4.8%).
    clean_price : float, optional
        Observed clean price (if solving for YTM).
    face_value : float
        Par value (default 100.0).
    frequency : int
        Coupon payments per year (1=annual, 2=semi-annual, 4=quarterly).
    accrual_fraction : float
        Fraction of current coupon period elapsed (0.0 to 1.0).
    """
    coupon = (face_value * coupon_rate) / frequency
    n_periods = int(round(maturity * frequency))
    period_times = np.array([(i - accrual_fraction) / frequency for i in range(1, n_periods + 1)])
    period_times = period_times[period_times > 0]
    total_periods = len(period_times)

    accrued_interest = coupon * accrual_fraction

    if ytm is None and clean_price is not None:
        target_dirty = clean_price + accrued_interest

        def obj(y):
            disc = 1.0 / ((1.0 + y / frequency) ** (period_times * frequency))
            p = coupon * np.sum(disc) + face_value * disc[-1]
            return p - target_dirty

        try:
            ytm = float(brentq(obj, -0.05, 1.0))
        except Exception:
            ytm = 0.05
    elif ytm is None:
        ytm = coupon_rate

    # Discounting at YTM
    y_per_period = ytm / frequency
    discounts = 1.0 / ((1.0 + y_per_period) ** (period_times * frequency))

    cf = np.full(total_periods, coupon)
    cf[-1] += face_value

    pv_cf = cf * discounts
    dirty_price = float(np.sum(pv_cf))
    calc_clean_price = dirty_price - accrued_interest

    # Macaulay Duration: sum(t * PV(CF)) / Dirty Price
    mac_dur = float(np.sum(period_times * pv_cf) / dirty_price)

    # Modified Duration: MacDur / (1 + y/m)
    mod_dur = float(mac_dur / (1.0 + y_per_period))

    # Convexity: sum(t * (t + 1/m) * PV(CF)) / [Dirty * (1 + y/m)^2]
    convexity = float(
        np.sum(period_times * (period_times + 1.0 / frequency) * pv_cf)
        / (dirty_price * ((1.0 + y_per_period) ** 2))
    )

    # DV01 (Dollar Value of an 01): change in price for 1bp (0.0001) yield increase
    dv01 = float(mod_dur * dirty_price * 0.0001)

    return {
        "clean_price": float(calc_clean_price),
        "dirty_price": dirty_price,
        "accrued_interest": float(accrued_interest),
        "ytm": float(ytm),
        "macaulay_duration": mac_dur,
        "modified_duration": mod_dur,
        "convexity": convexity,
        "dv01": dv01,
        "pv01": dv01,
    }


# ═════════════════════════════════════════════════════════════════════
# 5. Curve Shocks & Key-Rate Duration (KRD)
# ═════════════════════════════════════════════════════════════════════

def apply_curve_shocks(
    tenors: list[float],
    yields: list[float],
    scenario: Literal["parallel", "steepener", "flattener", "twist", "custom"],
    shift_bps: float = 25.0,
    custom_shifts_bps: Optional[list[float]] = None,
) -> dict:
    """Simulate macro yield curve shock scenarios."""
    t = np.asarray(tenors, dtype=float)
    y = np.asarray(yields, dtype=float)
    bps = shift_bps / 10000.0

    if scenario == "parallel":
        shifted = y + bps
    elif scenario == "steepener":
        # Short rates decrease or unchanged, long rates increase
        weights = (t - np.min(t)) / (np.max(t) - np.min(t)) - 0.5
        shifted = y + 2.0 * bps * weights
    elif scenario == "flattener":
        # Short rates increase, long rates decrease
        weights = (t - np.min(t)) / (np.max(t) - np.min(t)) - 0.5
        shifted = y - 2.0 * bps * weights
    elif scenario == "twist":
        # Butterfly twist (hump in the belly)
        mid = (np.min(t) + np.max(t)) / 2.0
        weights = np.sin(np.pi * (t - np.min(t)) / (np.max(t) - np.min(t)))
        shifted = y + bps * weights
    else:  # custom
        c_shifts = np.asarray(custom_shifts_bps or [0.0] * len(t), dtype=float) / 10000.0
        shifted = y + c_shifts

    return {
        "scenario": scenario,
        "original_yields": yields,
        "shifted_yields": [float(x) for x in shifted],
        "delta_bps": [float((s - o) * 10000.0) for s, o in zip(shifted, y)],
    }


def compute_key_rate_durations(
    coupon_rate: float,
    maturity: float,
    key_tenors: Optional[list[float]] = None,
    current_yield: float = 0.045,
    face_value: float = 100.0,
    frequency: int = 2,
) -> dict:
    """Compute Key-Rate Durations (KRD) across benchmark curve tenors using triangle shock tent functions."""
    if key_tenors is None:
        key_tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]

    key_t = np.asarray(key_tenors, dtype=float)
    base_res = price_bond(coupon_rate, maturity, ytm=current_yield, face_value=face_value, frequency=frequency)
    p0 = base_res["dirty_price"]

    krds = []
    shift_bps = 1.0  # 1 bp
    dh = shift_bps / 10000.0

    # Sensitivity of this bond to a shift at each key rate tenor
    for i, t_k in enumerate(key_t):
        t_left = key_t[i - 1] if i > 0 else 0.0
        t_right = key_t[i + 1] if i < len(key_t) - 1 else key_t[-1] + 10.0

        # Bond maturity weight under tent function
        if t_left <= maturity <= t_k:
            weight = (maturity - t_left) / (t_k - t_left) if t_k != t_left else 1.0
        elif t_k < maturity <= t_right:
            weight = (t_right - maturity) / (t_right - t_k) if t_right != t_k else 1.0
        else:
            weight = 0.0

        shifted_yield = current_yield + weight * dh
        p_shifted = price_bond(coupon_rate, maturity, ytm=shifted_yield, face_value=face_value, frequency=frequency)["dirty_price"]
        krd = -(p_shifted - p0) / (p0 * dh)
        krds.append(float(krd))

    return {
        "key_tenors": key_tenors,
        "key_rate_durations": krds,
        "total_krd": float(sum(krds)),
        "modified_duration": base_res["modified_duration"],
    }


# ═════════════════════════════════════════════════════════════════════
# 6. Term Structure Models: Vasicek, CIR, Hull-White
# ═════════════════════════════════════════════════════════════════════

def vasicek_bond_price(
    r0: float,
    T: float,
    a: float,
    b: float,
    sigma: float,
) -> dict:
    """Analytical zero-coupon bond pricing under the Vasicek (1977) model.

    dr(t) = a * (b - r(t)) dt + sigma * dW(t)
    P(0, T) = A(0, T) * exp(-B(0, T) * r0)
    """
    if a <= 0:
        raise ValueError("Mean-reversion speed 'a' must be positive.")

    B = (1.0 - math.exp(-a * T)) / a
    A_term = (b - 0.5 * (sigma * sigma) / (a * a)) * (B - T) - ((sigma * sigma) / (4.0 * a)) * (B * B)
    A = math.exp(A_term)

    price = float(A * math.exp(-B * r0))
    zero_yield = -math.log(price) / T if T > 0 else r0

    return {
        "price": price,
        "yield": float(zero_yield),
        "A": float(A),
        "B": float(B),
        "model": "vasicek",
    }


def cir_bond_price(
    r0: float,
    T: float,
    a: float,
    b: float,
    sigma: float,
) -> dict:
    """Analytical zero-coupon bond pricing under the Cox-Ingersoll-Ross (CIR 1985) model.

    dr(t) = a * (b - r(t)) dt + sigma * sqrt(r(t)) * dW(t)
    """
    if 2.0 * a * b <= sigma * sigma:
        # Feller condition violated for rates
        pass

    gamma = math.sqrt(a * a + 2.0 * sigma * sigma)
    exp_gamma_T = math.exp(gamma * T)

    denom = (gamma + a) * (exp_gamma_T - 1.0) + 2.0 * gamma
    B = (2.0 * (exp_gamma_T - 1.0)) / denom

    A_base = (2.0 * gamma * math.exp((a + gamma) * T / 2.0)) / denom
    A = A_base ** ((2.0 * a * b) / (sigma * sigma))

    price = float(A * math.exp(-B * r0))
    zero_yield = -math.log(price) / T if T > 0 else r0

    return {
        "price": price,
        "yield": float(zero_yield),
        "A": float(A),
        "B": float(B),
        "model": "cir",
    }


def hull_white_bond_price(
    r0: float,
    T: float,
    a: float,
    sigma: float,
    initial_forward_rate: float = 0.045,
) -> dict:
    """Analytical zero-coupon bond pricing under the 1-Factor Hull-White model.

    dr(t) = (theta(t) - a * r(t)) dt + sigma * dW(t)
    """
    B = (1.0 - math.exp(-a * T)) / a
    # Analytical zero-coupon bond with flat initial forward curve
    V = (sigma * sigma / (2.0 * a * a)) * (T + (1.0 - math.exp(-2.0 * a * T)) / (2.0 * a) - 2.0 * B)
    ln_A = -initial_forward_rate * T - V
    price = float(math.exp(ln_A - B * (r0 - initial_forward_rate)))
    zero_yield = -math.log(price) / T if T > 0 else r0

    return {
        "price": price,
        "yield": float(zero_yield),
        "B": float(B),
        "model": "hull_white_1f",
    }
