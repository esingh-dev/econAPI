"""Exotic derivatives analytical and approximation engine.

Implements:
- Asian options: Exact closed-form for geometric average (Kemna-Vorst) & Turnbull-Wakeman approximation for arithmetic average
- Barrier options: Complete Reiner-Rubinstein (1991) analytical solutions for all 8 barrier types (Up/Down, In/Out, Call/Put) with rebate
- Binary/Digital options: Cash-or-nothing and Asset-or-nothing calls and puts
"""

from __future__ import annotations

import math
from typing import Literal, Optional
import numpy as np
from scipy.stats import norm


# ═════════════════════════════════════════════════════════════════════
# 1. Asian Options
# ═════════════════════════════════════════════════════════════════════

def geometric_asian_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
    n_obs: int = 252,
) -> dict:
    """Exact closed-form Kemna-Vorst (1990) price for continuous/discrete Geometric Asian option.

    Parameters
    ----------
    S, K, T, r, sigma, q : standard inputs
    n_obs : int
        Number of averaging fixings over [0, T].
    """
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2.")

    # Kemna-Vorst adjusted volatility and drift for geometric mean
    sigma_g = sigma * math.sqrt((2.0 * n_obs + 1.0) / (6.0 * (n_obs + 1.0)))
    mu_g = (r - q - 0.5 * sigma * sigma) * 0.5 + 0.5 * sigma_g * sigma_g

    # Equivalent BSM parameters
    d1 = (math.log(S / K) + (mu_g + 0.5 * sigma_g * sigma_g) * T) / (sigma_g * math.sqrt(T))
    d2 = d1 - sigma_g * math.sqrt(T)

    disc_r = math.exp(-r * T)
    disc_mu = math.exp((mu_g - r) * T)

    if option_type == "call":
        price = S * disc_mu * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    else:
        price = K * disc_r * norm.cdf(-d2) - S * disc_mu * norm.cdf(-d1)

    return {
        "price": float(max(0.0, price)),
        "averaging_type": "geometric",
        "sigma_adj": float(sigma_g),
        "option_type": option_type,
    }


def arithmetic_asian_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
    n_obs: int = 252,
) -> dict:
    """Turnbull-Wakeman (1991) analytical two-moment approximation for Arithmetic Asian option."""
    dt = T / n_obs
    b = r - q

    # First moment M1: E[A_T]
    if abs(b) < 1e-8:
        M1 = S
    else:
        M1 = (S / (n_obs * (math.exp(b * dt) - 1.0))) * (math.exp(b * (T + dt)) - math.exp(b * dt))

    # Second moment M2: E[A_T^2]
    sig2 = sigma * sigma
    exp_b_dt = math.exp(b * dt)
    exp_2b_sig2_dt = math.exp((2.0 * b + sig2) * dt)

    sum_term = 0.0
    for i in range(1, n_obs + 1):
        for j in range(1, n_obs + 1):
            min_ij = min(i, j)
            max_ij = max(i, j)
            term = math.exp(b * (i + j) * dt + sig2 * min_ij * dt)
            sum_term += term

    M2 = (S * S / (n_obs * n_obs)) * sum_term

    # Approximate lognormal parameters for A_T
    var_A = max(1e-12, M2 - M1 * M1)
    sigma_a = math.sqrt(math.log(M2 / (M1 * M1)) / T)

    d1 = (math.log(M1 / K) + 0.5 * sigma_a * sigma_a * T) / (sigma_a * math.sqrt(T))
    d2 = d1 - sigma_a * math.sqrt(T)
    disc_r = math.exp(-r * T)

    if option_type == "call":
        price = disc_r * (M1 * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = disc_r * (K * norm.cdf(-d2) - M1 * norm.cdf(-d1))

    return {
        "price": float(max(0.0, price)),
        "averaging_type": "arithmetic",
        "expected_average": float(M1),
        "implied_vol_avg": float(sigma_a),
        "option_type": option_type,
    }


# ═════════════════════════════════════════════════════════════════════
# 2. Barrier Options (Reiner-Rubinstein 1991 analytical)
# ═════════════════════════════════════════════════════════════════════

def barrier_option_price(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    barrier_type: Literal["up_and_out", "up_and_in", "down_and_out", "down_and_in"],
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
    rebate: float = 0.0,
) -> dict:
    """Exact analytical pricing for all standard single barrier options (Reiner & Rubinstein 1991).

    Parameters
    ----------
    S, K, T, r, sigma, q : standard inputs
    H : float
        Barrier level (H > 0).
    barrier_type : {"up_and_out", "up_and_in", "down_and_out", "down_and_in"}
    option_type : {"call", "put"}
    rebate : float
        Cash rebate paid at expiry if the barrier condition is triggered (default: 0.0).
    """
    if H <= 0:
        raise ValueError("Barrier level H must be positive.")

    # Check immediate breaching at t=0
    is_up = "up" in barrier_type
    is_out = "out" in barrier_type

    if is_up and S >= H:
        # Already knocked in or out
        price = rebate * math.exp(-r * T) if is_out else 0.0  # vanilla payoff if knocked in
        # For knocked in, price standard vanilla
        from app.engines.black_scholes import bsm_price
        if not is_out:
            price = bsm_price(S, K, T, r, sigma, option_type, q)["price"]
        return {"price": float(price), "breached_at_start": True}

    if not is_up and S <= H:
        from app.engines.black_scholes import bsm_price
        if is_out:
            price = rebate * math.exp(-r * T)
        else:
            price = bsm_price(S, K, T, r, sigma, option_type, q)["price"]
        return {"price": float(price), "breached_at_start": True}

    # Reiner-Rubinstein analytical components
    sig2 = sigma * sigma
    sqrt_T = math.sqrt(T)
    b = r - q
    mu = (b - 0.5 * sig2) / sig2
    lam = math.sqrt(mu * mu + 2.0 * r / sig2)

    # Helper probabilities A, B, C, D, E, F
    def f_A(phi, eta):
        d1 = (math.log(S / K) + (b + 0.5 * sig2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return phi * S * math.exp(-q * T) * norm.cdf(phi * d1) - phi * K * math.exp(-r * T) * norm.cdf(phi * d2)

    def f_B(phi, eta):
        d1 = (math.log(S / H) + (b + 0.5 * sig2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return phi * S * math.exp(-q * T) * norm.cdf(phi * d1) - phi * K * math.exp(-r * T) * norm.cdf(phi * d2)

    def f_C(phi, eta):
        y = (math.log(H * H / (S * K)) + (b + 0.5 * sig2) * T) / (sigma * sqrt_T)
        y2 = y - sigma * sqrt_T
        return phi * S * math.exp(-q * T) * ((H / S) ** (2.0 * (mu + 1.0))) * norm.cdf(eta * y) - phi * K * math.exp(-r * T) * ((H / S) ** (2.0 * mu)) * norm.cdf(eta * y2)

    def f_D(phi, eta):
        y = (math.log(H / S) + (b + 0.5 * sig2) * T) / (sigma * sqrt_T)
        y2 = y - sigma * sqrt_T
        return phi * S * math.exp(-q * T) * ((H / S) ** (2.0 * (mu + 1.0))) * norm.cdf(eta * y) - phi * K * math.exp(-r * T) * ((H / S) ** (2.0 * mu)) * norm.cdf(eta * y2)

    def f_E(eta):
        if rebate <= 0:
            return 0.0
        x = (math.log(H / S) - mu * sig2 * T) / (sigma * sqrt_T)
        return rebate * math.exp(-r * T) * (norm.cdf(eta * x) - ((H / S) ** (2.0 * mu)) * norm.cdf(eta * (x - 2.0 * math.log(H / S) / (sigma * sqrt_T))))

    def f_F(eta):
        if rebate <= 0:
            return 0.0
        x1 = (math.log(H / S) + lam * sig2 * T) / (sigma * sqrt_T)
        x2 = (math.log(H / S) - lam * sig2 * T) / (sigma * sqrt_T)
        return rebate * (((H / S) ** (mu + lam)) * norm.cdf(eta * x1) + ((H / S) ** (mu - lam)) * norm.cdf(eta * x2))

    phi = 1.0 if option_type == "call" else -1.0

    # Pricing formula routing according to relative positions of K and H
    price = 0.0

    if barrier_type == "down_and_out":
        eta = 1.0
        if option_type == "call":
            if K >= H:
                price = f_A(1, 1) - f_C(1, 1) + f_F(1)
            else:
                price = f_B(1, 1) - f_D(1, 1) + f_F(1)
        else:  # put
            if K >= H:
                price = f_A(-1, 1) - f_B(-1, 1) + f_C(-1, 1) - f_D(-1, 1) + f_F(1)
            else:
                price = f_F(1)

    elif barrier_type == "down_and_in":
        from app.engines.black_scholes import bsm_price
        vanilla = bsm_price(S, K, T, r, sigma, option_type, q)["price"]
        out_res = barrier_option_price(S, K, H, T, r, sigma, "down_and_out", option_type, q, rebate=0.0)
        price = vanilla - out_res["price"] + f_E(1)

    elif barrier_type == "up_and_out":
        eta = -1.0
        if option_type == "call":
            if K >= H:
                price = f_F(-1)
            else:
                price = f_A(1, -1) - f_B(1, -1) + f_C(1, -1) - f_D(1, -1) + f_F(-1)
        else:  # put
            if K >= H:
                price = f_B(-1, -1) - f_D(-1, -1) + f_F(-1)
            else:
                price = f_A(-1, -1) - f_C(-1, -1) + f_F(-1)

    elif barrier_type == "up_and_in":
        from app.engines.black_scholes import bsm_price
        vanilla = bsm_price(S, K, T, r, sigma, option_type, q)["price"]
        out_res = barrier_option_price(S, K, H, T, r, sigma, "up_and_out", option_type, q, rebate=0.0)
        price = vanilla - out_res["price"] + f_E(-1)

    return {
        "price": float(max(0.0, price)),
        "barrier_type": barrier_type,
        "option_type": option_type,
        "barrier_level": H,
        "rebate": rebate,
        "breached_at_start": False,
    }


# ═════════════════════════════════════════════════════════════════════
# 3. Binary / Digital Options
# ═════════════════════════════════════════════════════════════════════

def binary_option_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    payoff_type: Literal["cash_or_nothing", "asset_or_nothing"] = "cash_or_nothing",
    option_type: Literal["call", "put"] = "call",
    cash_amount: float = 1.0,
    q: float = 0.0,
) -> dict:
    """Exact analytical pricing for Cash-or-Nothing and Asset-or-Nothing binary options."""
    if T <= 0:
        if option_type == "call":
            hit = S >= K
        else:
            hit = S <= K
        payoff = cash_amount if payoff_type == "cash_or_nothing" else S
        return {"price": float(payoff if hit else 0.0), "delta": 0.0}

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if payoff_type == "cash_or_nothing":
        if option_type == "call":
            price = cash_amount * disc_r * norm.cdf(d2)
            delta = cash_amount * disc_r * norm.pdf(d2) / (S * sigma * sqrt_T)
        else:
            price = cash_amount * disc_r * norm.cdf(-d2)
            delta = -cash_amount * disc_r * norm.pdf(-d2) / (S * sigma * sqrt_T)
    else:  # asset_or_nothing
        if option_type == "call":
            price = S * disc_q * norm.cdf(d1)
            delta = disc_q * norm.cdf(d1) + disc_q * norm.pdf(d1) / (sigma * sqrt_T)
        else:
            price = S * disc_q * norm.cdf(-d1)
            delta = disc_q * norm.cdf(-d1) - disc_q * norm.pdf(-d1) / (sigma * sqrt_T)

    return {
        "price": float(max(0.0, price)),
        "delta": float(delta),
        "payoff_type": payoff_type,
        "option_type": option_type,
    }
