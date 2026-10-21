"""Black-Scholes-Merton analytical option pricing and Greeks engine with safeguards.

Implements the closed-form BSM model for European vanilla options with continuous
dividend yield q, complete first- and higher-order Greeks, finite-difference validation,
put-call parity verification, no-arbitrage bounds checking, and boundary condition handling.
"""

from __future__ import annotations

import math
from typing import Literal, Optional
import numpy as np
from scipy.stats import norm


def _compute_d1_d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> tuple[float, float]:
    """Compute d1 and d2 components of the generalized BSM formula (Merton 1973).

    d1 = [ln(S/K) + (r - q + sigma^2 / 2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    """
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
) -> dict:
    """Compute Black-Scholes-Merton option price with continuous dividend yield q.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset (S >= 0, units: currency).
    K : float
        Strike price (K >= 0, units: currency).
    T : float
        Time to expiration in years (T >= 0, units: years).
    r : float
        Continuously compounded annualized risk-free interest rate (e.g. 0.05 for 5%).
    sigma : float
        Annualized volatility (sigma >= 0, e.g. 0.20 for 20%).
    option_type : {"call", "put"}
        Option type.
    q : float, optional
        Continuous dividend or foreign interest rate yield (default: 0.0).

    Returns
    -------
    dict
        {"price": float, "d1": float, "d2": float, "warnings": list[str]}
    """
    warnings: list[str] = []

    # Parameter checks
    if S < 0:
        raise ValueError("Spot price S must be non-negative.")
    if K < 0:
        raise ValueError("Strike price K must be non-negative.")
    if sigma < 0:
        raise ValueError("Volatility sigma must be non-negative.")
    if T < 0:
        raise ValueError("Time to expiration T must be non-negative.")

    # Boundary cases:
    # 1. At or past expiry (T <= 0)
    if T <= 1e-12:
        warnings.append("Option at or past expiry; returning intrinsic payoff.")
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return {"price": float(intrinsic), "d1": float("nan"), "d2": float("nan"), "warnings": warnings}

    # 2. Spot price S -> 0
    if S <= 1e-12:
        warnings.append("Spot price is zero.")
        price = 0.0 if option_type == "call" else K * math.exp(-r * T)
        return {"price": float(price), "d1": float("-inf"), "d2": float("-inf"), "warnings": warnings}

    # 3. Strike price K -> 0
    if K <= 1e-12:
        warnings.append("Strike price is zero.")
        price = S * math.exp(-q * T) if option_type == "call" else 0.0
        return {"price": float(price), "d1": float("inf"), "d2": float("inf"), "warnings": warnings}

    # 4. Zero volatility (sigma -> 0)
    if sigma <= 1e-12:
        warnings.append("Volatility is zero; pricing deterministic forward discounted payoff.")
        forward = S * math.exp((r - q) * T)
        discount = math.exp(-r * T)
        if option_type == "call":
            price = max(forward - K, 0.0) * discount
        else:
            price = max(K - forward, 0.0) * discount
        return {"price": float(price), "d1": float("nan"), "d2": float("nan"), "warnings": warnings}

    # Standard analytical calculation
    d1, d2 = _compute_d1_d2(S, K, T, r, sigma, q)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if option_type == "call":
        price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    else:
        price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)

    # Clean numerical precision near 0
    price = max(float(price), 0.0)

    return {"price": price, "d1": float(d1), "d2": float(d2), "warnings": warnings}


def bsm_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
) -> dict:
    """Compute analytical first- and higher-order Greeks under generalized BSM.

    First-order: Delta, Gamma, Vega, Theta, Rho.
    Higher-order: Vanna, Volga (vomma), Charm (delta decay), Veta, Color, Speed, Ultima.
    """
    if sigma <= 1e-12:
        raise ValueError("Volatility sigma must be positive for Greeks calculation.")
    if T <= 1e-12:
        raise ValueError("Time to expiration T must be positive for Greeks calculation.")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive for Greeks calculation.")

    d1, d2 = _compute_d1_d2(S, K, T, r, sigma, q)
    sqrt_T = math.sqrt(T)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_neg_d1 = norm.cdf(-d1)
    N_neg_d2 = norm.cdf(-d2)

    # ── First-order Greeks ──────────────────────────────────────────────
    # Gamma & Vega
    gamma = (disc_q * n_d1) / (S * sigma * sqrt_T)
    vega = S * disc_q * n_d1 * sqrt_T

    if option_type == "call":
        delta = disc_q * N_d1
        theta = (
            -(S * disc_q * n_d1 * sigma) / (2.0 * sqrt_T)
            - r * K * disc_r * N_d2
            + q * S * disc_q * N_d1
        )
        rho = K * T * disc_r * N_d2
        charm = disc_q * (
            q * N_d1 - n_d1 * ((2.0 * (r - q) * T - d2 * sigma * sqrt_T) / (2.0 * T * sigma * sqrt_T))
        )
    else:
        delta = -disc_q * N_neg_d1
        theta = (
            -(S * disc_q * n_d1 * sigma) / (2.0 * sqrt_T)
            + r * K * disc_r * N_neg_d2
            - q * S * disc_q * N_neg_d1
        )
        rho = -K * T * disc_r * N_neg_d2
        charm = -disc_q * (
            q * N_neg_d1 + n_d1 * ((2.0 * (r - q) * T - d2 * sigma * sqrt_T) / (2.0 * T * sigma * sqrt_T))
        )

    # ── Higher-order Greeks ─────────────────────────────────────────────
    # Vanna: dVega/dS = dDelta/dSigma = -disc_q * n(d1) * d2 / sigma
    vanna = -disc_q * n_d1 * d2 / sigma

    # Volga (Vomma): dVega/dSigma = Vega * d1 * d2 / sigma
    volga = vega * d1 * d2 / sigma

    # Veta: -dVega/dT = -S * disc_q * n(d1) * sqrt_T * [q + (r - q)*d1/(sigma*sqrt_T) - (1 + d1*d2)/(2*T)]
    veta = -S * disc_q * n_d1 * sqrt_T * (
        q + ((r - q) * d1) / (sigma * sqrt_T) - (1.0 + d1 * d2) / (2.0 * T)
    )

    # Speed: dGamma/dS = -Gamma/S * (d1 / (sigma * sqrt_T) + 1)
    speed = -gamma / S * (d1 / (sigma * sqrt_T) + 1.0)

    # Color: dGamma/dT
    color = (
        -gamma
        * (
            q
            + (r - q) * d1 / (sigma * sqrt_T)
            + (1.0 - d1 * d2) / (2.0 * T)
        )
    )

    # Ultima: dVolga/dSigma = -Vega/sigma^2 * [d1*d2*(1 - d1*d2) + d1^2 + d2^2]
    ultima = (-vega / (sigma * sigma)) * (
        d1 * d2 * (1.0 - d1 * d2) + d1 * d1 + d2 * d2
    )

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
        "vanna": float(vanna),
        "volga": float(volga),
        "charm": float(charm),
        "veta": float(veta),
        "color": float(color),
        "speed": float(speed),
        "ultima": float(ultima),
    }


def validate_greeks_finite_difference(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
    eps: float = 1e-4,
) -> dict:
    """Validate analytical Greeks against central finite difference numerical derivatives.

    Returns comparison dict with analytical values, numerical values, and relative errors.
    """
    ana = bsm_greeks(S, K, T, r, sigma, option_type, q)

    def p(s=S, t=T, v=sigma, rate=r):
        return bsm_price(s, K, t, rate, v, option_type, q)["price"]

    # Delta: dP/dS
    num_delta = (p(s=S * (1.0 + eps)) - p(s=S * (1.0 - eps))) / (2.0 * S * eps)
    # Gamma: d2P/dS2
    num_gamma = (
        p(s=S * (1.0 + eps)) - 2.0 * p() + p(s=S * (1.0 - eps))
    ) / ((S * eps) ** 2)
    # Vega: dP/dSigma
    num_vega = (p(v=sigma + eps) - p(v=sigma - eps)) / (2.0 * eps)
    # Theta: -dP/dT
    num_theta = -(p(t=T + eps) - p(t=T - eps)) / (2.0 * eps)
    # Rho: dP/dr
    num_rho = (p(rate=r + eps) - p(rate=r - eps)) / (2.0 * eps)

    # Vanna: d2P / (dS dSigma)
    p_pp = p(s=S * (1.0 + eps), v=sigma + eps)
    p_pm = p(s=S * (1.0 + eps), v=sigma - eps)
    p_mp = p(s=S * (1.0 - eps), v=sigma + eps)
    p_mm = p(s=S * (1.0 - eps), v=sigma - eps)
    num_vanna = (p_pp - p_pm - p_mp + p_mm) / (4.0 * S * eps * eps)

    # Volga: d2P / dSigma2
    num_volga = (p(v=sigma + eps) - 2.0 * p() + p(v=sigma - eps)) / (eps * eps)

    def rel_err(a: float, n: float) -> float:
        denom = max(abs(a), abs(n), 1e-6)
        return abs(a - n) / denom

    checks = {
        "delta": {"analytical": ana["delta"], "numerical": num_delta, "rel_error": rel_err(ana["delta"], num_delta)},
        "gamma": {"analytical": ana["gamma"], "numerical": num_gamma, "rel_error": rel_err(ana["gamma"], num_gamma)},
        "vega": {"analytical": ana["vega"], "numerical": num_vega, "rel_error": rel_err(ana["vega"], num_vega)},
        "theta": {"analytical": ana["theta"], "numerical": num_theta, "rel_error": rel_err(ana["theta"], num_theta)},
        "rho": {"analytical": ana["rho"], "numerical": num_rho, "rel_error": rel_err(ana["rho"], num_rho)},
        "vanna": {"analytical": ana["vanna"], "numerical": num_vanna, "rel_error": rel_err(ana["vanna"], num_vanna)},
        "volga": {"analytical": ana["volga"], "numerical": num_volga, "rel_error": rel_err(ana["volga"], num_volga)},
    }
    all_passed = all(c["rel_error"] < 0.02 for c in checks.values())

    return {
        "all_passed": all_passed,
        "max_rel_error": max(c["rel_error"] for c in checks.values()),
        "checks": checks,
    }


def verify_put_call_parity(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    tol: float = 1e-4,
) -> dict:
    """Verify Put-Call Parity: C - P = S * exp(-q*T) - K * exp(-r*T)."""
    lhs = call_price - put_price
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    diff = abs(lhs - rhs)
    return {
        "satisfied": bool(diff <= tol),
        "lhs_c_minus_p": float(lhs),
        "rhs_forward_diff": float(rhs),
        "discrepancy": float(diff),
        "tolerance": tol,
    }


def check_no_arbitrage_bounds(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
) -> dict:
    """Verify standard European option no-arbitrage lower and upper bounds."""
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if option_type == "call":
        lower = max(0.0, S * disc_q - K * disc_r)
        upper = S * disc_q
    else:
        lower = max(0.0, K * disc_r - S * disc_q)
        upper = K * disc_r

    is_valid = (lower - 1e-7 <= price <= upper + 1e-7)
    return {
        "valid": bool(is_valid),
        "price": float(price),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "violation": None if is_valid else ("Below lower bound" if price < lower else "Above upper bound"),
    }


def bsm_full(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
    validate_greeks: bool = False,
    extended: bool = False,
) -> dict:
    """Compute BSM price and Greeks in a single pass.

    Returns the standard 6 keys by default, or full extended higher-order Greeks
    and validation when extended=True or validate_greeks=True.
    """
    price_res = bsm_price(S, K, T, r, sigma, option_type, q)
    greeks_res = bsm_greeks(S, K, T, r, sigma, option_type, q)

    result = {
        "price": price_res["price"],
        "delta": greeks_res["delta"],
        "gamma": greeks_res["gamma"],
        "vega": greeks_res["vega"],
        "theta": greeks_res["theta"],
        "rho": greeks_res["rho"],
    }

    if extended or validate_greeks or q != 0.0:
        result.update({
            "vanna": greeks_res["vanna"],
            "volga": greeks_res["volga"],
            "charm": greeks_res["charm"],
            "veta": greeks_res["veta"],
            "color": greeks_res["color"],
            "speed": greeks_res["speed"],
            "ultima": greeks_res["ultima"],
            "d1": price_res["d1"],
            "d2": price_res["d2"],
            "no_arbitrage_bounds": check_no_arbitrage_bounds(price_res["price"], S, K, T, r, option_type, q),
            "warnings": price_res["warnings"],
        })

    if validate_greeks:
        result["greeks_validation"] = validate_greeks_finite_difference(
            S, K, T, r, sigma, option_type, q
        )

    return result
