"""Black-Scholes-Merton analytical option pricing engine.

Implements the closed-form BSM model for European vanilla options,
including all first-order Greeks (Delta, Gamma, Vega, Theta, Rho).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from typing import Literal


def _compute_d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> tuple[float, float]:
    """Compute d1 and d2 components of the BSM formula.

    d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return float(d1), float(d2)


def bsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> dict:
    """Compute Black-Scholes-Merton option price.

    Parameters
    ----------
    S : float
        Current spot price of the underlying (S > 0).
    K : float
        Strike price (K > 0).
    T : float
        Time to expiration in years (T > 0).
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualized volatility (sigma > 0).
    option_type : {"call", "put"}
        Type of option.

    Returns
    -------
    dict
        {"price": float, "d1": float, "d2": float}
    """
    if sigma <= 0:
        raise ValueError("Volatility (sigma) must be positive.")
    if T <= 0:
        # At or past expiry: return intrinsic value.
        if option_type == "call":
            return {"price": float(max(S - K, 0.0)), "d1": np.nan, "d2": np.nan}
        return {"price": float(max(K - S, 0.0)), "d1": np.nan, "d2": np.nan}

    d1, d2 = _compute_d1_d2(S, K, T, r, sigma)
    discount = np.exp(-r * T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * discount * norm.cdf(d2)
    else:
        price = K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)

    return {"price": float(price), "d1": d1, "d2": d2}


def bsm_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> dict:
    """Compute all analytical BSM Greeks.

    Parameters
    ----------
    S, K, T, r, sigma, option_type
        See :func:`bsm_price`.

    Returns
    -------
    dict
        {"delta": float, "gamma": float, "vega": float,
         "theta": float, "rho": float}

    Notes
    -----
    Delta_call = N(d1)           Delta_put = N(d1) - 1
    Gamma      = n(d1) / (S * sigma * sqrt(T))        (same for both)
    Vega       = S * n(d1) * sqrt(T)                   (same for both)
    Theta_call = -(S * n(d1) * sigma)/(2*sqrt(T)) - r*K*exp(-r*T)*N(d2)
    Theta_put  = -(S * n(d1) * sigma)/(2*sqrt(T)) + r*K*exp(-r*T)*N(-d2)
    Rho_call   = K * T * exp(-r*T) * N(d2)
    Rho_put    = -K * T * exp(-r*T) * N(-d2)
    """
    if sigma <= 0:
        raise ValueError("Volatility (sigma) must be positive.")
    if T <= 0:
        raise ValueError("Time to expiration (T) must be positive for Greeks calculation.")

    d1, d2 = _compute_d1_d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    discount = np.exp(-r * T)
    n_d1 = norm.pdf(d1)

    # Gamma & Vega are identical for calls and puts
    gamma = n_d1 / (S * sigma * sqrt_T)
    vega = S * n_d1 * sqrt_T

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = -(S * n_d1 * sigma) / (2.0 * sqrt_T) - r * K * discount * norm.cdf(d2)
        rho = K * T * discount * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        theta = -(S * n_d1 * sigma) / (2.0 * sqrt_T) + r * K * discount * norm.cdf(-d2)
        rho = -K * T * discount * norm.cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def bsm_full(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> dict:
    """Compute BSM price and all Greeks in a single pass.

    Returns
    -------
    dict
        {"price", "delta", "gamma", "vega", "theta", "rho"}
    """
    price_result = bsm_price(S, K, T, r, sigma, option_type)
    greeks_result = bsm_greeks(S, K, T, r, sigma, option_type)

    return {
        "price": price_result["price"],
        "delta": greeks_result["delta"],
        "gamma": greeks_result["gamma"],
        "vega": greeks_result["vega"],
        "theta": greeks_result["theta"],
        "rho": greeks_result["rho"],
    }
