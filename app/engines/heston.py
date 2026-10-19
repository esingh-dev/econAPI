"""Heston stochastic volatility model pricing engine.

Implements option pricing under the Heston model via semi-analytical
Fourier / numerical quadrature integration of the characteristic function,
with implied volatility extraction via Brent's method.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.optimize import brentq
from typing import Literal


def heston_characteristic_function(
    u: complex,
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> complex:
    """Evaluate the Heston characteristic function φ(u).

    Uses the formulation of Gatheral (2006) / Albrecher et al., choosing
    the numerically stable branch with log-transform.

    Parameters
    ----------
    u : complex
        Fourier variable.
    S : float
        Spot price.
    K : float
        Strike price (unused here but kept for interface consistency).
    T : float
        Time to expiration.
    r : float
        Risk-free rate.
    v0 : float
        Initial instantaneous variance.
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run variance.
    xi : float
        Volatility of variance (vol-of-vol).
    rho : float
        Correlation between asset and variance Brownian motions.

    Returns
    -------
    complex
        Value of the characteristic function at u.
    """
    xi2 = xi * xi
    iu = 1j * u

    d = np.sqrt(
        (rho * xi * iu - kappa) ** 2 + xi2 * (iu + u * u)
    )

    g_num = kappa - rho * xi * iu - d
    g_den = kappa - rho * xi * iu + d

    # Protect against zero denominator
    if abs(g_den) < 1e-30:
        g_den = 1e-30

    g = g_num / g_den

    exp_neg_dT = np.exp(-d * T)

    # C and D coefficients
    C = (kappa * theta / xi2) * (
        g_num * T - 2.0 * np.log((1.0 - g * exp_neg_dT) / (1.0 - g))
    )

    D = (g_num / xi2) * (
        (1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT)
    )

    phi = np.exp(C + D * v0 + iu * (np.log(S) + r * T))
    return phi


def _heston_P(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    j: int,
) -> float:
    """Compute probability Pj (j=1 or j=2) via numerical integration.

    For j=1: uses the characteristic function for the asset measure.
    For j=2: uses the characteristic function for the risk-neutral measure.

    The call price is then C = S·P1 − K·exp(−rT)·P2.
    """
    def integrand(u: float) -> float:
        if j == 1:
            # Characteristic function under asset measure:
            # φ(u − i) / φ(−i)  where φ(−i) = S·exp(rT)
            phi_u_minus_i = heston_characteristic_function(
                u - 1j, S, K, T, r, v0, kappa, theta, xi, rho
            )
            phi_neg_i = S * np.exp(r * T)
            cf = phi_u_minus_i / phi_neg_i
        else:
            cf = heston_characteristic_function(
                u, S, K, T, r, v0, kappa, theta, xi, rho
            )

        return float(np.real(np.exp(-1j * u * np.log(K)) * cf / (1j * u)))

    integral, _ = integrate.quad(integrand, 1e-15, 200.0, limit=200)
    return 0.5 + integral / np.pi


def heston_price_by_integration(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Compute European option price under the Heston model.

    Uses semi-analytical Fourier inversion with scipy.integrate.quad.

    Parameters
    ----------
    S, K, T, r : float
        Standard option parameters.
    v0 : float
        Initial variance.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-run variance level.
    xi : float
        Vol-of-vol.
    rho : float
        Correlation ∈ [−1, 1].
    option_type : {"call", "put"}
        Option type.

    Returns
    -------
    float
        Option price.
    """
    P1 = _heston_P(S, K, T, r, v0, kappa, theta, xi, rho, j=1)
    P2 = _heston_P(S, K, T, r, v0, kappa, theta, xi, rho, j=2)

    call_price = S * P1 - K * np.exp(-r * T) * P2

    if option_type == "call":
        return float(call_price)
    else:
        # Put-call parity: P = C - S + K·exp(−rT)
        put_price = call_price - S + K * np.exp(-r * T)
        return float(put_price)


def heston_implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Extract BSM implied volatility from a given option price via Brent's method.

    Parameters
    ----------
    price : float
        Market (or model) option price.
    S, K, T, r : float
        Standard option parameters.
    option_type : {"call", "put"}
        Option type.

    Returns
    -------
    float
        Implied volatility.
    """
    from app.engines.black_scholes import bsm_price as _bsm_price

    def objective(sigma: float) -> float:
        return _bsm_price(S, K, T, r, sigma, option_type)["price"] - price

    try:
        iv = brentq(objective, 1e-4, 5.0, xtol=1e-12, maxiter=200)
    except ValueError:
        iv = float("nan")
    return float(iv)


def heston_full(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: Literal["call", "put"] = "call",
) -> dict:
    """Compute Heston price and back-out BSM implied volatility.

    Returns
    -------
    dict
        {"price": float, "implied_volatility": float}
    """
    price = heston_price_by_integration(
        S, K, T, r, v0, kappa, theta, xi, rho, option_type
    )
    iv = heston_implied_vol(price, S, K, T, r, option_type)

    return {
        "price": price,
        "implied_volatility": iv,
    }
