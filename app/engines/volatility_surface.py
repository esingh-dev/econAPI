"""Volatility Surface modeling, SVI parametrization, interpolation, and arbitrage diagnostics.

Implements:
- SVI (Stochastic Volatility Inspired) raw parametrization (Gatheral 2004)
- SVI calibration to market strikes / log-moneyness quotes
- Roger Lee wing-slope bounds check (b(1 + |rho|) < 2)
- Butterfly arbitrage check (Gatheral non-negativity of risk-neutral density g(k) >= 0)
- Calendar spread arbitrage check (monotonicity of total variance w(k, T2) >= w(k, T1))
- Bicubic / bilinear 2D surface interpolation across moneyness and expiry
"""

from __future__ import annotations

import math
from typing import Optional
import numpy as np
from scipy.optimize import minimize


def svi_total_variance(
    k: np.ndarray | float, a: float, b: float, rho: float, m: float, sigma: float
) -> np.ndarray | float:
    """Evaluate raw SVI total implied variance w(k) = sigma_BS^2 * T.

    w(k) = a + b * [ rho * (k - m) + sqrt((k - m)^2 + sigma^2) ]

    Parameters
    ----------
    k : ndarray or float
        Log-moneyness k = ln(K / F) where F is the forward price.
    a : float
        Vertical shift (a >= 0).
    b : float
        Angle between the two asymptotes (b >= 0).
    rho : float
        Orientation / skew parameter (-1 <= rho <= 1).
    m : float
        Horizontal translation.
    sigma : float
        Smoothness / curvature parameter (sigma > 0).
    """
    diff = k - m
    return a + b * (rho * diff + np.sqrt(diff * diff + sigma * sigma))


def check_svi_arbitrage(
    a: float, b: float, rho: float, m: float, sigma: float, k_grid: Optional[np.ndarray] = None
) -> dict:
    """Check Roger Lee wing bounds and Durrleman/Gatheral butterfly arbitrage condition."""
    warnings: list[str] = []

    # 1. Roger Lee bounds: b * (1 + |rho|) < 2
    roger_lee_stat = b * (1.0 + abs(rho))
    roger_lee_ok = bool(roger_lee_stat < 2.0)
    if not roger_lee_ok:
        warnings.append(
            f"Roger Lee wing slope bound violated: b*(1+|rho|) = {roger_lee_stat:.4f} >= 2.0"
        )

    # 2. Strict minimum variance non-negativity: a + b*sigma*sqrt(1 - rho^2) >= 0
    min_var = a + b * sigma * math.sqrt(max(0.0, 1.0 - rho * rho))
    non_negative_var = bool(min_var >= 0.0)
    if not non_negative_var:
        warnings.append(f"Negative total variance minimum detected ({min_var:.4f} < 0)")

    # 3. Butterfly arbitrage check via Gatheral's g(k) formula:
    # g(k) = (1 - k*w' / (2*w))^2 - (w'/4)*(1/w + 1/4) + w''/2
    if k_grid is None:
        k_grid = np.linspace(-1.5, 1.5, 200)

    diff = k_grid - m
    sq_term = np.sqrt(diff * diff + sigma * sigma)
    w = a + b * (rho * diff + sq_term)
    w_prime = b * (rho + diff / sq_term)
    w_second = b * (sigma * sigma) / (sq_term**3)

    # Protect division by zero
    w_safe = np.maximum(w, 1e-8)
    term1 = (1.0 - k_grid * w_prime / (2.0 * w_safe)) ** 2
    term2 = (w_prime * w_prime / 4.0) * (1.0 / w_safe + 0.25)
    term3 = 0.5 * w_second

    g_k = term1 - term2 + term3
    min_g = float(np.min(g_k))
    butterfly_arb_free = bool(min_g >= 0.0)

    if not butterfly_arb_free:
        warnings.append(
            f"Butterfly arbitrage violation: minimum density g(k) = {min_g:.4f} < 0.0"
        )

    return {
        "butterfly_arbitrage_free": butterfly_arb_free,
        "roger_lee_bound_satisfied": roger_lee_ok,
        "roger_lee_stat": float(roger_lee_stat),
        "min_variance": float(min_var),
        "min_density_g": float(min_g),
        "warnings": warnings,
    }


def calibrate_svi_slice(
    strikes: np.ndarray,
    implied_vols: np.ndarray,
    forward: float,
    T: float,
) -> dict:
    """Calibrate raw SVI slice parameters to market implied volatilities for a single expiry T.

    Parameters
    ----------
    strikes : ndarray
        Option strikes.
    implied_vols : ndarray
        Market implied volatilities as decimals.
    forward : float
        Forward price for expiry T.
    T : float
        Maturity in years.
    """
    k = np.log(np.asarray(strikes, dtype=float) / forward)
    w_market = (np.asarray(implied_vols, dtype=float) ** 2) * T

    # Initial guess
    atm_w = float(np.interp(0.0, k, w_market))
    p0 = [atm_w * 0.5, 0.1, -0.3, 0.0, 0.1]

    # Bounds: a, b, rho, m, sigma
    bounds = [
        (0.0, 2.0),
        (0.001, 1.5),
        (-0.99, 0.99),
        (-1.0, 1.0),
        (0.001, 1.0),
    ]

    def loss(p):
        a, b, rho, m, sigma = p
        w_model = svi_total_variance(k, a, b, rho, m, sigma)
        sq_err = np.sum((w_model - w_market) ** 2)
        # Penalize Roger Lee bound violation
        rl_pen = max(0.0, b * (1.0 + abs(rho)) - 2.0) * 100.0
        return sq_err + rl_pen

    res = minimize(loss, p0, bounds=bounds, method="L-BFGS-B")
    a_opt, b_opt, rho_opt, m_opt, sigma_opt = res.x

    fitted_w = svi_total_variance(k, a_opt, b_opt, rho_opt, m_opt, sigma_opt)
    fitted_vols = np.sqrt(np.maximum(1e-8, fitted_w) / T)
    rmse = float(np.sqrt(np.mean((fitted_vols - implied_vols) ** 2)))

    arb_checks = check_svi_arbitrage(a_opt, b_opt, rho_opt, m_opt, sigma_opt)

    return {
        "a": float(a_opt),
        "b": float(b_opt),
        "rho": float(rho_opt),
        "m": float(m_opt),
        "sigma": float(sigma_opt),
        "rmse": rmse,
        "fitted_vols": fitted_vols.tolist(),
        "converged": bool(res.success),
        "arbitrage_checks": arb_checks,
    }


def check_calendar_arbitrage(
    expiries: list[float],
    svi_slices: list[dict],
    k_grid: Optional[np.ndarray] = None,
) -> dict:
    """Check calendar spread arbitrage: total variance must be non-decreasing in tenor T.

    w(k, T_{i+1}) >= w(k, T_i) for T_{i+1} > T_i.
    """
    if k_grid is None:
        k_grid = np.linspace(-1.0, 1.0, 100)

    warnings: list[str] = []
    total_vars = []
    for s in svi_slices:
        w = svi_total_variance(k_grid, s["a"], s["b"], s["rho"], s["m"], s["sigma"])
        total_vars.append(w)

    violations = 0
    max_drop = 0.0
    for i in range(len(expiries) - 1):
        diff = total_vars[i] - total_vars[i + 1]  # positive if calendar arb exists
        worst = float(np.max(diff))
        if worst > 1e-6:
            violations += 1
            max_drop = max(max_drop, worst)
            warnings.append(
                f"Calendar arbitrage detected between T={expiries[i]:.2f} and T={expiries[i+1]:.2f} (max drop: {worst:.4f})"
            )

    return {
        "calendar_arbitrage_free": bool(violations == 0),
        "num_slice_violations": violations,
        "max_variance_drop": float(max_drop),
        "warnings": warnings,
    }
