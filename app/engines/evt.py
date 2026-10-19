"""Extreme Value Theory — Peaks-Over-Threshold GPD fitting.

Fits a Generalised Pareto Distribution to tail losses, then computes
analytical EVT-based Value-at-Risk and Expected Shortfall (CVaR).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def fit_gpd_pot(
    returns: np.ndarray,
    threshold_quantile: float = 0.95,
) -> dict:
    """Fit GPD to tail losses via Peaks-Over-Threshold.

    Parameters
    ----------
    returns : ndarray, shape (N,)
        Historical return series (positive = gain, negative = loss).
    threshold_quantile : float
        Quantile level for the threshold (e.g. 0.95).

    Returns
    -------
    dict
        xi : float — GPD shape parameter
        beta : float — GPD scale parameter
        threshold : float — threshold u
        n_exceedances : int
        n_total : int
    """
    losses = -returns  # large losses become large positive numbers
    u = float(np.quantile(losses, threshold_quantile))

    exceedances = losses[losses > u] - u

    if len(exceedances) < 5:
        raise ValueError(
            f"Only {len(exceedances)} exceedances found above the "
            f"{threshold_quantile:.0%} threshold. Need at least 5 for a "
            f"reliable GPD fit. Lower threshold_quantile or provide more data."
        )

    # scipy.stats.genpareto: shape=c (=xi), loc, scale (=beta)
    # Fix loc=0 since exceedances are already shifted
    xi, _loc, beta = stats.genpareto.fit(exceedances, floc=0)

    return {
        "xi": float(xi),
        "beta": float(beta),
        "threshold": float(u),
        "n_exceedances": int(len(exceedances)),
        "n_total": int(len(returns)),
    }


def compute_var(gpd_params: dict, confidence_level: float = 0.99) -> float:
    """Compute EVT-based Value-at-Risk.

    VaR_α = u + (β/ξ) · [ ((1−α) / p_u)^(−ξ) − 1 ]

    where p_u = n_exceedances / n_total.

    Parameters
    ----------
    gpd_params : dict
        Output from :func:`fit_gpd_pot`.
    confidence_level : float
        Confidence level α (e.g. 0.99 for 99% VaR).

    Returns
    -------
    float
        Value-at-Risk (expressed as a positive loss).
    """
    xi = gpd_params["xi"]
    beta = gpd_params["beta"]
    u = gpd_params["threshold"]
    n_exc = gpd_params["n_exceedances"]
    n_total = gpd_params["n_total"]

    p_u = n_exc / n_total  # tail probability

    if abs(xi) < 1e-12:
        # Exponential limit (xi → 0)
        var = u + beta * np.log(p_u / (1.0 - confidence_level))
    else:
        var = u + (beta / xi) * (
            ((1.0 - confidence_level) / p_u) ** (-xi) - 1.0
        )

    return float(var)


def compute_cvar(var_value: float, gpd_params: dict) -> float:
    """Compute EVT-based Expected Shortfall (CVaR).

    CVaR_α = (VaR_α + β − ξ·u) / (1 − ξ)

    Requires ξ < 1 for a finite expectation.

    Parameters
    ----------
    var_value : float
        Value-at-Risk from :func:`compute_var`.
    gpd_params : dict
        Output from :func:`fit_gpd_pot`.

    Returns
    -------
    float
        Expected Shortfall / Conditional VaR.
    """
    xi = gpd_params["xi"]
    beta = gpd_params["beta"]
    u = gpd_params["threshold"]

    if xi >= 1.0:
        raise ValueError(
            f"GPD shape parameter ξ = {xi:.4f} ≥ 1; "
            f"the mean excess is infinite and CVaR is undefined."
        )

    cvar = (var_value + beta - xi * u) / (1.0 - xi)
    return float(cvar)


def evt_analysis(
    returns: np.ndarray,
    confidence_level: float = 0.99,
    threshold_quantile: float = 0.95,
) -> dict:
    """Full EVT tail-risk analysis.

    Parameters
    ----------
    returns : ndarray, shape (N,)
        Historical return series.
    confidence_level : float
        VaR / CVaR confidence level.
    threshold_quantile : float
        POT threshold quantile.

    Returns
    -------
    dict
        gpd_params : dict
        var : float
        cvar : float
        confidence_level : float
    """
    gpd_params = fit_gpd_pot(returns, threshold_quantile)
    var = compute_var(gpd_params, confidence_level)
    cvar = compute_cvar(var, gpd_params)

    return {
        "gpd_params": gpd_params,
        "var": var,
        "cvar": cvar,
        "confidence_level": confidence_level,
    }
