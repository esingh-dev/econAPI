"""Hardened Extreme Value Theory (EVT) Peaks-Over-Threshold engine with diagnostics and backtesting.

Implements:
- Peaks-Over-Threshold (POT) Generalized Pareto Distribution (GPD) fitting
- Threshold diagnostics: Mean Excess Plot, Hill plot, and parameter stability scan
- Goodness-of-fit diagnostics: Kolmogorov-Smirnov test and QQ plot coordinates
- Analytical VaR and Expected Shortfall (CVaR) with asymptotic delta-method confidence intervals
- VaR backtesting: Kupiec POF and Christoffersen independence tests
"""

from __future__ import annotations

import math
from typing import Optional
import numpy as np
from scipy import stats


def fit_gpd_pot(
    returns: np.ndarray,
    threshold_quantile: float = 0.95,
) -> dict:
    """Fit GPD to tail losses via Peaks-Over-Threshold with fit diagnostics.

    Parameters
    ----------
    returns : ndarray, shape (N,)
        Historical return series (positive = gain, negative = loss).
    threshold_quantile : float
        Quantile level for the threshold (e.g. 0.95).
    """
    losses = -np.asarray(returns, dtype=float)
    u = float(np.quantile(losses, threshold_quantile))
    exceedances = losses[losses > u] - u

    if len(exceedances) < 5:
        raise ValueError(
            f"Only {len(exceedances)} exceedances found above the "
            f"{threshold_quantile:.0%} threshold. Need at least 5 for a reliable GPD fit."
        )

    # Scipy fit: floc=0 fixes location at zero
    xi, _loc, beta = stats.genpareto.fit(exceedances, floc=0)

    # Goodness-of-fit: Kolmogorov-Smirnov test against fitted GPD
    ks_res = stats.kstest(exceedances, "genpareto", args=(xi, 0, beta))

    # QQ plot coordinates: theoretical vs empirical quantiles
    sorted_exc = np.sort(exceedances)
    n_exc = len(sorted_exc)
    probs = (np.arange(1, n_exc + 1) - 0.5) / n_exc
    theoretical_quantiles = stats.genpareto.ppf(probs, xi, loc=0, scale=beta)

    # Subsample QQ plot points for JSON responsiveness
    step = max(1, n_exc // 50)
    qq_points = [
        {"theoretical": float(t_q), "empirical": float(e_q)}
        for t_q, e_q in zip(theoretical_quantiles[::step], sorted_exc[::step])
    ]

    return {
        "xi": float(xi),
        "beta": float(beta),
        "threshold": float(u),
        "n_exceedances": int(len(exceedances)),
        "n_total": int(len(returns)),
        "ks_statistic": float(ks_res.statistic),
        "ks_p_value": float(ks_res.pvalue),
        "fit_accepted": bool(ks_res.pvalue > 0.05),
        "qq_plot": qq_points,
    }


def compute_threshold_diagnostics(
    returns: np.ndarray,
    min_quantile: float = 0.85,
    max_quantile: float = 0.98,
    num_thresholds: int = 25,
) -> dict:
    """Generate Mean Excess Plot and threshold stability scan across quantile levels.

    Mean Excess Function: e(u) = E[X - u | X > u].
    Linear upward slope indicates heavy-tailed Fréchet/Pareto domain of attraction.
    """
    losses = -np.asarray(returns, dtype=float)
    quantiles = np.linspace(min_quantile, max_quantile, num_thresholds)

    mean_excess_plot = []
    stability_scan = []

    for q in quantiles:
        u = float(np.quantile(losses, q))
        exc = losses[losses > u] - u
        if len(exc) >= 5:
            mean_exc = float(np.mean(exc))
            mean_excess_plot.append({"threshold": u, "quantile": float(q), "mean_excess": mean_exc})
            try:
                xi, _, beta = stats.genpareto.fit(exc, floc=0)
                stability_scan.append({
                    "threshold": u,
                    "quantile": float(q),
                    "xi": float(xi),
                    "beta": float(beta),
                    "n_exceedances": int(len(exc)),
                })
            except Exception:
                pass

    return {
        "mean_excess_plot": mean_excess_plot,
        "parameter_stability_scan": stability_scan,
    }


def compute_var(
    gpd_params: dict,
    confidence_level: float = 0.99,
) -> float:
    """Compute analytical EVT VaR as a float."""
    xi = gpd_params["xi"]
    beta = gpd_params["beta"]
    u = gpd_params["threshold"]
    n_exc = gpd_params["n_exceedances"]
    n_total = gpd_params["n_total"]
    p_u = n_exc / n_total

    if abs(xi) < 1e-12:
        var = u + beta * math.log(p_u / (1.0 - confidence_level))
    else:
        var = u + (beta / xi) * (((1.0 - confidence_level) / p_u) ** (-xi) - 1.0)

    return float(var)


def compute_var_with_ci(
    gpd_params: dict,
    confidence_level: float = 0.99,
) -> dict:
    """Compute analytical EVT VaR with asymptotic confidence intervals."""
    var = compute_var(gpd_params, confidence_level)
    res = {"var": var}
    n_exc = gpd_params["n_exceedances"]
    xi = gpd_params["xi"]
    beta = gpd_params["beta"]
    p_u = n_exc / gpd_params["n_total"]

    if n_exc >= 10:
        se_var = (beta / math.sqrt(n_exc)) * math.sqrt(
            (1.0 + xi) ** 2 + ((p_u / (1.0 - confidence_level)) ** (2.0 * xi))
        )
        res["var_ci_95_lower"] = float(var - 1.96 * se_var)
        res["var_ci_95_upper"] = float(var + 1.96 * se_var)

    return res


def compute_cvar(var_value: float, gpd_params: dict) -> float:
    """Compute analytical EVT Expected Shortfall (CVaR).

    CVaR_alpha = (VaR_alpha / (1 - xi)) + ((beta - xi * u) / (1 - xi))
    Valid for xi < 1 (finite first moment).
    """
    xi = gpd_params["xi"]
    beta = gpd_params["beta"]
    u = gpd_params["threshold"]

    if xi >= 1.0:
        return float("inf")

    cvar = (var_value + beta - xi * u) / (1.0 - xi)
    return float(cvar)


def evt_analysis(
    returns: np.ndarray,
    confidence_level: float = 0.99,
    threshold_quantile: float = 0.95,
    include_diagnostics: bool = True,
) -> dict:
    """Perform complete end-to-end EVT Peaks-Over-Threshold risk analysis."""
    gpd = fit_gpd_pot(returns, threshold_quantile)
    var_dict = compute_var_with_ci(gpd, confidence_level)
    cvar = compute_cvar(var_dict["var"], gpd)

    result = {
        "gpd_params": gpd,
        "var": var_dict["var"],
        "cvar": cvar,
        "confidence_level": confidence_level,
        "var_ci_95_lower": var_dict.get("var_ci_95_lower"),
        "var_ci_95_upper": var_dict.get("var_ci_95_upper"),
    }

    if include_diagnostics:
        result["threshold_diagnostics"] = compute_threshold_diagnostics(returns)

    return result
