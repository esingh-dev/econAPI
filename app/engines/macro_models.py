"""Macro-econometric models — Nelson-Siegel yield curve fitting and Cointegration analytics.

Fits the three-factor Nelson-Siegel parametric term structure model to observed bond yields
with stable Diebold-Li initialization, unit convention support, fit diagnostics (R^2, RMSE, MAE, AIC/BIC),
robust Huber loss estimation, and parameter recovery benchmarking.
"""

from __future__ import annotations

import math
from typing import Literal, Optional
import numpy as np
from scipy.optimize import curve_fit, minimize


def nelson_siegel(
    tau: np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    lam: float,
) -> np.ndarray:
    """Evaluate the Nelson-Siegel yield curve.

    y(tau) = beta0 + beta1 * [(1 - exp(-tau/lam)) / (tau/lam)]
                   + beta2 * [(1 - exp(-tau/lam)) / (tau/lam) - exp(-tau/lam)]

    Parameters
    ----------
    tau : ndarray
        Maturities in years.
    beta0 : float
        Level (long-term asymptotic rate).
    beta1 : float
        Slope factor (short-term component).
    beta2 : float
        Curvature factor (medium-term component).
    lam : float
        Decay parameter lambda > 0.
    """
    tau = np.asarray(tau, dtype=float)
    x = np.where(tau > 1e-12, tau / lam, 1e-12)
    exp_neg_x = np.exp(-x)

    factor1 = np.where(tau > 1e-12, (1.0 - exp_neg_x) / x, 1.0)
    factor2 = factor1 - exp_neg_x

    return beta0 + beta1 * factor1 + beta2 * factor2


def fit_nelson_siegel(
    tenors: np.ndarray,
    yields: np.ndarray,
    unit_convention: Literal["auto", "percent", "decimal"] = "auto",
    robust_loss: Literal["linear", "huber", "soft_l1"] = "linear",
) -> dict:
    """Fit the Nelson-Siegel model to an observed term structure with fit diagnostics.

    Parameters
    ----------
    tenors : ndarray
        Maturity tenors in years.
    yields : ndarray
        Observed market yields.
    unit_convention : {"auto", "percent", "decimal"}
    robust_loss : {"linear", "huber", "soft_l1"}
    """
    tenors = np.asarray(tenors, dtype=float)
    y_raw = np.asarray(yields, dtype=float)

    if len(tenors) != len(y_raw):
        raise ValueError("tenors and yields must have the same length.")
    if len(tenors) < 4:
        raise ValueError("Need at least 4 data points for Nelson-Siegel fit.")

    # Unit convention detection
    warnings = []
    scale = 1.0
    detected_unit = "percent"
    if unit_convention == "decimal" or (unit_convention == "auto" and np.max(np.abs(y_raw)) < 0.25):
        detected_unit = "decimal"
        # Fit internally in percent for optimal numerical scaling
        scale = 100.0
        warnings.append("Yields detected as decimal; scaled to percentage internally for stable numerical optimization.")

    y = y_raw * scale

    # 1. Diebold-Li two-step grid search over lambda to find global minimum basin
    best_rmse = float("inf")
    best_p0 = [float(y[-1]), float(y[0] - y[-1]), 0.0, 1.5]

    for lam_candidate in np.linspace(0.1, 10.0, 60):
        x = np.where(tenors > 1e-12, tenors / lam_candidate, 1e-12)
        exp_neg = np.exp(-x)
        f1 = np.where(tenors > 1e-12, (1.0 - exp_neg) / x, 1.0)
        f2 = f1 - exp_neg
        X = np.column_stack([np.ones_like(tenors), f1, f2])
        try:
            betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ betas
            curr_rmse = np.sqrt(np.mean((y - pred) ** 2))
            if curr_rmse < best_rmse:
                best_rmse = curr_rmse
                best_p0 = [float(betas[0]), float(betas[1]), float(betas[2]), float(lam_candidate)]
        except np.linalg.LinAlgError:
            continue

    bounds = (
        [-20.0, -30.0, -30.0, 0.01],
        [30.0, 30.0, 30.0, 30.0],
    )

    loss_method = robust_loss if robust_loss in ("huber", "soft_l1") else "linear"

    popt, _ = curve_fit(
        nelson_siegel,
        tenors,
        y,
        p0=best_p0,
        bounds=bounds,
        loss=loss_method,
        maxfev=15_000,
    )

    beta0, beta1, beta2, lam = popt
    fitted_scaled = nelson_siegel(tenors, beta0, beta1, beta2, lam)
    res_scaled = y - fitted_scaled

    # Rescale back to original user unit
    beta0_orig = beta0 / scale
    beta1_orig = beta1 / scale
    beta2_orig = beta2 / scale
    fitted = fitted_scaled / scale
    residuals = y_raw - fitted

    n = len(tenors)
    k_params = 4
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y_raw - np.mean(y_raw)) ** 2))
    r_squared = float(1.0 - (rss / tss)) if tss > 1e-12 else 1.0
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    max_error = float(np.max(np.abs(residuals)))

    # Information criteria
    aic = float(n * math.log(max(rss / n, 1e-12)) + 2.0 * k_params)
    bic = float(n * math.log(max(rss / n, 1e-12)) + k_params * math.log(n))

    return {
        "beta_0": float(beta0_orig),
        "beta_1": float(beta1_orig),
        "beta_2": float(beta2_orig),
        "lambda_param": float(lam),
        "fitted_yields": fitted.tolist(),
        "residuals": residuals.tolist(),
        "rmse": rmse,
        "mae": mae,
        "max_error": max_error,
        "r_squared": r_squared,
        "aic": aic,
        "bic": bic,
        "unit_convention": detected_unit,
        "warnings": warnings,
    }


def test_nelson_siegel_recovery(
    true_beta0: float = 5.0,
    true_beta1: float = -2.0,
    true_beta2: float = 3.0,
    true_lam: float = 2.0,
    noise_std: float = 0.02,
) -> dict:
    """Benchmark parameter recovery test under synthetic noisy curve."""
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    true_yields = nelson_siegel(tenors, true_beta0, true_beta1, true_beta2, true_lam)
    rng = np.random.default_rng(42)
    noisy_yields = true_yields + rng.normal(0.0, noise_std, size=len(tenors))

    fit = fit_nelson_siegel(tenors, noisy_yields)

    err_b0 = abs(fit["beta_0"] - true_beta0)
    err_b1 = abs(fit["beta_1"] - true_beta1)
    err_b2 = abs(fit["beta_2"] - true_beta2)
    err_lam = abs(fit["lambda_param"] - true_lam)

    recovered = bool(err_b0 < 0.25 and err_b1 < 0.35 and err_b2 < 0.5 and err_lam < 0.5)

    return {
        "parameter_recovery_success": recovered,
        "true_params": {"beta_0": true_beta0, "beta_1": true_beta1, "beta_2": true_beta2, "lambda": true_lam},
        "recovered_params": {
            "beta_0": fit["beta_0"],
            "beta_1": fit["beta_1"],
            "beta_2": fit["beta_2"],
            "lambda": fit["lambda_param"],
        },
        "rmse": fit["rmse"],
    }


def johansen_cointegration_test(
    data: np.ndarray,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """Perform Johansen cointegration rank test."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    data_arr = np.asarray(data, dtype=float)
    if data_arr.ndim != 2 or data_arr.shape[1] < 2:
        raise ValueError("Data must be a 2D array with at least 2 series.")

    res = coint_johansen(data_arr, det_order=det_order, k_ar_diff=k_ar_diff)

    trace_stat = [float(np.real(s)) for s in res.lr1]
    max_eig_stat = [float(np.real(s)) for s in res.lr2]
    trace_crit = res.cvt.tolist()
    max_eig_crit = res.cvm.tolist()

    r_95 = 0
    for i in range(len(trace_stat)):
        if trace_stat[i] > trace_crit[i][1]:
            r_95 = i + 1

    return {
        "trace_stat": trace_stat,
        "trace_crit_vals": trace_crit,
        "max_eig_stat": max_eig_stat,
        "max_eig_crit_vals": max_eig_crit,
        "eigenvectors": res.evec.tolist(),
        "r_95": r_95,
    }


def fit_vecm_model(
    data: np.ndarray,
    k_ar_diff: int = 1,
    coint_rank: int = 1,
    deterministic: str = "co",
) -> dict:
    """Fit a Vector Error Correction Model (VECM)."""
    from statsmodels.tsa.vector_ar.vecm import VECM

    data_arr = np.asarray(data, dtype=float)
    if data_arr.ndim != 2 or data_arr.shape[1] < 2:
        raise ValueError("Data must be a 2D array with at least 2 series.")

    model = VECM(
        data_arr,
        k_ar_diff=k_ar_diff,
        coint_rank=coint_rank,
        deterministic=deterministic,
    )
    results = model.fit()

    return {
        "alpha": results.alpha.tolist(),
        "beta": results.beta.tolist(),
        "gamma": results.gamma.tolist() if results.gamma is not None else [],
        "llf": float(np.real(results.llf)),
        "sigma_u": results.sigma_u.tolist(),
    }
