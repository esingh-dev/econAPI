"""Macro-econometric models — Nelson-Siegel yield curve fitting.

Fits the three-factor Nelson-Siegel parametric term structure model
to observed bond yields via non-linear least squares.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def nelson_siegel(
    tau: np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    lam: float,
) -> np.ndarray:
    """Evaluate the Nelson-Siegel yield curve.

    y(τ) = β₀ + β₁ · [(1 − e^(−τ/λ)) / (τ/λ)]
              + β₂ · [(1 − e^(−τ/λ)) / (τ/λ) − e^(−τ/λ)]

    Parameters
    ----------
    tau : ndarray
        Maturities in years.
    beta0 : float
        Level (long-term rate).
    beta1 : float
        Slope factor.
    beta2 : float
        Curvature factor.
    lam : float
        Decay parameter λ > 0.

    Returns
    -------
    ndarray
        Fitted yields of same shape as *tau*.
    """
    tau = np.asarray(tau, dtype=float)

    # Protect against division by zero when τ = 0
    # L'Hôpital: lim_{τ→0} (1 − e^{-τ/λ})/(τ/λ) = 1
    x = np.where(tau > 1e-12, tau / lam, 1e-12)
    exp_neg_x = np.exp(-x)

    factor1 = np.where(tau > 1e-12, (1.0 - exp_neg_x) / x, 1.0)
    factor2 = factor1 - exp_neg_x

    return beta0 + beta1 * factor1 + beta2 * factor2


def fit_nelson_siegel(
    tenors: np.ndarray,
    yields: np.ndarray,
) -> dict:
    """Fit the Nelson-Siegel model to an observed term structure.

    Uses ``scipy.optimize.curve_fit`` with sensible initial guesses and
    bounded parameter space.

    Parameters
    ----------
    tenors : ndarray
        Maturity tenors in years (must be > 0).
    yields : ndarray
        Observed market yields (same length as *tenors*).

    Returns
    -------
    dict
        beta_0, beta_1, beta_2, lambda_param : fitted parameters
        fitted_yields : list[float]
        residuals : list[float]
        rmse : float
    """
    tenors = np.asarray(tenors, dtype=float)
    yields = np.asarray(yields, dtype=float)

    if len(tenors) != len(yields):
        raise ValueError("tenors and yields must have the same length.")
    if len(tenors) < 4:
        raise ValueError("Need at least 4 data points for Nelson-Siegel fit.")

    # Diebold-Li style two-step initialization:
    # Grid search over lambda to initialize beta parameters via OLS
    best_rmse = float("inf")
    best_p0 = [yields[-1], yields[0] - yields[-1], 0.0, 1.0]

    for lam_candidate in np.linspace(0.1, 10.0, 50):
        x = np.where(tenors > 1e-12, tenors / lam_candidate, 1e-12)
        exp_neg = np.exp(-x)
        f1 = np.where(tenors > 1e-12, (1.0 - exp_neg) / x, 1.0)
        f2 = f1 - exp_neg
        X = np.column_stack([np.ones_like(tenors), f1, f2])
        try:
            betas, _, _, _ = np.linalg.lstsq(X, yields, rcond=None)
            pred = X @ betas
            curr_rmse = np.sqrt(np.mean((yields - pred) ** 2))
            if curr_rmse < best_rmse:
                best_rmse = curr_rmse
                best_p0 = [float(betas[0]), float(betas[1]), float(betas[2]), float(lam_candidate)]
        except np.linalg.LinAlgError:
            continue

    bounds = (
        [-20.0, -30.0, -30.0, 0.01],   # lower
        [ 30.0,  30.0,  30.0, 30.0],   # upper
    )

    popt, _ = curve_fit(
        nelson_siegel,
        tenors,
        yields,
        p0=best_p0,
        bounds=bounds,
        maxfev=10_000,
    )

    beta0, beta1, beta2, lam = popt
    fitted = nelson_siegel(tenors, beta0, beta1, beta2, lam)
    residuals = yields - fitted
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "beta_0": float(beta0),
        "beta_1": float(beta1),
        "beta_2": float(beta2),
        "lambda_param": float(lam),
        "fitted_yields": fitted.tolist(),
        "residuals": residuals.tolist(),
        "rmse": rmse,
    }


def johansen_cointegration_test(
    data: np.ndarray,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """Perform Johansen cointegration rank test.

    Parameters
    ----------
    data : ndarray, shape (T, K)
        Multivariate time series data (T periods, K series).
    det_order : int
        Deterministic terms: -1 for no constant, 0 for constant, 1 for trend.
    k_ar_diff : int
        Number of lagged differences in VAR representation.

    Returns
    -------
    dict
        trace_stat : list[float]
        trace_crit_vals : list[list[float]] (90%, 95%, 99%)
        max_eig_stat : list[float]
        max_eig_crit_vals : list[list[float]]
        eigenvectors : list[list[float]]
        r_95 : int (number of cointegrating vectors at 95% level)
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    data_arr = np.asarray(data, dtype=float)
    if data_arr.ndim != 2 or data_arr.shape[1] < 2:
        raise ValueError("Data must be a 2D array with at least 2 series.")

    res = coint_johansen(data_arr, det_order=det_order, k_ar_diff=k_ar_diff)

    trace_stat = [float(np.real(s)) for s in res.lr1]
    max_eig_stat = [float(np.real(s)) for s in res.lr2]
    trace_crit = res.cvt.tolist()  # shape (K, 3)
    max_eig_crit = res.cvm.tolist()

    # Determine cointegration rank at 95% significance (column index 1 in cvt)
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
    """Fit a Vector Error Correction Model (VECM).

    Parameters
    ----------
    data : ndarray, shape (T, K)
        Multivariate time series data.
    k_ar_diff : int
        Number of lagged difference terms.
    coint_rank : int
        Cointegration rank (number of cointegrating vectors).
    deterministic : str
        Deterministic specification: 'nc', 'co', 'ci', 'lo', 'li'.

    Returns
    -------
    dict
        alpha : list[list[float]] (speed of adjustment matrix)
        beta : list[list[float]] (cointegrating vectors)
        gamma : list[list[float]] (short-run dynamic coefficients)
        aic : float
        bic : float
    """
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
