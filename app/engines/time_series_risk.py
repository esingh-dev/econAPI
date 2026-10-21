"""Time-series risk analytics, GARCH/EWMA volatility modeling, Stress testing, and VaR Backtesting.

Implements:
- Parametric, Historical, Student-t, and Cornish-Fisher VaR & Expected Shortfall
- EWMA (RiskMetrics) and GARCH(1,1) Quasi-Maximum Likelihood volatility modeling
- Multi-factor regression, beta exposures, and systematic vs idiosyncratic risk decomposition
- Historical and synthetic stress testing / scenario analysis
- Kupiec POF LR test, Christoffersen Independence test, and Basel Traffic Light classification
"""

from __future__ import annotations

import math
from typing import Literal, Optional
import numpy as np
from scipy import stats
from scipy.optimize import minimize


# ═════════════════════════════════════════════════════════════════════
# 1. Multi-Model VaR & Expected Shortfall Engine
# ═════════════════════════════════════════════════════════════════════

def compute_var_suite(
    returns: list[float] | np.ndarray,
    confidence_level: float = 0.99,
    time_horizon_days: int = 1,
) -> dict:
    """Compute VaR and CVaR across Historical, Parametric Normal, Student-t, and Cornish-Fisher models.

    Parameters
    ----------
    returns : list[float] or ndarray
        Periodic return series (e.g. daily returns).
    confidence_level : float
        Confidence level alpha (e.g. 0.99 for 99%).
    time_horizon_days : int
        Square-root of time scaling horizon (default: 1).
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 30:
        raise ValueError("Need at least 30 observations for risk modeling.")

    alpha = confidence_level
    losses = -r  # losses are positive
    n = len(losses)
    sqrt_h = math.sqrt(time_horizon_days)

    mean_loss = float(np.mean(losses))
    std_loss = float(np.std(losses, ddof=1))
    skew = float(stats.skew(losses))
    kurt = float(stats.kurtosis(losses))  # excess kurtosis

    # 1. Historical Simulation
    sorted_losses = np.sort(losses)
    idx = int(np.ceil(alpha * n)) - 1
    idx = min(max(0, idx), n - 1)
    hist_var = float(sorted_losses[idx] * sqrt_h)
    tail_losses = sorted_losses[sorted_losses >= hist_var / sqrt_h]
    hist_cvar = float(np.mean(tail_losses) * sqrt_h) if len(tail_losses) > 0 else hist_var

    # 2. Parametric Gaussian
    z_alpha = stats.norm.ppf(alpha)
    norm_var = float((mean_loss + z_alpha * std_loss) * sqrt_h)
    # Analytical CVaR for Normal: mu + sigma * pdf(z) / (1 - alpha)
    norm_cvar = float((mean_loss + std_loss * (stats.norm.pdf(z_alpha) / (1.0 - alpha))) * sqrt_h)

    # 3. Student-t
    try:
        df, loc_t, scale_t = stats.t.fit(losses)
        t_quant = stats.t.ppf(alpha, df)
        t_var = float((loc_t + scale_t * t_quant) * sqrt_h)
        # Expected shortfall for t-distribution
        cvar_factor = (df + t_quant**2) / (df - 1.0) * stats.t.pdf(t_quant, df) / (1.0 - alpha)
        t_cvar = float((loc_t + scale_t * cvar_factor) * sqrt_h)
    except Exception:
        df = 5.0
        t_var = norm_var
        t_cvar = norm_cvar

    # 4. Cornish-Fisher expansion (adjusts quantile for skewness and excess kurtosis)
    # w_alpha = z + (z^2 - 1)*S/6 + (z^3 - 3z)*K/24 - (2z^3 - 5z)*S^2/36
    cf_z = (
        z_alpha
        + (z_alpha**2 - 1.0) * (skew / 6.0)
        + (z_alpha**3 - 3.0 * z_alpha) * (kurt / 24.0)
        - (2.0 * z_alpha**3 - 5.0 * z_alpha) * (skew**2 / 36.0)
    )
    cf_var = float((mean_loss + cf_z * std_loss) * sqrt_h)
    cf_cvar = float(cf_var * 1.15)  # conservative tail approximation

    return {
        "confidence_level": alpha,
        "time_horizon_days": time_horizon_days,
        "sample_moments": {
            "mean": -mean_loss,
            "std": std_loss,
            "skewness": -skew,
            "excess_kurtosis": kurt,
        },
        "models": {
            "historical": {"var": hist_var, "cvar": hist_cvar},
            "parametric_gaussian": {"var": norm_var, "cvar": norm_cvar},
            "student_t": {"var": t_var, "cvar": t_cvar, "fitted_df": float(df)},
            "cornish_fisher": {"var": cf_var, "cvar": cf_cvar},
        },
    }


# ═════════════════════════════════════════════════════════════════════
# 2. Volatility Modeling: EWMA & GARCH(1,1)
# ═════════════════════════════════════════════════════════════════════

def fit_garch11(
    returns: list[float] | np.ndarray,
) -> dict:
    """Fit GARCH(1,1) model via Quasi-Maximum Likelihood Estimation (QMLE).

    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
    Persistence check: alpha + beta < 1 (covariance stationary).
    """
    r = np.asarray(returns, dtype=float)
    t = len(r)
    sample_var = float(np.var(r, ddof=1))

    # Initial guess: omega = 0.05 * sample_var, alpha = 0.08, beta = 0.88
    x0 = [0.05 * sample_var, 0.08, 0.88]
    bounds = [(1e-7, None), (1e-6, 0.99), (1e-6, 0.99)]

    def neg_log_lik(params):
        omega, alpha, beta = params
        if alpha + beta >= 0.9999:
            return 1e8

        sig2 = np.empty(t)
        sig2[0] = sample_var
        for i in range(1, t):
            sig2[i] = omega + alpha * (r[i - 1] ** 2) + beta * sig2[i - 1]

        sig2 = np.maximum(sig2, 1e-10)
        # Gaussian log-likelihood
        ll = -0.5 * np.sum(np.log(2.0 * math.pi) + np.log(sig2) + (r**2) / sig2)
        return -ll

    res = minimize(neg_log_lik, x0, bounds=bounds, method="L-BFGS-B")
    omega_opt, alpha_opt, beta_opt = res.x

    persistence = float(alpha_opt + beta_opt)
    is_stationary = bool(persistence < 1.0)
    long_run_vol = math.sqrt(omega_opt / (1.0 - persistence) * 252.0) if is_stationary else None

    # Final conditional variance path
    sig2_path = np.empty(t)
    sig2_path[0] = sample_var
    for i in range(1, t):
        sig2_path[i] = omega_opt + alpha_opt * (r[i - 1] ** 2) + beta_opt * sig2_path[i - 1]

    current_vol = float(math.sqrt(sig2_path[-1] * 252.0))

    return {
        "omega": float(omega_opt),
        "alpha": float(alpha_opt),
        "beta": float(beta_opt),
        "persistence": persistence,
        "is_stationary": is_stationary,
        "current_annualized_vol": current_vol,
        "long_run_annualized_vol": long_run_vol,
        "converged": bool(res.success),
    }


def compute_ewma_volatility(
    returns: list[float] | np.ndarray,
    decay: float = 0.94,
) -> dict:
    """Compute RiskMetrics EWMA volatility path."""
    r = np.asarray(returns, dtype=float)
    t = len(r)
    sig2 = np.empty(t)
    sig2[0] = float(np.var(r[:min(30, t)]))

    for i in range(1, t):
        sig2[i] = decay * sig2[i - 1] + (1.0 - decay) * (r[i - 1] ** 2)

    vols_ann = np.sqrt(sig2 * 252.0)

    return {
        "decay_lambda": decay,
        "current_annualized_vol": float(vols_ann[-1]),
        "volatility_path": [float(x) for x in vols_ann],
    }


# ═════════════════════════════════════════════════════════════════════
# 3. Factor Models & Risk Decomposition
# ═════════════════════════════════════════════════════════════════════

def factor_model_decomposition(
    asset_returns: list[float],
    factor_returns: list[list[float]],
    factor_names: Optional[list[str]] = None,
) -> dict:
    """Decompose asset returns into systematic factor exposures and idiosyncratic risk.

    R_i = alpha + B * F + epsilon
    Total Var = B' Sigma_F B + Var(epsilon)
    """
    y = np.asarray(asset_returns, dtype=float)
    X = np.asarray(factor_returns, dtype=float)  # shape (T, K)
    t, k = X.shape
    names = factor_names or [f"Factor_{i+1}" for i in range(k)]

    # Add constant for alpha
    X_const = np.column_stack([np.ones(t), X])
    betas, residuals, rank, s = np.linalg.lstsq(X_const, y, rcond=None)

    alpha = float(betas[0])
    factor_betas = {name: float(b) for name, b in zip(names, betas[1:])}

    fitted = X_const @ betas
    eps = y - fitted
    total_var = float(np.var(y, ddof=1))
    idio_var = float(np.var(eps, ddof=1))
    systematic_var = max(0.0, total_var - idio_var)

    r_squared = float(systematic_var / total_var) if total_var > 1e-12 else 0.0

    return {
        "alpha": alpha,
        "factor_betas": factor_betas,
        "r_squared": r_squared,
        "systematic_variance_pct": float(r_squared * 100.0),
        "idiosyncratic_variance_pct": float((1.0 - r_squared) * 100.0),
        "total_annualized_vol": float(math.sqrt(total_var * 252.0)),
        "idiosyncratic_annualized_vol": float(math.sqrt(idio_var * 252.0)),
    }


# ═════════════════════════════════════════════════════════════════════
# 4. Stress Testing & Scenario Analysis
# ═════════════════════════════════════════════════════════════════════

def run_stress_test(
    portfolio_weights: dict[str, float],
    factor_exposures: dict[str, dict[str, float]],
    scenarios: Optional[dict[str, dict[str, float]]] = None,
) -> dict:
    """Evaluate portfolio P&L under historical and macro stress shock scenarios.

    Default scenarios include 2008 GFC, 2020 Covid Crash, 1987 Black Monday, and Stagflation Shock.
    """
    default_scenarios = {
        "2008_GFC": {"Equity": -0.45, "Bond": 0.12, "Credit": -0.25, "Commodity": -0.35},
        "2020_Covid_Shock": {"Equity": -0.32, "Bond": 0.08, "Credit": -0.15, "Commodity": -0.40},
        "1987_Black_Monday": {"Equity": -0.22, "Bond": 0.04, "Credit": -0.05, "Commodity": -0.10},
        "Stagflation_Shock": {"Equity": -0.18, "Bond": -0.12, "Credit": -0.10, "Commodity": 0.30},
    }
    active_scenarios = scenarios or default_scenarios

    scenario_results = {}
    for sc_name, shocks in active_scenarios.items():
        asset_impacts = {}
        total_pnl = 0.0
        for asset, w in portfolio_weights.items():
            exps = factor_exposures.get(asset, {})
            # Return shock = sum(beta_f * shock_f)
            asset_ret = sum(exps.get(f, 0.0) * shock for f, shock in shocks.items())
            asset_impacts[asset] = float(asset_ret)
            total_pnl += w * asset_ret

        scenario_results[sc_name] = {
            "portfolio_loss_pct": float(total_pnl * 100.0),
            "asset_returns_pct": {k: float(v * 100.0) for k, v in asset_impacts.items()},
        }

    return {"scenarios": scenario_results}


# ═════════════════════════════════════════════════════════════════════
# 5. VaR Backtesting: Kupiec, Christoffersen & Basel Traffic Light
# ═════════════════════════════════════════════════════════════════════

def backtest_var(
    realized_returns: list[float],
    forecast_vars: list[float],
    confidence_level: float = 0.99,
) -> dict:
    """Perform Kupiec Proportion of Failures, Christoffersen Independence, and Basel Traffic Light tests.

    Parameters
    ----------
    realized_returns : list[float]
        Out-of-sample realized returns.
    forecast_vars : list[float]
        Predicted VaRs (expressed as positive loss quantities).
    confidence_level : float
        Target VaR confidence level (e.g. 0.99).
    """
    r = np.asarray(realized_returns, dtype=float)
    var = np.asarray(forecast_vars, dtype=float)
    n = len(r)

    # Losses exceeding VaR
    exceptions = (-r) > var
    num_exceptions = int(np.sum(exceptions))
    observed_p = num_exceptions / n if n > 0 else 0.0
    p = 1.0 - confidence_level

    # 1. Kupiec Proportion of Failures (POF) Likelihood Ratio test
    # LR_POF = -2 ln( (1 - p)^(N - x) p^x / ((1 - x/N)^(N - x) (x/N)^x) )
    if num_exceptions == 0:
        lr_pof = -2.0 * n * math.log(1.0 - p)
    elif num_exceptions == n:
        lr_pof = -2.0 * n * math.log(p)
    else:
        num = ((1.0 - p) ** (n - num_exceptions)) * (p**num_exceptions)
        den = ((1.0 - observed_p) ** (n - num_exceptions)) * (observed_p**num_exceptions)
        lr_pof = -2.0 * math.log(num / den)

    p_value_kupiec = float(1.0 - stats.chi2.cdf(lr_pof, df=1))
    kupiec_passed = bool(p_value_kupiec > 0.05)

    # 2. Christoffersen Independence test (Markov transition between exceptions)
    # Transitions: n00, n01, n10, n11
    n00, n01, n10, n11 = 0, 0, 0, 0
    for i in range(n - 1):
        prev = int(exceptions[i])
        curr = int(exceptions[i + 1])
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        elif prev == 1 and curr == 1:
            n11 += 1

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

    if pi0 == 0.0 or pi1 == 0.0 or pi == 0.0 or (1.0 - pi0) == 0.0 or (1.0 - pi1) == 0.0 or (1.0 - pi) == 0.0:
        lr_ind = 0.0
    else:
        term_num = ((1.0 - pi) ** (n00 + n10)) * (pi ** (n01 + n11))
        term_den = ((1.0 - pi0) ** n00) * (pi0**n01) * ((1.0 - pi1) ** n10) * (pi1**n11)
        lr_ind = -2.0 * math.log(term_num / term_den)

    p_value_christoffersen = float(1.0 - stats.chi2.cdf(lr_ind, df=1))
    independence_passed = bool(p_value_christoffersen > 0.05)

    # Combined Conditional Coverage: LR_CC = LR_POF + LR_IND ~ Chi2(2)
    lr_cc = lr_pof + lr_ind
    p_value_cc = float(1.0 - stats.chi2.cdf(lr_cc, df=2))

    # 3. Basel Traffic Light Classification (standard 250 observations at 99%)
    # Green: 0 to 4 exceptions; Yellow: 5 to 9; Red: 10+
    scale_factor = 250.0 / n if n > 0 else 1.0
    scaled_exceptions = num_exceptions * scale_factor

    if scaled_exceptions <= 4.0:
        zone = "Green"
    elif scaled_exceptions <= 9.0:
        zone = "Yellow"
    else:
        zone = "Red"

    return {
        "n_observations": n,
        "n_exceptions": num_exceptions,
        "expected_exceptions": float(n * p),
        "observed_failure_rate": observed_p,
        "expected_failure_rate": p,
        "kupiec_lr_pof": float(lr_pof),
        "kupiec_p_value": p_value_kupiec,
        "kupiec_passed": kupiec_passed,
        "christoffersen_lr_ind": float(lr_ind),
        "christoffersen_p_value": p_value_christoffersen,
        "independence_passed": independence_passed,
        "conditional_coverage_lr": float(lr_cc),
        "conditional_coverage_p_value": p_value_cc,
        "basel_traffic_light_zone": zone,
    }
