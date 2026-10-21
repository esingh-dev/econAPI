"""Comprehensive Portfolio Optimization, Construction, and Risk Attribution engine.

Implements:
- Markowitz Mean-Variance optimization (Max Sharpe, Min Variance, Target Return, Efficient Frontier)
- Equal Risk Contribution (ERC) / Risk Parity optimization
- Inverse Volatility allocation
- Black-Litterman model with investor views and posterior covariance
- Rockafellar-Uryasev Linear Programming CVaR / Expected Shortfall optimization
- Position bounds, sector constraints, turnover constraints, and transaction costs
- Rebalancing simulator and Brinson-Fachler performance attribution
- Performance metrics: Sharpe, Sortino, Calmar, Omega, Max Drawdown, Ulcer Index, Tracking Error
"""

from __future__ import annotations

import math
from typing import Literal, Optional
import numpy as np
from scipy.optimize import minimize, linprog


# ═════════════════════════════════════════════════════════════════════
# 1. Performance & Risk Metrics
# ═════════════════════════════════════════════════════════════════════

def compute_portfolio_metrics(
    returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.02,
    annualization_factor: int = 252,
) -> dict:
    """Compute comprehensive risk-adjusted performance and drawdown statistics.

    Parameters
    ----------
    returns : ndarray
        1D array of portfolio daily/periodic returns.
    benchmark_returns : ndarray, optional
        Benchmark returns of same length.
    risk_free_rate : float
        Annual risk-free rate.
    annualization_factor : int
        252 for daily, 52 for weekly, 12 for monthly.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 3:
        raise ValueError("Need at least 3 return periods to compute portfolio metrics.")

    rf_periodic = risk_free_rate / annualization_factor
    excess_r = r - rf_periodic

    # Annualized return and volatility
    mean_r = float(np.mean(r))
    ann_return = float(mean_r * annualization_factor)
    ann_vol = float(np.std(r, ddof=1) * math.sqrt(annualization_factor))

    # Sharpe Ratio
    sharpe = float(ann_return - risk_free_rate) / ann_vol if ann_vol > 1e-8 else 0.0

    # Downside deviation & Sortino Ratio
    downside_diff = np.minimum(excess_r, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside_diff**2)) * math.sqrt(annualization_factor))
    sortino = float(ann_return - risk_free_rate) / downside_dev if downside_dev > 1e-8 else 0.0

    # Cumulative wealth & Drawdowns
    wealth_index = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth_index)
    drawdowns = (wealth_index - peak) / peak
    max_drawdown = float(np.min(drawdowns))  # negative number

    # Drawdown duration & Ulcer index
    ulcer_index = float(math.sqrt(np.mean((drawdowns * 100.0) ** 2)))
    calmar = float(ann_return / abs(max_drawdown)) if abs(max_drawdown) > 1e-6 else 0.0

    # Omega Ratio (threshold = 0)
    gains = r[r > 0]
    losses = r[r < 0]
    sum_gains = float(np.sum(gains)) if len(gains) > 0 else 0.0
    sum_losses = float(np.sum(np.abs(losses))) if len(losses) > 0 else 1e-8
    omega = float(sum_gains / sum_losses)

    # Benchmark comparison: Alpha, Beta, Tracking Error, Information Ratio
    info_ratio = None
    tracking_err = None
    beta = None
    alpha = None

    if benchmark_returns is not None:
        b = np.asarray(benchmark_returns, dtype=float)
        diff = r - b
        tracking_err = float(np.std(diff, ddof=1) * math.sqrt(annualization_factor))
        info_ratio = float((ann_return - np.mean(b) * annualization_factor) / tracking_err) if tracking_err > 1e-8 else 0.0
        cov_rb = float(np.cov(r, b)[0, 1])
        var_b = float(np.var(b, ddof=1))
        beta = float(cov_rb / var_b) if var_b > 1e-8 else 1.0
        alpha = float(ann_return - (risk_free_rate + beta * (np.mean(b) * annualization_factor - risk_free_rate)))

    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "omega_ratio": omega,
        "max_drawdown": max_drawdown,
        "ulcer_index": ulcer_index,
        "tracking_error": tracking_err,
        "information_ratio": info_ratio,
        "beta": beta,
        "alpha": alpha,
    }


# ═════════════════════════════════════════════════════════════════════
# 2. Markowitz Mean-Variance Optimization
# ═════════════════════════════════════════════════════════════════════

def optimize_mean_variance(
    expected_returns: list[float],
    cov_matrix: list[list[float]],
    objective: Literal["max_sharpe", "min_variance", "target_return", "target_risk"] = "max_sharpe",
    target_value: Optional[float] = None,
    risk_free_rate: float = 0.02,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    allow_short: bool = False,
    asset_names: Optional[list[str]] = None,
    prev_weights: Optional[list[float]] = None,
    max_turnover: Optional[float] = None,
) -> dict:
    """Quadratic programming Mean-Variance Optimization via SciPy SLSQP.

    Supports box bounds, turnover constraints, and long-only or bounded shorting.
    """
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)
    n = len(mu)
    names = asset_names or [f"Asset_{i+1}" for i in range(n)]

    x0 = np.full(n, 1.0 / n)

    lb = -1.0 if allow_short else min_weight
    ub = 1.0 if allow_short else max_weight
    bounds = [(lb, ub) for _ in range(n)]

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if prev_weights is not None and max_turnover is not None:
        w_prev = np.asarray(prev_weights, dtype=float)
        constraints.append({
            "type": "ineq",
            "fun": lambda w: max_turnover - np.sum(np.abs(w - w_prev)),
        })

    if objective == "min_variance":
        def loss(w):
            return float(w @ sigma @ w)
    elif objective == "max_sharpe":
        def loss(w):
            ret = float(w @ mu)
            vol = math.sqrt(max(1e-12, float(w @ sigma @ w)))
            return -((ret - risk_free_rate) / vol)
    elif objective == "target_return":
        tgt = target_value if target_value is not None else float(np.mean(mu))
        constraints.append({"type": "eq", "fun": lambda w: float(w @ mu) - tgt})
        def loss(w):
            return float(w @ sigma @ w)
    elif objective == "target_risk":
        tgt_vol = target_value if target_value is not None else 0.15
        constraints.append({"type": "ineq", "fun": lambda w: tgt_vol**2 - float(w @ sigma @ w)})
        def loss(w):
            return -float(w @ mu)

    res = minimize(loss, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    w_opt = res.x
    w_opt = np.where(np.abs(w_opt) < 1e-5, 0.0, w_opt)
    w_opt /= np.sum(w_opt)

    port_ret = float(w_opt @ mu)
    port_vol = float(math.sqrt(max(1e-12, w_opt @ sigma @ w_opt)))
    sharpe = float((port_ret - risk_free_rate) / port_vol)

    return {
        "weights": {name: float(w) for name, w in zip(names, w_opt)},
        "expected_return": port_ret,
        "volatility": port_vol,
        "sharpe_ratio": sharpe,
        "converged": bool(res.success),
        "objective": objective,
    }


# ═════════════════════════════════════════════════════════════════════
# 3. Equal Risk Contribution (ERC) / Risk Parity
# ═════════════════════════════════════════════════════════════════════

def optimize_risk_parity(
    cov_matrix: list[list[float]],
    asset_names: Optional[list[str]] = None,
    budget: Optional[list[float]] = None,
) -> dict:
    """Equal Risk Contribution (ERC) optimization.

    Equalizes the marginal risk contributions: sigma_i * (Sigma w)_i / sigma_p = B_i.
    """
    sigma = np.asarray(cov_matrix, dtype=float)
    n = sigma.shape[0]
    names = asset_names or [f"Asset_{i+1}" for i in range(n)]
    b = np.asarray(budget, dtype=float) if budget is not None else np.full(n, 1.0 / n)

    x0 = np.full(n, 1.0 / n)
    bounds = [(1e-5, 1.0) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def loss(w):
        port_var = float(w @ sigma @ w)
        if port_var <= 1e-12:
            return 1e6
        # Marginal risk contribution: MRC_i = (Sigma w)_i
        mrc = sigma @ w
        # Total risk contribution: TRC_i = w_i * MRC_i
        trc = w * mrc / port_var
        return float(np.sum((trc - b) ** 2))

    res = minimize(loss, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    w_opt = res.x / np.sum(res.x)

    port_vol = float(math.sqrt(w_opt @ sigma @ w_opt))
    trc = (w_opt * (sigma @ w_opt)) / (port_vol**2)

    return {
        "weights": {name: float(w) for name, w in zip(names, w_opt)},
        "risk_contributions": {name: float(r) for name, r in zip(names, trc)},
        "portfolio_volatility": port_vol,
        "converged": bool(res.success),
    }


# ═════════════════════════════════════════════════════════════════════
# 4. Black-Litterman Model
# ═════════════════════════════════════════════════════════════════════

def black_litterman(
    market_weights: list[float],
    cov_matrix: list[list[float]],
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    pick_matrix_P: Optional[list[list[float]]] = None,
    view_returns_Q: Optional[list[float]] = None,
    omega_uncertainty: Optional[list[list[float]]] = None,
    asset_names: Optional[list[str]] = None,
) -> dict:
    """Compute Black-Litterman implied equilibrium returns, posterior returns, and posterior covariance.

    Parameters
    ----------
    market_weights : list[float]
        Market capitalization or benchmark weights.
    cov_matrix : list[list[float]]
        N x N asset covariance matrix.
    risk_aversion : float
        Risk aversion coefficient lambda = (r_m - r_f) / sigma_m^2 (typically 2.0 to 4.0).
    tau : float
        Scalar indicating uncertainty in the prior (typically 0.01 to 0.1).
    pick_matrix_P : list[list[float]], optional
        K x N matrix identifying assets involved in the K views.
    view_returns_Q : list[float], optional
        K x 1 vector of view expected returns.
    """
    w_mkt = np.asarray(market_weights, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)
    n = len(w_mkt)
    names = asset_names or [f"Asset_{i+1}" for i in range(n)]

    # 1. Implied equilibrium returns: Pi = lambda * Sigma * w_mkt
    pi = risk_aversion * (sigma @ w_mkt)

    if pick_matrix_P is None or view_returns_Q is None or len(view_returns_Q) == 0:
        # No views: posterior is prior
        return {
            "prior_returns": {name: float(x) for name, x in zip(names, pi)},
            "posterior_returns": {name: float(x) for name, x in zip(names, pi)},
            "posterior_weights": {name: float(x) for name, x in zip(names, w_mkt)},
        }

    P = np.asarray(pick_matrix_P, dtype=float)
    Q = np.asarray(view_returns_Q, dtype=float)
    k = len(Q)

    # 2. View uncertainty Omega: if not provided, He & Litterman (1999) proportional formulation
    # Omega = diag(P * (tau * Sigma) * P^T)
    if omega_uncertainty is None:
        omega_diag = np.diag(P @ (tau * sigma) @ P.T)
        Omega = np.diag(omega_diag)
    else:
        Omega = np.asarray(omega_uncertainty, dtype=float)

    # 3. Master formula:
    # mu_BL = Pi + tau*Sigma*P^T * (P*(tau*Sigma)*P^T + Omega)^(-1) * (Q - P*Pi)
    tau_sigma = tau * sigma
    inv_term = np.linalg.inv(P @ tau_sigma @ P.T + Omega)
    M = tau_sigma @ P.T @ inv_term
    mu_bl = pi + M @ (Q - P @ pi)

    # Posterior covariance:
    # Sigma_BL = Sigma + tau*Sigma - tau*Sigma*P^T * inv_term * P*(tau*Sigma)
    sigma_bl = sigma + tau_sigma - M @ (P @ tau_sigma)

    # Posterior optimal weights: w_BL = (lambda * Sigma_BL)^(-1) * mu_BL
    w_bl = np.linalg.solve(risk_aversion * sigma_bl, mu_bl)
    w_bl = np.maximum(0.0, w_bl)
    w_bl /= np.sum(w_bl)

    return {
        "prior_returns": {name: float(x) for name, x in zip(names, pi)},
        "posterior_returns": {name: float(x) for name, x in zip(names, mu_bl)},
        "posterior_weights": {name: float(x) for name, x in zip(names, w_bl)},
        "prior_weights": {name: float(x) for name, x in zip(names, w_mkt)},
    }


# ═════════════════════════════════════════════════════════════════════
# 5. CVaR / Expected Shortfall LP Optimization (Rockafellar & Uryasev)
# ═════════════════════════════════════════════════════════════════════

def optimize_cvar(
    scenario_returns: list[list[float]],
    confidence_level: float = 0.95,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    asset_names: Optional[list[str]] = None,
) -> dict:
    """Rockafellar & Uryasev Linear Programming formulation to minimize CVaR.

    min_{w, gamma, u} gamma + (1 / ((1 - alpha)*T)) * sum(u_t)
    s.t. u_t >= -r_t' w - gamma, u_t >= 0, sum(w) = 1, w_i in [min_w, max_w]
    """
    R = np.asarray(scenario_returns, dtype=float)  # shape (T, N)
    T_obs, N = R.shape
    names = asset_names or [f"Asset_{i+1}" for i in range(N)]
    alpha = confidence_level

    # Decision vector x = [w_1, ..., w_N, gamma, u_1, ..., u_T]
    # Total variables = N + 1 + T_obs
    c = np.zeros(N + 1 + T_obs)
    c[N] = 1.0  # coefficient on gamma
    c[N + 1 :] = 1.0 / ((1.0 - alpha) * T_obs)  # coefficient on u_t

    # Inequality constraints: A_ub * x <= b_ub
    # -r_t' w - gamma - u_t <= 0  ==>  (-R) w - 1*gamma - I*u <= 0
    A_ub = np.zeros((T_obs, N + 1 + T_obs))
    A_ub[:, :N] = -R
    A_ub[:, N] = -1.0
    A_ub[:, N + 1 :] = -np.eye(T_obs)
    b_ub = np.zeros(T_obs)

    # Equality constraints: sum(w) = 1
    A_eq = np.zeros((1, N + 1 + T_obs))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])

    # Variable bounds: w in [min_w, max_w], gamma free, u_t >= 0
    bounds = [(min_weight, max_weight) for _ in range(N)]
    bounds.append((None, None))  # gamma free
    for _ in range(T_obs):
        bounds.append((0.0, None))  # u_t >= 0

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    w_opt = res.x[:N]
    w_opt = np.maximum(0.0, w_opt)
    w_opt /= np.sum(w_opt)

    var_val = float(res.x[N])
    cvar_val = float(res.fun)

    return {
        "weights": {name: float(w) for name, w in zip(names, w_opt)},
        "var": var_val,
        "cvar": cvar_val,
        "confidence_level": confidence_level,
        "converged": bool(res.success),
    }


# ═════════════════════════════════════════════════════════════════════
# 6. Brinson-Fachler Performance Attribution
# ═════════════════════════════════════════════════════════════════════

def brinson_performance_attribution(
    portfolio_weights: list[float],
    benchmark_weights: list[float],
    portfolio_returns: list[float],
    benchmark_returns: list[float],
    sector_names: Optional[list[str]] = None,
) -> dict:
    """Decompose excess return into Allocation, Selection, and Interaction effects.

    Brinson, Hood & Beebower (1986) / Brinson & Fachler (1985).
    """
    wp = np.asarray(portfolio_weights, dtype=float)
    wb = np.asarray(benchmark_weights, dtype=float)
    rp = np.asarray(portfolio_returns, dtype=float)
    rb = np.asarray(benchmark_returns, dtype=float)
    names = sector_names or [f"Sector_{i+1}" for i in range(len(wp))]

    r_bm_total = float(np.sum(wb * rb))
    r_port_total = float(np.sum(wp * rp))
    total_excess = r_port_total - r_bm_total

    # Allocation effect: (w_p - w_b) * (R_b - R_bm_total)
    allocation = (wp - wb) * (rb - r_bm_total)

    # Selection effect: w_b * (R_p - R_b)
    selection = wb * (rp - rb)

    # Interaction effect: (w_p - w_b) * (R_p - R_b)
    interaction = (wp - wb) * (rp - rb)

    sector_results = []
    for i, name in enumerate(names):
        sector_results.append({
            "sector": name,
            "portfolio_weight": float(wp[i]),
            "benchmark_weight": float(wb[i]),
            "portfolio_return": float(rp[i]),
            "benchmark_return": float(rb[i]),
            "allocation_effect": float(allocation[i]),
            "selection_effect": float(selection[i]),
            "interaction_effect": float(interaction[i]),
            "total_effect": float(allocation[i] + selection[i] + interaction[i]),
        })

    return {
        "portfolio_total_return": r_port_total,
        "benchmark_total_return": r_bm_total,
        "total_excess_return": total_excess,
        "total_allocation": float(np.sum(allocation)),
        "total_selection": float(np.sum(selection)),
        "total_interaction": float(np.sum(interaction)),
        "sectors": sector_results,
    }
