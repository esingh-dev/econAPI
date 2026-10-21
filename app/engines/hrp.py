"""Hardened Hierarchical Risk Parity (HRP) portfolio optimization engine.

Implements:
- Multiple covariance estimators: Ledoit-Wolf shrinkage, standard sample covariance, EWMA covariance, semi-covariance
- Missing-data imputation: forward-fill, dropna, mean imputation
- Single-linkage clustering, quasi-diagonalization (seriation), and recursive bisection
- Minimum and maximum weight bounds (box constraints) via iterative redistribution
- Turnover-aware rebalancing and transaction cost deduction
- Walk-forward out-of-sample backtesting
"""

from __future__ import annotations

import math
from typing import Literal, Optional
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


def clean_missing_data(
    returns: np.ndarray,
    method: Literal["dropna", "forward_fill", "mean_impute"] = "dropna",
) -> np.ndarray:
    """Handle missing (NaN / Inf) values in historical return series."""
    arr = np.asarray(returns, dtype=float)
    if not np.any(np.isnan(arr)) and not np.any(np.isinf(arr)):
        return arr

    if method == "dropna":
        mask = ~np.any(np.isnan(arr) | np.isinf(arr), axis=1)
        cleaned = arr[mask]
        if len(cleaned) < 5:
            raise ValueError("Too few non-missing observations after dropna.")
        return cleaned

    elif method == "forward_fill":
        cleaned = arr.copy()
        for col in range(cleaned.shape[1]):
            last_val = 0.0
            for row in range(cleaned.shape[0]):
                if np.isnan(cleaned[row, col]) or np.isinf(cleaned[row, col]):
                    cleaned[row, col] = last_val
                else:
                    last_val = cleaned[row, col]
        return cleaned

    else:  # mean_impute
        cleaned = arr.copy()
        col_means = np.nanmean(np.where(np.isinf(cleaned), np.nan, cleaned), axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        for col in range(cleaned.shape[1]):
            mask = np.isnan(cleaned[:, col]) | np.isinf(cleaned[:, col])
            cleaned[mask, col] = col_means[col]
        return cleaned


def compute_shrunk_covariance(returns: np.ndarray) -> np.ndarray:
    """Apply Ledoit-Wolf shrinkage estimator (legacy alias)."""
    return estimate_covariance(returns, method="ledoit_wolf")


def estimate_covariance(
    returns: np.ndarray,
    method: Literal["ledoit_wolf", "sample", "ewma", "semi_covariance"] = "ledoit_wolf",
    decay: float = 0.94,
) -> np.ndarray:
    """Compute covariance matrix using specified estimation technique."""
    if method == "ledoit_wolf":
        lw = LedoitWolf().fit(returns)
        cov = lw.covariance_
    elif method == "sample":
        cov = np.cov(returns, rowvar=False)
    elif method == "ewma":
        t, n = returns.shape
        weights = np.array([(1.0 - decay) * (decay ** (t - 1 - i)) for i in range(t)])
        weights /= np.sum(weights)
        mean_r = returns.T @ weights
        centered = returns - mean_r
        cov = (centered.T * weights) @ centered
    elif method == "semi_covariance":
        # Downside covariance (only negative returns relative to mean)
        means = np.mean(returns, axis=0)
        downside = np.minimum(returns - means, 0.0)
        cov = (downside.T @ downside) / returns.shape[0]

    # Protect positive-definiteness
    cov = (cov + cov.T) * 0.5
    min_eig = np.min(np.linalg.eigvalsh(cov))
    if min_eig < 1e-8:
        cov += np.eye(cov.shape[0]) * (abs(min_eig) + 1e-6)

    return cov


def correlation_to_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to proper metric distance matrix d_ij = sqrt(0.5 * (1 - rho_ij))."""
    corr_clipped = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr_clipped)))
    np.fill_diagonal(dist, 0.0)
    return dist


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def quasi_diagonalize(link: np.ndarray) -> list[int]:
    """Reorder assets by traversing the dendrogram (seriation)."""
    n = link.shape[0] + 1

    def _recurse(node_id: int) -> list[int]:
        if node_id < n:
            return [int(node_id)]
        row = int(node_id - n)
        left = int(link[row, 0])
        right = int(link[row, 1])
        return _recurse(left) + _recurse(right)

    root = 2 * n - 2
    return _recurse(root)


def _cluster_variance(cov_sub: np.ndarray) -> float:
    """Inverse-variance weighted portfolio variance for a cluster."""
    diag = np.diag(cov_sub)
    ivp = 1.0 / np.maximum(diag, 1e-8)
    ivp /= np.sum(ivp)
    return float(ivp @ cov_sub @ ivp)


def recursive_bisection(cov: np.ndarray, sorted_indices: list[int]) -> np.ndarray:
    """Assign weights to sorted assets using recursive bisection allocation."""
    w = np.ones(len(sorted_indices))
    clusters = [sorted_indices]

    while len(clusters) > 0:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            c_left = cluster[:mid]
            c_right = cluster[mid:]

            cov_left = cov[np.ix_(c_left, c_left)]
            cov_right = cov[np.ix_(c_right, c_right)]

            v_left = _cluster_variance(cov_left)
            v_right = _cluster_variance(cov_right)

            alpha = 1.0 - v_left / (v_left + v_right)

            for i in c_left:
                w[sorted_indices.index(i)] *= alpha
            for i in c_right:
                w[sorted_indices.index(i)] *= 1.0 - alpha

            new_clusters.append(c_left)
            new_clusters.append(c_right)

        clusters = new_clusters

    return w


def apply_weight_constraints(
    weights: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    max_iter: int = 50,
) -> np.ndarray:
    """Apply box constraints [min_weight, max_weight] iteratively while preserving sum(w)=1."""
    w = weights.copy()
    n = len(w)

    if min_weight * n > 1.0 or max_weight * n < 1.0:
        return w / np.sum(w)

    for _ in range(max_iter):
        clipped = np.clip(w, min_weight, max_weight)
        excess = 1.0 - np.sum(clipped)
        if abs(excess) < 1e-7:
            return clipped

        free_mask = (w > min_weight + 1e-7) & (w < max_weight - 1e-7)
        if not np.any(free_mask):
            return clipped / np.sum(clipped)

        clipped[free_mask] += excess / np.sum(free_mask)
        w = clipped

    return w / np.sum(w)


def hrp_optimize(
    returns: np.ndarray,
    asset_names: Optional[list[str]] = None,
    cov_estimator: Literal["ledoit_wolf", "sample", "ewma", "semi_covariance"] = "ledoit_wolf",
    missing_method: Literal["dropna", "forward_fill", "mean_impute"] = "dropna",
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    prev_weights: Optional[dict[str, float]] = None,
    tx_cost_bps: float = 0.0,
) -> dict:
    """Full HRP optimization pipeline with covariance estimation, constraints, and turnover."""
    arr = np.asarray(returns, dtype=float)
    if len(arr) < 5:
        raise ValueError(f"Need at least 5 return observations for HRP, got {len(arr)}")

    clean_r = clean_missing_data(arr, method=missing_method)
    cov = estimate_covariance(clean_r, method=cov_estimator)
    corr = _cov_to_corr(cov)
    dist = correlation_to_distance(corr)

    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method="single")
    sorted_idx = quasi_diagonalize(link)
    raw_weights = recursive_bisection(cov, sorted_idx)

    # Map back to original asset order
    orig_weights = np.empty_like(raw_weights)
    for sort_pos, orig_pos in enumerate(sorted_idx):
        orig_weights[orig_pos] = raw_weights[sort_pos]

    # Apply box constraints
    final_weights = apply_weight_constraints(orig_weights, min_weight, max_weight)

    n_assets = len(final_weights)
    names = asset_names or [f"Asset_{i}" for i in range(n_assets)]
    weight_dict = {name: float(w) for name, w in zip(names, final_weights)}

    port_var = float(final_weights @ cov @ final_weights)
    mean_ret = float(np.mean(clean_r @ final_weights) * 252.0)
    port_vol = float(math.sqrt(port_var * 252.0))
    sharpe = mean_ret / port_vol if port_vol > 1e-8 else 0.0

    # Turnover & transaction cost calculation
    turnover = None
    tx_cost = 0.0
    if prev_weights is not None:
        pw = np.array([prev_weights.get(name, 0.0) for name in names])
        turnover = float(np.sum(np.abs(final_weights - pw)))
        tx_cost = float(turnover * (tx_cost_bps / 10000.0))

    return {
        "weights": weight_dict,
        "portfolio_variance": port_var,
        "portfolio_volatility": port_vol,
        "sharpe_ratio": sharpe,
        "annualized_return": mean_ret,
        "turnover": turnover,
        "transaction_cost": tx_cost,
        "covariance_estimator": cov_estimator,
        "asset_order": [names[i] for i in sorted_idx],
    }


def hrp_walk_forward_backtest(
    returns: np.ndarray,
    train_window: int = 120,
    rebalance_freq: int = 20,
    asset_names: Optional[list[str]] = None,
    cov_estimator: Literal["ledoit_wolf", "sample", "ewma"] = "ledoit_wolf",
) -> dict:
    """Out-of-sample rolling backtest of HRP."""
    t_total, n_assets = returns.shape
    if t_total <= train_window + rebalance_freq:
        raise ValueError("Insufficient observations for walk-forward backtest.")

    portfolio_returns = []
    dates = []
    current_weights = np.full(n_assets, 1.0 / n_assets)

    for t in range(train_window, t_total):
        # Rebalance check
        if (t - train_window) % rebalance_freq == 0:
            window_r = returns[t - train_window : t]
            hrp_res = hrp_optimize(window_r, asset_names=asset_names, cov_estimator=cov_estimator)
            current_weights = np.array([hrp_res["weights"][name] for name in (asset_names or [f"Asset_{i+1}" for i in range(n_assets)])])

        day_ret = float(np.sum(returns[t] * current_weights))
        portfolio_returns.append(day_ret)

    port_r_arr = np.array(portfolio_returns)
    from app.engines.portfolio_optimization import compute_portfolio_metrics
    metrics = compute_portfolio_metrics(port_r_arr)

    return {
        "out_of_sample_returns": portfolio_returns,
        "metrics": metrics,
        "n_periods": len(portfolio_returns),
    }
