"""Hierarchical Risk Parity (HRP) portfolio optimisation engine.

Implements the López de Prado HRP algorithm with Ledoit-Wolf covariance
shrinkage, single-linkage hierarchical clustering, quasi-diagonalisation
(seriation), and recursive bisection weight allocation.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from typing import Optional


def compute_shrunk_covariance(returns: np.ndarray) -> np.ndarray:
    """Apply Ledoit-Wolf shrinkage estimator.

    Parameters
    ----------
    returns : ndarray, shape (T, N)
        T observations for N assets.

    Returns
    -------
    ndarray, shape (N, N)
        Shrunk covariance matrix (guaranteed positive-definite).
    """
    lw = LedoitWolf().fit(returns)
    return lw.covariance_


def correlation_to_distance(corr: np.ndarray) -> np.ndarray:
    """Convert a correlation matrix to a distance matrix.

    d_ij = sqrt(2 * (1 − ρ_ij))

    Parameters
    ----------
    corr : ndarray, shape (N, N)

    Returns
    -------
    ndarray, shape (N, N)
    """
    # Clip to handle floating-point noise outside [-1, 1]
    corr_clipped = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(2.0 * (1.0 - corr_clipped))
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
    """Reorder assets by traversing the dendrogram (seriation).

    Parameters
    ----------
    link : ndarray
        Linkage matrix from ``scipy.cluster.hierarchy.linkage``.

    Returns
    -------
    list[int]
        Sorted leaf indices.
    """
    n = link.shape[0] + 1  # number of original observations

    def _recurse(node_id: int) -> list[int]:
        if node_id < n:
            return [int(node_id)]
        row = int(node_id - n)
        left = int(link[row, 0])
        right = int(link[row, 1])
        return _recurse(left) + _recurse(right)

    root = 2 * n - 2  # root node index
    return _recurse(root)


def _cluster_variance(cov_sub: np.ndarray) -> float:
    """Inverse-variance weighted portfolio variance for a cluster.

    w_i ∝ 1 / σ²_i, then cluster_var = w' Σ w.
    """
    ivp = 1.0 / np.diag(cov_sub)
    ivp /= ivp.sum()
    return float(ivp @ cov_sub @ ivp)


def recursive_bisection(cov: np.ndarray, sorted_indices: list[int]) -> np.ndarray:
    """Allocate weights via top-down recursive bisection.

    At each split the weight budget is divided inversely proportional
    to the cluster variance of each half.

    Parameters
    ----------
    cov : ndarray, shape (N, N)
        Full covariance matrix (original indexing).
    sorted_indices : list[int]
        Quasi-diagonalised ordering of asset indices.

    Returns
    -------
    ndarray, shape (N,)
        Weight vector (sums to 1).
    """
    n = len(sorted_indices)
    weights = np.ones(n, dtype=float)

    # Work list of clusters represented as index-ranges into sorted_indices
    clusters: list[tuple[int, int]] = [(0, n)]

    while clusters:
        new_clusters: list[tuple[int, int]] = []
        for start, end in clusters:
            if end - start <= 1:
                continue
            mid = (start + end) // 2

            left_ids = sorted_indices[start:mid]
            right_ids = sorted_indices[mid:end]

            # Sub-covariance matrices in original asset space
            cov_left = cov[np.ix_(left_ids, left_ids)]
            cov_right = cov[np.ix_(right_ids, right_ids)]

            var_left = _cluster_variance(cov_left)
            var_right = _cluster_variance(cov_right)

            total_var = var_left + var_right
            if total_var == 0:
                alpha = 0.5
            else:
                alpha = 1.0 - var_left / total_var  # left gets MORE if its var is LESS

            weights[start:mid] *= alpha
            weights[mid:end] *= 1.0 - alpha

            new_clusters.append((start, mid))
            new_clusters.append((mid, end))

        clusters = new_clusters

    # Map back to original asset ordering
    result = np.zeros(cov.shape[0], dtype=float)
    for i, idx in enumerate(sorted_indices):
        result[idx] = weights[i]

    return result


def hrp_optimize(
    returns: np.ndarray,
    asset_names: Optional[list[str]] = None,
) -> dict:
    """Full Hierarchical Risk Parity pipeline.

    Steps
    -----
    1. Ledoit-Wolf shrunk covariance estimation.
    2. Correlation → distance matrix.
    3. Single-linkage hierarchical clustering.
    4. Quasi-diagonalisation (seriation).
    5. Recursive bisection → weight vector.

    Parameters
    ----------
    returns : ndarray, shape (T, N)
        Historical return matrix. T ≥ 2, N ≥ 2.
    asset_names : list[str] | None
        Optional human-readable names for N assets.

    Returns
    -------
    dict
        weights : dict[str, float]
        portfolio_variance : float
        sharpe_ratio : float  (annualised, assuming rf = 0)
    """
    T, N = returns.shape
    if N < 2:
        raise ValueError("Need at least 2 assets.")
    if T < 2:
        raise ValueError("Need at least 2 time periods.")

    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(N)]

    # 1. Covariance
    cov = compute_shrunk_covariance(returns)

    # 2. Distance
    corr = _cov_to_corr(cov)
    dist = correlation_to_distance(corr)

    # 3. Clustering  — condensed distance form required by linkage
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")

    # 4. Quasi-diagonalisation
    sorted_idx = quasi_diagonalize(link)

    # 5. Recursive bisection
    w = recursive_bisection(cov, sorted_idx)

    # Portfolio statistics
    portfolio_variance = float(w @ cov @ w)
    daily_returns = returns @ w  # (T,) vector of portfolio daily returns
    mean_daily_return = float(np.mean(daily_returns))
    annualised_return = mean_daily_return * 252
    annualised_vol = np.sqrt(portfolio_variance * 252)

    if annualised_vol > 0:
        sharpe_ratio = annualised_return / annualised_vol
    else:
        sharpe_ratio = 0.0

    weights_dict = {name: float(w[i]) for i, name in enumerate(asset_names)}

    return {
        "weights": weights_dict,
        "portfolio_variance": portfolio_variance,
        "sharpe_ratio": float(sharpe_ratio),
    }
