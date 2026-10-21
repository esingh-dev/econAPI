"""Monte Carlo derivatives pricing engine with variance reduction.

Implements:
- Geometric Brownian Motion path simulator
- Antithetic variates variance reduction
- Control variate variance reduction using analytical BSM as covariate
- Standard error and 95% asymptotic confidence interval
- Support for European vanilla, Asian, and Barrier payoffs
- Deterministic seed control for reproducibility
"""

from __future__ import annotations

import math
from typing import Literal, Optional
import numpy as np


def monte_carlo_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    payoff_style: Literal["european", "asian_arithmetic", "barrier_up_out", "barrier_down_out"] = "european",
    barrier_level: Optional[float] = None,
    q: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 100,
    use_antithetic: bool = True,
    use_control_variate: bool = True,
    seed: Optional[int] = 42,
) -> dict:
    """Monte Carlo pricing with variance reduction techniques.

    Parameters
    ----------
    S, K, T, r, sigma, q : standard market parameters
    option_type : {"call", "put"}
    payoff_style : {"european", "asian_arithmetic", "barrier_up_out", "barrier_down_out"}
    barrier_level : float, optional
    n_paths : int
        Number of simulated paths (default: 50,000).
    n_steps : int
        Discretization steps per path.
    use_antithetic : bool
    use_control_variate : bool
    seed : int, optional
    """
    if n_paths < 1000 or n_paths > 500_000:
        raise ValueError("n_paths must be between 1,000 and 500,000.")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    disc = math.exp(-r * T)
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol_step = sigma * math.sqrt(dt)

    num_sim_paths = n_paths // 2 if use_antithetic else n_paths

    # Standard normal increments: shape (num_sim_paths, n_steps)
    z = rng.standard_normal((num_sim_paths, n_steps))
    if use_antithetic:
        z = np.vstack([z, -z])

    actual_paths = z.shape[0]

    # Generate log paths
    log_returns = drift + vol_step * z
    log_paths = np.zeros((actual_paths, n_steps + 1))
    log_paths[:, 0] = math.log(S)
    log_paths[:, 1:] = math.log(S) + np.cumsum(log_returns, axis=1)
    paths = np.exp(log_paths)

    s_T = paths[:, -1]

    # Payoff evaluation
    if payoff_style == "european":
        if option_type == "call":
            payoffs = np.maximum(s_T - K, 0.0)
        else:
            payoffs = np.maximum(K - s_T, 0.0)

    elif payoff_style == "asian_arithmetic":
        # Average spot over all time steps
        avg_s = np.mean(paths, axis=1)
        if option_type == "call":
            payoffs = np.maximum(avg_s - K, 0.0)
        else:
            payoffs = np.maximum(K - avg_s, 0.0)

    elif payoff_style == "barrier_up_out":
        H = barrier_level or S * 1.2
        max_s = np.max(paths, axis=1)
        breached = max_s >= H
        base_payoff = np.maximum(s_T - K, 0.0) if option_type == "call" else np.maximum(K - s_T, 0.0)
        payoffs = np.where(breached, 0.0, base_payoff)

    elif payoff_style == "barrier_down_out":
        H = barrier_level or S * 0.8
        min_s = np.min(paths, axis=1)
        breached = min_s <= H
        base_payoff = np.maximum(s_T - K, 0.0) if option_type == "call" else np.maximum(K - s_T, 0.0)
        payoffs = np.where(breached, 0.0, base_payoff)

    discounted_payoffs = disc * payoffs

    # Control variates (using European vanilla BSM analytical price)
    if use_control_variate and payoff_style != "european":
        from app.engines.black_scholes import bsm_price
        ana_bsm = bsm_price(S, K, T, r, sigma, option_type, q)["price"]
        eur_payoffs = disc * (np.maximum(s_T - K, 0.0) if option_type == "call" else np.maximum(K - s_T, 0.0))
        cov_matrix = np.cov(discounted_payoffs, eur_payoffs)
        var_eur = cov_matrix[1, 1]
        if var_eur > 1e-12:
            c = cov_matrix[0, 1] / var_eur
            discounted_payoffs = discounted_payoffs - c * (eur_payoffs - ana_bsm)

    price = float(np.mean(discounted_payoffs))
    std_err = float(np.std(discounted_payoffs, ddof=1) / math.sqrt(actual_paths))
    ci_95 = (float(price - 1.96 * std_err), float(price + 1.96 * std_err))

    return {
        "price": price,
        "standard_error": std_err,
        "ci_95_lower": ci_95[0],
        "ci_95_upper": ci_95[1],
        "n_paths": actual_paths,
        "n_steps": n_steps,
        "antithetic_used": use_antithetic,
        "control_variate_used": use_control_variate,
        "payoff_style": payoff_style,
    }
