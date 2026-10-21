"""Binomial and Trinomial tree pricing engine for European and American options.

Implements:
- Cox-Ross-Rubinstein (CRR) binomial tree with continuous dividend yield q
- Jarrow-Rudd (JR) equal-probability binomial tree
- Trinomial lattice (Kamrad-Ritchken) for enhanced convergence
- Early exercise premium computation for American options
- Lattice-derived Greeks (Delta, Gamma, Theta)
"""

from __future__ import annotations

import math
from typing import Literal, Optional
import numpy as np


def binomial_tree_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: int = 200,
    option_type: Literal["call", "put"] = "call",
    style: Literal["european", "american"] = "american",
    tree_type: Literal["crr", "jarrow_rudd"] = "crr",
    q: float = 0.0,
) -> dict:
    """Price European or American options via Binomial Tree.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard option inputs.
    steps : int
        Number of time steps in the tree (default 200).
    option_type : {"call", "put"}
    style : {"european", "american"}
    tree_type : {"crr", "jarrow_rudd"}
    q : float
        Continuous dividend yield.

    Returns
    -------
    dict
        price, early_exercise_premium, delta, gamma, theta.
    """
    if steps < 5 or steps > 2000:
        raise ValueError("Tree steps must be between 5 and 2000.")

    dt = T / steps
    disc = math.exp(-r * dt)

    if tree_type == "crr":
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        p = (math.exp((r - q) * dt) - d) / (u - d)
    else:  # Jarrow-Rudd
        drift = (r - q - 0.5 * sigma * sigma) * dt
        vol_step = sigma * math.sqrt(dt)
        u = math.exp(drift + vol_step)
        d = math.exp(drift - vol_step)
        p = 0.5

    if p < 0.0 or p > 1.0:
        # Arb warning/fallback
        p = max(0.0, min(1.0, p))

    q_prob = 1.0 - p

    # Asset prices at maturity (step N)
    # S_j = S * u^j * d^(N - j) for j in 0..N
    asset_prices = np.array([S * (u**j) * (d ** (steps - j)) for j in range(steps + 1)])

    if option_type == "call":
        values = np.maximum(asset_prices - K, 0.0)
    else:
        values = np.maximum(K - asset_prices, 0.0)

    values_eur = values.copy()

    # Store values at step 1 and step 2 for Greek estimation
    val_step2 = None
    val_step1 = None

    # Backward induction
    for i in range(steps - 1, -1, -1):
        # Discounted continuation value
        values_eur = disc * (p * values_eur[1 : i + 2] + q_prob * values_eur[0 : i + 1])
        cont_val = disc * (p * values[1 : i + 2] + q_prob * values[0 : i + 1])

        current_spots = np.array([S * (u**j) * (d ** (i - j)) for j in range(i + 1)])
        if option_type == "call":
            exercise_val = np.maximum(current_spots - K, 0.0)
        else:
            exercise_val = np.maximum(K - current_spots, 0.0)

        if style == "american":
            values = np.maximum(cont_val, exercise_val)
        else:
            values = cont_val

        if i == 2:
            val_step2 = values.copy()
        elif i == 1:
            val_step1 = values.copy()

    price = float(values[0])
    eur_price = float(values_eur[0])
    early_exercise_premium = max(0.0, price - eur_price) if style == "american" else 0.0

    # Greeks from lattice nodes
    # Step 1: S_u = S*u, S_d = S*d
    delta = float((val_step1[1] - val_step1[0]) / (S * u - S * d))

    # Gamma from Step 2:
    # Nodes: S_uu = S*u^2, S_ud = S*u*d, S_dd = S*d^2
    h1 = S * u * d - S * d * d
    h2 = S * u * u - S * u * d
    d_up = (val_step2[2] - val_step2[1]) / h2
    d_down = (val_step2[1] - val_step2[0]) / h1
    gamma = float((d_up - d_down) / (0.5 * (S * u * u - S * d * d)))

    # Theta from center node at step 2 vs step 0:
    theta = float((val_step2[1] - price) / (2.0 * dt))

    return {
        "price": price,
        "european_price": eur_price,
        "early_exercise_premium": early_exercise_premium,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "tree_type": tree_type,
        "steps": steps,
        "style": style,
    }


def trinomial_tree_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: int = 150,
    option_type: Literal["call", "put"] = "call",
    style: Literal["european", "american"] = "american",
    q: float = 0.0,
) -> dict:
    """Price European or American options via Trinomial Tree (Kamrad-Ritchken formulation)."""
    if steps < 5 or steps > 1500:
        raise ValueError("Trinomial steps must be between 5 and 1500.")

    dt = T / steps
    disc = math.exp(-r * dt)
    dx = sigma * math.sqrt(3.0 * dt)
    u = math.exp(dx)
    d = 1.0 / u

    # Risk-neutral transition probabilities
    drift = r - q - 0.5 * sigma * sigma
    nu = drift * dt / dx
    p_u = 1.0 / 6.0 + 0.5 * (nu**2 + nu)
    p_m = 2.0 / 3.0 - (nu**2)
    p_d = 1.0 / 6.0 + 0.5 * (nu**2 - nu)

    # Protect probabilities
    p_u = max(0.0, min(1.0, p_u))
    p_m = max(0.0, min(1.0, p_m))
    p_d = max(0.0, min(1.0, p_d))

    # At maturity: 2 * steps + 1 nodes
    # Node j in -steps..steps: spot = S * exp(j * dx)
    spots = np.array([S * math.exp(j * dx) for j in range(-steps, steps + 1)])

    if option_type == "call":
        values = np.maximum(spots - K, 0.0)
    else:
        values = np.maximum(K - spots, 0.0)

    values_eur = values.copy()

    for i in range(steps - 1, -1, -1):
        # Trinomial expectation: p_d * V_{j-1} + p_m * V_j + p_u * V_{j+1}
        num_nodes = 2 * i + 1
        cont_val = disc * (
            p_d * values[0:num_nodes]
            + p_m * values[1 : num_nodes + 1]
            + p_u * values[2 : num_nodes + 2]
        )
        cont_eur = disc * (
            p_d * values_eur[0:num_nodes]
            + p_m * values_eur[1 : num_nodes + 1]
            + p_u * values_eur[2 : num_nodes + 2]
        )
        values_eur = cont_eur

        node_spots = np.array([S * math.exp(j * dx) for j in range(-i, i + 1)])
        if option_type == "call":
            exercise = np.maximum(node_spots - K, 0.0)
        else:
            exercise = np.maximum(K - node_spots, 0.0)

        if style == "american":
            values = np.maximum(cont_val, exercise)
        else:
            values = cont_val

    price = float(values[0])
    eur_price = float(values_eur[0])

    return {
        "price": price,
        "european_price": eur_price,
        "early_exercise_premium": max(0.0, price - eur_price) if style == "american" else 0.0,
        "tree_type": "trinomial",
        "steps": steps,
        "style": style,
    }
