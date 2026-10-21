"""Institutional Walk-Forward Portfolio Backtesting Engine with Slippage and Turnover."""

from __future__ import annotations

import numpy as np
from app.engines.hrp import hrp_optimize


def run_walk_forward_backtest(
    returns_matrix: list[list[float]],
    asset_names: list[str],
    strategy: str = "hrp",
    train_window: int = 60,
    rebalance_freq: int = 20,
    tx_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> dict:
    """Walk-forward backtest simulating out-of-sample portfolio execution.

    Parameters
    ----------
    returns_matrix : list[list[float]]
        T x N historical returns.
    asset_names : list[str]
        Names of the N assets.
    strategy : str
        "hrp" | "equal_weight" | "inverse_vol"
    train_window : int
        Number of past trading periods used for calibration (e.g. 60 days).
    rebalance_freq : int
        Periods between rebalancing (e.g. 20 days for monthly).
    tx_cost_bps : float
        Transaction fee in basis points.
    slippage_bps : float
        Slippage penalty in basis points per unit turnover.

    Returns
    -------
    dict with out-of-sample returns, equity curve, drawdown, turnover, and performance ratios.
    """
    R = np.asarray(returns_matrix, dtype=float)
    T, N = R.shape

    if T < train_window + rebalance_freq:
        raise ValueError(f"Total periods ({T}) must be >= train_window ({train_window}) + rebalance_freq ({rebalance_freq}).")

    oos_returns = []
    oos_dates_idx = []
    equity_curve = [1.0]
    turnovers = []
    weights_history = []

    current_weights = np.ones(N) / N
    total_cost_factor = (tx_cost_bps + slippage_bps) / 10000.0

    step = 0
    t = train_window

    while t < T:
        # 1. Point-in-time calibration window (strictly past data)
        train_slice = R[t - train_window : t, :]

        # 2. Strategy allocation
        if strategy == "hrp":
            try:
                hrp_out = hrp_optimize(train_slice.tolist(), asset_names, cov_estimator="ledoit_wolf")
                target_weights = np.array([hrp_out["weights"][name] for name in asset_names])
            except Exception:
                target_weights = np.ones(N) / N
        elif strategy == "inverse_vol":
            vols = np.std(train_slice, axis=0)
            inv_vols = np.where(vols > 1e-8, 1.0 / vols, 1.0)
            target_weights = inv_vols / np.sum(inv_vols)
        else:  # equal_weight
            target_weights = np.ones(N) / N

        # 3. Turnover and transaction friction
        turnover = float(np.sum(np.abs(target_weights - current_weights)))
        cost = turnover * total_cost_factor
        turnovers.append(turnover)
        weights_history.append({name: round(float(w), 4) for name, w in zip(asset_names, target_weights)})

        current_weights = target_weights

        # 4. Out-of-sample forward evaluation block
        rebal_horizon = min(rebalance_freq, T - t)
        for h in range(rebal_horizon):
            period_ret = float(np.dot(current_weights, R[t + h, :]))
            if h == 0:
                period_ret -= cost  # deduct friction at rebalance event
            oos_returns.append(period_ret)
            equity_curve.append(equity_curve[-1] * (1.0 + period_ret))
            oos_dates_idx.append(t + h)

            # Drift weights with returns
            asset_growths = 1.0 + R[t + h, :]
            drifted = current_weights * asset_growths
            current_weights = drifted / np.sum(drifted)

        t += rebalance_freq

    oos_arr = np.array(oos_returns)
    eq_arr = np.array(equity_curve)

    # Compute Drawdown Series
    running_max = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))

    # Metrics
    mean_daily = float(np.mean(oos_arr))
    std_daily = float(np.std(oos_arr))
    ann_return = float((equity_curve[-1] ** (252.0 / len(oos_arr))) - 1.0) if len(oos_arr) > 0 and equity_curve[-1] > 0 else 0.0
    ann_vol = float(std_daily * np.sqrt(252))
    sharpe = float(ann_return / ann_vol) if ann_vol > 1e-6 else 0.0

    downside = oos_arr[oos_arr < 0]
    downside_vol = float(np.std(downside) * np.sqrt(252)) if len(downside) > 1 else ann_vol
    sortino = float(ann_return / downside_vol) if downside_vol > 1e-6 else 0.0
    calmar = float(abs(ann_return / max_drawdown)) if abs(max_drawdown) > 1e-6 else 0.0

    return {
        "strategy": strategy,
        "n_periods": len(oos_returns),
        "cumulative_return": float(equity_curve[-1] - 1.0),
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_drawdown,
        "mean_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "equity_curve": [round(x, 5) for x in equity_curve],
        "drawdowns": [round(x, 5) for x in drawdowns.tolist()],
        "weights_snapshots": weights_history,
    }
