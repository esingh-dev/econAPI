"""Executive Benchmark Verification Runner for econAPI.

Directly queries the econAPI endpoints via FastAPI TestClient and compares
outputs against verified literature benchmarks:
1. John C. Hull (2018) - Options, Futures, and Other Derivatives (BSM & Greeks)
2. Albrecher, Mayer, Schoutens, & Tistaert (2007) - Heston Model Table 1
3. Marcos López de Prado (2016) - Hierarchical Risk Parity Cluster Allocation
4. McNeil, Frey & Embrechts (2005) - Quantitative Risk Management EVT Closed Form
5. Nelson & Siegel (1987) - Canonical Term Structure Model Calibration
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.engines.black_scholes import bsm_full
from app.engines.heston import heston_full
from app.engines.hrp import hrp_optimize
from app.engines.evt import compute_var, compute_cvar
from app.engines.macro_models import nelson_siegel, fit_nelson_siegel

client = TestClient(app)


def print_header(title: str):
    print(f"\n{'='*80}\n {title}\n{'='*80}")


def format_row(metric: str, target: float, actual: float, tolerance: float) -> str:
    diff = abs(target - actual)
    rel_err = (diff / abs(target) * 100.0) if target != 0 else diff
    status = "PASS [OK]" if diff <= tolerance else "FAIL [X]"
    return f"{metric:<26} | Target: {target:>10.5f} | API: {actual:>10.5f} | Err: {diff:>9.6f} ({rel_err:>6.3f}%) | {status}"


def run_all_benchmarks():
    all_passed = True

    # ─────────────────────────────────────────────────────────────────
    # 1. BSM & GREEKS: JOHN C. HULL (2018)
    # ─────────────────────────────────────────────────────────────────
    print_header("1. BLACK-SCHOLES-MERTON & GREEKS — JOHN C. HULL BENCHMARK")
    print("Reference: Options, Futures, and Other Derivatives (S=42, K=40, T=0.5, r=0.10, sigma=0.20)")
    
    res_call = client.post("/v1/derivatives/bsm", json={
        "S": 42.0, "K": 40.0, "T": 0.5, "r": 0.10, "sigma": 0.20, "option_type": "call"
    }).json()

    res_put = client.post("/v1/derivatives/bsm", json={
        "S": 42.0, "K": 40.0, "T": 0.5, "r": 0.10, "sigma": 0.20, "option_type": "put"
    }).json()

    bsm_tests = [
        ("Call Price", 4.75942, res_call["price"], 1e-4),
        ("Put Price", 0.80860, res_put["price"], 1e-4),
        ("Delta (Call)", 0.77914, res_call["delta"], 1e-3),
        ("Delta (Put)", -0.22086, res_put["delta"], 1e-3),
        ("Gamma", 0.04996, res_call["gamma"], 1e-4),
        ("Vega", 8.81342, res_call["vega"], 1e-3),
        ("Theta (Call / yr)", -4.55909, res_call["theta"], 1e-3),
        ("Rho (Call)", 13.98205, res_call["rho"], 1e-3),
        ("Rho (Put)", -5.04254, res_put["rho"], 1e-3),
    ]

    for metric, target, actual, tol in bsm_tests:
        row = format_row(metric, target, actual, tol)
        print(row)
        if abs(target - actual) > tol:
            all_passed = False

    # ─────────────────────────────────────────────────────────────────
    # 2. HESTON MODEL: ALBRECHER ET AL. (2007) / HESTON (1993)
    # ─────────────────────────────────────────────────────────────────
    print_header("2. HESTON STOCHASTIC VOLATILITY — ALBRECHER ET AL. (2007) BENCHMARK")
    print("Reference: S=100, T=1.0, r=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7")

    res_atm = client.post("/v1/derivatives/heston", json={
        "S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "v0": 0.04,
        "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.7, "option_type": "call"
    }).json()

    res_otm = client.post("/v1/derivatives/heston", json={
        "S": 100.0, "K": 110.0, "T": 1.0, "r": 0.05, "v0": 0.04,
        "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.7, "option_type": "call"
    }).json()

    res_itm = client.post("/v1/derivatives/heston", json={
        "S": 100.0, "K": 90.0, "T": 1.0, "r": 0.05, "v0": 0.04,
        "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.7, "option_type": "call"
    }).json()

    heston_tests = [
        ("ATM Call (K=100)", 10.15463, res_atm["price"], 0.01),
        ("OTM Call (K=110)", 4.86618, res_otm["price"], 0.01),
        ("ITM Call (K=90)", 17.14006, res_itm["price"], 0.01),
        ("ATM Implied Vol", 0.19047, res_atm["implied_volatility"], 0.01),
    ]

    for metric, target, actual, tol in heston_tests:
        row = format_row(metric, target, actual, tol)
        print(row)
        if abs(target - actual) > tol:
            all_passed = False

    # ─────────────────────────────────────────────────────────────────
    # 3. HRP: MARCOS LÓPEZ DE PRADO (2016)
    # ─────────────────────────────────────────────────────────────────
    print_header("3. HIERARCHICAL RISK PARITY — LÓPEZ DE PRADO (2016) BENCHMARK")
    print("Reference: Two independent assets with variances sigma_A^2=0.01 and sigma_B^2=0.04")

    rng = np.random.default_rng(12345)
    ret_A = rng.normal(0.0, 0.10, size=10000)
    ret_B = rng.normal(0.0, 0.20, size=10000)
    returns_2asset = np.column_stack([ret_A, ret_B]).tolist()

    res_hrp = client.post("/v1/portfolio/hrp", json={
        "returns": returns_2asset,
        "asset_names": ["AssetA", "AssetB"]
    }).json()

    hrp_tests = [
        ("Weight AssetA (Target 80%)", 0.80000, res_hrp["weights"]["AssetA"], 0.02),
        ("Weight AssetB (Target 20%)", 0.20000, res_hrp["weights"]["AssetB"], 0.02),
        ("Weight Sum (Budget 100%)", 1.00000, sum(res_hrp["weights"].values()), 1e-6),
    ]

    for metric, target, actual, tol in hrp_tests:
        row = format_row(metric, target, actual, tol)
        print(row)
        if abs(target - actual) > tol:
            all_passed = False

    # ─────────────────────────────────────────────────────────────────
    # 4. EVT: MCNEIL, FREY & EMBRECHTS (2005) CLOSED FORM
    # ─────────────────────────────────────────────────────────────────
    print_header("4. EXTREME VALUE THEORY — MCNEIL ET AL. (2005) CLOSED FORM")
    print("Reference: Exact GPD analytical quantile at alpha=0.99, u=0.03, pu=0.05, xi=0.25, beta=0.02")

    gpd_bench = {"xi": 0.25, "beta": 0.02, "threshold": 0.03, "n_exceedances": 50, "n_total": 1000}
    calc_var = compute_var(gpd_bench, confidence_level=0.99)
    calc_cvar = compute_cvar(calc_var, gpd_bench)

    evt_tests = [
        ("VaR (alpha=0.99)", 0.069628, calc_var, 1e-5),
        ("CVaR (Expected Shortfall)", 0.109504, calc_cvar, 1e-5),
    ]

    for metric, target, actual, tol in evt_tests:
        row = format_row(metric, target, actual, tol)
        print(row)
        if abs(target - actual) > tol:
            all_passed = False

    # ─────────────────────────────────────────────────────────────────
    # 5. NELSON-SIEGEL: CANONICAL CALIBRATION (1987)
    # ─────────────────────────────────────────────────────────────────
    print_header("5. NELSON-SIEGEL YIELD CURVE — CANONICAL MODEL (1987)")
    print("Reference: beta0=8.40, beta1=-2.10, beta2=1.60, lambda=2.50 across 10 tenors")

    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    canonical_yields = nelson_siegel(np.array(tenors), 8.40, -2.10, 1.60, 2.50).tolist()

    res_ns = client.post("/v1/macro/nelson-siegel", json={
        "tenors": tenors,
        "yields": canonical_yields
    }).json()

    ns_tests = [
        ("Recovered beta_0 (Level)", 8.40000, res_ns["beta_0"], 0.05),
        ("Recovered beta_1 (Slope)", -2.10000, res_ns["beta_1"], 0.05),
        ("Recovered beta_2 (Curv)", 1.60000, res_ns["beta_2"], 0.05),
        ("Recovered lambda (Decay)", 2.50000, res_ns["lambda_param"], 0.05),
        ("Fitting RMSE (Ground Truth)", 0.00000, res_ns["rmse"], 1e-3),
    ]

    for metric, target, actual, tol in ns_tests:
        row = format_row(metric, target, actual, tol)
        print(row)
        if abs(target - actual) > tol:
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print(" ALL ESTABLISHED BENCHMARKS VERIFIED SUCCESSFULLY WITH 100% PASS RATE")
    else:
        print(" SOME BENCHMARKS FAILED TO MEET CONVERGENCE CRITERIA")
    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_all_benchmarks())
