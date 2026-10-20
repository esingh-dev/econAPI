# econAPI: Production Quantitative Finance & Risk Microservice

[![CI Pipeline](https://github.com/your-username/econAPI/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/econAPI/actions)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 66 Passed](https://img.shields.io/badge/tests-66%20passed-brightgreen.svg)](tests/)
[![Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7.svg)](render.yaml)
[![RapidAPI](https://img.shields.io/badge/Monetized%20on-RapidAPI-0055FF.svg)](#3-rapidapi-studio-integration--monetization)

**econAPI** is a production-grade financial engineering microservice engineered with **FastAPI**, **NumPy**, **SciPy**, and **Statsmodels**. It provides high-performance calculation engines for Black-Scholes-Merton option pricing, Heston stochastic volatility with implied volatility inversion, Hierarchical Risk Parity (HRP) portfolio optimization, Extreme Value Theory (EVT) tail-risk metrics, and Nelson-Siegel yield curve calibration.

---

## Table of Contents

1. [Local Installation & Pytest Verification](#1-local-installation--pytest-verification)
2. [Connecting GitHub to Render (Zero-Cost Hosting)](#2-connecting-github-to-render-zero-cost-hosting)
3. [RapidAPI Studio Integration & Monetization](#3-rapidapi-studio-integration--monetization)
4. [Configuring the 4 RapidAPI Pricing Tiers](#4-configuring-the-4-rapidapi-pricing-tiers)
5. [Python SDK Client Usage (`client.py`)](#5-python-sdk-client-usage-clientpy)
6. [API Endpoints Reference](#6-api-endpoints-reference)
7. [Environment Variables](#7-environment-variables)
8. [License](#8-license)

---

## 1. Local Installation & Pytest Verification

### Prerequisites
- Python 3.11 or 3.12
- Git

### Setup Steps
```bash
# 1. Clone the repository
git clone https://github.com/your-username/econAPI.git
cd econAPI

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Start local development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation is available immediately:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI JSON**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### Running Automated Test Suites
econAPI includes **66 comprehensive automated tests** verified against published literature benchmarks (John C. Hull, Albrecher et al., Marcos López de Prado, McNeil/Frey/Embrechts, and Nelson & Siegel):

```bash
# Run all 66 unit and integration tests
pytest -v

# Run the executive literature benchmark comparison report
python scripts/verify_benchmarks.py
```

---

## 2. Connecting GitHub to Render (Zero-Cost Hosting)

Render provides free-tier web service hosting with automatic SSL, zero-downtime deploys on Git push, and native support for Render Blueprints via `render.yaml`.

```
[Local Code] ─── git push ───> [GitHub Repository] ─── webhook ───> [Render Free Cloud]
```

### Step 1: Push Code to GitHub
```bash
git init
git branch -M main
git add .
git commit -m "feat: complete econAPI with Render and RapidAPI rollout config"
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/econAPI.git
git push -u origin main
```

### Step 2: Deploy on Render via Blueprint (1-Click)
1. Log into your free account at [render.com](https://render.com).
2. Click **Blueprints** in the top navigation bar.
3. Click **New Blueprint Instance** and select your GitHub `econAPI` repository.
4. Render will automatically detect [`render.yaml`](render.yaml) and configure:
   - **Service Name**: `econapi`
   - **Environment**: Python 3.11
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check**: `/health`
5. Click **Apply**. Render will build and deploy your container.
6. Once deployed, note your public URL:
   `https://econapi.onrender.com`

---

## 3. RapidAPI Studio Integration & Monetization

RapidAPI serves as your monetization gateway, handling user management, credit card billing, rate limiting, and DDoS protection for **$0 upfront cost**.

### Step 1: Create a Provider Account
1. Sign up for free at [rapidapi.com](https://rapidapi.com).
2. Go to **My APIs** $\to$ **Add New API** (or open [RapidAPI Studio](https://studio.rapidapi.com)).
3. Enter API Details:
   - **API Name**: `econAPI`
   - **Short Description**: `Institutional Quantitative Finance, Derivative Pricing & Tail Risk Microservice`
   - **Category**: `Finance`

### Step 2: Import OpenAPI Specification
1. In RapidAPI Studio, navigate to **Hub Listing** $\to$ **Definition** $\to$ **Endpoints**.
2. Click **Import** $\to$ Select **OpenAPI (Swagger)**.
3. Choose **Import from URL** and paste your live Render schema URL:
   `https://econapi.onrender.com/openapi.json`
4. RapidAPI will instantly parse all schemas, types, and endpoints:
   - `GET /health`
   - `POST /v1/derivatives/bsm`
   - `POST /v1/derivatives/heston`
   - `POST /v1/portfolio/hrp`
   - `POST /v1/risk/evt-cvar`
   - `POST /v1/macro/nelson-siegel`

### Step 3: Configure Base Target URL & Gateway Secret
1. Under **Target URLs**, enter your Render host:
   `https://econapi.onrender.com`
2. Under **Transformations / Headers**, add a forwarded secret header to authenticate requests from RapidAPI to your Render backend:
   - Header Name: `X-API-Key`
   - Header Value: Match the key specified in your `ECONAPI_VALID_API_KEYS` on Render (e.g. `rapidapi-gateway-secret-key-prod`).

---

## 4. Configuring the 4 RapidAPI Pricing Tiers

Navigate to **Plans & Pricing** in RapidAPI Studio and configure these 4 monetization tiers:

| Tier Name | Monthly Price | Monthly Request Quota | Rate Limit | Hard / Soft Limit | Features & Restrictions | Target Audience |
|:---|:---:|:---:|:---:|:---:|:---|:---|
| **Free Sandbox** | **$0.00 / mo** | **500 requests** | 1 req / sec | **Hard Limit** (Stops on cap) | • Access to `/health` and `/v1/derivatives/bsm`<br>• Single-asset Greeks | Academic researchers, students, exploratory developers |
| **Pro** | **$49.00 / mo** | **25,000 requests** | 10 req / sec | **Soft Limit** ($0.003 / extra req) | • All endpoints unlocked<br>• HRP up to 20 assets<br>• EVT Tail Risk (VaR/CVaR) | Individual quants, crypto traders, boutique wealth advisors |
| **Ultra** | **$149.00 / mo** | **150,000 requests** | 30 req / sec | **Soft Limit** ($0.0015 / extra req) | • High-priority compute queue<br>• HRP up to 100 assets<br>• Full Heston & Nelson-Siegel | Systematic prop trading desks, hedge fund analysts |
| **Enterprise** | **$499.00 / mo** | **500,000 requests** | 100 req / sec | **Soft Limit** ($0.0008 / extra req) | • Institutional throughput<br>• Unlimited asset clusters<br>• 99.9% uptime SLA<br>• Email & Discord support | FinTech platforms, neo-banks, asset management firms |

---

## 5. Python SDK Client Usage (`client.py`)

A clean, strongly-typed client library is provided in [`client.py`](client.py). Developers can use it directly or bundle it into a package for PyPI.

### Basic Initialization
```python
from client import EconAPI

# Initialize client using your RapidAPI Key
client = EconAPI(
    api_key="YOUR_RAPIDAPI_KEY",
    rapidapi_host="econapi.p.rapidapi.com"
)

# Check service status
health = client.health()
print("Service Status:", health)  # {"status": "healthy", "version": "1.0.0"}
```

### 1. Black-Scholes-Merton Pricing & Analytical Greeks
```python
bsm = client.price_bsm(
    spot=100.0,
    strike=100.0,
    time_to_maturity=1.0,
    risk_free_rate=0.05,
    volatility=0.20,
    option_type="call"
)

print(f"Call Price: ${bsm['price']:.4f}")
print(f"Delta: {bsm['delta']:.4f} | Gamma: {bsm['gamma']:.4f} | Vega: {bsm['vega']:.4f}")
```

### 2. Heston Stochastic Volatility & Implied Volatility
```python
heston = client.price_heston(
    spot=100.0,
    strike=100.0,
    time_to_maturity=1.0,
    risk_free_rate=0.05,
    v0=0.04,          # initial variance
    kappa=2.0,        # mean-reversion speed
    theta=0.04,       # long-term variance
    xi=0.30,          # vol-of-vol
    rho=-0.70,        # asset-variance correlation
    option_type="call"
)

print(f"Heston Price: ${heston['price']:.4f}")
print(f"Extracted Implied Volatility: {heston['implied_volatility']*100:.2f}%")
```

### 3. Hierarchical Risk Parity (HRP) Portfolio Allocation
```python
import numpy as np

# 200 daily returns across 3 assets
rng = np.random.default_rng(42)
returns = rng.normal(0.0005, 0.02, size=(200, 3)).tolist()

hrp = client.optimize_hrp(
    returns=returns,
    asset_names=["SPY", "TLT", "GLD"]
)

print("Optimal Portfolio Weights:", hrp["weights"])
print(f"Portfolio Annualized Sharpe Ratio: {hrp['sharpe_ratio']:.2f}")
```

### 4. Extreme Value Theory (EVT) VaR & Expected Shortfall
```python
# Historical return distribution with tail risk
losses = rng.normal(0.0005, 0.02, size=500).tolist()

evt = client.compute_evt_cvar(
    returns=losses,
    confidence_level=0.99,
    threshold_quantile=0.95
)

print(f"GPD Shape (xi): {evt['gpd_params']['xi']:.4f}")
print(f"99% Value-at-Risk (VaR): {evt['var']*100:.2f}%")
print(f"99% Expected Shortfall (CVaR): {evt['cvar']*100:.2f}%")
```

### 5. Nelson-Siegel Yield Curve Calibration
```python
tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
yields = [5.25, 5.10, 4.85, 4.50, 4.35, 4.20, 4.15, 4.10, 4.30, 4.40]

curve = client.fit_yield_curve(tenors=tenors, yields=yields)

print(f"Level (beta0): {curve['beta_0']:.4f}")
print(f"Slope (beta1): {curve['beta_1']:.4f}")
print(f"Curvature (beta2): {curve['beta_2']:.4f}")
print(f"Decay (lambda): {curve['lambda_param']:.4f}")
print(f"Calibration RMSE: {curve['rmse']:.6f}%")
```

---

## 6. API Endpoints Reference

All endpoints expect JSON payloads and return JSON responses.

### `POST /v1/derivatives/bsm`
- **Request Body**:
  - `S` (float, > 0): Spot price.
  - `K` (float, > 0): Strike price.
  - `T` (float, > 0): Expiration in years.
  - `r` (float): Continuously compounded risk-free rate.
  - `sigma` (float, > 0): Annual volatility.
  - `option_type` (string): `"call"` or `"put"`.
- **Response**: `price`, `delta`, `gamma`, `vega`, `theta`, `rho`.

### `POST /v1/derivatives/heston`
- **Request Body**: `S`, `K`, `T`, `r`, `v0`, `kappa`, `theta`, `xi`, `rho` $\in [-1, 1]$, `option_type`.
- **Response**: `price`, `implied_volatility`.

### `POST /v1/portfolio/hrp`
- **Request Body**:
  - `returns` (list of lists of floats): Matrix of historical returns $(T \ge 2, N \ge 2)$.
  - `asset_names` (list of str, optional): Identifiers for each asset.
- **Response**: `weights` (dict), `portfolio_variance`, `sharpe_ratio`.

### `POST /v1/risk/evt-cvar`
- **Request Body**:
  - `returns` (list of floats, $\ge 50$ observations): Historical asset return series.
  - `confidence_level` (float, $0.5 < \alpha < 1.0$): VaR confidence level (default `0.99`).
  - `threshold_quantile` (float, $0.5 < u < 1.0$): POT threshold quantile (default `0.95`).
- **Response**: `gpd_params` (`xi`, `beta`, `threshold`, `n_exceedances`, `n_total`), `var`, `cvar`, `confidence_level`.

### `POST /v1/macro/nelson-siegel`
- **Request Body**:
  - `tenors` (list of floats, $\ge 4$ tenors): Maturities in years.
  - `yields` (list of floats, same length): Observed yields.
- **Response**: `beta_0`, `beta_1`, `beta_2`, `lambda_param`, `fitted_yields`, `rmse`.

---

## 7. Environment Variables

All settings use the `ECONAPI_` prefix and can be configured via a `.env` file or cloud platform settings:

| Variable | Type | Default | Description |
|:---|:---:|:---:|:---|
| `ECONAPI_APP_NAME` | string | `econAPI` | Application display name |
| `ECONAPI_VERSION` | string | `1.0.0` | API release version |
| `ECONAPI_ENVIRONMENT` | string | `development` | `development` (bypasses auth) or `production` (enforces API key) |
| `ECONAPI_VALID_API_KEYS` | JSON list | `[]` | List of allowed API keys (e.g. `["rapidapi-gateway-secret-key-prod"]`) |
| `ECONAPI_CORS_ORIGINS` | JSON list | `["*"]` | Allowed CORS origins for browser dashboards |
| `ECONAPI_HOST` | string | `0.0.0.0` | Bind host interface |
| `ECONAPI_PORT` | integer | `8000` | Bind port (overridden dynamically by Render via `$PORT`) |

---

## 8. License

This project is licensed under the [MIT License](LICENSE).
