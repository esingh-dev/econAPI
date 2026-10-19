================================================================================
econAPI: High-Performance Quantitative Finance & Risk Microservice
================================================================================

Version: 1.0.0
Author: econAPI Contributors
License: MIT License
Language: Python 3.11+ / FastAPI / NumPy / SciPy / Scikit-Learn / Statsmodels

--------------------------------------------------------------------------------
1. OVERVIEW
--------------------------------------------------------------------------------
econAPI is a production-ready financial engineering microservice providing
sub-millisecond derivative pricing, portfolio optimization, yield curve
calibration, and tail-risk analytics.

Mathematical Modules Included:
- Black-Scholes-Merton: Analytical vanilla pricing and analytical Greeks
  (Delta, Gamma, Vega, Theta, Rho) with boundary condition handling.
- Heston Model: Semi-analytical pricing via numerical quadrature integration
  of the characteristic function (Albrecher/Gatheral stable formulation),
  plus BSM implied volatility inversion via Brent's method.
- Hierarchical Risk Parity (HRP): López de Prado asset allocation using
  Ledoit-Wolf shrinkage, angular distance correlation, single-linkage
  clustering, quasi-diagonalization (seriation), and recursive bisection.
- Extreme Value Theory (EVT): Peaks-Over-Threshold (POT) tail risk fitting
  to Generalized Pareto Distribution (GPD) with closed-form VaR and CVaR.
- Nelson-Siegel Yield Curve: 4-parameter term structure calibration using
  Diebold-Li two-step grid initialization with non-linear least squares.
- Macro Econometrics: Johansen Cointegration Rank Test & Vector Error
  Correction Model (VECM) estimation.

--------------------------------------------------------------------------------
2. QUICK START (LOCAL RUN)
--------------------------------------------------------------------------------
Prerequisites: Python 3.11 or higher installed.

Step 1: Clone or navigate to the directory
    cd econAPI

Step 2: Create and activate virtual environment
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Linux / macOS:
    source .venv/bin/activate

Step 3: Install dependencies
    pip install --upgrade pip
    pip install -r requirements.txt

Step 4: Run the API server
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Step 5: Access Interactive Documentation
    Swagger UI: http://localhost:8000/docs
    ReDoc:      http://localhost:8000/redoc
    Health:     http://localhost:8000/health

--------------------------------------------------------------------------------
3. RUNNING IN DOCKER
--------------------------------------------------------------------------------
Build image:
    docker build -t econapi:latest .

Run container:
    docker run -d -p 8000:8000 --name econapi econapi:latest

Run with production API key authentication:
    docker run -d -p 8000:8000 \
      -e ECONAPI_ENVIRONMENT=production \
      -e ECONAPI_VALID_API_KEYS='["your-secret-api-key"]' \
      --name econapi econapi:latest

--------------------------------------------------------------------------------
4. VERIFICATION & TESTS
--------------------------------------------------------------------------------
Execute the complete test suite (66 tests across all engines):
    pytest -v

Execute the literature benchmark verification runner:
    python scripts/verify_benchmarks.py

--------------------------------------------------------------------------------
5. API ENDPOINTS SUMMARY
--------------------------------------------------------------------------------
GET  /health
     Check microservice liveness and version.

POST /v1/derivatives/bsm
     Price European call/put options & calculate Greeks under BSM.
     Payload: {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "sigma": 0.2, "option_type": "call"}

POST /v1/derivatives/heston
     Price European options under Heston stochastic volatility & extract implied vol.
     Payload: {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7, "option_type": "call"}

POST /v1/portfolio/hrp
     Compute Hierarchical Risk Parity asset weights and Sharpe ratio.
     Payload: {"returns": [[0.01, -0.02], [0.02, 0.01], ...], "asset_names": ["SPY", "TLT"]}

POST /v1/risk/evt-cvar
     Compute Value-at-Risk and Expected Shortfall using POT GPD tail fitting.
     Payload: {"returns": [-0.02, 0.01, -0.03, ...], "confidence_level": 0.99, "threshold_quantile": 0.95}

POST /v1/macro/nelson-siegel
     Fit Nelson-Siegel parametric yield curve across maturities.
     Payload: {"tenors": [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], "yields": [5.25, 5.10, 4.85, 4.50, 4.35, 4.20, 4.15, 4.10, 4.30, 4.40]}

--------------------------------------------------------------------------------
6. ZERO-COST CLOUD DEPLOYMENT (HUGGING FACE SPACES / RENDER)
--------------------------------------------------------------------------------
Deploy for 100% free on Hugging Face Spaces (2 vCPUs, 16GB RAM):
1. Create a free account on huggingface.co
2. Create New Space -> Name: econAPI -> SDK: Docker (Blank) -> Free Hardware
3. Push your repository:
   git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/econAPI
   git push space main
4. Your API is live with free HTTPS at:
   https://<YOUR_USERNAME>-econapi.hf.space/docs
================================================================================
