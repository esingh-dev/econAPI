# econAPI: Production Quantitative Finance & Risk Microservice

[![CI Pipeline](https://github.com/your-username/econAPI/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/econAPI/actions)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 116 Passed](https://img.shields.io/badge/tests-116%20passed-brightgreen.svg)](tests/)
[![Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7.svg)](render.yaml)

**econAPI** is a production-grade financial engineering microservice engineered with **FastAPI**, **NumPy**, **SciPy**, **Scikit-learn**, and **Statsmodels**. It provides high-performance, hardened calculation engines for:

1. **Derivatives & Volatility**: Generalized Black-Scholes-Merton with continuous dividends $q$, complete 1st and higher-order Greeks (Vanna, Volga, Charm, Veta, Color, Speed, Ultima), finite-difference verification, no-arbitrage bounds, Heston stochastic volatility with continuous Little Trap formulation, market calibration, Binomial (CRR / Jarrow-Rudd) and Trinomial American trees, Exotic options (Asian, Barrier, Digital), Monte Carlo with antithetic & control variate variance reduction, and SVI surface calibration with butterfly & calendar arbitrage checks.
2. **Yield Curves & Fixed Income**: Nelson-Siegel and 6-parameter Nelson-Siegel-Svensson (NSS) calibration, zero-coupon, spot, forward, and par yield curves, curve bootstrapping from money market deposits, futures, and swaps, bond pricing, accrued interest (Actual/Actual & 30/360), YTM (Brent solver), Macaulay/Modified duration, Convexity, DV01/PV01, curve shock scenarios, Key-Rate Duration (KRD), and short-rate models (Vasicek, CIR, Hull-White 1-Factor).
3. **Portfolio Construction & Optimization**: Markowitz Mean-Variance (Max Sharpe, Min Variance, Target Return/Risk), Equal Risk Contribution (ERC) Risk Parity, Inverse Volatility allocation, Black-Litterman model with investor views and posterior covariance, Rockafellar-Uryasev Linear Programming CVaR optimization, box weight constraints, turnover limits, proportional transaction cost deduction, walk-forward out-of-sample backtesting, Brinson-Fachler attribution, and comprehensive performance metrics (Sharpe, Sortino, Calmar, Omega, Max Drawdown, Ulcer Index).
4. **Time Series Risk & Analytics**: Multi-model VaR & Expected Shortfall (Historical, Parametric Gaussian, Student-t, Cornish-Fisher expansion), RiskMetrics EWMA volatility, GARCH(1,1) Quasi-Maximum Likelihood Estimation (QMLE), multi-factor regression risk decomposition, historical & macro stress testing, EVT Peaks-Over-Threshold (GPD) with Mean Excess diagnostics, Kolmogorov-Smirnov fit tests, QQ plots, asymptotic confidence intervals, and regulatory VaR backtesting (Kupiec POF LR test, Christoffersen Independence test, Basel Traffic Light classification).

---

## Model Assumptions, Mathematical Formulations & Safeguards

### 1. Options and Derivatives

#### Generalized Black-Scholes-Merton (Merton 1973)
- **Model Assumptions**: Geometric Brownian Motion $dS_t = (r - q) S_t dt + \sigma S_t dW_t$ with constant risk-free rate $r$, constant continuous dividend yield $q$, constant volatility $\sigma$, frictionless trading, and continuous rebalancing.
- **Formulation**:
  $$d_1 = \frac{\ln(S/K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$
  $$C = S e^{-q T} N(d_1) - K e^{-r T} N(d_2), \quad P = K e^{-r T} N(-d_2) - S e^{-q T} N(-d_1)$$
- **Put-Call Parity**: $C - P = S e^{-q T} - K e^{-r T}$.
- **No-Arbitrage Bounds**: $\max(0, S e^{-q T} - K e^{-r T}) \le C \le S e^{-q T}$ and $\max(0, K e^{-r T} - S e^{-q T}) \le P \le K e^{-r T}$.
- **Higher-Order Greeks**:
  - Vanna: $\frac{\partial^2 V}{\partial S \partial \sigma} = -e^{-q T} n(d_1) \frac{d_2}{\sigma}$
  - Volga (Vomma): $\frac{\partial^2 V}{\partial \sigma^2} = \text{Vega} \cdot \frac{d_1 d_2}{\sigma}$
  - Charm: $-\frac{\partial \Delta}{\partial T} = e^{-q T} \left[ q N(d_1) - n(d_1) \frac{2(r-q)T - d_2 \sigma \sqrt{T}}{2 T \sigma \sqrt{T}} \right]$
  - Veta: $-\frac{\partial \text{Vega}}{\partial T}$
  - Color: $\frac{\partial \text{Gamma}}{\partial T}$, Speed: $\frac{\partial \text{Gamma}}{\partial S}$, Ultima: $\frac{\partial \text{Volga}}{\partial \sigma}$.
- **Safeguards**: Analytical Greeks are verified against central finite differences with max relative error tolerance $< 1\%$. Expiry ($T \le 0$), zero volatility ($\sigma \to 0$), zero strike ($K \to 0$), and zero spot ($S \to 0$) are resolved via exact limit functions.

#### Heston Stochastic Volatility Model (1993)
- **Model Assumptions**:
  $$dS_t = (r - q) S_t dt + \sqrt{v_t} S_t dW_t^S$$
  $$dv_t = \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v, \quad d\langle W^S, W^v \rangle_t = \rho dt$$
- **Numerical Stability**: Uses Albrecher et al. (2007) continuous "Little Trap" formulation of the characteristic function $\phi(u)$ to eliminate complex branch-cut discontinuities.
- **Feller Condition**: $2\kappa\theta > \xi^2$ guarantees variance stays strictly positive. Violation triggers a clear advisory warning while maintaining stable integration.
- **Quadrature Convergence**: Adaptive Gauss-Kronrod numerical quadrature tracking absolute and relative integration error tolerances ($\le 10^{-4}$).
- **Implied Volatility Inversion**: Brent-Decker bracketed root-finder backed by rational asymptotic fallbacks when approaching deep out-of-the-money bounds.
- **Market Calibration**: L-BFGS-B / SLSQP optimization across strike and maturity grids minimizing price root-mean-square error (RMSE) with Feller regularization.

#### American Options via Binomial & Trinomial Trees
- **Binomial (CRR & Jarrow-Rudd)**: Discrete backward induction lattices evaluating $\max(\text{Continuation}, \text{Exercise})$.
- **Trinomial (Kamrad-Ritchken)**: 3-branch spatial grid ($\Delta x = \sigma \sqrt{3 \Delta t}$) with second-order accuracy.
- **Early Exercise Premium**: Output reports $P_{\text{American}} - P_{\text{European}}$.

#### Exotic Derivatives
- **Asian Options**: Geometric average price/strike exact closed form (Kemna-Vorst 1990) and arithmetic average two-moment approximation (Turnbull-Wakeman 1991).
- **Barrier Options**: Complete analytical closed-form solutions (Reiner & Rubinstein 1991) for Up/Down, In/Out, Call/Put with cash rebates and immediate breach detection.
- **Digital / Binary Options**: Cash-or-Nothing and Asset-or-Nothing analytical formulas.

#### Monte Carlo Engine
- **Variance Reduction**: Antithetic Variates paired with Control Variates using analytical BSM as a covariate.
- **Reproducibility**: Deterministic PRNG seed parameter ensuring reproducible simulation runs.
- **Diagnostics**: Standard error and 95% asymptotic confidence intervals.

#### Volatility Surface & SVI
- **Raw SVI**: $w(k) = a + b[\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}]$.
- **Arbitrage Diagnostics**:
  - Roger Lee extreme strike wing slope bounds: $b(1 + |\rho|) < 2$.
  - Gatheral non-negativity of risk-neutral density: $g(k) \ge 0$.
  - Calendar spread arbitrage: total variance non-decreasing in tenor $w(k, T_2) \ge w(k, T_1)$.

---

### 2. Yield Curves and Fixed Income

#### Nelson-Siegel & Nelson-Siegel-Svensson (NSS)
- **Nelson-Siegel (3-Factor)**:
  $$y(\tau) = \beta_0 + \beta_1 \frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} + \beta_2 \left(\frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} - e^{-\tau/\lambda}\right)$$
- **Nelson-Siegel-Svensson (4-Factor / 6-Parameter)**:
  $$y(\tau) = \text{NS}(\tau) + \beta_3 \left(\frac{1 - e^{-\tau/\lambda_2}}{\tau/\lambda_2} - e^{-\tau/\lambda_2}\right)$$
- **Optimization Stability**: Diebold-Li two-step grid search over decay parameters $(\lambda_1, \lambda_2)$ with conditional OLS for linear betas $(\beta_0, \beta_1, \beta_2, \beta_3)$, fine-tuned via bounded L-BFGS-B.
- **Unit Convention**: Explicit support for `auto`, `percent` ($5.25$), and `decimal` ($0.0525$) with auto-scaling to avoid numerical divergence.
- **Diagnostics**: RMSE, MAE, Max Error, $R^2$, AIC, and BIC.
- **Robust Loss**: Optional Huber and Soft-$L_1$ loss functions for noisy or sparse sovereign debt data.

#### Curve Bootstrapping & Analytics
- **Bootstrapping**: Unwinds zero-coupon discount factors sequentially from short-term money market deposits, futures/FRAs, and par interest rate swaps.
- **Bond Pricing**: Accrued interest (Actual/Actual and 30/360), dirty/clean price, Yield to Maturity (Brent solver), Macaulay duration, Modified duration, Convexity, and DV01 / PV01.
- **Key-Rate Duration (KRD)**: Triangular tent-function shifts decomposing interest rate exposure across benchmark maturities ($0.25\text{Y}$ to $30\text{Y}$).
- **Term Structure Models**:
  - **Vasicek (1977)**: Mean-reverting Gaussian short-rate model with analytical bond pricing.
  - **Cox-Ingersoll-Ross (1985)**: Square-root diffusion ensuring non-negative nominal interest rates.
  - **Hull-White (1-Factor)**: Analytical zero-coupon bond pricing exactly calibrated to initial forward curve.

---

### 3. Portfolio Construction & Optimization

#### Hierarchical Risk Parity (HRP)
- **Algorithm (López de Prado 2016)**:
  1. Tree clustering using single-linkage on metric correlation distance $d_{ij} = \sqrt{\frac{1}{2}(1 - \rho_{ij})}$.
  2. Quasi-diagonalization (seriation) ordering assets along the dendrogram leaf order.
  3. Recursive bisection allocating weights inversely proportional to cluster variance.
- **Enhancements**:
  - Covariance estimators: Ledoit-Wolf shrinkage, standard sample covariance, EWMA covariance, and semi-covariance.
  - Missing-data imputation: forward-fill, dropna, mean imputation.
  - Box constraints $[w_{\min}, w_{\max}]$ via iterative redistribution.
  - Turnover tracking and transaction cost deduction.
  - Walk-forward rolling out-of-sample backtesting.

#### Markowitz Mean-Variance Optimization
- Quadratic programming (SciPy SLSQP) supporting Max Sharpe, Global Minimum Variance (GMV), Target Return, and Target Risk with box bounds and turnover constraints.

#### Equal Risk Contribution (ERC) / Risk Parity
- Equalizes marginal risk contributions: $\text{TRC}_i = w_i (\Sigma w)_i / \sigma_p^2 = B_i$.

#### Black-Litterman Model
- Implied market equilibrium returns $\Pi = \lambda \Sigma w_{\text{mkt}}$ blended with investor views $P \mu = Q + \Omega$ to produce posterior expected returns $\mu_{\text{BL}}$ and covariance $\Sigma_{\text{BL}}$.

#### CVaR Optimization (Rockafellar & Uryasev 2000)
- Exact Linear Programming formulation minimizing Conditional Value-at-Risk at confidence level $\alpha$ under arbitrary scenario matrices.

#### Attribution & Performance
- **Brinson-Fachler Attribution**: Allocation effect, Selection effect, and Interaction effect vs benchmark.
- **Risk Metrics**: Sharpe, Sortino (downside deviation), Calmar, Omega, Max Drawdown, Ulcer Index, Information Ratio, and Tracking Error.

---

### 4. Risk and Time Series Models

#### Multi-Model Value-at-Risk & Expected Shortfall
- **Historical Simulation**: Empirical quantile of historical return losses.
- **Parametric Gaussian**: $\text{VaR}_\alpha = \mu + z_\alpha \sigma$.
- **Student-t**: Heavy-tailed parametric quantile accounting for degrees of freedom.
- **Cornish-Fisher Expansion**: Polynomial adjustment accounting for skewness $S$ and excess kurtosis $K$.

#### Volatility Modeling
- **EWMA (RiskMetrics)**: $\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_{t-1}^2$ (standard $\lambda = 0.94$).
- **GARCH(1,1)**: Quasi-Maximum Likelihood Estimation of $\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$ with stationarity check ($\alpha + \beta < 1$) and unconditional long-run variance.

#### Factor Models & Stress Testing
- **Factor Models**: Ordinary least squares regression decomposing systematic vs idiosyncratic risk.
- **Stress Testing**: Revaluation under historical crises (2008 GFC, 2020 Covid, 1987 Black Monday, Stagflation) and custom macro factor shocks.

#### VaR Backtesting & Model Validation
- **Kupiec POF LR Test**: Likelihood ratio test evaluating whether observed exception rate equals $(1 - \alpha)$.
- **Christoffersen Independence Test**: Markov transition test checking whether exceptions cluster in time.
- **Basel Traffic Light System**: Regulatory classification into Green, Yellow, and Red zones.

#### Extreme Value Theory (EVT) Peaks-Over-Threshold
- **GPD Formulation**: Fits Generalized Pareto Distribution to excess losses over a high threshold $u$.
- **Diagnostics**: Mean Excess Plot, parameter stability scan, Kolmogorov-Smirnov goodness-of-fit test, and QQ plot coordinates.
- **Confidence Intervals**: 95% asymptotic delta-method confidence intervals for VaR.

---

## Complete API Endpoints Catalog

### Derivatives & Options (`/v1/derivatives`)
- `POST /v1/derivatives/bsm`: Generalized BSM pricing, 1st & higher-order Greeks, no-arbitrage bounds, finite-difference verification.
- `POST /v1/derivatives/heston`: Heston stochastic volatility with continuous Little Trap formulation, Feller check, implied volatility.
- `POST /v1/derivatives/heston/calibrate`: Calibrate $(v_0, \kappa, \theta, \xi, \rho)$ to market option price quotes.
- `POST /v1/derivatives/tree`: Binomial (CRR / Jarrow-Rudd) and Trinomial trees for European and American options.
- `POST /v1/derivatives/asian`: Geometric (exact) and Arithmetic (Turnbull-Wakeman) Asian option pricing.
- `POST /v1/derivatives/barrier`: Reiner-Rubinstein analytical pricing for Up/Down, In/Out barrier options with rebates.
- `POST /v1/derivatives/binary`: Cash-or-Nothing and Asset-or-Nothing digital/binary option pricing.
- `POST /v1/derivatives/monte-carlo`: Monte Carlo option pricing with Antithetic and Control Variate variance reduction.
- `POST /v1/derivatives/svi/calibrate`: Calibrate raw SVI volatility slice with Roger Lee wing bounds and butterfly checks.
- `POST /v1/derivatives/svi/calendar-arbitrage`: Verify absence of calendar spread arbitrage across volatility surface slices.

### Fixed Income & Yield Curves (`/v1/fixed-income`)
- `POST /v1/fixed-income/nelson-siegel`: 3-Factor Nelson-Siegel curve calibration with unit detection and fit diagnostics.
- `POST /v1/fixed-income/nss`: 4-Factor / 6-Parameter Nelson-Siegel-Svensson curve calibration.
- `POST /v1/fixed-income/curve-suite`: Spot, instantaneous forward, par yield curves, and discount factors.
- `POST /v1/fixed-income/bootstrap`: Bootstrap zero curve from deposits, futures/FRAs, and interest rate swaps.
- `POST /v1/fixed-income/bond-price`: Bond clean/dirty pricing, accrued interest, YTM solver, Macaulay/Modified duration, Convexity, DV01.
- `POST /v1/fixed-income/curve-shock`: Parallel, steepener, flattener, and butterfly twist curve shock scenarios.
- `POST /v1/fixed-income/key-rate-duration`: Key-Rate Duration (KRD) profile across benchmark tenors.
- `POST /v1/fixed-income/models/vasicek`: Analytical zero-coupon bond pricing under the Vasicek model.
- `POST /v1/fixed-income/models/cir`: Analytical zero-coupon bond pricing under the Cox-Ingersoll-Ross (CIR) model.
- `POST /v1/fixed-income/models/hull-white`: Analytical zero-coupon bond pricing under the Hull-White 1-Factor model.

### Portfolio Construction & Optimization (`/v1/portfolio`)
- `POST /v1/portfolio/hrp`: Hierarchical Risk Parity with covariance shrinkage, constraints, and turnover tracking.
- `POST /v1/portfolio/hrp/backtest`: Walk-forward out-of-sample backtest of HRP portfolio allocation.
- `POST /v1/portfolio/mean-variance`: Markowitz Mean-Variance optimization (Max Sharpe, Min Variance, Target Return/Risk).
- `POST /v1/portfolio/risk-parity`: Equal Risk Contribution (ERC) / Risk Parity optimization.
- `POST /v1/portfolio/black-litterman`: Black-Litterman model blending market equilibrium with investor views.
- `POST /v1/portfolio/cvar-optimization`: Rockafellar-Uryasev LP Conditional Value-at-Risk optimization.
- `POST /v1/portfolio/attribution`: Brinson-Fachler performance attribution (allocation, selection, interaction).
- `POST /v1/portfolio/metrics`: Risk-adjusted performance metrics (Sharpe, Sortino, Calmar, Max Drawdown, Ulcer Index).

### Risk Analytics & Macro (`/v1/risk`)
- `POST /v1/risk/evt-cvar`: Peaks-Over-Threshold tail risk VaR & CVaR with KS test, QQ plot, Mean Excess diagnostics, and CIs.
- `POST /v1/risk/var-suite`: Multi-model VaR & Expected Shortfall (Historical, Gaussian, Student-t, Cornish-Fisher).
- `POST /v1/risk/garch`: GARCH(1,1) Quasi-Maximum Likelihood conditional volatility modeling.
- `POST /v1/risk/ewma`: RiskMetrics EWMA volatility modeling and path generation.
- `POST /v1/risk/factor-model`: Multi-factor regression, beta sensitivities, and systematic vs idiosyncratic variance.
- `POST /v1/risk/stress-test`: Historical (2008 GFC, 2020 Covid, 1987 Crash) and synthetic stress testing.
- `POST /v1/risk/var-backtest`: Kupiec POF, Christoffersen Independence, and Basel Traffic Light VaR backtesting.
- `POST /v1/macro/nelson-siegel`: Legacy Nelson-Siegel calibration endpoint.

---

## Local Installation & Verification

```bash
# 1. Clone the repository
git clone https://github.com/your-username/econAPI.git
cd econAPI

# 2. Activate virtual environment
.venv\Scripts\activate

# 3. Run complete automated test suite (116 tests across all engines)
pytest -v

# 4. Start high-performance development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation is available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Interactive Studio UI**: [http://localhost:8000/](http://localhost:8000/)

---

## License

This project is licensed under the [MIT License](LICENSE).
