"""econAPI Python Client SDK.

Official Python client wrapper library for econAPI via RapidAPI or direct deployment.
Provides typed, validated interfaces for Black-Scholes, Heston Stochastic Volatility,
Hierarchical Risk Parity (HRP), Extreme Value Theory (EVT) tail-risk, and Nelson-Siegel
yield curve modeling.

Example Usage:
--------------
    from client import EconAPI

    # Initialize with your RapidAPI key
    api = EconAPI(api_key="YOUR_RAPIDAPI_KEY")

    # 1. Price an option with Black-Scholes-Merton and analytical Greeks
    bsm = api.price_bsm(
        spot=100.0, strike=100.0, time_to_maturity=1.0,
        risk_free_rate=0.05, volatility=0.20, option_type="call"
    )
    print("Call Price:", bsm["price"], "Delta:", bsm["delta"])

    # 2. Price under Heston Stochastic Volatility & extract implied vol
    heston = api.price_heston(
        spot=100.0, strike=100.0, time_to_maturity=1.0, risk_free_rate=0.05,
        v0=0.04, kappa=2.0, theta=0.04, xi=0.30, rho=-0.70, option_type="call"
    )
    print("Heston Price:", heston["price"], "Implied Vol:", heston["implied_volatility"])
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
import requests


# ═══════════════════════════════════════════════════════════════════
# Custom SDK Exceptions
# ═══════════════════════════════════════════════════════════════════

class EconAPIError(Exception):
    """Base exception for all econAPI errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(EconAPIError):
    """Raised when the API key is missing, invalid, or unauthorized (HTTP 403)."""
    pass


class ValidationError(EconAPIError):
    """Raised when mathematical inputs fail parameter bounds validation (HTTP 400 / 422)."""
    pass


class RateLimitError(EconAPIError):
    """Raised when API quota or rate limits are exceeded (HTTP 429)."""
    pass


# ═══════════════════════════════════════════════════════════════════
# EconAPI Client Class
# ═══════════════════════════════════════════════════════════════════

class EconAPI:
    """Client for interacting with the econAPI quantitative microservice."""

    def __init__(
        self,
        api_key: str,
        rapidapi_host: str = "econapi.p.rapidapi.com",
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Initialize the EconAPI client.

        Parameters
        ----------
        api_key : str
            Your RapidAPI key or direct service API key.
        rapidapi_host : str, default "econapi.p.rapidapi.com"
            The RapidAPI host header associated with your marketplace listing.
        base_url : str, optional
            Direct URL if bypassing RapidAPI (e.g., 'https://econapi.onrender.com').
            Defaults to 'https://{rapidapi_host}'.
        timeout : float, default 30.0
            Request timeout in seconds.
        session : requests.Session, optional
            Optional pre-configured requests session for connection pooling.
        """
        if not api_key or not isinstance(api_key, str):
            raise ValueError("api_key must be a non-empty string.")

        self.api_key = api_key.strip()
        self.rapidapi_host = rapidapi_host.strip()
        self.base_url = (base_url or f"https://{self.rapidapi_host}").rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _headers(self) -> Dict[str, str]:
        """Construct standard HTTP headers for RapidAPI or direct authentication."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "econAPI-Python-SDK/1.0.0",
        }
        if "rapidapi.com" in self.base_url:
            headers["X-RapidAPI-Key"] = self.api_key
            headers["X-RapidAPI-Host"] = self.rapidapi_host
        else:
            # Direct backend deployment authentication
            headers["X-API-Key"] = self.api_key
        return headers

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute HTTP request with error translation."""
        url = f"{self.base_url}{path}"
        headers = self._headers()

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=payload if payload is not None else None,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise EconAPIError(f"Network error connecting to {url}: {exc}") from exc

        # Translate HTTP status codes to typed exceptions
        if response.status_code == 200:
            return response.json()

        try:
            body = response.json()
            detail = body.get("detail", response.text)
        except Exception:
            body = response.text
            detail = response.text

        if response.status_code == 403:
            raise AuthenticationError(
                f"Authentication failed: {detail}. Verify your API key.",
                status_code=403,
                response_body=body,
            )
        elif response.status_code in (400, 422):
            raise ValidationError(
                f"Validation error ({response.status_code}): {detail}",
                status_code=response.status_code,
                response_body=body,
            )
        elif response.status_code == 429:
            raise RateLimitError(
                f"Rate limit exceeded: {detail}. Please upgrade your subscription tier.",
                status_code=429,
                response_body=body,
            )
        else:
            raise EconAPIError(
                f"API error ({response.status_code}): {detail}",
                status_code=response.status_code,
                response_body=body,
            )

    # ─────────────────────────────────────────────────────────────────
    # System & Diagnostics
    # ─────────────────────────────────────────────────────────────────

    def health(self) -> Dict[str, str]:
        """Check API service health and current release version.

        Returns
        -------
        dict
            {"status": "healthy", "version": "1.0.0"}
        """
        return self._request("GET", "/health")

    # ─────────────────────────────────────────────────────────────────
    # Derivatives Pricing
    # ─────────────────────────────────────────────────────────────────

    def price_bsm(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        option_type: Literal["call", "put"] = "call",
    ) -> Dict[str, float]:
        """Price a European vanilla option using the Black-Scholes-Merton model.

        Parameters
        ----------
        spot : float
            Current price of underlying asset (S > 0).
        strike : float
            Strike price (K > 0).
        time_to_maturity : float
            Maturity horizon in years (T > 0).
        risk_free_rate : float
            Continuously compounded annual risk-free rate.
        volatility : float
            Annualized standard deviation of returns (sigma > 0).
        option_type : {"call", "put"}, default "call"
            Type of derivative option.

        Returns
        -------
        dict
            {"price": float, "delta": float, "gamma": float,
             "vega": float, "theta": float, "rho": float}
        """
        payload = {
            "S": float(spot),
            "K": float(strike),
            "T": float(time_to_maturity),
            "r": float(risk_free_rate),
            "sigma": float(volatility),
            "option_type": option_type,
        }
        return self._request("POST", "/v1/derivatives/bsm", payload)

    def price_heston(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        option_type: Literal["call", "put"] = "call",
    ) -> Dict[str, float]:
        """Price a European option under the Heston Stochastic Volatility model.

        Parameters
        ----------
        spot : float
            Current asset price (S > 0).
        strike : float
            Strike price (K > 0).
        time_to_maturity : float
            Time to maturity in years (T > 0).
        risk_free_rate : float
            Risk-free interest rate (r).
        v0 : float
            Initial instantaneous variance (v0 > 0).
        kappa : float
            Mean-reversion rate (kappa > 0).
        theta : float
            Long-term variance level (theta > 0).
        xi : float
            Volatility of variance / vol-of-vol (xi > 0).
        rho : float
            Correlation between asset and variance processes (rho in [-1, 1]).
        option_type : {"call", "put"}, default "call"
            Option type.

        Returns
        -------
        dict
            {"price": float, "implied_volatility": float}
        """
        payload = {
            "S": float(spot),
            "K": float(strike),
            "T": float(time_to_maturity),
            "r": float(risk_free_rate),
            "v0": float(v0),
            "kappa": float(kappa),
            "theta": float(theta),
            "xi": float(xi),
            "rho": float(rho),
            "option_type": option_type,
        }
        return self._request("POST", "/v1/derivatives/heston", payload)

    # ─────────────────────────────────────────────────────────────────
    # Portfolio Optimization
    # ─────────────────────────────────────────────────────────────────

    def optimize_hrp(
        self,
        returns: Union[List[List[float]], Sequence[Sequence[float]], Any],
        asset_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Allocate optimal weights using Hierarchical Risk Parity (HRP).

        Parameters
        ----------
        returns : 2D array-like of shape (T, N)
            Historical returns matrix where T >= 2 is time periods and N >= 2 is assets.
        asset_names : list of str, optional
            Human-readable asset tickers (length must equal N).

        Returns
        -------
        dict
            {"weights": {name: weight}, "portfolio_variance": float, "sharpe_ratio": float}
        """
        # Convert numpy arrays or nested iterables to standard JSON lists
        if hasattr(returns, "tolist"):
            ret_list = returns.tolist()
        else:
            ret_list = [list(row) for row in returns]

        payload: Dict[str, Any] = {"returns": ret_list}
        if asset_names is not None:
            payload["asset_names"] = list(asset_names)

        return self._request("POST", "/v1/portfolio/hrp", payload)

    # ─────────────────────────────────────────────────────────────────
    # Tail Risk & Extreme Value Theory
    # ─────────────────────────────────────────────────────────────────

    def compute_evt_cvar(
        self,
        returns: Union[List[float], Sequence[float], Any],
        confidence_level: float = 0.99,
        threshold_quantile: float = 0.95,
    ) -> Dict[str, Any]:
        """Compute tail risk VaR and CVaR (Expected Shortfall) via GPD Peaks-Over-Threshold.

        Parameters
        ----------
        returns : 1D array-like of shape (T,)
            Historical return series (minimum 50 observations).
        confidence_level : float, default 0.99
            Confidence level alpha in (0.5, 1.0).
        threshold_quantile : float, default 0.95
            Tail threshold quantile u in (0.5, 1.0).

        Returns
        -------
        dict
            {"gpd_params": {"xi": float, "beta": float, "threshold": float,
             "n_exceedances": int, "n_total": int},
             "var": float, "cvar": float, "confidence_level": float}
        """
        if hasattr(returns, "tolist"):
            ret_list = returns.tolist()
        else:
            ret_list = list(returns)

        payload = {
            "returns": [float(r) for r in ret_list],
            "confidence_level": float(confidence_level),
            "threshold_quantile": float(threshold_quantile),
        }
        return self._request("POST", "/v1/risk/evt-cvar", payload)

    # ─────────────────────────────────────────────────────────────────
    # Macro Econometrics & Yield Curve Calibration
    # ─────────────────────────────────────────────────────────────────

    def fit_yield_curve(
        self,
        tenors: Union[List[float], Sequence[float], Any],
        yields: Union[List[float], Sequence[float], Any],
    ) -> Dict[str, Any]:
        """Calibrate the Nelson-Siegel parametric yield curve to market yields.

        Parameters
        ----------
        tenors : 1D array-like of shape (M,)
            Maturity horizons in years (minimum 4 tenors).
        yields : 1D array-like of shape (M,)
            Observed annual yields matching tenors.

        Returns
        -------
        dict
            {"beta_0": float, "beta_1": float, "beta_2": float,
             "lambda_param": float, "fitted_yields": list[float], "rmse": float}
        """
        tenors_list = tenors.tolist() if hasattr(tenors, "tolist") else list(tenors)
        yields_list = yields.tolist() if hasattr(yields, "tolist") else list(yields)

        payload = {
            "tenors": [float(t) for t in tenors_list],
            "yields": [float(y) for y in yields_list],
        }
        return self._request("POST", "/v1/macro/nelson-siegel", payload)


# ═══════════════════════════════════════════════════════════════════
# Self-Test Demo Block
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing EconAPI SDK initialization...")
    client = EconAPI(api_key="demo-key-12345", base_url="http://localhost:8000")
    print(f"Client configured with base URL: {client.base_url}")
    print("Ready for deployment to PyPI and RapidAPI.")
