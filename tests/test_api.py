"""End-to-end API integration tests for econAPI endpoints and middleware."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ═══════════════════════════════════════════════════════════════════
# System & Health Checks
# ═══════════════════════════════════════════════════════════════════

def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == settings.VERSION


# ═══════════════════════════════════════════════════════════════════
# Derivatives Endpoints
# ═══════════════════════════════════════════════════════════════════

def test_api_bsm_call(client: TestClient):
    payload = {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2,
        "option_type": "call",
    }
    response = client.post("/v1/derivatives/bsm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert abs(data["price"] - 10.4506) < 0.01
    assert abs(data["delta"] - 0.6368) < 0.01
    assert data["gamma"] > 0
    assert data["vega"] > 0
    assert "theta" in data
    assert "rho" in data


def test_api_bsm_put(client: TestClient):
    payload = {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2,
        "option_type": "put",
    }
    response = client.post("/v1/derivatives/bsm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert abs(data["price"] - 5.5735) < 0.01
    assert data["delta"] < 0


def test_api_heston(client: TestClient):
    payload = {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "xi": 0.3,
        "rho": -0.7,
        "option_type": "call",
    }
    response = client.post("/v1/derivatives/heston", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["price"] > 0
    assert data["implied_volatility"] > 0


# ═══════════════════════════════════════════════════════════════════
# Portfolio Endpoints
# ═══════════════════════════════════════════════════════════════════

def test_api_hrp(client: TestClient):
    # Simulated 2-asset returns
    returns = [
        [0.01, -0.005],
        [-0.02, 0.01],
        [0.015, 0.02],
        [-0.005, -0.01],
        [0.03, 0.005],
        [0.01, -0.015],
    ]
    payload = {
        "returns": returns,
        "asset_names": ["Equity", "Bonds"],
    }
    response = client.post("/v1/portfolio/hrp", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "weights" in data
    assert "Equity" in data["weights"]
    assert "Bonds" in data["weights"]
    assert abs(sum(data["weights"].values()) - 1.0) < 1e-6
    assert data["portfolio_variance"] > 0


# ═══════════════════════════════════════════════════════════════════
# Risk & Macro Endpoints
# ═══════════════════════════════════════════════════════════════════

def test_api_evt_cvar(client: TestClient):
    import numpy as np
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.02, size=200).tolist()
    payload = {
        "returns": returns,
        "confidence_level": 0.99,
        "threshold_quantile": 0.95,
    }
    response = client.post("/v1/risk/evt-cvar", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["var"] > 0
    assert data["cvar"] >= data["var"]
    assert data["gpd_params"]["n_total"] == 200


def test_api_nelson_siegel(client: TestClient):
    payload = {
        "tenors": [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0],
        "yields": [5.25, 5.10, 4.85, 4.50, 4.35, 4.20, 4.15, 4.10, 4.30, 4.40],
    }
    response = client.post("/v1/macro/nelson-siegel", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "beta_0" in data
    assert "beta_1" in data
    assert "beta_2" in data
    assert "lambda_param" in data
    assert len(data["fitted_yields"]) == 10
    assert data["rmse"] < 0.5


# ═══════════════════════════════════════════════════════════════════
# Authentication Middleware in Production Mode
# ═══════════════════════════════════════════════════════════════════

def test_auth_middleware_production(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "VALID_API_KEYS", ["secret-prod-key-123"])

    # Exempt route /health should pass without key
    res = client.get("/health")
    assert res.status_code == 200

    # Non-exempt route without key should return 403 Forbidden
    payload = {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2, "option_type": "call"}
    res_no_key = client.post("/v1/derivatives/bsm", json=payload)
    assert res_no_key.status_code == 403
    assert res_no_key.json()["error_type"] == "AuthenticationError"

    # Non-exempt route with invalid key should return 403 Forbidden
    res_bad_key = client.post("/v1/derivatives/bsm", json=payload, headers={"X-API-Key": "wrong"})
    assert res_bad_key.status_code == 403

    # Non-exempt route with valid key should succeed (200)
    res_valid_key = client.post(
        "/v1/derivatives/bsm", json=payload, headers={"X-API-Key": "secret-prod-key-123"}
    )
    assert res_valid_key.status_code == 200
