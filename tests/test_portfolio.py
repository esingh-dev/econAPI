"""Tests for Hierarchical Risk Parity portfolio optimisation engine."""

from __future__ import annotations

import numpy as np
import pytest

from app.engines.hrp import hrp_optimize, compute_shrunk_covariance


# ═══════════════════════════════════════════════════════════════════
# Weight Allocation Tests
# ═══════════════════════════════════════════════════════════════════

class TestHRPWeights:
    """Verify HRP produces valid, sensible portfolio allocations."""

    @pytest.fixture()
    def random_returns(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.normal(0.0005, 0.02, size=(200, 5))

    def test_weights_sum_to_one(self, random_returns: np.ndarray):
        result = hrp_optimize(random_returns)
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 1e-10

    def test_weights_all_positive(self, random_returns: np.ndarray):
        result = hrp_optimize(random_returns)
        for w in result["weights"].values():
            assert w > 0, f"Weight {w} is not positive"

    def test_portfolio_variance_positive(self, random_returns: np.ndarray):
        result = hrp_optimize(random_returns)
        assert result["portfolio_variance"] > 0

    def test_two_assets_lower_variance_gets_higher_weight(self):
        """Lower-variance asset should receive a larger weight."""
        rng = np.random.default_rng(99)
        low_vol = rng.normal(0.001, 0.01, size=(500, 1))
        high_vol = rng.normal(0.001, 0.05, size=(500, 1))
        returns = np.hstack([low_vol, high_vol])

        result = hrp_optimize(returns, asset_names=["Low", "High"])
        assert result["weights"]["Low"] > result["weights"]["High"]

    def test_with_asset_names(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, size=(100, 3))
        names = ["SPY", "TLT", "GLD"]
        result = hrp_optimize(returns, asset_names=names)
        assert set(result["weights"].keys()) == set(names)

    def test_default_asset_names(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, size=(100, 3))
        result = hrp_optimize(returns)
        assert "Asset_0" in result["weights"]
        assert "Asset_1" in result["weights"]
        assert "Asset_2" in result["weights"]

    def test_many_assets(self):
        """HRP should handle moderate dimensionality (30 assets)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, size=(300, 30))
        result = hrp_optimize(returns)
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 1e-10
        assert len(result["weights"]) == 30


# ═══════════════════════════════════════════════════════════════════
# Covariance Shrinkage Tests
# ═══════════════════════════════════════════════════════════════════

class TestLedoitWolf:
    def test_positive_definite(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, size=(100, 5))
        cov = compute_shrunk_covariance(returns)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0), "Covariance matrix is not positive definite"

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, size=(100, 5))
        cov = compute_shrunk_covariance(returns)
        assert np.allclose(cov, cov.T)

    def test_shape(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, size=(100, 7))
        cov = compute_shrunk_covariance(returns)
        assert cov.shape == (7, 7)


# ═══════════════════════════════════════════════════════════════════
# Edge-Case / Validation Tests
# ═══════════════════════════════════════════════════════════════════

class TestHRPValidation:
    def test_too_few_periods_raises(self):
        with pytest.raises(ValueError):
            hrp_optimize(np.array([[0.01, 0.02]]))

    def test_single_asset_raises(self):
        with pytest.raises(ValueError):
            hrp_optimize(np.array([[0.01], [0.02]]))
