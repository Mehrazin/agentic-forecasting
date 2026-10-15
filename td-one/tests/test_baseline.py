"""Tests for M0 historical baseline with Beta-Binomial."""

import pytest
import numpy as np

from td_one.models.baseline import HistoricalBaseline


def test_m0_predicts_rate() -> None:
    """M0 computes historical switching rate."""
    # 100 renewals, 20 switches over 6 months
    model = HistoricalBaseline()
    model.fit(
        n_renewals=np.array([15, 18, 16, 17, 16, 18]),
        n_switches=np.array([3, 4, 3, 4, 3, 3]),
    )
    # Expected rate ~20/100 = 0.20
    assert abs(model.expected_rate - 0.20) < 0.05


def test_m0_beta_binomial_bounds() -> None:
    """Predictive distribution respects [0, N] bounds."""
    model = HistoricalBaseline()
    model.fit(
        n_renewals=np.array([100] * 6),
        n_switches=np.array([20] * 6),
    )
    # Predict for N=50 upcoming renewals
    dist = model.predict_counts(n_upcoming=50, n_samples=1000)
    assert (dist >= 0).all()
    assert (dist <= 50).all()
    # Mean should be ~10 (20% of 50)
    assert abs(dist.mean() - 10) < 2


def test_m0_jeffreys_prior() -> None:
    """Jeffreys Beta(0.5, 0.5) prior prevents zero-count collapse."""
    model = HistoricalBaseline()
    # Zero switches observed (edge case)
    model.fit(
        n_renewals=np.array([10] * 6),
        n_switches=np.array([0] * 6),
    )
    # Should not predict exactly 0 (prior smooths)
    assert model.expected_rate > 0
    assert model.expected_rate < 0.1


def test_m0_quantiles() -> None:
    """80% prediction interval from Beta-Binomial."""
    model = HistoricalBaseline()
    model.fit(
        n_renewals=np.array([100] * 6),
        n_switches=np.array([20] * 6),
    )
    q10, q50, q90 = model.predict_quantiles(n_upcoming=100)
    assert q10 <= q50 <= q90
    assert q10 >= 0
    assert q90 <= 100
