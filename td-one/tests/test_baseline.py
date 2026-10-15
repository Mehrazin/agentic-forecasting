"""Tests for M0 historical baseline with Beta-Binomial."""

import numpy as np
import pytest

from td_one.models.baseline import HistoricalBaseline


def test_m0_predicts_rate() -> None:
    """M0 computes the trailing-window switching rate."""
    # 100 renewals, 20 switches over 6 months
    model = HistoricalBaseline()
    model.fit(
        n_renewals=np.array([15, 18, 16, 17, 16, 18]),
        n_switches=np.array([3, 4, 3, 4, 3, 3]),
    )
    # Posterior mean: (20 + 0.5) / (100 + 1) = 0.2030
    assert abs(model.expected_rate - 0.2030) < 0.005


def test_m0_trailing_window() -> None:
    """M0 pools only the trailing window (matches its docstring).

    Regression test: the old code pooled all 12 periods despite the
    docstring promising a trailing 6-month window.
    """
    model = HistoricalBaseline(trailing_window=6)
    # First 6 periods at 10%, last 6 at 30%
    model.fit(
        n_renewals=np.array([100] * 12),
        n_switches=np.array([10] * 6 + [30] * 6),
    )
    # Trailing-6 posterior: (180 + 0.5) / (600 + 1) = 0.3002
    assert abs(model.expected_rate - 0.3002) < 0.005


def test_m0_unfitted_raises() -> None:
    """Unfitted model raises (error-path coverage per repo philosophy)."""
    model = HistoricalBaseline()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_quantiles(n_upcoming=10)
    with pytest.raises(RuntimeError, match="fit"):
        _ = model.expected_rate


def test_m0_rejects_invalid_counts() -> None:
    """A > N or misaligned inputs raise."""
    model = HistoricalBaseline()
    with pytest.raises(ValueError, match="0 <= A <= N"):
        model.fit(np.array([10, 10]), np.array([3, 15]))
    with pytest.raises(ValueError, match="non-empty"):
        model.fit(np.array([]), np.array([]))


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
