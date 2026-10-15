"""Tests for evaluation metrics (proper scoring rules)."""

import numpy as np

from td_one.evaluation.metrics import (
    brier_score,
    crps_counts,
    interval_coverage,
    mae,
)


def test_brier_perfect() -> None:
    """Brier = 0 for perfect prediction."""
    assert brier_score(np.array([1.0]), np.array([1])) == 0.0
    assert brier_score(np.array([0.0]), np.array([0])) == 0.0


def test_brier_worst() -> None:
    """Brier = 1 for perfectly wrong prediction."""
    assert brier_score(np.array([1.0]), np.array([0])) == 1.0


def test_mae() -> None:
    """MAE computes correctly."""
    pred = np.array([10.0, 20.0, 30.0])
    actual = np.array([12.0, 18.0, 33.0])
    assert abs(mae(pred, actual) - 2.333) < 0.01


def test_interval_coverage() -> None:
    """Coverage fraction computed correctly."""
    q10 = np.array([5.0, 10.0, 15.0])
    q90 = np.array([15.0, 20.0, 25.0])
    actual = np.array([10.0, 25.0, 20.0])  # 2/3 inside
    cov = interval_coverage(q10, q90, actual)
    assert abs(cov - 0.666) < 0.01


def test_crps_counts() -> None:
    """CRPS for count distributions (sanity)."""
    # Perfect prediction: all mass on true value
    samples = np.array([[10] * 100])  # 1 segment, 100 samples all = 10
    actual = np.array([10])
    score = crps_counts(samples, actual)
    assert score[0] < 0.1  # Near zero for perfect
