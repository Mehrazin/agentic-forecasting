"""Evaluation metrics: proper scoring rules for count/rate forecasts.

- Brier score: for binary outcomes (switched or not)
- CRPS: for count distributions (proper scoring rule)
- MAE/RMSE: point forecast accuracy
- Interval coverage: calibration of 80% PIs
"""

from __future__ import annotations

import numpy as np


def brier_score(
    prob_pred: np.ndarray,
    actual: np.ndarray,
) -> float:
    """Brier score for probabilistic binary predictions.

    BS = mean((p - y)^2). Lower is better. Strictly proper.

    Args:
        prob_pred: Predicted probabilities in [0, 1].
        actual: Binary outcomes (0/1).

    Returns:
        Mean Brier score.

    """
    p = np.asarray(prob_pred, dtype=float)
    y = np.asarray(actual, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {y.shape}")
    if bool((p < 0).any() or (p > 1).any()):
        raise ValueError("prob_pred must be in [0, 1]")
    if not bool(((y == 0) | (y == 1)).all()):
        raise ValueError("actual must be binary (0/1)")
    return float(np.mean((p - y) ** 2))


def mae(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(pred - actual)))


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def interval_coverage(
    q10: np.ndarray,
    q90: np.ndarray,
    actual: np.ndarray,
) -> float:
    """Empirical coverage of 80% prediction intervals.

    Args:
        q10: Lower bounds (10th percentile).
        q90: Upper bounds (90th percentile).
        actual: Realized values.

    Returns:
        Fraction inside [q10, q90]. Target: 0.80 ± 0.05.

    """
    inside = (actual >= q10) & (actual <= q90)
    return float(np.mean(inside))


def crps_counts(
    samples: np.ndarray,
    actual: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """CRPS for count predictive distributions.

    Uses the exact sample-based estimator:
        CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent draws from the predictive distribution.

    The pairwise term uses the exact O(M log M) sorted L-moment identity
    (no subsampling, no diagonal bias, seed-free):
        0.5 * E|X - X'| = sum_{i}(2i - M + 1) * s_(i) / M^2
    where s_(i) are the sorted samples.

    Args:
        samples: Shape (n_segments, n_samples) of predictive draws.
        actual: Shape (n_segments,) of realized counts.
        seed: Accepted for backward compatibility; ignored (deterministic).

    Returns:
        CRPS per segment (lower is better).

    """
    n_segments, n_samples = samples.shape
    # E|X - y|
    term1 = np.mean(np.abs(samples - actual[:, None]), axis=1)
    # 0.5 * E|X - X'| via exact sorted identity (no subsample, no diagonal).
    s = np.sort(samples, axis=1)
    m = float(n_samples)
    i = np.arange(n_samples, dtype=float)
    term2 = np.sum((2.0 * i - m + 1.0) * s, axis=1) / (m * m)
    return term1 - term2
