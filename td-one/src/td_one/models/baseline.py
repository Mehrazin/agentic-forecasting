"""M0: Historical-rate baseline with Jeffreys Beta-Binomial.

The simplest model: trailing 6-month empirical switching rate with a
Jeffreys Beta(0.5, 0.5) prior. Predictive distribution for counts is
Beta-Binomial, guaranteeing 0 <= A <= N coherence.

This is the K=0 non-agentic baseline in the ablation ladder.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


class HistoricalBaseline:
    """Trailing historical switching rate forecaster."""

    def __init__(self) -> None:
        """Initialize unfitted baseline."""
        self._alpha: float | None = None
        self._beta: float | None = None
        self._fitted = False

    def fit(
        self,
        n_renewals: np.ndarray,
        n_switches: np.ndarray,
    ) -> None:
        """Fit on historical (N, A) pairs.

        Uses Jeffreys prior Beta(0.5, 0.5):
            alpha = 0.5 + sum(A)
            beta = 0.5 + sum(N - A)

        Args:
            n_renewals: Array of renewal counts per period.
            n_switches: Array of switch counts per period.
        """
        total_n = int(n_renewals.sum())
        total_a = int(n_switches.sum())
        # Jeffreys prior prevents collapse on zero counts
        self._alpha = 0.5 + total_a
        self._beta = 0.5 + (total_n - total_a)
        self._fitted = True

    @property
    def expected_rate(self) -> float:
        """Posterior mean switching rate."""
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        assert self._alpha is not None and self._beta is not None
        return self._alpha / (self._alpha + self._beta)

    def predict_counts(
        self,
        n_upcoming: int,
        n_samples: int = 1000,
        seed: int = 42,
    ) -> np.ndarray:
        """Sample from Beta-Binomial predictive distribution.

        Args:
            n_upcoming: Number of upcoming renewals (N).
            n_samples: Number of Monte Carlo samples.
            seed: Random seed.

        Returns:
            Array of sampled switch counts (each in [0, N]).
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        assert self._alpha is not None and self._beta is not None
        rng = np.random.default_rng(seed)
        # Beta-Binomial: sample p ~ Beta(alpha, beta), then A ~ Binomial(N, p)
        p_samples = rng.beta(self._alpha, self._beta, size=n_samples)
        counts = rng.binomial(n_upcoming, p_samples)
        return counts

    def predict_quantiles(
        self,
        n_upcoming: int,
    ) -> tuple[float, float, float]:
        """80% prediction interval (q10, q50, q90) for counts.

        Args:
            n_upcoming: Number of upcoming renewals.

        Returns:
            (q10, q50, q90) quantiles.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        assert self._alpha is not None and self._beta is not None
        # Analytical quantiles via scipy
        dist = stats.betabinom(n_upcoming, self._alpha, self._beta)
        return (
            float(dist.ppf(0.10)),
            float(dist.ppf(0.50)),
            float(dist.ppf(0.90)),
        )
