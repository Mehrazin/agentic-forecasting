"""Exponentially-weighted M0: recency-weighted Beta-Binomial.

Replaces the hard trailing window with exponential decay: month i
(counting back from the most recent) gets weight
``0.5 ** (i / half_life)``. Same Jeffreys Beta-Binomial machinery as
M0, but recent months count more -- a smooth alternative to choosing
a window cutoff.

In the limit half_life -> inf the weights are uniform (plain pooling);
as half_life -> 0 only the most recent month matters.
"""

from __future__ import annotations

import numpy as np

from td_one.models.baseline import HistoricalBaseline


def decay_weights(n_periods: int, half_life: float) -> np.ndarray:
    """Exponential decay weights, most-recent-last.

    Args:
        n_periods: Number of trailing periods.
        half_life: Periods for the weight to halve; must be positive.

    Returns:
        Weights in (0, 1], with the last entry exactly 1.0.

    """
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1")
    ages = np.arange(n_periods - 1, -1, -1, dtype=float)  # 0 for most recent
    return np.power(0.5, ages / half_life)


class EWMHistoricalBaseline(HistoricalBaseline):
    """M0 with exponential-decay weighting instead of a hard window."""

    def __init__(self, half_life: float = 2.0) -> None:
        """Initialize unfitted EWM baseline.

        Args:
            half_life: Months for a period's weight to halve.

        """
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        # Bypass HistoricalBaseline.__init__ (no trailing window here).
        self.half_life = half_life
        self.trailing_window = 0  # unused; fit() ignores it
        self._alpha: float | None = None
        self._beta: float | None = None
        self._n_periods = 0
        self._fitted = False

    def fit(
        self,
        n_renewals: np.ndarray,
        n_switches: np.ndarray,
    ) -> None:
        """Fit with decay-weighted effective counts.

        Effective counts: A_eff = sum(w * A), N_eff = sum(w * N), then
        alpha = 0.5 + A_eff, beta = 0.5 + (N_eff - A_eff) as in M0.

        Args:
            n_renewals: Renewal counts per period (oldest first).
            n_switches: Switch counts per period (oldest first).

        """
        n = np.asarray(n_renewals, dtype=float)
        a = np.asarray(n_switches, dtype=float)
        if n.shape != a.shape or n.size == 0:
            raise ValueError("n_renewals and n_switches must be non-empty and aligned")
        if bool((a < 0).any() or (a > n).any()):
            raise ValueError("switch counts must satisfy 0 <= A <= N")
        w = decay_weights(n.size, self.half_life)
        n_eff = float((w * n).sum())
        a_eff = float((w * a).sum())
        self._alpha = 0.5 + a_eff
        self._beta = 0.5 + (n_eff - a_eff)
        self._n_periods = n.size
        self._fitted = True

    @property
    def effective_periods(self) -> float:
        """Sum of decay weights over the fitted history (effective months)."""
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        return float(decay_weights(self._n_periods, self.half_life).sum())
