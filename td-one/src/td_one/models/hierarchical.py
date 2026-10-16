"""Hierarchical M0: partial pooling across segments.

Each segment's switching rate is shrunk toward a pooled across-segment
mean. The shared Beta prior is estimated empirically (method of moments
on trailing-window segment rates, net of binomial sampling noise), so
small segments borrow strength while large segments dominate their own
posterior.

Posterior for segment s (trailing window w):
    alpha_s = a_prior + A_s[-w:].sum()
    beta_s  = b_prior + (N_s[-w:] - A_s[-w:]).sum()
with (a_prior, b_prior) from the moment-matched Beta(mu, v_between).

If no between-segment heterogeneity is detectable (v_between <= 0),
falls back to the Jeffreys prior -- i.e. plain M0.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from td_one.models.baseline import TRAILING_WINDOW

#: Floor on the between-segment variance to avoid degenerate priors.
_MIN_BETWEEN_VAR = 1e-12


class HierarchicalBaseline:
    """Partial-pooling Beta-Binomial forecaster over segment panels."""

    def __init__(self, trailing_window: int = TRAILING_WINDOW) -> None:
        """Initialize unfitted hierarchical baseline.

        Args:
            trailing_window: Number of most recent periods pooled per segment.

        """
        if trailing_window < 1:
            raise ValueError("trailing_window must be >= 1")
        self.trailing_window = trailing_window
        self._prior_a: float | None = None
        self._prior_b: float | None = None
        self._segments: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._fitted = False

    @property
    def prior_params(self) -> tuple[float, float]:
        """The empirical-Bayes (a, b) of the shared Beta prior."""
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        assert self._prior_a is not None and self._prior_b is not None
        return self._prior_a, self._prior_b

    @property
    def pooled_rate(self) -> float:
        """Exposure-weighted mean rate across segments (prior mean)."""
        a, b = self.prior_params
        return a / (a + b)

    def fit(self, segments: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Estimate the shared prior and store per-segment trailing counts.

        Args:
            segments: Mapping segment id -> (n_renewals, n_switches)
                arrays (full history; the trailing window is sliced here).

        Raises:
            ValueError: On empty input, misaligned arrays, or any
                segment violating 0 <= A <= N.

        """
        if not segments:
            raise ValueError("segments must be non-empty")
        w = self.trailing_window
        rates: list[float] = []
        exposures: list[float] = []
        total_a = 0.0
        total_n = 0.0
        cleaned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for seg, (n_raw, a_raw) in segments.items():
            n = np.asarray(n_raw, dtype=float)
            a = np.asarray(a_raw, dtype=float)
            if n.shape != a.shape or n.size == 0:
                raise ValueError(f"segment {seg!r}: n/a must be non-empty and aligned")
            if bool((a < 0).any() or (a > n).any()):
                raise ValueError(f"segment {seg!r}: must satisfy 0 <= A <= N")
            n_w, a_w = n[-w:], a[-w:]
            n_sum = float(n_w.sum())
            a_sum = float(a_w.sum())
            if n_sum <= 0:
                raise ValueError(f"segment {seg!r}: trailing window has no renewals")
            cleaned[seg] = (n_w, a_w)
            rates.append(a_sum / n_sum)
            exposures.append(n_sum)
            total_a += a_sum
            total_n += n_sum

        # Empirical-Bayes prior via method of moments, net of sampling noise.
        mu = total_a / total_n
        r = np.asarray(rates)
        e = np.asarray(exposures)
        v_total = float(np.sum(e * (r - mu) ** 2) / np.sum(e))
        # Expected sampling variance under homogeneity: sum_s n_s * mu(1-mu)/n_s / sum n_s
        v_sampling = mu * (1.0 - mu) * len(r) / float(np.sum(e))
        v_between = v_total - v_sampling
        if v_between <= _MIN_BETWEEN_VAR:
            # No detectable heterogeneity: Jeffreys prior (plain M0).
            self._prior_a, self._prior_b = 0.5, 0.5
        else:
            concentration = mu * (1.0 - mu) / v_between - 1.0
            concentration = max(concentration, _MIN_BETWEEN_VAR)
            self._prior_a = mu * concentration
            self._prior_b = (1.0 - mu) * concentration
        self._segments = cleaned
        self._fitted = True

    def _posterior(self, segment_id: str) -> tuple[float, float]:
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        assert self._prior_a is not None and self._prior_b is not None
        try:
            n_w, a_w = self._segments[segment_id]
        except KeyError as exc:
            raise KeyError(f"unknown segment {segment_id!r}") from exc
        return self._prior_a + float(a_w.sum()), self._prior_b + float((n_w - a_w).sum())

    def expected_rate(self, segment_id: str) -> float:
        """Posterior mean switching rate for a segment."""
        alpha, beta = self._posterior(segment_id)
        return alpha / (alpha + beta)

    def shrinkage(self, segment_id: str) -> float:
        """Fraction of the posterior mean coming from the pooled prior.

        0 = segment data dominates; 1 = fully shrunk to the pooled mean.
        """
        alpha, beta = self._posterior(segment_id)
        assert self._prior_a is not None and self._prior_b is not None
        prior_n = self._prior_a + self._prior_b
        return prior_n / (alpha + beta)

    def predict_rate_quantiles(
        self,
        segment_id: str,
        n_upcoming: int,
        levels: np.ndarray,
    ) -> dict[float, float]:
        """Posterior predictive rate quantiles for one segment.

        Args:
            segment_id: Segment fitted in :meth:`fit`.
            n_upcoming: Upcoming renewals (exposure, >= 1).
            levels: Quantile levels in (0, 1).

        Returns:
            Mapping level -> rate quantile (monotone by construction).

        """
        if n_upcoming < 1:
            raise ValueError("n_upcoming must be >= 1")
        alpha, beta = self._posterior(segment_id)
        lv = np.asarray(levels, dtype=float)
        if bool((lv <= 0.0).any() or (lv >= 1.0).any()):
            raise ValueError("levels must be in (0, 1)")
        dist = stats.betabinom(n_upcoming, alpha, beta)
        counts = dist.ppf(lv)
        rates = np.clip(counts / n_upcoming, 0.0, 1.0)
        return {float(level): float(r) for level, r in zip(lv, rates, strict=True)}
