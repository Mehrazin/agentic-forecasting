"""Hybrid readout: combines numerical backend with LLM contextual adjustment.

The agent's evidence-grounded log-odds shift Δ (bounded to [-0.75, +0.75])
is applied to the base rate forecast:
    logit(p_adj) = logit(p_base) + Δ
    p_adj = sigmoid(logit(p_adj))

Then counts: A ~ BetaBinomial(N, α, β) where α, β from p_adj.

The bound prevents runaway hallucinated adjustments (literature finding).
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import logit


class HybridReadout:
    """Combines TSFM/baseline rate with bounded LLM adjustment."""

    # Bound from plan §4.3 and literature (prevents calibration destruction)
    MAX_DELTA = 0.75

    def apply_adjustment(
        self,
        base_rate: float,
        delta_logit: float,
    ) -> float:
        """Apply bounded log-odds shift to base rate.

        Args:
            base_rate: Base predicted switching rate in [0, 1].
                Degenerate 0/1 values are epsilon-clipped, not rejected —
                the numerical backend may legitimately emit them.
            delta_logit: LLM's contextual adjustment (clipped to ±0.75).

        Returns:
            Adjusted rate in (0, 1).

        """
        # Clip delta (prevents runaway adjustments)
        delta = float(np.clip(delta_logit, -self.MAX_DELTA, self.MAX_DELTA))
        # Epsilon-clip for numerical stability (logit undefined at 0/1)
        base_clipped = float(np.clip(base_rate, 1e-6, 1 - 1e-6))
        adjusted_logit = logit(base_clipped) + delta
        return float(sigmoid(adjusted_logit))

    def predict_counts(
        self,
        base_rate: float,
        delta_logit: float,
        n_upcoming: int,
        n_samples: int = 1000,
        seed: int = 42,
        concentration: float | None = None,
    ) -> np.ndarray:
        """Sample switch counts with adjustment.

        Args:
            base_rate: Base rate from numerical backend.
            delta_logit: LLM adjustment.
            n_upcoming: Number of renewals.
            n_samples: Monte Carlo samples.
            seed: Random seed.
            concentration: Beta concentration (alpha+beta). If None, derived
                from the base rate's implied precision — callers doing
                ablations should pass the backend's own concentration so
                interval widths stay comparable across rungs.

        Returns:
            Array of counts in [0, N].

        """
        adjusted = self.apply_adjustment(base_rate, delta_logit)
        # Default concentration keeps the posterior modestly informative;
        # ablations should override with the backend's concentration.
        conc = 20.0 if concentration is None else float(concentration)
        alpha = adjusted * conc
        beta = (1.0 - adjusted) * conc
        rng = np.random.default_rng(seed)
        p_samples = rng.beta(alpha, beta, size=n_samples)
        counts = rng.binomial(n_upcoming, p_samples)
        return counts
