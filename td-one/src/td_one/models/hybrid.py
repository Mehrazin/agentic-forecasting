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
from scipy import stats
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
            base_rate: Base predicted switching rate in (0, 1).
            delta_logit: LLM's contextual adjustment (clipped to ±0.75).

        Returns:
            Adjusted rate in (0, 1).
        """
        if not 0 < base_rate < 1:
            raise ValueError(f"base_rate must be in (0, 1), got {base_rate}")
        # Clip delta (prevents runaway adjustments)
        delta = float(np.clip(delta_logit, -self.MAX_DELTA, self.MAX_DELTA))
        # Logit -> shift -> sigmoid
        # Clip base_rate away from 0/1 for numerical stability
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
    ) -> np.ndarray:
        """Sample switch counts with adjustment.

        Uses Beta-Binomial with mean = adjusted rate. For simplicity,
        we use a fixed concentration (alpha+beta=20) — in production,
        this would come from the TSFM's uncertainty.

        Args:
            base_rate: Base rate from numerical backend.
            delta_logit: LLM adjustment.
            n_upcoming: Number of renewals.
            n_samples: Monte Carlo samples.
            seed: Random seed.

        Returns:
            Array of counts in [0, N].
        """
        adjusted = self.apply_adjustment(base_rate, delta_logit)
        # Beta parameters: mean=adjusted, concentration=20
        # alpha = mean * conc, beta = (1-mean) * conc
        conc = 20.0
        alpha = adjusted * conc
        beta = (1.0 - adjusted) * conc
        rng = np.random.default_rng(seed)
        p_samples = rng.beta(alpha, beta, size=n_samples)
        counts = rng.binomial(n_upcoming, p_samples)
        return counts
