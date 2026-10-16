"""Shrinkage and regime gating for the M3 hybrid readout.

M3's observed failure modes (exp4, real Groq): the agent's delta is
directionally right ~4/6 but (a) occasionally gets the sign wrong,
(b) spuriously adjusts benign scenarios, (c) undershoots magnitude.
The ±0.75 bound is necessary but not sufficient — these mechanisms add
a "do nothing" prior.

1. :class:`ShrunkHybridReadout` — empirical-Bayes-style shrinkage of the
   proposed delta toward 0: ``delta' = k * delta`` with ``k`` in [0, 1]
   tuned on data. ``k = 1`` recovers the parent; ``k = 0`` is pure M0.
   Shrinkage is applied *before* the ±0.75 clip (via the parent's
   ``apply_adjustment``), so a shrunk delta can never exceed the bound.
2. :func:`gate_delta` — a two-stage rule. A two-proportion z-test
   (:func:`regime_z_stat`) compares the recent-window switching rate
   against the trailing rate; if no significant shift is detected
   (``|z| < threshold``) the delta is forced to 0 no matter what the
   agent proposed. This directly targets the benign-scenario failure
   (spurious +0.30 on true 0.00).
"""

from __future__ import annotations

import numpy as np

from td_one.models.hybrid import HybridReadout


class ShrunkHybridReadout(HybridReadout):
    """HybridReadout with the agent delta shrunk toward zero.

    Args:
        shrinkage: Shrinkage factor ``k`` in [0, 1]. The proposed delta
            is multiplied by ``k`` before the ±0.75 bound is applied.

    Raises:
        ValueError: If ``shrinkage`` is outside [0, 1].

    """

    def __init__(self, shrinkage: float = 1.0) -> None:
        """Initialize with a shrinkage factor."""
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")
        self.shrinkage = float(shrinkage)

    def apply_adjustment(self, base_rate: float, delta_logit: float) -> float:
        """Apply the shrunk, bounded log-odds shift to the base rate."""
        shrunk = float(delta_logit) * self.shrinkage
        return super().apply_adjustment(base_rate, shrunk)


def regime_z_stat(
    n_renewals: np.ndarray,
    n_switches: np.ndarray,
    recent_months: int = 3,
) -> float:
    """Two-proportion z-statistic: recent window vs trailing history.

    Tests whether the switching rate in the last ``recent_months``
    periods differs from the rate in all earlier periods. Positive z
    means the recent rate is higher.

    Returns 0.0 (no evidence) when the panel is too short, either side
    has no renewals, or the standard error is degenerate.

    Args:
        n_renewals: Renewal counts per period.
        n_switches: Switch counts per period.
        recent_months: Size of the recent window.

    Returns:
        The z-statistic (float).

    """
    n = np.asarray(n_renewals, dtype=float)
    a = np.asarray(n_switches, dtype=float)
    if recent_months < 1 or n.size < recent_months + 1:
        return 0.0
    n1 = float(n[-recent_months:].sum())
    a1 = float(a[-recent_months:].sum())
    n0 = float(n[:-recent_months].sum())
    a0 = float(a[:-recent_months].sum())
    if n1 <= 0 or n0 <= 0:
        return 0.0
    p1 = a1 / n1
    p0 = a0 / n0
    p_pool = (a1 + a0) / (n1 + n0)
    se = float(np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n0)))
    if se <= 0:
        return 0.0
    return float((p1 - p0) / se)


def gate_delta(
    delta: float,
    n_renewals: np.ndarray,
    n_switches: np.ndarray,
    threshold: float = 2.0,
    recent_months: int = 3,
) -> float:
    """Force the delta to 0 unless the panel shows a regime shift.

    If ``|regime_z_stat| < threshold`` the agent's delta is discarded
    (returned as 0.0); otherwise it passes through unchanged. The gate
    sees only the panel — never the scenario label — so it cannot leak.

    Args:
        delta: The agent's proposed log-odds adjustment.
        n_renewals: Renewal counts per period.
        n_switches: Switch counts per period.
        threshold: Minimum |z| to let the delta through.
        recent_months: Size of the recent window for the z-test.

    Returns:
        ``delta`` if a shift is detected, else 0.0.

    """
    if abs(regime_z_stat(n_renewals, n_switches, recent_months)) < threshold:
        return 0.0
    return float(delta)
