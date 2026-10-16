"""Classical M0 ensemble: average of quantile lattices.

Combines component rate-quantile lattices (M0 at several windows plus
the EWM variant) by weighted averaging at each level. Averaging
non-decreasing quantile functions preserves monotonicity, so the
ensemble lattice is valid by construction.

Two weighting schemes:
- equal weights (the default; no tuning, no lookahead);
- CRPS weights: inverse trailing-origin CRPS, updated per origin by the
  caller via :meth:`set_weights` (weights always normalized to sum 1).
"""

from __future__ import annotations

import numpy as np


def average_lattices(
    lattices: list[dict[float, float]],
    weights: list[float] | None = None,
) -> dict[float, float]:
    """Weighted average of rate-quantile lattices.

    Args:
        lattices: Non-empty list of {level: rate} mappings sharing the
            same level grid.
        weights: Optional non-negative weights (normalized internally).
            Defaults to equal weights.

    Returns:
        Averaged lattice on the shared grid (monotone if inputs are).

    Raises:
        ValueError: On empty input, mismatched grids, or invalid weights.

    """
    if not lattices:
        raise ValueError("lattices must be non-empty")
    ref = sorted(lattices[0].keys())
    for lat in lattices[1:]:
        if sorted(lat.keys()) != ref:
            raise ValueError("all lattices must share the same level grid")
    w = np.ones(len(lattices)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(lattices),) or bool((w < 0).any()) or not bool((w > 0).any()):
        raise ValueError("weights must be non-negative with at least one positive")
    w = w / w.sum()
    out: dict[float, float] = {}
    for level in ref:
        vals = np.array([lat[level] for lat in lattices])
        out[float(level)] = float(np.clip((w * vals).sum(), 0.0, 1.0))
    return out


def crps_weights(recent_crps: list[list[float]], floor: float = 1e-6) -> list[float]:
    """Inverse-CRPS weights from trailing-origin scores.

    Args:
        recent_crps: Per-component lists of recent per-origin mean CRPS
            (most recent last); each list non-empty.
        floor: Minimum CRPS to avoid division blowups.

    Returns:
        Normalized weights; lower recent CRPS -> higher weight.

    """
    if not recent_crps or any(len(c) == 0 for c in recent_crps):
        raise ValueError("recent_crps needs a non-empty history per component")
    means = np.array([float(np.mean(c)) for c in recent_crps])
    inv = 1.0 / np.maximum(means, floor)
    return [float(x) for x in inv / inv.sum()]


class LatticeEnsemble:
    """Weighted-average ensemble over component forecast functions."""

    def __init__(self, weights: list[float] | None = None) -> None:
        """Initialize with optional fixed weights (equal if None)."""
        self._weights = None if weights is None else [float(x) for x in weights]
        if self._weights is not None:
            w = np.asarray(self._weights)
            if bool((w < 0).any()) or not bool((w > 0).any()):
                raise ValueError("weights must be non-negative with a positive sum")

    def set_weights(self, weights: list[float]) -> None:
        """Update weights (e.g. from trailing-origin CRPS)."""
        w = np.asarray([float(x) for x in weights], dtype=float)
        if bool((w < 0).any()) or not bool((w > 0).any()):
            raise ValueError("weights must be non-negative with a positive sum")
        self._weights = [float(x) for x in w]

    @property
    def weights(self) -> list[float] | None:
        """Current weights (None = equal weighting)."""
        return None if self._weights is None else list(self._weights)

    def combine(self, lattices: list[dict[float, float]]) -> dict[float, float]:
        """Average component lattices with the current weights."""
        if self._weights is not None and len(self._weights) != len(lattices):
            raise ValueError(f"{len(self._weights)} weights for {len(lattices)} lattices")
        return average_lattices(lattices, self._weights)
