"""Classical machinery for applying regime readings (M4).

The LLM's :class:`RegimeReading` is treated as a noisy sensor. The M4
forecast starts from M0 and intervenes only when two independent
confirmations agree:

1. **LLM confidence**: ``reading.confidence >= confidence_threshold``.
   Low-confidence readings are discarded (abstention → M0).
2. **Panel evidence** (optional): ``|regime_z_stat| >= z_threshold``.
   The panel's own recent history must corroborate a shift.

If either check fails — or the reading is ``NONE`` — the forecast is
exactly M0. When both pass, the data-calibrated class delta
(``CLASS_DELTAS``, magnitudes from the exp2 fixed-grid probe, not from
the LLM) is applied through the bounded hybrid readout.

The LLM never touches a number: it supplies a class and a confidence;
every threshold, delta, and forecast lives here, in tested code.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from td_one.agent.regime_reader import REGIMES, RegimeClass, RegimeReading
from td_one.evaluation.metrics import crps_counts
from td_one.models.baseline import HistoricalBaseline
from td_one.models.hybrid import HybridReadout
from td_one.models.shrinkage import regime_z_stat

#: Data-calibrated deltas per predicted class. Magnitudes come from the
#: exp2 fixed-grid probe's best point (+0.25), symmetric for DOWN, not
#: from any LLM judgment. NONE -> hard zero (the "do nothing" prior).
#: ABSTAIN -> hard zero (mechanism unclear; never indexed in practice
#: because should_intervene blocks it, but defined for completeness).
CLASS_DELTAS: dict[str, float] = {
    "UP": 0.25,
    "DOWN": -0.25,
    "NONE": 0.0,
    "ABSTAIN": 0.0,
}


class M4Forecast(TypedDict):
    """M4 forecast pieces for one scenario."""

    base_rate: float
    reading: RegimeReading
    intervened: bool
    delta: float
    adjusted_rate: float
    samples: np.ndarray
    q10: float
    q50: float
    q90: float


def should_intervene(
    reading: RegimeReading,
    n_renewals: np.ndarray,
    n_switches: np.ndarray,
    confidence_threshold: float = 0.7,
    z_threshold: float = 1.25,
    use_panel_gate: bool = True,
) -> bool:
    """Decide whether a regime reading justifies intervening on M0.

    Intervention requires the LLM to be confident *and* (optionally)
    the panel to corroborate a shift. ``NONE`` and ``ABSTAIN`` readings
    never intervene — they are explicit "do nothing" signals (NONE =
    confident no-change, ABSTAIN = mechanism unclear).

    Args:
        reading: The LLM's regime reading.
        n_renewals: Renewal counts per period.
        n_switches: Switch counts per period.
        confidence_threshold: Minimum LLM confidence to act.
        z_threshold: Minimum |z| for the panel gate.
        use_panel_gate: If False, skip the panel corroboration check.

    Returns:
        True if the forecast should be adjusted, else False.

    """
    if reading.regime in ("NONE", "ABSTAIN"):
        return False
    if reading.confidence < confidence_threshold:
        return False
    if use_panel_gate:
        z = regime_z_stat(n_renewals, n_switches)
        if abs(z) < z_threshold:
            return False
    return True


def apply_regime_reading(
    n_renewals: np.ndarray,
    n_switches: np.ndarray,
    n_upcoming: int,
    reading: RegimeReading,
    confidence_threshold: float = 0.7,
    z_threshold: float = 1.25,
    use_panel_gate: bool = True,
    seed: int = 42,
) -> M4Forecast:
    """Apply a regime reading to an M0 forecast (M4).

    Fits M0 on the panel; intervenes with the data-calibrated class
    delta only if :func:`should_intervene` passes. Otherwise returns
    the M0 forecast unchanged.

    Args:
        n_renewals: Renewal counts per period.
        n_switches: Switch counts per period.
        n_upcoming: Upcoming renewals in the test period.
        reading: The LLM's regime reading.
        confidence_threshold: Minimum LLM confidence to act.
        z_threshold: Minimum |z| for the panel gate.
        use_panel_gate: If False, skip the panel corroboration check.
        seed: Random seed for posterior sampling.

    Returns:
        The M4 forecast pieces.

    """
    if reading.regime not in REGIMES:
        raise ValueError(f"unknown regime {reading.regime!r}")
    m0 = HistoricalBaseline()
    m0.fit(n_renewals, n_switches)
    base_rate = m0.expected_rate
    readout = HybridReadout()

    intervened = should_intervene(
        reading,
        n_renewals,
        n_switches,
        confidence_threshold=confidence_threshold,
        z_threshold=z_threshold,
        use_panel_gate=use_panel_gate,
    )
    delta = CLASS_DELTAS[reading.regime] if intervened else 0.0
    adjusted = readout.apply_adjustment(base_rate, delta)
    concentration = 1.0 + float(np.asarray(n_renewals)[-6:].sum())
    samples = readout.predict_counts(
        base_rate,
        delta,
        n_upcoming,
        concentration=concentration,
        seed=seed,
    )
    return M4Forecast(
        base_rate=base_rate,
        reading=reading,
        intervened=intervened,
        delta=delta,
        adjusted_rate=adjusted,
        samples=samples,
        q10=float(np.quantile(samples, 0.10)),
        q50=float(np.quantile(samples, 0.50)),
        q90=float(np.quantile(samples, 0.90)),
    )


def m4_crps(
    forecast: M4Forecast,
    actual: int,
) -> float:
    """CRPS of an M4 forecast against an observed switch count."""
    samples = np.asarray(forecast["samples"])
    return float(crps_counts(samples[None, :], np.array([actual]))[0])


def _regime_class_for_shift(true_shift: float) -> RegimeClass:
    """Map a true logit shift to its regime class (for mock experiments)."""
    if true_shift > 0.05:
        return "UP"
    if true_shift < -0.05:
        return "DOWN"
    return "NONE"
