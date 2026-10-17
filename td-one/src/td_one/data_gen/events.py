"""Exogenous event timeline generator for the event-augmented simulator.

Events are external shocks (policy changes, rate surprises, housing corrections,
lender campaigns, macro surprises) that affect mortgage outcomes through causal
channels. Unlike the simulator's "planted shocks" (conditional branches in the
DGP code), these are genuine exogenous perturbations: they arrive at random
times, affect heterogeneous cohorts, and are observable through a noisy news
feed — creating the information asymmetry that justifies agentic methods.

PIT safety: the full event timeline is generated FIRST, before any outcomes.
Downstream code (simulator, news synthesis) must only reference events with
``event_date <= asof_date``. The ``events_visible_asof`` helper enforces this.

Event types (matching AGENTIC_ROADMAP.md Part 2):
1. POLICY_SHOCK — OSFI stress-test changes; affects p(switch) for uninsured
   borrowers after a 2-month implementation lag.
2. RATE_SHIFT — BoC surprise decisions (off-calendar); affects rate_spread for
   borrowers resetting within 3 months; 2-4 week transmission to mortgage rates.
3. HOUSING_DISRUPTION — Regional price drops; affects LTV → p(switch) through
   the equity channel; news delayed ~1 month with survivorship bias.
4. LENDER_SHIFT — Bank retention campaign; affects p(switch) for that bank's
   borrowers only; asymmetric news coverage (may get zero articles).
5. MACRO_SURPRISE — Unemployment spike in a sector/region; affects
   payment_shock through income loss; DGP effect immediate, news delayed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

import numpy as np


class EventType(StrEnum):
    """The five exogenous event types."""

    POLICY_SHOCK = "policy_shock"
    RATE_SHIFT = "rate_shift"
    HOUSING_DISRUPTION = "housing_disruption"
    LENDER_SHIFT = "lender_shift"
    MACRO_SURPRISE = "macro_surprise"


#: Default transmission lags (months) from event_date to DGP effect onset.
DEFAULT_TRANSMISSION_LAG: dict[EventType, int] = {
    EventType.POLICY_SHOCK: 2,
    EventType.RATE_SHIFT: 1,  # 2-4 weeks ≈ 1 month at monthly granularity
    EventType.HOUSING_DISRUPTION: 1,
    EventType.LENDER_SHIFT: 0,  # immediate for the bank's borrowers
    EventType.MACRO_SURPRISE: 0,  # immediate income effect
}

#: How long the event's effect persists (months) after onset.
DEFAULT_EFFECT_DURATION: dict[EventType, int] = {
    EventType.POLICY_SHOCK: 12,
    EventType.RATE_SHIFT: 6,
    EventType.HOUSING_DISRUPTION: 9,
    EventType.LENDER_SHIFT: 4,
    EventType.MACRO_SURPRISE: 6,
}

#: Valid cohort selectors per event type.
VALID_COHORTS: dict[EventType, tuple[str, ...]] = {
    EventType.POLICY_SHOCK: ("uninsured_720plus", "uninsured_all", "all_borrowers"),
    EventType.RATE_SHIFT: ("all_borrowers", "variable_rate", "resetting_3mo"),
    EventType.HOUSING_DISRUPTION: (
        "toronto_condo",
        "vancouver_detached",
        "alberta_all",
        "ontario_all",
    ),
    EventType.LENDER_SHIFT: ("bank_td", "bank_rbc", "bank_bmo", "bank_scotia"),
    EventType.MACRO_SURPRISE: (
        "ontario_manufacturing",
        "alberta_energy",
        "bc_tech",
        "quebec_all",
    ),
}


@dataclass(frozen=True)
class Event:
    """A single exogenous event.

    Attributes:
        event_id: Unique identifier (e.g. "EVT-0007").
        event_type: One of the five EventType values.
        event_date: Month-end date when the event occurs (the "as of" for PIT).
        magnitude: Signed size of the shock in natural units:
            - POLICY_SHOCK: pp change in stress-test qualifying rate.
            - RATE_SHIFT: pp change in BoC policy rate (signed).
            - HOUSING_DISRUPTION: fractional price drop (positive = drop).
            - LENDER_SHIFT: retention campaign intensity 0-1.
            - MACRO_SURPRISE: pp increase in regional unemployment.
        affected_cohort: Cohort selector string (see VALID_COHORTS).
        transmission_lag_months: Months from event_date to DGP effect onset.
        effect_duration_months: Months the effect persists after onset.
        effect_size: Logit-scale shift applied to p(switch) for affected
            borrowers during the effect window (signed).
        metadata: Free-form dict for type-specific details (not used by DGP).

    """

    event_id: str
    event_type: EventType
    event_date: date
    magnitude: float
    affected_cohort: str
    transmission_lag_months: int
    effect_duration_months: int
    effect_size: float
    metadata: dict[str, object] = field(default_factory=dict)

    def effect_window(self) -> tuple[date, date]:
        """Return (onset_date, end_date) of the DGP effect window.

        Both are month-end dates. The effect applies to maturities with
        ``onset_date <= maturity_month_end <= end_date``.
        """
        onset = _add_months(self.event_date, self.transmission_lag_months)
        end = _add_months(onset, self.effect_duration_months - 1)
        return onset, end

    def is_active(self, asof: date) -> bool:
        """Whether the event's DGP effect is active at the given month-end date."""
        onset, end = self.effect_window()
        return onset <= asof <= end


def _add_months(d: date, n: int) -> date:
    """Add n months to a date, clamping to month-end."""
    total = (d.year * 12 + d.month - 1) + n
    y, m = divmod(total, 12)
    m += 1
    # Last day of target month
    if m == 12:
        last = 31
    elif m in (4, 6, 9, 11):
        last = 30
    elif m == 2:
        leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
        last = 29 if leap else 28
    else:
        last = 31
    return date(y, m, min(d.day, last))


def _last_day(year: int, month: int) -> date:
    """Last calendar day of the given month."""
    if month == 12:
        return date(year, 12, 31)
    return date.fromordinal(date(year, month + 1, 1).toordinal() - 1)


def generate_event_timeline(
    start_date: date,
    n_months: int,
    seed: int = 42,
    event_rate_per_year: float = 3.0,
) -> list[Event]:
    """Generate a random exogenous event timeline.

    Events are drawn from a Poisson process (``event_rate_per_year`` expected
    events per 12 months), each assigned a random type, magnitude, cohort, and
    effect size. Magnitudes are calibrated so that a single event moves the
    affected cohort's switch rate by roughly 3-10pp — large enough to be
    detectable by an oracle, small enough that a naive model won't trivially
    pick it up from the aggregate series alone.

    Args:
        start_date: First month of the timeline (any day; snapped to month-end).
        n_months: Number of months in the timeline.
        seed: Random seed for reproducibility.
        event_rate_per_year: Expected events per 12-month period.

    Returns:
        List of Event, sorted by event_date. May be empty.

    """
    rng = np.random.default_rng(seed)
    # Snap start to month-end
    cur = _last_day(start_date.year, start_date.month)

    n_events = int(rng.poisson(event_rate_per_year * n_months / 12))
    events: list[Event] = []
    for i in range(n_events):
        # Uniform random month in the timeline
        offset = int(rng.integers(0, n_months))
        ev_date = _add_months(cur, offset)
        ev_date = _last_day(ev_date.year, ev_date.month)

        ev_type = EventType(rng.choice([e.value for e in EventType]))
        cohort = str(rng.choice(list(VALID_COHORTS[ev_type])))
        magnitude, effect_size = _draw_magnitude(ev_type, rng)

        events.append(
            Event(
                event_id=f"EVT-{i:04d}",
                event_type=ev_type,
                event_date=ev_date,
                magnitude=magnitude,
                affected_cohort=cohort,
                transmission_lag_months=DEFAULT_TRANSMISSION_LAG[ev_type],
                effect_duration_months=DEFAULT_EFFECT_DURATION[ev_type],
                effect_size=effect_size,
                metadata={"cohort": cohort},
            )
        )
    events.sort(key=lambda e: (e.event_date, e.event_id))
    # Re-number in chronological order for stable IDs
    renumbered = [
        Event(
            event_id=f"EVT-{j:04d}",
            event_type=e.event_type,
            event_date=e.event_date,
            magnitude=e.magnitude,
            affected_cohort=e.affected_cohort,
            transmission_lag_months=e.transmission_lag_months,
            effect_duration_months=e.effect_duration_months,
            effect_size=e.effect_size,
            metadata=e.metadata,
        )
        for j, e in enumerate(events)
    ]
    return renumbered


def _draw_magnitude(
    ev_type: EventType, rng: np.random.Generator
) -> tuple[float, float]:
    """Draw (magnitude, logit effect_size) for an event type.

    Effect sizes are calibrated so the affected cohort's switch probability
    moves by ~3-10pp from a ~10% baseline (logit(-2.2) ≈ 0.10). A +0.4 logit
    shift moves 10% → ~14%; +0.8 moves 10% → ~20%.
    """
    if ev_type == EventType.POLICY_SHOCK:
        # Stress-test qualifying rate change: ±50-150bp
        mag = float(rng.uniform(0.5, 1.5)) * float(rng.choice([-1.0, 1.0]))
        # Tighter test (positive mag) → fewer qualify → more switch shopping
        effect = 0.45 * np.sign(mag) * float(rng.uniform(0.7, 1.3))
        return mag, float(effect)
    if ev_type == EventType.RATE_SHIFT:
        # BoC surprise: ±25-75bp
        mag = float(rng.uniform(0.25, 0.75)) * float(rng.choice([-1.0, 1.0]))
        # Hike (positive) → higher spreads → more switching
        effect = 0.9 * mag * float(rng.uniform(0.8, 1.2))
        return mag, float(effect)
    if ev_type == EventType.HOUSING_DISRUPTION:
        # Price drop: 5-20%
        mag = float(rng.uniform(0.05, 0.20))
        # Bigger drop → more underwater → fewer can switch (negative), but
        # distressed sales increase switching (positive). Net: distress dominates.
        effect = 2.0 * mag * float(rng.uniform(0.8, 1.2))
        return mag, float(effect)
    if ev_type == EventType.LENDER_SHIFT:
        # Campaign intensity 0.3-1.0
        mag = float(rng.uniform(0.3, 1.0))
        # Retention campaign → LESS switching (negative effect)
        effect = -0.8 * mag * float(rng.uniform(0.8, 1.2))
        return mag, float(effect)
    # MACRO_SURPRISE
    mag = float(rng.uniform(1.0, 4.0))  # pp unemployment increase
    effect = 0.18 * mag * float(rng.uniform(0.8, 1.2))
    return mag, float(effect)


def events_visible_asof(events: list[Event], asof: date) -> list[Event]:
    """Return events with ``event_date <= asof`` (PIT-safe filter).

    This is the ONLY sanctioned way for downstream code to access the timeline
    at a given as-of date. Anything else is a PIT violation.

    Args:
        events: Full event timeline.
        asof: The as-of date (typically a month-end origin).

    Returns:
        Events with event_date <= asof, in chronological order.

    """
    return [e for e in events if e.event_date <= asof]


def active_effect_shift(
    events: list[Event],
    asof: date,
    cohort_matcher: object = None,
) -> float:
    """Total logit shift from events active at ``asof`` for a borrower.

    Sums ``effect_size`` over events where ``is_active(asof)``. The caller is
    responsible for cohort matching (whether this borrower is in the event's
    ``affected_cohort``); pass only matching events, or filter inside.

    Args:
        events: Candidate events (already PIT-filtered and cohort-matched).
        asof: Month-end date of the outcome being generated.
        cohort_matcher: Unused; reserved for future per-borrower matching.

    Returns:
        Sum of effect_size for active events (0.0 if none).

    """
    return float(sum(e.effect_size for e in events if e.is_active(asof)))


def cohort_matches(
    affected_cohort: str,
    province: str,
    insured_flag: bool,
    credit_score_band: str,
    rate_type: str,
    occupancy_type: str = "owner_occupied",
    employment_status: str = "employed",
    borrower_id: str = "",
) -> bool:
    """Check whether a borrower belongs to an event's affected cohort.

    This is the heterogeneous-effect mechanism: events don't hit everyone
    equally. Agents must reason about *who* is affected, not just *when*.

    Args:
        affected_cohort: Cohort selector from the Event.
        province: Borrower's province (ON/BC/AB/QC/MB).
        insured_flag: Whether the mortgage is insured.
        credit_score_band: Credit band string.
        rate_type: "fixed" or "variable".
        occupancy_type: Occupancy type string.
        employment_status: Employment status string.
        borrower_id: Borrower ID (for deterministic lender assignment).

    Returns:
        True if the borrower is in the affected cohort.

    """
    c = affected_cohort
    if c == "all_borrowers":
        return True
    if c == "uninsured_720plus":
        return (not insured_flag) and credit_score_band == "720_plus"
    if c == "uninsured_all":
        return not insured_flag
    if c == "variable_rate":
        return rate_type == "variable"
    if c == "resetting_3mo":
        # Approximated by variable-rate borrowers (they reset continuously).
        # The simulator refines this with actual maturity proximity.
        return rate_type == "variable"
    if c == "toronto_condo":
        return province == "ON"  # proxy: ON ≈ GTA concentration
    if c == "vancouver_detached":
        return province == "BC"
    if c == "alberta_all":
        return province == "AB"
    if c == "ontario_all":
        return province == "ON"
    if c in ("bank_td", "bank_rbc", "bank_bmo", "bank_scotia"):
        # Lender is not in the base schema; assign deterministically by
        # hashing borrower_id so the DGP is reproducible per borrower.
        # Each bank gets ~25% of borrowers.
        import hashlib

        h = int(hashlib.md5(borrower_id.encode()).hexdigest(), 16)
        bank_idx = h % 4
        banks = ("bank_td", "bank_rbc", "bank_bmo", "bank_scotia")
        return c == banks[bank_idx]
    if c == "ontario_manufacturing":
        return province == "ON" and employment_status in ("employed", "unemployed")
    if c == "alberta_energy":
        return province == "AB"
    if c == "bc_tech":
        return province == "BC"
    if c == "quebec_all":
        return province == "QC"
    return False
