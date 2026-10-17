"""Tests for the event-augmented simulator: events and news synthesis.

Covers: event timeline generation, PIT-safety invariants, cohort matching,
news feed generation, and the simulator's event integration.
"""

from datetime import date

import pytest

from td_one.data_gen.events import (
    Event,
    EventType,
    active_effect_shift,
    cohort_matches,
    events_visible_asof,
    generate_event_timeline,
)
from td_one.data_gen.news_synthesis import (
    articles_visible_asof,
    coverage_stats,
    generate_news_feed,
    strip_eval_fields,
)


@pytest.fixture
def timeline() -> list[Event]:
    """Small deterministic timeline for tests."""
    return generate_event_timeline(
        start_date=date(2022, 1, 31),
        n_months=24,
        seed=123,
        event_rate_per_year=6.0,
    )


# --- Event timeline ---


def test_timeline_sorted_and_bounded(timeline: list[Event]) -> None:
    """Events are sorted by date and within the requested window."""
    dates = [e.event_date for e in timeline]
    assert dates == sorted(dates)
    for e in timeline:
        assert date(2022, 1, 31) <= e.event_date <= date(2024, 1, 31)


def test_timeline_reproducible() -> None:
    """Same seed → same timeline."""
    t1 = generate_event_timeline(date(2022, 1, 31), 24, seed=7)
    t2 = generate_event_timeline(date(2022, 1, 31), 24, seed=7)
    assert [(e.event_id, e.event_date, e.event_type) for e in t1] == [
        (e.event_id, e.event_date, e.event_type) for e in t2
    ]


def test_timeline_ids_chronological(timeline: list[Event]) -> None:
    """Event IDs are numbered in chronological order."""
    ids = [e.event_id for e in timeline]
    assert ids == sorted(ids)


def test_event_types_valid(timeline: list[Event]) -> None:
    """All events have valid types and cohorts."""
    from td_one.data_gen.events import VALID_COHORTS

    for e in timeline:
        assert isinstance(e.event_type, EventType)
        assert e.affected_cohort in VALID_COHORTS[e.event_type]


# --- PIT safety ---


def test_events_visible_asof_pit(timeline: list[Event]) -> None:
    """events_visible_asof only returns events with event_date <= asof."""
    asof = date(2022, 6, 30)
    visible = events_visible_asof(timeline, asof)
    for e in visible:
        assert e.event_date <= asof
    # And it returns ALL such events (no silent dropping)
    expected = [e for e in timeline if e.event_date <= asof]
    assert len(visible) == len(expected)


def test_events_visible_asof_empty_before_start(timeline: list[Event]) -> None:
    """No events visible before the timeline starts."""
    assert events_visible_asof(timeline, date(2020, 1, 31)) == []


def test_effect_window_respects_lag() -> None:
    """Effect window starts after transmission lag, not at event_date."""
    e = Event(
        event_id="EVT-0000",
        event_type=EventType.POLICY_SHOCK,
        event_date=date(2022, 3, 31),
        magnitude=1.0,
        affected_cohort="uninsured_720plus",
        transmission_lag_months=2,
        effect_duration_months=12,
        effect_size=0.5,
    )
    onset, end = e.effect_window()
    assert onset == date(2022, 5, 31)
    assert end == date(2023, 4, 30)
    # Not active at event_date (lag not yet elapsed)
    assert not e.is_active(date(2022, 3, 31))
    assert not e.is_active(date(2022, 4, 30))
    assert e.is_active(date(2022, 5, 31))
    assert e.is_active(date(2023, 4, 30))
    assert not e.is_active(date(2023, 5, 31))


def test_active_effect_shift_sums() -> None:
    """active_effect_shift sums effect_size over active events."""
    e1 = Event("EVT-0000", EventType.RATE_SHIFT, date(2022, 1, 31), 0.5,
              "all_borrowers", 0, 6, 0.4)
    e2 = Event("EVT-0001", EventType.RATE_SHIFT, date(2022, 3, 31), -0.25,
              "all_borrowers", 0, 6, -0.2)
    # Both active in May 2022
    assert active_effect_shift([e1, e2], date(2022, 5, 31)) == pytest.approx(0.2)
    # Only e2 active in Aug 2022 (e1's 6-month window ended Jul 2022)
    assert active_effect_shift([e1, e2], date(2022, 8, 31)) == pytest.approx(-0.2)


# --- Cohort matching ---


def test_cohort_matches_uninsured() -> None:
    assert cohort_matches("uninsured_720plus", "ON", False, "720_plus", "fixed")
    assert not cohort_matches("uninsured_720plus", "ON", True, "720_plus", "fixed")
    assert not cohort_matches("uninsured_720plus", "ON", False, "660_719", "fixed")
    assert cohort_matches("uninsured_all", "BC", False, "600_659", "variable")
    assert not cohort_matches("uninsured_all", "BC", True, "600_659", "variable")


def test_cohort_matches_geography() -> None:
    assert cohort_matches("toronto_condo", "ON", True, "720_plus", "fixed")
    assert not cohort_matches("toronto_condo", "BC", True, "720_plus", "fixed")
    assert cohort_matches("alberta_all", "AB", True, "720_plus", "fixed")
    assert cohort_matches("all_borrowers", "MB", True, "below_600", "variable")


def test_cohort_matches_lender_deterministic() -> None:
    """Lender assignment is deterministic per borrower_id."""
    r1 = cohort_matches("bank_td", "ON", True, "720_plus", "fixed", borrower_id="C00001")
    r2 = cohort_matches("bank_td", "ON", True, "720_plus", "fixed", borrower_id="C00001")
    assert r1 == r2
    # Exactly one bank matches per borrower
    matches = sum(
        cohort_matches(b, "ON", True, "720_plus", "fixed", borrower_id="C00001")
        for b in ("bank_td", "bank_rbc", "bank_bmo", "bank_scotia")
    )
    assert matches == 1


# --- News synthesis ---


def test_news_feed_pit_safe(timeline: list[Event]) -> None:
    """No article is dated before its event."""
    articles = generate_news_feed(timeline, seed=99)
    by_id = {e.event_id: e for e in timeline}
    for a in articles:
        ev = by_id[str(a["event_id"])]
        assert a["article_date"] >= ev.event_date, (
            f"article {a['headline'][:40]} predates event {ev.event_id}"
        )


def test_news_feed_reproducible(timeline: list[Event]) -> None:
    a1 = generate_news_feed(timeline, seed=5)
    a2 = generate_news_feed(timeline, seed=5)
    assert [(x["headline"], str(x["article_date"])) for x in a1] == [
        (x["headline"], str(x["article_date"])) for x in a2
    ]


def test_strip_eval_fields_removes_event_id(timeline: list[Event]) -> None:
    articles = generate_news_feed(timeline, seed=5)
    stripped = strip_eval_fields(articles)
    assert articles  # non-empty (high event rate in fixture)
    for a in stripped:
        assert "event_id" not in a
        assert "headline" in a and "body" in a and "article_date" in a


def test_articles_visible_asof_pit(timeline: list[Event]) -> None:
    articles = generate_news_feed(timeline, seed=5)
    asof = date(2022, 6, 15)
    visible = articles_visible_asof(articles, asof)
    for a in visible:
        assert a["article_date"] <= asof


def test_coverage_stats(timeline: list[Event]) -> None:
    articles = generate_news_feed(timeline, seed=5)
    stats = coverage_stats(articles, timeline)
    assert stats["n_events"] == len(timeline)
    assert stats["n_articles"] == len(articles)
    assert 0.0 <= stats["coverage_rate"] <= 1.0


def test_lender_events_may_have_zero_coverage() -> None:
    """LENDER_SHIFT has coverage_prob 0.4 — with few events, zero coverage is likely."""
    ev = Event("EVT-0000", EventType.LENDER_SHIFT, date(2022, 6, 30), 0.8,
              "bank_td", 0, 4, -0.5)
    # Try several seeds; at least one should give zero articles (p=0.6 each)
    zero_counts = sum(
        len(generate_news_feed([ev], seed=s)) == 0 for s in range(10)
    )
    assert zero_counts >= 1


# --- Simulator integration ---


def test_simulator_without_events_unchanged() -> None:
    """No event_timeline → identical outcomes to pre-event behavior."""
    from td_one.data_gen.simulator import MortgageSimulator

    sim = MortgageSimulator(n_customers=200, n_months=24, seed=42)
    hist = sim.generate_history()
    # Reset RNG to the same state before each generate_outcomes call by
    # re-seeding: outcomes must not depend on whether event_timeline=None
    # was passed.
    import numpy as np

    sim.rng = np.random.default_rng(999)
    o1 = sim.generate_outcomes(history=hist)
    sim.rng = np.random.default_rng(999)
    o2 = sim.generate_outcomes(history=hist, event_timeline=None)
    assert [o.outcome for o in o1] == [o.outcome for o in o2]
    # Empty list behaves like None
    sim.rng = np.random.default_rng(999)
    o3 = sim.generate_outcomes(history=hist, event_timeline=[])
    assert [o.outcome for o in o1] == [o.outcome for o in o3]


def test_simulator_with_events_shifts_outcomes() -> None:
    """A strong positive event increases switch rate for the affected cohort."""
    import numpy as np

    from td_one.data_gen.events import _last_day
    from td_one.data_gen.simulator import MortgageSimulator

    # Single large rate-hike event affecting everyone, active for the whole window
    big_event = Event(
        event_id="EVT-0000",
        event_type=EventType.RATE_SHIFT,
        event_date=_last_day(2021, 6),
        magnitude=0.75,
        affected_cohort="all_borrowers",
        transmission_lag_months=0,
        effect_duration_months=60,
        effect_size=1.5,  # large: ~10% → ~35%
    )
    sim = MortgageSimulator(n_customers=500, n_months=24, seed=42)
    hist = sim.generate_history()
    sim.rng = np.random.default_rng(999)
    base = sim.generate_outcomes(history=hist, event_timeline=None)
    sim.rng = np.random.default_rng(999)
    with_events = sim.generate_outcomes(history=hist, event_timeline=[big_event])
    # Same maturities produce outcomes in both
    assert len(base) == len(with_events) and len(base) > 0
    base_rate = sum(o.switched_lender_flag for o in base) / len(base)
    event_rate = sum(o.switched_lender_flag for o in with_events) / len(with_events)
    assert event_rate > base_rate, f"{event_rate=} not > {base_rate=}"


def test_simulator_events_pit_safe() -> None:
    """Events dated after a borrower's maturity do not affect their outcome."""
    import numpy as np

    from td_one.data_gen.events import _last_day
    from td_one.data_gen.simulator import MortgageSimulator

    # Event far in the future — should have zero effect on all outcomes
    future_event = Event(
        event_id="EVT-0000",
        event_type=EventType.RATE_SHIFT,
        event_date=_last_day(2030, 1),
        magnitude=0.75,
        affected_cohort="all_borrowers",
        transmission_lag_months=0,
        effect_duration_months=60,
        effect_size=5.0,  # huge, would dominate if leaked
    )
    sim = MortgageSimulator(n_customers=200, n_months=24, seed=42)
    hist = sim.generate_history()
    sim.rng = np.random.default_rng(999)
    base = sim.generate_outcomes(history=hist, event_timeline=None)
    sim.rng = np.random.default_rng(999)
    with_future = sim.generate_outcomes(history=hist, event_timeline=[future_event])
    assert [o.outcome for o in base] == [o.outcome for o in with_future]
