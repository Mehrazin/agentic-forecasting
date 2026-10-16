"""Tests for the synthetic mortgage data generator.

Includes the planted-shock uplift test (now real, since the simulator has
a structural outcome DGP) and realism invariants.
"""

from datetime import date

import pytest

from td_one.data_gen.amortization import canadian_balance_after_payment
from td_one.data_gen.simulator import MortgageSimulator
from td_one.schemas import MortgageHistoryRecord


@pytest.fixture
def small_sim() -> MortgageSimulator:
    """Small simulator for fast tests (100 customers, 12 months)."""
    return MortgageSimulator(
        n_customers=100,
        n_months=12,
        seed=42,
    )


def test_simulator_generates_history(small_sim: MortgageSimulator) -> None:
    """Generates mortgage_history records with unique customers."""
    records = small_sim.generate_history()
    assert len(records) == 100 * 12
    assert len({r.mortgage_id for r in records}) == 100
    assert len({r.snapshot_date for r in records}) == 12


def test_snapshot_dates_monotonic_per_customer(small_sim: MortgageSimulator) -> None:
    """Per-customer snapshots are chronological (PIT invariant)."""
    records = small_sim.generate_history()
    by_customer: dict[str, list[date]] = {}
    for r in records:
        by_customer.setdefault(r.mortgage_id, []).append(r.snapshot_date)
    for dates in by_customer.values():
        assert dates == sorted(dates)


def test_balance_follows_amortization(small_sim: MortgageSimulator) -> None:
    """Balance_{t+1} is consistent with the Canadian amortization formula."""
    records = small_sim.generate_history()
    by_customer: dict[str, list[MortgageHistoryRecord]] = {}
    for r in records:
        by_customer.setdefault(r.mortgage_id, []).append(r)
    # Check a few customers; skip variable-rate (rate changes mid-stream)
    checked = 0
    for recs in by_customer.values():
        if recs[0].rate_type != "fixed" or checked >= 5:
            continue
        for prev, cur in zip(recs, recs[1:]):  # noqa: B905 - shifted window, lengths differ by 1 by construction
            expected = canadian_balance_after_payment(
                prev.mortgage_balance_cad,
                prev.mortgage_rate_pct,
                prev.mortgage_payment_monthly_cad,
            )
            # abs=0.5: the simulator rounds balance to cents each month,
            # so recomputation from rounded values accumulates small error
            assert cur.mortgage_balance_cad == pytest.approx(expected, abs=0.5)
        checked += 1
    assert checked > 0


def test_amortization_decrements(small_sim: MortgageSimulator) -> None:
    """Remaining amortization strictly decreases month to month."""
    records = small_sim.generate_history()
    by_customer: dict[str, list[MortgageHistoryRecord]] = {}
    for r in records:
        by_customer.setdefault(r.mortgage_id, []).append(r)
    for recs in list(by_customer.values())[:10]:
        amorts = [r.remaining_amortization_months for r in recs]
        pairs = zip(amorts, amorts[1:])  # noqa: B905 - shifted window, lengths differ by 1
        assert all(b == a - 1 for a, b in pairs)


def test_insured_flag_threaded(small_sim: MortgageSimulator) -> None:
    """insured_flag is populated (was generated then dropped — regression)."""
    records = small_sim.generate_history()
    flags = {r.insured_flag for r in records}
    assert None not in flags
    assert flags == {True, False}


def test_planted_rate_shock() -> None:
    """Low-rate 2021 cohort switches measurably more than control.

    Structural DGP test: the planted shock must be detectable as an
    outcome uplift, not just a cohort attribute.
    """
    sim = MortgageSimulator(n_customers=2000, n_months=66, seed=123)
    history = sim.generate_history()
    outcomes = sim.generate_outcomes(history)
    assert len(outcomes) > 100, "need enough maturities for a powered test"

    first: dict[str, MortgageHistoryRecord] = {}
    for r in history:
        first.setdefault(r.mortgage_id, r)

    def is_low_rate(mid: str) -> bool:
        r = first[mid]
        return (
            r.rate_type == "fixed"
            and r.origination_date.year == 2021
            and r.mortgage_rate_pct <= 3.0
        )

    low = [o for o in outcomes if is_low_rate(o.mortgage_id)]
    ctl = [o for o in outcomes if not is_low_rate(o.mortgage_id)]
    assert len(low) > 50 and len(ctl) > 50

    p1 = sum(1 for o in low if o.switched_lender_flag) / len(low)
    p0 = sum(1 for o in ctl if o.switched_lender_flag) / len(ctl)
    # Binomial z-test for the uplift (fixed seed -> deterministic)
    se = (p1 * (1 - p1) / len(low) + p0 * (1 - p0) / len(ctl)) ** 0.5
    z = (p1 - p0) / se if se > 0 else 0.0
    assert z > 2.0, f"no detectable shock: low_rate={p1:.3f} control={p0:.3f} z={z:.2f}"


def test_outcomes_have_observation_lag() -> None:
    """Outcomes are observed after the event (PIT maskable)."""
    sim = MortgageSimulator(n_customers=500, n_months=66, seed=7)
    outcomes = sim.generate_outcomes()
    assert len(outcomes) > 0
    for o in outcomes:
        assert o.outcome_observed_at is not None and o.event_date is not None
        assert o.outcome_observed_at >= o.event_date


def test_deterministic_with_seed() -> None:
    """Same seed -> same output."""
    sim1 = MortgageSimulator(n_customers=50, n_months=6, seed=99)
    sim2 = MortgageSimulator(n_customers=50, n_months=6, seed=99)
    r1 = sim1.generate_history()
    r2 = sim2.generate_history()
    assert len(r1) == len(r2)
    assert r1[0].customer_id == r2[0].customer_id
    assert r1[0].mortgage_balance_cad == r2[0].mortgage_balance_cad


def test_maturities_spread_across_calendar() -> None:
    """Check maturities spread across calendar months.

    Regression: originations were hardcoded to June, collapsing all
    maturities into ~5 distinct months at 10k x 66 scale.
    """
    sim = MortgageSimulator(n_customers=2000, n_months=66, seed=123)
    histories = sim.generate_history()
    maturities = {(r.maturity_date.year, r.maturity_date.month) for r in histories}
    assert len(maturities) >= 24, f"only {len(maturities)} distinct maturity months"
    june_share = sum(1 for y, m in maturities if m == 6) / len(maturities)
    assert june_share < 0.5, f"maturities still June-concentrated: {june_share:.2f}"
