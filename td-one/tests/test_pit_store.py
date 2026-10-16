"""Tests for the point-in-time DuckDB store."""

from datetime import date
from pathlib import Path

import pytest

from td_one.data_gen.simulator import MortgageSimulator
from td_one.data_service.pit_store import PITStore
from td_one.schemas import RenewalOutcomeRecord


@pytest.fixture
def pit_store(tmp_path: Path) -> PITStore:
    """PIT store with small synthetic data."""
    db_path = tmp_path / "test.duckdb"
    store = PITStore(db_path)
    sim = MortgageSimulator(n_customers=50, n_months=12, seed=42)
    records = sim.generate_history()
    store.ingest_history(records)
    return store


def test_pit_filters_by_snapshot_date(pit_store: PITStore) -> None:
    """History queries respect snapshot_date <= run_date."""
    run_date = date(2021, 8, 28)
    df = pit_store.get_history(run_date)
    assert len(df) > 0
    assert (df["snapshot_date"] <= run_date.isoformat()).all()


def test_pit_no_future_data(pit_store: PITStore) -> None:
    """Cannot see future snapshots."""
    early = date(2021, 5, 28)
    late = date(2021, 10, 28)
    df_early = pit_store.get_history(early)
    df_late = pit_store.get_history(late)
    assert len(df_late) > len(df_early)


def test_pit_segment_query(pit_store: PITStore) -> None:
    """Segment filtering works with PIT."""
    run_date = date(2021, 8, 28)
    df = pit_store.get_segment(
        run_date=run_date,
        province="ON",
    )
    assert len(df) > 0
    assert (df["province"] == "ON").all()


def _outcome(mortgage_id: str, record_asof: date, observed_at: date | None) -> RenewalOutcomeRecord:
    return RenewalOutcomeRecord(
        customer_id="C001",
        mortgage_id=mortgage_id,
        renewal_episode_id="E001-01",
        scheduled_maturity_date=date(2021, 6, 30),
        outcome="switched_lender" if observed_at else "pending",
        switched_lender_flag=True if observed_at else None,
        event_date=date(2021, 6, 30) if observed_at else None,
        outcome_observed_at=observed_at,
        record_asof=record_asof,
    )


def test_pit_outcomes_mask_unobserved(tmp_path: Path) -> None:
    """Outcomes with outcome_observed_at > run_date are masked.

    Regression test: the outcomes table and its dual guards
    (record_asof AND outcome_observed_at) were documented but missing.
    """
    store = PITStore(tmp_path / "o.duckdb")
    store.ingest_outcomes(
        [
            _outcome("M1", date(2021, 7, 1), date(2021, 8, 15)),  # observed late
            _outcome("M2", date(2021, 7, 1), date(2021, 6, 30)),  # observed
            _outcome("M3", date(2021, 9, 1), date(2021, 6, 30)),  # recorded late
        ]
    )
    # At 2021-07-15: M1's outcome not yet observed -> masked;
    # M3 not yet recorded -> masked. Only M2 visible.
    df = store.get_outcomes(date(2021, 7, 15))
    assert set(df["mortgage_id"]) == {"M2"}
    # At 2021-09-15: all three visible.
    df2 = store.get_outcomes(date(2021, 9, 15))
    assert set(df2["mortgage_id"]) == {"M1", "M2", "M3"}
    store.close()
