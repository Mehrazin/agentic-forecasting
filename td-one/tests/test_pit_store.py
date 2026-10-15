"""Tests for the point-in-time DuckDB store."""

import pytest
from datetime import date
from pathlib import Path
import tempfile

from td_one.data_service.pit_store import PITStore
from td_one.data_gen.simulator import MortgageSimulator


@pytest.fixture
def pit_store() -> PITStore:
    """PIT store with small synthetic data."""
    tmp = Path(tempfile.mkdtemp())
    db_path = tmp / "test.duckdb"
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
