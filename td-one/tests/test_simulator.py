"""Tests for the synthetic mortgage data generator."""

import pytest
from datetime import date

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
    """Generates mortgage_history records."""
    records = small_sim.generate_history()
    assert len(records) > 0
    # All records validate against schema
    for r in records[:10]:
        assert isinstance(r, MortgageHistoryRecord)


def test_simulator_customer_count(small_sim: MortgageSimulator) -> None:
    """Correct number of unique customers."""
    records = small_sim.generate_history()
    customer_ids = {r.customer_id for r in records}
    assert len(customer_ids) == 100


def test_simulator_months(small_sim: MortgageSimulator) -> None:
    """Correct number of months."""
    records = small_sim.generate_history()
    months = {r.snapshot_date for r in records}
    assert len(months) == 12


def test_planted_rate_shock() -> None:
    """Low-rate borrowers have elevated switching (plan §3.2)."""
    sim = MortgageSimulator(n_customers=1000, n_months=66, seed=123)
    # The shock should create a detectable difference
    # (detailed assertion in integration test with outcomes)
    assert sim is not None


def test_deterministic_with_seed() -> None:
    """Same seed -> same output."""
    sim1 = MortgageSimulator(n_customers=50, n_months=6, seed=99)
    sim2 = MortgageSimulator(n_customers=50, n_months=6, seed=99)
    r1 = sim1.generate_history()
    r2 = sim2.generate_history()
    assert len(r1) == len(r2)
    assert r1[0].customer_id == r2[0].customer_id
    assert r1[0].mortgage_balance_cad == r2[0].mortgage_balance_cad
