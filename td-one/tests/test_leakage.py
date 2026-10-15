"""Tests for the temporal leakage auditor."""

from datetime import date

from td_one.evaluation.leakage import LeakageAuditor


def test_no_leakage_clean_trace() -> None:
    """Clean trace passes audit."""
    auditor = LeakageAuditor(run_date=date(2026, 10, 15))
    trace = [
        {"timestamp": "2026-10-10", "content": "BoC held rates"},
        {"timestamp": "2026-10-14", "content": "CPI data released"},
    ]
    assert auditor.audit(trace) == 0.0  # 0% leakage


def test_leakage_detected() -> None:
    """Post-run_date timestamp is flagged."""
    auditor = LeakageAuditor(run_date=date(2026, 10, 15))
    trace = [
        {"timestamp": "2026-10-10", "content": "OK"},
        {"timestamp": "2026-10-20", "content": "Future data!"},
    ]
    assert auditor.audit(trace) > 0.0


def test_leakage_rate_calculation() -> None:
    """Leakage rate = leaked / total."""
    auditor = LeakageAuditor(run_date=date(2026, 10, 15))
    trace = [
        {"timestamp": "2026-10-10", "content": "a"},
        {"timestamp": "2026-10-16", "content": "b"},
        {"timestamp": "2026-10-17", "content": "c"},
        {"timestamp": "2026-10-14", "content": "d"},
    ]
    # 2/4 = 50% leakage
    assert auditor.audit(trace) == 0.5


def test_leakage_unparseable_timestamp_is_leakage() -> None:
    """Fail-closed: malformed timestamps count as leakage, not safe.

    Regression test: the old code silently treated unparseable dates as
    safe, letting sloppy or injected timestamps launder through.
    """
    auditor = LeakageAuditor(run_date=date(2026, 10, 15))
    trace = [
        {"timestamp": "not-a-date", "content": "x"},
        {"timestamp": "2026-10-10", "content": "y"},
    ]
    assert auditor.audit(trace) == 0.5


def test_leakage_boundary_inclusive() -> None:
    """Timestamp == run_date is allowed (PIT semantics are inclusive)."""
    auditor = LeakageAuditor(run_date=date(2026, 10, 15))
    trace = [{"timestamp": "2026-10-15", "content": "same day"}]
    assert auditor.audit(trace) == 0.0


def test_assert_no_leakage_raises() -> None:
    """The raise path (explicit error-path coverage per repo philosophy)."""
    import pytest

    auditor = LeakageAuditor(run_date=date(2026, 10, 15))
    with pytest.raises(ValueError, match="Temporal leakage"):
        auditor.assert_no_leakage([{"timestamp": "2026-10-20", "content": "x"}])
    # Clean trace does not raise
    auditor.assert_no_leakage([{"timestamp": "2026-10-10", "content": "x"}])
