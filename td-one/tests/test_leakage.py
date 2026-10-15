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
