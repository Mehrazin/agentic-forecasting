"""Tests for td_one.agent.risk_reporter."""

import pytest

from td_one.agent.risk_reporter import (
    RiskReporter,
    build_explanation_prompt,
    check_citations,
    compute_risk_bins,
    widen_for_uncertainty,
)


def test_compute_risk_bins() -> None:
    train = [0.05, 0.06, 0.07, 0.10, 0.11, 0.12, 0.15, 0.16, 0.20]
    bins = compute_risk_bins({"a": 0.05, "b": 0.11, "c": 0.20}, train)
    assert bins == {"a": "LOW", "b": "MEDIUM", "c": "HIGH"}


def test_compute_risk_bins_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_risk_bins({"a": 0.1}, [])


def test_widen_for_uncertainty_downgrades() -> None:
    bins = {"a": "HIGH", "b": "MEDIUM", "c": "LOW"}
    intervals = {"a": (0.05, 0.25), "b": (0.08, 0.12), "c": (0.01, 0.20)}
    out = widen_for_uncertainty(bins, intervals, width_threshold=0.05)
    assert out["a"] == "MEDIUM"  # wide -> downgraded
    assert out["b"] == "MEDIUM"  # narrow -> kept
    assert out["c"] == "LOW"  # LOW stays LOW


def test_check_citations_ok() -> None:
    text = (
        "Attrition is forecast at 0.12 [stat]. "
        "Ontario buyers face headwinds [url:0]."
    )
    assert check_citations(text) is True


def test_check_citations_rejects_uncited() -> None:
    text = "Attrition is forecast at 0.12 [stat]. Rates will explode next year."
    assert check_citations(text) is False


def test_check_citations_skips_structural() -> None:
    assert check_citations("In summary, risk is elevated [stat].") is True


def test_build_explanation_prompt_marks_citables() -> None:
    system, user = build_explanation_prompt(
        "g1",
        "province == 'ON'",
        {"predicted_mean_attrition": 0.12},
        [{"title": "BoC holds", "url": "http://x"}],
    )
    assert "[stat]" in user
    assert "[url:0]" in user
    assert "citation" in system.lower()


class GoodLLM:
    def complete(self, prompt: str, system=None) -> str:
        return "Risk is HIGH at 0.18 [stat]. News notes pressure [url:0]."


class BadLLM:
    def complete(self, prompt: str, system=None) -> str:
        return "Risk is HIGH. Trust me."


def test_risk_reporter_valid() -> None:
    r = RiskReporter(GoodLLM())
    rep = r.report("g1", "province == 'ON'", 0.18, (0.15, 0.21), "HIGH",
                   articles=[{"title": "t", "url": "u"}])
    assert rep.citations_valid is True
    assert rep.risk_level == "HIGH"


def test_risk_reporter_flags_uncited() -> None:
    r = RiskReporter(BadLLM())
    rep = r.report("g1", "province == 'ON'", 0.18, (0.15, 0.21), "HIGH")
    assert rep.citations_valid is False
