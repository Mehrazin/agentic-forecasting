"""Tests for the TD-One agent controller.

Covers the safety-critical adjustment parsing/clipping and the fail-safe
behavior — the paths that determine whether the M3 ablation stays honest.
"""

from unittest.mock import MagicMock

import pytest

from td_one.agent.controller import ForecastAgent


def _agent_with_text(text: str) -> ForecastAgent:
    mock_llm = MagicMock()
    mock_llm.complete.return_value = text
    return ForecastAgent(llm=mock_llm)


def test_agent_retains_provider() -> None:
    """Agent keeps the LLM provider (contract for the future ReAct loop)."""
    mock_llm = MagicMock()
    agent = ForecastAgent(llm=mock_llm)
    assert agent.llm is mock_llm


def test_agent_builds_evidence_prompt() -> None:
    """Evidence prompt pins the run date next to the temporal guard."""
    agent = ForecastAgent(llm=MagicMock())
    prompt = agent._build_evidence_prompt(
        segment_id="low_rate",
        run_date="2026-10-15",
        segment_profile="Fixed rate <=3%, deposits < $5k",
    )
    assert "low_rate" in prompt
    # Structural contract: run_date adjacent to the guard clause
    assert "BEFORE 2026-10-15" in prompt


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0.35", 0.35),
        ("-0.10", -0.10),
        ("2.5", 0.75),  # clipped
        ("-1.2", -0.75),  # clipped
        ("0.0", 0.0),
    ],
)
def test_estimate_adjustment_parsing_and_clipping(text: str, expected: float) -> None:
    """LLM text -> bounded delta."""
    agent = _agent_with_text(text)
    delta = agent.estimate_adjustment("s", "2026-10-15", "p", 0.2, [{"fact": "x"}])
    assert delta == pytest.approx(expected)


def test_estimate_adjustment_takes_last_float() -> None:
    """Models echo context first; the trailing number is the verdict.

    Regression test: the old first-float regex turned
    "based on 3 sources, delta 0.25" into delta=3 -> clipped to 0.75.
    """
    agent = _agent_with_text("based on 3 sources, I estimate delta 0.25")
    delta = agent.estimate_adjustment("s", "2026-10-15", "p", 0.2, [{"fact": "x"}])
    assert delta == pytest.approx(0.25)


def test_estimate_adjustment_no_evidence_returns_zero_without_call() -> None:
    """No evidence -> 0.0 and the LLM is never consulted."""
    mock_llm = MagicMock()
    agent = ForecastAgent(llm=mock_llm)
    assert agent.estimate_adjustment("s", "2026-10-15", "p", 0.2, []) == 0.0
    mock_llm.complete.assert_not_called()


def test_estimate_adjustment_llm_failure_returns_none() -> None:
    """LLM outage -> None (distinct from a genuine 'no adjustment' 0.0).

    Regression test: the old code swallowed all exceptions into 0.0,
    silently degrading M3 -> M1 during Groq outages.
    """
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("429 rate limited")
    agent = ForecastAgent(llm=mock_llm)
    assert agent.estimate_adjustment("s", "2026-10-15", "p", 0.2, [{"fact": "x"}]) is None


def test_estimate_adjustment_unparseable_returns_none() -> None:
    """No number in the response -> None, not a silent 0.0."""
    agent = _agent_with_text("no number here")
    assert agent.estimate_adjustment("s", "2026-10-15", "p", 0.2, [{"fact": "x"}]) is None


def test_estimate_adjustment_retries_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transient LLM failures are retried; success on retry returns delta."""
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [RuntimeError("429"), "0.30"]
    agent = ForecastAgent(llm=mock_llm)
    delta = agent.estimate_adjustment("s", "2026-10-15", "p", 0.2, [{"fact": "x"}])
    assert delta == pytest.approx(0.30)
    assert mock_llm.complete.call_count == 2


def test_estimate_adjustment_gives_up_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistent failure -> None after exhausting retries (not silent 0.0)."""
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("500")
    agent = ForecastAgent(llm=mock_llm)
    assert agent.estimate_adjustment("s", "2026-10-15", "p", 0.2, [{"fact": "x"}]) is None
    assert mock_llm.complete.call_count == 3
