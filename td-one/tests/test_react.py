"""Tests for the ReAct orchestration loop (agent/react.py).

The loop is decoupled from td_one.tools: tools are injected as plain
callables, so every behavior here is driven by a scripted MockLLMClient.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date

from td_one.agent.react import MockLLMClient, ReActAgent


def _tools(log: list[str]) -> dict[str, Callable[[dict[str, object]], str]]:
    def data_segment(args: dict[str, object]) -> str:
        log.append(f"data_segment:{args}")
        return '{"n_renewals": [120, 130, 125], "n_switches": [12, 14, 11]}'

    def evidence(args: dict[str, object]) -> str:
        log.append(f"evidence:{args}")
        return '[{"source": "boc", "fact": "rate held"}]'

    def forecasting(args: dict[str, object]) -> str:
        log.append(f"forecasting:{args}")
        return '{"q10": 0.08, "q50": 0.10, "q90": 0.13}'

    return {
        "data_segment": data_segment,
        "evidence": evidence,
        "forecasting": forecasting,
    }


def _script(*responses: str) -> MockLLMClient:
    return MockLLMClient(list(responses))


def test_scripted_three_step_trace_ends_with_forecast() -> None:
    """A scripted tool-use trace should dispatch in order and finalize."""
    calls: list[str] = []
    llm = _script(
        json.dumps({"action": "data_segment", "args": {"segment": "seg-a"}}),
        json.dumps({"action": "evidence", "args": {"query": "BoC rate"}}),
        json.dumps({"action": "forecasting", "args": {"segment": "seg-a", "horizon": 1}}),
        json.dumps(
            {
                "action": "finalize",
                "forecast": {"0.1": 0.08, "0.5": 0.10, "0.9": 0.12},
            }
        ),
    )
    agent = ReActAgent(llm, _tools(calls), max_steps=6)
    result = agent.run("Forecast switching for segment seg-a", date(2025, 6, 30))

    assert result["forecast"] == {0.1: 0.08, 0.5: 0.10, 0.9: 0.12}
    assert result["steps"] == 4
    actions = [s["action"] for s in result["trace"]]
    assert actions == ["data_segment", "evidence", "forecasting", "finalize"]
    # tools were actually dispatched in script order
    assert calls[0].startswith("data_segment:")
    assert calls[1].startswith("evidence:")
    assert calls[2].startswith("forecasting:")
    # observations were recorded back into the trace
    assert "n_switches" in result["trace"][0]["observation"]


def test_malformed_json_recovery_continues_loop() -> None:
    """A non-JSON LLM reply is recorded as a failed step; the loop continues."""
    llm = _script(
        "this is not json at all",
        json.dumps({"action": "finalize", "forecast": {"0.1": 0.05, "0.5": 0.1, "0.9": 0.2}}),
    )
    agent = ReActAgent(llm, _tools([]), max_steps=6)
    result = agent.run("task", date(2025, 6, 30))

    assert result["forecast"] == {0.1: 0.05, 0.5: 0.1, 0.9: 0.2}
    assert result["steps"] == 2
    failed = result["trace"][0]
    assert failed["action"] == "parse_error"
    assert "observation" in failed and failed["observation"]


def test_unknown_action_recorded_not_crash() -> None:
    """An unknown action name is recorded as an error observation; loop continues."""
    llm = _script(
        json.dumps({"action": "web_search", "args": {}}),
        json.dumps({"action": "finalize", "forecast": {"0.1": 0.05, "0.5": 0.1, "0.9": 0.2}}),
    )
    agent = ReActAgent(llm, _tools([]), max_steps=6)
    result = agent.run("task", date(2025, 6, 30))

    assert result["forecast"] == {0.1: 0.05, 0.5: 0.1, 0.9: 0.2}
    assert "unknown action" in result["trace"][0]["observation"].lower()


def test_max_steps_exhaustion_returns_none() -> None:
    """An LLM that never finalizes yields forecast None after max_steps."""
    # mock repeats the data_segment action forever; the loop must stop anyway
    llm = _script(json.dumps({"action": "data_segment", "args": {}}))
    agent = ReActAgent(llm, _tools([]), max_steps=2)
    result = agent.run("task", date(2025, 6, 30))

    assert result["forecast"] is None
    assert result["steps"] == 2
    assert len(result["trace"]) == 2


def test_mock_llm_client_repeats_last_when_exhausted() -> None:
    """MockLLMClient contract: repeats its last response instead of crashing."""
    llm = MockLLMClient(['{"action": "finalize", "forecast": {}}'])
    assert llm.complete("p1") == '{"action": "finalize", "forecast": {}}'
    assert llm.complete("p2") == '{"action": "finalize", "forecast": {}}'
    assert llm.calls == 2


def test_finalize_with_empty_forecast_still_returns_empty_dict() -> None:
    """An empty finalize forecast is returned as an empty dict, not None."""
    llm = _script('{"action": "finalize", "forecast": {}}')
    agent = ReActAgent(llm, _tools([]), max_steps=6)
    result = agent.run("task", date(2025, 6, 30))
    assert result["forecast"] == {}
