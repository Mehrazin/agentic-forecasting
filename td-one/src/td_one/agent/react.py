"""ReAct orchestration loop for the TD-One forecast agent.

The loop is intentionally decoupled from ``td_one.tools``: tools are
injected as plain callables ``dict[str, Callable[[dict[str, object]], str]]``
so the orchestration logic is testable with a scripted mock LLM and the
experiments can wire whatever tool implementations land later.

Protocol per step: the LLM replies with exactly one JSON object:
- ``{"action": "data_segment"|"evidence"|"forecasting", "args": {...}}``
  dispatches to the injected tool and records its observation.
- ``{"action": "finalize", "forecast": {"0.1": q10, "0.5": q50, "0.9": q90}}``
  emits the final rate-quantile forecast and ends the loop.

Malformed replies are recorded as failed steps and the loop continues;
if ``max_steps`` is exceeded without a finalize, the forecast is None.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import date
from typing import Protocol, TypedDict

logger = logging.getLogger(__name__)

#: Tool actions the loop will dispatch.
_TOOL_ACTIONS = ("data_segment", "evidence", "forecasting")


class LLMClient(Protocol):
    """Minimal LLM interface for the ReAct loop."""

    def complete(self, prompt: str) -> str:
        """Complete a prompt, returning the raw model text."""
        ...


class TraceStep(TypedDict):
    """One recorded step of the ReAct trace."""

    action: str
    args: dict[str, object]
    observation: str


class ReActResult(TypedDict):
    """Outcome of a ReAct run."""

    forecast: dict[float, float] | None
    trace: list[TraceStep]
    steps: int


class MockLLMClient:
    """Scripted LLM for tests and mock-mode experiments.

    Returns the scripted responses in order; once exhausted, repeats the
    last response (deterministic, and safe because the loop is bounded by
    max_steps).
    """

    def __init__(self, responses: list[str]) -> None:
        """Initialize with the scripted reply sequence.

        Args:
            responses: Non-empty list of raw reply strings.

        """
        if not responses:
            raise ValueError("MockLLMClient needs at least one scripted response")
        self._responses = responses
        self.calls = 0

    def complete(self, prompt: str) -> str:  # noqa: ARG002 - mock ignores prompt
        """Return the next scripted response (last one repeats)."""
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


def _render_prompt(task: str, run_date: date, trace: list[TraceStep]) -> str:
    """Build the per-step prompt from the task and the trace so far."""
    lines = [
        "You are a mortgage-switching forecast agent.",
        f"Run date (PIT cutoff): {run_date.isoformat()}.",
        f"Task: {task}",
        "",
        "Reply with exactly ONE JSON object per step. Available actions:",
        '  {"action": "data_segment", "args": {...}} — pull segment history',
        '  {"action": "evidence", "args": {...}} — retrieve macro/policy evidence',
        '  {"action": "forecasting", "args": {...}} — run the numerical forecast',
        '  {"action": "finalize", "forecast": {"0.1": q10, "0.5": q50, "0.9": q90}}',
        "    — emit the final switching-rate quantiles and stop.",
        "",
        "Trace so far:",
    ]
    if not trace:
        lines.append("  (no steps yet)")
    for i, step in enumerate(trace):
        lines.append(
            f"  [{i}] action={step['action']} args={json.dumps(step['args'])}"
            f" observation={step['observation'][:400]}"
        )
    lines.append("")
    lines.append("Next JSON action:")
    return "\n".join(lines)


def _parse_forecast(raw: object) -> dict[float, float] | None:
    """Parse and validate a finalize forecast mapping, or None if invalid."""
    if not isinstance(raw, dict):
        return None
    parsed: dict[float, float] = {}
    for key, value in raw.items():
        try:
            q = float(key)
            rate = float(value)
        except (TypeError, ValueError):
            return None
        if not 0.0 <= q <= 1.0 or not 0.0 <= rate <= 1.0:
            return None
        parsed[q] = rate
    return parsed


class ReActAgent:
    """Reason-Act loop that drives injected forecasting tools via an LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        tools: dict[str, Callable[[dict[str, object]], str]],
        max_steps: int = 6,
    ) -> None:
        """Initialize the agent.

        Args:
            llm_client: LLM with a ``complete(prompt) -> str`` method.
            tools: Mapping of tool action name to callable taking an args
                dict and returning an observation string.
            max_steps: Hard cap on loop iterations.

        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.llm_client = llm_client
        self.tools = tools
        self.max_steps = max_steps

    def _record(
        self, trace: list[TraceStep], action: str, args: dict[str, object], obs: str
    ) -> None:
        trace.append(TraceStep(action=action, args=args, observation=obs))

    def run(self, task: str, run_date: date) -> ReActResult:
        """Run the ReAct loop for a forecasting task.

        Args:
            task: Natural-language forecasting task.
            run_date: Point-in-time cutoff for all evidence and data pulls.

        Returns:
            ReActResult with the finalized rate-quantile forecast (None if
            the loop exhausted max_steps without a finalize), the full
            trace, and the step count.

        """
        trace: list[TraceStep] = []
        for step in range(self.max_steps):
            prompt = _render_prompt(task, run_date, trace)
            try:
                reply = self.llm_client.complete(prompt)
            except Exception as e:  # noqa: BLE001 - record, don't crash the run
                self._record(trace, "llm_error", {}, f"LLM call failed: {e}")
                continue
            action_obj = self._parse_action(reply)
            if action_obj is None:
                self._record(
                    trace,
                    "parse_error",
                    {},
                    f"Could not parse JSON action from reply: {reply[:400]!r}",
                )
                continue
            action = action_obj.get("action")
            if action == "finalize":
                forecast = _parse_forecast(action_obj.get("forecast"))
                if forecast is None:
                    self._record(
                        trace,
                        "parse_error",
                        {},
                        "finalize had an invalid forecast mapping",
                    )
                    continue
                self._record(trace, "finalize", {}, json.dumps(action_obj["forecast"]))
                return ReActResult(forecast=forecast, trace=trace, steps=step + 1)
            if action in _TOOL_ACTIONS:
                tool = self.tools.get(str(action))
                if tool is None:
                    self._record(
                        trace,
                        str(action),
                        {},
                        f"tool '{action}' is not wired",
                    )
                    continue
                args = action_obj.get("args")
                args_dict = args if isinstance(args, dict) else {}
                try:
                    observation = tool(args_dict)
                except Exception as e:  # noqa: BLE001 - record, don't crash
                    observation = f"tool '{action}' raised: {e}"
                self._record(trace, str(action), args_dict, observation)
                continue
            self._record(
                trace,
                "unknown_action",
                {},
                f"unknown action {action!r} in reply: {reply[:400]!r}",
            )
        return ReActResult(forecast=None, trace=trace, steps=self.max_steps)

    @staticmethod
    def _parse_action(reply: str) -> dict[str, object] | None:
        """Extract the JSON action object from a raw LLM reply."""
        text = reply.strip()
        # Tolerate markdown code fences.
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.debug("ReAct parse failure on reply: %r", reply[:200])
            return None
        return parsed if isinstance(parsed, dict) else None
