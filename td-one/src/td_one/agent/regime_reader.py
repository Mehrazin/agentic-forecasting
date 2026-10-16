"""Regime-reader agent (M4): the LLM as a noisy sensor, never a forecaster.

Redesigns the agent's role after the honest-M2 negative
(``experiments/exp3_results_honest_backend.json``): the ReAct agent
overrides decent forecasts with worse judgment. The M4 agent never
emits a number that touches a forecast. It outputs a structured regime
reading — a class, a confidence, and a reason — and all numerical work
(the abstention decision, the delta magnitude, the forecast itself)
lives in tested classical code (``td_one.models.regime_blend``).

The key mechanism is *abstention*: the LLM is allowed to say "I'm not
sure", and low-confidence readings are ignored. This directly targets
exp7's failure mode, where hard UP/DOWN/NONE classification at 67%
accuracy produced catastrophic wrong-direction errors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import numpy as np

RegimeClass = Literal["UP", "DOWN", "NONE"]

#: Valid regime classes.
REGIMES: tuple[str, str, str] = ("UP", "DOWN", "NONE")


@dataclass(frozen=True)
class RegimeReading:
    """A structured regime judgment from the LLM.

    Attributes:
        regime: Predicted direction of the switching-rate shift.
        confidence: Self-reported confidence in [0, 1].
        reason: One-sentence justification (for audit, not for math).

    """

    regime: RegimeClass
    confidence: float
    reason: str


def parse_regime_reading(raw: str) -> RegimeReading:
    """Parse and validate an LLM regime reading.

    Accepts a JSON object with ``regime`` ("UP"|"DOWN"|"NONE"),
    ``confidence`` (number in [0, 1]), and ``reason`` (string).
    Tolerates markdown code fences and surrounding whitespace.

    Args:
        raw: The LLM's raw text output.

    Returns:
        A validated :class:`RegimeReading`.

    Raises:
        ValueError: If the output is not parseable or fails validation.

    """
    text = raw.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        lines = text.split("\n")
        # Drop the opening fence line and the closing fence line.
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"regime reading is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("regime reading must be a JSON object")
    regime = cast(RegimeClass, data.get("regime"))
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"confidence must be a number, got {confidence!r}")
    confidence_f = float(confidence)
    if not 0.0 <= confidence_f <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence_f}")
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    return RegimeReading(
        regime=regime,
        confidence=confidence_f,
        reason=reason.strip(),
    )


REGIME_SYSTEM = (
    "You are a mortgage market analyst. You output structured regime "
    "judgments as JSON. You NEVER output numerical forecasts or deltas."
)

REGIME_PROMPT = (
    "Given the macro scenario below and the current switching rate, "
    "judge the expected DIRECTION of the borrower switching-rate shift "
    "and state your confidence.\n"
    "\n"
    "Scenario: {description}\n"
    "Current switching rate: {base_rate:.3f}\n"
    "\n"
    "Think step by step:\n"
    "1. Does this scenario make switching MORE attractive (borrowers "
    "chase better deals), LESS attractive (barriers rise), or is there "
    "no material change?\n"
    "2. Note: policies that make switching HARDER (re-qualification, "
    "tighter rules) REDUCE switching even if they sound restrictive.\n"
    "3. How confident are you? Report LOW confidence (<0.5) when the "
    "scenario is ambiguous or the mechanism is unclear. It is always "
    "acceptable to report low confidence.\n"
    "\n"
    'Respond with JSON only: {{"regime": "UP" | "DOWN" | "NONE", '
    '"confidence": <0.0-1.0>, "reason": "<one sentence>"}}'
)


class MockRegimeReader:
    """Scripted regime reader with a calibrated error model (development only).

    Simulates an LLM whose confidence is *calibrated*: correct predictions
    draw confidence from Beta(8, 2) (mean 0.8) and wrong predictions from
    Beta(2, 5) (mean ~0.29). Set ``calibrated=False`` to simulate a
    miscalibrated LLM (confidence uniform, uncorrelated with correctness).

    Args:
        accuracy: Probability of predicting the true class.
        calibrated: Whether confidence correlates with correctness.
        seed: Random seed.

    """

    def __init__(self, accuracy: float = 0.67, calibrated: bool = True, seed: int = 42) -> None:
        """Initialize the mock reader."""
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"accuracy must be in [0, 1], got {accuracy}")
        self.accuracy = float(accuracy)
        self.calibrated = bool(calibrated)
        self.rng = np.random.default_rng(seed)

    def read(self, true_class: RegimeClass) -> RegimeReading:
        """Generate a scripted reading for a scenario with known truth."""
        if true_class not in REGIMES:
            raise ValueError(f"unknown true class {true_class!r}")
        if self.rng.random() < self.accuracy:
            regime: RegimeClass = true_class
            correct = True
        else:
            wrong = [c for c in REGIMES if c != true_class]
            regime = str(self.rng.choice(wrong))  # type: ignore[assignment]
            correct = False
        if self.calibrated:
            confidence = float(self.rng.beta(8, 2) if correct else self.rng.beta(2, 5))
        else:
            confidence = float(self.rng.uniform(0.3, 0.9))
        return RegimeReading(
            regime=regime,
            confidence=confidence,
            reason=f"mock reading (correct={correct})",
        )


class LLMClient(Protocol):
    """Minimal protocol for an LLM chat client."""

    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        """Send one chat completion; return the assistant's text."""
        ...


def read_regime_real(
    client: LLMClient,
    description: str,
    base_rate: float,
) -> RegimeReading:
    """Get a regime reading from a real LLM client.

    Args:
        client: An LLM client (e.g. :class:`GroqChatClient`).
        description: The scenario description.
        base_rate: The current switching rate.

    Returns:
        The parsed :class:`RegimeReading`.

    Raises:
        ValueError: If the LLM output fails validation.
        RuntimeError: If the LLM call fails.

    """
    prompt = REGIME_PROMPT.format(description=description, base_rate=base_rate)
    raw = client.complete(prompt, system=REGIME_SYSTEM, json_mode=True)
    return parse_regime_reading(raw)
