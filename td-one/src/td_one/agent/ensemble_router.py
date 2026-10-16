"""Ensemble router agent (M5, REATS-style): the LLM as a weight router.

The literature survey (§2, §6.3) ranks this as the best-supported agentic
role: the LLM outputs *weights over classical/TSFM forecasts* with
natural-language rationale, never forecast values. This is directly
testable against inverse-CRPS weighting (exp5: 4.078).

The LLM sees model cards (qualitative strengths/weaknesses from our
experiments), trailing CRPS per model, and a scenario description. It
outputs weights + rationale. Classical code validates the weights and
combines the lattices via :func:`average_lattices`.

The LLM never produces a forecast number — only judgments about which
models to trust.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import numpy as np

#: Candidate model ids (must match the order used by the experiment).
MODEL_IDS: tuple[str, ...] = ("m0_w3", "conf_w6", "chronos2", "timesfm25")


@dataclass(frozen=True)
class RouterWeights:
    """LLM-routed ensemble weights.

    Attributes:
        weights: Weight per model id in ``MODEL_IDS`` order. Sums to 1.
        rationale: Natural-language justification (for audit, not math).

    """

    weights: tuple[float, ...]
    rationale: str

    def as_dict(self) -> dict[str, float]:
        """Map model id -> weight."""
        return dict(zip(MODEL_IDS, self.weights, strict=True))


def parse_router_weights(raw: str) -> RouterWeights:
    """Parse and validate LLM-routed ensemble weights.

    Accepts a JSON object with ``weights`` (object mapping model id to
    non-negative number) and ``rationale`` (non-empty string). Weights
    are normalized to sum to 1. Tolerates markdown fences.

    Args:
        raw: The LLM's raw text output.

    Returns:
        A validated :class:`RouterWeights`.

    Raises:
        ValueError: If the output is not parseable or fails validation.

    """
    text = raw.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.split("\n") if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"router weights are not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("router weights must be a JSON object")
    w_raw = data.get("weights")
    if not isinstance(w_raw, dict):
        raise ValueError("weights must be a JSON object mapping model id to number")
    weights: list[float] = []
    for mid in MODEL_IDS:
        v = w_raw.get(mid)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"weight for {mid!r} must be a number, got {v!r}")
        fv = float(v)
        if fv < 0.0 or not np.isfinite(fv):
            raise ValueError(f"weight for {mid!r} must be finite non-negative")
        weights.append(fv)
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    normalized = tuple(w / total for w in weights)
    rationale = data.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")
    return RouterWeights(weights=normalized, rationale=rationale.strip())


ROUTER_SYSTEM = (
    "You are a forecast ensemble manager. You output portfolio weights "
    "over forecasting models as JSON, with a rationale. You NEVER output "
    "forecast values or numbers about the future."
)

ROUTER_PROMPT = (
    "You manage an ensemble of 4 switching-rate forecast models. "
    "Given their recent performance and the current scenario, allocate "
    "weights (sum to 1.0). Favor models whose strengths match the scenario.\n"
    "\n"
    "Model cards:\n"
    "- m0_w3: short-window Beta-Binomial baseline. Sharpest intervals. "
    "Best single model in stable regimes (seed-0 CRPS 3.20). Slow to "
    "adapt to sudden shifts.\n"
    "- conf_w6: conformalized baseline (wider 80% intervals). Fixes "
    "undercoverage. Best when uncertainty is high or regimes are volatile.\n"
    "- chronos2: pretrained time-series foundation model. Tends to "
    "OVERCOVER (intervals too wide, 0.90-0.97 coverage). Decent point "
    "accuracy.\n"
    "- timesfm25: pretrained time-series foundation model. Best "
    "calibrated (|coverage-0.80|=0.06). Adapts well to regime shifts.\n"
    "\n"
    "Trailing mean CRPS (lower is better):\n"
    "{trailing_crps}\n"
    "\n"
    "Current scenario: {description}\n"
    "\n"
    "Think step by step:\n"
    "1. Which models' strengths match this scenario?\n"
    "2. Does trailing performance confirm or contradict the model cards?\n"
    "3. Allocate weights. It is acceptable to concentrate weight when "
    "one model clearly fits, or spread weight when uncertain.\n"
    "\n"
    'Respond with JSON only: {{"weights": '
    '{{"m0_w3": <w>, "conf_w6": <w>, "chronos2": <w>, "timesfm25": <w>}}, '
    '"rationale": "<one or two sentences>"}}'
)


class LLMClient(Protocol):
    """Minimal protocol for an LLM chat client."""

    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        """Send one chat completion; return the assistant's text."""
        ...


class MockRouter:
    """Scripted ensemble router (development only).

    Simulates an LLM that usually replicates inverse-CRPS weighting but
    sometimes applies qualitative insight: with probability
    ``insight_prob`` it tilts 60% of the weight toward the best model
    for the scenario type.

    Args:
        insight_prob: Probability of applying qualitative insight.
        seed: Random seed.

    """

    def __init__(self, insight_prob: float = 0.5, seed: int = 42) -> None:
        """Initialize the mock router."""
        if not 0.0 <= insight_prob <= 1.0:
            raise ValueError(f"insight_prob must be in [0, 1], got {insight_prob}")
        self.insight_prob = float(insight_prob)
        self.rng = np.random.default_rng(seed)

    def route(
        self,
        trailing_crps: dict[str, float],
        best_model: str,
    ) -> RouterWeights:
        """Generate scripted weights.

        Args:
            trailing_crps: Model id -> trailing mean CRPS.
            best_model: The (hidden) best model for this scenario type.

        Returns:
            A :class:`RouterWeights`.

        """
        inv = np.array([1.0 / max(trailing_crps[mid], 1e-6) for mid in MODEL_IDS])
        base = inv / inv.sum()
        if self.rng.random() < self.insight_prob:
            tilt = np.zeros(len(MODEL_IDS))
            tilt[MODEL_IDS.index(best_model)] = 0.6
            w = 0.4 * base + tilt
            w = w / w.sum()
            rationale = f"mock insight: tilted toward {best_model}"
        else:
            w = base
            rationale = "mock: inverse-CRPS weights"
        return RouterWeights(weights=tuple(float(x) for x in w), rationale=rationale)


def read_weights_real(
    client: LLMClient,
    trailing_crps: dict[str, float],
    description: str,
) -> RouterWeights:
    """Get ensemble weights from a real LLM client.

    Args:
        client: An LLM client (e.g. :class:`GroqChatClient`).
        trailing_crps: Model id -> trailing mean CRPS.
        description: The scenario description.

    Returns:
        The parsed :class:`RouterWeights`.

    """
    crps_lines = "\n".join(f"- {mid}: {trailing_crps[mid]:.3f}" for mid in MODEL_IDS)
    prompt = ROUTER_PROMPT.format(trailing_crps=crps_lines, description=description)
    raw = client.complete(prompt, system=ROUTER_SYSTEM, json_mode=True)
    return parse_router_weights(raw)


def fallback_weights() -> RouterWeights:
    """Safe fallback when the LLM output is unusable: equal weights."""
    n = len(MODEL_IDS)
    return RouterWeights(
        weights=tuple(1.0 / n for _ in range(n)),
        rationale="fallback: equal weights (LLM output unusable)",
    )
