"""Ensemble router agent (M5, REATS-style): the LLM as a weight router.

The literature survey (§2, §6.3) ranks this as the best-supported agentic
role: the LLM outputs *weights over classical/TSFM forecasts* with
natural-language rationale, never forecast values. This is directly
testable against inverse-CRPS weighting (exp5: 4.078).

V1 (``read_weights_real``): the LLM outputs weights directly from model
cards + trailing CRPS + scenario. CLEAN NEGATIVE in exp12 (real Groq 2.60
vs inverse-CRPS 2.32) — see docs/AGENT_AUDIT_M5.md for the post-mortem.

V2 (``read_weights_critic``): the LLM acts as a *critic* of the
inverse-CRPS baseline rather than replacing it. It sees the trailing
CRPS explicitly RANKED (so it cannot misread the ordering), the baseline
weights, and the scenario. It may AGREE (keep baseline) or ADJUST a
single model up/down by a bounded step. The adjustment is applied
mechanically with clipping. This constrains the LLM's degrees of freedom
to a single bounded intervention it must justify against the ranking.
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


# ---------------------------------------------------------------------------
# V2: critic-constrained router
# ---------------------------------------------------------------------------

#: Bounded adjustment steps for the v2 critic. The LLM may move a single
#: model's weight by one of these amounts (taken proportionally from the
#: others). Small enough that a wrong call cannot blow up the ensemble.
ADJUST_STEPS: dict[str, float] = {"small": 0.08, "medium": 0.15}

#: Hard bounds on final v2 weights. Prevents the LLM from zeroing a model
#: or concentrating excessively via repeated adjustments.
#: Feasibility: with 4 models, max + 3*min must be >= 1.0.
V2_WEIGHT_MIN = 0.05
V2_WEIGHT_MAX = 0.85


@dataclass(frozen=True)
class CriticDecision:
    """The LLM critic's verdict on the inverse-CRPS baseline.

    Attributes:
        decision: "agree" (keep baseline) or "adjust".
        model: Model id to adjust, or None if agreeing.
        direction: "up" or "down", or None if agreeing.
        strength: "small" or "medium", or None if agreeing.
        rationale: Natural-language justification (must cite the CRPS ranking).

    """

    decision: str
    model: str | None
    direction: str | None
    strength: str | None
    rationale: str


def parse_critic_decision(raw: str) -> CriticDecision:
    """Parse and validate a v2 critic decision.

    Accepts a JSON object with ``decision`` ("agree" | "adjust"),
    ``rationale`` (non-empty), and, if adjusting, ``model`` (a MODEL_IDS
    entry), ``direction`` ("up" | "down"), ``strength`` ("small" | "medium").

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
        raise ValueError(f"critic decision is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("critic decision must be a JSON object")
    decision = data.get("decision")
    if decision not in ("agree", "adjust"):
        raise ValueError(f"decision must be 'agree' or 'adjust', got {decision!r}")
    rationale = data.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")
    model = data.get("model")
    direction = data.get("direction")
    strength = data.get("strength")
    if decision == "agree":
        if model is not None or direction is not None or strength is not None:
            raise ValueError("agree must not specify model/direction/strength")
        return CriticDecision(
            decision="agree",
            model=None,
            direction=None,
            strength=None,
            rationale=rationale.strip(),
        )
    # decision == "adjust": all three fields required.
    if model not in MODEL_IDS:
        raise ValueError(f"model must be one of {MODEL_IDS}, got {model!r}")
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    if strength not in ADJUST_STEPS:
        raise ValueError(f"strength must be one of {list(ADJUST_STEPS)}, got {strength!r}")
    return CriticDecision(
        decision="adjust",
        model=model,
        direction=direction,
        strength=strength,
        rationale=rationale.strip(),
    )


def apply_critic_decision(
    baseline: dict[str, float],
    decision: CriticDecision,
) -> RouterWeights:
    """Apply a v2 critic decision to inverse-CRPS baseline weights.

    The adjustment is mechanical: move ``step`` weight to/from the target
    model, taking from (or giving to) the other models proportionally to
    their baseline weights. Final weights are clipped to
    [V2_WEIGHT_MIN, V2_WEIGHT_MAX] and renormalized.

    Args:
        baseline: Model id -> baseline weight (sums to 1).
        decision: The parsed critic decision.

    Returns:
        Validated :class:`RouterWeights`.

    """
    w = {mid: float(baseline[mid]) for mid in MODEL_IDS}
    rationale = decision.rationale
    if decision.decision == "adjust":
        assert decision.model is not None
        assert decision.direction is not None
        assert decision.strength is not None
        step = ADJUST_STEPS[decision.strength]
        target = decision.model
        others = [m for m in MODEL_IDS if m != target]
        if decision.direction == "up":
            # Take `step` from others proportionally, give to target.
            other_total = sum(w[m] for m in others)
            if other_total <= 0:
                raise ValueError("cannot adjust up: other weights sum to zero")
            take = min(step, other_total * 0.9)  # never drain others entirely
            for m in others:
                w[m] -= take * (w[m] / other_total)
            w[target] += take
        else:  # down
            # Take `step` from target, distribute to others proportionally.
            give = min(step, w[target] * 0.9)  # never zero the target
            w[target] -= give
            other_total = sum(w[m] for m in others)
            if other_total > 0:
                for m in others:
                    w[m] += give * (w[m] / other_total)
            else:
                for m in others:
                    w[m] += give / len(others)
        # Clip and renormalize (iterative: renormalization after clipping
        # can push a weight back outside bounds, so repeat until stable).
        for _ in range(10):
            for mid in MODEL_IDS:
                w[mid] = min(max(w[mid], V2_WEIGHT_MIN), V2_WEIGHT_MAX)
            total = sum(w.values())
            w = {mid: v / total for mid, v in w.items()}
            in_bounds = all(V2_WEIGHT_MIN - 1e-9 <= v <= V2_WEIGHT_MAX + 1e-9 for v in w.values())
            if in_bounds:
                break
        rationale = (
            f"[v2 critic: {decision.direction} {decision.model} {decision.strength}] {rationale}"
        )
    else:
        rationale = f"[v2 critic: agree] {rationale}"
    return RouterWeights(
        weights=tuple(w[mid] for mid in MODEL_IDS),
        rationale=rationale,
    )


CRITIC_SYSTEM = (
    "You are a forecast ensemble reviewer. You review baseline weights "
    "and decide whether a single bounded adjustment is warranted. You "
    "NEVER output forecast values or numbers about the future."
)

CRITIC_PROMPT = (
    "You review an ensemble of 4 switching-rate forecast models. "
    "The baseline weights come from inverse trailing CRPS (a strong "
    "statistical rule). Your job is to AGREE with the baseline or "
    "propose ONE bounded adjustment.\n"
    "\n"
    "Trailing mean CRPS, RANKED (lower is better):\n"
    "{ranked_crps}\n"
    "\n"
    "Baseline inverse-CRPS weights:\n"
    "{baseline_weights}\n"
    "\n"
    "Model cards (qualitative, secondary to the numbers above):\n"
    "- m0_w3: short-window Beta-Binomial. Sharpest intervals. Best in "
    "stable regimes. Slow to adapt to sudden shifts.\n"
    "- conf_w6: conformalized baseline (wider intervals). Best when "
    "uncertainty is high or regimes are volatile.\n"
    "- chronos2: pretrained TSFM. Tends to OVERCOVER (0.90-0.97). "
    "Decent point accuracy but systematically wide.\n"
    "- timesfm25: pretrained TSFM. Best calibrated. Adapts well to "
    "regime shifts.\n"
    "\n"
    "Current scenario: {description}\n"
    "\n"
    "Think step by step:\n"
    "1. Who is ranked 1st vs 4th by trailing CRPS? How big are the gaps?\n"
    "2. Does the baseline weighting reflect that ranking?\n"
    "3. Is there a SPECIFIC, evidence-backed reason to override the "
    "ranking for this scenario? (e.g., the scenario is a kind the "
    "trailing data did not cover). Vague model-card matching is NOT "
    "enough — the numbers come first.\n"
    "4. If yes: which ONE model, which direction, and how strongly? "
    "If no: AGREE.\n"
    "\n"
    'Respond with JSON only: {{"decision": "agree" or "adjust", '
    '"model": "<id or null>", "direction": "up"/"down"/null, '
    '"strength": "small"/"medium"/null, '
    '"rationale": "<cite the CRPS ranking explicitly>"}}'
)


def _ranked_crps_lines(trailing_crps: dict[str, float]) -> str:
    """Format trailing CRPS as an explicit ranking (best to worst)."""
    ranked = sorted(MODEL_IDS, key=lambda m: trailing_crps[m])
    best = trailing_crps[ranked[0]]
    lines = []
    for i, mid in enumerate(ranked, start=1):
        gap = trailing_crps[mid] - best
        lines.append(f"{i}. {mid}: {trailing_crps[mid]:.3f} (+{gap:.3f} vs best)")
    return "\n".join(lines)


def read_weights_critic(
    client: LLMClient,
    trailing_crps: dict[str, float],
    description: str,
) -> RouterWeights:
    """Get v2 critic-constrained ensemble weights from an LLM client.

    The LLM reviews the inverse-CRPS baseline and may propose a single
    bounded adjustment. The adjustment is applied mechanically.

    Args:
        client: An LLM client (e.g. :class:`GroqChatClient`).
        trailing_crps: Model id -> trailing mean CRPS.
        description: The scenario description.

    Returns:
        The final :class:`RouterWeights` (baseline ± bounded adjustment).

    """
    from td_one.models.ensemble import crps_weights

    baseline_list = crps_weights([[trailing_crps[m]] for m in MODEL_IDS])
    baseline = dict(zip(MODEL_IDS, baseline_list, strict=True))
    ranked = _ranked_crps_lines(trailing_crps)
    base_lines = "\n".join(f"- {mid}: {baseline[mid]:.3f}" for mid in MODEL_IDS)
    prompt = CRITIC_PROMPT.format(
        ranked_crps=ranked,
        baseline_weights=base_lines,
        description=description,
    )
    raw = client.complete(prompt, system=CRITIC_SYSTEM, json_mode=True)
    decision = parse_critic_decision(raw)
    return apply_critic_decision(baseline, decision)
