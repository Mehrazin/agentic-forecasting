"""Tests for the M5 ensemble router (LLM weights, never values)."""

import json as _json

import numpy as np
import pytest

from td_one.agent.ensemble_router import (
    ADJUST_STEPS,
    MODEL_IDS,
    V2_WEIGHT_MAX,
    V2_WEIGHT_MIN,
    MockRouter,
    apply_critic_decision,
    fallback_weights,
    parse_critic_decision,
    parse_router_weights,
)


def _valid_json(**overrides: object) -> str:
    """Build a valid router JSON string with optional overrides."""
    import json

    payload: dict[str, object] = {
        "weights": {"m0_w3": 0.5, "conf_w6": 0.2, "chronos2": 0.1, "timesfm25": 0.2},
        "rationale": "m0_w3 fits the stable regime",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_parse_valid_weights() -> None:
    """Well-formed weights parse and normalize."""
    r = parse_router_weights(_valid_json())
    assert r.weights == pytest.approx((0.5, 0.2, 0.1, 0.2))
    assert r.as_dict()["m0_w3"] == pytest.approx(0.5)


def test_parse_normalizes() -> None:
    """Weights are normalized to sum to 1."""
    r = parse_router_weights(
        _valid_json(weights={"m0_w3": 2.0, "conf_w6": 1.0, "chronos2": 1.0, "timesfm25": 0.0})
    )
    assert sum(r.weights) == pytest.approx(1.0)
    assert r.weights[0] == pytest.approx(0.5)


def test_parse_strips_fences() -> None:
    """Markdown fences are tolerated."""
    r = parse_router_weights("```json\n" + _valid_json() + "\n```")
    assert sum(r.weights) == pytest.approx(1.0)


def test_parse_rejects_malformed() -> None:
    """Garbage raises ValueError."""
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_router_weights("m0_w3: 0.9 or something")


def test_parse_rejects_missing_model() -> None:
    """All four model ids must be present."""
    with pytest.raises(ValueError, match="must be a number"):
        parse_router_weights(_valid_json(weights={"m0_w3": 1.0, "conf_w6": 0.0, "chronos2": 0.0}))


def test_parse_rejects_negative() -> None:
    """Negative weights are rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        parse_router_weights(
            _valid_json(weights={"m0_w3": 0.5, "conf_w6": -0.1, "chronos2": 0.3, "timesfm25": 0.3})
        )


def test_parse_rejects_all_zero() -> None:
    """All-zero weights are rejected."""
    with pytest.raises(ValueError, match="positive value"):
        parse_router_weights(
            _valid_json(weights={"m0_w3": 0.0, "conf_w6": 0.0, "chronos2": 0.0, "timesfm25": 0.0})
        )


def test_parse_rejects_empty_rationale() -> None:
    """Empty rationale is rejected."""
    with pytest.raises(ValueError, match="rationale must be a non-empty string"):
        parse_router_weights(_valid_json(rationale="  "))


def test_fallback_is_equal_weights() -> None:
    """Fallback gives uniform weights summing to 1."""
    r = fallback_weights()
    assert r.weights == pytest.approx(tuple(0.25 for _ in MODEL_IDS))
    assert sum(r.weights) == pytest.approx(1.0)


def test_mock_router_no_insight_replicates_inv_crps() -> None:
    """With insight_prob=0 the mock is pure inverse-CRPS."""
    router = MockRouter(insight_prob=0.0, seed=0)
    trailing = {"m0_w3": 3.0, "conf_w6": 6.0, "chronos2": 4.0, "timesfm25": 5.0}
    r = router.route(trailing, best_model="m0_w3")
    inv = np.array([1 / 3.0, 1 / 6.0, 1 / 4.0, 1 / 5.0])
    expected = inv / inv.sum()
    assert np.allclose(r.weights, expected)


def test_mock_router_insight_tilts_to_best() -> None:
    """With insight_prob=1 the mock tilts 60% to the best model."""
    router = MockRouter(insight_prob=1.0, seed=0)
    trailing = {"m0_w3": 3.0, "conf_w6": 6.0, "chronos2": 4.0, "timesfm25": 5.0}
    r = router.route(trailing, best_model="timesfm25")
    assert r.weights[3] > 0.6  # timesfm25 gets 0.6 tilt + share of 0.4 base
    assert sum(r.weights) == pytest.approx(1.0)


def test_mock_router_rejects_bad_prob() -> None:
    """insight_prob outside [0, 1] is rejected."""
    with pytest.raises(ValueError, match="insight_prob must be in"):
        MockRouter(insight_prob=1.5)


# ---------------------------------------------------------------------------
# V2 critic-constrained router
# ---------------------------------------------------------------------------


def _critic_json(**overrides: object) -> str:
    """Build a valid v2 critic JSON string with optional overrides."""
    payload: dict[str, object] = {
        "decision": "agree",
        "model": None,
        "direction": None,
        "strength": None,
        "rationale": "timesfm25 ranked 1st at 3.328; baseline reflects this; no override warranted",
    }
    payload.update(overrides)
    return _json.dumps(payload)


def _baseline() -> dict[str, float]:
    return {"m0_w3": 0.15, "conf_w6": 0.22, "chronos2": 0.28, "timesfm25": 0.35}


def test_parse_critic_agree() -> None:
    """Agree decision parses with no adjustment fields."""
    d = parse_critic_decision(_critic_json())
    assert d.decision == "agree"
    assert d.model is None
    assert "timesfm25" in d.rationale


def test_parse_critic_adjust() -> None:
    """Adjust decision requires model/direction/strength."""
    d = parse_critic_decision(
        _critic_json(
            decision="adjust",
            model="timesfm25",
            direction="up",
            strength="small",
            rationale="timesfm25 ranked 1st; shift scenario favors it further",
        )
    )
    assert d.decision == "adjust"
    assert d.model == "timesfm25"
    assert d.direction == "up"
    assert d.strength == "small"


def test_parse_critic_rejects_bad_decision() -> None:
    """Unknown decision values are rejected."""
    with pytest.raises(ValueError, match="agree.*adjust"):
        parse_critic_decision(_critic_json(decision="maybe"))


def test_parse_critic_rejects_adjust_missing_fields() -> None:
    """Adjust without model/direction/strength is rejected."""
    with pytest.raises(ValueError, match="model must be one of"):
        parse_critic_decision(_critic_json(decision="adjust"))


def test_parse_critic_rejects_agree_with_fields() -> None:
    """Agree must not carry adjustment fields."""
    with pytest.raises(ValueError, match="must not specify"):
        parse_critic_decision(_critic_json(model="timesfm25"))


def test_parse_critic_rejects_empty_rationale() -> None:
    """Rationale is required (must cite the ranking)."""
    with pytest.raises(ValueError, match="rationale"):
        parse_critic_decision(_critic_json(rationale=""))


def test_apply_agree_keeps_baseline() -> None:
    """Agree returns the baseline weights unchanged."""
    base = _baseline()
    d = parse_critic_decision(_critic_json())
    r = apply_critic_decision(base, d)
    assert r.as_dict() == pytest.approx(base)
    assert "[v2 critic: agree]" in r.rationale


def test_apply_adjust_up_small() -> None:
    """Adjust-up moves exactly `small` weight to the target."""
    base = _baseline()
    d = parse_critic_decision(
        _critic_json(decision="adjust", model="timesfm25", direction="up", strength="small")
    )
    r = apply_critic_decision(base, d)
    out = r.as_dict()
    # timesfm25 gains `small`, others lose proportionally.
    assert out["timesfm25"] == pytest.approx(base["timesfm25"] + ADJUST_STEPS["small"])
    assert sum(r.weights) == pytest.approx(1.0)
    # Others decreased but stayed positive.
    for m in ("m0_w3", "conf_w6", "chronos2"):
        assert 0.0 < out[m] < base[m]


def test_apply_adjust_down_medium() -> None:
    """Adjust-down removes weight from the target (bounded by 0.9 guard)."""
    base = _baseline()
    d = parse_critic_decision(
        _critic_json(decision="adjust", model="m0_w3", direction="down", strength="medium")
    )
    r = apply_critic_decision(base, d)
    out = r.as_dict()
    # The 0.9 guard prevents zeroing: gives min(step, 0.9 * weight).
    expected_give = min(ADJUST_STEPS["medium"], base["m0_w3"] * 0.9)
    # After clipping (0.015 -> 0.05) and renormalization, m0_w3 is at min bound.
    assert out["m0_w3"] == pytest.approx(V2_WEIGHT_MIN, abs=0.01)
    assert sum(r.weights) == pytest.approx(1.0)
    # Others gained weight.
    for m in ("conf_w6", "chronos2", "timesfm25"):
        assert out[m] > base[m]


def test_apply_clips_to_bounds() -> None:
    """Final weights respect [V2_WEIGHT_MIN, V2_WEIGHT_MAX]."""
    # Extreme baseline: one model dominates.
    base = {"m0_w3": 0.02, "conf_w6": 0.03, "chronos2": 0.05, "timesfm25": 0.90}
    d = parse_critic_decision(
        _critic_json(decision="adjust", model="timesfm25", direction="up", strength="medium")
    )
    r = apply_critic_decision(base, d)
    for w in r.weights:
        assert V2_WEIGHT_MIN - 1e-9 <= w <= V2_WEIGHT_MAX + 1e-9
    assert sum(r.weights) == pytest.approx(1.0)


def test_apply_is_deterministic() -> None:
    """Same inputs give same outputs (no hidden randomness)."""
    base = _baseline()
    d = parse_critic_decision(
        _critic_json(decision="adjust", model="chronos2", direction="down", strength="small")
    )
    r1 = apply_critic_decision(base, d)
    r2 = apply_critic_decision(base, d)
    assert r1.weights == r2.weights
    assert r1.rationale == r2.rationale


def test_critic_decision_is_frozen() -> None:
    """CriticDecision is immutable (audit trail integrity)."""
    d = parse_critic_decision(_critic_json())
    with pytest.raises(AttributeError):
        d.decision = "adjust"  # type: ignore[misc]
