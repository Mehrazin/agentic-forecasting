"""Tests for the M5 ensemble router (LLM weights, never values)."""

import numpy as np
import pytest

from td_one.agent.ensemble_router import (
    MODEL_IDS,
    MockRouter,
    fallback_weights,
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
