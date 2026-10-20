"""Tests for the anticipatory LLM shift adjuster (exp69).

Contracts under test:
- PIT: the classifier prompt contains only articles with
  article_date <= origin; future articles never leak in.
- Schema: malformed LLM output -> NULL_SIGNAL (do-no-harm).
- Overlay math: steps before predicted onset untouched; steps at/after
  shifted by exactly the calibrated delta; direction flips the sign;
  quantiles stay in [0, 1] and monotone.
- Timing: "low" confidence abstains; ramp gives partial weight.
- Calibration: recovers the known level change from synthetic episodes.
"""

from __future__ import annotations

import json

import numpy as np

from td_one.agent.anticipatory_adjuster import (
    MAGNITUDE_BUCKETS,
    NULL_SIGNAL,
    AnticipatoryClassifier,
    ShiftSignal,
    apply_overlay,
    calibrate_level_shifts,
    month_label,
)

LEVELS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
CAL = {("rate", "large"): 0.10, ("rate", "medium"): 0.05, ("policy", "large"): 0.08}


class RecordingClient:
    """Mock LLM client that records prompts and returns canned JSON."""

    def __init__(self, payload: object):
        self.payload = payload
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


def _articles() -> list[dict[str, object]]:
    return [
        {
            "title": "Rate hike signalled",
            "body": "The Bank signalled a 100bp hike taking effect in 2 months.",
            "article_date": month_label(28),
        },
        {
            "title": "Future article",
            "body": "This article is from the future and must never be seen.",
            "article_date": month_label(35),
        },
        {
            "title": "Oil prices steady",
            "body": "Unrelated filler.",
            "article_date": month_label(20),
        },
    ]


def test_classifier_pit_safe() -> None:
    client = RecordingClient(
        {
            "shift_type": "rate",
            "direction": "up",
            "magnitude_bucket": "large",
            "months_from_now": 2,
            "timing_confidence": "high",
            "rationale": "hike signalled",
        }
    )
    clf = AnticipatoryClassifier(client)
    sig = clf(_articles(), origin=29, origin_label=month_label(29))
    assert sig.is_actionable
    assert "Future article" not in client.prompts[0]
    assert "Rate hike signalled" in client.prompts[0]


def test_classifier_malformed_json_is_null() -> None:
    clf = AnticipatoryClassifier(RecordingClient("not json at all"))
    sig = clf(_articles(), origin=29, origin_label=month_label(29))
    assert sig == NULL_SIGNAL


def test_classifier_unknown_bucket_is_null() -> None:
    clf = AnticipatoryClassifier(
        RecordingClient(
            {
                "shift_type": "rate",
                "direction": "up",
                "magnitude_bucket": "gigantic",
                "months_from_now": 1,
                "timing_confidence": "high",
            }
        )
    )
    sig = clf(_articles(), origin=29, origin_label=month_label(29))
    assert sig == NULL_SIGNAL


def test_classifier_hedged_signal_is_null() -> None:
    # Type set but direction "none" -> hedging -> not actionable.
    clf = AnticipatoryClassifier(
        RecordingClient(
            {
                "shift_type": "rate",
                "direction": "none",
                "magnitude_bucket": "large",
                "months_from_now": 1,
                "timing_confidence": "high",
            }
        )
    )
    sig = clf(_articles(), origin=29, origin_label=month_label(29))
    assert sig == NULL_SIGNAL


def test_classifier_negative_timing_is_null() -> None:
    clf = AnticipatoryClassifier(
        RecordingClient(
            {
                "shift_type": "rate",
                "direction": "up",
                "magnitude_bucket": "large",
                "months_from_now": -1,
                "timing_confidence": "high",
            }
        )
    )
    sig = clf(_articles(), origin=29, origin_label=month_label(29))
    assert sig == NULL_SIGNAL


def _values(h: int = 6) -> np.ndarray:
    base = np.array([0.08, 0.10, 0.12, 0.14, 0.16])
    return np.tile(base[:, None], (1, h))


def test_overlay_step_function() -> None:
    sig = ShiftSignal("rate", "up", "large", 2, "high")
    out = apply_overlay(LEVELS, _values(), sig, origin=28, calibration=CAL)
    # Onset = 30. Months 29 (k=0) untouched; 30..34 (k>=1) shifted +0.10.
    np.testing.assert_allclose(out[:, 0], _values()[:, 0])
    np.testing.assert_allclose(out[:, 1:], _values()[:, 1:] + 0.10)


def test_overlay_direction_down() -> None:
    sig = ShiftSignal("rate", "down", "medium", 1, "high")
    out = apply_overlay(LEVELS, _values(), sig, origin=28, calibration=CAL)
    np.testing.assert_allclose(out[:, 0], _values()[:, 0] - 0.05)


def test_overlay_low_confidence_abstains() -> None:
    sig = ShiftSignal("rate", "up", "large", 1, "low")
    out = apply_overlay(LEVELS, _values(), sig, origin=28, calibration=CAL)
    np.testing.assert_allclose(out, _values())


def test_overlay_ramp_partial_weight() -> None:
    sig = ShiftSignal("rate", "up", "large", 2, "medium")
    out = apply_overlay(LEVELS, _values(), sig, origin=28, calibration=CAL)
    # Onset = 30. Ramp: month 29 (k=0) -> w=0.5; month 30+ -> w=1.
    np.testing.assert_allclose(out[:, 0], _values()[:, 0] + 0.05)
    np.testing.assert_allclose(out[:, 1:], _values()[:, 1:] + 0.10)


def test_overlay_no_ramp_ablation() -> None:
    sig = ShiftSignal("rate", "up", "large", 2, "medium")
    out = apply_overlay(LEVELS, _values(), sig, origin=28, calibration=CAL, use_ramp=False)
    np.testing.assert_allclose(out[:, 0], _values()[:, 0])
    np.testing.assert_allclose(out[:, 1:], _values()[:, 1:] + 0.10)


def test_overlay_clips_and_stays_monotone() -> None:
    vals = np.tile(np.array([0.90, 0.93, 0.95, 0.97, 0.99])[:, None], (1, 6))
    sig = ShiftSignal("rate", "up", "large", 1, "high")
    out = apply_overlay(LEVELS, vals, sig, origin=28, calibration=CAL)
    assert out.max() <= 1.0 and out.min() >= 0.0
    assert np.all(np.diff(out, axis=0) >= 0)


def test_overlay_missing_calibration_is_noop() -> None:
    sig = ShiftSignal("volatility", "up", "large", 1, "high")
    out = apply_overlay(LEVELS, _values(), sig, origin=28, calibration=CAL)
    np.testing.assert_allclose(out, _values())


def test_overlay_null_signal_is_noop() -> None:
    out = apply_overlay(LEVELS, _values(), NULL_SIGNAL, origin=28, calibration=CAL)
    np.testing.assert_allclose(out, _values())


def test_calibration_recovers_level_change() -> None:
    # Rate series: 0.10 pre-onset (30), 0.20 post.
    rate = np.concatenate([np.full(30, 0.10), np.full(18, 0.20)])
    cal = calibrate_level_shifts([("rate", "large", rate, 30)] * 3)
    assert abs(cal[("rate", "large")] - 0.10) < 1e-9
    assert set(cal) == {("rate", "large")}


def test_month_label_ordering() -> None:
    # Lexicographic order must equal chronological (PIT filter correctness).
    labels = [month_label(m) for m in range(60)]
    assert labels == sorted(labels)
    assert set(MAGNITUDE_BUCKETS) == {"small", "medium", "large"}


def test_overlay_post_origin_is_noop() -> None:
    # Origin already includes the predicted onset: the TSFM has seen the
    # shock; the overlay must not double-count (exp68 lesson).
    sig = ShiftSignal("rate", "up", "large", 0, "high")
    out = apply_overlay(LEVELS, _values(), sig, origin=31, calibration=CAL)
    np.testing.assert_allclose(out, _values())


def test_classifier_strips_markdown_fences() -> None:
    # Regression: Gemini wraps JSON in ```json fences; the classifier must
    # parse it, not fall back to NULL_SIGNAL (exp69 real-run bug).
    payload = {
        "shift_type": "rate",
        "direction": "up",
        "magnitude_bucket": "large",
        "months_from_now": 1,
        "timing_confidence": "high",
        "rationale": "hike",
    }
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    clf = AnticipatoryClassifier(RecordingClient(fenced))
    sig = clf(_articles(), origin=29, origin_label=month_label(29))
    assert sig.is_actionable
    assert sig.magnitude_bucket == "large"
    assert sig.months_from_now == 1


def test_classifier_plain_fence() -> None:
    payload = {
        "shift_type": "policy",
        "direction": "up",
        "magnitude_bucket": "medium",
        "months_from_now": 2,
        "timing_confidence": "medium",
    }
    fenced = "```\n" + json.dumps(payload) + "\n```"
    clf = AnticipatoryClassifier(RecordingClient(fenced))
    sig = clf(_articles(), origin=29, origin_label=month_label(29))
    assert sig.is_actionable
    assert sig.shift_type == "policy"


def test_classifier_parses_driving_trends() -> None:
    payload = {
        "shift_type": "rate",
        "direction": "up",
        "magnitude_bucket": "large",
        "months_from_now": 1,
        "timing_confidence": "high",
        "rationale": "hike",
        "driving_trends": ["rate hike expected", "recession risk rising", "housing cooling"],
    }
    clf = AnticipatoryClassifier(RecordingClient(payload))
    sig = clf(_articles(), origin=29, origin_label=month_label(29))
    assert sig.is_actionable
    assert sig.driving_trends == ("rate hike expected", "recession risk rising", "housing cooling")


def test_classifier_trends_capped_and_sanitized() -> None:
    payload = {
        "shift_type": "policy",
        "direction": "up",
        "magnitude_bucket": "medium",
        "months_from_now": 2,
        "timing_confidence": "medium",
        "driving_trends": ["a", "b", "c", "d", "e", "f", "g", ""],
    }
    clf = AnticipatoryClassifier(RecordingClient(payload))
    sig = clf(_articles(), origin=29, origin_label=month_label(29))
    assert sig.driving_trends == ("a", "b", "c", "d", "e")


def test_classifier_trends_default_empty() -> None:
    payload = {
        "shift_type": "none",
        "direction": "none",
        "magnitude_bucket": "none",
        "months_from_now": None,
        "timing_confidence": "low",
    }
    clf = AnticipatoryClassifier(RecordingClient(payload))
    sig = clf(_articles(), origin=29, origin_label=month_label(29))
    assert not sig.is_actionable
    assert sig.driving_trends == ()
