"""Tests for the M4 regime-reader agent (LLM as sensor, never forecaster)."""

import numpy as np
import pytest

from td_one.agent.regime_reader import (
    MockRegimeReader,
    RegimeReading,
    parse_regime_reading,
)
from td_one.models.regime_blend import (
    CLASS_DELTAS,
    _regime_class_for_shift,
    apply_regime_reading,
    should_intervene,
)

# ---------------------------------------------------------------------------
# parse_regime_reading
# ---------------------------------------------------------------------------


def test_parse_valid_reading() -> None:
    """Well-formed JSON parses to a RegimeReading."""
    r = parse_regime_reading('{"regime": "UP", "confidence": 0.85, "reason": "rates up"}')
    assert r == RegimeReading(regime="UP", confidence=0.85, reason="rates up")


def test_parse_strips_markdown_fences() -> None:
    """Markdown code fences are tolerated."""
    r = parse_regime_reading('```json\n{"regime": "DOWN", "confidence": 0.6, "reason": "x"}\n```')
    assert r.regime == "DOWN"
    assert r.confidence == pytest.approx(0.6)


def test_parse_rejects_malformed_json() -> None:
    """Garbage text raises ValueError."""
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_regime_reading("UP, I think, maybe 0.8")


def test_parse_rejects_non_object() -> None:
    """A JSON array is not a reading."""
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_regime_reading('["UP", 0.8]')


def test_parse_rejects_bad_regime() -> None:
    """Unknown regime classes are rejected."""
    with pytest.raises(ValueError, match="regime must be one of"):
        parse_regime_reading('{"regime": "SIDEWAYS", "confidence": 0.5, "reason": "x"}')


def test_parse_rejects_confidence_out_of_range() -> None:
    """Confidence outside [0, 1] is rejected."""
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="confidence must be in"):
            parse_regime_reading(f'{{"regime": "UP", "confidence": {bad}, "reason": "x"}}')


def test_parse_rejects_bool_confidence() -> None:
    """Bool is not a valid confidence (bool is a subclass of int)."""
    with pytest.raises(ValueError, match="confidence must be a number"):
        parse_regime_reading('{"regime": "UP", "confidence": true, "reason": "x"}')


def test_parse_rejects_empty_reason() -> None:
    """Empty reasons are rejected (audit trail matters)."""
    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        parse_regime_reading('{"regime": "UP", "confidence": 0.5, "reason": "  "}')


def test_parse_accepts_boundary_confidences() -> None:
    """0.0 and 1.0 are valid confidences."""
    for c in (0.0, 1.0):
        r = parse_regime_reading(f'{{"regime": "NONE", "confidence": {c}, "reason": "x"}}')
        assert r.confidence == c


# ---------------------------------------------------------------------------
# should_intervene
# ---------------------------------------------------------------------------


def _shifted_panel() -> tuple[np.ndarray, np.ndarray]:
    """Panel with a clear recent upshift (z >> 0)."""
    n = np.full(12, 200)
    a = np.full(12, 20)
    a[-3:] = 60  # recent tripling
    return n, a


def _flat_panel() -> tuple[np.ndarray, np.ndarray]:
    """Stationary panel (z ~ 0)."""
    n = np.full(12, 200)
    a = np.full(12, 20)
    return n, a


def test_none_never_intervenes() -> None:
    """A NONE reading is an explicit do-nothing, even at confidence 1."""
    n, a = _shifted_panel()
    reading = RegimeReading(regime="NONE", confidence=1.0, reason="flat")
    assert not should_intervene(reading, n, a)


def test_low_confidence_abstains() -> None:
    """Below-threshold confidence abstains even with a panel shift."""
    n, a = _shifted_panel()
    reading = RegimeReading(regime="UP", confidence=0.4, reason="unsure")
    assert not should_intervene(reading, n, a, confidence_threshold=0.7)


def test_confident_reading_with_panel_shift_intervenes() -> None:
    """Confident UP + corroborating panel -> intervene."""
    n, a = _shifted_panel()
    reading = RegimeReading(regime="UP", confidence=0.9, reason="rates up")
    assert should_intervene(reading, n, a)


def test_panel_gate_blocks_uncorroborated_reading() -> None:
    """Confident reading on a flat panel is blocked by the panel gate."""
    n, a = _flat_panel()
    reading = RegimeReading(regime="UP", confidence=0.95, reason="macro")
    assert not should_intervene(reading, n, a, use_panel_gate=True)
    # ... but passes when the panel gate is disabled.
    assert should_intervene(reading, n, a, use_panel_gate=False)


# ---------------------------------------------------------------------------
# apply_regime_reading
# ---------------------------------------------------------------------------


def test_abstain_returns_m0_forecast() -> None:
    """Abstention -> delta 0, intervened False, rate == M0 rate."""
    n, a = _flat_panel()
    reading = RegimeReading(regime="UP", confidence=0.2, reason="unsure")
    fc = apply_regime_reading(n, a, 200, reading)
    assert not fc["intervened"]
    assert fc["delta"] == 0.0
    assert fc["adjusted_rate"] == pytest.approx(fc["base_rate"])


def test_intervention_applies_class_delta() -> None:
    """Intervention applies the data-calibrated class delta, not an LLM number."""
    n, a = _shifted_panel()
    reading = RegimeReading(regime="DOWN", confidence=0.9, reason="cuts")
    fc = apply_regime_reading(n, a, 200, reading)
    assert fc["intervened"]
    assert fc["delta"] == CLASS_DELTAS["DOWN"] == -0.25
    assert fc["adjusted_rate"] < fc["base_rate"]


def test_forecast_samples_shape() -> None:
    """Samples are valid count draws."""
    n, a = _shifted_panel()
    reading = RegimeReading(regime="UP", confidence=0.9, reason="x")
    fc = apply_regime_reading(n, a, 200, reading, seed=1)
    assert fc["samples"].shape == (1000,)
    assert np.all((fc["samples"] >= 0) & (fc["samples"] <= 200))
    assert fc["q10"] <= fc["q50"] <= fc["q90"]


# ---------------------------------------------------------------------------
# MockRegimeReader
# ---------------------------------------------------------------------------


def test_mock_reader_rejects_bad_accuracy() -> None:
    """Accuracy outside [0, 1] is rejected."""
    with pytest.raises(ValueError, match="accuracy must be in"):
        MockRegimeReader(accuracy=1.5)


def test_mock_reader_calibration() -> None:
    """Calibrated mock: correct predictions are more confident on average."""
    reader = MockRegimeReader(accuracy=0.7, calibrated=True, seed=0)
    correct_conf, wrong_conf = [], []
    for _ in range(400):
        r = reader.read("UP")
        # Reconstruct correctness from the mock's reason string.
        if "correct=True" in r.reason:
            correct_conf.append(r.confidence)
        else:
            wrong_conf.append(r.confidence)
    assert len(correct_conf) > 200  # ~70% of 400
    assert float(np.mean(correct_conf)) > float(np.mean(wrong_conf)) + 0.2


def test_mock_reader_miscalibrated_confidence_flat() -> None:
    """Miscalibrated mock: confidence uncorrelated with correctness."""
    reader = MockRegimeReader(accuracy=0.7, calibrated=False, seed=0)
    correct_conf, wrong_conf = [], []
    for _ in range(400):
        r = reader.read("UP")
        if "correct=True" in r.reason:
            correct_conf.append(r.confidence)
        else:
            wrong_conf.append(r.confidence)
    gap = abs(float(np.mean(correct_conf)) - float(np.mean(wrong_conf)))
    assert gap < 0.15


def test_regime_class_for_shift() -> None:
    """True-shift -> class mapping used by mock experiments."""
    assert _regime_class_for_shift(0.6) == "UP"
    assert _regime_class_for_shift(-0.5) == "DOWN"
    assert _regime_class_for_shift(0.0) == "NONE"
    assert _regime_class_for_shift(0.04) == "NONE"  # dead zone
