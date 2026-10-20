"""Tests for the agentic shift guard (exp68).

Covers: CUSUM detection timing (fires before the target moves), false-positive
control on calm DGPs, PIT safety (the guard only sees data <= asof and
PIT-filtered events), conformal widening calibration, and the LLM hook
(structure-only output).
"""

from __future__ import annotations

import numpy as np
import pytest

from td_one.agent.shift_guard import (
    Classification,
    Detection,
    LLMShiftClassifier,
    ResponseTemplate,
    ShiftGuard,
    ShiftKind,
    calibrate_widening,
    cusum_detect,
    heuristic_classifier,
    llm_classifier_adapter,
    widen_quantiles,
)


def _covariate_with_shift(n: int, t_shift: int, mag: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = 2.0 + 0.1 * rng.standard_normal(n)
    x[t_shift:] += mag
    return x


def _calm_covariate(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 2.0 + 0.1 * rng.standard_normal(n)


def _events(t_announce: int) -> list[dict]:
    return [{"t": t_announce, "type": "RATE_SHIFT", "description": "BoC policy rate hike"}]


# ---------------------------------------------------------------- detection


def test_cusum_fires_before_target_moves() -> None:
    """Covariate shifts at T-2, target at T: detection must land in between."""
    t_cov, t_target = 28, 30
    x = _covariate_with_shift(48, t_cov, 0.8, seed=7)
    det = cusum_detect(x, asof=32)
    assert det.detected
    assert det.onset_idx is not None and det.detection_idx is not None
    # Onset estimated within k=3 of the true covariate shift...
    assert abs(det.onset_idx - t_cov) <= 3
    # ...and detection fires strictly before the target moves.
    assert det.detection_idx < t_target


def test_cusum_detection_latency_bounded() -> None:
    """A 4σ covariate jump is caught within 3 periods, 9/10 seeds."""
    late = 0
    for seed in range(10):
        x = _covariate_with_shift(48, 28, 0.8, seed=seed)
        det = cusum_detect(x, asof=35)
        if not det.detected or det.detection_idx is None or det.detection_idx - 28 > 3:
            late += 1
    assert late <= 1


def test_false_positive_rate_controlled_on_calm_dgp() -> None:
    """CUSUM must stay quiet on calm covariates: FPR < 10% over 50 draws."""
    fps = 0
    for seed in range(50):
        x = _calm_covariate(48, seed=seed)
        if cusum_detect(x, asof=47).detected:
            fps += 1
    assert fps / 50 < 0.10


def test_no_detection_without_baseline() -> None:
    det = cusum_detect(np.array([1.0, 1.0]), asof=1)
    assert not det.detected


# --------------------------------------------------------------- PIT safety


def test_guard_ignores_future_events() -> None:
    """Events after asof must not change the guard's behavior (PIT gate)."""
    rng = np.random.default_rng(3)
    ctx = 0.05 + 0.005 * rng.standard_normal(40)
    cov = _covariate_with_shift(40, 30, 0.8, seed=3)
    tsfm = lambda c, h, s: (  # noqa: E731
        np.linspace(0.05, 0.95, 5),
        np.tile(np.linspace(0.04, 0.06, 5)[:, None], (1, h)),
    )
    classical = tsfm
    guard = ShiftGuard()
    events_past = _events(28)
    events_future = _events(28) + [{"t": 39, "type": "RATE_SHIFT", "description": "future hike"}]
    asof = 35
    r1 = guard.guard(ctx, cov, events_past, asof, tsfm, classical, 6, 0)
    r2 = guard.guard(ctx, cov, events_future, asof, tsfm, classical, 6, 0)
    assert r1.template == r2.template
    assert r1.context_start == r2.context_start
    assert np.array_equal(r1.values, r2.values)


def test_guard_never_indexes_beyond_asof() -> None:
    """Appending future covariate points must not change the forecast."""
    rng = np.random.default_rng(11)
    ctx = 0.05 + 0.005 * rng.standard_normal(30)
    cov = _covariate_with_shift(30, 20, 0.8, seed=11)
    seen = {}

    def tsfm(c, h, s):
        seen["len"] = len(c)
        lv = np.linspace(0.05, 0.95, 5)
        return lv, np.tile(np.linspace(0.04, 0.06, 5)[:, None], (1, h))

    guard = ShiftGuard()
    asof = 24
    guard.guard(ctx, cov, [], asof, tsfm, tsfm, 6, 0)
    assert seen["len"] == asof + 1  # exactly the visible prefix


def test_cusum_ignores_post_asof_covariate() -> None:
    """A covariate shift after asof is invisible to the detector."""
    x = _covariate_with_shift(48, 40, 2.0, seed=5)
    assert not cusum_detect(x, asof=35).detected
    assert cusum_detect(x, asof=47).detected


# ---------------------------------------------------------------- response


def test_truncate_uses_post_detection_context() -> None:
    cov = _covariate_with_shift(48, 28, 0.8, seed=7)
    rng = np.random.default_rng(7)
    ctx = 0.05 + 0.005 * rng.standard_normal(48)
    seen = {}

    def tsfm(c, h, s):
        seen["len"] = len(c)
        lv = np.linspace(0.05, 0.95, 5)
        return lv, np.tile(np.linspace(0.04, 0.06, 5)[:, None], (1, h))

    guard = ShiftGuard()
    res = guard.guard(ctx, cov, _events(26), 40, tsfm, tsfm, 6, 0)
    assert res.template == ResponseTemplate.TRUNCATE
    assert res.context_start >= 26  # post-detection
    assert seen["len"] == 41 - res.context_start


def test_widen_fallback_when_too_little_post_data() -> None:
    """Detection 1 period before origin -> WIDEN, not TRUNCATE."""
    cov = _covariate_with_shift(48, 39, 1.5, seed=7)
    ctx = np.full(48, 0.05)
    tsfm = lambda c, h, s: (  # noqa: E731
        np.linspace(0.05, 0.95, 5),
        np.tile(np.linspace(0.04, 0.06, 5)[:, None], (1, h)),
    )
    guard = ShiftGuard(widen_factor=1.5)
    res = guard.guard(ctx, cov, _events(37), 40, tsfm, tsfm, 6, 0)
    assert res.template == ResponseTemplate.WIDEN
    assert res.widen_factor == 1.5
    assert res.context_start == 0


def test_no_detection_no_action() -> None:
    cov = _calm_covariate(48, seed=1)
    ctx = np.full(48, 0.05)
    tsfm = lambda c, h, s: (  # noqa: E731
        np.linspace(0.05, 0.95, 5),
        np.tile(np.linspace(0.04, 0.06, 5)[:, None], (1, h)),
    )
    guard = ShiftGuard()
    res = guard.guard(ctx, cov, [], 40, tsfm, tsfm, 6, 0)
    assert res.template == ResponseTemplate.NONE
    assert res.widen_factor == 1.0
    assert res.context_start == 0


# ---------------------------------------------------------------- widening


def test_calibrate_widening_restores_coverage() -> None:
    """On synthetic shift residuals, the calibrated factor hits ~90% coverage."""
    rng = np.random.default_rng(0)
    # Model is overconfident by 2x under shift: |resid|/halfwidth ~ |N(0,2)|.
    scores = np.abs(rng.standard_normal(2000)) * 2.0
    q_hat = calibrate_widening(scores, np.ones_like(scores), alpha=0.1)
    assert 1.5 < q_hat < 3.5
    # Widened intervals cover ~90%.
    assert np.mean(scores <= q_hat) == pytest.approx(0.9, abs=0.02)


def test_calibrate_widening_never_shrinks() -> None:
    q_hat = calibrate_widening(np.array([0.1, 0.2]), np.array([1.0, 1.0]))
    assert q_hat == 1.0


def test_widen_quantiles_symmetric_around_median() -> None:
    lv = np.array([0.1, 0.5, 0.9])
    v = np.array([[0.04], [0.05], [0.06]])
    w = widen_quantiles(lv, v, 2.0)
    assert w[1, 0] == pytest.approx(0.05)
    assert w[0, 0] == pytest.approx(0.03)
    assert w[2, 0] == pytest.approx(0.07)


# ---------------------------------------------------------------- LLM hook


class _MockClient:
    def __init__(self, payload: str):
        self._payload = payload

    def complete(self, prompt: str) -> str:
        assert "shift_kind" in prompt  # structure-only request
        assert "90" not in prompt and "0.9" not in prompt
        return self._payload


def test_llm_hook_returns_structure_only() -> None:
    client = _MockClient(
        '{"shift_kind": "level", "template": "truncate", '
        '"confidence": "high", "rationale": "rate hike announced"}'
    )
    clf = LLMShiftClassifier(client)
    out = clf({"detection": {"detected": True}})
    assert isinstance(out, Classification)
    assert out.kind == ShiftKind.LEVEL
    assert out.template == ResponseTemplate.TRUNCATE
    assert out.confidence == "high"


def test_llm_hook_bad_json_falls_back() -> None:
    clf = LLMShiftClassifier(_MockClient("not json"))
    out = clf({})
    assert out.kind == ShiftKind.UNKNOWN
    assert out.template == ResponseTemplate.WIDEN


def test_llm_hook_rejects_unknown_values() -> None:
    clf = LLMShiftClassifier(
        _MockClient(
            '{"shift_kind": "meteor", "template": "panic", '
            '"confidence": "extreme", "rationale": "x"}'
        )
    )
    out = clf({})
    assert out.kind == ShiftKind.UNKNOWN
    assert out.template == ResponseTemplate.WIDEN
    assert out.confidence == "low"


def test_llm_adapter_signature() -> None:
    clf = LLMShiftClassifier(
        _MockClient(
            '{"shift_kind": "none", "template": "none", "confidence": "high", "rationale": "calm"}'
        )
    )
    adapted = llm_classifier_adapter(clf)
    det = cusum_detect(_calm_covariate(48, seed=2), asof=47)
    out = adapted(det, [], 0.0)
    assert out.kind == ShiftKind.NONE
    assert out.template == ResponseTemplate.NONE


def test_heuristic_classifier_template_mapping() -> None:
    det = cusum_detect(_covariate_with_shift(48, 28, 0.8, seed=7), asof=40)
    cls = heuristic_classifier(det, _events(26), 0.8)
    assert cls.kind == ShiftKind.LEVEL
    assert cls.template == ResponseTemplate.TRUNCATE
    cls2 = heuristic_classifier(Detection(False, None, None, 0.0), [], 0.0)
    assert cls2.template == ResponseTemplate.NONE
