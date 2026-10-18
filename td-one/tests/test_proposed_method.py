"""Tests for the TD-One proposed method (exp47).

Covers: fit/predict roundtrip, gate behavior, news/LLM gating,
regime spec labeling, and the ForecastOutput contract.
"""

import numpy as np
import pytest

from td_one.agent.proposed_method import (
    ForecastOutput,
    ProposedForecaster,
    generic_regime_spec,
    m4_regime_spec,
)

FEATURE_NAMES = [
    "n_changepoints",
    "n_recent_cps",
    "trend_strength",
    "seasonal_strength",
    "vol_ratio",
    "is_stationary",
    "is_nonstationary",
    "acf1",
    "spectral_entropy",
    "hurst",
]


def _fake_record(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    feats = rng.random(len(FEATURE_NAMES)).tolist()
    # make seasonal_strength sometimes high so regimes vary
    feats[3] = float(rng.choice([0.1, 0.8]))
    feats[2] = float(rng.choice([0.2, 0.9]))
    quantiles = {
        m: (rng.random((9, 1)) * 0.2 + 0.4).tolist() for m in ["naive", "m0", "ets", "arima"]
    }
    crps = {m: float(rng.random()) for m in ["naive", "m0", "ets", "arima"]}
    return {
        "features": feats,
        "quantiles": quantiles,
        "crps": crps,
        "actual": [0.5],
    }


def _feats_dict(rec: dict) -> dict:
    return dict(zip(FEATURE_NAMES, rec["features"], strict=True))


def test_fit_predict_roundtrip():
    recs = [_fake_record(s) for s in range(60)]
    fc = ProposedForecaster(h=1, regime_spec=m4_regime_spec())
    fc.fit(recs, FEATURE_NAMES)
    r = recs[0]
    qm = {m: np.asarray(q, dtype=float) for m, q in r["quantiles"].items()}
    out = fc.predict(np.zeros(30) + 0.5, qm, _feats_dict(r), FEATURE_NAMES)
    assert isinstance(out, ForecastOutput)
    assert out.quantiles.shape == (9, 1)
    assert abs(sum(out.regime_probs.values()) - 1.0) < 1e-9
    assert out.active_expert in ["naive", "m0", "ets", "arima"]
    assert out.trust in ("EASY", "HARD")
    assert isinstance(out.explanation, str) and len(out.explanation) > 0


def test_gate_fires_on_uncertainty():
    # craft features that make the router uncertain: all regimes ~equal
    recs = [_fake_record(s) for s in range(60)]
    fc = ProposedForecaster(h=1, regime_spec=generic_regime_spec())
    fc.fit(recs, FEATURE_NAMES)
    r = recs[0]
    qm = {m: np.asarray(q, dtype=float) for m, q in r["quantiles"].items()}
    # force uncertainty: monkeypatch router to output uniform probs
    orig = fc._regime_probs

    def uniform(feats, fns, nf=None, lr=None):
        return (dict.fromkeys(["calm", "stressed"], 0.5), False, False)

    fc._regime_probs = uniform
    out = fc.predict(np.zeros(30) + 0.5, qm, _feats_dict(r), FEATURE_NAMES, t_stat=0.5, shift=0.001)
    fc._regime_probs = orig
    assert out.gated is True
    assert out.gate_reason == "uncertain_router_weak_evidence"


def test_news_gated_off_by_default():
    recs = [_fake_record(s) for s in range(60)]
    fc = ProposedForecaster(h=1, regime_spec=m4_regime_spec(), use_news=False)
    fc.fit(recs, FEATURE_NAMES)
    r = recs[0]
    qm = {m: np.asarray(q, dtype=float) for m, q in r["quantiles"].items()}
    out = fc.predict(
        np.zeros(30) + 0.5,
        qm,
        _feats_dict(r),
        FEATURE_NAMES,
        news_features={"coverage": 0.9, "stress_signal": 1.0},
    )
    assert out.news_used is False  # use_news=False -> gate closed


def test_news_gated_on_density():
    recs = [_fake_record(s) for s in range(60)]
    fc = ProposedForecaster(h=1, regime_spec=m4_regime_spec(), use_news=True)
    fc.fit(recs, FEATURE_NAMES)
    r = recs[0]
    feats = _feats_dict(r)
    p0, _, _ = fc._regime_probs(feats, FEATURE_NAMES, None, None)
    # dense news
    p1, used1, _ = fc._regime_probs(
        feats, FEATURE_NAMES, {"coverage": 0.9, "stress_signal": 1.0}, None
    )
    assert used1 is True
    # sparse news -> gate closed
    p2, used2, _ = fc._regime_probs(
        feats, FEATURE_NAMES, {"coverage": 0.1, "stress_signal": 1.0}, None
    )
    assert used2 is False
    assert p2 == p0


def test_llm_abstention():
    recs = [_fake_record(s) for s in range(60)]
    fc = ProposedForecaster(h=1, regime_spec=m4_regime_spec(), use_llm=True)
    fc.fit(recs, FEATURE_NAMES)
    feats = _feats_dict(recs[0])
    # low confidence -> abstain
    _, _, used_lo = fc._regime_probs(
        feats, FEATURE_NAMES, None, {"regime": "noisy", "confidence": 0.5}
    )
    assert used_lo is False
    # high confidence -> used
    _, _, used_hi = fc._regime_probs(
        feats, FEATURE_NAMES, None, {"regime": "noisy", "confidence": 0.9}
    )
    assert used_hi is True


def test_regime_specs_label():
    m4 = m4_regime_spec()
    assert m4.label_fn({"seasonal_strength": 0.8, "trend_strength": 0.1}) == "seasonal"
    assert m4.label_fn({"seasonal_strength": 0.1, "trend_strength": 0.9}) == "trending"
    assert m4.label_fn({"seasonal_strength": 0.1, "trend_strength": 0.1}) == "noisy"
    g = generic_regime_spec()
    assert g.label_fn({"n_recent_cps": 2, "vol_ratio": 0.5}) == "stressed"
    assert g.label_fn({"n_recent_cps": 0, "vol_ratio": 0.5}) == "calm"


def test_predict_requires_fit():
    fc = ProposedForecaster()
    with pytest.raises(AssertionError):
        fc.predict(np.zeros(30), {}, {}, FEATURE_NAMES)


def test_boc_forecaster_easy_uses_rule():
    from td_one.agent.proposed_method import BoCForecaster

    recs = [{"regime": "EASY", "rule_pred": "HIKE", "actual": "HIKE"} for _ in range(10)]
    recs += [{"regime": "HARD", "rule_pred": "HIKE", "actual": "HOLD"} for _ in range(10)]
    bf = BoCForecaster().fit(recs)
    out = bf.predict("EASY", "HIKE")
    assert out.prediction == "HIKE"
    assert out.regime == "EASY"


def test_boc_forecaster_hard_uses_climatology():
    from td_one.agent.proposed_method import BoCForecaster

    recs = [{"regime": "EASY", "rule_pred": "HIKE", "actual": "HIKE"} for _ in range(5)]
    recs += [{"regime": "HARD", "rule_pred": "HIKE", "actual": "HOLD"} for _ in range(15)]
    bf = BoCForecaster().fit(recs)
    assert bf._climatology_class == "HOLD"
    out = bf.predict("HARD", "HIKE")
    assert out.prediction == "HOLD"


def test_gate_thresholds_tunable():
    from td_one.agent.proposed_method import GATE_SHIFT, GATE_T_STAT, GATE_TOP_P

    assert GATE_TOP_P == 0.6
    assert GATE_T_STAT == 3.5
    assert GATE_SHIFT == 0.03


def test_loss_regime_spec():
    from td_one.agent.tools.regimes import loss_regime_spec

    rng = np.random.default_rng(0)
    recs = []
    for _ in range(30):
        crps = {m: float(rng.uniform(0.1, 0.3)) for m in ["naive", "m0", "ets", "arima"]}
        recs.append({"crps": crps})
    spec = loss_regime_spec(recs, ["naive", "m0", "ets", "arima"], n_regimes=3)
    assert len(spec["winner_per_regime"]) == 3
    assert all(w in ["naive", "m0", "ets", "arima"] for w in spec["winner_per_regime"])
