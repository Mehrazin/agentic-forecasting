"""Tests for LLM risk-identification tools."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from td_one.agent import risk_tools as rt


def test_predicted_class_boundaries():
    cutoffs = (0.3, 0.6)
    assert rt.predicted_class(0.1, cutoffs) == "LOW"
    assert rt.predicted_class(0.3, cutoffs) == "LOW"
    assert rt.predicted_class(0.5, cutoffs) == "MEDIUM"
    assert rt.predicted_class(0.9, cutoffs) == "HIGH"


def test_training_proba_cutoffs_tertiles():
    proba = np.linspace(0, 1, 301)
    t1, t2 = rt.training_proba_cutoffs(proba)
    assert t1 == pytest.approx(1 / 3, abs=0.01)
    assert t2 == pytest.approx(2 / 3, abs=0.01)


def test_format_shap_drivers():
    summary = {
        "delta_logodds": 0.42,
        "top_up": [{"feature": "rate_type", "mean_shap": 0.31}],
        "top_down": [{"feature": "household_income_cad", "mean_shap": -0.12}],
    }
    text = rt.format_shap_drivers(summary)
    assert "rate_type" in text and "+0.310" in text
    assert "household_income_cad" in text and "-0.120" in text


def _flip_model_data(seed: int = 0):
    # x0 cleanly separates classes around 0; x1 is noise.
    rng = np.random.default_rng(seed)
    n = 400
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    y = (x0 > 0).astype(int)
    x = pd.DataFrame({"x0": x0, "x1": x1})
    model = GradientBoostingClassifier(n_estimators=20, random_state=0).fit(x, y)
    return model, x


def test_counterfactual_flips_numeric():
    model, x = _flip_model_data()
    group = x[(x["x0"] > 0.2) & (x["x0"] < 0.9)].reset_index(drop=True)
    p0 = model.predict_proba(group)[:, 1].mean()
    assert p0 > 0.66  # group starts HIGH
    cutoffs = (0.33, 0.66)
    base_map = {"x0": "x0", "x1": "x1"}
    feature_std = {"x0": float(x["x0"].std()), "x1": float(x["x1"].std())}
    findings = rt.counterfactual_flips(
        model, group, ["x0", "x1"], base_map, feature_std, cutoffs, top_k=2
    )
    assert any("x0" in f and "HIGH->" in f for f in findings), findings


def test_counterfactual_flips_categorical():
    rng = np.random.default_rng(1)
    n = 400
    lvl = rng.choice(["a", "b"], n)
    y = (lvl == "a").astype(int)
    x = pd.DataFrame({"lvl_a": (lvl == "a").astype(float), "lvl_b": (lvl == "b").astype(float)})
    model = GradientBoostingClassifier(n_estimators=20, random_state=0).fit(x, y)
    group = x[x["lvl_a"] == 1.0].reset_index(drop=True)
    cutoffs = (0.33, 0.66)
    findings = rt.counterfactual_flips(
        model, group, ["lvl"], {"lvl_a": "lvl", "lvl_b": "lvl"}, {}, cutoffs, top_k=1
    )
    assert any("lvl" in f and "HIGH->LOW" in f for f in findings), findings


@dataclass
class _Hit:
    title: str
    pub_date: str
    score: float


def test_news_risk_alignment_formats_hits():
    def fake_search(query, before, top_k):
        return [_Hit("Rates rise", "2024-01-01", 0.82)]

    text = rt.news_risk_alignment("profile", fake_search, before="2024-06-01")
    assert "Rates rise" in text and "0.820" in text and "[url:0]" in text


def test_news_risk_alignment_failure_graceful():
    def bad_search(query, before, top_k):
        raise RuntimeError("no index")

    text = rt.news_risk_alignment("profile", bad_search, before="2024-06-01")
    assert "failed" in text


def test_uplift_amenability_text_ranking():
    text = rt.uplift_amenability_text(0.031, rank=2, n_groups=10)
    assert "rank 2/10" in text and "+0.0310" in text
