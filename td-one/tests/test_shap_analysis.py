"""Tests for SHAP-based attrition analysis."""

import numpy as np
import pandas as pd
import pytest

from td_one.evaluation import shap_analysis as sa


def _toy_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "mortgage_id": [f"m{i}" for i in range(n)],
            "rate_type": rng.choice(["fixed", "variable"], n),
            "credit_score_band": rng.choice(["720_plus", "650_719"], n),
            "province": rng.choice(["ON", "BC"], n),
            "household_income_cad": rng.normal(100_000, 30_000, n),
            "mortgage_rate_pct": rng.normal(4.0, 1.0, n),
            "switched_lender_flag": rng.integers(0, 2, n),
        }
    )


def test_prepare_features_excludes_outcome_and_ids():
    x, names, base_map = sa.prepare_features(_toy_df())
    assert "switched_lender_flag" not in names
    assert "mortgage_id" not in names
    assert set(base_map) == set(names)
    # determinism: same column order twice
    _, names2, _ = sa.prepare_features(_toy_df())
    assert names == names2


def test_base_attribute_handles_underscored_values():
    df = _toy_df()
    _, names, base_map = sa.prepare_features(df)
    # find the one-hot column for the 720_plus level
    col = next(c for c in names if c.startswith("credit_score_band_"))
    assert sa.base_attribute(col, base_map) == "credit_score_band"
    # numeric columns map to themselves
    assert sa.base_attribute("mortgage_rate_pct", base_map) == "mortgage_rate_pct"


def test_prepare_features_excludes_id_like_and_zero_variance():
    df = _toy_df().copy()
    df["household_id"] = [f"h{i}" for i in range(len(df))]  # ID-like: 300 unique
    df["renewal_episode_id"] = [f"e{i}" for i in range(len(df))]
    df["constant_col"] = 1  # zero variance
    x, names, _ = sa.prepare_features(df)
    assert "household_id" not in names
    assert not any(n.startswith("household_id_") for n in names)
    assert not any(n.startswith("renewal_episode_id_") for n in names)
    assert "constant_col" not in names
    assert len(names) < 40  # no one-hot explosion


def test_top_interactions_finds_planted_interaction():
    # DGP: y depends on rate_type x mortgage_rate_pct interaction only.
    rng = np.random.default_rng(1)
    n = 800
    df = pd.DataFrame(
        {
            "mortgage_id": [f"m{i}" for i in range(n)],
            "rate_type": rng.choice(["fixed", "variable"], n),
            "province": rng.choice(["ON", "BC"], n),
            "mortgage_rate_pct": rng.normal(4.0, 1.0, n),
            "household_income_cad": rng.normal(100_000, 30_000, n),
        }
    )
    logit = (df["rate_type"] == "variable").astype(float) * (df["mortgage_rate_pct"] - 4.0) * 3.0
    y = (logit + rng.normal(0, 0.5, n) > 0).astype(int)
    x, names, base_map = sa.prepare_features(df)
    model = sa.fit_attrition_model(x, y, seed=0)
    iv = sa.shap_interaction_values(model, x.iloc[:200])
    assert iv.shape == (200, len(names), len(names))
    top = sa.top_interactions(iv, names, k=5, base_map=base_map)
    pairs = {t["attrs"] for t in top}
    assert frozenset({"rate_type", "mortgage_rate_pct"}) in pairs


def test_planted_pair_recall():
    top = [
        {"attrs": frozenset({"a", "b"})},
        {"attrs": frozenset({"c", "d"})},
    ]
    # planted pairs: (rate_type, mortgage_rate_pct) etc.; none match here
    assert sa.planted_pair_recall(top, k=2) == 0.0
    top2 = [{"attrs": frozenset({"rate_type", "mortgage_rate_pct"})}]
    planted = sa.planted_interaction_pairs()
    assert frozenset({"rate_type", "mortgage_rate_pct"}) in planted
    assert sa.planted_pair_recall(top2, k=1) == pytest.approx(1 / len(planted))


def test_cluster_on_shap_deterministic():
    rng = np.random.default_rng(2)
    vals = rng.normal(size=(100, 8))
    a = sa.cluster_on_shap(vals, n_clusters=3, seed=42)
    b = sa.cluster_on_shap(vals, n_clusters=3, seed=42)
    assert (a == b).all()
    assert len(set(a)) == 3


def test_group_shap_driver_summary_signs():
    rng = np.random.default_rng(3)
    vals = rng.normal(size=(50, 4))
    names = ["f0", "f1", "f2", "f3"]
    vals[:, 0] += 2.0  # f0 pushes this group up
    summary = sa.group_shap_driver_summary(
        vals, names, np.arange(50), portfolio_mean=-1.0, group_mean=0.5
    )
    assert summary["delta_logodds"] == pytest.approx(1.5)
    assert summary["top_up"][0]["feature"] == "f0"
