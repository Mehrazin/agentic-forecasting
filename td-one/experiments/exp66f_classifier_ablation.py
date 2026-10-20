"""exp66f: LLM risk classifier ablation + risk-tool evaluation.

Follow-up to exp66d Task 2 (LLM classifier directionally better than
bins but n=10 and info-asymmetric: "bins vs bins+LLM+news").

**Ablation 1 — isolate LLM vs news** (seeds 42/123/7):
  A. bins            — deterministic training-tertile bins (no LLM)
  B. llm-stats       — LLM + aggregate stats only (no news, no tools)
  C. llm-stats-news  — LLM + stats + PIT-safe news (exp66d setup)

**Ablation 2 — risk-identification tools** (seed 42; tools from
``agent.risk_tools``):
  D. llm-stats+shap           — per-group SHAP risk drivers
  E. llm-stats+counterfactual — single-feature class-flip sensitivity
  F. llm-stats+uplift         — intervention amenability ranking
  G. llm-stats+newsalign      — semantic news retrieval (as articles)
  H. llm-stats+all-tools      — D+E+F with semantic news articles

Metric per config per seed: Spearman rank correlation between risk
level (HIGH=3/MED=2/LOW=1) and test realized attrition across groups,
plus a monotonicity flag. Pooled bootstrap 95% CI for config
differences (groups pooled across seeds; honest caveat: groups within
a seed are not independent).

PIT: all tools fit on training-period data; news gated
``pub_date < cutoff``.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp66d_cohort_recovery import _group_train_meta  # noqa: E402

from td_one.agent.deterministic_grouper import fit_deterministic_groups  # noqa: E402
from td_one.agent.llm_risk_classifier import classify_risk_llm  # noqa: E402
from td_one.agent.risk_reporter import compute_risk_bins, widen_for_uncertainty  # noqa: E402
from td_one.agent.risk_tools import (  # noqa: E402
    counterfactual_flips,
    format_counterfactuals,
    format_shap_drivers,
    news_risk_alignment,
    predicted_class,
    training_proba_cutoffs,
    uplift_amenability_text,
)
from td_one.evaluation.shap_analysis import (  # noqa: E402
    expected_log_odds,
    fit_attrition_model,
    group_shap_driver_summary,
    prepare_features,
    shap_values,
)
from td_one.evaluation.uplift import (  # noqa: E402
    fit_t_learner,
    uplift_scores,
    variable_rate_treatment,
)
from td_one.rag.semantic import SemanticNewsIndex  # noqa: E402

SEEDS = [42, 123, 7]
PER_TOOL_SEED = 42
N_CUSTOMERS = 4000
N_MONTHS = 36
TEST_MONTHS = 9
MIN_GROUP_N = 200
N_CLUSTERS = 5
GROUPER_SEED = 42
NEWS_TOP_K = 5
N_BOOTSTRAP = 2000

LEVEL_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


def _simulate(seed: int):
    """Mirror exp66d's simulate() with a seed parameter."""
    from td_one.data_gen.simulator_v2 import MortgageSimulatorV2

    sim = MortgageSimulatorV2(
        n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=seed, use_fred_macro=False
    )
    history = sim.generate_history()
    outcomes = sim.generate_outcomes(history)
    hist_df = pd.DataFrame([r.model_dump() for r in history])
    hist_df["snapshot_date"] = pd.to_datetime(hist_df["snapshot_date"])
    out_df = pd.DataFrame(
        {
            "mortgage_id": [o.mortgage_id for o in outcomes],
            "outcome_observed_at": [o.outcome_observed_at for o in outcomes],
            "switched_lender_flag": [bool(o.switched_lender_flag) for o in outcomes],
        }
    )
    out_df["month"] = pd.to_datetime(out_df["outcome_observed_at"]).dt.date.apply(
        lambda d: date(d.year, d.month, 1)
    )
    months = sorted(out_df["month"].unique())
    cutoff = months[-TEST_MONTHS]
    return hist_df, out_df, cutoff


def _training_label(train_meta: pd.DataFrame, out_df: pd.DataFrame, cutoff: date) -> np.ndarray:
    train_out = out_df[out_df["month"] < cutoff]
    switched = set(train_out[train_out["switched_lender_flag"]]["mortgage_id"].astype(str))
    return train_meta["mortgage_id"].astype(str).isin(switched).to_numpy(dtype=int)


def _spearman(levels: dict[str, str], realized: dict[str, float]) -> float:
    gids = [g for g in levels if g in realized]
    if len(gids) < 4:
        return float("nan")
    ranks = [LEVEL_RANK[levels[g]] for g in gids]
    vals = [realized[g] for g in gids]
    r, _ = spearmanr(ranks, vals)
    return float(r)


def _monotonic(levels: dict[str, str], realized: dict[str, float]) -> bool:
    agg: dict[str, list[float]] = {}
    for g, lvl in levels.items():
        if g in realized:
            agg.setdefault(lvl, []).append(realized[g])
    means = {lvl: float(np.mean(v)) for lvl, v in agg.items() if v}
    order = ["LOW", "MEDIUM", "HIGH"]
    present = [lvl for lvl in order if lvl in means]
    return all(means[present[i]] <= means[present[i + 1]] for i in range(len(present) - 1))


def _bootstrap_delta(
    obs_a: list[tuple[float, float]],
    obs_b: list[tuple[float, float]],
    n: int = N_BOOTSTRAP,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for Spearman(A) - Spearman(B) on pooled (rank, realized) pairs."""
    rng = np.random.default_rng(seed)

    def sp(obs: list[tuple[float, float]]) -> float:
        r, _ = spearmanr([o[0] for o in obs], [o[1] for o in obs])
        return float(r)

    base = sp(obs_a) - sp(obs_b)
    diffs = []
    m = min(len(obs_a), len(obs_b))
    for _ in range(n):
        ia = rng.integers(0, len(obs_a), m)
        ib = rng.integers(0, len(obs_b), m)
        diffs.append(sp([obs_a[i] for i in ia]) - sp([obs_b[i] for i in ib]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return base, float(lo), float(hi)


def _title_search_news(province: str, cutoff: date) -> list[dict]:
    index = SemanticNewsIndex()
    arts = index.search(
        f"Canada mortgage {province} interest rates", before=cutoff, top_k=NEWS_TOP_K
    )
    return [{"title": a.title, "pub_date": a.pub_date, "url": a.url} for a in arts]


def run_seed(seed: int, llm, tool_configs: bool) -> dict:
    """Run one seed: fit everything, classify under each config."""
    print(f"\n=== seed {seed} ===")
    hist_df, out_df, cutoff = _simulate(seed)
    train_meta = _group_train_meta(hist_df, cutoff)
    y = _training_label(train_meta, out_df, cutoff)
    test_out = out_df[out_df["month"] >= cutoff]
    train_out = out_df[out_df["month"] < cutoff]

    det = fit_deterministic_groups(
        train_meta, n_clusters=N_CLUSTERS, seed=GROUPER_SEED, min_group_n=MIN_GROUP_N
    )
    gids = [g.group_id for g in det.groups]
    assign = np.asarray(det.assignment.values).astype(str)
    print(f"  {len(gids)} groups")

    # test assignment: test mortgages -> group via training assignment
    mid_to_group = dict(
        zip(train_meta["mortgage_id"].astype(str), assign, strict=True)
    )
    test_assign = {
        str(mid): mid_to_group[str(mid)]
        for mid in test_out["mortgage_id"].astype(str).unique()
        if str(mid) in mid_to_group
    }
    group_test = test_out.copy()
    group_test["group_id"] = group_test["mortgage_id"].astype(str).map(test_assign)
    test_realized = {
        g: float(v)
        for g, v in group_test.groupby("group_id")["switched_lender_flag"].mean().items()
    }
    test_n = group_test.groupby("group_id").size().to_dict()

    tg = train_out.copy()
    tg["group_id"] = tg["mortgage_id"].astype(str).map(mid_to_group)
    train_realized = {
        g: float(tg.groupby("group_id")["switched_lender_flag"].mean().get(g, 0.0))
        for g in gids
    }

    # ---- deterministic bins (config A) ----
    bins = compute_risk_bins(train_realized, list(train_realized.values()))

    # ---- model + tools ----
    x, feature_names, base_map = prepare_features(train_meta)
    model = fit_attrition_model(x, y, seed=seed)
    sv = shap_values(model, x)
    ev = expected_log_odds(model)
    proba = model.predict_proba(x)[:, 1]
    cutoffs = training_proba_cutoffs(proba)
    feature_std = {
        base_map[c]: float(x[c].std())
        for c in x.columns
        if base_map[c] == c and pd.api.types.is_numeric_dtype(x[c])
    }
    treatment = variable_rate_treatment(train_meta)
    ufit = fit_t_learner(x, treatment, y, seed=seed)
    usc = uplift_scores(ufit, x)
    index = SemanticNewsIndex()

    # per-group tool sections (computed once, reused across configs)
    group_mean_uplift = {}
    for g in gids:
        mask = assign == g
        group_mean_uplift[g] = float(usc[mask].mean()) if mask.any() else 0.0
    uplift_rank = {
        g: r
        for r, g in enumerate(
            sorted(gids, key=lambda gg: -group_mean_uplift[gg]), start=1
        )
    }

    def tool_sections(g: str, rule: str, stats: dict) -> dict[str, str]:
        mask = assign == g
        sections: dict[str, str] = {}
        # SHAP drivers
        group_mean = ev + sv[mask].sum(axis=1).mean()
        summary = group_shap_driver_summary(
            sv, feature_names, mask, ev, group_mean, top_k=5, base_map=base_map
        )
        sections["shap"] = format_shap_drivers(summary)
        mean_shap = sv[mask].mean(axis=0)
        top_attrs = []
        for i in np.argsort(-np.abs(mean_shap))[:12]:
            from td_one.evaluation.shap_analysis import base_attribute

            a = base_attribute(feature_names[i], base_map)
            if a not in top_attrs:
                top_attrs.append(a)
            if len(top_attrs) == 5:
                break
        findings = counterfactual_flips(
            model, x.iloc[mask], top_attrs, base_map, feature_std, cutoffs
        )
        sections["counterfactual"] = format_counterfactuals(findings)
        # uplift amenability
        sections["uplift"] = uplift_amenability_text(
            group_mean_uplift[g], uplift_rank[g], len(gids)
        )
        return sections

    def classify(g: str, rule: str, stats: dict, articles: list[dict], extra: list[str]) -> str:
        try:
            out = classify_risk_llm(llm, g, rule, stats, articles, extra_sections=extra or None)
            return out.level
        except Exception as e:  # noqa: BLE001 — one bad group must not kill the run
            print(f"    LLM failed for {g}: {e}; defaulting MEDIUM")
            return "MEDIUM"

    levels: dict[str, dict[str, str]] = {}
    train_meta_g = train_meta.copy()
    train_meta_g["group_id"] = assign
    for g in gids:
        gd = next(gg for gg in det.groups if gg.group_id == g)
        members = train_meta_g[train_meta_g["group_id"] == g]
        stats = {
            "n_members": len(members),
            "mean_household_income_cad": round(float(members["household_income_cad"].mean()), 0),
            "dominant_province": members["province"].mode().iloc[0] if len(members) else "n/a",
            "train_realized_attrition": round(float(train_realized.get(g, 0.0)), 4),
            "predicted_mean_attrition": round(float(proba[assign == g].mean()), 4),
            "n_test_episodes": int(test_n.get(g, 0)),
        }
        province = str(stats["dominant_province"])
        try:
            news_title = _title_search_news(province, cutoff)
        except Exception:  # noqa: BLE001
            news_title = []

        cfg_levels: dict[str, str] = {}
        cfg_levels["bins"] = bins.get(g, "MEDIUM")
        cfg_levels["llm-stats"] = classify(g, gd.rule, stats, [], [])
        cfg_levels["llm-stats-news"] = classify(g, gd.rule, stats, news_title, [])
        if tool_configs:
            sec = tool_sections(g, gd.rule, stats)
            # semantic news alignment: retrieved articles become the
            # config's article list (consistent [url:N] citations)
            profile = (
                f"Borrower cohort: {gd.rule}. Predicted attrition "
                f"{stats['predicted_mean_attrition']:.3f} vs training realized "
                f"{stats['train_realized_attrition']:.3f}. Dominant province "
                f"{stats['dominant_province']}."
            )
            sem_arts = index.search(profile, before=cutoff, top_k=NEWS_TOP_K)
            sem_dicts = [
                {"title": a.title, "pub_date": a.pub_date, "url": a.url} for a in sem_arts
            ]
            cfg_levels["llm-stats+shap"] = classify(g, gd.rule, stats, [], [sec["shap"]])
            cfg_levels["llm-stats+counterfactual"] = classify(
                g, gd.rule, stats, [], [sec["counterfactual"]]
            )
            cfg_levels["llm-stats+uplift"] = classify(g, gd.rule, stats, [], [sec["uplift"]])
            cfg_levels["llm-stats+newsalign"] = classify(g, gd.rule, stats, sem_dicts, [])
            cfg_levels["llm-stats+all-tools"] = classify(
                g,
                gd.rule,
                stats,
                sem_dicts,
                [sec["shap"], sec["counterfactual"], sec["uplift"]],
            )
        levels[g] = cfg_levels
        print(f"  {g}: " + " ".join(f"{k}={v}" for k, v in cfg_levels.items()))

    # metrics per config
    metrics = {}
    for cfg in next(iter(levels.values())):
        lv = {g: levels[g][cfg] for g in gids}
        metrics[cfg] = {
            "spearman": _spearman(lv, test_realized),
            "monotonic": _monotonic(lv, test_realized),
            "n_groups": len([g for g in gids if g in test_realized]),
        }
    print(f"  metrics: {json.dumps(metrics, indent=2)}")
    return {
        "seed": seed,
        "cutoff": str(cutoff),
        "levels": levels,
        "metrics": metrics,
        "test_realized": test_realized,
    }


def main() -> None:
    from td_one.llm_rungs.vector_client import VectorProxyClient

    t0 = time.time()
    out_dir = Path(__file__).resolve().parent
    llm = VectorProxyClient(model="flash-lite", call_budget=400)
    seed_results = []
    for seed in SEEDS:
        tool_configs = seed == PER_TOOL_SEED
        seed_results.append(run_seed(seed, llm, tool_configs))

    # pooled bootstrap for key comparisons
    def pooled(cfg: str) -> list[tuple[float, float]]:
        obs = []
        for sr in seed_results:
            for g, lv in sr["levels"].items():
                if g in sr["test_realized"] and cfg in lv:
                    obs.append((LEVEL_RANK[lv[cfg]], sr["test_realized"][g]))
        return obs

    comparisons = {}
    for a, b, name in [
        ("llm-stats", "bins", "llm-stats_vs_bins"),
        ("llm-stats-news", "llm-stats", "news_effect"),
        ("llm-stats-news", "bins", "llm-stats-news_vs_bins"),
    ]:
        base, lo, hi = _bootstrap_delta(pooled(a), pooled(b))
        comparisons[name] = {"delta_spearman": base, "ci_lo": lo, "ci_hi": hi}
        print(f"  {name}: delta={base:+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}]")
    # tool configs only exist on PER_TOOL_SEED
    sr42 = next(sr for sr in seed_results if sr["seed"] == PER_TOOL_SEED)
    for cfg in ["llm-stats+shap", "llm-stats+counterfactual", "llm-stats+uplift",
                "llm-stats+newsalign", "llm-stats+all-tools"]:
        lv = {g: sr42["levels"][g][cfg] for g in sr42["levels"]}
        print(
            f"  {cfg}: spearman={_spearman(lv, sr42['test_realized']):+.3f} "
            f"monotonic={_monotonic(lv, sr42['test_realized'])}"
        )

    summary = {
        "config": {
            "seeds": SEEDS,
            "per_tool_seed": PER_TOOL_SEED,
            "n_customers": N_CUSTOMERS,
            "n_months": N_MONTHS,
        },
        "seeds": [
            {"seed": sr["seed"], "metrics": sr["metrics"], "levels": sr["levels"]}
            for sr in seed_results
        ],
        "bootstrap_comparisons": comparisons,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "exp66f_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote exp66f_results.json ({summary['elapsed_s']}s)")


if __name__ == "__main__":
    main()
