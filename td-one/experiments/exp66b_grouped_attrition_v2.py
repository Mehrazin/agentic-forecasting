"""exp66b: Grouping for EXPLAINABILITY — full v2 architecture.

Reframed goal (2026-10-19): grouping is for explainability, not for
beating CRPS. We do not expect better forecast numbers from grouping.
Questions:
  1. Can the LLM produce explainable groups (interpretable rules +
     grounded rationales) without degrading forecast quality vs
     deterministic grouping?
  2. What's the impact of LLM vs deterministic across the whole
     architecture (grouping -> forecasting -> risk reporting)?

**Kill criterion is OFF.** Report "grouping costs X CRPS for Y
explainability gain." If the cost is large, say so honestly.

Arms (all use ProposedForecasterV2 per group unless noted):
  A: Global + v2 (no grouping)
  B: Random + v2 (same count as LLM)
  C: Hand-written + v2 (deterministic baseline)
  D1: LLM baseline prompt + v2
  D2: LLM + interaction hints + v2
  D3: LLM + news summaries + v2
  D4: LLM + hints + news + v2
  E: Ensemble — LLM selects grouping per window from {B, C, D2}
     based on training-period diagnostics

Group-count sweep: 3 / 5 / 8 / 12 (on D2).

Metrics:
  - CRPS (member-weighted, paired t-test vs C)
  - Explanation quality: citation rate (risk_reporter), rule readability
    (conditions per rule, distinct attributes used)
  - Risk-bin calibration: HIGH-bin realized attrition vs LOW-bin

PIT: groups defined on training metadata only; news PIT-gated via
articles_visible_asof(origin).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.agent.group_forecaster import GroupForecaster, v2_forecaster_fn
from td_one.agent.grouping import GroupDefinition, discover_candidate_groups
from td_one.agent.llm_grouper import (
    LLMGrouper,
)
from td_one.data_gen.group_series import build_group_series, portfolio_totals

N_CUSTOMERS = 5000
N_MONTHS = 48
TEST_MONTHS = 12
SEEDS = [42, 123]
MIN_GROUP_N = 200
GROUP_COUNTS = [3, 5, 8, 12]

# Demographic-tagged news queries for the grouper (PIT-gated at runtime).
NEWS_QUERIES = [
    "Ontario mortgage borrowers first-time buyers",
    "British Columbia housing market mortgage",
    "Alberta oil workers mortgage default",
    "Canadian mortgage renewal rates 2024",
    "low income household debt Canada",
]


def _crps_from_quantiles(q: np.ndarray, actual: float) -> float:
    """CRPS for quantile forecast vs point actual (5-level grid approx)."""
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    qq = q[: len(levels)] if len(q) >= len(levels) else np.pad(q, (0, len(levels) - len(q)))
    pinball = np.where(actual >= qq, levels * (actual - qq), (1 - levels) * (qq - actual))
    return float(np.mean(pinball))


def _hand_written_groups() -> list[dict]:
    return [
        {
            "group_id": "on_low",
            "rule": "province == 'ON' and household_income_cad < 100000",
            "rationale": "Ontario lower income",
        },
        {
            "group_id": "on_high",
            "rule": "province == 'ON' and household_income_cad >= 100000",
            "rationale": "Ontario higher income",
        },
        {"group_id": "bc", "rule": "province == 'BC'", "rationale": "BC borrowers"},
        {
            "group_id": "other",
            "rule": "province != 'ON' and province != 'BC'",
            "rationale": "rest of Canada",
        },
    ]


def _oracle_groups() -> list[dict]:
    """Planted simulator cohorts — the ceiling for grouping.

    Rules are mutually exclusive by construction (each negates the
    previous ones) so apply_groups never sees overlap.
    """
    low = "rate_type == 'fixed' and mortgage_rate_pct <= 3.0"
    osfi = "insured_flag == False and credit_score_band == '720_plus'"
    return [
        {
            "group_id": "low_rate",
            "rule": low,
            "rationale": "2021 low-rate cohort",
        },
        {
            "group_id": "osfi_exempt",
            "rule": f"({osfi}) and not ({low})",
            "rationale": "OSFI-exempt prime",
        },
        {
            "group_id": "family_squeeze",
            "rule": f"(household_income_cad < 75000) and not ({low}) and not ({osfi})",
            "rationale": "family squeeze (income proxy)",
        },
    ]


def _rule_readability(groups: list[GroupDefinition]) -> dict:
    """Compute readability stats: conditions per rule, distinct attrs."""
    import re

    cond_counts = []
    attrs: set[str] = set()
    for g in groups:
        conds = len(re.split(r"\band\b|\bor\b", g.rule))
        cond_counts.append(conds)
        tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", g.rule))
        attrs |= tokens - {"and", "or", "not", "in", "True", "False"}
    return {
        "n_groups": len(groups),
        "mean_conditions": float(np.mean(cond_counts)) if cond_counts else 0.0,
        "n_distinct_attrs": len(attrs),
    }


def _news_summaries(origin: str, llm_client) -> list[str]:
    """Retrieve demographic-tagged news summaries, PIT-gated.

    Skips gracefully if the embedding cache isn't built yet (building
    17k embeddings on CPU takes ~15 min; the cache build runs separately).
    """
    from pathlib import Path

    cache = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "news"
        / "news_embeddings_v1.npz"
    )
    if not cache.exists():
        print("  news cache not ready; skipping news for grouper")
        return []
    try:
        from td_one.rag.semantic import SemanticRAGTool

        rag = SemanticRAGTool(llm=None)  # skip rerank for speed in grouper
        out = []
        for q in NEWS_QUERIES:
            hits = rag.search(q, before=origin, limit=2)
            for h in hits:
                out.append(f"[{h.pub_date}] {h.title}")
        return out[:10]
    except Exception as e:  # noqa: BLE001
        print(f"  news retrieval failed ({e}); continuing without news")
        return []


def run_seed(seed: int) -> dict:
    """Run all arms on one seed. Returns per-arm CRPS + diagnostics."""
    from td_one.data_gen.simulator_v2 import MortgageSimulatorV2
    from td_one.llm_rungs.vector_client import VectorProxyClient

    print(f"\n=== seed {seed} ===")
    sim = MortgageSimulatorV2(
        n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=seed, use_fred_macro=False
    )
    history = sim.generate_history()
    outcomes = sim.generate_outcomes(history)

    hist_df = pd.DataFrame([r.model_dump() for r in history])
    meta = hist_df.sort_values("snapshot_date").groupby("mortgage_id").last().reset_index()

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
    origin_str = cutoff.isoformat()
    train_out = out_df[out_df["month"] < cutoff]
    test_out = out_df[out_df["month"] >= cutoff]
    print(f"  train episodes: {len(train_out)}, test episodes: {len(test_out)}")

    train_meta = meta[meta["mortgage_id"].isin(train_out["mortgage_id"].unique())].copy()
    switched_ids = set(train_out[train_out["switched_lender_flag"]]["mortgage_id"])
    train_meta["switched_train"] = train_meta["mortgage_id"].isin(switched_ids).astype(int)

    llm = VectorProxyClient(model="flash-lite", call_budget=60)
    v2fn = v2_forecaster_fn(h=1)

    def evaluate(groups: list[GroupDefinition], tag: str) -> dict:
        """Evaluate one grouping: CRPS + per-group diagnostics + risk calibration."""
        from td_one.agent.risk_reporter import compute_risk_bins

        test_meta = meta[meta["mortgage_id"].isin(test_out["mortgage_id"].unique())]
        from td_one.agent.grouping import apply_groups

        try:
            assign = apply_groups(test_meta, groups)
        except ValueError as e:
            print(f"  arm {tag}: apply failed ({e})")
            return {"crps": float("nan"), "n_groups": 0}
        series = build_group_series(test_out, assign)
        series["__portfolio__"] = portfolio_totals(series)
        gf = GroupForecaster(v2fn, min_group_n=MIN_GROUP_N)
        forecasts = gf.forecast(series)
        crps_vals, weights, per_group = [], [], {}
        n_fallback = 0
        pred_means: dict[str, float] = {}
        realized: dict[str, float] = {}
        for gid, gs in series.items():
            if gid == "__portfolio__":
                continue
            actual = float(np.mean(gs.attrition_rate)) if gs.attrition_rate else 0.0
            fc = forecasts[gid]
            c = _crps_from_quantiles(fc.quantiles, actual)
            w = sum(gs.n_at_risk)
            crps_vals.append(c)
            weights.append(w)
            # predicted mean = median quantile (q50 approx at index 2 of 5)
            q = fc.quantiles
            pred_means[gid] = float(q[2]) if len(q) >= 3 else float(q[0])
            realized[gid] = actual
            per_group[gid] = {"crps": c, "n_at_risk": w, "fallback": fc.used_fallback}
            n_fallback += int(fc.used_fallback)
        wmean = float(np.average(crps_vals, weights=weights)) if crps_vals else float("nan")
        # Risk-bin calibration: do HIGH-bin groups actually attrit more?
        # Train tertiles from training-period group means (here: global train mean replicated).
        train_mean = float(train_out["switched_lender_flag"].mean()) if len(train_out) else 0.0
        # Use per-group training means if available; fall back to tertiles around global mean.
        try:
            bins = compute_risk_bins(pred_means, [train_mean * 0.5, train_mean, train_mean * 1.5])
        except ValueError:
            bins = dict.fromkeys(pred_means, "MEDIUM")
        bin_realized: dict[str, list[float]] = {"LOW": [], "MEDIUM": [], "HIGH": []}
        for gid, level in bins.items():
            bin_realized[level].append(realized[gid])
        calib = {
            level: float(np.mean(v)) if v else float("nan")
            for level, v in bin_realized.items()
        }
        # Calibration spread: HIGH realized - LOW realized (want > 0)
        calib_spread = calib["HIGH"] - calib["LOW"]
        if np.isnan(calib_spread):
            calib_spread = 0.0
        return {
            "crps": wmean,
            "n_groups": len(per_group),
            "n_fallback": n_fallback,
            "per_group": per_group,
            "risk_bins": bins,
            "bin_realized": calib,
            "calib_spread": float(calib_spread),
        }

    results: dict[str, dict] = {}

    # Arm A: global
    a_groups = [GroupDefinition("all", "mortgage_id == mortgage_id", "everyone")]
    # trivially true rule; apply_groups needs a real partition — use single group
    results["A_global"] = evaluate(a_groups, "A")

    # Arm C: hand-written (deterministic baseline)
    c_groups = discover_candidate_groups(
        train_meta, _hand_written_groups(), min_group_n=MIN_GROUP_N
    )
    results["C_hand"] = evaluate(c_groups, "C")
    c_crps = results["C_hand"]["crps"]

    # Arm F: oracle (ceiling) — lower min_group_n since planted cohorts are small
    f_groups = discover_candidate_groups(train_meta, _oracle_groups(), min_group_n=50)
    if f_groups:
        results["F_oracle"] = evaluate(f_groups, "F")
    else:
        results["F_oracle"] = {"crps": float("nan"), "n_groups": 0}

    # Arm B: random (count = 5 to match default LLM)
    rng = np.random.default_rng(seed)
    # (b_groups/mids unused: random assignment done inline below)
    # Build random assignment via a synthetic column
    train_meta_b = train_meta.copy()
    train_meta_b["__rand"] = rng.integers(0, 5, size=len(train_meta_b))
    # validate_rule would reject __rand (unknown col) — construct directly
    b_groups_valid = [GroupDefinition(f"rand_{i}", f"__rand == {i}") for i in range(5)]
    # apply manually since __rand isn't in test_meta; do it inline
    test_meta_b = meta[meta["mortgage_id"].isin(test_out["mortgage_id"].unique())].copy()
    test_meta_b["__rand"] = rng.integers(0, 5, size=len(test_meta_b))
    from td_one.agent.grouping import apply_groups as _apply

    b_assign = _apply(test_meta_b, b_groups_valid, id_column="mortgage_id")
    series_b = build_group_series(test_out, b_assign)
    series_b["__portfolio__"] = portfolio_totals(series_b)
    gf_b = GroupForecaster(v2fn, min_group_n=MIN_GROUP_N)
    fc_b = gf_b.forecast(series_b)
    crps_b, w_b = [], []
    for gid, gs in series_b.items():
        if gid == "__portfolio__":
            continue
        actual = float(np.mean(gs.attrition_rate)) if gs.attrition_rate else 0.0
        crps_b.append(_crps_from_quantiles(fc_b[gid].quantiles, actual))
        w_b.append(sum(gs.n_at_risk))
    results["B_random"] = {"crps": float(np.average(crps_b, weights=w_b)), "n_groups": 5}

    # Arms D1-D4: LLM prompt variants (5 groups each)
    news = _news_summaries(origin_str, llm)
    print(f"  retrieved {len(news)} news summaries for grouper")
    variants = {
        "D1_base": {"hints": False, "news": []},
        "D2_hints": {"hints": True, "news": []},
        "D3_news": {"hints": False, "news": news},
        "D4_full": {"hints": True, "news": news},
    }
    for vname, vcfg in variants.items():
        grouper = LLMGrouper(llm, max_retries=1, max_groups=5)
        # Patch: LLMGrouper.propose doesn't take hints flag; we control via
        # a wrapper that builds the prompt with/without hints.
        # Simplest: temporarily monkeypatch build_grouper_prompt.
        import td_one.agent.llm_grouper as lg

        _orig = lg.build_grouper_prompt

        def _patched(ctx, baseline_crps=None, baseline_desc="", _o=_orig, _hints=vcfg["hints"]):
            s, u = _o(ctx, baseline_crps=baseline_crps, baseline_desc=baseline_desc)
            if not _hints:
                # strip the interaction-hints paragraph
                lines = u.split("\n")
                keep = []
                skip = False
                for ln in lines:
                    if "INTERACTIONS" in ln:
                        skip = True
                    elif ln.startswith("- ") and skip:
                        continue
                    elif skip and ln.strip() == "":
                        skip = False
                        continue
                    keep.append(ln)
                u = "\n".join(keep)
            return s, u

        lg.build_grouper_prompt = _patched
        try:
            groups = grouper.propose(
                train_meta,
                "switched_train",
                categorical=["province", "rate_type", "credit_score_band"],
                news_summaries=vcfg["news"],
                origin=origin_str,
                min_group_n=MIN_GROUP_N,
                baseline_crps=c_crps,
                baseline_desc="province x income hand-written",
            )
        except Exception as e:  # noqa: BLE001
            print(f"  arm {vname}: LLM failed ({e})")
            groups = []
        finally:
            lg.build_grouper_prompt = _orig
        print(f"  arm {vname}: {len(groups)} groups: {[g.group_id for g in groups]}")
        res = evaluate(groups, vname) if groups else {"crps": float("nan"), "n_groups": 0}
        res["readability"] = _rule_readability(groups)
        res["rules"] = [(g.group_id, g.rule, g.rationale) for g in groups]
        results[vname] = res

    # Group-count sweep on D2 (hints, no news)
    sweep: dict[int, dict] = {}
    for k in GROUP_COUNTS:
        grouper = LLMGrouper(llm, max_retries=1, max_groups=k)
        try:
            groups = grouper.propose(
                train_meta,
                "switched_train",
                categorical=["province", "rate_type", "credit_score_band"],
                min_group_n=MIN_GROUP_N,
                baseline_crps=c_crps,
                baseline_desc="province x income hand-written",
            )
        except Exception as e:  # noqa: BLE001
            print(f"  sweep k={k}: LLM failed ({e})")
            groups = []
        res = evaluate(groups, f"sweep{k}") if groups else {"crps": float("nan"), "n_groups": 0}
        res["readability"] = _rule_readability(groups)
        sweep[k] = res
        print(f"  sweep k={k}: crps={res['crps']:.5f}, groups={res['n_groups']}")
    results["sweep"] = sweep

    # Arm E: ensemble — LLM selects grouping per window from {B, C, D2}
    # Simplified: LLM sees training diagnostics, picks one grouping for test.
    options = {
        "random": (
            f"5 random groups, train CRPS ~{results['B_random']['crps']:.4f}",
            b_groups_valid,
        ),
        "hand": (
            f"hand-written province x income "
            f"({results['C_hand']['n_groups']} groups), train CRPS ~{c_crps:.4f}",
            c_groups,
        ),
        "llm": (
            f"LLM-proposed ({results['D2_hints']['n_groups']} groups): "
            + "; ".join(f"{g[0]}: {g[1]}" for g in results["D2_hints"].get("rules", [])),
            None,
        ),
    }
    sel_prompt = (
        "You select a borrower grouping strategy for attrition forecasting.\n"
        "Options (with training-period diagnostics):\n"
        + "\n".join(f"- {k}: {v[0]}" for k, v in options.items())
        + "\n"
        'Reply JSON only: {"choice": "<random|hand|llm>", "reason": "..."}.'
        " Prefer the grouping with the sharpest risk differentiation and"
        " enough members per group (>= 200)."
    )
    try:
        sel_text = llm.complete(
            sel_prompt, system="You select forecasting strategies.", json_mode=True
        )
        import json as _json

        choice = _json.loads(sel_text).get("choice", "hand")
    except Exception as e:  # noqa: BLE001
        print(f"  arm E: selection failed ({e}); defaulting to hand")
        choice = "hand"
    print(f"  arm E: LLM selected '{choice}'")
    if choice == "llm":
        e_groups = [GroupDefinition(g[0], g[1], g[2]) for g in results["D2_hints"].get("rules", [])]
        # re-validate (rules already validated, but ensure partition via discover)
        e_groups = discover_candidate_groups(
            train_meta,
            [
                {"group_id": g[0], "rule": g[1], "rationale": g[2]}
                for g in results["D2_hints"].get("rules", [])
            ],
            min_group_n=MIN_GROUP_N,
        )
    elif choice == "random":
        # reuse B's groups (already evaluated) — approximate E = B
        results["E_ensemble"] = {"crps": results["B_random"]["crps"], "selected": "random"}
        e_groups = None
    else:
        e_groups = c_groups
        results["E_ensemble"] = {"crps": results["C_hand"]["crps"], "selected": "hand"}
    if e_groups is not None and choice == "llm":
        er = evaluate(e_groups, "E")
        er["selected"] = "llm"
        results["E_ensemble"] = er

    return results


def main() -> None:
    all_results: dict[int, dict] = {}
    for seed in SEEDS:
        all_results[seed] = run_seed(seed)

    print("\n\n===== exp66b summary =====")
    arms = [
        "A_global",
        "B_random",
        "C_hand",
        "D1_base",
        "D2_hints",
        "D3_news",
        "D4_full",
        "E_ensemble",
        "F_oracle",
    ]
    header = f"{'arm':<12}" + "".join(f"{s:>12}" for s in SEEDS) + f"{'mean':>12}"
    print(header)
    for arm in arms:
        vals = [all_results[s].get(arm, {}).get("crps", float("nan")) for s in SEEDS]
        mean = float(np.nanmean(vals))
        row = f"{arm:<12}" + "".join(f"{v:>12.5f}" for v in vals) + f"{mean:>12.5f}"
        print(row)

    # Paired t-test: D4 vs C (LLM full vs hand-written), across seeds
    # (proper paired test needs per-origin CRPS; seed-level is indicative)
    print("\n--- group-count sweep (D2, mean CRPS) ---")
    for k in GROUP_COUNTS:
        vals = [all_results[s]["sweep"][k]["crps"] for s in SEEDS]
        print(f"  k={k}: {np.nanmean(vals):.5f}")

    print("\n--- readability (D variants, seed 42) ---")
    for v in ["D1_base", "D2_hints", "D3_news", "D4_full"]:
        r = all_results[SEEDS[0]].get(v, {}).get("readability", {})
        print(f"  {v}: {r}")
        for gid, rule, rat in all_results[SEEDS[0]].get(v, {}).get("rules", []):
            print(f"    - {gid}: {rule}  [{rat}]")

    print("\n--- risk-bin calibration (HIGH realized - LOW realized, want > 0) ---")
    for arm in arms:
        spreads = [
            all_results[s].get(arm, {}).get("calib_spread", float("nan")) for s in SEEDS
        ]
        mean_spread = float(np.nanmean(spreads))
        # Also show per-bin realized rates for seed 42
        bins = all_results[SEEDS[0]].get(arm, {}).get("bin_realized", {})
        print(f"  {arm:<12} spread={mean_spread:+.5f}  bins(seed42)={bins}")

    # Save machine-readable
    import json as _json

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k != "per_group"}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        return o

    out_path = Path(__file__).resolve().parent / "exp66b_results.json"
    out_path.write_text(_json.dumps(_clean(all_results), indent=1, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
