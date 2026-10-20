"""exp66d: cohort recovery + LLM risk classifier (exp66c follow-up).

Task 1 — Cohort recovery: the simulator plants ground-truth cohorts
(``low_rate``, ``osfi_exempt``, ``family_squeeze``, ``control``).
Measure whether the deterministic grouper recovers the DGP's real
structure (ARI + per-cohort precision/recall). Low ARI => the
"explainability" is explaining noise — say so honestly.

Task 2 — LLM as at-risk classifier: deterministic training-tertile
bins were inverted on test (exp66c honest negative). Give the LLM each
group's aggregate stats + training attrition + PIT-safe news chunks and
ask for HIGH/MEDIUM/LOW. Compare LLM classes vs deterministic bins on
test-set calibration. Honest framing: if the signal does not transfer,
the LLM cannot fix it — the question is whether news/context helps.

Task 3 — Worked examples for docs/GROUPING_EXAMPLES.md.

PIT: grouper fit on training metadata; news gated ``pub_date < cutoff``;
cohort labels use origination fields (known at origination) + last
pre-maturity snapshot fields — same attributes the grouper saw.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.agent.deterministic_grouper import fit_deterministic_groups
from td_one.agent.group_forecaster import GroupForecaster, v2_forecaster_fn
from td_one.agent.llm_risk_classifier import classify_risk_llm
from td_one.agent.risk_reporter import (
    build_explanation_prompt,
    check_citations,
    compute_risk_bins,
    widen_for_uncertainty,
)
from td_one.data_gen.cohort_labels import cohort_label
from td_one.data_gen.group_series import build_group_series, portfolio_totals
from td_one.evaluation.cohort_recovery import (
    adjusted_rand_index,
    best_match_per_cohort,
)

N_CUSTOMERS = 4000
N_MONTHS = 36
TEST_MONTHS = 9
SEED = 42
MIN_GROUP_N = 200
N_CLUSTERS = 5
GROUPER_SEED = 42
NEWS_TOP_K = 5


def simulate() -> tuple[pd.DataFrame, pd.DataFrame, date]:
    """Simulate borrowers; return (history_df, outcomes_df, test cutoff)."""
    from td_one.data_gen.simulator_v2 import MortgageSimulatorV2

    sim = MortgageSimulatorV2(
        n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=SEED, use_fred_macro=False
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


def cohort_labels_for(hist_df: pd.DataFrame) -> dict[str, str]:
    """Recompute planted cohort per mortgage from history snapshots.

    Mirrors ``generate_outcomes``: origination fields from the FIRST
    snapshot, insured/credit fields from the LAST snapshot.
    """
    labels: dict[str, str] = {}
    for mid, grp in hist_df.sort_values("snapshot_date").groupby("mortgage_id"):
        orig = grp.iloc[0]
        rec = grp.iloc[-1]
        orig_year = pd.Timestamp(orig["origination_date"]).year
        labels[str(mid)] = cohort_label(
            rate_type=str(orig["rate_type"]),
            origination_year=int(orig_year),
            mortgage_rate_pct=float(orig["mortgage_rate_pct"]),
            insured_flag=bool(rec["insured_flag"]),
            credit_score_band=str(rec["credit_score_band"]),
            household_income_cad=float(orig["household_income_cad"]),
            n_children=int(orig["children_under6"]) + int(orig["children_6to17"]),
        )
    return labels


def _group_train_meta(hist_df: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Last pre-cutoff snapshot per borrower (grouper input)."""
    return (
        hist_df[hist_df["snapshot_date"] < pd.Timestamp(cutoff)]
        .sort_values("snapshot_date")
        .groupby("mortgage_id")
        .last()
        .reset_index()
    )


def _crps_from_quantiles(q: np.ndarray, actual: float) -> float:
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    qq = q[: len(levels)] if len(q) >= len(levels) else np.pad(q, (0, len(levels) - len(q)))
    pinball = np.where(actual >= qq, levels * (actual - qq), (1 - levels) * (qq - actual))
    return float(np.mean(pinball))


def run_recovery() -> dict:
    """Task 1: fit groups, measure cohort recovery."""
    print("exp66d: simulating...")
    hist_df, out_df, cutoff = simulate()
    train_meta = _group_train_meta(hist_df, cutoff)
    print(f"  train borrowers: {len(train_meta)}, cutoff: {cutoff}")

    det = fit_deterministic_groups(
        train_meta, n_clusters=N_CLUSTERS, seed=GROUPER_SEED, min_group_n=MIN_GROUP_N
    )
    print(f"  {len(det.groups)} deterministic groups")

    labels = cohort_labels_for(hist_df)
    mids = train_meta["mortgage_id"].astype(str).tolist()
    true = [labels[m] for m in mids]
    # det.assignment is indexed by train_meta's original index
    pred = [str(det.assignment.loc[idx]) for idx in train_meta.index]

    ari = adjusted_rand_index(true, pred)
    per_cohort = best_match_per_cohort(true, pred)
    cohort_counts = pd.Series(true).value_counts().to_dict()
    print(f"  ARI: {ari:.4f}")
    for c, m in per_cohort.items():
        print(
            f"  {c}: best={m['group_id']} P={m['precision']:.2f} "
            f"R={m['recall']:.2f} F1={m['f1']:.2f}"
        )
    return {
        "ari": ari,
        "per_cohort": per_cohort,
        "cohort_counts": cohort_counts,
        "hist_df": hist_df,
        "out_df": out_df,
        "cutoff": cutoff,
        "train_meta": train_meta,
        "det": det,
    }


def _news_for_group(province: str, cutoff: date) -> list[dict]:
    """PIT-safe news chunks for a group (title search, no rerank)."""
    from td_one.rag.semantic import SemanticNewsIndex

    index = SemanticNewsIndex()  # llm=None -> semantic order, no rerank
    arts = index.search(
        f"Canada mortgage {province} interest rates", before=cutoff, top_k=NEWS_TOP_K
    )
    return [{"title": a.title, "pub_date": a.pub_date, "url": a.url} for a in arts]


def run_classifier(rec: dict) -> dict:
    """Task 2: LLM risk classifier vs deterministic bins on test calibration."""
    from td_one.llm_rungs.vector_client import VectorProxyClient

    det = rec["det"]
    train_meta: pd.DataFrame = rec["train_meta"]
    out_df: pd.DataFrame = rec["out_df"]
    cutoff: date = rec["cutoff"]
    test_out = out_df[out_df["month"] >= cutoff]
    train_out = out_df[out_df["month"] < cutoff]

    # Group forecasts (v2) + per-group realized rates.
    member_of = det.assignment
    train_meta_g = train_meta.copy()
    train_meta_g["group_id"] = np.asarray(member_of.values)
    test_mids = set(test_out["mortgage_id"].astype(str))
    test_assign = {
        str(mid): str(member_of.loc[train_meta[train_meta["mortgage_id"] == mid].index[0]])
        for mid in test_mids
        if (train_meta["mortgage_id"] == mid).any()
    }
    forecaster = GroupForecaster(v2_forecaster_fn(), min_group_n=MIN_GROUP_N)
    series = build_group_series(test_out, test_assign)
    series["__portfolio__"] = portfolio_totals(series)
    forecasts = forecaster.forecast(series)
    predicted = {gid: float(np.asarray(fc.quantiles)[2]) for gid, fc in forecasts.items()}

    group_test = test_out.copy()
    group_test["group_id"] = group_test["mortgage_id"].astype(str).map(test_assign)
    test_realized = group_test.groupby("group_id")["switched_lender_flag"].mean().to_dict()

    mid_to_group = dict(
        zip(
            train_meta["mortgage_id"].astype(str),
            np.asarray(member_of.values).astype(str),
            strict=True,
        )
    )
    tg = train_out.copy()
    tg["group_id"] = tg["mortgage_id"].astype(str).map(mid_to_group)
    train_realized = {
        gid: float(tg.groupby("group_id")["switched_lender_flag"].mean().get(gid, 0.0))
        for gid in predicted
    }

    # Deterministic bins (training tertiles) — the exp66c baseline.
    bins = compute_risk_bins(train_realized, list(train_realized.values()))
    intervals = {
        gid: (
            max(0.0, float(np.asarray(forecasts[gid].quantiles)[0])),
            float(np.asarray(forecasts[gid].quantiles)[-1]),
        )
        for gid in predicted
    }
    bins = widen_for_uncertainty(bins, intervals, width_threshold=0.05)

    # LLM classifier: same stats + PIT-safe news.
    llm = VectorProxyClient(model="flash-lite", call_budget=20)
    llm_levels: dict[str, str] = {}
    llm_just: dict[str, str] = {}
    for gd in det.groups:
        gid = gd.group_id
        members = train_meta_g[train_meta_g["group_id"] == gid]
        stats = {
            "n_members": len(members),
            "mean_household_income_cad": round(float(members["household_income_cad"].mean()), 0),
            "dominant_province": members["province"].mode().iloc[0] if len(members) else "n/a",
            "train_realized_attrition": round(float(train_realized.get(gid, 0.0)), 4),
            "predicted_mean_attrition": round(float(predicted.get(gid, 0.0)), 4),
            "n_test_episodes": int((group_test["group_id"] == gid).sum()),
        }
        province = str(stats["dominant_province"])
        try:
            articles = _news_for_group(province, cutoff)
        except Exception as e:  # noqa: BLE001 — news is a bonus, not a gate
            print(f"  news failed for {gid} ({e}); continuing without")
            articles = []
        try:
            out = classify_risk_llm(llm, gid, gd.rule, stats, articles)
            llm_levels[gid] = out.level
            llm_just[gid] = out.justification
        except Exception as e:  # noqa: BLE001 — one bad group must not kill the run
            print(f"  LLM classifier failed for {gid} ({e})")
            llm_levels[gid] = "MEDIUM"
            llm_just[gid] = ""
        print(f"  {gid}: bins={bins.get(gid)} llm={llm_levels[gid]}")

    def calibration(levels: dict[str, str]) -> dict[str, float]:
        agg: dict[str, list[float]] = {}
        for gid, lvl in levels.items():
            if gid in test_realized:
                agg.setdefault(lvl, []).append(float(test_realized[gid]))
        return {lvl: float(np.mean(v)) for lvl, v in agg.items() if v}

    bins_cal = calibration(bins)
    llm_cal = calibration(llm_levels)
    print(f"  bins test calibration: {bins_cal}")
    print(f"  llm  test calibration: {llm_cal}")
    return {
        "bins": bins,
        "llm_levels": llm_levels,
        "llm_justifications": llm_just,
        "bins_test_calibration": bins_cal,
        "llm_test_calibration": llm_cal,
        "train_realized": train_realized,
        "test_realized": {k: float(v) for k, v in test_realized.items()},
        "predicted": predicted,
        "test_assign": test_assign,
        "forecasts": {k: np.asarray(v.quantiles).tolist() for k, v in forecasts.items()},
    }


def _explain_group(llm, gd, stats: dict, articles: list[dict]) -> tuple[str, bool]:
    system, user = build_explanation_prompt(gd.group_id, gd.rule, stats, articles)
    user = user + "\nEvery sentence must end with a [stat] or [url:N] citation."
    text = llm.complete(user, system=system)
    return text, check_citations(text)


def run_examples(rec: dict, clf: dict, out_dir: Path) -> list[dict]:
    """Task 3: 4 worked examples for docs/GROUPING_EXAMPLES.md."""
    from td_one.llm_rungs.vector_client import VectorProxyClient

    det = rec["det"]
    train_meta: pd.DataFrame = rec["train_meta"]
    out_df: pd.DataFrame = rec["out_df"]
    cutoff: date = rec["cutoff"]
    per_cohort: dict = rec["per_cohort"]
    test_out = out_df[out_df["month"] >= cutoff]

    member_of = det.assignment
    train_meta_g = train_meta.copy()
    train_meta_g["group_id"] = np.asarray(member_of.values)
    test_assign = clf["test_assign"]
    group_test = test_out.copy()
    group_test["group_id"] = group_test["mortgage_id"].astype(str).map(test_assign)

    # Pick: (1) best cohort mapping, (2) worst cohort mapping,
    # (3) bin/LLM disagreement, (4) highest test realized.
    best_c = max(per_cohort.items(), key=lambda kv: kv[1]["f1"])
    worst_c = min(per_cohort.items(), key=lambda kv: kv[1]["f1"])
    picks = [best_c[1]["group_id"], worst_c[1]["group_id"]]
    for gid in [g.group_id for g in det.groups]:
        if clf["bins"].get(gid) != clf["llm_levels"].get(gid) and gid not in picks:
            picks.append(gid)
            break
    top_test = max(clf["test_realized"].items(), key=lambda kv: kv[1])[0]
    if top_test not in picks:
        picks.append(top_test)
    picks = picks[:4]
    # Annotate which cohort each pick best-matches (for the doc).
    pick_cohort = {}
    for c, m in per_cohort.items():
        pick_cohort.setdefault(m["group_id"], []).append((c, m["f1"]))
    print(f"  example groups: {picks}")

    llm = VectorProxyClient(model="flash-lite", call_budget=10)
    examples = []
    for gid in picks:
        gd = next(g for g in det.groups if g.group_id == gid)
        members = train_meta_g[train_meta_g["group_id"] == gid]
        stats = {
            "n_members": len(members),
            "mean_household_income_cad": round(float(members["household_income_cad"].mean()), 0),
            "dominant_province": members["province"].mode().iloc[0] if len(members) else "n/a",
            "train_realized_attrition": round(float(clf["train_realized"].get(gid, 0.0)), 4),
            "n_test_episodes": int((group_test["group_id"] == gid).sum()),
            "realized_test_attrition": round(float(clf["test_realized"].get(gid, 0.0)), 4),
            "predicted_mean_attrition": round(float(clf["predicted"].get(gid, 0.0)), 4),
            "risk_level_bins": clf["bins"].get(gid, "MEDIUM"),
            "risk_level_llm": clf["llm_levels"].get(gid, "MEDIUM"),
        }
        try:
            articles = _news_for_group(str(stats["dominant_province"]), cutoff)
        except Exception:  # noqa: BLE001
            articles = []
        try:
            text, valid = _explain_group(llm, gd, stats, articles)
        except Exception as e:  # noqa: BLE001
            text, valid = f"(explanation failed: {e})", False
        examples.append(
            {
                "group_id": gid,
                "rule": gd.rule,
                "best_cohort_match": sorted(pick_cohort.get(gid, []), key=lambda t: -t[1]),
                "stats": stats,
                "explanation": text,
                "citations_valid": valid,
            }
        )
        print(f"  {gid}: explanation citations_valid={valid}")
    (out_dir / "exp66d_examples.json").write_text(json.dumps(examples, indent=2))
    return examples


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rec = run_recovery()
    clf = run_classifier(rec)
    examples = run_examples(rec, clf, out_dir)

    summary = {
        "config": {
            "n_customers": N_CUSTOMERS,
            "n_months": N_MONTHS,
            "test_months": TEST_MONTHS,
            "seed": SEED,
            "n_clusters": N_CLUSTERS,
            "grouper_seed": GROUPER_SEED,
        },
        "cohort_recovery": {
            "ari": rec["ari"],
            "per_cohort": rec["per_cohort"],
            "cohort_counts": rec["cohort_counts"],
        },
        "classifier": {
            "bins": clf["bins"],
            "llm_levels": clf["llm_levels"],
            "bins_test_calibration": clf["bins_test_calibration"],
            "llm_test_calibration": clf["llm_test_calibration"],
        },
        "examples": [e["group_id"] for e in examples],
    }
    (out_dir / "exp66d_results.json").write_text(json.dumps(summary, indent=2))
    print("\nwrote exp66d_results.json + exp66d_examples.json")


if __name__ == "__main__":
    main()
