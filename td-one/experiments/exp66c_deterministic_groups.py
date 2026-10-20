"""exp66c: deterministic grouping + LLM explanations (the exp66b pivot).

exp66b found LLM-proposed groupings are not reproducible run-to-run,
undermining their use for explainability. This experiment implements
the pivot: deterministic clustering (kmeans + tree-surrogate rules)
for group *discovery*, LLM (via risk_reporter) for group
*explanation*.

Arms (all per-group ProposedForecasterV2 unless noted):
  A: Global (no grouping)
  B: Random (same count as G)
  C: Hand-written (deterministic baseline)
  G: Deterministic kmeans+tree groups

Metrics:
  - CRPS (member-weighted, per seed)
  - Explanation quality (arm G): citation pass rate, conditions per
    rule, distinct attributes, human rubric scores
  - Risk-bin calibration: HIGH-bin realized vs LOW-bin realized

PIT: grouper fit on training metadata only; rules applied to test
rows. No outcome columns enter grouping.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.agent.deterministic_grouper import fit_deterministic_groups
from td_one.agent.group_forecaster import GroupForecaster, v2_forecaster_fn
from td_one.agent.grouping import GroupDefinition, discover_candidate_groups
from td_one.agent.risk_reporter import (
    build_explanation_prompt,
    check_citations,
    compute_risk_bins,
    widen_for_uncertainty,
)
from td_one.data_gen.group_series import build_group_series, portfolio_totals

N_CUSTOMERS = 4000
N_MONTHS = 36
TEST_MONTHS = 9
SEEDS = [42, 123]
MIN_GROUP_N = 200
N_CLUSTERS = 5
GROUPER_SEED = 42


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


def _assign_by_rules(
    test_df: pd.DataFrame,
    groups: list[GroupDefinition],
    id_column: str = "mortgage_id",
) -> dict[str, str]:
    """Vectorized rule application; every row must match exactly one rule."""
    assign: dict[str, str] = {}
    for g in groups:
        for mid in test_df.query(g.rule)[id_column].astype(str):
            if mid in assign:
                raise ValueError(f"mortgage {mid} matches multiple groups")
            assign[mid] = g.group_id
    missing = set(test_df[id_column].astype(str)) - set(assign)
    if missing:
        raise ValueError(f"{len(missing)} rows match no group")
    return assign


def _arm_crps(
    test_out: pd.DataFrame,
    assign: dict[str, str],
    forecaster: GroupForecaster,
) -> tuple[float, dict[str, float], dict[str, np.ndarray]]:
    """Member-weighted CRPS + per-group CRPS + per-group quantile forecasts."""
    series = build_group_series(test_out, assign)
    series["__portfolio__"] = portfolio_totals(series)
    forecasts = forecaster.forecast(series)
    per_group: dict[str, float] = {}
    per_q: dict[str, np.ndarray] = {}
    wvals, weights = [], []
    for gid, gs in series.items():
        if gid == "__portfolio__":
            continue
        actual = float(np.mean(gs.attrition_rate)) if gs.attrition_rate else 0.0
        c = _crps_from_quantiles(forecasts[gid].quantiles, actual)
        per_group[gid] = c
        per_q[gid] = np.asarray(forecasts[gid].quantiles, dtype=float)
        w = sum(gs.n_at_risk)
        wvals.append(c)
        weights.append(w)
    return float(np.average(wvals, weights=weights)), per_group, per_q


def run_seed(seed: int) -> dict:
    from td_one.data_gen.simulator_v2 import MortgageSimulatorV2

    print(f"\n=== seed {seed} ===")
    sim = MortgageSimulatorV2(
        n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=seed, use_fred_macro=False
    )
    history = sim.generate_history()
    outcomes = sim.generate_outcomes(history)

    hist_df = pd.DataFrame([r.model_dump() for r in history])

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
    # PIT-clean training metadata: last snapshot per borrower with
    # snapshot_date strictly before the test cutoff. Group definitions
    # must not see test-period attributes.
    hist_df["snapshot_date"] = pd.to_datetime(hist_df["snapshot_date"])
    meta_train = (
        hist_df[hist_df["snapshot_date"] < pd.Timestamp(cutoff)]
        .sort_values("snapshot_date")
        .groupby("mortgage_id")
        .last()
        .reset_index()
    )
    train_out = out_df[out_df["month"] < cutoff]
    test_out = out_df[out_df["month"] >= cutoff]
    test_mids = test_out["mortgage_id"].unique()
    # Grouper fit on the FULL training portfolio (all borrowers), not
    # switchers-only: groups partition the portfolio. Assignments for
    # test borrowers use their training-period attributes (known at
    # origin) — PIT-clean.
    train_meta = meta_train.copy()
    test_meta = meta_train[meta_train["mortgage_id"].isin(test_mids)].copy()
    print(f"  train episodes: {len(train_out)}, test episodes: {len(test_out)}")
    print(f"  train borrowers: {len(train_meta)}, test borrowers: {len(test_meta)}")

    forecaster = GroupForecaster(v2_forecaster_fn(), min_group_n=MIN_GROUP_N)

    # Arm G: deterministic groups (fit on TRAIN meta only).
    det = fit_deterministic_groups(
        train_meta,
        n_clusters=N_CLUSTERS,
        seed=GROUPER_SEED,
        min_group_n=MIN_GROUP_N,
    )
    print(f"  arm G: {len(det.groups)} deterministic groups")

    # Arm C: hand-written.
    c_groups = discover_candidate_groups(
        train_meta, _hand_written_groups(), min_group_n=MIN_GROUP_N
    )

    # Arm B: random, same count as G.
    rng = np.random.default_rng(seed)
    b_assign = {mid: f"rand_{rng.integers(0, len(det.groups))}" for mid in test_mids}

    results: dict = {"seed": seed, "arms": {}}
    g_assign = _assign_by_rules(test_meta, det.groups)
    arms: dict[str, dict[str, str]] = {
        "A_global": dict.fromkeys(test_mids, "all"),
        "B_random": b_assign,
        "C_hand": _assign_by_rules(test_meta, c_groups),
        "G_deterministic": g_assign,
    }
    g_quantiles: dict[str, np.ndarray] = {}
    for name, assign in arms.items():
        wmean, per_group, per_q = _arm_crps(test_out, assign, forecaster)
        results["arms"][name] = {"weighted_crps": wmean, "per_group": per_group}
        if name == "G_deterministic":
            g_quantiles = per_q
        print(f"  arm {name}: weighted CRPS = {wmean:.5f} ({len(per_group)} groups)")
    results["_g_quantiles"] = g_quantiles
    results["_g_test_assign"] = g_assign

    # Arm G diagnostics: rule readability.
    readability = []
    for gd in det.groups:
        n_cond = len(re.split(r"\band\b", gd.rule, flags=re.IGNORECASE))
        attrs = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", gd.rule)) - {
            "and",
            "or",
            "not",
            "True",
            "False",
        }
        readability.append({"group_id": gd.group_id, "conditions": n_cond, "attrs": sorted(attrs)})
    results["readability"] = readability
    results["rules"] = [{"group_id": gd.group_id, "rule": gd.rule} for gd in det.groups]
    # Keep the deterministic grouping for the explanation stage.
    results["_det_groups"] = det.groups
    results["_det_assignment"] = det.assignment
    results["_train_meta"] = train_meta
    results["_train_out"] = train_out
    results["_test_out"] = test_out
    return results


def explain_groups(seed_results: dict, out_dir: Path) -> dict:
    """LLM explanations for arm G groups (one seed). Returns quality stats."""
    from td_one.llm_rungs.vector_client import VectorProxyClient

    det_groups: list[GroupDefinition] = seed_results["_det_groups"]
    test_out: pd.DataFrame = seed_results["_test_out"]
    train_meta: pd.DataFrame = seed_results["_train_meta"]
    test_assign: dict[str, str] = seed_results["_g_test_assign"]
    g_quantiles: dict[str, np.ndarray] = seed_results["_g_quantiles"]

    group_test = test_out.copy()
    group_test["group_id"] = group_test["mortgage_id"].map(test_assign)
    realized = group_test.groupby("group_id")["switched_lender_flag"].mean().to_dict()

    # Predicted mean attrition = median (q50) of each group's forecast.
    predicted = {
        gid: float(q[2]) if len(q) >= 3 else float(q[len(q) // 2]) for gid, q in g_quantiles.items()
    }
    # Risk bins from TRAINING-period realized attrition per group
    # (PIT-clean). Binning the v2 predicted means is degenerate: the
    # per-group forecasts are near-identical, so tertiles collapse and
    # every group lands HIGH. Training realized rates differentiate.
    train_out_seed = seed_results["_train_out"]
    mid_to_group = dict(
        zip(
            seed_results["_train_meta"]["mortgage_id"].astype(str),
            np.asarray(seed_results["_det_assignment"].values),
            strict=True,
        )
    )
    tg = train_out_seed.copy()
    tg["group_id"] = tg["mortgage_id"].astype(str).map(mid_to_group)
    _train_realized = tg.groupby("group_id")["switched_lender_flag"].mean().to_dict()
    train_realized = {gid: float(_train_realized.get(gid, 0.0)) for gid in predicted}
    bins = compute_risk_bins(train_realized, list(train_realized.values()))
    intervals = {
        gid: (
            max(0.0, float(q[0])) if len(q) else 0.0,
            float(q[-1]) if len(q) else 0.0,
        )
        for gid, q in g_quantiles.items()
    }
    bins = widen_for_uncertainty(bins, intervals, width_threshold=0.05)

    member_of = seed_results["_det_assignment"]
    train_meta_g = train_meta.copy()
    train_meta_g["group_id"] = np.asarray(member_of.values)
    llm = VectorProxyClient(model="flash-lite", call_budget=20)
    reports = []
    cited = 0
    for gd in det_groups:
        gid = gd.group_id
        members = train_meta_g[train_meta_g["group_id"] == gid]
        stats: dict = {
            "n_members": len(members),
            "mean_household_income_cad": round(float(members["household_income_cad"].mean()), 0),
            "dominant_province": members["province"].mode().iloc[0] if len(members) else "n/a",
            "train_realized_attrition": round(float(train_realized.get(gid, 0.0)), 4),
            "n_test_episodes": int((group_test["group_id"] == gid).sum()),
            "realized_test_attrition": round(float(realized.get(gid, 0.0)), 4),
            "predicted_mean_attrition": round(float(predicted.get(gid, 0.0)), 4),
            "risk_level": bins.get(gid, "MEDIUM"),
        }
        system, user = build_explanation_prompt(gid, gd.rule, stats, articles=[])
        # Stricter contract than the module default: the checker cannot
        # distinguish interpretive closers from factual claims, so every
        # sentence must carry a citation.
        user = user + "\nEvery sentence must end with a [stat] or [url:N] citation."
        text = llm.complete(user, system=system)
        valid = check_citations(text)
        cited += int(valid)
        reports.append(
            {
                "group_id": gid,
                "rule": gd.rule,
                "risk_level": bins.get(gid, "MEDIUM"),
                "stats": stats,
                "explanation": text,
                "citations_valid": valid,
            }
        )
        print(f"  {gid}: citations_valid={valid}")
    (out_dir / "exp66c_sample_risk_report.json").write_text(json.dumps(reports, indent=2))
    return {
        "n_groups": len(reports),
        "citation_pass_rate": cited / len(reports),
        "reports": reports,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    all_results = [run_seed(s) for s in SEEDS]

    # Explanation stage on the first seed.
    quality = explain_groups(all_results[0], out_dir)
    print(f"\ncitation pass rate: {quality['citation_pass_rate']:.2f}")

    summary = {
        "config": {
            "n_customers": N_CUSTOMERS,
            "n_months": N_MONTHS,
            "test_months": TEST_MONTHS,
            "seeds": SEEDS,
            "n_clusters": N_CLUSTERS,
            "grouper_seed": GROUPER_SEED,
        },
        "seeds": [
            {
                "seed": r["seed"],
                "arms": {name: a["weighted_crps"] for name, a in r["arms"].items()},
                "readability": r["readability"],
                "rules": r["rules"],
            }
            for r in all_results
        ],
        "explanation_quality": {
            "n_groups": quality["n_groups"],
            "citation_pass_rate": quality["citation_pass_rate"],
        },
    }
    (out_dir / "exp66c_results.json").write_text(json.dumps(summary, indent=2))
    print("\nwrote exp66c_results.json")


if __name__ == "__main__":
    main()
