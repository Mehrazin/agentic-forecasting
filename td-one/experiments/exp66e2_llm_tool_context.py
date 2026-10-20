"""exp66e Phase 2: LLM grouper with tool context vs without.

Phase 1 verdict: SHAP clustering doubled ARI (0.091 vs 0.046) but
interaction recall@10 = 0 and uplift deciles = 0.002. The
pre-registered gate (SHAP-cluster ARI >= 0.05) passes, so we test
whether giving the LLM the analyst tools' findings improves its
proposed groupings' cohort recovery.

Configs (3 trials each; the LLM grouper is flaky per exp66b, so
per-trial ARIs are reported, not just means):
  - baseline: LLMGrouper with attribute summaries only
  - tools:    + SHAP top drivers, top interactions, SHAP-segment
              profiles, uplift-decile summary as tool context

Metric: ARI of the proposed partition (with catch-all) vs planted
cohorts. Honest framing: the tools describe model structure, not the
planted cohorts (Phase 1 showed interactions miss the planted pairs),
so any gain comes from better-grounded proposals, not from leaking
the answer — the LLM never sees cohort labels.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp66d_cohort_recovery import (  # noqa: E402
    _group_train_meta,
    cohort_labels_for,
    simulate,
)

from td_one.agent.llm_grouper import LLMGrouper  # noqa: E402
from td_one.evaluation.cohort_recovery import adjusted_rand_index  # noqa: E402
from td_one.evaluation.shap_analysis import (  # noqa: E402
    base_attribute,
    cluster_on_shap,
    fit_attrition_model,
    group_shap_driver_summary,
    expected_log_odds,
    prepare_features,
    shap_interaction_values,
    shap_values,
    top_interactions,
)
from td_one.evaluation.uplift import (  # noqa: E402
    fit_t_learner,
    uplift_deciles,
    uplift_scores,
    variable_rate_treatment,
)

SEED = 42
N_TRIALS = 3
MIN_GROUP_N = 200
INTERACTION_SUBSAMPLE = 500


def _training_label(train_meta: pd.DataFrame, out_df: pd.DataFrame, cutoff) -> np.ndarray:
    train_out = out_df[out_df["month"] < cutoff]
    switched = set(train_out[train_out["switched_lender_flag"]]["mortgage_id"].astype(str))
    return train_meta["mortgage_id"].astype(str).isin(switched).to_numpy(dtype=int)


def _fmt_pair(pair: tuple[str, str], score: float) -> str:
    return f"{pair[0]} x {pair[1]} ({score:.3f})"


def build_tool_context(
    train_meta: pd.DataFrame,
    x: pd.DataFrame,
    feature_names: list[str],
    base_map: dict[str, str],
    model,
    sv: np.ndarray,
    y: np.ndarray,
) -> list[str]:
    """Analyst-tool findings as LLM prompt context (no cohort labels)."""
    sections: list[str] = []
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)
    top_feats = [
        f"{base_attribute(feature_names[i], base_map)} ({mean_abs[i]:.3f})"
        for i in order[:8]
    ]
    sections.append("Top attrition drivers (mean |SHAP|, log-odds): " + "; ".join(top_feats))

    rng = np.random.default_rng(7)
    sub = rng.choice(len(x), size=min(INTERACTION_SUBSAMPLE, len(x)), replace=False)
    iv = shap_interaction_values(model, x.iloc[sub])
    top = top_interactions(iv, feature_names, k=10, base_map=base_map)
    sections.append(
        "Strongest feature interactions (mean |SHAP interaction|): "
        + "; ".join(_fmt_pair(t["pair"], t["score"]) for t in top)
    )

    ev = expected_log_odds(model)
    cl = cluster_on_shap(sv, n_clusters=5, seed=SEED)
    seg_lines = []
    for k in range(5):
        mask = cl == k
        if not mask.any():
            continue
        gm = ev + sv[mask].sum(axis=1).mean()
        summary = group_shap_driver_summary(
            sv, feature_names, mask, ev, gm, top_k=2, base_map=base_map
        )
        ups = ", ".join(d["feature"] for d in summary["top_up"]) or "none"
        downs = ", ".join(d["feature"] for d in summary["top_down"]) or "none"
        seg_lines.append(
            f"segment {k} (n={int(mask.sum())}): up: {ups}; down: {downs}"
        )
    sections.append("SHAP segments (borrowers grouped by why they switch): " + " | ".join(seg_lines))

    treatment = variable_rate_treatment(train_meta)
    ufit = fit_t_learner(x, treatment, y, seed=SEED)
    usc = uplift_scores(ufit, x)
    dec = uplift_deciles(usc, n_deciles=10)
    d9 = float(usc[dec == 9].mean())
    d0 = float(usc[dec == 0].mean())
    sections.append(
        f"Uplift segments (T-learner, variable vs fixed rate): highest-uplift decile "
        f"mean uplift {d9:+.4f} (n={int((dec == 9).sum())}); lowest {d0:+.4f} "
        f"(n={int((dec == 0).sum())}). High-uplift borrowers respond most to rate moves."
    )
    return sections


def _partition_assignment(
    train_meta: pd.DataFrame, groups: list
) -> dict[str, str]:
    """First-match partition for ARI: each borrower -> first matching group.

    LLM proposals may overlap slightly (resolve_overlaps tolerates small
    Jaccard); this yields a deterministic partition where ``apply_groups``
    would raise. Unmatched borrowers go to ``__catch_all__``.
    """
    mids = train_meta["mortgage_id"].astype(str)
    assignment = pd.Series("__catch_all__", index=mids.tolist())
    claimed: set[str] = set()
    for g in groups:
        try:
            hits = set(train_meta.query(g.rule)["mortgage_id"].astype(str))
        except Exception:  # noqa: BLE001 — query failures are the validation signal
            continue
        new = sorted(hits - claimed)
        if new:
            assignment.loc[new] = g.group_id
            claimed.update(new)
    return assignment.to_dict()


def main() -> None:
    from td_one.llm_rungs.vector_client import VectorProxyClient

    t0 = time.time()
    out_dir = Path(__file__).resolve().parent
    print("exp66e2: simulating...")
    hist_df, out_df, cutoff = simulate()
    train_meta = _group_train_meta(hist_df, cutoff)
    y = _training_label(train_meta, out_df, cutoff)
    labels = cohort_labels_for(hist_df)
    true = [labels[m] for m in train_meta["mortgage_id"].astype(str)]

    print("exp66e2: fitting model + SHAP...")
    x, feature_names, base_map = prepare_features(train_meta)
    model = fit_attrition_model(x, y, seed=SEED)
    sv = shap_values(model, x)

    df = train_meta.copy()
    df["switched_train"] = y.astype(int)

    llm = VectorProxyClient(model="flash-lite", call_budget=40)
    grouper = LLMGrouper(llm, max_retries=2, max_groups=5)

    results: dict = {"trials": []}
    for trial in range(N_TRIALS):
        for cfg, tools in (("baseline", None), ("tools", "build")):
            ctx = build_tool_context(train_meta, x, feature_names, base_map, model, sv, y) if tools else None
            print(f"  trial {trial} config {cfg}: proposing...")
            try:
                groups = grouper.propose(
                    df,
                    "switched_train",
                    categorical=["province", "rate_type", "credit_score_band"],
                    tool_context=ctx,
                    min_group_n=MIN_GROUP_N,
                )
            except Exception as e:  # noqa: BLE001
                print(f"    propose raised: {e}")
                groups = []
            if not groups:
                print("    no groups proposed")
                results["trials"].append(
                    {"trial": trial, "config": cfg, "ari": None, "n_groups": 0}
                )
                continue
            assignment = _partition_assignment(train_meta, groups)
            pred = [assignment[m] for m in train_meta["mortgage_id"].astype(str)]
            ari = adjusted_rand_index(true, pred)
            print(f"    {len(groups)} groups, ARI={ari:.4f}")
            for g in groups:
                print(f"      {g.group_id}: {g.rule[:90]}")
            results["trials"].append(
                {
                    "trial": trial,
                    "config": cfg,
                    "ari": ari,
                    "n_groups": len(groups),
                    "rules": [g.rule for g in groups],
                }
            )

    for cfg in ("baseline", "tools"):
        aris = [t["ari"] for t in results["trials"] if t["config"] == cfg and t["ari"] is not None]
        print(f"  {cfg}: ARIs={[f'{a:.4f}' for a in aris]} mean={np.mean(aris):.4f}" if aris else f"  {cfg}: no successful trials")
        results[f"{cfg}_mean_ari"] = float(np.mean(aris)) if aris else None
    results["elapsed_s"] = round(time.time() - t0, 1)
    (out_dir / "exp66e2_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote exp66e2_results.json ({results['elapsed_s']}s)")


if __name__ == "__main__":
    main()
