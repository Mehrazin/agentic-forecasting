"""Re-run the exp66d LLM risk classifier to capture justifications.

The main exp66d run saved levels but not justification texts. This
re-simulates (seed 42 -> identical data/groups) and re-classifies,
persisting justifications to exp66d_justifications.json. One-off
script; not part of the experiment suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp66d_cohort_recovery import (  # noqa: E402
    _group_train_meta,
    _news_for_group,
    simulate,
)

from td_one.agent.deterministic_grouper import fit_deterministic_groups  # noqa: E402
from td_one.agent.llm_risk_classifier import classify_risk_llm  # noqa: E402
from td_one.llm_rungs.vector_client import VectorProxyClient  # noqa: E402

N_CLUSTERS = 5
GROUPER_SEED = 42
MIN_GROUP_N = 200


def main() -> None:
    hist_df, out_df, cutoff = simulate()
    train_meta = _group_train_meta(hist_df, cutoff)
    det = fit_deterministic_groups(
        train_meta, n_clusters=N_CLUSTERS, seed=GROUPER_SEED, min_group_n=MIN_GROUP_N
    )
    train_out = out_df[out_df["month"] < cutoff]
    member_of = det.assignment
    train_meta_g = train_meta.copy()
    train_meta_g["group_id"] = np.asarray(member_of.values)
    mid_to_group = dict(
        zip(
            train_meta["mortgage_id"].astype(str),
            np.asarray(member_of.values).astype(str),
            strict=True,
        )
    )
    tg = train_out.copy()
    tg["group_id"] = tg["mortgage_id"].astype(str).map(mid_to_group)
    train_realized = tg.groupby("group_id")["switched_lender_flag"].mean().to_dict()

    llm = VectorProxyClient(model="flash-lite", call_budget=15)
    out = {}
    for gd in det.groups:
        gid = gd.group_id
        members = train_meta_g[train_meta_g["group_id"] == gid]
        stats = {
            "n_members": len(members),
            "mean_household_income_cad": round(float(members["household_income_cad"].mean()), 0),
            "dominant_province": members["province"].mode().iloc[0] if len(members) else "n/a",
            "train_realized_attrition": round(float(train_realized.get(gid, 0.0)), 4),
        }
        try:
            articles = _news_for_group(str(stats["dominant_province"]), cutoff)
        except Exception:  # noqa: BLE001 — news is a bonus, not a gate
            articles = []
        try:
            res = classify_risk_llm(llm, gid, gd.rule, stats, articles)
            out[gid] = {
                "level": res.level,
                "justification": res.justification,
                "citations_valid": res.citations_valid,
                "rule": gd.rule,
            }
        except Exception as e:  # noqa: BLE001
            out[gid] = {"level": "ERROR", "justification": str(e), "rule": gd.rule}
        print(f"  {gid}: {out[gid]['level']}")
    out_path = Path(__file__).resolve().parent / "exp66d_justifications.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
