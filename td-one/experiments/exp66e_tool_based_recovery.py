"""exp66e: tool-based cohort recovery (Phase 1).

The deterministic grouper clustered on *raw features* and failed to
recover the simulator's planted cohorts (ARI 0.010, chance level —
exp66d) because it clusters on feature similarity, not attrition
relevance. This experiment tests whether tools that measure *what
drives attrition heterogeneity* do better:

- **SHAP clustering:** cluster on per-borrower SHAP vectors (why they
  switch) instead of raw features (what they look like).
- **SHAP interactions:** the planted cohorts are sharp interactions;
  do the top SHAP interaction pairs recover the planted pairs?
- **Uplift segmentation:** T-learner uplift (variable vs fixed rate)
  deciles — segment by heterogeneous treatment response.

Compares ARI vs planted cohorts: raw-feature KMeans (exp66d baseline,
recomputed in-run) vs SHAP-cluster KMeans vs uplift deciles.

PIT: attrition model, SHAP, and uplift are fit on training-period rows
and the training-window switching label only.

Phase 2 (LLM grouper with tool context) runs only if Phase 1 shows
promise: SHAP-cluster ARI >= 0.05 or planted recall@10 >= 0.33.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp66d_cohort_recovery import (  # noqa: E402
    _group_train_meta,
    cohort_labels_for,
    simulate,
)

from td_one.evaluation.cohort_recovery import (  # noqa: E402
    adjusted_rand_index,
    best_match_per_cohort,
)
from td_one.evaluation.shap_analysis import (  # noqa: E402
    cluster_on_shap,
    fit_attrition_model,
    planted_interaction_pairs,
    planted_pair_recall,
    prepare_features,
    shap_interaction_values,
    shap_values,
    top_interactions,
)
from td_one.evaluation.uplift import (  # noqa: E402
    decile_cohort_table,
    fit_t_learner,
    uplift_deciles,
    uplift_scores,
    variable_rate_treatment,
)

SEED = 42
N_CLUSTERS = 5
INTERACTION_SUBSAMPLE = 500
INTERACTION_SEED = 7


def _training_label(train_meta: pd.DataFrame, out_df: pd.DataFrame, cutoff) -> np.ndarray:
    train_out = out_df[out_df["month"] < cutoff]
    switched = set(train_out[train_out["switched_lender_flag"]]["mortgage_id"].astype(str))
    return train_meta["mortgage_id"].astype(str).isin(switched).to_numpy(dtype=int)


def main() -> None:
    t0 = time.time()
    out_dir = Path(__file__).resolve().parent
    print("exp66e: simulating (same config as exp66d)...")
    hist_df, out_df, cutoff = simulate()
    train_meta = _group_train_meta(hist_df, cutoff)
    y = _training_label(train_meta, out_df, cutoff)
    print(f"  borrowers: {len(train_meta)}, switch rate: {y.mean():.3f}")

    labels = cohort_labels_for(hist_df)
    mids = train_meta["mortgage_id"].astype(str).tolist()
    true = [labels[m] for m in mids]

    print("exp66e: preparing features + fitting attrition GBM...")
    x, feature_names, base_map = prepare_features(train_meta)
    print(f"  {len(feature_names)} encoded features")
    model = fit_attrition_model(x, y, seed=SEED)
    train_acc = model.score(x, y)
    print(f"  train accuracy: {train_acc:.3f}")

    results: dict = {
        "config": {"seed": SEED, "n_clusters": N_CLUSTERS, "train_accuracy": train_acc},
    }

    # ---- baseline: raw-feature KMeans (exp66d protocol) ----
    print("exp66e: baseline raw-feature KMeans...")
    from sklearn.preprocessing import StandardScaler

    x_std = StandardScaler().fit_transform(x.to_numpy(dtype=float))
    raw_labels = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit_predict(x_std)
    raw_ari = adjusted_rand_index(true, [str(v) for v in raw_labels])
    print(f"  raw-feature ARI: {raw_ari:.4f}")
    results["raw_feature_kmeans_ari"] = raw_ari

    # ---- SHAP values + clustering ----
    print("exp66e: computing SHAP values...")
    sv = shap_values(model, x)
    print(f"  shap shape: {sv.shape}")
    for k in (N_CLUSTERS, 10):
        shap_labels = cluster_on_shap(sv, n_clusters=k, seed=SEED)
        ari = adjusted_rand_index(true, [str(v) for v in shap_labels])
        print(f"  SHAP-cluster ARI (k={k}): {ari:.4f}")
        results[f"shap_cluster_ari_k{k}"] = ari
    results["shap_cluster_per_cohort_k5"] = best_match_per_cohort(
        true, [str(v) for v in cluster_on_shap(sv, n_clusters=N_CLUSTERS, seed=SEED)]
    )

    # ---- SHAP interactions: do they expose the planted pairs? ----
    print("exp66e: computing SHAP interaction values (subsample)...")
    rng = np.random.default_rng(INTERACTION_SEED)
    sub = rng.choice(len(x), size=min(INTERACTION_SUBSAMPLE, len(x)), replace=False)
    iv = shap_interaction_values(model, x.iloc[sub])
    top = top_interactions(iv, feature_names, k=20, base_map=base_map)
    print("  top-10 interactions:")
    for entry in top[:10]:
        print(f"    {entry['pair'][0]} x {entry['pair'][1]}: {entry['score']:.5f}")
    planted = planted_interaction_pairs()
    print(f"  planted pairs ({len(planted)}): {[tuple(sorted(p)) for p in planted]}")
    for kk in (5, 10, 20):
        rec = planted_pair_recall(top, k=kk)
        print(f"  planted recall@{kk}: {rec:.3f}")
        results[f"planted_recall@{kk}"] = rec
    results["top_interactions"] = [
        {"pair": list(e["pair"]), "score": e["score"]} for e in top
    ]

    # ---- T-learner uplift deciles ----
    print("exp66e: fitting T-learner uplift...")
    treatment = variable_rate_treatment(train_meta)
    print(f"  treated (variable) rate: {treatment.mean():.3f}")
    ufit = fit_t_learner(x, treatment, y, seed=SEED)
    scores = uplift_scores(ufit, x)
    dec = uplift_deciles(scores, n_deciles=10)
    uplift_ari = adjusted_rand_index(true, [str(v) for v in dec])
    print(f"  uplift-decile ARI: {uplift_ari:.4f}")
    results["uplift_decile_ari"] = uplift_ari
    results["uplift_decile_per_cohort"] = best_match_per_cohort(
        true, [str(v) for v in dec]
    )
    tab = decile_cohort_table(dec, true)
    print("  decile x cohort crosstab:")
    print(tab.to_string())
    results["uplift_decile_crosstab"] = tab.to_dict()

    # ---- verdict ----
    shap_ari5 = results["shap_cluster_ari_k5"]
    rec10 = results["planted_recall@10"]
    phase2 = bool(shap_ari5 >= 0.05 or rec10 >= 0.33)
    results["phase2_recommended"] = phase2
    print("\n--- ARI comparison ---")
    print(f"  raw-feature KMeans (k=5): {raw_ari:.4f}  [exp66d baseline]")
    print(f"  SHAP-cluster (k=5):       {shap_ari5:.4f}")
    print(f"  SHAP-cluster (k=10):      {results['shap_cluster_ari_k10']:.4f}")
    print(f"  uplift deciles (10):      {uplift_ari:.4f}")
    print(f"  planted recall@10:        {rec10:.3f}")
    print(f"  Phase 2 (LLM tool context): {'GO' if phase2 else 'NO-GO'}")
    print(f"  elapsed: {time.time() - t0:.0f}s")

    (out_dir / "exp66e_results.json").write_text(json.dumps(results, indent=2, default=str))
    print("wrote exp66e_results.json")


if __name__ == "__main__":
    main()
