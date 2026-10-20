# exp66e: Uplift + SHAP Tools for Cohort Recovery — Report

**Date:** 2026-10-20
**Status:** Complete. Phases 1+2 plus the exp66d classifier follow-up (exp66f).
**Setup:** seed 42, 4,000 customers, 36 months, same protocol as exp66d.
**Dependency note:** `shap>=0.45.0` installed via uv and declared in
`pyproject.toml`; `uv.lock` regeneration is pending on Vandad's laptop.

## Phase 1 — Do the tools recover the cohorts? (exp66e)

The deterministic grouper failed cohort recovery (ARI 0.010, exp66d)
because it clusters on feature *similarity*. Hypothesis: tools that
measure *what drives attrition heterogeneity* — SHAP attributions and
uplift scores — should cut closer to the DGP's causal joints.

**ARI vs planted cohorts (higher = better recovery; ~0 = chance):**

| Method | ARI | Notes |
|---|---|---|
| Raw-feature KMeans (k=5) | 0.046 | exp66d protocol recomputed in-run (exp66d's 0.010 was on 10 tree-leaf groups; both ≈ chance) |
| **SHAP-cluster KMeans (k=5)** | **0.091** | ~2× raw baseline, but still weak in absolute terms |
| SHAP-cluster KMeans (k=10) | 0.066 | more clusters don't help |
| Uplift deciles (T-learner) | 0.002 | ❌ chance level |

**Per-cohort best match, SHAP clustering (k=5):**

| Cohort | Precision | Recall | F1 |
|---|---|---|---|
| `low_rate` | 0.53 | 0.57 | 0.55 |
| `control` | 0.71 | 0.45 | 0.55 |
| `family_squeeze` | 0.09 | 0.44 | 0.15 |
| `osfi_exempt` | 0.12 | 0.42 | 0.18 |

`low_rate` is now *partially* recovered (F1 0.45 → 0.55 vs exp66d) —
the SHAP space does carry cohort signal. `family_squeeze` and
`osfi_exempt` are still missed.

**SHAP interactions — the specific hypothesis fails.** Planted-pair
recall@10 = **0.000** (recall@5 = 0.000, @20 = 0.000). The top
interactions are dominated by `term_months` pairs
(`origination_year × term_months` 0.20, `mortgage_rate_pct ×
term_months` 0.12, `children × term_months` 0.04). Planted attributes
*appear* (origination_year, mortgage_rate_pct, children) but with the
wrong partners. Explanation: `term_months` drives maturity timing and
seasoning in the DGP, and the training-label construction
(switch-before-cutoff) makes term a strong selection variable — the
interaction detector finds the model's real structure, which is not
the planted cohort rules.

**Uplift — honest negative.** T-learner uplift (variable vs fixed
rate) deciles give ARI 0.002. Variable/fixed exposure does not segment
borrowers along cohort joints. Per-cohort best-match F1 ≤ 0.19 for all
cohorts.

**Model sanity:** GBM train accuracy 0.99, 3-fold CV AUC 0.92. The
synthetic DGP's switching logit is mostly determined by observable
quantities (offer spread S-curve, rate spread, payment shock), so a
high AUC is genuine signal, not memorization — feature importance is
spread across plausible drivers with no single smoking-gun leak.
**Caveat (documented, not fixed):** features are last-pre-cutoff
snapshots, not decision-point-aligned; 60% of switched borrowers'
snapshots post-date their outcome month. History is generated before
outcomes in the simulator, so this is a measurement-timing caveat of
the exp66d protocol, not label leakage. Changing the protocol
mid-ablation would confound every comparison against exp66c/d.

**Phase 1 verdict:** ⚠️ **Mixed.** SHAP clustering is the first method
to beat the raw-feature baseline (2×), and partially recovers
`low_rate`. The interaction-recovery hypothesis and the uplift
hypothesis both fail cleanly. Pre-registered Phase 2 gate (SHAP ARI
≥ 0.05) passed on the clustering result alone.

## Phase 2 — Give the tools to the LLM grouper (exp66e2)

`LLMGrouper` gained a `tool_context` input (SHAP top drivers, top
interactions, SHAP-segment profiles, uplift summary). 3 trials per
config, ARI of the proposed partition vs planted cohorts:

| Config | Trial ARIs | Mean |
|---|---|---|
| LLM baseline (attribute summaries only) | 0.056, 0.026, −0.006 | 0.026 |
| LLM + tool context | 0.015, 0.035, (no groups) | 0.025 |

**❌ Honest negative.** Tool context does not improve LLM group
recovery — both configs sit at chance level, *below* deterministic
SHAP clustering (0.091) and even raw KMeans (0.046). The LLM proposes
plausible-sounding rules that don't carve the DGP's joints, and
analyst findings in the prompt don't change that. This extends the
exp66b/exp66d pattern: the LLM is a poor *discoverer* of group
structure. The tools' value is for the analyst layer (below), not
the grouper.

## exp66f — LLM risk classifier ablation + risk-identification tools

Follow-up to exp66d's info-asymmetric comparison ("bins vs
bins+LLM+news", n=10). Now 3 seeds × ~10 groups per config, Spearman
correlation of risk level vs test realized attrition.

**Ablation 1 — isolate LLM vs news:**

| Config | Seed 42 | Seed 123 | Seed 7 |
|---|---|---|---|
| bins | −0.379 | +0.517 | +0.299 |
| llm-stats | +0.120 | +0.382 | +0.261 |
| llm-stats-news | +0.055 | +0.359 | +0.454 |

Pooled bootstrap 95% CIs (2,000 reps): llm-stats − bins = +0.154
[−0.378, +0.667]; **news effect (llm-stats-news − llm-stats) =
+0.031 [−0.447, +0.499]** — the interval covers zero comfortably.
exp66d's apparent LLM gain does **not** survive the controlled
ablation: it was seed-specific noise (seed 42's bins were inverted;
on seeds 123/7 the bins are fine and the LLM adds nothing measurable).
News adds nothing on top of stats either.

**Ablation 2 — risk-identification tools** (`agent/risk_tools.py`,
seed 42 only):

| Config | Spearman | Monotonic |
|---|---|---|
| llm-stats (baseline) | +0.120 | ❌ |
| + SHAP drivers | +0.104 | ❌ |
| **+ counterfactual sensitivity** | **+0.348** | ✅ |
| + uplift amenability | +0.120 | ❌ |
| + news alignment (semantic) | −0.174 | ❌ |
| + all tools | +0.165 | ❌ |

The counterfactual tool is the only one that moved the needle on
seed 42 (and the only monotonic config there). But n=10 groups, one
seed — this is a directional signal, not a result. News alignment
*hurts* (−0.174), consistent with the news-track honest negatives
elsewhere in the program. SHAP drivers and uplift amenability are
neutral as prompt context.

## Verdict — do these tools belong in the agent's kit?

1. **SHAP analysis (`evaluation/shap_analysis.py`): YES, for analysts.**
   SHAP clustering is the best cohort-recovery method we have (ARI
   0.091, `low_rate` F1 0.55) and per-group SHAP drivers are the
   grounded way to explain *why* a segment is risky. Keep as an
   evaluation/explanation tool. Do NOT claim interaction recovery —
   recall@10 = 0 kills that story.
2. **Uplift (`evaluation/uplift.py`): NO for recovery, keep on shelf.**
   ARI 0.002 as segmentation; neutral as LLM context. The module is
   tested and reusable if a real treatment (retention offers) is ever
   run, but it earns no place in the current pipeline.
3. **LLM grouper + tool context: NO.** Mean ARI 0.025 vs 0.026
   baseline — the tools don't fix LLM group discovery. Grouping
   stays deterministic (exp66c pivot stands).
4. **Risk tools for the classifier: counterfactuals only, provisionally.**
   +0.348 vs +0.120 on one seed is worth a multi-seed confirmation
   before any claim. News alignment should not be used (−0.174).
5. **exp66d's LLM-classifier result is downgraded.** With 3 seeds and
   controlled information, the LLM-vs-bins advantage is +0.154 with CI
   covering zero, and the news effect is +0.031 [−0.45, +0.50]. The
   honest statement is now: *no measurable LLM or news advantage for
   group risk classification at this scale.*

## Files

- Code: [shap_analysis.py](sandbox://workspace/agentic-forecasting/td-one/src/td_one/evaluation/shap_analysis.py), [uplift.py](sandbox://workspace/agentic-forecasting/td-one/src/td_one/evaluation/uplift.py), [risk_tools.py](sandbox://workspace/agentic-forecasting/td-one/src/td_one/agent/risk_tools.py)
- Experiments: [exp66e_tool_based_recovery.py](sandbox://workspace/agentic-forecasting/td-one/experiments/exp66e_tool_based_recovery.py), [exp66e2_llm_tool_context.py](sandbox://workspace/agentic-forecasting/td-one/experiments/exp66e2_llm_tool_context.py), [exp66f_classifier_ablation.py](sandbox://workspace/agentic-forecasting/td-one/experiments/exp66f_classifier_ablation.py)
- Results: `exp66e_results.json`, `exp66e2_results.json`, `exp66f_results.json`
- Tests: `tests/test_shap_analysis.py` (7), `tests/test_uplift.py` (5),
  `tests/test_risk_tools.py` (8), `tests/test_llm_grouper.py` (+1 tool-context)
