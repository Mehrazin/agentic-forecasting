# exp66: LLM Grouping + Group-Level Attrition Forecasting

**Date:** 2026-10-19
**Verdict:** ❌ **Honest negative** — LLM-discovered groups do not beat random groups.

## Setup

4-arm ablation on synthetic borrower data (2,000 customers, 36 months,
6 held-out test months, 140 test episodes):

| Arm | Description | Weighted CRPS |
|---|---|---|
| A: Global | No grouping | **0.00227** |
| B: Random | Random assignment, 2 groups | 0.00690 |
| C: Hand-written | Province × income cohorts | 0.01164 |
| D: LLM-discovered | Gemini Flash proposes rules | 0.01705 |

**Pre-registered kill criterion:** D must beat B significantly (paired
t-test). **Result: D (0.01705) is worse than B (0.00690). Kill criterion
triggered.** The LLM grouping is dead — do not iterate the prompt.

## What the LLM proposed

Gemini Flash (via Vector proxy) proposed 5 groups from training metadata
+ attrition summaries; 2 validated after overlap resolution:

- `high_rate_variable`: `rate_type == 'variable'`
- `high_debt_service`: `rate_type == 'fixed' and mortgage_payment_monthly_cad > 3108.24`
- (+ catch-all)

The rules are plausible but the groups are too coarse: variable-rate
borrowers are a large heterogeneous pool, and the payment threshold
splits on a continuous variable without capturing the underlying
rate-sensitivity mechanism.

## Why it failed (hypotheses)

1. **Small test set:** 140 episodes over 6 months is thin for group-level
   evaluation. Per-group CRPS is noisy.
2. **Stub forecaster:** the trailing-mean forecaster used here is weak;
   group differences may emerge with the full v2 router.
3. **LLM optimizes for interpretability, not predictiveness:** the rules
   "make sense" (variable rates are riskier) but don't carve the
   attrition DGP at its joints.

## What was built (all tested, all pushed)

- `src/td_one/data_gen/group_series.py` — histories → monthly per-group
  series (aggregation identity tested).
- `src/td_one/rag/semantic.py` — local embedding RAG (all-MiniLM-L6-v2;
  Vector proxy has no embedding access). Top-k tuned: semantic_k=50
  (recall 0.818), final k=5.
- `src/td_one/agent/grouping.py` — `GroupDefinition`, `validate_rule()`,
  `apply_groups()`, `discover_candidate_groups()` (closes Argon Finding 3).
- `src/td_one/agent/llm_grouper.py` — prompt builder + validation loop.
- `src/td_one/agent/group_forecaster.py` — per-group forecasting with
  global fallback.
- `src/td_one/agent/risk_reporter.py` — deterministic bins + citation-
  enforced explanations.
- `scripts/fetch_demographic_news.py` — Perplexity fetcher (cached,
  resume-safe).
- `scripts/tune_rag_k.py` — top-k tuning harness.

## Next steps

The infrastructure is solid and reusable. The grouping hypothesis needs:
(1) larger test sets, (2) the full v2 forecaster per group, and
(3) possibly LLM-proposed groups evaluated on more distinct DGPs.
Track 2's shift scenarios (exp67) can stress-test whether grouping helps
under distribution shift even if it doesn't help in-distribution.
