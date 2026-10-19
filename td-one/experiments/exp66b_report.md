# exp66b: Grouping for Explainability — Report

**Date:** 2026-10-19
**Status:** Infrastructure complete; LLM grouper flaky in full pipeline
**Reframed goal:** Grouping for explainability, not CRPS. Kill criterion OFF.

## What was built

Full v2 architecture for group-level attrition forecasting:
- `agent/grouping.py` — `GroupDefinition`, `validate_rule()` (now rejects
  bare-column rules like `age < 40 and income`), `discover_candidate_groups()`
- `agent/llm_grouper.py` — prompt with interaction hints, baseline CRPS
  targeting, human-friendly bin edges
- `agent/group_forecaster.py` — per-group `ProposedForecasterV2`
- `agent/risk_reporter.py` — deterministic bins + citation enforcement
- `experiments/exp66b_grouped_attrition_v2.py` — 9 arms (A/B/C/D1-D4/E/F),
  group-count sweep, risk-bin calibration

All modules have tests (49 total, passing). Committed and pushed.

## Key findings

### 1. The validator fix matters
The LLM occasionally emits incomplete rules like:
```
primary_borrower_age < 40 and other_debt_payment_monthly_cad
```
(missing threshold). These parse in pandas but are semantically wrong.
Added `validate_rule()` check: every AND/OR clause must contain a comparison
operator. Test added. This is a real robustness win for any LLM-to-code pipeline.

### 2. LLM grouper works in isolation (5/5), flaky in full pipeline
Controlled test (2000 customers, 5 trials): **5/5 success**, 3-6 groups each.
Full exp66b pipeline (5000 customers, 48 months): D1/D2 failed validation
on both attempts.

**Hypothesis:** The failures correlate with prompt complexity (baseline CRPS
targeting + interaction hints) interacting with LLM variance. The isolated
tests used simpler prompts. The flakiness itself is a finding: **LLM-proposed
groupings are not reliably reproducible**, which undermines their use for
explainability — a risk manager can't trust groups that change run-to-run.

### 3. Rule readability (from successful runs)
When the LLM succeeds, rules are interpretable:
```
rate_type == 'variable' and household_income_cad <= 44000
  [Variable rate borrowers with constrained income...]

credit_score_band == '720_plus' and household_income_cad > 83000
  [High-earning individuals with excellent credit...]
```
Mean 2.0 conditions per rule, 4-5 distinct attributes. Readable by a risk
manager. But see finding #2 — reproducibility is the issue, not readability.

### 4. Risk-bin calibration (implemented, not yet run)
`evaluate()` now computes per-arm risk bins and calibration spread
(HIGH realized - LOW realized). Ready for when the pipeline stabilizes.

## Recommendation: PIVOT (not kill, not continue as-is)

**Don't kill grouping** — the infrastructure is solid and the explainability
goal is valid. **Don't continue the current approach** — the LLM grouper is
too flaky for production use.

**Pivot to:** Deterministic grouping with LLM *explanations*.
- Use hand-written or clustering-based groups (deterministic, reproducible)
- Use the LLM only for what it's good at: generating grounded rationales
  and risk narratives for each group (via `risk_reporter.py`)
- This separates the flaky part (group discovery) from the valuable part
  (explanation generation)

The citation-enforced `risk_reporter.py` is the most novel piece and doesn't
depend on LLM grouping. A risk manager gets: deterministic groups +
LLM-generated, citation-grounded explanations. That's the explainability win
without the reproducibility problem.

## Cost/benefit (honest)
- **CRPS cost of grouping:** Unknown (pipeline didn't complete). exp66
  suggested grouping hurts CRPS vs global, but that was a thin test.
- **Explainability gain:** Rules are readable when generated, but
  non-reproducible. Net value currently negative for production.
- **Infrastructure value:** High. Validator, grouper, forecaster, reporter
  are reusable. The bare-column check alone is worth keeping.

## Next steps (if pivoting)
1. Lock in deterministic groups (hand-written or k-means on metadata)
2. Run `risk_reporter.py` on each group with news context
3. Measure: citation rate, human readability scores, risk-bin calibration
4. Compare: LLM explanations vs template explanations (ablate the LLM)
