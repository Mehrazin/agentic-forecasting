# exp22: M2 v2 (code-executing) vs M2 v1 (3-tool) — Report

**Date:** 2026-10-17
**Status:** Complete. 60 origins × 5 seeds. Script: `experiments/exp22_m2_comparison.py`, results: `experiments/exp22_results.json`.

## Question

Does giving the ReAct agent code execution + 10 statistical tools (v2) beat the
original 3-tool agent (v1)? Both driven by deterministic scripted policies (no
real LLM) to isolate *capabilities* from LLM quality.

- **v1 policy:** data_segment → evidence → forecasting (m0) → finalize (m0 output verbatim).
- **v2 policy:** data_segment → code_execute (changepoint + stationarity + recent-shift
  t-test) → evidence (news triangulation, always) → code_execute (post-break
  adjustment, only if a significant recent shift is found) or forecasting
  (honest fallback) → finalize.

Data: event-augmented simulator (Phase 1). One-step-ahead monthly switch-rate
forecasts, scored by CRPS on {0.1, 0.5, 0.9} quantiles.

## Results

| Metric | Value |
|---|---|
| Origins | 60 (5 seeds × 12) |
| v1 mean CRPS | 0.04939 |
| v2 mean CRPS | 0.04877 |
| Overall improvement | +1.3% (paired t=0.34, p=0.73, n.s.) |
| v2 code_execute used | 100% |
| v2 adjustment rate | 15% (9/60) |

Stratified by whether v2's analysis fired:

| Stratum | n | v1 CRPS | v2 CRPS | Improvement | v2 wins |
|---|---|---|---|---|---|
| Adjusted | 9 | 0.06313 | 0.05895 | **+6.6%** | 6/9 |
| Fallback (no shift found) | 51 | 0.04697 | 0.04697 | +0.0% | 0/51 (exact ties) |

All 9 adjustments came from the recent-shift t-test (last-3 vs prior-6 months);
the changepoint detector never fired within the 5-month recency window (its
`min_distance=6` makes it blind to very recent breaks — a design lesson).

Per-origin adjusted results (mode=shift for all):

| Seed | Origin | v1 CRPS | v2 CRPS | Δ |
|---|---|---|---|---|
| 42 | 2024-05-31 | 0.03058 | 0.00928 | +69.7% |
| 42 | 2024-06-30 | 0.02894 | 0.00391 | +86.5% |
| 7 | 2024-10-31 | 0.04496 | 0.00789 | +82.5% |
| 7 | 2024-11-30 | 0.05201 | 0.02994 | +42.4% |
| 99 | 2024-02-29 | 0.15187 | 0.09875 | +35.0% |
| 1 | 2024-09-30 | 0.09783 | 0.08819 | +9.9% |
| 99 | 2024-04-30 | 0.04646 | 0.07379 | −58.8% |
| 7 | 2024-05-31 | 0.06415 | 0.11971 | −86.6% |
| 1 | 2024-10-31 | 0.05133 | 0.09908 | −93.0% |

## What v2 did differently (trace analysis)

1. **Real analysis, every origin.** v2 ran changepoint detection, ADF/KPSS
   stationarity tests, and a recent-shift t-test in the sandbox at all 60
   origins. v1 cannot do any of this.
2. **Selective action.** v2 adjusted at 9/60 origins and fell back to m0 at
   51/60 — correctly refusing to hallucinate adjustments when the evidence
   wasn't there (all 51 fallbacks tie v1 exactly).
3. **News triangulation didn't fire.** Zero news articles were retrieved in the
   60 days before any of the 9 adjusted origins. v2 acted on statistical
   evidence alone; the news channel (the information-asymmetry advantage) was
   too sparse/delayed to corroborate at these horizons.

## Diagnosis: why the wins are high-variance

The 6 wins are large (+35% to +87%) but the 3 losses are also large (−59% to
−93%). Two compounding causes:

1. **Overconfident intervals.** v2's quantile width comes from the sd of 3
   recent months. When those 3 months happen to be tight, v2 emits a very
   narrow interval (e.g., width 0.030 vs v1's 0.235). Right → excellent CRPS;
   wrong → catastrophic CRPS. The 3-point sd is too noisy for calibration.
2. **No corroboration gating.** v2 adjusts on the t-test alone (p<0.10,
   |shift|>2pp). With noisy monthly data, some "shifts" are false positives
   that mean-revert (e.g., seed 1, 2024-10-31: v2 moved the median the wrong
   way). News corroboration — the designed triangulation step — never fired
   because no recent articles existed.

## Verdict

**Capability confirmed, forecast value mixed.** v2 demonstrably does things v1
cannot: it runs statistical analysis, detects shifts, and acts selectively
(15% fire rate, 100% of fallbacks correct). But naive post-break extrapolation
is high-variance: +6.6% mean CRPS on adjusted origins with 3/9 substantial
losses driven by overconfident intervals.

**The lesson is not "agents don't help" — it's "detection ≠ good forecasting."**
The capability upgrade is real; the *decision policy* needs work:
- Shrinkage: blend recent-mean and trailing-mean instead of full switch.
- Calibration: wider intervals when the level estimate comes from few points
  (e.g., scale sd by sqrt(1 + 1/n)).
- Corroboration gating: only adjust when news or a second independent signal
  confirms (currently never fires — the news feed may need denser coverage).

This is the honest v1→v2 story: the agent became a data scientist, ran real
analysis, and its judgment is directionally right but overconfident. Phase 3
(exp23+) should target the decision policy, not more tools.

## Files

- Script: `experiments/exp22_m2_comparison.py` (ruff clean)
- Results: `experiments/exp22_results.json` (60 rows, full traces metadata)
- This report: `experiments/exp22_report.md`
