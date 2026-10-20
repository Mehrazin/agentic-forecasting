# exp68: Agentic Shift Guard vs TSFM-Alone — Report

**Date:** 2026-10-20
**Status:** COMPLETE — honest negative.
**Data:** `exp68_results.json` (192 rows, deduplicated).
**Pre-registration:** locked in `exp67_report.md` before running; reproduced below.

## Pre-registered design

- **Dev shifts:** S1, S4, S6. **Eval (holdout) shifts:** S2, S3, S5, S7.
- **Disjoint RNG seeds** from exp67 (exp67: 0–1; exp68: 100–101).
- **Dev magnitudes 1–2σ; holdout magnitudes 2–4σ.**
- **Claim under test:** agent+TSFM beats TSFM-alone on post-shift origins at
  p<0.05 (paired t-test), with a do-no-harm gate (no significant degradation
  on pre-shift origins or on the no-break shift S5).

## The guard

`src/td_one/agent/shift_guard.py`: CUSUM on a policy-rate covariate (shifts 2
months before the attrition target for S1/S2/S6) + PIT-filtered announcements
→ heuristic classifier → context truncation (TSFM on post-detection data only)
+ conformal quantile widening calibrated on dev shift episodes.
TSFM = Chronos-2 (real weights). The LLM hook (`LLMShiftClassifier`) is
implemented and tested (structure-only output) but the experiment uses the
deterministic heuristic classifier for reproducibility.

Dev calibration: `widen_factor=1.000` — the guard's truncated forecasts on dev
shifts were already calibrated (no widening applied in eval).

## Verdict: FAIL (honest negative)

| Test | Result |
|---|---|
| **Primary:** guard+TSFM vs TSFM-alone, post-shift, eval shifts | mean ΔCRPS −0.00024 (+0.5% rel.), **p=0.824**, win rate **14.6%** → **FAIL** |
| Do-no-harm (pre-shift origins) | Δ=0.00000, p=n/a → PASS |
| Do-no-harm (S5 no-break shift) | Δ=+0.00210, p=0.166 → PASS (n.s.) |

Per-shift (post-shift, n=12 each):

| Shift | Guard action | Rel. improvement | p | Win rate |
|---|---|---|---|---|
| S2 | truncate 12/12 (correct detection) | +0.3% | 0.942 | 50% |
| S3 | none 10/12, truncate 2/12 (false +) | −0.9% | 0.321 | 0% |
| S5 | none 10/12, truncate 2/12 (false +) | −4.6% | 0.166 | 0% |
| S7 | none 10/12, truncate 2/12 (false +) | +4.6% | 0.419 | 8% |

## What the negative teaches

1. **Context truncation hurts.** The guard's core response — dropping
   pre-shift history — loses 85% of the time. Chronos-2 handles regime
   shifts better with full (contaminated) context than with truncated
   post-shift context: the pre-shift data carries seasonal/structural
   information that outweighs the regime contamination. Nine post-shift
   points are not enough for the TSFM to re-estimate seasonality.
2. **Correct detection ≠ benefit.** On S2 the guard fired correctly on all
   12 forecasts (covariate lead works as designed) yet gained exactly
   nothing (+0.3%, coin flip). Detection is necessary but not sufficient;
   the response was wrong.
3. **False positives are costly.** The 2 spurious truncations on S3/S5/S7
   (CUSUM false alarms on flat covariates) account for most of the guard's
   losses (win rate 0% where it fired without signal).
4. **Widening never engaged.** Dev calibration returned 1.0 — under these
   severities the TSFM's intervals were already adequate, so the
   conformal-widening component contributed nothing.

## Implications

- Do **not** ship context truncation as a shift response. A guard that only
  widens intervals on detection (no truncation) might do no harm, but this
  experiment gives no evidence it would help either.
- The exp67 finding stands: no model is robust to 4σ level shifts, and the
  best defense remains the TSFM's own full-context adaptation.
- Per the pre-registration: no iteration on this design.

## Reproducibility notes

- `experiments/exp68_agent_vs_shift.py --mode real` (Chronos-2, CPU).
  Mock mode (`--mode mock`) reproduces the pipeline in seconds.
- A resume-checkpoint duplication bug double-counted eval_post rows in the
  first write; fixed in code and deduplicated in the committed file
  (192 rows). Values unchanged.
- `errors: 0/192` rows. Guard module: 18/18 tests green
  (`tests/test_shift_guard.py`); ruff + mypy clean.
