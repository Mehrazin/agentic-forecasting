# exp67: TSFM Red-Team on Loan-Attrition Shifts — Report

**Date:** 2026-10-20
**Status:** COMPLETE — real-TSFM grid finished (32 combos + S3 control).
**Data:** `exp67_results_a.json` (S1–S4 + placebo + S3ctrl, 864 rows),
`exp67_results_b.json` (S5–S7, 432 rows). Analysis: `experiments/exp67_analyze.py`
→ `experiments/exp67_figs/breaking_points.csv` + `breaking_point_curves.png`.

## Protocol

7 shift types (S1–S7) × severities [1σ, 4σ] × 2 seeds × 9 rolling origins
(3 pre / 3 straddle / 3 post, h=6, onset month 30 of 48). Models: Chronos-2 and
TimesFM-2.5 (real weights, CPU) vs naive-trailing and M0 (Jeffreys
Beta-Binomial) on identical shifted series. Placebo (no-shift DGP) control
included; S3 uses a dedicated undiverged-regional-average control (`S3ctrl`)
because its aggregate is built from different regional seeds.

**Metric (placebo-controlled):** for each (shift, severity, model, seed, origin),
CRPS(shifted) − CRPS(control at same seed/origin). This differences out
origin/context-length effects. Paired t-test on the diffs (n=6 per cell:
2 seeds × 3 positional pairs). Pre-shift class doubles as the PIT/leakage
gate — shifted and control series are bit-identical before onset, so pre
degradation must be ≈0.

## PIT / placebo gate: PASS

Pre-shift degradation is **+0% in all 28 (shift, severity, model) cells**.
No pre-onset leakage in any shift constructor (verified bit-identical), the
S3 control aggregation was fixed to ratio-of-sums (average-of-ratios leaked
0.01 pre-onset; caught by the gate, fixed, re-ran with an assertion), and the
analysis pairing was fixed after a severity-duplicate placebo misalignment
was caught by the same gate. The harness is trustworthy.

## Breaking-point curves

Straddle-origin CRPS degradation vs control (placebo-controlled; `*` = p<0.05):

| Shift | 1σ (all models) | 4σ (all models) | Verdict |
|---|---|---|---|
| S1 rate shock | +7–13%, n.s. | **+185–228%** `*` | Breaks between 1σ–4σ; catastrophic at 4σ |
| S2 policy intervention | −1–+4%, n.s. | **+65–79%** `*` | Breaks at 4σ only |
| S3 regional divergence | −7–−11%, n.s. | **+30–59%** `*` | Breaks at 4σ (aggregate barely moves!) |
| S4 seasonal break | −10–−13%, n.s. | −11–−13%, n.s. | **Never breaks** — break *helps* slightly |
| S5 correlation breakdown | −5–−11%, n.s. | −1–−8%, n.s. | **Never breaks** — univariate models blind (expected) |
| S6 volatility regime | +16–29% (m0 `*`) | +33–50% (m0 `*`) | Partial — m0 breaks, TSFMs degrade n.s. |
| S7 composition drift | −1–−5%, n.s. | +32–37%, n.s. | Marginal at 4σ (post-shift: +61–75% `*` for TSFMs) |

![Breaking-point curves](exp67_figs/breaking_point_curves.png)

## Per-shift verdicts

- **S1** is the universal killer: every model, including both TSFMs, suffers
  ~3× CRPS at 4σ on straddle origins. 80% coverage collapses (chronos2
  0.89→0.53, m0 0.61→**0.08**, naive 0.81→0.33). A plain level shift defeats
  all four forecasters — there is no "robust model" to switch to.
- **S2/S3** break all models at 4σ but are harmless at 1σ. S3 is notable: the
  portfolio aggregate is *designed* to cancel, yet every model still degrades
  +30–59% — regional divergence injects aggregate noise the univariate
  forecasters cannot explain.
- **S4/S5 are honest negatives for the red-team:** no model breaks at either
  severity. S4's seasonal break slightly *improves* CRPS (the post-break
  seasonal pattern is simpler); S5's correlation breakdown is invisible to
  univariate series by construction. A guard must **not** fire aggressively
  on these — do-no-harm.
- **S6** breaks m0 (+50% `*` at 4σ) while TSFMs degrade (+33–36%) without
  reaching significance — the one shift where TSFMs show a (weak) relative
  edge.

## TSFM vs classical: relative fragility, absolute dominance

In *relative* terms TSFMs enjoy **no robustness advantage**: at 4σ S1,
chronos2 +185% and timesfm25 +200% vs m0 +228% and naive +193% — everyone
breaks together. But in *absolute* terms at straddle/4σ, **Chronos-2 ranks #1
in 6 of 8 shifts** (S1, S2, S3, S3ctrl, S6, calm) and timesfm25 #1 in S4;
**m0 ranks last almost everywhere**. TSFMs fall further because they start
from a much better baseline, but they still land softest. Implication for the
guard: the win is not "switch to a robust model" (none exists) but **early
detection + interval widening + context surgery**.

## What this means for the guard (exp68)

1. **Target the breaking region:** S1/S2/S3/S6/S7 at ≥2σ. S4/S5 are no-fire
   zones (do-no-harm gate).
2. **Coverage collapse is the sharpest failure mode** (m0 0.61→0.08 at S1/4σ)
   — shift-aware quantile widening from past shift episodes' residuals is the
   highest-value guard component.
3. **Detection must precede the target move:** S1/S2 are visible in policy
   rates / announcements before attrition reacts — the CUSUM-on-covariates
   design is validated by this finding.

## Pre-registration for exp68 (locked before running)

- **Dev shifts:** S1, S4, S6. **Eval (holdout) shifts:** S2, S3, S5, S7.
- **Disjoint RNG seeds** from exp67 (exp67 used seeds 0–1; exp68 uses 100+).
- **Dev magnitudes 1–2σ; holdout magnitudes 2–4σ** (exp67 ran 1σ/4σ; exp68
  adds the 2σ rung for both).
- **Claim under test:** agent+TSFM beats TSFM-alone on post-shift origins at
  p<0.05 (paired t-test on placebo-controlled CRPS diffs), with a do-no-harm
  gate (no significant degradation on pre-shift origins or on S4/S5).
- If the claim fails: honest negative, no iteration.

## Reproducibility notes

- Two VM recycles hit the grid mid-run; per-combo atomic checkpoints
  (`--resume`) lost zero completed combos. `exp67_fix_naive.py` repaired the
  naive baseline (horizon-tiling bug, regression test in
  `tests/test_exp67_harness.py`); `exp67_s3_control.py` built the S3 control
  with a PIT assertion.
- `errors: 0/1224` rows. n=6 paired diffs per breaking-point cell (2 seeds) —
  significance flags are indicative, not definitive; the *pattern* across
  models (all break together) is the robust finding.
