# exp69: Anticipatory LLM Shift Adjustment — Report

**Date:** 2026-10-20
**Status:** COMPLETE — honest negative (pre-registered kill triggered).
**Data:** `experiments/exp69_results.json` (192 rows, 0 errors). Analysis inline
in `experiments/exp69_anticipatory_llm.py::analyze`.

## Pre-registered design (locked before running)

exp68's *reactive* guard (detect after onset → truncate context) failed
(+0.5%, p=0.824): truncation hurts, correct detection ≠ benefit. exp69 tests
the *anticipatory* variant the user proposed: the LLM reads PIT-safe news
*before* the shock and a calibrated judgmental overlay shifts the TSFM's
forecast for the horizon steps the shock will hit.

**Division of labor (the key design choice):** the LLM outputs structure
only (shift type, direction, magnitude bucket, timing — never numbers);
a calibration table maps (shift_type, bucket) → level shift in rate space,
learned as the median post-vs-pre level change on *past* shock episodes
(calibration seeds 200–202, disjoint from test seeds 300–302). The LLM
classifies; statistics quantify.

- **Shifts:** S1 (rate shock), S2 (policy intervention). **Severities:** 2σ/4σ.
- **Origins:** STRADDLE=(27,28,29) primary (window covers onset=30, shock
  news published at months 27–29); NONEWS=(24,25,26) do-no-harm (window
  covers onset but no news yet — classifier must output none);
  POST=(31,33) do-no-harm (shock already in context — overlay must not fire
  by construction).
- **Arms:** A=Chronos-2 alone; B=A+LLM overlay; C=A+oracle overlay (perfect
  classification — mechanism ceiling); B_noramp (ramp off); B_fixed
  (fixed Δ=0.05 instead of calibrated buckets).
- **Synthetic news** announces each shock at months 27–29 with magnitude
  cues (50bp/100bp, "targeted"/"sweeping") and relative timing ("in 3
  months"/"next month"), wired through `articles_visible_asof` (PIT-safe).
  Distractor articles included. Every article predates the shock (asserted).
- **Timing rule:** the shift applies only to horizon steps at/after the
  predicted onset, and only when the origin precedes the predicted onset
  (no double-counting once the shock is in context — the exp68 lesson).
  "low" timing confidence abstains.
- **Primary claim:** B beats A on straddle origins (paired t-test, p<0.05).
- **Kill:** if B doesn't beat A at p<0.05, dead — no iteration.

## Verdict: FAIL (honest negative)

| Test | Result |
|---|---|
| **Primary:** B vs A, straddle origins | mean ΔCRPS −0.0050 (**+10.2%** rel.), **p=0.231**, win rate **36%** → **FAIL** |
| Mechanism ceiling: C vs A | −0.0056 (**+11.4%**), p=0.214, win 44% → FAIL |
| LLM error cost: B vs C | +0.0006, p=0.731 — LLM ≈ oracle |
| Ablation ramp off (B_noramp vs A) | identical to B (all signals "high" confidence) |
| Ablation fixed Δ (B_fixed vs A) | +7.8%, p=0.072, win 42% → FAIL |
| Do-no-harm (NONEWS origins) | Δ=0.0000 exactly, p=1.0 → **PASS** |
| Do-no-harm (POST origins) | Δ=0.0000 exactly, p=1.0 → **PASS** |

Per the pre-registration: **dead, no iteration.**

## What the negative teaches

1. **The mechanism is the limit, not the LLM.** The oracle (perfect shift
   type, direction, bucket, timing) reaches only +11.4% at p=0.21 with a 44%
   win rate. Even perfect anticipation doesn't significantly help CRPS here:
   the calibrated *median* level shift is a blunt instrument against
   per-episode noise. Realized S1/4σ level changes across three test seeds
   were 0.12 / 0.25 / 0.17 while the calibration said 0.14 — and monthly
   Beta-Binomial noise is ±0.1. A fixed median shift frequently overshoots
   or undershoots the episode it is applied to.
2. **It works where the shift dwarfs the noise, and only there.** S1/4σ:
   B +30% (win 67%), C +35% (win 89%). S1/2σ: B **−17%** (win 33%);
   S2/2σ: −4%; S2/4σ: −11%. Below 4σ on a clean level shift, the calibrated
   shift adds more error than it removes. The policy shift (S2, smaller
   calibrated Δs + 2-month lag structure) never pays.
3. **Earlier origins are worse.** Win rate by origin: 27 → 0%, 28 → 50%,
   29 → 58%. Three months out, even the oracle hurts more often than not —
   the forecast window's pre-shock steps are accurate and the shifted steps
   inherit the full per-episode level noise.
4. **LLM classification errors are real but not the binding constraint.**
   B≈C overall (p=0.73). Two genuine LLM errors observed: (a) at origin 27,
   S1/4σ, the LLM reported `months_from_now=0` instead of 3 (arithmetic
   slip) — the guard correctly abstained, but it cost the single biggest
   oracle win (0.127→0.073 on seed 301); (b) S2/2σ was bucketed "small",
   for which the calibration table has no entry (design gap: the schema
   allows "small" but calibration only covered medium/large) — the overlay
   no-op'd, which accidentally protected B since the oracle's "medium"
   shift hurt on all three seeds.
5. **Do-no-harm held.** The classifier never hallucinated a shift from
   distractors (36/36 none at NONEWS origins) and never fired post-onset
   (24/24). The abstention logic (low confidence / origin ≥ predicted
   onset / missing calibration) works as designed.

## Engineering notes

- **Parser bug caught by the first real run:** Gemini wraps JSON in
  ` ```json ` fences; the classifier's `json.loads` failed silently and 27/36
  straddle signals fell back to NULL — the first run measured the parser,
  not the LLM. Fixed with `_extract_json` fence-stripping + 2 regression
  tests (`tests/test_anticipatory_adjuster.py`, 18/18 green). The LLM's
  actual classifications are excellent (correct type/direction/bucket and
  month arithmetic in every probed case).
- **Mock-mode bugs caught before the real run:** (a) the LLM cache key
  omitted article bodies, colliding 2σ/4σ classifications; (b) the mock
  didn't anchor "next month" to the article's publication month, firing the
  overlay at post origins. Both fixed; the real prompt carries the same
  anchoring instruction.
- `experiments/exp69_llm_cache.json` (32 cached classifications) is a local
  cost-saving artifact, not committed. 32 real LLM calls (flash-lite).
- New module: `src/td_one/agent/anticipatory_adjuster.py`
  (`ShiftSignal`, `AnticipatoryClassifier`, `calibrate_level_shifts`,
  `apply_overlay`). 18/18 tests green; ruff clean.

## Implications

- Do **not** pursue anticipatory level-shift overlays on this surface. The
  oracle ceiling (+11%, n.s.) says the information isn't worth the
  miscalibration risk at these noise levels — and that conclusion already
  assumes perfect news, perfect timing, and a cooperative DGP.
- The one place the mechanism showed life (S1/4σ, +30–35%) is documented
  above as an observation, not an iteration license — per the
  pre-registration, no follow-up on this design.
- Standing result across exp67→exp69: **no tested intervention beats
  Chronos-2-alone on shifted series.** exp67: no robust model exists.
  exp68: reactive truncation hurts. exp69: anticipatory shifting doesn't
  significantly help. The TSFM's own full-context forecast remains the
  best defense.

## Naming note

`docs/LOAN_ATTRITION_PLAN.md` previously reserved "exp69" for the Track 1+2
integration (aspirational, no code). This experiment takes the exp69 number
per the task; the integration plan is renumbered to **exp70**.

## Follow-up: exp69b (broad synthesis)

User-directed variant: the LLM synthesizes across a broad macro landscape
(semantic RAG over 12+ topics) instead of rate news only. Result: synthesis
*hurt* (D vs B′ −5.0%, win 5.6%) — corroborating context dilutes timing
precision. See `experiments/exp69b_report.md`.
