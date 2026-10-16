# M5 Ensemble Router: Post-Mortem and V2 Redesign

## The failure (exp12)

Real Groq: **2.60** mean CRPS vs inverse-CRPS: **2.32**. The LLM router lost
to a simple statistical baseline. Clean negative.

### Per-scenario breakdown

| Scenario | Type | Equal | Inv-CRPS | LLM | Verdict |
|---|---|---|---|---|---|
| s_stable_1 | stable | 2.507 | 2.884 | **1.784** | LLM won |
| s_shift_up_1 | shift | 4.017 | 2.887 | 4.183 | LLM lost |
| s_shift_dn_1 | shift | 1.140 | 1.310 | 1.308 | Tie |
| s_vol_1 | volatile | 2.665 | 2.207 | 3.107 | LLM lost badly |

Trailing CRPS ranking (the signal the LLM had): timesfm25 (3.328) <
chronos2 (3.733) < conf_w6 (4.345) < m0_w3 (5.097).

### Root causes

**1. The LLM misread the quantitative inputs.**
In s_vol_1, it gave chronos2 the highest weight (0.35) with rationale
"chronos2 and timesfm25 have strong recent CRPS" — but chronos2 was
2nd-worst (3.733 vs timesfm25's 3.328). The trailing CRPS was presented
as an unordered list; the LLM failed to rank it correctly.

**2. Qualitative heuristics overrode quantitative signals.**
In s_shift_up_1, the LLM split weight between conf_w6 (0.35) and
timesfm25 (0.35), rationale: "conf_w6 and timesfm25 match the volatile,
regime-shift scenario". It conflated "shift" (bias regime, timesfm25 best)
with "volatile" (noise regime, conf_w6 best). The model card heuristic
("conf_w6 best when volatile") became an excuse to ignore the trailing
numbers.

**3. Wrong output space.**
Asking for 4 precise weights summing to 1.0 forces the LLM to do
arithmetic it's bad at. It spreads weights thinly (0.1/0.35/0.2/0.35)
instead of concentrating on the best model.

**4. No anchoring.**
Starting from scratch each time lets the LLM drift far from the strong
inverse-CRPS baseline. There was no default it had to justify deviating
from.

The one win (s_stable_1) is telling: the LLM correctly concentrated on
m0_w3 (0.6) because the model-card heuristic ("m0_w3 best in stable")
happened to align with the right answer. When heuristics and numbers
disagree, the LLM follows heuristics — and loses.

## V2: critic-constrained router

The LLM acts as a **critic** of the inverse-CRPS baseline, not a
replacement.

### What changed

1. **Anchored on the strong baseline.** V2 starts with inverse-CRPS
   weights. The LLM can only AGREE (keep them) or propose ONE bounded
   adjustment.

2. **Ranked CRPS presentation.** Trailing CRPS is shown explicitly ranked
   (1st/2nd/3rd/4th with gaps vs best), making it impossible to misread
   the ordering that caused the s_vol_1 failure.

3. **Constrained output space.** Instead of 4 weights, the LLM outputs a
   single decision: agree, or adjust one model up/down by small (±0.08)
   or medium (±0.15). The adjustment is applied mechanically with
   clipping to [0.05, 0.85].

4. **Numbers-first prompting.** The prompt explicitly states: "Vague
   model-card matching is NOT enough — the numbers come first." The LLM
   must cite the CRPS ranking in its rationale.

### Why it might work

- The degrees of freedom collapse from "4 continuous weights" to "1 of
  9 discrete actions" (agree + 4 models × 2 directions × 2 strengths,
  though strength is part of the action).
- A wrong call moves at most 0.15 weight — it cannot blow up the ensemble
  the way v1's 0.35-on-chronos2 did.
- When the LLM has no strong reason to intervene, it agrees and we get
  the baseline (2.32) for free.
- The failure mode is now "LLM agrees too often" (harmless) rather than
  "LLM makes bad bets" (harmful).

### Why it might still fail

- If the LLM systematically adjusts in the wrong direction, v2 could
  still underperform the pure baseline. The bounded step limits damage
  to ~0.15 weight shift per scenario.
- The critic pattern assumes the baseline is usually right. If
  inverse-CRPS is systematically wrong in a regime, v2 inherits that.

## Implementation

- `src/td_one/agent/ensemble_router.py`: v1 kept as `read_weights_real`;
  v2 as `read_weights_critic` + `parse_critic_decision` +
  `apply_critic_decision`. `CriticDecision` is frozen (audit trail).
- `tests/test_ensemble_router.py`: 12 new v2 tests (parsing, bounded
  application, clipping feasibility, determinism, immutability).
- All 24 tests pass; ruff clean; mypy --strict clean.

## Next step

Run exp12 with v2 (`read_weights_critic` instead of `read_weights_real`)
on the same 4 scenarios. If v2 ≥ inverse-CRPS, the critic pattern is
validated. If v2 < inverse-CRPS, the LLM adds no value even when
constrained — consider killing M5 entirely.
