# Nonsignificant Positives Audit: exp1–exp50

**Date:** 2026-10-18
**Purpose:** Complete inventory of every *positive-direction but nonsignificant*
(or otherwise excluded) effect across the TD-One program, with a verdict on
whether each is honestly combinable into the proposed method. This is the
evidence base for exp51 (kitchen sink).

**Rule:** a component enters the kitchen sink only if (a) its direction was
positive, (b) no *stronger later* experiment overturned it, and (c) it can be
implemented without inventing signal (no synthetic inputs, no unfunded
assumptions).

---

## Candidate table

| # | Exp | Effect (verified) | p / n | Why excluded from proposed | Combinable? |
|---|---|---|---|---|---|
| 1 | exp18b | TimesFM_cal −12.2% vs ETS on SMM (0.000497 vs 0.000566, 32/50 wins) | p=0.076, n=50 | n.s.; exp50 found 0% selection as regime expert on M4 | **YES — SMM only** |
| 2 | exp33 | Agent macro features (ff_3m_chg, term_spread) +1.21% held-out on SMM (6/12 wins) | n.s., n=12 | tiny n, n.s. | **YES — SMM only** |
| 3 | exp22 | M2 v2 code-exec agent +6.6% when active (9/60 origins, 6/9 wins); 3 catastrophic losses (−59% to −93%) | p=0.73 overall, n=60 | high variance; needs live LLM per origin | **NO** — see note A |
| 4 | exp41 | Learned selection +0.9% gap closed vs hand-rules; tied with best single | n/a (no p reported) | superseded by exp43 (regime router does this better) | **YES — as meta-blend** |
| 5 | exp44 | News-as-selector: synthetic 0.367→0.522 accuracy | p=0.0068 synthetic | **exp46 supersedes**: dense real news, p=0.48, no help | **NO** — see note B |
| 6 | exp36 | Agent discriminates seasonality (produce 0.375 vs headline 0.000) | diagnostic | not a forecast mechanism | **NO** — diagnostic only |
| 7 | exp11 | LLM regime sensor 3.59 vs 4.82 (M4a vs M0); 1 confident error catastrophic | fragile, n=200 mock | already in proposed as *gated* optional module | **Already integrated** (gated); live-LLM CV infeasible |
| 8 | exp5 | Conformal calibration: coverage 0.671→0.914 | calibration | already in proposed | **Already integrated** |
| 9 | exp10 | Hierarchical reconciliation +5–10% on thin segments | niche | pipeline has no hierarchy | **NO** — no hierarchy in data |
| 10 | exp13 | TSFM residual macro-conditioning: 1/6 helped, 4/6 hurt | p=0.375 (sign test) | direction *negative* overall | **NO** — was negative, not positive |
| 11 | exp4 | M3 hybrid 3/6 beat M0 (real Groq); direction right 4/6 | n=6 | fixed/superseded by exp6 gate (in proposed) | **Already integrated** (via gate) |
| 12 | exp17 | News regime reader pilot 3/4 correct | n.s., n=6 | pilot; superseded by exp44→exp46 (killed) | **NO** — superseded by kill |
| 13 | exp50 | Learned gate thresholds (0.5/2.5/0.02) vs hand (0.6/3.5/0.03) | p=0.32 | n.s. | **YES** — cheap, function exists |
| 14 | exp50 | Loss-regime clustering vs hand regimes | p=0.44 | n.s., hand wins | **YES** — cheap, but expected null |

### Note A (exp22)
The code-executing agent's *net* effect was +1.3% (n.s.); the +6.6% is
conditional on the 9/60 origins where it acted, with 3 catastrophic losses
from overconfident intervals. There is no honest deterministic proxy: the
adjustment direction came from the LLM's analysis. Its calibrated descendant
— corroboration gating + James-Stein shrinkage (`decision_policy.py`) — is
already the "do nothing" prior in the proposed method. Running 600+ live
Groq calls inside blocked CV is infeasible and would not be the same
mechanism anyway.

### Note B (exp44 → exp46)
exp44's positive is *synthetic-only*. exp46 is a strictly stronger test
(real BoC, 86% news coverage, dense variants): news features don't help
regime classification (p=0.48) or forecasting on any target. Including news
in the kitchen sink would mean ignoring stronger later evidence, not testing
a significance threshold. Excluded on those grounds.

---

## What the kitchen sink (exp51) actually tests

| Component | Source | Where it applies |
|---|---|---|
| + TSFM_cal 5th expert (TimesFM-2.5 native quantiles, normalized space) | exp18b | SMM only |
| + Macro features: ff_3m_chg, term_spread (PIT-safe, month-end FRED) | exp33 | SMM only |
| + Meta-learner blend: 0.5·w_router + 0.5·P_meta(model\|features) | exp41 | all continuous |
| + Learned gate thresholds (grid search per fold) | exp50 | all continuous |
| + Loss-regime spec (cluster loss vectors) | exp50 | all continuous (expected null) |

Design: ablations (each alone) + full kitchen sink, vs proposed baseline,
blocked 5-fold CV, paired t-tests, on M4 / Synthetic / SMM / Food CPI.
BoC: N/A (discrete forecaster already includes its regime gate; no
nonsignificant positive applies).

**Key question:** does adding nonsignificant positives help (real but
underpowered) or hurt (noise)? A worse kitchen sink is a finding — it
validates the significance filter.
