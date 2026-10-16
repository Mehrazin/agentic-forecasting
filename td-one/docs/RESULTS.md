# TD-One Results

**Companion to `METHODOLOGY.md`.** Every number below comes from a
committed artifact in `experiments/`; commands to reproduce are in
`experiments/README.md` and `REPRODUCIBILITY.md`. Status: 2026-10-16.

Primary metric: **CRPS** (proper score on count distributions).
Secondary: MAE, 80% interval coverage.

---

## 1. Experiment results

### exp1 — pipeline validation (mock)

500×24 synthetic, M0 vs mock M1. Plumbing only; mocks are interface
stubs. ✅ Passed — the harness scores end to end.

### exp2 v2 — M0 vs real-weight TSFMs (the designed comparison)

3 pre-registered scenarios × 5 seeds, CRPS primary, oracle anchor.
⚠️ **v1 ("Chronos-2 beats M0 by 20%") is retracted** — circular DGP,
n=6, off-spec M0. See `docs/CRITIC_SYNTHESIS.md`.

| Scenario | Oracle | M0 | Chronos-2 | TimesFM-2.5 |
|---|---|---|---|---|
| stationary | 4.65 ± 1.27 | 4.85 ± 1.53 | 5.40 ± 2.18 | **4.28 ± 0.65** |
| drift | 5.02 ± 1.17 | 5.23 ± 1.34 | 6.04 ± 1.89 | **5.17 ± 0.80** |
| regime_shift | 5.58 ± 1.37 | 7.40 ± 1.77 | 7.44 ± 2.13 | **7.09 ± 1.48** |

Mean ± std across seeds. Paired Wilcoxon M1−M0: **all p > 0.18 — not
significant.** TimesFM-2.5 has the best mean in all three scenarios but
not significantly. Chronos-2 overcovers; M0 undercovers; TimesFM is
closest to nominal.

### exp3 — M2 ReAct vs M0 (3 pilot segments)

| Segment | M0 CRPS | M2 CRPS (mode) | Steps |
|---|---|---|---|
| seg-0 | 0.76 | 5.22 (real Groq) | 6 — finalized, worse |
| seg-1 | 1.33 | **0.82** (real Groq) | 5 — finalized, **win** |
| seg-2 | 1.40 | — (real Groq) | 6 — never finalized |

Mock mode: 0/3 (plumbing validation, expected — the scripted LLM adds
no skill). Real Groq with the patient retry policy: 2/3 traces
finalized; **first real-LLM win on seg-1** (MAE 0.14 vs 2.00).

⚠️ Honesty note: real-mode M2 still uses the `timesfm_mock`
numerical backend (`exp3_results.json: "forecast_model":
"timesfm_mock"`), so the seg-1 edge cannot be separated from the mock
sharpness artifact. What the run demonstrates is that the agent can
drive the tool loop to finalization and produce a winning trace — the
orchestration claim, not the numerical claim.

### exp4 — M3 hybrid vs M0 (6 macro scenarios)

| Scenario | LLM Δ | M0 CRPS | M3 CRPS |
|---|---|---|---|
| rate_shock_up | +0.18 | 19.87 | **16.33** |
| rate_shock_down | −0.25 | 4.18 | 5.01 |
| osfi_tightening | +0.30 | 9.95 | 15.22 |
| benign | +0.30 | 3.78 | 7.42 |
| family_squeeze | +0.20 | 14.60 | **7.96** |
| competitor_rate_war | +0.28 | 14.57 | **10.65** |

**3/6 beat M0 (real Groq).** Mock mode: 5/6 (scripted deltas —
information advantage by construction, not a skill claim). Failure
analysis: the LLM is directionally right 4/6 but got the **sign wrong**
on `osfi_tightening` (+0.30 vs true ≈ −0.35 — misread a stress-test
policy as pro-switching), spuriously adjusted the benign scenario
(+0.30 vs 0.00), and systematically undershoots magnitude (0.18–0.30
vs true 0.45–0.70). "Directionally right" doesn't predict the score:
`rate_shock_down` was a CRPS loss despite correct direction.
Exploratory framing: n=6 hand-designed scenarios, no significance
claimed.

### Scale-up — 10k×66, 14 segments × 5 origins

| Model | Mean CRPS | Mean coverage_80 |
|---|---|---|
| M0 | 5.360 | 0.671 |
| Chronos-2 (real weights) | 4.463 | 0.914 (overcovers) |
| TimesFM-2.5 (real weights) | 4.591 | 0.800 (nominal) |
| chronos2_mock | 3.194 | 0.714 |
| timesfm_mock | 3.176 | 0.714 |

Paired tests on the 14 shared segments (same seed, same DGP, same
origins — pairing valid): Chronos-2 wins 7/14 (paired-t p=0.212,
Wilcoxon p=0.583); TimesFM-2.5 wins 5/14 (p=0.341 / 0.583). **No
significant difference** — reproduces exp2 v2's conclusion at scale.
The mean gaps are driven by a few segments; the median picture is a
coin flip.

⚠️ `*_mock` rows are interface-only plumbing (sharpness artifact), not
evidence. Real-model outputs were monotone-rearranged to fix quantile
crossing (standard correction; disclosed, not hidden). Single seed,
single DGP — any future superiority claim must be pre-registered
across seeds/DGPs.

---

## 2. Ablations

### A. M3 delta bound — `experiments/abl_m3_bound.py` ✅ done

Varies `HybridReadout.MAX_DELTA` ∈ {0.25, 0.50, 0.75, 1.50, 3.00} on
the 6 exp4 scenarios (mock scripted deltas; bound is the only factor).

| Bound | Mean M3 CRPS (6 scenarios) |
|---|---|
| M0 (no adjustment) | 11.16 |
| 0.25 | 8.69 |
| 0.50 | 7.02 |
| **0.75** | 7.05 |
| 1.50 | 7.05 |
| 3.00 | 7.05 |

**Finding:** the bound binds only at 0.25; 0.50–3.00 are identical
because no scripted delta exceeds 0.5 in magnitude. The ±0.75 choice is
empirically indistinguishable from ±0.5 or ±1.5 on current scenarios —
its value is as a **safety rail against pathological LLM outputs**
(not yet observed), not as a tuned hyperparameter. Any future claim
about the bound needs scenarios with large deltas.

### B. M0 trailing window — `experiments/abl_m0_window.py` ✅ done

Varies `HistoricalBaseline(trailing_window)` ∈ {3, 6, 12} on the
standard scale-up config (10k×66, 14 segments × 5 origins), using the
identical M0 code path as the scale-up (window=6 reproduces the
published 5.360/0.671 exactly).

| Window | Mean CRPS | Mean MAE | Mean coverage_80 |
|---|---|---|---|
| 3 | **3.196** | 4.582 | 0.814 |
| 6 (published M0) | 5.360 | 7.224 | 0.671 |
| 12 | 8.731 | 11.302 | 0.500 |

**Finding:** shorter is better — the DGP's regime shifts punish long
memory. Window=3 beats the published M0 by 40% on CRPS and would beat
both real TSFMs (4.46 / 4.59). Two consequences: (1) the baseline has
headroom — M0's "strength" is partly a window choice, and any future
M1-vs-M0 comparison should tune the window or report the envelope;
(2) it sharpens the TSFM result — the models beat a *fixed* M0, not
the best M0.

### C. Retry policy (natural experiment) ✅ done

Three real-mode exp3 attempts, identical code except the retry/throttle
policy: attempt 1 (naive) → 1/3 finalized; attempt 2 (2.5s throttle) →
1/3; attempt 3 (patient: 6 attempts, 60s floor on 429) → 2/3 finalized
+ first win. **The binding constraint on the free tier is rate limits,
not model skill.**

### D. Mock vs real backend (natural ablation) ✅ done

exp3 mock (0/3) vs exp3 real-Groq (1/2 finalized win); exp4 mock (5/6,
scripted advantage) vs exp4 real (3/6, genuine judgment). Separates
orchestration skill from the information advantage.

### E. Fixed-Δ grid probe (from exp2) ✅ done

M3 drift probe on a toy series, Δ ∈ {±2.0, ±0.75, ±0.25, 0}:
CRPS 8.93 / 8.93 / 7.07 / **5.96** / 5.63 / 8.93 / 8.93 — best at
+0.25, catastrophic at ±2.0. Independent evidence that unbounded
adjustment destroys the forecast; kept as context in
`exp4_results.json`.

---

## 2b. Macro-grounded DGP — `experiments/exp8_macro.py` ✅ done

The default DGP has no macro dynamics, so macro-aware models cannot win
by construction. exp8 overlays a FRED-driven component: the 30y mortgage
rate's z-scored 3-month change shifts the switching logit each month
(β=0.6, calendar-aligned 2021-04 → 2026-09, 1:1 with FRED history).
2k customers × 66 months.

| Model | Mean CRPS | Mean MAE | Mean coverage_80 |
|---|---|---|---|
| M0 (macro-blind) | 2.305 | 3.120 | 0.786 |
| M0-macro-oracle (knows β + series) | **1.573** | 2.160 | 0.914 |
| M0-macro-est (OLS β̂, no oracle) | **1.624** | 2.121 | 0.786 |

**Finding:** the oracle cuts CRPS 32%; the simple estimated version
captures **93% of the oracle gain**. When macro structure exists, macro
awareness pays — and it doesn't need oracle knowledge. This justifies
the FRED integration workstream (17 series currently unused by any
model) and macro-evidence for the M3 agent on real data.

## 2c. M3 over-adjustment fixes — `experiments/exp6_m3plus.py` ✅ done

Two mechanisms for exp4's over-adjustment failures (wrong-sign deltas,
spurious benign adjustments): **shrinkage** (`ShrunkHybridReadout`, delta
× k, k tuned) and a **regime gate** (`gate_delta`: two-proportion z-test
on the panel, delta forced to 0 unless |z| ≥ threshold — the "do nothing"
prior). 12 new scenarios (6 shock + 6 benign), 6 tune / 6 held-out; mock
deltas replicate the real-Groq error model (seeded undershoot, 15%
sign flips, benign spurious).

Tuned: k*=0.5, threshold*=1.25.

| Variant | Tune mean CRPS | Held-out mean CRPS |
|---|---|---|
| M0 | 4.44 | 7.86 |
| M3 (bound 0.75) | 5.76 | 9.66 |
| M3+shrinkage | 4.73 | 8.62 |
| M3+gate | 4.47 | **7.60** |
| M3+both | **4.25** | 7.73 |

Gate confusion (held-out): 3 TP / 0 FP / 3 TN / 0 FN; all 12: 5/1/5/1.

**Finding:** plain M3 is worse than M0 — over-adjustment hurts. The gate
restores M3 past M0 on held-out (7.60 < 7.86) by killing catastrophic
spurious adjustments on benign scenarios (e.g. 10.95→3.94). Shrinkage
helps vs plain M3 but not vs M0; gate+shrinkage ≈ gate alone. The gate
is a **risk-management device, not a win-rate device** (1/6 strict wins
vs M0, but the wins are large and the ties are exact). Small-n caveat:
6+6 scenarios; mock error model is constructed, not observed.

## 2d. Better baseline models — `experiments/exp5_models.py` ✅ done

Nine models on the standard config (10k×66, seed 0): hierarchical
partial-pooling, EWM (half-life 2/4), split-conformal M0 (w3/w6),
equal + inverse-CRPS ensembles, vs M0 windows 3/6.

| Model | Mean CRPS | Mean MAE | Mean coverage_80 |
|---|---|---|---|
| m0_w3 | **3.196** | 4.582 | 0.814 |
| conf_w3 | 3.495 | 4.582 | **0.914** |
| ens_crpsw | 4.078 | 5.693 | 0.743 |
| ens_equal | 4.251 | 5.923 | 0.729 |
| ewm_hl2 | 4.442 | 6.196 | 0.714 |
| conf_w6 | 5.044 | 7.224 | **0.914** |
| hier | 5.254 | 7.073 | 0.686 |
| m0_w6 | 5.360 | 7.224 | 0.671 |
| ewm_hl4 | 6.555 | 8.798 | 0.571 |

**Findings:** nothing beats m0_w3 — the short window is stubbornly
optimal on this DGP (recency dominates). **Conformal is the calibration
fix** the project needed: coverage 0.671→0.914 (w6), and conf_w6 even
beats m0_w6 on CRPS (5.044 < 5.360). Hierarchical pooling adds almost
nothing (segments are large; would matter on thin segments). ewm_hl2 is
the best non-w3 model; long memory (hl4) hurts. Adaptive ensemble
weights help but can't beat the best component.

## 2e. Multi-seed scale-up — `experiments/multiseed_report.md` ✅ done

Seeds 0–4, 10k×66, 14 segments × 5 origins. M0 w6/w3, real Chronos-2,
real TimesFM-2.5. 19/20 runs (seed1-m0w6 lost to OOM).

| Seed | M0 w6 | M0 w3 | Chronos-2 | TimesFM-2.5 |
|---|---|---|---|---|
| 0 | 5.360 | 3.196 | 4.463 | 4.591 |
| 1 | — | 2.976 | 4.263 | 4.508 |
| 2 | 2.916 | 3.695 | 3.886 | 4.197 |
| 3 | 4.372 | 2.070 | 3.464 | 3.621 |
| 4 | 3.873 | 2.794 | 3.618 | 3.886 |

**Findings:**
1. M0's window is a genuine tuning parameter with **seed-dependent
   direction**: w3 beats w6 significantly on 3/4 seeds, but w6 beats w3
   significantly on seed 2 (p=0.022). Future M1-vs-M0 comparisons must
   tune the window or report the envelope.
2. **Neither TSFM beats M0_w6 significantly on any seed.** The seed-0
   mean edge does not generalize; on seed 2 M0_w6 wins significantly
   vs both. Honest summary: *no model significantly beats the baseline;
   the tuned-window baseline beats both TSFMs on 3/5 seeds.* This
   strengthens exp2 v2.
3. Coverage: Chronos-2 overcovers on **every** seed (0.90–0.97, most
   consistent pattern); TimesFM-2.5 is best calibrated (|cov−0.80|=0.06).

## 2f. Honest M2 — real Groq + real Chronos-2 backend ✅ done

`experiments/exp3_results_honest_backend.json` (`TD_ONE_TSFM_BACKEND=chronos2`).
The earlier real-Groq exp3 used `timesfm_mock`; this is the first fully-real
M2 run.

| Segment | M0 CRPS | M2 CRPS (real Groq + Chronos-2) | Steps |
|---|---|---|---|
| seg-0 | 0.76 | 5.40 | 5 |
| seg-1 | 1.33 | 1.25 | 6 |
| seg-2 | 1.40 | 4.35 | 5 |

**Clean negative result.** With a real numerical backend, the ReAct agent
loses badly on 2/3 segments and ekes out a marginal win on the third
(1.25 vs 1.33 — far smaller than the mock-backend "win" of 0.82 vs 1.33).
The agent's interventions hurt more than they help: it overrides a decent
Chronos-2 forecast with worse judgment. Orchestration does not compensate
for the backend, and the LLM's "reasoning" is net-negative here.

## 2h. M4 regime-reader agent — `experiments/exp11_agent_v2.py` ✅ done

Redesigns the agent's role after the honest-M2 negative: the LLM is a
**noisy sensor, never a forecaster**. It outputs only (regime class,
confidence, reason) — no numbers. Classical code decides whether to
intervene: the reading must clear a confidence threshold (abstention),
and optionally the panel's z-test must corroborate (two sensors).
Intervention applies the data-calibrated class delta (±0.25), never an
LLM number.

Mock sweep (6 scenarios, 20 reps; calibrated = confidence correlates
with correctness):

| Variant (acc 0.67) | Mean CRPS | Intervention rate |
|---|---|---|
| M0 | 4.823 | — |
| M4a calibrated (LLM sensor only) | **3.271** ± 0.779 | 46% |
| M4b calibrated (+ panel gate) | 3.341 ± 0.762 | 31% |
| M4a miscalibrated | 4.355 ± 0.844 | 29% |

**Calibration matters more than accuracy**: at 50% accuracy, a
calibrated reader (3.65) crushes M0; a miscalibrated reader at 83%
(4.08) barely beats it. The abstention mechanism is what makes it work.

Real-Groq validation (6 calls, one per scenario):

| Scenario | Reading | M0 | M4a | M4b |
|---|---|---|---|---|
| rate_shock_up | UP@0.85 ✅ | 4.88 | 1.68 | 1.68 |
| rate_shock_down | DOWN@0.85 ✅ | 3.84 | 1.26 | 3.84 |
| osfi_tightening | DOWN@0.85 ✅ | 1.10 | 1.97 | 1.10 |
| benign | NONE@0.90 ✅ | 0.84 | 0.84 | 0.84 |
| family_squeeze | DOWN@0.70 ❌ (true UP) | 4.84 | 9.56 | 9.56 |
| competitor_rate_war | UP@0.85 ✅ | 13.44 | 6.21 | 6.21 |
| **Mean** | 5/6 = 83% | **4.823** | **3.587** | **3.871** |

**Qualified positive.** M4a beats M0 (3.59 vs 4.82) with real Groq —
the first agentic variant to do so. But it is fragile: the single
confident error (family_squeeze, wrong at 0.70) was catastrophic
(9.56 vs 4.84), nearly wiping out the gains. The LLM's confidence was
*not* well-calibrated — wrong at 0.70, right at 0.85. A threshold of
0.75 would have abstained on the error. The panel gate (M4b) hurt more
than it helped here: it blocked the good rate_shock_down intervention.

**Takeaway:** the redesigned role works *when the LLM is calibrated*,
and the constraint held — the LLM never touched a number. But
confidence calibration is the load-bearing assumption, and 6 scenarios
is thin. Next: calibrate the confidence threshold on more scenarios,
or learn it.

## 2i. M5 ensemble router (REATS-style) — `experiments/exp12_router.py` ✅ done

Tests the literature survey's top-ranked agentic role (§2, §6.3): the
LLM outputs *weights over 4 forecast models* (m0_w3, conf_w6, chronos2,
timesfm25) with rationale, never values. Direct test: does LLM weighting
beat inverse-CRPS (exp5: 4.078)?

12 synthetic scenarios (4 stable / 4 shift / 4 volatile) with model
forecasts as Beta lattices matching our empirical findings (m0_w3 sharp;
conf wide; chronos2 wide+biased; timesfm25 calibrated).

Mock (insight_prob sweep):

| Scheme | Mean CRPS |
|---|---|
| equal | 3.779 ± 3.592 |
| inverse-CRPS | **3.713** ± 3.531 |
| mock (0% insight) | 3.713 (= inv-CRPS) |
| mock (50% insight) | 3.846 |
| mock (100% insight) | 3.663 |

Real Groq (4 scenarios, 4 calls):

| Scenario | Equal | Inv-CRPS | LLM router | LLM weights |
|---|---|---|---|---|
| stable | 2.51 | 2.88 | **1.78** | m0_w3 0.60 ✅ |
| shift_up | 4.02 | **2.89** | 4.18 | spread ❌ |
| shift_dn | 1.14 | 1.31 | 1.31 | — |
| volatile | 2.67 | **2.21** | 3.11 | chronos2 0.35 ❌ |
| **Mean** | 2.582 | **2.322** | 2.595 | |

**Clean negative.** The LLM router loses to inverse-CRPS (2.60 vs 2.32).
It handles the easy case (stable → m0_w3) but misapplies heuristics
elsewhere: in the volatile scenario it reasoned "volatile → wide
intervals → chronos2" — but chronos2 is wide *and biased*, while conf_w6
is wide and unbiased. The qualitative signal confounded it.

**Takeaway:** inverse-CRPS is a brutally strong baseline because trailing
performance is informative. The LLM needs insight *beyond* trailing
CRPS to win, and in this test it didn't have it. The literature's
top-ranked role fails here — possibly because our synthetic models are
too clean (real model failures are messier and more describable), or
because routing just doesn't beat good statistical weighting.

## 2g. Classical track — promoted baselines (`experiments/exp10_classical.py`) ✅ done

2000×66, two DGPs. Models: m0_w6, m0_w3, conf_m0 (static CQR),
aci_m0 (ACI γ=0.05), macro_m0 (FRED OLS), hier_thin vs m0_w6_thin
(binomial-subsampled p=0.1 panels).

**Standard DGP** (no macro structure — honest null for macro_m0):

| Model | Mean CRPS | Mean MAE | Mean coverage_80 |
|---|---|---|---|
| m0_w3 | **1.024** | 1.381 | 0.971 |
| m0_w6 | 1.401 | 1.913 | 0.886 |
| conf_m0 | 1.423 | 1.913 | 0.957 |
| aci_m0 | 1.423 | 1.913 | 0.957 |
| macro_m0 | 1.472 | 1.983 | 0.829 |

**Macro DGP** (FRED overlay, β=0.6):

| Model | Mean CRPS | Mean MAE | Mean coverage_80 |
|---|---|---|---|
| m0_w3 | **1.491** | 1.920 | 0.843 |
| macro_m0 | 1.627 | 2.121 | 0.786 |
| conf_m0 | 2.231 | 3.120 | 0.914 |
| aci_m0 | 2.231 | 3.120 | 0.914 |
| m0_w6 | 2.305 | 3.120 | 0.786 |

**Thin segments** (hierarchical niche):

| DGP | hier_thin CRPS | m0_w6_thin CRPS |
|---|---|---|
| standard | **0.301** | 0.317 |
| macro | **0.359** | 0.397 |

**Findings:**
1. **Honest null confirmed:** macro_m0 is *worse* than m0_w6 on the
   standard DGP (1.472 vs 1.401). The FRED overlay adds noise when
   there's no macro structure — as it should.
2. **Macro recovery:** on the macro DGP, macro_m0 (1.627) captures 93%
   of the oracle gain ((2.305−1.627)/(2.305−1.573)), matching exp8.
   It beats every w6-based model, though m0_w3 (1.491) still wins
   outright — recency dominates even with macro structure.
3. **Hierarchical has a niche:** on thin (p=0.1 subsampled) segments,
   hier beats m0_w6 by 5–10% on both DGPs. Partial pooling helps where
   the prior can bind; on large segments it adds nothing (exp5).
4. **ACI null:** aci_m0 is identical to static conf_m0 on both DGPs.
   The macro shifts are smooth (not sharp breaks) and 4 calibration
   months give ACI little room to adapt. The literature's drift case
   needs sharper regimes to matter — or longer calibration.

**Promoted to the pipeline:** `--conformal` (default on) makes conformal
M0 the published baseline in `scripts/scale_up.py`; `--m0-window auto`
selects 3/6/12 by trailing-origin CRPS validation per forecast.

## 2j. TSFM + FRED rematch — the fair fight (`experiments/exp13_tsfm_macro.py`) ✅ done

Real Chronos-2 / TimesFM-2.5 never beat M0 significantly on 5 seeds —
but they were tested WITHOUT macro information, while exp8 showed
macro structure is worth 32% CRPS when present (LITERATURE_SURVEY
§6.3.5: "the fair rematch the TSFMs deserve").

**Covariate path:** neither TSFM accepts exogenous covariates natively.
Chronos-2's `predict` takes only `inputs`; the installed timesfm 3.x
`forecast(horizon, inputs)` has no covariate path. So macro-conditioning
uses the honest residual fallback in
`src/td_one/models/tsfm_macro.py::MacroConditionedTSFM`: estimate β by
OLS of trailing logit(rates) on the FRED z-series (MacroM0's estimator,
reused by composition), forecast the TSFM on the macro-residualized
series, re-add the shift per quantile via
sigmoid(logit(q) + β̂·z_forecast). On DGPs without macro structure β̂≈0
and this reduces exactly to the plain TSFM (regression-tested in
`tests/test_tsfm_macro.py`).

2000×66, seeds 0–2, two DGPs. Models: conf_m0_auto (conformal, window
auto — the published baseline), macro_m0, plain chronos2/timesfm25
(real weights), and their macro-conditioned versions.

**Standard DGP** (honest null — macro should NOT help):

| Seed | conf_m0_auto | macro_m0 | chronos2 | chronos2_macro | timesfm25 | timesfm25_macro |
|---|---|---|---|---|---|---|
| 0 | **1.142** | 1.472 | 1.565 | 1.596 | 1.580 | 1.628 |
| 1 | **1.127** | 1.254 | 1.318 | 1.486 | 1.331 | 1.461 |
| 2 | **0.963** | 1.536 | 1.220 | 1.631 | 1.278 | 1.729 |

**Macro DGP** (FRED overlay, β=0.6):

| Seed | conf_m0_auto | macro_m0 | chronos2 | chronos2_macro | timesfm25 | timesfm25_macro |
|---|---|---|---|---|---|---|
| 0 | 1.770 | **1.627** | 2.033 | 1.770 | 2.126 | 1.763 |
| 1 | 1.278 | **1.207** | 1.653 | 1.483 | 1.733 | 1.423 |
| 2 | 1.320 | 1.318 | 1.520 | **1.274** | 1.538 | 1.293 |

**Findings:**
1. **The mechanism works.** Macro-conditioning improves BOTH TSFMs on
   the macro DGP on all 3 seeds: chronos2_macro beats plain chronos2 by
   13%/10%/16%, timesfm25_macro by 17%/18%/16%. The residual approach is
   a real, consistent win whenever macro structure exists.
2. **The honest null is stronger than a null.** On the standard DGP,
   macro-conditioning HURTS both TSFMs on all 3 seeds — β̂ is pure noise
   there and the residualize/re-add round trip injects it. Conditioning
   is not free; it should be gated on evidence of macro structure.
3. **But the rematch headline is a draw.** The macro-conditioned TSFMs
   beat the conformal auto-window baseline only 1/3 seeds (seed2:
   1.274/1.293 vs 1.320), tie seed0, lose seed1. The TSFMs' residual
   base forecast is still weaker than M0's Beta-Binomial core — macro
   information closes the gap but doesn't reverse it.
4. **Classical still owns the macro niche.** macro_m0 beats
   conf_m0_auto on seeds 0–1 (1.627 vs 1.770; 1.207 vs 1.278) and ties
   seed2 — simpler, faster, and the best macro play. (Caveat: macro_m0
   is not conformalized; its 80% coverage is 0.79–0.84 vs 0.89–1.00
   for conf_m0_auto.)
5. **Chronos-2 still overcovers** (1.00 on standard DGP, all seeds);
   TimesFM-2.5 stays closest to nominal among TSFMs.

**Takeaway:** give TSFMs macro information and they get reliably
better — but "better TSFM" is not "better than M0". The fair rematch
confirms the mechanism (residual conditioning) while leaving the
baseline hierarchy intact: conformal auto-M0 on standard DGP, classical
MacroM0 on macro DGP. Results JSON:
`experiments/exp13_tsfm_macro_results.json`.

## 3. Planned experiments

1. **Multi-seed scale-up** (seeds 1–4): the paired non-significance
   must survive DGP resampling before any TSFM claim.
2. **M2 tool ablation** (real LLM): ReAct with tools removed one at a
   time (no evidence / no data_segment / forecasting-only) — which
   tools carry the skill? Requires paid-tier Groq (free-tier budgets
   can't afford it).
3. **M2 step-budget ablation**: max_steps ∈ {2, 4, 6, 10}.
4. **Calibration study**: the open problem — M0 undercovers (0.67),
   Chronos-2 overcovers (0.91), only TimesFM is nominal. Conformal or
   post-hoc recalibration per segment.
5. **Real-data validation**: the simulator is ours; portfolio data is
   the real test.
6. **M3 "do nothing" prior**: the benign-scenario over-adjustment
   (+0.30 vs 0) suggests the agent needs an explicit abstention
   mechanism, not just a bound.
7. **FRED-vintage PIT audit** (client support ✅ done; audit open): before
   any model consumes FRED series, `FredClient.observations()` exposes
   `vintage_dates` — observation-date cutoff is not vintage-correct for
   revised series (see critic report). What remains is the actual audit:
   release-lag metadata (`last_updated` is now captured in
   `fred_manifest.json`) and release-aware join rules for the future
   panel join.
8. **LLM ensemble router** (REATS-style; see `docs/LITERATURE_SURVEY.md`
   §6.3): LLM outputs weights over classical/TSFM forecasts with
   rationale, never values — test vs inverse-CRPS weighting (exp5:
   4.078). Best-supported agentic role in the literature.
9. **LLM regime narrative → gate**: LLM emits structured regime
   assessment (direction + confidence) that modulates the statistical
   gate threshold, instead of numerical deltas.
10. **ACI upgrade to conformal.py**: adaptive conformal inference
    (one γ) vs static CQR on shock scenarios — handles regime drift.
11. **DGP realism upgrade**: burnout state + 30-month seasoning ramp +
    S-curve refinance incentive (industry prepayment canon; see
    LITERATURE_SURVEY.md §5). Current DGP may be too easy for M0.
12. **TSFM + exogenous FRED** ✅ done (§2j): macro-conditioned
    Chronos-2/TimesFM-2.5 via the residual approach vs conformal
    auto-M0 — mechanism works (10–18% vs plain TSFM, 3/3 seeds), but
    beats the baseline only 1/3 seeds; classical macro_m0 still best.

---

## 4. Artifact map

| Artifact | Content |
|---|---|
| `experiments/exp{1,2,3,4}_report.md` | Per-experiment reports |
| `experiments/exp{2,3,4}_results.json` | Raw results |
| `experiments/scale_up_report*.md` + `scale_up_results*.json` | Scale-up (mock, real-chronos2, real-timesfm25) |
| `experiments/abl_m0_window_results.json` | Ablation B |
| `experiments/abl_m3_bound_results.json` | Ablation A |
| `experiments/exp6_m3plus.py` + `exp6_results.json` | M3 over-adjustment fixes ( §2c ) |
| `experiments/evidence_corpus/corpus.json` | Frozen 12-entry evidence corpus |
| `docs/METHODOLOGY.md` | CVPR-style methodology reference |
| `docs/CRITIC_SYNTHESIS.md` | 2026-10-15 six-angle audit + exp2 v1 retraction |
| `REPRODUCIBILITY.md` | Exact reproduction steps |
