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
