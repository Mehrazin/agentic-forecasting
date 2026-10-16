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
| `experiments/evidence_corpus/corpus.json` | Frozen 12-entry evidence corpus |
| `docs/METHODOLOGY.md` | CVPR-style methodology reference |
| `docs/CRITIC_SYNTHESIS.md` | 2026-10-15 six-angle audit + exp2 v1 retraction |
| `REPRODUCIBILITY.md` | Exact reproduction steps |
