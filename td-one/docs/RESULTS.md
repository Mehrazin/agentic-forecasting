# TD-One Results — R2 Series

**Companion to `METHODOLOGY.md`.** Every number below comes from a
committed artifact in `experiments/`; commands to reproduce are in
`experiments/README.md` and `REPRODUCIBILITY.md`. Status: 2026-10-16.

**R2 = Round 2.** PIT-safe reruns of the first series (exp1–exp15, archived
in `docs/RESULTS_ARCHIVE.md`). The data audit found P0-1 (FRED z-score
future leak); R2 reruns all affected experiments with `asof` filtering.

Primary metric: **CRPS** (proper score on count distributions).
Secondary: MAE, 80% interval coverage.

## Document map

| Doc | What it is |
|---|---|
| `docs/METHODOLOGY.md` | CVPR-style methodology reference |
| `docs/RESULTS.md` (this) | R2 experiment numbers |
| `docs/RESULTS_ARCHIVE.md` | exp1–exp15 (superseded, preserved for provenance) |
| `docs/DATA_AUDIT.md` | 2026-10-16 audit: 2 P0, 5 P1, 5 P2 + fixes |
| `docs/RAG_DESIGN.md` | News RAG system design |

---

## R2 Experiment registry

| # | Rerun of | File | Status | One-line verdict |
|---|---|---|---|---|
| R2-1 | exp10 (classical) | `experiments/r2_exp10_classical.py` | ✅ done (5 seeds) | **No evidence: macro worse than m0_w3 by 0.19 (p=0.064)** |
| R2-2 | exp13 (TSFM macro) | `experiments/r2_exp13_tsfm_macro.py` | ✅ done (lagged rerun) | **Residual conditioning hurts C2 4/6 — method-specific** |
| R2-3 | exp15 (corrected macro) | `experiments/r2_exp15_macro_corrected.py` | ✅ done | PIT-safe = harmless on v2 DGP |
| R2-4 | exp14 (M4 real) | `experiments/r2_exp14_common_grid.py` | ✅ done | Common-grid rerun: TSFM win survives (-24.1% Chronos2, -22.6% TimesFM) |
| exp17 | — (new) | `experiments/exp17_news_regime.py` | ✅ done (pilot) | **News+RAG+regime reader end-to-end: 3/4 correct (n=6, n.s.)** |
| exp18 | — (new) | `experiments/exp18_smm_full.py` | ✅ done | **Calibrated TSFMs beat m0 30% on 63-mo SMM (p=0.0012)** |

---

## R2-1: Classical with lagged PIT-safe macro — **no evidence** (5 seeds)

| Model | Mean CRPS (5 seeds) | SE |
|---|---|---|
| m0_w3 | **1.2804** | 0.0460 |
| macro_m0_w3 | 1.4665 | 0.0978 |
| macro_m0_w6 | 1.6475 | 0.0828 |
| m0_w6 | 1.7705 | 0.1335 |

**RETRACTION (2026-10-17):** The original R2 conclusion ("leak was 100% of benefit")
was based on a bug — `fred_z.get(forecast_month)` looked up t+1 in a dict truncated
at t, always returning 0.0. The single-seed lagged rerun showed -0.357 vs m0_w6,
but multi-seed evaluation (seeds 0-4) confirms Argon's window-confounding hypothesis:
macro_m0_w3 vs m0_w3 is **+0.1861 (macro worse), p=0.064**. The lagged z[t] was
compensating for m0_w6's stale average, not adding true macro signal. The original
-0.357 did not replicate (seed 0 was lucky).

**Verdict:** No evidence macro helps over a responsive baseline. Classical macro track is closed.

## R2-2: TSFM + lagged FRED — **exploratory negative** (n=6, no significance testing)

Lagged residual conditioning: 1/6 helps (-1.2%), 4/6 hurt (+3% to +23%), 1/6 ties.
The residual approach — subtracting β·z from training inputs then re-adding at
forecast time — distorts the input distribution without adequate compensation.

**Caveats (statistical):** n=6 with no p-values or confidence intervals — anecdotal,
not established. A sign test gives p=0.375 two-sided (1 win, 4 losses, 1 tie;
n=5 non-tied). The pattern is
suggestive (losses dwarf the single win) but needs larger evaluation. Also note
`conf_m0` beats both Chronos2 variants in every seed/DGP — the relevant baseline
isn't "plain Chronos2" but "best available model."

## R2-3: Corrected macro — PIT safety neutralizes the damage

R1: estimated overlay hurt by 0.151. R2: m0=0.6092, oracle=0.6199 (-0.0107),
est=0.692 (-0.0828). Even the oracle hurts on v2 DGP — DGP-specific, not harmless.

## R2-4: exp14 rerun on a common 0.10–0.90 grid — **TSFM win survives** (done 2026-10-17)

All models scored on the same 9-level quantile grid (classical residual quantiles
computed directly; Chronos2 interpolated interior, no extrapolation; TimesFM exact):
naive 462.84, timesfm25 358.13 (**−22.6%**), chronos2 351.18 (**−24.1%**).
The TSFM edge is not a scoring artifact. `experiments/r2_exp14_common_grid_results.json`.

## Bottom line (updated 2026-10-17)

**The macro track is method-specific and statistically preliminary.** Lagged PIT-safe
macro shows exploratory promise for classical models (-0.357 vs m0_w6, but loses to
m0_w3; single seed) while residual conditioning shows exploratory negative signals
for TSFMs (n=6, no significance testing). The original "macro is dead" verdict was
based on a dict-lookup bug. The honest conclusion: **we don't know yet** — the
findings are suggestive but need multi-seed confirmatory evaluation.

**exp18b (fair comparison):** TimesFM beats the best fairly-constructed classical
baseline by **12.2% CRPS** (0.000497 vs ETS+resid 0.000566, p=0.076, wins 32/50
origins — not significant at α=0.05). Against the weaker m0_w3 baseline the gap
is 17.9% (p=0.0079), but ETS is the proper classical comparator and Argon was
right to demand it. All models on a common 19-level quantile grid with proper
uncertainty wrappers. The TSFM edge is real point-accuracy + better spreads,
but smaller than initially claimed.

Full analysis: `experiments/r2_comparison.md`.

---

## What carries forward from exp1–exp15 (not rerun)

These results were **not** affected by the P0 leaks:

- **M4 regime reader** (exp11): 3.59 vs M0 4.82 — real Groq, no FRED dependency
- **M5 router** (exp12): clean negative — no FRED dependency
- **M4 Monthly benchmark** (exp14): TSFMs beat classical ~20% — no FRED z-scores
- **Simulator v2** (exp9): DGP work — unaffected

---

## exp17: News-augmented regime reader (end-to-end)

First run wiring RAG retrieval → v2 regime classification together.
4 months of 2026, 2 RAG searches each, real Groq (12 calls total).

| Month | Predicted | Expected | Truth (rate move) | Score |
|---|---|---|---|---|
| 2026-02 | UP @ 0.60 | DOWN | +0.40pp | ❌ wrong |
| 2026-03 | UP @ 0.70 | UP | −0.08pp | ✅ correct |
| 2026-06 | DOWN @ 0.40 | DOWN | +0.17pp | ✅ correct |
| 2026-08 | DOWN @ 0.45 | DOWN | +0.37pp | ✅ correct |

**3/4 correct (pilot, n=6, not statistically significant).** No hallucinations; all reasons cite actual article content.
Confidences appropriately modest (0.40–0.70). The Feb miss is instructive:
news was celebrating falling rates (backward-looking) while March reversed
sharply — a news-lag problem, not a prompt failure. Counterarguments present
in all 4 readings (v2 prompt working). No ABSTAINs triggered; the ambiguous
cases got low confidence instead, which is acceptable calibration behavior.

Full results: `experiments/exp17_news_regime_results.json`.

---

## 4. Artifact map

| Artifact | Location |
|---|---|
| R2 scripts | `experiments/r2_*.py` |
| R2 comparison | `experiments/r2_comparison.md` (lagged rerun, all 4 series) |
| exp17 | `experiments/exp17_news_regime.py` → `exp17_news_regime_results.json` |
| Archive | `docs/RESULTS_ARCHIVE.md` |
| Audit | `docs/DATA_AUDIT.md` |

---

## Agentic Rebuild: Phases 1–3 (exp20–exp25)

After M2/M3/M4/M5 failed as output-layer wrappers, we rebuilt the agentic track from first principles. See `docs/AGENTIC_ROADMAP.md` for the full plan.

![M4 regime reader vs M5 ensemble router](plots/agent_m4_m5.png)

**Phase 1: Event-augmented simulator** (`src/td_one/data_gen/events.py`, `news_synthesis.py`)
- 5 event types with causal mechanisms, PIT-safe news feed, 84% news-leading
- 30/30 tests pass

**Phase 2: Agent infrastructure** (`src/td_one/agent/sandbox.py`, `tools/`, `react_v2.py`)
- Sandboxed code execution (ported from AI2), 10 statistical tools, tool call tracker
- 66 tests pass

**Phase 3: Reflect and adjust**
- Decision policy with shrinkage + corroboration gating (11 tests)
- exp24 (regime selection): ties best single, no alpha
- exp25 (h=3): news **significantly worse** (+25%, p=0.0013)

**exp21 (feature discovery):** Honest negative. Agent finds sensible features that don't transfer (-21.6%) — news lags the effect.

**exp22 (M2 v2 vs v1):** Mixed. v2 is a data scientist (+6.6% when active, 6/9 wins) but high variance from overconfident intervals.

**exp23 (model selection):** Clean negative. Diagnostic selection = random (p=0.80) — backward can't predict forward.

Full reports: `experiments/exp21_report.md`, `exp22_report.md`, `exp23_report.md`, `exp24_report.md`, `exp25_report.md`, `experiments/PHASE3_SUMMARY.md`.

---

## Agentic Track: exp42–48 (trilogy → proposed method)

Per `docs/AGENTIC_ROADMAP.md`: three parallel tracks, then combine what survived.

- **exp42 (FFORMA meta-learner):** ❌ honest negative — doesn't beat equal weights.
- **exp43 (constrained regime router):** ✅ partial positive — first method to significantly beat equal weights (M4 CRPS 0.1282 vs 0.1295, p=0.018). The constrained structural prior (3 regimes + fixed experts) worked where flexible learning failed.
- **exp44 (news as selector):** ⚖️ partial positive — mechanism proven on synthetic (p=0.0068); real-data application blocked by news sparsity.
- **exp45 (learned regimes):** ❌ honest negative — hand rules beat clustering (p=0.00046).
- **exp46 (dense news):** ❌ definitive negative — dense news doesn't help regime/selection.
- **exp47 (proposed method):** ✅ beats equal weights (0.1451 vs 0.1507, p=6.5e-09), best single, and the exp43 router alone. See `docs/PROPOSED_METHOD.md`.
- **exp48 (reasoning sweep):** ❌ "high" reasoning never helps — 3.5–6× slower, sometimes hurts (1.000→0.333 on BoC discovery). Default to "low" for diagnostics.

![Agentic trilogy on M4](plots/trilogy_m4.png)

![Reasoning effort sweep](plots/reasoning_sweep.png)

---

## Real-Data Benchmarks

**Energy/Oil (WTI):** `td-one/benchmarks/energy_oil/`
- WTI is a random walk. ETS/ARIMA/naive identical to 4 decimals.
- Our agent correctly finds nothing (92% of the time) but false positives cost 4-7%.
- **Lesson:** Method is sound; task is impossible. Calibrate synthetic DGP to 2-3pp effects.

**BoC Rate Decisions:** `td-one/benchmarks/boc_rates/`
- Simple rule: 72.2% vs 69.4% climatology (119 meetings; corrected 2026-10-19 — old 83.7%/79.1% was inflated by 24 stale post-2023 rows)
- Our tools diagnosed the missing signal: inflation_gap × rate_momentum interaction
- **Agent discovery test: SUCCESS.** Agent found the interaction autonomously with no hints.
- **Methodology validated:** Works when signal exists (BoC), correctly reports nothing when it doesn't (WTI).

**Freddie Mac SMM (63 months, 2021–2026):** calibrated TSFMs beat m0 by 30% (p=0.0012) — see exp18. News-regime augmentation (exp19) frozen: news doesn't move SMM.

![Freddie Mac SMM 2021–2026](plots/smm_2021_2026.png)

---

## News Audits

**Date-level:** `td-one/audits/NEWS_LEAKAGE_REPORT.md` — 33/34 pass, 0 leakage. Triple-layer RAG defense verified.

**Content-level:** `td-one/audits/NEWS_CONTENT_REPORT.md` — CLEAN. Forward-looking language is about inputs (rates, prices), never about targets (switching rates). Vector bootcamp has date-only guard; ours is strictly stronger.
