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
| R2-1 | exp10 (classical) | `experiments/r2_exp10_classical.py` | ✅ done (lagged rerun) | **Lagged macro helps: -0.357 (53% of leaky gain)** |
| R2-2 | exp13 (TSFM macro) | `experiments/r2_exp13_tsfm_macro.py` | ✅ done (lagged rerun) | **Residual conditioning hurts C2 4/6 — method-specific** |
| R2-3 | exp15 (corrected macro) | `experiments/r2_exp15_macro_corrected.py` | ✅ done | PIT-safe = harmless on v2 DGP |
| R2-4 | exp14 (M4 real) | — | 📝 planned | Soundness-fix rerun |
| exp17 | — (new) | `experiments/exp17_news_regime.py` | ✅ done (pilot) | **News+RAG+regime reader end-to-end: 3/4 correct (n=6, n.s.)** |
| exp18 | — (new) | `experiments/exp18_smm_full.py` | ✅ done | **Calibrated TSFMs beat m0 30% on 63-mo SMM (p=0.0012)** |

---

## R2-1: Classical with lagged PIT-safe macro — **exploratory positive** (single seed)

| Model | R1 (leaky) | R2-old (buggy) | R2-new (lagged) |
|---|---|---:|---:|
| m0_w3 | — | — | **1.457** |
| m0_w6 | 2.305 | 2.305 | 2.28 |
| macro_m0 | 1.627 | 2.305 | **1.923** |

**RETRACTION (2026-10-17):** The original R2 conclusion ("leak was 100% of benefit")
was based on a bug — `fred_z.get(forecast_month)` looked up t+1 in a dict truncated
at t, always returning 0.0. With the correct lagged implementation (predictive
regression on z[t]), macro_m0 achieves **-0.357 CRPS vs m0_w6** (~53% of leaky gain).

**Caveats (statistical):** Single seed (seed=0), no uncertainty quantification.
Against the best M0 window (m0_w3=1.457), macro_m0 (1.923) is **worse by 32%**.
Window tuning matters more than the macro overlay. This is an exploratory finding,
not a confirmatory result.

## R2-2: TSFM + lagged FRED — **exploratory negative** (n=6, no significance testing)

Lagged residual conditioning: 1/6 helps (-1.2%), 4/6 hurt (+3% to +23%), 1/6 ties.
The residual approach — subtracting β·z from training inputs then re-adding at
forecast time — distorts the input distribution without adequate compensation.

**Caveats (statistical):** n=6 with no p-values or confidence intervals — anecdotal,
not established. A sign test gives p=0.11 (not significant). The pattern is
suggestive (losses dwarf the single win) but needs larger evaluation. Also note
`conf_m0` beats both Chronos2 variants in every seed/DGP — the relevant baseline
isn't "plain Chronos2" but "best available model."

## R2-3: Corrected macro — PIT safety neutralizes the damage

R1: estimated overlay hurt by 0.151. R2: m0=0.6092, oracle=0.6199 (-0.0107),
est=0.692 (-0.0828). Even the oracle hurts on v2 DGP — DGP-specific, not harmless.

## Bottom line (updated 2026-10-17)

**The macro track is method-specific and statistically preliminary.** Lagged PIT-safe
macro shows exploratory promise for classical models (-0.357 vs m0_w6, but loses to
m0_w3; single seed) while residual conditioning shows exploratory negative signals
for TSFMs (n=6, no significance testing). The original "macro is dead" verdict was
based on a dict-lookup bug. The honest conclusion: **we don't know yet** — the
findings are suggestive but need multi-seed confirmatory evaluation.

**exp18:** Calibrated TimesFM beats m0 by 30% on 63 months of real Freddie Mac SMM
(p=0.0012, survives Bonferroni) — the first statistically robust TSFM win. Chronos2
(p=0.011) is nominally significant but doesn't survive multiple-comparison correction.

**Baseline caveat:** exp18 compares against trailing-window means (m0_w3/6/12) only.
No ETS, ARIMA, or Theta baseline was run — "beats trailing mean" is not "beats
classical." A competent ETS would likely erase much of the 30% gap. This is the
strongest TSFM evidence in the project, but the baseline bar was low.

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
| R2 comparison | `experiments/r2_comparison.md` (when complete) |
| exp17 | `experiments/exp17_news_regime.py` → `exp17_news_regime_results.json` |
| Archive | `docs/RESULTS_ARCHIVE.md` |
| Audit | `docs/DATA_AUDIT.md` |
