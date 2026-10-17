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
| R2-1 | exp10 (classical) | `experiments/r2_exp10_classical.py` | ✅ done | **Leak was 100% of macro benefit** |
| R2-2 | exp13 (TSFM macro) | `experiments/r2_exp13_tsfm_macro.py` | ✅ done | **Benefit reversed to penalty** |
| R2-3 | exp15 (corrected macro) | `experiments/r2_exp15_macro_corrected.py` | ✅ done | PIT-safe = harmless |
| R2-4 | exp14 (M4 real) | — | 📝 planned | Soundness-fix rerun |
| exp17 | — (new) | `experiments/exp17_news_regime.py` | ✅ done | **News+RAG+regime reader end-to-end: 3/4 correct** |

---

## R2-1: Classical with PIT-safe macro — the leak WAS the benefit

| Model | R1 (leaky) | R2 (PIT-safe) | Delta |
|---|---|---:|---:|
| m0_w6 | 2.305 | 2.305 | 0.000 |
| macro_m0 | 1.627 | 2.305 | **+0.678** |

The entire 0.678 CRPS "macro benefit" was the model seeing `z[forecast_month]`
from the future. PIT-safe: forecast-month z is unknowable → zero shift → pure M0.

## R2-2: TSFM + FRED — benefit reversed

Chronos2 macro-conditioning beat plain on all 3 seeds (R1). PIT-safe: **worse**
on all 3 seeds (+0.257 to +0.762 CRPS). The residual approach needs `z[T+1]`,
which is unknowable — training gets distorted without compensation.

## R2-3: Corrected macro — PIT safety neutralizes the damage

R1: estimated overlay hurt by 0.151. R2: exactly equals M0 (harmless).
The damage was leak-amplified wrong-sign beta; PIT safety kills it.

## Bottom line

**None of the published macro results survive PIT-safe reruns.** The FRED overlay
as designed (contemporaneous `z[forecast_month]`) is not a realizable forecasting
strategy. A PIT-safe macro model needs to forecast the macro itself or use lagged z.
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

**3/4 correct.** No hallucinations; all reasons cite actual article content.
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
