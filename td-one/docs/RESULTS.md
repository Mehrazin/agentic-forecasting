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
| R2-1 | exp10 (classical) | `experiments/r2_exp10_classical.py` | ⏳ running | PIT-safe macro vs M0 |
| R2-2 | exp13 (TSFM macro) | `experiments/r2_exp13_tsfm_macro.py` | ⏳ running | PIT-safe TSFM+FRED |
| R2-3 | exp15 (corrected macro) | `experiments/r2_exp15_macro_corrected.py` | ⏳ running | PIT-safe corrected sign |
| R2-4 | exp14 (M4 real) | — | 📝 planned | Soundness-fix rerun |

---

## R2-1: Classical with PIT-safe macro

⏳ Running. Will compare against archived exp10 (§2g in `RESULTS_ARCHIVE.md`).

**Key question:** Did the z-score leak inflate the macro benefit in exp10?

---

## R2-2: TSFM + FRED (PIT-safe)

⏳ Running. Will compare against archived exp13 (§2j in `RESULTS_ARCHIVE.md`).

**Key question:** Does macro-conditioned TSFM still help without the leak?

---

## R2-3: Corrected-sign macro (PIT-safe)

⏳ Running. Will compare against archived exp15 (§2m in `RESULTS_ARCHIVE.md`).

**Key question:** exp15 already killed the macro claim; does PIT-safety change anything?

---

## What carries forward from exp1–exp15 (not rerun)

These results were **not** affected by the P0 leaks:

- **M4 regime reader** (exp11): 3.59 vs M0 4.82 — real Groq, no FRED dependency
- **M5 router** (exp12): clean negative — no FRED dependency
- **M4 Monthly benchmark** (exp14): TSFMs beat classical ~20% — no FRED z-scores
- **Simulator v2** (exp9): DGP work — unaffected

---

## 4. Artifact map

| Artifact | Location |
|---|---|
| R2 scripts | `experiments/r2_*.py` |
| R2 comparison | `experiments/r2_comparison.md` (when complete) |
| Archive | `docs/RESULTS_ARCHIVE.md` |
| Audit | `docs/DATA_AUDIT.md` |
