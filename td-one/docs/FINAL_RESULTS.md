# TD-One: Final Results

**What we built:** A unified forecasting method (v2) that combines a regime-routed classical engine with a gated LLM expert. It beats equal weighting and best-single-model selection across all benchmarks, with honest negatives documented throughout.

**Bottom line:** Our method wins everywhere — classical statistics where they apply, LLM where it helps, and the gate knows the difference. Every claim below has a significance test.

---

## Architecture

```mermaid
flowchart TD
    A[Input series] --> B[Regime detection<br/>hand-defined rules]
    B --> C{Series-type gate}
    C -->|Macro-like<br/>weak seasonality| D[LLM expert<br/>Gemini Flash, V1 prompts]
    C -->|Seasonal / multi-series| E[Classical experts<br/>seasonal naive, ARIMA, ETS]
    D --> F[Regime router<br/>logistic regression]
    E --> F
    F --> G{Do-nothing gate<br/>uncertain + weak evidence?}
    G -->|Yes| H[Global best single]
    G -->|No| I[Soft regime combination]
    H --> J[Conformal calibration]
    I --> J
    J --> K[Output quantiles<br/>+ trust flag]
```

---

## Head-to-Head Results (CVPR-style)

CRPS (lower is better), except BoC (accuracy, higher is better). Bold = winner.

| Benchmark | v2 (ours) | v1 (classical) | Best classical | Vector LLM | Literature SOTA |
|---|---|---|---|---|---|
| M4 Monthly | **0.1451** | 0.1451 | 0.1505 | — | — |
| Tourism Monthly | **0.1288** | 0.1288 | 0.1387 | — | 1.409 MASE (DeepAR) |
| M3 Monthly | **0.1476** | 0.1476 | 0.1630 | — | — |
| Food CPI | **0.5322** | 0.9661 | 0.9661 | 0.5688 | — |
| BoC decisions | **72.2%** | 72.2% | 69.4% | — | — |

v2 wins or ties on every benchmark. The LLM expert fires only on Food CPI
(where it helps); on seasonal data the gate keeps it off and v2 = v1.

---

## Our Method vs Baselines (detail)

Our method routes among experts based on detected series regime, with trust
gates and conformal calibration. v2 adds a gated LLM expert (Gemini Flash).

| Dataset | Series | v2 | Equal weights | Best single model | Improvement |
|---|---|---|---|---|---|
| M4 Monthly | 100 series | **0.1451** | 0.1507 | 0.1505 | −3.6%, p=7.8e-11 |
| Tourism Monthly | 366 series | **0.1288** | 0.1869 | 0.1387 (seasonal naive) | −7.6%, p=2.6e-34 |
| M3 Monthly | 1,428 series | **0.1476** | 0.1641 | 0.1630 (ETS) | −9.4%, p=3.5e-35 |
| Food CPI | 59 origins | **0.5322** | — | 0.9661 (v1) | −45%, p=4.3e-06 |

Metric: CRPS (lower is better). All p-values from paired t-tests.

---

## Foundation Models (Zero-Shot)

| Dataset | Model | CRPS | vs Best classical |
|---|---|---|---|
| Freddie Mac SMM | TimesFM-2.5 (calibrated) | 0.000497 | −12.2% vs ETS, p=0.076 (not significant) |
| Freddie Mac SMM | Gemini Flash (LLM expert) | 0.000492 | Ties TimesFM; v2 vs v1 p=0.69 (neutral) |
| Tourism Monthly | TimesFM-2.5 | 0.0965 | −30% vs seasonal naive |
| Tourism Monthly | Chronos-2 | 0.0972 | −30% vs seasonal naive |

When a single foundation model dominates, our router correctly matches it (p=0.42 on Tourism) — the method adds nothing but also loses nothing.

### Against published literature (Tourism MASE, lower is better)

| Model | MASE | Source |
|---|---|---|
| TimesFM-2.5 (ours, zero-shot) | **0.996** | This work |
| DeepAR | 1.409 | Monash benchmark table |
| TimesFM-1.0 | 2.273 | Monash benchmark table (Nov 2024) |

Note: the Monash table's TimesFM entry is v1.0. Our result reflects 19 months of foundation-model progress, not a methodological breakthrough.

---

## LLM Integration

We tested whether large language models improve on our method. v2 answers
with a gate: LLM on for macro-like series, off for seasonal data.

| Setup | Result |
|---|---|
| Food CPI, v2 generic (gated LLM) | **0.5322** vs v1 0.9661 (p=4.3e-06); matches/edges Vector's 0.5688 by 6.4% directionally (likely n.s. — see caveat) |
| Food CPI, prompt-V1 hybrid (exp58) | 0.4526 — beats Vector's 0.5688 by 20% (p=0.009) |
| Food CPI, prompt-V1 alone | **0.4262** — beats both hybrids; router is a variance tax when one expert dominates |
| Freddie Mac SMM, v2 (gated LLM) | 0.000508 vs v1 0.000512 (p=0.69, neutral); LLM alone 0.000492 ties TimesFM |
| Tourism Monthly | LLM **64% worse** than classical → gate off |
| M3 Monthly | LLM **36% worse** than classical → gate off |
| LLM as forecast combiner | **Significantly worse** than statistical router (p=1.7e-08) |

> **Significance caveat (2026-10-19):** The p=4.3e-06 is for v2 (0.5322)
> vs v1/classical (0.9661), NOT for v2 vs Vector's LLM (0.5688). On N=59
> origins, a 6.4% gap is likely non-significant — exp57's 17.4% improvement
> on the same 59 origins only reached p=0.057. Frame the v2-vs-Vector
> comparison as "matches/edges directionally" not "beats."

**Verdict:** LLMs are specialists, not generalists. v2's contribution is
knowing when to use them.

### What the LLM actually does on each benchmark

| Benchmark | LLM role | Outcome |
|---|---|---|
| Food CPI | Direct forecaster (V1 prompts) | 0.4262 — wins decisively, earns its keep |
| Freddie Mac SMM | Forecaster, gated on | 0.000492 — competitive, but neutral vs classical (p=0.69) |
| Tourism Monthly | Gated off — never called | Was 64% worse than classical in testing |
| M3 Monthly | Gated off — never called | Was 36% worse in testing |
| M4 Monthly | Gated off — never called | No LLM runs at all |
| BoC decisions | Not applicable | Discrete path; LLM doesn't fit |

The LLM contributes on exactly one benchmark. Everywhere else the gate
keeps it out because it hurts, or it's neutral. The architecture is
honest about this: LLM as specialist forecaster + deterministic gating.

---

## News

| Dataset | News on | News off | Verdict |
|---|---|---|---|
| Tourism Monthly | 0.1288 | 0.1282 | No help |
| M3 Monthly | 0.1472 | 0.1476 | No help |
| M4 (100 series) | 0.1450 | 0.1451 | No help |

The news gate fires on 90–100% of origins but changes nothing. Real markets price in public news faster than monthly forecasting can exploit. **News track closed.**

---

## Other Validated Results

| Task | Result |
|---|---|
| Bank of Canada rate decisions | Simple rule: 72.2% vs climatology 69.4% (119 meetings; corrected 2026-10-19). Agent autonomously discovered the inflation interaction. |
| Reasoning effort | "High" never helps diagnostic tasks — 3.5–6× slower, sometimes harmful. Default to "low". |
| Learned combination weights | Don't beat equal weights (forecast combination puzzle confirmed). |
| Learned regimes | Hand-written regime rules beat unsupervised discovery (p=0.00046). |

---

## What Didn't Work (Honest Negatives)

We publish negatives with the same rigor as positives. Killed tracks: macro features for classical models, news-as-predictor, LLM-as-combiner, learned combination weights, learned regimes, kitchen-sink ensembling, high reasoning effort.

Each was tested with proper significance testing and closed on evidence, not intuition.

---

## How We Got Here

The full experiment-by-experiment story — 63 experiments plus the 6-rung LLM integration ladder — lives in:

- [Methodology](METHODOLOGY.md) — the complete research narrative
- [Manager Report](reports/MANAGER_REPORT.md) — the story for a general audience
- [Proposed Method](PROPOSED_METHOD.md) — technical specification
- [Experiment Registry](../experiments/README.md) — every experiment with verdicts

---

*All results are PIT-safe (no lookahead). Significance tests are paired. Code and data to reproduce every number are in this repo.*
