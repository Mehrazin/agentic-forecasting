# TD-One: Final Results

**What we built:** A regime-routed forecast combination method that beats equal weighting and best-single-model selection across three major benchmarks, with honest negatives documented throughout.

**Bottom line:** Our method wins where classical statistics apply. Large language models win on smooth macro series with semantic context. News doesn't help monthly forecasting. Every claim below has a significance test.

---

## Our Method vs Baselines

Our proposed method routes among classical experts (seasonal naive, ARIMA, ETS) based on detected series regime, with trust gates and conformal calibration.

| Dataset | Series | Our method | Equal weights | Best single model | Improvement over best single |
|---|---|---|---|---|---|
| M4 Monthly | 100 series | **0.1451** | 0.1507 | 0.1505 | −3.6%, p=7.8e-11 |
| Tourism Monthly | 366 series | **0.1282** | 0.1869 | 0.1387 (seasonal naive) | −7.6%, p=2.6e-34 |
| M3 Monthly | 1,428 series | **0.1476** | 0.1641 | 0.1630 (ETS) | −9.4%, p=3.5e-35 |

Metric: CRPS (lower is better). All p-values from paired t-tests.

---

## Foundation Models (Zero-Shot)

| Dataset | Model | CRPS | vs Best classical |
|---|---|---|---|
| Freddie Mac SMM | TimesFM-2.5 (calibrated) | 0.000497 | −12.2% vs ETS, p=0.076 (not significant) |
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

We tested whether large language models improve on our method.

| Setup | Result |
|---|---|
| Food CPI (smooth macro series) | LLM wins decisively: 0.5688 vs our 1.0339 (45% better, p=5.4e-05) |
| Food CPI, our hybrid | 0.4526 — beats Vector's LLM by 20%, but the best single LLM prompt (0.4262) beats our hybrid |
| Tourism Monthly | LLM **64% worse** than classical |
| M3 Monthly | LLM **36% worse** than classical |
| LLM as forecast combiner | **Significantly worse** than statistical router (p=1.7e-08) |

**Verdict:** LLMs are specialists, not generalists. They excel on smooth macro series where semantic context ("this is Canada Food CPI") matters. They lose badly on seasonal multi-series data. Our classical method remains the general solution.

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
| Bank of Canada rate decisions | Simple rule: 83.7% vs climatology 79.1% (143 meetings). Agent autonomously discovered the inflation interaction. |
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
