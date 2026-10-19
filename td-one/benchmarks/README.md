# `benchmarks/` — real-data benchmarks

Each benchmark is self-contained: survey, experiment scripts, results, report.

| Benchmark | What it is |
|---|---|
| [`boc_rates/`](boc_rates/README.md) | Bank of Canada rate decisions — methodology validation (agent discovered the inflation interaction autonomously) |
| [`energy_oil/`](energy_oil/README.md) | WTI crude — random walk; method sound, task impossible |
| [`food_cpi/`](food_cpi/README.md) | Food CPI forecasting — classical discriminates seasonality; LLM (exp53) beats proposed 45%; ladder exp55–59: LLM is a smooth-macro specialist |
| [`m4/`](m4/README.md) | M4 Monthly — the agentic track's real-data home |
| [`smm/`](smm/README.md) | Freddie Mac SMM — TimesFM_cal beats best classical 12.2% (p=0.076 n.s.); agent features +1.21% (n.s.); real events hurt (−13%, priced in) |

Synthesis: [`docs/reports/FINAL_SYNTHESIS.md`](../docs/reports/FINAL_SYNTHESIS.md).
Up: [`td-one/`](../README.md).
