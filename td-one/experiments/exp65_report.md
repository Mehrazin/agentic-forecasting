# exp65: SMM with LLM Expert

**Date:** 2026-10-19
**Method:** Proposed v2 (router + gated LLM expert, Gemini Flash via Vector proxy)
**Dataset:** Freddie Mac SMM, 63 months, 50 origins

## Results (CRPS, lower is better)

| Method | Mean CRPS |
|---|---|
| v2 (router + LLM) | 0.000508 |
| v1 (classical only) | 0.000512 |
| LLM alone | 0.000492 |

**Verdict: LLM NEUTRAL**

- v2 vs v1: -0.000004 (p=0.6931, win 0.50)
- v2 vs LLM: +0.000016 (p=0.4738, win 0.54)

### Classical experts (for context)

| Expert | Mean CRPS |
|---|---|
| naive | 0.000510 |
| snaive | 0.001312 |
| ets | 0.000639 |
| arima | 0.000510 |

## Notes

- LLM prompts: V1-style (history stats in words), SMM in percent units.
- 50/50 LLM origins succeeded.
- Gate: dataset="smm" is macro-like → LLM expert enabled (llm_gate).
