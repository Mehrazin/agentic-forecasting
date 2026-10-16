# TD-One Experiments

End-to-end results for the mortgage attrition forecasting ladder. Every
experiment is PIT-clean (no lookahead) and reproducible — see
`REPRODUCIBILITY.md` for the full reproduction steps.

## The ladder

| Rung | Model | What it is |
|---|---|---|
| M0 | Historical baseline | Trailing-6 Jeffreys Beta-Binomial on segment panels |
| M1 | Zero-shot TSFMs | Chronos-2 / TimesFM-2.5 on panel switching-rate series |
| M2 | ReAct LLM agent | Groq-driven tool loop (data_segment → evidence → forecasting) |
| M3 | Hybrid | M0 + bounded log-odds adjustment Δ ∈ [-0.75, +0.75] proposed by the agent |

Primary metric everywhere is **CRPS** (proper score on count distributions);
secondary: MAE and 80% interval coverage.

## Experiment index

| Exp | Comparison | Scale | Mode | Report |
|---|---|---|---|---|
| exp1 | M0 vs M1 (mocks) | 500×24 synthetic | mock | `exp1_report.md` |
| exp2 v2 | M0 vs M1 (real weights) | 3 scenarios × 5 seeds | real weights | `exp2_report.md` |
| exp3 | M2 vs M0 | 3 pilot segments | mock + real Groq | `exp3_report.md` |
| exp4 | M3 vs M0 | 6 macro scenarios | mock + real Groq | `exp4_report.md` |
| scale-up | M0 vs `*_mock` | 10k×66, 15 seg × 6 origins | mock (plumbing) | `scale_up_report.md` |
| scale-up | M0 vs real Chronos-2 | 10k×66, 15 seg × 5–6 origins | real weights | `scale_up_report_real-chronos2.md` |
| scale-up | M0 vs real TimesFM-2.5 | 10k×66, 15 seg × 6 origins | real weights | _in progress_ |

## Key findings (read this first)

1. **exp2 v1's "Chronos-2 beats M0 by 20%" is retracted.** The critic
   audit found a circular DGP hand-tuned to flatter trend-extrapolating
   TSFMs, n=6 test points on one seed, and an off-spec M0. The redesigned
   v2 (3 pre-registered scenarios, 5 seeds, CRPS primary, oracle anchor)
   finds **no model significantly beats M0** (all paired p > 0.18).
   TimesFM-2.5 has the best mean CRPS in all three scenarios but not
   significantly. Chronos-2 overcovers; M0 undercovers; TimesFM is
   closest to nominal.

2. **Mock `*_mock` scale-up wins over M0 are a sharpness artifact, not
   evidence.** The mocks center on the last observed rate with tight
   spreads, so they score better on CRPS in stable-rate environments.
   The real M0-vs-M1 comparison is exp2 v2.

3. **Real Chronos-2 at scale overcovers** (coverage 1.0/0.8 across
   segments), matching exp2's finding. CRPS is mixed vs M0 segment by
   segment. Real TimesFM-2.5 scale-up is running.

4. **Real-Groq M2 (exp3): first real-LLM win.** With the patient retry
   policy, the agent finalized 2/3 traces; seg-1 M2 CRPS 0.82 vs M0
   1.33. seg-0 finalized worse than M0; seg-2 never finalized. Mock
   mode remains the plumbing validation (0/3 pilot segments beat M0 —
   expected, the scripted LLM adds no skill).

5. **Real-Groq M3 (exp4): 3/6 scenarios beat M0.** The LLM's deltas are
   directionally right (e.g. −0.25 on the rate-drop scenario) but it
   over-adjusts — it added +0.30 even on the benign scenario where the
   right answer is ~0.

6. **Calibration is the open work.** Coverage sits below the 80%
   nominal for M0 and the mocks at scale; Chronos-2 overcovers.

## Reproducing

```bash
cd td-one
uv run python experiments/exp1_mvp1.py          # mock M0 vs M1
uv run python experiments/exp2_real_weights.py  # real weights (needs models/weights/)
TD_ONE_REAL_LLM=1 uv run python experiments/exp3_m2.py   # real-Groq ReAct (≤20 calls)
TD_ONE_REAL_LLM=1 uv run python experiments/exp4_m3.py   # real-Groq M3 (≤10 calls)
uv run python scripts/scale_up.py --n-customers 10000 --n-months 66 --models m0
# real-weight scale-ups, one model per invocation (RAM):
uv run --with torch --with chronos-forecasting \
  python scripts/scale_up.py --models chronos2 --tag real-chronos2
uv run --with torch --with timesfm \
  python scripts/scale_up.py --models timesfm25 --tag real-timesfm25
```

Real-LLM modes go through the vault-surrogate Groq client
(`src/td_one/agent/groq_client.py`): key never in process, per-run call
budgets, 429 backoff with 60s floor, 2.5s process-wide throttle.

## Artifacts

- `exp{1,2,3,4}_report.md` + `exp{2,3,4}_results.json` — per-experiment reports and raw results
- `scale_up_report*.md` + `scale_up_results*.json` — scale-up reports (mock, real-chronos2, real-timesfm25)
- `evidence_corpus/corpus.json` — frozen 12-entry illustrative evidence corpus (PIT-filtered at `run_date`)
- `REPRODUCIBILITY.md` — weights bootstrap, exact commands, pinned revisions
- `../docs/CRITIC_SYNTHESIS.md` — the 2026-10-15 six-angle audit that retracted exp2 v1
- `../docs/SCALE_UP_RUNBOOK.md` — pipeline runbook (prereqs, timings, troubleshooting)
