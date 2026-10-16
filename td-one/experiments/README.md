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
| exp6 | M3+shrinkage/gate vs M3/M0 | 12 scenarios (6 tune + 6 held-out) | mock (mechanism test) | `exp6_results.json` |
| scale-up | M0 vs `*_mock` | 10k×66, 14 seg × 5 origins | mock (plumbing) | `scale_up_report.md` |
| scale-up | M0 vs real Chronos-2 | 10k×66, 14 seg × 5 origins | real weights | `scale_up_report_real-chronos2.md` |
| scale-up | M0 vs real TimesFM-2.5 | 10k×66, 14 seg × 5 origins | real weights | `scale_up_report_real-timesfm25.md` |

Scale note: 15 segments are designed; `term_short` never scores (no
observable training data at these origins), so all runs report **14
segments**. Six origins are configured; one per segment is skipped for
the same reason, leaving **5 scored origins per segment**.

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

3. **Real Chronos-2 at scale overcovers** (coverage values {0.6, 0.8,
   1.0} across segments, mean 0.914), matching exp2's finding. Paired
   tests on the 14 shared segments (same seed, same DGP, same origins —
   pairing valid): Chronos-2 wins 7/14 (paired-t p=0.212, Wilcoxon
   p=0.583); TimesFM-2.5 wins 5/14 (p=0.341 / 0.583) — **not
   significant**, reproducing exp2 v2's conclusion at scale. The mean
   CRPS gaps (4.463 / 4.591 vs M0 5.360) are driven by a few segments;
   the median picture is a coin flip. TimesFM mean coverage is 0.80
   (nominal). ⚠️ Single seed, single DGP — any future superiority claim
   must be pre-registered across seeds/DGP variants.

4. **Real-Groq M2 (exp3): first real-LLM win — with a caveat.** With
   the patient retry policy (the third real-mode attempt; attempts 1–2
   were rate-limit-bound), the agent finalized 2/3 traces; seg-1 M2 CRPS
   0.82 vs M0 1.33. seg-0 finalized worse than M0; seg-2 never
   finalized. **Honesty note:** real-mode M2 still uses the
   `timesfm_mock` numerical backend
   (`exp3_results.json: "forecast_model": "timesfm_mock"`), so the
   seg-1 edge cannot be separated from the mock sharpness artifact
   (finding #2). What the run demonstrates is orchestration — the agent
   driving the tool loop to finalization — not numerical superiority.
   Mock mode remains the plumbing validation (0/3 pilot segments beat
   M0 — expected, the scripted LLM adds no skill).

5. **Real-Groq M3 (exp4): 3/6 scenarios beat M0 (exploratory — n=6
   hand-designed scenarios, no significance claimed).** The LLM's
   deltas are directionally right 4/6 (e.g. −0.25 on the rate-drop
   scenario) but: it got the **sign wrong** on `osfi_tightening` (+0.30
   vs true ≈ −0.35, misread a stress-test policy as pro-switching), it
   over-adjusts (+0.30 on the benign scenario where the right answer is
   ~0), and it systematically undershoots magnitude (0.18–0.30 vs true
   0.45–0.70). Direction doesn't predict the score: `rate_shock_down`
   was a CRPS loss despite correct direction.

6. **M3 over-adjustment fixes (exp6, mock mechanism test):** plain M3 is
   worse than M0 (held-out mean CRPS 9.66 vs 7.86) — over-adjustment
   hurts. A **regime gate** (two-proportion z-test on the panel; delta
   forced to 0 unless |z| ≥ 1.25) restores M3 to 7.60, beating M0 by
   killing catastrophic spurious adjustments on benign scenarios
   (gate confusion on held-out: 3 TP / 0 FP / 3 TN / 0 FN).
   **Shrinkage** (k*=0.5, tuned) helps vs plain M3 (8.62) but not vs M0;
   gate+shrinkage (7.73) ≈ gate alone. The gate is a risk-management
   device, not a win-rate device. Small-n (6+6 scenarios); mock deltas
   replicate the real-Groq error model.

7. **Calibration is the open work.** Coverage sits below the 80%
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
- `exp6_m3plus.py` + `exp6_results.json` — M3 over-adjustment fixes (shrinkage + regime gate), 6 tune + 6 held-out scenarios
- `scale_up_report*.md` + `scale_up_results*.json` — scale-up reports (mock, real-chronos2, real-timesfm25)
- `evidence_corpus/corpus.json` — frozen 12-entry illustrative evidence corpus (PIT-filtered at `run_date`)
- `REPRODUCIBILITY.md` — weights bootstrap, exact commands, pinned revisions
- `../docs/CRITIC_SYNTHESIS.md` — the 2026-10-15 six-angle audit that retracted exp2 v1
- `../docs/SCALE_UP_RUNBOOK.md` — pipeline runbook (prereqs, timings, troubleshooting)
