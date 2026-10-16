# TD-One Scale-Up Runbook

End-to-end pipeline: synthetic population → point-in-time store → segments →
backtest → experiments. Run at the **current pilot sample size first** (cheap,
fast), then scale to the full 10k×66 population.

**Status legend:** ✅ implemented & tested · ⏳ in progress (build agents) ·
❌ not started. Only the ✅ steps can be run as written; the rest name the
expected entry point so the docs stay accurate when the code lands.

## 0. Prerequisites

- Python 3.12+, `uv` package manager
- Repo checked out; work from `td-one/`
- TSFM weights (✅): `scripts/bootstrap_models.py` → `models/weights/`
  (gitignored; pinned revisions in `models/MANIFEST.json`). See
  `experiments/REPRODUCIBILITY.md` §2.
- Groq LLM (for M2/M3 rungs only): real-LLM calls go through the secure
  `custom.groq` connector — keys never appear in code, env files, or logs.
  Groq free tier is rate-limited (429s); see Troubleshooting.

## 1. ✅ Simulate the population

`src/td_one/data_gen/simulator.py` — `MortgageSimulator(n_customers=10000,
n_months=66, seed=...)`, then `generate_history()` + `generate_outcomes(history)`.

```python
from td_one.data_gen.simulator import MortgageSimulator
sim = MortgageSimulator(n_customers=10000, n_months=66, seed=42)
history = sim.generate_history()        # ~660k records
outcomes = sim.generate_outcomes(history)
```

**Runtime (measured on the 2-vCPU VM):** ~5.4s per 1k×66 → **~1 min for the
full 10k×66**; outcomes are near-instant. The simulator is the *fast* part of
the pipeline; the TSFMs are the bottleneck (see §5).

## 2. ✅ Ingest into the PIT store

`src/td_one/data_service/pit_store.py` — `PITStore(db_path)`, then
`ingest_history(records)` / `ingest_outcomes(records)`. Queries enforce
`record_asof <= run_date` and `outcome_observed_at <= run_date`
(guards are regression-tested; the leakage auditor in
`src/td_one/evaluation/leakage.py` verifies 0% leakage).

```python
from td_one.data_service.pit_store import PITStore
store = PITStore("data/pit.duckdb")
store.ingest_history(history)
store.ingest_outcomes(outcomes)
```

## 3. ✅ Segment discovery

`td_one.segments.discovery.discover_segments` partitions renewals into
interpretable segments (`payment_shock_q1..5`, `rate_type_fixed/variable`,
`region_<province>`, `term_short/medium/long`) and builds monthly
`[month, n_renewals, n_switches]` panels. It accepts `observed_asof` for
per-origin PIT masking: training months only count outcomes with
`outcome_observed_at <= origin` (leakage guard); future months keep actuals
as scoring targets. The agent tools (`data_segment`, `evidence`,
`forecasting`) are implemented in `td_one/tools/`; `data_segment_tool`
reads panels from `data/segments/`.

## 4. ✅ Backtest runner

`td_one.backtest.runner.run_backtest(panels, forecast_fn, origins)` runs the
rolling-origin backtest with a defensive cutoff check, scoring CRPS
(counts), MAE, and 80% interval coverage per segment × origin. The end-to-end
entry point is `scripts/scale_up.py`:

```bash
cd td-one && uv run python scripts/scale_up.py --n-customers 10000 --n-months 66 --models m0
# smoke: --n-customers 500 --n-months 24 --models m0 --skip-ingest
# M0 trailing window: --m0-window 3 | 6 | 12 | auto (default 6;
#   'auto' selects 3/6/12 by trailing-origin CRPS validation per forecast)
# Conformal baseline: --conformal / --no-conformal (default: on).
#   The published M0 is the conformal M0 (fixes undercoverage 0.671->0.914).
# multi-seed: --seed 1 --tag seed1-m0w6
```

## 5. ✅/⏳ Experiments

| Experiment | Status | Entry point | Notes |
|---|---|---|---|
| exp1 — M0 vs M1 (mock TSFMs) | ✅ | `experiments/exp1_mvp1.py` | 500×24, mock mode; pilot-era, kept for regression |
| exp2 — M0 vs M1 (real weights) | ✅ | `experiments/exp2_real_weights.py` | 3 scenarios × 5 seeds; `TD_ONE_MOCK=1` for CI |
| exp3 — M2 ReAct (mock + real Groq) | ✅ | `experiments/exp3_m2.py` | Mock: scripted ReAct traces against real tools (plumbing validation). Real: `TD_ONE_REAL_LLM=1`, budget 20 calls; 2/3 traces finalized, first real-LLM win |
| exp4 — M3 hybrid driver | ✅ | `experiments/exp4_m3.py` | 6 macro scenarios, bounded Δ; real-Groq exercised (`TD_ONE_REAL_LLM=1`, budget 10) — 3/6 beat M0 |
| exp9 — richer DGP (simulator v2) | ✅ | `experiments/exp9_richdgp.py` | 2000×66 on v2 (offers, burnout, seasoning, S-curve); window ranking reverses (m0_w6 wins) |
| exp10 — classical track | ✅ | `experiments/exp10_classical.py` | conformal default, ACI, macro_m0, hierarchical niche |
| exp11 — M4 regime-reader | ✅ | `experiments/exp11_agent_v2.py` | Real Groq, budget 6; qualified positive |
| exp12 — M5 ensemble router | ✅ | `experiments/exp12_router.py` | Real Groq, budget 4; clean negative vs inverse-CRPS |
| exp13 — TSFM+FRED rematch | ✅ | `experiments/exp13_tsfm_macro.py` | Macro-conditioned TSFs, 2 DGPs × 3 seeds |
| exp14 — M4 real-data benchmark | ✅ | `experiments/exp14_m4_real.py` | 2000 M4 Monthly series, real weights; TSFMs beat classical ~20% |
| scale-up — full 10k×66 | ✅ | `scripts/scale_up.py` | Segment backtest; writes `experiments/scale_up_results.json` + `scale_up_report.md` |

**TSFM runtime (measured, 2-vCPU VM, real weights, 45 origins):** M0 ~0.1s;
Chronos-2 ~26s; **TimesFM-2.5 ~972s** (compile-heavy; fast once compiled).
For iteration, run exp2 with `TD_ONE_MOCK=1` (seconds) and reserve real
weights for the final confirmation run.

**Groq (M2/M3):** real-LLM calls go through `td_one/agent/groq_client.py`
(AI2-grade retry: 6 attempts, 60s floor on 429, 2.5s process-wide
throttle; the old controller is gone). Free-tier rate limits apply —
expect 429s; do not hammer. Budget ~1–2s per call plus retry delays.

## 6. Artifact map

| Artifact | Written by | Location |
|---|---|---|
| Weights (gitignored) | `scripts/bootstrap_models.py` | `models/weights/chronos2/`, `models/weights/timesfm25/` |
| Pin manifest | `scripts/bootstrap_models.py` | `models/MANIFEST.json` |
| PIT database | §2 | `data/pit.duckdb` (gitignored) |
| exp results JSON | each experiment script | `experiments/exp<N>_results.json` |
| exp reports | each experiment script / author | `experiments/exp<N>_report.md` |
| Reproducibility guide | — | `experiments/REPRODUCIBILITY.md` |
| Critic audit | — | `docs/CRITIC_SYNTHESIS.md` |

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| HF download fails with `Invalid port` | Sanitize proxy vars: `export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"` (bracketed IPv6 entries break `huggingface_hub`) |
| HF offline / rate-limited bootstrap | `scripts/bootstrap_models.py` resolves SHAs explicitly and fails closed with a not-reproducible status instead of silently using `main`; retry when network returns |
| OOM during TSFM inference | Models already load one-at-a-time with explicit unload; on smaller VMs use `TD_ONE_MOCK=1`. For full scale-ups, run one model per invocation with `--tag` (e.g. `--models chronos2 --tag real-chronos2`) so only one weight set is ever resident |
| `forecast quantiles must be non-decreasing` | Real TSFMs can emit quantile crossing; `tools/forecasting._real_forecast` sorts values ascending to enforce monotonicity (regression-tested) |
| `models/weights/` missing | Re-run the bootstrap; the experiment scripts fail fast if `MANIFEST.json` is absent |
| Groq 429 (rate limit) | `td_one/agent/groq_client.py` retries with AI2's patient free-tier policy (6 attempts, 60s floor on 429, 2.5s process-wide throttle). Do not hammer — the free tier resets on a window |
| `timesfm` import error | `uv run --with timesfm` and `timesfm>=3.0` (2.x `TimesFm` class no longer exists) |
| Long TimesFM runs | Expected (~16 min / 45 origins); use mock mode for iteration, real weights only for final runs |

## 8. Verified timings (this VM, 2 vCPU)

- Simulator 10k×66: ~1 min (measured at 1k×66: 5.4s)
- exp2 v2 real weights: ~17 min total (TimesFM dominates)
- Real-weight scale-up (10k×66, 14 seg × 5 origins): Chronos-2 ~144s;
  TimesFM 764.7s (~13 min, measured; run per-model with `--tag` on
  RAM-constrained VMs)
- exp2 mock: seconds
- Test suite: 316 collected (measured 2026-10-16; 3 skipped = gated live-weight/network tests)
