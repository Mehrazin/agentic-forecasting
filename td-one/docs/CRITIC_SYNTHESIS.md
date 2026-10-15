# Critic Synthesis — TD-One Audit 2026-10-15

Six critics audited the project. This is the consolidated, prioritized
improvement list. All items are being implemented; the pilot data size
stays fixed per user direction.

## P0 — Bugs / false claims (fixing now)

### Models (`tsfm.py`)
1. **Chronos real path unreachable**: `if use_mock or _pipeline is None` silently
   returns mock in real mode. Fix: branch on `use_mock` only, then `_load()`.
2. **Chronos returns 21 fixed quantiles, not `n_samples` draws**: the real
   `predict()` has no `num_samples` param; output is a quantile lattice
   (levels 0.01..0.99). Fix: real path becomes `forecast_quantiles()` exposing
   true levels; mock/real divergence documented.
3. **TimesFM quantile mislabeling**: interpolation assumed source levels span
   [0,1]; actual levels are [0.1..0.9]. Fix: interpolate from true levels,
   raise on unexpected count.
4. **`weights_dir` handling**: store in `__init__`, accept in `forecast*`;
   no more private-`_load` calls from experiment code.

### Experiment design (exp2)
5. **Circular DGP**: drift DGP hand-tuned to flatter TSFMs. Fix: 3 pre-registered
   scenarios (stationary / drift / regime-shift), report all.
6. **n=6, one seed**: noise dominates. Fix: K seeds, paired tests, oracle-MAE
   noise-floor anchor.
7. **M0 spec/code mismatch**: docstring says trailing-6-month, code pools 12.
   Fix: implement trailing-6.
8. **MAE-only**: discards probabilistic forecasts. Fix: CRPS primary, 80% PI
   coverage secondary.

### Data
9. **No outcome DGP**: simulator generates history only; "planted shocks" are
   decorative. Fix: `generate_outcomes()` with structural logistic DGP +
   observation lag + behavioral uplift test.
10. **PIT outcomes table missing**: docstring promises guards that don't exist.
    Fix: add table + `get_outcomes(run_date)` with both predicates + leakage
    regression test.
11. **`insured_flag` dropped**: generated, never emitted. Fix: add 9 missing
    schema fields, thread through simulator.

### Safety / correctness
12. **`estimate_adjustment` silent 0.0**: all failures → "no adjustment".
    Fix: log warning, return Optional, distinguish failure from neutral.
13. **Leakage auditor fail-open** on unparseable timestamps. Fix: fail-closed.

### Docs (house-rule hard requirement)
14. README/PLAN_SOLO aspirational claims (tools, backtest runner, 4-table
    schemas, ReAct loop, FRED cache). Fix: status pass (done/not-done).
15. Root `plan.md` ↔ `PLAN_SOLO.md` canonical pointer both directions.
16. `pyproject.toml`: `timesfm>=3.0.0`, add pytest to dev deps.
17. REPRODUCIBILITY.md wrong timesfm troubleshooting row.

## P1 — Tuning (implementing now, pilot stays small)

- **Realistic DGP**: regime-shift + seasonal N + Beta-Binomial overdispersion +
  AR(1) noise; 5+ seeds, mean±sd.
- **Simulator persistence**: sticky delinquency (~0.3%/mo Markov), autocorrelated
  deposits, variable-rate share (~30%) tied to BoC path, decrementing
  amortization, episode `-02` at maturity, term heterogeneity {36,60,84},
  lognormal income.
- **M3 probe**: fixed Δ grid {±2.0, ±0.75, ±0.25, 0} on M1a rate → CRPS/coverage
  vs Δ, testing the bounded-adjustment hypothesis with negative control.
- **Rolling-origin evaluation** instead of single 12/6 split.
- **Hybrid**: concentration as parameter (default from backend, not hardcoded 20);
  drop the {0,1} raise, keep epsilon clip.
- **Controller**: first-float regex → last float / strict parse; `import re` at
  top; TypedDict for evidence; timeout+retry on LLM call.
- **Metrics**: `crps_counts` seed parameter (deterministic).
- **Ruff/mypy**: 92 ruff errors → clean (per-file-ignores for scripts/experiments
  prints, fix D413); mypy strict clean with optional-dep overrides.
- **Bootstrap**: TypedDict, loud unresolved-revision warning, drop dead vars.
- **Tests**: real-path contract tests (stub pipeline), quantile-level test,
  adjustment clipping tests, leakage fail-closed tests, M0 unfitted test;
  cut shallow tests (interface attributes, assert-is-not-None, Pydantic
  construction positives).

## P2 — Deferred / polish
- Per-forecast cost/latency instrumentation (do timing; cost when Groq wired to M2).
- Darts-vs-raw rationale note.
- tmp_path fixture cleanup, boundary timestamp test.

## What the critics got wrong / judged too harshly
- None of the P0s were disputed on re-read; all are real.
- The "circular DGP" charge is fair for the *claim* ("beats baseline on this task
  family") but the pilot was always a smoke test; the fix (3 scenarios) is cheap.
