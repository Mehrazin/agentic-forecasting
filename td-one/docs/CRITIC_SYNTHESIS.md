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

---

# Critic Synthesis — TD-One Audit Round 2 (2026-10-16)

Six critics audited the new code since the 2026-10-15 audit: AI2-grade
retry (`agent/retry.py`), Groq client rewrite, live FRED client,
real-TSFM wiring, real-Groq exp3/exp4, and the new docs
(`experiments/README.md`, `docs/METHODOLOGY.md`). Verdicts: 4 fix-first,
1 ship (test soundness), 1 fix-first (code quality). No blocks; nothing
fabricated. All P0s and justified P1s below are fixed in this round.

## P0 — fixed

1. **"Real-LLM win" used the mock numerical backend — undisclosed.**
   `exp3_results.json` records `"forecast_model": "timesfm_mock"` with
   `"mode": "real"`. The seg-1 edge (0.82 vs 1.33) cannot be separated
   from the mock sharpness artifact that finding #2 debunks. Fix:
   `experiments/README.md` finding #4, `exp3_report.md`,
   `docs/METHODOLOGY.md` §6.2, and `docs/RESULTS.md` now state that
   real-mode M2 used the mock backend — the run demonstrates
   orchestration (driving the loop to finalization), not numerical
   superiority.
2. **`exp3_report.md` mislabeled its own mode.** Header said
   "Mode: real (Groq)"; body said "scripted ReAct agent" and "mock LLM;
   plumbing validation only"; evidence table showed −1 (broken
   sentinel). Fix: full rewrite for real mode; the −1 diagnostic is
   flagged as a known data-quality gap.
3. **`PLAN_SOLO.md` contradicted itself on ladder status.** Line 30 said
   M2/M3 ✅ done (mock); line 33 said M2/M3 ❌ planned. Fix: both read
   ✅ done with real-Groq runs committed; FRED architecture line updated
   to live vault surrogate; build order and call budgets corrected
   (exp3: 20, exp4: 10; 6 attempts, 60s floor on 429).

## P1 — fixed

**Framing / statistics**
4. Paired tests added to `experiments/README.md` and both
   `scale_up_report_real-*.md`: Chronos-2 wins 7/14 (paired-t p=0.212,
   Wilcoxon p=0.583); TimesFM-2.5 wins 5/14 (p=0.341/0.583) — **not
   significant**, reproducing exp2 v2 at scale. Multiple-comparisons
   noted (no correction; doesn't change the conclusion).
5. exp4 selective reporting corrected: osfi_tightening got the SIGN
   wrong (+0.30 vs true ≈ −0.35); benign spuriously adjusted (+0.30 vs
   0.00); systematic magnitude undershoot (0.18–0.30 vs true 0.45–0.70);
   `rate_shock_down` was a CRPS loss despite correct direction.
   Exploratory framing added (n=6 hand-designed, no significance).
6. Scale-up reports' Interpretation sections rewritten (were copied from
   the mock template): removed `*_mock` discussion, fixed the false
   "coverage below nominal for every model", added the monotone-sort
   methods disclosure, framed as "count-aware Bayes vs rate-only
   zero-shot on short panels".
7. Attempt history disclosed: this was the third real-mode exp3 attempt;
   the patient retry policy was the difference between 1/3 and 2/3
   finalized traces.
8. Single-seed/single-DGP limitation stated wherever scale-up findings
   appear; 14-vs-15 segment count corrected everywhere
   (15 designed; `term_short` never scores; 5 of 6 origins scored).

**Code**
9. `_validate_rates` rejects NaN (was silently passed; NaN flowed into
   real wrappers and `np.sort` parked it on the top quantile).
   Defensive guard in `_panel_rates`. Regression tests.
10. Empty Groq content → new `GroqEmptyContentError`, classified
    transient and retried (was bare RuntimeError, zero retries despite
    being a known gpt-oss failure mode). Regression tests.
11. `URLError` added to retry `transient_names` (urllib wraps timeouts/
    DNS failures in URLError — the "timeouts are retried" claim was
    empirically false). Retry-After now actually honored: the wrapped
    HTTPError carries the parsed value through (was dead code; the 60s
    floor always applied). Retry attempts now logged (were silent).
12. FRED `observations()`: raises `FredUnavailableError` on FRED error
    payloads (was silent empty Series); `ValueError` on `start > end`;
    new `vintage_dates` parameter with a docstring stating
    observation-date cutoff ≠ vintage-correct (~6/17 series revised).
    Manifest gains FRED `last_updated`; empty frames record null
    (not "nan" strings); per-series fetch retried (3 attempts);
    unknown series ids raise ValueError. Regression tests.
13. Strict mypy: 8 errors on new src/ code fixed (untyped helper
    returns, bare `dict` generics, wrapper Protocol instead of
    `dict[str, object]` + wrong ignore, `llm: LLMClient` annotation).
    `sys.path.insert` hoisted to module level with dedup guard
    (was per-attempt/per-request, unbounded). Shared `_repo_root()`
    helper replaces hand-counted `parents[3]` in 4 modules.
14. Tests: weights test gets `skipif` (fresh clones go red otherwise);
    monotone test pins the exact level→value mapping (was the weak
    sortedness invariant); 2.5s throttle covered with a fake-clock
    test; worst-case wall-clock documented on `complete()` (budget
    bounds spend, not time).

**Docs**
15. `docs/SCALE_UP_RUNBOOK.md`: exp3/exp4 rows, Groq retry paragraph,
    TimesFM timing (765s ≈ 13 min, not ~35), test-suite count (157+).
16. `REPRODUCIBILITY.md`: `--with huggingface_hub` is bootstrap-only.

## New results this round (ablations)

- **M3 bound** (`experiments/abl_m3_bound.py`): MAX_DELTA ∈
  {0.25, 0.5, 0.75, 1.5, 3.0} on exp4 mock scenarios. The bound binds
  only at 0.25 (mean CRPS 8.69); 0.5–3.0 identical (7.01–7.05) — no
  scripted delta exceeds 0.5. The ±0.75 choice is a safety rail against
  pathological LLM outputs, not a tuned hyperparameter.
- **M0 window** (`experiments/abl_m0_window.py`): trailing_window ∈
  {3, 6, 12} on the scale-up config. window=6 reproduces the published
  M0 exactly (5.360/0.671). **window=3 reaches CRPS 3.196** — beating
  both real TSFMs (4.46/4.59). The baseline has headroom; future
  M1-vs-M0 comparisons should tune the window or report the envelope.

## Deferred to a later pass (P2/P3)

*Status 2026-10-16: "Multi-seed scale-up; calibration study; real-data
validation" below are now DONE — see `RESULTS.md` §2e (multi-seed),
§2d/§2g (calibration → conformal default), §2k (M4 real-data). The
remaining items are still open.*
- Crossing-frequency instrumentation for the monotone rearrangement.
- FRED `load_series(sid, asof)` helper making the PIT cutoff structural
  (currently a documented convention on full-history parquet).
- `max_tokens` escalation on empty-content retry (P2-8).
- Minimum-context guard for real TSFMs (degenerate 1–2 point panels).
- M2 tool ablation and step-budget ablation (need paid-tier Groq).
- Multi-seed scale-up; calibration study; real-data validation.

## What the critics got wrong / judged too harshly

- The "counts-vs-rates asymmetry" (model soundness P1-2) is a valid
  framing constraint, but the reports already present the comparison
  as a system benchmark; the added "count-aware Bayes vs rate-only
  zero-shot on short panels" line closes it without redesign.
- Code quality P2-8 (`parents[3]` → shared helper) was implemented even
  though the pattern was consistent — cheap insurance against the
  exact bug class that bit us before.
