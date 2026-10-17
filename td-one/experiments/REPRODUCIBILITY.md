# Reproducibility Guide — TD-One Experiments

How to reproduce TD-One's model-weight bootstrap and every major experiment
from scratch. Commands assume `cd td-one` from the repo root.

## 0. Read this first: environment quirks

Three environment-specific gotchas that will waste your time if you don't know them:

| Quirk | Rule |
|---|---|
| `uv run` hangs at 0% CPU in this VM | **Always use `.venv/bin/python` directly**, never `uv run`. Every command below uses this form. |
| `/tmp` is a 512MB tmpfs | Extract large archives (Freddie Mac 7z files, ExpressVPN parts, etc.) under `~/workspace/`, never `/tmp`. |
| `huggingface_hub` fails with `Invalid port: ':1]'` | Your shell's `no_proxy` has IPv6 entries (e.g. `[::1]`). Sanitize first: `export no_proxy="localhost,127.0.0.1"` and `export NO_PROXY="localhost,127.0.0.1"` before any HF download. |

## 1. Prerequisites

- Python 3.12+, with the project venv at `td-one/.venv/` (`uv sync` to create)
- ~3 GB free disk for weights (Chronos-2 ~500MB, TimesFM-2.5 ~800MB)
- ~2 GB free RAM (models load one at a time on CPU)
- Network access to huggingface.co (via egress proxy if set; sanitize `no_proxy` per §0)
- Freddie Mac raw files in `td-one/data/freddie/raw/` (gitignored, 21 vintages 2021Q1–2026Q1) — needed for exp16/exp18 data prep only; the derived `smm_monthly_2021_2026.csv` is committed
- Groq vault credential (`custom.groq`) — needed only for exp17 (regime reader LLM calls)

## 2. Bootstrap model weights (pinned SHAs)

```bash
cd td-one
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
.venv/bin/python scripts/bootstrap_models.py
```

What it does:
1. Downloads `amazon/chronos-2` → `models/weights/chronos2/`
2. Downloads `google/timesfm-2.5-200m-pytorch` → `models/weights/timesfm25/`
3. Writes `models/MANIFEST.json` with revisions, sizes, directory hashes, UTC timestamp.

**Weight revisions are pinned in the script** (not floating `main`):

| Key | HF repo | Pinned SHA | Size |
|-----|---------|------------|------|
| chronos2 | `amazon/chronos-2` | `29ec3766d36d6f73f0696f85560a422f50e8498c` | ~456MB |
| timesfm25 | `google/timesfm-2.5-200m-pytorch` | `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` | ~882MB |

All TSFM results in this repo (exp2 v2, exp14, exp16, exp18, r2_exp13, scale-ups)
were produced with exactly these snapshots. The script is idempotent —
re-running verifies existing snapshots instead of re-downloading.

Weights are gitignored (1.3GB); `models/MANIFEST.json` is committed.

### Mock mode (no downloads, for CI)

Most TSFM wrappers accept `use_mock=True`, and several experiment scripts
honor `TD_ONE_MOCK=1` to run on calibrated mocks. Mock results are marked
`"mode": "mock"` in output JSONs. Mock numbers do **not** reproduce real-weight
numbers — they only exercise the pipeline.

## 3. Experiment 2 — M0 vs real TSFM weights (synthetic, seed=42)

```bash
cd td-one
.venv/bin/python experiments/exp2_real_weights.py
```

1. Loads `models/MANIFEST.json` (fails fast if weights missing).
2. Generates synthetic 18-period switching data (seed=42, mild upward drift).
3. **M0**: Jeffreys Beta-Binomial on periods 1–12, predicts 13–18.
4. **M1a**: Chronos-2 zero-shot → unloads. **M1b**: TimesFM-2.5 zero-shot → unloads.
5. Writes `experiments/exp2_results.json` (MAE per model, mode, seed) and `experiments/exp2_report.md`.

Models load one at a time and are explicitly unloaded between runs to stay
within RAM limits on small VMs. Mock mode: `TD_ONE_MOCK=1 .venv/bin/python experiments/exp2_real_weights.py`.

Expected (real weights): Chronos-2 MAE ~4.00, TimesFM-2.5 MAE ~4.62, M0 MAE ~5.00
on the synthetic DGP. (exp2 v1's "20% win" was retracted — circular DGP; see
`experiments/README.md` finding #1.)

## 4. Experiment 16 — first real-data SMM forecast (proof of concept)

```bash
cd td-one
.venv/bin/python experiments/exp16_freddie_smm.py
```

Forecasts monthly SMM (single-month mortality / prepayment rate) for
origination-vintage × FICO-band cohorts on Freddie Mac loan-level data.

**Honest limitations (by design):** only 3 months of perf data (2026-01 to
2026-03, vintage 2026Q1). Train on Jan+Feb, forecast Mar — 2 training points
per cohort. This is a pipeline proof-of-concept, NOT a definitive model
comparison.

- Data: `data/freddie/raw/2026Q1/` (gitignored) via `td_one.freddie.monthly_prepay_rate`.
- Models: m0 (trailing mean), m0_conformal (trailing mean ± residual quantiles), chronos2 (real weights, 21 quantiles).
- Scoring: CRPS via pinball estimator (`crps_from_quantiles` in `td_one/evaluation/real_benchmark.py`); CRPS = MAE for point forecasts.
- PIT: cohort SMM for month m uses only perf rows with period == m.
- Writes `experiments/exp16_freddie_smm_results.json`.

Expected: m0 CRPS ~0.00060; uncalibrated chronos2 ~0.017 (29× worse — this
motivated the `LogScaleCalibrator` used in exp18). Superseded by exp18 for the
full 63-month series.

## 5. Experiment 17 — news-augmented regime reader, end-to-end (pilot, n=4)

```bash
cd td-one
.venv/bin/python experiments/exp17_news_regime.py
```

Wires three components that were built but never ran together:
1. `RAGTool` (BM25 → Groq rerank → abstain) over the news corpus
2. `NewsSearchTool(backend="rag")` as the agent-facing interface
3. `read_regime_with_news` (v2 prompt: ABSTAIN class, counterargument step, news discipline)

For 4 historical months in 2026, retrieves PIT-safe news (strictly before each
month's cutoff), feeds it to the regime reader, and compares the predicted
switching-rate direction against the actual next-month 30Y mortgage rate move
(FRED MORTGAGE30US, hardcoded month-end values in the script).

- Requires the Groq vault credential (`custom.groq`); ~12 Groq calls total (well within free tier).
- Requires the news DB at `data/news/news_2026.db` (gitignored; multipart zip in `data/news_dist/` — see that directory's README for reassembly).
- Writes `experiments/exp17_news_regime_results.json`.

Expected: 3/4 correct direction classifications (2026-05, 2026-06, 2026-08 correct;
2026-02 miss — news celebrated falling rates while March reversed). **This is a
qualitative pilot (n=4), not a backtest — do not cite as a statistically
established result.**

## 6. Experiment 18 — 63-month Freddie Mac SMM benchmark (flagship)

```bash
cd td-one
.venv/bin/python experiments/exp18_smm_full.py
```

Rolling-origin backtest on the full Freddie Mac monthly prepayment (SMM)
series, Jan 2021 – Mar 2026 (63 months, `data/freddie/smm_monthly_2021_2026.csv`
— committed). 50 rolling origins (2022-01 … 2026-02); train on months ≤ origin,
forecast origin+1.

Models:
- `m0_w3` / `m0_w6` / `m0_w12`: trailing-mean point forecasts (baselines)
- `m0_conformal`: trailing-12 mean ± empirical residual quantiles (80% nominal)
- `chronos2_cal`: Chronos-2 (real weights) + `LogScaleCalibrator`
- `timesfm25_cal`: TimesFM-2.5 (real weights) + `LogScaleCalibrator`

**The `LogScaleCalibrator` is required:** raw TSFM spreads are calibrated in
absolute terms and blow up 29× on near-zero SMM (exp16). Log-space forecasting
makes spreads relative. It depends on the `validate_input=False` output-clip fix
in `td_one/models/tsfm.py` (without it, the calibrator produces all-1.0
forecasts on real weights — see commit history).

Scoring: CRPS via pinball estimator (`crps_from_quantiles`); MAE on
median/point; 80% interval coverage. PIT: the SMM CSV is precomputed per
calendar month; slicing train to months ≤ origin guarantees no lookahead.

**Reproducibility notes:**
- M0 models are deterministic. TSFM inference now accepts a `seed` parameter
  (`forecast_quantiles(..., seed=...)`, threading `torch.manual_seed`); the
  committed results predate seeding, so **exact bit-reproduction of the TSFM
  CRPS values is not guaranteed** — the ranking (TSFM > m0) is robust to this noise.
- Runtime: tens of minutes on CPU (TimesFM ~15s/forecast × 50 origins).
- Writes `experiments/exp18_smm_full_results.json`.

Expected (from `exp18_smm_full_results.json`):

| Model | mean CRPS | mean MAE | 80% coverage |
|---|---|---:|---:|
| m0_w3 | 0.000740 | 0.000740 | n/a (point) |
| m0_w6 | 0.000886 | 0.000886 | n/a (point) |
| m0_w12 | 0.001016 | 0.001016 | n/a (point) |
| m0_conformal | 0.000778 | 0.001016 | **0.50** (miscalibrated — see below) |
| chronos2_cal | 0.000523 | 0.000662 | 0.88 |
| timesfm25_cal | **0.000515** | 0.000644 | 0.84 |

TimesFM_cal beats m0_w3 by 30% (paired t, p=0.0012, survives Bonferroni over
8 comparisons; wins 35/50 origins, every year 2022–2026). Chronos2_cal beats
m0_w3 by 29% (p=0.011 — nominally significant, does not survive Bonferroni).

**Caveats:** baselines are trailing-window means only — no ETS/ARIMA/Theta was
run, so "beats trailing mean" ≠ "beats classical." The m0_conformal 80%
interval covers at 0.50 (5.3σ below nominal) because symmetric residual
intervals fail on the trending 2024–2026 SMM — do not cite this implementation
as evidence that "conformal works."

## 7. R2 reruns — PIT-safe macro experiments

These rerun the R1 macro experiments with PIT-safe **lagged** macro
(`beta_hat * z[origin]`, predictive regression) after the Argon audit found the
R1/R2-old implementation had a dict-lookup bug (`fred_z.get(forecast_month)`
returned 0.0 always). All use the mortgage simulator; all record seeds in
their results JSONs.

### r2_exp10 — classical lagged macro (SEED = 0)

```bash
cd td-one
.venv/bin/python experiments/r2_exp10_classical.py
```

`MortgageSimulator(n_customers, n_months, seed=0)`; `run_backtest(..., seed=0)`.
Compares m0 windows vs `macro_m0` (OLS of trailing logit-rates on lagged
FRED z-scores, direct logit shift) on standard and macro DGPs.

Expected (macro DGP): macro_m0 1.923 vs m0_w6 2.28 (**-0.357**, ~53% of the
leaky R1 gain). **Caveats:** single seed, no uncertainty quantification;
m0_w3 (1.457) still beats macro_m0 — window tuning matters more than the
overlay. Exploratory, not confirmatory.

### r2_exp13 — TSFM + lagged macro, 3 seeds (SEEDS = (0, 1, 2))

```bash
cd td-one
.venv/bin/python experiments/r2_exp13_tsfm_macro.py
```

Residual conditioning: subtract β·z from training inputs, re-add at forecast
time. Requires real TSFM weights (§2). **Slow: 20+ minutes** (TimesFM
`model.compile()` ~2 min on CPU alone); use `.venv/bin/python`, never `uv run`.

Expected (Chronos2, 6 seed/DGP combos): 1/6 helps (-1.2%), 4/6 hurt
(+3% to +23%), 1/6 ties. **Caveats:** n=6, no significance testing (sign test
p=0.11) — suggestive pattern, not an established finding. Note `conf_m0`
beats both Chronos2 variants in every combo.

### r2_exp15 — corrected macro on v2 DGP (SEEDS = (0, 1, 2))

```bash
cd td-one
.venv/bin/python experiments/r2_exp15_macro_corrected.py
```

Uses `MortgageSimulatorV2` (harder DGP: richer heterogeneity, baked-in FRED
macro). Compares m0 vs oracle-β overlay vs estimated-β overlay.

Expected: m0 0.6092, oracle 0.6199 (-0.0107), estimated 0.692 (-0.0828).
Even the oracle hurts on v2 — DGP-specific, not a general verdict on macro.

## 8. Data dependencies

| Experiment | Data needed | In git? |
|---|---|---|
| exp2 | synthetic (generated in-script) | n/a |
| exp16 | `data/freddie/raw/2026Q1/` vintage | ❌ gitignored |
| exp17 | `data/news/news_2026.db` (reassemble from `data/news_dist/` zip parts) | ❌ gitignored (zip parts committed) |
| exp18 | `data/freddie/smm_monthly_2021_2026.csv` | ✅ committed |
| r2_exp10/13/15 | synthetic (generated in-script) | n/a |
| all TSFM runs | `models/weights/` via §2 bootstrap | ❌ gitignored (manifest committed) |

Freddie Mac raw files cannot be redistributed (data-use agreement) — they stay
gitignored. The derived monthly SMM CSV is committed.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| Command appears to hang at start | You're using `uv run` — it hangs at 0% CPU in this VM. Use `.venv/bin/python` (§0). |
| `Invalid port: ':1]'` on HF download | Sanitize `no_proxy`/`NO_PROXY` (§0, §2). |
| OOM during TSFM inference | Models already load sequentially and unload between runs; reduce origins or use mock mode. |
| `chronos-forecasting` import error | Install into the venv: `.venv/bin/pip install chronos-forecasting` (or `uv sync`). |
| `TimesFM_2p5_200M_torch` import fails | Ensure `timesfm>=3.0` — the 3.x API moved to `timesfm.timesfm_2p5.timesfm_2p5_torch.TimesFM_2p5_200M_torch`; the 2.x `TimesFm` class no longer exists. |
| Gated repo 401/403 | Set `HF_TOKEN` or run `hf auth login`. |
| `FileNotFoundError: models/MANIFEST.json` | Run the §2 bootstrap first; TSFM scripts fail fast without it. |
| `exp18_smm_full_results.json` has `NaN` values | Known: m0 point-forecast coverage is NaN (no interval). Strict JSON parsers choke — use Python's `json` (allows NaN) or sanitize. |
| `FileNotFoundError: data/news/news_2026.db` (exp17) | Reassemble from `data/news_dist/` — see that directory's README. |
| Extraction fails with "no space" | You're extracting to `/tmp` (512MB tmpfs). Extract under `~/workspace/` (§0). |

## 10. Version history

- 2026-10-15: Initial bootstrap (Chronos-2 + TimesFM-2.5), Experiment 2.
- 2026-10-17: Full rewrite. Pinned weight SHAs in bootstrap script; added TSFM
  `seed` parameter; documented exp16 (real SMM POC), exp17 (news regime pilot),
  exp18 (63-month SMM flagship), and r2_exp10/r2_exp13/r2_exp15 (PIT-safe macro
  reruns). Documented `uv run` hang, `/tmp` tmpfs limit, and NaN-in-JSON quirks.
