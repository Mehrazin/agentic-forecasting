# Reproducibility Guide — TD-One Model Weights & Experiments

This document describes how to reproduce the TD-One model-weight bootstrap
and Experiment 2 (M0 vs M1 with real TSFM weights) from scratch.

## 1. Prerequisites

- Python 3.12+
- `uv` package manager (`~/.local/bin/uv`)
- ~3 GB free disk (weights: Chronos-2 ~500MB, TimesFM-2.5 ~800MB)
- ~2 GB free RAM (models load one at a time on CPU)
- Network access to huggingface.co (via egress proxy if set)

## 2. Bootstrap model weights

```bash
cd td-one

# NOTE: if your shell has IPv6 entries in no_proxy (e.g. [::1]),
# huggingface_hub fails with "Invalid port". Sanitize first:
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

uv run --with huggingface_hub python scripts/bootstrap_models.py
```

What it does:
1. Downloads `amazon/chronos-2` → `models/weights/chronos2/`
2. Downloads `google/timesfm-2.5-200m-pytorch` → `models/weights/timesfm25/`
3. Resolves exact commit SHAs and writes `models/MANIFEST.json`
   with revisions, sizes, directory hashes, and UTC timestamp.

The script is idempotent — re-running verifies existing snapshots
instead of re-downloading.

**Pinned models (recorded in MANIFEST.json at bootstrap time):**

| Key | HF repo | Params | Role |
|-----|---------|--------|------|
| chronos2 | `amazon/chronos-2` | 120M | M1a (primary, best calibration) |
| timesfm25 | `google/timesfm-2.5-200m-pytorch` | 200M | M1b (secondary, independent family) |

## 3. Run Experiment 2 (real weights)

```bash
cd td-one

uv run --with torch --with chronos-forecasting \
    --with timesfm python experiments/exp2_real_weights.py
```

(`huggingface_hub` is only needed for the §2 bootstrap, not for the
run — weights load from the local snapshot pinned in
`models/MANIFEST.json`.)

What it does:
1. Loads `models/MANIFEST.json` for weights paths (fails fast if missing).
2. Generates synthetic 18-period switching data (seed=42, mild upward drift).
3. **M0**: fits Jeffreys Beta-Binomial on periods 1–12, predicts 13–18.
4. **M1a**: loads Chronos-2 → zero-shot forecast of rates → unloads.
5. **M1b**: loads TimesFM-2.5 → zero-shot forecast of rates → unloads.
6. Reports MAE for each; writes `experiments/exp2_results.json`.

Models load **one at a time** and are explicitly unloaded between runs
to stay within RAM limits on small VMs.

### Mock mode (no downloads, for CI)

```bash
TD_ONE_MOCK=1 uv run python experiments/exp2_real_weights.py
```

Uses calibrated mocks instead of real weights. Results are marked
`"mode": "mock"` in the output JSON.

## 4. Expected outputs

| File | Contents |
|------|----------|
| `models/MANIFEST.json` | Pinned revisions, sizes, hashes, timestamp |
| `models/weights/chronos2/` | Chronos-2 snapshot |
| `models/weights/timesfm25/` | TimesFM-2.5 snapshot |
| `experiments/exp2_results.json` | MAE per model, mode, seed |
| `experiments/exp2_report.md` | Human-readable report |

## 5. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Invalid port: ':1]'` on download | Sanitize `no_proxy`/`NO_PROXY` (see §2) |
| OOM during inference | Reduce `n_samples` in exp2 script; models already load sequentially |
| `chronos-forecasting` import error | Ensure `--with chronos-forecasting` is passed to `uv run` |
| `TimesFM_2p5_200M_torch` import fails | Ensure `timesfm>=3.0` — the 3.x API moved to `timesfm.timesfm_2p5.timesfm_2p5_torch.TimesFM_2p5_200M_torch`; the 2.x `TimesFm` class no longer exists |
| Gated repo 401/403 | Set `HF_TOKEN` env var or run `hf auth login` |

## 6. Version history

- 2026-10-15: Initial bootstrap (Chronos-2 + TimesFM-2.5), Experiment 2.
