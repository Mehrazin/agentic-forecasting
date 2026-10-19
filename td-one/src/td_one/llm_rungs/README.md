# `td_one.llm_rungs` — LLM Integration Ladder

Infrastructure for the 6-rung LLM integration ladder (exp55–exp59): the
program to give the proposed method what Vector's LLM predictor has —
semantic context and better probabilistic judgment — without losing the
classical method's generalizability.

The pattern: LLMs are called once per (origin, prompt variant) and their
quantile grids are cached to JSON. Everything downstream (combination,
evaluation, ablations) runs **offline** on those caches — no LLM calls, fully
reproducible. Rung 1 is the dataset registry; rungs 2–4 are LLM variants;
rung 5 combines them with the classical experts; rung 6 tests whether the
combination generalizes to new datasets.

LLMs are called via `vector_client.py` (Vector proxy → Gemini Flash, the
default — no quota walls) or `groq_client.py` (Groq fallback).

## Modules

| Module | What it does |
|---|---|
| `datasets.py` | Rung-1 dataset registry: description, regime spec, expert set, data path per dataset |
| `prompts.py` | PIT-safe prompt templates — pure functions of (dataset, origin, history). V1 (metadata + history stats in words), V2 (V1 + macro lookup). Every field traced to a pre-origin source |
| `runner.py` | `run_variant()` — runs one LLM variant over origins; resume-safe via JSON cache; `--backend {groq,vector}` and `--vector-model {flash-lite,flash}` flags |
| `vector_client.py` | `VectorProxyClient` — drop-in replacement for `GroqChatClient`, routes through `custom.vector-llm-proxy` to Gemini Flash. Same `complete()` protocol |
| `cache.py` | Cache load/save, `grid_to_crps()` for recomputing CRPS from stored grids |
| `combine.py` | Offline combination: inject LLM grids into classical per-origin records, blocked CV with `ProposedForecaster`. Used by exp58 |

## Experiments

| Exp | Rung | Script | Verdict |
|---|---|---|---|
| exp55 | 2: LLM as expert | `experiments/exp55_run_llm.py` | LLM beats classical on Food CPI |
| exp56 | 3: LLM as calibrator | `experiments/exp56_run_llm.py` | Calibrator worse than naive — killed |
| exp57 | 4: Per-dataset prompts | `experiments/exp57_run_llm.py` | V1 (0.4262) > V2 (0.4599) > generic (0.5158) |
| exp58 | 5: Hybrid router | `experiments/exp58_hybrid.py` | 0.4526 beats Vector's 0.5688 by 20% — but plain llm-V1 (0.4262) beats the hybrid |
| exp59 | 6: Generalizability | `experiments/exp59_generalize.py` | Clean negative: LLM 64% worse (Tourism), 36% worse (M3). Specialist, not generalist |

Orchestrator: `experiments/run_llm_ladder.sh` (`BACKEND=vector` for the proxy).

## Docs

- `docs/LLM_INTEGRATION_LADDER.md` — ladder plan and design
- `docs/PROMPT_LEAKAGE_AUDIT.md` — PIT audit for all prompt templates

Up: [`td_one/`](../README.md).
