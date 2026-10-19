# `td_one.llm_rungs` — LLM Integration Ladder

Offline + online infrastructure for the 6-rung LLM integration ladder
(exp55–exp59). LLMs are called via `vector_client.py` (Vector proxy →
Gemini Flash) or `groq_client.py` (Groq fallback); everything else runs
offline on cached quantile grids.

## Modules

| Module | What it does |
|---|---|
| `vector_client.py` | `VectorProxyClient` — drop-in replacement for `GroqChatClient`, routes through `custom.vector-llm-proxy` to Gemini Flash. Same `complete()` protocol. |
| `runner.py` | `run_variant()` — runs one LLM variant over origins; resume-safe via JSON cache; `--backend {groq,vector}` flag. |
| `cache.py` | Cache load/save, `grid_to_crps()` for recomputing CRPS from stored grids. |
| `combine.py` | Offline combination: inject LLM grids into classical records, blocked CV with `ProposedForecaster`. Used by exp58. |
| `datasets.py` | Rung-1 dataset registry: description, regime spec, expert set, data path per dataset. |
| `prompts.py` | PIT-safe prompt templates (pure functions of dataset/origin/history). V1/V2 variants. |

## Experiments

| Exp | Rung | Script |
|---|---|---|
| exp55 | 2: LLM as expert | `experiments/exp55_run_llm.py` |
| exp56 | 3: LLM as calibrator | `experiments/exp56_run_llm.py` |
| exp57 | 4: Per-dataset prompts | `experiments/exp57_run_llm.py` |
| exp58 | 5: Hybrid router | `experiments/exp58_hybrid.py` |
| exp59 | 6: Generalizability | `experiments/exp59_generalize.py` |

Orchestrator: `experiments/run_llm_ladder.sh` (`BACKEND=vector` for proxy).

## Docs

- `docs/LLM_INTEGRATION_LADDER.md` — ladder plan and design
- `docs/PROMPT_LEAKAGE_AUDIT.md` — PIT audit for all prompt templates
