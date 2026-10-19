"""Prompt templates and JSON cache helpers for the LLM integration ladder.

Rungs (see ``docs/LLM_INTEGRATION_LADDER.md``):
- exp55 — rung 2: LLM as a regime-router expert (generic prompt, exp53 replication)
- exp56 — rung 3: LLM as a calibrator of a classical seasonal-naive grid
- exp57 — rung 4: per-dataset prompts V1 (history stats in words) and
  V2 (V1 + PIT-safe macro context)

All prompt builders are pure functions of (dataset, origin date, history
``<= origin``). No I/O happens here; the PIT-safe macro lookup lives in the
experiment scripts and is documented in ``docs/PROMPT_LEAKAGE_AUDIT.md``.

Offline side (no LLM calls): ``datasets`` (rung 1 registry),
``combine`` (``build_records_with_llm`` + ``blocked_cv_predict``), and the
evaluation half of ``cache`` (``save_grid_cache`` / ``load_grid_cache`` /
``grid_to_crps``).
"""

from td_one.llm_rungs.cache import (
    append_origin_record,
    done_origins,
    grid_to_crps,
    load_cache,
    load_grid_cache,
    new_cache,
    save_cache,
    save_grid_cache,
)
from td_one.llm_rungs.combine import blocked_cv_predict, build_records_with_llm
from td_one.llm_rungs.datasets import DATASET_SPECS, DatasetSpec
from td_one.llm_rungs.prompts import (
    LEVELS_9,
    LEVELS_11,
    calibrator_grid_text,
    macro_context_block,
    serialize_history,
    system_prompt_calibrator,
    system_prompt_forecaster,
    user_prompt_calibrator,
    user_prompt_generic,
    user_prompt_v1,
    user_prompt_v2,
    words_stats_block,
)

__all__ = [
    "DATASET_SPECS",
    "DatasetSpec",
    "LEVELS_11",
    "LEVELS_9",
    "append_origin_record",
    "blocked_cv_predict",
    "build_records_with_llm",
    "calibrator_grid_text",
    "done_origins",
    "grid_to_crps",
    "load_cache",
    "load_grid_cache",
    "macro_context_block",
    "new_cache",
    "save_cache",
    "save_grid_cache",
    "serialize_history",
    "system_prompt_calibrator",
    "system_prompt_forecaster",
    "user_prompt_calibrator",
    "user_prompt_generic",
    "user_prompt_v1",
    "user_prompt_v2",
    "words_stats_block",
]
