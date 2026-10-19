"""Prompt templates for the LLM integration ladder (exp55–exp57).

Pure functions only — no I/O. Every builder takes (dataset description,
origin date, history ``<= origin``) and returns a string. PIT safety is
audited in ``docs/PROMPT_LEAKAGE_AUDIT.md``: each template field is traced
to a data source whose as-of date is ``<= origin`` by construction.

Prompt lineage: the forecaster system prompt and the generic user-prompt
shape replicate Vector's ``QuantileGridLLMPredictor`` (``_build_system_prompt``
/ ``_build_user_prompt`` in
``aieng-forecasting/aieng/forecasting/methods/llm_processes/quantile_grid.py``);
adaptations are documented in ``docs/PROMPT_LEAKAGE_AUDIT.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

#: 11-level standard quantile grid elicited from the LLM (Vector's STANDARD_QUANTILES).
LEVELS_11: list[float] = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

#: 9-level subset used for CRPS scoring (matches exp52/exp53).
LEVELS_9: list[float] = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def serialize_history(values: Sequence[float], dates: Sequence[str], precision: int = 2) -> str:
    """Render a cutoff-filtered history as one ``YYYY-MM: value`` line per row.

    Args:
        values: Observed values, all ``<= origin`` (caller-enforced cutoff).
        dates: Matching month-start date strings (``YYYY-MM-DD`` accepted,
            rendered as ``YYYY-MM``).
        precision: Decimal places for values.

    """
    lines = [f"{str(d)[:7]}: {float(v):.{precision}f}" for v, d in zip(values, dates, strict=True)]
    return "\n".join(lines)


def system_prompt_forecaster() -> str:
    """System prompt for LLM-as-forecaster (rungs 2 and 4).

    Replicates Vector's quantile-grid system prompt: JSON only, exact array
    length, monotone quantiles, same units as the input series.
    """
    return (
        "You are a probabilistic time-series forecaster. Given a historical series and a "
        "task description, return calibrated predictive quantiles for every requested "
        "forecast step.\n"
        "\n"
        "Rules:\n"
        "- Return ONLY a JSON object of the form "
        '{"forecasts": [{"q05": ..., "q10": ..., "q20": ..., "q30": ..., "q40": ..., '
        '"q50": ..., "q60": ..., "q70": ..., "q80": ..., "q90": ..., "q95": ...}]}. '
        "No prose, no markdown.\n"
        "- The 'forecasts' array MUST have exactly the requested number of elements, one "
        "per forecast step in chronological order.\n"
        "- Each forecast object MUST contain q05, q10, q20, q30, q40, q50, q60, q70, "
        "q80, q90, and q95.\n"
        "- Quantiles must be in the same units as the input series.\n"
        "- Quantiles must be monotone non-decreasing within each forecast step."
    )


def system_prompt_calibrator() -> str:
    """System prompt for LLM-as-calibrator (rung 3).

    Same output contract as the forecaster, but the role is framed as
    adjusting a statistical model's quantile grid rather than forecasting
    from scratch.
    """
    return (
        "You are a forecast calibrator. Below is a statistical model's quantile forecast "
        "for a monthly economic series, together with the observed history. Your job is to "
        "return a calibrated version of that forecast: keep what the statistical model "
        "gets right, and fix its spread and centering using the history.\n"
        "\n"
        "Rules:\n"
        "- Return ONLY a JSON object of the form "
        '{"forecasts": [{"q05": ..., "q10": ..., "q20": ..., "q30": ..., "q40": ..., '
        '"q50": ..., "q60": ..., "q70": ..., "q80": ..., "q90": ..., "q95": ...}]}. '
        "No prose, no markdown.\n"
        "- The 'forecasts' array MUST have exactly the requested number of elements, one "
        "per forecast step in chronological order.\n"
        "- Each forecast object MUST contain q05, q10, q20, q30, q40, q50, q60, q70, "
        "q80, q90, and q95.\n"
        "- Quantiles must be in the same units as the input series.\n"
        "- Quantiles must be monotone non-decreasing within each forecast step."
    )


def user_prompt_generic(
    *,
    task_line: str,
    series_desc: str,
    units: str,
    frequency: str,
    history_str: str,
    forecast_start: str,
    forecast_end: str,
    n_steps: int,
    suffix: str = "",
) -> str:
    """Vector-style generic user prompt: task + meta + history + forecast instruction.

    This is the exp53 replication prompt (rung 2). ``history_str`` must contain
    only observations ``<= origin``.
    """
    meta_block = f"Series: {series_desc}\nUnits: {units}\nFrequency: {frequency}"
    base = (
        f"Task: {task_line}\n"
        "\n"
        f"{meta_block}\n"
        "\n"
        "History:\n"
        f"{history_str}\n"
        "\n"
        f"Forecast the next {n_steps} {'value' if n_steps == 1 else 'values'} "
        f"({forecast_start} through {forecast_end}).\n"
        "Return a JSON object with a 'forecasts' array of length "
        f"{n_steps}; each item contains the standard quantile fields q05 through q95."
    )
    if suffix:
        base = f"{base}\n\n{suffix.lstrip(chr(10))}"
    return base


def calibrator_grid_text(grid: Sequence[float], levels: Sequence[float]) -> str:
    """Render a classical quantile grid as ``level: value`` lines for the prompt."""
    return "\n".join(f"{float(lv):.2f}: {float(v):.2f}" for lv, v in zip(levels, grid, strict=True))


def user_prompt_calibrator(
    *,
    task_line: str,
    series_desc: str,
    units: str,
    frequency: str,
    history_str: str,
    classical_grid_str: str,
    forecast_start: str,
    forecast_end: str,
    n_steps: int,
) -> str:
    """Rung 3 calibrator user prompt: meta + history + classical grid + calibrate instruction."""
    meta_block = f"Series: {series_desc}\nUnits: {units}\nFrequency: {frequency}"
    return (
        f"Task: {task_line}\n"
        "\n"
        f"{meta_block}\n"
        "\n"
        "History:\n"
        f"{history_str}\n"
        "\n"
        "Statistical model's 9-quantile forecast (levels 0.10 to 0.90):\n"
        f"{classical_grid_str}\n"
        "\n"
        f"Return the calibrated 11-quantile grid for the next {n_steps} "
        f"{'value' if n_steps == 1 else 'values'} ({forecast_start} through {forecast_end}): "
        "a 'forecasts' array of length "
        f"{n_steps}, each item with quantile fields q05 through q95."
    )


def words_stats_block(
    *,
    last_value: float,
    last_date: str,
    change_12m_pct: float,
    ref_12m: str,
    change_3m_pct: float,
    ref_3m: str,
    max_12m: float,
    min_12m: float,
    as_of: str,
) -> str:
    """Rung 4 V1 "Recent behavior (in words)" block.

    Every number is derived from the pre-origin history (which ends at
    ``as_of``). Reference months are printed so a fallback to an earlier
    month (missing observation) is visible rather than silent. No post-origin
    information can enter.
    """
    return (
        f"Recent behavior (in words) — all computed from observations up to {as_of}:\n"
        f"- Last observed value: {last_value:.2f} ({last_date}).\n"
        f"- 12-month change: {change_12m_pct:+.2f}% (vs {ref_12m}).\n"
        f"- 3-month change: {change_3m_pct:+.2f}% (vs {ref_3m}).\n"
        f"- Highest value in the last 12 months: {max_12m:.2f}; "
        f"lowest: {min_12m:.2f}."
    )


def macro_context_block(*, facts: Sequence[tuple[str, str, str, str]], as_of: str) -> str:
    """Rung 4 V2 macro-context block.

    Args:
        facts: (label, value_str, fact_as_of, source) tuples. Every fact must
            carry its own as-of date ``<= origin``; the caller (experiment
            script) performs the PIT-safe lookup.
        as_of: Origin date; nothing in the prompt may be newer than this.

    """
    lines = [
        "Macro context — every fact is labeled with its as-of date; "
        f"nothing here is newer than {as_of}:"
    ]
    for label, value_str, fact_as_of, source in facts:
        lines.append(f"- {label}: {value_str} (as of {fact_as_of}; source: {source}).")
    return "\n".join(lines)


def user_prompt_v1(
    *,
    task_line: str,
    series_desc: str,
    units: str,
    frequency: str,
    history_str: str,
    stats_block: str,
    forecast_start: str,
    forecast_end: str,
    n_steps: int,
) -> str:
    """Rung 4 V1: generic prompt + a "Recent behavior (in words)" block."""
    return user_prompt_generic(
        task_line=task_line,
        series_desc=series_desc,
        units=units,
        frequency=frequency,
        history_str=history_str,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        n_steps=n_steps,
        suffix=stats_block,
    )


def user_prompt_v2(
    *,
    task_line: str,
    series_desc: str,
    units: str,
    frequency: str,
    history_str: str,
    stats_block: str,
    macro_block: str,
    forecast_start: str,
    forecast_end: str,
    n_steps: int,
) -> str:
    """Rung 4 V2: V1 + PIT-safe macro context block."""
    return user_prompt_generic(
        task_line=task_line,
        series_desc=series_desc,
        units=units,
        frequency=frequency,
        history_str=history_str,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        n_steps=n_steps,
        suffix=f"{stats_block}\n\n{macro_block}",
    )
