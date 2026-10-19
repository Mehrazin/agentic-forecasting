"""Offline combination framework for the LLM integration ladder (rungs 2-5).

The LLM worker produces one JSON grid cache per (origin, prompt variant);
everything here runs offline on those caches — no LLM calls:

- :func:`build_records_with_llm`: inject cached LLM grids into per-origin
  classical records as an extra ``"llm"`` expert.
- :func:`blocked_cv_predict`: blocked K-fold CV over per-origin records,
  replicating the evaluation pattern of ``experiments/exp49_final.py`` (via
  exp47's ``run_dataset_cv``): fit a fresh forecaster per fold on train
  records, score each test origin, return per-origin CRPS plus paired stats
  vs a baseline.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.stats import ttest_rel

from td_one.agent.proposed_method import LEVELS, ProposedForecaster
from td_one.agent.tools.evaluation import combine_quantiles
from td_one.llm_rungs.cache import grid_to_crps
from td_one.llm_rungs.prompts import LEVELS_9

#: Train-window stand-in for ``ProposedForecaster.predict`` (records carry no
#: train window; mirrors exp47's ``train_proxy = np.zeros(30)``).
_TRAIN_PROXY = np.zeros(30)


def _subset_indices(cache_levels: Sequence[float]) -> list[int]:
    """Return the indices of our 9 levels [0.1..0.9] inside the cache's level grid."""
    idx: list[int] = []
    for lv in LEVELS_9:
        hits = [i for i, c in enumerate(cache_levels) if abs(float(c) - lv) < 1e-9]
        if not hits:
            raise ValueError(f"level {lv} not present in cache levels {list(cache_levels)}")
        idx.append(hits[0])
    return idx


def build_records_with_llm(
    classical_records: list[dict[str, Any]],
    llm_cache: dict[str, Any],
    feature_names: list[str],
    expert_name: str = "llm",
) -> list[dict[str, Any]]:
    """Inject cached LLM quantile grids into classical per-origin records.

    Each classical record keeps its ``ProposedForecaster.fit`` format
    (``features``, ``quantiles`` {model: (9,H)}, ``crps`` {model: float},
    ``actual``, plus ``origin``) and gains ``quantiles[expert_name]``
    ((9,1)) and ``crps[expert_name]``. The LLM's level grid (e.g. 11
    levels) is subset to our 9 levels (0.1..0.9); the CRPS is recomputed
    from the subset grid with :func:`grid_to_crps` so it is consistent
    with the stored quantiles. ``expert_name`` lets several caches (e.g.
    prompt variants) be injected as distinct experts.

    Origins with no usable LLM entry (missing, or ``ok=False``) are dropped;
    every returned record keeps its ``"origin"`` so callers can align
    evaluations. Input records are copied, never mutated.
    """
    if "levels" not in llm_cache or "per_origin" not in llm_cache:
        raise ValueError("llm_cache must have 'levels' and 'per_origin' keys")
    idx = _subset_indices(llm_cache["levels"])
    by_origin = {int(e["origin"]): e for e in llm_cache["per_origin"] if e.get("ok", False)}

    out: list[dict[str, Any]] = []
    for r in classical_records:
        entry = by_origin.get(int(r["origin"]))
        if entry is None:
            continue  # no usable LLM grid for this origin
        if len(r["features"]) != len(feature_names):
            raise ValueError(
                f"origin {r['origin']}: {len(r['features'])} features, "
                f"expected {len(feature_names)} ({feature_names})"
            )
        h = int(np.asarray(next(iter(r["quantiles"].values())), dtype=float).shape[1])
        if h != 1:
            raise ValueError(
                f"origin {r['origin']}: LLM caches are single-horizon, classical record has H={h}"
            )
        actual = float(np.ravel(np.asarray(r["actual"], dtype=float))[0])
        grid9 = np.asarray(entry["grid"], dtype=float)[idx]
        crps_llm = grid_to_crps(grid9, LEVELS_9, actual)

        nr = dict(r)
        nr["quantiles"] = {**r["quantiles"], expert_name: grid9.reshape(9, 1).tolist()}
        nr["crps"] = {**r["crps"], expert_name: crps_llm}
        out.append(nr)
    return out


def _time_block_folds(
    records: list[dict[str, Any]], n_folds: int
) -> list[tuple[list[int], list[int]]]:
    """Contiguous time-block folds; replicates exp47's non-series branch."""
    order = sorted(range(len(records)), key=lambda i: records[i]["origin"])
    folds: list[tuple[list[int], list[int]]] = []
    chunk = len(order) // n_folds
    for f in range(n_folds):
        te = order[f * chunk : (f + 1) * chunk if f < n_folds - 1 else len(order)]
        te_set = set(te)
        tr = [i for i in order if i not in te_set]
        folds.append((tr, te))
    return folds


def _series_folds(
    records: list[dict[str, Any]], n_folds: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    """Series-grouped folds; replicates exp42's ``kfold_by_series``."""
    rng = np.random.default_rng(seed)
    series_ids = sorted({r["series_id"] for r in records})
    rng.shuffle(series_ids)
    buckets: list[list[str]] = [[] for _ in range(n_folds)]
    for i, sid in enumerate(series_ids):
        buckets[i % n_folds].append(sid)
    idx = list(range(len(records)))
    out: list[tuple[list[int], list[int]]] = []
    for f in range(n_folds):
        test_sids = set(buckets[f])
        te = [i for i in idx if records[i]["series_id"] in test_sids]
        tr = [i for i in idx if records[i]["series_id"] not in test_sids]
        out.append((tr, te))
    return out


def _equal_weights_via_tool(r: dict[str, Any], models: list[str]) -> np.ndarray:
    """Equal-weights combination via the ``combine_quantiles`` tool.

    Replicates exp49's ``_equal_weights_via_tool`` exactly (it routes the
    combination through the tool rather than a manual mean).
    """
    qm = {m: np.asarray(r["quantiles"][m], dtype=float) for m in models}
    h = next(iter(qm.values())).shape[1]
    out = np.zeros((len(LEVELS_9), h))
    for s in range(h):
        mq = {
            m: {float(q): float(v) for q, v in zip(LEVELS_9, qm[m][:, s], strict=True)}
            for m in models
        }
        out[:, s] = np.array(combine_quantiles(mq).values)
    return out


def _mean_crps(qs: np.ndarray, actual: np.ndarray) -> float:
    """Mean CRPS via 2*mean pinball over levels and horizons.

    Body replicates ``exp42_fforma._crps_quantile`` exactly (flattened
    (9,H) loss array, single ``np.mean``) so CV scores match exp47/exp49
    bit-for-bit. For H=1 this agrees with :func:`grid_to_crps` exactly.
    """
    qs = np.asarray(qs, dtype=float)
    y = np.asarray(actual, dtype=float)
    losses = np.where(
        y[None, :] >= qs,
        (y[None, :] - qs) * LEVELS[:, None],
        (qs - y[None, :]) * (1.0 - LEVELS[:, None]),
    )
    return 2.0 * float(np.mean(losses))


def _paired_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired comparison of two per-origin CRPS vectors (exp47's ``cmp``)."""
    t, p = ttest_rel(a, b)
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(a - b)),
        "t": float(t),
        "p": float(p),
        "win_rate": float(np.mean(a < b)),
    }


def blocked_cv_predict(
    records: list[dict[str, Any]],
    feature_names: list[str],
    forecaster_factory: Callable[[], ProposedForecaster],
    n_folds: int = 5,
    *,
    baseline: str | Callable[[], ProposedForecaster] = "best_single",
    fold_mode: str = "time",
    seed: int = 0,
) -> dict[str, Any]:
    """Blocked K-fold CV: fit per fold on train records, predict test origins.

    Replicates the evaluation pattern of ``experiments/exp49_final.py`` (via
    exp47's ``run_dataset_cv``): fold splits are deterministic given the
    records, so two calls on the same origin set produce aligned per-origin
    CRPS vectors suitable for paired tests.

    Args:
        records: per-origin dicts (``features``, ``quantiles`` {model:
            (9,H)}, ``crps`` {model: float}, ``actual``, ``origin``).
        feature_names: ordered feature names matching each record's features.
        forecaster_factory: zero-arg callable returning an unfitted
            ``ProposedForecaster`` (a fresh one is fit per fold).
        n_folds: number of folds.
        baseline: ``"best_single"`` (fold's train-best model, the exp47
            default), ``"equal"`` (mean over the forecaster's models),
            ``"equal_tool"`` (equal weights via the ``combine_quantiles``
            tool — the exp49 variant), or a zero-arg forecaster factory used
            as the baseline forecaster.
        fold_mode: ``"time"`` (contiguous blocks sorted by origin — the
            exp47 branch for single-series datasets) or ``"series"``
            (grouped by ``series_id`` — the exp47 branch for M4/synthetic).
        seed: seed for the series-shuffle in ``"series"`` mode.

    Returns:
        Dict with ``crps`` / ``baseline_crps`` (per-origin, aligned to the
        input record order), paired ``stats`` (method vs baseline), and
        ``per_fold`` summaries.

    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n_folds > len(records):
        raise ValueError(f"n_folds={n_folds} > n_records={len(records)}")
    if fold_mode == "time":
        folds = _time_block_folds(records, n_folds)
    elif fold_mode == "series":
        folds = _series_folds(records, n_folds, seed)
    else:
        raise ValueError(f"fold_mode must be 'time' or 'series', got {fold_mode!r}")
    if isinstance(baseline, str) and baseline not in ("best_single", "equal", "equal_tool"):
        raise ValueError(
            "baseline must be 'best_single', 'equal', 'equal_tool' or a forecaster "
            f"factory, got {baseline!r}"
        )

    method_crps = np.full(len(records), np.nan)
    base_crps = np.full(len(records), np.nan)
    per_fold: list[dict[str, Any]] = []
    base_name = baseline if isinstance(baseline, str) else "forecaster"

    for f, (tr_idx, te_idx) in enumerate(folds):
        if not te_idx:
            raise ValueError(f"fold {f}: empty test set")
        tr = [records[i] for i in tr_idx]

        fc = forecaster_factory()
        fc.fit(tr, feature_names)

        base_fc: ProposedForecaster | None = None
        best = ""
        if callable(baseline):
            base_fc = baseline()
            base_fc.fit(tr, feature_names)
        elif baseline == "best_single":
            mean_crps = {m: float(np.mean([r["crps"][m] for r in tr])) for m in fc.models}
            best = min(mean_crps, key=lambda m: mean_crps[m])
            base_name = f"best_single({best})"

        fold_c, fold_b = [], []
        for i in te_idx:
            r = records[i]
            actual = np.asarray(r["actual"], dtype=float)
            qm = {m: np.asarray(r["quantiles"][m], dtype=float) for m in fc.models}
            feats = dict(zip(feature_names, r["features"], strict=True))

            out = fc.predict(_TRAIN_PROXY, qm, feats, feature_names)
            c = _mean_crps(out.quantiles, actual)
            method_crps[i] = c
            fold_c.append(c)

            if base_fc is not None:
                b_out = base_fc.predict(_TRAIN_PROXY, qm, feats, feature_names)
                b = _mean_crps(b_out.quantiles, actual)
            elif baseline == "best_single":
                b = _mean_crps(np.asarray(r["quantiles"][best], dtype=float), actual)
            elif baseline == "equal":
                q_mean = np.mean(
                    [np.asarray(r["quantiles"][m], dtype=float) for m in fc.models], axis=0
                )
                b = _mean_crps(q_mean, actual)
            else:  # "equal_tool"
                b = _mean_crps(_equal_weights_via_tool(r, fc.models), actual)
            base_crps[i] = b
            fold_b.append(b)

        per_fold.append(
            {
                "fold": f,
                "n_test": len(te_idx),
                "mean_crps": float(np.mean(fold_c)),
                "mean_baseline_crps": float(np.mean(fold_b)),
            }
        )

    return {
        "n": len(records),
        "n_folds": n_folds,
        "fold_mode": fold_mode,
        "baseline": base_name,
        "crps": [float(v) for v in method_crps],
        "baseline_crps": [float(v) for v in base_crps],
        "stats": _paired_stats(method_crps, base_crps),
        "per_fold": per_fold,
    }
