#!/usr/bin/env python3
r"""Scale-up pipeline: simulate -> PIT ingest -> segment -> backtest -> report.

End-to-end run at full simulator scale (default 10,000 customers x 66
months):

1. Simulate mortgage histories + renewal outcomes (structural DGP).
2. Ingest into the DuckDB PIT store via a parquet fast path
   (``--skip-ingest`` to skip).
3. Discover segments -> monthly count panels, PIT-masked per forecast
   origin (``observed_asof``): training months only include outcomes that
   had been observed by the origin (leakage guard); future months keep
   actuals as scoring targets (the backtest runner never feeds them to
   the forecast function — enforced by its cutoff check).
4. Rolling-origin backtest per model (default origins: last 6 month-ends).
5. Write ``experiments/scale_up_results.json`` and
   ``experiments/scale_up_report.md``.

Models are wired as ``forecast_fn`` callables returning a next-month
*rate* quantile lattice ``{level: rate}``. ``m0`` is always available
(trailing-6-month Jeffreys Beta posterior). ``td_one.tools.forecasting``
is attempted inside ``main``; when it is unavailable (not yet implemented)
a clear note is printed and only ``m0`` runs.

Usage:
  cd td-one && uv run python scripts/scale_up.py --n-customers 10000 --n-months 66
  # smoke test from repo root:
  uv run --project td-one python td-one/scripts/scale_up.py \\
      --n-customers 500 --n-months 24 --models m0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from td_one.backtest.runner import ForecastFn, run_backtest  # noqa: E402
from td_one.data_gen.simulator import MortgageSimulator  # noqa: E402
from td_one.data_service.pit_store import PITStore  # noqa: E402
from td_one.models.baseline import HistoricalBaseline  # noqa: E402
from td_one.schemas import MortgageHistoryRecord, RenewalOutcomeRecord  # noqa: E402
from td_one.segments.discovery import discover_segments  # noqa: E402

#: Rate quantile levels emitted by the M0 forecast function.
M0_LEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

#: PIT table column orders (must match PITStore._init_schema).
_PIT_HISTORY_COLS = [
    "customer_id",
    "mortgage_id",
    "snapshot_date",
    "province",
    "mortgage_rate_pct",
    "td_liquid_deposits_cad",
    "credit_score_band",
    "household_income_cad",
    "children_under6",
    "children_6to17",
    "mortgage_balance_cad",
    "insured_flag",
]
_PIT_OUTCOME_COLS = [
    "customer_id",
    "mortgage_id",
    "renewal_episode_id",
    "scheduled_maturity_date",
    "outcome",
    "switched_lender_flag",
    "event_date",
    "outcome_observed_at",
    "record_asof",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="TD-One scale-up pipeline")
    p.add_argument("--n-customers", type=int, default=10000)
    p.add_argument("--n-months", type=int, default=66)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--origins",
        nargs="*",
        default=None,
        help="Forecast origins as YYYY-MM-DD (default: last 6 panel month-ends).",
    )
    p.add_argument(
        "--models",
        default="m0",
        help="Comma-separated model names (default: m0; only m0 is wired).",
    )
    p.add_argument(
        "--tag",
        default="",
        help="Tag inserted into output filenames (e.g. --tag real -> "
        "scale_up_results_real.json). Lets per-model runs coexist.",
    )
    p.add_argument(
        "--m0-window",
        default="6",
        help="Trailing window (months) for the M0 baseline: an integer "
        "(default: 6) or 'auto' to select 3/6/12 by trailing-origin "
        "CRPS validation per forecast.",
    )
    p.add_argument(
        "--conformal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the split-conformal M0 as the baseline (default: on). "
        "Conformal M0 fixes M0's undercoverage (0.671 -> 0.914 on seed 0) "
        "and is the published baseline. Use --no-conformal for plain M0.",
    )
    p.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip the PIT store ingest step.",
    )
    args = p.parse_args(argv)
    if args.n_customers <= 0 or args.n_months <= 0:
        p.error("--n-customers and --n-months must be positive")
    return args


def _parse_m0_window(raw: str) -> int | str:
    """Parse --m0-window: a positive int, or the string "auto".

    Raises:
        SystemExit: On invalid values.

    """
    if raw == "auto":
        return "auto"
    try:
        w = int(raw)
    except ValueError:
        raise SystemExit(f"bad --m0-window {raw!r} (want a positive int or 'auto')") from None
    if w < 1:
        raise SystemExit(f"bad --m0-window {raw!r} (want a positive int or 'auto')")
    return w


def _parse_origins(raw: list[str] | None) -> list[date] | None:
    """Parse YYYY-MM-DD origin strings; None stays None (derive later)."""
    if raw is None:
        return None
    try:
        return [date.fromisoformat(s) for s in raw]
    except ValueError as exc:
        raise SystemExit(f"bad --origins date (want YYYY-MM-DD): {exc}") from exc


def _m0_rate_forecast(train: pd.DataFrame, origin: date, seed: int = 0) -> dict[float, float]:
    """M0 as a ``forecast_fn``: trailing-6m Jeffreys Beta rate quantiles.

    Fits :class:`HistoricalBaseline` on the training panel and returns
    quantiles of the Beta posterior predictive rate. (Reads the fitted
    posterior shape the same way ``experiments/exp2_real_weights.py``
    does.)
    """
    del origin, seed  # M0 is a trailing-window empirical rate.
    model = HistoricalBaseline(trailing_window=6)
    model.fit(train["n_renewals"].to_numpy(), train["n_switches"].to_numpy())
    assert model._alpha is not None and model._beta is not None  # set by fit()
    q = stats.beta.ppf(M0_LEVELS, model._alpha, model._beta)
    return {float(level): float(value) for level, value in zip(M0_LEVELS, q, strict=True)}


def _m0_window_forecast(train: pd.DataFrame, origin: date, window: int) -> dict[float, float]:
    """M0 as a ``forecast_fn`` with a parametrized trailing window.

    Mirrors the ``forecast("m0", ...)`` path in
    ``td_one.tools.forecasting`` exactly (``_panel_rates`` filtering,
    ``TIMESFM_LEVELS`` grid, ``n_upcoming`` from the trailing window
    mean), except the trailing window is parametrized. At window=6 the
    output is identical to the tools path (verified: CRPS 5.360,
    coverage 0.671 on seed 0).
    """
    from td_one.models.tsfm import TIMESFM_LEVELS
    from td_one.tools.forecasting import _panel_rates

    n, a = _panel_rates(train, origin)
    model = HistoricalBaseline(trailing_window=window)
    model.fit(n, a)
    n_upcoming = max(1, int(round(float(n[-window:].mean()))))
    return model.predict_rate_quantiles(n_upcoming, np.asarray(TIMESFM_LEVELS, dtype=float))


def _m0_conformal_forecast(train: pd.DataFrame, origin: date, window: int) -> dict[float, float]:
    """Conformal M0 as a ``forecast_fn`` (the published baseline).

    Same wiring as :func:`_m0_window_forecast` but with
    :class:`ConformalM0`'s split-conformal interval widening. Fixes M0's
    chronic undercoverage (0.671 -> 0.914 on seed 0) and beats plain M0
    on CRPS at window 6 (5.044 < 5.360).
    """
    from td_one.models.conformal import ConformalM0
    from td_one.models.tsfm import TIMESFM_LEVELS
    from td_one.tools.forecasting import _panel_rates

    n, a = _panel_rates(train, origin)
    model = ConformalM0(trailing_window=window)
    model.fit(n, a)
    n_upcoming = max(1, int(round(float(n[-window:].mean()))))
    return model.predict_rate_quantiles(n_upcoming, np.asarray(TIMESFM_LEVELS, dtype=float))


#: Candidate windows for --m0-window auto selection.
_AUTO_WINDOWS = (3, 6, 12)
#: Pseudo-origins used for trailing validation.
_AUTO_VAL_ORIGINS = 4
#: Minimum history months for auto-tuning; below this, fall back to 6.
_AUTO_MIN_HISTORY = 12
#: MC samples for the validation CRPS.
_AUTO_MC_SAMPLES = 1000


def _beta_binom_crps(alpha: float, beta: float, n: int, actual: int, seed: int = 0) -> float:
    """Fast MC CRPS of a Beta-Binomial predictive vs an observed count.

    CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|, with the pairwise term via
    the sorting trick (O(m log m)). Deterministic given the seed.
    """
    rng = np.random.default_rng(seed)
    p = rng.beta(alpha, beta, size=_AUTO_MC_SAMPLES)
    counts = rng.binomial(n, p)
    diff1 = float(np.abs(counts - actual).mean())
    s = np.sort(counts)
    i = np.arange(_AUTO_MC_SAMPLES)
    diff2 = 2.0 * float(np.sum((2 * i - _AUTO_MC_SAMPLES + 1) * s)) / (_AUTO_MC_SAMPLES**2)
    return diff1 - 0.5 * diff2


def _m0_auto_window_forecast(
    train: pd.DataFrame, origin: date, conformal: bool = True
) -> dict[float, float]:
    """M0 with the trailing window selected by trailing-origin validation.

    For each candidate window in (3, 6, 12), the last
    ``_AUTO_VAL_ORIGINS`` months of the training panel are used as
    pseudo-origins: fit on history before the pseudo-origin, score CRPS
    against the pseudo-origin month's actual. The window with the best
    mean validation CRPS forecasts the real origin. With too little
    history, falls back to window 6. The selection uses only data
    available at each pseudo-origin (no leakage).
    """
    from td_one.models.conformal import ConformalM0
    from td_one.models.tsfm import TIMESFM_LEVELS
    from td_one.tools.forecasting import _panel_rates

    n, a = _panel_rates(train, origin)
    n = np.asarray(n, dtype=float)
    a = np.asarray(a, dtype=float)

    selected = 6
    if len(n) >= _AUTO_MIN_HISTORY:
        best_score = float("inf")
        for w in _AUTO_WINDOWS:
            scores: list[float] = []
            for v in range(1, _AUTO_VAL_ORIGINS + 1):
                # Pseudo-origin v months before the panel end.
                fit_n, fit_a = n[:-v], a[:-v]
                val_n, val_a = n[-v], a[-v]
                if len(fit_n) < w or val_n <= 0:
                    continue
                model = HistoricalBaseline(trailing_window=w)
                model.fit(fit_n, fit_a)
                assert model._alpha is not None and model._beta is not None
                scores.append(
                    _beta_binom_crps(model._alpha, model._beta, int(val_n), int(val_a), seed=v)
                )
            if scores:
                mean_score = float(np.mean(scores))
                if mean_score < best_score:
                    best_score = mean_score
                    selected = w

    if conformal:
        model_c = ConformalM0(trailing_window=selected)
        model_c.fit(n, a)
        n_upcoming = max(1, int(round(float(n[-selected:].mean()))))
        return model_c.predict_rate_quantiles(n_upcoming, np.asarray(TIMESFM_LEVELS, dtype=float))
    model = HistoricalBaseline(trailing_window=selected)
    model.fit(n, a)
    n_upcoming = max(1, int(round(float(n[-selected:].mean()))))
    return model.predict_rate_quantiles(n_upcoming, np.asarray(TIMESFM_LEVELS, dtype=float))


def _load_forecast_fns(
    model_names: list[str],
    seed: int,
    m0_window: int | str = 6,
    conformal: bool = True,
) -> tuple[dict[str, ForecastFn], list[str]]:
    """Wire model names to ``forecast_fn`` callables.

    ``td_one.tools.forecasting`` is imported here (not at module import)
    so the pipeline degrades gracefully while the tools package is empty.
    Each model is wired as ``functools.partial(forecast, name)``, matching
    the runner's ``(train_panel, origin) -> {level: rate}`` contract.

    Args:
        model_names: Model names to wire.
        seed: Base RNG seed (used by the local m0 fallback only).
        m0_window: Trailing window for M0: an int, or "auto" to select
            3/6/12 by trailing-origin CRPS validation per forecast.
        conformal: Use the split-conformal M0 as the baseline (default).
            Fixes M0's undercoverage; the published baseline.

    Returns:
        (forecast functions, names wired via td_one.tools.forecasting).

    """
    try:
        from td_one.tools import forecasting as tools_mod

        supported = tools_mod.SUPPORTED_MODELS
        print(f"[models] td_one.tools.forecasting loaded (models: {', '.join(supported)})")
    except ImportError as exc:
        print(f"[models] NOTE: td_one.tools.forecasting unavailable ({exc}); 'm0' fallback only.")
        tools_mod = None
        supported = ()

    auto = isinstance(m0_window, str) and m0_window == "auto"
    window_int = 6 if auto else int(m0_window)

    fns: dict[str, ForecastFn] = {}
    via_tools: list[str] = []
    for name in model_names:
        if tools_mod is not None and name in supported and name != "m0":
            fns[name] = partial(tools_mod.forecast, name)
            via_tools.append(name)
            print(f"[models] wired {name!r} via td_one.tools.forecasting")
        elif name == "m0" and auto:
            # Window selected per forecast by trailing validation.
            fns[name] = partial(_m0_auto_window_forecast, conformal=conformal)
            print(f"[models] wired 'm0' via auto-window validation (conformal={conformal})")
        elif name == "m0" and conformal:
            # Published baseline: conformal M0 with the requested window.
            fns[name] = partial(_m0_conformal_forecast, window=window_int)
            print(f"[models] wired 'm0' via conformal M0 (window={window_int})")
        elif name == "m0" and window_int != 6:
            # Parametrized trailing window (mirrors the tools m0 path).
            fns[name] = partial(_m0_window_forecast, window=window_int)
            print(f"[models] wired 'm0' via local windowed forecast (window={window_int})")
        elif name == "m0":
            # Local fallback while td_one.tools.forecasting is unavailable.
            fns[name] = partial(_m0_rate_forecast, seed=seed)
            print("[models] wired 'm0' via local fallback (_m0_rate_forecast)")
        else:
            raise SystemExit(
                f"unknown model {name!r} (tools models: {', '.join(supported) or 'none'}; "
                "fallback: m0)"
            )
    return fns, via_tools


def _default_origins(panels: dict[str, pd.DataFrame], k: int = 6) -> list[date]:
    """Last ``k`` month-ends across all segment panels."""
    months = sorted({m for panel in panels.values() for m in panel["month"]})
    return list(months[-k:])


def _ingest(
    histories: list[MortgageHistoryRecord],
    outcomes: list[RenewalOutcomeRecord],
    db_path: Path,
    work_dir: Path,
) -> float:
    """Bulk ingest into the PIT store via a parquet fast path.

    ``PITStore.ingest_history``/``ingest_outcomes`` use Python-level
    ``executemany`` with ``INSERT OR REPLACE`` (~40 rows/s measured: ~4.5h
    for 660k rows). Instead, records are dumped to parquet (Rust-speed,
    PIT-column subset only) and bulk-loaded with DuckDB-native
    ``INSERT ... SELECT FROM read_parquet(...)`` inside one transaction.
    The PIT store remains the system of record; row counts are verified
    after load. Returns elapsed seconds.
    """
    import duckdb

    t0 = time.time()
    work_dir.mkdir(parents=True, exist_ok=True)
    # Schema comes from PITStore (single source of truth for the tables).
    PITStore(db_path).close()

    hist_pq = work_dir / "histories.parquet"
    out_pq = work_dir / "outcomes.parquet"
    hist_df = pd.DataFrame([r.model_dump(include=set(_PIT_HISTORY_COLS)) for r in histories])[
        _PIT_HISTORY_COLS
    ]
    hist_df.to_parquet(hist_pq, index=False)
    print(f"  histories -> {hist_pq} ({len(hist_df)} rows)", flush=True)
    out_df = pd.DataFrame([r.model_dump(include=set(_PIT_OUTCOME_COLS)) for r in outcomes])[
        _PIT_OUTCOME_COLS
    ]
    out_df.to_parquet(out_pq, index=False)
    print(f"  outcomes -> {out_pq} ({len(out_df)} rows)", flush=True)

    con = duckdb.connect(str(db_path))
    try:
        con.execute("BEGIN")
        con.execute("DELETE FROM renewal_outcomes")
        con.execute("DELETE FROM mortgage_history")
        con.execute(
            "INSERT INTO mortgage_history SELECT * FROM read_parquet(?)",
            [str(hist_pq)],
        )
        con.execute(
            "INSERT INTO renewal_outcomes SELECT * FROM read_parquet(?)",
            [str(out_pq)],
        )
        con.execute("COMMIT")
        n_h = con.execute("SELECT COUNT(*) FROM mortgage_history").fetchone()[0]
        n_o = con.execute("SELECT COUNT(*) FROM renewal_outcomes").fetchone()[0]
    finally:
        con.close()
    if n_h != len(histories) or n_o != len(outcomes):
        raise RuntimeError(
            f"PIT row-count mismatch: history {n_h}/{len(histories)}, "
            f"outcomes {n_o}/{len(outcomes)}"
        )
    return time.time() - t0


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per-segment x model means over origins."""
    return (
        results.groupby(["segment", "model"], as_index=False)
        .agg(
            crps=("crps", "mean"),
            mae=("mae", "mean"),
            coverage_80=("coverage_80", "mean"),
            n_origins=("origin", "count"),
            mean_n_renewals=("n_renewals", "mean"),
        )
        .sort_values(["model", "segment"])
        .reset_index(drop=True)
    )


def main(argv: list[str] | None = None) -> int:
    """Run the scale-up pipeline. Returns process exit code."""
    args = parse_args(argv)
    model_names = list(filter(None, map(str.strip, args.models.split(","))))
    if not model_names:
        print("SCALE-UP FAILED: --models is empty", file=sys.stderr)
        return 2
    origins_arg = _parse_origins(args.origins)
    timings: dict[str, float] = {}
    t_start = time.time()

    try:
        # ---- 1. Simulate ----
        print(
            f"[1/5] Simulating {args.n_customers} customers x {args.n_months} months "
            f"(seed={args.seed}) ...",
            flush=True,
        )
        t0 = time.time()
        sim = MortgageSimulator(
            n_customers=args.n_customers, n_months=args.n_months, seed=args.seed
        )
        histories = sim.generate_history()
        outcomes = sim.generate_outcomes(histories)
        timings["simulate_s"] = round(time.time() - t0, 1)
        print(
            f"      {len(histories)} history records, {len(outcomes)} outcomes "
            f"in {timings['simulate_s']}s",
            flush=True,
        )

        # ---- 2. PIT ingest ----
        if args.skip_ingest:
            print("[2/5] PIT ingest skipped (--skip-ingest)", flush=True)
            timings["ingest_s"] = 0.0
        else:
            scale_up_dir = ROOT / "data" / "scale_up"
            db_path = scale_up_dir / "pit.duckdb"
            print(f"[2/5] Ingesting into PIT store {db_path} ...", flush=True)
            timings["ingest_s"] = round(_ingest(histories, outcomes, db_path, scale_up_dir), 1)
            print(f"      ingest done in {timings['ingest_s']}s", flush=True)

        # ---- 3. Segment discovery (per origin, PIT-masked) ----
        print("[3/5] Discovering segments (per-origin PIT masking) ...", flush=True)
        t0 = time.time()
        panels_full = discover_segments(histories, outcomes)
        origins = origins_arg or _default_origins(panels_full)
        origin_str = ", ".join(o.isoformat() for o in origins)
        timings["segments_s"] = round(time.time() - t0, 1)
        print(
            f"      {len(panels_full)} segments, {len(origins)} origins "
            f"[{origin_str}] in {timings['segments_s']}s",
            flush=True,
        )

        # ---- 4. Backtest per model ----
        print(f"[4/5] Backtesting models={model_names} ...", flush=True)
        m0_window = _parse_m0_window(args.m0_window)
        fns, via_tools = _load_forecast_fns(model_names, args.seed, m0_window, args.conformal)
        all_results: list[pd.DataFrame] = []
        t0 = time.time()
        for name in model_names:
            for origin in origins:
                # PIT-correct training side for this origin; future months
                # keep actuals as scoring targets. Origins with no
                # observable training data (everything still unobserved)
                # are skipped.
                try:
                    panels = discover_segments(histories, outcomes, observed_asof=origin)
                except ValueError as exc:
                    print(
                        f"      {name} @ {origin.isoformat()}: skipped ({exc})",
                        flush=True,
                    )
                    continue
                r = run_backtest(panels, fns[name], [origin], seed=args.seed)
                r["model"] = name
                all_results.append(r)
            # Free real-TSFM weights before the next model (RAM).
            try:
                from td_one.tools.forecasting import unload_real_models

                unload_real_models()
            except ImportError:
                pass
            print(f"      {name}: done", flush=True)
        timings["backtest_s"] = round(time.time() - t0, 1)
        if not all_results:
            print(
                "SCALE-UP FAILED: no scored cells (no origin had observable data)", file=sys.stderr
            )
            return 2
        results = pd.concat(all_results, ignore_index=True)
        print(f"      backtest done in {timings['backtest_s']}s", flush=True)

        # ---- 5. Write outputs ----
        print("[5/5] Writing outputs ...", flush=True)
        summary = _summarize(results)
        exp_dir = ROOT / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{args.tag}" if args.tag else ""
        results_json = exp_dir / f"scale_up_results{suffix}.json"
        report_md = exp_dir / f"scale_up_report{suffix}.md"
        payload = {
            "config": {
                "n_customers": args.n_customers,
                "n_months": args.n_months,
                "seed": args.seed,
                "models": model_names,
                "m0_window": args.m0_window,
                "conformal": args.conformal,
                "origins": [o.isoformat() for o in origins],
                "skip_ingest": args.skip_ingest,
            },
            "timings_s": timings,
            "total_runtime_s": round(time.time() - t_start, 1),
            "n_segments": len(panels_full),
            "results": results.to_dict(orient="records"),
            "summary": summary.to_dict(orient="records"),
            "notes": [
                "Segment panels are PIT-masked per forecast origin "
                "(observed_asof): training months only count outcomes with "
                "outcome_observed_at <= origin; future months keep actuals "
                "as scoring targets (never forecaster inputs).",
                "Models wired via td_one.tools.forecasting: "
                + (", ".join(via_tools) if via_tools else "none (local m0 fallback)."),
                "PIT ingest used the parquet fast path (native DuckDB bulk load); "
                "PITStore.executemany is ~40 rows/s and infeasible at 660k rows.",
            ],
        }
        # JSON needs plain types for dates.
        for row in payload["results"]:
            row["origin"] = str(row["origin"])
        results_json.write_text(json.dumps(payload, indent=2, default=str))

        lines = [
            "# Scale-up Report: TD-One segment backtest",
            "",
            f"**Date:** {datetime.now(UTC).date().isoformat()}. "
            f"**Config:** n_customers={args.n_customers}, n_months={args.n_months}, "
            f"seed={args.seed}, models={','.join(model_names)}, "
            f"m0_window={args.m0_window}, conformal={args.conformal}.",
            f"**Origins:** {', '.join(o.isoformat() for o in origins)}.",
            f"**Runtime:** {payload['total_runtime_s']}s "
            f"(simulate {timings['simulate_s']}s, ingest {timings['ingest_s']}s, "
            f"segments {timings['segments_s']}s, backtest {timings['backtest_s']}s).",
            "",
            "## Per-segment x model scores (mean over origins)",
            "",
            "| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, s in summary.iterrows():
            lines.append(
                f"| {s['segment']} | {s['model']} | {s['crps']:.3f} | {s['mae']:.2f} | "
                f"{s['coverage_80']:.2f} | {int(s['n_origins'])} | {s['mean_n_renewals']:.1f} |"
            )
        lines += [
            "",
            "## Interpretation (read before comparing models)",
            "",
            "- **The `*_mock` models are interface-only plumbing.** They center"
            " on the last observed rate with fixed tight probit spreads, so in"
            " a stable-rate environment they score better on CRPS than M0's"
            " wider Jeffreys posterior. That gap is a sharpness artifact of the"
            " placeholder, NOT evidence about Chronos-2/TimesFM. The real"
            " M0-vs-M1 comparison is `experiments/exp2_report.md` (real"
            " weights, 3 pre-registered scenarios).",
            "- **The published M0 is the conformal M0** (`--conformal`, default on)."
            " Split-conformal widening fixes M0's chronic undercoverage"
            " (0.671 -> 0.914 on seed 0); use `--no-conformal` for plain M0.",
            "- **M0's trailing window is a tuning parameter** (`--m0-window`,"
            " default 6, or 'auto' for per-forecast trailing validation)."
            " Its direction is seed-dependent (w3 wins 3/4 seeds, w6 wins 1/4);"
            " report the w3/w6 envelope or tune it, never a single window.",
            "- Segment panels are PIT-masked per origin (`observed_asof`):"
            " training months count only outcomes with `outcome_observed_at <="
            " origin`. Future months keep actuals purely as scoring targets.",
            "- Maturities spread year-round (simulator draws a uniform"
            " origination month per customer); origins are the last 6 panel"
            " month-ends.",
            "",
            "## Notes",
            "",
            f"- Models wired via `td_one.tools.forecasting`: "
            f"{', '.join(via_tools) if via_tools else 'none (local m0 fallback)'}.",
            "- PIT ingest used the parquet fast path (DuckDB-native bulk load "
            "into the PIT schema); the `PITStore` row-wise API is too slow at "
            "this scale (~40 rows/s).",
        ]
        report_md.write_text("\n".join(lines) + "\n")
        print(f"      wrote {results_json}", flush=True)
        print(f"      wrote {report_md}", flush=True)

        print(f"DONE in {payload['total_runtime_s']}s", flush=True)
        return 0
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 -- top-level failure boundary: any
        # unexpected error must exit non-zero with a clear message.
        print(f"SCALE-UP FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
