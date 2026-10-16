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
        "--skip-ingest",
        action="store_true",
        help="Skip the PIT store ingest step.",
    )
    args = p.parse_args(argv)
    if args.n_customers <= 0 or args.n_months <= 0:
        p.error("--n-customers and --n-months must be positive")
    return args


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


def _load_forecast_fns(
    model_names: list[str], seed: int
) -> tuple[dict[str, ForecastFn], list[str]]:
    """Wire model names to ``forecast_fn`` callables.

    ``td_one.tools.forecasting`` is imported here (not at module import)
    so the pipeline degrades gracefully while the tools package is empty.
    Each model is wired as ``functools.partial(forecast, name)``, matching
    the runner's ``(train_panel, origin) -> {level: rate}`` contract.

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

    fns: dict[str, ForecastFn] = {}
    via_tools: list[str] = []
    for name in model_names:
        if tools_mod is not None and name in supported:
            fns[name] = partial(tools_mod.forecast, name)
            via_tools.append(name)
            print(f"[models] wired {name!r} via td_one.tools.forecasting")
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
        fns, via_tools = _load_forecast_fns(model_names, args.seed)
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
            f"seed={args.seed}, models={','.join(model_names)}.",
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
            "- **Coverage is below nominal for every model** (target 0.80)."
            " M0 undercovers and the mocks undercover; interval calibration"
            " remains open work at this sample size.",
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
