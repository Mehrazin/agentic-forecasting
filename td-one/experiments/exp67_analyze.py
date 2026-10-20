"""exp67 breaking-point analysis + plots.

Merges exp67_results_a.json and exp67_results_b.json (the two parallel real-TSFM
grid halves), prints per-model breaking-point tables, and writes degradation
curves + coverage plots to experiments/exp67_figs/.

Usage:
    .venv/bin/python experiments/exp67_analyze.py [--outdir experiments/exp67_figs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

EXP_DIR = Path(__file__).resolve().parent


def load_merged(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for p in paths:
        if not p.exists():
            print(f"  [skip] {p.name} not found")
            continue
        rows.extend(json.loads(p.read_text())["rows"])
    df = pd.DataFrame(rows)
    # Placebo ('none') is severity-invariant: one copy per (seed, origin, model),
    # otherwise positional pairing against shifted rows misaligns.
    plac_mask = df["shift"] == "none"
    df_plac = df[plac_mask].drop_duplicates(
        subset=["shift", "seed", "origin", "model"], keep="last"
    )
    df = pd.concat([df[~plac_mask], df_plac], ignore_index=True)
    # De-duplicate: same (shift, severity, seed, origin, model) from a rerun.
    df = df.drop_duplicates(
        subset=["shift", "severity", "seed", "origin", "model"], keep="last"
    )
    return df


def breaking_point_table(df: pd.DataFrame) -> pd.DataFrame:
    """Placebo-controlled breaking points.

    For each (shift, severity, model): at each (seed, origin), the shifted CRPS
    minus the placebo ('none' shift) CRPS at the SAME seed and origin. This
    controls for origin/context-length effects. Pairing is positional within
    (seed, origin_class): i-th origin, sorted.

    The pre-shift class doubles as a PIT/leakage check: shifted and placebo
    series are bit-identical before onset, so pre-class degradation must be ~0.
    """
    from scipy import stats

    out: list[dict] = []
    ok = df[df["status"] == "ok"]
    # S3's aggregate is built from different regional seeds than the placebo;
    # its valid control is the undiverged regional average ('S3ctrl').
    control_for = {"S3": "S3ctrl"}
    for (sh, sev, model), g in ok.groupby(["shift", "severity", "model"]):
        if sh in ("none", "S3ctrl"):
            continue
        ctl = ok[ok["shift"] == control_for.get(sh, "none")]
        # S3ctrl rows carry the S3 severities; 'none' is severity-invariant.
        if sh in control_for:
            ctl = ctl[ctl["severity"] == sev]
        row: dict = {"shift": sh, "severity": sev, "model": model}
        for oclass in ("pre", "straddle", "post"):
            diffs: list[float] = []
            for sd in sorted(g["seed"].unique()):
                gs = (
                    g[(g["seed"] == sd) & (g["origin_class"] == oclass)]
                    .sort_values("origin")
                )
                gp = ctl[
                    (ctl["seed"] == sd)
                    & (ctl["model"] == model)
                    & (ctl["origin_class"] == oclass)
                ].sort_values("origin")
                n = min(len(gs), len(gp))
                if n:
                    diffs.extend(
                        (gs["crps"].to_numpy()[:n] - gp["crps"].to_numpy()[:n]).tolist()
                    )
            d = np.asarray(diffs)
            if len(d) >= 4:
                _, p = stats.ttest_1samp(d, 0.0)
                base = ctl[
                    (ctl["model"] == model) & (ctl["origin_class"] == oclass)
                ]["crps"].mean()
                row.update(
                    {
                        f"{oclass}_n": len(d),
                        f"{oclass}_degrad_rel": (
                            float(d.mean() / base) if base else float("nan")
                        ),
                        f"{oclass}_p": float(p),
                    }
                )
        cov_s = g[g["origin_class"] == "straddle"]["coverage80"].mean()
        cov_p = ctl[(ctl["model"] == model) & (ctl["origin_class"] == "straddle")][
            "coverage80"
        ].mean()
        row.update({"cov_straddle_shifted": cov_s, "cov_straddle_placebo": cov_p})
        out.append(row)
    return pd.DataFrame(out).sort_values(["model", "shift", "severity"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(EXP_DIR / "exp67_figs"))
    ap.add_argument(
        "--inputs",
        default=",".join(["exp67_results_a.json", "exp67_results_b.json", "exp67_results.json"]),
    )
    args = ap.parse_args()
    if pd is None:
        raise SystemExit("pandas required")

    df = load_merged([EXP_DIR / n for n in args.inputs.split(",")])
    print(f"loaded {len(df)} rows, models={sorted(df['model'].unique())}")
    print(f"shifts={sorted(df['shift'].unique())} severities={sorted(df['severity'].unique())}")
    n_err = int((df["status"] != "ok").sum())
    print(f"errors: {n_err}/{len(df)}")
    if n_err:
        print(df[df["status"] != "ok"].groupby("model")["error"].agg(["count", "first"]))

    bp = breaking_point_table(df)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bp.to_csv(outdir / "breaking_points.csv", index=False)

    # Console: per model, per shift, the severity ladder with significance flags.
    # straddle = causal shift effect; pre = leakage check (must be ~0).
    for model in sorted(bp["model"].unique()):
        print(f"\n=== {model} (placebo-controlled; pre must be ~0) ===")
        sub = bp[bp["model"] == model]
        for sh in sorted(sub["shift"].unique()):
            row = sub[sub["shift"] == sh].sort_values("severity")
            cells = "  ".join(
                f"{r.severity:g}σ:pre={r.pre_degrad_rel:+.0%}"
                f" straddle={r.straddle_degrad_rel:+.0%}{'*' if r.straddle_p < 0.05 else ''}"
                f" post={r.post_degrad_rel:+.0%}{'*' if r.post_p < 0.05 else ''}"
                for r in row.itertuples()
            )
            print(f"  {sh}: {cells}")

    # Coverage: shifted vs placebo at straddle origins.
    print("\n=== 80% coverage at straddle origins (shifted vs placebo) ===")
    print(
        bp[["model", "shift", "severity", "cov_straddle_shifted", "cov_straddle_placebo"]]
        .to_string(index=False)
    )

    # Plots.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shifts = sorted(bp["shift"].unique())
        models = sorted(bp["model"].unique())
        fig, axes = plt.subplots(
            len(shifts), 1, figsize=(9, 3.2 * len(shifts)), sharex=True, squeeze=False
        )
        for ax, sh in zip(axes[:, 0], shifts, strict=True):
            for model in models:
                sub = bp[(bp["shift"] == sh) & (bp["model"] == model)].sort_values("severity")
                if len(sub):
                    ax.plot(
                        sub["severity"],
                        sub["straddle_degrad_rel"] * 100,
                        marker="o",
                        label=model,
                    )
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"{sh}: placebo-controlled CRPS degradation at straddle origins")
            ax.set_ylabel("degradation %")
            ax.legend(fontsize=8)
        axes[-1, 0].set_xlabel("severity (σ)")
        fig.tight_layout()
        fig.savefig(outdir / "breaking_point_curves.png", dpi=120)
        print(f"\nwrote {outdir / 'breaking_point_curves.png'}")
    except ImportError:
        print("[skip plots] matplotlib not installed")


if __name__ == "__main__":
    main()
