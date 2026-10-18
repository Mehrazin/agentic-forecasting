"""exp51: Kitchen sink — combine ALL positive-but-nonsignificant effects.

The proposed method (exp47/49) includes only statistically significant
components. This experiment tests whether the significance filter left real
gains on the table: add every *honestly combinable* positive-direction
component (see docs/NONSIGNIFICANT_POSITIVES.md) and compare vs proposed.

Components (all verified positive-direction, none overturned by stronger evidence):
  +tsfm   SMM only: TimesFM-2.5 as 5th expert (exp18b: -12.2% vs ETS, p=0.076)
  +macro  SMM only: ff_3m_chg + term_spread features (exp33: +1.21%, n.s.)
  +meta   all: meta-learner blend 0.5*w_router + 0.5*P_meta(model|features) (exp41)
  +gates  all: learned gate thresholds via grid search (exp50: p=0.32, n.s.)
  +lossreg all: loss-vector regime spec instead of hand rules (exp50: p=0.44)

Excluded with documented reasons (see NONSIGNIFICANT_POSITIVES.md):
  exp22 (needs live LLM), exp44 (killed by exp46), exp36 (diagnostic only),
  exp11/exp5/exp4 (already integrated), exp10 (no hierarchy), exp13 (was negative),
  exp17 (superseded by kill).

Design: ablations (each alone) + full kitchen sink, blocked 5-fold CV,
paired t-tests, vs the proposed baseline run on identical records/folds.

Usage:
    cd td-one && .venv/bin/python experiments/exp51_kitchen_sink.py [--quick]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import exp42_fforma as f42  # noqa: E402

from td_one.agent.proposed_method import (  # noqa: E402
    GATE_SHIFT,
    GATE_T_STAT,
    GATE_TOP_P,
    ForecastOutput,
    ProposedForecaster,
    RegimeSpec,
    generic_regime_spec,
    m4_regime_spec,
)
from td_one.agent.tools.data import series_features  # noqa: E402
from td_one.agent.tools.regimes import (  # noqa: E402
    loss_regime_classifier,
    loss_regime_spec,
)

MODELS_4 = ["naive", "m0", "ets", "arima"]
FEATURE_NAMES = list(f42.FEATURE_NAMES)
MACRO_FEATURES = ["ff_3m_chg", "term_spread"]
Z_SCORES = np.array(
    [
        -1.2815515655446004,
        -0.8416212335729143,
        -0.5244005127080409,
        -0.2533471031357997,
        0.0,
        0.2533471031357997,
        0.5244005127080409,
        0.8416212335729143,
        1.2815515655446004,
    ]
)


# ---------------------------------------------------------------------------
# Record generation (mirrors f42.generate_records exactly, keeps train window)
# ---------------------------------------------------------------------------


def generate_records_with_train(
    series_list: list[tuple[str, np.ndarray]],
    train_window: int,
    horizon: int,
    n_origins: int,
    dataset: str,
) -> list[dict]:
    """Mirror f42.generate_records exactly, plus keep the normalized train window."""
    records = []
    for sid, full in series_list:
        full = np.asarray(full, dtype=float)
        max_origin = len(full) - horizon
        origins = range(max_origin - n_origins + 1, max_origin + 1)
        for o in origins:
            train_raw = full[o - train_window : o]
            actual_raw = full[o : o + horizon]
            if len(train_raw) < train_window or len(actual_raw) < horizon:
                continue
            train, actual = f42._normalize(train_raw, actual_raw)
            feats = series_features(train).as_dict()
            feat_list = [float(feats[k]) for k in FEATURE_NAMES]
            quantiles: dict[str, list] = {}
            crps: dict[str, float] = {}
            for m, fn in f42.MODEL_FNS.items():
                try:
                    point, resid = fn(train, horizon)
                    resid_std = float(np.std(resid)) if len(resid) else 0.1
                    qs = f42._quantiles_from_point(point, resid_std, horizon)
                except Exception:  # noqa: BLE001
                    qs = np.tile(np.linspace(0.05, 0.95, 9)[:, None], (1, horizon))
                quantiles[m] = qs.tolist()
                crps[m] = f42._crps_quantile(actual, qs)
            records.append(
                {
                    "dataset": dataset,
                    "series_id": sid,
                    "origin": int(o),
                    "features": feat_list,
                    "quantiles": quantiles,
                    "actual": actual.tolist(),
                    "crps": crps,
                    "train": train.tolist(),
                }
            )
    return records


# ---------------------------------------------------------------------------
# +tsfm: TimesFM-2.5 quantiles for SMM records (exp18b mechanism)
# ---------------------------------------------------------------------------


def add_tsfm_to_smm(records: list[dict], horizon: int = 1) -> list[dict]:
    """Run TimesFM-2.5 (real weights) on each SMM record's normalized train.

    Native 9-level quantiles, clipped to [0,1] (the pipeline's normalized
    space). No LogScaleCalibrator: normalization already handles the scale
    issue it was built for (raw SMM ~ 1e-3).
    """
    from td_one.models.tsfm import TimesFMWrapper

    # disk cache: TSFM inference is ~15s/origin; reuse across runs
    cache_path = ROOT / "experiments" / "exp51_tsfm_cache.json"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        try:
            with open(cache_path) as fh:
                cache = json.load(fh)
            print(f"[exp51] tsfm cache: {len(cache)} entries", flush=True)
        except Exception:  # noqa: BLE001
            cache = {}

    wdir = ROOT / "models" / "weights" / "timesfm25"
    tfm = TimesFMWrapper(use_mock=False, weights_dir=str(wdir))
    for n, r in enumerate(records, start=1):
        key = f"{r['series_id']}_{r['origin']}"
        if key in cache:
            q = np.asarray(cache[key]["quantiles"], dtype=float)
        else:
            train = np.asarray(r["train"], dtype=float)
            try:
                _, qv = tfm.forecast_quantiles(train, horizon=horizon, seed=42)
                q = np.asarray(qv, dtype=float)
                if q.shape != (9, horizon):
                    q = np.tile(np.linspace(0.05, 0.95, 9)[:, None], (1, horizon))
                q = np.clip(q, 0.0, 1.0)
                q = np.maximum.accumulate(q, axis=0)
            except Exception:  # noqa: BLE001
                q = np.tile(np.linspace(0.05, 0.95, 9)[:, None], (1, horizon))
            cache[key] = {"quantiles": q.tolist()}
        actual = np.asarray(r["actual"], dtype=float)
        r["quantiles"]["tsfm_cal"] = q.tolist()
        r["crps"]["tsfm_cal"] = f42._crps_quantile(actual, q)
        if n % 10 == 0:
            print(f"[exp51] tsfm {n}/{len(records)}", flush=True)
    tfm.unload()
    with contextlib.suppress(Exception), open(cache_path, "w") as fh:
        json.dump(cache, fh)
    return records


# ---------------------------------------------------------------------------
# +macro: PIT-safe FRED macro features for SMM records (exp33 mechanism)
# ---------------------------------------------------------------------------


def add_macro_to_smm(records: list[dict]) -> tuple[list[dict], list[str]]:
    """Append ff_3m_chg + term_spread (month-end FRED, strictly <= origin-1).

    Mirrors benchmarks/smm/exp33_agent_features.py build_features PIT logic.
    """
    fred_root = ROOT / "data" / "fred"
    frames = {}
    for fid in ("FEDFUNDS", "T10Y2Y"):
        p = fred_root / f"{fid}.parquet"
        d = pd.read_parquet(p)
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date").drop_duplicates("date", keep="last")
        m = d.set_index("date")["value"].resample("ME").last().dropna()
        frames[fid] = m
    ff = frames["FEDFUNDS"]
    ts = frames["T10Y2Y"]

    smm_df = pd.read_csv(ROOT / "data" / "freddie" / "smm_monthly_2021_2026.csv")
    # date col is 'period' as YYYYMM ints (e.g. 202101); fall back to any
    # date-like column
    if "period" in smm_df.columns:
        smm_dates = pd.to_datetime(
            smm_df["period"].astype(str), format="%Y%m"
        ) + pd.offsets.MonthEnd(0)
    else:
        date_col = next(c for c in smm_df.columns if "date" in c.lower() or "month" in c.lower())
        smm_dates = pd.to_datetime(smm_df[date_col]).dt.to_period("M").dt.to_timestamp("M")

    feat_names = FEATURE_NAMES + MACRO_FEATURES
    for r in records:
        o = int(r["origin"])  # index into the 63-month array
        # last fully-known month = o - 1 -> find its month-end in FRED frames
        m_end = smm_dates.iloc[o - 1] if o - 1 < len(smm_dates) else smm_dates.iloc[-1]
        # ffill: last observation on/before month-end
        ff_m = ff.loc[:m_end].iloc[-1] if len(ff.loc[:m_end]) else np.nan
        # 3m change needs the value 3 months earlier
        m_3 = m_end - pd.DateOffset(months=3)
        ff_m3 = ff.loc[:m_3].iloc[-1] if len(ff.loc[:m_3]) else np.nan
        ts_m = ts.loc[:m_end].iloc[-1] if len(ts.loc[:m_end]) else np.nan
        r["features"] = list(r["features"]) + [
            float(ff_m - ff_m3) if np.isfinite(ff_m) and np.isfinite(ff_m3) else 0.0,
            float(ts_m) if np.isfinite(ts_m) else 0.0,
        ]
    return records, feat_names


# ---------------------------------------------------------------------------
# KitchenSinkForecaster: ProposedForecaster + meta-learner blend (exp41)
# ---------------------------------------------------------------------------


@dataclass
class KitchenSinkForecaster(ProposedForecaster):
    """Proposed method + meta-learner blend (exp41 mechanism).

    w(m) = (1-a) * w_router(m) + a * P_meta(m | features), a=0.5.
    w_router(m) = sum_reg P(reg) * [expert(reg) == m] — identical to the
    parent's soft-regime combination when a=0, so meta_blend=False
    reproduces the proposed method exactly.
    """

    meta_blend: bool = False
    meta_alpha: float = 0.5
    _meta_clf: LogisticRegression | None = field(default=None, repr=False)

    def fit(self, records: list[dict], feature_names: list[str]) -> KitchenSinkForecaster:
        super().fit(records, feature_names)
        if self.meta_blend:
            x = np.array([r["features"] for r in records], dtype=float)
            # best model per record (train CRPS) — same labels exp41 used
            y = np.array([min(self.models, key=lambda m: float(r["crps"][m])) for r in records])
            if len(np.unique(y)) < 2:
                # degenerate: single best model on train -> no meta signal
                self._meta_clf = None
                return self
            # drop models never best (keeps the multinomial well-posed)
            self._meta_clf = LogisticRegression(max_iter=1000, C=1.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._meta_clf.fit(x, y)
        return self

    def predict(
        self,
        train: np.ndarray,
        model_quantiles: dict[str, np.ndarray],
        feats: dict,
        feature_names: list[str],
        news_features: dict | None = None,
        llm_reading: dict | None = None,
        t_stat: float = 0.0,
        shift: float = 0.0,
    ) -> ForecastOutput:
        assert self._fitted, "call fit() first"
        spec = self.regime_spec
        probs, news_used, llm_used = self._regime_probs(
            feats, feature_names, news_features, llm_reading
        )

        # "do nothing" gate (same as parent)
        top_p = max(probs.values())
        gated, gate_reason = False, ""
        weak = (
            top_p < self.gate_top_p
            and abs(t_stat) < self.gate_t_stat
            and abs(shift) < self.gate_shift
        )
        if weak:
            gated, gate_reason = True, "uncertain_router_weak_evidence"

        qm = {m: np.asarray(v, dtype=float) for m, v in model_quantiles.items()}
        q0 = next(iter(qm.values()))
        hh = q0.shape[1]
        combined = np.zeros((9, hh))
        if gated:
            combined = qm[self._global_best].copy()
            blend_note = ""
        else:
            # router-implied model weights
            w_router = dict.fromkeys(self.models, 0.0)
            for reg, p in probs.items():
                w_router[self._expert_per_regime[reg]] += p
            if self.meta_blend and self._meta_clf is not None:
                x = np.array([[feats[k] for k in feature_names]], dtype=float)
                p_meta = self._meta_clf.predict_proba(x)[0]
                classes = list(self._meta_clf.classes_)
                w_meta = dict.fromkeys(self.models, 0.0)
                for i, c in enumerate(classes):
                    if c in w_meta:
                        w_meta[c] = float(p_meta[i])
                a = self.meta_alpha
                w = {m: (1 - a) * w_router[m] + a * w_meta[m] for m in self.models}
                blend_note = " | meta-blend"
            else:
                w = w_router
                blend_note = ""
            for m in self.models:
                combined += w[m] * qm[m]

        # conformal calibration (same as parent, exp5)
        hist_vol = float(np.std(np.diff(np.asarray(train, dtype=float)))) or 1e-6
        width_floor = 0.5 * hist_vol
        med = combined[4]
        half = np.maximum((combined[8] - combined[0]) / 2.0, width_floor)
        for i, z in enumerate(Z_SCORES):
            combined[i] = med + z * half / Z_SCORES[8]

        active_expert = self._expert_per_regime[max(probs, key=probs.get)]
        trust = "EASY" if top_p >= 0.7 else "HARD"
        explanation = (
            f"KitchenSink({spec.name}): "
            + ", ".join(f"{r}={p:.2f}" for r, p in sorted(probs.items()))
            + f" | active: {active_expert} | trust: {trust}"
            + blend_note
            + (f" | GATED ({gate_reason})" if gated else "")
        )
        return ForecastOutput(
            quantiles=combined,
            regime_probs=probs,
            expert_per_regime=dict(self._expert_per_regime),
            active_expert=active_expert,
            trust=trust,
            trust_margin=0.0,
            explanation=explanation,
            gated=gated,
            gate_reason=gate_reason,
            news_used=news_used,
            llm_used=llm_used,
        )


# ---------------------------------------------------------------------------
# +gates: learned gate thresholds (exp50 mechanism, models-parametric)
# ---------------------------------------------------------------------------


def learn_gates(
    records: list[dict],
    feature_names: list[str],
    spec: RegimeSpec,
    models: list[str],
    meta_blend: bool = False,
) -> dict[str, float]:
    """Grid-search gate thresholds with 2-fold CV (like learn_gate_thresholds)."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    idx = np.arange(len(records))
    best, best_score = None, float("inf")
    for top_p in (0.5, 0.6, 0.7):
        for t_thr in (2.5, 3.0, 3.5, 4.0):
            for s_thr in (0.02, 0.03, 0.04, 0.05):
                fold_scores = []
                for tr, va in kf.split(idx):
                    f = KitchenSinkForecaster(
                        regime_spec=spec,
                        models=models,
                        meta_blend=meta_blend,
                        gate_top_p=top_p,
                        gate_t_stat=t_thr,
                        gate_shift=s_thr,
                    )
                    f.fit([records[i] for i in tr], feature_names)
                    sc = []
                    for i in va:
                        r = records[i]
                        out = f.predict(
                            train=np.asarray(r["train"], dtype=float),
                            model_quantiles=r["quantiles"],
                            feats=dict(zip(feature_names, r["features"], strict=True)),
                            feature_names=feature_names,
                        )
                        actual = np.asarray(r["actual"], dtype=float)
                        q = out.quantiles
                        sc.append(
                            float(
                                np.mean(
                                    [
                                        f42._crps_quantile(q[:, h], actual[h])
                                        for h in range(q.shape[1])
                                    ]
                                )
                            )
                        )
                    fold_scores.append(float(np.mean(sc)))
                ms = float(np.mean(fold_scores))
                if ms < best_score:
                    best_score, best = (
                        ms,
                        {
                            "gate_top_p": top_p,
                            "gate_t_stat": t_thr,
                            "gate_shift": s_thr,
                        },
                    )
    return best or {"gate_top_p": GATE_TOP_P, "gate_t_stat": GATE_T_STAT, "gate_shift": GATE_SHIFT}


# ---------------------------------------------------------------------------
# +lossreg: loss-vector regime spec adapter (exp50 mechanism)
# ---------------------------------------------------------------------------


def loss_regime_spec_fn(
    train_records: list[dict], models: list[str], feature_names: list[str]
) -> RegimeSpec:
    """Build a RegimeSpec from loss-vector clustering (fit on train only)."""
    ls = loss_regime_spec(train_records, models, n_regimes=3, seed=42)
    clf = loss_regime_classifier(train_records, feature_names, ls, seed=42)
    regimes = [f"lossreg_{i}" for i in range(3)]

    def label(feats: dict) -> str:
        x = np.array([[float(feats[k]) for k in feature_names]], dtype=float)
        return regimes[int(clf.predict(x)[0])]

    return RegimeSpec(
        name="lossreg",
        regimes=regimes,
        label_fn=label,
        description="KMeans on per-model loss vectors (exp50)",
    )


# ---------------------------------------------------------------------------
# CV harness: variants vs proposed on identical records/folds
# ---------------------------------------------------------------------------

# variant -> config
VARIANTS: dict[str, dict] = {
    "proposed": {},  # baseline
    "meta": {"meta_blend": True},
    "gates": {"learned_gates": True},
    "lossreg": {"loss_regimes": True},
}


def _folds(name: str, records: list[dict], n_folds: int = 5, seed: int = 0):
    if name in ("m4", "synthetic"):
        return f42.kfold_by_series(records, n_folds=n_folds, seed=seed)
    order = sorted(range(len(records)), key=lambda i: records[i]["origin"])
    folds = []
    chunk = len(order) // n_folds
    for f in range(n_folds):
        te = order[f * chunk : (f + 1) * chunk if f < n_folds - 1 else len(order)]
        tr = [i for i in order if i not in set(te)]
        folds.append((tr, te))
    return folds


def run_variant_cv(
    name: str,
    records: list[dict],
    base_spec_fn,
    variant: dict,
    models: list[str],
    feature_names: list[str],
) -> dict:
    """Blocked CV for one variant. Returns per-record CRPS + diagnostics."""
    folds = _folds(name, records)
    crps, gate_rates = [], []
    for tr_idx, te_idx in folds:
        tr = [records[i] for i in tr_idx]
        te = [records[i] for i in te_idx]
        spec = (
            loss_regime_spec_fn(tr, models, feature_names)
            if variant.get("loss_regimes")
            else base_spec_fn()
        )
        fc = KitchenSinkForecaster(
            regime_spec=spec,
            models=models,
            meta_blend=bool(variant.get("meta_blend")),
        )
        if variant.get("learned_gates"):
            g = learn_gates(tr, feature_names, spec, models, fc.meta_blend)
            fc.gate_top_p, fc.gate_t_stat, fc.gate_shift = (
                g["gate_top_p"],
                g["gate_t_stat"],
                g["gate_shift"],
            )
        fc.fit(tr, feature_names)
        for r in te:
            actual = np.asarray(r["actual"], dtype=float)
            out = fc.predict(
                train=np.asarray(r["train"], dtype=float),
                model_quantiles=r["quantiles"],
                feats=dict(zip(feature_names, r["features"], strict=True)),
                feature_names=feature_names,
            )
            crps.append(f42._crps_quantile(actual, out.quantiles))
            gate_rates.append(1.0 if out.gated else 0.0)
    return {
        "crps": np.array(crps),
        "gate_fire_rate": float(np.mean(gate_rates)),
        "n": len(records),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    n_m4 = 20 if args.quick else 100
    synth_n = 12 if args.quick else 40
    print("[exp51] generating records (identical pipeline to exp49)...", flush=True)
    m4 = generate_records_with_train(f42.load_m4(n=n_m4), 48, 6, 6, "m4")
    synth = generate_records_with_train(
        f42.load_synthetic(n_series=synth_n), 30, 1, 12, "synthetic"
    )
    smm = generate_records_with_train([f42.load_smm()[0]], 36, 1, 50, "smm")
    fcpi = generate_records_with_train([f42.load_food_cpi()[0]], 48, 1, 24, "food_cpi")
    print(
        f"[exp51] records: m4={len(m4)} synth={len(synth)} smm={len(smm)} fcpi={len(fcpi)}",
        flush=True,
    )

    # SMM-only components
    smm_variants = dict(VARIANTS)
    smm_models = list(MODELS_4)
    smm_feat_names = list(FEATURE_NAMES)
    if not args.quick:
        print("[exp51] +tsfm: TimesFM-2.5 inference on SMM origins...", flush=True)
        smm = add_tsfm_to_smm(smm)
        smm_models = MODELS_4 + ["tsfm_cal"]
        smm_variants["tsfm"] = {"extra_model": "tsfm_cal"}
        print("[exp51] +macro: FRED macro features for SMM...", flush=True)
        smm, smm_feat_names = add_macro_to_smm(smm)
        smm_variants["macro"] = {"extra_features": True}
    smm_variants["sink"] = {
        "extra_model": None if args.quick else "tsfm_cal",
        "extra_features": not args.quick,
        "meta_blend": True,
        "learned_gates": True,
        "loss_regimes": True,
    }

    datasets = {
        "m4": (
            m4,
            m4_regime_spec,
            MODELS_4,
            FEATURE_NAMES,
            dict(VARIANTS, sink={"meta_blend": True, "learned_gates": True, "loss_regimes": True}),
        ),
        "synthetic": (
            synth,
            m4_regime_spec,
            MODELS_4,
            FEATURE_NAMES,
            dict(VARIANTS, sink={"meta_blend": True, "learned_gates": True, "loss_regimes": True}),
        ),
        "food_cpi": (
            fcpi,
            generic_regime_spec,
            MODELS_4,
            FEATURE_NAMES,
            dict(VARIANTS, sink={"meta_blend": True, "learned_gates": True, "loss_regimes": True}),
        ),
        "smm": (smm, generic_regime_spec, smm_models, smm_feat_names, smm_variants),
    }

    results: dict = {}
    for name, (records, spec_fn, models, feat_names, variants) in datasets.items():
        print(f"[exp51] {name}: {len(variants)} variants x 5 folds...", flush=True)
        # NOTE: +tsfm/+macro change the shared record objects (models/features).
        # Run order: ablations that need the 4-model/10-feat records FIRST is
        # impossible once mutated — so run 'proposed/meta/gates/lossreg' on
        # pristine copies, and tsfm/macro/sink on the augmented records.
        # Simplest honest approach: deep-copy records per variant group.
        import copy

        variant_out: dict[str, dict] = {}
        # group 1: base records (4 models, 10 feats) — strip augmentations
        base_records = []
        for r in records:
            rb = copy.deepcopy(r)
            rb["quantiles"] = {m: rb["quantiles"][m] for m in MODELS_4}
            rb["crps"] = {m: rb["crps"][m] for m in MODELS_4}
            rb["features"] = list(rb["features"][: len(FEATURE_NAMES)])
            base_records.append(rb)
        for vname in ("proposed", "meta", "gates", "lossreg"):
            if vname not in variants:
                continue
            cfg = variants[vname]
            variant_out[vname] = run_variant_cv(
                name, base_records, spec_fn, cfg, MODELS_4, FEATURE_NAMES
            )
        # group 2: augmented records (SMM: 5 models, 12 feats)
        for vname in ("tsfm", "macro", "sink"):
            if vname not in variants:
                continue
            cfg = variants[vname]
            use_models = models
            use_feats = feat_names
            if vname == "tsfm":
                # tsfm only: 5 models, 10 feats
                recs = []
                for r in records:
                    rb = copy.deepcopy(r)
                    rb["quantiles"] = {m: rb["quantiles"][m] for m in MODELS_4 + ["tsfm_cal"]}
                    rb["crps"] = {m: rb["crps"][m] for m in MODELS_4 + ["tsfm_cal"]}
                    rb["features"] = list(rb["features"][: len(FEATURE_NAMES)])
                    recs.append(rb)
                use_models, use_feats = MODELS_4 + ["tsfm_cal"], FEATURE_NAMES
            elif vname == "macro":
                # macro only: 4 models, 12 feats
                recs = []
                for r in records:
                    rb = copy.deepcopy(r)
                    rb["quantiles"] = {m: rb["quantiles"][m] for m in MODELS_4}
                    rb["crps"] = {m: rb["crps"][m] for m in MODELS_4}
                    recs.append(rb)
                use_models, use_feats = MODELS_4, feat_names
            else:  # sink: everything
                recs, use_models, use_feats = records, models, feat_names
            # strip variant keys that aren't forecaster config
            fcfg = {
                k: v for k, v in cfg.items() if k in ("meta_blend", "learned_gates", "loss_regimes")
            }
            variant_out[vname] = run_variant_cv(name, recs, spec_fn, fcfg, use_models, use_feats)

        # paired comparisons vs proposed
        prop = variant_out["proposed"]["crps"]
        summary = {}
        for vname, vout in variant_out.items():
            c = vout["crps"]
            if vname == "proposed":
                summary[vname] = {
                    "mean_crps": float(np.mean(c)),
                    "gate_fire_rate": vout["gate_fire_rate"],
                    "n": vout["n"],
                }
            else:
                t, p = ttest_rel(c, prop)
                summary[vname] = {
                    "mean_crps": float(np.mean(c)),
                    "vs_proposed": {
                        "mean_diff": float(np.mean(c - prop)),
                        "t": float(t),
                        "p": float(p),
                        "win_rate": float(np.mean(c < prop)),
                    },
                    "gate_fire_rate": vout["gate_fire_rate"],
                    "n": vout["n"],
                }
        results[name] = summary
        for vname, s in summary.items():
            if vname == "proposed":
                print(f"[exp51]   {name}/{vname}: {s['mean_crps']:.4f}", flush=True)
            else:
                vp = s["vs_proposed"]
                print(
                    f"[exp51]   {name}/{vname}: {s['mean_crps']:.4f} "
                    f"diff={vp['mean_diff']:+.5f} p={vp['p']:.4g} win={vp['win_rate']:.2f}",
                    flush=True,
                )

    out = {
        "date": datetime.now(UTC).isoformat(),
        "quick": args.quick,
        "elapsed_s": time.time() - t0,
        "components": {
            "tsfm": "exp18b TimesFM-2.5 5th expert (SMM only)",
            "macro": "exp33 ff_3m_chg + term_spread (SMM only)",
            "meta": "exp41 meta-learner blend",
            "gates": "exp50 learned gate thresholds",
            "lossreg": "exp50 loss-vector regimes",
            "sink": "all applicable combined",
        },
        "results": results,
    }
    with open(ROOT / "experiments" / "exp51_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[exp51] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
