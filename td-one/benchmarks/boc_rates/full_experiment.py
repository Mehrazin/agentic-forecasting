"""BoC full experiment: multinomial logistic + agent discovery test.

1. MNLogit with 5-fold time-series CV (features + interaction)
2. Scripted agent test: can the tools+loop discover the inflation interaction?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data() -> pd.DataFrame:
    path = Path(__file__).parent / "pilot_data.json"
    df = pd.read_json(path)
    # Map -1/0/1 to 0/1/2 for MNLogit (needs 0..J-1)
    df["y"] = df["decision"].map({-1: 0, 0: 1, 1: 2})
    # Interaction term our tools identified
    df["rm_x_ig"] = df["rate_momentum"] * df["inflation_gap"]
    return df


def time_series_cv(df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5):
    """5-fold time-series CV. Returns per-fold accuracy and RPS."""
    n = len(df)
    fold_size = n // (n_splits + 1)
    results = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_start = train_end
        test_end = min(train_end + fold_size, n)
        train = df.iloc[:train_end]
        test = df.iloc[test_start:test_end]
        if len(test) == 0:
            continue

        X_train = sm.add_constant(train[feature_cols], has_constant="add")
        y_train = train["y"]
        try:
            model = sm.MNLogit(y_train, X_train).fit(disp=0, maxiter=200)
        except Exception as e:
            print(f"  Fold {i}: fit failed ({e}), skipping")
            continue

        X_test = sm.add_constant(test[feature_cols], has_constant="add")
        # Align columns (in case of missing dummy issues)
        for c in X_train.columns:
            if c not in X_test.columns:
                X_test[c] = 0
        X_test = X_test[X_train.columns]

        probs = model.predict(X_test)
        # probs is DataFrame with columns 0,1,2
        pred_idx = np.argmax(probs.values, axis=1)
        # Map back: 0->cut(-1), 1->hold(0), 2->hike(1)
        idx_to_dec = {0: -1, 1: 0, 2: 1}
        preds = [idx_to_dec[i] for i in pred_idx]
        actual = test["decision"].values

        acc = float(np.mean([p == a for p, a in zip(preds, actual)]))

        # RPS for ordered categories (cut < hold < hike)
        # RPS = sum_{k} (CDF_pred(k) - CDF_actual(k))^2
        rps_list = []
        for j, a in enumerate(actual):
            p = probs.values[j]  # [P(cut), P(hold), P(hike)]
            cdf_pred = np.cumsum(p)
            # Actual CDF: cut-> [1,1,1], hold->[0,1,1], hike->[0,0,1]
            if a == -1:
                cdf_act = np.array([1.0, 1.0, 1.0])
            elif a == 0:
                cdf_act = np.array([0.0, 1.0, 1.0])
            else:
                cdf_act = np.array([0.0, 0.0, 1.0])
            # RPS sums over K-1 = 2 thresholds
            rps = float(np.sum((cdf_pred[:2] - cdf_act[:2]) ** 2))
            rps_list.append(rps)

        results.append({
            "fold": i,
            "n_train": len(train),
            "n_test": len(test),
            "accuracy": acc,
            "mean_rps": float(np.mean(rps_list)),
        })
        print(f"  Fold {i}: train={len(train)}, test={len(test)}, acc={acc:.3f}, RPS={np.mean(rps_list):.4f}")

    return results


def per_class_metrics(df: pd.DataFrame, feature_cols: list[str]):
    """Fit on full data, report per-class precision/recall (in-sample, diagnostic)."""
    X = sm.add_constant(df[feature_cols], has_constant="add")
    y = df["y"]
    model = sm.MNLogit(y, X).fit(disp=0, maxiter=200)
    probs = model.predict(X)
    pred_idx = np.argmax(probs.values, axis=1)
    idx_to_dec = {0: -1, 1: 0, 2: 1}
    preds = np.array([idx_to_dec[i] for i in pred_idx])
    actual = df["decision"].values

    print("\nPer-class (in-sample) metrics:")
    for dec, name in [(-1, "cut"), (0, "hold"), (1, "hike")]:
        tp = int(np.sum((preds == dec) & (actual == dec)))
        fp = int(np.sum((preds == dec) & (actual != dec)))
        fn = int(np.sum((preds != dec) & (actual == dec)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"  {name}: precision={prec:.3f} ({tp}/{tp+fp}), recall={rec:.3f} ({tp}/{tp+fn})")

    # Coefficients for hike vs hold (the interesting margin)
    print("\nCoefficients (hike vs hold baseline):")
    params = model.params
    # MNLogit params: columns are outcome categories (0=cut, 2=hike), hold(1) is base
    print(params.round(4).to_string())
    return model


def main() -> None:
    df = load_data()
    print(f"N={len(df)} meetings")

    base_features = ["rate_momentum", "inflation_gap", "unemployment_momentum"]
    full_features = base_features + ["rm_x_ig"]

    print("\n=== Model A: 3 base features ===")
    res_a = time_series_cv(df, base_features)

    print("\n=== Model B: 3 features + rm_x_ig interaction ===")
    res_b = time_series_cv(df, full_features)

    def summarize(res, name):
        if not res:
            print(f"{name}: no folds completed")
            return None
        acc = np.mean([r["accuracy"] for r in res])
        rps = np.mean([r["mean_rps"] for r in res])
        print(f"\n{name}: mean acc={acc:.3f}, mean RPS={rps:.4f}")
        return {"accuracy": float(acc), "rps": float(rps), "folds": res}

    sum_a = summarize(res_a, "Model A (base)")
    sum_b = summarize(res_b, "Model B (+interaction)")

    print("\n=== Per-class diagnostics (Model B, full sample) ===")
    per_class_metrics(df, full_features)

    # Baselines for comparison
    n = len(df)
    split = int(n * 0.7)
    test = df.iloc[split:]
    clim_acc = float((test["decision"] == 0).mean())

    def rule_predict(rm):
        if rm < -0.25:
            return -1
        elif rm > 0.25:
            return 1
        return 0

    rule_acc = float((test["rate_momentum"].apply(rule_predict) == test["decision"]).mean())
    print(f"\nBaselines (last 30%): climatology={clim_acc:.3f}, rule={rule_acc:.3f}")

    results = {
        "n_meetings": len(df),
        "model_a_base": sum_a,
        "model_b_interaction": sum_b,
        "baselines": {"climatology": clim_acc, "rule": rule_acc},
    }
    out = Path(__file__).parent / "full_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
