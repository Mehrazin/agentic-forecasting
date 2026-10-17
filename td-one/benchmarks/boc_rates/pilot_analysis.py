"""BoC pilot part 2: Apply TD-One agent tools to the failures.

Where does the simple rule baseline fail? Can our tools diagnose it?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_pilot_data() -> pd.DataFrame:
    path = Path(__file__).parent / "pilot_data.json"
    return pd.read_json(path)


def main() -> None:
    df = load_pilot_data()
    print(f"Loaded {len(df)} meetings")
    
    # Recompute rule predictions
    def rule_predict(rm):
        if rm < -0.25:
            return -1
        elif rm > 0.25:
            return 1
        else:
            return 0
    
    df["rule_pred"] = df["rate_momentum"].apply(rule_predict)
    df["correct"] = df["rule_pred"] == df["decision"]
    
    # Where does it fail?
    failures = df[~df["correct"]]
    print(f"\nFailures: {len(failures)} / {len(df)} ({len(failures)/len(df):.1%})")
    print("\nFailure breakdown (actual -> predicted):")
    for _, row in failures.iterrows():
        print(f"  {row['meeting_date']}: actual={int(row['decision'])}, pred={int(row['rule_pred'])}, "
              f"rm={row['rate_momentum']:.2f}, ig={row['inflation_gap']:.2f}, um={row['unemployment_momentum']:.2f}")
    
    # Can our tools help? Let's try:
    # 1. Are failures clustered in time? (changepoint)
    # 2. Do failures correlate with inflation_gap or unemployment_momentum?
    
    print("\n--- Tool 1: Failure correlation with other features ---")
    for feat in ["inflation_gap", "unemployment_momentum"]:
        corr = df[feat].corr((~df["correct"]).astype(float))
        print(f"  {feat} vs failure: r={corr:.3f}")
    
    # 2. Among holds predicted as hikes (false positives), what's different?
    fp = df[(df["decision"] == 0) & (df["rule_pred"] == 1)]
    tn = df[(df["decision"] == 0) & (df["rule_pred"] == 0)]
    print(f"\n--- Tool 2: False positives (n={len(fp)}) vs true negatives (n={len(tn)}) ---")
    if len(fp) > 0 and len(tn) > 0:
        for feat in ["inflation_gap", "unemployment_momentum", "rate_momentum"]:
            print(f"  {feat}: FP mean={fp[feat].mean():.2f}, TN mean={tn[feat].mean():.2f}")
    
    # 3. Among hikes predicted as holds (false negatives), what's different?
    fn = df[(df["decision"] == 1) & (df["rule_pred"] == 0)]
    tp = df[(df["decision"] == 1) & (df["rule_pred"] == 1)]
    print(f"\n--- Tool 3: False negatives (n={len(fn)}) vs true positives (n={len(tp)}) ---")
    if len(fn) > 0 and len(tp) > 0:
        for feat in ["inflation_gap", "unemployment_momentum", "rate_momentum"]:
            print(f"  {feat}: FN mean={fn[feat].mean():.2f}, TP mean={tp[feat].mean():.2f}")
    
    # Key insight: can inflation_gap or unemployment_momentum rescue the failures?
    # Let's see if a simple 2-feature rule does better
    print("\n--- Tool 4: Can we improve with inflation_gap? ---")
    # If rate_momentum is borderline but inflation is high, maybe hold not hike?
    def improved_rule(row):
        rm, ig = row["rate_momentum"], row["inflation_gap"]
        # Original rule
        if rm < -0.25:
            base = -1
        elif rm > 0.25:
            base = 1
        else:
            base = 0
        # Refinement: if predicting hike but inflation is very low, maybe hold?
        # (BoC won't hike if inflation is below target)
        if base == 1 and ig < -0.5:
            return 0
        # If predicting hold but inflation very high and rm positive, maybe hike?
        if base == 0 and ig > 2.0 and rm > 0.1:
            return 1
        return base
    
    df["improved_pred"] = df.apply(improved_rule, axis=1)
    n = len(df)
    split = int(n * 0.7)
    test_df = df.iloc[split:]
    imp_acc = (test_df["improved_pred"] == test_df["decision"]).mean()
    rule_acc = (test_df["rule_pred"] == test_df["decision"]).mean()
    print(f"  Original rule accuracy: {rule_acc:.3f}")
    print(f"  Improved rule accuracy: {imp_acc:.3f}")
    print(f"  Gain: {imp_acc - rule_acc:+.3f}")
    
    # Save results
    results = {
        "n_meetings": len(df),
        "n_failures": len(failures),
        "failure_rate": float(len(failures) / len(df)),
        "rule_accuracy_test": float(rule_acc),
        "improved_accuracy_test": float(imp_acc),
        "improvement": float(imp_acc - rule_acc),
    }
    out_path = Path(__file__).parent / "pilot_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
