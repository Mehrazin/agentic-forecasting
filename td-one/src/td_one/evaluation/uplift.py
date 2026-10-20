"""T-learner uplift for heterogeneous treatment response.

The deterministic grouper clustered on *raw features* and failed to
recover the simulator's planted cohorts (ARI 0.010, exp66d) because it
clusters on feature similarity, not attrition relevance. The planted
cohorts respond *differently to treatments*: the ``low_rate`` cohort
is all fixed-and-cheap (rate-locked), ``family_squeeze`` is
payment-stressed (offer-sensitive), etc. A T-learner estimates
borrower-level uplift — ``P(switch | treated, X) - P(switch |
control, X)`` — and segments borrowers by *heterogeneous response*,
which should cut closer to the DGP's causal joints than raw-feature
clustering.

Treatment here: variable-rate exposure (``rate_type == 'variable'``)
vs fixed. Variable borrowers are exposed to rate volatility; fixed
borrowers are not. This is a *descriptive* heterogeneity-revealing
segmentation, not a causal identification — variable vs fixed is
self-selected and confounded, so uplift scores must not be read as
causal effects. Say so in any report that uses them.

PIT: the two response models are fit on training-period rows and the
training-window switching label only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


@dataclass(frozen=True)
class UpliftFit:
    """Fitted T-learner: two response models + training metadata."""

    model_treated: GradientBoostingClassifier
    model_control: GradientBoostingClassifier
    treatment_rate: float
    n_treated: int
    n_control: int
    seed: int


def variable_rate_treatment(df: pd.DataFrame) -> np.ndarray:
    """Treatment indicator: 1 if variable rate (rate-volatile), else 0.

    PIT-safe: ``rate_type`` is a training-period snapshot attribute.
    """
    return (df["rate_type"].astype(str) == "variable").to_numpy(dtype=int)


def fit_t_learner(
    x: pd.DataFrame,
    treatment: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> UpliftFit:
    """Fit a T-learner: one GBM per treatment arm.

    Raises ``ValueError`` if either arm has fewer than 50 rows (the
    response model would be noise).
    """
    treatment = np.asarray(treatment, dtype=int)
    y = np.asarray(y, dtype=float)
    x_t, y_t = x.iloc[treatment == 1], y[treatment == 1]
    x_c, y_c = x.iloc[treatment == 0], y[treatment == 0]
    if len(x_t) < 50 or len(x_c) < 50:
        raise ValueError(
            f"T-learner needs >= 50 rows per arm, got {len(x_t)} treated / {len(x_c)} control"
        )
    model_treated = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed
    ).fit(x_t, y_t)
    model_control = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed + 1
    ).fit(x_c, y_c)
    return UpliftFit(
        model_treated=model_treated,
        model_control=model_control,
        treatment_rate=float(treatment.mean()),
        n_treated=int(len(x_t)),
        n_control=int(len(x_c)),
        seed=seed,
    )


def uplift_scores(fit: UpliftFit, x: pd.DataFrame) -> np.ndarray:
    """Borrower-level uplift: P(switch|treated,X) - P(switch|control,X)."""
    p_t = fit.model_treated.predict_proba(x)[:, 1]
    p_c = fit.model_control.predict_proba(x)[:, 1]
    return np.asarray(p_t - p_c, dtype=float)


def uplift_deciles(scores: np.ndarray, n_deciles: int = 10) -> np.ndarray:
    """Segment borrowers into uplift deciles (0 = lowest uplift).

    Deterministic: ``pandas.qcut`` on ranks; ties broken by order.
    """
    scores = np.asarray(scores, dtype=float)
    ranks = pd.Series(scores).rank(method="first")
    labels = pd.qcut(ranks, q=n_deciles, labels=False, duplicates="drop")
    return np.asarray(labels, dtype=int)


def decile_cohort_table(deciles: np.ndarray, cohort_labels: list[str]) -> pd.DataFrame:
    """Crosstab of uplift decile x planted cohort (counts)."""
    return pd.crosstab(
        pd.Series(deciles, name="uplift_decile"),
        pd.Series(cohort_labels, name="cohort"),
    )
