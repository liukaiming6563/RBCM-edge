"""Population decoding utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_trial_feature_matrix(
    response_table: pd.DataFrame,
    label_column: str = "paradigm",
    value_column: str = "firing_rate_hz",
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert long response rows into trial-level population vectors."""
    key_cols = [
        col
        for col in ["run_id", "paradigm", "rep", "dir_idx", "step", "phase"]
        if col in response_table.columns
    ]
    features = response_table.pivot_table(
        index=key_cols,
        columns="unit_id",
        values=value_column,
        aggfunc="mean",
        fill_value=0.0,
    )
    labels = features.index.to_frame(index=False)[label_column]
    return features.reset_index(drop=True), labels.reset_index(drop=True)


def logistic_decoding_cv(
    features: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Run stratified cross-validated logistic decoding."""
    model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, features, labels, cv=cv, scoring="accuracy")
    return {"accuracy_mean": float(scores.mean()), "accuracy_std": float(scores.std())}
