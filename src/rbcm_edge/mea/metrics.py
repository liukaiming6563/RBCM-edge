"""MEA response metrics used by the RBCM paper analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def modulation_index(
    response_context: float | np.ndarray,
    response_single: float | np.ndarray,
    epsilon: float = 1e-6,
) -> float | np.ndarray:
    """Compute signed boundary-context modulation index."""
    return (response_context - response_single) / (
        response_context + response_single + epsilon
    )


def add_modulation_index_columns(
    matched: pd.DataFrame,
    context_col: str = "response_context",
    single_col: str = "response_single",
    output_col: str = "modulation_index",
) -> pd.DataFrame:
    """Add an MI column to a table with matched context and single responses."""
    matched = matched.copy()
    matched[output_col] = modulation_index(matched[context_col], matched[single_col])
    return matched


def summarize_response_table(response_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize firing rates by run, paradigm, direction, phase, and step."""
    group_cols = [
        col
        for col in ["run_id", "paradigm", "dir_idx", "dir_name_en", "phase", "step"]
        if col in response_table.columns
    ]
    return (
        response_table.groupby(group_cols, dropna=False)["firing_rate_hz"]
        .agg(["mean", "std", "sem", "count"])
        .reset_index()
    )


def preferred_direction_index(direction_responses: np.ndarray) -> float:
    """Compute a simple direction selectivity index from direction responses."""
    responses = np.asarray(direction_responses, dtype=float)
    if responses.size == 0 or np.nansum(responses) <= 0:
        return np.nan
    preferred = np.nanmax(responses)
    null = np.nanmin(responses)
    return float((preferred - null) / (preferred + null + 1e-6))
