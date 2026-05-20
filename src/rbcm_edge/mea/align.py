"""Spike-event alignment and response table construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def count_spikes_in_window(spike_times_s: np.ndarray, start_s: float, end_s: float) -> int:
    """Count spikes in a half-open time window `[start_s, end_s)`."""
    left = np.searchsorted(spike_times_s, start_s, side="left")
    right = np.searchsorted(spike_times_s, end_s, side="left")
    return int(right - left)


def build_response_table(
    events: pd.DataFrame,
    spikes_by_cluster: dict[int, np.ndarray],
    run_id: str,
    paradigm: str,
) -> pd.DataFrame:
    """Build a long-form unit by event response table."""
    required = {"t0_s", "t1_s"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Event table is missing required columns: {sorted(missing)}")
    event_columns = [
        col
        for col in [
            "event_id",
            "bar_id",
            "segment_id",
            "rep",
            "dir_idx",
            "dir_code",
            "dir_name_en",
            "dir_deg",
            "step",
            "step_global",
            "phase",
            "phase_step",
            "t0_s",
            "t1_s",
            "dur_s",
            "source_event_file",
        ]
        if col in events.columns
    ]
    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        start_s = float(event["t0_s"])
        end_s = float(event["t1_s"])
        duration_s = float(event.get("dur_s", end_s - start_s))
        base = {col: event[col] for col in event_columns}
        base.update({"run_id": run_id, "paradigm": paradigm})
        for unit_id, spike_times_s in spikes_by_cluster.items():
            count = count_spikes_in_window(spike_times_s, start_s, end_s)
            row = dict(base)
            row.update(
                {
                    "unit_id": int(unit_id),
                    "spike_count": count,
                    "firing_rate_hz": count / duration_s if duration_s > 0 else np.nan,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def response_matrix(
    response_table: pd.DataFrame,
    value_column: str = "firing_rate_hz",
) -> pd.DataFrame:
    """Pivot a long response table into a unit by direction-step matrix."""
    table = response_table.copy()
    if {"dir_idx", "step"}.issubset(table.columns):
        table["dir_step"] = table["dir_idx"].astype(str) + "_" + table["step"].astype(str)
    else:
        table["dir_step"] = table.index.astype(str)
    return table.pivot_table(
        index="unit_id",
        columns="dir_step",
        values=value_column,
        aggfunc="mean",
    )
