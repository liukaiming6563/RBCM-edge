"""Stimulus event table loading and normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rbcm_edge.utils.io import read_csv_auto


def load_events(stimulus_dir: str | Path, visible_only: bool = True) -> pd.DataFrame:
    """Load stimulus timing information from bars_only.csv or events_tidy.csv."""
    stimulus_dir = Path(stimulus_dir)
    preferred = "bars_only.csv" if visible_only else "events_tidy.csv"
    path = stimulus_dir / preferred
    if not path.exists() and visible_only:
        path = stimulus_dir / "events_tidy.csv"
    if not path.exists():
        raise FileNotFoundError(f"No event CSV found in {stimulus_dir}")
    events = read_csv_auto(path)
    events = normalize_event_columns(events)
    events["source_event_file"] = path.name
    return events


def normalize_event_columns(events: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column names without discarding original information."""
    events = events.copy()
    rename_map = {
        "repeat": "rep",
        "direction": "dir_idx",
        "direction_idx": "dir_idx",
        "dir_name": "dir_name_en",
        "on_start_s": "t0_s",
        "on_end_s": "t1_s",
    }
    events = events.rename(columns={k: v for k, v in rename_map.items() if k in events.columns})
    if "dur_s" not in events.columns and {"t0_s", "t1_s"}.issubset(events.columns):
        events["dur_s"] = events["t1_s"] - events["t0_s"]
    if "event_type" in events.columns:
        visible = events["event_type"].astype(str).str.lower().eq("bar")
        if visible.any():
            events["is_visible_step"] = visible
    elif "source_event_id" in events.columns or "bar_id" in events.columns:
        events["is_visible_step"] = True
    return events
