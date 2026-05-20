"""Kilosort output loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class KilosortRun:
    """In-memory representation of the essential Kilosort outputs."""

    run_id: str
    kilosort_dir: Path
    sample_rate_hz: float
    spike_times_samples: np.ndarray
    spike_times_s: np.ndarray
    spike_clusters: np.ndarray
    cluster_table: pd.DataFrame

    @property
    def unit_ids(self) -> np.ndarray:
        """Return sorted cluster ids present in the spike table."""
        return np.sort(np.unique(self.spike_clusters))


def load_kilosort_run(
    kilosort_dir: str | Path,
    run_id: str,
    sample_rate_hz: float = 20000.0,
) -> KilosortRun:
    """Load spike times, spike clusters, and cluster labels for one run."""
    kilosort_dir = Path(kilosort_dir)
    spike_times_path = kilosort_dir / "spike_times.npy"
    spike_clusters_path = kilosort_dir / "spike_clusters.npy"
    if not spike_times_path.exists() or not spike_clusters_path.exists():
        raise FileNotFoundError(
            f"Missing spike_times.npy or spike_clusters.npy in {kilosort_dir}"
        )
    spike_times_samples = np.load(spike_times_path).reshape(-1)
    spike_clusters = np.load(spike_clusters_path).reshape(-1)
    spike_times_s = spike_times_samples.astype(np.float64) / float(sample_rate_hz)
    cluster_table = load_cluster_table(kilosort_dir)
    return KilosortRun(
        run_id=run_id,
        kilosort_dir=kilosort_dir,
        sample_rate_hz=sample_rate_hz,
        spike_times_samples=spike_times_samples,
        spike_times_s=spike_times_s,
        spike_clusters=spike_clusters,
        cluster_table=cluster_table,
    )


def load_cluster_table(kilosort_dir: str | Path) -> pd.DataFrame:
    """Load and merge available Kilosort/Phy cluster metadata tables."""
    kilosort_dir = Path(kilosort_dir)
    tables: list[pd.DataFrame] = []
    for filename in [
        "cluster_group.tsv",
        "cluster_KSLabel.tsv",
        "cluster_Amplitude.tsv",
        "cluster_ContamPct.tsv",
    ]:
        path = kilosort_dir / filename
        if path.exists():
            table = pd.read_csv(path, sep="\t")
            if "cluster_id" not in table.columns:
                table = table.rename(columns={table.columns[0]: "cluster_id"})
            tables.append(table)
    if not tables:
        return pd.DataFrame(columns=["cluster_id"])
    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table, on="cluster_id", how="outer")
    return merged.sort_values("cluster_id").reset_index(drop=True)


def select_units(
    cluster_table: pd.DataFrame,
    allowed_labels: Iterable[str] = ("good",),
) -> np.ndarray:
    """Select cluster ids by manual or Kilosort labels."""
    if cluster_table.empty:
        return np.array([], dtype=int)
    allowed = {str(label).lower() for label in allowed_labels}
    masks = []
    for column in ["group", "KSLabel"]:
        if column in cluster_table.columns:
            masks.append(cluster_table[column].astype(str).str.lower().isin(allowed))
    if not masks:
        return cluster_table["cluster_id"].dropna().astype(int).sort_values().to_numpy()
    keep = masks[0]
    for mask in masks[1:]:
        keep = keep | mask
    return cluster_table.loc[keep, "cluster_id"].dropna().astype(int).sort_values().to_numpy()


def spikes_by_unit(run: KilosortRun, unit_ids: Iterable[int] | None = None) -> dict[int, np.ndarray]:
    """Group spike times in seconds by cluster id."""
    if unit_ids is None:
        unit_ids = run.unit_ids
    result: dict[int, np.ndarray] = {}
    for unit_id in unit_ids:
        mask = run.spike_clusters == int(unit_id)
        result[int(unit_id)] = np.sort(run.spike_times_s[mask])
    return result
