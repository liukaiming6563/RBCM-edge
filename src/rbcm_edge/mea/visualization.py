"""Visualization helpers for MEA response tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rbcm_edge.utils.io import ensure_dir


def plot_step_response_summary(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Plot mean firing rate by step for each direction."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    if "dir_idx" in summary.columns:
        for dir_idx, part in summary.groupby("dir_idx"):
            part = part.sort_values("step")
            label = str(part.get("dir_name_en", pd.Series([dir_idx])).iloc[0])
            ax.plot(part["step"], part["mean"], marker="o", linewidth=1.5, label=label)
    else:
        part = summary.sort_values("step")
        ax.plot(part["step"], part["mean"], marker="o", linewidth=1.5)
    ax.set_xlabel("Stimulus step")
    ax.set_ylabel("Mean firing rate (Hz)")
    ax.set_title("Step-wise population response")
    ax.grid(True, alpha=0.25)
    if "dir_idx" in summary.columns:
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_modulation_index_histogram(mi_table: pd.DataFrame, output_path: str | Path) -> None:
    """Plot a histogram of modulation index values."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(mi_table["modulation_index"].dropna(), bins=40, color="#3366aa", alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Modulation index")
    ax.set_ylabel("Count")
    ax.set_title("Boundary-context modulation")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
