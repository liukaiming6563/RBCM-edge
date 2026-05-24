"""Plot loss and validation metric curves from a train_log.csv file.

PyCharm usage:
1. Edit `DEFAULT_ARGS` below if you want to point to another experiment.
2. Right-click this file and choose Run.

Command-line usage:
```powershell
python edge_model\\tools\\analyze_training_log.py --log-csv outputs\\edge_detection\\local_rbcm_fuse_demo\\logs\\train_log.csv
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ARGS = {
    "log_csv": PROJECT_ROOT
    / "outputs"
    / "edge_detection"
    / "local_rbcm_fuse_demo"
    / "logs"
    / "train_log.csv",
    "output_dir": None,
    "show": False,
}


def parse_args() -> argparse.Namespace:
    """Parse arguments while keeping editable defaults for PyCharm runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-csv", type=Path, default=DEFAULT_ARGS["log_csv"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARGS["output_dir"],
        help="Where to save plots. Default: <experiment_root>/plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=DEFAULT_ARGS["show"],
        help="Show figures interactively after saving them.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Load the training CSV and save trend plots."""
    log_csv = args.log_csv if args.log_csv.is_absolute() else PROJECT_ROOT / args.log_csv
    output_dir = args.output_dir
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    analyze_training_log(log_csv=log_csv, output_dir=output_dir)

    if args.show:
        plt.show()
    else:
        plt.close("all")


def analyze_training_log(log_csv: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    """Read a train_log.csv file and save all training trend plots.

    Args:
        log_csv: Path to the CSV written by `edge_model/train.py`.
        output_dir: Optional plot output directory. When omitted, plots are saved
            under `<experiment_root>/plots`.

    Returns:
        Mapping from plot name to saved file path. This return value is useful
        when `train.py` calls the function automatically after training.
    """
    log_csv = Path(log_csv)
    if not log_csv.exists():
        raise FileNotFoundError(
            f"Cannot find training log: {log_csv}. "
            "Run training first or set DEFAULT_ARGS['log_csv'] to your train_log.csv."
        )

    if output_dir is None:
        output_dir = log_csv.parents[1] / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = pd.read_csv(log_csv)
    if "epoch" not in log.columns or "split" not in log.columns:
        raise ValueError("train_log.csv must contain at least 'epoch' and 'split' columns.")
    log = normalize_training_log(log)

    print(f"Loaded {len(log)} rows from {log_csv}")
    print(f"Saving plots to {output_dir}")

    saved_paths = {
        "loss_curves": output_dir / "loss_curves.png",
        "validation_metrics": output_dir / "validation_metrics.png",
        "training_summary": output_dir / "training_summary.png",
    }
    plot_loss_curves(log, saved_paths["loss_curves"])
    plot_metric_curves(log, saved_paths["validation_metrics"], metrics=["ODS", "OIS", "AP"])
    plot_summary_panel(log, saved_paths["training_summary"])
    return saved_paths


def plot_loss_curves(log: pd.DataFrame, output_path: Path) -> None:
    """Plot train and validation loss curves."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for split, part in log.groupby("split"):
        split_name = str(split).lower()
        candidates = ["loss", "total"] if split_name == "val" else ["total", "loss"]
        value_col = _first_present_numeric_column(part, candidates)
        if value_col is None:
            continue
        ax.plot(part["epoch"], part[value_col], marker="o", linewidth=1.8, label=f"{split} {value_col}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved {output_path}")


def plot_metric_curves(log: pd.DataFrame, output_path: Path, metrics: list[str]) -> None:
    """Plot validation ODS/OIS/AP curves when those columns exist."""
    val = log[log["split"].astype(str).str.lower().eq("val")].copy()
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    plotted = False
    for metric in metrics:
        if _has_numeric_values(val, metric):
            ax.plot(val["epoch"], val[metric], marker="o", linewidth=1.8, label=metric)
            plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No validation metric columns found", ha="center", va="center")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Edge Metrics")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved {output_path}")


def plot_summary_panel(log: pd.DataFrame, output_path: Path) -> None:
    """Save a compact 2x2 panel for quick experiment review."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    axes = axes.ravel()

    train = log[log["split"].astype(str).str.lower().eq("train")]
    val = log[log["split"].astype(str).str.lower().eq("val")]

    _plot_column(axes[0], train, "epoch", "total", "Train Total Loss")
    _plot_column(axes[1], val, "epoch", "loss", "Val Loss")
    _plot_column(axes[2], val, "epoch", "ODS", "Val ODS")
    _plot_multiple_columns(axes[3], val, "epoch", ["OIS", "AP"], "Val OIS / AP")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved {output_path}")


def _plot_column(ax, data: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    """Plot one column if it exists; otherwise show a placeholder."""
    ax.set_title(title)
    if data.empty or not _has_numeric_values(data, y_col):
        ax.text(0.5, 0.5, f"Missing {y_col}", ha="center", va="center")
        return
    ax.plot(data[x_col], data[y_col], marker="o", linewidth=1.8)


def _plot_multiple_columns(ax, data: pd.DataFrame, x_col: str, y_cols: list[str], title: str) -> None:
    """Plot multiple columns on one axis when present."""
    ax.set_title(title)
    plotted = False
    for y_col in y_cols:
        if _has_numeric_values(data, y_col):
            ax.plot(data[x_col], data[y_col], marker="o", linewidth=1.8, label=y_col)
            plotted = True
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Missing metrics", ha="center", va="center")


def normalize_training_log(log: pd.DataFrame) -> pd.DataFrame:
    """Normalize current and legacy train_log.csv schemas for plotting.

    New logs contain both training loss columns and validation metric columns.
    A legacy writer bug saved validation rows using the training-loss header:
    `total=ODS`, `final_bce=OIS`, `final_dice=AP`, and `local=loss`.
    This function detects that layout and reconstructs the intended columns for
    analysis without modifying the original CSV on disk.
    """
    log = log.copy()
    metric_cols = ["ODS", "OIS", "AP", "loss"]
    for col in ["total", "final_bce", "final_dice", "local", "gate_sparsity", *metric_cols]:
        if col in log.columns:
            log[col] = pd.to_numeric(log[col], errors="coerce")

    val_mask = log["split"].astype(str).str.lower().eq("val")
    has_metric_cols = any(col in log.columns and log.loc[val_mask, col].notna().any() for col in metric_cols)
    legacy_cols = {"total", "final_bce", "final_dice", "local"}.issubset(log.columns)
    if val_mask.any() and not has_metric_cols and legacy_cols:
        log.loc[val_mask, "ODS"] = log.loc[val_mask, "total"]
        log.loc[val_mask, "OIS"] = log.loc[val_mask, "final_bce"]
        log.loc[val_mask, "AP"] = log.loc[val_mask, "final_dice"]
        log.loc[val_mask, "loss"] = log.loc[val_mask, "local"]
        log.loc[val_mask, ["total", "final_bce", "final_dice", "local"]] = pd.NA

    return log


def _first_present_numeric_column(data: pd.DataFrame, columns: list[str]) -> str | None:
    """Return the first candidate column containing at least one numeric value."""
    for column in columns:
        if _has_numeric_values(data, column):
            return column
    return None


def _has_numeric_values(data: pd.DataFrame, column: str) -> bool:
    """Check whether a column exists and has at least one non-null value."""
    return column in data.columns and pd.to_numeric(data[column], errors="coerce").notna().any()


if __name__ == "__main__":
    main(parse_args())
