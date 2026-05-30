"""读取 RBCM-edge MEA 正式分析结果。

绘图代码只基于 outputs/MEA_analysis/tables 下已经生成的 CSV，
不重复计算 firing-rate 或 grid-level 指标。这样主分析和论文图导出
彼此独立，便于复现、排错和后期改图。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fig_style import PAIR_LABELS


REQUIRED_TABLES = {
    "grid": "grid_cell_level_results_all_scales_extended.csv",
    "paired_overall": "paired_retina_overall_summary.csv",
    "paired_direction": "paired_retina_direction_summary.csv",
    "threshold_pair": "threshold_sensitivity_pair_overall_summary.csv",
    "threshold_direction": "threshold_sensitivity_grid_direction_summary.csv",
    "spatial": "spatial_map_similarity.csv",
    "spatial_overall": "spatial_map_similarity_pair_overall.csv",
    "robust": "robust_metric_grid_summary.csv",
    "ndi": "NDI_summary.csv",
    "valid": "valid_grid_and_unit_count_summary.csv",
    "permutation": "permutation_null_summary.csv",
}


@dataclass
class MEAResults:
    """所有绘图所需结果表。"""

    grid: pd.DataFrame
    paired_overall: pd.DataFrame
    paired_direction: pd.DataFrame
    threshold_pair: pd.DataFrame
    threshold_direction: pd.DataFrame
    spatial: pd.DataFrame
    spatial_overall: pd.DataFrame
    robust: pd.DataFrame
    ndi: pd.DataFrame
    valid: pd.DataFrame
    permutation: pd.DataFrame


def _read_table(input_dir: Path, filename: str) -> pd.DataFrame:
    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required MEA result table: {path}")
    return pd.read_csv(path)


def _standardize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    """添加论文图中使用的可读标签列。"""

    out = df.copy()
    if "pair_id" in out.columns:
        out["pair_label"] = out["pair_id"].map(PAIR_LABELS).fillna(out["pair_id"])
    if "grid_n" in out.columns:
        out["grid_label"] = out["grid_n"].astype(int).astype(str) + "x" + out["grid_n"].astype(int).astype(str)
        out["grid_label_pretty"] = out["grid_n"].astype(int).astype(str) + "×" + out["grid_n"].astype(int).astype(str)
    if "direction_code" in out.columns:
        out["direction_code"] = out["direction_code"].astype(str)
    return out


def _check_columns(df: pd.DataFrame, name: str, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Table {name} is missing required columns: {sorted(missing)}")


def load_all_results(input_dir: str | Path) -> MEAResults:
    """读取并校验所有正式分析结果表。"""

    input_dir = Path(input_dir)
    tables = {key: _standardize_common_columns(_read_table(input_dir, filename)) for key, filename in REQUIRED_TABLES.items()}

    _check_columns(
        tables["grid"],
        "grid",
        {
            "grid_n",
            "pair_id",
            "direction_code",
            "grid_x",
            "grid_y",
            "valid_grid",
            "UME_mean_fr_hz",
            "CME_mean_fr_hz",
            "delta_mean_fr_hz",
            "NDI_mean",
            "delta_nonzero_fraction",
        },
    )
    _check_columns(
        tables["paired_overall"],
        "paired_overall",
        {
            "grid_n",
            "pair_id",
            "mean_different_fraction",
            "mean_abs_delta_mean_fr_hz",
            "mean_abs_NDI_mean",
            "mean_normalized_MAE",
            "total_valid_grids",
            "total_UME_higher",
            "total_CME_higher",
            "total_similar",
        },
    )
    _check_columns(
        tables["threshold_pair"],
        "threshold_pair",
        {"grid_n", "pair_id", "threshold_hz", "mean_different_fraction", "mean_UME_higher_fraction", "mean_CME_higher_fraction"},
    )
    _check_columns(
        tables["spatial"],
        "spatial",
        {"grid_n", "pair_id", "direction_code", "pearson_r", "normalized_MAE", "normalized_RMSE", "mean_abs_delta_mean_fr_hz"},
    )

    return MEAResults(**tables)


def select_representative_directions(results: MEAResults, grid_n: int = 12) -> dict[str, str]:
    """为每个 paired retina 选择一个代表方向。

    选择规则：在指定 grid scale 下，选择 mean_abs_delta_mean_fr_hz 最大的方向。
    这样主图中的代表性 ΔFR map 由数据驱动选择，而不是手工 cherry-pick。
    """

    sub = results.spatial[(results.spatial["grid_n"] == grid_n)].copy()
    reps: dict[str, str] = {}
    for pair_id, df in sub.groupby("pair_id"):
        row = df.sort_values("mean_abs_delta_mean_fr_hz", ascending=False).iloc[0]
        reps[pair_id] = str(row["direction_code"])
    return reps

