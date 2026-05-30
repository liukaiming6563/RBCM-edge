"""
扩展版 MEA permutation test：比较不同移动步骤窗口与 ON/OFF phase 条件下的 UME/CME 差异。

本脚本回答的问题：
    在原始主分析只取“边缘移动到视野中间那一步 + ON phase”的基础上，
    如果改用更宽的移动步骤窗口，或者改用 OFF / ON+OFF 响应窗口，
    UME 与 CME 的 grid-level population response 差异是否更容易通过 permutation test？

重要解释边界：
    1. UME 与 CME 来自不同 recording / spike sorting，不能做 unit-level pairing。
    2. permutation test 在每个 spatial grid 内随机打乱 UME/CME 标签，同时保持两侧 unit 数不变。
    3. 该检验回答的是：真实 UME/CME 标签下的局部群体差异是否大于随机标签分配预期。
    4. ON_OFF 在本脚本中定义为 ON 与 OFF 两类事件窗口合并后一起平均，不是 ON - OFF。

输入：
    D:/study/project/RBCM-Edge/MEA_data/<exp_id>/segment_result/processed_segment/good_on/output_fre.npy
    D:/study/project/RBCM-Edge/MEA_data/<exp_id>/segment_result/processed_segment/good_off/output_fre.npy

输出：
    D:/study/project/RBCM-Edge/outputs/MEA_analysis/permutation_window_phase_analysis/
        tables/
        figures/
        reports/
        logs/
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# 复用正式 MEA 分析中的空间坐标重建、网格分配、聚合与基础常量。
from run_MEA_grid_full_analysis import (
    MEA_DIR,
    PAIR_CONFIG,
    GRID_SCALES,
    DIRECTION_CONFIG,
    MIN_UNITS_PER_GRID_PER_STIM,
    MAIN_THRESHOLD_HZ,
    EPSILON,
    RANDOM_SEED,
    load_unit_positions,
    assign_grid,
    build_unit_table,
    aggregate_grid,
    ndi,
    classify_delta,
)


# =============================================================================
# 1. 输出目录与分析配置
# =============================================================================

PROJECT_DIR = Path(r"D:\study\project\RBCM-Edge")
ANALYSIS_DIR = PROJECT_DIR / "MEA_analysis"
OUT_DIR = PROJECT_DIR / "outputs" / "MEA_analysis" / "permutation_window_phase_analysis"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"
LOG_DIR = OUT_DIR / "logs"

# 与主分析一致：每个有效 grid 中 UME 和 CME 各至少 3 个 good units。
# 如果以后要测试 min_units=1，可单独改这里。
MIN_UNITS_PER_GRID_PER_STIM_EXTENDED = MIN_UNITS_PER_GRID_PER_STIM

# 置换次数。与主分析保持一致，兼顾运行时间与稳定性。
N_PERM = 1000

# 6 种移动步骤窗口。
# 注意：这里使用 Python 半开区间 [start, stop)，与用户写的 0-7、7-13 等描述对应。
STEP_WINDOWS = {
    "all_steps": {
        "label_cn": "所有移动步骤",
        "UME": (0, 13),
        "CME": (0, 11),
    },
    "approach_to_center": {
        "label_cn": "边缘逐步靠近视野中间",
        "UME": (0, 7),
        "CME": (0, 6),
    },
    "depart_from_center": {
        "label_cn": "边缘逐步远离视野中间",
        "UME": (7, 13),
        "CME": (6, 11),
    },
    "late_approach": {
        "label_cn": "靠近视野中间的后半程",
        "UME": (4, 7),
        "CME": (3, 6),
    },
    "early_depart": {
        "label_cn": "远离视野中间的前半程",
        "UME": (7, 10),
        "CME": (6, 9),
    },
    "middle_motion": {
        "label_cn": "整个移动过程的中间环节",
        "UME": (4, 10),
        "CME": (3, 9),
    },
}

# 3 种 phase 条件。
# ON_OFF 表示将 ON 与 OFF 事件窗口合并后一起平均，不是 ON - OFF。
PHASE_MODES = ["ON", "OFF", "ON_OFF"]


mpl.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.family": "Arial",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


LOG_LINES: list[str] = []


def log(message: str) -> None:
    """同时打印和记录日志。"""

    print(message)
    LOG_LINES.append(message)


def ensure_dirs() -> None:
    """创建本任务单独输出文件夹。"""

    for d in [OUT_DIR, TABLE_DIR, FIG_DIR, REPORT_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. firing-rate 读取与移动步骤窗口提取
# =============================================================================


def _phase_dir_name(phase: str) -> str:
    """将 phase 名称转换为 processed_segment 子目录名称。"""

    if phase == "ON":
        return "good_on"
    if phase == "OFF":
        return "good_off"
    raise ValueError(f"_phase_dir_name only supports ON/OFF, got {phase}")


def load_phase_fr_array(exp_id: str, phase: str) -> np.ndarray:
    """读取某个 experiment 在 ON 或 OFF 条件下的 good-unit firing-rate 数组。

    返回数组 shape = [repeat, good_unit, event]。
    """

    path = MEA_DIR / exp_id / "segment_result" / "processed_segment" / _phase_dir_name(phase) / "output_fre.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing firing-rate array: {path}")
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"{path} should have shape repeat x unit x event, got {arr.shape}")
    if np.nanmin(arr) < 0:
        raise ValueError(f"{path} contains negative firing rates")
    return arr.astype(float, copy=False)


def event_indices_for_window(stimulus: str, direction_zero_based: int, window_name: str) -> np.ndarray:
    """计算某个 stimulus / direction / window 对应的 event indices。

    event 轴的组织方式与 metadata 一致：
        UME: 8 个方向，每个方向 13 步，共 104 events；
        CME: 8 个方向，每个方向 11 步，共 88 events。
    """

    if stimulus not in {"UME", "CME"}:
        raise ValueError(f"Unknown stimulus: {stimulus}")
    if window_name not in STEP_WINDOWS:
        raise ValueError(f"Unknown window: {window_name}")

    steps_per_dir = 13 if stimulus == "UME" else 11
    start, stop = STEP_WINDOWS[window_name][stimulus]
    if not (0 <= start < stop <= steps_per_dir):
        raise ValueError(f"Invalid step range for {stimulus}/{window_name}: {(start, stop)}")
    return direction_zero_based * steps_per_dir + np.arange(start, stop)


def load_window_phase_fr(
    exp_id: str,
    stimulus: str,
    direction_zero_based: int,
    window_name: str,
    phase_mode: str,
) -> np.ndarray:
    """提取某个条件下每个 unit 的平均 firing rate。

    输出 shape = [n_good_units]。

    averaging 维度：
        - repeat 维度；
        - 所选移动步骤 event 维度；
        - 如果 phase_mode == ON_OFF，则 ON 与 OFF 的 event 窗口合并后一起平均。
    """

    event_idx = event_indices_for_window(stimulus, direction_zero_based, window_name)

    def extract_from_phase(phase: str) -> np.ndarray:
        arr = load_phase_fr_array(exp_id, phase)
        if int(event_idx.max()) >= arr.shape[2]:
            raise IndexError(
                f"{exp_id} {stimulus} {phase} {window_name}: event index {int(event_idx.max())} "
                f"out of range for shape {arr.shape}"
            )
        # selected shape = [repeat, unit, selected_events]
        return arr[:, :, event_idx]

    if phase_mode in {"ON", "OFF"}:
        selected = extract_from_phase(phase_mode)
    elif phase_mode == "ON_OFF":
        selected = np.concatenate([extract_from_phase("ON"), extract_from_phase("OFF")], axis=2)
    else:
        raise ValueError(f"Unknown phase_mode: {phase_mode}")

    # 对 repeat 与事件窗口平均，每个 unit 得到一个 firing-rate 值。
    return np.nanmean(selected, axis=(0, 2))


# =============================================================================
# 3. grid metrics 与 permutation test
# =============================================================================


def compute_grid_and_unit_table(
    grid_n: int,
    pair_id: str,
    pair_cfg: dict[str, str],
    direction: dict,
    window_name: str,
    phase_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算一个条件组合下的 grid-level 明细与 unit-level 表。"""

    cme_exp = pair_cfg["CME"]
    ume_exp = pair_cfg["UME"]
    direction_zero_based = direction["id"] - 1

    ume_units = load_unit_positions(ume_exp, "single_edge")
    cme_units = load_unit_positions(cme_exp, "double_edge")

    # 同一 pair 中 UME/CME 共享空间边界，保证 spatial grid matched。
    all_x = pd.concat([ume_units["x"], cme_units["x"]], ignore_index=True)
    all_y = pd.concat([ume_units["y"], cme_units["y"]], ignore_index=True)
    x_edges = np.linspace(all_x.min(), all_x.max(), grid_n + 1)
    y_edges = np.linspace(all_y.min(), all_y.max(), grid_n + 1)
    ume_units = assign_grid(ume_units, x_edges, y_edges, grid_n)
    cme_units = assign_grid(cme_units, x_edges, y_edges, grid_n)

    ume_fr = load_window_phase_fr(ume_exp, "UME", direction_zero_based, window_name, phase_mode)
    cme_fr = load_window_phase_fr(cme_exp, "CME", direction_zero_based, window_name, phase_mode)
    ume_unit_table = build_unit_table(ume_units, ume_fr, "UME")
    cme_unit_table = build_unit_table(cme_units, cme_fr, "CME")
    unit_df = pd.concat([ume_unit_table, cme_unit_table], ignore_index=True)

    ume_grid = aggregate_grid(ume_unit_table, "UME")
    cme_grid = aggregate_grid(cme_unit_table, "CME")

    full_grid = pd.DataFrame(
        [
            {"grid_x": gx, "grid_y": gy, "grid_id": f"{gy:02d}_{gx:02d}"}
            for gy in range(grid_n)
            for gx in range(grid_n)
        ]
    )
    grid = full_grid.merge(ume_grid, on=["grid_id", "grid_x", "grid_y"], how="left")
    grid = grid.merge(cme_grid, on=["grid_id", "grid_x", "grid_y"], how="left")

    for col in [
        "UME_unit_count",
        "CME_unit_count",
        "UME_zero_count",
        "CME_zero_count",
        "UME_nonzero_count",
        "CME_nonzero_count",
    ]:
        grid[col] = grid[col].fillna(0).astype(int)

    grid["grid_scale"] = f"{grid_n}x{grid_n}"
    grid["grid_n"] = grid_n
    grid["pair_id"] = pair_id
    grid["UME_exp"] = ume_exp
    grid["CME_exp"] = cme_exp
    grid["direction_id"] = f"{direction['id']:02d}"
    grid["direction_code"] = direction["code"]
    grid["direction_name"] = direction["name"]
    grid["window_name"] = window_name
    grid["window_label_cn"] = STEP_WINDOWS[window_name]["label_cn"]
    grid["phase_mode"] = phase_mode
    grid["UME_step_start"] = STEP_WINDOWS[window_name]["UME"][0]
    grid["UME_step_stop"] = STEP_WINDOWS[window_name]["UME"][1]
    grid["CME_step_start"] = STEP_WINDOWS[window_name]["CME"][0]
    grid["CME_step_stop"] = STEP_WINDOWS[window_name]["CME"][1]
    grid["min_units_per_grid_per_stim"] = MIN_UNITS_PER_GRID_PER_STIM_EXTENDED
    grid["valid_grid"] = (
        (grid["UME_unit_count"] >= MIN_UNITS_PER_GRID_PER_STIM_EXTENDED)
        & (grid["CME_unit_count"] >= MIN_UNITS_PER_GRID_PER_STIM_EXTENDED)
    )

    grid["delta_mean_fr_hz"] = grid["UME_mean_fr_hz"] - grid["CME_mean_fr_hz"]
    grid["abs_delta_mean_fr_hz"] = grid["delta_mean_fr_hz"].abs()
    grid["NDI_mean"] = ndi(grid["UME_mean_fr_hz"], grid["CME_mean_fr_hz"])
    grid["abs_NDI_mean"] = grid["NDI_mean"].abs()
    grid["difference_class_main"] = classify_delta(
        grid["delta_mean_fr_hz"],
        grid["valid_grid"],
        MAIN_THRESHOLD_HZ,
    )

    unit_df["grid_scale"] = f"{grid_n}x{grid_n}"
    unit_df["grid_n"] = grid_n
    unit_df["pair_id"] = pair_id
    unit_df["UME_exp"] = ume_exp
    unit_df["CME_exp"] = cme_exp
    unit_df["direction_id"] = f"{direction['id']:02d}"
    unit_df["direction_code"] = direction["code"]
    unit_df["direction_name"] = direction["name"]
    unit_df["window_name"] = window_name
    unit_df["phase_mode"] = phase_mode
    return grid, unit_df


def permutation_for_condition(grid: pd.DataFrame, unit_df: pd.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    """对一个条件组合做 permutation null comparison。"""

    valid = grid[grid["valid_grid"]].copy()
    if valid.empty:
        return {
            "real_mean_abs_delta": np.nan,
            "null_mean_mean_abs_delta": np.nan,
            "null_std_mean_abs_delta": np.nan,
            "p_mean_abs_delta": np.nan,
            "real_different_fraction": np.nan,
            "null_mean_different_fraction": np.nan,
            "null_std_different_fraction": np.nan,
            "p_different_fraction": np.nan,
            "real_mean_abs_NDI": np.nan,
            "null_mean_mean_abs_NDI": np.nan,
            "null_std_mean_abs_NDI": np.nan,
            "p_mean_abs_NDI": np.nan,
        }

    real_mean_abs_delta = float(valid["abs_delta_mean_fr_hz"].mean())
    real_different_fraction = float((valid["difference_class_main"].isin(["UME_higher", "CME_higher"])).mean())
    real_mean_abs_NDI = float(valid["abs_NDI_mean"].mean())

    # 预先把每个 valid grid 内的 firing-rate 数组和 UME sample size 存起来，减少循环中 pandas 操作。
    grid_samples: list[tuple[np.ndarray, int]] = []
    for gid in valid["grid_id"]:
        sub = unit_df[unit_df["grid_id"] == gid]
        ume_vals = sub.loc[sub["stimulus"] == "UME", "firing_rate_hz"].to_numpy(dtype=float)
        cme_vals = sub.loc[sub["stimulus"] == "CME", "firing_rate_hz"].to_numpy(dtype=float)
        if len(ume_vals) < MIN_UNITS_PER_GRID_PER_STIM_EXTENDED or len(cme_vals) < MIN_UNITS_PER_GRID_PER_STIM_EXTENDED:
            continue
        merged = np.concatenate([ume_vals, cme_vals])
        grid_samples.append((merged, len(ume_vals)))

    null_abs = np.empty(N_PERM, dtype=float)
    null_diff = np.empty(N_PERM, dtype=float)
    null_ndi = np.empty(N_PERM, dtype=float)

    for p in range(N_PERM):
        abs_delta_values = []
        different_count = 0
        ndi_values = []
        for merged, n_ume in grid_samples:
            perm = rng.permutation(merged)
            pseudo_ume = perm[:n_ume]
            pseudo_cme = perm[n_ume:]
            m_ume = float(np.mean(pseudo_ume))
            m_cme = float(np.mean(pseudo_cme))
            delta = m_ume - m_cme
            abs_delta_values.append(abs(delta))
            if abs(delta) > MAIN_THRESHOLD_HZ:
                different_count += 1
            ndi_values.append(abs(delta / (m_ume + m_cme + EPSILON)))

        null_abs[p] = float(np.mean(abs_delta_values)) if abs_delta_values else np.nan
        null_diff[p] = float(different_count / len(grid_samples)) if grid_samples else np.nan
        null_ndi[p] = float(np.mean(ndi_values)) if ndi_values else np.nan

    def p_value(null_arr: np.ndarray, real: float) -> float:
        arr = null_arr[np.isfinite(null_arr)]
        if arr.size == 0 or not np.isfinite(real):
            return np.nan
        return float((np.sum(arr >= real) + 1) / (arr.size + 1))

    return {
        "real_mean_abs_delta": real_mean_abs_delta,
        "null_mean_mean_abs_delta": float(np.nanmean(null_abs)),
        "null_std_mean_abs_delta": float(np.nanstd(null_abs, ddof=1)),
        "p_mean_abs_delta": p_value(null_abs, real_mean_abs_delta),
        "real_different_fraction": real_different_fraction,
        "null_mean_different_fraction": float(np.nanmean(null_diff)),
        "null_std_different_fraction": float(np.nanstd(null_diff, ddof=1)),
        "p_different_fraction": p_value(null_diff, real_different_fraction),
        "real_mean_abs_NDI": real_mean_abs_NDI,
        "null_mean_mean_abs_NDI": float(np.nanmean(null_ndi)),
        "null_std_mean_abs_NDI": float(np.nanstd(null_ndi, ddof=1)),
        "p_mean_abs_NDI": p_value(null_ndi, real_mean_abs_NDI),
    }


# =============================================================================
# 4. 汇总表与图
# =============================================================================


def summarize_significance(perm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成显著性数量汇总。"""

    group_cols = ["window_name", "window_label_cn", "phase_mode"]
    rows = []
    for keys, sub in perm.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_tests"] = int(len(sub))
        for metric in ["p_mean_abs_delta", "p_different_fraction", "p_mean_abs_NDI"]:
            vals = sub[metric].dropna().astype(float)
            row[f"{metric}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{metric}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{metric}_sig_count_0p05"] = int((vals < 0.05).sum())
            row[f"{metric}_sig_fraction_0p05"] = float((vals < 0.05).mean()) if len(vals) else np.nan
        rows.append(row)
    by_condition = pd.DataFrame(rows)

    rows = []
    for keys, sub in perm.groupby(["window_name", "window_label_cn", "phase_mode", "grid_scale", "grid_n", "pair_id"]):
        row = dict(zip(["window_name", "window_label_cn", "phase_mode", "grid_scale", "grid_n", "pair_id"], keys))
        row["n_direction_tests"] = int(len(sub))
        for metric in ["p_mean_abs_delta", "p_different_fraction", "p_mean_abs_NDI"]:
            vals = sub[metric].dropna().astype(float)
            row[f"{metric}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{metric}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{metric}_sig_count_0p05"] = int((vals < 0.05).sum())
        rows.append(row)
    by_pair_scale = pd.DataFrame(rows)
    return by_condition, by_pair_scale


def make_condition_heatmap(summary: pd.DataFrame, metric: str, stem: str) -> None:
    """绘制 phase × window 的显著比例热图。"""

    pivot = summary.pivot(index="phase_mode", columns="window_name", values=f"{metric}_sig_fraction_0p05")
    # 固定顺序，方便读图。
    pivot = pivot.reindex(index=PHASE_MODES, columns=list(STEP_WINDOWS.keys()))
    fig, ax = plt.subplots(figsize=(11, 3.0))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "NA", ha="center", va="center", color="white", fontsize=8)
    ax.set_title(f"Fraction of tests with {metric} < 0.05")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("significant fraction")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def make_min_p_heatmap(summary: pd.DataFrame, metric: str, stem: str) -> None:
    """绘制 phase × window 的最小 p 值热图。"""

    pivot = summary.pivot(index="phase_mode", columns="window_name", values=f"{metric}_min")
    pivot = pivot.reindex(index=PHASE_MODES, columns=list(STEP_WINDOWS.keys()))
    values = pivot.to_numpy(dtype=float)
    plot_values = -np.log10(np.clip(values, 1e-6, 1.0))
    fig, ax = plt.subplots(figsize=(11, 3.0))
    im = ax.imshow(plot_values, cmap="magma", vmin=0, vmax=max(2.0, np.nanmax(plot_values)), aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.3g}" if np.isfinite(value) else "NA", ha="center", va="center", color="white", fontsize=8)
    ax.set_title(f"Minimum {metric} across all grid/pair/direction tests")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("-log10(min p)")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_report(perm: pd.DataFrame, summary: pd.DataFrame) -> None:
    """写一个简短 Markdown 报告，方便快速判断哪些条件更有希望。"""

    lines = []
    lines.append("# MEA permutation test under alternative step-window and phase conditions")
    lines.append("")
    lines.append("## Analysis definition")
    lines.append("")
    lines.append("- UME/CME are compared at spatially matched grid-level local population scale.")
    lines.append("- ON_OFF means ON and OFF event windows were pooled and averaged, not ON minus OFF.")
    lines.append(f"- Permutations per test: {N_PERM}.")
    lines.append(f"- Valid grid rule: UME and CME each have at least {MIN_UNITS_PER_GRID_PER_STIM_EXTENDED} good units.")
    lines.append(f"- Main threshold for different_fraction: |DeltaFR| > {MAIN_THRESHOLD_HZ} Hz.")
    lines.append("")

    pcols = ["p_mean_abs_delta", "p_different_fraction", "p_mean_abs_NDI"]
    for pcol in pcols:
        vals = perm[pcol].dropna().astype(float)
        lines.append(f"## {pcol}")
        lines.append("")
        lines.append(f"- Significant tests (p < 0.05): {(vals < 0.05).sum()} / {len(vals)} ({(vals < 0.05).mean():.3f})")
        lines.append(f"- Min p: {vals.min():.4g}")
        lines.append(f"- Median p: {vals.median():.4g}")
        top = perm.sort_values(pcol).head(12)
        lines.append("")
        lines.append("| window | phase | grid | pair | dir | p | real | null_mean |")
        lines.append("|---|---|---|---|---|---:|---:|---:|")
        real_col = {
            "p_mean_abs_delta": "real_mean_abs_delta",
            "p_different_fraction": "real_different_fraction",
            "p_mean_abs_NDI": "real_mean_abs_NDI",
        }[pcol]
        null_col = {
            "p_mean_abs_delta": "null_mean_mean_abs_delta",
            "p_different_fraction": "null_mean_different_fraction",
            "p_mean_abs_NDI": "null_mean_mean_abs_NDI",
        }[pcol]
        for r in top.itertuples(index=False):
            lines.append(
                f"| {r.window_name} | {r.phase_mode} | {r.grid_scale} | {r.pair_id} | {r.direction_code} | "
                f"{getattr(r, pcol):.4g} | {getattr(r, real_col):.4g} | {getattr(r, null_col):.4g} |"
            )
        lines.append("")

    best = summary.copy()
    best["any_sig_fraction"] = best[
        [
            "p_mean_abs_delta_sig_fraction_0p05",
            "p_different_fraction_sig_fraction_0p05",
            "p_mean_abs_NDI_sig_fraction_0p05",
        ]
    ].max(axis=1)
    best = best.sort_values("any_sig_fraction", ascending=False).head(10)
    lines.append("## Best condition-level significant fractions")
    lines.append("")
    lines.append("| window | phase | sig frac abs_delta | sig frac diff_fraction | sig frac NDI |")
    lines.append("|---|---|---:|---:|---:|")
    for r in best.itertuples(index=False):
        lines.append(
            f"| {r.window_name} | {r.phase_mode} | "
            f"{r.p_mean_abs_delta_sig_fraction_0p05:.3f} | "
            f"{r.p_different_fraction_sig_fraction_0p05:.3f} | "
            f"{r.p_mean_abs_NDI_sig_fraction_0p05:.3f} |"
        )

    (REPORT_DIR / "permutation_window_phase_report.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 5. 主流程
# =============================================================================


def main() -> None:
    """运行完整扩展 permutation 分析。"""

    ensure_dirs()
    rng = np.random.default_rng(RANDOM_SEED)
    log("Start extended MEA permutation window/phase analysis")
    log(f"Output directory: {OUT_DIR}")

    config = {
        "STEP_WINDOWS": STEP_WINDOWS,
        "PHASE_MODES": PHASE_MODES,
        "GRID_SCALES": GRID_SCALES,
        "PAIR_CONFIG": PAIR_CONFIG,
        "DIRECTION_CONFIG": DIRECTION_CONFIG,
        "MIN_UNITS_PER_GRID_PER_STIM": MIN_UNITS_PER_GRID_PER_STIM_EXTENDED,
        "MAIN_THRESHOLD_HZ": MAIN_THRESHOLD_HZ,
        "N_PERM": N_PERM,
        "RANDOM_SEED": RANDOM_SEED,
        "ON_OFF_definition": "pooled average of ON and OFF event windows, not ON minus OFF",
    }
    (OUT_DIR / "analysis_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    grid_summaries = []
    n_total = len(STEP_WINDOWS) * len(PHASE_MODES) * len(GRID_SCALES) * len(PAIR_CONFIG) * len(DIRECTION_CONFIG)
    counter = 0

    for window_name in STEP_WINDOWS:
        for phase_mode in PHASE_MODES:
            for grid_n in GRID_SCALES:
                for pair_id, pair_cfg in PAIR_CONFIG.items():
                    for direction in DIRECTION_CONFIG:
                        counter += 1
                        log(
                            f"[{counter}/{n_total}] window={window_name}, phase={phase_mode}, "
                            f"grid={grid_n}x{grid_n}, pair={pair_id}, dir={direction['code']}"
                        )
                        grid, unit_df = compute_grid_and_unit_table(
                            grid_n=grid_n,
                            pair_id=pair_id,
                            pair_cfg=pair_cfg,
                            direction=direction,
                            window_name=window_name,
                            phase_mode=phase_mode,
                        )
                        result = permutation_for_condition(grid, unit_df, rng)
                        base = {
                            "window_name": window_name,
                            "window_label_cn": STEP_WINDOWS[window_name]["label_cn"],
                            "phase_mode": phase_mode,
                            "grid_scale": f"{grid_n}x{grid_n}",
                            "grid_n": grid_n,
                            "pair_id": pair_id,
                            "direction_id": f"{direction['id']:02d}",
                            "direction_code": direction["code"],
                            "direction_name": direction["name"],
                            "n_perm": N_PERM,
                            "valid_grid_count": int(grid["valid_grid"].sum()),
                            "total_grid_count": int(len(grid)),
                            "UME_step_start": STEP_WINDOWS[window_name]["UME"][0],
                            "UME_step_stop": STEP_WINDOWS[window_name]["UME"][1],
                            "CME_step_start": STEP_WINDOWS[window_name]["CME"][0],
                            "CME_step_stop": STEP_WINDOWS[window_name]["CME"][1],
                        }
                        rows.append({**base, **result})
                        valid = grid[grid["valid_grid"]]
                        grid_summaries.append(
                            {
                                **base,
                                "mean_abs_delta_mean_fr_hz": float(valid["abs_delta_mean_fr_hz"].mean()) if len(valid) else np.nan,
                                "mean_abs_NDI_mean": float(valid["abs_NDI_mean"].mean()) if len(valid) else np.nan,
                                "different_fraction": float(valid["difference_class_main"].isin(["UME_higher", "CME_higher"]).mean())
                                if len(valid)
                                else np.nan,
                            }
                        )

    perm = pd.DataFrame(rows)
    grid_summary = pd.DataFrame(grid_summaries)
    condition_summary, pair_scale_summary = summarize_significance(perm)

    perm_path = TABLE_DIR / "permutation_window_phase_summary.csv"
    grid_path = TABLE_DIR / "grid_metric_window_phase_summary.csv"
    cond_path = TABLE_DIR / "permutation_window_phase_condition_significance.csv"
    pair_path = TABLE_DIR / "permutation_window_phase_pair_scale_significance.csv"
    perm.to_csv(perm_path, index=False, encoding="utf-8-sig")
    grid_summary.to_csv(grid_path, index=False, encoding="utf-8-sig")
    condition_summary.to_csv(cond_path, index=False, encoding="utf-8-sig")
    pair_scale_summary.to_csv(pair_path, index=False, encoding="utf-8-sig")

    for pcol in ["p_mean_abs_delta", "p_different_fraction", "p_mean_abs_NDI"]:
        make_condition_heatmap(condition_summary, pcol, f"significant_fraction_heatmap_{pcol}")
        make_min_p_heatmap(condition_summary, pcol, f"min_p_heatmap_{pcol}")

    write_report(perm, condition_summary)

    shutil.copy2(Path(__file__).resolve(), OUT_DIR / "run_MEA_permutation_window_phase_analysis_code_snapshot.py")
    (LOG_DIR / "analysis_log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")

    log(f"Saved permutation table: {perm_path}")
    log(f"Saved condition significance table: {cond_path}")
    log("Finished extended MEA permutation window/phase analysis")
    # 写入最后日志。
    (LOG_DIR / "analysis_log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
