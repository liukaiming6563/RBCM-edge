"""
MEA permutation test for log ON/OFF ratio response.

本脚本只检验一种响应定义：

    response = log((FR_ON + EPS_RATIO_HZ) / (FR_OFF + EPS_RATIO_HZ))

其中：
    - FR_ON / FR_OFF 分别在指定移动步骤窗口内，对 repeat 和 step 取平均；
    - EPS_RATIO_HZ 默认为 0.1 Hz，用于避免 OFF firing rate 为 0 或极小时 ratio 爆炸；
    - response > 0 表示 ON 阶段相对 OFF 阶段增强；
    - response < 0 表示 ON 阶段相对 OFF 阶段降低。

Permutation test 的单位仍然是 spatially matched grid-level local population：
    在每个有效 grid 内合并 UME/CME unit response，然后随机打乱 UME/CME 标签，
    保持 UME 与 CME 的 unit 数不变，检验真实标签下的差异是否大于随机预期。

注意：
    1. UME 与 CME 来自不同 recording / spike sorting，不能做 unit-level pairing。
    2. 本脚本不检验单独 ON 或单独 OFF，只检验 log ON/OFF ratio。
    3. 因为 response 可以为负，归一化差异使用：
       norm_diff = (UME_mean - CME_mean) / (abs(UME_mean) + abs(CME_mean) + EPSILON)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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
    classify_delta,
)


# =============================================================================
# 1. 配置区
# =============================================================================

PROJECT_DIR = Path(r"D:\study\project\RBCM-Edge")
OUT_DIR = PROJECT_DIR / "outputs" / "MEA_analysis" / "permutation_log_onoff_ratio_analysis"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"
LOG_DIR = OUT_DIR / "logs"

N_PERM = 1000
EPS_RATIO_HZ = 0.1
MIN_UNITS_PER_GRID_PER_STIM_EXTENDED = MIN_UNITS_PER_GRID_PER_STIM

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
    print(message)
    LOG_LINES.append(message)


def ensure_dirs() -> None:
    for d in [OUT_DIR, TABLE_DIR, FIG_DIR, REPORT_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. firing-rate 读取与 log ON/OFF ratio 提取
# =============================================================================


def load_phase_fr_array(exp_id: str, phase: str) -> np.ndarray:
    """读取 good_on 或 good_off firing-rate 数组，shape = repeat x unit x event。"""

    if phase not in {"ON", "OFF"}:
        raise ValueError(f"phase should be ON or OFF, got {phase}")
    subdir = "good_on" if phase == "ON" else "good_off"
    path = MEA_DIR / exp_id / "segment_result" / "processed_segment" / subdir / "output_fre.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing firing-rate array: {path}")
    arr = np.load(path).astype(float, copy=False)
    if arr.ndim != 3:
        raise ValueError(f"{path} should have shape repeat x unit x event, got {arr.shape}")
    if np.nanmin(arr) < 0:
        raise ValueError(f"{path} contains negative firing rates")
    return arr


def event_indices_for_window(stimulus: str, direction_zero_based: int, window_name: str) -> np.ndarray:
    """计算某个 stimulus / direction / step-window 的 event indices。"""

    steps_per_dir = 13 if stimulus == "UME" else 11
    start, stop = STEP_WINDOWS[window_name][stimulus]
    if not (0 <= start < stop <= steps_per_dir):
        raise ValueError(f"Invalid window {window_name} for {stimulus}: {(start, stop)}")
    return direction_zero_based * steps_per_dir + np.arange(start, stop)


def load_log_onoff_ratio_response(
    exp_id: str,
    stimulus: str,
    direction_zero_based: int,
    window_name: str,
) -> np.ndarray:
    """提取每个 good unit 的 log ON/OFF ratio response。

    输出 shape = [n_good_units]。

    计算步骤：
        1. 在指定 step-window 内分别取 ON 和 OFF firing-rate 数组；
        2. 对 repeat 和 step/event 维度求平均，得到每个 unit 的 FR_ON 和 FR_OFF；
        3. 计算 log((FR_ON + EPS_RATIO_HZ) / (FR_OFF + EPS_RATIO_HZ))。
    """

    event_idx = event_indices_for_window(stimulus, direction_zero_based, window_name)
    on = load_phase_fr_array(exp_id, "ON")
    off = load_phase_fr_array(exp_id, "OFF")
    if int(event_idx.max()) >= on.shape[2] or int(event_idx.max()) >= off.shape[2]:
        raise IndexError(
            f"{exp_id} {stimulus} {window_name}: event index {int(event_idx.max())} "
            f"out of range, ON shape={on.shape}, OFF shape={off.shape}"
        )
    on_mean = np.nanmean(on[:, :, event_idx], axis=(0, 2))
    off_mean = np.nanmean(off[:, :, event_idx], axis=(0, 2))
    return np.log((on_mean + EPS_RATIO_HZ) / (off_mean + EPS_RATIO_HZ))


# =============================================================================
# 3. grid metric 与 permutation
# =============================================================================


def normalized_signed_difference(a: pd.Series, b: pd.Series) -> pd.Series:
    """适用于可正可负 response 的归一化差异。"""

    return (a - b) / (a.abs() + b.abs() + EPSILON)


def compute_grid_and_unit_table(
    grid_n: int,
    pair_id: str,
    pair_cfg: dict[str, str],
    direction: dict,
    window_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算一个 grid/pair/direction/window 条件下的 grid 与 unit 表。"""

    cme_exp = pair_cfg["CME"]
    ume_exp = pair_cfg["UME"]
    direction_zero_based = direction["id"] - 1

    ume_units = load_unit_positions(ume_exp, "single_edge")
    cme_units = load_unit_positions(cme_exp, "double_edge")

    all_x = pd.concat([ume_units["x"], cme_units["x"]], ignore_index=True)
    all_y = pd.concat([ume_units["y"], cme_units["y"]], ignore_index=True)
    x_edges = np.linspace(all_x.min(), all_x.max(), grid_n + 1)
    y_edges = np.linspace(all_y.min(), all_y.max(), grid_n + 1)
    ume_units = assign_grid(ume_units, x_edges, y_edges, grid_n)
    cme_units = assign_grid(cme_units, x_edges, y_edges, grid_n)

    ume_response = load_log_onoff_ratio_response(ume_exp, "UME", direction_zero_based, window_name)
    cme_response = load_log_onoff_ratio_response(cme_exp, "CME", direction_zero_based, window_name)
    ume_unit_table = build_unit_table(ume_units, ume_response, "UME")
    cme_unit_table = build_unit_table(cme_units, cme_response, "CME")
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
    for col in ["UME_unit_count", "CME_unit_count"]:
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
    grid["UME_step_start"] = STEP_WINDOWS[window_name]["UME"][0]
    grid["UME_step_stop"] = STEP_WINDOWS[window_name]["UME"][1]
    grid["CME_step_start"] = STEP_WINDOWS[window_name]["CME"][0]
    grid["CME_step_stop"] = STEP_WINDOWS[window_name]["CME"][1]
    grid["response_metric"] = "log_onoff_ratio"
    grid["eps_ratio_hz"] = EPS_RATIO_HZ
    grid["min_units_per_grid_per_stim"] = MIN_UNITS_PER_GRID_PER_STIM_EXTENDED
    grid["valid_grid"] = (
        (grid["UME_unit_count"] >= MIN_UNITS_PER_GRID_PER_STIM_EXTENDED)
        & (grid["CME_unit_count"] >= MIN_UNITS_PER_GRID_PER_STIM_EXTENDED)
    )

    grid["delta_response"] = grid["UME_mean_fr_hz"] - grid["CME_mean_fr_hz"]
    grid["abs_delta_response"] = grid["delta_response"].abs()
    grid["norm_diff"] = normalized_signed_difference(grid["UME_mean_fr_hz"], grid["CME_mean_fr_hz"])
    grid["abs_norm_diff"] = grid["norm_diff"].abs()
    grid["difference_class_main"] = classify_delta(grid["delta_response"], grid["valid_grid"], MAIN_THRESHOLD_HZ)

    unit_df["grid_scale"] = f"{grid_n}x{grid_n}"
    unit_df["grid_n"] = grid_n
    unit_df["pair_id"] = pair_id
    unit_df["UME_exp"] = ume_exp
    unit_df["CME_exp"] = cme_exp
    unit_df["direction_id"] = f"{direction['id']:02d}"
    unit_df["direction_code"] = direction["code"]
    unit_df["direction_name"] = direction["name"]
    unit_df["window_name"] = window_name
    unit_df["response_metric"] = "log_onoff_ratio"
    return grid, unit_df


def permutation_for_condition(grid: pd.DataFrame, unit_df: pd.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    """对一个条件做 permutation null comparison。"""

    valid = grid[grid["valid_grid"]].copy()
    if valid.empty:
        return {k: np.nan for k in [
            "real_mean_abs_delta", "null_mean_mean_abs_delta", "null_std_mean_abs_delta", "p_mean_abs_delta",
            "real_different_fraction", "null_mean_different_fraction", "null_std_different_fraction", "p_different_fraction",
            "real_mean_abs_norm_diff", "null_mean_mean_abs_norm_diff", "null_std_mean_abs_norm_diff", "p_mean_abs_norm_diff",
        ]}

    real_mean_abs_delta = float(valid["abs_delta_response"].mean())
    real_different_fraction = float(valid["difference_class_main"].isin(["UME_higher", "CME_higher"]).mean())
    real_mean_abs_norm_diff = float(valid["abs_norm_diff"].mean())

    grid_samples: list[tuple[np.ndarray, int]] = []
    for gid in valid["grid_id"]:
        sub = unit_df[unit_df["grid_id"] == gid]
        ume_vals = sub.loc[sub["stimulus"] == "UME", "firing_rate_hz"].to_numpy(dtype=float)
        cme_vals = sub.loc[sub["stimulus"] == "CME", "firing_rate_hz"].to_numpy(dtype=float)
        if len(ume_vals) < MIN_UNITS_PER_GRID_PER_STIM_EXTENDED or len(cme_vals) < MIN_UNITS_PER_GRID_PER_STIM_EXTENDED:
            continue
        grid_samples.append((np.concatenate([ume_vals, cme_vals]), len(ume_vals)))

    null_abs = np.empty(N_PERM, dtype=float)
    null_diff = np.empty(N_PERM, dtype=float)
    null_norm = np.empty(N_PERM, dtype=float)

    for i in range(N_PERM):
        abs_values = []
        norm_values = []
        different_count = 0
        for merged, n_ume in grid_samples:
            perm = rng.permutation(merged)
            pseudo_ume = perm[:n_ume]
            pseudo_cme = perm[n_ume:]
            m_ume = float(np.mean(pseudo_ume))
            m_cme = float(np.mean(pseudo_cme))
            delta = m_ume - m_cme
            abs_values.append(abs(delta))
            norm_values.append(abs(delta / (abs(m_ume) + abs(m_cme) + EPSILON)))
            if abs(delta) > MAIN_THRESHOLD_HZ:
                different_count += 1
        null_abs[i] = float(np.mean(abs_values)) if abs_values else np.nan
        null_diff[i] = float(different_count / len(grid_samples)) if grid_samples else np.nan
        null_norm[i] = float(np.mean(norm_values)) if norm_values else np.nan

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
        "real_mean_abs_norm_diff": real_mean_abs_norm_diff,
        "null_mean_mean_abs_norm_diff": float(np.nanmean(null_norm)),
        "null_std_mean_abs_norm_diff": float(np.nanstd(null_norm, ddof=1)),
        "p_mean_abs_norm_diff": p_value(null_norm, real_mean_abs_norm_diff),
    }


# =============================================================================
# 4. 汇总、画图、报告
# =============================================================================


def summarize_significance(perm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for window_name, sub in perm.groupby("window_name"):
        row = {
            "window_name": window_name,
            "window_label_cn": STEP_WINDOWS[window_name]["label_cn"],
            "n_tests": int(len(sub)),
        }
        for pcol in ["p_mean_abs_delta", "p_different_fraction", "p_mean_abs_norm_diff"]:
            vals = sub[pcol].dropna().astype(float)
            row[f"{pcol}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{pcol}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{pcol}_sig_count_0p05"] = int((vals < 0.05).sum())
            row[f"{pcol}_sig_fraction_0p05"] = float((vals < 0.05).mean()) if len(vals) else np.nan
        rows.append(row)
    by_window = pd.DataFrame(rows)

    rows = []
    for keys, sub in perm.groupby(["window_name", "grid_scale", "grid_n", "pair_id"]):
        row = dict(zip(["window_name", "grid_scale", "grid_n", "pair_id"], keys))
        row["window_label_cn"] = STEP_WINDOWS[row["window_name"]]["label_cn"]
        row["n_direction_tests"] = int(len(sub))
        for pcol in ["p_mean_abs_delta", "p_different_fraction", "p_mean_abs_norm_diff"]:
            vals = sub[pcol].dropna().astype(float)
            row[f"{pcol}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{pcol}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{pcol}_sig_count_0p05"] = int((vals < 0.05).sum())
        rows.append(row)
    by_window_pair_scale = pd.DataFrame(rows)
    return by_window, by_window_pair_scale


def make_window_heatmap(summary: pd.DataFrame, pcol: str, stem: str) -> None:
    """绘制每个 step-window 下显著比例与最小 p 值概览图。"""

    order = list(STEP_WINDOWS.keys())
    sub = summary.set_index("window_name").reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))

    sig = sub[f"{pcol}_sig_fraction_0p05"].to_numpy(dtype=float)[None, :]
    im0 = axes[0].imshow(sig, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order, rotation=35, ha="right")
    axes[0].set_yticks([])
    axes[0].set_title(f"p<0.05 fraction: {pcol}")
    for j, value in enumerate(sig[0]):
        axes[0].text(j, 0, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.04, pad=0.02)

    minp = sub[f"{pcol}_min"].to_numpy(dtype=float)
    minus_log = -np.log10(np.clip(minp, 1e-6, 1.0))[None, :]
    im1 = axes[1].imshow(minus_log, cmap="magma", aspect="auto")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(order, rotation=35, ha="right")
    axes[1].set_yticks([])
    axes[1].set_title(f"minimum p: {pcol}")
    for j, value in enumerate(minp):
        axes[1].text(j, 0, f"{value:.3g}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.04, pad=0.02, label="-log10(min p)")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_report(perm: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = []
    lines.append("# MEA permutation test for log ON/OFF ratio response")
    lines.append("")
    lines.append("Response definition:")
    lines.append("")
    lines.append(f"`log((FR_ON + {EPS_RATIO_HZ}) / (FR_OFF + {EPS_RATIO_HZ}))`")
    lines.append("")
    lines.append(f"- Permutations per test: {N_PERM}")
    lines.append(f"- Valid grid rule: UME and CME each have at least {MIN_UNITS_PER_GRID_PER_STIM_EXTENDED} good units")
    lines.append(f"- Difference threshold for different_fraction: |Delta response| > {MAIN_THRESHOLD_HZ}")
    lines.append("")

    for pcol, real_col, null_col in [
        ("p_mean_abs_delta", "real_mean_abs_delta", "null_mean_mean_abs_delta"),
        ("p_different_fraction", "real_different_fraction", "null_mean_different_fraction"),
        ("p_mean_abs_norm_diff", "real_mean_abs_norm_diff", "null_mean_mean_abs_norm_diff"),
    ]:
        vals = perm[pcol].dropna().astype(float)
        lines.append(f"## {pcol}")
        lines.append("")
        lines.append(f"- Significant tests: {(vals < 0.05).sum()} / {len(vals)} ({(vals < 0.05).mean():.4f})")
        lines.append(f"- Min p: {vals.min():.4g}")
        lines.append(f"- Median p: {vals.median():.4g}")
        lines.append("")
        lines.append("| window | grid | pair | dir | p | real | null mean |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        for r in perm.sort_values(pcol).head(15).itertuples(index=False):
            lines.append(
                f"| {r.window_name} | {r.grid_scale} | {r.pair_id} | {r.direction_code} | "
                f"{getattr(r, pcol):.4g} | {getattr(r, real_col):.4g} | {getattr(r, null_col):.4g} |"
            )
        lines.append("")

    lines.append("## Window-level summary")
    lines.append("")
    lines.append("| window | sig abs-delta | sig different-fraction | sig norm-diff | min p abs-delta | min p diff | min p norm |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in summary.sort_values("p_mean_abs_delta_sig_fraction_0p05", ascending=False).itertuples(index=False):
        lines.append(
            f"| {r.window_name} | {r.p_mean_abs_delta_sig_fraction_0p05:.4f} | "
            f"{r.p_different_fraction_sig_fraction_0p05:.4f} | "
            f"{r.p_mean_abs_norm_diff_sig_fraction_0p05:.4f} | "
            f"{r.p_mean_abs_delta_min:.4g} | {r.p_different_fraction_min:.4g} | "
            f"{r.p_mean_abs_norm_diff_min:.4g} |"
        )

    (REPORT_DIR / "permutation_log_onoff_ratio_report.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 5. 主流程
# =============================================================================


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(RANDOM_SEED)
    log("Start MEA permutation test for log ON/OFF ratio response")
    log(f"Output directory: {OUT_DIR}")

    config = {
        "response_metric": "log_onoff_ratio",
        "formula": f"log((FR_ON + {EPS_RATIO_HZ}) / (FR_OFF + {EPS_RATIO_HZ}))",
        "EPS_RATIO_HZ": EPS_RATIO_HZ,
        "STEP_WINDOWS": STEP_WINDOWS,
        "GRID_SCALES": GRID_SCALES,
        "PAIR_CONFIG": PAIR_CONFIG,
        "DIRECTION_CONFIG": DIRECTION_CONFIG,
        "MIN_UNITS_PER_GRID_PER_STIM": MIN_UNITS_PER_GRID_PER_STIM_EXTENDED,
        "MAIN_THRESHOLD_HZ": MAIN_THRESHOLD_HZ,
        "N_PERM": N_PERM,
        "RANDOM_SEED": RANDOM_SEED,
    }
    (OUT_DIR / "analysis_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    grid_rows = []
    total = len(STEP_WINDOWS) * len(GRID_SCALES) * len(PAIR_CONFIG) * len(DIRECTION_CONFIG)
    counter = 0
    for window_name in STEP_WINDOWS:
        for grid_n in GRID_SCALES:
            for pair_id, pair_cfg in PAIR_CONFIG.items():
                for direction in DIRECTION_CONFIG:
                    counter += 1
                    log(f"[{counter}/{total}] window={window_name}, grid={grid_n}x{grid_n}, pair={pair_id}, dir={direction['code']}")
                    grid, unit_df = compute_grid_and_unit_table(grid_n, pair_id, pair_cfg, direction, window_name)
                    result = permutation_for_condition(grid, unit_df, rng)
                    base = {
                        "window_name": window_name,
                        "window_label_cn": STEP_WINDOWS[window_name]["label_cn"],
                        "grid_scale": f"{grid_n}x{grid_n}",
                        "grid_n": grid_n,
                        "pair_id": pair_id,
                        "direction_id": f"{direction['id']:02d}",
                        "direction_code": direction["code"],
                        "direction_name": direction["name"],
                        "n_perm": N_PERM,
                        "valid_grid_count": int(grid["valid_grid"].sum()),
                        "total_grid_count": int(len(grid)),
                        "response_metric": "log_onoff_ratio",
                        "eps_ratio_hz": EPS_RATIO_HZ,
                        "UME_step_start": STEP_WINDOWS[window_name]["UME"][0],
                        "UME_step_stop": STEP_WINDOWS[window_name]["UME"][1],
                        "CME_step_start": STEP_WINDOWS[window_name]["CME"][0],
                        "CME_step_stop": STEP_WINDOWS[window_name]["CME"][1],
                    }
                    rows.append({**base, **result})
                    valid = grid[grid["valid_grid"]]
                    grid_rows.append(
                        {
                            **base,
                            "mean_abs_delta_response": float(valid["abs_delta_response"].mean()) if len(valid) else np.nan,
                            "mean_abs_norm_diff": float(valid["abs_norm_diff"].mean()) if len(valid) else np.nan,
                            "different_fraction": float(valid["difference_class_main"].isin(["UME_higher", "CME_higher"]).mean())
                            if len(valid)
                            else np.nan,
                        }
                    )

    perm = pd.DataFrame(rows)
    grid_summary = pd.DataFrame(grid_rows)
    by_window, by_window_pair_scale = summarize_significance(perm)

    perm_path = TABLE_DIR / "permutation_log_onoff_ratio_summary.csv"
    grid_path = TABLE_DIR / "grid_metric_log_onoff_ratio_summary.csv"
    win_path = TABLE_DIR / "permutation_log_onoff_ratio_window_significance.csv"
    pair_path = TABLE_DIR / "permutation_log_onoff_ratio_pair_scale_significance.csv"
    perm.to_csv(perm_path, index=False, encoding="utf-8-sig")
    grid_summary.to_csv(grid_path, index=False, encoding="utf-8-sig")
    by_window.to_csv(win_path, index=False, encoding="utf-8-sig")
    by_window_pair_scale.to_csv(pair_path, index=False, encoding="utf-8-sig")

    for pcol in ["p_mean_abs_delta", "p_different_fraction", "p_mean_abs_norm_diff"]:
        make_window_heatmap(by_window, pcol, f"log_onoff_ratio_window_summary_{pcol}")

    write_report(perm, by_window)

    shutil.copy2(Path(__file__).resolve(), OUT_DIR / "run_MEA_permutation_log_onoff_ratio_analysis_code_snapshot.py")
    log(f"Saved permutation table: {perm_path}")
    log(f"Saved report: {REPORT_DIR / 'permutation_log_onoff_ratio_report.md'}")
    log("Finished MEA permutation test for log ON/OFF ratio response")
    (LOG_DIR / "analysis_log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
