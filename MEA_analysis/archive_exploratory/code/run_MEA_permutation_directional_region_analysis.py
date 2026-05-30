"""Directional region permutation analysis for RBCM-edge MEA data.

This script extends the spatially matched grid-level MEA analysis by adding
direction-dependent retinal region selection and mirror hypotheses.

Analysis idea
-------------
The user wants to compare UME and CME responses only within the spatial region
that is likely covered by the moving edge during a selected motion interval.
For example, for left-to-right motion (R), the selected region can be the
leftmost 70% or 60% of the retinal field. For diagonal or vertical directions,
we apply the same idea by projecting each unit's spatial position onto the
motion axis and selecting units whose normalized progress along that axis falls
within the selected interval.

Important interpretation boundary
---------------------------------
This is still NOT a unit-level paired analysis. UME and CME sorted units are not
matched one-to-one. Each valid spatial grid is treated as a local RGC
population region, and UME/CME labels are shuffled within each valid grid to
build the permutation null.
"""

from __future__ import annotations

import math
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from run_MEA_grid_full_analysis import (
    MEA_DIR,
    PAIR_CONFIG,
    DIRECTION_CONFIG,
    MIN_UNITS_PER_GRID_PER_STIM,
    MAIN_THRESHOLD_HZ,
    EPSILON,
    RANDOM_SEED,
    load_unit_positions,
)


# =============================================================================
# 1. User-facing configuration
# =============================================================================

PROJECT_DIR = Path(r"D:\study\project\RBCM-Edge")
OUT_DIR = PROJECT_DIR / "outputs" / "MEA_analysis" / "permutation_directional_region_analysis_reduced"
TABLE_DIR = OUT_DIR / "tables"
REPORT_DIR = OUT_DIR / "reports"
LOG_DIR = OUT_DIR / "logs"
CODE_DIR = OUT_DIR / "code_snapshot"

# Use the same number of permutations as the previous log ON/OFF ratio analysis.
N_PERM = 1000
EPS_RATIO_HZ = 0.1

# Movement windows are half-open intervals: [start, stop).
STEP_WINDOWS = {
    "approach_to_center": {"UME": (0, 7), "CME": (0, 6), "label_cn": "靠近中心全过程"},
    "late_approach": {"UME": (4, 7), "CME": (3, 6), "label_cn": "靠近中心后半程"},
}

# Region intervals are defined on normalized motion progress p in [0, 1].
# p=0 means the motion-start side and p=1 means the motion-end side.
REGION_CONFIG = {
    "start_60": {"interval": (0.0, 0.60), "label_cn": "起始侧0-60%"},
    "band_30_60": {"interval": (0.30, 0.60), "label_cn": "运动进程30-60%"},
}

# User-requested reduced grid scales for this directional-region analysis.
# Keeping fewer scales makes the enlarged condition grid easier to interpret:
# 6 movement windows x 4 mirror hypotheses x 4 region choices x 5 grid scales
# x 3 paired retinas x 8 directions = 11520 permutation tests.
GRID_SCALES_TO_RUN = [8, 12]

# Mirror hypotheses. The transform is applied to normalized retinal coordinates.
MIRROR_CONFIG = {
    "none": {"flip_x": False, "flip_y": False, "label_cn": "不翻转"},
    "flip_x": {"flip_x": True, "flip_y": False, "label_cn": "左右翻转"},
    "flip_y": {"flip_x": False, "flip_y": True, "label_cn": "上下翻转"},
    "flip_xy": {"flip_x": True, "flip_y": True, "label_cn": "左右+上下翻转"},
}

DIRECTION_ORDER = ["R", "RU", "U", "LU", "L", "LD", "D", "RD"]

RNG = np.random.default_rng(RANDOM_SEED)
LOG_LINES: list[str] = []


def log(message: str) -> None:
    """Print and store a log message."""

    print(message)
    LOG_LINES.append(message)


def ensure_dirs() -> None:
    """Create output directories."""

    for directory in [OUT_DIR, TABLE_DIR, REPORT_DIR, LOG_DIR, CODE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. Data loading and response extraction
# =============================================================================


@lru_cache(maxsize=None)
def load_phase_fr_array(exp_id: str, phase: str) -> np.ndarray:
    """Load firing-rate array with shape repeat x good_unit x event.

    Parameters
    ----------
    exp_id:
        Recording id such as "000031".
    phase:
        "ON" or "OFF".
    """

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


@lru_cache(maxsize=None)
def cached_unit_positions(exp_id: str, stim_class: str) -> pd.DataFrame:
    """Load good-unit spatial coordinates and cache them."""

    return load_unit_positions(exp_id, stim_class).copy()


def event_indices(stimulus: str, direction_zero_based: int, window_name: str) -> np.ndarray:
    """Return event indices for a stimulus, direction, and step window."""

    steps_per_dir = 13 if stimulus == "UME" else 11
    start, stop = STEP_WINDOWS[window_name][stimulus]
    if not (0 <= start < stop <= steps_per_dir):
        raise ValueError(f"Invalid window for {stimulus} {window_name}: {(start, stop)}")
    return direction_zero_based * steps_per_dir + np.arange(start, stop)


@lru_cache(maxsize=None)
def load_log_onoff_response(
    exp_id: str,
    stimulus: str,
    direction_zero_based: int,
    window_name: str,
) -> np.ndarray:
    """Compute per-unit log ON/OFF ratio response.

    The output has shape [n_good_units]. For each unit:

        response = log((mean_ON + 0.1) / (mean_OFF + 0.1))

    where mean_ON and mean_OFF are averaged across repeats and selected steps.
    """

    idx = event_indices(stimulus, direction_zero_based, window_name)
    on = load_phase_fr_array(exp_id, "ON")
    off = load_phase_fr_array(exp_id, "OFF")
    if int(idx.max()) >= on.shape[2] or int(idx.max()) >= off.shape[2]:
        raise IndexError(f"Event index out of range for {exp_id} {stimulus} {window_name}")
    on_mean = np.nanmean(on[:, :, idx], axis=(0, 2))
    off_mean = np.nanmean(off[:, :, idx], axis=(0, 2))
    return np.log((on_mean + EPS_RATIO_HZ) / (off_mean + EPS_RATIO_HZ))


# =============================================================================
# 3. Spatial transforms and directional region selection
# =============================================================================


def normalize_and_mirror(
    units: pd.DataFrame,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    mirror_name: str,
) -> pd.DataFrame:
    """Normalize unit coordinates to [0, 1] and apply a mirror hypothesis."""

    cfg = MIRROR_CONFIG[mirror_name]
    out = units.copy()
    out["x_norm"] = (out["x"] - x_min) / (x_max - x_min + EPSILON)
    out["y_norm"] = (out["y"] - y_min) / (y_max - y_min + EPSILON)
    out["x_norm"] = out["x_norm"].clip(0.0, 1.0)
    out["y_norm"] = out["y_norm"].clip(0.0, 1.0)
    if cfg["flip_x"]:
        out["x_norm"] = 1.0 - out["x_norm"]
    if cfg["flip_y"]:
        out["y_norm"] = 1.0 - out["y_norm"]
    return out


def motion_progress(x: pd.Series, y: pd.Series, direction_code: str) -> pd.Series:
    """Compute normalized progress from motion-start side to motion-end side.

    Canonical convention:
    - R starts at left and moves right: p = x
    - U starts at bottom and moves top: p = 1 - y
    - Diagonal directions use the average progress along the two axes.

    Because the experimental display/retina orientation is uncertain, this
    canonical convention is repeated under all mirror hypotheses.
    """

    if direction_code == "R":
        return x
    if direction_code == "RU":
        return (x + (1.0 - y)) / 2.0
    if direction_code == "U":
        return 1.0 - y
    if direction_code == "LU":
        return ((1.0 - x) + (1.0 - y)) / 2.0
    if direction_code == "L":
        return 1.0 - x
    if direction_code == "LD":
        return ((1.0 - x) + y) / 2.0
    if direction_code == "D":
        return y
    if direction_code == "RD":
        return (x + y) / 2.0
    raise ValueError(f"Unknown direction code: {direction_code}")


def assign_grid_from_norm(units: pd.DataFrame, grid_n: int) -> pd.DataFrame:
    """Assign grid ids on normalized mirrored coordinates."""

    out = units.copy()
    out["grid_x"] = np.clip(np.floor(out["x_norm"].to_numpy() * grid_n).astype(int), 0, grid_n - 1)
    out["grid_y"] = np.clip(np.floor(out["y_norm"].to_numpy() * grid_n).astype(int), 0, grid_n - 1)
    out["grid_id"] = out["grid_y"].astype(str).str.zfill(2) + "_" + out["grid_x"].astype(str).str.zfill(2)
    return out


def build_unit_table(units: pd.DataFrame, response: np.ndarray, stimulus: str) -> pd.DataFrame:
    """Attach response values to unit coordinates."""

    max_idx = int(units["unit_row_idx"].max())
    if max_idx >= response.size:
        raise ValueError(f"{stimulus} unit_row_idx exceeds response length: {max_idx} >= {response.size}")
    out = units.copy()
    out["stimulus"] = stimulus
    out["response"] = response[out["unit_row_idx"].to_numpy(dtype=int)]
    return out


# =============================================================================
# 4. Grid aggregation and permutation test
# =============================================================================


def aggregate_pair_grid(unit_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate unit responses to matched grid-level UME/CME means."""

    ume = (
        unit_df[unit_df["stimulus"].eq("UME")]
        .groupby(["grid_id", "grid_x", "grid_y"], as_index=False)
        .agg(UME_unit_count=("response", "size"), UME_mean_response=("response", "mean"))
    )
    cme = (
        unit_df[unit_df["stimulus"].eq("CME")]
        .groupby(["grid_id", "grid_x", "grid_y"], as_index=False)
        .agg(CME_unit_count=("response", "size"), CME_mean_response=("response", "mean"))
    )
    grid = pd.merge(ume, cme, on=["grid_id", "grid_x", "grid_y"], how="outer")
    grid["UME_unit_count"] = grid["UME_unit_count"].fillna(0).astype(int)
    grid["CME_unit_count"] = grid["CME_unit_count"].fillna(0).astype(int)
    grid["valid_grid"] = (
        (grid["UME_unit_count"] >= MIN_UNITS_PER_GRID_PER_STIM)
        & (grid["CME_unit_count"] >= MIN_UNITS_PER_GRID_PER_STIM)
    )
    grid["delta_response"] = grid["UME_mean_response"] - grid["CME_mean_response"]
    grid["abs_delta_response"] = grid["delta_response"].abs()
    grid["abs_norm_diff"] = (
        grid["delta_response"].abs()
        / (grid["UME_mean_response"].abs() + grid["CME_mean_response"].abs() + EPSILON)
    )
    return grid, unit_df[unit_df["grid_id"].isin(grid.loc[grid["valid_grid"], "grid_id"])]


def permutation_test(grid_valid: pd.DataFrame, unit_valid: pd.DataFrame) -> dict[str, float]:
    """Permutation null by shuffling UME/CME labels within each valid grid."""

    real_abs_delta = grid_valid["abs_delta_response"].to_numpy(dtype=float)
    real_abs_norm = grid_valid["abs_norm_diff"].to_numpy(dtype=float)
    real_mean_abs_delta = float(np.nanmean(real_abs_delta)) if real_abs_delta.size else np.nan
    real_different_fraction = float(np.nanmean(real_abs_delta > MAIN_THRESHOLD_HZ)) if real_abs_delta.size else np.nan
    real_mean_abs_norm_diff = float(np.nanmean(real_abs_norm)) if real_abs_norm.size else np.nan

    grid_arrays: list[tuple[np.ndarray, int]] = []
    for grid_id in grid_valid["grid_id"]:
        values = unit_valid.loc[unit_valid["grid_id"].eq(grid_id), ["stimulus", "response"]]
        ume_n = int(values["stimulus"].eq("UME").sum())
        cme_n = int(values["stimulus"].eq("CME").sum())
        if ume_n >= MIN_UNITS_PER_GRID_PER_STIM and cme_n >= MIN_UNITS_PER_GRID_PER_STIM:
            grid_arrays.append((values["response"].to_numpy(dtype=float), ume_n))

    if not grid_arrays:
        return {
            "n_perm": N_PERM,
            "valid_grid_count": 0,
            "real_mean_abs_delta": np.nan,
            "null_mean_abs_delta": np.nan,
            "p_mean_abs_delta": np.nan,
            "real_different_fraction": np.nan,
            "null_different_fraction": np.nan,
            "p_different_fraction": np.nan,
            "real_mean_abs_norm_diff": np.nan,
            "null_mean_abs_norm_diff": np.nan,
            "p_mean_abs_norm_diff": np.nan,
        }

    null_abs_delta = np.empty(N_PERM, dtype=float)
    null_diff_fraction = np.empty(N_PERM, dtype=float)
    null_abs_norm = np.empty(N_PERM, dtype=float)

    for i in range(N_PERM):
        deltas = []
        norm_diffs = []
        for arr, ume_n in grid_arrays:
            shuffled = RNG.permutation(arr)
            ume = shuffled[:ume_n]
            cme = shuffled[ume_n:]
            ume_mean = float(np.nanmean(ume))
            cme_mean = float(np.nanmean(cme))
            delta = ume_mean - cme_mean
            deltas.append(delta)
            norm_diffs.append(abs(delta) / (abs(ume_mean) + abs(cme_mean) + EPSILON))
        deltas = np.asarray(deltas, dtype=float)
        norm_diffs = np.asarray(norm_diffs, dtype=float)
        null_abs_delta[i] = float(np.nanmean(np.abs(deltas)))
        null_diff_fraction[i] = float(np.nanmean(np.abs(deltas) > MAIN_THRESHOLD_HZ))
        null_abs_norm[i] = float(np.nanmean(norm_diffs))

    def p_greater(null_values: np.ndarray, real_value: float) -> float:
        if not np.isfinite(real_value):
            return np.nan
        null_values = null_values[np.isfinite(null_values)]
        if null_values.size == 0:
            return np.nan
        return float((np.sum(null_values >= real_value) + 1) / (null_values.size + 1))

    return {
        "n_perm": N_PERM,
        "valid_grid_count": int(len(grid_arrays)),
        "real_mean_abs_delta": real_mean_abs_delta,
        "null_mean_abs_delta": float(np.nanmean(null_abs_delta)),
        "p_mean_abs_delta": p_greater(null_abs_delta, real_mean_abs_delta),
        "real_different_fraction": real_different_fraction,
        "null_different_fraction": float(np.nanmean(null_diff_fraction)),
        "p_different_fraction": p_greater(null_diff_fraction, real_different_fraction),
        "real_mean_abs_norm_diff": real_mean_abs_norm_diff,
        "null_mean_abs_norm_diff": float(np.nanmean(null_abs_norm)),
        "p_mean_abs_norm_diff": p_greater(null_abs_norm, real_mean_abs_norm_diff),
    }


# =============================================================================
# 5. Analysis loop and summaries
# =============================================================================


def compute_combo(
    window_name: str,
    mirror_name: str,
    region_name: str,
    grid_n: int,
    pair_id: str,
    direction: dict,
) -> dict[str, object]:
    """Compute one condition combination."""

    pair_cfg = PAIR_CONFIG[pair_id]
    ume_exp = pair_cfg["UME"]
    cme_exp = pair_cfg["CME"]
    direction_zero_based = direction["id"] - 1

    ume_units = cached_unit_positions(ume_exp, "single_edge")
    cme_units = cached_unit_positions(cme_exp, "double_edge")
    x_min = float(min(ume_units["x"].min(), cme_units["x"].min()))
    x_max = float(max(ume_units["x"].max(), cme_units["x"].max()))
    y_min = float(min(ume_units["y"].min(), cme_units["y"].min()))
    y_max = float(max(ume_units["y"].max(), cme_units["y"].max()))

    ume_units = normalize_and_mirror(ume_units, x_min, x_max, y_min, y_max, mirror_name)
    cme_units = normalize_and_mirror(cme_units, x_min, x_max, y_min, y_max, mirror_name)
    ume_units = assign_grid_from_norm(ume_units, grid_n)
    cme_units = assign_grid_from_norm(cme_units, grid_n)

    ume_units["motion_progress"] = motion_progress(ume_units["x_norm"], ume_units["y_norm"], direction["code"])
    cme_units["motion_progress"] = motion_progress(cme_units["x_norm"], cme_units["y_norm"], direction["code"])
    start, stop = REGION_CONFIG[region_name]["interval"]
    ume_units = ume_units[(ume_units["motion_progress"] >= start) & (ume_units["motion_progress"] <= stop)].copy()
    cme_units = cme_units[(cme_units["motion_progress"] >= start) & (cme_units["motion_progress"] <= stop)].copy()

    ume_response = load_log_onoff_response(ume_exp, "UME", direction_zero_based, window_name)
    cme_response = load_log_onoff_response(cme_exp, "CME", direction_zero_based, window_name)
    unit_df = pd.concat(
        [
            build_unit_table(ume_units, ume_response, "UME"),
            build_unit_table(cme_units, cme_response, "CME"),
        ],
        ignore_index=True,
    )

    grid, unit_valid = aggregate_pair_grid(unit_df)
    grid_valid = grid[grid["valid_grid"]].copy()
    stats = permutation_test(grid_valid, unit_valid)
    stats.update(
        {
            "window_name": window_name,
            "window_label_cn": STEP_WINDOWS[window_name]["label_cn"],
            "mirror_name": mirror_name,
            "mirror_label_cn": MIRROR_CONFIG[mirror_name]["label_cn"],
            "region_name": region_name,
            "region_label_cn": REGION_CONFIG[region_name]["label_cn"],
            "region_start": start,
            "region_stop": stop,
            "grid_scale": f"{grid_n}x{grid_n}",
            "grid_n": grid_n,
            "pair_id": pair_id,
            "UME_exp": ume_exp,
            "CME_exp": cme_exp,
            "direction_id": f"{direction['id']:02d}",
            "direction_code": direction["code"],
            "direction_name": direction["name"],
            "UME_region_unit_count": int((unit_df["stimulus"] == "UME").sum()),
            "CME_region_unit_count": int((unit_df["stimulus"] == "CME").sum()),
            "response_metric": "log_onoff_ratio",
            "eps_ratio_hz": EPS_RATIO_HZ,
        }
    )
    return stats


def bh_fdr(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction."""

    p = values.to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    p_valid = p[mask]
    if p_valid.size == 0:
        return pd.Series(q, index=values.index)
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    out = np.empty_like(ranked)
    prev = 1.0
    n = ranked.size
    for i in range(n - 1, -1, -1):
        rank = i + 1
        prev = min(prev, ranked[i] * n / rank)
        out[i] = min(prev, 1.0)
    q_valid = np.empty_like(out)
    q_valid[order] = out
    q[mask] = q_valid
    return pd.Series(q, index=values.index)


def category_from_p(p: float, q: float) -> str:
    """Convert p/q values into the user's simple qualitative labels."""

    if np.isfinite(q) and q < 0.05:
        return "显著不一样"
    if np.isfinite(p) and p < 0.05:
        return "有点不一样"
    if np.isfinite(p) and p < 0.15:
        return "几乎一样"
    return "完全一样"


def write_markdown_report(results: pd.DataFrame, condition_summary: pd.DataFrame) -> Path:
    """Write a concise Markdown report with the best conditions."""

    report_path = REPORT_DIR / "directional_region_permutation_report.md"
    top = condition_summary.sort_values(
        ["sig_q_count", "sig_p_count", "median_p_mean_abs_delta"],
        ascending=[False, False, True],
    ).head(20)

    lines = [
        "# Directional-region log ON/OFF ratio permutation analysis",
        "",
        "响应指标：`log((ON + 0.1) / (OFF + 0.1))`。",
        "",
        "区域筛选：对每个方向沿运动方向计算 normalized progress，并筛选起始侧 0-70%、起始侧 0-60%、40-70%、30-60%。",
        "",
        "镜像条件：none、flip_x、flip_y、flip_xy。",
        "",
        "检验单位：spatially matched grid-level local population，不是 unit-level pairing。",
        "",
        "## Top conditions",
        "",
        top[
            [
                "window_name",
                "mirror_name",
                "region_name",
                "sig_q_count",
                "sig_p_count",
                "total_tests",
                "sig_q_fraction",
                "sig_p_fraction",
                "median_p_mean_abs_delta",
                "mean_real_mean_abs_delta",
            ]
        ].to_markdown(index=False),
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    ensure_dirs()
    log(f"Output directory: {OUT_DIR}")
    total = (
        len(STEP_WINDOWS)
        * len(MIRROR_CONFIG)
        * len(REGION_CONFIG)
        * len(GRID_SCALES_TO_RUN)
        * len(PAIR_CONFIG)
        * len(DIRECTION_CONFIG)
    )
    log(f"Total tests: {total}")

    rows = []
    i = 0
    for window_name in STEP_WINDOWS:
        for mirror_name in MIRROR_CONFIG:
            for region_name in REGION_CONFIG:
                for grid_n in GRID_SCALES_TO_RUN:
                    for pair_id in PAIR_CONFIG:
                        for direction in DIRECTION_CONFIG:
                            i += 1
                            if i == 1 or i % 200 == 0 or i == total:
                                log(
                                    f"[{i}/{total}] window={window_name}, mirror={mirror_name}, "
                                    f"region={region_name}, grid={grid_n}x{grid_n}"
                                )
                            rows.append(compute_combo(window_name, mirror_name, region_name, grid_n, pair_id, direction))

    results = pd.DataFrame(rows)
    results["q_global_mean_abs_delta"] = bh_fdr(results["p_mean_abs_delta"])
    results["category_global"] = [
        category_from_p(p, q) for p, q in zip(results["p_mean_abs_delta"], results["q_global_mean_abs_delta"])
    ]
    result_path = TABLE_DIR / "directional_region_permutation_summary.csv"
    results.to_csv(result_path, index=False, encoding="utf-8-sig")
    log(f"Saved full result table: {result_path}")

    condition_summary = (
        results.groupby(["window_name", "mirror_name", "region_name"], as_index=False)
        .agg(
            total_tests=("p_mean_abs_delta", "size"),
            sig_p_count=("p_mean_abs_delta", lambda s: int(np.sum(np.asarray(s, dtype=float) < 0.05))),
            sig_q_count=("q_global_mean_abs_delta", lambda s: int(np.sum(np.asarray(s, dtype=float) < 0.05))),
            median_p_mean_abs_delta=("p_mean_abs_delta", "median"),
            mean_real_mean_abs_delta=("real_mean_abs_delta", "mean"),
            mean_valid_grid_count=("valid_grid_count", "mean"),
        )
    )
    condition_summary["sig_p_fraction"] = condition_summary["sig_p_count"] / condition_summary["total_tests"]
    condition_summary["sig_q_fraction"] = condition_summary["sig_q_count"] / condition_summary["total_tests"]
    summary_path = TABLE_DIR / "directional_region_condition_summary.csv"
    condition_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    log(f"Saved condition summary: {summary_path}")

    # Per-pair summary is useful for deciding whether a condition is supported by
    # all three paired retina preparations or by one pair only.
    pair_summary = (
        results.groupby(["window_name", "mirror_name", "region_name", "pair_id"], as_index=False)
        .agg(
            total_tests=("p_mean_abs_delta", "size"),
            sig_p_count=("p_mean_abs_delta", lambda s: int(np.sum(np.asarray(s, dtype=float) < 0.05))),
            sig_q_count=("q_global_mean_abs_delta", lambda s: int(np.sum(np.asarray(s, dtype=float) < 0.05))),
            median_p_mean_abs_delta=("p_mean_abs_delta", "median"),
            mean_real_mean_abs_delta=("real_mean_abs_delta", "mean"),
        )
    )
    pair_summary["sig_p_fraction"] = pair_summary["sig_p_count"] / pair_summary["total_tests"]
    pair_summary["sig_q_fraction"] = pair_summary["sig_q_count"] / pair_summary["total_tests"]
    pair_summary_path = TABLE_DIR / "directional_region_pair_summary.csv"
    pair_summary.to_csv(pair_summary_path, index=False, encoding="utf-8-sig")
    log(f"Saved pair summary: {pair_summary_path}")

    report_path = write_markdown_report(results, condition_summary)
    log(f"Saved report: {report_path}")

    code_snapshot = CODE_DIR / "run_MEA_permutation_directional_region_analysis_code_snapshot.py"
    shutil.copy2(Path(__file__), code_snapshot)
    (LOG_DIR / "analysis_log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")
    log("Finished directional-region permutation analysis.")


if __name__ == "__main__":
    main()
