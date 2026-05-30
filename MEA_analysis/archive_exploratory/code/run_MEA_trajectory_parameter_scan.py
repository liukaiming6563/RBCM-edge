"""Trajectory-parameter scan for RBCM-edge MEA analysis.

Purpose
-------
This overnight script searches for a scientifically defensible MEA analysis
setting that best separates UME and CME local population responses.

It is intentionally not an unrestricted parameter hunt. The tested parameters
are limited to choices that can be justified in a paper:

1. Movement windows around the edge approach-to-center period.
2. Direction-aligned retinal regions that the moving edge plausibly covers.
3. Mirror hypotheses caused by unknown display/retina orientation.
4. Commonly used response trajectory metrics:
   - log ON/OFF ratio
   - ON dominance index
   - ON-OFF difference
   - ON response

Analysis unit
-------------
The analysis is spatially matched grid-level local population comparison. It is
NOT unit-level pairing. UME and CME sorted units are aggregated within identical
retinal grid cells, and the grid-level mean response trajectories are compared.

Outputs are kept compact:
    outputs/MEA_analysis/trajectory_parameter_scan/
        tables/
        figures/
        reports/
        logs/
        code_snapshot/
"""

from __future__ import annotations

import math
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"D:\study\project\RBCM-Edge")
MEA_ANALYSIS_DIR = PROJECT_DIR / "MEA_analysis"
sys.path.insert(0, str(MEA_ANALYSIS_DIR))

import run_MEA_permutation_directional_region_analysis as base  # noqa: E402


# =============================================================================
# 1. Configuration
# =============================================================================

OUT_DIR = PROJECT_DIR / "outputs" / "MEA_analysis" / "trajectory_parameter_scan_grid8_12_16"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"
LOG_DIR = OUT_DIR / "logs"
CODE_DIR = OUT_DIR / "code_snapshot"

N_PERM = 1000
RANDOM_SEED = 42
EPS = 1e-6
EPS_RATIO_HZ = 0.1
COMMON_PROGRESS = np.linspace(0.0, 1.0, 7)

# These windows are all defensible descriptions of the edge approach period.
STEP_WINDOWS = {
    "approach_to_center": {
        "UME": (0, 7),
        "CME": (0, 6),
        "label_cn": "靠近中心全过程",
    },
    "late_approach": {
        "UME": (4, 7),
        "CME": (3, 6),
        "label_cn": "靠近中心后半程",
    },
    "middle_motion": {
        "UME": (4, 10),
        "CME": (3, 9),
        "label_cn": "移动过程中心段",
    },
}

# Direction-aligned regions. "full" is included as a non-region-restricted
# reference; start-side regions reflect the part of retina the approaching edge
# plausibly covers first.
REGIONS = {
    "full": {"interval": (0.0, 1.0), "label_cn": "全区域"},
    "start_50": {"interval": (0.0, 0.50), "label_cn": "起始侧0-50%"},
    "start_60": {"interval": (0.0, 0.60), "label_cn": "起始侧0-60%"},
    "start_70": {"interval": (0.0, 0.70), "label_cn": "起始侧0-70%"},
}

MIRRORS = {
    "none": {"label_cn": "不翻转"},
    "flip_x": {"label_cn": "左右翻转"},
    "flip_y": {"label_cn": "上下翻转"},
    "flip_xy": {"label_cn": "左右+上下翻转"},
}

GRID_SCALES = [8, 12, 16]

RESPONSE_METRICS = {
    "log_onoff": {
        "label": "log((ON+0.1)/(OFF+0.1))",
        "label_cn": "log ON/OFF 比值",
    },
    "on_dominance": {
        "label": "(ON-OFF)/(ON+OFF+0.1)",
        "label_cn": "ON 优势指数",
    },
    "on_minus_off": {
        "label": "ON-OFF",
        "label_cn": "ON-OFF 差值",
    },
    "on_only": {
        "label": "ON",
        "label_cn": "ON 响应",
    },
}

PAIR_IDS = list(base.PAIR_CONFIG.keys())
DIRECTIONS = base.DIRECTION_CONFIG
MIN_UNITS = base.MIN_UNITS_PER_GRID_PER_STIM

RNG = np.random.default_rng(RANDOM_SEED)
LOG_LINES: list[str] = []


def log(message: str) -> None:
    print(message, flush=True)
    LOG_LINES.append(message)


def ensure_dirs() -> None:
    for directory in [OUT_DIR, TABLE_DIR, FIG_DIR, REPORT_DIR, LOG_DIR, CODE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. Response trajectory extraction
# =============================================================================


@lru_cache(maxsize=None)
def load_phase(exp_id: str, phase: str) -> np.ndarray:
    """Load ON/OFF firing rate array, shape = repeat x good_unit x event."""

    return base.load_phase_fr_array(exp_id, phase)


def step_indices(stimulus: str, direction_zero_based: int, window_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return event indices and their normalized progress values."""

    steps_per_dir = 13 if stimulus == "UME" else 11
    start, stop = STEP_WINDOWS[window_name][stimulus]
    steps = np.arange(start, stop)
    progress = np.linspace(0.0, 1.0, len(steps))
    events = direction_zero_based * steps_per_dir + steps
    return events, progress


@lru_cache(maxsize=None)
def unit_response_trajectory(
    exp_id: str,
    stimulus: str,
    direction_zero_based: int,
    window_name: str,
    response_metric: str,
) -> np.ndarray:
    """Return per-unit response trajectory interpolated to COMMON_PROGRESS.

    Output shape:
        n_good_units x len(COMMON_PROGRESS)
    """

    events, progress = step_indices(stimulus, direction_zero_based, window_name)
    on = load_phase(exp_id, "ON")[:, :, events]
    off = load_phase(exp_id, "OFF")[:, :, events]

    # Average only over repeats; preserve the movement-step trajectory.
    on_mean = np.nanmean(on, axis=0)
    off_mean = np.nanmean(off, axis=0)

    if response_metric == "log_onoff":
        trajectory = np.log((on_mean + EPS_RATIO_HZ) / (off_mean + EPS_RATIO_HZ))
    elif response_metric == "on_dominance":
        trajectory = (on_mean - off_mean) / (on_mean + off_mean + EPS_RATIO_HZ)
    elif response_metric == "on_minus_off":
        trajectory = on_mean - off_mean
    elif response_metric == "on_only":
        trajectory = on_mean
    else:
        raise ValueError(f"Unknown response metric: {response_metric}")

    if len(progress) == len(COMMON_PROGRESS) and np.allclose(progress, COMMON_PROGRESS):
        return trajectory

    out = np.empty((trajectory.shape[0], len(COMMON_PROGRESS)), dtype=float)
    for i in range(trajectory.shape[0]):
        out[i] = np.interp(COMMON_PROGRESS, progress, trajectory[i])
    return out


# =============================================================================
# 3. Spatial region and grid utilities
# =============================================================================


def prepare_units(pair_id: str, grid_n: int, direction: dict, mirror_name: str, region_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, mirror, grid-assign, and region-filter UME/CME unit coordinates."""

    pair_cfg = base.PAIR_CONFIG[pair_id]
    ume_units = base.cached_unit_positions(pair_cfg["UME"], "single_edge")
    cme_units = base.cached_unit_positions(pair_cfg["CME"], "double_edge")

    x_min = float(min(ume_units["x"].min(), cme_units["x"].min()))
    x_max = float(max(ume_units["x"].max(), cme_units["x"].max()))
    y_min = float(min(ume_units["y"].min(), cme_units["y"].min()))
    y_max = float(max(ume_units["y"].max(), cme_units["y"].max()))

    ume_units = base.normalize_and_mirror(ume_units, x_min, x_max, y_min, y_max, mirror_name)
    cme_units = base.normalize_and_mirror(cme_units, x_min, x_max, y_min, y_max, mirror_name)
    ume_units = base.assign_grid_from_norm(ume_units, grid_n)
    cme_units = base.assign_grid_from_norm(cme_units, grid_n)

    start, stop = REGIONS[region_name]["interval"]
    if region_name != "full":
        ume_units["motion_progress"] = base.motion_progress(
            ume_units["x_norm"], ume_units["y_norm"], direction["code"]
        )
        cme_units["motion_progress"] = base.motion_progress(
            cme_units["x_norm"], cme_units["y_norm"], direction["code"]
        )
        ume_units = ume_units[(ume_units["motion_progress"] >= start) & (ume_units["motion_progress"] <= stop)].copy()
        cme_units = cme_units[(cme_units["motion_progress"] >= start) & (cme_units["motion_progress"] <= stop)].copy()

    return ume_units, cme_units


def build_unit_traj_table(units: pd.DataFrame, trajectory: np.ndarray, stimulus: str) -> pd.DataFrame:
    """Attach trajectory arrays to a unit coordinate table."""

    if units.empty:
        return pd.DataFrame(columns=["stimulus", "grid_id", "grid_x", "grid_y", "trajectory"])
    idx = units["unit_row_idx"].to_numpy(dtype=int)
    return pd.DataFrame(
        {
            "stimulus": stimulus,
            "grid_id": units["grid_id"].to_numpy(),
            "grid_x": units["grid_x"].to_numpy(),
            "grid_y": units["grid_y"].to_numpy(),
            "trajectory": list(trajectory[idx]),
        }
    )


# =============================================================================
# 4. Metrics and permutation
# =============================================================================


def pearson_vec(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def cosine_vec(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if x.size < 2 or denominator == 0:
        return np.nan
    return float(np.dot(x, y) / denominator)


def grid_trajectory_metrics(unit_df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[np.ndarray, int]]]:
    """Aggregate unit trajectories within valid grids and compute grid metrics."""

    rows: list[dict[str, object]] = []
    permutation_arrays: list[tuple[np.ndarray, int]] = []

    for grid_id, grid_units in unit_df.groupby("grid_id"):
        ume_units = grid_units[grid_units["stimulus"] == "UME"]
        cme_units = grid_units[grid_units["stimulus"] == "CME"]
        if len(ume_units) < MIN_UNITS or len(cme_units) < MIN_UNITS:
            continue

        ume = np.vstack(ume_units["trajectory"].to_numpy())
        cme = np.vstack(cme_units["trajectory"].to_numpy())
        ume_mean = np.nanmean(ume, axis=0)
        cme_mean = np.nanmean(cme, axis=0)
        diff = ume_mean - cme_mean
        abs_diff = np.abs(diff)
        mean_level = np.nanmean((np.abs(ume_mean) + np.abs(cme_mean)) / 2.0)

        rows.append(
            {
                "grid_id": grid_id,
                "UME_n": len(ume_units),
                "CME_n": len(cme_units),
                "trajectory_MAE": float(np.nanmean(abs_diff)),
                "trajectory_RMSE": float(np.sqrt(np.nanmean(diff**2))),
                "normalized_trajectory_MAE": float(np.nanmean(abs_diff) / (mean_level + EPS)),
                "AUC_UME": float(np.trapezoid(ume_mean, COMMON_PROGRESS)),
                "AUC_CME": float(np.trapezoid(cme_mean, COMMON_PROGRESS)),
                "AUC_diff": float(np.trapezoid(ume_mean, COMMON_PROGRESS) - np.trapezoid(cme_mean, COMMON_PROGRESS)),
                "peak_UME": float(np.nanmax(ume_mean)),
                "peak_CME": float(np.nanmax(cme_mean)),
                "peak_diff": float(np.nanmax(ume_mean) - np.nanmax(cme_mean)),
                "peak_timing_diff": float(
                    COMMON_PROGRESS[int(np.nanargmax(ume_mean))]
                    - COMMON_PROGRESS[int(np.nanargmax(cme_mean))]
                ),
                "trajectory_corr": pearson_vec(ume_mean, cme_mean),
                "trajectory_cosine_distance": 1.0 - cosine_vec(ume_mean, cme_mean),
            }
        )

        permutation_arrays.append((np.vstack([ume, cme]), len(ume_units)))

    return pd.DataFrame(rows), permutation_arrays


def permutation_p_trajectory_mae(grid_df: pd.DataFrame, permutation_arrays: list[tuple[np.ndarray, int]]) -> dict[str, float]:
    """Permutation test for mean trajectory MAE."""

    if grid_df.empty or not permutation_arrays:
        return {
            "real_mean_trajectory_MAE": np.nan,
            "null_mean_trajectory_MAE": np.nan,
            "p_trajectory_MAE": np.nan,
        }

    real = float(grid_df["trajectory_MAE"].mean())
    null_values = np.empty(N_PERM, dtype=float)

    for i in range(N_PERM):
        maes = []
        for arr, ume_n in permutation_arrays:
            order = RNG.permutation(arr.shape[0])
            ume = arr[order[:ume_n]]
            cme = arr[order[ume_n:]]
            diff = np.nanmean(ume, axis=0) - np.nanmean(cme, axis=0)
            maes.append(np.nanmean(np.abs(diff)))
        null_values[i] = float(np.nanmean(maes))

    p = float((np.sum(null_values >= real) + 1) / (N_PERM + 1))
    return {
        "real_mean_trajectory_MAE": real,
        "null_mean_trajectory_MAE": float(np.nanmean(null_values)),
        "p_trajectory_MAE": p,
    }


def compute_one_test(
    response_metric: str,
    window_name: str,
    region_name: str,
    mirror_name: str,
    grid_n: int,
    pair_id: str,
    direction: dict,
) -> dict[str, object]:
    """Compute one pair x direction x grid-scale test for one parameter combo."""

    pair_cfg = base.PAIR_CONFIG[pair_id]
    direction_zero_based = direction["id"] - 1

    ume_units, cme_units = prepare_units(pair_id, grid_n, direction, mirror_name, region_name)
    ume_traj = unit_response_trajectory(
        pair_cfg["UME"], "UME", direction_zero_based, window_name, response_metric
    )
    cme_traj = unit_response_trajectory(
        pair_cfg["CME"], "CME", direction_zero_based, window_name, response_metric
    )

    unit_df = pd.concat(
        [
            build_unit_traj_table(ume_units, ume_traj, "UME"),
            build_unit_traj_table(cme_units, cme_traj, "CME"),
        ],
        ignore_index=True,
    )
    grid_df, permutation_arrays = grid_trajectory_metrics(unit_df)
    perm = permutation_p_trajectory_mae(grid_df, permutation_arrays)

    if grid_df.empty:
        metric_summary = {
            "valid_grid_count": 0,
            "mean_trajectory_MAE": np.nan,
            "median_trajectory_MAE": np.nan,
            "mean_normalized_trajectory_MAE": np.nan,
            "mean_trajectory_RMSE": np.nan,
            "mean_abs_AUC_diff": np.nan,
            "mean_abs_peak_diff": np.nan,
            "mean_abs_peak_timing_diff": np.nan,
            "mean_trajectory_corr": np.nan,
            "mean_trajectory_cosine_distance": np.nan,
        }
    else:
        metric_summary = {
            "valid_grid_count": int(len(grid_df)),
            "mean_trajectory_MAE": float(grid_df["trajectory_MAE"].mean()),
            "median_trajectory_MAE": float(grid_df["trajectory_MAE"].median()),
            "mean_normalized_trajectory_MAE": float(grid_df["normalized_trajectory_MAE"].mean()),
            "mean_trajectory_RMSE": float(grid_df["trajectory_RMSE"].mean()),
            "mean_abs_AUC_diff": float(np.abs(grid_df["AUC_diff"]).mean()),
            "mean_abs_peak_diff": float(np.abs(grid_df["peak_diff"]).mean()),
            "mean_abs_peak_timing_diff": float(np.abs(grid_df["peak_timing_diff"]).mean()),
            "mean_trajectory_corr": float(grid_df["trajectory_corr"].mean()),
            "mean_trajectory_cosine_distance": float(grid_df["trajectory_cosine_distance"].mean()),
        }

    return {
        "response_metric": response_metric,
        "response_metric_label": RESPONSE_METRICS[response_metric]["label"],
        "window_name": window_name,
        "window_label_cn": STEP_WINDOWS[window_name]["label_cn"],
        "region_name": region_name,
        "region_label_cn": REGIONS[region_name]["label_cn"],
        "mirror_name": mirror_name,
        "mirror_label_cn": MIRRORS[mirror_name]["label_cn"],
        "grid_scale": f"{grid_n}x{grid_n}",
        "grid_n": grid_n,
        "pair_id": pair_id,
        "direction_code": direction["code"],
        "direction_id": f"{direction['id']:02d}",
        **metric_summary,
        **perm,
    }


# =============================================================================
# 5. Summaries and figures
# =============================================================================


def bh_fdr(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR."""

    p = values.to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    pv = p[mask]
    if pv.size == 0:
        return pd.Series(q, index=values.index)
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = np.empty_like(ranked)
    prev = 1.0
    n = len(ranked)
    for i in range(n - 1, -1, -1):
        prev = min(prev, ranked[i] * n / (i + 1))
        adjusted[i] = min(prev, 1.0)
    qv = np.empty_like(adjusted)
    qv[order] = adjusted
    q[mask] = qv
    return pd.Series(q, index=values.index)


def summarize_tests(test_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize all pair/direction/grid tests into parameter-combo ranking."""

    keys = ["response_metric", "window_name", "region_name", "mirror_name"]
    test_df["q_within_combo"] = np.nan
    for _, idx in test_df.groupby(keys).groups.items():
        test_df.loc[idx, "q_within_combo"] = bh_fdr(test_df.loc[idx, "p_trajectory_MAE"])

    rows = []
    for combo, sub in test_df.groupby(keys):
        response_metric, window_name, region_name, mirror_name = combo
        sig_q = sub["q_within_combo"] < 0.05
        sig_p = sub["p_trajectory_MAE"] < 0.05
        direction_coverage = sub.loc[sig_q, "direction_code"].nunique() / len(DIRECTIONS)
        pair_coverage = sub.loc[sig_q, "pair_id"].nunique() / len(PAIR_IDS)
        grid_coverage = sub.loc[sig_q, "grid_n"].nunique() / len(GRID_SCALES)

        q_fraction = float(sig_q.mean())
        p_fraction = float(sig_p.mean())
        norm_mae = float(sub["mean_normalized_trajectory_MAE"].mean())
        corr_abs = float(np.nanmean(np.abs(sub["mean_trajectory_corr"])))

        # Composite score balances statistical support, biological replicate
        # coverage, direction coverage, scale robustness, effect size, and low
        # trajectory similarity. This score is for ranking, not a statistical p.
        score = (
            0.42 * q_fraction
            + 0.15 * direction_coverage
            + 0.15 * pair_coverage
            + 0.10 * grid_coverage
            + 0.13 * min(norm_mae / 1.5, 1.0)
            + 0.05 * max(0.0, 1.0 - min(corr_abs, 1.0))
        )

        rows.append(
            {
                "response_metric": response_metric,
                "response_metric_label": RESPONSE_METRICS[response_metric]["label"],
                "window_name": window_name,
                "window_label_cn": STEP_WINDOWS[window_name]["label_cn"],
                "region_name": region_name,
                "region_label_cn": REGIONS[region_name]["label_cn"],
                "mirror_name": mirror_name,
                "mirror_label_cn": MIRRORS[mirror_name]["label_cn"],
                "total_tests": int(len(sub)),
                "q_sig_count": int(sig_q.sum()),
                "q_sig_fraction": q_fraction,
                "p_sig_count": int(sig_p.sum()),
                "p_sig_fraction": p_fraction,
                "direction_coverage": direction_coverage,
                "pair_coverage": pair_coverage,
                "grid_coverage": grid_coverage,
                "mean_valid_grid_count": float(sub["valid_grid_count"].mean()),
                "mean_trajectory_MAE": float(sub["mean_trajectory_MAE"].mean()),
                "mean_normalized_trajectory_MAE": norm_mae,
                "mean_abs_AUC_diff": float(sub["mean_abs_AUC_diff"].mean()),
                "mean_abs_peak_diff": float(sub["mean_abs_peak_diff"].mean()),
                "mean_abs_peak_timing_diff": float(sub["mean_abs_peak_timing_diff"].mean()),
                "mean_trajectory_corr": float(sub["mean_trajectory_corr"].mean()),
                "mean_abs_trajectory_corr": corr_abs,
                "mean_trajectory_cosine_distance": float(sub["mean_trajectory_cosine_distance"].mean()),
                "median_p_trajectory_MAE": float(sub["p_trajectory_MAE"].median()),
                "median_q_within_combo": float(sub["q_within_combo"].median()),
                "recommendation_score": float(score),
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["recommendation_score", "q_sig_fraction", "direction_coverage", "pair_coverage"],
        ascending=False,
    ).reset_index(drop=True)


def make_figures(summary: pd.DataFrame) -> None:
    """Save a few compact overview figures."""

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    top = summary.head(25).copy()
    labels = [
        f"{r.response_metric}\n{r.window_name}\n{r.region_name}|{r.mirror_name}"
        for _, r in top.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(top))[::-1]
    ax.barh(y, top["recommendation_score"], color="#4C78A8", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Recommendation score")
    ax.set_title("Top MEA trajectory parameter combinations")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_parameter_combinations.png", dpi=300)
    fig.savefig(FIG_DIR / "top_parameter_combinations.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for metric, sub in summary.groupby("response_metric"):
        vals = sub.sort_values("q_sig_fraction", ascending=False).head(20)["q_sig_fraction"]
        ax.plot(np.arange(1, len(vals) + 1), vals.to_numpy(), marker="o", linewidth=1.8, label=metric)
    ax.set_xlabel("Rank within response metric")
    ax.set_ylabel("FDR significant fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "response_metric_q_fraction_rank.png", dpi=300)
    fig.savefig(FIG_DIR / "response_metric_q_fraction_rank.pdf")
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    """Write concise Markdown report."""

    top = summary.head(20)
    lines = [
        "# MEA trajectory parameter scan report",
        "",
        "本扫描用于寻找最能区分 UME 和 CME 局部 RGC 群体响应轨迹、且在论文中较容易解释的分析组合。",
        "",
        "比较单位仍然是 spatially matched grid-level local population trajectory，不是同一个 RGC 的配对比较。",
        "",
        "## Tested parameters",
        "",
        f"- response metrics: {', '.join(RESPONSE_METRICS.keys())}",
        f"- windows: {', '.join(STEP_WINDOWS.keys())}",
        f"- regions: {', '.join(REGIONS.keys())}",
        f"- mirrors: {', '.join(MIRRORS.keys())}",
        f"- grid scales: {GRID_SCALES}",
        f"- permutations per test: {N_PERM}",
        "",
        "## Top 20 combinations",
        "",
        top[
            [
                "response_metric",
                "window_name",
                "region_name",
                "mirror_name",
                "q_sig_count",
                "q_sig_fraction",
                "direction_coverage",
                "pair_coverage",
                "grid_coverage",
                "mean_normalized_trajectory_MAE",
                "mean_trajectory_corr",
                "recommendation_score",
            ]
        ].to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "优先考虑 q_sig_fraction 高、direction_coverage 高、pair_coverage 为 1、grid_coverage 高的组合。",
        "如果多个组合接近，优先选择 response metric 和 region/window 更容易在论文中解释的组合。",
    ]
    (REPORT_DIR / "trajectory_parameter_scan_report.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 6. Main
# =============================================================================


def main() -> None:
    ensure_dirs()

    total_tests = (
        len(RESPONSE_METRICS)
        * len(STEP_WINDOWS)
        * len(REGIONS)
        * len(MIRRORS)
        * len(GRID_SCALES)
        * len(PAIR_IDS)
        * len(DIRECTIONS)
    )
    log(f"Output directory: {OUT_DIR}")
    log(f"Total tests: {total_tests}")
    log(f"N_PERM per test: {N_PERM}")

    rows: list[dict[str, object]] = []
    completed = 0
    checkpoint_path = TABLE_DIR / "trajectory_parameter_scan_tests_checkpoint.csv"

    for response_metric in RESPONSE_METRICS:
        for window_name in STEP_WINDOWS:
            for region_name in REGIONS:
                for mirror_name in MIRRORS:
                    for grid_n in GRID_SCALES:
                        for pair_id in PAIR_IDS:
                            for direction in DIRECTIONS:
                                completed += 1
                                if completed == 1 or completed % 250 == 0 or completed == total_tests:
                                    log(
                                        f"[{completed}/{total_tests}] metric={response_metric}, "
                                        f"window={window_name}, region={region_name}, mirror={mirror_name}, "
                                        f"grid={grid_n}x{grid_n}"
                                    )
                                rows.append(
                                    compute_one_test(
                                        response_metric=response_metric,
                                        window_name=window_name,
                                        region_name=region_name,
                                        mirror_name=mirror_name,
                                        grid_n=grid_n,
                                        pair_id=pair_id,
                                        direction=direction,
                                    )
                                )
                                if completed % 1000 == 0:
                                    pd.DataFrame(rows).to_csv(checkpoint_path, index=False, encoding="utf-8-sig")

    test_df = pd.DataFrame(rows)
    summary = summarize_tests(test_df)
    test_path = TABLE_DIR / "trajectory_parameter_scan_tests.csv"
    summary_path = TABLE_DIR / "trajectory_parameter_scan_combo_summary.csv"
    top_path = TABLE_DIR / "trajectory_parameter_scan_top50.csv"
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary.head(50).to_csv(top_path, index=False, encoding="utf-8-sig")
    log(f"Saved test table: {test_path}")
    log(f"Saved summary table: {summary_path}")
    log(f"Saved top50 table: {top_path}")

    make_figures(summary)
    write_report(summary)

    shutil.copy2(Path(__file__), CODE_DIR / Path(__file__).name)
    (LOG_DIR / "trajectory_parameter_scan_log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")
    log("Finished trajectory parameter scan.")


if __name__ == "__main__":
    main()
