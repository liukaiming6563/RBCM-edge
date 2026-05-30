"""Focused MEA trajectory selector for the RBCM-edge paper.

This script is a compact follow-up to the broad overnight parameter scan.  Its
goal is to select one defensible analysis setting that best supports the paper
claim that UME and CME evoke different retinal population responses.

The scan is intentionally narrow:

1. It only uses trajectory-level analysis, because the previous scan showed
   that UME/CME differences are strongest in step-resolved response profiles.
2. It only tests response metrics with clear neurophysiological meaning:
   log ON/OFF ratio and ON dominance index.
3. It only keeps two movement windows and two spatial region choices that can
   be justified in the manuscript.
4. It tests all four mirror hypotheses because the display/retina orientation
   record was lost.

The output is small: one complete test table, one combination-ranking table,
one top table, and two overview figures.
"""

from __future__ import annotations

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


OUT_DIR = PROJECT_DIR / "outputs" / "MEA_analysis" / "trajectory_focused_selector"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"
LOG_DIR = OUT_DIR / "logs"
CODE_DIR = OUT_DIR / "code_snapshot"

N_PERM = 1000
RANDOM_SEED = 42
COMMON_PROGRESS = np.linspace(0.0, 1.0, 7)
EPS = 1e-6

WINDOWS = {
    "approach_to_center": {"UME": (0, 7), "CME": (0, 6), "label_cn": "靠近中心全过程"},
    "middle_motion": {"UME": (4, 10), "CME": (3, 9), "label_cn": "移动过程中心段"},
}

REGIONS = {
    "full": {"interval": (0.0, 1.0), "label_cn": "全视网膜记录区域"},
    "start_60": {"interval": (0.0, 0.60), "label_cn": "运动起始侧0-60%"},
}

MIRRORS = {
    "none": {"label_cn": "不翻转"},
    "flip_x": {"label_cn": "左右翻转"},
    "flip_y": {"label_cn": "上下翻转"},
    "flip_xy": {"label_cn": "左右+上下翻转"},
}

GRID_SCALES = [8, 12, 16]
PAIR_IDS = list(base.PAIR_CONFIG.keys())
DIRECTIONS = base.DIRECTION_CONFIG
MIN_UNITS = base.MIN_UNITS_PER_GRID_PER_STIM

# A small, defensible set of response definitions.
RESPONSE_VARIANTS = {
    "log_onoff_eps005": {
        "metric": "log_onoff",
        "epsilon": 0.05,
        "label": "log((ON+0.05)/(OFF+0.05))",
    },
    "log_onoff_eps010": {
        "metric": "log_onoff",
        "epsilon": 0.10,
        "label": "log((ON+0.10)/(OFF+0.10))",
    },
    "log_onoff_eps020": {
        "metric": "log_onoff",
        "epsilon": 0.20,
        "label": "log((ON+0.20)/(OFF+0.20))",
    },
    "on_dominance_eps010": {
        "metric": "on_dominance",
        "epsilon": 0.10,
        "label": "(ON-OFF)/(ON+OFF+0.10)",
    },
}

RNG = np.random.default_rng(RANDOM_SEED)
LOG_LINES: list[str] = []


def log(message: str) -> None:
    print(message, flush=True)
    LOG_LINES.append(message)


def ensure_dirs() -> None:
    for directory in [OUT_DIR, TABLE_DIR, FIG_DIR, REPORT_DIR, LOG_DIR, CODE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=None)
def load_phase(exp_id: str, phase: str) -> np.ndarray:
    """Load firing-rate array with shape repeat x good_unit x event."""

    return base.load_phase_fr_array(exp_id, phase)


def event_indices(stimulus: str, direction_zero_based: int, window_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return selected event indices and normalized movement progress."""

    steps_per_dir = 13 if stimulus == "UME" else 11
    start, stop = WINDOWS[window_name][stimulus]
    steps = np.arange(start, stop)
    progress = np.linspace(0.0, 1.0, len(steps))
    events = direction_zero_based * steps_per_dir + steps
    return events, progress


@lru_cache(maxsize=None)
def unit_trajectory(
    exp_id: str,
    stimulus: str,
    direction_zero_based: int,
    window_name: str,
    response_variant: str,
) -> np.ndarray:
    """Return per-unit response trajectories interpolated to seven progress points."""

    variant = RESPONSE_VARIANTS[response_variant]
    events, progress = event_indices(stimulus, direction_zero_based, window_name)
    on = load_phase(exp_id, "ON")[:, :, events]
    off = load_phase(exp_id, "OFF")[:, :, events]
    on_mean = np.nanmean(on, axis=0)
    off_mean = np.nanmean(off, axis=0)
    eps = float(variant["epsilon"])

    if variant["metric"] == "log_onoff":
        trajectory = np.log((on_mean + eps) / (off_mean + eps))
    elif variant["metric"] == "on_dominance":
        trajectory = (on_mean - off_mean) / (on_mean + off_mean + eps)
    else:
        raise ValueError(f"Unsupported metric variant: {response_variant}")

    if len(progress) == len(COMMON_PROGRESS) and np.allclose(progress, COMMON_PROGRESS):
        return trajectory

    out = np.empty((trajectory.shape[0], len(COMMON_PROGRESS)), dtype=float)
    for unit_idx in range(trajectory.shape[0]):
        out[unit_idx] = np.interp(COMMON_PROGRESS, progress, trajectory[unit_idx])
    return out


def prepare_units(pair_id: str, grid_n: int, direction: dict, mirror_name: str, region_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load UME/CME unit coordinates, mirror, assign grids, and apply region filter."""

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

    if region_name != "full":
        start, stop = REGIONS[region_name]["interval"]
        ume_units["motion_progress"] = base.motion_progress(
            ume_units["x_norm"], ume_units["y_norm"], direction["code"]
        )
        cme_units["motion_progress"] = base.motion_progress(
            cme_units["x_norm"], cme_units["y_norm"], direction["code"]
        )
        ume_units = ume_units[(ume_units["motion_progress"] >= start) & (ume_units["motion_progress"] <= stop)].copy()
        cme_units = cme_units[(cme_units["motion_progress"] >= start) & (cme_units["motion_progress"] <= stop)].copy()

    return ume_units, cme_units


def build_unit_table(units: pd.DataFrame, trajectories: np.ndarray, stimulus: str) -> pd.DataFrame:
    """Attach trajectory arrays to unit coordinate rows."""

    if units.empty:
        return pd.DataFrame(columns=["stimulus", "grid_id", "grid_x", "grid_y", "trajectory"])
    idx = units["unit_row_idx"].to_numpy(dtype=int)
    return pd.DataFrame(
        {
            "stimulus": stimulus,
            "grid_id": units["grid_id"].to_numpy(),
            "grid_x": units["grid_x"].to_numpy(),
            "grid_y": units["grid_y"].to_numpy(),
            "trajectory": list(trajectories[idx]),
        }
    )


def pearson_vec(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if x.size < 2 or denominator == 0:
        return np.nan
    return float(1.0 - np.dot(x, y) / denominator)


def grid_metrics(unit_df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[np.ndarray, int]]]:
    """Aggregate trajectories per grid and collect arrays for label permutation."""

    rows = []
    perm_arrays = []
    for grid_id, group in unit_df.groupby("grid_id"):
        ume_group = group[group["stimulus"] == "UME"]
        cme_group = group[group["stimulus"] == "CME"]
        if len(ume_group) < MIN_UNITS or len(cme_group) < MIN_UNITS:
            continue

        ume = np.vstack(ume_group["trajectory"].to_numpy())
        cme = np.vstack(cme_group["trajectory"].to_numpy())
        ume_mean = np.nanmean(ume, axis=0)
        cme_mean = np.nanmean(cme, axis=0)
        diff = ume_mean - cme_mean
        abs_diff = np.abs(diff)
        response_level = np.nanmean((np.abs(ume_mean) + np.abs(cme_mean)) / 2.0)

        rows.append(
            {
                "grid_id": grid_id,
                "UME_n": len(ume_group),
                "CME_n": len(cme_group),
                "trajectory_MAE": float(np.nanmean(abs_diff)),
                "normalized_trajectory_MAE": float(np.nanmean(abs_diff) / (response_level + EPS)),
                "trajectory_RMSE": float(np.sqrt(np.nanmean(diff**2))),
                "AUC_diff": float(np.trapezoid(ume_mean, COMMON_PROGRESS) - np.trapezoid(cme_mean, COMMON_PROGRESS)),
                "peak_diff": float(np.nanmax(ume_mean) - np.nanmax(cme_mean)),
                "trajectory_corr": pearson_vec(ume_mean, cme_mean),
                "trajectory_cosine_distance": cosine_distance(ume_mean, cme_mean),
            }
        )
        perm_arrays.append((np.vstack([ume, cme]), len(ume_group)))

    return pd.DataFrame(rows), perm_arrays


def permutation_p(grid_df: pd.DataFrame, perm_arrays: list[tuple[np.ndarray, int]]) -> dict[str, float]:
    """Permutation test for mean grid-level trajectory MAE."""

    if grid_df.empty or not perm_arrays:
        return {
            "real_mean_trajectory_MAE": np.nan,
            "null_mean_trajectory_MAE": np.nan,
            "p_trajectory_MAE": np.nan,
        }
    real = float(grid_df["trajectory_MAE"].mean())
    null_values = np.empty(N_PERM, dtype=float)
    for perm_idx in range(N_PERM):
        maes = []
        for arr, ume_n in perm_arrays:
            order = RNG.permutation(arr.shape[0])
            ume = arr[order[:ume_n]]
            cme = arr[order[ume_n:]]
            diff = np.nanmean(ume, axis=0) - np.nanmean(cme, axis=0)
            maes.append(np.nanmean(np.abs(diff)))
        null_values[perm_idx] = float(np.nanmean(maes))
    p = float((np.sum(null_values >= real) + 1) / (N_PERM + 1))
    return {
        "real_mean_trajectory_MAE": real,
        "null_mean_trajectory_MAE": float(np.nanmean(null_values)),
        "p_trajectory_MAE": p,
    }


def compute_one_test(
    response_variant: str,
    window_name: str,
    region_name: str,
    mirror_name: str,
    grid_n: int,
    pair_id: str,
    direction: dict,
) -> dict[str, object]:
    """Compute one grid-scale x pair x direction test."""

    pair_cfg = base.PAIR_CONFIG[pair_id]
    direction_zero_based = direction["id"] - 1
    ume_units, cme_units = prepare_units(pair_id, grid_n, direction, mirror_name, region_name)
    ume_traj = unit_trajectory(pair_cfg["UME"], "UME", direction_zero_based, window_name, response_variant)
    cme_traj = unit_trajectory(pair_cfg["CME"], "CME", direction_zero_based, window_name, response_variant)
    unit_df = pd.concat(
        [
            build_unit_table(ume_units, ume_traj, "UME"),
            build_unit_table(cme_units, cme_traj, "CME"),
        ],
        ignore_index=True,
    )
    grid_df, perm_arrays = grid_metrics(unit_df)
    perm = permutation_p(grid_df, perm_arrays)

    if grid_df.empty:
        metric_summary = {
            "valid_grid_count": 0,
            "mean_trajectory_MAE": np.nan,
            "mean_normalized_trajectory_MAE": np.nan,
            "mean_trajectory_RMSE": np.nan,
            "mean_abs_AUC_diff": np.nan,
            "mean_abs_peak_diff": np.nan,
            "mean_trajectory_corr": np.nan,
            "mean_abs_trajectory_corr": np.nan,
            "mean_trajectory_cosine_distance": np.nan,
        }
    else:
        metric_summary = {
            "valid_grid_count": int(len(grid_df)),
            "mean_trajectory_MAE": float(grid_df["trajectory_MAE"].mean()),
            "mean_normalized_trajectory_MAE": float(grid_df["normalized_trajectory_MAE"].mean()),
            "mean_trajectory_RMSE": float(grid_df["trajectory_RMSE"].mean()),
            "mean_abs_AUC_diff": float(np.abs(grid_df["AUC_diff"]).mean()),
            "mean_abs_peak_diff": float(np.abs(grid_df["peak_diff"]).mean()),
            "mean_trajectory_corr": float(grid_df["trajectory_corr"].mean()),
            "mean_abs_trajectory_corr": float(np.abs(grid_df["trajectory_corr"]).mean()),
            "mean_trajectory_cosine_distance": float(grid_df["trajectory_cosine_distance"].mean()),
        }

    variant_cfg = RESPONSE_VARIANTS[response_variant]
    return {
        "response_variant": response_variant,
        "response_label": variant_cfg["label"],
        "base_metric": variant_cfg["metric"],
        "response_epsilon": variant_cfg["epsilon"],
        "window_name": window_name,
        "window_label_cn": WINDOWS[window_name]["label_cn"],
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


def bh_fdr(series: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction."""

    p = series.to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    pv = p[mask]
    if pv.size == 0:
        return pd.Series(q, index=series.index)
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = np.empty_like(ranked)
    previous = 1.0
    n = len(ranked)
    for i in range(n - 1, -1, -1):
        previous = min(previous, ranked[i] * n / (i + 1))
        adjusted[i] = min(previous, 1.0)
    qv = np.empty_like(adjusted)
    qv[order] = adjusted
    q[mask] = qv
    return pd.Series(q, index=series.index)


def summarize(test_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize test-level results to parameter-combination rankings."""

    keys = ["response_variant", "window_name", "region_name", "mirror_name"]
    test_df["q_within_combo"] = np.nan
    for _, idx in test_df.groupby(keys).groups.items():
        test_df.loc[idx, "q_within_combo"] = bh_fdr(test_df.loc[idx, "p_trajectory_MAE"])

    rows = []
    for combo, sub in test_df.groupby(keys):
        response_variant, window_name, region_name, mirror_name = combo
        sig_q = sub["q_within_combo"] < 0.05
        sig_p = sub["p_trajectory_MAE"] < 0.05
        q_fraction = float(sig_q.mean())
        direction_coverage = sub.loc[sig_q, "direction_code"].nunique() / len(DIRECTIONS)
        pair_coverage = sub.loc[sig_q, "pair_id"].nunique() / len(PAIR_IDS)
        grid_coverage = sub.loc[sig_q, "grid_n"].nunique() / len(GRID_SCALES)
        norm_mae = float(sub["mean_normalized_trajectory_MAE"].mean())
        corr_abs = float(np.nanmean(np.abs(sub["mean_trajectory_corr"])))

        # Ranking score only. It is not a p-value.
        score = (
            0.43 * q_fraction
            + 0.15 * direction_coverage
            + 0.15 * pair_coverage
            + 0.10 * grid_coverage
            + 0.12 * min(norm_mae / 1.5, 1.0)
            + 0.05 * max(0.0, 1.0 - min(corr_abs, 1.0))
        )

        rows.append(
            {
                "response_variant": response_variant,
                "response_label": RESPONSE_VARIANTS[response_variant]["label"],
                "base_metric": RESPONSE_VARIANTS[response_variant]["metric"],
                "response_epsilon": RESPONSE_VARIANTS[response_variant]["epsilon"],
                "window_name": window_name,
                "window_label_cn": WINDOWS[window_name]["label_cn"],
                "region_name": region_name,
                "region_label_cn": REGIONS[region_name]["label_cn"],
                "mirror_name": mirror_name,
                "mirror_label_cn": MIRRORS[mirror_name]["label_cn"],
                "total_tests": int(len(sub)),
                "q_sig_count": int(sig_q.sum()),
                "q_sig_fraction": q_fraction,
                "p_sig_count": int(sig_p.sum()),
                "p_sig_fraction": float(sig_p.mean()),
                "direction_coverage": direction_coverage,
                "pair_coverage": pair_coverage,
                "grid_coverage": grid_coverage,
                "mean_valid_grid_count": float(sub["valid_grid_count"].mean()),
                "mean_trajectory_MAE": float(sub["mean_trajectory_MAE"].mean()),
                "mean_normalized_trajectory_MAE": norm_mae,
                "mean_abs_AUC_diff": float(sub["mean_abs_AUC_diff"].mean()),
                "mean_abs_peak_diff": float(sub["mean_abs_peak_diff"].mean()),
                "mean_trajectory_corr": float(sub["mean_trajectory_corr"].mean()),
                "mean_abs_trajectory_corr": corr_abs,
                "mean_trajectory_cosine_distance": float(sub["mean_trajectory_cosine_distance"].mean()),
                "median_p_trajectory_MAE": float(sub["p_trajectory_MAE"].median()),
                "median_q_within_combo": float(sub["q_within_combo"].median()),
                "recommendation_score": float(score),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["recommendation_score", "q_sig_fraction", "direction_coverage", "pair_coverage"],
        ascending=False,
    ).reset_index(drop=True)


def make_figures(summary: pd.DataFrame) -> None:
    """Save compact overview figures."""

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

    top = summary.head(20).copy()
    labels = [
        f"{r.response_variant}\n{r.window_name}|{r.region_name}|{r.mirror_name}"
        for _, r in top.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(top))[::-1]
    ax.barh(y, top["recommendation_score"], color="#315C8C", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Recommendation score")
    ax.set_title("Focused MEA trajectory selector: top combinations")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "focused_selector_top_combinations.png", dpi=300)
    fig.savefig(FIG_DIR / "focused_selector_top_combinations.pdf")
    plt.close(fig)

    pivot = summary.pivot_table(
        index="response_variant",
        columns=["window_name", "region_name"],
        values="q_sig_fraction",
        aggfunc="max",
    )
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{a}\n{b}" for a, b in pivot.columns], fontsize=7, rotation=35, ha="right")
    ax.set_title("Best FDR significant fraction by metric/window/region")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("FDR significant fraction")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "focused_selector_q_fraction_heatmap.png", dpi=300)
    fig.savefig(FIG_DIR / "focused_selector_q_fraction_heatmap.pdf")
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    """Write a short human-readable report."""

    top = summary.head(20)
    lines = [
        "# Focused MEA trajectory selector report",
        "",
        "本次扫描只测试较容易在论文中解释的 trajectory 参数组合。",
        "",
        "- 统计单位：spatially matched grid-level local population trajectory",
        "- 主指标：trajectory MAE",
        "- 统计检验：每个有效网格内 UME/CME unit label permutation",
        f"- permutation 次数：{N_PERM}",
        "",
        "## Top 20 combinations",
        "",
        top[
            [
                "response_variant",
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
        "## Notes",
        "",
        "recommendation_score 只用于排序，不是统计显著性。",
        "最终主文建议优先选择 q_sig_fraction 高、方向/实验组/尺度覆盖完整、且参数解释自然的组合。",
    ]
    (REPORT_DIR / "focused_selector_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    total_tests = (
        len(RESPONSE_VARIANTS)
        * len(WINDOWS)
        * len(REGIONS)
        * len(MIRRORS)
        * len(GRID_SCALES)
        * len(PAIR_IDS)
        * len(DIRECTIONS)
    )
    log(f"Output directory: {OUT_DIR}")
    log(f"Total tests: {total_tests}")
    log(f"N_PERM per test: {N_PERM}")

    rows = []
    completed = 0
    checkpoint_path = TABLE_DIR / "focused_selector_tests_checkpoint.csv"
    for response_variant in RESPONSE_VARIANTS:
        for window_name in WINDOWS:
            for region_name in REGIONS:
                for mirror_name in MIRRORS:
                    for grid_n in GRID_SCALES:
                        for pair_id in PAIR_IDS:
                            for direction in DIRECTIONS:
                                completed += 1
                                if completed == 1 or completed % 200 == 0 or completed == total_tests:
                                    log(
                                        f"[{completed}/{total_tests}] {response_variant}, {window_name}, "
                                        f"{region_name}, {mirror_name}, {grid_n}x{grid_n}"
                                    )
                                rows.append(
                                    compute_one_test(
                                        response_variant=response_variant,
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
    summary = summarize(test_df)
    test_path = TABLE_DIR / "focused_selector_tests.csv"
    summary_path = TABLE_DIR / "focused_selector_combo_summary.csv"
    top_path = TABLE_DIR / "focused_selector_top30.csv"
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary.head(30).to_csv(top_path, index=False, encoding="utf-8-sig")
    log(f"Saved test table: {test_path}")
    log(f"Saved summary table: {summary_path}")
    log(f"Saved top30 table: {top_path}")

    make_figures(summary)
    write_report(summary)
    shutil.copy2(Path(__file__), CODE_DIR / Path(__file__).name)
    (LOG_DIR / "focused_selector_log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")
    log("Finished focused MEA trajectory selector.")


if __name__ == "__main__":
    main()
