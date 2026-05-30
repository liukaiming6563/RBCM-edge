"""绘制 RBCM-edge MEA 论文补充图 S1-S10。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig_style import (
    DIR_ORDER,
    PAIR_COLORS,
    add_compact_colorbar,
    add_panel_label,
    make_delta_cmap,
    matrix_from_grid,
    mm_to_inch,
    pair_label,
    plot_masked_grid_map,
    prettify_axis,
    robust_symmetric_vmax,
    save_figure,
    set_paper_style,
)
from load_results import MEAResults, select_representative_directions


def _all_direction_maps(
    results: MEAResults,
    out_dir: Path,
    manifest_rows: list[dict],
    *,
    grid_n: int,
    value_col: str,
    figure_id: str,
    label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """3 pair × 8 direction 的统一热图面板。"""

    set_paper_style()
    sub = results.grid[results.grid["grid_n"] == grid_n].copy()
    if vmin is None or vmax is None:
        vmax_auto = robust_symmetric_vmax(sub.loc[sub["valid_grid"], value_col], percentile=95, floor=1.0)
        vmin, vmax = -vmax_auto, vmax_auto
    cmap = make_delta_cmap()
    pair_ids = ["pair_31_32", "pair_34_35", "pair_37_38"]

    fig, axes = plt.subplots(3, 8, figsize=mm_to_inch(180, 78), constrained_layout=True)
    im = None
    for r, pair_id in enumerate(pair_ids):
        for c, direction in enumerate(DIR_ORDER):
            ax = axes[r, c]
            df = sub[(sub["pair_id"] == pair_id) & (sub["direction_code"] == direction)]
            im = plot_masked_grid_map(ax, df, value_col, grid_n, cmap=cmap, vmin=vmin, vmax=vmax, title=direction if r == 0 else "")
            if c == 0:
                ax.set_ylabel(pair_label(pair_id), rotation=0, ha="right", va="center", labelpad=18)
            else:
                ax.set_ylabel("")
    add_compact_colorbar(fig, im, axes.ravel().tolist(), label, orientation="horizontal", fraction=0.035, pad=0.04)
    files = save_figure(fig, out_dir / figure_id, tiff=True)
    manifest_rows.append({"figure_id": figure_id, "figure_type": "supplementary", "files": ";".join(map(str, files)), "description": f"All-direction {value_col} maps at {grid_n}x{grid_n}."})
    plt.close(fig)


def plot_supp_S1_all_delta_maps_12x12(results: MEAResults, out_dir: Path, manifest_rows: list[dict]) -> None:
    _all_direction_maps(results, out_dir, manifest_rows, grid_n=12, value_col="delta_mean_fr_hz", figure_id="supp_fig_S1_all_direction_delta_maps_12x12", label="ΔFR (Hz), UME - CME")


def plot_supp_S2_all_delta_maps_16x16(results: MEAResults, out_dir: Path, manifest_rows: list[dict]) -> None:
    _all_direction_maps(results, out_dir, manifest_rows, grid_n=16, value_col="delta_mean_fr_hz", figure_id="supp_fig_S2_all_direction_delta_maps_16x16", label="ΔFR (Hz), UME - CME")


def plot_supp_S3_grid_scale_robustness(results: MEAResults, out_dir: Path, manifest_rows: list[dict]) -> None:
    """S3：三个多尺度鲁棒性指标。"""

    set_paper_style()
    df = results.paired_overall.copy()
    metrics = [
        ("mean_different_fraction", "Different fraction", (0, 1)),
        ("mean_abs_delta_mean_fr_hz", "Mean |ΔFR| (Hz)", None),
        ("mean_normalized_MAE", "Mean normalized MAE", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=mm_to_inch(180, 55), constrained_layout=True)
    for ax, (col, ylabel, ylim) in zip(axes, metrics):
        for pair_id, sub in df.groupby("pair_id"):
            sub = sub.sort_values("grid_n")
            ax.plot(sub["grid_n"], sub[col], marker="o", ms=4.5, color=PAIR_COLORS[pair_id], label=pair_label(pair_id))
        ax.axvspan(11.5, 16.5, color="#EFEFEF", zorder=0)
        ax.set_xlabel("Grid N")
        ax.set_ylabel(ylabel)
        ax.set_xticks([8, 10, 12, 16, 20, 25, 30])
        if ylim:
            ax.set_ylim(*ylim)
        prettify_axis(ax, grid=True)
    axes[0].legend(frameon=False, loc="best")
    for label, ax in zip(["A", "B", "C"], axes):
        add_panel_label(ax, label, x=-0.16, y=1.08)
    files = save_figure(fig, out_dir / "supp_fig_S3_grid_scale_robustness", tiff=True)
    manifest_rows.append({"figure_id": "supp_fig_S3_grid_scale_robustness", "figure_type": "supplementary", "files": ";".join(map(str, files)), "description": "Grid-scale robustness across multiple summary metrics."})
    plt.close(fig)


def plot_supp_S4_threshold_sensitivity(results: MEAResults, out_dir: Path, manifest_rows: list[dict]) -> None:
    """S4：阈值敏感性。"""

    set_paper_style()
    df = results.threshold_pair.copy()
    fig = plt.figure(figsize=mm_to_inch(180, 62))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.08], wspace=0.42)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, grid_n in zip(axes[:2], [12, 16]):
        sub = df[df["grid_n"] == grid_n]
        for pair_id, psub in sub.groupby("pair_id"):
            psub = psub.sort_values("threshold_hz")
            ax.plot(psub["threshold_hz"], psub["mean_different_fraction"], marker="o", ms=4.5, color=PAIR_COLORS[pair_id], label=pair_label(pair_id))
        ax.set_xlabel("Threshold (Hz)")
        ax.set_ylabel("Different fraction")
        ax.set_ylim(0, 1)
        ax.set_title(f"{grid_n}×{grid_n}", pad=2)
        prettify_axis(ax, grid=True)
    axes[0].legend(frameon=False, loc="best")

    heat = df.groupby(["grid_n", "threshold_hz"], as_index=False)["mean_different_fraction"].mean()
    pivot = heat.pivot(index="grid_n", columns="threshold_hz", values="mean_different_fraction").sort_index()
    im = axes[2].imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axes[2].set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
    axes[2].set_yticks(range(len(pivot.index)), pivot.index.astype(str))
    axes[2].set_xlabel("Threshold (Hz)")
    axes[2].set_ylabel("Grid N")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            axes[2].text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")
    add_compact_colorbar(fig, im, axes[2], "Different fraction", orientation="vertical", fraction=0.055, pad=0.03)
    for label, ax in zip(["A", "B", "C"], axes):
        add_panel_label(ax, label, x=-0.18, y=1.10)
    files = save_figure(fig, out_dir / "supp_fig_S4_threshold_sensitivity", tiff=True)
    manifest_rows.append({"figure_id": "supp_fig_S4_threshold_sensitivity", "figure_type": "supplementary", "files": ";".join(map(str, files)), "description": "Threshold sensitivity of grid classification."})
    plt.close(fig)


def plot_supp_S5_NDI_maps(results: MEAResults, out_dir: Path, manifest_rows: list[dict]) -> None:
    _all_direction_maps(results, out_dir, manifest_rows, grid_n=12, value_col="NDI_mean", figure_id="supp_fig_S5_NDI_maps_12x12", label="NDI", vmin=-1, vmax=1)


def plot_supp_S6_response_fraction_maps(results: MEAResults, out_dir: Path, manifest_rows: list[dict]) -> None:
    _all_direction_maps(results, out_dir, manifest_rows, grid_n=12, value_col="delta_nonzero_fraction", figure_id="supp_fig_S6_response_fraction_delta_maps_12x12", label="Δ responsive fraction", vmin=-1, vmax=1)


def plot_supp_S7_robust_metric_comparison(results: MEAResults, out_dir: Path, manifest_rows: list[dict]) -> None:
    """S7：每个 pair 一个代表方向，比较 5 种指标 map。"""

    set_paper_style()
    grid_n = 12
    reps = select_representative_directions(results, grid_n=grid_n)
    pair_ids = ["pair_31_32", "pair_34_35", "pair_37_38"]
    metrics = [
        ("delta_mean_fr_hz", "Δ mean FR", "Hz", None),
        ("delta_median_fr_hz", "Δ median FR", "Hz", None),
        ("delta_nonzero_mean_fr_hz", "Δ nonzero mean FR", "Hz", None),
        ("delta_nonzero_fraction", "Δ responsive fraction", "Fraction", (-1, 1)),
        ("NDI_mean", "NDI", "Index", (-1, 1)),
    ]
    fig, axes = plt.subplots(3, 5, figsize=mm_to_inch(180, 92), constrained_layout=True)
    cmap = make_delta_cmap()
    im_by_col = []
    for c, (col, title, _, limits) in enumerate(metrics):
        values = results.grid[(results.grid["grid_n"] == grid_n) & results.grid["valid_grid"]][col]
        if limits is None:
            vmax = robust_symmetric_vmax(values, percentile=95, floor=1.0)
            limits = (-vmax, vmax)
        im_last = None
        for r, pair_id in enumerate(pair_ids):
            direction = reps[pair_id]
            df = results.grid[(results.grid["grid_n"] == grid_n) & (results.grid["pair_id"] == pair_id) & (results.grid["direction_code"] == direction)]
            im_last = plot_masked_grid_map(axes[r, c], df, col, grid_n, cmap=cmap, vmin=limits[0], vmax=limits[1], title=title if r == 0 else "")
            if c == 0:
                axes[r, c].set_ylabel(f"{pair_label(pair_id)}\n{direction}", rotation=0, ha="right", va="center", labelpad=26)
        im_by_col.append((im_last, metrics[c][2]))
    for c, (im, label) in enumerate(im_by_col):
        add_compact_colorbar(fig, im, axes[:, c].tolist(), label, orientation="horizontal", fraction=0.035, pad=0.04)
    files = save_figure(fig, out_dir / "supp_fig_S7_robust_metric_comparison", tiff=True)
    manifest_rows.append({"figure_id": "supp_fig_S7_robust_metric_comparison", "figure_type": "supplementary", "files": ";".join(map(str, files)), "description": "Representative robust metric maps."})
    plt.close(fig)


def plot_supp_S8_map_similarity_scatter(results: MEAResults, out_dir: Path, manifest_rows: list[dict]) -> None:
    """S8：UME map 与 CME map 的 grid scatter。"""

    set_paper_style()
    pair_ids = ["pair_31_32", "pair_34_35", "pair_37_38"]
    grid_ns = [12, 16]
    fig, axes = plt.subplots(3, 2, figsize=mm_to_inch(125, 125), constrained_layout=True)
    for r, pair_id in enumerate(pair_ids):
        for c, grid_n in enumerate(grid_ns):
            ax = axes[r, c]
            sub = results.grid[(results.grid["pair_id"] == pair_id) & (results.grid["grid_n"] == grid_n) & results.grid["valid_grid"]]
            x = sub["UME_mean_fr_hz"].to_numpy(dtype=float)
            y = sub["CME_mean_fr_hz"].to_numpy(dtype=float)
            ax.scatter(x, y, s=8, color=PAIR_COLORS[pair_id], alpha=0.25, edgecolor="none")
            maxv = np.nanpercentile(np.r_[x, y], 99) if x.size else 1
            ax.plot([0, maxv], [0, maxv], "--", color="#555555", lw=0.9)
            sim = results.spatial_overall[(results.spatial_overall["pair_id"] == pair_id) & (results.spatial_overall["grid_n"] == grid_n)].iloc[0]
            ax.text(0.04, 0.94, f"r={sim['mean_pearson_r']:.2f}\nNMAE={sim['mean_normalized_MAE']:.2f}", transform=ax.transAxes, va="top", fontsize=8)
            ax.set_xlim(0, maxv)
            ax.set_ylim(0, maxv)
            ax.set_xlabel("UME mean FR (Hz)")
            ax.set_ylabel("CME mean FR (Hz)")
            ax.set_title(f"{pair_label(pair_id)}  {grid_n}×{grid_n}", pad=2)
            prettify_axis(ax, grid=True)
    files = save_figure(fig, out_dir / "supp_fig_S8_UME_CME_map_scatter", tiff=True)
    manifest_rows.append({"figure_id": "supp_fig_S8_UME_CME_map_scatter", "figure_type": "supplementary", "files": ";".join(map(str, files)), "description": "Grid-level UME vs CME mean FR scatter."})
    plt.close(fig)


def plot_supp_S9_valid_grid_qc(results: MEAResults, out_dir: Path, manifest_rows: list[dict]) -> None:
    """S9：有效网格与 unit count QC。"""

    set_paper_style()
    valid = results.valid.copy()
    fig, axes = plt.subplots(1, 4, figsize=mm_to_inch(180, 52), constrained_layout=True)
    metrics = [
        ("valid_grid_fraction", "Valid grid fraction", (0, 1)),
        ("mean_UME_unit_count_per_valid_grid", "Mean UME units/grid", None),
        ("mean_CME_unit_count_per_valid_grid", "Mean CME units/grid", None),
    ]
    for ax, (col, ylabel, ylim) in zip(axes[:3], metrics):
        summary = valid.groupby(["grid_n", "pair_id"], as_index=False)[col].mean()
        for pair_id, sub in summary.groupby("pair_id"):
            sub = sub.sort_values("grid_n")
            ax.plot(sub["grid_n"], sub[col], marker="o", ms=4, color=PAIR_COLORS[pair_id], label=pair_label(pair_id))
        ax.set_xlabel("Grid N")
        ax.set_ylabel(ylabel)
        ax.set_xticks([8, 10, 12, 16, 20, 25, 30])
        if ylim:
            ax.set_ylim(*ylim)
        prettify_axis(ax, grid=True)
    axes[0].legend(frameon=False, loc="best")

    unit_counts = results.grid[results.grid["valid_grid"]][["UME_unit_count", "CME_unit_count"]].melt(var_name="Condition", value_name="Units per valid grid")
    colors = {"UME_unit_count": "#D55E00", "CME_unit_count": "#0072B2"}
    for key, df in unit_counts.groupby("Condition"):
        axes[3].hist(df["Units per valid grid"], bins=30, alpha=0.55, color=colors[key], label=key.replace("_unit_count", ""))
    axes[3].set_xlabel("Units per valid grid")
    axes[3].set_ylabel("Count")
    axes[3].legend(frameon=False)
    prettify_axis(axes[3], grid=False)
    for label, ax in zip(["A", "B", "C", "D"], axes):
        add_panel_label(ax, label, x=-0.16, y=1.08)
    files = save_figure(fig, out_dir / "supp_fig_S9_valid_grid_unit_count_QC", tiff=True)
    manifest_rows.append({"figure_id": "supp_fig_S9_valid_grid_unit_count_QC", "figure_type": "supplementary", "files": ";".join(map(str, files)), "description": "Valid grid and unit count QC."})
    plt.close(fig)


def plot_supp_S10_permutation_null(results: MEAResults, out_dir: Path, manifest_rows: list[dict]) -> None:
    """S10：permutation null 的辅助 QC 图。

    正式分析表只保存 null mean/std，没有保存每次 permutation 的完整数组。
    因此这里用正态近似画出 null summary distribution，用于补充说明，
    不作为 RBCM 动机的主要证据。
    """

    set_paper_style()
    grid_n = 12
    reps = select_representative_directions(results, grid_n=grid_n)
    pair_ids = ["pair_31_32", "pair_34_35", "pair_37_38"]
    fig, axes = plt.subplots(1, 3, figsize=mm_to_inch(165, 48), constrained_layout=True)
    rng = np.random.default_rng(42)
    for ax, pair_id in zip(axes, pair_ids):
        direction = reps[pair_id]
        row = results.permutation[(results.permutation["grid_n"] == grid_n) & (results.permutation["pair_id"] == pair_id) & (results.permutation["direction_code"] == direction)].iloc[0]
        null = rng.normal(row["null_mean_mean_abs_delta"], row["null_std_mean_abs_delta"], 4000)
        null = null[np.isfinite(null)]
        ax.hist(null, bins=30, color="#BDBDBD", edgecolor="white", linewidth=0.4)
        ax.axvline(row["real_mean_abs_delta"], color="#B2182B", lw=2.0, label="Real")
        ax.axvline(row["null_mean_mean_abs_delta"], color="#333333", lw=1.2, ls="--", label="Null mean")
        ax.set_title(f"{pair_label(pair_id)}  {direction}", pad=2)
        ax.set_xlabel("Mean |ΔFR| (Hz)")
        ax.set_ylabel("Approx. null count")
        ax.text(0.04, 0.94, f"p={row['p_mean_abs_delta']:.3f}", transform=ax.transAxes, va="top", fontsize=8)
        prettify_axis(ax)
    axes[0].legend(frameon=False)
    files = save_figure(fig, out_dir / "supp_fig_S10_permutation_null", tiff=True)
    manifest_rows.append({"figure_id": "supp_fig_S10_permutation_null", "figure_type": "supplementary", "files": ";".join(map(str, files)), "description": "Permutation null QC based on stored null mean/std approximation."})
    plt.close(fig)


def plot_all_supplementary_figures(results: MEAResults, out_dir: str | Path, manifest_rows: list[dict]) -> None:
    """导出所有补充图。"""

    out_dir = Path(out_dir) / "supplementary_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_supp_S1_all_delta_maps_12x12(results, out_dir, manifest_rows)
    plot_supp_S2_all_delta_maps_16x16(results, out_dir, manifest_rows)
    plot_supp_S3_grid_scale_robustness(results, out_dir, manifest_rows)
    plot_supp_S4_threshold_sensitivity(results, out_dir, manifest_rows)
    plot_supp_S5_NDI_maps(results, out_dir, manifest_rows)
    plot_supp_S6_response_fraction_maps(results, out_dir, manifest_rows)
    plot_supp_S7_robust_metric_comparison(results, out_dir, manifest_rows)
    plot_supp_S8_map_similarity_scatter(results, out_dir, manifest_rows)
    plot_supp_S9_valid_grid_qc(results, out_dir, manifest_rows)
    plot_supp_S10_permutation_null(results, out_dir, manifest_rows)

