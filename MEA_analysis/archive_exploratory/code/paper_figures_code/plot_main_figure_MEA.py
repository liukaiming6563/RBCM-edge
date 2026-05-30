"""绘制 RBCM-edge MEA 论文主图。

主图叙事：
1. UME/CME 刺激和 MEA 记录流程；
2. 空间匹配网格局部群体分析，而非 unit-level pairing；
3. 代表性 ΔFR map；
4. paired-retina 层级汇总；
5. spatial map dissimilarity；
6. grid-scale robustness。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from fig_style import (
    CME_COLOR,
    GRID_COLOR,
    UME_COLOR,
    PAIR_COLORS,
    add_compact_colorbar,
    add_panel_label,
    make_delta_cmap,
    mm_to_inch,
    pair_label,
    plot_masked_grid_map,
    prettify_axis,
    robust_symmetric_vmax,
    save_figure,
    set_paper_style,
)
from load_results import MEAResults, select_representative_directions


def _stimulus_screen(ax, x: float, y: float, w: float, h: float, mode: str) -> None:
    """画 UME/CME 刺激示意小屏幕。"""

    ax.add_patch(Rectangle((x, y), w, h, facecolor="#F6F6F6", edgecolor="#333333", lw=0.8))
    if mode == "UME":
        ax.add_patch(Rectangle((x, y), w * 0.48, h, facecolor="#D9D9D9", edgecolor="none"))
        ax.add_patch(Rectangle((x + w * 0.48, y), w * 0.52, h, facecolor="#FFFFFF", edgecolor="none"))
        label = "UME"
        subtitle = "uniform bg"
        color = UME_COLOR
    else:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#EFEFEF", edgecolor="none"))
        ax.add_patch(Rectangle((x, y), w * 0.35, h, facecolor="#D0D0D0", edgecolor="none", alpha=0.85))
        ax.add_patch(Rectangle((x + w * 0.64, y), w * 0.36, h, facecolor="#C6C6C6", edgecolor="none", alpha=0.7))
        ax.add_patch(Rectangle((x + w * 0.48, y), w * 0.06, h, facecolor="#FFFFFF", edgecolor="#555555", lw=0.2))
        label = "CME"
        subtitle = "static context"
        color = CME_COLOR
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor="#333333", lw=0.8))
    ax.arrow(x + w * 0.30, y + h + 0.025, w * 0.32, 0, head_width=0.020, head_length=0.020, color="#555555", lw=0.8)
    ax.text(x + 0.012, y + h - 0.015, label, ha="left", va="top", fontweight="bold", fontsize=8.0, color=color)
    ax.text(x + w + 0.020, y + h / 2, subtitle, ha="left", va="center", fontsize=6.3, color="#444444")


def plot_panel_A_stimulus_workflow(ax) -> None:
    """Panel A：UME/CME 刺激与 MEA 记录流程示意。"""

    ax.set_axis_off()
    _stimulus_screen(ax, 0.035, 0.62, 0.16, 0.135, "UME")
    _stimulus_screen(ax, 0.035, 0.37, 0.16, 0.135, "CME")

    ax.add_patch(FancyArrowPatch((0.30, 0.69), (0.405, 0.55), arrowstyle="->", mutation_scale=10, lw=0.85, color="#333333", connectionstyle="arc3,rad=-0.16"))
    ax.add_patch(FancyArrowPatch((0.30, 0.44), (0.405, 0.50), arrowstyle="->", mutation_scale=10, lw=0.85, color="#333333", connectionstyle="arc3,rad=0.16"))

    # MEA array + retina schematic.
    ax.add_patch(Rectangle((0.42, 0.39), 0.155, 0.22, facecolor="#FAFAFA", edgecolor="#333333", lw=0.75))
    for i in range(5):
        for j in range(5):
            ax.add_patch(Circle((0.44 + i * 0.026, 0.420 + j * 0.033), 0.004, color="#777777", alpha=0.75))
    ax.add_patch(Circle((0.497, 0.500), 0.062, facecolor="#F7DDC7", edgecolor="#C97940", lw=0.8, alpha=0.9))
    ax.text(0.497, 0.31, "MEA", ha="center", va="top", fontsize=7.2)

    ax.add_patch(FancyArrowPatch((0.60, 0.50), (0.705, 0.50), arrowstyle="->", mutation_scale=10, lw=0.85, color="#333333"))

    ax.add_patch(Rectangle((0.73, 0.405), 0.14, 0.205, facecolor="#F7F7F7", edgecolor="#333333", lw=0.75))
    rng = np.random.default_rng(3)
    for row in range(5):
        xs = 0.748 + np.sort(rng.random(18)) * 0.105
        ys = 0.428 + row * 0.034 + rng.normal(0, 0.003, len(xs))
        ax.vlines(xs, ys - 0.012, ys + 0.012, color="#333333", lw=0.45)
    ax.text(0.80, 0.31, "good units", ha="center", va="top", fontsize=7.2)
    ax.text(0.47, 0.12, "ON phase  |  center step  |  repeat-averaged FR", ha="center", fontsize=6.7, color="#444444")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_panel_B_grid_method(ax, results: MEAResults) -> None:
    """Panel B：空间网格局部群体分析示意。"""

    ax.set_axis_off()
    rng = np.random.default_rng(12)
    x0, y0, w, h = 0.05, 0.16, 0.52, 0.66
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="#FBFBFB", edgecolor="#333333", lw=0.8))
    for i in range(13):
        xx = x0 + w * i / 12
        yy = y0 + h * i / 12
        ax.plot([xx, xx], [y0, y0 + h], color=GRID_COLOR, lw=0.35, alpha=0.8)
        ax.plot([x0, x0 + w], [yy, yy], color=GRID_COLOR, lw=0.35, alpha=0.8)
    ume_x = x0 + rng.random(130) * w
    ume_y = y0 + rng.random(130) * h
    cme_x = x0 + rng.random(120) * w
    cme_y = y0 + rng.random(120) * h
    ax.scatter(ume_x, ume_y, s=7, color="#D55E00", alpha=0.42, edgecolor="none", label="UME")
    ax.scatter(cme_x, cme_y, s=7, color=CME_COLOR, alpha=0.42, edgecolor="none", label="CME")
    ax.add_patch(Rectangle((x0 + w * 6 / 12, y0 + h * 5 / 12), w / 12, h / 12, facecolor="#FFD966", edgecolor="#111111", lw=1.1, alpha=0.65))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98), handletextpad=0.2, borderaxespad=0, ncol=2, columnspacing=0.8)

    ax.add_patch(Rectangle((0.64, 0.25), 0.31, 0.46, facecolor="#FFFFFF", edgecolor="#333333", lw=0.75))
    ax.text(0.795, 0.65, "Spatial grid cell", ha="center", va="center", fontweight="bold", fontsize=8.3)
    ax.text(0.795, 0.55, "local RGC population\n(no unit-level pairing)", ha="center", va="center", fontsize=6.9, color="#444444", linespacing=0.9)
    ax.text(0.795, 0.415, "ΔFR = FR$_{UME}$ - FR$_{CME}$", ha="center", va="center", fontsize=7.7)
    ax.text(0.795, 0.315, "NDI = ΔFR /\n(FR$_{UME}$ + FR$_{CME}$ + ε)", ha="center", va="center", fontsize=6.8, linespacing=0.85)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _representative_map_rows(results: MEAResults, grid_n: int = 12):
    reps = select_representative_directions(results, grid_n=grid_n)
    rows = []
    for pair_id in ["pair_31_32", "pair_34_35", "pair_37_38"]:
        direction = reps[pair_id]
        df = results.grid[(results.grid["grid_n"] == grid_n) & (results.grid["pair_id"] == pair_id) & (results.grid["direction_code"] == direction)]
        rows.append((pair_id, direction, df))
    return rows


def plot_panel_C_representative_maps(fig, axes, results: MEAResults, grid_n: int = 12):
    """Panel C：代表性 ΔFR maps，三个 pair 各一个数据驱动选择的方向。"""

    cmap = make_delta_cmap()
    main_values = results.grid[(results.grid["grid_n"] == grid_n) & results.grid["valid_grid"]]["delta_mean_fr_hz"]
    vmax = robust_symmetric_vmax(main_values, percentile=95, floor=2.0)
    ims = []
    for ax, (pair_id, direction, df) in zip(axes, _representative_map_rows(results, grid_n)):
        im = plot_masked_grid_map(
            ax,
            df,
            "delta_mean_fr_hz",
            grid_n,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            title=f"{pair_label(pair_id)}  {direction}",
        )
        ims.append(im)
    add_compact_colorbar(fig, ims[-1], axes, "ΔFR (Hz), UME - CME", orientation="horizontal", fraction=0.08, pad=0.08)
    return ims[-1]


def plot_panel_D_paired_summary(axs, results: MEAResults) -> None:
    """Panel D：paired-retina 层级 small multiples。"""

    metrics = [
        ("mean_different_fraction", "Different grids", "Fraction", (0, 1)),
        ("mean_abs_delta_mean_fr_hz", "|ΔFR|", "Hz", None),
        ("mean_abs_NDI_mean", "|NDI|", "Index", (0, None)),
        ("mean_normalized_MAE", "Norm. MAE", "Norm. MAE", (0, None)),
    ]
    df = results.paired_overall[results.paired_overall["grid_n"].isin([12, 16])].copy()
    pair_ids = ["pair_31_32", "pair_34_35", "pair_37_38"]
    x = np.arange(len(pair_ids))
    for ax, (col, title, ylabel, ylim) in zip(axs, metrics):
        for pair_idx, pair_id in enumerate(pair_ids):
            sub = df[df["pair_id"] == pair_id].sort_values("grid_n")
            vals = sub[col].to_numpy()
            ax.plot([pair_idx, pair_idx], vals, color="#AAAAAA", lw=0.8, zorder=1)
            for grid_n, marker in [(12, "o"), (16, "s")]:
                row = sub[sub["grid_n"] == grid_n].iloc[0]
                ax.scatter(pair_idx, row[col], s=38, marker=marker, color=PAIR_COLORS[pair_id], edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_title(title, pad=2)
        ax.set_xticks(x, [pair_label(p) for p in pair_ids], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            low, high = ylim
            ax.set_ylim(bottom=low)
            if high is not None:
                ax.set_ylim(top=high)
        prettify_axis(ax, grid=True)
    # Marker meaning is described in the figure legend to keep panel D uncluttered.


def plot_panel_E_spatial_dissimilarity(ax, results: MEAResults) -> None:
    """Panel E：normalized MAE dot plot。"""

    df = results.paired_overall[results.paired_overall["grid_n"].isin([12, 16])].copy()
    pair_ids = ["pair_31_32", "pair_34_35", "pair_37_38"]
    x = np.arange(len(pair_ids))
    for pair_idx, pair_id in enumerate(pair_ids):
        sub = df[df["pair_id"] == pair_id].sort_values("grid_n")
        vals = sub["mean_normalized_MAE"].to_numpy()
        ax.plot([pair_idx, pair_idx], vals, color="#AAAAAA", lw=0.8)
        for grid_n, marker in [(12, "o"), (16, "s")]:
            row = sub[sub["grid_n"] == grid_n].iloc[0]
            ax.scatter(pair_idx, row["mean_normalized_MAE"], s=48, marker=marker, color=PAIR_COLORS[pair_id], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x, [pair_label(p) for p in pair_ids], rotation=30, ha="right")
    ax.set_ylabel("Normalized MAE")
    ax.set_title("Spatial map dissimilarity", pad=4)
    prettify_axis(ax, grid=True)


def plot_panel_F_grid_scale_robustness(ax, results: MEAResults) -> None:
    """Panel F：多尺度鲁棒性曲线。"""

    df = results.paired_overall.copy()
    for pair_id, sub in df.groupby("pair_id"):
        sub = sub.sort_values("grid_n")
        ax.plot(sub["grid_n"], sub["mean_different_fraction"], marker="o", ms=4.5, color=PAIR_COLORS[pair_id], label=pair_label(pair_id))
    ax.axvspan(11.5, 16.5, color="#EEEEEE", zorder=0)
    ax.text(14, 0.06, "main\nscales", ha="center", va="bottom", fontsize=7, color="#666666")
    ax.set_xlabel("Grid N")
    ax.set_ylabel("Different fraction")
    ax.set_ylim(0, 1)
    ax.set_xticks([8, 10, 12, 16, 20, 25, 30])
    ax.set_title("Grid-scale robustness", pad=4)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.02), handlelength=1.4, borderaxespad=0.0)
    prettify_axis(ax, grid=True)


def export_individual_main_panels(results: MEAResults, out_dir: Path, manifest_rows: list[dict]) -> None:
    """单独导出 A-F panel 文件，方便后期拼版修改。"""

    set_paper_style()
    panel_dir = out_dir / "source_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=mm_to_inch(85, 55))
    plot_panel_A_stimulus_workflow(ax)
    add_panel_label(ax, "A", x=-0.02, y=1.02)
    files = save_figure(fig, panel_dir / "panel_A_workflow")
    manifest_rows.append({"figure_id": "panel_A_workflow", "figure_type": "main_panel", "files": ";".join(map(str, files)), "description": "UME/CME stimulus and MEA workflow schematic."})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=mm_to_inch(95, 55))
    plot_panel_B_grid_method(ax, results)
    add_panel_label(ax, "B", x=-0.02, y=1.02)
    files = save_figure(fig, panel_dir / "panel_B_grid_method")
    manifest_rows.append({"figure_id": "panel_B_grid_method", "figure_type": "main_panel", "files": ";".join(map(str, files)), "description": "Spatially matched grid-level local population analysis schematic."})
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=mm_to_inch(150, 45), constrained_layout=True)
    plot_panel_C_representative_maps(fig, axes, results, grid_n=12)
    add_panel_label(axes[0], "C", x=-0.18, y=1.10)
    files = save_figure(fig, panel_dir / "panel_C_representative_delta_maps")
    manifest_rows.append({"figure_id": "panel_C_representative_delta_maps", "figure_type": "main_panel", "files": ";".join(map(str, files)), "description": "Representative grid-level delta FR maps."})
    plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=mm_to_inch(170, 50), constrained_layout=True)
    plot_panel_D_paired_summary(axs, results)
    add_panel_label(axs[0], "D", x=-0.36, y=1.14)
    files = save_figure(fig, panel_dir / "panel_D_paired_retina_summary")
    manifest_rows.append({"figure_id": "panel_D_paired_retina_summary", "figure_type": "main_panel", "files": ";".join(map(str, files)), "description": "Paired-retina-level summary at main grid scales."})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=mm_to_inch(65, 50))
    plot_panel_E_spatial_dissimilarity(ax, results)
    add_panel_label(ax, "E", x=-0.20, y=1.08)
    files = save_figure(fig, panel_dir / "panel_E_spatial_dissimilarity")
    manifest_rows.append({"figure_id": "panel_E_spatial_dissimilarity", "figure_type": "main_panel", "files": ";".join(map(str, files)), "description": "Spatial map dissimilarity quantified by normalized MAE."})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=mm_to_inch(70, 50))
    plot_panel_F_grid_scale_robustness(ax, results)
    add_panel_label(ax, "F", x=-0.20, y=1.08)
    files = save_figure(fig, panel_dir / "panel_F_grid_scale_robustness")
    manifest_rows.append({"figure_id": "panel_F_grid_scale_robustness", "figure_type": "main_panel", "files": ";".join(map(str, files)), "description": "Grid-scale robustness of different-grid fraction."})
    plt.close(fig)


def plot_main_figure(results: MEAResults, out_dir: str | Path, manifest_rows: list[dict]) -> None:
    """生成完整主图和拆分 panel。"""

    set_paper_style()
    out_dir = Path(out_dir)
    main_dir = out_dir / "main_figures"
    main_dir.mkdir(parents=True, exist_ok=True)
    export_individual_main_panels(results, out_dir, manifest_rows)

    fig = plt.figure(figsize=mm_to_inch(180, 238))
    gs = fig.add_gridspec(4, 6, height_ratios=[0.95, 1.03, 0.92, 0.95], hspace=0.56, wspace=0.58)

    ax_a = fig.add_subplot(gs[0, :2])
    plot_panel_A_stimulus_workflow(ax_a)
    add_panel_label(ax_a, "A", x=-0.04, y=1.04)

    ax_b = fig.add_subplot(gs[0, 2:])
    plot_panel_B_grid_method(ax_b, results)
    add_panel_label(ax_b, "B", x=-0.04, y=1.04)

    axes_c = [fig.add_subplot(gs[1, i * 2 : (i + 1) * 2]) for i in range(3)]
    plot_panel_C_representative_maps(fig, axes_c, results, grid_n=12)
    add_panel_label(axes_c[0], "C", x=-0.15, y=1.12)

    sub_d = gs[2, :].subgridspec(1, 4, wspace=0.40)
    axs_d = [fig.add_subplot(sub_d[0, i]) for i in range(4)]
    plot_panel_D_paired_summary(axs_d, results)
    add_panel_label(axs_d[0], "D", x=-0.25, y=1.17)

    ax_e = fig.add_subplot(gs[3, :2])
    plot_panel_E_spatial_dissimilarity(ax_e, results)
    add_panel_label(ax_e, "E", x=-0.18, y=1.13)

    ax_f = fig.add_subplot(gs[3, 2:])
    plot_panel_F_grid_scale_robustness(ax_f, results)
    add_panel_label(ax_f, "F", x=-0.12, y=1.13)

    files = save_figure(fig, main_dir / "main_figure_MEA_edge_context", tiff=True)
    manifest_rows.append({"figure_id": "main_figure_MEA_edge_context", "figure_type": "main_figure", "files": ";".join(map(str, files)), "description": "Complete MEA main figure for edge-context-dependent RGC population responses."})
    plt.close(fig)
