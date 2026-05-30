"""
绘制 RBCM-edge 论文 Fig. 2A: Representative spatial DeltaFR maps.

本脚本基于已经完成的 MEA grid-level 正式分析结果绘图，而不是重新计算 firing rate。
固定使用：
    - grid size: 20 x 20
    - direction: left-to-right, 在当前分析表中对应 direction_code == "R"
    - paired retina: pair_31_32, pair_34_35, pair_37_38
    - DeltaFR = FR_UME - FR_CME

图像设计目标：
    展示 UME 和 CME 在空间匹配的局部视网膜网格中诱发不同的 RGC 群体响应模式。
    这不是 unit-level pairing；每个网格代表同一局部空间区域内的 sorted-unit population。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch


# =============================================================================
# 1. 路径与固定分析参数
# =============================================================================

PROJECT_ROOT = Path(r"D:\study\project\RBCM-Edge")

# 输入表：正式 MEA grid-cell 明细结果。
# 该表已经包含每个 grid cell 的 UME/CME mean firing rate、DeltaFR、valid_grid 等字段。
INPUT_TABLE = (
    PROJECT_ROOT
    / "outputs"
    / "MEA_analysis"
    / "tables"
    / "grid_cell_level_results_all_scales_extended.csv"
)

# 输出路径：用户指定的临时 figure 目录。
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "MEA_analysis" / "figure_tmp"

# 固定绘图参数。
GRID_N = 20
PAIR_ORDER = ["pair_31_32", "pair_34_35", "pair_37_38"]
PAIR_LABELS = {
    "pair_31_32": "P31–32",
    "pair_34_35": "P34–35",
    "pair_37_38": "P37–38",
}

# left-to-right 在当前 MEA 分析表中对应 R/right。
# 这里保留多个候选名，方便以后如果表格字段名变化仍可自动识别。
LEFT_TO_RIGHT_CANDIDATES = {"R", "dir-01", "rightward", "left_to_right", "right"}

# DeltaFR 接近 0 的计数阈值，仅用于终端核查，不参与 colormap 归一化。
NEAR_ZERO_THRESHOLD_HZ = 0.5


# =============================================================================
# 2. 颜色与论文风格参数
# =============================================================================

COLOR_CME_HIGHER = "#39FF14"  # DeltaFR < 0, fluorescent green
COLOR_NEAR_ZERO = "#073B4C"   # DeltaFR ~= 0, deep cyan-blue
COLOR_UME_HIGHER = "#FFF44F"  # DeltaFR > 0, lemon yellow
COLOR_INVALID = "#073B4C"     # invalid/background, same deep cyan-blue per current figure style
COLOR_GRID = "#808080"        # kept only for legacy helper legend edge; no grid lines are drawn
COLOR_TEXT = "#333333"        # labels and outlines
COLOR_BG = "#FFFFFF"

DPI = 600


@dataclass(frozen=True)
class FigureVariant:
    """一张 Fig. 2A 候选图的版式设置。"""

    name: str
    suffix: str
    figsize: tuple[float, float]
    show_helper_legend: bool
    colorbar_label: str
    title_pad: float


VARIANTS = [
    FigureVariant(
        name="main",
        suffix="main",
        figsize=(6.6, 2.35),
        show_helper_legend=False,
        colorbar_label=r"$\Delta$FR (Hz)",
        title_pad=4.0,
    ),
    FigureVariant(
        name="annotated",
        suffix="annotated",
        figsize=(7.0, 2.65),
        show_helper_legend=True,
        colorbar_label=r"$\Delta$FR = FR$_{UME}$ - FR$_{CME}$ (Hz)",
        title_pad=5.0,
    ),
]


# =============================================================================
# 3. 通用绘图工具
# =============================================================================

def set_paper_style() -> None:
    """设置接近期刊主图的 Matplotlib 风格。"""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.9,
            "figure.facecolor": COLOR_BG,
            "axes.facecolor": COLOR_BG,
            "savefig.facecolor": COLOR_BG,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_delta_cmap() -> LinearSegmentedColormap:
    """构建指定的 fluorescent-green -> deep-cyan-blue -> lemon-yellow 发散色图。

    负值表示 CME higher，正值表示 UME higher，0 附近和背景为深青蓝色。
    masked/invalid grids 由 cmap.set_bad 单独设置为同样的深青蓝背景色。
    """

    cmap = LinearSegmentedColormap.from_list(
        "fig2a_cme_deepteal_ume_smooth",
        [
            (0.0, COLOR_CME_HIGHER),
            (0.42, "#12D78E"),
            (0.50, COLOR_NEAR_ZERO),
            (0.58, "#B9D95C"),
            (1.0, COLOR_UME_HIGHER),
        ],
        N=256,
    )
    cmap.set_bad(COLOR_INVALID)
    return cmap


def nice_symmetric_limit(values: Iterable[float]) -> float:
    """用 |DeltaFR| 的 95th percentile 计算好看的对称色条范围。

    返回值 V 用于 [-V, +V]。为了便于读图，V 会向上取到较整齐的刻度。
    """

    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0

    raw = float(np.percentile(np.abs(arr), 95))
    if raw <= 1:
        step = 0.25
    elif raw <= 3:
        step = 0.5
    elif raw <= 8:
        step = 1.0
    else:
        step = 2.0
    return float(np.ceil(raw / step) * step)


def resolve_left_to_right_direction(df: pd.DataFrame) -> tuple[int, str, str]:
    """从分析表中自动识别 left-to-right 方向。

    当前正式分析使用 direction_code == "R" 和 direction_name == "right"。
    为了让脚本对后续表格命名更稳健，这里同时检查 code 与 name 的多个候选写法。
    """

    direction_cols = ["direction_code", "direction_name"]
    for col in direction_cols:
        normalized = df[col].astype(str).str.strip()
        candidates = normalized.str.lower().isin({x.lower() for x in LEFT_TO_RIGHT_CANDIDATES})
        hit = df.loc[candidates, ["direction_id", "direction_code", "direction_name"]].drop_duplicates()
        if not hit.empty:
            row = hit.iloc[0]
            return int(row["direction_id"]), str(row["direction_code"]), str(row["direction_name"])
    raise ValueError(
        "Cannot resolve left-to-right direction. "
        f"Available direction_code values: {sorted(df['direction_code'].dropna().unique())}"
    )


def load_selected_grid_results() -> tuple[pd.DataFrame, int, str, str]:
    """读取正式分析表，并筛选 Fig. 2A 所需的固定条件。"""

    if not INPUT_TABLE.exists():
        raise FileNotFoundError(f"Input table not found: {INPUT_TABLE}")

    df = pd.read_csv(INPUT_TABLE)
    required_columns = {
        "grid_n",
        "pair_id",
        "direction_id",
        "direction_code",
        "direction_name",
        "grid_x",
        "grid_y",
        "valid_grid",
        "delta_mean_fr_hz",
        "UME_unit_count",
        "CME_unit_count",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {INPUT_TABLE}: {missing}")

    direction_id, direction_code, direction_name = resolve_left_to_right_direction(df)
    selected = df[
        (df["grid_n"] == GRID_N)
        & (df["pair_id"].isin(PAIR_ORDER))
        & (df["direction_id"] == direction_id)
    ].copy()

    if selected.empty:
        raise RuntimeError(
            f"No rows selected for grid_n={GRID_N}, pairs={PAIR_ORDER}, direction_id={direction_id}"
        )

    # 保证 paired retina 的显示顺序固定。
    selected["pair_id"] = pd.Categorical(selected["pair_id"], categories=PAIR_ORDER, ordered=True)
    return selected, direction_id, direction_code, direction_name


def grid_to_matrix(pair_df: pd.DataFrame) -> np.ma.MaskedArray:
    """将一个 pair 的 grid-cell 长表转换为 20x20 masked matrix。

    matrix[row, col] 对应 grid_y, grid_x。
    valid_grid == False 的格子会被 mask，绘图时显示为 COLOR_INVALID。
    """

    mat = np.full((GRID_N, GRID_N), np.nan, dtype=float)
    mask = np.ones((GRID_N, GRID_N), dtype=bool)

    for row in pair_df.itertuples(index=False):
        gx = int(row.grid_x)
        gy = int(row.grid_y)
        if 0 <= gx < GRID_N and 0 <= gy < GRID_N:
            mat[gy, gx] = float(row.delta_mean_fr_hz)
            mask[gy, gx] = not bool(row.valid_grid)

    return np.ma.array(mat, mask=mask)


def summarize_pair(pair_df: pd.DataFrame, pair_id: str) -> dict[str, float | int | str]:
    """生成终端核查用的 pair-level DeltaFR 摘要。"""

    valid = pair_df[pair_df["valid_grid"].astype(bool)].copy()
    vals = valid["delta_mean_fr_hz"].astype(float)
    return {
        "pair": pair_id,
        "valid_grid_count": int(len(valid)),
        "delta_min": float(vals.min()) if len(vals) else np.nan,
        "delta_max": float(vals.max()) if len(vals) else np.nan,
        "delta_mean": float(vals.mean()) if len(vals) else np.nan,
        "delta_median": float(vals.median()) if len(vals) else np.nan,
        "positive_count": int((vals > 0).sum()),
        "negative_count": int((vals < 0).sum()),
        "near_zero_count": int((vals.abs() <= NEAR_ZERO_THRESHOLD_HZ).sum()),
    }


def draw_variant(
    selected: pd.DataFrame,
    summaries: list[dict[str, float | int | str]],
    vlim: float,
    variant: FigureVariant,
) -> list[Path]:
    """绘制并保存一个 Fig. 2A 版本。"""

    cmap = make_delta_cmap()
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    fig = plt.figure(figsize=variant.figsize, dpi=DPI)
    width_ratios = [1.0, 1.0, 1.0, 0.075]
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=width_ratios,
        left=0.045,
        right=0.93 if not variant.show_helper_legend else 0.89,
        bottom=0.16 if not variant.show_helper_legend else 0.23,
        top=0.86,
        wspace=0.11,
    )

    mesh = None
    for i, pair_id in enumerate(PAIR_ORDER):
        ax = fig.add_subplot(gs[0, i])
        pair_df = selected[selected["pair_id"].astype(str) == pair_id]
        matrix = grid_to_matrix(pair_df)

        # 使用 imshow + bicubic 插值绘制更平滑的空间响应图。
        # 这里不绘制网格线和外框线，使 20x20 离散网格呈现为柔和的局部响应图。
        # invalid grid 在当前视觉方案中作为深青蓝背景处理；
        # 因此绘图时将 masked grid 填为 0，再做插值，避免 imshow 对 masked array
        # 的插值把孤立有效格过度吞掉。
        smooth_matrix = matrix.filled(0.0)
        mesh = ax.imshow(
            smooth_matrix,
            cmap=cmap,
            norm=norm,
            interpolation="bicubic",
            origin="upper",
            extent=(0, GRID_N, GRID_N, 0),
            resample=True,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, GRID_N)
        ax.set_ylim(0, GRID_N)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(PAIR_LABELS[pair_id], color=COLOR_TEXT, fontweight="bold", pad=variant.title_pad)

        for spine in ax.spines.values():
            spine.set_visible(False)

    cax = fig.add_subplot(gs[0, 3])
    assert mesh is not None
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.outline.set_linewidth(0.7)
    cbar.outline.set_edgecolor(COLOR_TEXT)
    cbar.ax.tick_params(length=2.5, width=0.7, colors=COLOR_TEXT, labelsize=8)
    cbar.set_ticks([-vlim, 0, vlim])
    cbar.set_ticklabels([f"-{vlim:g}", "0", f"{vlim:g}"])
    cbar.set_label(variant.colorbar_label, color=COLOR_TEXT, labelpad=5)

    if variant.show_helper_legend:
        handles = [
            Patch(facecolor=COLOR_UME_HIGHER, edgecolor="none", label="UME higher"),
            Patch(facecolor=COLOR_CME_HIGHER, edgecolor="none", label="CME higher"),
            Patch(facecolor=COLOR_NEAR_ZERO, edgecolor="none", label="near zero / background"),
            Patch(facecolor=COLOR_INVALID, edgecolor="none", label="invalid grid"),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.47, 0.055),
            ncol=4,
            frameon=False,
            handlelength=1.15,
            columnspacing=1.35,
            fontsize=8.2,
        )

    out_base = OUTPUT_DIR / f"fig2A_representative_deltaFR_maps_20x20_R_{variant.suffix}"
    saved_paths = []
    for ext in ["png", "pdf", "svg"]:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.035)
        saved_paths.append(out_path)
    plt.close(fig)
    return saved_paths


def main() -> None:
    """脚本入口：读取数据、核查、绘制两个版本并保存。"""

    set_paper_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected, direction_id, direction_code, direction_name = load_selected_grid_results()

    valid_delta = selected.loc[selected["valid_grid"].astype(bool), "delta_mean_fr_hz"].astype(float)
    vlim = nice_symmetric_limit(valid_delta)

    summaries = []
    for pair_id in PAIR_ORDER:
        pair_df = selected[selected["pair_id"].astype(str) == pair_id]
        summaries.append(summarize_pair(pair_df, pair_id))

    all_saved = []
    for variant in VARIANTS:
        all_saved.extend(draw_variant(selected, summaries, vlim, variant))

    # 终端核查信息：方便确认方向、有效网格、DeltaFR 统计和输出路径。
    # 同时写入 txt 日志，方便后续整理论文图时复查。
    log_lines = []
    log_lines.append("Fig. 2A Representative spatial DeltaFR maps")
    log_lines.append(f"Input table: {INPUT_TABLE}")
    log_lines.append(f"Grid size: {GRID_N}x{GRID_N}")
    log_lines.append(
        "Resolved left-to-right direction: "
        f"direction_id={direction_id}, direction_code={direction_code}, direction_name={direction_name}"
    )
    log_lines.append(f"Near-zero threshold for QC counts: |DeltaFR| <= {NEAR_ZERO_THRESHOLD_HZ:g} Hz")
    log_lines.append(f"Unified colorbar range: [-{vlim:g}, +{vlim:g}] Hz")
    log_lines.append("")
    for item in summaries:
        log_lines.append(
            f"{PAIR_LABELS[str(item['pair'])]} ({item['pair']}): "
            f"valid={item['valid_grid_count']}, "
            f"min={item['delta_min']:.3f}, max={item['delta_max']:.3f}, "
            f"mean={item['delta_mean']:.3f}, median={item['delta_median']:.3f}, "
            f"positive={item['positive_count']}, negative={item['negative_count']}, "
            f"near_zero={item['near_zero_count']}"
        )
    log_lines.append("")
    log_lines.append("Output files:")
    for path in all_saved:
        log_lines.append(str(path))

    log_path = OUTPUT_DIR / "fig2A_representative_deltaFR_maps_20x20_R_qc_log.txt"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log_lines.append(str(log_path))
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
