"""
批量绘制 Fig. 2A 候选空间 DeltaFR 图。

用途：
    用户希望在当前 Fig. 2A 平滑视觉风格基础上，额外查看 12x12 和 16x16
    两个网格尺度下 8 个运动方向的所有候选图，后续手动挑选最适合论文主图的版本。

固定规则：
    - 输入使用正式 MEA grid-cell 明细表；
    - 每张候选图包含三个 paired retina: P31-32 / P34-35 / P37-38；
    - DeltaFR = FR_UME - FR_CME；
    - 正值使用柠檬黄，负值使用荧光绿，0/背景/invalid grid 使用深青蓝；
    - 去掉网格线和外框线，用 bicubic 插值呈现更平滑的空间响应图。

注意：
    这些图是为了视觉挑选主图候选，不改变正式分析中的数值统计和有效网格定义。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


# =============================================================================
# 1. 路径与固定参数
# =============================================================================

PROJECT_ROOT = Path(r"D:\study\project\RBCM-Edge")
INPUT_TABLE = (
    PROJECT_ROOT
    / "outputs"
    / "MEA_analysis"
    / "tables"
    / "grid_cell_level_results_all_scales_extended.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "MEA_analysis" / "figure_tmp"

GRID_LIST = [12, 16]
PAIR_ORDER = ["pair_31_32", "pair_34_35", "pair_37_38"]
PAIR_LABELS = {
    "pair_31_32": "P31-32",
    "pair_34_35": "P34-35",
    "pair_37_38": "P37-38",
}

DIRECTION_ORDER = ["R", "RU", "U", "LU", "L", "LD", "D", "RD"]
NEAR_ZERO_THRESHOLD_HZ = 0.5
DPI = 600


# =============================================================================
# 2. 当前 Fig. 2A 平滑版视觉风格
# =============================================================================

COLOR_CME_HIGHER = "#39FF14"  # DeltaFR < 0, fluorescent green
COLOR_NEAR_ZERO = "#073B4C"   # DeltaFR ~= 0, deep cyan-blue
COLOR_UME_HIGHER = "#FFF44F"  # DeltaFR > 0, lemon yellow
COLOR_INVALID = "#073B4C"     # invalid/background, same deep cyan-blue
COLOR_TEXT = "#333333"
COLOR_BG = "#FFFFFF"


@dataclass(frozen=True)
class DirectionInfo:
    """方向元信息，用于文件命名、标题和核查输出。"""

    direction_id: int
    direction_code: str
    direction_name: str


def set_paper_style() -> None:
    """设置统一的论文图风格。"""

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
    """构建荧光绿 -> 深青蓝 -> 柠檬黄的自定义发散色图。"""

    cmap = LinearSegmentedColormap.from_list(
        "fig2a_smooth_green_teal_yellow",
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


def nice_symmetric_limit(values: np.ndarray) -> float:
    """根据 |DeltaFR| 的 95th percentile 得到整齐的对称色条范围。"""

    arr = np.asarray(values, dtype=float)
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


def load_results() -> tuple[pd.DataFrame, list[DirectionInfo]]:
    """读取正式 grid-cell 明细表，并整理 8 个方向的信息。"""

    if not INPUT_TABLE.exists():
        raise FileNotFoundError(f"Input table not found: {INPUT_TABLE}")
    df = pd.read_csv(INPUT_TABLE)

    required = {
        "grid_n",
        "pair_id",
        "direction_id",
        "direction_code",
        "direction_name",
        "grid_x",
        "grid_y",
        "valid_grid",
        "delta_mean_fr_hz",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {INPUT_TABLE}: {missing}")

    directions = (
        df[["direction_id", "direction_code", "direction_name"]]
        .drop_duplicates()
        .sort_values("direction_id")
    )
    order_map = {code: i for i, code in enumerate(DIRECTION_ORDER)}
    directions["sort_key"] = directions["direction_code"].map(order_map).fillna(999).astype(int)
    directions = directions.sort_values(["sort_key", "direction_id"])

    infos = [
        DirectionInfo(int(r.direction_id), str(r.direction_code), str(r.direction_name))
        for r in directions.itertuples(index=False)
        if str(r.direction_code) in DIRECTION_ORDER
    ]
    if len(infos) != 8:
        raise RuntimeError(f"Expected 8 directions, got {len(infos)}: {infos}")
    return df, infos


def grid_to_matrix(pair_df: pd.DataFrame, grid_n: int) -> np.ndarray:
    """把某一 pair/direction/scale 的长表转成 grid_n x grid_n 矩阵。

    invalid grid 用 0 填充，使其在当前视觉方案中显示为深青蓝背景。
    """

    mat = np.zeros((grid_n, grid_n), dtype=float)
    for row in pair_df.itertuples(index=False):
        gx = int(row.grid_x)
        gy = int(row.grid_y)
        if 0 <= gx < grid_n and 0 <= gy < grid_n and bool(row.valid_grid):
            mat[gy, gx] = float(row.delta_mean_fr_hz)
    return mat


def summarize_pair(pair_df: pd.DataFrame, grid_n: int, info: DirectionInfo, pair_id: str) -> dict[str, object]:
    """记录每张候选图中每个 pair 的基础核查信息。"""

    valid = pair_df[pair_df["valid_grid"].astype(bool)].copy()
    vals = valid["delta_mean_fr_hz"].astype(float)
    return {
        "grid_n": grid_n,
        "direction_id": info.direction_id,
        "direction_code": info.direction_code,
        "direction_name": info.direction_name,
        "pair_id": pair_id,
        "valid_grid_count": int(len(valid)),
        "delta_min": float(vals.min()) if len(vals) else np.nan,
        "delta_max": float(vals.max()) if len(vals) else np.nan,
        "delta_mean": float(vals.mean()) if len(vals) else np.nan,
        "delta_median": float(vals.median()) if len(vals) else np.nan,
        "positive_count": int((vals > 0).sum()),
        "negative_count": int((vals < 0).sum()),
        "near_zero_count": int((vals.abs() <= NEAR_ZERO_THRESHOLD_HZ).sum()),
    }


def draw_single_direction_panel(
    df: pd.DataFrame,
    grid_n: int,
    info: DirectionInfo,
    vlim: float,
) -> tuple[list[Path], list[dict[str, object]]]:
    """绘制一个 grid scale + 一个方向下的 3-pair 横排候选图。"""

    selected = df[
        (df["grid_n"] == grid_n)
        & (df["pair_id"].isin(PAIR_ORDER))
        & (df["direction_id"] == info.direction_id)
    ].copy()

    cmap = make_delta_cmap()
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    fig = plt.figure(figsize=(6.6, 2.35), dpi=DPI)
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.075],
        left=0.045,
        right=0.93,
        bottom=0.16,
        top=0.86,
        wspace=0.11,
    )

    summaries: list[dict[str, object]] = []
    image = None
    for i, pair_id in enumerate(PAIR_ORDER):
        pair_df = selected[selected["pair_id"] == pair_id]
        summaries.append(summarize_pair(pair_df, grid_n, info, pair_id))
        mat = grid_to_matrix(pair_df, grid_n)

        ax = fig.add_subplot(gs[0, i])
        image = ax.imshow(
            mat,
            cmap=cmap,
            norm=norm,
            interpolation="bicubic",
            origin="upper",
            extent=(0, grid_n, grid_n, 0),
            resample=True,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, grid_n)
        ax.set_ylim(0, grid_n)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(PAIR_LABELS[pair_id], color=COLOR_TEXT, fontweight="bold", pad=4.0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cax = fig.add_subplot(gs[0, 3])
    assert image is not None
    cbar = fig.colorbar(image, cax=cax)
    cbar.outline.set_linewidth(0.7)
    cbar.outline.set_edgecolor(COLOR_TEXT)
    cbar.ax.tick_params(length=2.5, width=0.7, colors=COLOR_TEXT, labelsize=8)
    cbar.set_ticks([-vlim, 0, vlim])
    cbar.set_ticklabels([f"-{vlim:g}", "0", f"{vlim:g}"])
    cbar.set_label(r"$\Delta$FR (Hz)", color=COLOR_TEXT, labelpad=5)

    out_base = OUTPUT_DIR / (
        f"fig2A_candidate_deltaFR_maps_{grid_n}x{grid_n}_"
        f"dir{info.direction_id:02d}_{info.direction_code}_smooth"
    )
    saved = []
    for ext in ["png", "pdf", "svg"]:
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.035)
        saved.append(path)
    plt.close(fig)
    return saved, summaries


def draw_overview(df: pd.DataFrame, grid_n: int, directions: list[DirectionInfo], vlim: float) -> list[Path]:
    """额外绘制一个总览图：8 个方向 x 3 个 paired retina，方便快速挑选。"""

    cmap = make_delta_cmap()
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    fig = plt.figure(figsize=(7.4, 13.0), dpi=DPI)
    gs = fig.add_gridspec(
        len(directions),
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.07],
        left=0.07,
        right=0.92,
        bottom=0.035,
        top=0.975,
        wspace=0.08,
        hspace=0.12,
    )

    image = None
    for r, info in enumerate(directions):
        selected = df[
            (df["grid_n"] == grid_n)
            & (df["pair_id"].isin(PAIR_ORDER))
            & (df["direction_id"] == info.direction_id)
        ].copy()
        for c, pair_id in enumerate(PAIR_ORDER):
            ax = fig.add_subplot(gs[r, c])
            pair_df = selected[selected["pair_id"] == pair_id]
            mat = grid_to_matrix(pair_df, grid_n)
            image = ax.imshow(
                mat,
                cmap=cmap,
                norm=norm,
                interpolation="bicubic",
                origin="upper",
                extent=(0, grid_n, grid_n, 0),
                resample=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(PAIR_LABELS[pair_id], color=COLOR_TEXT, fontweight="bold", pad=4)
            if c == 0:
                ax.text(
                    -0.08,
                    0.5,
                    info.direction_code,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=COLOR_TEXT,
                )

    cax = fig.add_subplot(gs[:, 3])
    assert image is not None
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_ticks([-vlim, 0, vlim])
    cbar.set_ticklabels([f"-{vlim:g}", "0", f"{vlim:g}"])
    cbar.set_label(r"$\Delta$FR (Hz)", color=COLOR_TEXT, labelpad=5)
    cbar.outline.set_linewidth(0.7)
    cbar.outline.set_edgecolor(COLOR_TEXT)

    out_base = OUTPUT_DIR / f"fig2A_candidate_deltaFR_maps_{grid_n}x{grid_n}_all8dirs_overview_smooth"
    saved = []
    for ext in ["png", "pdf", "svg"]:
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.035)
        saved.append(path)
    plt.close(fig)
    return saved


def main() -> None:
    """批量生成候选图、manifest 和 QC 表。"""

    set_paper_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, directions = load_results()
    manifest_rows = []
    qc_rows = []

    for grid_n in GRID_LIST:
        grid_valid = df[
            (df["grid_n"] == grid_n)
            & (df["pair_id"].isin(PAIR_ORDER))
            & (df["direction_code"].isin(DIRECTION_ORDER))
            & (df["valid_grid"].astype(bool))
        ]["delta_mean_fr_hz"].astype(float)
        vlim = nice_symmetric_limit(grid_valid.to_numpy())

        for info in directions:
            saved, summaries = draw_single_direction_panel(df, grid_n, info, vlim)
            qc_rows.extend(summaries)
            for path in saved:
                manifest_rows.append(
                    {
                        "figure_type": "single_direction_panel",
                        "grid_n": grid_n,
                        "direction_id": info.direction_id,
                        "direction_code": info.direction_code,
                        "path": str(path),
                        "colorbar_vlim_hz": vlim,
                    }
                )

        overview_saved = draw_overview(df, grid_n, directions, vlim)
        for path in overview_saved:
            manifest_rows.append(
                {
                    "figure_type": "all8dirs_overview",
                    "grid_n": grid_n,
                    "direction_id": "all",
                    "direction_code": "all",
                    "path": str(path),
                    "colorbar_vlim_hz": vlim,
                }
            )

        print(f"grid={grid_n}x{grid_n}, unified colorbar range=[-{vlim:g}, +{vlim:g}] Hz")

    manifest = pd.DataFrame(manifest_rows)
    qc = pd.DataFrame(qc_rows)
    manifest_path = OUTPUT_DIR / "fig2A_candidate_deltaFR_maps_12x12_16x16_manifest.csv"
    qc_path = OUTPUT_DIR / "fig2A_candidate_deltaFR_maps_12x12_16x16_qc.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

    print(f"Saved manifest: {manifest_path}")
    print(f"Saved QC table: {qc_path}")
    print(f"Generated {len(manifest)} files.")


if __name__ == "__main__":
    main()
