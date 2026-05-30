"""RBCM-edge MEA 论文图统一风格工具。

本文件集中定义论文图所需的字体、颜色、色图、panel label、
保存函数和热图绘制函数。所有主图与补充图都调用这里的函数，
避免不同图之间出现字体、颜色、线宽和色条风格不一致的问题。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# 颜色语义固定：UME 用暖色，CME 用冷色，pair 用色盲友好配色。
# 主图使用克制的 warm/cool 配色，避免默认 Matplotlib 色彩感。
UME_COLOR = "#C95F3F"
CME_COLOR = "#3B6FA1"
SIMILAR_COLOR = "#D9D9D9"
MASK_COLOR = "#EEEEEE"
GRID_COLOR = "#BDBDBD"
TEXT_COLOR = "#222222"

PAIR_COLORS = {
    "pair_31_32": "#3B6FA1",
    "pair_34_35": "#C95F3F",
    "pair_37_38": "#2A9D8F",
}

PAIR_LABELS = {
    "pair_31_32": "P31-32",
    "pair_34_35": "P34-35",
    "pair_37_38": "P37-38",
}

DIR_ORDER = ["R", "RU", "U", "LU", "L", "LD", "D", "RD"]


def set_paper_style() -> None:
    """设置全局 Matplotlib 风格。

    采用白底、无上/右边框、较细坐标轴和嵌入式 PDF 字体，
    方便后续在 Illustrator / Inkscape 中继续编辑。
    """

    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.8,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def mm_to_inch(width_mm: float, height_mm: float) -> tuple[float, float]:
    """毫米转英寸，便于按期刊双栏尺寸设置图大小。"""

    return width_mm / 25.4, height_mm / 25.4


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.06) -> None:
    """给 panel 左上角添加统一的大写字母标签。"""

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
        color=TEXT_COLOR,
    )


def save_figure(fig: plt.Figure, path_base: Path, tiff: bool = False) -> list[Path]:
    """同时保存 PDF、SVG、PNG，可选 TIFF。

    参数 path_base 不带扩展名，例如 ``.../main_figure``。
    返回实际生成的文件路径列表，便于写入 manifest。
    """

    path_base.parent.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for ext in ["pdf", "svg", "png"]:
        out = path_base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight")
        files.append(out)
    if tiff:
        out = path_base.with_suffix(".tiff")
        fig.savefig(out, dpi=600, bbox_inches="tight")
        files.append(out)
    return files


def make_delta_cmap() -> LinearSegmentedColormap:
    """构建 UME-CME 发散色图：负值蓝色，0 白色，正值橙红色。"""

    cmap = LinearSegmentedColormap.from_list(
        "ume_cme_delta",
        [
            (0.0, "#2B6CA3"),
            (0.48, "#F8F8F8"),
            (0.52, "#F8F8F8"),
            (1.0, "#B93A45"),
        ],
    )
    cmap.set_bad(MASK_COLOR)
    return cmap


def robust_symmetric_vmax(values: Iterable[float], percentile: float = 95.0, floor: float = 1.0) -> float:
    """根据绝对值分位数得到对称色条范围，避免少数极端值主导热图。"""

    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return floor
    return max(floor, float(np.nanpercentile(np.abs(arr), percentile)))


def matrix_from_grid(df, value_col: str, grid_n: int, mask_invalid: bool = True) -> np.ndarray:
    """将 grid-cell 长表转换为 N x N 矩阵。

    y 方向按照 grid_y 从小到大填入矩阵；绘图时使用 origin="lower"，
    因而图像下方对应较小的 grid_y。
    """

    mat = np.full((grid_n, grid_n), np.nan, dtype=float)
    for row in df.itertuples(index=False):
        value = getattr(row, value_col)
        valid = bool(getattr(row, "valid_grid")) if hasattr(row, "valid_grid") else True
        if mask_invalid and not valid:
            value = np.nan
        mat[int(getattr(row, "grid_y")), int(getattr(row, "grid_x"))] = value
    return mat


def plot_masked_grid_map(
    ax: plt.Axes,
    df,
    value_col: str,
    grid_n: int,
    *,
    cmap=None,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "",
    show_ticks: bool = False,
):
    """绘制 masked grid heatmap。

    无效网格用 colormap 的 bad color 显示为浅灰，避免和 0 值混淆。
    """

    cmap = cmap or make_delta_cmap()
    mat = matrix_from_grid(df, value_col, grid_n, mask_invalid=True)
    im = ax.imshow(mat, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, pad=3)
    if show_ticks:
        ax.set_xlabel("Retinal grid x")
        ax.set_ylabel("Retinal grid y")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#555555")
    return im


def add_compact_colorbar(
    fig: plt.Figure,
    im,
    ax_or_axes,
    label: str,
    *,
    orientation: str = "horizontal",
    fraction: float = 0.045,
    pad: float = 0.04,
):
    """添加紧凑色条。"""

    cb = fig.colorbar(im, ax=ax_or_axes, orientation=orientation, fraction=fraction, pad=pad)
    cb.set_label(label, labelpad=2)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, length=2)
    return cb


def prettify_axis(ax: plt.Axes, grid: bool = False) -> None:
    """统一坐标轴细节。"""

    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#333333")
    if grid:
        ax.grid(True, color="#E5E5E5", linewidth=0.6, zorder=0)
    else:
        ax.grid(False)


def pair_label(pair_id: str) -> str:
    """pair_31_32 -> P31-32。"""

    return PAIR_LABELS.get(pair_id, pair_id.replace("pair_", "P").replace("_", "-"))
