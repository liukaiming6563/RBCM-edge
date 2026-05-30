"""
绘制 UME/CME paired retina 的二维 sorted-unit 空间分布图。

本脚本用于论文图候选 panel：
    每张图对应一组 paired retina，将同一视网膜切片下 UME recording 与 CME recording
    的 good units 叠加在同一二维空间坐标中展示。

注意：
    - 这里展示的是 sorting 后的 good units 空间位置分布；
    - UME 与 CME 来自不同 recording / spike sorting，因此点不是一一对应的同一个 RGC；
    - 该图用于说明 spatially matched local population analysis 的空间采样基础。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# 1. 路径与配对配置
# =============================================================================

PROJECT_ROOT = Path(r"D:\study\project\RBCM-Edge")
DATA_ROOT = PROJECT_ROOT / "MEA_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "MEA_analysis" / "figure_tmp"

# 正式论文命名：
#   UME = Uniform-background moving edge，对应原 single-edge recording
#   CME = Contextual-background moving edge，对应原 double-edge recording
PAIR_CONFIG = {
    "pair_31_32": {"CME": "000031", "UME": "000032", "label": "P31-32"},
    "pair_34_35": {"CME": "000034", "UME": "000035", "label": "P34-35"},
    "pair_37_38": {"CME": "000037", "UME": "000038", "label": "P37-38"},
}


# =============================================================================
# 2. 视觉风格
# =============================================================================

DPI = 600
COLOR_UME = "#F4C430"  # warm lemon/yellow-orange
COLOR_CME = "#31C6B8"  # cyan-green
COLOR_TEXT = "#E8ECEF"
COLOR_BG = "#FFFFFF"


def set_paper_style() -> None:
    """设置接近期刊主图的基础 Matplotlib 风格。"""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 12,
            "legend.fontsize": 8.5,
            "figure.facecolor": COLOR_BG,
            "axes.facecolor": COLOR_BG,
            "savefig.facecolor": COLOR_BG,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


# =============================================================================
# 3. 数据读取与 unit 坐标重建
# =============================================================================

def read_cluster_group(path: Path) -> pd.DataFrame:
    """读取 cluster_group.tsv，返回 cluster_id 与 group 标签。

    不同 Kilosort/Phy 版本导出的列名可能略有不同，所以这里做了兼容处理。
    """

    df = pd.read_csv(path, sep="\t")

    cluster_col = "cluster_id" if "cluster_id" in df.columns else df.columns[0]
    group_col = "group" if "group" in df.columns else df.columns[-1]

    out = df[[cluster_col, group_col]].copy()
    out.columns = ["cluster_id", "group"]
    out["cluster_id"] = out["cluster_id"].astype(int)
    out["group"] = out["group"].astype(str).str.lower().str.strip()
    return out


def load_good_unit_positions(exp_id: str) -> pd.DataFrame:
    """从 spike-level 坐标重建 good sorted-unit 的二维位置。

    输入文件：
        spike_positions.npy: shape = [n_spikes, 2]，每个 spike 的二维位置；
        spike_clusters.npy: shape = [n_spikes]，每个 spike 的 cluster id；
        cluster_group.tsv: 每个 cluster 的 sorting 标签。

    对每个 good cluster，取该 cluster 所有 spike 位置的 median 作为 unit 位置。
    median 对少量离群 spike 更稳健。
    """

    ks_dir = DATA_ROOT / exp_id / "kilosort4"
    pos_path = ks_dir / "spike_positions.npy"
    clu_path = ks_dir / "spike_clusters.npy"
    group_path = ks_dir / "cluster_group.tsv"

    for path in [pos_path, clu_path, group_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    spike_positions = np.load(pos_path)
    spike_clusters = np.load(clu_path).astype(int)
    cluster_group = read_cluster_group(group_path)
    good_ids = set(cluster_group.loc[cluster_group["group"] == "good", "cluster_id"].tolist())

    records = []
    # 逐个 cluster 统计位置。当前数据规模下足够快，而且代码更容易核查。
    for cid in sorted(good_ids):
        mask = spike_clusters == cid
        if not np.any(mask):
            continue
        xy = spike_positions[mask]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if xy.size == 0:
            continue
        x = float(np.median(xy[:, 0]))
        y = float(np.median(xy[:, 1]))
        records.append(
            {
                "experiment": exp_id,
                "cluster_id": int(cid),
                "x": x,
                "y": y,
                "n_spikes": int(xy.shape[0]),
            }
        )

    out = pd.DataFrame(records)
    if out.empty:
        raise RuntimeError(f"No good units were found for experiment {exp_id}")
    return out


# =============================================================================
# 4. 绘图
# =============================================================================

def point_sizes(df: pd.DataFrame) -> np.ndarray:
    """用 spike 数量生成轻微变化的点大小，仅作为视觉辅助。"""

    vals = np.log1p(df["n_spikes"].to_numpy(dtype=float))
    low, high = np.percentile(vals, [5, 95])
    scaled = np.clip((vals - low) / (high - low + 1e-9), 0, 1)
    # 点不要太大，否则 UME/CME 重叠时后绘制的颜色会遮挡前一层。
    return 7 + scaled * 14


def plot_pair(pair_id: str, cfg: dict[str, str]) -> list[Path]:
    """绘制一组 paired retina 的 UME/CME good-unit 二维分布图。"""

    ume = load_good_unit_positions(cfg["UME"])
    cme = load_good_unit_positions(cfg["CME"])

    fig, ax = plt.subplots(figsize=(4.0, 3.55), dpi=DPI)

    # 先画 CME，再画 UME；二者 alpha 较低，重叠区域自然呈现混合。
    ax.scatter(
        cme["x"],
        cme["y"],
        s=point_sizes(cme),
        c=COLOR_CME,
        # 透明度降低，让 UME/CME 重叠区域仍能同时看见两类 unit。
        alpha=0.36,
        linewidths=0,
        label=f"CME good units (n={len(cme)})",
        rasterized=True,
    )
    ax.scatter(
        ume["x"],
        ume["y"],
        s=point_sizes(ume),
        c=COLOR_UME,
        alpha=0.34,
        linewidths=0,
        label=f"UME good units (n={len(ume)})",
        rasterized=True,
    )

    both = pd.concat([ume, cme], ignore_index=True)
    xmin, xmax = both["x"].min(), both["x"].max()
    ymin, ymax = both["y"].min(), both["y"].max()
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - 0.045 * dx, xmax + 0.045 * dx)
    ax.set_ylim(ymin - 0.045 * dy, ymax + 0.045 * dy)
    ax.set_aspect("equal", adjustable="box")

    # 论文 panel 风格：不显示标题和坐标刻度，只保留 unit 空间分布与简洁图例。
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLOR_UME,
               markeredgecolor="none", markersize=6, label=f"UME, n={len(ume)}"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLOR_CME,
               markeredgecolor="none", markersize=6, label=f"CME, n={len(cme)}"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        borderpad=0.2,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT)

    out_base = OUTPUT_DIR / f"{pair_id}_UME_CME_good_unit_spatial_distribution"
    saved = []
    for ext in ["png", "pdf", "svg"]:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.025)
        saved.append(out_path)
    plt.close(fig)

    print(
        f"{cfg['label']}: UME={cfg['UME']} good_units={len(ume)}, "
        f"CME={cfg['CME']} good_units={len(cme)}"
    )
    for path in saved:
        print(path)
    return saved


def main() -> None:
    """脚本入口：生成三张 paired retina unit 分布图。"""

    set_paper_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Paired UME/CME good-unit spatial distribution figures")
    print(f"Output directory: {OUTPUT_DIR}")
    for pair_id, cfg in PAIR_CONFIG.items():
        plot_pair(pair_id, cfg)


if __name__ == "__main__":
    main()
