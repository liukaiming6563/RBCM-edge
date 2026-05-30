"""一键导出 RBCM-edge MEA 论文风格主图、补充图和表格。

运行示例：

python D:/study/project/RBCM-Edge/MEA_analysis/paper_figures_code/export_all_figures.py

也可以指定路径：

python export_all_figures.py --input_dir D:/study/project/RBCM-Edge/outputs/MEA_analysis/tables --output_dir D:/study/project/RBCM-Edge/outputs/MEA_analysis/figures_paper_style
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from fig_style import set_paper_style
from load_results import load_all_results
from make_tables_MEA import export_all_tables
from plot_main_figure_MEA import plot_main_figure
from plot_supp_figures_MEA import plot_all_supplementary_figures


PROJECT_DIR = Path(r"D:\study\project\RBCM-Edge")
DEFAULT_INPUT_DIR = PROJECT_DIR / "outputs" / "MEA_analysis" / "tables"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "MEA_analysis" / "figures_paper_style"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-style MEA figures and tables for RBCM-edge.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing formal MEA result CSV tables.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for paper-style figures and tables.")
    parser.add_argument("--skip_supp", action="store_true", help="Only export main figure and tables.")
    return parser.parse_args()


def write_figure_legends(out_dir: Path) -> Path:
    """写出主图、补充图和表格图注草稿。"""

    legends_dir = out_dir / "legends"
    legends_dir.mkdir(parents=True, exist_ok=True)
    text = """# MEA Paper Figure Legends

## Main Figure

**Figure X. Edge-context-dependent retinal population responses measured by MEA.**
(A) Schematic of UME and CME stimulation and MEA recording workflow. UME denotes a uniform-background moving edge, while CME denotes a contextual-background moving edge containing a moving edge embedded in a static background context. (B) Spatially matched grid-level population analysis. Sorted units were not matched one-to-one across UME and CME recordings; instead, good units within each spatial grid were treated as a local RGC population. (C) Representative grid-level ΔFR maps, where ΔFR = FR_UME - FR_CME. Red indicates UME-dominant regions and blue indicates CME-dominant regions. Gray cells indicate insufficient units. Color scale was clipped using a robust symmetric range for visualization. (D) Paired-retina-level summary of different-grid fraction, mean absolute ΔFR, mean absolute NDI, and normalized MAE at 12×12 and 16×16 grid scales. (E) Spatial response map dissimilarity quantified by normalized MAE. (F) Grid-scale robustness of the different-grid fraction across multiple grid resolutions. Grid classification is descriptive and based on predefined firing-rate thresholds, not grid-wise statistical significance. These results indicate that edge context reshapes local RGC population response patterns rather than inducing a simple global gain change.

## Supplementary Figures

**Supplementary Figure S1. All-direction ΔFR maps at 12×12 grid scale.**
Grid-level ΔFR maps for three paired retinal preparations and eight movement directions. Red indicates UME-dominant local population responses, blue indicates CME-dominant responses, and gray indicates insufficient units.

**Supplementary Figure S2. All-direction ΔFR maps at 16×16 grid scale.**
Same as Supplementary Figure S1, but using a 16×16 spatial grid.

**Supplementary Figure S3. Grid-scale robustness.**
Mean different fraction, mean absolute ΔFR, and mean normalized MAE across grid scales. Each line represents one paired retinal preparation.

**Supplementary Figure S4. Threshold sensitivity.**
Different-grid fraction under multiple ΔFR thresholds. The heatmap summarizes threshold sensitivity across grid scales and paired retinal preparations.

**Supplementary Figure S5. NDI maps at 12×12 grid scale.**
Spatial maps of normalized difference index, NDI = (FR_UME - FR_CME) / (FR_UME + FR_CME + ε). Red indicates relative UME preference and blue indicates relative CME preference.

**Supplementary Figure S6. Responsive fraction difference maps.**
Maps of Δ responsive fraction, defined as UME_nonzero_fraction - CME_nonzero_fraction, at the 12×12 grid scale.

**Supplementary Figure S7. Robust metric comparison.**
Representative maps comparing Δ mean FR, Δ median FR, Δ nonzero mean FR, Δ responsive fraction, and NDI for one data-driven representative direction per paired retinal preparation.

**Supplementary Figure S8. UME versus CME spatial response map scatter.**
Each point represents a valid spatial grid. Dashed lines mark y = x. Pearson correlation and normalized MAE summarize map similarity/dissimilarity.

**Supplementary Figure S9. Valid grid and unit count quality control.**
Valid grid fraction and unit count summaries across grid scales and paired retinal preparations.

**Supplementary Figure S10. Permutation null QC.**
Approximate null summary plots for mean absolute ΔFR based on stored permutation null mean and standard deviation. This stricter permutation comparison was used as supplementary quality control and not as the primary evidence for RBCM motivation.

## Tables

**Table 1. Paired-retina-level summary at main grid scales.**
Summary of different fraction, mean absolute ΔFR, mean absolute NDI, mean normalized MAE, and grid-class counts for 12×12 and 16×16 grid scales.

**Supplementary Table S1. All-scale paired-retina summary.**
Paired-retina-level summary across all tested grid scales.

**Supplementary Table S2. Threshold sensitivity summary.**
Different fraction and grid-class fractions across ΔFR classification thresholds.

**Supplementary Table S3. Spatial response map similarity summary.**
Direction-level spatial map similarity and dissimilarity metrics, including Pearson r, Spearman r, cosine similarity, normalized MAE, normalized RMSE, mean absolute ΔFR, and mean absolute NDI.
"""
    path = legends_dir / "figure_legends_MEA.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["main_figures", "supplementary_figures", "source_panels", "tables", "legends", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "logs" / "paper_figure_export_log.txt"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Start export: {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"Input dir: {args.input_dir}\n")
        log.write(f"Output dir: {args.output_dir}\n")
        set_paper_style()
        results = load_all_results(args.input_dir)
        log.write("Loaded result tables.\n")

        figure_rows: list[dict] = []
        plot_main_figure(results, out_dir, figure_rows)
        log.write("Exported main figure and individual panels.\n")

        if not args.skip_supp:
            plot_all_supplementary_figures(results, out_dir, figure_rows)
            log.write("Exported supplementary figures.\n")

        figures_manifest = pd.DataFrame(figure_rows)
        figures_manifest_path = out_dir / "figures_manifest.csv"
        figures_manifest.to_csv(figures_manifest_path, index=False, encoding="utf-8-sig")
        log.write(f"Figure manifest: {figures_manifest_path}\n")

        table_manifest, _ = export_all_tables(results, out_dir / "tables")
        log.write(f"Exported tables: {len(table_manifest)} table groups.\n")

        legends_path = write_figure_legends(out_dir)
        log.write(f"Legends: {legends_path}\n")
        log.write(f"Finished export: {datetime.now().isoformat(timespec='seconds')}\n")
    print(f"Paper-style MEA figures exported to: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise

