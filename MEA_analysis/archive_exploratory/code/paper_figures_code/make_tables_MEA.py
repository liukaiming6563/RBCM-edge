"""生成 RBCM-edge MEA 论文主表和补充表。

表格以论文阅读为目标，字段命名更简洁，pair 名称使用 P31-32
这类短标签，并保留三位小数。原始完整结果仍保存在正式分析输出表中。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fig_style import pair_label
from load_results import MEAResults


def _round_numeric(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(digits)
    return out


def _export_table(df: pd.DataFrame, out_base: Path) -> list[Path]:
    """同时导出 csv/xlsx/tex/md。"""

    out_base.parent.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    csv_path = out_base.with_suffix(".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    files.append(csv_path)

    xlsx_path = out_base.with_suffix(".xlsx")
    try:
        df.to_excel(xlsx_path, index=False)
        files.append(xlsx_path)
    except Exception as exc:  # pragma: no cover - 取决于用户环境是否安装 openpyxl
        (out_base.with_suffix(".xlsx_error.txt")).write_text(str(exc), encoding="utf-8")

    tex_path = out_base.with_suffix(".tex")
    tex_path.write_text(df.to_latex(index=False, escape=False), encoding="utf-8")
    files.append(tex_path)

    md_path = out_base.with_suffix(".md")
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    files.append(md_path)
    return files


def make_table_1(results: MEAResults, out_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    """主文 Table 1：12x12 和 16x16 paired-retina summary。"""

    df = results.paired_overall[results.paired_overall["grid_n"].isin([12, 16])].copy()
    table = pd.DataFrame(
        {
            "Pair": df["pair_id"].map(pair_label),
            "Grid scale": df["grid_n"].astype(int).astype(str) + "×" + df["grid_n"].astype(int).astype(str),
            "Different fraction": df["mean_different_fraction"],
            "Mean |ΔFR| (Hz)": df["mean_abs_delta_mean_fr_hz"],
            "Mean |NDI|": df["mean_abs_NDI_mean"],
            "Mean normalized MAE": df["mean_normalized_MAE"],
            "Valid grids": df["total_valid_grids"].astype(int),
            "UME-dominant grids": df["total_UME_higher"].astype(int),
            "CME-dominant grids": df["total_CME_higher"].astype(int),
            "Similar grids": df["total_similar"].astype(int),
        }
    )
    table = _round_numeric(table)
    files = _export_table(table, out_dir / "table_1_paired_retina_summary")
    return table, files


def make_supp_table_s1(results: MEAResults, out_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Supplementary Table S1：所有 grid scale 的 paired-retina summary。"""

    df = results.paired_overall.copy()
    table = pd.DataFrame(
        {
            "Pair": df["pair_id"].map(pair_label),
            "Grid scale": df["grid_n"].astype(int).astype(str) + "×" + df["grid_n"].astype(int).astype(str),
            "Different fraction": df["mean_different_fraction"],
            "Mean |ΔFR| (Hz)": df["mean_abs_delta_mean_fr_hz"],
            "Mean |NDI|": df["mean_abs_NDI_mean"],
            "Mean normalized MAE": df["mean_normalized_MAE"],
            "Valid grids": df["total_valid_grids"].astype(int),
            "UME-dominant grids": df["total_UME_higher"].astype(int),
            "CME-dominant grids": df["total_CME_higher"].astype(int),
            "Similar grids": df["total_similar"].astype(int),
        }
    )
    table = _round_numeric(table)
    files = _export_table(table, out_dir / "supp_table_S1_all_scale_paired_summary")
    return table, files


def make_supp_table_s2(results: MEAResults, out_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Supplementary Table S2：阈值敏感性。"""

    df = results.threshold_pair.copy()
    table = pd.DataFrame(
        {
            "Pair": df["pair_id"].map(pair_label),
            "Grid scale": df["grid_n"].astype(int).astype(str) + "×" + df["grid_n"].astype(int).astype(str),
            "Threshold (Hz)": df["threshold_hz"],
            "Different fraction": df["mean_different_fraction"],
            "UME-dominant fraction": df["mean_UME_higher_fraction"],
            "CME-dominant fraction": df["mean_CME_higher_fraction"],
            "Similar fraction": df["mean_similar_fraction"],
            "Valid grids": df["total_valid_grids"].astype(int),
        }
    )
    table = _round_numeric(table)
    files = _export_table(table, out_dir / "supp_table_S2_threshold_sensitivity")
    return table, files


def make_supp_table_s3(results: MEAResults, out_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Supplementary Table S3：空间响应图相似性。"""

    df = results.spatial.copy()
    table = pd.DataFrame(
        {
            "Pair": df["pair_id"].map(pair_label),
            "Grid scale": df["grid_n"].astype(int).astype(str) + "×" + df["grid_n"].astype(int).astype(str),
            "Direction": df["direction_code"],
            "Pearson r": df["pearson_r"],
            "Spearman r": df["spearman_r"],
            "Cosine similarity": df["cosine_similarity"],
            "Normalized MAE": df["normalized_MAE"],
            "Normalized RMSE": df["normalized_RMSE"],
            "Mean |ΔFR| (Hz)": df["mean_abs_delta_mean_fr_hz"],
            "Mean |NDI|": df["mean_abs_NDI_mean"],
        }
    )
    table = _round_numeric(table)
    files = _export_table(table, out_dir / "supp_table_S3_spatial_map_similarity")
    return table, files


def export_all_tables(results: MEAResults, out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """导出全部论文表格，并返回 table manifest。"""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for table_id, maker, desc in [
        ("Table 1", make_table_1, "Main-scale paired-retina summary."),
        ("Supplementary Table S1", make_supp_table_s1, "All-scale paired-retina summary."),
        ("Supplementary Table S2", make_supp_table_s2, "Threshold sensitivity summary."),
        ("Supplementary Table S3", make_supp_table_s3, "Spatial response map similarity summary."),
    ]:
        table, files = maker(results, out_dir)
        manifest_rows.append(
            {
                "table_id": table_id,
                "n_rows": len(table),
                "files": ";".join(str(path) for path in files),
                "description": desc,
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "tables_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest, pd.DataFrame(manifest_rows)

