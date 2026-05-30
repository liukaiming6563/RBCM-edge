# MEA paper-style figure export

本目录用于把已经完成的 MEA 正式分析结果重绘为论文风格主图、补充图和表格。

输入数据来自：

```text
D:\study\project\RBCM-Edge\outputs\MEA_analysis\tables
```

输出结果保存到：

```text
D:\study\project\RBCM-Edge\outputs\MEA_analysis\figures_paper_style
```

运行：

```powershell
C:\App\anaconda\envs\lkm1\python.exe D:\study\project\RBCM-Edge\MEA_analysis\paper_figures_code\export_all_figures.py
```

代码结构：

- `fig_style.py`：统一字体、颜色、色图、保存和热图绘制函数。
- `load_results.py`：读取并校验正式分析 CSV 表格。
- `plot_main_figure_MEA.py`：绘制主图和 A-F 拆分 panel。
- `plot_supp_figures_MEA.py`：绘制 Supplementary Figure S1-S10。
- `make_tables_MEA.py`：导出 Table 1 和 Supplementary Tables。
- `export_all_figures.py`：一键运行全部导出。

正式命名：

- UME = Uniform-background moving edge
- CME = Contextual-background moving edge

注意：grid classification 是基于预设阈值的描述性分类，不是单个网格的显著性检验。
