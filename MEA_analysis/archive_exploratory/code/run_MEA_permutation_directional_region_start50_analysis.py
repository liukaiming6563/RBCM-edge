"""Small runner for directional-region permutation analysis: start-side 0-50%.

This file reuses the existing directional-region analysis implementation and
only changes the condition grid requested by the user:

- movement window: approach_to_center only
- unit region: motion-start side 0-50% only
- grid scales: 8x8 and 12x12
- mirror hypotheses, paired retina, directions: unchanged
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(r"D:\study\project\RBCM-Edge")
MEA_ANALYSIS_DIR = PROJECT_DIR / "MEA_analysis"
sys.path.insert(0, str(MEA_ANALYSIS_DIR))

import run_MEA_permutation_directional_region_analysis as analysis  # noqa: E402


analysis.OUT_DIR = PROJECT_DIR / "outputs" / "MEA_analysis" / "permutation_directional_region_start50_approach"
analysis.TABLE_DIR = analysis.OUT_DIR / "tables"
analysis.REPORT_DIR = analysis.OUT_DIR / "reports"
analysis.LOG_DIR = analysis.OUT_DIR / "logs"
analysis.CODE_DIR = analysis.OUT_DIR / "code_snapshot"

analysis.STEP_WINDOWS = {
    "approach_to_center": {
        "UME": (0, 7),
        "CME": (0, 6),
        "label_cn": "靠近中心全过程",
    }
}

analysis.REGION_CONFIG = {
    "start_50": {
        "interval": (0.0, 0.50),
        "label_cn": "起始侧0-50%",
    }
}

analysis.GRID_SCALES_TO_RUN = [8, 12]


if __name__ == "__main__":
    analysis.main()
