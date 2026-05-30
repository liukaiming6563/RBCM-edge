"""Resume the focused MEA trajectory selector from its checkpoint.

This helper is intentionally small. The original focused selector completed a
checkpoint table but stopped before writing the final summary. This script reads
that checkpoint, computes only missing parameter combinations, then reuses the
original selector's summary/figure/report functions.
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"D:\study\project\RBCM-Edge")
ANALYSIS_DIR = PROJECT_ROOT / "MEA_analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

import run_MEA_trajectory_focused_selector as selector  # noqa: E402


KEY_COLUMNS = [
    "response_variant",
    "window_name",
    "region_name",
    "mirror_name",
    "grid_n",
    "pair_id",
    "direction_code",
]


def build_all_jobs() -> list[dict]:
    """Build the full job list in the same nested-loop order as the main script."""

    jobs = []
    for response_variant in selector.RESPONSE_VARIANTS:
        for window_name in selector.WINDOWS:
            for region_name in selector.REGIONS:
                for mirror_name in selector.MIRRORS:
                    for grid_n in selector.GRID_SCALES:
                        for pair_id in selector.PAIR_IDS:
                            for direction in selector.DIRECTIONS:
                                jobs.append(
                                    {
                                        "response_variant": response_variant,
                                        "window_name": window_name,
                                        "region_name": region_name,
                                        "mirror_name": mirror_name,
                                        "grid_n": int(grid_n),
                                        "pair_id": pair_id,
                                        "direction_code": direction["code"],
                                        "direction": direction,
                                    }
                                )
    return jobs


def main() -> None:
    selector.ensure_dirs()
    checkpoint_path = selector.TABLE_DIR / "focused_selector_tests_checkpoint.csv"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint table: {checkpoint_path}")

    existing = pd.read_csv(checkpoint_path)
    existing["grid_n"] = existing["grid_n"].astype(int)
    done_keys = set(map(tuple, existing[KEY_COLUMNS].itertuples(index=False, name=None)))

    all_jobs = build_all_jobs()
    missing_jobs = []
    for job in all_jobs:
        key = (
            job["response_variant"],
            job["window_name"],
            job["region_name"],
            job["mirror_name"],
            int(job["grid_n"]),
            job["pair_id"],
            job["direction_code"],
        )
        if key not in done_keys:
            missing_jobs.append(job)

    selector.log(f"Checkpoint rows: {len(existing)}")
    selector.log(f"Total expected jobs: {len(all_jobs)}")
    selector.log(f"Missing jobs to compute: {len(missing_jobs)}")

    new_rows = []
    for idx, job in enumerate(missing_jobs, start=1):
        if idx == 1 or idx % 50 == 0 or idx == len(missing_jobs):
            selector.log(
                f"[resume {idx}/{len(missing_jobs)}] {job['response_variant']}, "
                f"{job['window_name']}, {job['region_name']}, {job['mirror_name']}, "
                f"{job['grid_n']}x{job['grid_n']}, {job['pair_id']}, {job['direction_code']}"
            )
        new_rows.append(
            selector.compute_one_test(
                response_variant=job["response_variant"],
                window_name=job["window_name"],
                region_name=job["region_name"],
                mirror_name=job["mirror_name"],
                grid_n=int(job["grid_n"]),
                pair_id=job["pair_id"],
                direction=job["direction"],
            )
        )

    if new_rows:
        combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        combined = existing.copy()

    combined = combined.drop_duplicates(KEY_COLUMNS, keep="last")
    combined = combined.sort_values(KEY_COLUMNS).reset_index(drop=True)
    if len(combined) != len(all_jobs):
        raise RuntimeError(f"Combined rows {len(combined)} != expected rows {len(all_jobs)}")

    summary = selector.summarize(combined)
    test_path = selector.TABLE_DIR / "focused_selector_tests.csv"
    summary_path = selector.TABLE_DIR / "focused_selector_combo_summary.csv"
    top_path = selector.TABLE_DIR / "focused_selector_top30.csv"
    combined.to_csv(test_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary.head(30).to_csv(top_path, index=False, encoding="utf-8-sig")

    selector.make_figures(summary)
    selector.write_report(summary)
    shutil.copy2(Path(__file__), selector.CODE_DIR / Path(__file__).name)
    (selector.LOG_DIR / "focused_selector_resume_log.txt").write_text(
        "\n".join(selector.LOG_LINES), encoding="utf-8"
    )
    selector.log("Resume completed and final tables/figures/reports were written.")


if __name__ == "__main__":
    main()
