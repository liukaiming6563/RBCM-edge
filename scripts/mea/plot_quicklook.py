"""Create quicklook MEA plots from generated response summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rbcm_edge.cli import add_path_argument
from rbcm_edge.mea.paths import DEFAULT_EXPERIMENTS, MeaProjectPaths
from rbcm_edge.mea.visualization import plot_step_response_summary

DEFAULT_ARGS = {
    "output_root": Path("outputs/mea"),
    "run_id": "all",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_argument(parser, "output-root", DEFAULT_ARGS["output_root"], "Analysis output root")
    parser.add_argument("--run-id", default=DEFAULT_ARGS["run_id"], help="Run id or 'all'.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    paths = MeaProjectPaths(data_root=Path("MEA_data"), output_root=args.output_root)
    run_ids = list(DEFAULT_EXPERIMENTS) if args.run_id == "all" else [args.run_id]
    for run_id in run_ids:
        paradigm = DEFAULT_EXPERIMENTS[run_id]
        out_dir = paths.per_run_output_dir(run_id, paradigm)
        summary_path = out_dir / "tables" / "response_summary.csv"
        if not summary_path.exists():
            print(f"Skipping {run_id}: missing {summary_path}")
            continue
        summary = pd.read_csv(summary_path, encoding="utf-8-sig")
        fig_path = out_dir / "figures" / "step_response_summary.png"
        plot_step_response_summary(summary, fig_path)
        print(f"Saved {fig_path}")


if __name__ == "__main__":
    main(parse_args())
