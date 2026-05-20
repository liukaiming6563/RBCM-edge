"""Inspect the local MEA data tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from rbcm_edge.cli import add_path_argument
from rbcm_edge.mea.paths import DEFAULT_EXPERIMENTS, MeaProjectPaths

DEFAULT_ARGS = {
    "data_root": Path("MEA_data"),
    "output_root": Path("outputs/mea"),
    "sample_rate_hz": 20000.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_argument(parser, "data-root", DEFAULT_ARGS["data_root"], "MEA data root")
    add_path_argument(parser, "output-root", DEFAULT_ARGS["output_root"], "Analysis output root")
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_ARGS["sample_rate_hz"])
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    paths = MeaProjectPaths(data_root=args.data_root, output_root=args.output_root)
    print(f"MEA data root: {paths.data_root.resolve()}")
    print(f"Output root:   {paths.output_root.resolve()}")
    print(f"Sample rate:   {args.sample_rate_hz:g} Hz")
    print()
    for run_id, paradigm in DEFAULT_EXPERIMENTS.items():
        run_root = paths.run_root(run_id)
        kilosort_dir = paths.kilosort_dir(run_id)
        stim_dir = paths.stimulus_dir(paradigm)
        required = [
            kilosort_dir / "spike_times.npy",
            kilosort_dir / "spike_clusters.npy",
            stim_dir / "events_tidy.csv",
            stim_dir / "bars_only.csv",
        ]
        status = "OK" if all(path.exists() for path in required) else "MISSING"
        print(f"{run_id}  {paradigm:12s}  {status}")
        print(f"  run_root:     {run_root}")
        print(f"  kilosort_dir: {kilosort_dir}")
        print(f"  stimulus_dir: {stim_dir}")
        for path in required:
            print(f"    [{'x' if path.exists() else ' '}] {path.name}")


if __name__ == "__main__":
    main(parse_args())
