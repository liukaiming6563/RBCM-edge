"""Build first-pass MEA response tables from Kilosort and stimulus timings."""

from __future__ import annotations

import argparse
from pathlib import Path

from rbcm_edge.cli import add_path_argument
from rbcm_edge.mea.align import build_response_table, response_matrix
from rbcm_edge.mea.event_loader import load_events
from rbcm_edge.mea.metrics import summarize_response_table
from rbcm_edge.mea.paths import DEFAULT_EXPERIMENTS, MeaProjectPaths
from rbcm_edge.mea.spike_loader import load_kilosort_run, select_units, spikes_by_unit
from rbcm_edge.utils.io import ensure_dir

DEFAULT_ARGS = {
    "data_root": Path("MEA_data"),
    "output_root": Path("outputs/mea"),
    "sample_rate_hz": 20000.0,
    "run_id": "all",
    "unit_labels": "good,mua",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_argument(parser, "data-root", DEFAULT_ARGS["data_root"], "MEA data root")
    add_path_argument(parser, "output-root", DEFAULT_ARGS["output_root"], "Analysis output root")
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_ARGS["sample_rate_hz"])
    parser.add_argument("--run-id", default=DEFAULT_ARGS["run_id"], help="Run id or 'all'.")
    parser.add_argument("--unit-labels", default=DEFAULT_ARGS["unit_labels"])
    parser.add_argument("--include-black-events", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    paths = MeaProjectPaths(data_root=args.data_root, output_root=args.output_root)
    labels = [item.strip() for item in args.unit_labels.split(",") if item.strip()]
    run_ids = list(DEFAULT_EXPERIMENTS) if args.run_id == "all" else [args.run_id]
    for run_id in run_ids:
        paradigm = DEFAULT_EXPERIMENTS[run_id]
        print(f"Processing {run_id} ({paradigm})")
        run = load_kilosort_run(paths.kilosort_dir(run_id), run_id, args.sample_rate_hz)
        unit_ids = select_units(run.cluster_table, allowed_labels=labels)
        if unit_ids.size == 0:
            print("  No labeled units found; using all units present in spike_clusters.npy")
            unit_ids = run.unit_ids
        grouped_spikes = spikes_by_unit(run, unit_ids)
        events = load_events(paths.stimulus_dir(paradigm), visible_only=not args.include_black_events)
        response = build_response_table(events, grouped_spikes, run_id=run_id, paradigm=paradigm)
        summary = summarize_response_table(response)
        matrix = response_matrix(response)
        out_dir = paths.per_run_output_dir(run_id, paradigm)
        ensure_dir(out_dir / "tables")
        response.to_csv(out_dir / "tables" / "response_table.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(out_dir / "tables" / "response_summary.csv", index=False, encoding="utf-8-sig")
        matrix.to_csv(out_dir / "tables" / "response_matrix.csv", encoding="utf-8-sig")
        print(f"  Saved {len(response):,} response rows to {out_dir / 'tables'}")


if __name__ == "__main__":
    main(parse_args())
