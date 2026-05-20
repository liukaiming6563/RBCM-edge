"""Evaluate a trained RBCM edge detection checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_model.core.config import deep_update, load_config, project_path
from edge_model.core.paths import make_run_paths
from edge_model.data.build import make_dataset, make_loader
from edge_model.engine.train_loop import append_metrics_csv, evaluate
from edge_model.models.build import build_model
from rbcm_edge.models.losses import EdgeDetectionLoss

DEFAULT_ARGS = {
    "config": PROJECT_ROOT / "edge_model" / "configs" / "eval_cross_dataset.yaml",
    "checkpoint": None,
    "eval_dataset": None,
    "batch_size": None,
    "input_size": None,
    "device": None,
}


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments with editable defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ARGS["config"])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARGS["checkpoint"])
    parser.add_argument("--eval-dataset", default=DEFAULT_ARGS["eval_dataset"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_ARGS["batch_size"])
    parser.add_argument("--input-size", type=int, default=DEFAULT_ARGS["input_size"])
    parser.add_argument("--device", default=DEFAULT_ARGS["device"])
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run checkpoint evaluation and optional prediction saving."""
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = load_config(config_path)
    config["paths"]["project_root"] = str(PROJECT_ROOT)
    config = deep_update(
        config,
        {
            "device": args.device,
            "dataset": {"eval_dataset": args.eval_dataset, "input_size": args.input_size},
            "loader": {"batch_size": args.batch_size},
            "eval": {"checkpoint": str(args.checkpoint) if args.checkpoint else None},
        },
    )

    device_name = config.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    checkpoint_path = project_path(config, config["eval"]["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", config)
    checkpoint_config = deep_update(checkpoint_config, {"paths": config["paths"], "dataset": config["dataset"]})

    model = build_model(checkpoint_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    dataset = make_dataset(
        config,
        dataset_name=config["dataset"]["eval_dataset"],
        split=config["dataset"].get("eval_split", "all"),
        training=False,
    )
    loader = make_loader(dataset, config, shuffle=False)

    output_root = project_path(config, config["paths"].get("output_root", "outputs/edge_detection"))
    run_paths = make_run_paths(output_root, config.get("experiment_name", "eval"))
    criterion = EdgeDetectionLoss()

    metrics = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        visual_dir=run_paths.visualizations if config.get("eval", {}).get("save_visualizations", True) else None,
        pred_dir=run_paths.predictions if config.get("eval", {}).get("save_predictions", True) else None,
        gate_dir=run_paths.gate_heatmaps if config.get("eval", {}).get("save_predictions", True) else None,
        max_visual_samples=int(config.get("eval", {}).get("max_visual_samples", 32)),
    )
    row = {"checkpoint": str(checkpoint_path), "dataset": config["dataset"]["eval_dataset"], **metrics}
    append_metrics_csv(run_paths.metrics / "eval_metrics.csv", row)
    print(row)


if __name__ == "__main__":
    main(parse_args())
