"""Train an RBCM edge detection model.

PyCharm usage:
1. Edit `DEFAULT_ARGS` below if needed.
2. Right-click this file and choose Run.

Command-line usage:
```powershell
python edge_model\\train.py --config edge_model\\configs\\local_3070ti.yaml --epochs 2
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_model.core.config import deep_update, load_config, project_path
from edge_model.core.paths import make_run_paths
from edge_model.core.seed import seed_everything
from edge_model.data.build import make_dataset, make_loader
from edge_model.engine.train_loop import append_metrics_csv, evaluate, train_one_epoch
from edge_model.models.build import build_model
from edge_model.tools.analyze_training_log import analyze_training_log
from rbcm_edge.models.losses import EdgeDetectionLoss

DEFAULT_ARGS = {
    "config": PROJECT_ROOT / "edge_model" / "configs" / "local_3070ti.yaml",
    "experiment_name": None,
    "train_dataset": None,
    "val_dataset": None,
    "epochs": None,
    "batch_size": None,
    "input_size": None,
    "learning_rate": None,
    "device": None,
}


def parse_args() -> argparse.Namespace:
    """Parse training arguments with editable defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ARGS["config"])
    parser.add_argument("--experiment-name", default=DEFAULT_ARGS["experiment_name"])
    parser.add_argument("--train-dataset", default=DEFAULT_ARGS["train_dataset"])
    parser.add_argument("--val-dataset", default=DEFAULT_ARGS["val_dataset"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_ARGS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_ARGS["batch_size"])
    parser.add_argument("--input-size", type=int, default=DEFAULT_ARGS["input_size"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_ARGS["learning_rate"])
    parser.add_argument("--device", default=DEFAULT_ARGS["device"])
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply command-line overrides to the YAML config."""
    updates = {
        "experiment_name": args.experiment_name,
        "device": args.device,
        "dataset": {
            "train_dataset": args.train_dataset,
            "val_dataset": args.val_dataset,
            "input_size": args.input_size,
        },
        "loader": {"batch_size": args.batch_size},
        "train": {"epochs": args.epochs, "learning_rate": args.learning_rate},
    }
    return deep_update(config, updates)


def main(args: argparse.Namespace) -> None:
    """Run the complete training workflow."""
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = apply_overrides(load_config(config_path), args)
    config["paths"]["project_root"] = str(PROJECT_ROOT)
    seed_everything(int(config.get("seed", 42)))

    device_name = config.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    output_root = project_path(config, config["paths"].get("output_root", "outputs/edge_detection"))
    run_paths = make_run_paths(output_root, config.get("experiment_name", "edge_experiment"))

    dataset_cfg = config["dataset"]
    train_dataset = make_dataset(
        config,
        dataset_name=dataset_cfg["train_dataset"],
        split=dataset_cfg.get("train_split", "train"),
        training=True,
    )
    val_dataset = make_dataset(
        config,
        dataset_name=dataset_cfg["val_dataset"],
        split=dataset_cfg.get("val_split", "all"),
        training=False,
    )
    train_loader = make_loader(train_dataset, config, shuffle=True)
    val_loader = make_loader(val_dataset, config, shuffle=False)

    model = build_model(config).to(device)
    loss_cfg = config.get("loss", {})
    criterion = EdgeDetectionLoss(
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        local_weight=float(loss_cfg.get("local_weight", 0.3)),
        gate_sparsity_weight=float(loss_cfg.get("gate_sparsity_weight", 1e-4)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("learning_rate", 1e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    use_amp = bool(config["train"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    best_ods = -1.0
    epochs = int(config["train"].get("epochs", 20))
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
            log_interval=int(config["train"].get("log_interval", 10)),
        )

        row = {"epoch": epoch, "split": "train", **train_metrics}
        append_metrics_csv(run_paths.logs / "train_log.csv", row)
        print(row)

        if epoch % int(config["train"].get("eval_interval", 1)) == 0:
            save_visuals = epoch % int(config["train"].get("save_visual_interval", 1)) == 0
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                visual_dir=run_paths.visualizations / f"epoch_{epoch:03d}" if save_visuals else None,
                pred_dir=None,
                gate_dir=run_paths.gate_heatmaps / f"epoch_{epoch:03d}" if save_visuals else None,
                max_visual_samples=int(config["train"].get("max_visual_samples", 8)),
            )
            val_row = {"epoch": epoch, "split": "val", **val_metrics}
            append_metrics_csv(run_paths.logs / "train_log.csv", val_row)
            print(val_row)

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "metrics": val_metrics,
            }
            torch.save(checkpoint, run_paths.checkpoints / "last.pt")
            if val_metrics.get("ODS", 0.0) > best_ods:
                best_ods = val_metrics["ODS"]
                torch.save(checkpoint, run_paths.checkpoints / "best.pt")
                print(f"Saved best checkpoint with ODS={best_ods:.4f}")

    if bool(config["train"].get("auto_plot_log", True)):
        log_csv = run_paths.logs / "train_log.csv"
        plot_dir = run_paths.root / "plots"
        print(f"Generating training trend plots from {log_csv}")
        analyze_training_log(log_csv=log_csv, output_dir=plot_dir)


if __name__ == "__main__":
    main(parse_args())
