"""Run inference on a folder of images and save prediction maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_model.core.config import load_config, project_path
from edge_model.data.transforms import make_eval_transform
from edge_model.engine.visualize import save_gate_heatmap, save_probability_map
from edge_model.models.build import build_model

DEFAULT_ARGS = {
    "config": PROJECT_ROOT / "edge_model" / "configs" / "eval_cross_dataset.yaml",
    "checkpoint": None,
    "image_dir": Path("Image_data/BSDS500/image"),
    "output_dir": Path("outputs/edge_detection/inference"),
    "input_size": 512,
    "device": "cuda",
}


def parse_args() -> argparse.Namespace:
    """Parse inference arguments with editable defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ARGS["config"])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARGS["checkpoint"])
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_ARGS["image_dir"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARGS["output_dir"])
    parser.add_argument("--input-size", type=int, default=DEFAULT_ARGS["input_size"])
    parser.add_argument("--device", default=DEFAULT_ARGS["device"])
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run inference and save probability maps plus signed gate heatmaps."""
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = load_config(config_path)
    config["paths"]["project_root"] = str(PROJECT_ROOT)
    if args.checkpoint is None:
        raise SystemExit("Please provide --checkpoint or edit DEFAULT_ARGS['checkpoint'].")

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    checkpoint = torch.load(project_path(config, args.checkpoint), map_location=device)
    model_config = checkpoint.get("config", config)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    image_dir = project_path(config, args.image_dir)
    output_dir = project_path(config, args.output_dir)
    pred_dir = output_dir / "predictions"
    gate_dir = output_dir / "gate_heatmaps"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gate_dir.mkdir(parents=True, exist_ok=True)

    transform = make_eval_transform(input_size=args.input_size)
    image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    with torch.no_grad():
        for image_path in sorted(image_paths):
            image = Image.open(image_path).convert("RGB")
            dummy_edge = Image.new("L", image.size, 0)
            image_tensor, _ = transform(image, dummy_edge)
            outputs = model(image_tensor.unsqueeze(0).to(device))
            probability = torch.sigmoid(outputs["logits"])[0, 0].cpu().numpy()
            save_probability_map(probability, pred_dir / f"{image_path.stem}.png")
            save_gate_heatmap(outputs["gate"][0], gate_dir / f"{image_path.stem}.png")
            print(f"Saved {image_path.stem}")


if __name__ == "__main__":
    main(parse_args())
