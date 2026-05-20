"""Run a normal Python sanity check for the edge dataloader."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_model.core.config import load_config
from edge_model.data.build import make_dataset, make_loader

DEFAULT_ARGS = {
    "config": PROJECT_ROOT / "edge_model" / "configs" / "local_3070ti.yaml",
}


def parse_args() -> argparse.Namespace:
    """Parse arguments with editable defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ARGS["config"])
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Load one training batch and print tensor shapes."""
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = load_config(config_path)
    config["paths"]["project_root"] = str(PROJECT_ROOT)
    dataset_cfg = config["dataset"]
    dataset = make_dataset(
        config,
        dataset_name=dataset_cfg["train_dataset"],
        split=dataset_cfg.get("train_split", "train"),
        training=True,
    )
    loader = make_loader(dataset, config, shuffle=True)
    batch = next(iter(loader))
    print("Dataloader check passed")
    print(f"  dataset size: {len(dataset)}")
    print(f"  image shape:  {tuple(batch['image'].shape)}")
    print(f"  edge shape:   {tuple(batch['edge'].shape)}")
    print(f"  sample ids:   {list(batch['sample_id'])[:4]}")


if __name__ == "__main__":
    main(parse_args())
