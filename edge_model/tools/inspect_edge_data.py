"""Inspect local edge detection datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

DEFAULT_ARGS = {
    "image_data_root": Path("Image_data"),
}


def parse_args() -> argparse.Namespace:
    """Parse arguments with editable defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-data-root", type=Path, default=DEFAULT_ARGS["image_data_root"])
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Print counts, formats, and sample dimensions for each dataset."""
    root = args.image_data_root
    for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        image_dir = dataset_dir / "image"
        edge_dir = dataset_dir / "edge"
        if not image_dir.exists() or not edge_dir.exists():
            continue
        images = sorted([p for p in image_dir.iterdir() if p.is_file()])
        edges = sorted([p for p in edge_dir.iterdir() if p.is_file()])
        image_stems = {p.stem for p in images}
        edge_stems = {p.stem for p in edges}
        print(f"\n{dataset_dir.name}")
        print(f"  images: {len(images)}")
        print(f"  edges:  {len(edges)}")
        print(f"  stems match: {image_stems == edge_stems}")
        if images:
            with Image.open(images[0]) as image:
                print(f"  sample image: {images[0].name}, size={image.size}, mode={image.mode}")
        if edges:
            with Image.open(edges[0]) as edge:
                print(f"  sample edge:  {edges[0].name}, size={edge.size}, mode={edge.mode}")


if __name__ == "__main__":
    main(parse_args())
