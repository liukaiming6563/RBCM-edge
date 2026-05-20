"""Dataset implementation for local edge detection folders.

Each dataset is expected to contain:

```text
DatasetName/
  image/
  edge/
  overlapping/
```

Only `image/` and `edge/` are used for training. `overlapping/` is left for
manual inspection and paper visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class EdgePair:
    """One input image and its ground-truth edge map."""

    image_path: Path
    edge_path: Path
    dataset_name: str
    sample_id: str


def collect_edge_pairs(dataset_root: str | Path, dataset_name: str) -> list[EdgePair]:
    """Collect image-edge pairs by matching file stems.

    Args:
        dataset_root: Folder containing `image/` and `edge/`.
        dataset_name: Human-readable dataset name written into each sample.

    Returns:
        Sorted list of matched image-edge pairs.
    """
    dataset_root = Path(dataset_root)
    image_dir = dataset_root / "image"
    edge_dir = dataset_root / "edge"
    if not image_dir.exists() or not edge_dir.exists():
        raise FileNotFoundError(f"Expected image/ and edge/ under {dataset_root}")

    images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    edges = [p for p in edge_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    edge_by_stem = {p.stem: p for p in edges}

    pairs: list[EdgePair] = []
    missing_edges: list[str] = []
    for image_path in sorted(images, key=lambda p: p.stem):
        edge_path = edge_by_stem.get(image_path.stem)
        if edge_path is None:
            missing_edges.append(image_path.name)
            continue
        pairs.append(
            EdgePair(
                image_path=image_path,
                edge_path=edge_path,
                dataset_name=dataset_name,
                sample_id=image_path.stem,
            )
        )

    if missing_edges:
        preview = ", ".join(missing_edges[:5])
        raise ValueError(f"{dataset_name} has images without edge maps: {preview}")
    return pairs


def split_pairs(
    pairs: list[EdgePair],
    split: str,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> list[EdgePair]:
    """Return a deterministic train/val/all split.

    This initial project scaffold does not assume official train/test text files.
    It uses deterministic random splits for local demos and supports `all` for
    cross-dataset testing. Official split support can be added later if needed.
    """
    if split == "all":
        return pairs
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")

    import random

    shuffled = pairs.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_items = shuffled[:val_count]
    train_items = shuffled[val_count:]
    return train_items if split == "train" else val_items


class EdgeFolderDataset(Dataset):
    """PyTorch dataset for RGB images and single-channel edge maps."""

    def __init__(
        self,
        pairs: list[EdgePair],
        transform: Callable | None = None,
    ) -> None:
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        pair = self.pairs[index]
        image = Image.open(pair.image_path).convert("RGB")
        edge = Image.open(pair.edge_path).convert("L")
        if self.transform is not None:
            image_tensor, edge_tensor = self.transform(image, edge)
        else:
            raise RuntimeError("A transform must be provided to convert PIL images to tensors.")

        return {
            "image": image_tensor,
            "edge": edge_tensor,
            "dataset": pair.dataset_name,
            "sample_id": pair.sample_id,
            "image_path": str(pair.image_path),
            "edge_path": str(pair.edge_path),
        }
