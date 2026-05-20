"""Generic edge detection dataset skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class EdgeSample:
    """One image and its edge label path."""

    image_path: Path
    edge_path: Path


class EdgeDataset:
    """A minimal PyTorch-compatible dataset for image-edge pairs."""

    def __init__(
        self,
        samples: list[EdgeSample],
        image_loader: Callable[[Path], object] | None = None,
        target_loader: Callable[[Path], object] | None = None,
        transform: Callable[[object, object], tuple[object, object]] | None = None,
    ) -> None:
        self.samples = samples
        self.image_loader = image_loader
        self.target_loader = target_loader
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, object]:
        sample = self.samples[index]
        if self.image_loader is None or self.target_loader is None:
            raise RuntimeError("Provide image_loader and target_loader before training.")
        image = self.image_loader(sample.image_path)
        target = self.target_loader(sample.edge_path)
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target
