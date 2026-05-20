"""Factories for datasets and data loaders."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from edge_model.data.edge_dataset import EdgeFolderDataset, collect_edge_pairs, split_pairs
from edge_model.data.transforms import make_eval_transform, make_train_transform


def make_dataset(config: dict, dataset_name: str, split: str, training: bool) -> EdgeFolderDataset:
    """Build an edge dataset from the configured Image_data folder."""
    project_root = Path(config.get("paths", {}).get("project_root", ".")).resolve()
    image_root = Path(config.get("paths", {}).get("image_data_root", "Image_data"))
    if not image_root.is_absolute():
        image_root = project_root / image_root

    dataset_root = image_root / dataset_name
    pairs = collect_edge_pairs(dataset_root, dataset_name)
    dataset_cfg = config.get("dataset", {})
    pairs = split_pairs(
        pairs,
        split=split,
        val_fraction=float(dataset_cfg.get("val_fraction", 0.15)),
        seed=int(config.get("seed", 42)),
    )

    if training:
        transform = make_train_transform(
            input_size=int(dataset_cfg.get("input_size", 384)),
            random_crop=bool(dataset_cfg.get("random_crop", True)),
            horizontal_flip=bool(dataset_cfg.get("horizontal_flip", True)),
            vertical_flip=bool(dataset_cfg.get("vertical_flip", False)),
        )
    else:
        transform = make_eval_transform(input_size=int(dataset_cfg.get("input_size", 384)))

    return EdgeFolderDataset(pairs, transform=transform)


def make_loader(dataset: EdgeFolderDataset, config: dict, shuffle: bool) -> DataLoader:
    """Create a PyTorch DataLoader with config-controlled defaults."""
    loader_cfg = config.get("loader", {})
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 4)),
        shuffle=shuffle,
        num_workers=int(loader_cfg.get("num_workers", 2)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=shuffle,
    )
