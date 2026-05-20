"""Prediction and gate visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def save_probability_map(probability: np.ndarray, output_path: str | Path) -> None:
    """Save a probability map as an 8-bit grayscale PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(output_path)


def save_gate_heatmap(gate: torch.Tensor, output_path: str | Path) -> None:
    """Save the channel-averaged signed gate as a red-blue heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gate_map = gate.detach().float().mean(dim=0).cpu().numpy()
    vmax = max(float(np.abs(gate_map).max()), 1e-6)
    plt.figure(figsize=(5, 5), dpi=150)
    plt.imshow(gate_map, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_triplet_visualization(
    image_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    probability: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save input image, GT edge, and prediction side by side."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = denormalize_image(image_tensor).permute(1, 2, 0).cpu().numpy()
    target = target_tensor.squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), dpi=150)
    axes[0].imshow(np.clip(image, 0, 1))
    axes[0].set_title("Image")
    axes[1].imshow(target, cmap="gray")
    axes[1].set_title("GT")
    axes[2].imshow(probability, cmap="gray")
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1)
    return image_tensor * std + mean
