"""Image and edge-map transforms for local training.

The transforms avoid torchvision so the project has fewer moving parts. Images
are converted to float tensors in `[0, 1]`, then normalized with ImageNet mean
and standard deviation. Edge maps become single-channel float tensors in `[0, 1]`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class EdgeTransform:
    """Joint transform for an RGB image and its edge map."""

    input_size: int = 384
    random_crop: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = False

    def __call__(self, image: Image.Image, edge: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply resize/crop/flip and convert to tensors."""
        image, edge = self._resize_or_crop(image, edge)

        if self.horizontal_flip and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            edge = edge.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.vertical_flip and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            edge = edge.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        image_tensor = self._image_to_tensor(image)
        edge_tensor = self._edge_to_tensor(edge)
        return image_tensor, edge_tensor

    def _resize_or_crop(self, image: Image.Image, edge: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Resize validation images and random-crop training images."""
        size = int(self.input_size)
        if self.random_crop:
            image, edge = self._resize_short_side(image, edge, size)
            width, height = image.size
            if width == size and height == size:
                return image, edge
            left = random.randint(0, max(0, width - size))
            top = random.randint(0, max(0, height - size))
            box = (left, top, left + size, top + size)
            return image.crop(box), edge.crop(box)

        return (
            image.resize((size, size), Image.Resampling.BILINEAR),
            edge.resize((size, size), Image.Resampling.NEAREST),
        )

    @staticmethod
    def _resize_short_side(
        image: Image.Image,
        edge: Image.Image,
        size: int,
    ) -> tuple[Image.Image, Image.Image]:
        """Resize so both dimensions are at least `size`, preserving aspect ratio."""
        width, height = image.size
        scale = max(size / width, size / height)
        new_size = (int(round(width * scale)), int(round(height * scale)))
        return (
            image.resize(new_size, Image.Resampling.BILINEAR),
            edge.resize(new_size, Image.Resampling.NEAREST),
        )

    @staticmethod
    def _image_to_tensor(image: Image.Image) -> torch.Tensor:
        """Convert PIL RGB image to normalized CHW tensor."""
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

    @staticmethod
    def _edge_to_tensor(edge: Image.Image) -> torch.Tensor:
        """Convert PIL grayscale edge map to `[1, H, W]` float tensor."""
        array = np.asarray(edge, dtype=np.float32) / 255.0
        array = (array > 0.5).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0).contiguous()


def make_train_transform(input_size: int, random_crop: bool, horizontal_flip: bool, vertical_flip: bool) -> EdgeTransform:
    """Create the transform used for training."""
    return EdgeTransform(
        input_size=input_size,
        random_crop=random_crop,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
    )


def make_eval_transform(input_size: int) -> EdgeTransform:
    """Create deterministic transform used for validation and testing."""
    return EdgeTransform(
        input_size=input_size,
        random_crop=False,
        horizontal_flip=False,
        vertical_flip=False,
    )
