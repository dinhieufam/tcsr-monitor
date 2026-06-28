"""Deterministic resize and normalize transforms shared across datasets."""

from __future__ import annotations

import cv2
import numpy as np


class ResizeNormalize:
    """Resize a BGR frame to target_size and normalize to ImageNet stats."""

    def __init__(
        self,
        target_size: tuple[int, int] = (512, 512),
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.target_size = target_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Return CHW float32 in [roughly -2, 2]."""
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return img.transpose(2, 0, 1)  # HWC → CHW


class ResizeMask:
    """Resize a binary mask to target_size (nearest-neighbour)."""

    def __init__(self, target_size: tuple[int, int] = (512, 512)):
        self.target_size = target_size

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        return cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
