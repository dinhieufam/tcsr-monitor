"""Severity-graded image corruptions for robustness evaluation (ImageNet-C style)."""

from __future__ import annotations

import io

import cv2
import numpy as np

CORRUPTION_TYPES = [
    "gaussian_blur",
    "motion_blur",
    "gaussian_noise",
    "brightness",
    "jpeg_compression",
    "contrast",
]
SEVERITIES = [1, 2, 3, 4, 5]


def apply_corruption(img: np.ndarray, corruption: str, severity: int) -> np.ndarray:
    """
    Apply a named corruption at a given severity (1=mild, 5=severe).

    img: BGR uint8 (H, W, 3).
    Returns a corrupted BGR uint8 array of the same shape.
    """
    if severity not in range(1, 6):
        raise ValueError(f"severity must be 1-5, got {severity}")

    s = severity - 1  # 0-based index into severity tables

    if corruption == "gaussian_blur":
        k = [3, 5, 7, 9, 11][s]
        return cv2.GaussianBlur(img, (k, k), sigmaX=(s + 1) * 0.5)

    elif corruption == "motion_blur":
        k = [3, 5, 7, 9, 15][s]
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        return cv2.filter2D(img, -1, kernel)

    elif corruption == "gaussian_noise":
        sigma = [5, 10, 20, 35, 50][s]
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(0, sigma, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    elif corruption == "brightness":
        # Simulate low-light / overexposure by scaling pixel values
        factor = [0.7, 0.5, 0.35, 0.2, 0.1][s]
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    elif corruption == "jpeg_compression":
        quality = [75, 50, 30, 15, 5][s]
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    elif corruption == "contrast":
        # Reduce contrast toward gray mean
        factor = [0.6, 0.4, 0.25, 0.15, 0.05][s]
        gray_mean = float(img.mean())
        return np.clip(
            factor * img.astype(np.float32) + (1.0 - factor) * gray_mean, 0, 255
        ).astype(np.uint8)

    else:
        raise ValueError(f"Unknown corruption type: {corruption!r}")
