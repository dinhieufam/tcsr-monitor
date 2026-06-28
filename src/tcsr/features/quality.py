"""Image quality features: brightness, blur, specular ratio, contrast."""

from __future__ import annotations

import cv2
import numpy as np


def compute_quality_features(bgr_img: np.ndarray) -> dict[str, float]:
    """
    bgr_img: (H, W, 3) uint8 BGR image.
    Returns dict of scalar quality features.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    brightness = float(gray.mean()) / 255.0
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())  # larger → sharper
    contrast = float(gray.std()) / 255.0

    # Specular highlight: very bright pixels (> 240 in any channel)
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    specular_ratio = float((hsv[:, :, 2] > 240).mean())

    return {
        "qual_brightness": brightness,
        "qual_blur_score": lap_var,
        "qual_contrast": contrast,
        "qual_specular_ratio": specular_ratio,
    }
