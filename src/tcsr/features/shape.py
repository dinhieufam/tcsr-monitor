"""Shape features: area, components, boundary, eccentricity, solidity."""

from __future__ import annotations

import numpy as np
from skimage import measure


def compute_shape_features(binary_mask: np.ndarray) -> dict[str, float]:
    """
    binary_mask: (H, W) uint8 with values 0 or 1.
    Returns dict of scalar shape features.
    """
    total_pixels = binary_mask.size
    area_frac = float(binary_mask.sum()) / total_pixels

    labeled = measure.label(binary_mask, connectivity=2)
    regions = measure.regionprops(labeled)
    n_components = len(regions)

    if not regions:
        return {
            "shape_area_frac": area_frac,
            "shape_n_components": 0.0,
            "shape_boundary_len": 0.0,
            "shape_max_eccentricity": 0.0,
            "shape_max_solidity": 0.0,
            "shape_perimeter": 0.0,
        }

    largest = max(regions, key=lambda r: r.area)
    perimeter = sum(r.perimeter for r in regions)

    return {
        "shape_area_frac": area_frac,
        "shape_n_components": float(n_components),
        "shape_boundary_len": float(perimeter) / total_pixels,
        "shape_max_eccentricity": float(largest.eccentricity),
        "shape_max_solidity": float(largest.solidity),
        "shape_perimeter": float(perimeter),
    }
