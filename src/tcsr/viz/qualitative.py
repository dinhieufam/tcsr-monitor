"""Qualitative visualization: correct / silent-failure / detected overlays."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tcsr.utils.logging import get_logger

log = get_logger(__name__)

_COLORS = {
    "correct": (0, 200, 0),      # green
    "silent_failure": (200, 0, 0),  # red
    "detected": (200, 200, 0),   # yellow
}


def overlay_mask(img: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.4) -> np.ndarray:
    vis = img.copy()
    colored = np.zeros_like(img)
    colored[mask > 0] = color
    return cv2.addWeighted(vis, 1 - alpha, colored, alpha, 0)


def make_qualitative_panel(
    examples: list[dict],
    output_path: Path,
) -> None:
    """
    examples: list of dicts with keys:
      frame_path, pred_mask_path, gt_mask_path, category (correct/silent_failure/detected)
    """
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, ex in zip(axes, examples):
        img = cv2.imread(ex["frame_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = cv2.imread(ex["pred_mask_path"], cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(ex["gt_mask_path"], cv2.IMREAD_GRAYSCALE)

        color = _COLORS.get(ex["category"], (128, 128, 128))
        vis = overlay_mask(img, pred > 0, color)
        if gt is not None:
            vis = cv2.addWeighted(vis, 0.85, cv2.cvtColor(
                cv2.applyColorMap((gt > 0).astype(np.uint8) * 255, cv2.COLORMAP_BONE), cv2.COLOR_BGR2RGB
            ) * 0, 0.15, 0)

        ax.imshow(vis)
        ax.set_title(ex["category"].replace("_", " ").title(), fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Qualitative panel saved: %s", output_path)
