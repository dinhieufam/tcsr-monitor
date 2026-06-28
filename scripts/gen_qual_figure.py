"""Generate the τ=0.75 qualitative failure panel for the ICARCV paper.

Produces results/figures/qual_failures_tau75.png — a 2×3 grid showing
raw endoscope frame | predicted-mask overlay | GT-mask overlay for each
of 6 selected confident failures at τ=0.75 (IoU ∈ [0.5, 0.75)).
"""
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR = ROOT / "data" / "processed" / "endovis2017" / "predictions"

# ── colours ──────────────────────────────────────────────────────────────────
PRED_COLOR  = np.array([255,  60,  60], dtype=np.uint8)   # red: predicted mask fill
GT_COLOR    = np.array([ 50, 220,  50], dtype=np.uint8)   # green: GT mask fill
FP_COLOR    = np.array([255, 165,   0], dtype=np.uint8)   # orange: false-positive pixels
ALPHA_FILL  = 0.52
ALPHA_FP    = 0.65


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_pred_mask(frame_id: str) -> np.ndarray:
    npz = np.load(PRED_DIR / frame_id / "probs.npz")
    probs = npz["probs"].astype(np.float32)
    return (probs >= 0.5).astype(np.uint8)


def load_gt_mask(mask_path: str, target_hw=(512, 512)) -> np.ndarray:
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (gt > 0).astype(np.uint8)


def add_contour(img: np.ndarray, mask: np.ndarray, color_bgr: tuple, thickness: int = 2) -> np.ndarray:
    """Draw mask boundary as a bright contour on the image (operates in-place copy)."""
    out = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Convert RGB image to BGR for cv2, draw, convert back
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.drawContours(out_bgr, contours, -1, color_bgr, thickness)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def overlay(frame_rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    out = frame_rgb.copy().astype(np.float32)
    mask3 = mask[:, :, None].astype(np.float32)
    out = out * (1 - alpha * mask3) + color.astype(np.float32) * (alpha * mask3)
    return np.clip(out, 0, 255).astype(np.uint8)


def pred_overlay_with_fp(frame_rgb, pred_mask, gt_mask):
    """Red fill for TP region, orange fill for FP region, red contour boundary."""
    tp = (pred_mask & gt_mask).astype(np.uint8)
    fp = (pred_mask & ~gt_mask).astype(np.uint8)
    out = overlay(frame_rgb, tp, PRED_COLOR, ALPHA_FILL)
    out = overlay(out,       fp, FP_COLOR,  ALPHA_FP)
    out = add_contour(out, pred_mask, (60, 20, 20), thickness=2)
    return out


def gt_overlay_clean(frame_rgb, gt_mask):
    out = overlay(frame_rgb, gt_mask, GT_COLOR, ALPHA_FILL)
    out = add_contour(out, gt_mask, (10, 80, 10), thickness=2)
    return out


def main():
    labels   = pd.read_parquet(ROOT / "data/processed/endovis2017/labels.parquet")
    manifest = pd.read_parquet(ROOT / "data/processed/endovis2017/manifest.parquet")
    test = manifest[manifest.split == "test"].merge(labels, on="frame_id", how="inner")

    tau75_fail = test[(test.iou >= 0.5) & (test.iou < 0.75)].copy().sort_values("iou").reset_index(drop=True)

    # Hand-picked for maximum visual diversity: alternate seq08/seq09 within each stack
    target_ids = [
        "endovis2017_seq08_00077",   # stack-A row-0: dark scene,  IoU=0.51
        "endovis2017_seq09_00267",   # stack-A row-1: pink scene,  IoU=0.54
        "endovis2017_seq08_00119",   # stack-B row-0: bright scene, IoU=0.54
        "endovis2017_seq09_00074",   # stack-B row-1: light scene, IoU=0.66
    ]
    selected = tau75_fail[tau75_fail.frame_id.isin(target_ids)].set_index("frame_id").loc[target_ids].reset_index()

    # ── layout: 2 stacks of 3×3 side by side ────────────────────────────────
    # Left stack: examples 0-2  |  Right stack: examples 3-5
    # Each stack: 3 rows × 3 cols (raw | pred | GT)
    # Total grid: 3 rows × 6 cols, with a small gap column between stacks

    CELL_W, CELL_H = 2.8, 2.6
    fig_w = 6 * CELL_W + 1.2   # extra for row labels
    fig_h = 2 * CELL_H + 0.6   # extra for col headers (2 rows now)

    fig = plt.figure(figsize=(fig_w, fig_h))

    import matplotlib.gridspec as gridspec

    # GridSpec: 2 rows × 7 cols — col 3 is a thin spacer
    col_widths = [1, 1, 1, 0.08, 1, 1, 1]   # stack-A | gap | stack-B
    gs = gridspec.GridSpec(2, 7, figure=fig,
                           width_ratios=col_widths,
                           hspace=0.06, wspace=0.04)

    def ax_idx(stack, col_in_stack):
        return col_in_stack if stack == 0 else col_in_stack + 4

    col_titles = ["Input", "Predicted", "Ground-truth"]

    axes_grid = {}
    for row in range(2):
        for stack in range(2):
            for c in range(3):
                ax = fig.add_subplot(gs[row, ax_idx(stack, c)])
                axes_grid[(stack, row, c)] = ax
                ax.axis("off")

    # Column headers (top row only) — large font
    for stack in range(2):
        for c, lbl in enumerate(col_titles):
            ax = axes_grid[(stack, 0, c)]
            ax.set_title(lbl, fontsize=16, fontweight="bold", pad=6)

    for idx, (_, row_data) in enumerate(selected.iterrows()):
        stack   = idx // 2
        row_pos = idx % 2

        frame_rgb = load_rgb(row_data.frame_path)
        pred_mask = load_pred_mask(row_data.frame_id)
        gt_mask   = load_gt_mask(row_data.mask_path)
        frame_512 = cv2.resize(frame_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)

        pred_vis = pred_overlay_with_fp(frame_512, pred_mask, gt_mask)
        gt_vis   = gt_overlay_clean(frame_512, gt_mask)

        iou_val = row_data.iou
        npz     = np.load(PRED_DIR / row_data.frame_id / "probs.npz")
        probs   = npz["probs"].astype(np.float32)
        fg_conf = float(probs[pred_mask.astype(bool)].mean()) if pred_mask.sum() > 0 else 0.0

        for c, img in enumerate([frame_512, pred_vis, gt_vis]):
            ax = axes_grid[(stack, row_pos, c)]
            ax.imshow(img)
            if c == 0:
                ax.set_ylabel(
                    f"IoU={iou_val:.2f}  conf={fg_conf:.2f}",
                    fontsize=13, labelpad=6, rotation=0, ha="right", va="center",
                    fontfamily="monospace",
                )


    out_path = OUT_DIR / "qual_failures_tau75.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
