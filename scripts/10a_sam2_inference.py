#!/usr/bin/env python
"""
Stage 10a: SAM2-Tiny zero-shot inference on EndoVis 2017 (all splits).

Protocol: SAM2 with GT bounding-box prompts (oracle-prompt, no fine-tuning).
This is the standard SAM evaluation protocol — the model is truly zero-shot
(no EndoVis training whatsoever); the GT bbox gives SAM2 the best possible
hint so that failures represent genuinely hard cases, not misprompting.

Output per frame (same layout as U-Net predictions, so all downstream
feature/label scripts work unchanged):
  data/processed/endovis2017/sam2_predictions/{frame_id}/probs.npz   (float16, 512x512)
  data/processed/endovis2017/sam2_predictions/{frame_id}/mask.png    (uint8 0/255)

Run from: tcsr-monitor/ with conda env tcsr-monitor active.
  CUDA_VISIBLE_DEVICES=3 python scripts/10a_sam2_inference.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.utils.io import save_npz
from tcsr.utils.logging import get_logger

log = get_logger(__name__)

PROCESSED_DIR = Path("data/processed/endovis2017")
MANIFEST_PATH = PROCESSED_DIR / "manifest.parquet"
SAM2_PRED_DIR = PROCESSED_DIR / "sam2_predictions"

SAM2_CHECKPOINT = Path(
    "/mnt/disk1/backup_user/hieu.ph/LISA/__Minh_Dang__/AutoNNUnet/sam2_hiera_tiny.pt"
)
SAM2_CONFIG = "configs/sam2/sam2_hiera_t.yaml"  # relative, resolved by sam2 package
IMG_SIZE = (512, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_gt_bbox(mask_path: str, pad: int = 5) -> tuple | None:
    """Return (x1,y1,x2,y2) of GT instrument pixels, padded. None if empty mask."""
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    gt = cv2.resize(gt, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST)
    if (gt > 0).sum() == 0:
        return None
    ys, xs = np.where(gt > 0)
    h, w = IMG_SIZE
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w - 1, int(xs.max()) + pad)
    y2 = min(h - 1, int(ys.max()) + pad)
    return (x1, y1, x2, y2)


def logits_to_prob_map(logits: np.ndarray, target_hw: tuple = (512, 512)) -> np.ndarray:
    """Upsample SAM2 low-res logits (256×256) → full-res sigmoid probability map."""
    t = torch.from_numpy(logits[0]).float().unsqueeze(0).unsqueeze(0)  # (1,1,256,256)
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return torch.sigmoid(t).squeeze().numpy().astype(np.float32)


def already_done(frame_id: str) -> bool:
    return (SAM2_PRED_DIR / frame_id / "probs.npz").exists()


def main() -> None:
    if not SAM2_CHECKPOINT.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {SAM2_CHECKPOINT}")

    # Build SAM2 image predictor
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    log.info("Loading SAM2-Tiny on %s ...", DEVICE)
    sam2_model = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    log.info("SAM2-Tiny ready")

    manifest = pd.read_parquet(MANIFEST_PATH)
    log.info("Manifest: %d frames (train=%d, cal=%d, test=%d)",
             len(manifest),
             (manifest.split == "train").sum(),
             (manifest.split == "cal").sum(),
             (manifest.split == "test").sum())

    n_done = sum(1 for _, row in manifest.iterrows() if already_done(row.frame_id))
    log.info("Already done: %d / %d", n_done, len(manifest))

    SAM2_PRED_DIR.mkdir(parents=True, exist_ok=True)

    n_empty = 0
    n_success = 0
    n_skip = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="SAM2 inference"):
        out_dir = SAM2_PRED_DIR / row.frame_id
        if already_done(row.frame_id):
            n_skip += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        bbox = compute_gt_bbox(row.mask_path)

        if bbox is None:
            # No instrument in GT mask → predict empty (perfect prediction)
            prob_map = np.zeros(IMG_SIZE, dtype=np.float32)
            mask_img = np.zeros(IMG_SIZE, dtype=np.uint8)
            n_empty += 1
        else:
            img_bgr = cv2.imread(row.frame_path)
            if img_bgr is None:
                log.warning("Missing frame: %s", row.frame_path)
                prob_map = np.zeros(IMG_SIZE, dtype=np.float32)
                mask_img = np.zeros(IMG_SIZE, dtype=np.uint8)
            else:
                img_bgr = cv2.resize(img_bgr, (IMG_SIZE[1], IMG_SIZE[0]))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                with torch.inference_mode():
                    predictor.set_image(img_rgb)
                    masks, scores, logits = predictor.predict(
                        box=np.array(bbox, dtype=np.float32),
                        multimask_output=False,
                    )

                # logits: (1, 256, 256); convert to full-res probability
                prob_map = logits_to_prob_map(logits)
                mask_img = (prob_map >= 0.5).astype(np.uint8) * 255
                n_success += 1

        save_npz({"probs": prob_map.astype(np.float16)}, out_dir / "probs.npz")
        cv2.imwrite(str(out_dir / "mask.png"), mask_img)

    total_processed = n_success + n_empty
    log.info(
        "Done: %d processed (%d non-empty, %d empty/no-instrument), %d skipped (already existed)",
        total_processed, n_success, n_empty, n_skip,
    )
    log.info("SAM2 predictions saved to %s", SAM2_PRED_DIR)


if __name__ == "__main__":
    main()
