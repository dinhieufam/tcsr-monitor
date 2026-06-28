#!/usr/bin/env python
"""
Stage 3c: Generate corrupted predictions for robustness evaluation.

For each corruption type × severity, applies the corruption to every
test-split frame (seqs 8-10), saves the corrupted image to disk, and
runs the original trained U-Net to produce corrupted predictions.

Outputs:
  data/processed/endovis2017/corrupted_predictions/{corruption}_sev{s}/
      {frame_id}/mask.png
      {frame_id}/probs.npz
  data/processed/endovis2017/corrupted_frames/{corruption}_sev{s}/{frame_id}.png
  data/processed/endovis2017/corrupted_manifest.parquet
    columns: frame_id, video_id, frame_idx_in_video, mask_path, split,
             corruption_type, severity, frame_path (→ corrupted image), pred_dir
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.data.corruptions import CORRUPTION_TYPES, SEVERITIES, apply_corruption
from tcsr.data.manifest import load_manifest
from tcsr.data.transforms import ResizeNormalize
from tcsr.segmentation.models import build_model
from tcsr.segmentation.predict import run_predictions
from tcsr.utils.io import save_npz
from tcsr.utils.logging import get_logger

log = get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed/endovis2017")
MANIFEST_PATH = PROCESSED_DIR / "manifest.parquet"
ORIG_CKPT = Path("experiments/2026-06-02_13-48-40/checkpoint_best.pth")
CORRUPTED_PRED_BASE = PROCESSED_DIR / "corrupted_predictions"
CORRUPTED_FRAME_BASE = PROCESSED_DIR / "corrupted_frames"
CORRUPTED_MANIFEST_PATH = PROCESSED_DIR / "corrupted_manifest.parquet"

IMG_SIZE = (512, 512)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEG_CFG = SimpleNamespace(
    architecture="Unet",
    encoder="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,
    checkpoint=str(ORIG_CKPT),
)


class CorruptedFrameDataset(Dataset):
    """Loads frames from disk, applies corruption in-memory, returns tensor + ids."""

    def __init__(self, df: pd.DataFrame, corruption: str, severity: int, transform):
        self.df = df.reset_index(drop=True)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row.frame_path)
        if img is None:
            raise FileNotFoundError(f"Missing frame: {row.frame_path}")
        img = apply_corruption(img, self.corruption, self.severity)
        return torch.from_numpy(self.transform(img)), row.frame_id, row.frame_path


def save_corrupted_frames(df: pd.DataFrame, corruption: str, severity: int, out_dir: Path) -> pd.Series:
    """Save corrupted images to disk; return mapping frame_id -> corrupted_frame_path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for _, row in df.iterrows():
        img = cv2.imread(row.frame_path)
        if img is None:
            continue
        img_c = apply_corruption(img, corruption, severity)
        img_c_resized = cv2.resize(img_c, (IMG_SIZE[1], IMG_SIZE[0]))
        out_path = out_dir / f"{row.frame_id}.png"
        cv2.imwrite(str(out_path), img_c_resized)
        paths[row.frame_id] = str(out_path)
    return pd.Series(paths)


@torch.no_grad()
def predict_corrupted(
    model: torch.nn.Module,
    df: pd.DataFrame,
    corruption: str,
    severity: int,
    transform,
    pred_dir: Path,
) -> None:
    """Run inference on corrupted frames and save mask.png + probs.npz."""
    model.eval().to(DEVICE)
    dataset = CorruptedFrameDataset(df, corruption, severity, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)

    for batch_imgs, frame_ids, _ in loader:
        batch_imgs = batch_imgs.to(DEVICE)
        logits = model(batch_imgs)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        for i, fid in enumerate(frame_ids):
            out_dir = pred_dir / fid
            out_dir.mkdir(parents=True, exist_ok=True)
            prob_map = probs[i]
            mask = (prob_map >= 0.5).astype(np.uint8) * 255
            cv2.imwrite(str(out_dir / "mask.png"), mask)
            save_npz({"probs": prob_map.astype(np.float16)}, out_dir / "probs.npz")


def main() -> None:
    if not ORIG_CKPT.exists():
        # Try to find any available checkpoint
        candidates = sorted(Path("experiments").glob("*/checkpoint_best.pth"))
        if not candidates:
            raise FileNotFoundError(f"Checkpoint not found at {ORIG_CKPT}")
        ORIG_CKPT_used = candidates[-1]
        log.warning("Using fallback checkpoint: %s", ORIG_CKPT_used)
        SEG_CFG.checkpoint = str(ORIG_CKPT_used)

    manifest = load_manifest(MANIFEST_PATH)
    test_df = manifest[manifest.split == "test"].copy()
    transform = ResizeNormalize(IMG_SIZE, MEAN, STD)

    log.info("Loading model from %s", SEG_CFG.checkpoint)
    model = build_model(SEG_CFG)
    model.eval().to(DEVICE)

    manifest_rows = []
    total = len(CORRUPTION_TYPES) * len(SEVERITIES)
    done = 0

    for corruption in CORRUPTION_TYPES:
        for severity in SEVERITIES:
            tag = f"{corruption}_sev{severity}"
            pred_dir = CORRUPTED_PRED_BASE / tag
            frame_dir = CORRUPTED_FRAME_BASE / tag

            log.info("[%d/%d] %s ...", done + 1, total, tag)

            # Save corrupted frames to disk (needed for quality features)
            frame_path_map = save_corrupted_frames(test_df, corruption, severity, frame_dir)

            # Run inference on corrupted frames
            predict_corrupted(model, test_df, corruption, severity, transform, pred_dir)

            # Record manifest rows
            for _, row in test_df.iterrows():
                manifest_rows.append({
                    "frame_id": row.frame_id,
                    "video_id": row.video_id,
                    "frame_idx_in_video": row.frame_idx_in_video,
                    "mask_path": row.mask_path,
                    "split": "test",
                    "corruption_type": corruption,
                    "severity": severity,
                    "frame_path": frame_path_map.get(row.frame_id, row.frame_path),
                    "pred_dir": str(pred_dir),
                })
            done += 1

    corrupted_manifest = pd.DataFrame(manifest_rows)
    corrupted_manifest.to_parquet(CORRUPTED_MANIFEST_PATH, index=False)
    log.info(
        "Corrupted manifest saved: %d conditions × %d frames = %d rows → %s",
        total, len(test_df), len(corrupted_manifest), CORRUPTED_MANIFEST_PATH,
    )


if __name__ == "__main__":
    main()
