#!/usr/bin/env python
"""
Stage 3b: Out-of-fold (OOF) predictions for training sequences.

For each training sequence i in {1..6}:
  1. Train U-Net on the remaining 5 training sequences (val on seq 7).
  2. Run inference on sequence i.
  3. Write predictions to data/processed/endovis2017/predictions/ (overwrites
     the in-sample predictions that were produced in Stage 3).

After this script completes, predictions/ contains:
  - seqs 1-6: OOF predictions (each from a model that never saw that sequence)
  - seqs 7-10: original predictions from the full model (untouched)

This gives the monitor training split real failure examples.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure package is importable from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.data.manifest import load_manifest
from tcsr.data.splits import make_oof_sequence_splits
from tcsr.data.transforms import ResizeMask, ResizeNormalize
from tcsr.segmentation.models import build_model
from tcsr.segmentation.predict import run_predictions
from tcsr.segmentation.train import train_model
from tcsr.utils.logging import get_logger

log = get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed/endovis2017")
MANIFEST_PATH = PROCESSED_DIR / "manifest.parquet"
PRED_DIR = PROCESSED_DIR / "predictions"
OOF_CKPT_DIR = Path("experiments/oof")
TRAIN_SEQS = [f"seq{i:02d}" for i in range(1, 7)]   # ["seq01"..."seq06"]
CAL_SEQ = "seq07"

IMG_SIZE = (512, 512)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEG_CFG = SimpleNamespace(
    architecture="Unet",
    encoder="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,
    checkpoint=None,  # fresh weights for each fold
)


# ── Dataset ──────────────────────────────────────────────────────────────────

class SegDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_transform, mask_transform):
        self.df = df.reset_index(drop=True)
        self.img_t = img_transform
        self.mask_t = mask_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import cv2
        row = self.df.iloc[idx]
        img = cv2.imread(row.frame_path)
        if img is None:
            raise FileNotFoundError(f"Missing frame: {row.frame_path}")
        mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Missing mask: {row.mask_path}")
        mask = (mask > 0).astype(np.uint8)
        return (
            torch.from_numpy(self.img_t(img)),
            torch.from_numpy(self.mask_t(mask)).long(),
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OOF_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(MANIFEST_PATH)
    img_transform = ResizeNormalize(IMG_SIZE, MEAN, STD)
    mask_transform = ResizeMask(IMG_SIZE)

    # Val loader is always seq 7 (fixed across folds)
    val_df = manifest[manifest.video_id == CAL_SEQ]
    val_loader = DataLoader(
        SegDataset(val_df, img_transform, mask_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    oof_folds = make_oof_sequence_splits(TRAIN_SEQS)
    log.info("Starting OOF cross-fitting: %d folds on device %s", len(oof_folds), DEVICE)

    for fold_idx, (fold_train_vids, held_out_vid) in enumerate(oof_folds):
        log.info(
            "=== Fold %d/%d: train on seqs %s | predict on seq %d ===",
            fold_idx + 1, len(oof_folds), fold_train_vids, held_out_vid,
        )

        # Training DataLoader for this fold
        fold_train_df = manifest[manifest.video_id.isin(fold_train_vids)]
        train_loader = DataLoader(
            SegDataset(fold_train_df, img_transform, mask_transform),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

        # Build fresh model (no pretrained checkpoint — ImageNet encoder weights only)
        model = build_model(SEG_CFG)

        ckpt_path = OOF_CKPT_DIR / f"fold_seq{held_out_vid}_checkpoint.pth"
        results = train_model(
            model,
            train_loader,
            val_loader,
            checkpoint_path=ckpt_path,
            device=DEVICE,
        )
        log.info(
            "Fold %d done: best_val_dice=%.4f epochs=%d",
            fold_idx + 1, results["best_val_dice"], results["epochs_trained"],
        )

        # Reload best checkpoint for inference
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)

        # Predict on held-out sequence → overwrites original in-sample predictions
        held_out_df = manifest[manifest.video_id == held_out_vid]
        run_predictions(
            model=model,
            manifest_df=held_out_df,
            output_dir=PRED_DIR,
            transform=img_transform,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            tta_runner=None,  # no TTA for training data
        )
        log.info("OOF predictions written for seq %d → %s", held_out_vid, PRED_DIR)

    log.info("OOF cross-fitting complete. All training-sequence predictions replaced.")
    log.info(
        "Run scripts/04_make_failure_labels.py next to recompute failure labels "
        "on the OOF predictions."
    )


if __name__ == "__main__":
    main()
