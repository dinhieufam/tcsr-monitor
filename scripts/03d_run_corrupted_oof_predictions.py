#!/usr/bin/env python
"""
Stage 3d: Generate corrupted predictions for training sequences using OOF fold checkpoints.

For each training sequence (seq01-seq06):
  - Load the corresponding OOF fold checkpoint (trained on all other training seqs)
  - For each corruption type × severity: apply corruption in memory, run inference
  - Save predictions to corrupted_train_predictions/{corruption}_sev{s}/
  - Save corrupted frames to corrupted_train_frames/{corruption}_sev{s}/
  - Record manifest rows for downstream feature extraction + labelling

Outputs:
  data/processed/endovis2017/corrupted_train_predictions/{tag}/{frame_id}/mask.png
  data/processed/endovis2017/corrupted_train_predictions/{tag}/{frame_id}/probs.npz
  data/processed/endovis2017/corrupted_train_frames/{tag}/{frame_id}.png
  data/processed/endovis2017/corrupted_train_manifest.parquet
    columns: frame_id, video_id, frame_idx_in_video, mask_path, split,
             corruption_type, severity, frame_path, pred_dir

This gives the monitor training split corruption-induced failures so it can learn
to use quality / shape / temporal features that entropy cannot see.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.data.corruptions import CORRUPTION_TYPES, SEVERITIES, apply_corruption
from tcsr.data.manifest import load_manifest
from tcsr.data.transforms import ResizeNormalize
from tcsr.segmentation.models import build_model
from tcsr.utils.io import save_npz
from tcsr.utils.logging import get_logger

log = get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed/endovis2017")
MANIFEST_PATH = PROCESSED_DIR / "manifest.parquet"
OOF_CKPT_DIR = Path("experiments/oof")
CORRUPTED_TRAIN_PRED_BASE = PROCESSED_DIR / "corrupted_train_predictions"
CORRUPTED_TRAIN_FRAME_BASE = PROCESSED_DIR / "corrupted_train_frames"
CORRUPTED_TRAIN_MANIFEST_PATH = PROCESSED_DIR / "corrupted_train_manifest.parquet"

TRAIN_SEQS = [f"seq{i:02d}" for i in range(1, 7)]
# OOF checkpoint for seqXX was trained on all training seqs EXCEPT seqXX
OOF_CKPTS = {seq: OOF_CKPT_DIR / f"fold_seq{seq}_checkpoint.pth" for seq in TRAIN_SEQS}

IMG_SIZE = (512, 512)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 16
DEVICE = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

SEG_CFG = SimpleNamespace(
    architecture="Unet",
    encoder="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,
    checkpoint=None,  # loaded per-fold below
)


def load_fold_model(seq: str) -> torch.nn.Module:
    ckpt_path = OOF_CKPTS[seq]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"OOF fold checkpoint missing: {ckpt_path}. Run 03b_run_oof_predictions.py first."
        )
    model = build_model(SEG_CFG)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    log.info("Loaded OOF fold model for %s from %s", seq, ckpt_path)
    return model


@torch.no_grad()
def predict_corrupted_batch(
    model: torch.nn.Module,
    seq_df: pd.DataFrame,
    corruption: str,
    severity: int,
    transform: ResizeNormalize,
    pred_dir: Path,
    frame_dir: Path,
) -> list[dict]:
    """Run inference on corrupted frames; save predictions + corrupted images.

    Returns list of manifest row dicts.
    """
    pred_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    buf_imgs: list[np.ndarray] = []
    buf_meta: list[dict] = []

    def flush(buf_imgs, buf_meta):
        if not buf_imgs:
            return
        batch = torch.from_numpy(np.stack(buf_imgs)).to(DEVICE)
        logits = model(batch)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        for i, meta in enumerate(buf_meta):
            prob_map = probs[i]
            out_dir = pred_dir / meta["frame_id"]
            out_dir.mkdir(parents=True, exist_ok=True)
            mask = (prob_map >= 0.5).astype(np.uint8) * 255
            cv2.imwrite(str(out_dir / "mask.png"), mask)
            save_npz({"probs": prob_map.astype(np.float16)}, out_dir / "probs.npz")

    for _, row in seq_df.sort_values("frame_idx_in_video").iterrows():
        img = cv2.imread(row.frame_path)
        if img is None:
            log.warning("Missing frame: %s", row.frame_path)
            continue

        img_c = apply_corruption(img, corruption, severity)

        # Save corrupted frame to disk (needed for quality feature extraction later)
        corrupted_frame_path = frame_dir / f"{row.frame_id}.png"
        img_c_resized = cv2.resize(img_c, (IMG_SIZE[1], IMG_SIZE[0]))
        cv2.imwrite(str(corrupted_frame_path), img_c_resized)

        buf_imgs.append(transform(img_c))
        buf_meta.append({
            "frame_id": row.frame_id,
            "corrupted_frame_path": str(corrupted_frame_path),
        })

        manifest_rows.append({
            "frame_id": row.frame_id,
            "video_id": row.video_id,
            "frame_idx_in_video": int(row.frame_idx_in_video),
            "mask_path": row.mask_path,
            "split": "train",
            "corruption_type": corruption,
            "severity": severity,
            "frame_path": str(corrupted_frame_path),
            "pred_dir": str(pred_dir),
        })

        if len(buf_imgs) == BATCH_SIZE:
            flush(buf_imgs, buf_meta)
            buf_imgs, buf_meta = [], []

    flush(buf_imgs, buf_meta)
    return manifest_rows


def main() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    transform = ResizeNormalize(IMG_SIZE, MEAN, STD)

    total_conditions = len(TRAIN_SEQS) * len(CORRUPTION_TYPES) * len(SEVERITIES)
    done = 0
    all_manifest_rows: list[dict] = []

    for seq in TRAIN_SEQS:
        model = load_fold_model(seq)
        seq_df = manifest[manifest.video_id == seq].copy()
        log.info("Processing seq %s (%d frames) × %d conditions ...",
                 seq, len(seq_df), len(CORRUPTION_TYPES) * len(SEVERITIES))

        for corruption in CORRUPTION_TYPES:
            for severity in SEVERITIES:
                tag = f"{corruption}_sev{severity}"
                pred_dir = CORRUPTED_TRAIN_PRED_BASE / tag
                frame_dir = CORRUPTED_TRAIN_FRAME_BASE / tag
                done += 1
                log.info("[%d/%d] %s/%s ...", done, total_conditions, seq, tag)

                rows = predict_corrupted_batch(
                    model, seq_df, corruption, severity, transform, pred_dir, frame_dir
                )
                all_manifest_rows.extend(rows)

        # Free GPU memory between fold models
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    corrupted_manifest = pd.DataFrame(all_manifest_rows)
    corrupted_manifest.to_parquet(CORRUPTED_TRAIN_MANIFEST_PATH, index=False)
    log.info(
        "Corrupted train manifest saved: %d seqs × %d conditions × ~%d frames = %d rows → %s",
        len(TRAIN_SEQS),
        len(CORRUPTION_TYPES) * len(SEVERITIES),
        len(manifest[manifest.video_id.isin(TRAIN_SEQS)]),
        len(corrupted_manifest),
        CORRUPTED_TRAIN_MANIFEST_PATH,
    )
    log.info("Run scripts/04c_make_corrupted_train_labels.py next.")


if __name__ == "__main__":
    main()
