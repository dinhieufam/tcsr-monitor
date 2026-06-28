"""Frozen inference: logits → per-frame probs/masks saved to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tcsr.utils.io import save_npz
from tcsr.utils.logging import get_logger

log = get_logger(__name__)


class FrameDataset(Dataset):
    def __init__(self, manifest_df, transform):
        self.df = manifest_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row.frame_path)
        if img is None:
            raise FileNotFoundError(row.frame_path)
        tensor = torch.from_numpy(self.transform(img))
        return tensor, row.frame_id, row.frame_path


@torch.no_grad()
def run_predictions(
    model: nn.Module,
    manifest_df,
    output_dir: Path,
    transform,
    batch_size: int = 8,
    device: str = "cuda",
    tta_runner=None,
) -> None:
    """Run frozen inference and save mask.png + probs.npz per frame."""
    model.eval().to(device)
    dataset = FrameDataset(manifest_df, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=False)

    for batch_imgs, frame_ids, frame_paths in tqdm(loader, desc="Predicting"):
        batch_imgs = batch_imgs.to(device)
        logits = model(batch_imgs)
        if hasattr(logits, "logits"):
            logits = logits.logits  # HF SegFormer

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (B, H, W)

        if tta_runner is not None:
            tta_var = tta_runner(model, batch_imgs).cpu().numpy()
        else:
            tta_var = None

        for i, (fid, fpath) in enumerate(zip(frame_ids, frame_paths)):
            out_dir = output_dir / fid
            out_dir.mkdir(parents=True, exist_ok=True)

            prob_map = probs[i]
            mask = (prob_map >= 0.5).astype(np.uint8) * 255
            cv2.imwrite(str(out_dir / "mask.png"), mask)

            arrays = {"probs": prob_map.astype(np.float16)}
            if tta_var is not None:
                arrays["tta_variance"] = tta_var[i].astype(np.float16)
            save_npz(arrays, out_dir / "probs.npz")

    log.info("Predictions saved to %s", output_dir)
