#!/usr/bin/env python
"""Stage 2 (optional): Train base segmentation model."""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from tcsr.data.manifest import load_manifest
from tcsr.data.transforms import ResizeNormalize, ResizeMask
from tcsr.segmentation.models import build_model
from tcsr.segmentation.train import train_model
from tcsr.utils import capture_run_metadata, get_logger, save_json, set_seed

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    capture_run_metadata(cfg, output_dir, cfg.seed)

    manifest = load_manifest(Path(cfg.data.processed_dir) / "manifest.parquet")
    train_df = manifest[manifest.split == "train"]
    val_df = manifest[manifest.split == "cal"]

    img_size = tuple(cfg.data.img_size)
    transform = ResizeNormalize(img_size, cfg.data.mean, cfg.data.std)
    mask_transform = ResizeMask(img_size)

    from tcsr.data.endovis2017 import load_frame, load_mask
    from torch.utils.data import Dataset

    class SegDataset(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            import cv2
            import numpy as np
            row = self.df.iloc[idx]
            img = cv2.imread(row.frame_path)
            mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)
            return (
                torch.from_numpy(transform(img)),
                torch.from_numpy(mask_transform(mask)).long(),
            )

    train_loader = DataLoader(SegDataset(train_df), batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(SegDataset(val_df), batch_size=8, shuffle=False, num_workers=0)

    model = build_model(cfg.seg_model)
    checkpoint_path = output_dir / "checkpoint_best.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = train_model(
        model, train_loader, val_loader,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    save_json(results, output_dir / "train_results.json")
    log.info("Training complete: %s", results)


if __name__ == "__main__":
    main()
