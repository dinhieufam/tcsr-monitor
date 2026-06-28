#!/usr/bin/env python
"""Stage 3: Frozen inference — probs/masks to disk."""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from tcsr.data.manifest import load_manifest
from tcsr.data.transforms import ResizeNormalize
from tcsr.segmentation.models import build_model
from tcsr.segmentation.predict import run_predictions
from tcsr.segmentation.tta import TTARunner
from tcsr.utils import capture_run_metadata, get_logger, set_seed

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    capture_run_metadata(cfg, output_dir, cfg.seed)

    manifest = load_manifest(Path(cfg.data.processed_dir) / "manifest.parquet")
    img_size = tuple(cfg.data.img_size)
    transform = ResizeNormalize(img_size, cfg.data.mean, cfg.data.std)

    model = build_model(cfg.seg_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tta_runner = TTARunner(n_augmentations=8)

    pred_dir = Path(cfg.data.processed_dir) / "predictions"
    run_predictions(
        model=model,
        manifest_df=manifest,
        output_dir=pred_dir,
        transform=transform,
        device=device,
        tta_runner=tta_runner,
    )
    log.info("Predictions complete → %s", pred_dir)


if __name__ == "__main__":
    main()
