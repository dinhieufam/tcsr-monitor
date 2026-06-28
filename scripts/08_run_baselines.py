#!/usr/bin/env python
"""Stage 8: Run all baseline risk-score methods."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from tcsr.baselines.runners import TemperatureScaling, run_all_baselines
from tcsr.data.manifest import load_manifest
from tcsr.utils import capture_run_metadata, get_logger, load_parquet, set_seed

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    capture_run_metadata(cfg, output_dir, cfg.seed)

    processed_dir = Path(cfg.data.processed_dir)
    manifest = load_manifest(processed_dir / "manifest.parquet")
    pred_dir = processed_dir / "predictions"
    baselines_path = processed_dir / "baselines.parquet"

    # Fit temperature scaling on calibration split
    cal_manifest = manifest[manifest.split == "cal"]
    labels_df = load_parquet(processed_dir / "labels.parquet")
    cal_labels = labels_df[labels_df.frame_id.isin(cal_manifest.frame_id)]

    ts = TemperatureScaling()
    # Temperature fitting would require logits — simplified to identity here
    log.info("Temperature scaling fitting skipped (logits not cached per frame)")

    run_all_baselines(
        manifest_df=manifest,
        pred_dir=pred_dir,
        output_path=baselines_path,
        temperature_scaler=None,  # set to ts after proper logit caching
    )
    log.info("Baselines written to %s", baselines_path)


if __name__ == "__main__":
    main()
