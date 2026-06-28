#!/usr/bin/env python
"""Stage 4: Compute per-frame failure labels y_t = 1[IoU(pred,gt) < tau]."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from tcsr.data.manifest import load_manifest
from tcsr.labels.failure_labels import compute_failure_labels
from tcsr.utils import capture_run_metadata, get_logger, set_seed

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    capture_run_metadata(cfg, output_dir, cfg.seed)

    processed_dir = Path(cfg.data.processed_dir)
    manifest = load_manifest(processed_dir / "manifest.parquet")
    pred_dir = processed_dir / "predictions"
    labels_path = processed_dir / "labels.parquet"

    compute_failure_labels(
        manifest_df=manifest,
        pred_dir=pred_dir,
        output_path=labels_path,
        taus=[0.5, 0.6, 0.75],
    )
    log.info("Labels written to %s", labels_path)


if __name__ == "__main__":
    main()
