#!/usr/bin/env python
"""Stage 5: Extract per-frame feature table."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from tcsr.data.manifest import load_manifest
from tcsr.features.build import build_feature_table
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
    features_path = processed_dir / f"features_{cfg.features.name}.parquet"

    feature_groups = dict(cfg.features.groups)

    shift_detector = None
    if feature_groups.get("shift", False):
        from tcsr.features.shift import ShiftDetector
        shift_detector = ShiftDetector(
            method=cfg.features.shift_method,
            n_neighbors=cfg.features.shift_n_neighbors,
        )
        # Note: shift detector fitting with embeddings is done in a separate pass
        # with the segmentation model; here we skip if no model is available
        log.warning(
            "Shift features require a model-embedding pass. "
            "Set features.shift=false or provide a fitted ShiftDetector."
        )
        shift_detector = None

    build_feature_table(
        manifest_df=manifest,
        pred_dir=pred_dir,
        output_path=features_path,
        feature_groups=feature_groups,
        shift_detector=shift_detector,
        temporal_window=cfg.features.temporal_window,
    )
    log.info("Features written to %s", features_path)


if __name__ == "__main__":
    main()
