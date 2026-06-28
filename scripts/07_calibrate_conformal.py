#!/usr/bin/env python
"""Stage 7: Conformal calibration — set threshold controlling missed failures."""

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from tcsr.conformal.split_conformal import calibrate_threshold
from tcsr.data.manifest import load_manifest
from tcsr.monitor.classifiers import load_monitor
from tcsr.utils import capture_run_metadata, get_logger, load_json, load_parquet, save_json, set_seed

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    capture_run_metadata(cfg, output_dir, cfg.seed)

    processed_dir = Path(cfg.data.processed_dir)
    manifest = load_manifest(processed_dir / "manifest.parquet")

    monitor_dir = Path(getattr(cfg, "monitor_dir", output_dir))
    monitor = load_monitor(monitor_dir / "monitor.pkl")
    feat_cols = load_json(monitor_dir / "feature_columns.json")

    features_df = load_parquet(processed_dir / f"features_{cfg.features.name}.parquet")
    labels_df = load_parquet(processed_dir / "labels.parquet")

    df = features_df.merge(labels_df, on="frame_id").merge(
        manifest[["frame_id", "split"]], on="frame_id"
    )
    cal_df = df[df.split == "cal"]

    X_cal = cal_df[feat_cols].values
    tau = getattr(cfg, "failure_tau", 0.5)
    tau_col = f"failure_tau{tau:.2f}".replace(".", "_")
    y_cal = cal_df[tau_col].values

    scores_cal = monitor.predict_proba(X_cal)
    result = calibrate_threshold(
        scores=scores_cal,
        labels=y_cal,
        alpha=cfg.conformal.alpha,
        score_type=cfg.conformal.score_type,
    )
    save_json(result, output_dir / "conformal_threshold.json")
    log.info("Conformal calibration: %s", result)


if __name__ == "__main__":
    main()
