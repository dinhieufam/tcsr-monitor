#!/usr/bin/env python
"""Stage 6: Train the failure monitor on train videos."""

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tcsr.data.manifest import load_manifest
from tcsr.monitor.train import train_monitor
from tcsr.utils import capture_run_metadata, get_logger, load_parquet, set_seed

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    capture_run_metadata(cfg, output_dir, cfg.seed)

    processed_dir = Path(cfg.data.processed_dir)
    manifest = load_manifest(processed_dir / "manifest.parquet")

    features_df = load_parquet(processed_dir / f"features_{cfg.features.name}.parquet")
    labels_df = load_parquet(processed_dir / "labels.parquet")

    # Attach split info
    features_df = features_df.merge(manifest[["frame_id", "split"]], on="frame_id")

    tau = getattr(cfg, "failure_tau", 0.5)
    tau_col = f"failure_tau{tau:.2f}".replace(".", "_")

    monitor = instantiate(cfg.monitor)
    results = train_monitor(
        monitor=monitor,
        features_df=features_df,
        labels_df=labels_df,
        tau_col=tau_col,
        output_dir=output_dir,
    )
    log.info("Monitor training complete: %s", results)


if __name__ == "__main__":
    main()
