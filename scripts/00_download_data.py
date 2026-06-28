#!/usr/bin/env python
"""Stage 0: Download and verify datasets."""

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from tcsr.data.download import verify_dataset
from tcsr.utils import capture_run_metadata, get_logger, set_seed

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    capture_run_metadata(cfg, output_dir, cfg.seed)

    raw_dir = Path(cfg.data.root)
    checksum_dir = Path("data/checksums")

    checksum_file = checksum_dir / f"{cfg.data.name}.sha256"
    if not raw_dir.exists():
        log.warning(
            "Raw data directory %s does not exist. "
            "Please download %s manually and place it there. "
            "See docs/datasets.md for instructions.",
            raw_dir,
            cfg.data.name,
        )
        return

    if checksum_file.exists():
        log.info("Verifying checksums for %s ...", cfg.data.name)
        verify_dataset(raw_dir, checksum_file)
    else:
        log.warning("No checksum file at %s — skipping verification.", checksum_file)

    log.info("Data stage complete for %s", cfg.data.name)


if __name__ == "__main__":
    main()
