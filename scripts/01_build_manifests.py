#!/usr/bin/env python
"""Stage 1: Build frame manifests with video-level splits."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from tcsr.utils import capture_run_metadata, get_logger, set_seed

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    capture_run_metadata(cfg, output_dir, cfg.seed)

    processed_dir = Path(cfg.data.processed_dir)
    manifest_path = processed_dir / "manifest.parquet"

    if cfg.data.name == "endovis2017":
        from tcsr.data.endovis2017 import build_endovis_manifest
        build_endovis_manifest(
            processed_dir=processed_dir,
            train_sequences=list(cfg.data.train_sequences),
            cal_sequences=list(cfg.data.cal_sequences),
            test_sequences=list(cfg.data.test_sequences),
            output_path=manifest_path,
        )
    elif cfg.data.name == "cholecseg8k":
        from tcsr.data.cholecseg8k import build_cholecseg_manifest
        build_cholecseg_manifest(
            processed_dir=processed_dir,
            test_sequences=list(cfg.data.test_sequences),
            output_path=manifest_path,
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.name}")

    log.info("Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
