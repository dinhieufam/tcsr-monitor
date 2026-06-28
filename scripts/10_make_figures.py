#!/usr/bin/env python
"""Stage 10: Regenerate all paper figures from saved metrics."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from tcsr.utils import capture_run_metadata, get_logger, set_seed
from tcsr.viz.figures import make_all_figures

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    capture_run_metadata(cfg, output_dir, cfg.seed)

    metrics_dir = Path("results/metrics")
    figures_dir = Path("results/figures")

    make_all_figures(metrics_dir=metrics_dir, output_dir=figures_dir)
    log.info("Figures written to %s", figures_dir)


if __name__ == "__main__":
    main()
