"""Structured run-metadata capture and logging setup."""

from __future__ import annotations

import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from tcsr.utils.io import save_json


def _git_info() -> dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        ))
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "dirty": False}


def _gpu_info() -> dict[str, str]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "driver": torch.version.cuda,
        }
    except ImportError:
        return {"available": False}


def capture_run_metadata(cfg: Any, output_dir: Path, seed: int) -> dict:
    from importlib.metadata import version, PackageNotFoundError

    packages = [
        "torch", "torchvision", "segmentation-models-pytorch", "numpy",
        "scikit-learn", "xgboost", "scikit-image", "pandas", "hydra-core",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "unknown"

    meta = {
        "timestamp": datetime.utcnow().isoformat(),
        "seed": seed,
        "git": _git_info(),
        "gpu": _gpu_info(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
    }
    save_json(meta, output_dir / "run_metadata.json")
    return meta


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
