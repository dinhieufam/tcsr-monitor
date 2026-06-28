"""Typed save/load helpers with content hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_parquet(df: pd.DataFrame, path: Path | str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return _sha256(path)


def load_parquet(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def save_npz(arrays: dict[str, np.ndarray], path: Path | str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return _sha256(path)


def load_npz(path: Path | str) -> dict[str, np.ndarray]:
    return dict(np.load(Path(path)))


def save_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path | str) -> Any:
    with open(Path(path)) as f:
        return json.load(f)
