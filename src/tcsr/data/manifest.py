"""Build and validate per-frame manifests (parquet)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tcsr.utils.io import save_parquet, load_parquet


MANIFEST_COLUMNS = [
    "frame_id",
    "video_id",
    "frame_idx_in_video",
    "frame_path",
    "mask_path",
    "split",
    "dataset",
]


def build_manifest(rows: list[dict], output_path: Path | str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    _validate(df)
    save_parquet(df, output_path)
    return df


def load_manifest(path: Path | str) -> pd.DataFrame:
    df = load_parquet(path)
    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    if df["frame_id"].duplicated().any():
        raise ValueError("Duplicate frame_ids in manifest")
    invalid_splits = set(df["split"].unique()) - {"train", "cal", "test"}
    if invalid_splits:
        raise ValueError(f"Invalid split values: {invalid_splits}")
