"""Video-level (not frame-level) split assignment to avoid temporal leakage."""

from __future__ import annotations

import pandas as pd


def make_oof_sequence_splits(train_video_ids: list) -> list[tuple]:
    """Leave-one-sequence-out: for each training seq, return (oof_train_ids, held_out_id)."""
    return [([v for v in train_video_ids if v != h], h) for h in train_video_ids]


def video_level_split(
    df: pd.DataFrame,
    train_video_ids: list,
    cal_video_ids: list,
    test_video_ids: list,
    video_col: str = "video_id",
) -> pd.DataFrame:
    """Add a 'split' column based on video-level assignment."""
    split_map: dict = {}
    for vid in train_video_ids:
        split_map[vid] = "train"
    for vid in cal_video_ids:
        split_map[vid] = "cal"
    for vid in test_video_ids:
        split_map[vid] = "test"

    unknown = set(df[video_col].unique()) - set(split_map.keys())
    if unknown:
        raise ValueError(f"Videos not assigned to any split: {unknown}")

    df = df.copy()
    df["split"] = df[video_col].map(split_map)
    return df
