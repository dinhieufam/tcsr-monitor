"""EndoVis 2017 dataset — frame/mask loaders and manifest builder."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from tcsr.data.splits import video_level_split
from tcsr.data.manifest import build_manifest
from tcsr.data.transforms import ResizeNormalize, ResizeMask


INSTRUMENT_CLASSES = list(range(1, 8))  # 7 instrument classes → binarize


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Any non-zero label → 1 (instrument present)."""
    return (mask > 0).astype(np.uint8)


def load_frame(path: Path | str, target_size: tuple[int, int] = (512, 512)) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


def load_mask(path: Path | str, target_size: tuple[int, int] = (512, 512)) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return binarize_mask(mask)


def build_endovis_manifest(
    processed_dir: Path,
    train_sequences: list[int],
    cal_sequences: list[int],
    test_sequences: list[int],
    output_path: Path,
) -> None:
    frames_dir = processed_dir / "frames"
    masks_dir = processed_dir / "masks"

    rows = []
    for seq_dir in sorted(frames_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq_id = int(re.search(r"\d+", seq_dir.name).group())
        for frame_path in sorted(seq_dir.glob("*.png")):
            frame_idx = int(frame_path.stem)
            mask_path = masks_dir / seq_dir.name / frame_path.name
            frame_id = f"endovis2017_seq{seq_id:02d}_{frame_idx:05d}"
            rows.append({
                "frame_id": frame_id,
                "video_id": f"seq{seq_id:02d}",
                "frame_idx_in_video": frame_idx,
                "frame_path": str(frame_path),
                "mask_path": str(mask_path),
                "split": "__TBD__",
                "dataset": "endovis2017",
            })

    import pandas as pd
    df = pd.DataFrame(rows)
    df = video_level_split(
        df,
        train_video_ids=[f"seq{s:02d}" for s in train_sequences],
        cal_video_ids=[f"seq{s:02d}" for s in cal_sequences],
        test_video_ids=[f"seq{s:02d}" for s in test_sequences],
    )
    build_manifest(df.to_dict("records"), output_path)
