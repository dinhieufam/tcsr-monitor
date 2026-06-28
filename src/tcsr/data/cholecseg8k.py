"""CholecSeg8k dataset — binarized instrument masks, manifest builder."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from tcsr.data.manifest import build_manifest
from tcsr.data.splits import video_level_split


# CholecSeg8k has 13 classes; instrument-related classes to keep as foreground
INSTRUMENT_CLASS_IDS = {5, 6, 7, 8}  # update per dataset docs


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    return np.isin(mask, list(INSTRUMENT_CLASS_IDS)).astype(np.uint8)


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


def build_cholecseg_manifest(
    processed_dir: Path,
    test_sequences: list[int],
    output_path: Path,
) -> None:
    frames_dir = processed_dir / "frames"
    masks_dir = processed_dir / "masks"

    rows = []
    for clip_dir in sorted(frames_dir.iterdir()):
        if not clip_dir.is_dir():
            continue
        clip_id = int(re.search(r"\d+", clip_dir.name).group())
        for frame_path in sorted(clip_dir.glob("*.png")):
            frame_idx = int(frame_path.stem)
            mask_path = masks_dir / clip_dir.name / frame_path.name
            frame_id = f"cholecseg8k_clip{clip_id:02d}_{frame_idx:05d}"
            rows.append({
                "frame_id": frame_id,
                "video_id": f"clip{clip_id:02d}",
                "frame_idx_in_video": frame_idx,
                "frame_path": str(frame_path),
                "mask_path": str(mask_path),
                "split": "__TBD__",
                "dataset": "cholecseg8k",
            })

    import pandas as pd
    df = pd.DataFrame(rows)
    df = video_level_split(
        df,
        train_video_ids=[],
        cal_video_ids=[],
        test_video_ids=[f"clip{s:02d}" for s in test_sequences],
    )
    build_manifest(df.to_dict("records"), output_path)
