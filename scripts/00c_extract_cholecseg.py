#!/usr/bin/env python
"""
Extract and preprocess CholecSeg8k from cholecseg8k.zip into:

  data/processed/cholecseg8k/
      frames/clip01/ ... clip17/   {frame_idx:05d}.png   RGB 512x512
      masks/ clip01/  ... clip17/  {frame_idx:05d}.png   grayscale class IDs (0-12)

Videos are mapped to sequential clip IDs:
  video01 → clip01, video09 → clip02, video12 → clip03, ...
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

RAW = Path("data/raw/cholecseg8k/cholecseg8k.zip")
OUT_FRAMES = Path("data/processed/cholecseg8k/frames")
OUT_MASKS = Path("data/processed/cholecseg8k/masks")
SIZE = (512, 512)


def resize_frame(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode frame")
    return cv2.resize(img, SIZE, interpolation=cv2.INTER_LINEAR)


def resize_mask_grayscale(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("cannot decode mask")
    return cv2.resize(img, SIZE, interpolation=cv2.INTER_NEAREST)


def save_png(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent.parent)

    print(f"Opening {RAW} ...")
    zf = zipfile.ZipFile(RAW)
    names = zf.namelist()

    # Find all unique top-level video directories (sorted)
    videos = sorted(set(n.split("/")[0] for n in names if "/" in n and n.split("/")[0].startswith("video")))
    print(f"Found {len(videos)} videos: {videos}")

    for clip_idx, video_name in enumerate(videos, start=1):
        clip_tag = f"clip{clip_idx:02d}"
        print(f"\n{video_name} → {clip_tag}")

        # Collect frame files across all sub-clip directories
        frame_entries = {}  # frame_idx → zip entry name
        mask_entries = {}   # frame_idx → zip entry name

        for entry in names:
            if not entry.startswith(video_name + "/"):
                continue
            fname = entry.split("/")[-1]
            if fname.endswith("_endo.png") and not fname.startswith("._"):
                m = re.search(r"frame_(\d+)_endo\.png", fname)
                if m:
                    frame_idx = int(m.group(1))
                    frame_entries[frame_idx] = entry
            elif fname.endswith("_endo_mask.png") and not fname.startswith("._"):
                m = re.search(r"frame_(\d+)_endo_mask\.png", fname)
                if m:
                    frame_idx = int(m.group(1))
                    mask_entries[frame_idx] = entry

        print(f"  {len(frame_entries)} frames, {len(mask_entries)} masks")

        for frame_idx in tqdm(sorted(frame_entries), desc=clip_tag, leave=False):
            out_name = f"{frame_idx:05d}.png"

            # Frame
            frame = resize_frame(zf.read(frame_entries[frame_idx]))
            save_png(frame, OUT_FRAMES / clip_tag / out_name)

            # Mask (grayscale class IDs 0-12)
            if frame_idx in mask_entries:
                mask = resize_mask_grayscale(zf.read(mask_entries[frame_idx]))
            else:
                mask = np.zeros(SIZE, dtype=np.uint8)
            save_png(mask, OUT_MASKS / clip_tag / out_name)

    zf.close()

    # Verify
    print("\nVerification:")
    for clip_idx in range(1, len(videos) + 1):
        clip_tag = f"clip{clip_idx:02d}"
        n_f = len(list((OUT_FRAMES / clip_tag).glob("*.png"))) if (OUT_FRAMES / clip_tag).exists() else 0
        n_m = len(list((OUT_MASKS / clip_tag).glob("*.png"))) if (OUT_MASKS / clip_tag).exists() else 0
        print(f"  {clip_tag}: {n_f:5d} frames  {n_m:5d} masks")

    print("\nDone. Run: python scripts/01_build_manifests.py data=cholecseg8k")
