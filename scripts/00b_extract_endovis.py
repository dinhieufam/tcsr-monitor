#!/usr/bin/env python
"""
Extract and preprocess EndoVis 2017 zip files into the directory structure
expected by 01_build_manifests.py:

  data/processed/endovis2017/
      frames/seq01/ ... seq10/   *.png  RGB 512x512
      masks/ seq01/  ... seq10/  *.png  binary 512x512 (0/255)

Sources:
  - Seqs 1-4  frames: instrument_1_4_training.zip (frames 000-224)
  - Seqs 5-8  frames: instrument_5_8_training.zip (frames 000-224)
  - Seqs 1-8  masks:  same training zips, ground_truth/* per-instrument labels (OR-merged)
  - Seqs 9-10 frames: instrument_9_10_testing.zip
  - Seqs 9-10 masks:  instrument_2017_test.zip  BinarySegmentation/
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

RAW = Path("data/raw/endovis2017")
OUT_FRAMES = Path("data/processed/endovis2017/frames")
OUT_MASKS = Path("data/processed/endovis2017/masks")
SIZE = (512, 512)


def resize_frame(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode frame")
    return cv2.resize(img, SIZE, interpolation=cv2.INTER_LINEAR)


def resize_mask(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("cannot decode mask")
    img = cv2.resize(img, SIZE, interpolation=cv2.INTER_NEAREST)
    return (img > 0).astype(np.uint8) * 255


def save_png(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def extract_training_seqs(zip_path: Path, dataset_ids: list[int]) -> None:
    """Extract frames 000-224 and binary masks for training-zip datasets."""
    print(f"\nExtracting {zip_path.name} (datasets {dataset_ids}) ...")
    zf = zipfile.ZipFile(zip_path)

    for ds in dataset_ids:
        seq_tag = f"seq{ds:02d}"
        prefix_frame = f"instrument_dataset_{ds}/left_frames/"
        prefix_gt = f"instrument_dataset_{ds}/ground_truth/"

        # Collect per-frame: frame bytes + list of GT label bytes
        frame_bytes: dict[str, bytes] = {}
        gt_by_frame: dict[str, list[bytes]] = {}

        for name in zf.namelist():
            if name.startswith(prefix_frame) and name.endswith(".png"):
                fname = Path(name).name
                frame_bytes[fname] = zf.read(name)
            elif name.startswith(prefix_gt) and name.endswith(".png"):
                fname = Path(name).name
                gt_by_frame.setdefault(fname, []).append(zf.read(name))

        print(f"  seq{ds:02d}: {len(frame_bytes)} frames, {len(gt_by_frame)} mask frames")

        for fname in tqdm(sorted(frame_bytes), desc=f"seq{ds:02d}", leave=False):
            frame_idx = int(Path(fname).stem.replace("frame", ""))
            out_fname = f"{frame_idx:05d}.png"

            # Frame
            frame = resize_frame(frame_bytes[fname])
            save_png(frame, OUT_FRAMES / seq_tag / out_fname)

            # Mask (OR all per-instrument label images)
            if fname in gt_by_frame:
                combined = np.zeros(SIZE, dtype=np.uint8)
                for label_bytes in gt_by_frame[fname]:
                    arr = np.frombuffer(label_bytes, np.uint8)
                    lbl = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                    if lbl is not None:
                        lbl_r = cv2.resize(lbl, SIZE, interpolation=cv2.INTER_NEAREST)
                        combined = np.maximum(combined, (lbl_r > 0).astype(np.uint8) * 255)
                save_png(combined, OUT_MASKS / seq_tag / out_fname)
            else:
                # No GT for this frame — save all-zero mask
                save_png(np.zeros(SIZE, dtype=np.uint8), OUT_MASKS / seq_tag / out_fname)

    zf.close()


def extract_test_seqs_9_10() -> None:
    """Extract frames for seqs 9-10 from testing zip."""
    zip_path = RAW / "instrument_9_10_testing.zip"
    print(f"\nExtracting {zip_path.name} (seqs 9-10 frames) ...")
    zf = zipfile.ZipFile(zip_path)

    for ds in [9, 10]:
        seq_tag = f"seq{ds:02d}"
        prefix = f"instrument_dataset_{ds}/left_frames/"
        frames = [(n, zf.read(n)) for n in zf.namelist() if n.startswith(prefix) and n.endswith(".png")]
        print(f"  seq{ds:02d}: {len(frames)} frames")

        for name, data in tqdm(frames, desc=seq_tag, leave=False):
            frame_idx = int(Path(name).stem.replace("frame", ""))
            frame = resize_frame(data)
            save_png(frame, OUT_FRAMES / seq_tag / f"{frame_idx:05d}.png")

    zf.close()


def extract_test_masks() -> None:
    """Extract binary masks for seqs 9-10 from instrument_2017_test.zip."""
    zip_path = RAW / "instrument_2017_test.zip"
    print(f"\nExtracting {zip_path.name} (binary masks for seqs 9-10) ...")
    zf = zipfile.ZipFile(zip_path)

    for ds in [9, 10]:
        seq_tag = f"seq{ds:02d}"
        prefix = f"instrument_2017_test/instrument_dataset_{ds}/BinarySegmentation/"
        masks = [(n, zf.read(n)) for n in zf.namelist() if n.startswith(prefix) and n.endswith(".png")]
        print(f"  seq{ds:02d}: {len(masks)} binary masks")

        for name, data in tqdm(masks, desc=seq_tag, leave=False):
            frame_idx = int(Path(name).stem.replace("frame", ""))
            mask = resize_mask(data)
            save_png(mask, OUT_MASKS / seq_tag / f"{frame_idx:05d}.png")

    zf.close()


def verify_output() -> None:
    print("\nVerification:")
    for ds in range(1, 11):
        seq_tag = f"seq{ds:02d}"
        n_frames = len(list((OUT_FRAMES / seq_tag).glob("*.png"))) if (OUT_FRAMES / seq_tag).exists() else 0
        n_masks = len(list((OUT_MASKS / seq_tag).glob("*.png"))) if (OUT_MASKS / seq_tag).exists() else 0
        print(f"  {seq_tag}: {n_frames:4d} frames  {n_masks:4d} masks")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent.parent)

    # Seqs 1-4: from instrument_1_4_training.zip
    extract_training_seqs(RAW / "instrument_1_4_training.zip", [1, 2, 3, 4])

    # Seqs 5-8: from instrument_5_8_training.zip
    extract_training_seqs(RAW / "instrument_5_8_training.zip", [5, 6, 7, 8])

    # Seqs 9-10: frames from testing zip, masks from test zip
    extract_test_seqs_9_10()
    extract_test_masks()

    verify_output()
    print("\nDone. Run: python scripts/01_build_manifests.py data=endovis2017")
