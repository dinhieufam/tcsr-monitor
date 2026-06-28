#!/usr/bin/env python
"""
Stage 6b: Train the failure monitor on combined clean OOF + corrupted training data.

Training data:
  - Clean OOF features (features_all.parquet, split=train):
      ~1350 frames, ~77 positives at τ=0.75
  - Corrupted training features (extracted inline from corrupted_train_manifest.parquet):
      ~40,500 rows (1350 frames × 30 conditions), ~9,000+ positives at τ=0.75

Calibration (for XGBoost early stopping): clean cal split (seq07, τ=0.75).

Usage:
  python scripts/06b_train_monitor_combined.py [--seed 0] [--tau 0.75] \
      [--output-dir experiments/combined_seed0]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.features.confidence import compute_confidence_features
from tcsr.features.quality import compute_quality_features
from tcsr.features.shape import compute_shape_features
from tcsr.features.temporal import compute_temporal_features, mask_iou
from tcsr.monitor.classifiers import XGBMonitor, save_monitor
from tcsr.utils.io import load_json, load_npz, load_parquet, save_json
from tcsr.utils.logging import get_logger
from tcsr.utils.seed import set_seed

log = get_logger(__name__)

PROCESSED_DIR = Path("data/processed/endovis2017")
_EPS = 1e-7

# Features the monitor expects (must match features_all.parquet column order)
FEATURE_PREFIXES = ("conf_", "shape_", "temp_", "qual_", "shift_")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]


def extract_features_for_condition(
    manifest_rows: pd.DataFrame,
    pred_dir: Path,
    feat_cols: list[str],
) -> np.ndarray:
    """Extract features for all frames in one (corruption, severity) condition.

    Temporal features reset at the start of each video within the condition
    to avoid cross-video leakage. Quality features come from the corrupted
    frame images stored on disk during 03d.

    Returns array of shape (N, len(feat_cols)).
    """
    rows_feat = []
    prev_mask = None
    prev_area = None
    prev_vid = None
    iou_history: list[float] = []

    for _, row in manifest_rows.iterrows():
        if row.video_id != prev_vid:
            prev_mask = None
            prev_area = None
            iou_history = []
            prev_vid = row.video_id

        npz_path = pred_dir / row.frame_id / "probs.npz"
        try:
            npz = load_npz(npz_path)
            prob_map = npz["probs"].astype(np.float32)
        except Exception:
            prob_map = np.zeros((512, 512), dtype=np.float32)

        curr_mask = (prob_map >= 0.5).astype(np.uint8)
        curr_area = float(curr_mask.mean())

        feat: dict = {}
        feat.update(compute_confidence_features(prob_map))
        feat.update(compute_shape_features(curr_mask))
        feat.update(compute_temporal_features(curr_mask, prev_mask, curr_area, prev_area, iou_history[-5:]))

        # Quality features from corrupted frame saved to disk by 03d
        try:
            bgr = cv2.imread(row.frame_path)
            if bgr is not None:
                feat.update(compute_quality_features(bgr))
        except Exception:
            pass

        if prev_mask is not None:
            iou_history.append(mask_iou(curr_mask.astype(bool), prev_mask.astype(bool)))

        prev_mask = curr_mask
        prev_area = curr_area
        rows_feat.append(feat)

    feat_df = pd.DataFrame(rows_feat)
    for col in feat_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    return feat_df[feat_cols].fillna(0.0).values.astype(np.float32)


def _load_manifest_splits() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "manifest.parquet")[["frame_id", "split"]]


def load_clean_train_data(
    feat_cols: list[str],
    tau_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load clean OOF train features + labels (split=train only)."""
    features_df = load_parquet(PROCESSED_DIR / "features_all.parquet")
    labels_df = load_parquet(PROCESSED_DIR / "labels.parquet")
    splits_df = _load_manifest_splits()

    df = features_df.merge(labels_df[["frame_id", tau_col]], on="frame_id").merge(splits_df, on="frame_id")
    train_df = df[df["split"] == "train"]

    X = train_df[feat_cols].values.astype(np.float32)
    y = train_df[tau_col].values
    log.info("Clean OOF train: %d frames, %d positives (%.1f%%)", len(y), y.sum(), 100 * y.mean())
    return X, y


def load_clean_cal_data(
    feat_cols: list[str],
    tau_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load clean calibration split features + labels (for XGBoost early stopping)."""
    features_df = load_parquet(PROCESSED_DIR / "features_all.parquet")
    labels_df = load_parquet(PROCESSED_DIR / "labels.parquet")
    splits_df = _load_manifest_splits()

    df = features_df.merge(labels_df[["frame_id", tau_col]], on="frame_id").merge(splits_df, on="frame_id")
    cal_df = df[df["split"] == "cal"]

    if len(cal_df) == 0:
        return None, None
    X = cal_df[feat_cols].values.astype(np.float32)
    y = cal_df[tau_col].values
    log.info("Clean cal: %d frames, %d positives (%.1f%%)", len(y), y.sum(), 100 * y.mean())
    return X, y


def load_corrupted_train_data(
    feat_cols: list[str],
    tau_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from corrupted training predictions inline and return (X, y)."""
    c_manifest_path = PROCESSED_DIR / "corrupted_train_manifest.parquet"
    c_labels_path = PROCESSED_DIR / "corrupted_train_labels.parquet"

    if not c_manifest_path.exists():
        raise FileNotFoundError(f"Run 03d first. Missing: {c_manifest_path}")
    if not c_labels_path.exists():
        raise FileNotFoundError(f"Run 04c first. Missing: {c_labels_path}")

    c_manifest = pd.read_parquet(c_manifest_path)
    c_labels = pd.read_parquet(c_labels_path)

    c_manifest = c_manifest.merge(
        c_labels[["frame_id", "corruption_type", "severity", tau_col]],
        on=["frame_id", "corruption_type", "severity"],
        how="left",
    )

    conditions = c_manifest.groupby(["corruption_type", "severity"])
    n_conditions = c_manifest[["corruption_type", "severity"]].drop_duplicates().shape[0]

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for (corruption, severity), grp in tqdm(conditions, total=n_conditions, desc="Corrupted train features"):
        grp = grp.sort_values(["video_id", "frame_idx_in_video"]).reset_index(drop=True)
        pred_dir = Path(grp.pred_dir.iloc[0])

        y_cond = grp[tau_col].fillna(0).values.astype(int)
        X_cond = extract_features_for_condition(grp, pred_dir, feat_cols)

        X_parts.append(X_cond)
        y_parts.append(y_cond)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    log.info("Corrupted train: %d rows, %d positives (%.1f%%)", len(y), y.sum(), 100 * y.mean())
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train monitor on combined OOF + corrupted data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.75,
                        help="IoU failure threshold (0.5, 0.6, or 0.75)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to save monitor.pkl (default: experiments/combined_seed{seed})")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    tau_col = f"failure_tau{args.tau:.2f}".replace(".", "_")
    output_dir = args.output_dir or Path(f"experiments/combined_seed{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine feature columns from clean features table
    features_df = load_parquet(PROCESSED_DIR / "features_all.parquet")
    feat_cols = get_feature_columns(features_df)
    log.info("Feature columns (%d): %s", len(feat_cols), feat_cols)

    # Load training data
    X_clean, y_clean = load_clean_train_data(feat_cols, tau_col)
    X_corrupt, y_corrupt = load_corrupted_train_data(feat_cols, tau_col)
    X_cal, y_cal = load_clean_cal_data(feat_cols, tau_col)

    # Combine
    X_train = np.vstack([X_clean, X_corrupt])
    y_train = np.concatenate([y_clean, y_corrupt])
    log.info(
        "Combined training: %d frames, %d positives (%.1f%%)",
        len(y_train), y_train.sum(), 100 * y_train.mean(),
    )

    monitor = XGBMonitor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.seed,
    )

    if X_cal is not None and y_cal is not None:
        monitor.fit(X_train, y_train, X_val=X_cal, y_val=y_cal)
    else:
        monitor.fit(X_train, y_train)

    importances = monitor.feature_importances(feat_cols)
    save_monitor(monitor, output_dir / "monitor.pkl")
    save_json(feat_cols, output_dir / "feature_columns.json")
    save_json(importances, output_dir / "feature_importances.json")

    run_meta = {
        "seed": args.seed,
        "tau_col": tau_col,
        "n_train": int(len(y_train)),
        "n_train_pos": int(y_train.sum()),
        "n_train_pos_clean": int(y_clean.sum()),
        "n_train_pos_corrupt": int(y_corrupt.sum()),
        "feat_cols": feat_cols,
    }
    save_json(run_meta, output_dir / "run_metadata.json")
    log.info("Combined monitor saved to %s", output_dir)
    log.info("Run scripts/07_calibrate_conformal.py next with hydra.run.dir=%s", output_dir)

    # Print top-10 feature importances
    top10 = sorted(importances.items(), key=lambda x: -x[1])[:10]
    log.info("Top-10 feature importances:")
    for feat, imp in top10:
        log.info("  %-35s %.4f", feat, imp)


if __name__ == "__main__":
    main()
