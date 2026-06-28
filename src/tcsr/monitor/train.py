"""Fit the monitor on train videos, save model + feature column order."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tcsr.monitor.classifiers import save_monitor
from tcsr.utils.io import save_json
from tcsr.utils.logging import get_logger

log = get_logger(__name__)

FEATURE_COLUMN_PREFIX = ("conf_", "shape_", "temp_", "qual_", "shift_")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(c.startswith(p) for p in FEATURE_COLUMN_PREFIX)]


def train_monitor(
    monitor,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    tau_col: str,
    output_dir: Path,
    val_split: str = "cal",
) -> dict:
    df = features_df.merge(labels_df[["frame_id", tau_col]], on="frame_id")
    feat_cols = get_feature_columns(df)

    train_df = df[df.get("split", "train") == "train"] if "split" in df.columns else df
    val_df = df[df.get("split", "") == val_split] if "split" in df.columns else None

    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df[tau_col].values

    X_val = val_df[feat_cols].values.astype(np.float32) if val_df is not None and len(val_df) else None
    y_val = val_df[tau_col].values if val_df is not None and len(val_df) else None

    n_pos = int(y_train.sum())
    if n_pos == 0:
        log.warning(
            "Training split has 0 positive examples (all IoU > τ). "
            "Monitor will predict from feature distributions only."
        )

    # GRUMonitor expects per-video sequences — group by video_id
    from tcsr.monitor.temporal_nn import GRUMonitor
    if isinstance(monitor, GRUMonitor):
        vid_col = "video_id" if "video_id" in train_df.columns else None
        if vid_col:
            X_seq = [g[feat_cols].values.astype(np.float32) for _, g in train_df.groupby(vid_col, sort=False)]
            y_seq = [g[tau_col].values for _, g in train_df.groupby(vid_col, sort=False)]
        else:
            X_seq, y_seq = [X_train], [y_train]
        monitor.fit(X_seq, y_seq)
    elif hasattr(monitor, "fit") and X_val is not None and "X_val" in monitor.fit.__code__.co_varnames:
        monitor.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        try:
            monitor.fit(X_train, y_train)
        except ValueError as exc:
            if "one class" in str(exc) or "out of bounds" in str(exc):
                log.warning("Classifier failed on single-class training data (%s); fitting on train+cal.", exc)
                combined = pd.concat([train_df, val_df]) if val_df is not None else train_df
                monitor.fit(
                    combined[feat_cols].values.astype(np.float32),
                    combined[tau_col].values,
                )
            else:
                raise

    importances = monitor.feature_importances(feat_cols)
    save_monitor(monitor, output_dir / "monitor.pkl")
    save_json(feat_cols, output_dir / "feature_columns.json")
    save_json(importances, output_dir / "feature_importances.json")
    log.info("Monitor saved to %s | train_n=%d", output_dir, len(X_train))
    return {"feat_cols": feat_cols, "train_n": len(X_train)}
