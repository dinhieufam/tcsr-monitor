#!/usr/bin/env python
"""
Stage 9b: Evaluate monitor on corrupted predictions (robustness sweep).

For each (corruption_type, severity) condition:
  - Extracts features from corrupted predictions
  - Applies trained OOF monitor
  - Computes baseline scores (entropy from corrupted probs)
  - Reports AUROC for monitor and entropy per condition

Aggregates results by severity (mean over corruption types) and saves:
  results/metrics/corruption_sweep.json
  results/metrics/corruption_detail.json   (per-type breakdown)
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.features.confidence import compute_confidence_features
from tcsr.features.shape import compute_shape_features
from tcsr.features.temporal import compute_temporal_features
from tcsr.monitor.classifiers import load_monitor
from tcsr.utils.io import load_json, load_npz
from tcsr.utils.logging import get_logger

log = get_logger(__name__)

PROCESSED_DIR = Path("data/processed/endovis2017")
CORRUPTED_MANIFEST_PATH = PROCESSED_DIR / "corrupted_manifest.parquet"
CORRUPTED_LABELS_PATH = PROCESSED_DIR / "corrupted_labels.parquet"
RESULTS_DIR = Path("results/metrics")
TAU_COL = "failure_tau0_50"
_EPS = 1e-7


def entropy_score(prob_map: np.ndarray) -> float:
    p = np.clip(prob_map.astype(np.float32), _EPS, 1 - _EPS)
    return float((-(p * np.log(p) + (1 - p) * np.log(1 - p))).mean())


def extract_features_for_condition(
    manifest_rows: pd.DataFrame,
    pred_dir: Path,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the feature columns that the monitor was trained on.
    Returns (X, entropy_scores) arrays of shape (N, n_feats) and (N,).
    """
    rows_feat = []
    entropy_scores = []

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
            # Missing prediction — skip
            prob_map = np.zeros((512, 512), dtype=np.float32)

        curr_mask = (prob_map >= 0.5).astype(np.uint8)
        curr_area = float(curr_mask.mean())

        feat: dict = {"frame_id": row.frame_id}
        feat.update(compute_confidence_features(prob_map))
        feat.update(compute_shape_features(curr_mask))
        feat.update(compute_temporal_features(curr_mask, prev_mask, curr_area, prev_area, iou_history[-5:]))

        # Quality features from corrupted frame (stored to disk by 03c)
        try:
            bgr = cv2.imread(row.frame_path)
            if bgr is not None:
                from tcsr.features.quality import compute_quality_features
                feat.update(compute_quality_features(bgr))
        except Exception:
            pass

        if prev_mask is not None:
            from tcsr.features.temporal import mask_iou
            iou_history.append(mask_iou(curr_mask.astype(bool), prev_mask.astype(bool)))

        prev_mask = curr_mask
        prev_area = curr_area
        entropy_scores.append(entropy_score(prob_map))
        rows_feat.append(feat)

    feat_df = pd.DataFrame(rows_feat)
    # Fill missing columns with 0
    for col in feat_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    X = feat_df[feat_cols].fillna(0.0).values.astype(np.float32)
    return X, np.array(entropy_scores, dtype=np.float32)


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    metric: str = "auroc",
) -> dict:
    """Bootstrap 95% CI for AUROC or AUPRC."""
    rng = np.random.default_rng(seed=42)
    stats = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y[idx], scores[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        if metric == "auroc":
            stats.append(roc_auc_score(yb, sb))
        else:
            stats.append(average_precision_score(yb, sb))
    if not stats:
        return {"ci_lo": float("nan"), "ci_hi": float("nan")}
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return {"ci_lo": lo, "ci_hi": hi}


def main() -> None:
    if not CORRUPTED_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Run 03c first. Missing: {CORRUPTED_MANIFEST_PATH}")
    if not CORRUPTED_LABELS_PATH.exists():
        raise FileNotFoundError(f"Run 04b first. Missing: {CORRUPTED_LABELS_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load monitor: prefer combined (trained on OOF + corrupted data), then tau=0.75 OOF, then tau=0.5
    monitor = None
    feat_cols = None
    for seed in [0, 1, 2]:
        for prefix in ["combined_seed", "oof_tau75_seed", "oof_seed"]:
            monitor_pkl = Path(f"experiments/{prefix}{seed}/monitor.pkl")
            feat_json = Path(f"experiments/{prefix}{seed}/feature_columns.json")
            if monitor_pkl.exists() and feat_json.exists():
                monitor = load_monitor(monitor_pkl)
                feat_cols = load_json(feat_json)
                log.info("Loaded monitor from experiments/%s%d", prefix, seed)
                break
        if monitor is not None:
            break

    if monitor is None:
        raise FileNotFoundError(
            "No OOF monitor found. Run 06_train_monitor.py with hydra.run.dir=experiments/oof_seed{0,1,2} first."
        )

    c_manifest = pd.read_parquet(CORRUPTED_MANIFEST_PATH)
    c_labels = pd.read_parquet(CORRUPTED_LABELS_PATH)

    # Merge labels into manifest
    c_manifest = c_manifest.merge(
        c_labels[["frame_id", "corruption_type", "severity", TAU_COL]],
        on=["frame_id", "corruption_type", "severity"],
        how="left",
    )

    conditions = c_manifest.groupby(["corruption_type", "severity"])
    detail: dict = {}

    for (corruption, severity), grp in tqdm(conditions, desc="Evaluating conditions"):
        grp = grp.sort_values(["video_id", "frame_idx_in_video"]).reset_index(drop=True)
        pred_dir = Path(grp.pred_dir.iloc[0])

        y = grp[TAU_COL].values.astype(int)
        X, ent_scores = extract_features_for_condition(grp, pred_dir, feat_cols)

        monitor_scores = monitor.predict_proba(X)

        key = f"{corruption}_sev{severity}"
        mon_auroc = safe_auroc(y, monitor_scores)
        ent_auroc = safe_auroc(y, ent_scores)
        mon_auprc = safe_auprc(y, monitor_scores)
        ent_auprc = safe_auprc(y, ent_scores)

        entry: dict = {
            "corruption_type": str(corruption),
            "severity": int(severity),
            "n_frames": int(len(y)),
            "failure_rate": float(y.mean()),
            "monitor_auroc": mon_auroc,
            "monitor_auprc": mon_auprc,
            "entropy_auroc": ent_auroc,
            "entropy_auprc": ent_auprc,
        }

        # Bootstrap CIs (only when enough positives)
        if int(y.sum()) >= 5:
            ci_m = bootstrap_ci(y, monitor_scores, n_boot=500, metric="auroc")
            ci_e = bootstrap_ci(y, ent_scores, n_boot=500, metric="auroc")
            entry["monitor_auroc_ci_lo"] = ci_m["ci_lo"]
            entry["monitor_auroc_ci_hi"] = ci_m["ci_hi"]
            entry["entropy_auroc_ci_lo"] = ci_e["ci_lo"]
            entry["entropy_auroc_ci_hi"] = ci_e["ci_hi"]

        detail[key] = entry
        log.info(
            "%s sev%d: fail=%.2f monitor_auroc=%.3f entropy_auroc=%.3f",
            corruption, severity,
            entry["failure_rate"], mon_auroc, ent_auroc,
        )

    # Save detailed results
    with open(RESULTS_DIR / "corruption_detail.json", "w") as f:
        json.dump(detail, f, indent=2)

    # Aggregate by severity (mean over corruption types)
    sweep: dict = {}
    for sev in sorted({int(d["severity"]) for d in detail.values()}):
        sub = [d for d in detail.values() if int(d["severity"]) == sev]
        sweep[f"sev_{sev}"] = {
            "severity": sev,
            "n_conditions": len(sub),
            "failure_rate": float(np.nanmean([d["failure_rate"] for d in sub])),
            "monitor_auroc": float(np.nanmean([d["monitor_auroc"] for d in sub])),
            "entropy_auroc": float(np.nanmean([d["entropy_auroc"] for d in sub])),
            "monitor_auprc": float(np.nanmean([d["monitor_auprc"] for d in sub])),
            "entropy_auprc": float(np.nanmean([d["entropy_auprc"] for d in sub])),
        }

    with open(RESULTS_DIR / "corruption_sweep.json", "w") as f:
        json.dump(sweep, f, indent=2)

    log.info("Corruption sweep saved → %s", RESULTS_DIR / "corruption_sweep.json")
    for k, v in sweep.items():
        log.info(
            "  %s: fail=%.2f | monitor=%.3f | entropy=%.3f",
            k, v["failure_rate"], v["monitor_auroc"], v["entropy_auroc"],
        )

    # Per-corruption-family summary (mean over severities, excluding NaN)
    family_map: dict[str, list] = {}
    for d in detail.values():
        family_map.setdefault(d["corruption_type"], []).append(d)

    family_summary: dict = {}
    for corruption_type, entries in family_map.items():
        def nanmean(vals):
            v = [x for x in vals if x == x]  # filter NaN
            return float(np.mean(v)) if v else float("nan")

        family_summary[corruption_type] = {
            "n_conditions": len(entries),
            "mean_failure_rate": nanmean([e["failure_rate"] for e in entries]),
            "monitor_auroc_mean": nanmean([e["monitor_auroc"] for e in entries]),
            "monitor_auprc_mean": nanmean([e["monitor_auprc"] for e in entries]),
            "entropy_auroc_mean": nanmean([e["entropy_auroc"] for e in entries]),
            "entropy_auprc_mean": nanmean([e["entropy_auprc"] for e in entries]),
            "monitor_wins_auroc": sum(
                1 for e in entries
                if e["monitor_auroc"] == e["monitor_auroc"]  # not NaN
                and e["entropy_auroc"] == e["entropy_auroc"]
                and e["monitor_auroc"] > e["entropy_auroc"]
            ),
        }

    with open(RESULTS_DIR / "corruption_family_summary.json", "w") as f:
        json.dump(family_summary, f, indent=2)

    log.info("Per-family summary:")
    for fam, v in family_summary.items():
        log.info(
            "  %-20s monitor_auroc=%.3f  entropy_auroc=%.3f  wins=%d/%d",
            fam, v["monitor_auroc_mean"], v["entropy_auroc_mean"],
            v["monitor_wins_auroc"], v["n_conditions"],
        )


if __name__ == "__main__":
    main()
