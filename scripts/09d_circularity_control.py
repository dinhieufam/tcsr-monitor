#!/usr/bin/env python
"""
Stage 9d: Circularity control.

Tests whether the monitor predicts failure, or merely detects that an image
was corrupted (regardless of whether the model actually failed).

Two experiments:
  1. Within corrupted-only frames: AUROC for model-correct vs model-failed.
     If near chance: monitor just detects corruption, not failure.
  2. False-alarm rate on corrupted-but-CORRECT frames (model still succeeded).
     A monitor that alarms on every corrupted frame regardless of outcome is
     useless as a safety monitor.

Uses cached corruption features from 09c if available, otherwise extracts.

Output: results/metrics/circularity_control.json
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.features.confidence import compute_confidence_features
from tcsr.features.quality import compute_quality_features
from tcsr.features.shape import compute_shape_features
from tcsr.features.temporal import compute_temporal_features, mask_iou
from tcsr.monitor.classifiers import load_monitor
from tcsr.utils.io import load_json, load_npz, load_parquet, save_json
from tcsr.utils.logging import get_logger
from tcsr.utils.seed import set_seed

log = get_logger(__name__)

PROCESSED_DIR = Path("data/processed/endovis2017")
RESULTS_DIR = Path("results/metrics")
CACHE_DIR = PROCESSED_DIR / "corruption_feature_cache"
CACHE_PATH = CACHE_DIR / "corrupted_test_features.parquet"

FEATURE_PREFIXES = ("conf_", "shape_", "temp_", "qual_", "shift_")
TAU_COLS = {0.5: "failure_tau0_50", 0.75: "failure_tau0_75"}
_EPS = 1e-7


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]


def entropy_score(prob_map: np.ndarray) -> float:
    p = np.clip(prob_map.astype(np.float32), _EPS, 1 - _EPS)
    return float((-(p * np.log(p) + (1 - p) * np.log(1 - p))).mean())


def extract_features_for_manifest(
    manifest: pd.DataFrame,
    feat_cols: list[str],
) -> pd.DataFrame:
    """Fallback extraction (used only if cache from 09c is missing)."""
    rows_out = []
    for (corruption, severity), grp in tqdm(
        manifest.groupby(["corruption_type", "severity"]),
        desc="Extracting corrupted test features",
    ):
        grp = grp.sort_values(["video_id", "frame_idx_in_video"]).reset_index(drop=True)
        pred_dir = Path(grp.pred_dir.iloc[0])

        prev_mask = None
        prev_area = None
        prev_vid = None
        iou_history: list[float] = []

        for _, row in grp.iterrows():
            if row.video_id != prev_vid:
                prev_mask, prev_area, iou_history, prev_vid = None, None, [], row.video_id

            npz_path = pred_dir / row.frame_id / "probs.npz"
            try:
                npz = load_npz(npz_path)
                prob_map = npz["probs"].astype(np.float32)
            except Exception:
                prob_map = np.zeros((512, 512), dtype=np.float32)

            curr_mask = (prob_map >= 0.5).astype(np.uint8)
            curr_area = float(curr_mask.mean())

            feat: dict = {
                "frame_id": row.frame_id,
                "corruption_type": corruption,
                "severity": int(severity),
                "entropy_score": entropy_score(prob_map),
            }
            feat.update(compute_confidence_features(prob_map))
            feat.update(compute_shape_features(curr_mask))
            feat.update(compute_temporal_features(curr_mask, prev_mask, curr_area, prev_area, iou_history[-5:]))
            try:
                bgr = cv2.imread(row.frame_path)
                if bgr is not None:
                    feat.update(compute_quality_features(bgr))
            except Exception:
                pass

            if prev_mask is not None:
                iou_history.append(mask_iou(curr_mask.astype(bool), prev_mask.astype(bool)))
            prev_mask, prev_area = curr_mask, curr_area
            rows_out.append(feat)

    result_df = pd.DataFrame(rows_out)
    for col in feat_cols:
        if col not in result_df.columns:
            result_df[col] = 0.0
    return result_df.fillna(0.0)


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def bootstrap_auroc_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    stats = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y[idx], s[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        stats.append(roc_auc_score(yb, sb))
    if not stats:
        return float("nan"), float("nan")
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main() -> None:
    set_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load combined monitor
    monitor_path = Path("experiments/combined_seed0/monitor.pkl")
    feat_json_path = Path("experiments/combined_seed0/feature_columns.json")
    if not monitor_path.exists():
        raise FileNotFoundError(f"Run 06b first. Missing: {monitor_path}")
    monitor = load_monitor(monitor_path)
    feat_cols = load_json(feat_json_path)
    log.info("Loaded combined monitor from %s", monitor_path)

    # Load conformal threshold
    conformal_path = Path("experiments/combined_seed0/conformal_threshold.json")
    if conformal_path.exists():
        conf = load_json(conformal_path)
        threshold = conf["threshold"]
    else:
        threshold = 0.5
    log.info("Using alarm threshold: %.4f", threshold)

    # Load or extract corrupted test features
    if CACHE_PATH.exists():
        log.info("Loading cached corrupted test features from %s", CACHE_PATH)
        feat_df = pd.read_parquet(CACHE_PATH)
    else:
        log.info("Cache not found; extracting features (run 09c first for speed)...")
        features_ref = load_parquet(PROCESSED_DIR / "features_all.parquet")
        feat_cols_ref = get_feature_columns(features_ref)
        manifest = pd.read_parquet(PROCESSED_DIR / "corrupted_manifest.parquet")
        feat_df = extract_features_for_manifest(manifest, feat_cols_ref)
        # Add labels
        labels = pd.read_parquet(PROCESSED_DIR / "corrupted_labels.parquet")
        feat_df = feat_df.merge(
            labels[["frame_id", "corruption_type", "severity", "failure_tau0_50", "failure_tau0_75"]],
            on=["frame_id", "corruption_type", "severity"],
            how="left",
        )

    # Ensure all feature columns present
    for col in feat_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    results = {}

    for tau, tau_col in TAU_COLS.items():
        if tau_col not in feat_df.columns:
            log.warning("tau_col %s not in cached features; skipping tau=%.2f", tau_col, tau)
            continue

        y_all = feat_df[tau_col].fillna(0).values.astype(int)
        X_all = feat_df[feat_cols].values.astype(np.float32)
        mon_scores = monitor.predict_proba(X_all)
        ent_scores = feat_df["entropy_score"].values if "entropy_score" in feat_df.columns else np.zeros(len(y_all))

        # ─── Experiment 1: correct-vs-failed within corrupted frames ───
        # All frames are corrupted; binary outcome is whether model failed
        n_corrupt_total = len(y_all)
        n_corrupt_failed = int(y_all.sum())
        n_corrupt_correct = n_corrupt_total - n_corrupt_failed

        mon_auroc = safe_auroc(y_all, mon_scores)
        ent_auroc = safe_auroc(y_all, ent_scores)
        mon_auprc = safe_auprc(y_all, mon_scores)
        ent_auprc = safe_auprc(y_all, ent_scores)
        ci_lo, ci_hi = bootstrap_auroc_ci(y_all, mon_scores)

        log.info(
            "τ=%.2f | Within-corrupted: n_total=%d n_failed=%d (%.1f%%)",
            tau, n_corrupt_total, n_corrupt_failed, 100 * y_all.mean(),
        )
        log.info(
            "  Monitor AUROC=%.3f [%.3f, %.3f] | Entropy AUROC=%.3f",
            mon_auroc, ci_lo, ci_hi, ent_auroc,
        )

        # ─── Experiment 2: FA rate on corrupted-but-correct frames ───
        correct_mask = y_all == 0
        mon_on_correct = mon_scores[correct_mask]
        alarms_on_correct = (mon_on_correct >= threshold).sum()
        fa_rate_on_correct = float(alarms_on_correct / max(correct_mask.sum(), 1))

        log.info(
            "  FA rate on corrupted-but-correct: %.3f (%d / %d alarmed)",
            fa_rate_on_correct, int(alarms_on_correct), int(correct_mask.sum()),
        )

        # ─── Per-condition breakdown ───
        per_condition: dict = {}
        for (corruption, severity), grp in feat_df.groupby(["corruption_type", "severity"]):
            y_c = grp[tau_col].fillna(0).values.astype(int)
            X_c = grp[feat_cols].values.astype(np.float32)
            s_c = monitor.predict_proba(X_c)
            e_c = grp["entropy_score"].values if "entropy_score" in grp.columns else np.zeros(len(y_c))

            correct_c = y_c == 0
            alarms_c = (s_c[correct_c] >= threshold).sum() if correct_c.sum() > 0 else 0

            per_condition[f"{corruption}_sev{severity}"] = {
                "severity": int(severity),
                "n_total": int(len(y_c)),
                "n_failed": int(y_c.sum()),
                "failure_rate": float(y_c.mean()),
                "monitor_auroc_within_corrupted": safe_auroc(y_c, s_c),
                "entropy_auroc_within_corrupted": safe_auroc(y_c, e_c),
                "n_correct_frames": int(correct_c.sum()),
                "fa_on_correct": int(alarms_c),
                "fa_rate_on_correct": float(alarms_c / max(correct_c.sum(), 1)),
            }

        # ─── Per-severity summary ───
        per_severity = {}
        for sev in range(1, 6):
            sub = [v for v in per_condition.values() if v["severity"] == sev]
            if not sub:
                continue
            per_severity[f"sev_{sev}"] = {
                "severity": sev,
                "monitor_auroc_mean": float(np.nanmean([v["monitor_auroc_within_corrupted"] for v in sub])),
                "entropy_auroc_mean": float(np.nanmean([v["entropy_auroc_within_corrupted"] for v in sub])),
                "fa_rate_mean": float(np.nanmean([v["fa_rate_on_correct"] for v in sub])),
            }

        results[f"tau_{tau:.2f}".replace(".", "_")] = {
            "tau": tau,
            "n_corrupted_total": n_corrupt_total,
            "n_corrupted_failed": n_corrupt_failed,
            "n_corrupted_correct": n_corrupt_correct,
            "failure_rate": float(y_all.mean()),
            "alarm_threshold": threshold,
            "within_corrupted_monitor_auroc": mon_auroc,
            "within_corrupted_monitor_auroc_ci_lo": ci_lo,
            "within_corrupted_monitor_auroc_ci_hi": ci_hi,
            "within_corrupted_entropy_auroc": ent_auroc,
            "within_corrupted_monitor_auprc": mon_auprc,
            "within_corrupted_entropy_auprc": ent_auprc,
            "fa_rate_on_correct": fa_rate_on_correct,
            "n_alarms_on_correct": int(alarms_on_correct),
            "per_condition": per_condition,
            "per_severity": per_severity,
        }

    # Overall interpretation note
    log.info("\n=== CIRCULARITY CONTROL SUMMARY ===")
    for tau_key, r in results.items():
        log.info("τ=%s: within-corrupted AUROC=%.3f [%.3f,%.3f] | FA-on-correct=%.3f",
                 r["tau"], r["within_corrupted_monitor_auroc"],
                 r["within_corrupted_monitor_auroc_ci_lo"],
                 r["within_corrupted_monitor_auroc_ci_hi"],
                 r["fa_rate_on_correct"])
        log.info("  Entropy within-corrupted AUROC=%.3f", r["within_corrupted_entropy_auroc"])
        if r["within_corrupted_monitor_auroc"] > 0.55:
            log.info("  ✓ Monitor discriminates correct-vs-failed within corrupted frames (not just detecting corruption)")
        else:
            log.info("  ✗ Monitor near chance within corrupted frames — likely detecting corruption, not failure")

    out_path = RESULTS_DIR / "circularity_control.json"
    save_json(results, out_path)
    log.info("Circularity control saved to %s", out_path)


if __name__ == "__main__":
    main()
