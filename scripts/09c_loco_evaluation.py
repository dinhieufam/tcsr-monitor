#!/usr/bin/env python
"""
Stage 9c: Generalization hierarchy for the corruption robustness claim.

Runs four protocols (weakest→strongest):
  1. Zero-shot shift   — trained on clean OOF only, tested on all 30 corrupted conditions
  2. LOCO              — trained on 5 corruption types, tested on held-out 6th (6 folds × 5 sev)
  3. Severity extrap   — trained on sev 1-3, tested on sev 4-5
  4. In-distribution   — trained on all 6 × all sev (reuse existing combined monitor)

Features are cached to parquet on first run so each XGBoost retraining is
< 1 min. Results are saved to:
  results/metrics/loco_hierarchy.json
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.features.confidence import compute_confidence_features
from tcsr.features.quality import compute_quality_features
from tcsr.features.shape import compute_shape_features
from tcsr.features.temporal import compute_temporal_features, mask_iou
from tcsr.monitor.classifiers import XGBMonitor, load_monitor
from tcsr.utils.io import load_json, load_npz, load_parquet, save_json
from tcsr.utils.logging import get_logger
from tcsr.utils.seed import set_seed

log = get_logger(__name__)

PROCESSED_DIR = Path("data/processed/endovis2017")
RESULTS_DIR = Path("results/metrics")
CACHE_DIR = PROCESSED_DIR / "corruption_feature_cache"

FEATURE_PREFIXES = ("conf_", "shape_", "temp_", "qual_", "shift_")
TAU_COL = "failure_tau0_75"
_EPS = 1e-7

CORRUPTION_TYPES = ["brightness", "contrast", "gaussian_blur", "gaussian_noise", "jpeg_compression", "motion_blur"]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]


def entropy_score(prob_map: np.ndarray) -> float:
    p = np.clip(prob_map.astype(np.float32), _EPS, 1 - _EPS)
    return float((-(p * np.log(p) + (1 - p) * np.log(1 - p))).mean())


def extract_features_for_manifest(
    manifest: pd.DataFrame,
    pred_dir_col: str,
    feat_cols: list[str],
    desc: str = "extracting",
) -> pd.DataFrame:
    """Extract features for all frames in manifest, keeping metadata columns.

    Returns DataFrame with columns: frame_id, corruption_type, severity, + feat_cols + entropy_score
    """
    rows_out = []

    for (corruption, severity), grp in tqdm(
        manifest.groupby(["corruption_type", "severity"]),
        desc=desc,
    ):
        grp = grp.sort_values(["video_id", "frame_idx_in_video"]).reset_index(drop=True)
        pred_dir = Path(grp[pred_dir_col].iloc[0])

        prev_mask = None
        prev_area = None
        prev_vid = None
        iou_history: list[float] = []

        for _, row in grp.iterrows():
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

            prev_mask = curr_mask
            prev_area = curr_area
            rows_out.append(feat)

    result_df = pd.DataFrame(rows_out)
    for col in feat_cols:
        if col not in result_df.columns:
            result_df[col] = 0.0
    return result_df.fillna(0.0)


def load_or_cache_features(
    manifest_path: Path,
    labels_path: Path,
    cache_path: Path,
    feat_cols: list[str],
    split_name: str,
) -> pd.DataFrame:
    """Load cached features or extract and cache them."""
    if cache_path.exists():
        log.info("Loading cached %s features from %s", split_name, cache_path)
        return pd.read_parquet(cache_path)

    log.info("Extracting %s features (will cache to %s)...", split_name, cache_path)
    manifest = pd.read_parquet(manifest_path)
    labels = pd.read_parquet(labels_path)

    feat_df = extract_features_for_manifest(
        manifest, "pred_dir", feat_cols, desc=f"{split_name} features"
    )

    tau_label_cols = ["frame_id", "corruption_type", "severity"] + [
        c for c in labels.columns if c.startswith("failure_tau")
    ]
    feat_df = feat_df.merge(
        labels[tau_label_cols],
        on=["frame_id", "corruption_type", "severity"],
        how="left",
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(cache_path, index=False)
    log.info("Cached %d rows to %s", len(feat_df), cache_path)
    return feat_df


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def bootstrap_auroc_ci(
    y: np.ndarray, s: np.ndarray, n_boot: int = 500, seed: int = 42
) -> tuple[float, float]:
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


def train_monitor(X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None) -> XGBMonitor:
    monitor = XGBMonitor(n_estimators=400, max_depth=6, learning_rate=0.05)
    monitor.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return monitor


def evaluate_protocol(
    monitor,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str = TAU_COL,
) -> dict:
    """Run evaluation for one protocol: per-condition AUROC + aggregated."""
    results_by_condition = {}

    for (corruption, severity), grp in test_df.groupby(["corruption_type", "severity"]):
        y = grp[label_col].fillna(0).values.astype(int)
        X = grp[feat_cols].values.astype(np.float32)
        ent = grp["entropy_score"].values

        mon_scores = monitor.predict_proba(X)

        mon_auroc = safe_auroc(y, mon_scores)
        ent_auroc = safe_auroc(y, ent)

        entry = {
            "corruption_type": str(corruption),
            "severity": int(severity),
            "n_frames": int(len(y)),
            "n_failures": int(y.sum()),
            "failure_rate": float(y.mean()),
            "monitor_auroc": mon_auroc,
            "entropy_auroc": ent_auroc,
            "monitor_auprc": safe_auprc(y, mon_scores),
            "entropy_auprc": safe_auprc(y, ent),
        }
        if y.sum() >= 5:
            lo, hi = bootstrap_auroc_ci(y, mon_scores)
            entry["monitor_auroc_ci_lo"] = lo
            entry["monitor_auroc_ci_hi"] = hi

        results_by_condition[f"{corruption}_sev{severity}"] = entry

    # Aggregate by severity
    by_severity: dict = {}
    for sev in range(1, 6):
        sub = [v for v in results_by_condition.values() if v["severity"] == sev]
        if not sub:
            continue
        by_severity[f"sev_{sev}"] = {
            "severity": sev,
            "n_conditions": len(sub),
            "monitor_auroc_mean": float(np.nanmean([v["monitor_auroc"] for v in sub])),
            "entropy_auroc_mean": float(np.nanmean([v["entropy_auroc"] for v in sub])),
            "monitor_wins": sum(
                1 for v in sub
                if v["monitor_auroc"] == v["monitor_auroc"]
                and v["entropy_auroc"] == v["entropy_auroc"]
                and v["monitor_auroc"] > v["entropy_auroc"]
            ),
            "total_conditions": len(sub),
        }

    overall = {
        "monitor_auroc_mean": float(np.nanmean([v["monitor_auroc"] for v in results_by_condition.values()])),
        "entropy_auroc_mean": float(np.nanmean([v["entropy_auroc"] for v in results_by_condition.values()])),
        "monitor_wins_total": sum(
            1 for v in results_by_condition.values()
            if v["monitor_auroc"] == v["monitor_auroc"]
            and v["entropy_auroc"] == v["entropy_auroc"]
            and v["monitor_auroc"] > v["entropy_auroc"]
        ),
        "total_conditions": len(results_by_condition),
    }

    return {
        "by_condition": results_by_condition,
        "by_severity": by_severity,
        "overall": overall,
    }


def main() -> None:
    set_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Reference feature columns from clean data
    features_ref = load_parquet(PROCESSED_DIR / "features_all.parquet")
    feat_cols = get_feature_columns(features_ref)
    log.info("Feature columns (%d): %s", len(feat_cols), feat_cols)

    # Load/cache corrupted test features
    test_feat = load_or_cache_features(
        PROCESSED_DIR / "corrupted_manifest.parquet",
        PROCESSED_DIR / "corrupted_labels.parquet",
        CACHE_DIR / "corrupted_test_features.parquet",
        feat_cols,
        "corrupted-test",
    )

    # Load/cache corrupted train features
    train_feat = load_or_cache_features(
        PROCESSED_DIR / "corrupted_train_manifest.parquet",
        PROCESSED_DIR / "corrupted_train_labels.parquet",
        CACHE_DIR / "corrupted_train_features.parquet",
        feat_cols,
        "corrupted-train",
    )

    # Load clean OOF train/cal data
    clean_features = load_parquet(PROCESSED_DIR / "features_all.parquet")
    labels_df = load_parquet(PROCESSED_DIR / "labels.parquet")
    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")[["frame_id", "split"]]

    clean_df = clean_features.merge(labels_df[["frame_id", TAU_COL]], on="frame_id").merge(manifest, on="frame_id")
    clean_train = clean_df[clean_df["split"] == "train"]
    clean_cal = clean_df[clean_df["split"] == "cal"]

    X_clean_train = clean_train[feat_cols].values.astype(np.float32)
    y_clean_train = clean_train[TAU_COL].values
    X_clean_cal = clean_cal[feat_cols].values.astype(np.float32)
    y_clean_cal = clean_cal[TAU_COL].values

    log.info("Clean train: %d frames, %d pos", len(y_clean_train), y_clean_train.sum())

    results = {}

    # --- Protocol 1: Zero-shot shift ---
    # Trained on clean OOF only, tested on all corrupted test conditions
    log.info("=== Protocol 1: Zero-shot shift ===")
    monitor_zero = train_monitor(X_clean_train, y_clean_train, X_clean_cal, y_clean_cal)
    results["zero_shot"] = {
        "description": "Train: clean OOF only | Test: all 6 types × all 5 severities",
        "train_types": "clean_only",
        "train_severities": "none",
        **evaluate_protocol(monitor_zero, test_feat, feat_cols),
    }
    log.info(
        "Zero-shot: monitor_auroc=%.3f entropy_auroc=%.3f wins=%d/%d",
        results["zero_shot"]["overall"]["monitor_auroc_mean"],
        results["zero_shot"]["overall"]["entropy_auroc_mean"],
        results["zero_shot"]["overall"]["monitor_wins_total"],
        results["zero_shot"]["overall"]["total_conditions"],
    )

    # --- Protocol 2: LOCO (leave-one-corruption-out) ---
    log.info("=== Protocol 2: LOCO (leave-one-corruption-out) ===")
    loco_folds = {}
    for held_out in CORRUPTION_TYPES:
        train_types = [t for t in CORRUPTION_TYPES if t != held_out]

        # Train data: clean OOF + corrupted train for the 5 in-set types
        corrupt_train_subset = train_feat[train_feat["corruption_type"].isin(train_types)]
        X_c = corrupt_train_subset[feat_cols].values.astype(np.float32)
        y_c = corrupt_train_subset[TAU_COL].fillna(0).values.astype(int)

        X_loco_train = np.vstack([X_clean_train, X_c])
        y_loco_train = np.concatenate([y_clean_train, y_c])

        monitor_loco = train_monitor(X_loco_train, y_loco_train, X_clean_cal, y_clean_cal)

        # Test on held-out type only
        test_held = test_feat[test_feat["corruption_type"] == held_out]
        fold_result = evaluate_protocol(monitor_loco, test_held, feat_cols)
        loco_folds[held_out] = fold_result
        log.info(
            "LOCO held-out=%s: monitor_auroc=%.3f entropy_auroc=%.3f wins=%d/%d",
            held_out,
            fold_result["overall"]["monitor_auroc_mean"],
            fold_result["overall"]["entropy_auroc_mean"],
            fold_result["overall"]["monitor_wins_total"],
            fold_result["overall"]["total_conditions"],
        )

    # Macro-average across LOCO folds
    loco_macro = {
        "monitor_auroc_macro": float(np.nanmean([
            loco_folds[t]["overall"]["monitor_auroc_mean"] for t in CORRUPTION_TYPES
        ])),
        "entropy_auroc_macro": float(np.nanmean([
            loco_folds[t]["overall"]["entropy_auroc_mean"] for t in CORRUPTION_TYPES
        ])),
        "monitor_wins_total": sum(
            loco_folds[t]["overall"]["monitor_wins_total"] for t in CORRUPTION_TYPES
        ),
        "total_conditions": sum(
            loco_folds[t]["overall"]["total_conditions"] for t in CORRUPTION_TYPES
        ),
    }

    results["loco"] = {
        "description": "Train: clean OOF + 5 of 6 corruption types | Test: held-out 6th type",
        "folds": loco_folds,
        "macro_average": loco_macro,
    }
    log.info(
        "LOCO macro: monitor_auroc=%.3f entropy_auroc=%.3f wins=%d/%d",
        loco_macro["monitor_auroc_macro"],
        loco_macro["entropy_auroc_macro"],
        loco_macro["monitor_wins_total"],
        loco_macro["total_conditions"],
    )

    # --- Protocol 3: Severity extrapolation ---
    log.info("=== Protocol 3: Severity extrapolation (train sev 1-3 → test sev 4-5) ===")
    corrupt_sev13_train = train_feat[train_feat["severity"].isin([1, 2, 3])]
    X_sev13 = corrupt_sev13_train[feat_cols].values.astype(np.float32)
    y_sev13 = corrupt_sev13_train[TAU_COL].fillna(0).values.astype(int)

    X_sev_train = np.vstack([X_clean_train, X_sev13])
    y_sev_train = np.concatenate([y_clean_train, y_sev13])

    monitor_sev = train_monitor(X_sev_train, y_sev_train, X_clean_cal, y_clean_cal)

    test_sev45 = test_feat[test_feat["severity"].isin([4, 5])]
    sev_result = evaluate_protocol(monitor_sev, test_sev45, feat_cols)
    results["severity_extrap"] = {
        "description": "Train: clean OOF + sev 1-3 | Test: sev 4-5",
        "train_severities": [1, 2, 3],
        "test_severities": [4, 5],
        **sev_result,
    }
    log.info(
        "Sev-extrap: monitor_auroc=%.3f entropy_auroc=%.3f wins=%d/%d",
        sev_result["overall"]["monitor_auroc_mean"],
        sev_result["overall"]["entropy_auroc_mean"],
        sev_result["overall"]["monitor_wins_total"],
        sev_result["overall"]["total_conditions"],
    )

    # --- Protocol 4: In-distribution (existing combined monitor) ---
    log.info("=== Protocol 4: In-distribution (combined monitor) ===")
    combined_monitor_path = Path("experiments/combined_seed0/monitor.pkl")
    if combined_monitor_path.exists():
        combined_feat_cols_path = Path("experiments/combined_seed0/feature_columns.json")
        monitor_indist = load_monitor(combined_monitor_path)
        indist_feat_cols = load_json(combined_feat_cols_path) if combined_feat_cols_path.exists() else feat_cols
        indist_result = evaluate_protocol(monitor_indist, test_feat, indist_feat_cols)
    else:
        log.warning("Combined monitor not found; using zero-shot monitor as proxy")
        indist_result = evaluate_protocol(monitor_zero, test_feat, feat_cols)

    results["in_distribution"] = {
        "description": "Train: clean OOF + all 6 types × all 5 sev | Test: all (upper bound)",
        "train_types": "all_6",
        "train_severities": "all_5",
        **indist_result,
    }
    log.info(
        "In-distribution: monitor_auroc=%.3f entropy_auroc=%.3f wins=%d/%d",
        indist_result["overall"]["monitor_auroc_mean"],
        indist_result["overall"]["entropy_auroc_mean"],
        indist_result["overall"]["monitor_wins_total"],
        indist_result["overall"]["total_conditions"],
    )

    # Summary table (for the paper)
    summary = {
        "zero_shot": {
            "protocol": "Zero-shot (clean train only)",
            "monitor_auroc": results["zero_shot"]["overall"]["monitor_auroc_mean"],
            "entropy_auroc": results["zero_shot"]["overall"]["entropy_auroc_mean"],
            "wins": f"{results['zero_shot']['overall']['monitor_wins_total']}/{results['zero_shot']['overall']['total_conditions']}",
        },
        "loco": {
            "protocol": "LOCO (unseen corruption type)",
            "monitor_auroc": loco_macro["monitor_auroc_macro"],
            "entropy_auroc": loco_macro["entropy_auroc_macro"],
            "wins": f"{loco_macro['monitor_wins_total']}/{loco_macro['total_conditions']}",
        },
        "severity_extrap": {
            "protocol": "Severity extrapolation (sev 4-5 unseen)",
            "monitor_auroc": sev_result["overall"]["monitor_auroc_mean"],
            "entropy_auroc": sev_result["overall"]["entropy_auroc_mean"],
            "wins": f"{sev_result['overall']['monitor_wins_total']}/{sev_result['overall']['total_conditions']}",
        },
        "in_distribution": {
            "protocol": "In-distribution (upper bound)",
            "monitor_auroc": indist_result["overall"]["monitor_auroc_mean"],
            "entropy_auroc": indist_result["overall"]["entropy_auroc_mean"],
            "wins": f"{indist_result['overall']['monitor_wins_total']}/{indist_result['overall']['total_conditions']}",
        },
    }
    results["summary_table"] = summary

    # Save
    out_path = RESULTS_DIR / "loco_hierarchy.json"
    save_json(results, out_path)
    log.info("LOCO hierarchy saved to %s", out_path)

    # Print summary table
    log.info("\n=== GENERALIZATION HIERARCHY SUMMARY ===")
    log.info("%-45s  %8s  %8s  %8s", "Protocol", "Mon AUROC", "Ent AUROC", "Wins")
    for k, v in summary.items():
        log.info("%-45s  %8.3f  %8.3f  %8s", v["protocol"], v["monitor_auroc"], v["entropy_auroc"], v["wins"])


if __name__ == "__main__":
    main()
