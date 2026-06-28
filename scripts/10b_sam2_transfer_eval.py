#!/usr/bin/env python
"""
Stage 10b: Cross-model transfer evaluation for SAM2.

Tests the core model-agnostic hypothesis: an XGBoost monitor trained entirely
on U-Net (ResNet34) failures can detect SAM2-Tiny failures without any SAM2
training data — i.e., the learned features (confidence, shape, temporal, quality)
are model-agnostic failure signals.

Three conditions evaluated on the clean test split (seqs 08-10):
  A. U-Net monitor (seed 0, combined OOF+corrupted) → SAM2 test features
       = zero-shot cross-model transfer
  B. Entropy baseline → SAM2 predictions (model-blind oracle baseline)
  C. SAM2-specific monitor (trained on SAM2 train-split features) → SAM2 test
       = in-distribution upper bound for SAM2

Also reports:
  - SAM2 vs U-Net failure rates (how much harder is SAM2 on this data?)
  - Feature-level distribution shift between U-Net and SAM2 predictions (mean ± std)

Prerequisite: run 10a first.

Output: results/metrics/sam2_transfer.json

Run from: tcsr-monitor/  with conda env tcsr-monitor active.
  python scripts/10b_sam2_transfer_eval.py
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
from tcsr.monitor.classifiers import XGBMonitor, load_monitor
from tcsr.utils.io import load_json, load_npz, save_json
from tcsr.utils.logging import get_logger
from tcsr.utils.seed import set_seed

log = get_logger(__name__)

PROCESSED_DIR = Path("data/processed/endovis2017")
SAM2_PRED_DIR = PROCESSED_DIR / "sam2_predictions"
RESULTS_DIR = Path("results/metrics")
MONITOR_PATH = Path("experiments/combined_seed0/monitor.pkl")
FEAT_JSON_PATH = Path("experiments/combined_seed0/feature_columns.json")

TAUS = {0.5: "failure_tau0_50", 0.75: "failure_tau0_75"}
FEATURE_PREFIXES = ("conf_", "shape_", "temp_", "qual_", "shift_")
_EPS = 1e-7


# ─── feature extraction ──────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]


def entropy_from_prob(prob_map: np.ndarray) -> float:
    p = np.clip(prob_map.astype(np.float32), _EPS, 1 - _EPS)
    return float((-(p * np.log(p) + (1 - p) * np.log(1 - p))).mean())


def extract_sam2_features(manifest: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Extract features from SAM2 predictions for all frames in manifest.

    Processes per video in frame order to compute temporal features correctly.
    """
    rows = []
    for video_id, vdf in tqdm(
        manifest.groupby("video_id"),
        desc="Extracting SAM2 features",
        total=manifest.video_id.nunique(),
    ):
        vdf = vdf.sort_values("frame_idx_in_video").reset_index(drop=True)
        prev_mask = None
        prev_area = None
        iou_history: list[float] = []

        for _, row in vdf.iterrows():
            npz_path = SAM2_PRED_DIR / row.frame_id / "probs.npz"
            try:
                npz = load_npz(npz_path)
                prob_map = npz["probs"].astype(np.float32)
            except Exception:
                prob_map = np.zeros((512, 512), dtype=np.float32)

            curr_mask = (prob_map >= 0.5).astype(np.uint8)
            curr_area = float(curr_mask.mean())

            feat: dict = {
                "frame_id": row.frame_id,
                "video_id": row.video_id,
                "split": row.split,
                "entropy_score": entropy_from_prob(prob_map),
            }
            feat.update(compute_confidence_features(prob_map))
            feat.update(compute_shape_features(curr_mask))
            feat.update(compute_temporal_features(
                curr_mask, prev_mask, curr_area, prev_area, iou_history[-5:]
            ))

            bgr = cv2.imread(row.frame_path)
            if bgr is not None:
                feat.update(compute_quality_features(bgr))

            if prev_mask is not None:
                iou_history.append(mask_iou(curr_mask.astype(bool), prev_mask.astype(bool)))

            prev_mask = curr_mask
            prev_area = curr_area
            rows.append(feat)

    df = pd.DataFrame(rows)
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df.fillna(0.0)


# ─── failure labels for SAM2 ─────────────────────────────────────────────────

def compute_sam2_labels(manifest: pd.DataFrame) -> pd.DataFrame:
    """Compute IoU(SAM2_pred, GT) per frame and derive failure labels at τ∈{0.5,0.75}."""
    rows = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="SAM2 labels"):
        mask_path = SAM2_PRED_DIR / row.frame_id / "mask.png"
        pred = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)

        if pred is None or gt is None:
            iou_val = 0.0
        else:
            if pred.shape != gt.shape:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            pred_b = pred > 0
            gt_b = gt > 0
            inter = (pred_b & gt_b).sum()
            union = (pred_b | gt_b).sum()
            iou_val = float(inter) / float(union) if union > 0 else 1.0

        entry = {"frame_id": row.frame_id, "sam2_iou": iou_val}
        for tau, tau_col in TAUS.items():
            entry[tau_col] = int(iou_val < tau)
        rows.append(entry)

    return pd.DataFrame(rows)


# ─── metrics ─────────────────────────────────────────────────────────────────

def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def bootstrap_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    auroc_b, auprc_b = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y[idx], s[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        auroc_b.append(roc_auc_score(yb, sb))
        auprc_b.append(average_precision_score(yb, sb))
    return {
        "auroc_ci_lo": float(np.percentile(auroc_b, 2.5)) if auroc_b else float("nan"),
        "auroc_ci_hi": float(np.percentile(auroc_b, 97.5)) if auroc_b else float("nan"),
        "auprc_ci_lo": float(np.percentile(auprc_b, 2.5)) if auprc_b else float("nan"),
        "auprc_ci_hi": float(np.percentile(auprc_b, 97.5)) if auprc_b else float("nan"),
    }


def eval_row(label: str, y: np.ndarray, s: np.ndarray) -> dict:
    auroc = safe_auroc(y, s)
    auprc = safe_auprc(y, s)
    ci = bootstrap_ci(y, s)
    return {
        "method": label,
        "n_total": int(len(y)),
        "n_failures": int(y.sum()),
        "failure_rate": float(y.mean()),
        "auroc": auroc,
        "auprc": auprc,
        **ci,
    }


# ─── SAM2-specific monitor (upper bound) ─────────────────────────────────────

def train_sam2_monitor(
    feat_df: pd.DataFrame, feat_cols: list[str], tau_col: str, seed: int = 0
) -> XGBMonitor:
    """Train XGBoost on SAM2 train-split features → upper-bound monitor."""
    train_df = feat_df[feat_df.split == "train"]
    cal_df = feat_df[feat_df.split == "cal"]

    X_tr = train_df[feat_cols].values.astype(np.float32)
    y_tr = train_df[tau_col].values.astype(int)

    X_cal = cal_df[feat_cols].values.astype(np.float32) if len(cal_df) > 0 else None
    y_cal = cal_df[tau_col].values.astype(int) if len(cal_df) > 0 else None

    log.info("SAM2 monitor train: %d frames, %d pos (%.1f%%)", len(y_tr), y_tr.sum(), 100*y_tr.mean())

    mon = XGBMonitor(n_estimators=400, max_depth=6, learning_rate=0.05, random_state=seed)
    mon.fit(X_tr, y_tr, X_val=X_cal, y_val=y_cal)
    return mon


# ─── U-Net feature distribution for comparison ───────────────────────────────

def feature_shift_summary(
    unet_feat: pd.DataFrame, sam2_feat: pd.DataFrame, feat_cols: list[str]
) -> dict:
    """Mean absolute difference between U-Net and SAM2 feature distributions on test split."""
    u = unet_feat[unet_feat.split == "test"][feat_cols]
    s = sam2_feat[sam2_feat.split == "test"][feat_cols]
    diff = {}
    for col in feat_cols:
        uc, sc = u[col].values, s[col].values
        diff[col] = {
            "unet_mean": float(uc.mean()),
            "sam2_mean": float(sc.mean()),
            "abs_diff": float(abs(uc.mean() - sc.mean())),
            "unet_std": float(uc.std()),
            "sam2_std": float(sc.std()),
        }
    return diff


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity-check SAM2 predictions exist
    n_sam2 = sum(1 for p in SAM2_PRED_DIR.rglob("probs.npz")) if SAM2_PRED_DIR.exists() else 0
    if n_sam2 == 0:
        raise FileNotFoundError(
            f"No SAM2 predictions found in {SAM2_PRED_DIR}. Run 10a first."
        )
    log.info("Found %d SAM2 prediction frames", n_sam2)

    # Load manifest + U-Net monitor
    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")
    unet_monitor = load_monitor(MONITOR_PATH)
    feat_cols = load_json(FEAT_JSON_PATH)
    log.info("Loaded U-Net monitor from %s", MONITOR_PATH)

    # Load U-Net clean features (for feature-shift comparison)
    unet_feat_df = pd.read_parquet(PROCESSED_DIR / "features_all.parquet")
    unet_feat_df = unet_feat_df.merge(
        manifest[["frame_id", "split"]], on="frame_id", how="left"
    )

    # Load U-Net failure labels (for failure-rate comparison)
    unet_labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")

    # ─── Extract SAM2 features ───────────────────────────────────────────────
    log.info("Extracting features from SAM2 predictions ...")
    sam2_feat_df = extract_sam2_features(manifest, feat_cols)

    # ─── Compute SAM2 failure labels ─────────────────────────────────────────
    log.info("Computing SAM2 failure labels ...")
    sam2_labels = compute_sam2_labels(manifest)
    sam2_feat_df = sam2_feat_df.merge(
        sam2_labels[["frame_id", "sam2_iou"] + list(TAUS.values())],
        on="frame_id", how="left"
    )

    # ─── Results container ───────────────────────────────────────────────────
    results: dict = {}

    # ─── Compare failure rates: SAM2 vs U-Net ────────────────────────────────
    test_mask = manifest[manifest.split == "test"]
    test_ids = set(test_mask.frame_id)
    sam2_test = sam2_feat_df[sam2_feat_df.frame_id.isin(test_ids)]
    unet_test_labels = unet_labels[unet_labels.frame_id.isin(test_ids)]

    results["model_comparison"] = {}
    for tau, tau_col in TAUS.items():
        unet_fr = float(unet_test_labels[tau_col].mean()) if tau_col in unet_test_labels.columns else float("nan")
        sam2_fr = float(sam2_test[tau_col].mean()) if tau_col in sam2_test.columns else float("nan")
        results["model_comparison"][f"tau_{tau:.2f}".replace(".", "_")] = {
            "unet_failure_rate": unet_fr,
            "sam2_failure_rate": sam2_fr,
            "sam2_harder_by": round(sam2_fr - unet_fr, 4) if not (np.isnan(unet_fr) or np.isnan(sam2_fr)) else None,
        }
        log.info("τ=%.2f — U-Net failure rate: %.3f | SAM2 failure rate: %.3f",
                 tau, unet_fr, sam2_fr)

    # ─── Feature shift ───────────────────────────────────────────────────────
    results["feature_shift"] = feature_shift_summary(unet_feat_df, sam2_feat_df, feat_cols)
    top_shift = sorted(
        results["feature_shift"].items(), key=lambda kv: -kv[1]["abs_diff"]
    )[:5]
    log.info("Top-5 feature shifts (U-Net → SAM2):")
    for feat, v in top_shift:
        log.info("  %-35s  unet=%.3f  sam2=%.3f  Δ=%.3f", feat, v["unet_mean"], v["sam2_mean"], v["abs_diff"])

    # ─── Per-τ evaluation ────────────────────────────────────────────────────
    for tau, tau_col in TAUS.items():
        if tau_col not in sam2_test.columns:
            log.warning("Missing %s in SAM2 features", tau_col)
            continue

        y_test = sam2_test[tau_col].fillna(0).values.astype(int)
        X_test = sam2_test[feat_cols].values.astype(np.float32)
        ent_test = sam2_test["entropy_score"].values

        tau_key = f"tau_{tau:.2f}".replace(".", "_")
        results[tau_key] = {}

        log.info("\n=== τ=%.2f  n_failures=%d / %d (%.1f%%) ===",
                 tau, y_test.sum(), len(y_test), 100 * y_test.mean())

        # A. U-Net monitor → SAM2 (zero-shot transfer)
        sam2_scores = unet_monitor.predict_proba(X_test)
        res_a = eval_row("UNet-monitor→SAM2 (transfer)", y_test, sam2_scores)
        results[tau_key]["unet_monitor_to_sam2_transfer"] = res_a
        log.info("  A. U-Net monitor → SAM2 (transfer): AUROC=%.3f [%.3f, %.3f]  AUPRC=%.3f",
                 res_a["auroc"], res_a["auroc_ci_lo"], res_a["auroc_ci_hi"], res_a["auprc"])

        # B. Entropy baseline
        res_b = eval_row("Entropy→SAM2", y_test, ent_test)
        results[tau_key]["entropy_baseline"] = res_b
        log.info("  B. Entropy baseline:                 AUROC=%.3f [%.3f, %.3f]  AUPRC=%.3f",
                 res_b["auroc"], res_b["auroc_ci_lo"], res_b["auroc_ci_hi"], res_b["auprc"])

        # C. SAM2-specific monitor (upper bound)
        sam2_monitor = train_sam2_monitor(sam2_feat_df, feat_cols, tau_col, seed=0)
        sam2_specific_scores = sam2_monitor.predict_proba(X_test)
        res_c = eval_row("SAM2-monitor→SAM2 (upper-bound)", y_test, sam2_specific_scores)
        results[tau_key]["sam2_monitor_upper_bound"] = res_c
        log.info("  C. SAM2-specific monitor (UB):       AUROC=%.3f [%.3f, %.3f]  AUPRC=%.3f",
                 res_c["auroc"], res_c["auroc_ci_lo"], res_c["auroc_ci_hi"], res_c["auprc"])

        # Transfer efficiency = transfer AUROC / upper-bound AUROC
        if not np.isnan(res_a["auroc"]) and not np.isnan(res_c["auroc"]) and res_c["auroc"] > 0:
            transfer_eff = res_a["auroc"] / res_c["auroc"]
            results[tau_key]["transfer_efficiency"] = round(transfer_eff, 4)
            log.info("  Transfer efficiency (A/C): %.3f", transfer_eff)

    # ─── Summary ─────────────────────────────────────────────────────────────
    log.info("\n=== SAM2 TRANSFER SUMMARY ===")
    for tau_key, r in results.items():
        if not tau_key.startswith("tau_"):
            continue
        tau_str = tau_key.replace("tau_", "τ=").replace("_", ".")
        a = r.get("unet_monitor_to_sam2_transfer", {})
        b = r.get("entropy_baseline", {})
        c = r.get("sam2_monitor_upper_bound", {})
        log.info(
            "%s | Transfer AUROC=%.3f  Entropy=%.3f  UB=%.3f  efficiency=%.3f",
            tau_str,
            a.get("auroc", float("nan")),
            b.get("auroc", float("nan")),
            c.get("auroc", float("nan")),
            r.get("transfer_efficiency", float("nan")),
        )

    out_path = RESULTS_DIR / "sam2_transfer.json"
    save_json(results, out_path)
    log.info("SAM2 transfer evaluation saved to %s", out_path)


if __name__ == "__main__":
    main()
