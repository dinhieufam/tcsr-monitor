#!/usr/bin/env python
"""Audit baseline risk scores and write corrected baseline artifacts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.baselines.runners import (  # noqa: E402
    _entropy_score,
    _max_softmax_score,
    _temporal_heuristic_score,
    _tta_variance_score,
)
from tcsr.utils.io import load_npz, save_parquet  # noqa: E402


PROCESSED_DIR = Path("data/processed/endovis2017")
RESULTS_DIR = Path("results/metrics/baseline_audit")
EPS = 1e-12


def _safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y) or np.nanstd(scores) <= EPS:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _safe_auprc(y: np.ndarray, scores: np.ndarray) -> float:
    if y.sum() == 0 or np.nanstd(scores) <= EPS:
        return float("nan")
    return float(average_precision_score(y, scores))


def _ci(y: np.ndarray, scores: np.ndarray, metric: str, n_boot: int = 1000) -> dict[str, float | int]:
    rng = np.random.default_rng(0)
    values: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yb = y[idx]
        sb = scores[idx]
        if yb.sum() == 0 or yb.sum() == len(yb) or np.nanstd(sb) <= EPS:
            continue
        if metric == "auroc":
            values.append(float(roc_auc_score(yb, sb)))
        else:
            values.append(float(average_precision_score(yb, sb)))
    if len(values) < 10:
        return {"lo": float("nan"), "hi": float("nan"), "n_bootstraps": len(values)}
    return {
        "lo": float(np.percentile(values, 2.5)),
        "hi": float(np.percentile(values, 97.5)),
        "n_bootstraps": len(values),
    }


def _score_rows(manifest: pd.DataFrame, pred_dir_col: str | None = None) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    prev_masks: dict[str, np.ndarray | None] = {}
    missing_probs = 0
    tta_present = 0

    iterator = manifest.sort_values(["video_id", "frame_idx_in_video"]).iterrows()
    for _, row in tqdm(iterator, total=len(manifest), desc="Scoring baselines", disable=True):
        video_id = str(row.video_id)
        prev_mask = prev_masks.get(video_id)
        pred_dir = Path(row[pred_dir_col]) if pred_dir_col else PROCESSED_DIR / "predictions"
        npz_path = pred_dir / row.frame_id / "probs.npz"

        try:
            npz = load_npz(npz_path)
            prob_map = npz["probs"].astype(np.float32)
        except Exception:
            missing_probs += 1
            continue

        curr_mask = (prob_map >= 0.5).astype(np.uint8)
        entry = {
            "frame_id": row.frame_id,
            "bl_max_softmax": _max_softmax_score(prob_map),
            "bl_entropy": _entropy_score(prob_map),
            "bl_temporal_heuristic": _temporal_heuristic_score(curr_mask, prev_mask),
        }
        if pred_dir_col:
            entry["corruption_type"] = row.corruption_type
            entry["severity"] = int(row.severity)

        tta = _tta_variance_score(npz)
        if tta is not None:
            entry["bl_tta_variance"] = tta
            tta_present += 1

        rows.append(entry)
        prev_masks[video_id] = curr_mask

    return pd.DataFrame(rows), {"missing_probs": missing_probs, "tta_present": tta_present}


def _status_for_column(df: pd.DataFrame, col: str, required_cache: str) -> dict:
    if col not in df.columns:
        return {
            "status": "removed",
            "reason": f"required cache unavailable: {required_cache}",
            "n_unique": 0,
            "std": float("nan"),
        }
    values = df[col].astype(float).values
    std = float(np.nanstd(values))
    n_unique = int(pd.Series(values).nunique(dropna=True))
    if n_unique <= 1 or std <= EPS:
        return {
            "status": "removed",
            "reason": "constant score on available frames; not a meaningful comparator",
            "n_unique": n_unique,
            "std": std,
        }
    return {"status": "kept", "reason": "implemented and non-constant", "n_unique": n_unique, "std": std}


def _metric_rows(
    scores_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    manifest: pd.DataFrame,
    kept_cols: list[str],
    split_name: str,
) -> list[dict]:
    rows: list[dict] = []
    if "split" in manifest.columns:
        frame_ids = manifest.loc[manifest.split == "test", "frame_id"]
    else:
        frame_ids = manifest["frame_id"]
    merged = (
        scores_df[scores_df.frame_id.isin(frame_ids)]
        .merge(labels_df, on=[c for c in ["frame_id", "corruption_type", "severity"] if c in scores_df.columns and c in labels_df.columns], how="inner")
    )
    for tau_col in ["failure_tau0_50", "failure_tau0_75"]:
        if tau_col not in merged.columns:
            continue
        y = merged[tau_col].astype(int).values
        for col in kept_cols:
            s = merged[col].astype(float).values
            auroc_ci = _ci(y, s, "auroc")
            auprc_ci = _ci(y, s, "auprc")
            rows.append(
                {
                    "split": split_name,
                    "tau": tau_col.replace("failure_tau", "").replace("_", "."),
                    "baseline": col,
                    "n_frames": int(len(y)),
                    "n_failures": int(y.sum()),
                    "failure_rate": float(y.mean()) if len(y) else float("nan"),
                    "auroc": _safe_auroc(y, s),
                    "auroc_ci_lo": auroc_ci["lo"],
                    "auroc_ci_hi": auroc_ci["hi"],
                    "auprc": _safe_auprc(y, s),
                    "auprc_ci_lo": auprc_ci["lo"],
                    "auprc_ci_hi": auprc_ci["hi"],
                    "n_bootstraps": int(min(auroc_ci["n_bootstraps"], auprc_ci["n_bootstraps"])),
                }
            )
    return rows


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")
    clean_scores, clean_inventory = _score_rows(manifest)

    corrupted_manifest_path = PROCESSED_DIR / "corrupted_manifest.parquet"
    corrupted_labels_path = PROCESSED_DIR / "corrupted_labels.parquet"
    corrupted_scores = None
    corrupted_inventory = {"missing_probs": None, "tta_present": None}
    if corrupted_manifest_path.exists() and corrupted_labels_path.exists():
        corrupted_manifest = pd.read_parquet(corrupted_manifest_path)
        corrupted_scores, corrupted_inventory = _score_rows(corrupted_manifest, pred_dir_col="pred_dir")
        corrupted_labels = pd.read_parquet(corrupted_labels_path)
    else:
        corrupted_manifest = None
        corrupted_labels = None

    candidate_cols = [
        ("bl_max_softmax", "probability maps"),
        ("bl_entropy", "probability maps"),
        ("bl_temporal_heuristic", "probability maps ordered by video"),
        ("bl_tta_variance", "TTA variance maps"),
        ("bl_temperature_scaling", "logits for calibration"),
        ("bl_feature_distance_ood", "encoder embeddings"),
    ]

    audit: dict = {
        "scope": "clean test and corrupted test baseline audit",
        "inputs": {
            "manifest": str(PROCESSED_DIR / "manifest.parquet"),
            "labels": str(PROCESSED_DIR / "labels.parquet"),
            "corrupted_manifest": str(corrupted_manifest_path) if corrupted_manifest_path.exists() else None,
            "corrupted_labels": str(corrupted_labels_path) if corrupted_labels_path.exists() else None,
        },
        "inventory": {
            "clean": clean_inventory,
            "corrupted": corrupted_inventory,
            "logits_found": False,
            "embeddings_found": False,
        },
        "baseline_status": {},
    }

    for col, required_cache in candidate_cols:
        audit["baseline_status"][col] = _status_for_column(clean_scores, col, required_cache)
        if (
            audit["baseline_status"][col]["status"] == "kept"
            and corrupted_scores is not None
            and col not in corrupted_scores.columns
        ):
            audit["baseline_status"][col] = {
                "status": "removed",
                "reason": "score is not available for the corrupted-test scope",
                "n_unique": audit["baseline_status"][col]["n_unique"],
                "std": audit["baseline_status"][col]["std"],
            }

    kept_cols = [col for col, _ in candidate_cols if audit["baseline_status"][col]["status"] == "kept"]
    corrected_clean = clean_scores[["frame_id", *kept_cols]].copy()
    save_parquet(corrected_clean, PROCESSED_DIR / "baselines_corrected.parquet")

    metric_rows = _metric_rows(corrected_clean, labels, manifest, kept_cols, "clean_test")

    if corrupted_scores is not None and corrupted_labels is not None and corrupted_manifest is not None:
        corrected_corrupted = corrupted_scores[["frame_id", "corruption_type", "severity", *kept_cols]].copy()
        save_parquet(corrected_corrupted, PROCESSED_DIR / "corrupted_baselines_corrected.parquet")
        metric_rows.extend(
            _metric_rows(corrected_corrupted, corrupted_labels, corrupted_manifest, kept_cols, "corrupted_test")
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULTS_DIR / "baseline_metrics_corrected.csv", index=False)

    audit["kept_baselines"] = kept_cols
    audit["removed_baselines"] = [
        col for col, _ in candidate_cols if audit["baseline_status"][col]["status"] != "kept"
    ]
    audit["outputs"] = {
        "clean_corrected_parquet": str(PROCESSED_DIR / "baselines_corrected.parquet"),
        "corrupted_corrected_parquet": str(PROCESSED_DIR / "corrupted_baselines_corrected.parquet")
        if corrupted_scores is not None
        else None,
        "metrics_csv": str(RESULTS_DIR / "baseline_metrics_corrected.csv"),
    }
    audit["paper_decision"] = (
        "Report max-softmax, entropy, and temporal heuristic as implemented baselines. Remove or "
        "explicitly mark TTA variance, temperature scaling, and feature-distance OOD as unavailable "
        "under the current cross-split cache state."
    )

    with open(RESULTS_DIR / "baseline_audit_2026-06-26.json", "w") as f:
        json.dump(_json_ready(audit), f, indent=2)

    print(json.dumps(_json_ready(audit["baseline_status"]), indent=2))
    print(f"Kept baselines: {', '.join(kept_cols)}")
    print(f"Wrote {RESULTS_DIR / 'baseline_metrics_corrected.csv'}")


if __name__ == "__main__":
    main()
