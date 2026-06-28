#!/usr/bin/env python
"""
Experiment E10: cached cross-dataset evaluation on CholecSeg8k.

This script executes the feasible E10 protocol using existing cached artifacts:
EndoVis-trained monitors are applied to CholecSeg8k feature tables. When the
external labels are one-class, AUROC/AUPRC are reported as unavailable rather
than forced into a misleading number.

Run from: tcsr-monitor/
  python scripts/12_external_dataset_eval.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.conformal.split_conformal import apply_threshold, calibrate_threshold
from tcsr.monitor.classifiers import load_monitor
from tcsr.utils.io import load_json


ENDOVIS_DIR = Path("data/processed/endovis2017")
EXTERNAL_DIR = Path("data/processed/cholecseg8k")
METRICS_DIR = Path("results/metrics/external")
FIGURES_DIR = Path("results/figures/external")
EXPERIMENT_DIR = Path("experiments/external_cholecseg8k")

FEATURE_PREFIXES = ("conf_", "shape_", "temp_", "qual_", "shift_")
TAU_COLS = ["failure_tau0_50", "failure_tau0_60", "failure_tau0_75"]

MONITORS = {
    "endovis_clean_seed0": {
        "dir": Path("experiments/seed0"),
        "description": "EndoVis clean-trained monitor, seed 0",
    },
    "endovis_combined_seed0": {
        "dir": Path("experiments/combined_seed0"),
        "description": "EndoVis clean+corrupted combined monitor, seed 0",
    },
}

BASELINE_COLS = ["bl_max_softmax", "bl_entropy", "bl_temporal_heuristic"]


def _safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def _corrs(score: np.ndarray, iou: np.ndarray) -> dict:
    df = pd.DataFrame({"score": score, "risk": 1.0 - iou})
    return {
        "pearson_score_vs_one_minus_iou": float(df["score"].corr(df["risk"], method="pearson")),
        "spearman_score_vs_one_minus_iou": float(df["score"].corr(df["risk"], method="spearman")),
    }


def _bootstrap_metric(score: np.ndarray, iou: np.ndarray, metric: str, n_boot: int = 1000) -> dict:
    rng = np.random.default_rng(42)
    vals: list[float] = []
    n = len(score)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if metric == "spearman":
            val = pd.Series(score[idx]).corr(pd.Series(1.0 - iou[idx]), method="spearman")
        elif metric == "pearson":
            val = pd.Series(score[idx]).corr(pd.Series(1.0 - iou[idx]), method="pearson")
        elif metric == "mean":
            val = float(np.mean(score[idx]))
        else:
            raise ValueError(metric)
        if not math.isnan(val):
            vals.append(float(val))
    if not vals:
        return {"ci_lo": float("nan"), "ci_hi": float("nan")}
    return {
        "ci_lo": float(np.percentile(vals, 2.5)),
        "ci_hi": float(np.percentile(vals, 97.5)),
    }


def _bootstrap_rate(values: np.ndarray, n_boot: int = 1000) -> dict:
    rng = np.random.default_rng(43)
    vals = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(float(values[idx].mean()))
    return {
        "ci_lo": float(np.percentile(vals, 2.5)),
        "ci_hi": float(np.percentile(vals, 97.5)),
    }


def _load_external() -> pd.DataFrame:
    manifest = pd.read_parquet(EXTERNAL_DIR / "manifest.parquet")
    features = pd.read_parquet(EXTERNAL_DIR / "features_all.parquet")
    labels = pd.read_parquet(EXTERNAL_DIR / "labels.parquet")
    baselines = pd.read_parquet(EXTERNAL_DIR / "baselines.parquet")
    return (
        features.merge(labels, on="frame_id")
        .merge(manifest, on="frame_id")
        .merge(baselines[["frame_id", *BASELINE_COLS]], on="frame_id")
    )


def _load_endovis_for_calibration() -> pd.DataFrame:
    manifest = pd.read_parquet(ENDOVIS_DIR / "manifest.parquet")
    labels = pd.read_parquet(ENDOVIS_DIR / "labels.parquet")
    baselines = pd.read_parquet(ENDOVIS_DIR / "baselines_corrected.parquet")
    features = pd.read_parquet(ENDOVIS_DIR / "features_all.parquet")
    return (
        features.merge(labels, on="frame_id")
        .merge(manifest[["frame_id", "split"]], on="frame_id")
        .merge(baselines[["frame_id", *BASELINE_COLS]], on="frame_id")
    )


def _score_monitors(external: pd.DataFrame, endovis: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    score_df = external[["frame_id", "video_id", "frame_idx_in_video", "iou", *TAU_COLS]].copy()
    monitor_records = {}
    for name, spec in MONITORS.items():
        monitor_dir = spec["dir"]
        monitor = load_monitor(monitor_dir / "monitor.pkl")
        feat_cols = load_json(monitor_dir / "feature_columns.json")
        missing = [c for c in feat_cols if c not in external.columns]
        if missing:
            raise ValueError(f"{name} missing external feature columns: {missing}")

        score = monitor.predict_proba(external[feat_cols].to_numpy(np.float32))
        score_df[name] = score

        cal_df = endovis[endovis["split"] == "cal"].copy()
        cal_score = monitor.predict_proba(cal_df[feat_cols].to_numpy(np.float32))
        threshold_path = monitor_dir / "conformal_threshold.json"
        threshold = load_json(threshold_path)["threshold"] if threshold_path.exists() else None
        threshold_source = str(threshold_path) if threshold_path.exists() else "not_available"

        monitor_records[name] = {
            "description": spec["description"],
            "feature_columns": feat_cols,
            "n_feature_columns": len(feat_cols),
            "threshold_source": threshold_source,
            "threshold": threshold,
            "endovis_calibrated_miss_rate_on_external_all_failures": (
                float((apply_threshold(score, threshold) == 0).mean())
                if threshold is not None
                else float("nan")
            ),
        }

        if threshold is not None:
            missed = (apply_threshold(score, threshold) == 0).astype(float)
            monitor_records[name].update({
                "external_alarm_rate": float(1.0 - missed.mean()),
                "external_miss_rate_ci": _bootstrap_rate(missed),
            })

        for tau_col in TAU_COLS:
            y = external[tau_col].to_numpy(int)
            monitor_records[name][tau_col] = {
                "prevalence": float(y.mean()),
                "n_failures": int(y.sum()),
                "n_total": int(len(y)),
                "auroc": _safe_auroc(y, score),
                "auprc": _safe_auprc(y, score),
            }

        monitor_records[name].update(_corrs(score, external["iou"].to_numpy(float)))
        monitor_records[name]["spearman_ci"] = _bootstrap_metric(
            score, external["iou"].to_numpy(float), "spearman"
        )

    return score_df, monitor_records


def _score_baselines(external: pd.DataFrame, endovis: pd.DataFrame) -> dict:
    records = {}
    for col in BASELINE_COLS:
        score = external[col].to_numpy(float)
        records[col] = {
            "prevalence_tau0_50": float(external["failure_tau0_50"].mean()),
            "auroc_tau0_50": _safe_auroc(external["failure_tau0_50"].to_numpy(int), score),
            "auprc_tau0_50": _safe_auprc(external["failure_tau0_50"].to_numpy(int), score),
            **_corrs(score, external["iou"].to_numpy(float)),
            "spearman_ci": _bootstrap_metric(score, external["iou"].to_numpy(float), "spearman"),
        }

        cal_df = endovis[endovis["split"] == "cal"]
        try:
            cal = calibrate_threshold(
                scores=cal_df[col].to_numpy(float),
                labels=cal_df["failure_tau0_50"].to_numpy(int),
                alpha=0.1,
                score_type="one_minus_prob",
            )
            threshold = cal["threshold"]
            missed = (apply_threshold(score, threshold) == 0).astype(float)
            records[col].update({
                "endovis_calibrated_threshold_tau0_50": threshold,
                "endovis_calibrated_miss_rate_on_external_all_failures": float(missed.mean()),
                "external_alarm_rate": float(1.0 - missed.mean()),
                "external_miss_rate_ci": _bootstrap_rate(missed),
            })
        except Exception as exc:
            records[col]["threshold_error"] = str(exc)
    return records


def _summary_rows(monitor_records: dict, baseline_records: dict) -> pd.DataFrame:
    rows = []
    for name, rec in monitor_records.items():
        tau = rec["failure_tau0_50"]
        rows.append({
            "method": name,
            "kind": "monitor",
            "n_total": tau["n_total"],
            "n_failures_tau0_50": tau["n_failures"],
            "prevalence_tau0_50": tau["prevalence"],
            "auroc_tau0_50": tau["auroc"],
            "auprc_tau0_50": tau["auprc"],
            "spearman_score_vs_one_minus_iou": rec["spearman_score_vs_one_minus_iou"],
            "spearman_ci_lo": rec["spearman_ci"]["ci_lo"],
            "spearman_ci_hi": rec["spearman_ci"]["ci_hi"],
            "endovis_calibrated_threshold": rec["threshold"],
            "external_miss_rate_all_failures": rec["endovis_calibrated_miss_rate_on_external_all_failures"],
            "external_alarm_rate": rec.get("external_alarm_rate", float("nan")),
        })
    for name, rec in baseline_records.items():
        rows.append({
            "method": name,
            "kind": "baseline",
            "n_total": 8080,
            "n_failures_tau0_50": 8080,
            "prevalence_tau0_50": rec["prevalence_tau0_50"],
            "auroc_tau0_50": rec["auroc_tau0_50"],
            "auprc_tau0_50": rec["auprc_tau0_50"],
            "spearman_score_vs_one_minus_iou": rec["spearman_score_vs_one_minus_iou"],
            "spearman_ci_lo": rec["spearman_ci"]["ci_lo"],
            "spearman_ci_hi": rec["spearman_ci"]["ci_hi"],
            "endovis_calibrated_threshold": rec.get("endovis_calibrated_threshold_tau0_50", float("nan")),
            "external_miss_rate_all_failures": rec.get("endovis_calibrated_miss_rate_on_external_all_failures", float("nan")),
            "external_alarm_rate": rec.get("external_alarm_rate", float("nan")),
        })
    return pd.DataFrame(rows)


def _make_figures(external: pd.DataFrame, score_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.hist(external["iou"], bins=40, color="#3b6ea8", alpha=0.85)
    ax.axvline(0.5, color="#9b2f2f", linestyle="--", linewidth=1.2)
    ax.set_xlabel("IoU")
    ax.set_ylabel("Frames")
    ax.set_title("CholecSeg8k external IoU distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cholecseg8k_iou_distribution.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(
        external["iou"],
        score_df["endovis_combined_seed0"],
        s=8,
        alpha=0.25,
        color="#2f7f6f",
        edgecolors="none",
    )
    ax.set_xlabel("IoU")
    ax.set_ylabel("EndoVis combined-monitor risk score")
    ax.set_title("External risk score versus segmentation quality")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cholecseg8k_score_vs_iou.png", dpi=220)
    plt.close(fig)

    plot_df = summary.sort_values("external_miss_rate_all_failures")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.barh(plot_df["method"], plot_df["external_miss_rate_all_failures"], color="#6b6f92")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Miss rate under EndoVis-calibrated threshold")
    ax.set_title("External all-failure alarm sensitivity")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cholecseg8k_external_miss_rates.png", dpi=220)
    plt.close(fig)


def _json_clean(obj):
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if math.isnan(val) or math.isinf(val) else val
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    external = _load_external()
    endovis = _load_endovis_for_calibration()
    score_df, monitor_records = _score_monitors(external, endovis)
    baseline_records = _score_baselines(external, endovis)
    summary = _summary_rows(monitor_records, baseline_records)

    by_clip = (
        score_df.merge(external[["frame_id", "split"]], on="frame_id")
        .groupby("video_id")
        .agg(
            n=("frame_id", "size"),
            mean_iou=("iou", "mean"),
            min_iou=("iou", "min"),
            max_iou=("iou", "max"),
            failure_rate_tau0_50=("failure_tau0_50", "mean"),
            combined_monitor_mean_score=("endovis_combined_seed0", "mean"),
        )
        .reset_index()
    )

    _make_figures(external, score_df, summary)

    score_path = EXPERIMENT_DIR / "cholecseg8k_external_scores.parquet"
    score_df.to_parquet(score_path, index=False)
    summary_path = METRICS_DIR / "cholecseg8k_external_summary.csv"
    by_clip_path = METRICS_DIR / "cholecseg8k_by_clip.csv"
    summary.to_csv(summary_path, index=False)
    by_clip.to_csv(by_clip_path, index=False)

    n = len(external)
    result = {
        "experiment": "E10 Cross-Dataset Evaluation",
        "status": "completed_with_one_class_label_limitation",
        "dataset": "cholecseg8k",
        "n_frames": int(n),
        "n_clips": int(external["video_id"].nunique()),
        "artifact_inventory": {
            "frames": int(n),
            "masks": int(n),
            "prediction_probability_maps": int(sum(1 for _ in (EXTERNAL_DIR / "predictions").rglob("probs.npz"))),
            "manifest": str(EXTERNAL_DIR / "manifest.parquet"),
            "features": str(EXTERNAL_DIR / "features_all.parquet"),
            "labels": str(EXTERNAL_DIR / "labels.parquet"),
            "baselines": str(EXTERNAL_DIR / "baselines.parquet"),
        },
        "segmentation_quality": {
            "mean_iou": float(external["iou"].mean()),
            "median_iou": float(external["iou"].median()),
            "std_iou": float(external["iou"].std()),
            "min_iou": float(external["iou"].min()),
            "max_iou": float(external["iou"].max()),
            "failure_rate_tau0_50": float(external["failure_tau0_50"].mean()),
            "failure_rate_tau0_60": float(external["failure_tau0_60"].mean()),
            "failure_rate_tau0_75": float(external["failure_tau0_75"].mean()),
        },
        "metrics_note": (
            "AUROC and AUPRC are unavailable because every CholecSeg8k frame is "
            "a failure at tau=0.50, tau=0.60, and tau=0.75 under the cached "
            "prediction/label pipeline."
        ),
        "monitor_results": monitor_records,
        "baseline_results": baseline_records,
        "outputs": {
            "summary_csv": str(summary_path),
            "by_clip_csv": str(by_clip_path),
            "score_parquet": str(score_path),
            "figures": [
                str(FIGURES_DIR / "cholecseg8k_iou_distribution.png"),
                str(FIGURES_DIR / "cholecseg8k_score_vs_iou.png"),
                str(FIGURES_DIR / "cholecseg8k_external_miss_rates.png"),
            ],
        },
        "publication_interpretation": (
            "Use as a limitations audit, not as positive external validation. "
            "The external segmenter/mask pipeline produces near-universal failure, "
            "so binary failure-detection discrimination cannot be assessed."
        ),
    }

    result_path = METRICS_DIR / "cholecseg8k_external_results.json"
    result_path.write_text(json.dumps(_json_clean(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": result["status"],
        "n_frames": result["n_frames"],
        "mean_iou": result["segmentation_quality"]["mean_iou"],
        "failure_rate_tau0_50": result["segmentation_quality"]["failure_rate_tau0_50"],
        "summary_csv": str(summary_path),
        "results_json": str(result_path),
    }, indent=2))


if __name__ == "__main__":
    main()
