#!/usr/bin/env python
"""Stage 9: Compute all evaluation metrics and dump to results/metrics/."""

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from tcsr.conformal.split_conformal import apply_threshold
from tcsr.data.manifest import load_manifest
from tcsr.evaluation.coverage_risk import coverage_risk_curve
from tcsr.evaluation.latency import benchmark_monitor
from tcsr.evaluation.lead_time import compute_lead_times
from tcsr.evaluation.monitor_metrics import bootstrap_ci, compute_monitor_metrics
from tcsr.evaluation.seg_metrics import compute_seg_metrics
from tcsr.evaluation.stratify import assign_failure_strata, per_stratum_auroc
from tcsr.monitor.classifiers import load_monitor
from tcsr.utils import (
    capture_run_metadata,
    get_logger,
    load_json,
    load_parquet,
    save_json,
    set_seed,
)

log = get_logger(__name__)

BASELINE_COLS = [
    "bl_max_softmax",
    "bl_entropy",
    "bl_temporal_heuristic",
]


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    capture_run_metadata(cfg, output_dir, cfg.seed)

    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = Path(cfg.data.processed_dir)
    manifest = load_manifest(processed_dir / "manifest.parquet")
    labels_df = load_parquet(processed_dir / "labels.parquet")

    tau = getattr(cfg, "failure_tau", 0.5)
    tau_col = f"failure_tau{tau:.2f}".replace(".", "_")
    fps = float(cfg.data.fps)

    # Seg metrics on test split
    test_manifest = manifest[manifest.split == "test"]
    test_labels = labels_df[labels_df.frame_id.isin(test_manifest.frame_id)]
    seg_metrics = compute_seg_metrics(test_labels)
    save_json(seg_metrics, results_dir / "seg_metrics.json")
    log.info("Seg metrics: %s", seg_metrics)

    # Monitor metrics
    monitor_dir = Path(getattr(cfg, "monitor_dir", output_dir))
    monitor = load_monitor(monitor_dir / "monitor.pkl")
    feat_cols = load_json(monitor_dir / "feature_columns.json")
    conformal = load_json(monitor_dir / "conformal_threshold.json") if (
        monitor_dir / "conformal_threshold.json"
    ).exists() else {"threshold": 0.5}

    features_df = load_parquet(processed_dir / f"features_{cfg.features.name}.parquet")
    df = features_df.merge(labels_df[[tau_col, "frame_id"]], on="frame_id").merge(
        manifest[["frame_id", "split", "video_id"]], on="frame_id"
    )
    test_df = df[df.split == "test"]
    X_test = test_df[feat_cols].values
    y_test = test_df[tau_col].values
    video_ids = test_df.video_id.values

    scores = monitor.predict_proba(X_test)
    threshold = conformal["threshold"]
    alarms = apply_threshold(scores, threshold)

    monitor_metrics = compute_monitor_metrics(scores, y_test, fps=fps, threshold=threshold)
    lead = compute_lead_times(alarms, y_test, video_ids, fps=fps)
    monitor_metrics.update(lead)
    save_json(monitor_metrics, results_dir / "monitor_metrics.json")

    # Coverage–risk curve
    curve = coverage_risk_curve(scores, y_test)
    curves_out = {
        "TCSR-Monitor": {
            "coverage": curve["coverage"].tolist(),
            "risk": curve["risk"].tolist(),
        }
    }

    # Baselines
    baselines_df = load_parquet(processed_dir / "baselines.parquet")
    test_bl = baselines_df[baselines_df.frame_id.isin(test_df.frame_id)]
    all_metrics: dict = {"TCSR-Monitor": monitor_metrics}
    latency: dict = {}

    for bl_col in BASELINE_COLS:
        if bl_col not in test_bl.columns:
            continue
        bl_scores = test_bl.set_index("frame_id").loc[test_df.frame_id, bl_col].values
        bl_metrics = compute_monitor_metrics(bl_scores, y_test, fps=fps, threshold=0.5)
        all_metrics[bl_col] = bl_metrics
        bl_curve = coverage_risk_curve(bl_scores, y_test)
        curves_out[bl_col] = {
            "coverage": bl_curve["coverage"].tolist(),
            "risk": bl_curve["risk"].tolist(),
        }

    save_json(all_metrics, results_dir / "monitor_metrics_summary.json")
    save_json(curves_out, results_dir / "coverage_risk_curves.json")

    # Latency benchmark
    latency["TCSR-Monitor"] = benchmark_monitor(monitor, X_test)
    save_json(latency, results_dir / "latency.json")

    # Bootstrap CIs on monitor and baselines
    if y_test.sum() > 5 and (len(y_test) - y_test.sum()) > 5:
        ci = bootstrap_ci(scores, y_test)
        monitor_metrics.update(ci)
        save_json(monitor_metrics, results_dir / "monitor_metrics.json")
        for bl_col in BASELINE_COLS:
            if bl_col in all_metrics:
                bl_scores_arr = test_bl.set_index("frame_id").loc[test_df.frame_id, bl_col].values
                bl_ci = bootstrap_ci(bl_scores_arr.astype(float), y_test)
                all_metrics[bl_col].update(bl_ci)
        save_json(all_metrics, results_dir / "monitor_metrics_summary.json")

    # Failure stratification
    features_test = features_df[features_df.frame_id.isin(test_df.frame_id)].set_index("frame_id").loc[test_df.frame_id].reset_index()
    entropy_scores_arr = test_bl.set_index("frame_id").loc[test_df.frame_id, "bl_entropy"].values if "bl_entropy" in test_bl.columns else np.zeros(len(y_test))
    strata = assign_failure_strata(
        labels=y_test,
        scores_entropy=entropy_scores_arr.astype(float),
        features_df=features_test,
    )
    stratum_metrics = per_stratum_auroc(
        labels=y_test,
        scores_monitor=scores,
        scores_entropy=entropy_scores_arr.astype(float),
        strata=strata,
    )
    stratum_metrics["n_total_failures"] = int(y_test.sum())
    save_json(stratum_metrics, results_dir / "stratified_metrics.json")
    log.info("Stratified metrics: %s", stratum_metrics)

    log.info("Evaluation complete. Results in %s", results_dir)


if __name__ == "__main__":
    main()
