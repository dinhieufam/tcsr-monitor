#!/usr/bin/env python
"""Run and aggregate the feature-family ablation experiment.

This script retrains each ablation monitor with the same split/seed protocol,
evaluates the matching model directory, and preserves per-seed metrics before
writing the aggregate table used by the paper figures.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


FEATURES = ["all", "confidence_only", "no_confidence", "no_shape", "no_temporal", "no_quality"]
SEEDS = [0, 1, 2]
RESULTS_DIR = Path("results/metrics")
OUT_FILE = RESULTS_DIR / "ablation_features.json"
OUT_CSV = RESULTS_DIR / "ablation_features.csv"
ABLATION_DIR = Path("experiments/ablation_feature_family")
FAILURE_TAU = 0.5
DATASET = "endovis2017"
MONITOR = "xgboost"
PYTHON = "/home/24hieu.pd/miniconda3/envs/tcsr-monitor/bin/python"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    conda_lib = "/home/24hieu.pd/miniconda3/envs/tcsr-monitor/lib"
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{conda_lib}:{existing}" if existing else conda_lib
    return env


def _run(cmd: list[str], label: str) -> None:
    print(f"\n[{label}] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=_env())
    if result.returncode != 0:
        print(result.stdout[-2000:], file=sys.stderr)
        print(result.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    if result.stdout.strip():
        print(result.stdout[-1000:])


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _mean_std(vals: np.ndarray) -> tuple[float, float]:
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict] = []

    for feat in FEATURES:
        for seed in SEEDS:
            run_name = f"ablation_feat_{feat}_seed{seed}"
            run_dir = Path("experiments") / run_name

            common = [
                f"data={DATASET}",
                f"features={feat}",
                f"monitor={MONITOR}",
                f"seed={seed}",
                f"+failure_tau={FAILURE_TAU}",
                f"hydra.run.dir={run_dir.as_posix()}",
            ]

            features_path = Path("data/processed") / DATASET / f"features_{feat}.parquet"
            if not features_path.exists():
                _run(
                    [
                        PYTHON,
                        "scripts/05_extract_features.py",
                        f"data={DATASET}",
                        f"features={feat}",
                        f"seed={seed}",
                        f"hydra.run.dir={run_dir.as_posix()}",
                    ],
                    f"extract features {feat}",
                )

            _run([PYTHON, "scripts/06_train_monitor.py", *common], f"train {feat} seed={seed}")
            _run(
                [
                    PYTHON,
                    "scripts/09_evaluate.py",
                    *common,
                    f"+monitor_dir={run_dir.as_posix()}",
                ],
                f"eval {feat} seed={seed}",
            )

            metrics = _load_json(RESULTS_DIR / "monitor_metrics.json")
            feature_cols = _load_json(run_dir / "feature_columns.json")
            row = {
                "feature_set": feat,
                "seed": seed,
                "n_features": len(feature_cols),
                "auroc": float(metrics.get("auroc", float("nan"))),
                "auprc": float(metrics.get("auprc", float("nan"))),
                "brier": float(metrics.get("brier", float("nan"))),
                "ece": float(metrics.get("ece", float("nan"))),
                "recall_at_cov90": float(metrics.get("recall_at_cov90", float("nan"))),
                "fa_per_min": float(metrics.get("fa_per_min", float("nan"))),
                "recall": float(metrics.get("recall", float("nan"))),
                "precision": float(metrics.get("precision", float("nan"))),
            }
            per_seed.append(row)

            per_run_dir = ABLATION_DIR / run_name
            per_run_dir.mkdir(parents=True, exist_ok=True)
            _copy_if_exists(RESULTS_DIR / "monitor_metrics.json", per_run_dir / "monitor_metrics.json")
            _copy_if_exists(RESULTS_DIR / "monitor_metrics_summary.json", per_run_dir / "monitor_metrics_summary.json")
            _copy_if_exists(RESULTS_DIR / "stratified_metrics.json", per_run_dir / "stratified_metrics.json")
            _copy_if_exists(run_dir / "feature_columns.json", per_run_dir / "feature_columns.json")
            _copy_if_exists(run_dir / "feature_importances.json", per_run_dir / "feature_importances.json")

            print(
                f"  {feat} seed={seed}: "
                f"AUROC={row['auroc']:.3f}, AUPRC={row['auprc']:.3f}, "
                f"features={row['n_features']}"
            )

    aggregate: dict[str, dict] = {}
    for feat in FEATURES:
        rows = [r for r in per_seed if r["feature_set"] == feat]
        aggregate[feat] = {"n_seeds": len(rows), "per_seed": rows}
        for metric in ["auroc", "auprc", "brier", "ece", "recall_at_cov90", "fa_per_min", "recall", "precision"]:
            vals = np.array([r[metric] for r in rows], dtype=float)
            mean, std = _mean_std(vals)
            aggregate[feat][metric] = mean
            aggregate[feat][f"{metric}_std"] = std
        aggregate[feat]["n_features"] = int(rows[0]["n_features"]) if rows else 0

    with open(OUT_FILE, "w") as f:
        json.dump(aggregate, f, indent=2)

    csv_fields = [
        "feature_set",
        "seed",
        "n_features",
        "auroc",
        "auprc",
        "brier",
        "ece",
        "recall_at_cov90",
        "fa_per_min",
        "recall",
        "precision",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(per_seed)

    print("\nAggregate summary")
    for feat in FEATURES:
        m = aggregate[feat]
        print(
            f"{feat:16s} "
            f"n_features={m['n_features']:2d} "
            f"AUROC={m['auroc']:.3f}±{m['auroc_std']:.3f} "
            f"AUPRC={m['auprc']:.3f}±{m['auprc_std']:.3f}"
        )
    print(f"\nSaved {OUT_FILE}")
    print(f"Saved {OUT_CSV}")
    print(f"Saved per-run metrics under {ABLATION_DIR}")


if __name__ == "__main__":
    main()
