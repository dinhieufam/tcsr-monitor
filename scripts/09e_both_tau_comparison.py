#!/usr/bin/env python
"""
Stage 9e: Both-τ comparison table.

Reports AUROC and AUPRC for every method (TCSR-Monitor, entropy, all baselines)
at BOTH τ=0.5 and τ=0.75 on the clean test split.

This removes the "threshold cherry-picking" objection: we show the combined
monitor wins at τ=0.5 AND τ=0.75, even though entropy structurally cannot win
at τ=0.75 (all failures are low-entropy there).

Also evaluates the combined monitor at τ=0.5 (previously unreported — only the
OOF-only monitor at τ=0.5 existed).

Output: results/metrics/both_tau_comparison.json
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.monitor.classifiers import load_monitor
from tcsr.utils.io import load_json, load_parquet, save_json
from tcsr.utils.logging import get_logger

log = get_logger(__name__)

PROCESSED_DIR = Path("data/processed/endovis2017")
RESULTS_DIR = Path("results/metrics")
_EPS = 1e-7

BASELINE_COLS = [
    "bl_max_softmax",
    "bl_entropy",
    "bl_temperature_scaling",
    "bl_tta_variance",
    "bl_temporal_heuristic",
    "bl_feature_distance_ood",
]

BASELINE_LABELS = {
    "bl_max_softmax": "Max-Softmax",
    "bl_entropy": "Entropy",
    "bl_temperature_scaling": "Temperature Scaling",
    "bl_tta_variance": "TTA Variance",
    "bl_temporal_heuristic": "Temporal Heuristic",
    "bl_feature_distance_ood": "Feature-Dist OOD",
}

TAUS = [0.5, 0.75]


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def bootstrap_ci(
    y: np.ndarray, s: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    auroc_stats, auprc_stats = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y[idx], s[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        auroc_stats.append(roc_auc_score(yb, sb))
        auprc_stats.append(average_precision_score(yb, sb))
    return {
        "auroc_ci_lo": float(np.percentile(auroc_stats, 2.5)) if auroc_stats else float("nan"),
        "auroc_ci_hi": float(np.percentile(auroc_stats, 97.5)) if auroc_stats else float("nan"),
        "auprc_ci_lo": float(np.percentile(auprc_stats, 2.5)) if auprc_stats else float("nan"),
        "auprc_ci_hi": float(np.percentile(auprc_stats, 97.5)) if auprc_stats else float("nan"),
        "n_bootstraps": len(auroc_stats),
    }


def evaluate_method(
    scores: np.ndarray, y: np.ndarray, method_name: str
) -> dict:
    auroc = safe_auroc(y, scores)
    auprc = safe_auprc(y, scores)
    ci = bootstrap_ci(y, scores)
    return {
        "method": method_name,
        "n_failures": int(y.sum()),
        "n_total": int(len(y)),
        "failure_rate": float(y.mean()),
        "auroc": auroc,
        "auprc": auprc,
        **ci,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load combined monitor (trained at τ=0.75)
    monitor_path = Path("experiments/combined_seed0/monitor.pkl")
    feat_json_path = Path("experiments/combined_seed0/feature_columns.json")

    if not monitor_path.exists():
        raise FileNotFoundError(f"Combined monitor not found at {monitor_path}. Run 06b first.")

    monitor = load_monitor(monitor_path)
    feat_cols = load_json(feat_json_path)
    log.info("Loaded combined monitor from %s", monitor_path)

    # Load data
    features_df = load_parquet(PROCESSED_DIR / "features_all.parquet")
    labels_df = load_parquet(PROCESSED_DIR / "labels.parquet")
    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")[["frame_id", "split"]]
    baselines_df = load_parquet(PROCESSED_DIR / "baselines.parquet")

    # Merge everything
    df = (
        features_df
        .merge(labels_df, on="frame_id")
        .merge(manifest, on="frame_id")
        .merge(baselines_df, on="frame_id", how="left")
    )
    test_df = df[df["split"] == "test"]

    X_test = test_df[feat_cols].values.astype(float)
    monitor_scores = monitor.predict_proba(X_test)

    results: dict = {}

    for tau in TAUS:
        tau_col = f"failure_tau{tau:.2f}".replace(".", "_")
        y = test_df[tau_col].values.astype(int)

        log.info("=== τ=%.2f  n_failures=%d (%.1f%%) ===", tau, y.sum(), 100 * y.mean())

        tau_key = f"tau_{tau:.2f}".replace(".", "_")
        results[tau_key] = {}

        # TCSR-Monitor
        results[tau_key]["TCSR-Monitor"] = evaluate_method(monitor_scores, y, "TCSR-Monitor")
        log.info(
            "  TCSR-Monitor AUROC=%.3f [%.3f, %.3f]",
            results[tau_key]["TCSR-Monitor"]["auroc"],
            results[tau_key]["TCSR-Monitor"]["auroc_ci_lo"],
            results[tau_key]["TCSR-Monitor"]["auroc_ci_hi"],
        )

        # Baselines
        for bl_col in BASELINE_COLS:
            if bl_col in test_df.columns:
                bl_scores = test_df[bl_col].fillna(0).values.astype(float)
                label = BASELINE_LABELS.get(bl_col, bl_col)
                results[tau_key][bl_col] = evaluate_method(bl_scores, y, label)
                log.info(
                    "  %-30s AUROC=%.3f [%.3f, %.3f]",
                    label,
                    results[tau_key][bl_col]["auroc"],
                    results[tau_key][bl_col]["auroc_ci_lo"],
                    results[tau_key][bl_col]["auroc_ci_hi"],
                )
            else:
                log.warning("Baseline column %s not found in baselines.parquet", bl_col)

    # Print summary comparison table
    log.info("\n=== BOTH-τ COMPARISON TABLE ===")
    log.info("%-30s  %8s  %8s  %8s  %8s", "Method", "AUROC@τ=0.5", "AUROC@τ=0.75", "AUPRC@τ=0.5", "AUPRC@τ=0.75")
    methods = list(results.get("tau_0_50", {}).keys())
    for method in methods:
        r05 = results.get("tau_0_50", {}).get(method, {})
        r75 = results.get("tau_0_75", {}).get(method, {})
        log.info(
            "%-30s  %8.3f  %8.3f  %8.3f  %8.3f",
            method,
            r05.get("auroc", float("nan")),
            r75.get("auroc", float("nan")),
            r05.get("auprc", float("nan")),
            r75.get("auprc", float("nan")),
        )

    out_path = RESULTS_DIR / "both_tau_comparison.json"
    save_json(results, out_path)
    log.info("Both-τ comparison saved to %s", out_path)


if __name__ == "__main__":
    main()
