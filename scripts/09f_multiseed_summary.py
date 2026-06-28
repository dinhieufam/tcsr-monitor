#!/usr/bin/env python
"""
Stage 9f: Multi-seed summary for the combined monitor.

Evaluates monitors trained at seeds 0, 1, 2 on the clean test set and reports
mean ± std across seeds for AUROC and AUPRC, with bootstrap CIs.

Requires: experiments/combined_seed{0,1,2}/monitor.pkl

Output: results/metrics/multiseed_summary.json
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
from tcsr.utils.seed import set_seed

log = get_logger(__name__)

PROCESSED_DIR = Path("data/processed/endovis2017")
RESULTS_DIR = Path("results/metrics")
SEEDS = [0, 1, 2]
TAUS = [0.5, 0.75]


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
    auroc_s, auprc_s = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y[idx], s[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        auroc_s.append(roc_auc_score(yb, sb))
        auprc_s.append(average_precision_score(yb, sb))
    return {
        "auroc_ci_lo": float(np.percentile(auroc_s, 2.5)) if auroc_s else float("nan"),
        "auroc_ci_hi": float(np.percentile(auroc_s, 97.5)) if auroc_s else float("nan"),
        "auprc_ci_lo": float(np.percentile(auprc_s, 2.5)) if auprc_s else float("nan"),
        "auprc_ci_hi": float(np.percentile(auprc_s, 97.5)) if auprc_s else float("nan"),
    }


def main() -> None:
    set_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    features_df = load_parquet(PROCESSED_DIR / "features_all.parquet")
    labels_df = load_parquet(PROCESSED_DIR / "labels.parquet")
    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")[["frame_id", "split"]]

    df = features_df.merge(labels_df, on="frame_id").merge(manifest, on="frame_id")
    test_df = df[df["split"] == "test"]

    results: dict = {"per_seed": {}, "aggregate": {}}

    for tau in TAUS:
        tau_col = f"failure_tau{tau:.2f}".replace(".", "_")
        tau_key = f"tau_{tau:.2f}".replace(".", "_")
        y_test = test_df[tau_col].values.astype(int)
        log.info("=== τ=%.2f  n_failures=%d ===", tau, y_test.sum())

        seed_aurocs, seed_auprcs = [], []

        for seed in SEEDS:
            monitor_path = Path(f"experiments/combined_seed{seed}/monitor.pkl")
            feat_json = Path(f"experiments/combined_seed{seed}/feature_columns.json")

            if not monitor_path.exists():
                log.warning("Seed %d monitor not found at %s — skipping", seed, monitor_path)
                continue

            monitor = load_monitor(monitor_path)
            feat_cols = load_json(feat_json)

            for col in feat_cols:
                if col not in test_df.columns:
                    test_df[col] = 0.0

            X_test = test_df[feat_cols].values.astype(float)
            scores = monitor.predict_proba(X_test)

            auroc = safe_auroc(y_test, scores)
            auprc = safe_auprc(y_test, scores)
            ci = bootstrap_ci(y_test, scores, seed=seed)

            seed_aurocs.append(auroc)
            seed_auprcs.append(auprc)

            seed_key = f"seed{seed}"
            if seed_key not in results["per_seed"]:
                results["per_seed"][seed_key] = {}
            results["per_seed"][seed_key][tau_key] = {
                "auroc": auroc,
                "auprc": auprc,
                **ci,
            }
            log.info("  Seed %d: AUROC=%.3f [%.3f,%.3f] AUPRC=%.3f",
                     seed, auroc, ci["auroc_ci_lo"], ci["auroc_ci_hi"], auprc)

        if seed_aurocs:
            results["aggregate"][tau_key] = {
                "n_seeds": len(seed_aurocs),
                "auroc_mean": float(np.nanmean(seed_aurocs)),
                "auroc_std": float(np.nanstd(seed_aurocs)),
                "auprc_mean": float(np.nanmean(seed_auprcs)),
                "auprc_std": float(np.nanstd(seed_auprcs)),
            }
            log.info(
                "  Aggregate: AUROC=%.3f±%.3f  AUPRC=%.3f±%.3f",
                results["aggregate"][tau_key]["auroc_mean"],
                results["aggregate"][tau_key]["auroc_std"],
                results["aggregate"][tau_key]["auprc_mean"],
                results["aggregate"][tau_key]["auprc_std"],
            )

    out_path = RESULTS_DIR / "multiseed_summary.json"
    save_json(results, out_path)
    log.info("Multi-seed summary saved to %s", out_path)


if __name__ == "__main__":
    main()
