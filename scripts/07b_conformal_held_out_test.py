#!/usr/bin/env python
"""
Stage 7b: Conformal coverage validation on held-out test split.

The existing calibration in 07 computes the conformal threshold on the cal
split and reports realized miss-rate on the *same* cal split — nearly circular.
This script validates on the held-out test split instead.

Procedure:
  - For N_REPEATS random splits of (cal ∪ test) into 50% cal / 50% test:
      1. Calibrate threshold on cal portion (target α=0.10)
      2. Compute realized miss-rate on test portion
  - Report: mean ± std and 95% CI of realized test miss-rate
  - Confirm it is ≤ α with CI lower bound

Also reports abstention rate (how many frames are alarmed at calibrated threshold).

Output: results/metrics/conformal_test_coverage.json
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.conformal.split_conformal import calibrate_threshold, apply_threshold
from tcsr.monitor.classifiers import load_monitor
from tcsr.utils.io import load_json, load_parquet, save_json
from tcsr.utils.logging import get_logger
from tcsr.utils.seed import set_seed

log = get_logger(__name__)

PROCESSED_DIR = Path("data/processed/endovis2017")
RESULTS_DIR = Path("results/metrics")

TAU = 0.75
TAU_COL = "failure_tau0_75"
ALPHA = 0.10
N_REPEATS = 200
CAL_FRACTION = 0.5


def main() -> None:
    set_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    monitor_path = Path("experiments/combined_seed0/monitor.pkl")
    feat_json_path = Path("experiments/combined_seed0/feature_columns.json")
    if not monitor_path.exists():
        raise FileNotFoundError(f"Run 06b first. Missing: {monitor_path}")

    monitor = load_monitor(monitor_path)
    feat_cols = load_json(feat_json_path)
    log.info("Loaded combined monitor from %s", monitor_path)

    features_df = load_parquet(PROCESSED_DIR / "features_all.parquet")
    labels_df = load_parquet(PROCESSED_DIR / "labels.parquet")
    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")[["frame_id", "split"]]

    df = features_df.merge(labels_df[[TAU_COL, "frame_id"]], on="frame_id").merge(manifest, on="frame_id")

    # Pool cal + test for the held-out validation experiment
    pool = df[df["split"].isin(["cal", "test"])].copy().reset_index(drop=True)
    X_pool = pool[feat_cols].values.astype(float)
    y_pool = pool[TAU_COL].values.astype(int)
    scores_pool = monitor.predict_proba(X_pool)

    log.info(
        "Pool (cal+test): %d frames, %d failures (%.1f%%)",
        len(y_pool), y_pool.sum(), 100 * y_pool.mean(),
    )

    rng = np.random.default_rng(42)
    miss_rates = []
    abstention_rates = []
    thresholds = []

    for rep in range(N_REPEATS):
        idx = np.arange(len(pool))
        rng.shuffle(idx)
        n_cal = int(len(idx) * CAL_FRACTION)
        cal_idx = idx[:n_cal]
        test_idx = idx[n_cal:]

        y_cal = y_pool[cal_idx]
        s_cal = scores_pool[cal_idx]
        y_test = y_pool[test_idx]
        s_test = scores_pool[test_idx]

        # Need at least some failures in cal
        if y_cal.sum() == 0:
            continue

        try:
            cal_result = calibrate_threshold(s_cal, y_cal, alpha=ALPHA)
        except Exception:
            continue
        thresh = cal_result["threshold"]

        # Evaluate on test
        failure_idx_test = y_test == 1
        if failure_idx_test.sum() == 0:
            continue

        alarms_test = apply_threshold(s_test, thresh)
        # Miss = failure frame that was NOT alarmed
        missed = float((alarms_test[failure_idx_test] == 0).mean())
        abstained = float(alarms_test.mean())  # fraction of frames alarmed

        miss_rates.append(missed)
        abstention_rates.append(abstained)
        thresholds.append(thresh)

    miss_rates = np.array(miss_rates)
    abstention_rates = np.array(abstention_rates)
    thresholds_arr = np.array(thresholds)

    mean_miss = float(miss_rates.mean())
    std_miss = float(miss_rates.std())
    ci_lo = float(np.percentile(miss_rates, 2.5))
    ci_hi = float(np.percentile(miss_rates, 97.5))
    mean_abstention = float(abstention_rates.mean())
    mean_threshold = float(thresholds_arr.mean())

    log.info("=== CONFORMAL TEST COVERAGE (%d repeats, α=%.2f) ===", len(miss_rates), ALPHA)
    log.info("Test miss-rate: mean=%.4f std=%.4f 95%%CI=[%.4f, %.4f]", mean_miss, std_miss, ci_lo, ci_hi)
    log.info("Abstention rate: mean=%.4f", mean_abstention)
    log.info("Mean calibrated threshold: %.4f", mean_threshold)

    certified = ci_hi <= ALPHA
    log.info("'α=%.2f certified' on test: %s (CI upper bound %.4f %s %.2f)",
             ALPHA, "✓" if certified else "✗", ci_hi, "≤" if certified else ">", ALPHA)

    # Also report for original cal-split threshold
    original_conf = None
    orig_path = Path("experiments/combined_seed0/conformal_threshold.json")
    if orig_path.exists():
        original_conf = load_json(orig_path)
        orig_thresh = original_conf["threshold"]
        # Apply to full test split
        test_only = df[df["split"] == "test"]
        X_test_only = test_only[feat_cols].values.astype(float)
        y_test_only = test_only[TAU_COL].values.astype(int)
        s_test_only = monitor.predict_proba(X_test_only)
        alarms_test_only = apply_threshold(s_test_only, orig_thresh)
        if y_test_only.sum() > 0:
            orig_miss = float((alarms_test_only[y_test_only == 1] == 0).mean())
            orig_abstention = float(alarms_test_only.mean())
            log.info(
                "Original cal-split threshold (%.4f): test miss-rate=%.4f abstention=%.4f",
                orig_thresh, orig_miss, orig_abstention,
            )

    result = {
        "alpha": ALPHA,
        "tau": TAU,
        "n_repeats": int(len(miss_rates)),
        "cal_fraction": CAL_FRACTION,
        "test_miss_rate_mean": mean_miss,
        "test_miss_rate_std": std_miss,
        "test_miss_rate_ci_lo": ci_lo,
        "test_miss_rate_ci_hi": ci_hi,
        "mean_abstention_rate": mean_abstention,
        "mean_calibrated_threshold": mean_threshold,
        "certified_at_alpha": bool(certified),
        "interpretation": (
            f"Realized test miss-rate {mean_miss:.4f} ± {std_miss:.4f} "
            f"[{ci_lo:.4f}, {ci_hi:.4f}], target α={ALPHA}. "
            f"{'Coverage guaranteed' if certified else 'Coverage not guaranteed at α=0.10 — motivates Mondrian conformal'}."
        ),
    }

    if original_conf is not None and "orig_miss" in dir():
        result["original_threshold_test_miss_rate"] = orig_miss  # type: ignore[possibly-undefined]
        result["original_threshold_abstention_rate"] = orig_abstention  # type: ignore[possibly-undefined]

    out_path = RESULTS_DIR / "conformal_test_coverage.json"
    save_json(result, out_path)
    log.info("Conformal test coverage saved to %s", out_path)


if __name__ == "__main__":
    main()
