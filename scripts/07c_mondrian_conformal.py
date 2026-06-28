#!/usr/bin/env python
"""
Stage 7c: Group-conditional (Mondrian) conformal calibration.

Demonstrates that a naive global conformal threshold loses coverage in severe
degradation bins, while group-conditional (Mondrian) conformal maintains
per-group coverage. This is the de-risking novelty contribution.

Groups are defined by:
  1. Image-quality severity bin (qual_blur_score + qual_brightness combined metric)
  2. Corruption severity level (1-5) on the corrupted test set

Procedure for corrupted test set:
  - Split corrupted test into cal/test (by frame, not condition)
  - Naive: one global threshold calibrated on cal
  - Mondrian: one threshold per severity level calibrated on cal
  - Compare per-severity realized miss-rate: naive degrades at severe levels,
    Mondrian holds per level

Also runs on clean data using quality bins derived from qual_blur_score.

Output: results/metrics/mondrian_conformal.json
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.conformal.split_conformal import calibrate_threshold
from tcsr.monitor.classifiers import load_monitor
from tcsr.utils.io import load_json, load_parquet, save_json
from tcsr.utils.logging import get_logger
from tcsr.utils.seed import set_seed

log = get_logger(__name__)

PROCESSED_DIR = Path("data/processed/endovis2017")
RESULTS_DIR = Path("results/metrics")
CACHE_DIR = PROCESSED_DIR / "corruption_feature_cache"

FEATURE_PREFIXES = ("conf_", "shape_", "temp_", "qual_", "shift_")
TAU_COL = "failure_tau0_75"
ALPHA = 0.10
N_REPEATS = 100
CAL_FRACTION = 0.5


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]


def calibrate_mondrian(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    alpha: float = 0.10,
) -> dict[int, float]:
    """Calibrate one threshold per group."""
    thresholds = {}
    for g in np.unique(groups):
        mask = groups == g
        if labels[mask].sum() == 0:
            thresholds[int(g)] = 0.0
            continue
        try:
            result = calibrate_threshold(scores[mask], labels[mask], alpha=alpha)
            thresholds[int(g)] = result["threshold"]
        except Exception:
            thresholds[int(g)] = 0.5
    return thresholds


def apply_mondrian(
    scores: np.ndarray,
    groups: np.ndarray,
    thresholds: dict[int, float],
    default_thresh: float = 0.5,
) -> np.ndarray:
    """Apply per-group thresholds."""
    alarms = np.zeros(len(scores), dtype=int)
    for g, thresh in thresholds.items():
        mask = groups == g
        alarms[mask] = (scores[mask] >= thresh).astype(int)
    # unseen groups
    for i, g in enumerate(groups):
        if int(g) not in thresholds:
            alarms[i] = int(scores[i] >= default_thresh)
    return alarms


def miss_rate_per_group(
    alarms: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Compute miss rate (P(no alarm | failure)) per group."""
    results = {}
    for g in sorted(np.unique(groups)):
        mask = (groups == g) & (labels == 1)
        if mask.sum() == 0:
            results[int(g)] = float("nan")
        else:
            results[int(g)] = float((alarms[mask] == 0).mean())
    return results


def run_experiment(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    n_repeats: int,
    rng: np.random.Generator,
) -> dict:
    """Run repeated cal/test splits for both naive and Mondrian."""
    naive_per_group: dict[int, list] = {int(g): [] for g in np.unique(groups)}
    mond_per_group: dict[int, list] = {int(g): [] for g in np.unique(groups)}
    naive_overall, mond_overall = [], []

    n = len(scores)

    for _ in range(n_repeats):
        idx = np.arange(n)
        rng.shuffle(idx)
        n_cal = int(n * CAL_FRACTION)
        cal_idx = idx[:n_cal]
        test_idx = idx[n_cal:]

        y_cal, s_cal, g_cal = labels[cal_idx], scores[cal_idx], groups[cal_idx]
        y_test, s_test, g_test = labels[test_idx], scores[test_idx], groups[test_idx]

        if y_cal.sum() == 0 or y_test.sum() == 0:
            continue

        # Naive global threshold
        try:
            naive_res = calibrate_threshold(s_cal, y_cal, alpha=alpha)
            naive_thresh = naive_res["threshold"]
        except Exception:
            naive_thresh = 0.5
        naive_alarms = (s_test >= naive_thresh).astype(int)

        # Mondrian per-group threshold
        mond_thresholds = calibrate_mondrian(s_cal, y_cal, g_cal, alpha=alpha)
        mond_alarms = apply_mondrian(s_test, g_test, mond_thresholds)

        # Per-group miss rates
        for g in np.unique(groups):
            gm = (g_test == g) & (y_test == 1)
            if gm.sum() == 0:
                continue
            naive_per_group[int(g)].append(float((naive_alarms[gm] == 0).mean()))
            mond_per_group[int(g)].append(float((mond_alarms[gm] == 0).mean()))

        # Overall miss rates
        fail_test = y_test == 1
        if fail_test.sum() > 0:
            naive_overall.append(float((naive_alarms[fail_test] == 0).mean()))
            mond_overall.append(float((mond_alarms[fail_test] == 0).mean()))

    def summarize(vals: list) -> dict:
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
        a = np.array(vals)
        return {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "ci_lo": float(np.percentile(a, 2.5)),
            "ci_hi": float(np.percentile(a, 97.5)),
            "coverage_ok": bool(np.percentile(a, 97.5) <= alpha),
        }

    return {
        "naive_overall": summarize(naive_overall),
        "mondrian_overall": summarize(mond_overall),
        "per_group": {
            str(g): {
                "naive": summarize(naive_per_group.get(int(g), [])),
                "mondrian": summarize(mond_per_group.get(int(g), [])),
            }
            for g in sorted(np.unique(groups))
        },
    }


def main() -> None:
    set_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    monitor_path = Path("experiments/combined_seed0/monitor.pkl")
    feat_json_path = Path("experiments/combined_seed0/feature_columns.json")
    if not monitor_path.exists():
        raise FileNotFoundError(f"Run 06b first. Missing: {monitor_path}")
    monitor = load_monitor(monitor_path)
    feat_cols = load_json(feat_json_path)

    rng = np.random.default_rng(42)
    results = {}

    # ─── Experiment A: Corruption severity bins on corrupted test set ───
    cache_path = CACHE_DIR / "corrupted_test_features.parquet"
    if cache_path.exists():
        log.info("Loading cached corrupted test features")
        feat_df = pd.read_parquet(cache_path)
    else:
        log.warning("Corrupted test feature cache not found. Run 09c first.")
        feat_df = None

    if feat_df is not None and TAU_COL in feat_df.columns:
        for col in feat_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0.0

        X = feat_df[feat_cols].values.astype(np.float32)
        y = feat_df[TAU_COL].fillna(0).values.astype(int)
        scores = monitor.predict_proba(X)
        groups_sev = feat_df["severity"].values.astype(int)

        log.info("=== Experiment A: Corrupted test, groups=severity levels ===")
        log.info("n=%d, n_failures=%d (%.1f%%)", len(y), y.sum(), 100 * y.mean())

        exp_a = run_experiment(scores, y, groups_sev, ALPHA, N_REPEATS, rng)
        results["corrupted_by_severity"] = {
            "description": "Groups = corruption severity (1-5); corrupted test set",
            "alpha": ALPHA,
            **exp_a,
        }

        log.info("Naive overall miss: %.3f [%.3f, %.3f] covered=%s",
                 exp_a["naive_overall"]["mean"],
                 exp_a["naive_overall"]["ci_lo"],
                 exp_a["naive_overall"]["ci_hi"],
                 exp_a["naive_overall"].get("coverage_ok"))
        log.info("Mondrian overall miss: %.3f [%.3f, %.3f] covered=%s",
                 exp_a["mondrian_overall"]["mean"],
                 exp_a["mondrian_overall"]["ci_lo"],
                 exp_a["mondrian_overall"]["ci_hi"],
                 exp_a["mondrian_overall"].get("coverage_ok"))

        for g_str, gv in exp_a["per_group"].items():
            log.info("  Sev %s | naive=%.3f [%.3f,%.3f] ok=%s | mondrian=%.3f [%.3f,%.3f] ok=%s",
                     g_str,
                     gv["naive"]["mean"], gv["naive"]["ci_lo"], gv["naive"]["ci_hi"],
                     gv["naive"].get("coverage_ok"),
                     gv["mondrian"]["mean"], gv["mondrian"]["ci_lo"], gv["mondrian"]["ci_hi"],
                     gv["mondrian"].get("coverage_ok"))

    # ─── Experiment B: Clean test with quality bins ───
    features_df = load_parquet(PROCESSED_DIR / "features_all.parquet")
    labels_df = load_parquet(PROCESSED_DIR / "labels.parquet")
    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")[["frame_id", "split"]]

    df = features_df.merge(labels_df[[TAU_COL, "frame_id"]], on="frame_id").merge(manifest, on="frame_id")
    # Pool cal + test for the experiment
    pool = df[df["split"].isin(["cal", "test"])].copy().reset_index(drop=True)

    for col in feat_cols:
        if col not in pool.columns:
            pool[col] = 0.0

    X_clean = pool[feat_cols].values.astype(np.float32)
    y_clean = pool[TAU_COL].values.astype(int)
    s_clean = monitor.predict_proba(X_clean)

    # Quality bins from blur_score (lower = more blurry = harder)
    if "qual_blur_score" in pool.columns:
        blur = pool["qual_blur_score"].values
        q33 = np.percentile(blur, 33)
        q67 = np.percentile(blur, 67)
        quality_group = np.where(blur < q33, 0, np.where(blur < q67, 1, 2))
        group_labels_b = {0: "low-quality (blur<p33)", 1: "mid-quality", 2: "high-quality (blur>p67)"}
    else:
        quality_group = np.zeros(len(pool), dtype=int)
        group_labels_b = {0: "all"}

    log.info("=== Experiment B: Clean cal+test, groups=image quality bins ===")
    log.info("n=%d, n_failures=%d (%.1f%%)", len(y_clean), y_clean.sum(), 100 * y_clean.mean())

    exp_b = run_experiment(s_clean, y_clean, quality_group, ALPHA, N_REPEATS, rng)
    results["clean_by_quality"] = {
        "description": "Groups = image quality bins (blur_score tertiles); clean cal+test",
        "alpha": ALPHA,
        "quality_bins": group_labels_b,
        **exp_b,
    }

    log.info("Naive overall miss: %.3f [%.3f, %.3f] covered=%s",
             exp_b["naive_overall"]["mean"],
             exp_b["naive_overall"]["ci_lo"],
             exp_b["naive_overall"]["ci_hi"],
             exp_b["naive_overall"].get("coverage_ok"))
    log.info("Mondrian overall miss: %.3f [%.3f, %.3f] covered=%s",
             exp_b["mondrian_overall"]["mean"],
             exp_b["mondrian_overall"]["ci_lo"],
             exp_b["mondrian_overall"]["ci_hi"],
             exp_b["mondrian_overall"].get("coverage_ok"))

    for g_str, gv in exp_b["per_group"].items():
        g_label = group_labels_b.get(int(g_str), f"group {g_str}")
        log.info("  %s | naive=%.3f [%.3f,%.3f] ok=%s | mondrian=%.3f [%.3f,%.3f] ok=%s",
                 g_label,
                 gv["naive"]["mean"], gv["naive"]["ci_lo"], gv["naive"]["ci_hi"],
                 gv["naive"].get("coverage_ok"),
                 gv["mondrian"]["mean"], gv["mondrian"]["ci_lo"], gv["mondrian"]["ci_hi"],
                 gv["mondrian"].get("coverage_ok"))

    out_path = RESULTS_DIR / "mondrian_conformal.json"
    save_json(results, out_path)
    log.info("Mondrian conformal results saved to %s", out_path)


if __name__ == "__main__":
    main()
