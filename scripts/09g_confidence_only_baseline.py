#!/usr/bin/env python
"""Experiment E3: learned confidence-only baseline.

This script trains the same XGBoost monitor with either the full feature set or
only confidence-derived features, then compares both against entropy on:
  1. clean EndoVis test frames at tau=0.75;
  2. leave-one-corruption-out (LOCO) corrupted-test folds at tau=0.75.

Outputs are written under results/metrics/confidence_only_baseline/ and
experiments/confidence_only_baseline/.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.monitor.classifiers import XGBMonitor, save_monitor


PROCESSED_DIR = Path("data/processed/endovis2017")
RESULTS_DIR = Path("results/metrics/confidence_only_baseline")
EXPERIMENT_DIR = Path("experiments/confidence_only_baseline")
TAU_COL = "failure_tau0_75"
SEED = 0
N_BOOT = 1000
CORRUPTION_TYPES = [
    "brightness",
    "contrast",
    "gaussian_blur",
    "gaussian_noise",
    "jpeg_compression",
    "motion_blur",
]
FEATURE_PREFIXES = ("conf_", "shape_", "temp_", "qual_", "shift_")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(c.startswith(prefix) for prefix in FEATURE_PREFIXES)]


def confidence_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("conf_")]


def safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, scores))


def safe_auprc(y: np.ndarray, scores: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, scores))


def metric_row(scope: str, model: str, y: np.ndarray, scores: np.ndarray, extra: dict | None = None) -> dict:
    row = {
        "scope": scope,
        "model": model,
        "n": int(len(y)),
        "n_failures": int(y.sum()),
        "failure_rate": float(np.mean(y)),
        "auroc": safe_auroc(y, scores),
        "auprc": safe_auprc(y, scores),
    }
    if extra:
        row.update(extra)
    return row


def train_xgb(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> XGBMonitor:
    monitor = XGBMonitor(n_estimators=400, max_depth=6, learning_rate=0.05, random_state=SEED)
    monitor.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return monitor


def bootstrap_delta_ci(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric: str,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas: list[float] = []
    metric_fn = safe_auroc if metric == "auroc" else safe_auprc
    point = metric_fn(y, scores_a) - metric_fn(y, scores_b)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        deltas.append(metric_fn(yb, scores_a[idx]) - metric_fn(yb, scores_b[idx]))

    if not deltas:
        lo = hi = float("nan")
    else:
        lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        f"delta_{metric}": float(point),
        f"delta_{metric}_ci_lo": float(lo),
        f"delta_{metric}_ci_hi": float(hi),
        f"delta_{metric}_n_boot": int(len(deltas)),
    }


def evaluate_scores(y: np.ndarray, full: np.ndarray, conf: np.ndarray, entropy: np.ndarray | None) -> dict:
    out = {
        "full": metric_row("aggregate", "full_monitor", y, full),
        "confidence_only": metric_row("aggregate", "learned_confidence_only", y, conf),
    }
    if entropy is not None:
        out["entropy"] = metric_row("aggregate", "entropy", y, entropy)
    out["full_minus_confidence_only"] = {
        **bootstrap_delta_ci(y, full, conf, "auroc"),
        **bootstrap_delta_ci(y, full, conf, "auprc"),
    }
    return out


def clean_test_experiment(full_cols: list[str], conf_cols: list[str]) -> tuple[dict, pd.DataFrame]:
    features = pd.read_parquet(PROCESSED_DIR / "features_all.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")[["frame_id", TAU_COL]]
    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")[["frame_id", "split"]]
    baselines = pd.read_parquet(PROCESSED_DIR / "baselines_corrected.parquet")

    df = features.merge(labels, on="frame_id").merge(manifest, on="frame_id")
    train_df = df[df["split"] == "train"]
    cal_df = df[df["split"] == "cal"]
    test_df = df[df["split"] == "test"].copy()

    y_train = train_df[TAU_COL].values.astype(int)
    y_cal = cal_df[TAU_COL].values.astype(int)
    y_test = test_df[TAU_COL].values.astype(int)

    full_monitor = train_xgb(
        train_df[full_cols].values.astype(np.float32),
        y_train,
        cal_df[full_cols].values.astype(np.float32),
        y_cal,
    )
    conf_monitor = train_xgb(
        train_df[conf_cols].values.astype(np.float32),
        y_train,
        cal_df[conf_cols].values.astype(np.float32),
        y_cal,
    )

    full_scores = full_monitor.predict_proba(test_df[full_cols].values.astype(np.float32))
    conf_scores = conf_monitor.predict_proba(test_df[conf_cols].values.astype(np.float32))
    entropy = baselines.set_index("frame_id").loc[test_df.frame_id, "bl_entropy"].values.astype(float)

    clean_dir = EXPERIMENT_DIR / "clean_tau75"
    clean_dir.mkdir(parents=True, exist_ok=True)
    save_monitor(full_monitor, clean_dir / "full_monitor.pkl")
    save_monitor(conf_monitor, clean_dir / "confidence_only_monitor.pkl")
    (clean_dir / "full_feature_columns.json").write_text(json.dumps(full_cols, indent=2))
    (clean_dir / "confidence_feature_columns.json").write_text(json.dumps(conf_cols, indent=2))

    scores_df = pd.DataFrame(
        {
            "frame_id": test_df.frame_id.values,
            TAU_COL: y_test,
            "full_monitor": full_scores,
            "learned_confidence_only": conf_scores,
            "entropy": entropy,
        }
    )
    scores_df.to_parquet(clean_dir / "clean_tau75_scores.parquet", index=False)

    result = evaluate_scores(y_test, full_scores, conf_scores, entropy)
    result["metadata"] = {
        "tau_col": TAU_COL,
        "n_train": int(len(train_df)),
        "n_cal": int(len(cal_df)),
        "n_test": int(len(test_df)),
        "n_train_failures": int(y_train.sum()),
        "n_cal_failures": int(y_cal.sum()),
        "n_test_failures": int(y_test.sum()),
    }
    return result, scores_df


def loco_experiment(full_cols: list[str], conf_cols: list[str]) -> tuple[dict, pd.DataFrame]:
    test_feat = pd.read_parquet(PROCESSED_DIR / "corruption_feature_cache/corrupted_test_features.parquet")
    train_feat = pd.read_parquet(PROCESSED_DIR / "corruption_feature_cache/corrupted_train_features.parquet")
    clean_features = pd.read_parquet(PROCESSED_DIR / "features_all.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")[["frame_id", TAU_COL]]
    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")[["frame_id", "split"]]

    clean_df = clean_features.merge(labels, on="frame_id").merge(manifest, on="frame_id")
    clean_train = clean_df[clean_df["split"] == "train"]
    clean_cal = clean_df[clean_df["split"] == "cal"]

    y_clean_train = clean_train[TAU_COL].values.astype(int)
    y_clean_cal = clean_cal[TAU_COL].values.astype(int)
    all_score_rows: list[pd.DataFrame] = []
    fold_summary: dict[str, dict] = {}

    loco_dir = EXPERIMENT_DIR / "loco_tau75"
    loco_dir.mkdir(parents=True, exist_ok=True)

    for held_out in CORRUPTION_TYPES:
        train_types = [t for t in CORRUPTION_TYPES if t != held_out]
        c_train = train_feat[train_feat["corruption_type"].isin(train_types)]
        y_c = c_train[TAU_COL].fillna(0).values.astype(int)

        y_train = np.concatenate([y_clean_train, y_c])
        full_train = np.vstack([
            clean_train[full_cols].values.astype(np.float32),
            c_train[full_cols].values.astype(np.float32),
        ])
        conf_train = np.vstack([
            clean_train[conf_cols].values.astype(np.float32),
            c_train[conf_cols].values.astype(np.float32),
        ])

        full_monitor = train_xgb(
            full_train,
            y_train,
            clean_cal[full_cols].values.astype(np.float32),
            y_clean_cal,
        )
        conf_monitor = train_xgb(
            conf_train,
            y_train,
            clean_cal[conf_cols].values.astype(np.float32),
            y_clean_cal,
        )

        held = test_feat[test_feat["corruption_type"] == held_out].copy()
        y_test = held[TAU_COL].fillna(0).values.astype(int)
        full_scores = full_monitor.predict_proba(held[full_cols].values.astype(np.float32))
        conf_scores = conf_monitor.predict_proba(held[conf_cols].values.astype(np.float32))
        entropy = held["entropy_score"].values.astype(float)

        fold_result = evaluate_scores(y_test, full_scores, conf_scores, entropy)
        fold_result["metadata"] = {
            "held_out": held_out,
            "train_types": train_types,
            "n_train": int(len(y_train)),
            "n_corrupted_train": int(len(c_train)),
            "n_test": int(len(held)),
            "n_train_failures": int(y_train.sum()),
            "n_test_failures": int(y_test.sum()),
        }
        fold_summary[held_out] = fold_result

        fold_dir = loco_dir / held_out
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_monitor(full_monitor, fold_dir / "full_monitor.pkl")
        save_monitor(conf_monitor, fold_dir / "confidence_only_monitor.pkl")

        all_score_rows.append(
            pd.DataFrame(
                {
                    "frame_id": held.frame_id.values,
                    "corruption_type": held.corruption_type.values,
                    "severity": held.severity.values.astype(int),
                    "held_out": held_out,
                    TAU_COL: y_test,
                    "full_monitor": full_scores,
                    "learned_confidence_only": conf_scores,
                    "entropy": entropy,
                }
            )
        )

    scores_df = pd.concat(all_score_rows, ignore_index=True)
    scores_df.to_parquet(loco_dir / "loco_tau75_scores.parquet", index=False)
    (loco_dir / "full_feature_columns.json").write_text(json.dumps(full_cols, indent=2))
    (loco_dir / "confidence_feature_columns.json").write_text(json.dumps(conf_cols, indent=2))

    y_all = scores_df[TAU_COL].values.astype(int)
    result = {
        "folds": fold_summary,
        "macro_average": {
            "full_monitor_auroc": float(np.nanmean([
                fold_summary[c]["full"]["auroc"] for c in CORRUPTION_TYPES
            ])),
            "confidence_only_auroc": float(np.nanmean([
                fold_summary[c]["confidence_only"]["auroc"] for c in CORRUPTION_TYPES
            ])),
            "entropy_auroc": float(np.nanmean([
                fold_summary[c]["entropy"]["auroc"] for c in CORRUPTION_TYPES
            ])),
            "full_monitor_auprc": float(np.nanmean([
                fold_summary[c]["full"]["auprc"] for c in CORRUPTION_TYPES
            ])),
            "confidence_only_auprc": float(np.nanmean([
                fold_summary[c]["confidence_only"]["auprc"] for c in CORRUPTION_TYPES
            ])),
            "entropy_auprc": float(np.nanmean([
                fold_summary[c]["entropy"]["auprc"] for c in CORRUPTION_TYPES
            ])),
        },
        "micro_average": evaluate_scores(
            y_all,
            scores_df["full_monitor"].values.astype(float),
            scores_df["learned_confidence_only"].values.astype(float),
            scores_df["entropy"].values.astype(float),
        ),
    }
    return result, scores_df


def write_summary_csv(clean_result: dict, loco_result: dict) -> None:
    rows = []
    for model_key, model_name in [
        ("entropy", "entropy"),
        ("confidence_only", "learned_confidence_only"),
        ("full", "full_monitor"),
    ]:
        clean = clean_result[model_key]
        rows.append({
            "scope": "clean_test_tau75",
            "model": model_name,
            "auroc": clean["auroc"],
            "auprc": clean["auprc"],
            "n": clean["n"],
            "n_failures": clean["n_failures"],
            "failure_rate": clean["failure_rate"],
        })

    macro = loco_result["macro_average"]
    for model_name, prefix in [
        ("entropy", "entropy"),
        ("learned_confidence_only", "confidence_only"),
        ("full_monitor", "full_monitor"),
    ]:
        rows.append({
            "scope": "loco_tau75_macro",
            "model": model_name,
            "auroc": macro[f"{prefix}_auroc"],
            "auprc": macro[f"{prefix}_auprc"],
            "n": "",
            "n_failures": "",
            "failure_rate": "",
        })

    micro = loco_result["micro_average"]
    for model_key, model_name in [
        ("entropy", "entropy"),
        ("confidence_only", "learned_confidence_only"),
        ("full", "full_monitor"),
    ]:
        row = micro[model_key]
        rows.append({
            "scope": "loco_tau75_micro",
            "model": model_name,
            "auroc": row["auroc"],
            "auprc": row["auprc"],
            "n": row["n"],
            "n_failures": row["n_failures"],
            "failure_rate": row["failure_rate"],
        })

    out_csv = RESULTS_DIR / "confidence_only_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scope", "model", "auroc", "auprc", "n", "n_failures", "failure_rate"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(PROCESSED_DIR / "features_all.parquet")
    full_cols = feature_columns(features)
    conf_cols = confidence_columns(features)
    if not conf_cols:
        raise RuntimeError("No confidence columns found. Expected columns with 'conf_' prefix.")

    clean_result, clean_scores = clean_test_experiment(full_cols, conf_cols)
    loco_result, loco_scores = loco_experiment(full_cols, conf_cols)

    payload = {
        "experiment": "E3 Learned Confidence-Only Baseline",
        "date": "2026-06-26",
        "tau_col": TAU_COL,
        "seed": SEED,
        "n_boot": N_BOOT,
        "feature_columns": {
            "full": full_cols,
            "confidence_only": conf_cols,
        },
        "clean_test_tau75": clean_result,
        "loco_tau75": loco_result,
        "outputs": {
            "clean_scores": str(EXPERIMENT_DIR / "clean_tau75" / "clean_tau75_scores.parquet"),
            "loco_scores": str(EXPERIMENT_DIR / "loco_tau75" / "loco_tau75_scores.parquet"),
            "summary_csv": str(RESULTS_DIR / "confidence_only_summary.csv"),
        },
    }

    with (RESULTS_DIR / "confidence_only_results.json").open("w") as f:
        json.dump(payload, f, indent=2)
    write_summary_csv(clean_result, loco_result)

    print("E3 learned confidence-only baseline complete")
    print(
        "Clean tau=0.75: "
        f"entropy AUROC={clean_result['entropy']['auroc']:.3f}, "
        f"confidence-only AUROC={clean_result['confidence_only']['auroc']:.3f}, "
        f"full AUROC={clean_result['full']['auroc']:.3f}"
    )
    print(
        "LOCO tau=0.75 macro: "
        f"entropy AUROC={loco_result['macro_average']['entropy_auroc']:.3f}, "
        f"confidence-only AUROC={loco_result['macro_average']['confidence_only_auroc']:.3f}, "
        f"full AUROC={loco_result['macro_average']['full_monitor_auroc']:.3f}"
    )
    print(f"Saved {RESULTS_DIR / 'confidence_only_results.json'}")
    print(f"Saved {RESULTS_DIR / 'confidence_only_summary.csv'}")


if __name__ == "__main__":
    main()
