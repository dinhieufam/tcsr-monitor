#!/usr/bin/env python
"""Experiment E6: learner ablation.

Compare logistic regression, random forest, and XGBoost on the same monitor
feature representation under clean-test and LOCO corrupted-test protocols.
The goal is to check whether the monitoring signal is tied specifically to
XGBoost or is recoverable by simpler/alternative classifiers.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.monitor.classifiers import LogRegMonitor, RFMonitor, XGBMonitor, save_monitor


PROCESSED_DIR = Path("data/processed/endovis2017")
RESULTS_DIR = Path("results/metrics/learner_ablation")
FIGURES_DIR = Path("results/figures/learner_ablation")
EXPERIMENT_DIR = Path("experiments/learner_ablation")

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
LEARNERS = ("logreg", "random_forest", "xgboost")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(c.startswith(prefix) for prefix in FEATURE_PREFIXES)]


def confidence_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("conf_")]


def make_learner(name: str):
    if name == "logreg":
        return LogRegMonitor(C=1.0, max_iter=2000, class_weight="balanced")
    if name == "random_forest":
        monitor = RFMonitor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
        )
        monitor.clf.set_params(random_state=SEED)
        return monitor
    if name == "xgboost":
        return XGBMonitor(n_estimators=400, max_depth=6, learning_rate=0.05, random_state=SEED)
    raise ValueError(f"Unknown learner: {name}")


def fit_learner(name: str, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    monitor = make_learner(name)
    start = time.perf_counter()
    if name == "xgboost":
        monitor.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        monitor.fit(X_train, y_train)
    return monitor, float(time.perf_counter() - start)


def safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, scores))


def safe_auprc(y: np.ndarray, scores: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, scores))


def metric_row(scope: str, feature_set: str, learner: str, y: np.ndarray, scores: np.ndarray, extra: dict | None = None) -> dict:
    row = {
        "scope": scope,
        "feature_set": feature_set,
        "learner": learner,
        "n": int(len(y)),
        "n_failures": int(y.sum()),
        "failure_rate": float(y.mean()),
        "auroc": safe_auroc(y, scores),
        "auprc": safe_auprc(y, scores),
    }
    if extra:
        row.update(extra)
    return row


def bootstrap_delta_ci(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric: str,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    metric_fn = safe_auroc if metric == "auroc" else safe_auprc
    point = metric_fn(y, scores_a) - metric_fn(y, scores_b)
    deltas: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        deltas.append(metric_fn(yb, scores_a[idx]) - metric_fn(yb, scores_b[idx]))

    if deltas:
        lo, hi = np.percentile(deltas, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return {
        f"delta_{metric}": float(point),
        f"delta_{metric}_ci_lo": float(lo),
        f"delta_{metric}_ci_hi": float(hi),
        f"delta_{metric}_n_boot": int(len(deltas)),
    }


def train_and_score(
    scope_dir: Path,
    feature_set: str,
    cols: list[str],
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scope: str,
    metadata_cols: list[str],
) -> tuple[list[dict], pd.DataFrame]:
    y_train = train_df[TAU_COL].fillna(0).values.astype(int)
    y_cal = cal_df[TAU_COL].fillna(0).values.astype(int)
    y_test = test_df[TAU_COL].fillna(0).values.astype(int)

    X_train = train_df[cols].values.astype(np.float32)
    X_cal = cal_df[cols].values.astype(np.float32)
    X_test = test_df[cols].values.astype(np.float32)

    score_data = {
        col: test_df[col].values
        for col in metadata_cols
        if col in test_df.columns
    }
    score_data[TAU_COL] = y_test
    rows: list[dict] = []

    for learner in LEARNERS:
        learner_dir = scope_dir / f"learner_ablation_{learner}_{feature_set}_seed{SEED}"
        learner_dir.mkdir(parents=True, exist_ok=True)
        monitor, train_seconds = fit_learner(learner, X_train, y_train, X_cal, y_cal)
        scores = monitor.predict_proba(X_test).astype(float)

        save_monitor(monitor, learner_dir / "monitor.pkl")
        (learner_dir / "feature_columns.json").write_text(json.dumps(cols, indent=2))
        score_data[learner] = scores
        rows.append(
            metric_row(
                scope,
                feature_set,
                learner,
                y_test,
                scores,
                {
                    "train_seconds": train_seconds,
                    "n_train": int(len(y_train)),
                    "n_train_failures": int(y_train.sum()),
                    "n_cal": int(len(y_cal)),
                    "n_cal_failures": int(y_cal.sum()),
                },
            )
        )

    scores_df = pd.DataFrame(score_data)
    scores_df.to_parquet(scope_dir / f"{scope}_{feature_set}_scores.parquet", index=False)
    return rows, scores_df


def clean_test_experiment(feature_sets: dict[str, list[str]]) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    features = pd.read_parquet(PROCESSED_DIR / "features_all.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")[["frame_id", TAU_COL]]
    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")[["frame_id", "video_id", "split"]]
    df = features.merge(labels, on="frame_id").merge(manifest, on="frame_id")

    train_df = df[df["split"] == "train"]
    cal_df = df[df["split"] == "cal"]
    test_df = df[df["split"] == "test"].copy()
    scope_dir = EXPERIMENT_DIR / "clean_tau75"
    scope_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    scores_by_feature_set: dict[str, pd.DataFrame] = {}
    for feature_set, cols in feature_sets.items():
        feature_rows, scores_df = train_and_score(
            scope_dir,
            feature_set,
            cols,
            train_df,
            cal_df,
            test_df,
            "clean_tau75",
            ["frame_id", "video_id"],
        )
        rows.extend(feature_rows)
        scores_by_feature_set[feature_set] = scores_df
    return rows, scores_by_feature_set


def loco_experiment(feature_sets: dict[str, list[str]]) -> tuple[list[dict], dict[str, pd.DataFrame], dict]:
    test_feat = pd.read_parquet(PROCESSED_DIR / "corruption_feature_cache/corrupted_test_features.parquet")
    train_feat = pd.read_parquet(PROCESSED_DIR / "corruption_feature_cache/corrupted_train_features.parquet")
    clean_features = pd.read_parquet(PROCESSED_DIR / "features_all.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")[["frame_id", TAU_COL]]
    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")[["frame_id", "video_id", "split"]]

    clean_df = clean_features.merge(labels, on="frame_id").merge(manifest, on="frame_id")
    clean_train = clean_df[clean_df["split"] == "train"]
    clean_cal = clean_df[clean_df["split"] == "cal"]

    scope_dir = EXPERIMENT_DIR / "loco_tau75"
    scope_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    all_scores: dict[str, list[pd.DataFrame]] = {feature_set: [] for feature_set in feature_sets}
    fold_metrics: dict = {}

    for held_out in CORRUPTION_TYPES:
        train_types = [t for t in CORRUPTION_TYPES if t != held_out]
        c_train = train_feat[train_feat["corruption_type"].isin(train_types)]
        held = test_feat[test_feat["corruption_type"] == held_out].copy()

        train_df = pd.concat([clean_train, c_train], ignore_index=True)
        cal_df = clean_cal
        fold_dir = scope_dir / held_out
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_metrics[held_out] = {}

        for feature_set, cols in feature_sets.items():
            feature_rows, scores_df = train_and_score(
                fold_dir,
                feature_set,
                cols,
                train_df,
                cal_df,
                held,
                f"loco_tau75_{held_out}",
                ["frame_id", "corruption_type", "severity"],
            )
            for row in feature_rows:
                row["held_out"] = held_out
                row["train_types"] = ",".join(train_types)
            metric_rows.extend(feature_rows)
            all_scores[feature_set].append(scores_df.assign(held_out=held_out))
            fold_metrics[held_out][feature_set] = feature_rows

    score_tables = {
        feature_set: pd.concat(frames, ignore_index=True)
        for feature_set, frames in all_scores.items()
    }
    for feature_set, scores_df in score_tables.items():
        scores_df.to_parquet(scope_dir / f"loco_tau75_{feature_set}_scores.parquet", index=False)

    return metric_rows, score_tables, fold_metrics


def add_micro_rows(score_tables: dict[str, pd.DataFrame]) -> list[dict]:
    rows: list[dict] = []
    for feature_set, scores_df in score_tables.items():
        y = scores_df[TAU_COL].values.astype(int)
        for learner in LEARNERS:
            rows.append(metric_row("loco_tau75_micro", feature_set, learner, y, scores_df[learner].values.astype(float)))
    return rows


def add_macro_rows(loco_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    df = pd.DataFrame(loco_rows)
    for (feature_set, learner), grp in df.groupby(["feature_set", "learner"]):
        rows.append(
            {
                "scope": "loco_tau75_macro",
                "feature_set": feature_set,
                "learner": learner,
                "n": "",
                "n_failures": "",
                "failure_rate": "",
                "auroc": float(np.nanmean(grp["auroc"].values.astype(float))),
                "auprc": float(np.nanmean(grp["auprc"].values.astype(float))),
                "train_seconds": float(np.sum(grp["train_seconds"].values.astype(float))),
                "n_train": "",
                "n_train_failures": "",
                "n_cal": "",
                "n_cal_failures": "",
            }
        )
    return rows


def paired_deltas(summary_rows: list[dict], scores_by_scope: dict[str, dict[str, pd.DataFrame]]) -> list[dict]:
    deltas: list[dict] = []
    for scope_name, feature_tables in scores_by_scope.items():
        for feature_set, scores_df in feature_tables.items():
            y = scores_df[TAU_COL].values.astype(int)
            xgb = scores_df["xgboost"].values.astype(float)
            for learner in ("logreg", "random_forest"):
                scores = scores_df[learner].values.astype(float)
                deltas.append(
                    {
                        "scope": scope_name,
                        "feature_set": feature_set,
                        "comparison": f"xgboost_minus_{learner}",
                        **bootstrap_delta_ci(y, xgb, scores, "auroc"),
                        **bootstrap_delta_ci(y, xgb, scores, "auprc"),
                    }
                )
    return deltas


def write_summary_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "scope",
        "feature_set",
        "learner",
        "held_out",
        "auroc",
        "auprc",
        "n",
        "n_failures",
        "failure_rate",
        "train_seconds",
        "n_train",
        "n_train_failures",
        "n_cal",
        "n_cal_failures",
        "train_types",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_delta_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_loco_micro(summary_rows: list[dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary_rows)
    df = df[(df["scope"] == "loco_tau75_micro") & (df["feature_set"] == "full")]
    learners = list(LEARNERS)
    x = np.arange(len(learners))
    auroc = [float(df[df["learner"] == learner]["auroc"].iloc[0]) for learner in learners]
    auprc = [float(df[df["learner"] == learner]["auprc"].iloc[0]) for learner in learners]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    width = 0.36
    ax.bar(x - width / 2, auroc, width, label="AUROC", color="#4C78A8")
    ax.bar(x + width / 2, auprc, width, label="AUPRC", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(["LogReg", "Random forest", "XGBoost"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Learner ablation, LOCO tau=0.75")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "learner_ablation_loco_tau75_full.png", dpi=200)
    plt.close(fig)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main() -> None:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(PROCESSED_DIR / "features_all.parquet")
    feature_sets = {
        "full": feature_columns(features),
        "confidence_only": confidence_columns(features),
    }
    if not feature_sets["full"] or not feature_sets["confidence_only"]:
        raise RuntimeError("Missing expected full or confidence-only feature columns.")

    clean_rows, clean_scores = clean_test_experiment(feature_sets)
    loco_rows, loco_scores, fold_metrics = loco_experiment(feature_sets)
    micro_rows = add_micro_rows(loco_scores)
    macro_rows = add_macro_rows(loco_rows)

    summary_rows = clean_rows + loco_rows + macro_rows + micro_rows
    deltas = paired_deltas(
        summary_rows,
        {
            "clean_tau75": clean_scores,
            "loco_tau75_micro": loco_scores,
        },
    )

    write_summary_csv(summary_rows, RESULTS_DIR / "learner_ablation_summary.csv")
    write_delta_csv(deltas, RESULTS_DIR / "learner_ablation_paired_deltas.csv")
    plot_loco_micro(summary_rows)

    payload = {
        "experiment": "E6 Learner Ablation",
        "date": "2026-06-27",
        "tau_col": TAU_COL,
        "seed": SEED,
        "n_boot": N_BOOT,
        "learners": LEARNERS,
        "feature_sets": feature_sets,
        "summary_rows": summary_rows,
        "paired_deltas": deltas,
        "fold_metrics": fold_metrics,
        "outputs": {
            "summary_csv": str(RESULTS_DIR / "learner_ablation_summary.csv"),
            "paired_deltas_csv": str(RESULTS_DIR / "learner_ablation_paired_deltas.csv"),
            "clean_scores_dir": str(EXPERIMENT_DIR / "clean_tau75"),
            "loco_scores_dir": str(EXPERIMENT_DIR / "loco_tau75"),
            "figure": str(FIGURES_DIR / "learner_ablation_loco_tau75_full.png"),
        },
    }
    with (RESULTS_DIR / "learner_ablation_results.json").open("w") as f:
        json.dump(to_jsonable(payload), f, indent=2)

    summary = pd.DataFrame(summary_rows)
    clean_full = summary[(summary.scope == "clean_tau75") & (summary.feature_set == "full")]
    loco_full = summary[(summary.scope == "loco_tau75_micro") & (summary.feature_set == "full")]
    print("E6 learner ablation complete")
    print("Clean tau=0.75 full features:")
    for _, row in clean_full.iterrows():
        print(f"  {row.learner}: AUROC={row.auroc:.3f}, AUPRC={row.auprc:.3f}")
    print("LOCO tau=0.75 micro full features:")
    for _, row in loco_full.iterrows():
        print(f"  {row.learner}: AUROC={row.auroc:.3f}, AUPRC={row.auprc:.3f}")
    print(f"Saved {RESULTS_DIR / 'learner_ablation_results.json'}")


if __name__ == "__main__":
    main()
