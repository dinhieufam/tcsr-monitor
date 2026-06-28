#!/usr/bin/env python
"""Experiment E9: cross-sequence stress test.

Runs sequence-held-out monitor evaluations using cached feature tables:
  - clean leave-one-sequence-out over EndoVis sequences 1-10;
  - corrupted held-sequence evaluation over cached corrupted test sequences 8-10.

The corrupted cache is split by acquisition protocol in the existing project:
corrupted training features cover sequences 1-6, while corrupted test features
cover sequences 8-10. The corrupted part of this experiment is therefore a
held-test-sequence stress test over the available corrupted test sequences, not
a full leave-one-sequence-out corrupted retraining study over all ten videos.
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
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.monitor.classifiers import XGBMonitor, save_monitor


PROCESSED_DIR = Path("data/processed/endovis2017")
RESULTS_DIR = Path("results/metrics/sequence_stress")
FIGURES_DIR = Path("results/figures/sequence_stress")
EXPERIMENT_DIR = Path("experiments/sequence_stress")

TAU_COL = "failure_tau0_75"
SEED = 0
N_BOOT = 1000
FEATURE_PREFIXES = ("conf_", "shape_", "temp_", "qual_", "shift_")
FEATURE_SETS = ("full", "confidence_only")
TRAINING_CONDITIONS = ("clean_only", "combined")


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


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, metric: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    metric_fn = safe_auroc if metric == "auroc" else safe_auprc
    vals: list[float] = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), size=len(y))
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        vals.append(metric_fn(yb, scores[idx]))
    if vals:
        lo, hi = np.percentile(vals, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return {
        f"{metric}_ci_lo": float(lo),
        f"{metric}_ci_hi": float(hi),
        f"{metric}_n_boot": int(len(vals)),
    }


def bootstrap_delta_ci(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric: str,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    metric_fn = safe_auroc if metric == "auroc" else safe_auprc
    point = metric_fn(y, scores_a) - metric_fn(y, scores_b)
    vals: list[float] = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), size=len(y))
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        vals.append(metric_fn(yb, scores_a[idx]) - metric_fn(yb, scores_b[idx]))
    if vals:
        lo, hi = np.percentile(vals, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return {
        f"delta_{metric}": float(point),
        f"delta_{metric}_ci_lo": float(lo),
        f"delta_{metric}_ci_hi": float(hi),
        f"delta_{metric}_n_boot": int(len(vals)),
    }


def metric_row(
    scope: str,
    held_out_sequence: str,
    training_condition: str,
    feature_set: str,
    y: np.ndarray,
    scores: np.ndarray,
    extra: dict | None = None,
) -> dict:
    row = {
        "scope": scope,
        "held_out_sequence": held_out_sequence,
        "training_condition": training_condition,
        "feature_set": feature_set,
        "n": int(len(y)),
        "n_failures": int(y.sum()),
        "failure_rate": float(y.mean()),
        "auroc": safe_auroc(y, scores),
        "auprc": safe_auprc(y, scores),
        **bootstrap_ci(y, scores, "auroc", seed=SEED + 101),
        **bootstrap_ci(y, scores, "auprc", seed=SEED + 102),
    }
    if extra:
        row.update(extra)
    return row


def split_train_val(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[TAU_COL].fillna(0).values.astype(int)
    stratify = y if np.bincount(y, minlength=2).min() >= 2 else None
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def train_xgb(train_df: pd.DataFrame, val_df: pd.DataFrame, cols: list[str]) -> tuple[XGBMonitor, float]:
    monitor = XGBMonitor(n_estimators=400, max_depth=6, learning_rate=0.05, random_state=SEED)
    start = time.perf_counter()
    monitor.fit(
        train_df[cols].values.astype(np.float32),
        train_df[TAU_COL].fillna(0).values.astype(int),
        X_val=val_df[cols].values.astype(np.float32),
        y_val=val_df[TAU_COL].fillna(0).values.astype(int),
    )
    return monitor, float(time.perf_counter() - start)


def load_clean_df() -> pd.DataFrame:
    features = pd.read_parquet(PROCESSED_DIR / "features_all.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")[["frame_id", TAU_COL]]
    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")[
        ["frame_id", "video_id", "frame_idx_in_video", "split"]
    ]
    return features.merge(labels, on="frame_id").merge(manifest, on="frame_id")


def load_corrupted(kind: str) -> pd.DataFrame:
    if kind == "train":
        feat_path = PROCESSED_DIR / "corruption_feature_cache/corrupted_train_features.parquet"
        manifest_path = PROCESSED_DIR / "corrupted_train_manifest.parquet"
    elif kind == "test":
        feat_path = PROCESSED_DIR / "corruption_feature_cache/corrupted_test_features.parquet"
        manifest_path = PROCESSED_DIR / "corrupted_manifest.parquet"
    else:
        raise ValueError(kind)
    feat = pd.read_parquet(feat_path)
    manifest = pd.read_parquet(manifest_path)[
        ["frame_id", "corruption_type", "severity", "video_id", "frame_idx_in_video", "split"]
    ]
    return feat.merge(manifest, on=["frame_id", "corruption_type", "severity"], how="left")


def score_fold(
    scope_dir: Path,
    scope: str,
    held_seq: str,
    training_condition: str,
    feature_set: str,
    cols: list[str],
    train_pool: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata_cols: list[str],
    seed_offset: int,
) -> tuple[dict, pd.DataFrame]:
    fold_train, fold_val = split_train_val(train_pool, seed=SEED + seed_offset)
    monitor, train_seconds = train_xgb(fold_train, fold_val, cols)
    scores = monitor.predict_proba(test_df[cols].values.astype(np.float32)).astype(float)
    y_test = test_df[TAU_COL].fillna(0).values.astype(int)

    model_dir = scope_dir / held_seq / f"sequence_holdout_{held_seq}_{training_condition}_{feature_set}_seed{SEED}"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_monitor(monitor, model_dir / "monitor.pkl")
    (model_dir / "feature_columns.json").write_text(json.dumps(cols, indent=2))

    score_col = f"{training_condition}_{feature_set}"
    score_df = pd.DataFrame({col: test_df[col].values for col in metadata_cols if col in test_df.columns})
    score_df[TAU_COL] = y_test
    score_df[score_col] = scores
    score_df.to_parquet(model_dir / f"{scope}_{held_seq}_{training_condition}_{feature_set}_scores.parquet", index=False)

    row = metric_row(
        scope,
        held_seq,
        training_condition,
        feature_set,
        y_test,
        scores,
        {
            "score_column": score_col,
            "train_seconds": train_seconds,
            "n_train_pool": int(len(train_pool)),
            "n_train_pool_failures": int(train_pool[TAU_COL].fillna(0).sum()),
            "n_fit": int(len(fold_train)),
            "n_fit_failures": int(fold_train[TAU_COL].fillna(0).sum()),
            "n_val": int(len(fold_val)),
            "n_val_failures": int(fold_val[TAU_COL].fillna(0).sum()),
            "n_features": int(len(cols)),
        },
    )
    return row, score_df


def merge_scores(existing: pd.DataFrame | None, new_scores: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if existing is None:
        return new_scores
    return existing.merge(new_scores[keys + [c for c in new_scores.columns if c not in keys]], on=keys)


def clean_sequence_experiment(feature_sets: dict[str, list[str]], clean_df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    scope_dir = EXPERIMENT_DIR / "clean_tau75"
    scope_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    score_parts: list[pd.DataFrame] = []
    metadata_cols = ["frame_id", "video_id", "frame_idx_in_video"]

    for fold_idx, held_seq in enumerate(sorted(clean_df["video_id"].unique())):
        train_pool = clean_df[clean_df["video_id"] != held_seq].copy()
        test_df = clean_df[clean_df["video_id"] == held_seq].copy()
        fold_scores = None
        keys = ["frame_id", "video_id", "frame_idx_in_video", TAU_COL]
        for feature_set, cols in feature_sets.items():
            row, score_df = score_fold(
                scope_dir,
                "clean_loso_tau75",
                held_seq,
                "clean_only",
                feature_set,
                cols,
                train_pool,
                test_df,
                metadata_cols,
                seed_offset=1000 + fold_idx * 10 + len(rows),
            )
            rows.append(row)
            fold_scores = merge_scores(fold_scores, score_df, keys)
        assert fold_scores is not None
        fold_scores["held_out_sequence"] = held_seq
        (scope_dir / held_seq).mkdir(parents=True, exist_ok=True)
        fold_scores.to_parquet(scope_dir / held_seq / f"clean_loso_tau75_{held_seq}_scores.parquet", index=False)
        score_parts.append(fold_scores)

    scores = pd.concat(score_parts, ignore_index=True)
    scores.to_parquet(scope_dir / "clean_loso_tau75_sequence_scores.parquet", index=False)
    return rows, scores


def corrupted_sequence_experiment(
    feature_sets: dict[str, list[str]],
    clean_df: pd.DataFrame,
    corrupted_train: pd.DataFrame,
    corrupted_test: pd.DataFrame,
) -> tuple[list[dict], pd.DataFrame]:
    scope_dir = EXPERIMENT_DIR / "corrupted_tau75"
    scope_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    score_parts: list[pd.DataFrame] = []
    metadata_cols = ["frame_id", "video_id", "frame_idx_in_video", "corruption_type", "severity"]
    keys = ["frame_id", "video_id", "frame_idx_in_video", "corruption_type", "severity", TAU_COL]

    for fold_idx, held_seq in enumerate(sorted(corrupted_test["video_id"].dropna().unique())):
        test_df = corrupted_test[corrupted_test["video_id"] == held_seq].copy()
        fold_scores = None
        for training_condition in TRAINING_CONDITIONS:
            clean_pool = clean_df[clean_df["video_id"] != held_seq].copy()
            if training_condition == "clean_only":
                train_pool = clean_pool
            elif training_condition == "combined":
                corrupt_pool = corrupted_train[corrupted_train["video_id"] != held_seq].copy()
                train_pool = pd.concat([clean_pool, corrupt_pool], ignore_index=True)
            else:
                raise ValueError(training_condition)
            for feature_set, cols in feature_sets.items():
                row, score_df = score_fold(
                    scope_dir,
                    "corrupted_held_sequence_tau75",
                    held_seq,
                    training_condition,
                    feature_set,
                    cols,
                    train_pool,
                    test_df,
                    metadata_cols,
                    seed_offset=2000 + fold_idx * 100 + len(rows),
                )
                rows.append(row)
                fold_scores = merge_scores(fold_scores, score_df, keys)
        assert fold_scores is not None
        fold_scores["held_out_sequence"] = held_seq
        (scope_dir / held_seq).mkdir(parents=True, exist_ok=True)
        fold_scores.to_parquet(scope_dir / held_seq / f"corrupted_tau75_{held_seq}_scores.parquet", index=False)
        score_parts.append(fold_scores)

    scores = pd.concat(score_parts, ignore_index=True)
    scores.to_parquet(scope_dir / "corrupted_tau75_sequence_scores.parquet", index=False)
    return rows, scores


def add_macro_rows(fold_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    df = pd.DataFrame(fold_rows)
    for (scope, condition, feature_set), grp in df.groupby(["scope", "training_condition", "feature_set"], dropna=False):
        out.append(
            {
                "scope": f"{scope}_macro",
                "held_out_sequence": "macro_mean",
                "training_condition": condition,
                "feature_set": feature_set,
                "n": "",
                "n_failures": "",
                "failure_rate": "",
                "auroc": float(np.nanmean(grp["auroc"].astype(float))),
                "auroc_sequence_std": float(np.nanstd(grp["auroc"].astype(float), ddof=1)) if len(grp) > 1 else float("nan"),
                "auprc": float(np.nanmean(grp["auprc"].astype(float))),
                "auprc_sequence_std": float(np.nanstd(grp["auprc"].astype(float), ddof=1)) if len(grp) > 1 else float("nan"),
                "train_seconds": float(np.sum(grp["train_seconds"].astype(float))),
                "n_features": int(grp["n_features"].iloc[0]),
            }
        )
    return out


def add_micro_rows(clean_scores: pd.DataFrame, corrupted_scores: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    clean_y = clean_scores[TAU_COL].values.astype(int)
    for feature_set in FEATURE_SETS:
        col = f"clean_only_{feature_set}"
        out.append(metric_row("clean_loso_tau75_micro", "micro_pool", "clean_only", feature_set, clean_y, clean_scores[col].values.astype(float)))

    corr_y = corrupted_scores[TAU_COL].values.astype(int)
    for training_condition in TRAINING_CONDITIONS:
        for feature_set in FEATURE_SETS:
            col = f"{training_condition}_{feature_set}"
            out.append(
                metric_row(
                    "corrupted_held_sequence_tau75_micro",
                    "micro_pool",
                    training_condition,
                    feature_set,
                    corr_y,
                    corrupted_scores[col].values.astype(float),
                )
            )
    return out


def paired_deltas(clean_scores: pd.DataFrame, corrupted_scores: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    clean_y = clean_scores[TAU_COL].values.astype(int)
    rows.append(
        {
            "scope": "clean_loso_tau75_micro",
            "comparison": "full_minus_confidence_only",
            "training_condition": "clean_only",
            **bootstrap_delta_ci(clean_y, clean_scores["clean_only_full"].values.astype(float), clean_scores["clean_only_confidence_only"].values.astype(float), "auroc", SEED + 3001),
            **bootstrap_delta_ci(clean_y, clean_scores["clean_only_full"].values.astype(float), clean_scores["clean_only_confidence_only"].values.astype(float), "auprc", SEED + 3002),
        }
    )
    corr_y = corrupted_scores[TAU_COL].values.astype(int)
    for training_condition in TRAINING_CONDITIONS:
        rows.append(
            {
                "scope": "corrupted_held_sequence_tau75_micro",
                "comparison": "full_minus_confidence_only",
                "training_condition": training_condition,
                **bootstrap_delta_ci(corr_y, corrupted_scores[f"{training_condition}_full"].values.astype(float), corrupted_scores[f"{training_condition}_confidence_only"].values.astype(float), "auroc", SEED + 3011),
                **bootstrap_delta_ci(corr_y, corrupted_scores[f"{training_condition}_full"].values.astype(float), corrupted_scores[f"{training_condition}_confidence_only"].values.astype(float), "auprc", SEED + 3012),
            }
        )
    return rows


def positive_count_audit(clean_df: pd.DataFrame, corrupted_train: pd.DataFrame, corrupted_test: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for scope, df in [
        ("clean_all_sequences", clean_df),
        ("corrupted_train_cached", corrupted_train),
        ("corrupted_test_cached", corrupted_test),
    ]:
        grp = df.groupby("video_id")[TAU_COL].agg(["count", "sum", "mean"]).reset_index()
        grp.insert(0, "scope", scope)
        parts.append(grp.rename(columns={"count": "n", "sum": "n_failures", "mean": "failure_rate"}))
    return pd.concat(parts, ignore_index=True)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    preferred = [
        "scope",
        "held_out_sequence",
        "training_condition",
        "feature_set",
        "auroc",
        "auroc_ci_lo",
        "auroc_ci_hi",
        "auroc_sequence_std",
        "auprc",
        "auprc_ci_lo",
        "auprc_ci_hi",
        "auprc_sequence_std",
        "n",
        "n_failures",
        "failure_rate",
        "n_features",
        "train_seconds",
        "n_train_pool",
        "n_train_pool_failures",
        "n_fit",
        "n_fit_failures",
        "n_val",
        "n_val_failures",
        "score_column",
    ]
    fieldnames = [f for f in preferred if f in fieldnames] + [f for f in fieldnames if f not in preferred]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_sequence_dots(summary_rows: list[dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary_rows)
    folds = df[~df["held_out_sequence"].isin(["macro_mean", "micro_pool"])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), sharey=True)
    panels = [
        (axes[0], folds[(folds["scope"] == "clean_loso_tau75") & (folds["training_condition"] == "clean_only")], "Clean LOSO"),
        (axes[1], folds[(folds["scope"] == "corrupted_held_sequence_tau75") & (folds["feature_set"] == "full")], "Corrupted held sequence"),
    ]
    colors = {
        ("clean_only", "full"): "#4C78A8",
        ("clean_only", "confidence_only"): "#F58518",
        ("combined", "full"): "#54A24B",
        ("combined", "confidence_only"): "#B279A2",
    }
    for ax, sub, title in panels:
        labels = sorted(sub["held_out_sequence"].unique())
        x = np.arange(len(labels))
        for (condition, feature_set), grp in sub.groupby(["training_condition", "feature_set"]):
            y = []
            for seq in labels:
                row = grp[grp["held_out_sequence"] == seq]
                y.append(float(row["auprc"].iloc[0]) if len(row) else np.nan)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.4,
                label=f"{condition} / {feature_set}",
                color=colors.get((condition, feature_set)),
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("AUPRC")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sequence_stress_auprc_by_sequence.png", dpi=200)
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
    if isinstance(obj, float) and pd.isna(obj):
        return None
    return obj


def main() -> None:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    clean_df = load_clean_df()
    corrupted_train = load_corrupted("train")
    corrupted_test = load_corrupted("test")
    full_cols = feature_columns(clean_df)
    conf_cols = confidence_columns(clean_df)
    feature_sets = {"full": full_cols, "confidence_only": conf_cols}

    audit = positive_count_audit(clean_df, corrupted_train, corrupted_test)
    audit.to_csv(RESULTS_DIR / "sequence_positive_counts.csv", index=False)

    clean_rows, clean_scores = clean_sequence_experiment(feature_sets, clean_df)
    corrupted_rows, corrupted_scores = corrupted_sequence_experiment(feature_sets, clean_df, corrupted_train, corrupted_test)
    macro_rows = add_macro_rows(clean_rows + corrupted_rows)
    micro_rows = add_micro_rows(clean_scores, corrupted_scores)
    summary_rows = clean_rows + corrupted_rows + macro_rows + micro_rows
    deltas = paired_deltas(clean_scores, corrupted_scores)

    write_csv(summary_rows, RESULTS_DIR / "sequence_stress_summary.csv")
    write_csv(deltas, RESULTS_DIR / "sequence_stress_paired_deltas.csv")
    plot_sequence_dots(summary_rows)

    payload = {
        "experiment": "E9 Cross-Sequence Stress Test",
        "date": "2026-06-27",
        "tau_col": TAU_COL,
        "seed": SEED,
        "n_boot": N_BOOT,
        "feature_sets": {name: cols for name, cols in feature_sets.items()},
        "protocol": {
            "clean": "Leave-one-sequence-out over all clean EndoVis 2017 sequences, with validation split drawn from non-held training frames.",
            "corrupted": "Held-sequence evaluation over cached corrupted test sequences only: seq08, seq09, seq10.",
            "combined_training": "Clean non-held sequences plus cached corrupted training sequences 1-6.",
            "limitation": "Cached corrupted features do not cover all ten sequences; corrupted sequence stress is not a full corrupted LOSO study.",
        },
        "positive_count_audit": audit.to_dict(orient="records"),
        "summary_rows": summary_rows,
        "paired_deltas": deltas,
        "outputs": {
            "summary_csv": str(RESULTS_DIR / "sequence_stress_summary.csv"),
            "paired_deltas_csv": str(RESULTS_DIR / "sequence_stress_paired_deltas.csv"),
            "positive_counts_csv": str(RESULTS_DIR / "sequence_positive_counts.csv"),
            "clean_scores": str(EXPERIMENT_DIR / "clean_tau75" / "clean_loso_tau75_sequence_scores.parquet"),
            "corrupted_scores": str(EXPERIMENT_DIR / "corrupted_tau75" / "corrupted_tau75_sequence_scores.parquet"),
            "figure": str(FIGURES_DIR / "sequence_stress_auprc_by_sequence.png"),
        },
    }
    with (RESULTS_DIR / "sequence_stress_results.json").open("w") as f:
        json.dump(to_jsonable(payload), f, indent=2)

    summary = pd.DataFrame(summary_rows)
    print("E9 sequence stress test complete")
    for scope in ["clean_loso_tau75_micro", "clean_loso_tau75_macro", "corrupted_held_sequence_tau75_micro", "corrupted_held_sequence_tau75_macro"]:
        print(scope)
        sub = summary[summary["scope"] == scope]
        for _, row in sub.iterrows():
            print(
                f"  {row.training_condition}/{row.feature_set}: "
                f"AUROC={float(row.auroc):.3f}, AUPRC={float(row.auprc):.3f}"
            )
    print(f"Saved {RESULTS_DIR / 'sequence_stress_results.json'}")


if __name__ == "__main__":
    main()
