#!/usr/bin/env python
"""Experiment E7: temporal leakage and order stress test.

Runs temporal controls over the cached feature registry:
  - normal: original per-video temporal features;
  - no_temporal: remove temporal columns from train/cal/test;
  - reset: replace every temporal row with first-frame defaults;
  - shuffled: permute temporal feature rows within video/condition groups.

The shuffled control preserves temporal-feature marginals but breaks alignment
with the current frame. It is a sanity check, not a deployable model variant.
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

from tcsr.monitor.classifiers import XGBMonitor, save_monitor


PROCESSED_DIR = Path("data/processed/endovis2017")
RESULTS_DIR = Path("results/metrics/temporal_stress")
FIGURES_DIR = Path("results/figures/temporal_stress")
EXPERIMENT_DIR = Path("experiments/temporal_stress")

TAU_COL = "failure_tau0_75"
SEED = 0
N_BOOT = 1000
TEMPORAL_MODES = ("normal", "no_temporal", "reset", "shuffled")
CORRUPTION_TYPES = [
    "brightness",
    "contrast",
    "gaussian_blur",
    "gaussian_noise",
    "jpeg_compression",
    "motion_blur",
]
FEATURE_PREFIXES = ("conf_", "shape_", "temp_", "qual_", "shift_")
TEMP_DEFAULTS = {
    "temp_prev_iou": 1.0,
    "temp_centroid_jump": 0.0,
    "temp_area_delta": 0.0,
    "temp_rolling_iou_mean": 1.0,
    "temp_rolling_iou_std": 0.0,
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(c.startswith(prefix) for prefix in FEATURE_PREFIXES)]


def temporal_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("temp_")]


def mode_feature_columns(cols: list[str], mode: str) -> list[str]:
    if mode == "no_temporal":
        return [c for c in cols if not c.startswith("temp_")]
    return cols


def apply_temporal_mode(df: pd.DataFrame, mode: str, group_cols: list[str], seed: int) -> pd.DataFrame:
    out = df.copy()
    temp_cols = temporal_columns(out)
    if mode in ("normal", "no_temporal"):
        return out
    if mode == "reset":
        for col in temp_cols:
            out[col] = TEMP_DEFAULTS.get(col, 0.0)
        return out
    if mode == "shuffled":
        rng = np.random.default_rng(seed)
        for _, idx in out.groupby(group_cols, sort=False).groups.items():
            idx_arr = np.array(list(idx))
            if len(idx_arr) <= 1:
                continue
            perm = rng.permutation(idx_arr)
            out.loc[idx_arr, temp_cols] = out.loc[perm, temp_cols].to_numpy()
        return out
    raise ValueError(f"Unknown temporal mode: {mode}")


def safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, scores))


def safe_auprc(y: np.ndarray, scores: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, scores))


def metric_row(scope: str, mode: str, y: np.ndarray, scores: np.ndarray, extra: dict | None = None) -> dict:
    row = {
        "scope": scope,
        "temporal_mode": mode,
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


def train_xgb(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tuple[XGBMonitor, float]:
    monitor = XGBMonitor(n_estimators=400, max_depth=6, learning_rate=0.05, random_state=SEED)
    start = time.perf_counter()
    monitor.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return monitor, float(time.perf_counter() - start)


def train_and_score(
    scope_dir: Path,
    mode: str,
    cols: list[str],
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scope: str,
    metadata_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    y_train = train_df[TAU_COL].fillna(0).values.astype(int)
    y_cal = cal_df[TAU_COL].fillna(0).values.astype(int)
    y_test = test_df[TAU_COL].fillna(0).values.astype(int)
    monitor, train_seconds = train_xgb(
        train_df[cols].values.astype(np.float32),
        y_train,
        cal_df[cols].values.astype(np.float32),
        y_cal,
    )
    scores = monitor.predict_proba(test_df[cols].values.astype(np.float32)).astype(float)

    model_dir = scope_dir / f"temporal_{mode}_seed{SEED}"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_monitor(monitor, model_dir / "monitor.pkl")
    (model_dir / "feature_columns.json").write_text(json.dumps(cols, indent=2))

    score_data = {col: test_df[col].values for col in metadata_cols if col in test_df.columns}
    score_data[TAU_COL] = y_test
    score_data[mode] = scores
    scores_df = pd.DataFrame(score_data)
    return (
        metric_row(
            scope,
            mode,
            y_test,
            scores,
            {
                "train_seconds": train_seconds,
                "n_train": int(len(y_train)),
                "n_train_failures": int(y_train.sum()),
                "n_cal": int(len(y_cal)),
                "n_cal_failures": int(y_cal.sum()),
                "n_features": int(len(cols)),
            },
        ),
        scores_df,
    )


def load_clean_df() -> pd.DataFrame:
    features = pd.read_parquet(PROCESSED_DIR / "features_all.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")[["frame_id", TAU_COL]]
    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")[
        ["frame_id", "video_id", "frame_idx_in_video", "split"]
    ]
    return features.merge(labels, on="frame_id").merge(manifest, on="frame_id")


def load_corrupted_features(kind: str) -> pd.DataFrame:
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
        ["frame_id", "corruption_type", "severity", "video_id", "frame_idx_in_video"]
    ]
    return feat.merge(manifest, on=["frame_id", "corruption_type", "severity"], how="left")


def clean_test_experiment(base_cols: list[str]) -> tuple[list[dict], pd.DataFrame]:
    clean_df = load_clean_df()
    train_base = clean_df[clean_df["split"] == "train"]
    cal_base = clean_df[clean_df["split"] == "cal"]
    test_base = clean_df[clean_df["split"] == "test"]
    rows: list[dict] = []
    scores = None
    scope_dir = EXPERIMENT_DIR / "clean_tau75"
    scope_dir.mkdir(parents=True, exist_ok=True)

    for mode in TEMPORAL_MODES:
        cols = mode_feature_columns(base_cols, mode)
        train_df = apply_temporal_mode(train_base, mode, ["video_id"], seed=SEED + 11)
        cal_df = apply_temporal_mode(cal_base, mode, ["video_id"], seed=SEED + 12)
        test_df = apply_temporal_mode(test_base, mode, ["video_id"], seed=SEED + 13)
        row, score_df = train_and_score(
            scope_dir,
            mode,
            cols,
            train_df,
            cal_df,
            test_df,
            "clean_tau75",
            ["frame_id", "video_id", "frame_idx_in_video"],
        )
        rows.append(row)
        if scores is None:
            scores = score_df
        else:
            scores = scores.merge(score_df[["frame_id", mode]], on="frame_id")

    assert scores is not None
    scores.to_parquet(scope_dir / "clean_tau75_temporal_scores.parquet", index=False)
    return rows, scores


def loco_experiment(base_cols: list[str]) -> tuple[list[dict], pd.DataFrame, dict]:
    clean_df = load_clean_df()
    clean_train = clean_df[clean_df["split"] == "train"]
    clean_cal = clean_df[clean_df["split"] == "cal"]
    c_train_base = load_corrupted_features("train")
    c_test_base = load_corrupted_features("test")

    scope_dir = EXPERIMENT_DIR / "loco_tau75"
    scope_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    score_parts: list[pd.DataFrame] = []
    fold_metrics: dict = {}

    for held_out in CORRUPTION_TYPES:
        train_types = [t for t in CORRUPTION_TYPES if t != held_out]
        c_train_fold = c_train_base[c_train_base["corruption_type"].isin(train_types)]
        held = c_test_base[c_test_base["corruption_type"] == held_out]
        fold_dir = scope_dir / held_out
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_scores = None
        fold_metrics[held_out] = []

        for mode in TEMPORAL_MODES:
            cols = mode_feature_columns(base_cols, mode)
            clean_train_mode = apply_temporal_mode(clean_train, mode, ["video_id"], seed=SEED + 101)
            c_train_mode = apply_temporal_mode(
                c_train_fold,
                mode,
                ["corruption_type", "severity", "video_id"],
                seed=SEED + 102,
            )
            train_df = pd.concat([clean_train_mode, c_train_mode], ignore_index=True)
            cal_df = apply_temporal_mode(clean_cal, mode, ["video_id"], seed=SEED + 103)
            test_df = apply_temporal_mode(
                held,
                mode,
                ["corruption_type", "severity", "video_id"],
                seed=SEED + 104,
            )
            row, score_df = train_and_score(
                fold_dir,
                mode,
                cols,
                train_df,
                cal_df,
                test_df,
                f"loco_tau75_{held_out}",
                ["frame_id", "corruption_type", "severity", "video_id", "frame_idx_in_video"],
            )
            row["held_out"] = held_out
            row["train_types"] = ",".join(train_types)
            rows.append(row)
            fold_metrics[held_out].append(row)

            if fold_scores is None:
                fold_scores = score_df.assign(held_out=held_out)
            else:
                fold_scores = fold_scores.merge(score_df[["frame_id", "corruption_type", "severity", mode]], on=["frame_id", "corruption_type", "severity"])

        assert fold_scores is not None
        fold_scores.to_parquet(fold_dir / f"loco_tau75_{held_out}_temporal_scores.parquet", index=False)
        score_parts.append(fold_scores)

    scores = pd.concat(score_parts, ignore_index=True)
    scores.to_parquet(scope_dir / "loco_tau75_temporal_scores.parquet", index=False)
    return rows, scores, fold_metrics


def add_macro_rows(loco_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    df = pd.DataFrame(loco_rows)
    for mode, grp in df.groupby("temporal_mode"):
        rows.append(
            {
                "scope": "loco_tau75_macro",
                "temporal_mode": mode,
                "n": "",
                "n_failures": "",
                "failure_rate": "",
                "auroc": float(np.nanmean(grp["auroc"].astype(float))),
                "auprc": float(np.nanmean(grp["auprc"].astype(float))),
                "train_seconds": float(np.sum(grp["train_seconds"].astype(float))),
                "n_train": "",
                "n_train_failures": "",
                "n_cal": "",
                "n_cal_failures": "",
                "n_features": int(grp["n_features"].iloc[0]),
            }
        )
    return rows


def add_micro_rows(loco_scores: pd.DataFrame) -> list[dict]:
    y = loco_scores[TAU_COL].values.astype(int)
    return [
        metric_row("loco_tau75_micro", mode, y, loco_scores[mode].values.astype(float))
        for mode in TEMPORAL_MODES
    ]


def paired_deltas(clean_scores: pd.DataFrame, loco_scores: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    for scope, score_df in [("clean_tau75", clean_scores), ("loco_tau75_micro", loco_scores)]:
        y = score_df[TAU_COL].values.astype(int)
        normal = score_df["normal"].values.astype(float)
        for mode in ("no_temporal", "reset", "shuffled"):
            control = score_df[mode].values.astype(float)
            out.append(
                {
                    "scope": scope,
                    "comparison": f"normal_minus_{mode}",
                    **bootstrap_delta_ci(y, normal, control, "auroc"),
                    **bootstrap_delta_ci(y, normal, control, "auprc"),
                }
            )
    return out


def temporal_audit(clean_scores: pd.DataFrame, loco_scores: pd.DataFrame) -> dict:
    audit = {}
    for name, df in [("clean_test", clean_scores), ("loco_test", loco_scores)]:
        first = df["frame_idx_in_video"] == 0
        audit[name] = {
            "n": int(len(df)),
            "n_first_frames": int(first.sum()),
            "first_frame_fraction": float(first.mean()),
        }
    return audit


def write_summary_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "scope",
        "temporal_mode",
        "held_out",
        "auroc",
        "auprc",
        "n",
        "n_failures",
        "failure_rate",
        "n_features",
        "train_seconds",
        "n_train",
        "n_train_failures",
        "n_cal",
        "n_cal_failures",
        "train_types",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_delta_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary_rows: list[dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary_rows)
    rows = df[df["scope"].isin(["clean_tau75", "loco_tau75_micro"])].copy()
    labels = ["Normal", "No temporal", "Reset", "Shuffled"]
    modes = list(TEMPORAL_MODES)
    x = np.arange(len(modes))

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    for ax, scope, title in [
        (axes[0], "clean_tau75", "Clean tau=0.75"),
        (axes[1], "loco_tau75_micro", "LOCO tau=0.75 micro"),
    ]:
        sub = rows[rows["scope"] == scope]
        auroc = [float(sub[sub["temporal_mode"] == mode]["auroc"].iloc[0]) for mode in modes]
        auprc = [float(sub[sub["temporal_mode"] == mode]["auprc"].iloc[0]) for mode in modes]
        width = 0.36
        ax.bar(x - width / 2, auroc, width, label="AUROC", color="#4C78A8")
        ax.bar(x + width / 2, auprc, width, label="AUPRC", color="#F58518")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.0)
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "temporal_stress_clean_loco.png", dpi=200)
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
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    clean_df = load_clean_df()
    base_cols = feature_columns(clean_df)
    temp_cols = [c for c in base_cols if c.startswith("temp_")]
    if not temp_cols:
        raise RuntimeError("No temporal feature columns found.")

    clean_rows, clean_scores = clean_test_experiment(base_cols)
    loco_rows, loco_scores, fold_metrics = loco_experiment(base_cols)
    macro_rows = add_macro_rows(loco_rows)
    micro_rows = add_micro_rows(loco_scores)
    summary_rows = clean_rows + loco_rows + macro_rows + micro_rows
    deltas = paired_deltas(clean_scores, loco_scores)
    audit = temporal_audit(clean_scores, loco_scores)

    write_summary_csv(summary_rows, RESULTS_DIR / "temporal_stress_summary.csv")
    write_delta_csv(deltas, RESULTS_DIR / "temporal_stress_paired_deltas.csv")
    plot_summary(summary_rows)

    payload = {
        "experiment": "E7 Temporal Leakage and Order Stress Test",
        "date": "2026-06-27",
        "tau_col": TAU_COL,
        "seed": SEED,
        "n_boot": N_BOOT,
        "temporal_modes": TEMPORAL_MODES,
        "temporal_columns": temp_cols,
        "control_definitions": {
            "normal": "Use cached per-video temporal features.",
            "no_temporal": "Drop all temp_* columns from train/cal/test.",
            "reset": "Set every temp_* column to first-frame default values.",
            "shuffled": "Permute temporal feature rows within video/condition groups; preserves marginals but breaks current-frame alignment.",
        },
        "summary_rows": summary_rows,
        "paired_deltas": deltas,
        "fold_metrics": fold_metrics,
        "temporal_audit": audit,
        "outputs": {
            "summary_csv": str(RESULTS_DIR / "temporal_stress_summary.csv"),
            "paired_deltas_csv": str(RESULTS_DIR / "temporal_stress_paired_deltas.csv"),
            "clean_scores": str(EXPERIMENT_DIR / "clean_tau75" / "clean_tau75_temporal_scores.parquet"),
            "loco_scores": str(EXPERIMENT_DIR / "loco_tau75" / "loco_tau75_temporal_scores.parquet"),
            "figure": str(FIGURES_DIR / "temporal_stress_clean_loco.png"),
        },
    }
    with (RESULTS_DIR / "temporal_stress_results.json").open("w") as f:
        json.dump(to_jsonable(payload), f, indent=2)

    summary = pd.DataFrame(summary_rows)
    print("E7 temporal stress test complete")
    for scope in ["clean_tau75", "loco_tau75_micro", "loco_tau75_macro"]:
        print(scope)
        for _, row in summary[summary["scope"] == scope].iterrows():
            print(f"  {row.temporal_mode}: AUROC={row.auroc:.3f}, AUPRC={row.auprc:.3f}")
    print(f"Saved {RESULTS_DIR / 'temporal_stress_results.json'}")


if __name__ == "__main__":
    main()
