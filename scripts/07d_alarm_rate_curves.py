#!/usr/bin/env python
"""Experiment E5: frame-level alarm-rate and risk-coverage reporting.

This script extends the Mondrian conformal evaluation with operational alarm
burden metrics. It does not retrain the monitor. It uses cached monitor
features/scores, repeatedly calibrates global and group-conditional thresholds,
and reports the miss-rate cost together with false alarms on correct frames.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.conformal.split_conformal import calibrate_threshold
from tcsr.monitor.classifiers import load_monitor
from tcsr.utils.io import load_json, load_parquet, save_json
from tcsr.utils.seed import set_seed


ROOT = Path(".")
PROCESSED_DIR = ROOT / "data/processed/endovis2017"
CACHE_DIR = PROCESSED_DIR / "corruption_feature_cache"
METRICS_DIR = ROOT / "results/metrics/alarm_rates"
FIGURES_DIR = ROOT / "results/figures/alarm_rates"
MONITOR_DIR = ROOT / "experiments/combined_seed0"

TAU_COL = "failure_tau0_75"
ALPHAS = [0.05, 0.10, 0.20]
PRIMARY_ALPHA = 0.10
N_REPEATS = 200
CAL_FRACTION = 0.5
FPS = 25.0
RNG_SEED = 42


def _nan_summary(vals: list[float]) -> dict[str, float | int]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n": 0,
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n": int(arr.size),
    }


def _metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    alarms: np.ndarray,
    fps: float,
) -> dict[str, float | int]:
    labels = labels.astype(int)
    alarms = alarms.astype(int)
    fail = labels == 1
    correct = labels == 0
    n = int(labels.size)
    n_fail = int(fail.sum())
    n_correct = int(correct.sum())
    false_alarm_count = int(((alarms == 1) & correct).sum())
    missed_count = int(((alarms == 0) & fail).sum())
    alarm_count = int(alarms.sum())
    duration_minutes = n / (fps * 60.0)

    return {
        "n": n,
        "n_failures": n_fail,
        "n_correct": n_correct,
        "n_alarms": alarm_count,
        "n_missed_failures": missed_count,
        "n_false_alarms": false_alarm_count,
        "prevalence": float(n_fail / n) if n else float("nan"),
        "miss_rate": float(missed_count / n_fail) if n_fail else float("nan"),
        "failure_recall": float(1.0 - missed_count / n_fail) if n_fail else float("nan"),
        "false_alarm_rate": float(false_alarm_count / n_correct) if n_correct else float("nan"),
        "alarm_rate": float(alarm_count / n) if n else float("nan"),
        "alarms_per_minute": float(alarm_count / max(duration_minutes, 1e-12)),
        "false_alarms_per_minute": float(false_alarm_count / max(duration_minutes, 1e-12)),
        "precision": float(labels[alarms == 1].mean()) if alarm_count else float("nan"),
        "mean_score": float(np.mean(scores)) if n else float("nan"),
    }


def _calibrate_mondrian(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for g in sorted(np.unique(groups)):
        mask = groups == g
        if labels[mask].sum() == 0:
            thresholds[int(g)] = 1.0
            continue
        thresholds[int(g)] = calibrate_threshold(scores[mask], labels[mask], alpha=alpha)["threshold"]
    return thresholds


def _apply_mondrian(
    scores: np.ndarray,
    groups: np.ndarray,
    thresholds: dict[int, float],
    default_threshold: float,
) -> np.ndarray:
    alarms = np.zeros(scores.shape[0], dtype=int)
    for i, g in enumerate(groups):
        alarms[i] = int(scores[i] >= thresholds.get(int(g), default_threshold))
    return alarms


def _summarize_records(records: list[dict]) -> dict[str, dict]:
    metric_names = [
        "miss_rate",
        "failure_recall",
        "false_alarm_rate",
        "alarm_rate",
        "alarms_per_minute",
        "false_alarms_per_minute",
        "precision",
        "prevalence",
    ]
    out = {name: _nan_summary([float(r.get(name, float("nan"))) for r in records]) for name in metric_names}
    if records:
        first = records[0]
        out["counts"] = {
            "n_mean": _nan_summary([float(r["n"]) for r in records]),
            "n_failures_mean": _nan_summary([float(r["n_failures"]) for r in records]),
            "n_correct_mean": _nan_summary([float(r["n_correct"]) for r in records]),
        }
    return out


def _run_repeated_calibration(
    dataset_name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    group_labels: dict[int, str],
    alpha: float,
    rng: np.random.Generator,
) -> dict:
    n = len(scores)
    overall_records: dict[str, list[dict]] = {"global": [], "mondrian": []}
    group_records: dict[str, dict[int, list[dict]]] = {
        "global": {int(g): [] for g in sorted(np.unique(groups))},
        "mondrian": {int(g): [] for g in sorted(np.unique(groups))},
    }
    threshold_records: list[dict] = []

    for repeat in range(N_REPEATS):
        idx = np.arange(n)
        rng.shuffle(idx)
        n_cal = int(n * CAL_FRACTION)
        cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

        y_cal, s_cal, g_cal = labels[cal_idx], scores[cal_idx], groups[cal_idx]
        y_test, s_test, g_test = labels[test_idx], scores[test_idx], groups[test_idx]
        if y_cal.sum() == 0 or y_test.sum() == 0:
            continue

        global_threshold = calibrate_threshold(s_cal, y_cal, alpha=alpha)["threshold"]
        global_alarms = (s_test >= global_threshold).astype(int)
        mondrian_thresholds = _calibrate_mondrian(s_cal, y_cal, g_cal, alpha=alpha)
        mondrian_alarms = _apply_mondrian(s_test, g_test, mondrian_thresholds, global_threshold)

        threshold_records.append(
            {
                "repeat": repeat,
                "global_threshold": float(global_threshold),
                "mondrian_thresholds": {str(k): float(v) for k, v in mondrian_thresholds.items()},
            }
        )

        for method, alarms in (("global", global_alarms), ("mondrian", mondrian_alarms)):
            overall_records[method].append(_metrics(s_test, y_test, alarms, FPS))
            for g in sorted(np.unique(groups)):
                gm = g_test == g
                if gm.sum() == 0:
                    continue
                group_records[method][int(g)].append(_metrics(s_test[gm], y_test[gm], alarms[gm], FPS))

    return {
        "dataset": dataset_name,
        "alpha": alpha,
        "n_repeats_requested": N_REPEATS,
        "n_repeats_used": len(overall_records["global"]),
        "cal_fraction": CAL_FRACTION,
        "fps": FPS,
        "alarm_aggregation": "frame_level",
        "group_labels": {str(k): v for k, v in group_labels.items()},
        "overall": {method: _summarize_records(recs) for method, recs in overall_records.items()},
        "per_group": {
            str(g): {
                "label": group_labels.get(int(g), str(g)),
                "global": _summarize_records(group_records["global"][int(g)]),
                "mondrian": _summarize_records(group_records["mondrian"][int(g)]),
            }
            for g in sorted(np.unique(groups))
        },
        "thresholds": {
            "global": _nan_summary([float(r["global_threshold"]) for r in threshold_records]),
            "mondrian": {
                str(g): _nan_summary(
                    [
                        float(r["mondrian_thresholds"][str(g)])
                        for r in threshold_records
                        if str(g) in r["mondrian_thresholds"]
                    ]
                )
                for g in sorted(np.unique(groups))
            },
        },
    }


def _risk_coverage_curve(scores: np.ndarray, labels: np.ndarray, fps: float, n_points: int = 101) -> pd.DataFrame:
    thresholds = np.quantile(scores, np.linspace(1.0, 0.0, n_points))
    rows = []
    for threshold in thresholds:
        alarms = (scores >= threshold).astype(int)
        row = _metrics(scores, labels, alarms, fps)
        row["threshold"] = float(threshold)
        row["coverage"] = row["alarm_rate"]
        row["risk"] = row["miss_rate"]
        rows.append(row)
    curve = pd.DataFrame(rows).drop_duplicates(subset=["threshold", "coverage", "risk"])
    return curve.sort_values("coverage").reset_index(drop=True)


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    order = np.argsort(x[mask])
    return float(np.trapezoid(y[mask][order], x[mask][order]))


def _flatten_summary(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        for scope in ["overall"]:
            for method, summary in result[scope].items():
                rows.append(
                    {
                        "dataset": result["dataset"],
                        "alpha": result["alpha"],
                        "scope": scope,
                        "group": "all",
                        "group_label": "all",
                        "method": method,
                        **{
                            f"{metric}_{stat}": value
                            for metric, metric_summary in summary.items()
                            if isinstance(metric_summary, dict) and metric != "counts"
                            for stat, value in metric_summary.items()
                        },
                    }
                )
        for g, group_result in result["per_group"].items():
            for method in ["global", "mondrian"]:
                summary = group_result[method]
                rows.append(
                    {
                        "dataset": result["dataset"],
                        "alpha": result["alpha"],
                        "scope": "per_group",
                        "group": g,
                        "group_label": group_result["label"],
                        "method": method,
                        **{
                            f"{metric}_{stat}": value
                            for metric, metric_summary in summary.items()
                            if isinstance(metric_summary, dict) and metric != "counts"
                            for stat, value in metric_summary.items()
                        },
                    }
                )
    return pd.DataFrame(rows)


def _load_clean_quality_scores(monitor, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    features = load_parquet(PROCESSED_DIR / "features_all.parquet")
    labels = load_parquet(PROCESSED_DIR / "labels.parquet")
    manifest = load_parquet(PROCESSED_DIR / "manifest.parquet")[["frame_id", "split"]]
    df = features.merge(labels[["frame_id", TAU_COL]], on="frame_id").merge(manifest, on="frame_id")
    pool = df[df["split"].isin(["cal", "test"])].copy().reset_index(drop=True)

    for col in feat_cols:
        if col not in pool.columns:
            pool[col] = 0.0
    scores = monitor.predict_proba(pool[feat_cols].values.astype(np.float32))
    labels_arr = pool[TAU_COL].fillna(0).values.astype(int)

    blur = pool["qual_blur_score"].values
    q33, q67 = np.percentile(blur, [33, 67])
    groups = np.where(blur < q33, 0, np.where(blur < q67, 1, 2)).astype(int)
    group_labels = {
        0: "low-quality blur<p33",
        1: "mid-quality",
        2: "high-quality blur>p67",
    }
    return scores, labels_arr, groups, group_labels


def _load_corrupted_severity_scores(monitor, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    feat_df = pd.read_parquet(CACHE_DIR / "corrupted_test_features.parquet")
    for col in feat_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    scores = monitor.predict_proba(feat_df[feat_cols].values.astype(np.float32))
    labels = feat_df[TAU_COL].fillna(0).values.astype(int)
    groups = feat_df["severity"].values.astype(int)
    group_labels = {int(g): f"severity {int(g)}" for g in sorted(np.unique(groups))}
    return scores, labels, groups, group_labels


def _write_figures(summary_df: pd.DataFrame, curves: dict[str, pd.DataFrame]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    primary = summary_df[
        (summary_df["alpha"] == PRIMARY_ALPHA)
        & (summary_df["scope"] == "per_group")
        & (summary_df["dataset"] == "corrupted_by_severity")
    ].copy()
    if not primary.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        methods = ["global", "mondrian"]
        colors = {"global": "#4C78A8", "mondrian": "#F58518"}
        groups = sorted(primary["group"].unique(), key=lambda x: int(x))
        x = np.arange(len(groups))
        width = 0.36
        for offset, method in [(-width / 2, "global"), (width / 2, "mondrian")]:
            part = primary[primary["method"] == method].set_index("group").loc[groups]
            axes[0].bar(x + offset, part["miss_rate_mean"], width, label=method, color=colors[method])
            axes[1].bar(x + offset, part["false_alarm_rate_mean"], width, label=method, color=colors[method])
        axes[0].axhline(PRIMARY_ALPHA, color="#555555", linestyle="--", linewidth=1)
        axes[0].set_title("Miss-rate")
        axes[1].set_title("False-alarm rate")
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            ax.set_xlabel("Corruption severity")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Frame fraction")
        axes[1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "miss_vs_false_alarm_by_severity_alpha0_10.png", dpi=220)
        plt.close(fig)

    for dataset, curve in curves.items():
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.plot(curve["coverage"], curve["risk"], color="#4C78A8", linewidth=2)
        ax.set_xlabel("Alarm coverage / alarm rate")
        ax.set_ylabel("Miss-rate among failures")
        ax.set_title(dataset.replace("_", " "))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"risk_coverage_{dataset}.png", dpi=220)
        plt.close(fig)


def main() -> None:
    set_seed(RNG_SEED)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    monitor = load_monitor(MONITOR_DIR / "monitor.pkl")
    feat_cols = load_json(MONITOR_DIR / "feature_columns.json")
    rng = np.random.default_rng(RNG_SEED)

    datasets = {
        "corrupted_by_severity": _load_corrupted_severity_scores(monitor, feat_cols),
        "clean_by_quality": _load_clean_quality_scores(monitor, feat_cols),
    }

    all_results = []
    curves: dict[str, pd.DataFrame] = {}
    curve_summary = []
    for dataset_name, (scores, labels, groups, group_labels) in datasets.items():
        curve = _risk_coverage_curve(scores, labels, FPS)
        curves[dataset_name] = curve
        curve.to_csv(METRICS_DIR / f"risk_coverage_{dataset_name}.csv", index=False)
        curve_summary.append(
            {
                "dataset": dataset_name,
                "coverage_risk_auc": _auc(curve["coverage"].values, curve["risk"].values),
                "min_miss_rate": float(curve["risk"].min()),
                "max_false_alarms_per_minute": float(curve["false_alarms_per_minute"].max()),
                "n": int(len(labels)),
                "n_failures": int(labels.sum()),
                "prevalence": float(labels.mean()),
            }
        )

        for alpha in ALPHAS:
            result = _run_repeated_calibration(dataset_name, scores, labels, groups, group_labels, alpha, rng)
            all_results.append(result)
            alpha_tag = str(alpha).replace(".", "_")
            save_json(result, METRICS_DIR / f"alarm_rates_{dataset_name}_alpha{alpha_tag}.json")

    summary_df = _flatten_summary(all_results)
    summary_df.to_csv(METRICS_DIR / "alarm_rate_summary.csv", index=False)
    pd.DataFrame(curve_summary).to_csv(METRICS_DIR / "risk_coverage_auc.csv", index=False)

    results = {
        "experiment": "E5_alarm_rate_and_risk_coverage_reporting",
        "tau": 0.75,
        "alphas": ALPHAS,
        "primary_alpha": PRIMARY_ALPHA,
        "fps": FPS,
        "n_repeats": N_REPEATS,
        "cal_fraction": CAL_FRACTION,
        "alarm_aggregation": "frame_level",
        "limitation": (
            "Frame-level alarm rates may overcount sustained clinical events; "
            "event-level alert aggregation is not evaluated here."
        ),
        "datasets": all_results,
        "risk_coverage_auc": curve_summary,
    }
    save_json(results, METRICS_DIR / "alarm_rate_results.json")

    with (METRICS_DIR / "risk_coverage_curves.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                name: {col: df[col].tolist() for col in df.columns}
                for name, df in curves.items()
            },
            f,
            indent=2,
        )

    _write_figures(summary_df, curves)
    print(f"Wrote E5 alarm-rate results to {METRICS_DIR}")
    print(f"Wrote E5 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
