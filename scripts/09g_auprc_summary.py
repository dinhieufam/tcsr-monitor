#!/usr/bin/env python
"""Experiment E4: AUPRC and prevalence-aware reporting.

This script builds a conservative score registry for rare-event reporting:
  1. clean-test PR/AUPRC from recomputed main monitor and corrected baselines;
  2. LOCO tau=0.75 PR/AUPRC from E3 paired score files;
  3. aggregate-only circularity and SAM2 transfer rows where score-level PR
     curves are not cached.

Outputs are written under results/metrics/auprc/ and results/figures/auprc/.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.monitor.classifiers import load_monitor  # noqa: E402
from tcsr.utils.io import load_json  # noqa: E402


PROCESSED_DIR = Path("data/processed/endovis2017")
RESULTS_DIR = Path("results/metrics/auprc")
FIGURES_DIR = Path("results/figures/auprc")
E3_DIR = Path("experiments/confidence_only_baseline")
N_BOOT = 1000
SEED = 42
TAUS = [0.5, 0.75]

BASELINES = {
    "bl_max_softmax": "Max-softmax",
    "bl_entropy": "Entropy",
    "bl_temporal_heuristic": "Temporal heuristic",
}


def _tau_col(tau: float) -> str:
    return f"failure_tau{tau:.2f}".replace(".", "_")


def _tau_key(tau: float) -> str:
    return f"tau_{tau:.2f}".replace(".", "_")


def _safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y) or np.nanstd(scores) <= 1e-12:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _safe_auprc(y: np.ndarray, scores: np.ndarray) -> float:
    if y.sum() == 0 or np.nanstd(scores) <= 1e-12:
        return float("nan")
    return float(average_precision_score(y, scores))


def _bootstrap_metric_ci(
    y: np.ndarray,
    scores: np.ndarray,
    metric: str,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    metric_fn = _safe_auroc if metric == "auroc" else _safe_auprc
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yb = y[idx]
        sb = scores[idx]
        value = metric_fn(yb, sb)
        if not math.isnan(value):
            values.append(value)
    if len(values) < 10:
        return {f"{metric}_ci_lo": float("nan"), f"{metric}_ci_hi": float("nan"), "n_bootstraps": len(values)}
    return {
        f"{metric}_ci_lo": float(np.percentile(values, 2.5)),
        f"{metric}_ci_hi": float(np.percentile(values, 97.5)),
        "n_bootstraps": len(values),
    }


def _bootstrap_delta_ci(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric: str,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    metric_fn = _safe_auroc if metric == "auroc" else _safe_auprc
    point = metric_fn(y, scores_a) - metric_fn(y, scores_b)
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yb = y[idx]
        da = metric_fn(yb, scores_a[idx])
        db = metric_fn(yb, scores_b[idx])
        if not math.isnan(da) and not math.isnan(db):
            deltas.append(da - db)
    if len(deltas) < 10:
        lo = hi = float("nan")
    else:
        lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        f"delta_{metric}": float(point),
        f"delta_{metric}_ci_lo": float(lo),
        f"delta_{metric}_ci_hi": float(hi),
        f"delta_{metric}_n_boot": int(len(deltas)),
    }


def _metric_row(protocol: str, tau: float | None, model: str, y: np.ndarray, scores: np.ndarray, source: str) -> dict:
    auroc_ci = _bootstrap_metric_ci(y, scores, "auroc")
    auprc_ci = _bootstrap_metric_ci(y, scores, "auprc")
    return {
        "protocol": protocol,
        "tau": tau,
        "model": model,
        "source": source,
        "n_frames": int(len(y)),
        "n_failures": int(y.sum()),
        "failure_rate": float(np.mean(y)),
        "prevalence_baseline_auprc": float(np.mean(y)),
        "auroc": _safe_auroc(y, scores),
        "auroc_ci_lo": auroc_ci["auroc_ci_lo"],
        "auroc_ci_hi": auroc_ci["auroc_ci_hi"],
        "auprc": _safe_auprc(y, scores),
        "auprc_ci_lo": auprc_ci["auprc_ci_lo"],
        "auprc_ci_hi": auprc_ci["auprc_ci_hi"],
        "n_bootstraps": int(min(auroc_ci["n_bootstraps"], auprc_ci["n_bootstraps"])),
    }


def _clean_score_registry() -> dict[float, pd.DataFrame]:
    monitor = load_monitor(Path("experiments/combined_seed0/monitor.pkl"))
    feat_cols = load_json(Path("experiments/combined_seed0/feature_columns.json"))

    features = pd.read_parquet(PROCESSED_DIR / "features_all.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")
    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")[["frame_id", "split"]]
    baselines = pd.read_parquet(PROCESSED_DIR / "baselines_corrected.parquet")

    df = features.merge(labels, on="frame_id").merge(manifest, on="frame_id").merge(
        baselines, on="frame_id", how="left"
    )
    test_df = df[df["split"] == "test"].copy()
    test_df["TCSR-Monitor"] = monitor.predict_proba(test_df[feat_cols].values.astype(float))

    out: dict[float, pd.DataFrame] = {}
    for tau in TAUS:
        tau_col = _tau_col(tau)
        cols = ["frame_id", tau_col, "TCSR-Monitor", *BASELINES.keys()]
        scores_df = test_df[cols].copy()
        scores_df.to_parquet(RESULTS_DIR / f"clean_{_tau_key(tau)}_scores.parquet", index=False)
        out[tau] = scores_df
    return out


def _write_pr_curve(protocol: str, tau: float, model: str, y: np.ndarray, scores: np.ndarray) -> Path:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    padded_thresholds = np.concatenate([thresholds, [np.nan]])
    curve = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": padded_thresholds,
        }
    )
    safe_model = model.lower().replace(" ", "_").replace("-", "_")
    out = RESULTS_DIR / f"pr_curve_{protocol}_{_tau_key(tau)}_{safe_model}.csv"
    curve.to_csv(out, index=False)
    return out


def _plot_pr_curves(protocol: str, tau: float, curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    plt.figure(figsize=(5.2, 4.0))
    for model, (y, scores) in curves.items():
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = _safe_auprc(y, scores)
        plt.plot(recall, precision, linewidth=2, label=f"{model} AP={ap:.3f}")
    prevalence = float(next(iter(curves.values()))[0].mean())
    plt.axhline(prevalence, color="0.35", linestyle="--", linewidth=1.2, label=f"Prevalence={prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{protocol.replace('_', ' ').title()} tau={tau:.2f}")
    plt.ylim(0, 1.02)
    plt.xlim(0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7.5, loc="best")
    plt.tight_layout()
    out = FIGURES_DIR / f"pr_curve_{protocol}_{_tau_key(tau)}.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def _aggregate_only_rows() -> list[dict]:
    rows: list[dict] = []
    circularity_path = Path("results/metrics/circularity_control.json")
    if circularity_path.exists():
        circularity = json.loads(circularity_path.read_text())
        tau75 = circularity.get("tau_0_75", {})
        if tau75:
            for model, prefix in [("TCSR-Monitor", "within_corrupted_monitor"), ("Entropy", "within_corrupted_entropy")]:
                rows.append(
                    {
                        "protocol": "circularity_control_within_corrupted",
                        "tau": 0.75,
                        "model": model,
                        "source": "aggregate_only:circularity_control.json",
                        "n_frames": tau75.get("n_corrupted_total"),
                        "n_failures": tau75.get("n_corrupted_failed"),
                        "failure_rate": tau75.get("failure_rate"),
                        "prevalence_baseline_auprc": tau75.get("failure_rate"),
                        "auroc": tau75.get(f"{prefix}_auroc"),
                        "auroc_ci_lo": tau75.get(f"{prefix}_auroc_ci_lo"),
                        "auroc_ci_hi": tau75.get(f"{prefix}_auroc_ci_hi"),
                        "auprc": tau75.get(f"{prefix}_auprc"),
                        "auprc_ci_lo": float("nan"),
                        "auprc_ci_hi": float("nan"),
                        "n_bootstraps": None,
                    }
                )

    sam2_path = Path("results/metrics/sam2_transfer.json")
    if sam2_path.exists():
        sam2 = json.loads(sam2_path.read_text())
        for tau_key, tau in [("tau_0_50", 0.5), ("tau_0_75", 0.75)]:
            for key in ["unet_monitor_to_sam2_transfer", "entropy_baseline", "sam2_monitor_upper_bound"]:
                if key not in sam2.get(tau_key, {}):
                    continue
                r = sam2[tau_key][key]
                rows.append(
                    {
                        "protocol": "sam2_transfer",
                        "tau": tau,
                        "model": r.get("method", key),
                        "source": "aggregate_only:sam2_transfer.json",
                        "n_frames": r.get("n_total"),
                        "n_failures": r.get("n_failures"),
                        "failure_rate": r.get("failure_rate"),
                        "prevalence_baseline_auprc": r.get("failure_rate"),
                        "auroc": r.get("auroc"),
                        "auroc_ci_lo": r.get("auroc_ci_lo"),
                        "auroc_ci_hi": r.get("auroc_ci_hi"),
                        "auprc": r.get("auprc"),
                        "auprc_ci_lo": r.get("auprc_ci_lo"),
                        "auprc_ci_hi": r.get("auprc_ci_hi"),
                        "n_bootstraps": r.get("n_bootstraps"),
                    }
                )
    return rows


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    delta_rows: list[dict] = []
    curve_outputs: list[str] = []
    figure_outputs: list[str] = []

    clean_scores = _clean_score_registry()
    for tau, scores_df in clean_scores.items():
        tau_col = _tau_col(tau)
        y = scores_df[tau_col].values.astype(int)
        curve_models: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        model_cols = {"TCSR-Monitor": "TCSR-Monitor", **{label: col for col, label in BASELINES.items()}}

        for model, col in model_cols.items():
            scores = scores_df[col].values.astype(float)
            summary_rows.append(_metric_row("clean_test", tau, model, y, scores, "score_file"))
            curve_outputs.append(str(_write_pr_curve("clean_test", tau, model, y, scores)))
            curve_models[model] = (y, scores)

        full = scores_df["TCSR-Monitor"].values.astype(float)
        for model, col in {label: col for col, label in BASELINES.items()}.items():
            baseline_scores = scores_df[col].values.astype(float)
            delta_rows.append(
                {
                    "protocol": "clean_test",
                    "tau": tau,
                    "comparison": f"TCSR-Monitor minus {model}",
                    **_bootstrap_delta_ci(y, full, baseline_scores, "auroc"),
                    **_bootstrap_delta_ci(y, full, baseline_scores, "auprc"),
                }
            )

        figure_outputs.append(str(_plot_pr_curves("clean_test", tau, curve_models)))

    e3_clean_path = E3_DIR / "clean_tau75" / "clean_tau75_scores.parquet"
    if e3_clean_path.exists():
        e3_clean = pd.read_parquet(e3_clean_path)
        y = e3_clean["failure_tau0_75"].values.astype(int)
        for model, col in [
            ("E3 paired full monitor", "full_monitor"),
            ("Learned confidence-only", "learned_confidence_only"),
            ("Entropy", "entropy"),
        ]:
            summary_rows.append(
                _metric_row("clean_test_e3_paired", 0.75, model, y, e3_clean[col].values.astype(float), "E3_score_file")
            )

    e3_loco_path = E3_DIR / "loco_tau75" / "loco_tau75_scores.parquet"
    if e3_loco_path.exists():
        loco = pd.read_parquet(e3_loco_path)
        y = loco["failure_tau0_75"].values.astype(int)
        curve_models = {}
        for model, col in [
            ("TCSR-Monitor", "full_monitor"),
            ("Learned confidence-only", "learned_confidence_only"),
            ("Entropy", "entropy"),
        ]:
            scores = loco[col].values.astype(float)
            summary_rows.append(_metric_row("loco_tau75_micro", 0.75, model, y, scores, "E3_score_file"))
            curve_outputs.append(str(_write_pr_curve("loco_tau75_micro", 0.75, model, y, scores)))
            curve_models[model] = (y, scores)
        figure_outputs.append(str(_plot_pr_curves("loco_tau75_micro", 0.75, curve_models)))

        full = loco["full_monitor"].values.astype(float)
        for model, col in [("Learned confidence-only", "learned_confidence_only"), ("Entropy", "entropy")]:
            delta_rows.append(
                {
                    "protocol": "loco_tau75_micro",
                    "tau": 0.75,
                    "comparison": f"TCSR-Monitor minus {model}",
                    **_bootstrap_delta_ci(y, full, loco[col].values.astype(float), "auroc"),
                    **_bootstrap_delta_ci(y, full, loco[col].values.astype(float), "auprc"),
                }
            )

        macro_rows = []
        for held_out, group in loco.groupby("held_out"):
            yg = group["failure_tau0_75"].values.astype(int)
            for model, col in [
                ("TCSR-Monitor", "full_monitor"),
                ("Learned confidence-only", "learned_confidence_only"),
                ("Entropy", "entropy"),
            ]:
                macro_rows.append(
                    {
                        "held_out": held_out,
                        "model": model,
                        "auroc": _safe_auroc(yg, group[col].values.astype(float)),
                        "auprc": _safe_auprc(yg, group[col].values.astype(float)),
                        "failure_rate": float(yg.mean()),
                    }
                )
        pd.DataFrame(macro_rows).to_csv(RESULTS_DIR / "loco_tau75_by_fold_auprc.csv", index=False)

    summary_rows.extend(_aggregate_only_rows())

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "auprc_summary.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(RESULTS_DIR / "paired_deltas.csv", index=False)

    payload = {
        "experiment": "E4 AUPRC and Prevalence-Aware Reporting",
        "date": "2026-06-26",
        "n_boot": N_BOOT,
        "positive_class": "failure",
        "score_polarity": "higher score means higher predicted failure risk",
        "summary_csv": str(RESULTS_DIR / "auprc_summary.csv"),
        "paired_deltas_csv": str(RESULTS_DIR / "paired_deltas.csv"),
        "curve_outputs": curve_outputs,
        "figure_outputs": figure_outputs,
        "notes": [
            "Clean-test rows use corrected E2 baselines and the combined_seed0 main monitor.",
            "LOCO rows use E3 paired score files.",
            "Circularity and SAM2 rows are aggregate-only because score-level PR curves are not cached in those artifacts.",
        ],
        "summary": summary.to_dict(orient="records"),
        "paired_deltas": pd.DataFrame(delta_rows).to_dict(orient="records"),
    }
    for tau in TAUS:
        tau_rows = summary[(summary["protocol"] == "clean_test") & (summary["tau"] == tau)]
        with (RESULTS_DIR / f"auprc_clean_{_tau_key(tau)}.json").open("w") as f:
            json.dump(_json_ready(tau_rows.to_dict(orient="records")), f, indent=2)
    loco_rows = summary[summary["protocol"].astype(str).str.startswith("loco")]
    with (RESULTS_DIR / "auprc_loco_tau0_75.json").open("w") as f:
        json.dump(_json_ready(loco_rows.to_dict(orient="records")), f, indent=2)
    with (RESULTS_DIR / "auprc_results.json").open("w") as f:
        json.dump(_json_ready(payload), f, indent=2)

    print("E4 AUPRC/prevalence reporting complete")
    clean75 = summary[(summary["protocol"] == "clean_test") & (summary["tau"] == 0.75)]
    for _, row in clean75.iterrows():
        print(f"Clean tau=0.75 {row['model']}: AUROC={row['auroc']:.3f}, AUPRC={row['auprc']:.3f}, prevalence={row['failure_rate']:.3f}")
    loco75 = summary[(summary["protocol"] == "loco_tau75_micro") & (summary["tau"] == 0.75)]
    for _, row in loco75.iterrows():
        print(f"LOCO tau=0.75 micro {row['model']}: AUROC={row['auroc']:.3f}, AUPRC={row['auprc']:.3f}, prevalence={row['failure_rate']:.3f}")
    print(f"Saved {RESULTS_DIR / 'auprc_results.json'}")
    print(f"Saved {RESULTS_DIR / 'auprc_summary.csv'}")


if __name__ == "__main__":
    main()
