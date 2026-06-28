"""Generate all paper figures from saved metrics (no recompute)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tcsr.utils.io import load_json
from tcsr.utils.logging import get_logger

log = get_logger(__name__)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = sns.color_palette("tab10")


def _save(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", path)


def fig_coverage_risk(metrics_dir: Path, output_dir: Path) -> None:
    """Fig 3: Coverage–risk curves for monitor vs. baselines."""
    curves_path = metrics_dir / "coverage_risk_curves.json"
    if not curves_path.exists():
        log.warning("Missing %s, skipping fig_coverage_risk", curves_path)
        return

    data = load_json(curves_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (name, curve) in enumerate(data.items()):
        ax.plot(curve["coverage"], curve["risk"], label=name, color=PALETTE[i % len(PALETTE)])
    ax.axhline(0.1, ls="--", color="gray", label="α=0.1")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Missed-failure rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Coverage–Risk (EndoVis 2017, τ=0.5)")
    _save(fig, output_dir / "fig_coverage_risk.pdf")


def fig_auroc_comparison(metrics_dir: Path, output_dir: Path) -> None:
    """Fig 4: AUROC/AUPRC comparison bar chart."""
    summary_path = metrics_dir / "monitor_metrics_summary.json"
    if not summary_path.exists():
        log.warning("Missing %s, skipping fig_auroc_comparison", summary_path)
        return

    data = load_json(summary_path)
    methods = list(data.keys())
    aurocs = [data[m].get("auroc", float("nan")) for m in methods]
    auprcs = [data[m].get("auprc", float("nan")) for m in methods]

    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, aurocs, width, label="AUROC", color=PALETTE[0])
    ax.bar(x + width / 2, auprcs, width, label="AUPRC", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_title("Failure Detection: AUROC & AUPRC")
    _save(fig, output_dir / "fig_auroc_comparison.pdf")


def fig_latency_scatter(metrics_dir: Path, output_dir: Path) -> None:
    """Fig 5: Latency vs. detection quality scatter."""
    latency_path = metrics_dir / "latency.json"
    summary_path = metrics_dir / "monitor_metrics_summary.json"
    if not latency_path.exists() or not summary_path.exists():
        log.warning("Missing latency or summary metrics, skipping fig_latency_scatter")
        return

    latency = load_json(latency_path)
    summary = load_json(summary_path)
    methods = list(set(latency.keys()) & set(summary.keys()))

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, m in enumerate(methods):
        ms = latency[m].get("mean_ms", float("nan"))
        auroc = summary[m].get("auroc", float("nan"))
        ax.scatter(ms, auroc, label=m, color=PALETTE[i % len(PALETTE)], s=80, zorder=5)
        ax.annotate(m, (ms, auroc), textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("Mean latency (ms/frame)")
    ax.set_ylabel("AUROC")
    ax.set_title("Latency vs. Detection Quality")
    _save(fig, output_dir / "fig_latency_scatter.pdf")


def fig_feature_ablation(metrics_dir: Path, output_dir: Path) -> None:
    """Fig 6: Feature-group ablation table/bar."""
    ablation_path = metrics_dir / "ablation_features.json"
    if not ablation_path.exists():
        log.warning("Missing %s, skipping fig_feature_ablation", ablation_path)
        return

    data = load_json(ablation_path)
    variants = list(data.keys())
    aurocs = [data[v].get("auroc", float("nan")) for v in variants]
    auprcs = [data[v].get("auprc", float("nan")) for v in variants]

    x = np.arange(len(variants))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - width / 2, aurocs, width, label="AUROC", color=PALETTE[0])
    ax.bar(x + width / 2, auprcs, width, label="AUPRC", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_title("Feature Ablation")
    _save(fig, output_dir / "fig_feature_ablation.pdf")


def fig_stratified_comparison(metrics_dir: Path, output_dir: Path) -> None:
    """Per-stratum AUROC bar: TCSR-Monitor vs entropy."""
    path = metrics_dir / "stratified_metrics.json"
    if not path.exists():
        log.warning("Missing %s, skipping fig_stratified_comparison", path)
        return

    data = load_json(path)
    strata = [s for s in data if s not in ("n_total_failures",)]
    monitor_aurocs = [data[s].get("monitor_auroc", float("nan")) for s in strata]
    entropy_aurocs = [data[s].get("entropy_auroc", float("nan")) for s in strata]
    ns = [data[s].get("n", 0) for s in strata]

    x = np.arange(len(strata))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, monitor_aurocs, width, label="TCSR-Monitor", color=PALETTE[0])
    bars2 = ax.bar(x + width / 2, entropy_aurocs, width, label="Entropy", color=PALETTE[2])
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={ns[i]})" for i, s in enumerate(strata)], rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC")
    ax.legend(fontsize=9)
    ax.set_title("Per-Stratum Failure Detection: Monitor vs Entropy")
    _save(fig, output_dir / "fig_stratified_comparison.pdf")


def fig_corruption_sweep(metrics_dir: Path, output_dir: Path) -> None:
    """Monitor vs entropy AUROC across corruption severity levels."""
    path = metrics_dir / "corruption_sweep.json"
    if not path.exists():
        log.warning("Missing %s, skipping fig_corruption_sweep", path)
        return

    data = load_json(path)
    severities = sorted(data.keys(), key=lambda k: int(k.split("_")[1]) if "_" in k else int(k))
    monitor_aurocs = [data[s].get("monitor_auroc", float("nan")) for s in severities]
    entropy_aurocs = [data[s].get("entropy_auroc", float("nan")) for s in severities]
    failure_rates = [data[s].get("failure_rate", float("nan")) for s in severities]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(severities, monitor_aurocs, "o-", color=PALETTE[0], label="TCSR-Monitor AUROC")
    ax1.plot(severities, entropy_aurocs, "s--", color=PALETTE[2], label="Entropy AUROC")
    ax1.axhline(0.5, ls=":", color="gray", lw=1)
    ax1.set_xlabel("Corruption severity")
    ax1.set_ylabel("AUROC")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(severities, failure_rates, "^:", color=PALETTE[3], alpha=0.6, label="Failure rate")
    ax2.set_ylabel("Failure rate", color=PALETTE[3])
    ax2.tick_params(axis="y", labelcolor=PALETTE[3])
    ax2.set_ylim(0, 1)

    ax1.set_title("Corruption Robustness Sweep (τ=0.5)")
    fig.tight_layout()
    _save(fig, output_dir / "fig_corruption_sweep.pdf")


def make_all_figures(metrics_dir: Path, output_dir: Path) -> None:
    fig_coverage_risk(metrics_dir, output_dir)
    fig_auroc_comparison(metrics_dir, output_dir)
    fig_latency_scatter(metrics_dir, output_dir)
    fig_feature_ablation(metrics_dir, output_dir)
    fig_stratified_comparison(metrics_dir, output_dir)
    fig_corruption_sweep(metrics_dir, output_dir)
    log.info("All figures written to %s", output_dir)
