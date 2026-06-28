#!/usr/bin/env python
"""
Experiment E8: second fine-tuned backbone transfer audit.

The EDD marks E8 as conditional: execute only if a trained second surgical
segmentation backbone or cached second-backbone predictions already exist. This
script checks those prerequisites and writes a reproducible no-go record when
the required artifacts are unavailable.

Run from: tcsr-monitor/
  python scripts/10c_second_backbone_transfer.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROCESSED_DIR = Path("data/processed/endovis2017")
CONFIG_DIR = Path("configs/seg_model")
RESULTS_DIR = Path("results/metrics/backbone_transfer")

CANDIDATES = {
    "deeplabv3p_resnet50": {
        "config": CONFIG_DIR / "deeplabv3p_resnet50.yaml",
        "prediction_dirs": [
            PROCESSED_DIR / "deeplabv3p_resnet50_predictions",
            PROCESSED_DIR / "predictions_deeplabv3p_resnet50",
        ],
    },
    "segformer_b0": {
        "config": CONFIG_DIR / "segformer_b0.yaml",
        "prediction_dirs": [
            PROCESSED_DIR / "segformer_b0_predictions",
            PROCESSED_DIR / "predictions_segformer_b0",
        ],
    },
    "unetpp_resnet34": {
        "config": CONFIG_DIR / "unetpp_resnet34.yaml",
        "prediction_dirs": [
            PROCESSED_DIR / "unetpp_resnet34_predictions",
            PROCESSED_DIR / "predictions_unetpp_resnet34",
        ],
    },
}

SAM2_DIR = PROCESSED_DIR / "sam2_predictions"


def read_checkpoint_from_config(path: Path) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.strip().startswith("checkpoint:"):
            value = line.split(":", 1)[1].strip()
            if value and value.lower() != "null":
                return value
            return None
    return None


def count_prob_maps(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("probs.npz"))


def candidate_row(name: str, spec: dict) -> dict:
    config_path = spec["config"]
    checkpoint = read_checkpoint_from_config(config_path)
    checkpoint_exists = bool(checkpoint) and Path(checkpoint).exists()
    pred_counts = {str(p): count_prob_maps(p) for p in spec["prediction_dirs"]}
    usable_prediction_dir = next((p for p, n in pred_counts.items() if n > 0), None)
    usable = checkpoint_exists and usable_prediction_dir is not None
    return {
        "candidate": name,
        "config_exists": config_path.exists(),
        "checkpoint": checkpoint,
        "checkpoint_exists": checkpoint_exists,
        "prediction_counts": pred_counts,
        "usable_prediction_dir": usable_prediction_dir,
        "usable_for_e8": usable,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [candidate_row(name, spec) for name, spec in CANDIDATES.items()]
    usable = [row for row in candidates if row["usable_for_e8"]]
    sam2_prob_maps = count_prob_maps(SAM2_DIR)

    status = "ready" if usable else "blocked_prerequisites_missing"
    recommendation = (
        "Proceed with second-backbone transfer using cached fine-tuned outputs."
        if usable
        else (
            "Do not run E8 as a publication experiment now. The available alternate "
            "configs do not point to trained checkpoints or cached prediction maps. "
            "SAM2 outputs exist, but SAM2 was already evaluated separately and is not "
            "the fine-tuned surgical-backbone transfer requested by E8."
        )
    )

    record = {
        "experiment": "E8 Second Fine-Tuned Backbone Transfer",
        "status": status,
        "date": "2026-06-27",
        "decision": "no_go" if not usable else "go",
        "candidate_backbones": candidates,
        "sam2_existing_prob_maps": sam2_prob_maps,
        "sam2_existing_result": str(Path("results/metrics/sam2_transfer.json")),
        "required_for_execution": [
            "trained second fine-tuned surgical segmentation checkpoint",
            "second-backbone clean test probability maps or logits",
            "feature cache compatible with the monitor feature schema",
            "failure labels computed from second-backbone masks against ground truth",
        ],
        "recommendation": recommendation,
    }

    json_path = RESULTS_DIR / "e8_second_backbone_input_audit.json"
    csv_path = RESULTS_DIR / "e8_second_backbone_input_audit.csv"
    json_path.write_text(json.dumps(record, indent=2) + "\n")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate",
                "config_exists",
                "checkpoint",
                "checkpoint_exists",
                "usable_prediction_dir",
                "usable_for_e8",
            ],
        )
        writer.writeheader()
        for row in candidates:
            writer.writerow({
                "candidate": row["candidate"],
                "config_exists": row["config_exists"],
                "checkpoint": row["checkpoint"] or "",
                "checkpoint_exists": row["checkpoint_exists"],
                "usable_prediction_dir": row["usable_prediction_dir"] or "",
                "usable_for_e8": row["usable_for_e8"],
            })

    print(json.dumps({
        "status": status,
        "decision": record["decision"],
        "usable_candidates": [row["candidate"] for row in usable],
        "sam2_existing_prob_maps": sam2_prob_maps,
        "json": str(json_path),
        "csv": str(csv_path),
    }, indent=2))


if __name__ == "__main__":
    main()
