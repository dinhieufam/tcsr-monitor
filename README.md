# TCSR-Monitor

**Temporal Conformal Surgical Risk Monitor** — a lightweight, model-agnostic runtime monitor that estimates calibrated failure risk for surgical instrument segmentation in real video.

Submitted to **ICARCV 2026**. Framed as a perception-reliability safety layer for vision-guided surgical robot autonomy.

---

## Problem

Deep learning segmenters fail silently: high-confidence wrong masks with no external alarm. TCSR-Monitor predicts *when* a segmenter should not be trusted, using only features observable at runtime (no ground truth), and wraps the risk score in a conformal guarantee controlling missed failures at a user-specified rate.

---

## Headline results

<!-- Auto-filled from results/metrics/ after `make eval` -->

| Method | AUROC ↑ | AUPRC ↑ | Recall@90% ↑ | FA/min ↓ | Latency (ms) ↓ |
|---|---|---|---|---|---|
| TCSR-Monitor (XGB, all features) | — | — | — | — | — |
| Max-softmax | — | — | — | — | — |
| Predictive entropy | — | — | — | — | — |
| Temperature scaling | — | — | — | — | — |
| TTA variance | — | — | — | — | — |
| Temporal heuristic | — | — | — | — | — |
| Feature-distance OOD | — | — | — | — | — |

*EndoVis 2017 test set, τ = 0.5, mean ± std over seeds {0,1,2}.*

---

## Install

```bash
# Recommended: uv
make setup

# Or: conda
conda env create -f environment.yml
conda activate tcsr-monitor
pip install -e .
```

## Data access

See [docs/datasets.md](docs/datasets.md) for download links, licenses, and checksums.

```bash
make data   # download + verify + build manifests (EndoVis 2017 + CholecSeg8k)
```

## Reproduce everything

```bash
make all    # full pipeline end-to-end
```

Or step-by-step — see [docs/reproduce.md](docs/reproduce.md).

---

## Repo map

```
configs/      Hydra config tree — change dataset/model/feature/baseline here
src/tcsr/     Installable package (data, segmentation, features, labels,
              monitor, conformal, baselines, evaluation, viz, utils)
scripts/      CLI entrypoints 00–10 (thin Hydra @main wrappers)
tests/        Unit tests + 20-frame smoke pipeline
results/      Committed metrics tables and paper figures
```

### Extend the system

| What to add | Where |
|---|---|
| New dataset | `src/tcsr/data/<name>.py` + `configs/data/<name>.yaml` |
| New segmentation model | `src/tcsr/segmentation/models.py` registry + `configs/seg_model/<name>.yaml` |
| New feature group | `src/tcsr/features/<name>.py` + `configs/features/all.yaml` |
| New baseline | `src/tcsr/baselines/runners.py` + `configs/baseline/<name>.yaml` |

---

## Citation

```bibtex
@inproceedings{tcsr2026,
  title     = {TCSR-Monitor: Temporal Conformal Surgical Risk Monitoring for Instrument Segmentation},
  booktitle = {ICARCV},
  year      = {2026},
}
```

### Dataset citations

- **EndoVis 2017:** Allan et al., 2019.
- **CholecSeg8k:** Hong et al., 2021.
