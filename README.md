# TCSR-Monitor

**Temporal Conformal Surgical Risk Monitor** — a lightweight, model-agnostic runtime monitor that estimates calibrated failure risk for surgical instrument segmentation in real video.

Accepted (poster) at the **MICCAI 2026 Workshop on Uncertainty for Safe Utilization of Machine Learning in Medical Imaging (UNSURE)**. Framed as a perception-reliability safety layer for vision-guided surgical robot autonomy.

---

## Problem

Deep learning segmenters fail silently: high-confidence wrong masks with no external alarm. TCSR-Monitor predicts *when* a segmenter should not be trusted, using only features observable at runtime (no ground truth), and wraps the risk score in a conformal guarantee controlling missed failures at a user-specified rate.

---

## Headline results

| Method | τ=0.5 AUROC ↑ | τ=0.5 AUPRC ↑ | τ=0.75 AUROC ↑ | τ=0.75 AUPRC ↑ |
|---|---|---|---|---|
| **TCSR-Monitor (ours)** | **0.877** | 0.468 | **0.793** | **0.301** |
| Predictive entropy | 0.764 | **0.635** | 0.483 | 0.220 |
| Max-softmax | 0.536 | 0.087 | 0.509 | 0.085 |
| Temporal heuristic | 0.338 | 0.013 | 0.460 | 0.074 |

*Held-out clean EndoVis 2017 test set (seed 0, point estimates — 825 frames from
3 held-out sequences; frame-level scores are correlated within a sequence, see
the paper for sequence-level bootstrap CIs). Failure prevalence: 1.7% at
τ=0.5, 6.9% at τ=0.75. Monitor inference latency: 0.36 ms/frame (mean, CPU
feature extraction + XGBoost).*

Under leave-one-corruption-out (LOCO) evaluation — the paper's core
robustness claim — the monitor generalizes to unseen acquisition
degradations: **0.814 AUROC** vs. **0.481** for predictive entropy (26/30
corruption × severity cells won). See the paper for the full breakdown,
circularity control, Mondrian conformal calibration, and SAM2 transfer
results.

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
  title     = {Beyond Uncertainty: Generalizable Failure Monitoring for Surgical
               Segmentation under Acquisition Degradation},
  author    = {Pham, Hieu D. and Cao, Dang P. M. and Huynh, Thanh Trung},
  booktitle = {MICCAI Workshop on Uncertainty for Safe Utilization of Machine
               Learning in Medical Imaging (UNSURE)},
  year      = {2026},
}
```

### Dataset citations

- **EndoVis 2017:** Allan et al., 2019.
- **CholecSeg8k:** Hong et al., 2021.
