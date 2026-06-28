# Reproducing Every Number in the Paper

All commands below assume you have followed the setup in the README.

## Environment

```bash
make setup
cp .env.example .env  # edit ENDOVIS2017_ROOT and CHOLECSEG8K_ROOT
```

## Full pipeline (EndoVis 2017, main experiment)

```bash
make all SEED=42
```

This runs stages 00–10 in order. Each stage writes its artifacts to
`data/processed/endovis2017/` and `results/`.

## Step-by-step

```bash
make data          # 00_download_data + 01_build_manifests
make predict       # 03_run_predictions (frozen U-Net)
make labels        # 04_make_failure_labels (tau in {0.5, 0.6, 0.75})
make features      # 05_extract_features (all feature groups)
make monitor       # 06_train_monitor + 07_calibrate_conformal
make baselines     # 08_run_baselines
make eval          # 09_evaluate → results/metrics/
make figures       # 10_make_figures → results/figures/
```

## Cross-dataset experiment

```bash
python scripts/06_train_monitor.py data=endovis2017 seed=42
python scripts/09_evaluate.py \
    data=cholecseg8k \
    monitor_dir=experiments/<endovis_run>/monitor \
    seed=42
```

Results reported without claiming the conformal guarantee transfers.

## Feature ablation

```bash
python scripts/06_train_monitor.py -m features=all,confidence_only,no_temporal,no_shift seed=0,1,2
```

Aggregated results → `results/metrics/ablation_features.json`.

## Monitor backbone sweep

```bash
python scripts/06_train_monitor.py -m monitor=logreg,random_forest,xgboost,gru_temporal seed=0,1,2
```

## Seed sweep (for mean ± std)

All headline numbers reported as mean ± std over seeds {0, 1, 2}.

```bash
for s in 0 1 2; do make all SEED=$s; done
```

Then aggregate with `scripts/09_evaluate.py` or a notebook in `notebooks/`.

## Regenerate figures only

```bash
make figures  # reads from results/metrics/, no model inference
```

## Run tests

```bash
make test              # unit tests + smoke pipeline
make test -k slow      # include slow smoke test (20-frame end-to-end)
```
