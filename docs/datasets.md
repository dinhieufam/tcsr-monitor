# Dataset Access, Licenses, and Download Instructions

## EndoVis 2017

**Task:** Robotic Instrument Segmentation Challenge, MICCAI 2017  
**License:** See challenge website. For research use only.  
**Citation:** Allan et al., "2017 Robotic Instrument Segmentation Challenge." arXiv:1902.06426, 2019.

### Download

1. Register at the challenge page: https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/
2. Download the training and test sequences.
3. Extract to `$ENDOVIS2017_ROOT` (set in `.env`).

Expected structure:
```
$ENDOVIS2017_ROOT/
├── instrument_dataset_1/
│   ├── left_frames/   # *.png RGB frames
│   └── ground_truth/  # *_label.png instrument masks
...
└── instrument_dataset_10/
```

4. Run preprocessing to produce `data/processed/endovis2017/frames/` and `masks/`:
```bash
# Add preprocessing script here
```

5. Record SHA-256 checksums:
```bash
sha256sum data/raw/endovis2017/*.zip > data/checksums/endovis2017.sha256
```

---

## CholecSeg8k

**Task:** Cholecystectomy instrument segmentation (13 classes → binarized)  
**License:** Creative Commons Attribution 4.0 (CC BY 4.0)  
**Citation:** Hong et al., "CholecSeg8k: A Semantic Segmentation Dataset for Laparoscopic Cholecystectomy Based on Cholec80." arXiv:2012.12453, 2021.

### Download

Available on Kaggle: https://www.kaggle.com/datasets/newslab/cholecseg8k

```bash
kaggle datasets download -d newslab/cholecseg8k -p $CHOLECSEG8K_ROOT
unzip "$CHOLECSEG8K_ROOT/cholecseg8k.zip" -d "$CHOLECSEG8K_ROOT"
sha256sum "$CHOLECSEG8K_ROOT"/*.zip > data/checksums/cholecseg8k.sha256
```

**Binarization:** For the cross-dataset protocol, the 13 semantic classes are reduced to binary instrument vs. background. Instrument-related class IDs: {5, 6, 7, 8} (update in `src/tcsr/data/cholecseg8k.py`).

---

## Optional: SurgiSR4K

Not load-bearing for the core ICARCV claim. Include only if time allows.

---

## Optional: SegSTRONG-C

Synthetic-corruption robustness benchmark. Results labeled as "generated-corruption robustness" in the paper — not a core claim.
