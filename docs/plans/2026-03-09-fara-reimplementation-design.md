# FARA Reimplementation & Baseline Comparison — Design Document

**Date**: 2026-03-09
**Status**: Approved
**Paper**: "A Robust Region-Aware Framework for Audio Forgery Localization" (Luo et al., IEEE TASLP 2026, DOI: 10.1109/TASLPRO.2026.3661237)

---

## 1. Goals

1. **Reproduce FARA** from scratch, matching published numbers (EER 5.98 on ASVPS, F1 95.09)
2. **Head-to-head comparison** of FARA vs BAM vs CFPRF vs PSDS across multiple datasets
3. All models trained on PartialSpoof only, cross-dataset eval on all others
4. **Exploratory** — results inform whether FARA is worth integrating into Explainability pipeline

## 2. Constraints

- **Never modify cloned baseline repos** — external wrappers only
- **Follow papers exactly** — contact authors for any ambiguity
- **Phase gating** — no proceeding until previous phase is completed, verified, tested, reviewed
- **Modular code** — shared modules, no duplicate functionality
- **Scientific rigor** — no shortcuts, no fabricated results, all claims cited

## 3. Datasets

### Available Now
| Dataset | Location | Frame Res. | Labels |
|---|---|---|---|
| PartialSpoof (ASVPS) | `/media/lab2208/ssd/datasets/PartialSpoof` | 20ms | Frame-level binary |
| LlamaPartialSpoof | `/media/lab2208/ssd/datasets/LlamaPartialSpoof` | 20ms | Frame-level binary |

### Available Later (user will download)
| Dataset | Source | Frame Res. | Notes |
|---|---|---|---|
| HQ-MPSD (English) | zenodo.org/records/17929533 | 30ms → resample to 20ms | Multilingual, artifact-controlled |
| PartialEdit | yzyouzhang.com/PartialEdit | 20ms | Neural speech editing |

## 4. Architecture — Modular Core Library

```
Localization/
├── core/                         # SHARED MODULES — used by ALL methods
│   ├── data/
│   │   ├── base_dataset.py       # Abstract dataset class
│   │   ├── partialspoof.py       # PartialSpoof loader (extends base)
│   │   ├── llamaspoof.py         # LlamaPS loader (extends base)
│   │   ├── hq_mpsd.py            # HQ-MPSD loader (added later)
│   │   ├── partialedit.py        # PartialEdit loader (added later)
│   │   └── collate.py            # Shared collation, padding, batching
│   ├── metrics/
│   │   ├── eer.py                # Equal Error Rate computation
│   │   ├── classification.py    # Precision, Recall, F1 (frame-level)
│   │   └── evaluate.py           # Unified evaluation entry point
│   ├── training/
│   │   ├── trainer.py            # Generic training loop (configurable)
│   │   ├── callbacks.py          # Checkpointing, logging, early stopping
│   │   └── scheduler.py          # LR scheduling utilities
│   ├── audio/
│   │   ├── io.py                 # Audio loading, resampling
│   │   └── augmentation.py       # Compression, noise (for future phases)
│   └── utils/
│       ├── config.py             # YAML config loading
│       ├── seed.py               # Reproducibility (seed=42)
│       └── logging.py            # Experiment logging
├── fara/                         # FARA reimplementation
│   ├── model/
│   │   ├── learnable_mask.py
│   │   ├── sincnet.py
│   │   ├── feature_fusion.py
│   │   ├── cmoe.py
│   │   ├── boundary_enhance.py
│   │   └── fara.py               # Full model assembly
│   ├── losses/
│   │   ├── group_contrastive.py
│   │   └── combined_loss.py
│   ├── train.py                  # Uses core/training/trainer.py
│   └── config.yaml
├── baselines/
│   ├── repos/                    # Git submodules — NEVER modified
│   │   ├── BAM/
│   │   ├── CFPRF/
│   │   └── PartialSpoof/
│   └── wrappers/
│       ├── base_wrapper.py       # Abstract wrapper interface
│       ├── bam_wrapper.py        # Data format adapter for BAM
│       ├── cfprf_wrapper.py      # Data format adapter for CFPRF
│       ├── psds_wrapper.py       # Data format adapter for PSDS
│       └── eval_bridge.py        # Captures baseline outputs → core/metrics
├── evaluation/
│   └── cross_dataset_eval.py     # Orchestrates eval matrix using core/metrics
├── configs/
│   ├── bam.yaml
│   ├── fara.yaml
│   ├── cfprf.yaml
│   └── psds.yaml
├── results/                      # Checkpoints, logs, result tables
├── scripts/
│   ├── download_hqmpsd.sh
│   ├── download_partialedit.sh
│   ├── train_bam.sh
│   ├── train_fara.sh
│   └── eval_all.sh
└── docs/
    └── plans/
        └── 2026-03-09-fara-reimplementation-design.md
```

### Reusability Matrix

| Module | Used By |
|---|---|
| `core/data/base_dataset.py` | All 4 dataset loaders |
| `core/metrics/*` | All methods + cross-dataset eval |
| `core/training/trainer.py` | FARA + any future custom models |
| `core/audio/io.py` | All dataset loaders + augmentation |
| `baselines/wrappers/base_wrapper.py` | All 3 baseline wrappers |
| `baselines/wrappers/eval_bridge.py` | All 3 baseline evals |

## 5. Baseline Integration — Wrapper Pattern

Wrappers never import or modify repo code. They:
1. Convert our unified dataset format → repo's expected format (file lists, directory structure, label format)
2. Invoke repo's original training/eval scripts as subprocesses
3. Parse repo's output files → our common prediction format
4. Feed predictions into `core/metrics/evaluate.py`

## 6. FARA Components — Paper Reference

| Component | Paper Section | Key Equations | Unknowns (need author contact) |
|---|---|---|---|
| Learnable Mask | III-A | Eq. 1 | Top-K value (k), noise distribution/scale (ε) |
| CMoE Router | III-B | Eq. 2, 3 | FAISS index type, initial centroid strategy |
| CMoE Experts | III-B | Eq. 4 | Expert MLP dimensions |
| Group Contrastive Loss | III-C | Eq. 5, 6, 7 | β (edge parameter) |
| Feature Fusion | III | — | Gating mechanism details |
| Boundary Enhancement | III | — | Conv kernel size, MLP hidden dims |
| Boundary Classify | III | — | MLP architecture |
| Classify Module | III | — | MLP architecture |

**Action required**: Draft email to corresponding author (Hongxia Wang, hxwang@scu.edu.cn) with all unknowns before starting Phase 2.

## 7. Training Specification (from paper)

- GPU: RTX 4080 16GB (paper used 2080Ti 11GB — we have more headroom)
- Optimizer: Adam (β1=0.9, β2=0.999, weight_decay=1e-4, lr=1e-5)
- Input: 16kHz, 20ms frame resolution
- Loss: L_train = L_spoof + 0.5 * L_boundary + 0.2 * L_CRL
- Seed: 42
- K=8 experts

## 8. Evaluation Protocol

**All models trained on PartialSpoof only.**

| Eval Dataset | Metric Targets (FARA) | Source |
|---|---|---|
| PartialSpoof (ASVPS) | EER 5.98, F1 95.09 | Table I |
| LlamaPartialSpoof | EER 33.17, F1 64.87 | Table III |
| HQ-MPSD | No published FARA numbers | New eval |
| PartialEdit | No published FARA numbers | New eval |

**BAM reproduction targets:**
- ASVPS: EER 8.43 (FARA paper Table I)
- PartialEdit: EER 4.07 (PartialEdit paper Table 3)

**Tolerance**: >10% relative deviation → investigate. >20% → debug, do not proceed.

## 9. Phases

| Phase | Description | Gate Criteria |
|---|---|---|
| 0 | Infrastructure: core modules, data loaders (PS + LlamaPS), metrics | Unit tests pass, metrics verified |
| 1 | BAM: clone, wrapper, train on PS, verify numbers | EER within 10% of 8.43 |
| 2 | FARA: author contact, implement components, unit tests | All components tested, forward pass verified |
| 3 | FARA training: train on PS, reproduce numbers | EER within 10% of 5.98 |
| 4 | Cross-dataset eval: FARA + BAM on LlamaPS (+ HQ-MPSD, PartialEdit when available) | Results collected and documented |
| 5 | CFPRF + PSDS: clone, wrappers, train, verify, full eval | Published numbers reproduced |
| 6 | Analysis: compile results, decision on Explainability integration | Report complete |
