# FARA Phase 2 — Implementation Design

**Date**: 2026-03-24
**Status**: Approved
**Paper**: Luo et al., "A Robust Region-Aware Framework for Audio Forgery Localization", IEEE/ACM TASLP 2026, DOI: 10.1109/TASLPRO.2026.3661237

---

## 1. Resolved Unknowns

The paper omits several architectural details. We resolve them with reasonable defaults, to be validated on the ASVPS validation set and swept if needed.

| Parameter | Default | Rationale | Sweep Range |
|---|---|---|---|
| LearnableMask k | 12 | Suppress half of 24 WavLM layers | {8, 10, 12, 14} |
| SincNet out_channels | 80 | Ravanelli & Bengio (2018) default | — |
| SincNet kernel_size | 251 | Original SincNet default (~15.7ms at 16kHz) | — |
| SincNet stride | 320 | 20ms at 16kHz, matching WavLM frame rate | — |
| Feature Fusion | Learned sigmoid gate + linear projection | Paper: "gating mechanism" | — |
| CMoE num_experts (K) | 8 | Paper Table VIII: K=8 achieves best EER (0.0095) | — |
| CMoE clustering | PyTorch K-means, no FAISS dependency | EMA centroid update α=0.1 (Eq. 2) | — |
| Expert MLP dims | d_model → 4*d_model → d_model | Standard transformer FFN expansion ratio | — |
| Boundary Enhance | Conv1D(kernel=3, padding=1) + ReLU + Linear | Paper: "1D convolution"; kernel=3 is minimal | {3, 5, 7} |
| Classify MLP | Linear(d, d//2) → ReLU → Dropout(0.1) → Linear(d//2, 2) | Standard classification head | — |
| BoundaryClassify MLP | Same architecture as Classify | Paper: "sharing the same structure" | — |
| GCL β (edge margin) | 0.3 | Common contrastive margin | {0.1, 0.2, 0.3, 0.5} |
| WavLM hidden states | 24 transformer layers (index 1-24, exclude CNN layer 0) | WavLM-Large architecture | — |

## 2. Data Flow

```
Raw Waveform (B, T_samples) at 16kHz
    │
    ├── WavLM-Large ──→ (B, T_frames, 24, 1024) all transformer hidden states
    │       │
    │       └── LearnableMask(k=12) ──→ (B, T_frames, 1024)
    │               noisy top-k gating, suppress 12 layers per frame
    │
    └── SincNet(80 filters, stride=320) ──→ (B, T_frames, 80)
            │
            ▼
    FeatureFusion ──→ (B, T_frames, 1024)
        σ(α) * wavlm_feat + (1-σ(α)) * Linear(sincnet_feat)
            │
            ▼
    CMoE Router ──→ W: (B, T_frames, K=8) expert weights
        K-means centroids (C×D), L2 distance, softmax routing (Eq. 3)
            │
    CMoE Experts (K=8 MLPs) ──→ O: (B, T_frames, 1024)
        Each expert: Linear(1024,4096) → GELU → Linear(4096,1024)
        Output: Σ W_j * E_j(x) (Eq. 4)
            │
            ├── BoundaryEnhance ──→ (B, T_frames, 1024)
            │       Conv1D(1024, 1024, k=3, pad=1) + ReLU + Linear
            │       │
            │       └── BoundaryClassify ──→ boundary_logits (B, T_frames, 2)
            │               generates Attention Mask for spoof branch
            │
            └── Classify ──→ spoof_logits (B, T_frames, 2)
                    modulated by Attention Mask from boundary branch
```

## 3. Loss Function (Eq. 8)

```
L_train = L_spoof + 0.5 * L_boundary + 0.2 * L_CRL

L_spoof:    CrossEntropyLoss(ignore_index=-1) on spoof_logits vs frame_labels
L_boundary: CrossEntropyLoss(ignore_index=-1) on boundary_logits vs boundary_labels
L_CRL:      Group Contrastive Loss (Eq. 5-7)
            - Groups defined by CMoE cluster assignments
            - Cosine similarity within groups
            - β=0.3 edge margin for negative pairs
```

## 4. Training Specification

- GPU: RTX 4080 16GB (paper used 2080Ti 11GB)
- Optimizer: Adam(β1=0.9, β2=0.999, weight_decay=1e-4, lr=1e-5)
- Input: 16kHz, 20ms frame resolution
- Seed: 42
- Batch size: 8 (adjust if VRAM constrained)
- Epochs: 50 (early stopping on val EER, patience=10)
- Gradient clipping: max_norm=1.0

## 5. Evaluation Targets

| Dataset | Metric | FARA Target | Tolerance |
|---|---|---|---|
| ASVPS eval | EER | 5.98% | ±10% (5.38-6.58) |
| ASVPS eval | F1 | 95.09% | ±10% |
| LlamaPS | EER | 33.17% | ±10% |
| LlamaPS | F1 | 64.87% | ±10% |

## 6. BAM Retrain at 0.02s

BAM was originally trained at 0.16s (its native resolution). For fair cross-method comparison, retrain at 0.02s matching FARA's resolution. The FARA paper states: "we set the model's localization time resolution to 20 ms" (Section IV-B).

## 7. File Structure

```
fara/
├── __init__.py
├── model/
│   ├── __init__.py
│   ├── learnable_mask.py    ← EXISTS (verified correct)
│   ├── sincnet.py           ← EXISTS (verified correct)
│   ├── feature_fusion.py    ← NEW
│   ├── cmoe.py              ← NEW
│   ├── boundary_enhance.py  ← NEW
│   └── fara.py              ← NEW (assembly)
├── losses/
│   ├── __init__.py
│   ├── group_contrastive.py ← NEW
│   └── combined_loss.py     ← NEW
└── train.py                 ← NEW

core/training/
├── __init__.py
├── trainer.py               ← NEW
└── callbacks.py             ← NEW

tests/
├── fara/
│   ├── __init__.py
│   ├── test_feature_fusion.py
│   ├── test_cmoe.py
│   ├── test_boundary_enhance.py
│   ├── test_fara_assembly.py
│   ├── test_group_contrastive.py
│   └── test_combined_loss.py
└── test_fara_integration.py  ← Phase 2 gate test
```
