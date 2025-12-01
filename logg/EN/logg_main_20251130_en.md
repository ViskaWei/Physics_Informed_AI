# Surface Gravity ($\log g$) Prediction: Master Experiment Log (as of 2025-12-01)

- Directory: `logg/`
- Last Updated: 2025-12-01
- Style Reference: `/home/swei20/VIT/docs`
- Description: This document serves as the **top-level index for all $\log g$ prediction experiments**, consolidating core conclusions from each subdirectory.

---

# ⚡ Quick Reference: Neural Network Design Guidelines

> **Executive Summary**: The $\log g$-flux mapping is inherently linear ($R^2=0.999$ @ noise=0), with information distributed across ~100-dimensional subspace; optimal NN strategy employs **small-kernel CNN (k=9, $R^2=0.657$)** or **Residual MLP + Linear Shortcut**; data volume is the critical leverage factor (32k→100k: +10.6%).

### Recommended NN Architectures (Priority-Ordered)

| Priority | Approach | Configuration | Expected $R^2$ | Key Points |
|:--------:|----------|---------------|----------------|------------|
| 🥇 | **Small-Kernel CNN** | k=9, 2L, lr=3e-3, AdaptiveAvgPool | **0.657** | Small receptive field + global pooling |
| 🥈 | **Residual MLP** | [256,64], learns Ridge residuals | **0.498** | Linear shortcut is critical |
| 🥉 | **Latent Probe** | enc\_pre\_latent + seg\_mean\_K8 | **0.55** | Extracted from Denoiser |
| 🆕 | **Top-K Window CNN** | K=256/512, W=17, Residual on Ridge | target ≥0.70 | [MVP-Local-1 in progress](gta/exp_topk_window_cnn_transformer_20251201.md) |
| 🆕 | **Global Feature Tower** | 158-dim (PCA+Ridge+TopK+Latent) | target ≥0.50 | [MVP-Global-1 in progress](gta/exp_global_feature_tower_mlp_20251201.md) |
| 🆕 | **Swin-1D** | Tiny (1-2M), patch=8, window=8 | surpass LGBM @100k | [swin_main planned](swin/swin_main_20251201.md) |

### Critical Design Principles

| Principle | Specific Recommendation | Evidence Source |
|-----------|------------------------|-----------------|
| **Linear Shortcut** | $\hat{y} = w^\top x + g_\theta(x)$ | Ridge $R^2=0.999$ @ noise=0 |
| **Small kernel superior to large kernel** | k ∈ {7, 9}, avoid k > 15 | CNN: k=9 (0.66) >> k=63 (0.02) |
| **Data volume prioritized** | 100k >> 32k (+10.6%) | Exceeds all architectural improvements |
| **Preserve wavelength locality** | Segmented pooling / TopK+window | Latent: seg\_mean +77.6% |

---

# Table of Contents

0. [Core Research Questions](#0-core-research-questions)
1. [Project Overview](#1-project-overview)
2. [Subdirectory Index](#2-subdirectory-index)
3. [Global Core Conclusions](#3-global-core-conclusions)
4. [Model Performance Leaderboard](#4-model-performance-leaderboard)
5. [Key Design Principles](#5-key-design-principles)
6. [Neural Network Best Practices](#6-neural-network-best-practices)
7. [Future Research Directions](#7-future-research-directions)
8. [Appendix: Quick Navigation](#8-appendix-quick-navigation)

---

# 0. Core Research Questions

## 0.1 Primary Objective

> **Predict stellar surface gravity $\log g$ from synthetic stellar spectra (4096-dimensional flux), providing systematic experimental support for Physics-Informed Neural Network architecture design.**

## 0.2 Hierarchical Research Questions

### Information Structure Layer (Addressed by ridge/, pca/, noise/)

| Question | Status | Answer |
|----------|--------|--------|
| Is the $\log g$-flux mapping inherently linear? | ✅ Answered | **Yes**, $R^2=0.999$ @ noise=0 |
| What is the effective dimensionality of $\log g$ information? | ✅ Answered | **~100-200 dimensions** (PCA experiments) |
| Is information distribution sparse or dispersed? | ✅ Answered | **Sparse**, 24% of pixels match full spectrum |

### Model Capability Layer (Addressed by lightgbm/, NN/, cnn/)

| Question | Status | Answer |
|----------|--------|--------|
| What is the ceiling for linear models? | ✅ Answered | Ridge $R^2=0.458$ @ noise=1.0 |
| How much improvement can nonlinear models provide? | ✅ Answered | LightGBM +17%, MLP +8.7% |
| Are CNNs suitable for spectroscopic tasks? | ✅ **Resolved** | **Small-kernel CNN (k=9) achieves $R^2=0.657$** |
| What is the optimal kernel size? | ✅ Answered | **k=7-9 optimal**, large kernels degrade performance |
| Is the receptive field hypothesis valid? | ✅ Answered | **No**, RF↑ → performance↓ (RF=25 optimal) |

### Architecture Design Layer (Addressed by gta/, distill/)

| Question | Status | Answer |
|----------|--------|--------|
| What capacity does Global Tower require? | 📋 In progress | Requires ~100-150 dim input |
| Can metadata predict $\log g$? | ✅ Answered | **No**, $R^2 \approx 0$ |
| Is the representation learning approach viable? | ✅ **Validated** | **Yes**, optimized extraction achieves $R^2=0.55$ (+150%) |
| Impact of pooling strategy? | ✅ Answered | Segmented pooling >> global mean (+77.6%) |

## 0.3 Experiment Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                Phase 1: Information Structure               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │  ridge/  │    │   pca/   │    │  noise/  │               │
│  │  Linear  │    │ Effective │   │  Sparse  │               │
│  │  Ceiling │    │   Dims    │   │  Distrib │               │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘               │
│       └───────────────┼───────────────┘                     │
│                       ▼                                      │
├─────────────────────────────────────────────────────────────┤
│                Phase 2: Model Capability                     │
│  ┌────────────┐         ┌──────────┐                        │
│  │ lightgbm/  │         │   NN/    │                        │
│  │  Nonlinear │         │ MLP/CNN  │                        │
│  │  Baseline  │         │          │                        │
│  └─────┬──────┘         └────┬─────┘                        │
│        └─────────┬───────────┘                              │
│                  ▼                                           │
├─────────────────────────────────────────────────────────────┤
│                Phase 3: Architecture Design                  │
│  ┌──────────┐    ┌────────────┐    ┌──────────┐             │
│  │   gta/   │    │  distill/  │    │  train/  │             │
│  │  Global  │    │ Represent. │    │ Training │             │
│  │  Tower   │    │  Learning  │    │ Strategy │             │
│  └──────────┘    └────────────┘    └──────────┘             │
└─────────────────────────────────────────────────────────────┘
```

---

# 1. Project Overview

## 1.1 Research Objectives

> **Predict stellar surface gravity $\log g$ from synthetic stellar spectra (4096-dimensional flux)**

Core scientific questions:
- How is $\log g$ information distributed in spectral space? (Sparse vs. dispersed)
- What performance ceiling can linear models achieve? How much improvement do nonlinear models provide?
- How does noise affect predictive performance? How to design noise-robust architectures?
- Can neural networks surpass LightGBM?

## 1.2 Data Summary

| Configuration | Value |
|--------------|-------|
| Training samples | 32,000 / 100,000 |
| Feature dimensionality | 4,096 (spectral flux) |
| Target parameter | $\log g$ (surface gravity) |
| Noise levels | $\sigma \in \{0, 0.1, 0.5, 1.0, 2.0\}$ |
| Data source | BOSZ synthetic spectral library |

## 1.3 Experiment Scale Statistics

| Subdirectory | # Experiments | Primary Model | Status | Core Finding |
|--------------|---------------|---------------|--------|--------------|
| `ridge/` | 52+ | Ridge Regression | ✅ Complete | Mapping is inherently linear |
| `pca/` | 13+ | PCA + Ridge | ✅ Complete | Effective dimensionality ~100 |
| `lightgbm/` | 180 | LightGBM | ✅ Complete | 32k SOTA $R^2=0.536$ |
| `noise/` | 104+ | TopK + LGBM/Ridge | ✅ Complete | 24% pixels sufficient |
| `NN/` | 53 | MLP | ✅ Complete | Residual +8.7% vs Ridge |
| `cnn/` | 28+ | 1D CNN | ✅ Complete | **k=9 achieves $R^2=0.657$** |
| `gta/` | 4 | Global Tower design | 🔄 Dual-tower MVP in progress | Metadata cannot predict; Top-K Window + Global Feature Tower |
| `distill/` | 4 | Representation learning | ✅ Complete | Optimized extraction $R^2=0.55$ |
| `train/` | 1 | Training strategy | ✅ Complete | val_size configured by noise |
| `swin/` | 0 | Swin-1D | 🔄 New MVP framework | Validate hierarchical attention |
| **Total** | **430+** | - | - | - |

---

# 2. Subdirectory Index

| Directory | Topic | Core Finding | Main File |
|-----------|-------|--------------|-----------|
| [ridge/](ridge/) | **Linear Baseline** | Mapping inherently linear, $R^2=0.999$ @ noise=0 | [ridge_main](ridge/ridge_main_20251130.md) |
| [pca/](pca/) | **Dimensionality Reduction** | Requires k≥100 PCs for $R^2≥0.99$ | [pca_main](pca/pca_main_20251130.md) |
| [lightgbm/](lightgbm/) | **Tree Model Baseline** | $R^2=0.9982$ @ noise=0, learning rate most critical | [lightgbm_main](lightgbm/lightgbm_main_20251130.md) |
| [noise/](noise/) | **Noise & Feature Selection** | K=1000 (24%) matches full spectrum | [noise_main](noise/noise_main_20251130.md) |
| [NN/](NN/) | **MLP Neural Networks** | Residual MLP exceeds Ridge by +8.7% | [NN_main](NN/NN_main_20251130.md) |
| [cnn/](cnn/) | **CNN Neural Networks** | **Small kernel (k=9) achieves $R^2=0.657$** ⭐ | [cnn_main](cnn/cnn_main_20251201.md) |
| [gta/](gta/) | **Global Tower Architecture** | Metadata cannot predict; **Dual-tower MVP in progress** | [gta_main](gta/gta_main_20251130.md) |
| [distill/](distill/) | **Representation Learning** | Optimized extraction achieves $R^2=0.55$ (+150%) | [distill_main](distill/distill_main_20251130.md) |
| [train/](train/) | **Training Strategy** | Optimal val_size depends on noise | [train_main](train/train_main_20251130.md) |
| [swin/](swin/) | **Swin-1D Architecture** | 🆕 Validate hierarchical attention | [swin_main](swin/swin_main_20251201.md) |

---

# 3. Global Core Conclusions

## 3.1 Information Structure

| Finding | Evidence | Source |
|---------|----------|--------|
| **Mapping is inherently linear** | Ridge @ noise=0 achieves $R^2=0.999$ | ridge/ |
| **Information is high-dimensional and dispersed** | Requires k≥100 PCs for $R^2≥0.99$ | pca/ |
| **Information distribution is sparse** | K=1000 (24%) matches full spectrum | noise/ |
| **Low-variance directions are important** | $\log g$ information resides in PC 20-200 | pca/ |

## 3.2 Model Capability

| Finding | Evidence | Source |
|---------|----------|--------|
| **Small-kernel CNN is optimal** | k=9 achieves $R^2=0.657$ @ noise=0.1 | cnn/ ⭐ |
| **Large kernels perform worse** | k=9 (0.66) >> k=63 (0.02) | cnn/ |
| **Receptive field hypothesis refuted** | RF increase → performance degradation | cnn/ |
| **LightGBM is 32k SOTA** | $R^2=0.536$ @ noise=1.0 | lightgbm/ |
| **MLP exceeds Ridge** | $R^2$: 0.498 vs 0.458 (+8.7%) | NN/ |
| **Latent representations contain information** | Optimized extraction achieves $R^2=0.55$ | distill/ |
| **Data volume is critical** | 32k→100k: $R^2$ +10.6% | NN/ |

## 3.3 Noise Impact

| Finding | Evidence | Source |
|---------|----------|--------|
| **Optimal α scales with noise** | 0.001 (N=0) → 1000 (N=2.0) | ridge/ |
| **Noise training is 20× TopK** | train_noise 0→1: $\Delta R^2 \approx 0.49$ | noise/ |
| **Feature stability at high noise** | noise≥0.5 correlation >0.95 | ridge/ |

---

# 4. Model Performance Leaderboard

## 4.1 noise=0 (Noise-Free)

| Rank | Model | $R^2$ | Source |
|:----:|-------|:-----:|--------|
| 🥇 | LightGBM | **0.998** | lightgbm/ |
| 🥈 | Ridge (α=0.001) | **0.999** | ridge/ |
| 🥉 | PCA(500) + Ridge | **0.9995** | pca/ |

## 4.2 noise=0.1 (Low Noise) ⭐ **Optimal NN Test Scenario**

| Rank | Model | $R^2$ | Parameters | Source |
|:----:|-------|:-----:|------------|--------|
| 🥇 | LightGBM | **0.962** | - | lightgbm/ |
| 🥈 | **Small-Kernel CNN (k=9)** | **0.657** | 27K | cnn/ ⭐ |
| 🥉 | Ridge (α=1.0) | **0.909** | - | ridge/ |
| 4 | Latent Probe (seg\_mean) | 0.55 | - | distill/ |

> **Note**: CNN experiments conducted at noise=0.1, slightly different from other model noise settings.

## 4.3 noise=1.0 (Standard Noise, 32k Data)

| Rank | Model | $R^2$ | Source |
|:----:|-------|:-----:|--------|
| 🥇 | **MLP (100k)** | **0.551** | NN/ (100k data) |
| 🥈 | **LightGBM** | **0.536** | lightgbm/ |
| 🥉 | **Residual MLP (32k)** | **0.498** | NN/ |
| 4 | Ridge (α=200) | 0.458 | ridge/ |

## 4.4 noise=2.0 (High Noise)

| Rank | Model | $R^2$ | Source |
|:----:|-------|:-----:|--------|
| 🥇 | LightGBM | **0.268** | MASTER_CONCLUSIONS |
| 🥈 | Ridge (α=1000) | **0.221** | ridge/ |

---

# 5. Key Design Principles

## 5.1 Architecture Design

| Principle | Recommendation | Source |
|-----------|----------------|--------|
| **Small-kernel CNN** | k ∈ {7, 9}, avoid k > 15 | cnn/ ⭐ |
| **Linear Shortcut** | $\hat{y} = w^\top x + g_\theta(x)$ | ridge/, pca/ |
| **Residual Strategy** | MLP learns Ridge residuals | NN/ |
| **AdaptiveAvgPool** | Use pooling for global aggregation, not large receptive fields | cnn/ |
| **Bottleneck ≥ 100** | Avoid excessive dimensionality reduction | pca/ |
| **2-3 layers sufficient** | Performance collapses at 4+ layers | NN/ |

## 5.2 Regularization Strategy

| Principle | Recommendation | Source |
|-----------|----------------|--------|
| **Weight decay scales with noise** | Higher noise → stronger regularization | ridge/ |
| **Variance-aware regularization** | PCA whitening | pca/ |
| **Dropout ≤ 0.3 (Residual)** | Excessive dropout prevents residual learning | NN/ |
| **Learning rate by kernel size** | Small kernel: lr=3e-3, large kernel: lr=1e-3 | cnn/ |

## 5.3 Input Design

| Principle | Recommendation | Source |
|-----------|----------------|--------|
| **Dual-channel input** | [flux, σ] | ridge/ (error experiments) |
| **TopK + window** | Extract neighborhood, not single points | noise/ |
| **Spectral data required** | Metadata cannot predict $\log g$ | gta/ |
| **Preserve wavelength locality** | Segmented pooling >> global mean | distill/ |

## 5.4 Training Strategy

| Principle | Recommendation | Source |
|-----------|----------------|--------|
| **Noise training prioritized** | train_noise ≈ 1.0-1.2 × test_noise | noise/ |
| **val_size configured by noise** | noise=0: use 128, noise=1: use 512 | train/ |
| **Data volume prioritized** | Prioritize acquiring more data | NN/ |

---

# 6. Neural Network Best Practices

> **This section consolidates NN design best practices derived from all experiments for quick reference.**

## 6.1 Recommended Architecture Configurations

### Option A: Small-Kernel CNN (Recommended for Initial Experiments)

```yaml
# Optimal configuration: R² = 0.657 @ noise=0.1
Architecture: 2-Layer CNN
  Conv1D: in=1, out=32, kernel=9, padding=same
  BatchNorm + ReLU
  Conv1D: in=32, out=64, kernel=9, padding=same
  BatchNorm + ReLU
  AdaptiveAvgPool1D(1)
  Linear(64, 1)

Hyperparameters:
  lr: 3e-3  # Critical: small kernel requires high lr
  weight_decay: 0
  batch_size: 2048
  epochs: 50
  early_stopping: patience=20

Parameters: ~27K
Receptive Field: RF=25
```

### Option B: Residual MLP (Classic Stable Approach)

```yaml
# Optimal configuration: R² = 0.498 @ noise=1.0, 32k data
Architecture: 2-Layer MLP + Ridge Residual
  Input: 4096 (full spectral flux)
  Linear(4096, 256) + ReLU + Dropout(0.3)
  Linear(256, 64) + ReLU + Dropout(0.3)
  Linear(64, 1)
  
Prediction: y_pred = Ridge_pred + MLP_output  # Linear Shortcut

Hyperparameters:
  lr: 0.001
  weight_decay: 0  # No wd for Residual mode
  dropout: 0.3
  batch_size: 2048
  epochs: 100
  
Parameters: ~1.1M
```

### Option C: Large-Data MLP (For 100k+ Data)

```yaml
# Optimal configuration: R² = 0.551 @ 100k data
Architecture: 3-Layer Wide MLP
  Linear(4096, 2048) + GELU + Dropout(0.4)
  Linear(2048, 1024) + GELU + Dropout(0.4)
  Linear(1024, 512) + GELU + Dropout(0.4)
  Linear(512, 1)

Hyperparameters:
  lr: 3e-4
  weight_decay: 1e-4
  activation: GELU
  dropout: 0.4
  
Parameters: ~11M
```

## 6.2 Critical Heuristics

### Recommended Practices

| Practice | Rationale | Evidence |
|----------|-----------|----------|
| Use small kernel (k=7-9) | Large kernels perform worse | k=9 (0.66) >> k=63 (0.02) |
| Use Linear Shortcut | Mapping is inherently linear | Ridge $R^2=0.999$ @ noise=0 |
| Prioritize data volume increase | Data effect >> architecture effect | 32k→100k: +10.6% |
| Use AdaptiveAvgPool | More effective than large receptive fields | CNN experiments validated |
| Preserve wavelength locality | Critical for $\log g$ | Segmented pooling +77.6% |

### Pitfalls to Avoid

| Pitfall | Experimental Evidence | Correct Approach |
|---------|----------------------|------------------|
| Increasing kernel to k>15 | k=63 is 30× worse than k=9 | Use k=7-9 |
| Using single lr for all architectures | Small kernel with lr=1e-3 yields $R^2=0.01$ | Adjust lr by kernel size |
| Deepening network to 4+ layers | 4-layer MLP $R^2$ drops from 0.49 to 0.10 | Use 2-3 layers |
| Using TopK point features | TopK MLP (0.46) < full spectrum (0.49) | Use TopK + window |
| Strong regularization on Residual | wd=1e-4 degrades performance | Use wd=0 for Residual |

## 6.3 True Relationship Between Receptive Field and Performance

> **Core Finding: Receptive field hypothesis refuted — large receptive fields degrade performance**

| Kernel | Receptive Field | Test $R^2$ | Parameters | Conclusion |
|--------|-----------------|-----------|------------|------------|
| k=7 | 19 | 0.603 | 23K | Good |
| k=9 | 25 | **0.657** | 27K | **Optimal** |
| k=21 | 61 | 0.035 | 52K | Significant degradation |
| k=45 | 133 | 0.008 | 102K | Failure |
| k=63 | 187 | -0.002 | 140K | Complete failure |

**Physical Interpretation**:
- $\log g$ information primarily derives from **local spectral line features** (width ~10-50 pixels)
- Small kernel (k=9) suffices to capture individual line shape
- **Global integration** achieved via AdaptiveAvgPool, no need for large convolutional receptive fields
- Large kernels cause **parameter explosion + overfitting**

---

# 7. Future Research Directions

## 7.1 High Priority (P0)

| Direction | Task | Expected Benefit | Status |
|-----------|------|------------------|--------|
| **MVP-Local-1** | Top-K Window + CNN/Transformer | Surpass full-spectrum CNN (0.657) @ noise=0.1 | In progress |
| **MVP-Global-1** | Global Feature Tower + MLP | Achieve R²≥0.50 @ noise=1.0 | In progress |
| **CNN noise=1.0 test** | Test optimal CNN config at noise=1.0 | Fair comparison with LightGBM/MLP | Pending |
| **100k fair comparison** | Retrain Ridge/LightGBM with 100k data | Fair comparison with NN | Pending |
| **Dilated CNN + lr=3e-3** | Retest dilated CNN with high learning rate | Validate dilation effectiveness | Pending |
| **Teacher-Student distillation** | Distill using new Teacher latent (seg\_mean) | Validate distillation viability | In progress |

## 7.2 Medium Priority (P1)

| Direction | Task | Expected Benefit |
|-----------|------|------------------|
| **Swin-1D** | Hierarchical local attention | Surpass LightGBM with 100k data |
| **CNN + Residual** | Small-kernel CNN learns Ridge residuals | Potential +3-5% |
| **TopK + window** | Extract TopK neighborhood windows (±8) | Preserve local context |
| **Dual-channel CNN** | [flux, error] dual-channel input | Leverage error information |
| **GTA Phase 2-5** | Complete F2-F6 feature family experiments | Guide Global Tower design |

## 7.3 Low Priority (P2)

| Direction | Task | Expected Benefit |
|-----------|------|------------------|
| **CNN + Attention** | CNN local features + Attention global integration | Hybrid architecture |
| **Position Encoding** | Add positional encoding to CNN | Leverage absolute wavelength position |
| **Physics-Informed ViT** | Patch design + physics positional encoding | Incorporate physical priors |

## 7.4 Hypotheses to Validate

| Hypothesis | Validation Experiment | Current Status |
|------------|-----------------------|----------------|
| Top-K Window surpasses full-spectrum CNN | MVP-Local-1 | In progress |
| 158-dim Global Feature achieves R²≥0.50 @ noise=1.0 | MVP-Global-1 | In progress |
| Swin-1D surpasses LightGBM with 100k data | Swin MVP | Planned |
| Small-kernel CNN remains superior to MLP at noise=1.0 | CNN @ noise=1.0 | Pending |
| Dilated CNN with high lr improves performance | Dilated + lr=3e-3 | Pending |
| CNN + Linear Shortcut further improves | CNN Residual | Pending |
| Distilled Student surpasses original MLP | Teacher-Student | In progress |

---

# 8. Appendix: Quick Navigation

## 8.1 By Model Type

| Model | Primary Directory | Core File | Best $R^2$ |
|-------|------------------|-----------|------------|
| **CNN (Small-Kernel)** | cnn/ | exp_cnn_dilated_kernel_sweep | **0.657** |
| **Swin-1D** | swin/ | swin_main | Planned |
| **MLP (Residual)** | NN/ | exp_nn_comprehensive_analysis | 0.498 |
| **Latent Probe** | distill/ | exp_latent_extraction_logg | 0.55 |
| **LightGBM** | lightgbm/ | exp_lightgbm_hyperparam_sweep | 0.536 |
| **Ridge** | ridge/ | exp_ridge_alpha_sweep | 0.458 |

## 8.2 By Research Topic

| Topic | Directory | Core File | Core Finding |
|-------|-----------|-----------|--------------|
| **CNN Architecture** | cnn/ | cnn_main | Small kernel optimal |
| **Swin-1D** | swin/ | swin_main | Hierarchical attention |
| **Dimensionality Reduction** | pca/ | pca_main | ~100 effective dims |
| **Feature Selection** | noise/ | noise_main | 24% pixels sufficient |
| **Representation Learning** | distill/ | distill_main | Optimized extraction +150% |
| **Architecture Design** | gta/ | gta_main | Spectral data required |

## 8.3 Key Experiment Report Index

| Experiment | Date | Core Finding |
|------------|------|--------------|
| **Swin-1D Architecture** | 2025-12-01 | Hierarchical attention validation |
| **Top-K Window CNN/Transformer** | 2025-12-01 | MVP-Local-1 experiment plan |
| **Global Feature Tower MLP** | 2025-12-01 | MVP-Global-1 experiment plan |
| CNN Kernel Sweep | 2025-12-01 | Small kernel (k=9) optimal |
| NN Comprehensive Analysis | 2025-11-30 | Residual MLP optimal |
| Latent Extraction Optimization | 2025-12-01 | Segmented pooling +77.6% |
| PCA Dimension Analysis | 2025-11-28 | Requires 100+ dims |
| Ridge α Sweep | 2025-11-27 | $R^2=0.999$ @ noise=0 |

---

# 9. Summary: Selecting the Optimal NN Architecture

```
Decision Tree: NN Architecture Selection

Start
  │
  ├── Data volume < 32k?
  │     │
  │     └── Yes → Use Ridge or LightGBM (insufficient data for NN)
  │
  ├── Pursuing maximum performance?
  │     │
  │     ├── Yes → Small-kernel CNN (k=9, lr=3e-3)
  │     │         Expected R² ≈ 0.65 (noise=0.1)
  │     │
  │     └── No → Residual MLP + Linear Shortcut
  │               Expected R² ≈ 0.50 (noise=1.0)
  │
  ├── Have pretrained Denoiser?
  │     │
  │     └── Yes → Latent Probe (enc_pre_latent + seg_mean_K8)
  │               Expected R² ≈ 0.55
  │
  ├── Require physical priors + local features?
  │     │
  │     └── Yes → Top-K Window CNN (K=256/512, W=17)
  │               Residual on Ridge, leverage Top-K physical priors
  │               Expected R² ≥ 0.70 (noise=0.1) [MVP-Local-1]
  │
  ├── Require global + local integration?
  │     │
  │     └── Yes → Dual-Tower Architecture: Global Tower + Local Tower
  │               Global: 158-dim features (PCA+Ridge+TopK+Latent)
  │               Local: Top-K Window CNN
  │               [MVP-Global-1 + MVP-Local-1 → MVP-Joint-1]
  │
  ├── Have 100k+ synthetic data + want to validate large models?
  │     │
  │     └── Yes → Swin-1D (Tiny: 1-2M params)
  │               patch=8, window=8, 2-3 stages
  │               Validate R² > 0.99 at noise=0 first
  │               Expected to surpass LightGBM with 100k data
  │
  └── Need higher performance?
        │
        ├── Increase data volume to 100k+ (+10.6%)
        ├── Try CNN + Residual strategy
        └── Try Swin-1D / Attention architectures
```

---

*Last Updated: 2025-12-01*  
*Total Experiments: 430+ configuration combinations*  
*Core Finding: Small-kernel CNN (k=9) achieves $R^2=0.657$ as current NN optimum; data volume is critical leverage; large receptive field hypothesis refuted*  
*Current Progress: MVP-Local-1 (Top-K Window CNN) & MVP-Global-1 (Global Feature Tower) & Swin-1D Architecture Experiments*

