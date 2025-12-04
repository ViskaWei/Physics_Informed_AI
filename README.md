# Physics-Informed AI for Stellar Parameter Prediction

<p align="center">
  <b>Systematic experiments on predicting stellar surface gravity (log g) from synthetic spectra</b>
</p>

---

## Overview

This repository serves as the **knowledge center** for physics-informed machine learning experiments focused on stellar parameter prediction. We systematically explore various model architectures—from linear baselines (Ridge, PCA) to neural networks (MLP, CNN, Swin Transformer) and advanced techniques (Mixture of Experts, knowledge distillation)—to predict stellar surface gravity ($\log g$) from 4096-dimensional synthetic spectral flux.

## Key Findings

| Discovery | Evidence | Implication |
|-----------|----------|-------------|
| **Mapping is inherently linear** | Ridge achieves $R^2=0.999$ @ noise=0 | Linear shortcut is essential for NN design |
| **Small-kernel CNN is optimal** | k=9 achieves $R^2=0.657$ @ noise=0.1 | Large receptive fields hurt performance |
| **Information is high-dimensional but sparse** | ~100 PCs needed; 24% wavelengths sufficient | Feature selection + local context matters |
| **Data volume is critical** | 32k→100k: +10.6% $R^2$ | Prioritize data over architecture complexity |
| **MoE soft routing works** | $\rho=1.00$ oracle gain retention | Physics-based expert partitioning is viable |

## 📖 English Reports

Comprehensive experiment reports are available in the [`EN/`](./EN/) directory:

| Report | Description |
|--------|-------------|
| [**Master Experiment Log**](./EN/logg_main_20251130_en.md) | Complete overview of all $\log g$ prediction experiments, model leaderboard, and NN design guidelines |
| [**MoE Research Hub**](./EN/moe_hub_20251203_en.md) | In-depth analysis of Mixture of Experts for stellar parameter prediction |

## Repository Structure

```
Physics_Informed_AI/
├── EN/                     # 📖 English reports (start here!)
│   ├── logg_main_*_en.md   # Master experiment log
│   └── moe_hub_*_en.md     # MoE research hub
│
├── logg/                   # Experiment logs (Chinese)
│   ├── cnn/                # CNN architecture experiments
│   ├── moe/                # Mixture of Experts experiments
│   ├── NN/                 # MLP neural network experiments
│   ├── ridge/              # Ridge regression baseline
│   ├── pca/                # PCA dimensionality reduction
│   ├── lightgbm/           # Tree model baseline
│   ├── noise/              # Noise robustness & feature selection
│   ├── gta/                # Global Tower Architecture design
│   ├── distill/            # Knowledge distillation & latent probing
│   ├── swin/               # Swin-1D Transformer experiments
│   ├── diffusion/          # Diffusion model experiments
│   └── train/              # Training strategy experiments
│
├── status/                 # Project status tracking
│   ├── kanban.md           # Experiment kanban board
│   └── next_steps.md       # Prioritized next steps
│
└── _backend/               # Templates and utilities
    └── template/           # Report templates
```

## Recommended Neural Network Architectures

Based on 430+ experiments, we recommend the following architectures in priority order:

| Priority | Approach | Configuration | Expected $R^2$ | Key Points |
|:--------:|----------|---------------|----------------|------------|
| 🥇 | **Small-Kernel CNN** | k=9, 2-layer, AdaptiveAvgPool | **0.657** | Small receptive field + global pooling |
| 🥈 | **Residual MLP** | [256,64], learns Ridge residuals | **0.498** | Linear shortcut is critical |
| 🥉 | **Latent Probe** | enc\_pre\_latent + seg\_mean\_K8 | **0.55** | Extracted from Denoiser |
| 🆕 | **MoE (9-Expert)** | Physics-window gate + soft routing | **0.931** | [M/H]-based expert partitioning |

## Critical Design Principles

1. **Always use Linear Shortcut**: $\hat{y} = w^\top x + g_\theta(x)$ — the mapping is fundamentally linear
2. **Small kernels outperform large kernels**: k ∈ {7, 9}, avoid k > 15
3. **Data volume over architecture complexity**: 100k >> 32k (+10.6%)
4. **Preserve wavelength locality**: Segmented pooling >> global mean (+77.6%)
5. **Soft routing for MoE**: Hard routing loses 28% of oracle gain

## Data

| Configuration | Value |
|--------------|-------|
| Training samples | 32,000 / 100,000 |
| Feature dimensionality | 4,096 (spectral flux) |
| Target parameter | $\log g$ (surface gravity) |
| Noise levels | $\sigma \in \{0, 0.1, 0.5, 1.0, 2.0\}$ |
| Data source | BOSZ synthetic spectral library |

## Citation

If you find this work useful, please consider citing:

```
@misc{physics_informed_ai_2025,
  author = {Viska Wei},
  title = {Physics-Informed AI for Stellar Parameter Prediction},
  year = {2025},
  url = {https://github.com/your-username/Physics_Informed_AI}
}
```

## Author

**Viska Wei**

---

*Last Updated: 2025-12-04*
