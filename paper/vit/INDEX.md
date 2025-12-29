# VIT Paper Materials Index

## Overview

This directory contains all technical documentation for writing the paper on Vision Transformer for stellar surface gravity estimation.

**Best Model Checkpoint**: `best_epoch=128-val_mae=0.3720-val_r2=0.7182.ckpt`  
**Model Location**: `/home/swei20/VIT/checkpoints/vit_1m/`  
**Code Repository**: `/home/swei20/VIT/`

---

## Quick Results

| Metric | Value |
|--------|-------|
| **Validation MAE** | 0.3720 dex |
| **Validation R²** | 0.7182 |
| **Best Epoch** | 128 / 200 |
| **Total Parameters** | 4.88M |
| **Training Data** | 200K samples |

---

## Document Index

### 1. [VIT_PAPER_TECHNICAL_REPORT.md](VIT_PAPER_TECHNICAL_REPORT.md)
**Comprehensive Technical Report**

Contains:
- Executive summary with key results
- Detailed model architecture (ViT configuration, parameter counts)
- Data pipeline description
- Training configuration (optimizer, scheduler, hyperparameters)
- Evaluation results
- Code structure overview
- Reproduction guide
- Mathematical formulation
- Appendix with full config

**Use for**: Understanding the complete system, reference during writing

---

### 2. [METHODS_DRAFT.md](METHODS_DRAFT.md)
**Paper Methods Section Draft**

Contains:
- Model architecture description (paper-ready)
- Data section with tables
- Training methodology
- Evaluation metrics
- Suggested figure captions
- References

**Use for**: Direct incorporation into paper methods/experiments sections

---

### 3. [DATA_DESCRIPTION.md](DATA_DESCRIPTION.md)
**Detailed Data Documentation**

Contains:
- Dataset summary and statistics
- HDF5 file structure
- Stellar parameter ranges
- Instrumental configuration (PFS MR)
- Noise properties
- Data loading code examples
- Generation pipeline details

**Use for**: Data section of paper, supplementary materials

---

### 4. [CODE_SNIPPETS.py](CODE_SNIPPETS.py)
**Key Code Implementations**

Contains:
- MyViT model class
- Conv1D patch tokenizer
- SpectraEmbeddings
- Training step implementation
- Data loading code
- Optimizer configuration
- Inference example
- ViT config factory

**Use for**: Methods section code descriptions, supplementary code

---

## Key Files in Repository

### Configuration
```
~/VIT/configs/exp/vit_1m_l1.yaml      # Training config (L1 loss)
~/VIT/configs/exp/vit_scaling_1M.yaml  # Alternative config (MSE loss)
```

### Model Code
```
~/VIT/src/models/specvit.py           # Main ViT model
~/VIT/src/models/embedding.py         # Patch embeddings
~/VIT/src/models/tokenization.py      # Patch tokenizers
~/VIT/src/models/builder.py           # Model factory
```

### Training Code
```
~/VIT/train_nn.py                     # Entry point
~/VIT/src/base/vit.py                 # Lightning module
~/VIT/src/nn/trainer.py               # Training utilities
~/VIT/src/opt/optimizer.py            # Optimizer configuration
```

### Data Code
```
~/VIT/src/dataloader/base.py          # Base dataset
~/VIT/src/dataloader/spec_datasets.py # Regression dataset
```

### Checkpoint
```
~/VIT/checkpoints/vit_1m/best_epoch=128-val_mae=0.3720-val_r2=0.7182.ckpt
```

---

## Data Locations

### Training Data
```
/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/train_200k_0/dataset.h5
```

### Validation Data
```
/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/val_1k/dataset.h5
```

### Test Data
```
/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/test_10k/dataset.h5
```

---

## Model Architecture Summary

```
Input: Spectrum (4096,)
    ↓
Conv1D Patch Tokenizer (16 → 256)
    ↓
256 patch tokens + 1 CLS token
    ↓
+ Learned Position Embeddings
    ↓
6 × Transformer Encoder Layer
    (256 dim, 8 heads, 1024 FFN)
    ↓
CLS Token Output (256,)
    ↓
Linear Regressor (256 → 1)
    ↓
Output: log_g prediction
```

---

## Key Hyperparameters

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | Patch size | 16 |
| | Hidden dim | 256 |
| | Layers | 6 |
| | Heads | 8 |
| **Training** | Epochs | 200 |
| | Batch size | 256 |
| | Precision | 16-mixed |
| **Optimizer** | Type | AdamW |
| | LR | 3e-4 |
| | Weight decay | 0.01 |
| | Scheduler | Cosine |
| **Data** | Train samples | 200,000 |
| | Label norm | Z-score |
| | Noise level | 1.0 |

---

## Suggested Paper Structure

### Abstract
- Task: Stellar parameter estimation from spectra
- Method: Vision Transformer adapted for 1D spectroscopy
- Results: MAE 0.37 dex, R² 0.72 on log g prediction

### Introduction
- Motivation for ML in stellar spectroscopy
- Vision Transformers for sequence modeling
- Contribution: ViT for 1D spectral regression

### Methods
- Model Architecture (see METHODS_DRAFT.md §2)
- Data (see METHODS_DRAFT.md §3)
- Training (see METHODS_DRAFT.md §4)

### Experiments
- Dataset description
- Evaluation metrics
- Results and analysis

### Discussion
- Comparison with baselines
- Limitations
- Future work

### Conclusion
- Summary of contributions

---

## Figure Suggestions

1. **Model Architecture Diagram**: ViT pipeline from spectrum to prediction
2. **Training Curves**: Loss and metrics over epochs
3. **Predictions vs True**: Scatter plot on test set
4. **Residual Analysis**: Residuals vs true log_g
5. **Attention Visualization**: Sample attention maps

---

## Reproduction Commands

```bash
# Train model
cd ~/VIT
python train_nn.py --config configs/exp/vit_1m_l1.yaml --gpu 0

# Evaluate checkpoint
python -c "
import torch
from src.models import get_model
from src.utils.utils import load_config

config = load_config('configs/exp/vit_1m_l1.yaml')
model = get_model(config)

ckpt = torch.load('checkpoints/vit_1m/best_epoch=128-val_mae=0.3720-val_r2=0.7182.ckpt')
model.load_state_dict({k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()})
print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters')
"
```

---

*Index generated: December 28, 2025*
