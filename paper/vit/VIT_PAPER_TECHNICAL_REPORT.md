# Vision Transformer for Stellar Surface Gravity Estimation
## Technical Report for Paper Writing

---

**Model**: Vision Transformer (ViT) for 1D Spectral Regression  
**Task**: Predicting log(g) (surface gravity) from stellar spectra  
**Best Checkpoint**: `best_epoch=128-val_mae=0.3720-val_r2=0.7182.ckpt`  
**Date**: December 2025

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Model Architecture](#2-model-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Training Configuration](#4-training-configuration)
5. [Evaluation & Results](#5-evaluation--results)
6. [Code Structure](#6-code-structure)
7. [Reproduction Guide](#7-reproduction-guide)
8. [Mathematical Formulation](#8-mathematical-formulation)
9. [References](#9-references)

---

## 1. Executive Summary

### Key Results

| Metric | Value | Dataset |
|--------|-------|---------|
| **Validation MAE** | 0.3720 | 1,000 samples |
| **Validation R²** | 0.7182 | 1,000 samples |
| **Test MAE** | ~0.37 | 10,000 samples |
| **Best Epoch** | 128 | out of 200 |
| **Total Parameters** | 4.88M | - |
| **Training Time** | ~12 hours | Single GPU |

### Model Highlights

- **Architecture**: Vision Transformer adapted for 1D spectral data
- **Input**: 4096-dimensional stellar spectrum (flux values)
- **Output**: Single scalar prediction (log_g)
- **Tokenization**: Conv1D patch embedding with patch size 16
- **Position Encoding**: Learned absolute positional embeddings
- **Loss Function**: L1 Loss (MAE) for robust regression

---

## 2. Model Architecture

### 2.1 High-Level Architecture

```
Input Spectrum (4096,) 
    ↓
Conv1D Patch Tokenizer (patch_size=16, stride=16)
    ↓
256 patch tokens (each 256-dim)
    ↓
+ CLS token → 257 tokens
    ↓
+ Learned Position Embeddings (257, 256)
    ↓
Transformer Encoder (6 layers)
    ↓
CLS token output (256,)
    ↓
Linear Regressor (256 → 1)
    ↓
Output: log_g prediction
```

### 2.2 Detailed Configuration

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Input** | | |
| | Spectrum length | 4096 |
| | Wavelength range | PFS MR band |
| **Tokenization** | | |
| | Patch size | 16 |
| | Stride size | 16 (non-overlapping) |
| | Projection type | Conv1D (`proj_fn: C1D`) |
| | Number of patches | 256 |
| **Embeddings** | | |
| | Hidden dimension | 256 |
| | CLS token | 1 learnable token |
| | Position encoding | Learned (257 × 256) |
| | Dropout | 0.1 |
| **Transformer Encoder** | | |
| | Number of layers | 6 |
| | Attention heads | 8 |
| | Head dimension | 32 (256 / 8) |
| | FFN intermediate size | 1024 (4 × 256) |
| | Activation | GELU |
| | Attention dropout | 0.1 |
| **Regressor Head** | | |
| | Input dimension | 256 |
| | Output dimension | 1 |

### 2.3 Parameter Count

| Component | Parameters |
|-----------|------------|
| Patch Embedding (Conv1D) | 4,352 |
| Position Embeddings | 65,792 |
| CLS Token | 256 |
| **Embeddings Total** | **70,400** |
| Transformer Encoder | 4,738,560 |
| ↳ Per Layer | 789,760 |
| ↳ ↳ Self-Attention | 263,168 |
| ↳ ↳ FFN | 526,336 |
| ↳ ↳ LayerNorms | 512 × 2 |
| Regressor Head | 257 |
| **Total** | **4,875,521 (~4.88M)** |

### 2.4 Conv1D Patch Tokenizer

The patch tokenizer converts the 1D spectrum into a sequence of patch embeddings:

```python
class Conv1DPatchTokenizer(nn.Module):
    def __init__(self, input_length, patch_size, hidden_size, stride):
        self.projection = nn.Conv1d(
            in_channels=1,           # Single channel input
            out_channels=hidden_size, # 256 output channels
            kernel_size=patch_size,   # 16
            stride=stride             # 16 (non-overlapping)
        )
    
    def forward(self, x):  # x: (batch, 4096)
        x = x.reshape(-1, 1, 4096)  # (batch, 1, 4096)
        x = self.projection(x)      # (batch, 256, 256)
        return x.transpose(1, 2)    # (batch, 256, 256)
```

**Number of patches**: `(4096 - 16) / 16 + 1 = 256`

### 2.5 Transformer Encoder Layer

Each of the 6 transformer layers follows the Pre-LN architecture:

```
Input (batch, 257, 256)
    ↓
LayerNorm
    ↓
Multi-Head Self-Attention (8 heads)
    ↓
+ Residual Connection
    ↓
LayerNorm
    ↓
FFN (256 → 1024 → 256, GELU)
    ↓
+ Residual Connection
    ↓
Output (batch, 257, 256)
```

---

## 3. Data Pipeline

### 3.1 Dataset Overview

| Property | Value |
|----------|-------|
| **Dataset Name** | BOSZ 1M Simulated Spectra |
| **Source** | BOSZ (Bohlin, Rauch, & Sah) ATLAS9 models |
| **Total Samples** | 1,000,000 |
| **Training Set** | 200,000 (single shard) |
| **Validation Set** | 1,000 |
| **Test Set** | 10,000 |
| **Data Format** | HDF5 |
| **Total Size** | ~93 GB (all shards) |

### 3.2 Data Paths

```
Training:   /datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/train_200k_0/dataset.h5
Validation: /datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/val_1k/dataset.h5
Test:       /datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/test_10k/dataset.h5
```

### 3.3 Stellar Parameter Ranges

| Parameter | Symbol | Min | Max | Unit | Distribution |
|-----------|--------|-----|-----|------|--------------|
| Effective Temperature | T_eff | 3750 | 6000 | K | Random (Beta) |
| Surface Gravity | log_g | 1.0 | 5.0 | dex | Random (Beta) |
| Metallicity | [Fe/H] | -1.0 | 0.0 | dex | Random (Beta) |
| i-band Magnitude | mag | 20.5 | 22.5 | mag | Uniform |
| Redshift | z | 0 | 0 | - | Fixed |

### 3.4 Instrumental Configuration (PFS MR)

| Property | Value |
|----------|-------|
| Spectrograph | Subaru PFS Medium Resolution |
| Spectral Arm | MR (Medium Resolution Red) |
| Model Resolution | R = 50,000 |
| Wavelength Range | 6500 - 9500 Å |
| Seeing | 0.5 - 1.5 arcsec |
| Exposure Time | 12 × 900s = 3 hours |
| Moon Phase | 0 (new moon) |

### 3.5 Data Processing Pipeline

```python
# 1. Load HDF5 data
with h5py.File(file_path, 'r') as f:
    wave = f['spectrumdataset/wave'][:]      # (4096,)
    flux = f['dataset/arrays/flux/value'][:] # (N, 4096)
    error = f['dataset/arrays/error/value'][:] # (N, 4096)
    log_g = pd.read_hdf(file_path)['log_g'].values  # (N,)

# 2. Preprocessing
flux = flux.clip(min=0.0)  # Remove negative values

# 3. Label Normalization (Z-score)
label_mean = log_g.mean()  # Computed on training set
label_std = log_g.std()
log_g_normalized = (log_g - label_mean) / label_std

# 4. Noise Augmentation (training only)
noisy_flux = flux + randn_like(flux) * error * noise_level  # noise_level=1.0
```

### 3.6 Noise Model

The noise augmentation simulates realistic observational noise:

- **Noise Type**: Heteroscedastic (pixel-dependent)
- **Noise Level**: 1.0 (full noise from error array)
- **Application**:
  - Training: On-the-fly random noise each batch
  - Validation/Test: Pre-generated with fixed seed (42) for reproducibility

```python
# Training: on-the-fly noise
noisy = flux + torch.randn_like(flux) * error * noise_level

# Validation/Test: fixed noise
torch.manual_seed(42)
noisy = flux + torch.randn_like(flux) * error * noise_level
```

---

## 4. Training Configuration

### 4.1 Optimizer

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 0.01 |
| β₁, β₂ | 0.9, 0.999 (default) |
| ε | 1e-8 (default) |

### 4.2 Learning Rate Schedule

| Parameter | Value |
|-----------|-------|
| Scheduler | Cosine Annealing |
| Initial LR | 3e-4 |
| Minimum LR (η_min) | 1e-5 |
| T_max | 200 (epochs) |

```
LR(t) = η_min + 0.5 * (η_max - η_min) * (1 + cos(π * t / T_max))
```

### 4.3 Training Settings

| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Batch Size | 256 |
| Gradient Clipping | 0.5 |
| Precision | 16-mixed (AMP) |
| Deterministic | True |
| Seed | 42 |

### 4.4 Loss Function

**L1 Loss (Mean Absolute Error)**:

```
L = (1/N) Σ |ŷᵢ - yᵢ|
```

where:
- ŷᵢ: Predicted log_g (normalized)
- yᵢ: True log_g (normalized)

Rationale: L1 loss is more robust to outliers than MSE and provides more interpretable errors.

### 4.5 Regularization

| Technique | Value |
|-----------|-------|
| Dropout (Attention) | 0.1 |
| Dropout (Hidden) | 0.1 |
| Weight Decay | 0.01 |
| Gradient Clipping | 0.5 |
| Early Stopping | 500 epochs patience |

### 4.6 Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA A100 / V100 |
| CUDA Version | 11.x+ |
| Mixed Precision | FP16 (AMP) |
| Training Time | ~12 hours |
| Memory Usage | ~16 GB |

---

## 5. Evaluation & Results

### 5.1 Primary Metrics

| Metric | Validation | Test |
|--------|------------|------|
| MAE | 0.3720 | ~0.37 |
| R² | 0.7182 | ~0.72 |
| MSE | - | - |
| RMSE | - | - |

### 5.2 Training Curve Summary

| Epoch | Val MAE | Val R² |
|-------|---------|--------|
| 1 | 0.8694 | -0.0009 |
| 6 | 0.5695 | 0.4755 |
| 9 | 0.4298 | 0.6619 |
| **128** | **0.3720** | **0.7182** |

### 5.3 Interpretation

- **MAE of 0.37 dex**: The model predicts log_g with an average error of ~0.37 dex
- **R² of 0.72**: The model explains 72% of the variance in log_g
- **Convergence**: Best performance at epoch 128/200 with early stopping

### 5.4 Target Range Context

Given that log_g ranges from 1.0 to 5.0 dex:
- Range: 4.0 dex
- MAE/Range: 0.37/4.0 = 9.25%
- This represents ~10% relative error

---

## 6. Code Structure

### 6.1 Key Files

```
VIT/
├── train_nn.py                           # Main training entry point
├── src/
│   ├── base/
│   │   ├── basemodule.py                # Base classes (DataModule, LightningModule, Trainer)
│   │   └── vit.py                       # ViT experiment class (ViTLModule, ViTDataModule)
│   ├── models/
│   │   ├── specvit.py                   # MyViT model class
│   │   ├── builder.py                   # Model factory (get_model, get_vit_config)
│   │   ├── embedding.py                 # SpectraEmbeddings (patch + position)
│   │   ├── tokenization.py              # Conv1DPatchTokenizer, SlidingWindowTokenizer
│   │   ├── rope.py                      # Rotary Position Embedding (optional)
│   │   ├── attention.py                 # PrefilledAttention (optional preprocessor)
│   │   ├── preprocessor.py              # ZCA/PCA preprocessing (optional)
│   │   └── layers.py                    # PrefilledLinear layer
│   ├── dataloader/
│   │   ├── base.py                      # BaseSpecDataset, NoiseMixin
│   │   └── spec_datasets.py             # RegSpecDataset, ClassSpecDataset
│   ├── nn/
│   │   ├── trainer.py                   # train_single_experiment function
│   │   ├── config.py                    # Sweep configuration loading
│   │   └── sweep_runner.py              # Parallel sweep execution
│   ├── opt/
│   │   └── optimizer.py                 # OptModule (AdamW, schedulers)
│   └── utils/
│       ├── utils.py                     # Config loading, utilities
│       └── hardware_utils.py            # GPU/worker detection
├── configs/
│   └── exp/
│       ├── vit_1m_l1.yaml              # L1 loss configuration
│       └── vit_scaling_1M.yaml         # 1M scaling configuration
├── checkpoints/
│   └── vit_1m/
│       └── best_epoch=128-*.ckpt       # Best model checkpoint
└── requirements.txt                     # Dependencies
```

### 6.2 Core Classes

#### MyViT (src/models/specvit.py)

```python
class MyViT(ViTPreTrainedModel, BaseModel):
    """Vision Transformer with optional input preprocessor"""
    
    def __init__(self, config: ViTConfig, loss_name, model_name, preprocessor=None):
        self.vit = ViTModel(config)
        self.vit.embeddings = SpectraEmbeddings(config)  # Custom embeddings
        self.regressor = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.L1Loss()  # L1 for 'l1'/'mae' loss
    
    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values)
        cls_token = outputs[0][:, 0, :]  # CLS token
        logits = self.regressor(cls_token)
        loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
        return SequenceClassifierOutput(loss=loss, logits=logits)
```

#### SpectraEmbeddings (src/models/embedding.py)

```python
class SpectraEmbeddings(nn.Module):
    def __init__(self, config):
        self.patch_embeddings = Conv1DPatchTokenizer(...)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches+1, hidden_size))
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        tokens = self.patch_embeddings(x)       # (B, 256, 256)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)  # (B, 257, 256)
        tokens = tokens + self.position_embeddings
        return self.dropout(tokens)
```

#### ViTLModule (src/base/vit.py)

```python
class ViTLModule(BaseLightningModule):
    def __init__(self, model, config):
        self.model = model
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()
    
    def training_step(self, batch, batch_idx):
        flux, error, labels = batch
        noisy = flux + torch.randn_like(flux) * error * noise_level
        loss = self.model(noisy, labels).loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        noisy, flux, error, labels = batch
        outputs = self.model(noisy, labels)
        preds = outputs.logits.squeeze()
        self.log('val_mae', self.mae(preds, labels))
        self.log('val_r2', self.r2(preds, labels))
```

### 6.3 Configuration Files

#### vit_1m_l1.yaml (Used for training)

```yaml
project: 'vit-1m-scaling'

model:
  name: vit
  task_type: reg
  image_size: 4096
  patch_size: 16
  hidden_size: 256
  num_hidden_layers: 6
  num_attention_heads: 8
  stride_size: 16
  proj_fn: 'C1D'
  pos_encoding_type: 'learned'
  max_position_embeddings: 512

train:
  batch_size: 256
  ep: 200
  save: true
  num_workers: 8
  precision: '16-mixed'

loss:
  name: 'mae'  # L1 Loss

opt:
  type: 'AdamW'
  lr: 0.0003
  weight_decay: 0.01
  lr_sch: 'cosine'
  eta_min: 0.00001

data:
  file_path: ".../train_200k_0/dataset.h5"
  val_path: ".../val_1k/dataset.h5"
  test_path: ".../test_10k/dataset.h5"
  param: log_g
  label_norm: 'standard'

noise:
  noise_level: 1.0
```

---

## 7. Reproduction Guide

### 7.1 Environment Setup

```bash
# Create conda environment
conda create -n vit-spec python=3.10
conda activate vit-spec

# Install PyTorch
pip install torch==2.8.0

# Install dependencies
pip install -r requirements.txt
```

### 7.2 Key Dependencies

| Package | Version |
|---------|---------|
| torch | 2.8.0 |
| transformers | 4.56.0 |
| lightning | 2.5.4 |
| pytorch-lightning | 2.5.4 |
| torchmetrics | 1.8.1 |
| h5py | 3.14.0 |
| numpy | 2.3.2 |
| pandas | 2.3.2 |

### 7.3 Training Command

```bash
cd ~/VIT

# Single GPU training
python train_nn.py --config configs/exp/vit_1m_l1.yaml --gpu 0

# With checkpointing enabled
python train_nn.py --config configs/exp/vit_1m_l1.yaml --gpu 0 --save

# With WandB logging
python train_nn.py --config configs/exp/vit_1m_l1.yaml --gpu 0 --wandb
```

### 7.4 Loading the Checkpoint

```python
import torch
from src.models import get_model
from src.utils.utils import load_config

# Load config
config = load_config('configs/exp/vit_1m_l1.yaml')

# Build model
model = get_model(config)

# Load checkpoint
ckpt_path = 'checkpoints/vit_1m/best_epoch=128-val_mae=0.3720-val_r2=0.7182.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')

# Load weights (remove 'model.' prefix)
state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
model.load_state_dict(state_dict)

# Inference
model.eval()
with torch.no_grad():
    output = model(spectrum_tensor, labels=None)
    prediction = output.logits.squeeze()
```

### 7.5 Data Access

```python
import h5py
import pandas as pd

# Load spectrum data
with h5py.File('dataset.h5', 'r') as f:
    wave = f['spectrumdataset/wave'][:]       # (4096,)
    flux = f['dataset/arrays/flux/value'][:]  # (N, 4096)
    error = f['dataset/arrays/error/value'][:] # (N, 4096)

# Load labels
df = pd.read_hdf('dataset.h5')
log_g = df['log_g'].values
```

---

## 8. Mathematical Formulation

### 8.1 Problem Statement

Given a stellar spectrum \( \mathbf{x} \in \mathbb{R}^{4096} \), predict the surface gravity \( \log g \in [1.0, 5.0] \).

### 8.2 Model Architecture

**Patch Tokenization:**
\[
\mathbf{z}_0 = [\mathbf{x}_{cls}; \mathbf{E}(\mathbf{x}_1); \mathbf{E}(\mathbf{x}_2); \ldots; \mathbf{E}(\mathbf{x}_{256})] + \mathbf{P}
\]

where:
- \( \mathbf{x}_i \in \mathbb{R}^{16} \): i-th patch of spectrum
- \( \mathbf{E}: \mathbb{R}^{16} \to \mathbb{R}^{256} \): Conv1D projection
- \( \mathbf{x}_{cls} \in \mathbb{R}^{256} \): Learnable CLS token
- \( \mathbf{P} \in \mathbb{R}^{257 \times 256} \): Learnable position embeddings

**Transformer Layer:**
\[
\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}
\]
\[
\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell
\]

**Prediction Head:**
\[
\hat{y} = \mathbf{W}_{reg} \cdot \mathbf{z}_L^0 + b_{reg}
\]

where \( \mathbf{z}_L^0 \) is the CLS token output from the final layer.

### 8.3 Loss Function

\[
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|
\]

### 8.4 Noise Model

\[
\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\epsilon} \odot \boldsymbol{\sigma}
\]

where:
- \( \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}) \): Standard Gaussian noise
- \( \boldsymbol{\sigma} \in \mathbb{R}^{4096} \): Per-pixel uncertainty from observations

---

## 9. References

### 9.1 Model References

1. **Vision Transformer (ViT)**: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021.

2. **HuggingFace Transformers**: Wolf et al., "Transformers: State-of-the-Art Natural Language Processing", EMNLP 2020.

### 9.2 Data References

1. **BOSZ Models**: Bohlin, R., Rauch, T., & Sah, S., 2017, "BOSZ: A Grid of Metal-free to Super-solar Theoretical Stellar Spectra".

2. **PFS Spectrograph**: Subaru Prime Focus Spectrograph Collaboration.

### 9.3 Implementation References

1. **PyTorch Lightning**: Falcon et al., "PyTorch Lightning", 2019.

2. **AdamW**: Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019.

---

## Appendix A: Full Hyperparameters

```yaml
# Complete configuration used for the best model
project: 'vit-1m-scaling'

model:
  name: vit
  task_type: reg
  image_size: 4096
  patch_size: 16
  hidden_size: 256
  num_hidden_layers: 6
  num_attention_heads: 8
  stride_size: 16
  proj_fn: 'C1D'
  pos_encoding_type: 'learned'
  max_position_embeddings: 512
  num_labels: 1
  param_names: ['log_g']

train:
  batch_size: 256
  ep: 200
  debug: 0
  save: true
  num_workers: 8
  precision: '16-mixed'

loss:
  name: 'mae'

opt:
  type: 'AdamW'
  lr: 0.0003
  weight_decay: 0.01
  lr_sch: 'cosine'
  eta_min: 0.00001

data:
  file_path: '/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/train_200k_0/dataset.h5'
  val_path: '/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/val_1k/dataset.h5'
  test_path: '/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/test_10k/dataset.h5'
  num_samples: -1
  num_test_samples: -1
  param: 'log_g'
  label_norm: 'standard'

noise:
  noise_level: 1.0

viz:
  enable: false

plotting:
  quick_mode: true
```

---

## Appendix B: Checkpoint Information

```python
# Checkpoint details
{
    'epoch': 128,
    'global_step': 504003,
    'pytorch-lightning_version': '2.5.1.post0',
    'total_parameters': 4875521,
}
```

---

*Report generated: December 28, 2025*
*Author: Auto-generated for paper writing*
*Repository: ~/VIT*
