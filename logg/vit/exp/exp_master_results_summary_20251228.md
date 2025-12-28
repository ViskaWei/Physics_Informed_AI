# Master Results Summary - log_g Prediction

> **Experiment ID**: VIT-20251228-master-summary
> **Date**: 2025-12-28
> **Task**: log_g prediction from stellar spectra
> **Data**: BOSZ synthetic spectra, 7100-8850Å, mag 20.5-22.5, noise=1.0

---

## 📊 1. Scaling Curve 比较 (test_10k, noise=1.0)

### 1.1 Best Results per Model

| Model | Data Size | Test R² | Test MAE | Notes |
|-------|-----------|---------|----------|-------|
| **ViT-p16_h256_L6** | 1M | **0.711** | 0.372 | Best overall |
| **ViT** | 500k | 0.709 | 0.383 | |
| **ViT** | 200k | 0.673 | 0.415 | |
| **ViT** | 100k | 0.596 | 0.472 | |
| **ViT** | 50k | 0.434 | 0.575 | |
| **LightGBM** (5000 trees) | 1M | 0.571 | 0.584 | |
| **LightGBM** | 500k | 0.574 | 0.579 | |
| **LightGBM** | 100k | 0.553 | 0.599 | |
| **LightGBM** | 50k | 0.488 | 0.647 | |
| **Ridge** (α=5000) | 1M | 0.500 | 0.633 | |
| **Ridge** | 500k | 0.490 | 0.638 | |
| **Ridge** | 100k | 0.475 | 0.648 | |

### 1.2 Key Findings

1. **ViT scales best** with data size (0.43 → 0.71 from 50k to 1M)
2. **ViT > LightGBM** by +0.14 R² at 1M data
3. **LightGBM saturates** around 0.57 R² regardless of data size
4. **Ridge saturates** around 0.50 R² (linear ceiling)

---

## 📊 2. MoE Oracle Results

### 2.1 By SNR Bins

| SNR Bin | n_test | Oracle R² | Global R² | Δ R² |
|---------|--------|-----------|-----------|------|
| SNR > 7 | 31 | 0.704 | 0.672 | +0.032 |
| **SNR ∈ (4,7]** | 292 | **0.710** | 0.614 | **+0.096** |
| SNR ∈ (2,4] | 377 | 0.511 | 0.504 | +0.007 |
| SNR ∈ (0,2] | 300 | 0.309 | 0.252 | +0.058 |

**Key**: Medium SNR (4-7) benefits most from MoE (+9.6% R²)

### 2.2 By Teff × MH (9-bin)

| Bin | Description | Oracle R² | Global R² | Δ R² |
|-----|-------------|-----------|-----------|------|
| 5 | Mid Metal-rich | **0.874** | 0.772 | +0.103 |
| 2 | Cool Metal-rich | 0.847 | 0.763 | +0.084 |
| 8 | Hot Metal-rich | 0.825 | 0.675 | +0.150 |
| 1 | Cool Solar | 0.796 | 0.650 | +0.146 |
| 7 | Hot Solar | 0.601 | 0.547 | +0.054 |
| 4 | Mid Solar | 0.583 | 0.545 | +0.038 |
| 0 | Cool Metal-poor | 0.543 | 0.351 | +0.193 |
| 6 | Hot Metal-poor | 0.447 | 0.276 | +0.171 |
| 3 | Mid Metal-poor | 0.307 | 0.138 | +0.169 |

**Key Findings**:
- **Metal-rich samples** are easiest to predict (R² > 0.82)
- **Metal-poor samples** have largest Δ R² (+0.17-0.19) from specialization
- **Cool + Metal-rich** = best combination

### 2.3 MoE with Learned Gate

| Bin | R² Classify | R² MLP Gate | R² Linear Gate |
|-----|-------------|-------------|----------------|
| Cool Metal-rich | 0.984 | **0.989** | 0.962 |
| Mid Metal-rich | 0.981 | 0.980 | 0.981 |
| Hot Metal-rich | 0.968 | **0.974** | 0.963 |
| Cool Solar | 0.967 | **0.969** | 0.943 |
| Mid Solar | 0.921 | **0.940** | 0.925 |
| Hot Solar | 0.943 | **0.960** | 0.916 |

**Key**: MLP gate outperforms linear gate on most bins

---

## 📊 3. De-leakage Analysis

| Feature Strategy | n_features | log_g R² | SNR R² | Passed |
|------------------|------------|----------|--------|--------|
| S0_baseline (raw error) | 4096 | 0.788 | **0.974** | ❌ |
| S2_template_scale | 2 | -0.001 | 0.808 | ✅ |
| S3_quantiles_11 | 11 | 0.049 | 0.814 | ✅ |
| S4_snr_only | 1 | 0.0 | 0.805 | ✅ |

**Key**: Raw error vector leaks SNR → need careful feature engineering

---

## 📊 4. Model Configurations

### ViT-1M (Best)
```yaml
model:
  hidden_size: 256
  num_hidden_layers: 6
  num_attention_heads: 8
  patch_size: 16
  intermediate_size: 1024

train:
  epochs: 200
  batch_size: 256
  lr: 0.0003
  weight_decay: 0.01
```

### LightGBM (Best)
```yaml
n_estimators: 5000
max_depth: 63
learning_rate: 0.05
feature_fraction: 0.8
bagging_fraction: 0.8
```

### Ridge (Best)
```yaml
alpha: 5000  # or 1e5 for some experiments
```

---

## 📊 5. Result Files Location

| Result Type | Path |
|-------------|------|
| VIT Scaling | `results/vit_scaling_summary.json` |
| Ridge Ceiling | `results/scaling_ml_ceiling/ridge_results.csv` |
| LightGBM Ceiling | `results/scaling_ml_ceiling/lightgbm_results.csv` |
| MoE SNR Oracle | `results/logg_snr_oracle_moe/per_bin_results.csv` |
| MoE 9-bin Oracle | `results/scaling_oracle_moe/per_bin_results.csv` |
| MoE Regression Gate | `results/moe/regression_gate/per_bin_results.csv` |
| De-leakage | `results/logg_snr_moe/deleakage_results.csv` |

---

## 🎯 Summary

| Metric | Best Model | Value |
|--------|------------|-------|
| **Overall R²** | ViT-1M | **0.711** |
| **Best MoE bin** | Mid Metal-rich | **0.874** |
| **Largest MoE gain** | Cool Metal-poor | **+0.193** |
| **ViT advantage over LGBM** | @ 1M data | **+0.14 R²** |

---

## 📌 Next Steps

1. [ ] Train ViT with MoE routing
2. [ ] Test on real SDSS data
3. [ ] Add uncertainty quantification
4. [ ] Multi-task: Teff + log_g + MH jointly

---

*Generated: 2025-12-28*
*Source: VIT experiments repository*
