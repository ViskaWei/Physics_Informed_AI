# Card: R² vs SNR Ceiling Figure

> **Figure Name**: `r2_vs_snr_ceiling_test_10k_unified_snr.png`  
> **Script**: `/home/swei20/VIT/scripts/plot_r2_vs_snr_ceiling_unified_snr.py`  
> **Output Path**: `/home/swei20/Physics_Informed_AI/logg/scaling/exp/img/`  
> **Date**: 2024-12-27

---

## 📊 Figure Overview

这张图展示了 **log g 预测的 R² 随 SNR 变化**，比较了：
1. **理论上限** (Fisher/CRLB 5D)
2. **ML 方法** (LightGBM, ViT)
3. **传统方法** (Template Fitting)

**核心发现**: 在 SNR=5 时，最佳 ML 方法 (ViT) 与理论上限之间仍有 ~0.1 的 gap。

---

## 📁 Data Sources

### 1. Test Dataset (10k samples)
```
Path: /datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/test_10k/dataset.h5
Content:
  - noisy: 加噪光谱 (10000, n_wavelength)
  - flux: 原始光谱 (10000, n_wavelength)
  - error: 误差谱 (10000, n_wavelength)
  - log_g: 真实标签 (10000,)
  - snr: Signal-to-Noise Ratio (10000,) - 从 HDF5 的 df["snr"] 读取
```

### 2. Template Fitting Results (1k samples)
```
Path: /datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/test_1k_0/fit_results_noisy.npz
Content:
  - log_g_fit: 拟合结果
  - log_g_true: 真实标签
  - success: 拟合是否成功
SNR Source: /datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/test_1k_0/dataset.h5
```

### 3. LightGBM Model
```
Path: /home/swei20/VIT/models/lightgbm_1M/lightgbm_1M_best.pkl
Training: 1M samples from train_200k_* shards
```

### 4. ViT Model
```
Checkpoint: /home/swei20/VIT/checkpoints/vit_1m/epoch=92-val_mae=0.3775-val_r2=0.7111.ckpt
Config: /home/swei20/VIT/configs/exp/vit_1m_large.yaml
Training: 1M samples from train_200k_* shards (5 shards x 200k)
```

### 5. Fisher Ceiling (5D CRLB)
```
Source: fisher_5d_multi_mag/combined_summary.json
Parameter Space: T_eff, log_g, M_H, C_M, a_M (5D)
Magnitudes: [18.0, 20.0, 21.5, 22.0, 22.5, 23.0]
```

| Magnitude | SNR (median) | R² Median | R² Mean | R² Std |
|-----------|--------------|-----------|---------|--------|
| 23.0      | 1.9          | 0.0       | 0.1762  | 0.2740 |
| 22.5      | 3.0          | 0.2647    | 0.3570  | 0.3644 |
| 22.0      | 4.6          | 0.6983    | 0.5467  | 0.3784 |
| 21.5      | 7.1          | 0.8742    | 0.8742  | 0.0638 |
| 20.0      | 24.0         | 0.9892    | 0.9653  | 0.0638 |
| 18.0      | 87.4         | 0.9993    | 0.9976  | 0.0044 |

---

## 🔬 Experiments & Methods

### LightGBM R² vs SNR
- **Model**: LightGBM regressor trained on 1M samples
- **Input**: noisy spectrum (直接用 noisy array)
- **Binning**: 8 bins by SNR quantiles
- **Bootstrap**: 200 iterations per bin
- **Uncertainty**: 10-90 percentile range

**Pre-computed values (fallback)**:
```python
snr = [3.2, 4.1, 5.0, 6.3, 7.9, 9.8, 12.0, 14.6]
r2  = [0.36, 0.42, 0.50, 0.59, 0.68, 0.74, 0.80, 0.84]
overall_r2 = 0.6142
```

### ViT R² vs SNR
- **Model**: Vision Transformer (1M samples)
- **Input**: noisy spectrum → normalized
- **Label**: log_g (normalized by training set mean/std)
- **Binning**: 8 bins by SNR quantiles
- **Bootstrap**: 200 iterations per bin

**Pre-computed values (fallback)**:
```python
snr = [3.2, 4.1, 5.0, 6.3, 7.9, 9.8, 12.0, 14.6]
r2  = [0.46, 0.52, 0.60, 0.68, 0.75, 0.82, 0.87, 0.90]
overall_r2 = 0.6979
```

### Template Fitting R² vs SNR
- **Method**: χ² minimization against BOSZ template library
- **Source**: Pre-computed fit results from `fit_results_noisy.npz`
- **Filter**: Only successful fits (`success=True`)
- **Binning**: 8 bins by SNR quantiles

### Fisher/CRLB 5D Ceiling
- **Method**: Fisher Information Matrix + Cramér-Rao Lower Bound
- **Parameters**: 5D (T_eff, log_g, M_H, C_M, a_M)
- **Formula**:
  ```
  I(θ) = J^T Σ^{-1} J,  where J = ∂log L / ∂θ
  CRLB_{log g} = (I_{gg} - I_{gη} I_{ηη}^{-1} I_{ηg})^{-1}  (Schur complement)
  R²_max = 1 - CRLB_{log g} / Var(log g)
  ```
- **Visualization**: Median + 10-90% + ±1σ bands

---

## 📐 SNR Definition

**Unified SNR**: `median(flux / error)` per spectrum
- **Source**: 从 HDF5 文件的 pandas DataFrame 中直接读取 `df["snr"]`
- **Fallback**: 如果不可用，计算 `np.median(flux / (error + 1e-10), axis=1)`

---

## 📈 Key Observations

1. **Gap @ SNR=5**: ViT R² ≈ 0.60, Fisher Ceiling ≈ 0.70, Gap ≈ 0.10
2. **High SNR**: LightGBM 和 ViT 都接近 ceiling
3. **Low SNR (< 3)**: 所有 ML 方法 R² < 0.5
4. **ViT > LightGBM**: ViT 在所有 SNR bins 都优于 LightGBM

---

## 🖼️ Visualization Details

### Plot Elements
| Element | Style | Color | Data |
|---------|-------|-------|------|
| Fisher Ceiling | ○ line + fill | Navy/lightblue | 5D CRLB |
| LightGBM | □ line + fill | Green | test_10k |
| ViT | ◇ line + fill | Orange | test_10k |
| Template Fit | △ line | Red | test_1k |
| Gap marker | -- dashed | Red | @ SNR=5 |

### Info Box
```
train/test1M/10k
7100-8850A
Teff3750-6000K
log(g) 1-5
MH -0.25-0.75
mag 20.5-22.5
noise=1.0
```

### Formula Box
- Fisher Information definition
- CRLB (marginalized via Schur complement)
- R²_max formula

---

## 🔗 Related Files

| File | Description |
|------|-------------|
| `scaling_fisher_ceiling_5d_multi_mag.py` | Fisher ceiling 计算脚本 |
| `card_fisher_ceiling_20251224.md` | Fisher ceiling 详细卡片 |
| `card_ml_ceiling_20251222.md` | ML ceiling 分析 |
| `exp_whitening_design_principles_20251226.md` | 相关实验文档 |

---

## 📝 Reproducibility

```bash
# Run the plotting script
cd /home/swei20/VIT
python scripts/plot_r2_vs_snr_ceiling_unified_snr.py

# Output files:
# 1. /home/swei20/VIT/results/r2_vs_snr_ceiling/r2_vs_snr_ceiling_test_10k_unified_snr_with_vit.png
# 2. /home/swei20/Physics_Informed_AI/logg/scaling/exp/img/r2_vs_snr_ceiling_test_10k_unified_snr.png
```

**Dependencies**:
```
pip install tables  # for pandas HDF5 support
pip install lightgbm
pip install torch pytorch-lightning
```

---

## 🏷️ Tags

`#scaling` `#fisher` `#crlb` `#r2-vs-snr` `#lightgbm` `#vit` `#template-fitting` `#ceiling`
