# Template Fitting 11K 整合实验报告

**Experiment ID**: VIT-20251227-template-fit-11k-01  
**Date**: 2024-12-27 (Updated: 2024-12-28)  
**Status**: Completed ✅

---

## 1. 实验目标

整合 Template Fitting 结果，合并 test_10k 和 test_1k_0 数据集，生成统一的 R² vs SNR 对比图。

## 2. 数据来源

| 数据集 | 文件 | 样本数 | 成功拟合数 | 成功率 |
|--------|------|--------|------------|--------|
| test_10k | `fit_results_noisy_merged.npz` | 10,000 | 9,986 | 99.86% |
| test_1k_0 | `fit_results_noisy.npz` | 1,000 | 997 | 99.70% |
| **合并** | `fit_results_noisy_11k.npz` | **11,000** | **10,983** | **99.85%** |

**数据路径**:
```
/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/
├── test_10k/
│   ├── fit_results_noisy_merged.npz  (10k)
│   └── fit_results_noisy_11k.npz     (11k 合并后)
└── test_1k_0/
    └── fit_results_noisy.npz         (1k)
```

## 3. SNR 分布

| 数据集 | SNR 范围 |
|--------|----------|
| test_10k | [2.79, 27.68] |
| test_1k_0 | [3.14, 20.43] |
| **合并** | [2.79, 27.68] |

## 4. Template Fitting R² vs SNR (Fixed Bins)

使用固定 SNR bins 进行分析，确保与其他方法（LightGBM, ViT）一致对比：

| SNR Bin | Center | 样本数 | R² | R² q10 | R² q90 |
|---------|--------|--------|-----|--------|--------|
| [2.5, 3.5] | 3 | 293 | 0.2132 | 0.1565 | 0.2615 |
| [3.5, 4.5] | 4 | 1,418 | 0.2283 | 0.1896 | 0.2616 |
| [4.5, 5.5] | 5 | 1,291 | 0.2809 | 0.2451 | 0.3209 |
| [5.5, 6.5] | 6 | 1,044 | 0.3569 | 0.3142 | 0.3950 |
| [6.5, 8.5] | 7 | 1,694 | 0.3594 | 0.3212 | 0.3997 |
| [8.5, 12.5] | 10 | 2,486 | 0.5128 | 0.4866 | 0.5424 |
| [12.5, 18.0] | 15 | 2,419 | 0.5645 | 0.5361 | 0.5881 |
| [18.0, 28.0] | 21 | 338 | 0.5387 | 0.4389 | 0.6116 |

**Overall R² = 0.4150** (10,983 successful fits)

### Python 代码复用

```python
# Template Fitting Results (fixed bins, consistent with other methods)
# Data source: fit_results_noisy_11k.npz, 10983 successful fits
# SNR from dataset.h5 (same as used for other methods)
TEMPLATE_RESULTS = {
    'snr': np.array([3, 4, 5, 6, 7, 10, 15, 21]),
    'r2': np.array([0.2132, 0.2283, 0.2809, 0.3569, 0.3594, 0.5128, 0.5645, 0.5387]),
    'r2_q10': np.array([0.1565, 0.1896, 0.2451, 0.3142, 0.3212, 0.4866, 0.5361, 0.4389]),
    'r2_q90': np.array([0.2615, 0.2616, 0.3209, 0.395, 0.3997, 0.5424, 0.5881, 0.6116]),
    'overall': 0.4150
}
```

## 5. 方法对比 (Overall R²)

| 方法 | Overall R² | 训练数据量 |
|------|-----------|-----------|
| Fisher/CRLB (理论上限) | ~0.99 | - |
| **ViT** | **0.7111** | 1M |
| LightGBM | 0.6140 | 1M |
| Ridge | 0.4957 | 1M |
| MLP | 0.4574 | 1M |
| CNN | 0.4289 | 1M |
| **Template Fitting** | **0.4150** | N/A (模板匹配) |

## 6. 关键发现

### 6.1 Template Fitting 特点
- **无需训练**: 直接使用物理模板进行拟合
- **低 SNR 表现差**: SNR < 5 时 R² < 0.30
- **高 SNR 饱和**: SNR > 15 时 R² ≈ 0.53-0.58，未继续提升

### 6.2 与 ML 方法对比
- Template Fitting 在所有 SNR 区间都低于 LightGBM 和 ViT
- 差距在低 SNR 时更明显 (SNR=3: Template=0.21 vs ViT=0.46)
- 高 SNR 时差距缩小但仍显著 (SNR=21: Template=0.54 vs ViT=0.91)

| SNR | Template | Ridge | LightGBM | ViT | Fisher |
|-----|----------|-------|----------|-----|--------|
| 3 | 0.21 | 0.12 | 0.27 | 0.46 | 0.26 |
| 4 | 0.23 | 0.24 | 0.36 | 0.46 | - |
| 5 | 0.28 | 0.40 | 0.43 | 0.54 | 0.70 |
| 7 | 0.36 | 0.53 | 0.59 | 0.71 | 0.87 |
| 10 | 0.51 | 0.58 | 0.72 | 0.80 | - |
| 15 | 0.56 | 0.63 | 0.82 | 0.89 | - |
| 21 | 0.54 | 0.67 | 0.87 | 0.91 | 0.99 |

### 6.3 理论上限差距
- SNR=5 时: Gap = Fisher_Ceiling(0.70) - ViT(0.54) ≈ 0.16
- 说明 ViT 在低 SNR 区域仍有提升空间

## 7. 生成的图表

### 主要图表
| 文件 | 描述 |
|------|------|
| `r2_vs_snr_ceiling_test_10k_unified_snr_with_vit.png` | 统一 SNR bins，包含所有方法对比 |
| `r2_vs_snr_all_methods.png` | 全方法对比图 |
| `r2_vs_snr_ceiling_test_10k_v3_with_vit.png` | 简化版对比图 |

### 图表位置
```
/home/swei20/VIT/results/r2_vs_snr_ceiling/
/home/swei20/Physics_Informed_AI/logg/scaling/exp/img/
```

## 8. 数据文件

### 输入文件
- `test_10k/fit_results_noisy_merged.npz` - 10k 模板拟合结果
- `test_1k_0/fit_results_noisy.npz` - 1k 模板拟合结果

### 输出文件
- `test_10k/fit_results_noisy_11k.npz` - 合并后的 11k 结果

### 字段说明
```python
{
    'log_g_fit': (11000,),      # 拟合的 log g
    'log_g_true': (11000,),     # 真实 log g
    'T_eff_fit': (11000,),      # 拟合的 T_eff
    'T_eff_true': (11000,),     # 真实 T_eff
    'M_H_fit': (11000,),        # 拟合的 [M/H]
    'M_H_true': (11000,),       # 真实 [M/H]
    'success': (11000,),        # 拟合成功标记
    'snr': (11000,),            # 每个样本的 SNR
    'idx': (11000,),            # 原始数据索引
    'source': (11000,),         # 0=test_10k, 1=test_1k_0
}
```

## 9. 复现命令

```bash
# 合并数据
python3 scripts/integrate_template_fit_results.py

# 生成图表
python3 scripts/plot_r2_vs_snr_ceiling_unified_snr.py
python3 scripts/plot_r2_vs_snr_all_methods.py
python3 scripts/plot_r2_vs_snr_ceiling_test10k_standalone.py
```

## 10. 结论

1. **Template Fitting 性能有限**: Overall R² = 0.4150，低于所有 ML 方法
2. **11k 数据集统一**: 合并 test_10k + test_1k_0 提供更稳健的评估
3. **ViT 最优**: 在所有 SNR 区间都显著优于其他方法
4. **理论上限仍有空间**: 与 Fisher/CRLB 相比，ML 方法仍有 0.1-0.2 的提升空间

---

**相关文档**:
- [R² vs SNR 对比图](/home/swei20/Physics_Informed_AI/logg/scaling/exp/img/r2_vs_snr_ceiling_test_10k_unified_snr.png)
- [全方法对比图](/home/swei20/Physics_Informed_AI/logg/scaling/exp/img/r2_vs_snr_all_methods.png)
