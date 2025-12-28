# Template Fitting Results Integration & R² vs SNR Analysis

**Experiment ID:** VIT-20251227-template-fit-integration  
**Date:** 2024-12-27  
**Status:** Completed ✅

---

## 1. 目标 (Objective)

整合现有 Template Fitting 结果，合并 `test_10k` 和 `test_1k_0` 两个测试集的拟合结果，生成统一的 R² vs SNR 分析图。

## 2. 数据来源 (Data Sources)

### 2.1 Template Fitting 结果

| 测试集 | 文件路径 | 样本数 | 成功拟合 | 成功率 |
|--------|----------|--------|----------|--------|
| test_10k | `/datascope/.../test_10k/fit_checkpoint_noisy.npz` | 8,600 | 8,586 | 99.8% |
| test_1k_0 | `/datascope/.../test_1k_0/fit_results_noisy.npz` | 1,000 | 997 | 99.7% |
| **Combined** | - | **9,600** | **9,583** | **99.8%** |

### 2.2 数据格式

每个 npz 文件包含：
- `log_g_fit`, `log_g_true`: 拟合值与真实值
- `T_eff_fit`, `T_eff_true`: 有效温度
- `M_H_fit`, `M_H_true`: 金属丰度
- `success`: 拟合是否成功 (bool)
- `idx`: 样本索引
- `*_err`: 拟合误差

## 3. 方法 (Methodology)

### 3.1 SNR 计算

使用 median per-pixel SNR 方法（与 LightGBM/ViT 一致）：

```python
snr = np.median(flux / error, axis=1)
```

### 3.2 R² vs SNR 分组

- 使用 **quantile-based binning** (8 bins)
- 每个 bin 约 1,198 个样本
- 确保各 bin 样本量均衡

### 3.3 整合流程

```
test_10k (8600 samples)  ─┐
                          ├──> Combined (9600 samples) ──> R² vs SNR
test_1k_0 (1000 samples) ─┘
```

## 4. 结果 (Results)

### 4.1 Overall Performance

| 数据集 | Overall R² (log_g) | SNR Range |
|--------|-------------------|-----------|
| test_1k_0 only | 0.4039 | [2.61, 18.02] |
| test_10k only | 0.4182 | [2.37, 18.81] |
| **Combined** | **0.4168** | **[2.37, 18.81]** |

### 4.2 R² vs SNR (Combined)

| Bin | SNR Range | Median SNR | R² | N Samples |
|-----|-----------|------------|-----|-----------|
| 1 | [2.4, 3.6] | 3.26 | 0.219 | 1,198 |
| 2 | [3.6, 4.5] | 4.07 | 0.256 | 1,198 |
| 3 | [4.5, 5.6] | 5.04 | 0.363 | 1,198 |
| 4 | [5.6, 7.0] | 6.31 | 0.342 | 1,197 |
| 5 | [7.0, 8.8] | 7.87 | 0.469 | 1,198 |
| 6 | [8.8, 10.9] | 9.76 | 0.533 | 1,198 |
| 7 | [10.9, 13.2] | 12.05 | 0.571 | 1,198 |
| 8 | [13.2, 18.8] | 14.61 | 0.584 | 1,198 |

### 4.3 与其他方法对比

| Method | Overall R² | 训练数据 | 备注 |
|--------|-----------|----------|------|
| Fisher/CRLB | ~0.99 | N/A | 理论最优上限 |
| ViT | 0.711 | 1M | 深度学习最佳 |
| LightGBM | 0.614 | 1M | 梯度提升树 |
| Ridge | 0.496 | 1M | 线性基线 |
| MLP | 0.457 | 1M | 多层感知机 |
| CNN | 0.429 | 1M | 一维卷积网络 |
| **Template Fit** | **0.417** | N/A | 传统拟合方法 |

### 4.4 关键发现

1. **Template Fitting 性能**
   - 在低 SNR 区域 (SNR < 5) 表现较差，R² < 0.4
   - 在高 SNR 区域 (SNR > 10) 逐渐改善，但仍低于 ML 方法
   - 存在非单调现象：SNR=6.31 时 R² 略有下降

2. **与 ML 方法的差距**
   - 在 SNR ≈ 5 时，ViT 领先 Template Fitting 约 0.23
   - 在 SNR ≈ 15 时，ViT 领先约 0.32
   - **差距随 SNR 增加而扩大**

3. **与理论上限的差距**
   - Fisher/CRLB 在 SNR=5 时 R² ≈ 0.7
   - Template Fitting 在 SNR=5 时 R² ≈ 0.36
   - **Gap ≈ 0.34**，表明 Template Fitting 远未达到理论最优

## 5. 输出文件 (Output Files)

### 5.1 数据文件

| 文件 | 路径 | 描述 |
|------|------|------|
| Combined Results | `/datascope/.../template_fit_combined/template_fit_combined.npz` | 合并的拟合结果 |
| R² vs SNR | `/datascope/.../template_fit_combined/r2_vs_snr_template.npz` | 预计算的 R² vs SNR |
| Project Copy | `/home/swei20/VIT/results/template_fit/` | 项目目录副本 |

### 5.2 图像文件

| 图像 | 路径 | 描述 |
|------|------|------|
| Unified SNR + ViT | `results/r2_vs_snr_ceiling/r2_vs_snr_ceiling_test_10k_unified_snr_with_vit.png` | 主图：含 ViT 和 Fisher ceiling |
| All Methods | `results/scaling_r2_vs_snr/r2_vs_snr_all_methods.png` | 所有方法对比 |
| Standalone | `results/r2_vs_snr_ceiling/r2_vs_snr_ceiling_test_10k_v3_with_vit.png` | 独立绘图版本 |

### 5.3 知识中心副本

所有图像已同步至：
```
~/Physics_Informed_AI/logg/scaling/exp/img/
├── r2_vs_snr_ceiling_test_10k_unified_snr.png
├── r2_vs_snr_ceiling_test_10k.png
└── r2_vs_snr_all_methods.png
```

## 6. 代码 (Scripts)

### 6.1 整合脚本

```bash
# 整合 template fitting 结果
python scripts/integrate_template_fit_results.py
```

### 6.2 绘图脚本

```bash
# Unified SNR 版本（主图）
python scripts/plot_r2_vs_snr_ceiling_unified_snr.py

# 所有方法对比
python scripts/plot_r2_vs_snr_all_methods.py

# Standalone 版本
python scripts/plot_r2_vs_snr_ceiling_test10k_standalone.py
```

## 7. 结论 (Conclusions)

### 7.1 Template Fitting 局限性

1. **SNR 敏感性**：在低 SNR 环境下性能急剧下降
2. **非单调性**：R² 随 SNR 增加并非严格单调，存在局部波动
3. **性能上限**：即使在高 SNR 区域，仍显著低于 ML 方法

### 7.2 ML 方法优势

1. **ViT 最优**：在所有 SNR 区间均显著领先
2. **LightGBM 稳健**：作为非深度学习方法，表现良好
3. **Gap 分析**：ML 方法更接近 Fisher/CRLB 理论上限

### 7.3 物理意义

- Template Fitting 依赖模板库的离散采样
- ML 方法能够学习连续的参数-光谱映射
- 在噪声环境下，ML 的正则化特性提供更好的泛化能力

## 8. 后续工作 (Future Work)

- [ ] 分析 Template Fitting 在不同 Teff/MH 区间的表现
- [ ] 研究 Template Fitting 失败案例的特征
- [ ] 探索 Template + ML 混合方法

---

## 附录：数据格式

### Combined NPZ 文件结构

```python
{
    'log_g_fit': np.array([...]),      # (9600,) 拟合值
    'log_g_true': np.array([...]),     # (9600,) 真实值
    'T_eff_fit': np.array([...]),      # (9600,) Teff 拟合
    'T_eff_true': np.array([...]),     # (9600,) Teff 真实
    'M_H_fit': np.array([...]),        # (9600,) MH 拟合
    'M_H_true': np.array([...]),       # (9600,) MH 真实
    'success': np.array([...]),        # (9600,) bool
    'snr': np.array([...]),            # (9600,) 计算的 SNR
    'idx': np.array([...]),            # (9600,) 原始索引
    'source': np.array([...]),         # (9600,) 'test_10k' or 'test_1k_0'
    'overall_r2_logg': 0.4168,         # 总体 R²
}
```

### R² vs SNR NPZ 文件结构

```python
{
    'snr_centers': np.array([3.26, 4.07, 5.04, 6.31, 7.87, 9.76, 12.05, 14.61]),
    'r2_values': np.array([0.219, 0.256, 0.363, 0.342, 0.469, 0.533, 0.571, 0.584]),
    'n_samples': np.array([1198, 1198, 1198, 1197, 1198, 1198, 1198, 1198]),
    'overall_r2': 0.4168,
    'method': 'template_fitting',
}
```

---

*Generated: 2024-12-27*  
*Author: Claude (Cursor AI)*  
*Project: VIT - Spectral Parameter Estimation*
