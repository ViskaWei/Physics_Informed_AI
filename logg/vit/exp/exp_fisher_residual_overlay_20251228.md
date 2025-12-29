# Fisher CRLB Residual Overlay Visualization

**Experiment ID**: `SCALING-20251228-fisher-residual-overlay`  
**Date**: 2025-12-28  
**Status**: ✅ Completed

---

## 📋 Overview

在 ViT 模型的 parity/residual 图上叠加 Fisher/CRLB 理论下界，直观展示模型性能与理论极限的对比。

---

## 🎯 核心图表

### Figure: Fisher CRLB Residual Overlay (Dual Magnitude)

![Fisher Residual Overlay](img/fisher_residual_overlay_real_dual_mag.png)

---

## 📊 图表说明

### 布局
| 子图 | 内容 |
|------|------|
| **左 (Parity Plot)** | 预测值 vs 真实值 + Fisher CRLB 包络带 |
| **中 (Residual Plot)** | 残差 vs 真实值 + Fisher CRLB 包络线 |
| **右 (Histogram)** | 残差分布 + 理论最小 σ 竖线 |

### 颜色编码
| 颜色 | 含义 |
|------|------|
| **Navy Blue (深蓝)** | Fisher CRLB @ mag=21.5 (SNR≈7) |
| **Steel Blue (浅蓝)** | Fisher CRLB @ mag=22.5 (SNR≈3) |
| **Orange** | ViT 模型预测点 |
| **Red (dashed)** | ViT 经验 σ |

### 关键指标
| 指标 | 值 |
|------|-----|
| **R² (ViT)** | 0.707 |
| **RMSE (ViT)** | 0.636 |
| **σ_min (mag=21.5)** | 0.43 |
| **σ_min (mag=22.5)** | 1.11 |

---

## 🔧 绘制方法

### 脚本路径
```
~/VIT/scripts/scaling_fisher_residual_overlay.py
```

### 运行命令
```bash
cd ~/VIT && source init.sh
python scripts/scaling_fisher_residual_overlay.py --real --no-show
```

### 关键参数
| 参数 | 值 |
|------|-----|
| `figsize` | (16, 5) |
| `xlim` (图1,2) | (1, 5) |
| `ylim` (图1) | (1, 5) |
| `ylim` (图2) | (-3, 3) |
| `xlim` (图3) | (-3, 3) |

### 数据来源
| 数据 | 路径 |
|------|------|
| **Fisher mag=21.5** | `results/SCALING-20251224-fisher-ceiling-02/fisher_results.csv` |
| **Fisher mag=22.5** | `results/fisher_5d_multi_mag/mag22.5/fisher_results.csv` |
| **VIT 模型** | `checkpoints/vit_1m/best_epoch=128-val_mae=0.3720-val_r2=0.7182.ckpt` |
| **测试数据** | `/datascope/.../mag205_225_lowT_1M/test_10k/dataset.h5` |

---

## 📈 核心代码逻辑

### 1. Fisher σ 计算
```python
# 从 CRLB 计算 σ_fisher
sigma_fisher = np.sqrt(crlb_logg_marginalized)

# 按 log_g bin 插值
sigma_per_bin = df.groupby(pd.cut(df['log_g'], bins))['sigma_fisher'].median()
```

### 2. 双重包络绘制
```python
# mag=22.5 (wider, lighter) - 先画
ax.fill_between(x, -sigma_225, sigma_225, color='#D4E6F1', alpha=0.4)

# mag=21.5 (narrower, darker) - 后画
ax.fill_between(x, -sigma_215, sigma_215, color='#B0C4DE', alpha=0.6)
```

### 3. 颜色方案
```python
COLOR_VIT = '#FF8C00'        # Orange
COLOR_FISHER_215 = 'navy'    # Navy blue
COLOR_FISHER_225 = 'steelblue'  # Lighter blue
```

---

## 🎨 图表设计要点

1. **无标题**: 三张子图均不显示标题
2. **图1无图例**: 左图 (parity plot) 不显示 legend
3. **图2图例**: 清晰标注模型名称 `ViT-p16_h256_L6_1.3M` 和 `ViT σ=0.64`
4. **图3图例**: 简短，放左上角避免遮挡直方图
5. **双重 Fisher bound**: mag=21.5 深色 + mag=22.5 浅色

---

## 📁 输出文件

```
~/VIT/results/fisher_residual_overlay/
├── fisher_residual_overlay_real_dual_mag.png
└── fisher_residual_overlay_real_dual_mag.pdf

~/Physics_Informed_AI/logg/vit/exp/img/
├── fisher_residual_overlay_real_dual_mag.png
└── fisher_residual_overlay_real_dual_mag.pdf
```

---

## 🔗 相关实验

- `SCALING-20251224-fisher-ceiling-02`: Fisher CRLB 计算 (mag=21.5)
- `fisher_5d_multi_mag`: 多 magnitude Fisher 计算 (5D 参数空间)
- `VIT-20251227-1m-scaling`: ViT 1M 训练实验

---

## 📝 Insights

1. **VIT 性能 vs 理论极限**: 
   - VIT σ=0.64 介于 mag=21.5 (σ=0.43) 和 mag=22.5 (σ=1.11) 之间
   - 测试数据 mag=20.5-22.5，模型表现符合预期

2. **残差分布**: 
   - 大部分残差落在 mag=21.5 的 Fisher bound 内
   - 少量 outliers 超出 mag=22.5 bound (低 SNR 样本)

3. **改进空间**:
   - 相对 mag=21.5 理论极限，还有约 50% 提升空间
   - 可通过 SNR-aware 训练策略进一步逼近
