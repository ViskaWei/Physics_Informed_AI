# 📇 Knowledge Card: Traditional ML Ceiling
> **Name:** ML Ceiling @ noise=1 | **ID:** `VIT-20251222-scaling-ml-ceiling-card`  
> **Topic:** `scaling` | **Source:** `exp_scaling_ml_ceiling_20251222.md` | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-22
```
💡 Ridge R²=0.46, LightGBM R²=0.57 @ 1M data, noise=1 — 传统 ML 存在明确性能天花板  
适用：高噪声场景下的 baseline 设定
```

---

## 🎯 问题与设置

**问题**: 传统 ML (Ridge, LightGBM) 在大数据+高噪声下能达到什么性能极限？

**设置**: 
- 数据: BOSZ 光谱, 1M train, 500 test, noise σ=1.0
- 模型: Ridge (α=5000), LightGBM (lr=0.05, trees=5000)
- 关键变量: 数据量 10k→1M

---

## 📊 关键结果

| # | 结果 | 数值 | 配置 |
|---|------|------|------|
| 1 | Ridge 最佳 R² | **0.46** | 1M, α=5000 |
| 2 | LightGBM 最佳 R² | **0.5709** | 1M, trees=1293 |
| 3 | Ridge ΔR² (1M vs 100k) | +0.024 | 边际增益小 |
| 4 | LightGBM ΔR² (1M vs 100k) | +0.018 | 边际增益小 |
| 5 | LightGBM vs Ridge 优势 | +7% | 非线性价值有限 |

---

## 💡 核心洞见

### 🏗️ 宏观层（架构设计）

- **传统 ML 天花板 ≈ 0.57**: 需要深度学习才能突破
- **数据量非瓶颈**: 100k 后增益 <3%，投资应转向模型改进

### 🔧 模型层（调参优化）

- **LightGBM 早停有效**: 1M 时仅用 26% 的最大树数
- **最优 Ridge α=5000**: 高噪声需要强正则化

### ⚙️ 工程层（实现细节）

- 极值区域 (log_g < 2 或 > 4) 预测偏差更大
- Ridge 预测更集中于均值附近（过度平滑）

---

## ➡️ 下一步

| 优先级 | 任务 | 相关 experiment_id |
|--------|------|-------------------|
| 🔴 P0 | 用 NN (MLP/CNN) 验证是否能突破 0.70 | MVP-2.0, 2.1 |
| 🟡 P1 | 测试不同 noise level 下的 scaling 规律 | MVP-1.4 |

---

## 🔗 相关链接

| 类型 | 路径 |
|------|------|
| 训练仓库 | `~/VIT/` |
| 结果目录 | `~/VIT/results/scaling_ml_ceiling/` |
| 完整报告 | `logg/scaling/exp/exp_scaling_ml_ceiling_20251222.md` |

