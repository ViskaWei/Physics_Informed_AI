# 📇 Knowledge Card: NN Baseline Framework
> **Name:** NN Baseline Framework | **ID:** `VIT-20251224-scaling-nn-baseline-card`  
> **Topic:** `scaling` | **Source:** `exp_scaling_nn_baseline_framework_20251224.md` | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-24
```
💡 MLP 100k R²=0.47 达到 Ridge baseline；CNN 弱于 MLP；Whitening 预处理导致训练崩溃  
适用：NN 架构选择和预处理决策
```

---

## 🎯 问题与设置

**问题**: MLP/CNN 能否突破传统 ML 的 R²=0.57 天花板？

**设置**: 
- 数据: BOSZ 100k/1M train, noise σ=1.0
- 模型: MLP (3L_1024, GELU), CNN (4L_k5_bn)
- 关键变量: 模型架构, 数据量, 输入预处理

---

## 📊 关键结果

| # | 结果 | 数值 | 配置 |
|---|------|------|------|
| 1 | MLP 100k Best R² | **0.4671** | 3L_1024, GELU |
| 2 | CNN 100k Best R² | 0.4122 | 4L_k5_bn |
| 3 | CNN 1M Best R² | 0.4337 | 4L_k5_wide |
| 4 | vs Oracle MoE gap | -0.15~0.19 | 0.62 target |
| 5 | Whitening 效果 | ❌ 完全失效 | R²≈0 |

**Baselines**: Ridge 0.46-0.50, LightGBM 0.50-0.57, Oracle MoE 0.62

---

## 💡 核心洞见

### 🏗️ 宏观层（架构设计）

- **MLP 达到 Ridge baseline**: 验证 NN 框架正常工作
- **单模型 NN 难以达到 Oracle 0.62**: Gap = 0.15-0.19
- **考虑 MoE-CNN**: 如果单模型无法突破

### 🔧 模型层（调参优化）

- **CNN 弱于 MLP**: 可能需要更多调参 (lr, wd, warmup)
- **只有 BatchNorm 版本 CNN 能正常训练**
- **MLP 1M 需要用 flux_only 模式重跑**

### ⚙️ 工程层（实现细节）

- ⚠️ Whitening `x = flux / (error × noise_level)` 导致极端值
- 建议改用 StandardScaler 或 log(1+x) 变换
- MLP 3L_1024: 4.8M params, 3.4min training

---

## ➡️ 下一步

| 优先级 | 任务 | 相关 experiment_id |
|--------|------|-------------------|
| 🔴 P0 | 修复 MLP 1M (用 flux_only 模式) | - |
| 🔴 P0 | 修复 whitening (用 StandardScaler) | - |
| 🟡 P1 | CNN 调参 (更大 lr, warmup) | - |
| 🟡 P1 | 考虑 MoE-CNN 架构 | - |

---

## 🔗 相关链接

| 类型 | 路径 |
|------|------|
| 训练仓库 | `~/VIT/` |
| 脚本 | `~/VIT/scripts/run_scaling_nn_baselines.py` |
| 数据模块 | `~/VIT/src/nn/scaling_data_adapter.py` |
| 完整报告 | `logg/scaling/exp/exp_scaling_nn_baseline_framework_20251224.md` |

