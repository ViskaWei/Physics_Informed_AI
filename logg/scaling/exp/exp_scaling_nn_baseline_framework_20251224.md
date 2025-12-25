# 📘 SCALING-20251224-nn-baseline-framework-01: NN Baseline Framework
> **Name:** TODO | **ID:** `VIT-20251224-scaling-01`  
> **Topic:** `scaling` | **MVP:** MVP-X.X | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-24 | **Status:** 🔄
```
💡 实验目的  
决定：影响的决策
```

---


## 🔗 Upstream Links
| Type | Link |
|------|------|
| 🧠 Hub | `logg/scaling/scaling_hub.md` |
| 🗺️ Roadmap | `logg/scaling/scaling_roadmap.md` |

---

## 📊 实验结果

### Summary Table

| Group | Model | Input | Test R² | MAE | vs Ridge | vs Oracle MoE |
|-------|-------|-------|---------|-----|----------|---------------|
| **MLP_100k** | mlp_3L_1024 | flux_only | **0.4671** | 0.645 | +0.007 ✅ | -0.153 |
| CNN_100k | cnn_4L_k5_bn | flux_only | 0.4122 | 0.704 | -0.048 | -0.208 |
| MLP_1M | mlp_2L_2048 | whitening | -0.0003 | 0.978 | -0.500 ❌ | -0.620 |
| **CNN_1M** | cnn_4L_k5_wide | whitening | **0.4337** | 0.681 | -0.066 | -0.186 |

### Baselines

| Model | R² (100k) | R² (1M) |
|-------|-----------|---------|
| Ridge | 0.46 | 0.50 |
| LightGBM | 0.50 | 0.57 |
| Oracle MoE | - | 0.62 |

---

## 🔍 关键发现

### C1: MLP 达到 Ridge Baseline ✅
- **发现**: MLP 3L_1024 with GELU @ 100k 达到 R²=0.4671
- **意义**: 验证 NN 框架正常工作，可以匹配传统 ML
- **参数**: 4.8M params, 3.4min training time

### C2: CNN 弱于 MLP ❌
- **发现**: Best CNN R²=0.4122，低于 MLP (0.4671)
- **原因分析**:
  1. 只有 BatchNorm 版本能正常训练
  2. 其他 CNN 变种 R²≈0 (训练不收敛)
  3. CNN 需要更多调参 (lr, wd, warmup)
- **结论**: 当前 CNN 架构未体现局部归纳偏置优势

### C3: Whitening 预处理失败 ⚠️
- **发现**: `x = flux / (error × noise_level)` 导致训练崩溃
- **原因**: 极端值 (error 很小时分母接近 0)
- **影响**: 所有 whitening 实验 R²≈0 或负值
- **建议**: 改用 StandardScaler 或 log(1+x) 变换

### C4: CNN 1M 有改善但不显著
- **发现**: CNN @ 1M 达到 R²=0.4337
- **vs 100k**: +0.02 R² (从 0.41 到 0.43)
- **vs Oracle MoE**: 仍有 -0.19 gap

### C5: MLP 1M 实验失败
- **原因**: MLP 使用 whitening 模式导致失败
- **需要**: 使用 flux_only 模式重跑 MLP 1M

---

## 📈 必须记录的 5 个数字

| # | 指标 | 值 |
|---|------|-----|
| 1 | **100k MLP Best R²** | 0.4671 |
| 2 | **100k CNN Best R²** | 0.4122 |
| 3 | **1M CNN Best R²** | 0.4337 |
| 4 | **vs Oracle gap (best)** | -0.15 (MLP 100k) |
| 5 | **whitening 敏感度** | 完全失效 ❌ |

---

## 🚦 止损判断

| 信号 | 状态 | 行动 |
|------|------|------|
| **MLP 止损** | ⬜ 未评估 | MLP 1M 需要用 flux_only 重跑 |
| **CNN 止损** | ⚠️ CNN < MLP | 检查 CNN 超参配置 |
| **vs Oracle MoE** | ❌ Gap = 0.15-0.19 | 单模型 NN 难以达到 0.62 |

---

## 📦 交付物

| 类型 | 路径 |
|------|------|
| 训练脚本 | `~/VIT/scripts/run_scaling_nn_baselines.py` |
| 数据模块 | `~/VIT/src/nn/scaling_data_adapter.py` |
| 结果 CSV | `~/VIT/results/scaling_nn_baselines/scaling_nn_results.csv` |

---

## 🔮 下一步建议

1. **修复 MLP 1M**: 使用 flux_only 模式重跑
2. **修复 whitening**: 改用 StandardScaler 或 clamp 极端值
3. **CNN 调参**: 尝试更大 lr (1e-2), warmup, 不同 kernel sizes
4. **考虑 MoE-CNN**: 如果单模型 CNN 无法突破，尝试 CNN 作为 expert

---

## 🏷️ 元数据

```yaml
experiment_id: SCALING-20251224-nn-baseline-framework-01
project: VIT
topic: nn
status: completed
metrics_summary: "MLP_100k R²=0.47, CNN_1M R²=0.43, vs Oracle gap=-0.15~0.19"
```
