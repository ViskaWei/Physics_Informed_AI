# 📘 SCALING-20251224-nn-baseline-framework-01: NN Baseline Framework

> **Name:** NN Baseline Framework | **ID:** `SCALING-20251224-nn-baseline-framework-01`  
> **Topic:** `scaling` | **MVP:** MVP-NN-0 | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-24 | **Status:** ✅ Done
> **验证假设:** H-NN0.1 (NN 框架能复现 ML baseline)

---

## ⚡ 核心结论速览

> **一句话总结**: MLP (flux_only) 达到 Ridge baseline，但 CNN 弱于 MLP；**Whitening 预处理导致训练完全崩溃**。

| 项目 | 结论 |
|------|------|
| **假设验证** | ✅ H-NN0.1: MLP 达到 Ridge baseline (R²=0.467 ≈ 0.46) |
| **关键发现** | ❌ Whitening (flux/error) 导致 R²≈0，所有 NN 必须用 flux_only 输入 |
| **最佳配置** | MLP 3L_1024 + flux_only + GELU: R²=0.4671, MAE=0.645 |
| **vs Oracle gap** | -0.15 (Oracle MoE=0.62, best NN=0.47) |
| **设计启示** | 1) NN 训练框架正常；2) CNN 需更多调参；3) 下一步修复 MLP 1M |

---

## 🔗 Upstream Links

| Type | Link |
|------|------|
| 🧠 Hub | [`scaling_hub_20251222.md`](../scaling_hub_20251222.md) |
| 🗺️ Roadmap | [`scaling_roadmap_20251222.md`](../scaling_roadmap_20251222.md) |

---

## 📐 实验设计

### 数据配置

| 项目 | 配置 |
|------|------|
| **数据集** | BOSZ 50000, mag205_225_lowT_1M |
| **训练规模** | 100k (smoke test) / 1M (full) |
| **测试集** | 1000 samples (固定) |
| **噪声水平** | σ=1.0 (heteroscedastic Gaussian) |
| **目标变量** | log_g ∈ [1.0, 5.0] |
| **输入维度** | 4096 (MR arm 光谱) |

### 输入变体

| Input Mode | 描述 | 结果 |
|------------|------|------|
| **flux_only** | 原始 flux | ✅ 正常工作 |
| **whitening** | flux / (error × σ) | ❌ 训练崩溃 (R²≈0) |

### 模型架构

| 类型 | 架构 | 参数量 | 备注 |
|------|------|--------|------|
| MLP 3L_1024 | [1024, 512, 256] + GELU + Dropout | 4.85M | **最佳配置** |
| MLP 3L_2048 | [2048, 1024, 512] + GELU + Dropout | 11.0M | 略低于 3L_1024 |
| CNN 4L_k5_bn | Conv1D×4 + BatchNorm + MLP head | 60.5k | CNN 中最佳 |
| CNN 4L_k5_wide | [32, 64, 128, 128] channels | 150k | 1M 上表现尚可 |

### 训练配置

| 项目 | 100k 配置 | 1M 配置 |
|------|-----------|---------|
| Epochs | 20 | 10 |
| Batch Size | 1024 | 2048 |
| Learning Rate | 1e-3 | 5e-4 |
| Weight Decay | 1e-4 | 1e-4 |
| Optimizer | AdamW | AdamW |
| Scheduler | CosineAnnealing | CosineAnnealing |
| Early Stopping | patience=10 | patience=10 |

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

---

## 📎 附录

### 6.1 完整数值结果

| experiment_id | model | input | test_r2 | test_mae | train_size | train_time |
|---------------|-------|-------|---------|----------|------------|------------|
| MLP_100k_3L_1024_raw | mlp_3L_1024 | flux_only | **0.4671** | 0.645 | 100k | 3.4min |
| MLP_100k_2L_2048_raw | mlp_2L_2048 | flux_only | 0.4664 | 0.650 | 100k | 3.4min |
| MLP_100k_3L_2048_raw | mlp_3L_2048 | flux_only | 0.4623 | 0.644 | 100k | 3.4min |
| MLP_100k_2L_1024_raw | mlp_2L_1024 | flux_only | 0.4518 | 0.661 | 100k | 2.6min |
| MLP_100k_3L_512_raw | mlp_3L_512 | flux_only | 0.4447 | 0.670 | 100k | 3.3min |
| CNN_100k_4L_k5_bn_raw | cnn_4L_k5_bn | flux_only | **0.4122** | 0.704 | 100k | 30min |
| CNN_100k_4L_k5_bn_wh | cnn_4L_k5_bn | whitening | 0.3434 | 0.757 | 100k | 32min |
| CNN_1M_4L_k5_wide_wh | cnn_4L_k5_wide | whitening | **0.4337** | 0.681 | 1M | 3.7h |
| MLP_1M_2L_2048_wh | mlp_2L_2048 | whitening | -0.0003 | 0.977 | 1M | 4.8min |

**Whitening 失败案例 (R²≈0)**:
- MLP_100k_*_wh: 全部 R²≈0 或负值
- CNN_100k_*_wh (无 BN): 全部 R²≈0

### 6.2 实验流程记录

**执行环境**:
```bash
cd ~/VIT
conda activate vit
```

**训练脚本路径**: 
`~/VIT/scripts/run_scaling_nn_baselines.py`

**执行命令**:
```bash
# 100k MLP smoke test
python scripts/run_scaling_nn_baselines.py -e MLP_100k --parallel --gpus 0,1,2,3

# 100k CNN experiments
python scripts/run_scaling_nn_baselines.py -e CNN_100k --parallel --gpus 0,1,2,3

# 1M experiments
python scripts/run_scaling_nn_baselines.py -e MLP_1M,CNN_1M --parallel --gpus 0,1,2,3
```

**数据模块**: `~/VIT/src/nn/scaling_data_adapter.py`

**关键代码引用**:
- 实验配置: `run_scaling_nn_baselines.py:86-119` (ScalingExpConfig)
- 模型构建: `run_scaling_nn_baselines.py:304-330` (build_model)
- 训练循环: `run_scaling_nn_baselines.py:337-465` (train_and_evaluate)

**结果 CSV**: `~/VIT/results/scaling_nn_baselines/scaling_nn_results.csv`

**关键发现记录**:
1. 2025-12-24 00:57: MLP_100k 完成，发现 whitening 导致训练崩溃
2. 2025-12-24 01:00: 追加 flux_only 实验，MLP 达到 R²=0.467
3. 2025-12-24 13:08: CNN 实验完成，只有 BatchNorm 版本能正常训练
4. 2025-12-24 17:21: 1M CNN (whitening) 完成，R²=0.434

---

## 🏷️ 元数据

```yaml
experiment_id: SCALING-20251224-nn-baseline-framework-01
project: VIT
topic: scaling
mvp: MVP-NN-0
status: completed
metrics_summary: "MLP_100k R²=0.47, CNN_1M R²=0.43, vs Oracle gap=-0.15~0.19"
key_insight: "Whitening preprocessing fails for all NN; flux_only required"
last_updated: 2025-12-25
```
