# 📘 Experiment Report: NN Baseline Framework (MLP + CNN)

---
> **Name:** NN Baseline Framework for logg Prediction @ noise=1  
> **ID:**  `SCALING-20251224-nn-baseline-framework-01`  
> **Topic ｜ MVP:** `VIT` | `scaling` ｜ MVP-NN-0, MVP-MLP-1, MVP-CNN-1  
> **Author:** Viska Wei  
> **Date:** 2025-12-24  
> **Project:** `VIT`  
> **Status:** 🔄 In Progress
---

## 🔗 Upstream Links

| Type | Link | Description |
|------|------|-------------|
| 🧠 Hub | [`scaling_hub_20251222.md`](../scaling_hub_20251222.md) | H-NN0~3 假设 |
| 🗺️ Roadmap | [`scaling_roadmap_20251222.md`](../scaling_roadmap_20251222.md) | MVP-NN-0~MoE-CNN-0 设计 |
| 📋 Kanban | [`kanban.md`](../../../status/kanban.md) | Experiment queue |
| 📚 Prerequisite | [exp_scaling_ml_ceiling](./exp_scaling_ml_ceiling_20251222.md) | MVP-1.0~1.2 ML baseline |

---
# 📑 Table of Contents

- [⚡ Key Findings](#-key-findings-for-hub-extraction)
- [1. 🎯 Objective](#1--objective)
- [2. 🧪 Experiment Design](#2--experiment-design)
- [3. 📊 Figures & Results](#3--figures--results)
- [4. 💡 Insights](#4--insights)
- [5. 📝 Conclusions](#5--conclusions)
- [6. 📎 Appendix](#6--appendix)

---


## ⚡ 核心结论速览（供 main 提取）

> **⏳ 待实验完成后填写**

### 一句话总结

> **TODO**

### 对假设的验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H-NN0.1: CNN whiten 100k ≥ Ridge 100k R² | ⏳ | - |
| H-MLP1.1: 100k→1M 提升 < +0.02 R² → MLP 不对 | ⏳ | - |
| H-CNN1.1: CNN 100k ≥ MLP + 0.05 R² | ⏳ | - |
| H-CNN1.2: CNN 1M ≥ 0.60 | ⏳ | - |

### 设计启示（1-2 条）

| 启示 | 具体建议 |
|------|---------|
| TODO | TODO |

### 关键数字

| 指标 | 值 |
|------|-----|
| MLP 100k R² | ⏳ |
| MLP 1M R² | ⏳ |
| CNN 100k R² | ⏳ |
| CNN 1M R² | ⏳ |
| ΔR² (100k→1M) MLP | ⏳ |
| ΔR² (100k→1M) CNN | ⏳ |
| vs Oracle MoE (0.62) gap | ⏳ |

---

# 1. 🎯 目标

## 1.1 实验目的

> 在 noise=1 条件下，用最小成本快速判断：
> 1. **单模型 NN 能不能接近/超过 Oracle MoE 的 0.62？**
> 2. 如果不能：是 **结构不对** 还是 **输入/归一化/目标设置不对**？

**核心问题**：NN 能否打破 ML ceiling (Ridge≈0.46, LGB≈0.57)，接近 Oracle MoE (0.62)？

**回答的问题**：
- Q1: MLP 全局架构是否注定不行？（止损信号：100k→1M < +0.02 R²）
- Q2: CNN 局部归纳偏置是否能带来质变？（CNN vs MLP ≥ +0.05 R²？）
- Q3: 输入 whitening 对 NN 训练的敏感度？
- Q4: 大数据 (1M) 对 NN 的收益如何？

**对应假设**：
- H-NN0.1: CNN whiten + 100k 能达到 Ridge 水平
- H-MLP1.1: MLP 在 100k→1M 提升 < +0.02 → 架构不对
- H-CNN1.1: CNN 100k 能明显超过 MLP (+0.05 R²)
- H-CNN1.2: CNN 1M 能接近 0.60

## 1.2 预期结果

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| ✅ 正常情况 | CNN 1M R² ≥ 0.58 | 接近 Oracle MoE (0.62) |
| ⚠️ 警告情况 | CNN 1M R² = 0.50~0.57 | 比 LGB 略好，需要多尺度 CNN |
| ❌ 异常情况 A | CNN 100k < Ridge | 80% 概率是输入/训练 bug |
| ❌ 异常情况 B | MLP 100k→1M > +0.05 R² | 意外发现，值得深入研究 |

---

# 2. 🧪 实验设计

## 2.0 总体原则（避免"结构不对，堆数据没用"）

> 🔴 **必须先锁死 3 个容易踩坑的点**：

### 2.0.1 输入 Whitening / 误差建模（noise=1 特别重要）

| 方案 | 公式 | 推荐 |
|------|------|------|
| **方案 A (推荐)** | `x = flux / (error * noise_level)` | ⭐ |
| 方案 B | 两通道 `[flux, error]` | 备选 |
| 方案 C | 两通道 `[flux, 1/error]` | 备选 |

> 否则网络会学到"噪声形态"而不是"谱线信息"。

### 2.0.2 输出目标的尺度

```python
# 标准化目标，训练更稳定更快收敛
y = (logg - mean) / std
```

### 2.0.3 评估要稳定

| 配置项 | 值 |
|--------|-----|
| Test set size | ≥ 20k |
| Random seed | 固定（42 或其他） |
| Stratification | 按 Teff/logg/[M/H] 分桶分层 |

## 2.1 数据

### 数据来源与规模

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **数据来源** | BOSZ 50000 合成光谱 (mag205_225_lowT_1M) | 与 Oracle MoE 实验一致 |
| **数据根目录** | `/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/` | |
| **训练样本数** | **1,000,000** (5 shards × 200k) | 全量训练 |
| **测试样本数** | **1,000** (test_1k_0) | 使用预生成的 noisy |
| **特征维度** | **4096** (MR arm) | 波长点数 ✅ |
| **波长范围** | MR arm (中分辨率) | |
| **星等范围** | mag 20.5-22.5 | |
| **温度范围** | Low T (3750-6000 K) | |
| **标签参数** | log_g (1.00 ~ 5.00 dex) | 主要目标 |
| **辅助参数** | Teff, [M/H] | 用于 stratification |

### 数据文件结构

```
/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/
├── train_200k_0/dataset.h5   # 200k samples, 19 GB
├── train_200k_1/dataset.h5   # 200k samples, 19 GB
├── train_200k_2/dataset.h5   # 200k samples, 19 GB
├── train_200k_3/dataset.h5   # 200k samples, 19 GB
├── train_200k_4/dataset.h5   # 200k samples, 19 GB
└── test_1k_0/dataset.h5      # 1k samples (预生成 noisy)
```

### HDF5 数据结构

```
dataset.h5
├── dataset/
│   ├── arrays/
│   │   ├── flux/value      # (N, 4096) - 原始光谱通量
│   │   ├── error/value     # (N, 4096) - 光谱误差
│   │   ├── noisy/value     # (N, 4096) - 预加噪光谱 (test 使用)
│   │   └── mask/value      # (N, 4096) - 掩码
│   └── params/table        # (N,) - 参数表
└── spectrumdataset/
    ├── wave                # (4096,) - 波长
    └── wave_edges          # (4097,) - 波长边界
```

### 噪声配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **噪声类型** | Heteroscedastic Gaussian | 异方差高斯噪声 |
| **噪声水平 noise_level** | **1.0** | 本次实验聚焦 |
| **训练噪声** | On-the-fly | 每次采样重新加噪 |
| **测试噪声** | **Pre-generated** | 使用 `noisy/value` 字段 |

**噪声添加公式**：

```python
# Heteroscedastic Gaussian noise
noise = noise_level * error * np.random.randn(*flux.shape)
noisy = flux + noise
noisy = np.clip(noisy, 0, None)  # Clip negative values
```

### 数据预处理

| 步骤 | 配置 |
|------|------|
| **输入归一化** | Whitening: `flux / (error * noise_level)` |
| **目标归一化** | StandardScaler: `(logg - mean) / std` |
| **Stratification** | 按 Teff/logg/[M/H] 分桶后分层抽样 |

## 2.2 模型与算法

### 2.2.1 MVP-NN-0: 可靠基线框架

> **目的**：建立 NN 训练管线 + 保证输入/评估没问题

| 配置项 | 值 |
|--------|-----|
| **训练规模** | 先 100k 做 smoke test |
| **通过条件** | 能复现 Ridge/LGBM 大致水平，train/val 曲线正常 |

### 2.2.2 MVP-MLP-1: 最小可行 MLP

**架构**：

```
Input (4096)
  ↓
Linear(4096→2048) → LayerNorm → GELU → Dropout(0.1)
  ↓
Linear(2048→1024) → GELU → Dropout(0.1)
  ↓
Linear(1024→512) → GELU → Dropout(0.1)
  ↓
Linear(512→1)
  ↓
Output (1)
```

| 超参数 | 值 |
|--------|-----|
| weight_decay | 1e-4 |
| dropout | 0.1 |
| LayerNorm | 第一层后 |

**🚨 止损信号**：
- 如果 **100k→1M 提升 < +0.02 R²** 且 val 曲线 plateau 很早：
  → 结论：**MLP 架构归纳偏置不对**，不要再在 MLP 上花时间

### 2.2.3 MVP-CNN-1: 最小 1D CNN

**架构**：

```
Input (1, 4096)  # (C=1, L=4096)
  ↓
[Stem] Conv1d(1→32, k=7, stride=1) → GELU
  ↓
[Block 1] Conv1d(32→64, k=5, dilation=1) → GELU → LayerNorm
          Conv1d(64→64, k=5, dilation=2) → GELU + Residual
  ↓
[Block 2] Conv1d(64→64, k=5, dilation=1) → GELU → LayerNorm
          Conv1d(64→64, k=5, dilation=2) → GELU + Residual
  ↓
[Block 3] Conv1d(64→64, k=5, dilation=1) → GELU → LayerNorm
          Conv1d(64→64, k=5, dilation=2) → GELU + Residual
  ↓
[Block 4] Conv1d(64→64, k=5, dilation=1) → GELU → LayerNorm
          Conv1d(64→64, k=5, dilation=2) → GELU + Residual
  ↓
[Pool] Global Average Pooling → (64,)
  ↓
[Head] Linear(64→128) → GELU → Linear(128→1)
  ↓
Output (1)
```

| 超参数 | 值 |
|--------|-----|
| Normalization | LayerNorm 或 GroupNorm |
| Residual | 简单加法 |
| weight_decay | 1e-4 |

**止损信号**：
- 如果 CNN 100k < Ridge/LGBM：80% 概率是 **输入/whitening/训练细节有 bug**

### 2.2.4 MVP-CNN-2: 多尺度 / 大感受野（可选）

> 仅当 MVP-CNN-1 效果不够好时启动

**增强方式 1**：dilation schedule `[1, 2, 4, 8]`

**增强方式 2**：多分支卷积核 k = `[3, 7, 15]` 并行分支后 concat（类似 Inception1D）

## 2.3 超参数配置

### 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **epochs** | 10~20 (100k), 5~10 (1M) | early stop 控制 |
| **batch_size** | 256 或 512 | 根据 GPU 内存调整 |
| **learning_rate** | 1e-3 → 1e-4 | cosine schedule |
| **optimizer** | AdamW | |
| **weight_decay** | 1e-4 | L2 正则 |
| **scheduler** | CosineAnnealingLR | 或 StepLR |
| **grad_clip** | 1.0 | 可选 |
| **early_stopping** | patience=3~5 epochs | val R² 不涨就停 |
| **random_seed** | 42 | 固定 |

### 扫描参数

| 实验 | 扫描参数 | 固定参数 |
|------|---------|---------|
| MVP-NN-0 | 无（验证框架） | 100k, whitening |
| MVP-MLP-1 | 数据规模: 100k → 1M | MLP 架构固定 |
| MVP-CNN-1 | 数据规模: 100k → 1M | CNN 架构固定 |
| MVP-CNN-2 | dilation/kernel | 1M |

## 2.4 评价指标

| 指标 | 公式 | 用途 |
|------|------|------|
| $R^2$ | $1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$ | **主要评价指标** |
| MAE | $\frac{1}{n}\sum\|y - \hat{y}\|$ | 参考 |
| RMSE | $\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}$ | 参考 |
| ΔR² (100k→1M) | R²_1M - R²_100k | 判断数据规模收益 |
| plateau epoch | 多少 epoch 到 plateau | 判断收敛效率 |

---

# 3. 📊 实验图表

> ⏳ 待实验完成后填写

### 图 1：MLP vs CNN Learning Curves

![TODO](./img/nn_baseline_learning_curves.png)

**Figure 1. MLP 和 CNN 在 100k/1M 上的训练曲线对比**

**关键观察**：
- TODO

---

### 图 2：Data Scaling Effect

![TODO](./img/nn_baseline_scaling.png)

**Figure 2. 100k → 1M 数据规模对 MLP/CNN 的影响**

**关键观察**：
- TODO

---

### 图 3：Model Comparison (同 test set)

![TODO](./img/nn_baseline_comparison.png)

**Figure 3. Ridge / LGB / MLP / CNN / Oracle MoE 在同一 test set 上的对比**

**关键观察**：
- TODO

---

# 4. 💡 关键洞见

> ⏳ 待实验完成后填写

## 4.1 宏观层洞见

TODO

## 4.2 模型层洞见

TODO

## 4.3 实验层细节洞见

TODO

---

# 5. 📝 结论

> ⏳ 待实验完成后填写

## 5.1 核心发现

> TODO

## 5.2 关键结论（2-4 条）

| # | 结论 | 证据 |
|---|------|------|
| 1 | TODO | TODO |
| 2 | TODO | TODO |

## 5.3 设计启示

TODO

## 5.4 物理解释

TODO

## 5.5 关键数字速查

| 指标 | 值 | 配置/条件 |
|------|-----|----------|
| MLP 100k R² | ⏳ | |
| MLP 1M R² | ⏳ | |
| CNN 100k R² | ⏳ | |
| CNN 1M R² | ⏳ | |
| Best NN vs Oracle MoE | ⏳ | |

## 5.6 下一步工作

| 方向 | 具体任务 | 优先级 | 对应 MVP |
|------|----------|--------|---------|
| 多尺度 CNN | 如果 CNN 1M < 0.60 | 🟡 P1 | MVP-CNN-2 |
| MoE-CNN | 如果 global CNN < 0.60 明显 | 🟢 P2 | MVP-MoE-CNN-0 |

---

# 6. 📎 附录

## 6.1 数值结果表

> ⏳ 待实验完成后填写

### 主要结果

| Model | Data Size | Test Size | R² | MAE | RMSE | 备注 |
|-------|-----------|-----------|-----|-----|------|------|
| Ridge | 1M | 1k | **0.4611** | 0.177 | 0.221 | ML baseline |
| LightGBM | 1M | 1k | **0.5749** | 0.154 | 0.196 | ML baseline |
| **Oracle MoE** | 1M | 1k | **0.6249** | 0.138 | 0.177 | 结构上限 (+0.16 vs Ridge) |
| MLP | 1M | 1k | ⏳ | | | |
| CNN | 1M | 1k | ⏳ | | | |
| Multi-scale CNN | 1M | 1k | ⏳ | | | |

### 100k → 1M Scaling

| Model | R²_100k | R²_1M | ΔR² | 判断 |
|-------|---------|-------|-----|------|
| MLP | ⏳ | ⏳ | ⏳ | |
| CNN | ⏳ | ⏳ | ⏳ | |

---

## 6.2 实验流程记录

### 6.2.1 环境与配置

| 项目 | 值 |
|------|-----|
| **仓库** | `~/VIT` |
| **Config 路径** | TODO |
| **输出路径** | `lightning_logs/version_X` |
| **Python** | 3.10+ |
| **关键依赖** | PyTorch 2.x, Lightning 2.x |

### 6.2.2 ✅ 输入格式已确认

> 参考: `exp_scaling_oracle_moe_noise1_20251223.md` + `/home/swei20/VIT/scripts/run_nn_baselines.py`

| 确认项 | 值 | 来源 |
|--------|-----|------|
| **波长点数** | **4096** (MR arm) | Oracle MoE 实验 |
| **训练输入** | `flux` + on-the-fly noise | 每次采样重新加噪 |
| **测试输入** | `noisy/value` (预生成) | test_1k_0 已有 noisy |
| **数据文件格式** | `.h5` (HDF5) | dataset.h5 |
| **1M 数据路径** | `/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/` | ✅ |
| **训练 shards** | `train_200k_{0-4}/dataset.h5` (5 × 200k = 1M) | ✅ |
| **测试文件** | `test_1k_0/dataset.h5` | 使用预生成 noisy |
| **Dataset class** | `RegSpecDataset` → `NNDataset` | ✅ 已有 |

### 6.2.3 执行命令

> 参考: `/home/swei20/VIT/scripts/run_nn_baselines.py`

```bash
cd ~/VIT

# ============================================================
# Step 0: Dry Run - 查看所有将运行的实验
# ============================================================
python scripts/run_nn_baselines.py --dry-run -e Step1

# ============================================================
# Step 1: MLP lr/wd 搜索 (32k, noise=1.0)
# 使用 Step1 实验组：固定架构 [256,64]，搜索 lr 和 weight_decay
# ============================================================
python scripts/run_nn_baselines.py -e Step1 --parallel --gpus 0,1,2,3

# ============================================================
# Step 2: MLP 架构搜索 (使用 Step1 最优 lr/wd)
# 搜索 depth, width, activation, init
# ============================================================
python scripts/run_nn_baselines.py -e Step2 --parallel --gpus 0,1,2,3

# ============================================================
# Step 3: MLP Deep/Wide 实验 (更大网络)
# ============================================================
python scripts/run_nn_baselines.py -e MLP_Deep --parallel --gpus 0,1,2,3
python scripts/run_nn_baselines.py -e MLP_Wide --parallel --gpus 0,1,2,3

# ============================================================
# Step 4: CNN 实验
# CNN_Stage1a: lr 搜索
# CNN_Stage1b: wd 搜索
# CNN_Stage2: 架构搜索
# ============================================================
python scripts/run_nn_baselines.py -e CNN_Stage1a --parallel --gpus 0,1,2,3
python scripts/run_nn_baselines.py -e CNN_Stage1b --parallel --gpus 0,1,2,3
python scripts/run_nn_baselines.py -e CNN_Stage2 --parallel --gpus 0,1,2,3

# ============================================================
# 100k/1M 规模实验 (使用最优配置)
# 需要设置 DATA_ROOT 指向 100k/1M 数据
# ============================================================
DATA_ROOT=/path/to/100k/data python scripts/run_nn_baselines.py -e MLP_Big --parallel
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-train` | 32000 | 训练样本数 |
| `--batch-size` | 2048 | 批大小（适配 V100） |
| `--epochs` | 100 | 最大训练轮数 |
| `--patience` | 50 | Early stopping patience |
| `--parallel` | True | 多 GPU 并行 |
| `--gpus` | 0,1,2,3,4,5,6,7 | 使用的 GPU |

### 数据路径

| 规模 | 路径 |
|------|------|
| 32k | `/srv/local/tmp/swei20/data/bosz50000/z0/train_32k/` |
| **1M (本次使用)** | `/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/` |
| 训练 shards | `train_200k_{0,1,2,3,4}/dataset.h5` (共 5 个) |
| 测试 | `test_1k_0/dataset.h5` (使用预生成 noisy) |

```python
# Python 数据加载示例
DATA_ROOT = "/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M"
TRAIN_SHARDS = [f"{DATA_ROOT}/train_200k_{i}/dataset.h5" for i in range(5)]
TEST_FILE = f"{DATA_ROOT}/test_1k_0/dataset.h5"
```

---

## 6.3 相关文件

| 类型 | 路径 | 说明 |
|------|------|------|
| Hub | `logg/scaling/scaling_hub_20251222.md` | 假设金字塔 |
| Roadmap | `logg/scaling/scaling_roadmap_20251222.md` | MVP 设计 |
| ML Baseline | `logg/scaling/exp/exp_scaling_ml_ceiling_20251222.md` | 前置实验 |
| 本报告 | `logg/scaling/exp/exp_scaling_nn_baseline_framework_20251224.md` | 当前文件 |
| 图表 | `logg/scaling/exp/img/` | 实验图表 |

---

## 6.4 必须记录的 5 个数字

| # | 指标 | 值 | 说明 |
|---|------|-----|------|
| 1 | **100k → 1M 的 ΔR²** | ⏳ | 每个模型一个 |
| 2 | **plateau epoch** | ⏳ | 训练效率 |
| 3 | **per-bin R²** | ⏳ | 特别是最差的 bin |
| 4 | **whitening 敏感度** | ⏳ | 有无 whitening 的差距 |
| 5 | **vs Oracle gap** | ⏳ | global CNN vs Oracle MoE |

---

## 6.5 推荐执行顺序

| 顺序 | MVP | 目的 | 时间预估 |
|------|-----|------|---------|
| 1 | MVP-NN-0 | 框架搭建 | 半天 |
| 2 | MVP-MLP-1 @100k + @1M | 快速止损/确认"MLP 不吃数据" | 1天 |
| 3 | MVP-CNN-1 @100k | 确认归纳偏置对不对 | 半天 |
| 4 | MVP-CNN-1 @1M | 看"大力出奇迹"是否成立 | 1天 |
| 5 | MVP-CNN-2 | 多尺度 CNN（如需） | 1天 |
| 6 | MVP-MoE-CNN-0 | 仅当 global CNN < 0.60 | 视情况 |

---

> **模板说明**：
> - 本文档为 NN baseline 实验框架，§1-2 已填写完整
> - §3-6 待实验完成后填写
> - 请在开始实验前确认 §6.2.2 中的输入格式问题

