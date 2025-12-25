# 🗺️ Experiment Roadmap: Data Scaling

> **Topic:** Data Scaling & Model Capacity  
> **Author:** Viska Wei  
> **Created:** 2025-12-22 | **Updated:** 2025-12-22  
> **Current Phase:** Phase 1

<!-- 
📝 Language Convention:
- Headers & section titles: English (keep as-is)
- Content (objectives, conclusions, notes): Chinese OK
- Table column headers: English (keep as-is)
- Table cell content: Chinese OK
-->

## 🔗 Related Files

| Type | File | Description |
|------|------|-------------|
| 🧠 Hub | [`scaling_hub_20251222.md`](./scaling_hub_20251222.md) | Knowledge navigation |
| 📋 Kanban | [`kanban.md`](../../status/kanban.md) | Global task board |
| 📗 Experiments | `exp/*.md` | Detailed reports |

## 📑 Contents

- [1. 🎯 Phase Overview](#1--phase-overview)
- [2. 📋 MVP List](#2--mvp-list)
- [3. 🔧 MVP Specifications](#3--mvp-specifications)
- [4. 📊 Progress Tracking](#4--progress-tracking)
- [5. 🔗 Cross-Repo Integration](#5--cross-repo-integration)
- [6. 📎 Appendix](#6--appendix)

---

# 1. 🎯 Phase Overview

> **Experiments organized by phase, each with clear objectives**

## 1.1 Phase List

| Phase | Objective | MVPs | Status | Key Output |
|-------|-----------|------|--------|------------|
| **Phase 1: ML Ceiling** | 验证传统 ML 在 1M 数据 + noise=1 下的性能瓶颈 | MVP-1.0~1.2 | ✅ | Ridge=0.50, LGB=0.57 |
| **Phase 1.x: ML Refinement** | 确认结果可信度 + 探索调优上限 | MVP-1.3~1.9 | 🔄 | 最终 ML 上限 |
| **Phase 2: NN Advantage** | 验证神经网络能突破 ML 瓶颈 | MVP-2.0~2.2 | ⏳ | NN 性能下限 |
| **Phase 3: Analysis** | 分析瓶颈本质和 scaling 规律 | MVP-3.0~3.2 | ⏳ | 设计原则 |
| **🔴 Phase 16: Ceiling 三层论证** | 理论上限 → 模型 ceiling → 结构上限 | MVP-16T/B/L/O/W/CNN | 🆕 | 可写入论文的证据链 |

## 1.2 Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                   MVP Experiment Dependencies               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   [Phase 1: ML Ceiling]                                     │
│   MVP-1.0 Ridge ──┬── MVP-1.2 Scaling Law                  │
│   MVP-1.1 LightGBM┘                                        │
│         │                                                   │
│         ▼                                                   │
│   [Phase 2: NN Advantage]                                   │
│   MVP-2.0 MLP ───┬── MVP-2.2 NN Scaling                    │
│   MVP-2.1 CNN ───┘                                         │
│         │                                                   │
│         ▼                                                   │
│   [Phase 3: Analysis]                                       │
│   MVP-3.0 Noise Info ── MVP-3.1 Capacity ── MVP-3.2 Feature│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 1.3 Decision Points

> **Key decision points based on experiment results**

| Point | Trigger | Option A | Option B |
|-------|---------|----------|----------|
| D1 | After Phase 1 | If ML R² < 0.6 → 确认瓶颈存在 | If ML R² ≥ 0.7 → 重新评估假设 |
| D2 | After Phase 2 | If NN R² > ML + 0.1 → 证明 NN 优势 | If ΔR² < 0.05 → 瓶颈可能是物理限制 |

---

# 2. 📋 MVP List

> **Overview of all MVPs for quick lookup and tracking**

## 2.1 Experiment Summary

| MVP                             | Name                             | Phase | Status | experiment_id                           | Report                                                     |
| ------------------------------- | -------------------------------- | ----- | ------ | --------------------------------------- | ---------------------------------------------------------- |
| MVP-1.0                         | Ridge 1M Ceiling                 | 1     | ✅      | `SCALING-20251222-ml-ceiling-01`        | [Link](./exp/exp_scaling_ml_ceiling_20251222.md)           |
| MVP-1.1                         | LightGBM 1M Ceiling              | 1     | ✅      | `SCALING-20251222-ml-ceiling-01`        | [Link](./exp/exp_scaling_ml_ceiling_20251222.md)           |
| MVP-1.2                         | ML Scaling Law                   | 1     | ✅      | `SCALING-20251222-ml-ceiling-01`        | [Link](./exp/exp_scaling_ml_ceiling_20251222.md)           |
| **MVP-1.3**                     | **Stats Validation (P0)**        | 1.x   | 🔴     | `SCALING-20251222-stats-01`             | [Link](./exp/exp_scaling_stats_validation_20251222.md)     |
| **MVP-1.4**                     | **Ridge α Extended (P0)**        | 1.x   | ✅      | `SCALING-20251222-ridge-alpha-01`       | [Link](./exp/exp_scaling_ridge_alpha_extended_20251222.md) |
| **MVP-1.5**                     | **LightGBM Param Extended (P0)** | 1.x   | ⏳      | `SCALING-20251222-lgbm-param-01`        | [Link](./exp/exp_scaling_lgbm_param_extended_20251222.md)  |
| **MVP-1.6**                     | **Whitening/SNR Input (P1)**     | 1.x   | ✅      | `SCALING-20251222-whitening-01`         | [Link](./exp/exp_scaling_whitening_snr_20251222.md)        |
| **MVP-1.7**                     | **PCA vs PLS 降维策略 (P1)**         | 1.x   | 🔴     | `SCALING-20251223-pca-pls-01`           | [Link](./exp/exp_scaling_pca_pls_comparison_20251223.md)   |
| MVP-1.8                         | MoE 分段建模 (P2)                    | 1.x   | ⏳      | -                                       | -                                                          |
| MVP-1.9                         | 物理特征工程 (P2)                      | 1.x   | ⏳      | -                                       | -                                                          |
| MVP-2.0                         | MLP 1M Performance               | 2     | ⏳      | -                                       | -                                                          |
| MVP-2.1                         | CNN 1M Performance               | 2     | ⏳      | -                                       | -                                                          |
| MVP-2.2                         | NN Scaling Law                   | 2     | ⏳      | -                                       | -                                                          |
| MVP-3.0                         | Noise Info Limit                 | 3     | ⏳      | -                                       | -                                                          |
| MVP-3.1                         | Model Capacity                   | 3     | ⏳      | -                                       | -                                                          |
| MVP-3.2                         | Feature Analysis                 | 3     | ⏳      | -                                       | -                                                          |
| **MVP-16T (V1)**                | **❌ Fisher/CRLB (失败-非网格数据)**     | 16    | ❌      | `SCALING-20251223-fisher-ceiling-01`    | [Link](./exp/exp_scaling_fisher_ceiling_20251223.md)       |
| **MVP-16T (V2)** ✅             | **✅ Fisher/CRLB (规则网格数据)**        | 16    | ✅     | `SCALING-20251224-fisher-ceiling-02`    | [Link](./exp/exp_scaling_fisher_ceiling_v2_20251224.md)    |
| **MVP-16B**                     | **🔴 Baseline 统计可信度 (P0)**       | 16    | 🔴     | `SCALING-20251223-baseline-stats-01`    | [Link](./exp/exp_scaling_baseline_stats_20251223.md)       |
| **MVP-16L**                     | **🟡 LMMSE 线性上限 (P1)**           | 16    | ⏳      | `SCALING-20251223-lmmse-ceiling-01`     | -                                                          |
| **MVP-16W**                     | **🟡 Whitening 表示 (P1)**         | 16    | ⏳      | `SCALING-20251223-whitening-noise1-01`  | -                                                          |
| **MVP-16A-0** 🆕               | **✅ Oracle MoE Structure Bonus (P0)** | 16    | ✅      | `SCALING-20251223-oracle-moe-noise1-01` | [Link](./exp/exp_scaling_oracle_moe_noise1_20251223.md)    |
| **MVP-16CNN**                   | **🟢 1D-CNN @ noise=1 (P2)**     | 16    | ⏳      | `SCALING-20251223-cnn-noise1-01`        | -                                                          |
|                                 |                                  |       |        |                                         |                                                            |
| **🔄 Phase T: Fisher 校准（V2 重新立项）** |                                  |       |        |                                         |                                                            |
| ~~MVP-T0~~                      | ~~Noise Monotonicity~~           | T     | ❌      | -                                       | 方法失败，取消                                                    |
| ~~MVP-T1~~                      | ~~Confounding Ablation~~         | T     | ❌      | -                                       | 方法失败，取消                                                    |
| **MVP-T2**                      | **🟡 LLR Jacobian (P1 降级)**      | T     | ⏳      | `SCALING-20251223-fisher-llr-01`        | -                                                          |
| **MVP-T3**                      | **🟢 Scale Audit (P2 快速)**       | T     | ⏳      | `SCALING-20251223-scale-audit-01`       | -                                                          |
|                                 |                                  |       |        |                                         |                                                            |
| **🆕 Phase D: 经验上限（替代 Fisher）** |                                  |       |        |                                         |                                                            |
| **MVP-D0**                      | **🔴 noise=0 Oracle 上限 (P0)**    | D     | 🔴     | `SCALING-20251223-noise0-oracle-01`     | -                                                          |
|                                 |                                  |       |        |                                         |                                                            |
| **🆕 Phase A: noise=1 MoE**     |                                  |       |        |                                         |                                                            |
| **MVP-16A-0**                   | **🔴 Oracle MoE @ noise=1 (P0)** | A     | ✅     | `SCALING-20251223-oracle-moe-noise1-01` | [exp](./exp/exp_scaling_oracle_moe_noise1_20251223.md)     |
| **MVP-16A-1**                   | **✅ Gate-feat Sanity (P1)**     | A     | ✅      | `SCALING-20251223-gate-feat-01`         | [exp](./exp/exp_scaling_gate_feat_sanity_20251224.md)      |
| **MVP-16A-2**                   | **🟡 Soft-gate MoE (P1)**        | A     | ⏳      | `SCALING-20251223-soft-moe-noise1-01`   | -                                                          |
|                                 |                                  |       |        |                                         |                                                            |
| **🆕 Phase NN: 神经网络 Baseline (2025-12-24 大立项)** |                                  |       |        |                                         |                                                            |
| **MVP-NN-0**                    | **✅ 可靠基线框架 (P0)**        | NN    | ✅     | `SCALING-20251224-nn-baseline-framework-01` | [Link](./exp/exp_scaling_nn_baseline_framework_20251224.md) |
| **MVP-MLP-1**                   | **🔴 最小可行 MLP (P0)**        | NN    | ⏳     | `SCALING-20251224-mlp-baseline-01`      | -                                                          |
| **MVP-CNN-1**                   | **🟡 最小 1D CNN (P1)**         | NN    | ⏳     | `SCALING-20251224-cnn-baseline-01`      | -                                                          |
| **MVP-CNN-2**                   | **🟡 多尺度 CNN (P1)**          | NN    | ⏳     | `SCALING-20251224-cnn-multiscale-01`    | -                                                          |
| **MVP-Compare**                 | **三件套同评估**                | NN    | ⏳     | `SCALING-20251224-nn-compare-01`        | -                                                          |
| **MVP-MoE-CNN-0**               | **🟢 MoE-CNN (P2, 条件启动)**   | NN    | ⏳     | `SCALING-20251224-moe-cnn-oracle-01`    | -                                                          |

**Status Legend:**
- ⏳ Planned | 🔴 Ready | 🚀 Running | ✅ Done | ❌ Cancelled | ⏸️ Paused

## 2.2 Configuration Reference

> **Key configurations across all MVPs**

| MVP | Data Size | Noise Level | Model | Key Variable | Acceptance |
|-----|-----------|-------------|-------|--------------|------------|
| MVP-1.0 | 1M train | σ=1.0 | Ridge | alpha sweep | R² < 0.6 |
| MVP-1.1 | 1M train | σ=1.0 | LightGBM | best config | R² < 0.65 |
| MVP-1.2 | 100k→1M | σ=1.0 | Ridge+LGB | data size | ΔR² < 0.03 |
| MVP-2.0 | 1M train | σ=1.0 | MLP | architecture | R² > 0.70 |
| MVP-2.1 | 1M train | σ=1.0 | CNN | architecture | R² > Ridge + 0.15 |
| MVP-2.2 | 100k→1M | σ=1.0 | MLP+CNN | data size | 持续提升 |

---

# 3. 🔧 MVP Specifications

> **Detailed specs for each MVP, ready for execution**

## Phase 1: ML Ceiling

### MVP-1.0: Ridge 1M Ceiling

| Item | Config |
|------|--------|
| **Objective** | 验证 Ridge 在 1M 数据 + noise=1 下的性能上限 |
| **Hypothesis** | H1.1.1: Ridge R² < 0.6 |
| **Data** | mag205_225_lowT_1M (1M train), noise_level=1.0, target=log_g |
| **Model** | Ridge Regression |
| **Features** | 全波段光谱 (~4000 维) |
| **Hyperparams** | alpha ∈ {0.01, 0.1, 1.0, 10, 100, 1000} |
| **Acceptance** | R² < 0.6 ⟹ 确认瓶颈 |
| **Early Stop** | N/A (Ridge 无迭代) |

**Expected Result:**
- Ridge 在 1M 数据下仍然 R² ≈ 0.55，与 100k 差别不大
- 证明线性模型无法从大数据中获益

**Steps:**
1. 加载 mag205_225_lowT_1M 全部 5 个 shard
2. 添加 noise_level=1.0 的高斯噪声
3. 扫描 alpha 参数
4. 记录最佳 R²

---

### MVP-1.1: LightGBM 1M Ceiling

| Item | Config |
|------|--------|
| **Objective** | 验证 LightGBM 在 1M 数据 + noise=1 下的性能上限 |
| **Hypothesis** | H1.2.1: LightGBM R² < 0.65 |
| **Data** | mag205_225_lowT_1M (1M train), noise_level=1.0, target=log_g |
| **Model** | LightGBM Regressor |
| **Features** | 全波段光谱 (~4000 维) |
| **Hyperparams** | lr=0.05, n_estimators=5000, early_stopping |
| **Acceptance** | R² < 0.65 ⟹ 确认瓶颈 |

**Expected Result:**
- LightGBM 略优于 Ridge，但仍受限
- 增加树数量不再提升性能

---

### MVP-1.2: ML Scaling Law

| Item | Config |
|------|--------|
| **Objective** | 对比 100k vs 1M 数据对 ML 方法的影响 |
| **Hypothesis** | H1.3.1: Ridge ΔR² < 0.02; H1.3.2: LightGBM ΔR² < 0.03 |
| **Data** | 100k 子集 vs 1M 全集, noise_level=1.0 |
| **Model** | Ridge + LightGBM (best config from 1.0, 1.1) |
| **Acceptance** | 边际收益递减明显 |

**Steps:**
1. 使用 MVP-1.0, 1.1 的最优配置
2. 分别在 100k 和 1M 上训练
3. 画 data size vs R² 曲线

---

## Phase 1.x: ML Refinement (P0/P1/P2)

### MVP-1.3: Stats Validation (🔴 P0)

| Item | Config |
|------|--------|
| **Objective** | 确认 "plateau" 是真实还是统计假象 |
| **Hypothesis** | H1.4.1: 多 seed 时 1M vs 500k 差异在误差棒内; H1.4.2: 扩大 test 后趋势不变 |
| **Method 1** | 多 seed 重复：200k, 500k, 1M 各跑 3-5 次不同 seed |
| **Method 2** | 扩大 test set：从 500 → 1000+ |
| **Key Metric** | R² 的 mean ± std |
| **Acceptance** | 如果 1M vs 500k 差异 < std，则确认 plateau |

**Expected Output:**
- 确认 plateau 是否真实
- 提供统计误差棒，指导后续实验的显著性判断

---

### MVP-1.4: Ridge α Extended Sweep (🔴 P0)

| Item | Config |
|------|--------|
| **Objective** | 找到 Ridge 在 noise=1 下的真正最优 α |
| **Hypothesis** | H1.5.1: 最优 α 在 5000~1e8 之间存在峰值后下降 |
| **Data** | 100k 和 1M 两个数据点 |
| **α Range** | `logspace(2, 8, 13)`: 1e2, 3e2, 1e3, ..., 1e8 |
| **Acceptance** | 观察到"峰值后下降"模式 |

**Expected Output:**
- Ridge 真正最优 α
- α vs R² 曲线图（应呈现倒 U 型）

---

### MVP-1.5: LightGBM Param Extended (🔴 P0)

| Item | Config |
|------|--------|
| **Objective** | 验证 LightGBM 参数空间是否探索完全 |
| **Hypothesis** | H1.6.1: num_leaves↑ 能提升; H1.6.2: lr↓ 能提升 |
| **Sweep 1** | num_leaves: 63 → 127 → 255 |
| **Sweep 2** | learning_rate: 0.05 → 0.02 → 0.01 |
| **Sweep 3** | min_data_in_leaf: 20 → 100 → 500 |
| **Control** | 固定训练轮数对比（不用 early stopping）做 sanity check |
| **Acceptance** | 任一配置 ΔR² > 0.01 |

**Expected Output:**
- LightGBM 真正最优配置
- 参数敏感度分析

---

### MVP-1.6: Whitening/SNR Input (🟡 P1)

| Item | Config |
|------|--------|
| **Objective** | 验证 Whitening (flux/error) 输入是否提升性能 |
| **Hypothesis** | H1.7.1: Whitening 能提升 R² > 0.02 |
| **Input Variants** | 1) raw flux, 2) StandardScaler, 3) flux/error (SNR), 4) (flux-μ)/error |
| **Models** | Ridge (best α from 1.4) + LightGBM (best config from 1.5) |
| **Acceptance** | Whitened > StandardScaled |

**Expected Output:**
- 最优输入表示方式
- 物理解释：SNR 归一化的意义

---

### MVP-1.7: PCA vs PLS 降维策略 (🟡 P1)

| Item | Config |
|------|--------|
| **Objective** | 对比监督降维 (PLS) vs 无监督降维 (PCA)，并探索 PCA 空间选择 |
| **Hypothesis** | H1.7.2: PLS 优于 PCA（相同维度）; H1.7.3: PCA 可能误伤低方差高信息特征; H1.7.4: Whitened/Denoised space 建 PCA 更稳健 |
| **experiment_id** | `SCALING-20251223-pca-pls-01` |
| **Report** | [Link](./exp/exp_scaling_pca_pls_comparison_20251223.md) |

#### 设计 1：PCA + Ridge K Sweep

| 配置项 | 值 |
|--------|-----|
| **降维方法** | PCA |
| **K 值** | 100, 200, 500, 1000 |
| **下游模型** | Ridge (best α from MVP-1.4) |
| **关键观察** | K 增大时 R² 是否先升后 plateau |

#### 设计 2：PLS vs PCA 对照

| 配置项 | 值 |
|--------|-----|
| **方法 A** | PCA + Ridge |
| **方法 B** | PLSRegression（监督降维） |
| **K 值** | 100, 200, 500, 1000 |
| **理论优势** | PLS 按 X-y 协方差找子空间，更适合"弱信号回归" |

#### 设计 3：PCA 空间选择

| PCA 空间 | 描述 | 推荐程度 |
|----------|------|----------|
| **Noisy space** | 直接在含噪光谱上 PCA | 默认，但有风险 |
| **Whitened space** | PCA((X - μ) / error) | ⭐ 推荐 |
| **Denoised space** | 平滑后 PCA，再投影 noisy | ⭐ 推荐 |

**⚠️ 核心风险**：
- PCA 保留的是**方差最大**的方向
- log_g 敏感特征可能是"**低方差但信息密度高**"的细谱线
- 高噪声下，PCA 可能把关键信号扔掉

**Expected Output:**
- K vs R² 曲线图（先升后 plateau？）
- PLS vs PCA 对比图
- PCA 空间选择对比图
- 最优降维策略建议

---

### MVP-1.8: MoE 分段建模 (🟢 P2)

| Item | Config |
|------|--------|
| **Objective** | 按 Teff/log_g 分段建模，改善极值区域 |
| **Method** | 粗分类（按 Teff 区间或 log_g bin）→ 每段独立模型 |
| **Risk** | 实现复杂，可能过拟合 |

---

**🆕 MVP-16A-0 Oracle MoE 结果 (2025-12-23 已完成):**

| Metric | Value |
|--------|-------|
| Global Ridge R² | 0.4611 (1k test) |
| Oracle MoE R² | **0.6249** |
| **ΔR²** | **+0.1637** (>>0.03 阈值) |
| Decision | ✅ MoE 路线继续 |

> **结论**: Oracle MoE 在 noise=1 + 1M 数据下展示强结构红利，所有 9 个 bin 都优于全局 Ridge。

---

### MVP-1.9: 物理特征工程 (🟢 P2)

| Item | Config |
|------|--------|
| **Objective** | 尝试物理驱动的特征工程 |
| **Features** | 等效宽度(EW)、线心/线翼比、局部卷积滤波响应 |
| **Risk** | 领域知识依赖重 |

---

## Phase 2: NN Advantage

### MVP-2.0: MLP 1M Performance

| Item | Config |
|------|--------|
| **Objective** | 验证 MLP 在 1M 数据下能否突破 ML 瓶颈 |
| **Hypothesis** | H2.1.1: MLP R² > 0.70 |
| **Data** | mag205_225_lowT_1M (1M train), noise_level=1.0, target=log_g |
| **Model** | MLP (3-4 layers, ReLU) |
| **Hyperparams** | hidden_dim=512, layers=3, batch=1024, lr=1e-3 |
| **Acceptance** | R² > 0.70 且 > Ridge + 0.10 |

---

### MVP-2.1: CNN 1M Performance

| Item | Config |
|------|--------|
| **Objective** | 验证 CNN 在 1M 数据下的性能 |
| **Hypothesis** | H2.2.1: CNN R² > Ridge + 0.15 |
| **Data** | mag205_225_lowT_1M (1M train), noise_level=1.0, target=log_g |
| **Model** | 1D CNN (dilated convolutions) |
| **Hyperparams** | 参考 cnn_main 的最优配置 |
| **Acceptance** | R² > 0.70 |

---

### MVP-2.2: NN Scaling Law

| Item | Config |
|------|--------|
| **Objective** | 验证 NN 的数据 scaling 是否持续有效 |
| **Data** | 100k / 200k / 500k / 1M |
| **Model** | MLP + CNN |
| **Acceptance** | NN 的 R² 持续上升，而 ML 饱和 |

---

## Phase 3: Analysis

### MVP-3.0: Noise Information Limit

| Item | Config |
|------|--------|
| **Objective** | 分析 noise=1 时的理论信息上限 |
| **Method** | 计算理论 SNR，估计最大可能 R² |

---

### MVP-3.1: Model Capacity Analysis

| Item | Config |
|------|--------|
| **Objective** | 分析不同模型的有效容量 |
| **Method** | 对比 parameter count vs performance |

---

### MVP-3.2: Feature Representation

| Item | Config |
|------|--------|
| **Objective** | 分析 NN 学到了什么 ML 学不到的特征 |
| **Method** | 特征可视化，attention map 分析 |

---

## 🔴 Phase 16: Ceiling 三层论证（2025-12-23 新增）

> **核心理念**：先推出理论上限 → 再证明 Ridge/LGBM ceiling → 再展示 MoE/NN 接近上限
> 
> **性价比优先三件套**：MVP-16T (Fisher) → MVP-16O (Oracle MoE) → MVP-16B (可信度)

### MVP-16T-V2: Fisher/CRLB 理论上限（🔴 P0 最高优先级 - 规则网格数据）

> **V1 失败原因**：BOSZ 连续采样数据导致邻近点差分法失效
> **V2 解决方案**：使用新生成的规则网格数据 `grid_mag215_lowT`

| Item | Config |
|------|--------|
| **Objective** | 使用规则网格数据计算 noise=1 时的理论可达上限 R²_max，量化 degeneracy |
| **Hypothesis** | H-16T.1: R²_max ≥ 0.75 (存在大 headroom) |
| **Hypothesis** | H-16T.2: degeneracy 显著 (log_g 与 Teff/[M/H] 纠缠) |
| **Data** | `/datascope/subaru/user/swei20/data/bosz50000/grid/grid_mag215_lowT/dataset.h5` (30,182 samples) |
| **Grid** | T_eff: 250K step, log_g: 0.5 step, [M/H]: 0.25 step |

**方法（最小可行版本）**：
1. 抽样 N=5k~20k 个参数点（不必用全 1M）
2. 对每个点，用 BOSZ forward model 在 θ±Δθ 做有限差分，得到 ∂μ/∂θ
3. 用 error×noise_level 组成 Σ（对角即可）
4. 计算 Fisher 信息矩阵：I(θ) = (∂μ/∂θ)ᵀ Σ⁻¹ (∂μ/∂θ)
5. 做 Schur complement，得到每个样本的 Var_min(log_g)
6. 聚合（均值/分位数），转成 R²_max 上界估计

**关键公式**：

$$R^2_{\max} \lesssim 1 - \frac{\mathbb{E}[\mathrm{CRLB}_{\log g}]}{\mathrm{Var}(\log g)}$$

**输出**：
- R²_max,CRLB（以及分布：median/90% 分位）
- degeneracy 指标：Fisher 条件数、log_g 与 Teff/[M/H] 的相关项强度

**止损规则**：
- 如果 R²_max ≈ 0.6 → "想大幅提升"基本不现实，目标改为"逼近上限 + 不确定度输出"
- 如果 R²_max ≥ 0.75 → 确实存在大 headroom，值得上 CNN/更强表征

**参考文献**：
- Fisher/CRLB：统计学经典推导
- van Trees 不等式（Bayesian CRLB）
- 天文应用：Gaia XP 光谱参数估计工作

---

### MVP-16B: Baseline 统计可信度（🔴 P0）

| Item | Config |
|------|--------|
| **Objective** | 把 "Ridge=0.50 / LGBM=0.57" 做成可信的 ceiling |
| **Hypothesis** | H-16B.1: 多 seed 确认 std < 0.01 |
| **Hypothesis** | H-16B.2: 扩大 test 后结论不变 |

**方法 B1（多 seed + 大 test）**：
- 训练集固定（1M 或 500k），换 5-10 个 seed
- test 从 500 扩到 5k~20k（至少 5k）
- 给出均值±std 或 CI

**方法 B2（LGBM 参数空间扩展）**：
- 扫 num_leaves, max_depth, lr, 更严格的正则
- 检查 early stopping 是否过早
- 输出：最优曲线与 plateau 证据

**输出**：
- R² 分布 + 方差解释
- LGBM 参数 plateau 证据

---

### MVP-16L: LMMSE 线性上限（🟡 P1）

| Item | Config |
|------|--------|
| **Objective** | 给 Ridge 一个"可证明的线性上限" |
| **Hypothesis** | H-16L.1: Ridge ≈ LMMSE (差 < 1%) |

**方法**：
- 用 1M 数据估计 Σ_xx, Σ_xy
- 计算最优线性预测器 w* = Σ_xx⁻¹ Σ_xy（或数值正规化）
- 计算其 test R²（这是"线性模型族"的上限）

**输出**：
- 如果 Ridge 与 LMMSE 差 < 0.005~0.01，可以写：
  "Ridge 已接近最优线性可达性能，因此线性模型族不可能再大幅提升"

---

### MVP-16W: Whitening 表示（🟡 P1）
| **MVP-16A-0** 🆕               | **✅ Oracle MoE Structure Bonus (P0)** | 16    | ✅      | `SCALING-20251223-oracle-moe-noise1-01` | [Link](./exp/exp_scaling_oracle_moe_noise1_20251223.md)    |

| Item | Config |
|------|--------|
| **Objective** | 验证 Whitening (flux/error) 在 noise=1 下的提升 |
| **Hypothesis** | H-16W.1: ΔR² ≥ 0.02 |

**输入变体**：
1. raw flux
2. StandardScaler
3. flux/error (SNR)
4. (flux-μ)/error

**模型**：Ridge (best α) + LightGBM (best config) + CNN

**决策规则**：
- 如果 ΔR² ≥ 0.02 → Whitening 应并入所有后续模型（包括 MoE/CNN）

---

### MVP-16CNN: 1D-CNN @ noise=1（🟢 P2）

| Item | Config |
|------|--------|
| **Objective** | 验证 CNN 能否从 0.57 往上冲一大截 (0.65~0.75) |
| **Hypothesis** | H-16CNN.1: CNN R² > 0.65 |
| **Hypothesis** | H-16CNN.2: CNN R² - Ridge R² > 0.10 |

**MVP 设计建议**：
1. 输入：whitened spectrum (flux/error)
2. 架构：小 ResNet1D 或 4-8 层 Conv1D + pooling + MLP head
3. 训练目标：只做 log_g 或 multi-task (Teff, [M/H], log_g)
   - multi-task 在 degeneracy 强时通常更稳
4. 评估：与 R²_max,CRLB 对齐，看 gap 还剩多少

**依赖**：
- 建议先完成 MVP-16T，确认 R²_max ≥ 0.75 后再投入 CNN

---

## ❌ Phase T: Fisher Ceiling 校准（已失败，重新规划）

> **失败原因**：MVP-16T 的方法论存在根本性缺陷
> - BOSZ 数据是**连续采样**（~40k 唯一值/参数），不是规则网格
> - 邻近点差分法在非网格数据上**完全失效**
> - CRLB 跨越 20 个数量级，R²_max=0.97 **不可靠**

### ❌ MVP-T0: Noise Monotonicity → **取消**

**取消原因**：底层方法已失败，扫 noise_level 无意义

---

### ❌ MVP-T1: Confounding Ablation → **取消**

**取消原因**：问题不是"约束太松"，而是**整个差分方法不适用于非网格数据**

---

### 🔄 MVP-T2: Local Linear Regression Jacobian → **升级为新方案**

| Item | Config |
|------|--------|
| **Objective** | 用局部多项式回归估计 Jacobian（exp.md 方案 B） |
| **Hypothesis** | H-T2.1: CRLB 分布合理（无 20 个数量级跨度） |
| **Method** | 对每个样本 i，找 K 个近邻，拟合 μ(θ) ≈ a + J·Δθ |
| **Status** | 🟡 P1（降级，等 16A-0 和 NN-0 先跑） |

**关键改进**：
- 用最小二乘拟合 J，而不是两点差分
- 天然处理近邻方向不正交的问题
- 需要足够多的近邻（K ≥ 10）

---

### ✅ MVP-T3: Scale Audit → **保留**

| Item | Config |
|------|--------|
| **Objective** | 确认 noise=1 实际 SNR |
| **Hypothesis** | H-T3.1: median(\|flux\|)/median(error×σ) ≈ 1 |
| **Method** | 打印 SNR 统计量，确认口径 |
| **Status** | 🟢 P2（快速验证，5 分钟内完成） |

---

## 🆕 Phase D: 经验上限替代方案（exp.md 方案 D）

> **核心思路**：既然理论 Fisher/CRLB 难以正确计算，改用**经验上限**
> 
> **优点**：实践可行，结果可信
> **缺点**：不是严格的理论上限

### MVP-D0: noise=0 Oracle 上限（🔴 P0 新增）

| Item | Config |
|------|--------|
| **Objective** | 用 noise=0 的最佳模型作为理论上限参照 |
| **Hypothesis** | H-D0.1: noise=0 时 Ridge R² > 0.95 |
| **Hypothesis** | H-D0.2: headroom = R²(noise=0) - R²(noise=1) > 0.40 |
| **Data** | 100k + 1M，noise_level = 0（或极小如 0.01） |
| **Models** | Ridge (best α), LightGBM (best config) |
| **Method** | 在无噪声数据上训练，记录 R² |
| **Acceptance** | 输出 noise=0 的 R² 作为经验上限 |

**逻辑**：
- noise=0 时的 R² 是**所有模型的经验上限**
- headroom = R²(noise=0) - R²(noise=1) = 可恢复的信息量
- 如果 headroom > 0.40 → 模型/表示改进有很大空间

**执行步骤**：
1. 复用 MVP-1.0/1.1 的脚本
2. 设置 noise_level = 0（或 0.01）
3. 训练 Ridge (α=1e5) 和 LightGBM
4. 记录 R²，计算 headroom

**预期结果**：
- Ridge @ noise=0: R² ≈ 0.98+（接近完美）
- Headroom = 0.98 - 0.50 = **0.48**（巨大空间）

---

## 🆕 Phase A: noise=1 MoE 结构红利（2025-12-23 新增）

> **核心问题**：noise=1 下 MoE 的结构红利是否还存在？
> 
> **策略**：先用 Oracle 确认 headroom，再决定是否做 soft gate

### MVP-16A-0: Oracle MoE @ noise=1（🔴 P0 最高优先级）

| Item | Config |
|------|--------|
| **Objective** | 不训练 gate，用真值路由，看结构红利 |
| **Hypothesis** | H-A0.1: ΔR² ≥ 0.03 vs Global Ridge |
| **Method** | 真值 (Teff×[M/H]) 分 9 bins，每 bin 训练 Ridge expert |
| **Acceptance** | ΔR² ≥ 0.03 → MoE 还有戏；ΔR² ≈ 0 → 放弃 MoE，转 NN |

**决策规则**:
- ✅ ΔR² ≥ 0.03: 继续 MVP-16A-1, A-2
- ❌ ΔR² < 0.03: MoE 路线关闭，专注 NN/表示学习

**可复用**:
- 低噪 MoE 的 bin 划分逻辑
- Ridge expert 超参（α ∈ [1e4, 1e5]）

---

### MVP-16A-1: Gate-feat Sanity @ noise=1（🟡 P1）

| Item | Config |
|------|--------|
| **Objective** | 评估 gate 特征在高噪下的信号 |
| **Hypothesis** | H-A1.1: Ca II triplet 等特征可区分 bins |
| **Method** | 不训练 MoE，只评估特征的分类/相关性 |
| **Risk** | 物理窗特征 SNR 可能崩，导致 gate 输入变成噪声 |

---

### MVP-16A-2: Soft-gate MoE @ noise=1（🟡 P1）

| Item | Config |
|------|--------|
| **Objective** | 复用低噪验证的 soft routing 配方 |
| **Hypothesis** | H-A2.1: Soft routing 能保持 ≥70% oracle 收益 |
| **Method** | 直接复用低噪的 soft gate 架构 |
| **依赖** | MVP-16A-0 ΔR² ≥ 0.03, MVP-16A-1 特征有信号 |

---

## 🆕 Phase NN: 神经网络 Baseline（2025-12-24 大立项）

> **核心问题**：单模型 NN 能否接近/超过 Oracle MoE 的 0.62？
> 
> **目标**：判断 (1) 结构不对 还是 (2) 输入/训练策略不对
> 
> **参考**：Oracle MoE @ noise=1 = **0.62**（结构性 headroom 存在）

### 🔑 总原则（避免"结构不对，堆数据没用"）

**三个容易踩坑的点必须锁死**：

| # | 坑点 | 解决方案 |
|---|------|---------|
| 1 | **输入 whitening** | `x = flux / (error * noise_level)` 或双通道 `[flux, error]` |
| 2 | **输出目标尺度** | `y = (logg - mean) / std` 标准化回归 |
| 3 | **评估稳定性** | 固定 test ≥ 20k，固定 seed |

---

### MVP-NN-0: 可靠基线框架（一天内完成）

| Item | Config |
|------|--------|
| **Objective** | 建立 NN 训练管线 + 保证输入/评估没问题 |
| **Data** | stratify split (按 Teff/logg/[M/H] 分桶分层抽样) |
| **Input Variants** | A: `flux_whiten = flux / (error × σ)` (推荐) <br> B: 双通道 `[flux, error]` |
| **Loss** | MSE（先别加物理项） |
| **Optimizer** | AdamW + cosine/step LR + early stopping (val R² 3-5 epoch 不涨就停) |
| **Training Scale** | **100k** 做 smoke test（别直接 1M） |
| **Acceptance** | 能在 100k 上稳定复现 Ridge/LGBM 水平，train/val 曲线正常 |
| **experiment_id** | `SCALING-20251224-nn-baseline-framework-01` |

---

### MVP-MLP-1: 最小可行 MLP + 明确止损

| Item | Config |
|------|--------|
| **Objective** | 快速验证"全局 MLP 是否注定不行" |
| **Hypothesis** | H-MLP1.1: 100k→1M 提升 < +0.02 R² → MLP 归纳偏置不对 |
| **Input** | **4096 维** (BOSZ 光谱长度) |
| **Architecture** | `Linear(4096→2048)→GELU→Dropout` → `2048→1024` → `1024→512` → `512→1` |
| **Regularization** | weight_decay=1e-4, dropout=0.1, **LayerNorm 放第一层后** |
| **Training** | 100k 先训练到收敛 (10-20 epochs)，再同结构上 1M |
| **Record** | train R², val R², test R², 收敛 step 数 |
| **experiment_id** | `SCALING-20251224-mlp-baseline-01` |

**🚨 MLP 止损信号（非常明确）**：
- 如果 **100k→1M 提升 < +0.02 R²** 且 val 曲线 plateau 很早：
  → 结论：**MLP 架构归纳偏置不对**，不要再在 MLP 上花时间
- 如果提升明显（+0.05 以上），才值得继续优化 MLP

---

### MVP-CNN-1: 最小 1D CNN（验证"局部结构"带来质变）

| Item | Config |
|------|--------|
| **Objective** | 看 CNN 能否明显超过 MLP / 接近 0.62 |
| **Hypothesis** | H-CNN1.1: CNN 100k 明显超过 MLP (≥+0.05 R²) |
| **Input** | 强烈建议 `flux_whiten`（或双通道） |
| **Architecture** | Stem: Conv1d(1→32, k=7) + GELU <br> Block × 4: Conv1d(32→64, k=5, dilation=1) → Conv1d(64→64, k=5, dilation=2) + 残差 + LayerNorm/GroupNorm <br> Pool: Global average pooling <br> Head: MLP 64→128→1 |
| **Training** | 先 100k，如果比 MLP 好很多（+0.05+）再上 1M |
| **experiment_id** | `SCALING-20251224-cnn-baseline-01` |

**CNN 止损信号**：
- 如果 CNN 100k 也不如 Ridge/LGBM，且怎么调 LR/正则都不行：
  → 80% 可能是 **输入/whitening/训练细节有 bug**，而不是 CNN 不行

---

### MVP-CNN-2: 多尺度 / 大感受野（专打 noise=1）

| Item | Config |
|------|--------|
| **Objective** | noise=1 核心：单条谱线信息不稳，需跨更宽波段累积证据 |
| **Hypothesis** | H-CNN2.1: 多尺度 CNN R² ≥ 0.60（逼近 Oracle MoE 0.62） |
| **增强方式 1** | dilation schedule: [1, 2, 4, 8] |
| **增强方式 2** | 多分支卷积核: k = [3, 7, 15] 并行分支后 concat（类似 Inception1D） |
| **experiment_id** | `SCALING-20251224-cnn-multiscale-01` |

---

### MVP-Compare: 三件套同评估

| Item | Config |
|------|--------|
| **Objective** | 在同一固定 test set 上比较，决定下一步路线 |
| **Models** | (1) Global Ridge/LGBM（已有） <br> (2) Oracle MoE = 0.62（已做） <br> (3) Global CNN（本次做） |
| **Decision** | - CNN ≥ 0.62 → 单模型 CNN 已吃掉结构红利，MoE 不必须 <br> - CNN 接近 0.62（差<0.02）→ 先做强 CNN，不急着 MoE <br> - CNN 明显低于 0.62（差≥0.05）→ MoE-CNN 才是正道 |

---

### MVP-MoE-CNN-0: 最保守 MoE-CNN（仅当 global CNN 明显打不过 oracle 时启动）

| Item | Config |
|------|--------|
| **Objective** | 验证 CNN expert 是否比 Ridge expert 更强 |
| **Hypothesis** | H-MoE-CNN.1: MoE(CNN experts) > MoE(Ridge experts) |
| **Experts** | 每个 bin 一个小 CNN（就是 MVP-CNN-1 的 CNN） |
| **Routing** | 先用 **真值路由（oracle）** 验证 CNN expert 效果 |
| **后续** | 然后再做 soft gate（复用之前成熟的 soft routing） |
| **experiment_id** | `SCALING-20251224-moe-cnn-oracle-01` |

---

### 📋 NN Baseline 必须记录的 5 个数字（写结论用）

| # | 指标 | 说明 |
|---|------|------|
| 1 | **100k → 1M 的 ΔR²** | 每个模型一个，判断数据规模收益 |
| 2 | **plateau epoch** | 训练到 plateau 需要多少 step/epoch |
| 3 | **per-bin R²** | 特别是最差的几个 bin |
| 4 | **whitening 敏感度** | 有无 whitening 的差距 |
| 5 | **vs Oracle gap** | global CNN vs Oracle MoE 的差距 |

---

### 🎯 推荐执行顺序

| 顺序 | MVP | 目的 | 时间预估 |
|------|-----|------|---------|
| 1 | MVP-NN-0 | 框架搭建 | 半天 |
| 2 | MVP-MLP-1 @100k + @1M | 快速止损/确认"MLP 不吃数据" | 1天 |
| 3 | MVP-CNN-1 @100k | 确认归纳偏置对不对 | 半天 |
| 4 | MVP-CNN-1 @1M | 看"大力出奇迹"是否成立 | 1天 |
| 5 | MVP-CNN-2 | 多尺度 CNN（如需） | 1天 |
| 6 | MVP-MoE-CNN-0 | 仅当 global CNN < 0.60 | 视情况 |

---

# 4. 📊 Progress Tracking

## 4.1 Kanban View

```
┌──────────────────┬──────────────────┬──────────────┬──────────────┬──────────────┐
│    ⏳ Planned    │     🔴 Ready     │  🚀 Running  │    ✅ Done   │  ❌ Cancelled │
├──────────────────┼──────────────────┼──────────────┼──────────────┼──────────────┤
│ MVP-CNN-1 (P1)   │ **MVP-MLP-1(P0)**│              │ MVP-1.0      │ MVP-T0       │
│ MVP-CNN-2 (P1)   │ MVP-D0 (P0)      │              │ MVP-1.1      │ MVP-T1       │
│ MVP-MoE-CNN-0    │ MVP-16B (P0)     │              │ MVP-1.2      │              │
│ MVP-16A-2 (P1)   │                  │              │ MVP-1.4 ✅   │              │
│ MVP-16L (P1)     │                  │              │ MVP-1.6 ✅   │              │
│ MVP-T2 (降级)    │                  │              │ MVP-16T-V2✅ │              │
│                  │                  │              │ MVP-16A-0 ✅ │              │
│                  │                  │              │ MVP-16A-1 ✅ │              │
│                  │                  │              │**MVP-NN-0✅**│              │
└──────────────────┴──────────────────┴──────────────┴──────────────┴──────────────┘
```

### 🔴 新 P0 优先级（2025-12-23 更新 v2 - Fisher 失败后）

> **核心策略**：放弃 Fisher ceiling，改用经验上限 + 直接验证 MoE/NN

**P0 三件套（决定路线）**：
1. **MVP-D0 (noise=0 Oracle)** → 用经验上限替代理论 ceiling
2. **MVP-16A-0 (Oracle MoE)** → 决定"noise=1 下 MoE 值不值"
3. **MVP-NN-0 (1D CNN)** → 验证"表示学习能否吃掉 headroom"

**决策树**：
```
MVP-D0 完成后
├─ noise=0 R² > 0.95 → 确认理论上可接近很高
│   └─ headroom = R²(noise=0) - R²(noise=1)
└─ noise=0 R² ≈ 0.80 → 物理上限本身不高
    └─ 调整预期

MVP-16A-0 完成后
├─ ΔR² ≥ 0.03 → MoE 有戏
│   └─ 继续 MVP-16A-1, A-2
└─ ΔR² < 0.03 → MoE 关闭
    └─ 专注 NN/表示学习

MVP-NN-0 完成后
├─ R² > 0.62 → NN 能吃 headroom
│   └─ 形成证据链
└─ R² ≈ 0.57 → 问题更深
    └─ 考虑 multi-task 解纠缠
```

**已取消**：
- ~~MVP-T0 (Noise Monotonicity)~~ → 方法失败
- ~~MVP-T1 (Confounding Ablation)~~ → 方法失败

**降级**：
- MVP-T2 (LLR Jacobian) → P1，等有时间再尝试
- MVP-16B (Baseline 可信度) → 可稍后

## 4.2 Key Conclusions Snapshot

> **One-line conclusion per completed MVP, synced to Hub**

| MVP | Conclusion | Key Metric | Synced to Hub |
|-----|------------|------------|---------------|
| MVP-1.0 | Ridge 在 1M + noise=1 下达到 R²=0.46 | R²=0.46 (1k test) | ✅ |
| MVP-1.1 | LightGBM 在 1M + noise=1 下达到 R²=0.57 | R²=0.5709 | ✅ |
| MVP-1.2 | 100k→1M 仅提升 2-3%，边际收益递减 | ΔR²<0.03 | ✅ |
| MVP-1.3 | TODO: 确认 plateau 统计可信度 | - | ⏳ |
| MVP-1.4 | 倒 U 型曲线确认，最优 α=1e4~1e5，优化提升仅 0.4%~2.5% | 100k: α=3.16e+04, R²=0.4856; 1M: α=1e+05, R²=0.5017 | ✅ |
| MVP-1.5 | TODO: 验证 LightGBM 参数极限 | - | ⏳ |
| **MVP-1.6** | **H1.7.1 ❌: SNR ΔR²=+0.015 未达阈值; ⚠️ StandardScaler 严重损害 LightGBM (-0.36)** | Ridge snr_centered: R²=0.5222; LightGBM raw: R²=0.5533 | ✅ |
| **MVP-16A-0** | **🔥 Oracle MoE 结构红利巨大！ΔR²=+0.16 >> 0.03 阈值，所有 9 bins 均正向提升** | Oracle R²=0.6249, Global R²=0.4611, ΔR²=+0.1637 | ✅ |
| **MVP-16T V2** | **✅ 理论上限 R²_max=0.89，headroom +32% vs LightGBM，继续投入 CNN 值得** | R²_max=0.8914, Schur=0.6906, CRLB跨2.9数量级 | ✅ |
| **MVP-NN-0** | **✅ MLP 达到 Ridge baseline (R²=0.467)；❌ Whitening 预处理失败导致 R²≈0；CNN 弱于 MLP** | MLP_100k R²=0.4671, CNN_100k R²=0.4122, vs Oracle gap=-0.15 | ✅ |

## 4.3 Timeline

| Date | Event | Notes |
|------|-------|-------|
| 2025-12-22 | Phase 1 完成 | Ridge=0.50, LGB=0.57 |
| 2025-12-22 | Phase 1.x 立项 | MVP-1.3~1.9 规划完成 |
| 2025-12-22 | P0 exp.md 框架创建 | stats, ridge-alpha, lgbm-param |
| 2025-12-23 | MVP-1.4 完成 | 最优 α=1e4~1e5，H1.5.1 验证 ✅ |
| 2025-12-23 | MVP-1.6 完成 | H1.7.1 ❌, LightGBM 必须用 raw 输入 |
| **2025-12-24** | **MVP-16A-0 完成** | 🔥 Oracle MoE ΔR²=+0.16, H-A0.1 ✅, H4.1.1 ✅, H4.1.2 ✅ |
| **2025-12-24** | **MVP-16T V2 完成** | ✅ R²_max=0.8914, Schur=0.6906, H-16T.1 ✅, H-16T.2 ✅ |
| **2025-12-24** | **MVP-NN-0 完成** | ✅ MLP=0.467, CNN=0.412; ❌ Whitening 失败 |

---

# 5. 🔗 Cross-Repo Integration

## 5.1 Experiment Index

> **Links to experiments_index/index.csv**

| experiment_id | project | topic | status | MVP |
|---------------|---------|-------|--------|-----|
| `SCALING-20251222-ridge-1m-01` | VIT | scaling | ⏳ | MVP-1.0 |
| `SCALING-20251222-lgbm-1m-01` | VIT | scaling | ⏳ | MVP-1.1 |

## 5.2 Repository Links

| Repo | Directory | Purpose |
|------|-----------|---------|
| VIT | `~/VIT/` | 训练 NN 模型 |
| This repo | `logg/scaling/` | 知识沉淀 |

## 5.3 Data Paths

| Dataset | Path | Size |
|---------|------|------|
| mag205_225_lowT_1M | `/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/` | 93 GB |
| train_200k_0 | `.../train_200k_0/dataset.h5` | 19 GB |
| train_200k_1 | `.../train_200k_1/dataset.h5` | 19 GB |
| train_200k_2 | `.../train_200k_2/dataset.h5` | 19 GB |
| train_200k_3 | `.../train_200k_3/dataset.h5` | 19 GB |
| train_200k_4 | `.../train_200k_4/dataset.h5` | 19 GB |

---

# 6. 📎 Appendix

## 6.1 Results Summary

> **Core metrics from all MVPs (to be filled)**

### ML vs NN Performance @ noise=1, 1M data

| Model | Config | $R^2$ | MAE | RMSE | ΔR² vs Ridge |
|-------|--------|-------|-----|------|--------------|
| Ridge | α=1e+05 (optimal) | 0.5017 | 0.6345 | - | baseline |
| LightGBM | (best config) | 0.5709 | - | - | +0.07 |
| MLP | (best arch) | - | - | - | - |
| CNN | (best arch) | - | - | - | - |

### Ridge α Sweep Results (MVP-1.4)

| Data Size | Baseline α~3162 R² | Optimal α | Optimal R² | Improvement |
|-----------|---------------------|-----------|------------|-------------|
| 100k | 0.4735 | 3.16e+04 | 0.4856 | +2.55% |
| 1M | 0.4997 | 1.00e+05 | 0.5017 | +0.42% |

### Data Scaling Effect (待填充)

| Data Size | Ridge R² | LGB R² | MLP R² | CNN R² |
|-----------|----------|--------|--------|--------|
| 100k | - | - | - | - |
| 200k | - | - | - | - |
| 500k | - | - | - | - |
| 1M | - | - | - | - |

---

## 6.2 File Index

| Type | Path | Description |
|------|------|-------------|
| Roadmap | `logg/scaling/scaling_roadmap_20251222.md` | This file |
| Hub | `logg/scaling/scaling_hub_20251222.md` | Knowledge navigation |
| MVP-1.0 | `logg/scaling/exp/exp_scaling_ml_ceiling_20251222.md` | ML ceiling |
| Images | `logg/scaling/img/` | Experiment figures |

---

## 6.3 Changelog

| Date | Change | Sections |
|------|--------|----------|
| 2025-12-22 | Created Roadmap | - |
| 2025-12-22 | Phase 1.x 规划完成 | §1.1, §2.1, §3 |
| 2025-12-22 | MVP-1.3~1.9 添加 | §2.1, §3 (Phase 1.x) |
| 2025-12-22 | P0 exp.md 框架创建 | §4 |
| 2025-12-22 | MVP-1.6 Whitening 立项 | §2.1, §4 |
| 2025-12-23 | MVP-1.4 完成，结果填充 | §2.1, §4.1, §4.2, §4.3, §6.1 |
| 2025-12-23 | MVP-1.7 PCA vs PLS 立项 (3 sub-designs) | §2.1, §3 |
| **2025-12-23** | **🔴 Phase 16 完整大立项：三层论证（理论上限→模型ceiling→结构上限）** | §1.1, §2.1, §3, §4.1 |
| 2025-12-23 | 添加 MVP-16T/B/L/O/W/CNN 完整规格 | §2.1, §3 (Phase 16) |
| 2025-12-23 | 更新 Kanban：Phase 16 P0 三件套优先 | §4.1 |
| 2025-12-23 | **MVP-1.6 Whitening 完成**: H1.7.1 ❌ REJECTED, SNR ΔR²=+0.0146 (Ridge) | §2.1, §4 |
| 2025-12-23 | 添加参考文献：Fisher/CRLB, van Trees, Gaia XP | §3 (MVP-16T) |
| **2025-12-23** | **MVP-16T ✅ 完成：R²_max=0.9661, Schur=0.2366** | §2.1, §4.1, §6.1 |
| **2025-12-23** | **🆕 Phase T/A/NN 大立项** | §2.1, §3, §4.1 |
| 2025-12-23 | 添加 MVP-T0/T1/T2/T3 (Fisher 校准) | §2.1, §3 |
| 2025-12-23 | 添加 MVP-16A-0/A-1/A-2 (MoE @ noise=1) | §2.1, §3 |
| 2025-12-23 | 添加 MVP-NN-0 (1D CNN whiten) | §2.1, §3 |
| 2025-12-23 | 更新 P0 优先级和决策树 | §4.1 |
| **2025-12-23** | **❌ MVP-16T 失败：方法论缺陷（非规则网格）** | §2.1, §3, §4.1 |
| 2025-12-23 | 取消 MVP-T0, T1；降级 T2, T3 | §2.1, §3 |
| 2025-12-23 | 新增 Phase D + MVP-D0 (经验上限) | §2.1, §3 |
| 2025-12-23 | 更新 P0 为 D0 + 16A-0 + NN-0 三件套 | §4.1 |
| **2025-12-24** | **🔄 MVP-16T-V2 立项：使用规则网格数据 grid_mag215_lowT 重做 Fisher** | §2.1, §3 |
| **2025-12-24** | **✅ MVP-16T-V2 完成：R²_max=0.8914, Schur=0.6906, 结果可信** | §2.1, §4.2, §4.3 |
| **2025-12-25** | **✅ MVP-NN-0 完成：MLP=0.467≈Ridge, CNN=0.412; Whitening 失败** | §2.1, §4.1, §4.2, §4.3 |

---

> **Template Usage:**
> 
> **Roadmap Scope:**
> - ✅ **Do:** MVP specs, execution tracking, kanban, cross-repo integration, metrics
> - ❌ **Don't:** Hypothesis management (→ hub.md), insight synthesis (→ hub.md), strategy (→ hub.md)
> 
> **Hub vs Roadmap:**
> - Hub = "What do we know? Where should we go?"
> - Roadmap = "What experiments are planned? What's the progress?"

---

## 📊 SCALING-20251222-ml-ceiling-01 实验结果

### 核心结论
传统 ML（Ridge, LightGBM）在 1M 数据 + noise=1 下分别达到 R²=0.46 和 R²=0.57，确认性能天花板存在。

### 关键数字
| 指标 | 值 |
|------|-----|
| Ridge 最佳 R² (1M) | 0.4997 |
| LightGBM 最佳 R² (1M) | 0.5709 |
| Ridge ΔR² (1M vs 100k) | +0.0244 |
| LightGBM ΔR² (1M vs 100k) | +0.0176 |

### 设计启示
1. **数据量非瓶颈**：100k→1M 仅提升 2-3%，应投资模型改进
2. **深度学习目标**：突破 R²=0.70 才算有意义提升
3. **Baseline 确立**：LightGBM R²=0.57 可作为 NN 的 baseline

### MVP 状态更新
- ✅ MVP-1.0 (Ridge @ 1M): Done
- ✅ MVP-1.1 (LightGBM @ 1M): Done
- ✅ MVP-1.2 (Scaling Law): Done

---

## 📊 SCALING-20251222-ridge-alpha-01 实验结果 (MVP-1.4)

### 核心结论
Ridge 最优 α 在 1e4~1e5 之间，比原 baseline (α=5000) 高 1-2 个数量级。倒 U 型曲线明确存在。

### 关键数字
| 数据量 | 最优 α | 最优 R² | vs baseline |
|--------|--------|---------|-------------|
| 100k | 3.16e+04 | 0.4856 | +2.55% |
| 1M | 1.00e+05 | 0.5017 | +0.42% |

### H1.5.1 验证结果
**✅ CONFIRMED** - 观察到明确的倒 U 型曲线：
- 100k: 峰值后下降 0.4849
- 1M: 峰值后下降 0.4663

### 设计启示
1. **Ridge α 应该更大**：推荐 α ∈ [1e4, 1e5]
2. **α 与数据量正相关**：更多数据 → 更大的最优 α
3. **优化空间有限**：α 调优仅提升 0.4%~2.5%，说明 Ridge ceiling 确实存在

### MVP-1.4 状态
- ✅ MVP-1.4 (Ridge α Extended): **Done**
- 图表位置: `logg/scaling/img/scaling_ridge_alpha_extended.png`
- 报告位置: `logg/scaling/exp/exp_scaling_ridge_alpha_extended_20251222.md`

---

# 📊 Phase 16 更新 (2025-12-23)

## MVP-16T 完成 ✅

| Item | Result |
|------|--------|
| **Status** | ✅ Done |
| **experiment_id** | SCALING-20251223-fisher-ceiling-01 |
| **R²_max (median)** | **0.9661** |
| **R²_max (90%)** | 0.9995 |
| **Schur decay** | 0.2366 (76% 信息因 degeneracy 损失) |
| **Fisher 条件数** | 8.65×10⁵ |
| **Gap vs Ridge** | +0.47 |
| **Gap vs LightGBM** | +0.40 |

### 假设验证

| Hypothesis | Criteria | Result | Status |
|------------|----------|--------|--------|
| H-16T.1: R²_max ≥ 0.75 | ≥ 0.75 | 0.9661 | ✅ |
| H-16T.2: degeneracy 显著 | Schur < 0.9 | 0.2366 | ✅ |

### 核心结论

1. **理论上限极高**：R²_max ≈ 0.97 远超当前最佳 (0.57)，巨大 headroom 存在
2. **degeneracy 是主要瓶颈**：边缘化后仅保留 24% Fisher 信息，需要 multi-task 解纠缠
3. **继续投入 CNN/Transformer 有意义**：理论上限证明提升空间巨大

### 下一步

| Direction | Priority | MVP |
|-----------|----------|-----|
| 继续 CNN | 🔴 P0 | MVP-16CNN |
| Multi-task 解纠缠 | 🟡 P1 | 后续 |
| Bayesian CRLB (van Trees) | 🟢 P2 | 可选 |

---

# ❌ MVP-16T 失败更新 (2025-12-23)

## 状态变更

| 项目 | 原状态 | 新状态 |
|------|--------|--------|
| MVP-16T | ✅ Done | ❌ **Failed** |

## 失败根因

1. **数据不是规则网格**：BOSZ 为连续采样，T_eff/log_g/[M/H] 各有 ~40k 唯一值
2. **邻近点差分法失效**：在非规则网格上无法正确估计 ∂μ/∂θ
3. **数值异常**：CRLB 跨越 20 个数量级，R²_max 呈双峰分布

## 结果状态

| 指标 | 值 | 可靠性 |
|------|-----|--------|
| R²_max (median) | 0.9661 | ❌ **不可信** |
| Schur decay | 0.2366 | ❌ **不可信** |

## 下一步

- **暂停 MVP-16T**：等待方法论改进
- 考虑替代方案：
  - 方案 A：BOSZ 前向模型数值微分
  - 方案 B：局部多项式回归
  - 方案 D：经验上限（noise=0 Oracle）

---

## 2025-12-23 Update: MVP-16A-0 Completed

### SCALING-20251223-oracle-moe-noise1-01

| Metric | Result |
|--------|--------|
| Global Ridge R² | 0.4316 (α=10000, CV) |
| Oracle MoE R² | **0.5838** |
| ΔR² | **+0.1522** ✅ |
| Hypothesis H-A0.1 | ✅ PASS (ΔR² ≥ 0.03) |
| Hypothesis H4.1.1 | ✅ PASS (R² > 0.55) |

**Decision**: ✅ STRONG STRUCTURE BONUS confirmed at noise=1. MoE route continues!

**Next**: MVP-16A-1 (Trainable Gate with Physical Features)


### SCALING-20251223-oracle-moe-noise1-01 (1M Data - Final)

| Metric | Result |
|--------|--------|
| Train Size | **1,000,000** |
| Global Ridge R² | 0.4611 (α=100000) |
| Oracle MoE R² | **0.6249** |
| ΔR² | **+0.1637** ✅ |
| Hypothesis H-A0.1 | ✅ PASS (ΔR² >> 0.03) |
| Hypothesis H4.1.1 | ✅ PASS (R² > 0.55) |
| All 9 bins positive ΔR² | ✅ YES |

**Decision**: ✅ STRONG STRUCTURE BONUS confirmed at noise=1, 1M scale.

| 2025-12-23 | 更新 P0 优先级和决策树 | §4.1 |
| **2025-12-23** | **❌ MVP-16T 失败：方法论缺陷（非规则网格）** | §2.1, §3, §4.1 |
| **2025-12-23** | **✅ MVP-16A-0 完成：Oracle MoE ΔR²=+0.16 >> 0.03** | §2.1, MVP-1.8 |
| **2025-12-23** | **🔄 Ridge 基准修正：1k test → R²=0.46 (原 500 test R²=0.50)** | 全文 |
| **2025-12-24** | **✅ Ridge Alpha Sweep (1k test): Best α=100k, R²=0.4551** | §2.1, MVP-1.0 |
| **2025-12-24** | **✅ Y-Scaling 实验: MinMaxScaler 对 R² 无影响** | §2.1, MVP-1.0 |

---

## 2025-12-24 Update: MVP-16A-1 Completed

### SCALING-20251223-gate-feat-01

| Metric | Result |
|--------|--------|
| Gate 9-class Accuracy | **87.8%** ✅ (>> 60% threshold) |
| F1 (macro) | 88.2% |
| Ca II F-statistic | **25,618** ✅ (>> 10 threshold) |
| Top F-statistic (PCA_1) | 287,966 |
| Avg SNR @ noise=1 | **6.21** ✅ (>> 1.0 threshold) |
| Total Gate Features | 37 (27 physical + 10 PCA) |

**Noise Sweep Results**:

| noise_level | accuracy | f1_macro |
|-------------|----------|----------|
| 0.0 | 98.3% | 98.3% |
| 0.2 | 96.8% | 96.7% |
| 0.5 | 92.5% | 92.7% |
| **1.0** | **88.3%** | **88.7%** |
| 2.0 | 75.1% | 76.0% |

**Hypothesis Verification**:

| Hypothesis | Criteria | Result | Status |
|------------|----------|--------|--------|
| H-A1.1 (Accuracy) | > 60% | 87.8% | ✅ PASS |
| H-A1.1 (F-stat) | > 10 | 25,618 | ✅ PASS |
| SNR threshold | > 1.0 | 6.21 | ✅ PASS |

**🔥 Decision**: ✅ GATE FEATURES USABLE @ noise=1 - Continue to MVP-16A-2 (Soft-gate MoE)

**Key Insight**: This was expected to be a "sanity check failure" showing gate features collapse at noise=1, but the result is surprisingly positive! Physical window features remain highly discriminative even under high noise conditions.

**Top 5 Most Discriminative Features** (by F-statistic):
1. PCA_1: 287,966 (global spectral shape)
2. PCA_3: 103,485
3. MgI_8806_mean: 83,547
4. MgI_8807_mean: 80,703
5. CaII_8542_mean: 71,738

**Report**: [exp_scaling_gate_feat_sanity_20251224.md](./exp/exp_scaling_gate_feat_sanity_20251224.md)

---

# 📊 MVP-16T V2 完成 (2025-12-24)

## 状态变更

| 项目 | V1 状态 | V2 状态 |
|------|---------|---------|
| MVP-16T | ❌ Failed | ✅ **Done** |

## V2 核心结果

| 指标 | V1 (异常) | V2 (可信) |
|------|----------|----------|
| **R²_max (median)** | 0.97 ⚠️ | **0.8914** ✅ |
| CRLB range (orders) | 20 | **2.9** ✅ |
| Condition number max | 5e+16 | 3.78e+06 ✅ |
| Schur decay | 0.24 ⚠️ | 0.6906 ✅ |

## 假设验证

| Hypothesis | Criteria | Result | Status |
|------------|----------|--------|--------|
| H-16T.1 (V2) | R²_max ≥ 0.75 | 0.8914 | ✅ |
| H-16T.2 (V2) | Schur decay < 0.9 | 0.6906 | ✅ |

## 核心结论

1. **理论上限高**：R²_max ≈ 0.89，继续投入 CNN/Transformer 值得
2. **Headroom 大**：当前 0.57 vs 理论 0.89，有 +32% 提升空间
3. **Degeneracy 中等**：Schur decay = 0.69，边缘化后保留 69% 信息

## 下一步

| 方向 | 优先级 | 说明 |
|------|--------|------|
| 继续 CNN | 🔴 P0 | 理论上限高，值得投入 |
| Multi-task | 🟡 P1 | Schur decay = 0.69，可能有帮助 |

---

### SCALING-20251224-nn-baseline-framework-01 Result (2025-12-24)

| Model | Train Size | Input | Test R² | vs Oracle (0.62) |
|-------|------------|-------|---------|------------------|
| MLP 3L_1024 | 100k | flux_only | **0.4671** | -0.153 |
| CNN 4L_k5_bn | 100k | flux_only | 0.4122 | -0.208 |
| CNN 4L_k5_wide | 1M | whitening | 0.4337 | -0.186 |

**Key Findings:**
- MLP matches Ridge baseline (✅ H-NN0.1 validated)
- CNN underperforms MLP by ~0.05 R²
- Whitening preprocessing fails (causes training collapse)
- Gap to Oracle MoE: **0.15-0.19 R²**

**Next Steps:**
- Fix MLP 1M (use flux_only instead of whitening)
- CNN needs better hyperparams (lr, warmup, bn required)
- Consider MoE-CNN if single-model CNN plateaus
