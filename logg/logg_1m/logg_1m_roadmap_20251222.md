# 🗺️ Experiment Roadmap: logg 1M Breakthrough

> **Topic:** logg Inference Breakthrough with 1M BOSZ→PFS Dataset  
> **Author:** Viska Wei  
> **Created:** 2025-12-22 | **Updated:** 2025-12-22  
> **Current Phase:** Phase 0 - Foundation

## 🔗 Related Files

| Type | File | Description |
|------|------|-------------|
| 🧠 Hub | [`logg_1m_hub_20251222.md`](./logg_1m_hub_20251222.md) | Knowledge navigation |
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
| **Phase 0: Foundation** | 建立可复现实验条件 + 上下限 | MVP-0.A, 0.B | ⏳ Planned | Low-noise 定义 + Baseline bounds |
| **Phase 1: Quick Wins** | 最可能立刻提升 logg 的 5 个方向 | MVP-1.1~1.5 | ⏳ Planned | 方向判定 + 初步提升 |
| **Phase 2: Breakthrough** | 结构性突破（预训练/多尺度） | MVP-2.1~2.3 | ⏳ Planned | 突破监督上限 |
| **Phase 3: Long-term** | 稳健性 + 未来真实数据适配 | MVP-3.1~3.6 | ⏳ Planned | 泛化保障 |

## 1.2 Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                       MVP Experiment Dependencies                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌───────────────────────────────────────────────────┐                            │
│   │           Phase 0: Foundation (必须先做)            │                            │
│   │   MVP-0.A (Low-noise 定义)    MVP-0.B (Baseline)   │                            │
│   └───────────────────────────────────────────────────┘                            │
│                            │                                                        │
│                            ▼                                                        │
│   ┌───────────────────────────────────────────────────────────────────────────┐    │
│   │                    Phase 1: Quick Wins (可并行)                             │    │
│   │                                                                            │    │
│   │   MVP-1.1         MVP-1.2         MVP-1.3         MVP-1.4        MVP-1.5  │    │
│   │   Fisher上限      Error输入       归一化对照       敏感窗口       多任务联合│    │
│   │      │               │               │               │              │     │    │
│   └──────┼───────────────┼───────────────┼───────────────┼──────────────┼─────┘    │
│          │               │               │               │              │          │
│          ▼               ▼               ▼               ▼              ▼          │
│   ┌─── Decision D1: 哪些方向有效？→ 决定 Phase 2 优先级 ─────────────────────┐      │
│   │                                                                          │      │
│   │   if MVP-1.1 显示大差距 → 继续 Phase 1-2                                 │      │
│   │   if MVP-1.1 显示接近上限 → 转向物理先验/多臂/真实数据                    │      │
│   │   if MVP-1.4 显示窗口有效 → Phase 2 优先 MVP-2.3 (多尺度)                 │      │
│   │   if MVP-1.5 显示多任务有效 → 后续实验默认用多任务                         │      │
│   │                                                                          │      │
│   └──────────────────────────────────────────────────────────────────────────┘      │
│                            │                                                        │
│                            ▼                                                        │
│   ┌───────────────────────────────────────────────────────────────────────────┐    │
│   │                  Phase 2: Structural Breakthrough                          │    │
│   │                                                                            │    │
│   │   MVP-2.1               MVP-2.2                MVP-2.3                    │    │
│   │   MSM 预训练            去噪预训练              多尺度 Token               │    │
│   │                                                                            │    │
│   └───────────────────────────────────────────────────────────────────────────┘    │
│                            │                                                        │
│                            ▼                                                        │
│   ┌───────────────────────────────────────────────────────────────────────────┐    │
│   │                  Phase 3: Long-term Directions                             │    │
│   │                                                                            │    │
│   │   MVP-3.1     MVP-3.2     MVP-3.3     MVP-3.4     MVP-3.5     MVP-3.6     │    │
│   │   λ PE        异方差      Error Mask  导数通道    窗口Attn    序回归       │    │
│   │                                                                            │    │
│   └───────────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 1.3 Decision Points

> **Key decision points based on experiment results**

| Point | Trigger | Option A | Option B |
|-------|---------|----------|----------|
| **D1** | After MVP-1.1 | If 模型误差 > 2× Fisher σ → 继续 Phase 1-2 | If 接近上限 → 转向物理先验/多臂 |
| **D2** | After MVP-1.4 | If 窗口 R² ≥ 全谱 → Phase 2 先做 MVP-2.3 | If 全谱更好 → 跳过窗口相关实验 |
| **D3** | After MVP-1.5 | If 多任务有效 → 后续默认多任务 | If 无效 → 保持单任务 |
| **D4** | After Phase 1 | 选择 2-3 个最有效方向进入 Phase 2 | 无效方向标记为 Closed |

---

# 2. 📋 MVP List

> **Overview of all MVPs for quick lookup and tracking**

## 2.1 Experiment Summary

| MVP | Name | Phase | Status | experiment_id | Report |
|-----|------|-------|--------|---------------|--------|
| **MVP-0.A** | Low-noise Protocol | 0 | ⏳ | - | - |
| **MVP-0.B** | Baseline Bounds (Scaling) | 0 | 🔴 Ready | `VIT-20251222-logg_1m-baseline-scaling-01` | [exp_logg_1m_baseline_scaling](./exp/exp_logg_1m_baseline_scaling_20251222.md) |
| **MVP-1.1** | Fisher/Sensitivity Upper Bound | 1 | ⏳ | - | - |
| **MVP-1.2** | SNR/Error as Input | 1 | ⏳ | - | - |
| **MVP-1.3** | Normalization Comparison | 1 | ⏳ | - | - |
| **MVP-1.4** | Sensitive Window vs Full Spectrum | 1 | ⏳ | - | - |
| **MVP-1.5** | Multi-task (Teff+FeH+logg) | 1 | ⏳ | - | - |
| **MVP-2.1** | Masked Spectrum Modeling Pretraining | 2 | ⏳ | - | - |
| **MVP-2.2** | Denoising Pretraining → logg | 2 | ⏳ | - | - |
| **MVP-2.3** | Multi-scale Token | 2 | ⏳ | - | - |
| **MVP-3.1** | λ/log λ Position Encoding | 3 | ⏳ | - | - |
| **MVP-3.2** | Heteroscedastic Regression | 3 | ⏳ | - | - |
| **MVP-3.3** | Error-based Masking | 3 | ⏳ | - | - |
| **MVP-3.4** | Derivative/High-pass Channel | 3 | ⏳ | - | - |
| **MVP-3.5** | Window Attention Bias | 3 | ⏳ | - | - |
| **MVP-3.6** | Ordinal Regression | 3 | ⏳ | - | - |

**Status Legend:**
- ⏳ Planned | 🔴 Ready | 🚀 Running | ✅ Done | ❌ Cancelled | ⏸️ Paused

## 2.2 Configuration Reference

> **Key configurations across all MVPs**

| MVP | Data Size | Model | Key Variable | Acceptance |
|-----|-----------|-------|--------------|------------|
| MVP-0.A | 1M (筛选) | - | SNR 分位数定义 | 可复现协议 |
| MVP-0.B | 5k-50k | Ridge/LGB/ViT/Window | - | 上下限建立 |
| MVP-1.1 | 5k-20k | 分析方法 | Fisher σ vs 模型误差 | 比值 > 2 |
| MVP-1.2 | 50k-200k | ViT | 输入通道 (flux/SNR/error) | ΔMAE ≥ 5% |
| MVP-1.3 | 50k-200k | ViT | 归一化方式 | R² 提升 |
| MVP-1.4 | 50k-200k | 小模型 | 窗口 vs 全谱 | R²(window) ≥ R²(full) |
| MVP-1.5 | 50k-200k | ViT | 单任务 vs 多任务 | MAE↓ + bias↓ |
| MVP-2.1 | 50k-200k | ViT+MSM head | 预训练 vs 直接监督 | 收敛快 + R²↑ |
| MVP-2.2 | 50k-200k | Denoiser+ViT | 去噪预训练 vs 直接 | 天空线区 MAE↓ |
| MVP-2.3 | 50k-200k | ViT | patch_size (25+200) | R²↑ + 稳健性↑ |
| MVP-3.1 | 50k-200k | ViT | PE (learnable vs log λ) | 不退化即可 |
| MVP-3.2 | 50k-200k | ViT | MSE vs NLL | outlier↓ |
| MVP-3.3 | 50k-200k | ViT | mask top 1/5/10% error | R²↑ |
| MVP-3.4 | 50k-200k | ViT | 1ch vs 2ch (flux+deriv) | R²↑ + 串扰↓ |
| MVP-3.5 | 50k-200k | ViT | window attn bias | R²↑ |
| MVP-3.6 | 50k-200k | ViT | 回归 vs 分箱分类+微调 | outlier↓ |

---

# 3. 🔧 MVP Specifications

> **Detailed specs for each MVP, ready for execution**

## Phase 0: Foundation

### MVP-0.A: Low-noise Protocol Definition

| Item | Config |
|------|--------|
| **Objective** | 定义可复现的"low-noise"实验条件 |
| **Hypothesis** | 不落实定义，后续实验结论会互相打架 |
| **Data** | 1M 全量数据 |
| **Method** | 计算每条谱 SNR = \|\|flux\|\| / \|\|error\|\| |
| **Output** | Low-noise 定义 = Top 20% SNR（或 Top 10%） |
| **Acceptance** | 有明确的筛选协议，test 分为 low-noise test + 全分布 test |

**Steps:**
1. 计算 1M 条谱的标量 SNR
2. 确定 SNR 分布的分位数（10%, 20%, 50%）
3. 固定协议：Top 20% SNR = low-noise
4. 后续所有实验在 low-noise test 上报主指标

---

### MVP-0.B: Baseline Bounds - Data Scaling at noise=1.0 🆕

| Item | Config |
|------|--------|
| **Objective** | 验证传统 ML (Ridge/LightGBM) 在 noise=1.0 下的 scaling 规律 |
| **Hypothesis** | 传统 ML 可能在大规模数据下存在性能天花板 |
| **Data** | 10k → 32k → 100k → 500k → 1M 逐步扩展 |
| **Models** | Ridge (PCA 降维) + LightGBM |
| **Noise** | noise_level = 1.0 (高噪声) |
| **Acceptance** | 得到完整 scaling curve，识别饱和点 |

**Scaling 配置:**

| 数据规模 | 训练样本 | 测试样本 |
|---------|---------|---------|
| 10k | 8,000 | 2,000 |
| 32k | 25,600 | 6,400 |
| 100k | 80,000 | 20,000 |
| 500k | 400,000 | 100,000 |
| 1M | 800,000 | 200,000 |

**Ridge 配置:**
- alpha: [0.1, 1.0, 10.0, 100.0]
- 特征: 全谱 / PCA-100 / PCA-200 / PCA-500

**LightGBM 配置:**
- n_estimators: 2500
- learning_rate: 0.05
- num_leaves: 31

---

## Phase 1: Quick Wins

### MVP-1.1: Fisher/Sensitivity Upper Bound

| Item | Config |
|------|--------|
| **Objective** | 确认模型误差距离 Fisher 理论上限有多远 |
| **Hypothesis** | H1.1: 模型误差 ≫ Fisher σ → 还有巨大提升空间 |
| **Data** | low-noise 子集 5k-20k 条 |
| **Method** | kNN (label-space) 找 Teff/FeH 近似但 logg 不同的近邻 → 有限差分估 ∂F/∂logg → Fisher → σ_logg_theory |
| **Acceptance** | 模型误差 / 理论σ > 2 → 继续优化; ≈ 1 → 转方向 |

**Steps:**
1. 在 low-noise 子集抽 5k-20k 条
2. 对每条谱，用 kNN 在 label-space 找近邻（Teff/FeH 接近，logg 有差）
3. 用有限差分估计 ∂F/∂logg
4. 用 error 估计 Fisher information → σ_logg_theory 分布
5. 对比模型误差分布 vs 理论 σ 分布

**→ Hypothesis Impact:** 
- If 模型误差 ≈ 理论σ → 别死磕模型，转去换波段/加物理先验/改任务定义
- If 模型误差 ≫ 理论σ → 继续优化表示学习/损失/归一化

---

### MVP-1.2: SNR/Error as Input

| Item | Config |
|------|--------|
| **Objective** | 让模型知道哪些像素可信 |
| **Hypothesis** | H1.2: logg 信号集中在窄特征，不知 error 会被局部高噪声污染 |
| **Data** | 50k-200k 样本 |
| **Model** | ViT (最小改动) |
| **Variants** | A: flux (现状) <br> B: flux / (error + eps) (SNR 谱) <br> C: concat(flux, log(error)) (2 channel) |
| **Acceptance** | low-noise logg MAE 下降 ≥ 5% |

**Steps:**
1. 准备三个数据版本（A/B/C）
2. 用同一 ViT 配置训练 3 次
3. 在 low-noise test 上对比 logg MAE/RMSE
4. 可选：可视化注意力/梯度在天空线区域的变化

---

### MVP-1.3: Normalization Comparison

| Item | Config |
|------|--------|
| **Objective** | 归一化方式是否压扁了 logg 线翼信号 |
| **Hypothesis** | H1.3: median norm 可能压扁局部对比 |
| **Data** | 50k-200k 样本 |
| **Model** | ViT (固定配置) |
| **Variants** | A: 全谱 median norm (现状) <br> B: 分块 z-score (chunk normalization) <br> C: 连续谱拟合后除去（低阶多项式/robust spline）→ 残差谱 |
| **Acceptance** | low-noise logg R² 提升或 MAE 下降 |

**Steps:**
1. 实现三种归一化预处理
2. 只改数据，模型配置不动
3. 训练 3 次对比
4. 可选：可视化模型对线区 vs 连续谱的关注差异

---

### MVP-1.4: Sensitive Window vs Full Spectrum

| Item | Config |
|------|--------|
| **Objective** | 验证"无关波段干扰"假设 |
| **Hypothesis** | H2.1: 全谱大量区域对 logg 是噪声，只喂敏感窗口反而更准 |
| **Data** | 50k-200k 样本 |
| **Model** | 同一小模型 (轻量 ViT 或 MLP) |
| **Window Selection** | 用 MVP-1.1 的 ∂F/∂logg 聚合 top-K 窗口 <br> 或用常识选：Ca II triplet (8498/8542/8662Å), H-α, 强分子带 |
| **Variants** | A: 只保留窗口（其他置零/mask） <br> B: 全谱 |
| **Acceptance** | R²(A) ≥ R²(B) |

**Steps:**
1. 确定敏感窗口（来自 MVP-1.1 或先验知识）
2. 构造两个数据版本（窗口/全谱）
3. 用同一小模型训练 3 次
4. 对比 logg R²

**→ Hypothesis Impact:**
- If A ≥ B → 下一步做自适应 token/窗口注意力 bias (Phase 2)
- If A < B → 暂时不做窗口相关优化

---

### MVP-1.5: Multi-task (Teff+FeH+logg)

| Item | Config |
|------|--------|
| **Objective** | logg 与 Teff/FeH 的耦合是否是精度瓶颈 |
| **Hypothesis** | H4.1: 单任务回归会把变化误归因，多任务可逼表示拆分因素 |
| **Data** | 50k-200k 样本 |
| **Model** | ViT + 3-head output (Teff/logg/FeH) |
| **Loss** | $w_1 \cdot MSE(logg) + w_2 \cdot MSE(Teff) + w_3 \cdot MSE(FeH)$ <br> 权重设成梯度量级接近 |
| **Acceptance** | low-noise logg MAE 下降 + logg 随 Teff/FeH 的系统偏差减小 |

**Steps:**
1. 改 head 输出 3 个标量
2. 设计 loss 权重（按梯度量级平衡）
3. 训练对比单任务 vs 多任务
4. 检查 logg 的 bias 图（logg_pred - logg_true vs Teff/FeH）

---

## Phase 2: Structural Breakthrough

### MVP-2.1: Masked Spectrum Modeling Pretraining

| Item | Config |
|------|--------|
| **Objective** | 自监督预训练是否能突破监督信号稀疏的限制 |
| **Hypothesis** | H3.1: logg 信号很"细"，MSM 能让 encoder 学到更稳的谱结构表征 |
| **Data** | 50k-200k（预训练可以用更多） |
| **Model** | ViT + MSM head (预训练) → logg head (微调) |
| **Pretraining** | 随机 mask 15-30% token/像素，让模型重建（L1/L2） |
| **Acceptance** | 同等监督训练预算下，logg R² 更高且收敛更快 |

**Steps:**
1. 加预训练阶段：mask + reconstruct
2. 预训练只跑少量 epoch（验证方向）
3. 加载权重做 logg 回归微调
4. 对比从头训练 vs 预训练后微调

---

### MVP-2.2: Denoising Pretraining → logg

| Item | Config |
|------|--------|
| **Objective** | 去噪预训练是否能减少天空线/系统误差干扰 |
| **Hypothesis** | H3.2: 即使 low-noise，结构化噪声仍妨碍线翼测量；先去噪能提升 logg |
| **Data** | 50k-200k |
| **Model** | Denoiser (blindspot/AE) + ViT |
| **Method** | 训练去噪器：输入 noisy flux → 输出 denoised flux <br> 冻结去噪器，把 denoised flux 喂给 logg 模型 |
| **Acceptance** | logg 提升 + 提升集中在天空线区域 |

**Steps:**
1. 训练去噪器（用 error 做加权 loss）
2. 冻结去噪器
3. 用 denoised flux 训练 logg 模型
4. 对比 raw flux vs denoised flux

---

### MVP-2.3: Multi-scale Token

| Item | Config |
|------|--------|
| **Objective** | 多尺度 token 是否兼顾线翼细节和上下文 |
| **Hypothesis** | H2.2: logg 依赖线翼细节（需小 patch）+ 宽上下文（需大 patch），单一 patch_size 两头不讨好 |
| **Data** | 50k-200k |
| **Model** | ViT with dual patch embedding |
| **Patch Config** | 小 patch: 25-50 像素（捕线翼） <br> 大 patch: 200-500 像素（上下文） |
| **Method** | 两套 patch embedding，拼接 token 序列后送同一 encoder |
| **Acceptance** | logg R² 提升 + 对窗口裁剪更稳 |

---

## Phase 3: Long-term Directions

### MVP-3.1: λ/log λ Position Encoding

| Item | Config |
|------|--------|
| **Objective** | 物理波长位置编码是否比 learnable PE 更稳 |
| **Hypothesis** | learnable index PE 对重采样/裁剪不稳；物理 λ PE 更符合谱结构 |
| **Data** | 50k-200k |
| **Model** | ViT |
| **Variants** | A: learnable index PE (现状) <br> B: sinusoidal PE with log λ <br> C: RoPE with log λ |
| **Acceptance** | 不退化即可；如果略升，说明"稳健性+小收益" |

---

### MVP-3.2: Heteroscedastic Regression

| Item | Config |
|------|--------|
| **Objective** | 输出不确定度是否能处理可辨识性差异 |
| **Hypothesis** | H4.2: 某些参数区域对 logg 不可辨识；学不确定度避免硬拟合噪声 |
| **Data** | 50k-200k |
| **Model** | ViT → output (μ, log σ²) |
| **Loss** | NLL (高斯): $\log \sigma + \frac{(y-\mu)^2}{2\sigma^2}$ |
| **Acceptance** | logg MAE 下降 或 outlier (\|Δlogg\|>0.5) 比例显著下降 |

---

### MVP-3.3: Error-based Masking

| Item | Config |
|------|--------|
| **Objective** | mask 最差像素是否能提升 logg |
| **Hypothesis** | 最脏的像素区域对 logg 是负贡献 |
| **Data** | 50k-200k (low-noise) |
| **Method** | 统计每个波长的 median(error)，mask top 1/5/10% |
| **Acceptance** | mask 少量点就能提升 logg → 做更系统的物理窗口化 |

---

### MVP-3.4: Derivative/High-pass Channel

| Item | Config |
|------|--------|
| **Objective** | 导数/高通能否放大 logg 的线翼信号 |
| **Hypothesis** | logg 信号更像"形状变化"而非绝对通量 |
| **Data** | 50k-200k |
| **Model** | ViT with 2-channel input |
| **Channels** | A: flux <br> B: flux - smooth(flux) 或 d flux / d log λ |
| **Acceptance** | logg R² 提升 + Teff/FeH 串扰下降 |

---

### MVP-3.5: Window Attention Bias

| Item | Config |
|------|--------|
| **Objective** | 软窗口引导（不是硬裁剪）是否能提升 logg |
| **Hypothesis** | 硬裁剪可能丢失上下文；软 bias 保留全谱但引导注意力 |
| **Data** | 50k-200k |
| **Model** | ViT |
| **Method** | 训练时：鼓励 attention map 在敏感窗口有更高 mass <br> 或：给窗口内 token 更高 loss 权重 |
| **Acceptance** | logg 提升 + 不伤害其他参数 |

---

### MVP-3.6: Ordinal Regression

| Item | Config |
|------|--------|
| **Objective** | 序回归/分箱分类是否比纯 MSE 更稳 |
| **Hypothesis** | logg 本质更像"有序等级"；先学排序/分箱更稳 |
| **Data** | 50k-200k |
| **Model** | ViT |
| **Method** | 1. logg 分箱分类（0.25 dex 一箱），CE loss <br> 2. CLS token → 回归 head 微调 |
| **Acceptance** | 回归误差下降 + outlier 率下降 |

---

# 4. 📊 Progress Tracking

## 4.1 Kanban View

```
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│  ⏳ Planned  │   🔴 Ready   │  🚀 Running  │    ✅ Done   │  ❌ Cancelled │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ MVP-0.A      │              │              │              │              │
│ MVP-0.B      │              │              │              │              │
│ MVP-1.1      │              │              │              │              │
│ MVP-1.2      │              │              │              │              │
│ MVP-1.3      │              │              │              │              │
│ MVP-1.4      │              │              │              │              │
│ MVP-1.5      │              │              │              │              │
│ MVP-2.1      │              │              │              │              │
│ MVP-2.2      │              │              │              │              │
│ MVP-2.3      │              │              │              │              │
│ MVP-3.1~3.6  │              │              │              │              │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

## 4.2 Key Conclusions Snapshot

> **One-line conclusion per completed MVP, synced to Hub**

| MVP | Conclusion | Key Metric | Synced to Hub |
|-----|------------|------------|---------------|
| - | [待实验] | - | - |

## 4.3 Timeline

| Date | Event | Notes |
|------|-------|-------|
| 2025-12-22 | 立项 | 创建 hub + roadmap |
| - | MVP-0.A start | - |
| - | MVP-0.B start | - |

---

# 5. 🔗 Cross-Repo Integration

## 5.1 Experiment Index

> **Links to experiments_index/index.csv**

| experiment_id | project | topic | status | MVP |
|---------------|---------|-------|--------|-----|
| [待分配] | VIT | logg_1m | ⏳ | MVP-0.A |

## 5.2 Repository Links

| Repo | Directory | Purpose |
|------|-----------|---------|
| VIT | `~/VIT/configs/logg_1m/` | Training configs |
| VIT | `~/VIT/data/` | mag205_225_lowT_1M |
| This repo | `logg/logg_1m/` | Knowledge base |

## 5.3 Run Path Records

> **Actual run paths for reproducibility**

| MVP | Repo | Script | Config | Output |
|-----|------|--------|--------|--------|
| - | - | - | - | - |

---

# 6. 📎 Appendix

## 6.1 Results Summary

> **Core metrics from all MVPs (待实验后填充)**

### Main Metrics Comparison

| MVP | Config | logg R² | logg MAE | logg RMSE | ΔR² vs Baseline |
|-----|--------|---------|----------|-----------|-----------------|
| MVP-0.B (下限) | Ridge/LGB | - | - | - | - |
| MVP-0.B (现状) | ViT | - | - | - | baseline |
| MVP-0.B (上限) | Window-only | - | - | - | - |

---

## 6.2 File Index

| Type | Path | Description |
|------|------|-------------|
| Roadmap | `logg/logg_1m/logg_1m_roadmap_20251222.md` | This file |
| Hub | `logg/logg_1m/logg_1m_hub_20251222.md` | Knowledge navigation |
| Experiments | `logg/logg_1m/exp/*.md` | Detailed reports |
| Images | `logg/logg_1m/img/` | Experiment figures |

---

## 6.3 Changelog

| Date | Change | Sections |
|------|--------|----------|
| 2025-12-22 | Created Roadmap with 16 MVPs across 4 phases | All |

---

> **Template Usage:**
> 
> **Roadmap Scope:**
> - ✅ **Do:** MVP specs, execution tracking, kanban, cross-repo integration, metrics
> - ❌ **Don't:** Hypothesis management (→ hub.md), insight synthesis (→ hub.md), strategy (→ hub.md)
> 
> **Update Triggers:**
> - Planning new MVP → update §2, §3
> - MVP status change → update §4
> - After experiment → record conclusion to §4.2, sync to Hub
> 
> **Hub vs Roadmap:**
> - Hub = "What do we know? Where should we go?"
> - Roadmap = "What experiments are planned? What's the progress?"

