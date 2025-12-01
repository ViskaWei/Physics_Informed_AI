# 📘 Neural Network 架构设计实验报告

---
> **实验名称：** Physics-Informed Neural Network Architecture Design for $\log g$ Prediction  
> **对应 MVP：** NN 架构系统性设计（多阶段）  
> **作者：** Viska Wei  
> **日期：** 2025-11-29  
> **数据版本：** HDF5 光谱数据（4096 像素合成光谱）  
> **模型版本：** Phase 1 - Baseline MLP/CNN (PyTorch)  
> **状态：** 🔄 进行中

---

# ⚡ 核心结论速览（供 main 提取）

### 一句话总结

> **综合前期实验，NN 设计核心原则：(1) 线性 shortcut 必需，(2) 噪声增强训练 >> 显式特征选择，(3) 双通道 [flux, σ] 输入，(4) bottleneck ≥ 100 维。**

### 对假设的验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| 线性成分是否主要？ | ✅ 是，noise=0 时 $R^2=0.999$ | 线性 shortcut 必需 |
| 噪声训练是否关键？ | ✅ 是，效果是 TopK 的 12 倍 | Noise augmentation 核心 |
| Error 通道是否有用？ | ✅ 是，单独 $R^2=0.91$ | 双通道输入必须 |
| 最小 bottleneck 维度？ | ✅ 100+，基于 PCA 实验 | 不能过度降维 |

### 设计启示（1-2 条）

| 启示 | 具体建议 |
|------|---------|
| **架构设计原则** | Linear shortcut + Learnable attention + [flux, σ] 双通道 |
| **训练策略** | 噪声增强训练 + 与噪声挂钩的 weight decay |

### 关键数字

| 指标 | 值 |
|------|-----|
| 线性 $R^2$ (noise=0) | **0.999** |
| 最小有效 K | **~50 (tokens)** |
| Error-only $R^2$ | **0.91** |
| 最小 bottleneck | **100 维** |

---

# 📑 目录

- [1. 🎯 目标](#1--目标)
- [2. 🧪 实验设计](#2--实验设计experiment-design)
- [3. 📊 实验图表](#3--实验图表)
- [4. 💡 关键洞见](#4--关键洞见key-insights)
- [5. 📝 结论](#5--结论conclusion)
- [6. 📎 附录](#6--附录)

---

# 1. 🎯 目标

## 1.1 背景与动机

当前我们已经用 Ridge Regression 和 LightGBM 系统性地分析了光谱到 $\log g$ 的映射关系，得到了一些比较清晰的结论。本实验旨在：

> **从简单到复杂地搭建神经网络（MLP → CNN → Vision Transformer），验证并利用这些线性 / 特征选择 / 噪声相关的结论，把它们转化成可工作的 NN 架构设计原则。**

更具体地说：
1. 以 **MLP / 1D CNN** 为起点，在相同训练集、噪声设定和评估指标下，系统比较它们与 Ridge / LightGBM 的表现
2. 分离并量化：
   - "**线性成分**"：可以直接由线性层 / 初始化继承自 Ridge
   - "**非线性修正成分**"：MLP/CNN 只在局部窗口、Top-K 区域做小的非线性校正
3. 探索 **输入结构**（全谱 vs Top-K vs Top-K+局部窗口）、**噪声水平** 与 **NN 架构选择**（深度、宽度、卷积感受野）的关系，为后续 Vision Transformer 的 token 设计/注意力稀疏化提供依据

### 🔬 前期实验的核心发现（支撑 NN 设计的 Insights）

| 来源实验 | 核心发现 | 对 NN 设计的启示 |
|---------|---------|-----------------|
| **Ridge α Sweep** | $\log g$-flux 映射本质线性 (noise=0 时 $R^2=0.999$) | 线性 shortcut 是必须的 |
| **Ridge α Sweep** | NN 主要任务是"忽略无关像素"，而非"提取信息" | 需要 Attention/Sparse/Denoising 机制 |
| **Ridge α Sweep** | 最优 $\alpha$ 随噪声单调增大 (跨越 6 个数量级) | Weight decay 需与噪声挂钩 |
| **PCA 实验** | $\log g$ 信息分散在 PC 20-200 的低方差方向 | Bottleneck ≥ 100 维 |
| **PCA 实验** | 需要 100+ PC 才能达到 $R^2 \geq 0.99$ | 不能过度降维 |
| **PCA 实验** | 前 5-10 PC 仅保留 67% 信息 | 不能简单用 PCA 预处理 |
| **Feature Stability** | N=0 是"孤岛"，与其他噪声完全不相关 | 避免在无噪声数据上训练 |
| **Feature Stability** | 高噪声下特征重要性高度稳定 | 高噪声训练可能更鲁棒 |
| **Top-K 实验** | 信息高度稀疏，~24% 像素 (K=1000) 即可 | Learnable soft mask / attention |
| **Top-K 实验** | 高噪声训练效果是 Top-K 的 12 倍 | Noise augmentation 是核心策略 |
| **Small K Limit** | 关键谱线：Ca II 8542, Mg I 8807, Na I 8183 | Patch 聚焦特定波长区域 |
| **Small K Limit** | K=50 达 $R^2=0.39$，K=100 达 $R^2=0.49$ | 最小 Token 数 ~50 |
| **Small K Limit** | 非线性组合必要：LGBM >> Ridge ($\Delta R^2 \approx 0.16$) | 需要非线性层 |
| **Error 实验** | Error $\sigma$ 单独可达 $R^2=0.91$ | 必须使用 [flux, $\sigma$] 双通道输入 |
| **Error 实验** | Error-$\log g$ 关系是非线性的 | 需要非线性层提取 error 信息 |
| **LGBM 实验** | 树模型自带特征选择，Top-K 边际收益小 | Attention 替代显式特征选择 |
| **LGBM Sweep** | 最优 LightGBM $R^2=0.9982$ | NN 基线需超过此值 |

### 🎯 设计哲学

基于以上发现，NN 设计应遵循以下核心原则：

1. **"Linear + Residual"**: 主干是线性映射，NN 只学习残差
2. **"Attention for Filtering"**: 使用注意力机制实现隐式特征选择
3. **"Variance-Aware"**: 对低方差但高信息量的方向给予足够权重
4. **"Noise-Adaptive"**: 根据噪声水平动态调整正则化强度
5. **"Dual-Channel"**: 同时利用 flux 和 error σ 作为输入

## 1.2 核心假设

> **核心假设：在给定相同输入和参数预算的前提下，小型 MLP/CNN 若显式利用"线性 baseline + Top-K 特征选择"，可以在各噪声水平上稳定超过 Ridge Regression / LightGBM 的 $R^2$，且性能提升主要来自对局部非线性 line profile 的建模，而不是重新学习整个线性映射。**

如果假设成立，意味着：
- $\log g$ 的 **主体信息确实是高维但近似线性的**，NN 只需在少数关键区域做局部非线性修正
- 显式注入先验（Ridge 权重初始化、Top-K 波长子集）是有效的：
  - 可以显著减少网络所需参数与训练数据量
  - 对于中高噪声，**"先选点再建模"** 比"在全谱上盲目堆深度/宽度"更优
- 为后续 ViT 提供设计方向：
  - 只需要对少量 **信息密集的 token / patch** 使用高分辨率注意力
  - 其它区域可以降权、下采样或粗略建模

如果假设不成立，则需要：
- 重新评估：传统 ML 得到的"线性 + Top-K"结构是否只是在模型家族限制下的"假线性"
- 是否存在大规模、分布广泛的 **强非线性模式**，只有足够 expressive 的 NN 才能挖出来
- 可能需要转向：更深/更宽的网络（更强表示能力）、更复杂的非线性结构（多尺度 CNN、Transformer、物理先验嵌入）

## 1.3 验证问题

> 这些问题是为"核心假设"服务的，每个问题都对应一个可量化的实验。
> 结果一栏实验后填：`✅/❌ + 关键数值（R² / MAE / 相对提升）`。

### 1.3.1 Baseline NN 验证问题（本阶段重点）

| # | 问题 | 验证目标 | 结果 |
|---|------|---------|------|
| Q1 | **在无噪声数据上，全谱输入的简单 MLP（2–3 层）能否达到或超过 Ridge baseline 的 $R^2$？** | 验证在 clean regime 下，NN 至少不比线性差，支撑"线性 + 小非线性修正"范式是合理起点 | [待填] |
| Q2 | **在中等噪声水平（noise=1.0）下，使用 Top-K 重要波长子集训练的 MLP，是否显著优于全谱 MLP？** | 验证"Top-K 特征选择可以减少 NN 在去噪上的浪费"，支持"先选点再建模"的策略 | [待填] |
| Q3 | **在相同输入（Top-K+局部窗口）和参数量的前提下，小型 1D CNN 是否能稳定超过 MLP 的 $R^2$？** | 验证"局部 line profile / continuum 结构是有用的"，CNN 的局部感受野是否可以更好利用这些信息 | [待填] |
| Q4 | **以 Ridge 权重初始化第一层线性层的 MLP/CNN，相比随机初始化，是否在收敛速度或最终 $R^2$ 上有收益？** | 验证"显式注入线性先验"是否能减少训练难度、提升数据效率，为后续"线性+Transformer"混合架构提供支持 | [待填] |
| Q5 | **在高噪声水平（noise=2.0）下，最优 NN（MLP/CNN）的提升主要集中在哪些噪声 regime？** | 定位"NN 相对传统 ML 的优势区域"，确认性能提升是否如假设所说主要来自中高噪声而非 clean regime | [待填] |

### 1.3.2 Physics-Informed 架构验证问题（后续阶段）

| # | 问题 | 验证目标 | 预期结果 |
|---|------|---------|----------|
| Q6 | Linear shortcut 是否显著提升性能？ | 验证"线性本质"假设 | $\Delta R^2 \geq 0.02$ vs 无 shortcut |
| Q7 | Learnable attention 是否优于 Full Spectrum？ | 验证"信息稀疏"假设 | noise=1.0 时 $\Delta R^2 \geq 0.05$ |
| Q8 | Dual-channel [flux, $\sigma$] 是否优于 flux-only？ | 验证"error 信息"假设 | $\Delta R^2 \geq 0.03$ |
| Q9 | Noise augmentation 是否提升噪声鲁棒性？ | 验证"噪声训练"假设 | 跨噪声泛化 $\Delta R^2 \geq 0.1$ |
| Q10 | 能否超越 LightGBM 基线？ | 整体架构验证 | noise=1.0 时 $R^2 > 0.52$ |

## 1.4 结论摘要（实验后填写）

### 1.4.1 实验结论

| 结论 | 说明 |
|------|------|
| TODO | TODO |

### 1.4.2 设计启示

| 设计原则 | 具体建议 |
|---------|---------|
| TODO | TODO |

> **一句话总结**：TODO

---

# 2. 🧪 实验设计（Experiment Design）

## 2.1 数据（Data）

| 配置项 | 值 |
|--------|-----|
| 训练样本数 | 32,000 |
| 验证样本数 | 10,000 |
| 测试样本数 | 10,000 |
| 特征维度 | 4,096 (flux) + 4,096 (error) = 8,192 |
| 标签参数 | $\log g$ |
| 噪声水平 | test: {0.0, 1.0, 2.0} |

### 2.1.1 数据格式

| 字段 | 维度 | 说明 |
|------|------|------|
| `flux` | (N, 4096) | 光谱流量向量 |
| `error` | (N, 4096) | 每个像素的测量误差 |
| `log_g` | (N,) | 目标标签 |

### 2.1.2 预处理

```python
# Flux 标准化（按训练集统计）
flux_normalized = (flux - flux_mean) / flux_std

# 噪声注入
noisy_flux = flux + randn() * error * noise_level
```

**噪声模型：**
$$
\text{noisy\_flux} = \text{flux} + \mathcal{N}(0, 1) \times \text{error} \times \text{noise\_level}
$$

### 2.1.3 噪声水平与 Ridge Baseline

| noise_level | 含义 | Ridge 最优 $\alpha$ | Ridge Test $R^2$ | Ridge Test MAE | Ridge Test RMSE |
|-------------|------|---------------------|------------------|----------------|-----------------|
| 0.0 | 无噪声 | 0.001 | **0.999** | 0.005 | 0.009 |
| 1.0 | 标准噪声 | 200.0 | **0.458** | 0.173 | 0.215 |
| 2.0 | 高噪声 | 1000.0 | **0.221** | 0.212 | 0.258 |

## 2.2 使用的特征类型

| 特征类型 | 维度 | 来源 Insight |
|---------|------|-------------|
| flux (原始光谱) | 4096 | 主要信息载体 |
| error σ (测量误差) | 4096 | Error 实验: $R^2=0.91$ |
| flux / error (SNR) | 4096 | Error 实验推荐 |
| PCA whitened flux | 200 | PCA 实验: 有效维度 ~200 |

## 2.3 模型与算法（Model & Algorithm）

### 2.3.1 架构 A: Linear + Residual MLP

**Insight 来源**: Ridge α Sweep ("线性本质") + PCA 实验 ("低方差信息")

$$
\hat{y} = \underbrace{w^\top x}_{\text{Linear Shortcut}} + \underbrace{g_\theta(x)}_{\text{Residual MLP}}
$$

```
Input (4096) → [Linear Shortcut] ────────────────────┐
      │                                              │
      └→ MLP(4096 → 512 → 128 → 32 → 1) ──────────→ [Add] → Output
                                                     
其中 Linear Shortcut: y_linear = w^T x + b (可初始化为 Ridge 解)
```

**关键设计**:
- Linear shortcut 初始化为 Ridge 最优解
- MLP 只需学习残差 (预期很小)
- MLP 深度浅 (2-3 层即可)

### 2.3.2 架构 B: Attention-based Feature Selection

**Insight 来源**: Top-K 实验 ("信息稀疏") + Small K Limit ("关键谱线")

$$
\hat{y} = f_\theta\left(\sum_{i=1}^{D} \alpha_i \cdot x_i\right), \quad \alpha_i = \text{softmax}(W_\alpha x)_i
$$

```
Input (4096) ──→ Attention Weights (learnable) ──→ Weighted Sum ──→ MLP ──→ Output
      │                                                             │
      └─────────────── Linear Shortcut ─────────────────────────────┘
```

**关键设计**:
- Learnable soft attention 替代 hard Top-K
- 注意力权重可解释 (应聚焦 Ca II, Mg I, Na I 区域)
- 保留线性 shortcut 作为后备

### 2.3.3 架构 C: Dual-Channel (Flux + Error)

**Insight 来源**: Error 实验 ("Error 含物理信息") + Top-K 实验 ("SNR 特征")

```
Flux (4096) ────→ Encoder_flux ──┐
                                 ├──→ Fusion Layer ──→ MLP ──→ Output
Error σ (4096) ──→ Encoder_err ──┘
                                 │
                     Linear Shortcut (from flux only)
```

**关键设计**:
- 两路独立 encoder (因为 error 信息是非线性的)
- 可选: 使用 SNR = flux / error 作为第三通道
- Error encoder 需要更深的非线性层

### 2.3.4 架构 D: Physics-Informed ViT Variant

**Insight 来源**: Small K Limit ("关键谱线聚焦") + PCA 实验 ("分布式编码")

```
Input (4096) → [Patch Embedding (patch_size=64, num_patches=64)]
                     ↓
              [Positional Encoding (wavelength-aware)]
                     ↓
              [Transformer Encoder (2-4 layers)]
                     ↓
              [CLS Token] → Linear → Output
                     │
         [Linear Shortcut from Global Average]
```

**关键设计**:
- Patch size 选择 ~64 像素 (覆盖单条谱线)
- 位置编码使用物理波长而非序号
- 关键区域 (Ca II, Mg I, Na I) 可使用专门的 patch tokens
- 层数 2-4 层即可 (信息维度 ~200)

### 2.3.5 架构 E: Noise-Adaptive Network

**Insight 来源**: Feature Stability ("噪声决定稳定性") + Ridge α Sweep ("最优 α 随噪声变化")

$$
\hat{y} = f_\theta(x; \hat{\sigma}), \quad \hat{\sigma} = \text{NoiseEstimator}(x)
$$

```
Input (4096) ──→ Noise Estimator ──→ σ̂ (estimated noise level)
      │                               │
      └──→ Main Network ←─────────────┘ (noise-conditional)
                │
           [Noise-dependent weight decay / dropout]
```

**关键设计**:
- 网络自动估计输入噪声水平
- 根据估计噪声调整内部正则化
- 类似于 Noise2Noise 的思想

## 2.4 超参数（Hyperparameters）

### 2.4.1 训练配置（固定）

```yaml
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-4
batch_size: 128

scheduler: ReduceLROnPlateau
  factor: 0.5
  patience: 10

early_stopping:
  patience: 20
  min_delta: 1e-5
  monitor: val_loss

max_epochs: 100
gradient_clip: 1.0
use_amp: true  # 混合精度训练
seed: 42
```

### 2.4.2 通用超参数

| 参数 | 值/搜索范围 | 来源 Insight |
|------|----------|-------------|
| Learning rate | **1e-3** (固定) | - |
| Weight decay | **1e-4** (固定) | Ridge: 最优 α 跨 6 个数量级 |
| Batch size | **128** (固定) | - |
| Epochs | **100** + Early Stopping (patience=20) | LGBM Sweep: 收益递减 |
| Dropout | **0.1** (固定) | - |

### 2.4.2 架构特定超参数

| 架构 | 参数 | 搜索范围 | 来源 Insight |
|------|------|----------|-------------|
| Linear+Residual | MLP hidden dims | [256, 512], [128, 256, 512] | PCA: 有效维度 ~200 |
| Attention | Temperature | [0.1, 1.0, 10.0] | Top-K: K=1000 最优 |
| Dual-Channel | Error encoder depth | [2, 3, 4] | Error: 非线性关系 |
| ViT | Patch size | [32, 64, 128] | Small K: ~50 像素有效 |
| ViT | Num layers | [2, 4, 6] | PCA: 维度 ~200 |
| Noise-Adaptive | Noise estimator arch | [MLP, CNN] | - |

### 2.4.3 训练策略

| 策略 | 配置 | 来源 Insight |
|------|------|-------------|
| **Noise Augmentation** | train_noise ∈ {0.0, 0.5, 1.0, 1.2} | Top-K: 高噪声训练效果最佳 |
| **Curriculum Learning** | 从低噪声到高噪声 | Feature Stability: 噪声决定稳定性 |
| **Linear Warmup** | 先训练 shortcut，再训练 residual | Ridge: 线性几乎足够 |
| **Variance-Aware Normalization** | PCA whitening 或 per-channel normalization | PCA: 低方差方向重要 |

## 2.5 评估指标与基线

### 2.5.1 基线模型

| 模型 | noise=0.0 | noise=1.0 | 来源 |
|------|-----------|-----------|------|
| Ridge (最优 α) | $R^2=0.999$ | $R^2=0.45$ | Ridge α Sweep |
| LightGBM | $R^2=0.998$ | $R^2=0.52$ | LGBM Sweep |
| Ridge + Top-K (K=1000) | - | $R^2=0.34$ | Top-K 实验 |
| Ridge (train_noise=1.2) | - | $R^2=0.47$ | Top-K 实验 |

### 2.5.2 评估矩阵

| train_noise | test_noise | 目标 $R^2$ | 对标基线 |
|-------------|------------|-----------|----------|
| 0.0 | 0.0 | ≥ 0.999 | Ridge |
| 1.0 | 0.0 | ≥ 0.85 | LGBM |
| 1.0 | 1.0 | **≥ 0.55** | LightGBM ($R^2=0.52$) |
| 1.2 | 1.0 | **≥ 0.55** | Ridge nz1.2 ($R^2=0.47$) |
| 1.0 | 2.0 | ≥ 0.35 | LGBM ($R^2=0.27$) |

## 2.6 Baseline NN 实验计划（第一批必须完成的实验）

> 本阶段聚焦于 MLP 和 1D CNN 的 baseline 实验，为后续 Physics-Informed 架构提供基准。
> **注意**：这是纯粹的 baseline NN，没有任何花哨设计！

**噪声水平设定**：`noise_levels = [0.0, 1.0, 2.0]`

### 实验总览

| Group | 实验数 | 验证问题 | 说明 |
|-------|--------|----------|------|
| A | 24 | Q1, Q4 | 全谱 MLP vs Ridge |
| B | 12 | Q2 | Top-K MLP |
| C | 6 | Q3 | CNN vs MLP |
| **总计** | **42** | Q1-Q5 | |

---

### 实验 Group A：全谱 MLP vs Ridge (24 个实验)

**目的**：对应验证问题 Q1、Q4 — 验证 NN 在全谱输入下能否匹配 Ridge

**设定**：

| 变量 | 取值 |
|------|------|
| 输入 | 全谱 4096 维 |
| 架构 | 2×256, 2×512, 3×256, 3×512 |
| 初始化 | random, ridge |
| 噪声 | 0.0, 1.0, 2.0 |

**MLP 架构变体**：

| 名称 | 层数 | 隐藏层配置 | 参数量 (全谱 4096 输入) |
|------|------|-----------|------------------------|
| `mlp_full_2x256` | 2 | [256, 128] | ~1.1M |
| `mlp_full_2x512` | 2 | [512, 256] | ~2.2M |
| `mlp_full_3x256` | 3 | [256, 256, 128] | ~1.1M |
| `mlp_full_3x512` | 3 | [512, 512, 256] | ~2.5M |

**初始化方式**：

| init_type | 说明 |
|-----------|------|
| `random` | PyTorch 默认初始化 |
| `ridge` | 第一层用 Ridge 回归权重初始化 |

**实验数**：4 架构 × 2 初始化 × 3 噪声 = **24 实验**

**预期结果分析**：

| 噪声 | Ridge $R^2$ | MLP 预期 | 假设 |
|------|------------|----------|------|
| 0.0 | 0.999 | ~0.99 | 线性已最优，MLP 可能略逊 |
| 1.0 | 0.458 | 0.45-0.55 | MLP 可能通过非线性学到更鲁棒表示 |
| 2.0 | 0.221 | 0.22-0.30 | 高噪声下 NN 可能有优势 |

---

### 实验 Group B：Top-K MLP (12 个实验)

**目的**：对应验证问题 Q2 — 测试 Top-K 特征选择是否对 NN 有帮助

**设定**：

| 变量 | 取值 |
|------|------|
| 输入 | Top-K 特征 (从 Ridge 重要性) |
| K 值 | 128, 256, 512, 1024 |
| 架构 | 2×256 固定 (hidden=[256, 128]) |
| 噪声 | 0.0, 1.0, 2.0 |

**实验数**：4 K值 × 3 噪声 = **12 实验**

**实验 ID 示例**：
- `B_mlp_topk_K128_nz0.0`
- `B_mlp_topk_K256_nz1.0`
- `B_mlp_topk_K1024_nz2.0`

---

### 实验 Group C：Top-K+Window CNN vs MLP (6 个实验)

**目的**：对应验证问题 Q3 — 测试 CNN 能否比 MLP 更好地捕获局部谱线结构

**设定**：

| 变量 | 取值 |
|------|------|
| K | 256 (固定) |
| window_size | ±8 像素 |
| 模型 | CNN, MLP |
| 噪声 | 0.0, 1.0, 2.0 |

**1D CNN 架构**：

```
输入: (batch, 1, seq_len)
    ↓
Conv1d(1→32, k=7, padding=3) → BatchNorm → ReLU → MaxPool(2)
    ↓
Conv1d(32→64, k=7, padding=3) → BatchNorm → ReLU → MaxPool(2)
    ↓
AdaptiveAvgPool1d(1)
    ↓
Linear(64→128) → ReLU → Dropout(0.1)
    ↓
Linear(128→1)
```

**实验数**：2 模型 × 3 噪声 = **6 实验**

**实验 ID 示例**：
- `C_cnn_topk_window_K256_nz0.0`
- `C_mlp_topk_window_K256_nz1.0`

---

### 完整实验清单 (42 个)

<details>
<summary>📋 点击展开完整实验清单</summary>

#### Group A: Full Spectrum MLP (24 个)

| # | experiment_id | 架构 | 初始化 | 噪声 |
|---|---------------|------|--------|------|
| 1 | A_mlp_2x256_rand_nz0.0 | 2×256 | random | 0.0 |
| 2 | A_mlp_2x256_ridge_nz0.0 | 2×256 | ridge | 0.0 |
| 3 | A_mlp_2x512_rand_nz0.0 | 2×512 | random | 0.0 |
| 4 | A_mlp_2x512_ridge_nz0.0 | 2×512 | ridge | 0.0 |
| 5 | A_mlp_3x256_rand_nz0.0 | 3×256 | random | 0.0 |
| 6 | A_mlp_3x256_ridge_nz0.0 | 3×256 | ridge | 0.0 |
| 7 | A_mlp_3x512_rand_nz0.0 | 3×512 | random | 0.0 |
| 8 | A_mlp_3x512_ridge_nz0.0 | 3×512 | ridge | 0.0 |
| 9-16 | ... | ... | ... | 1.0 |
| 17-24 | ... | ... | ... | 2.0 |

#### Group B: Top-K MLP (12 个)

| # | experiment_id | K | 噪声 |
|---|---------------|---|------|
| 25 | B_mlp_topk_K128_nz0.0 | 128 | 0.0 |
| 26 | B_mlp_topk_K256_nz0.0 | 256 | 0.0 |
| 27 | B_mlp_topk_K512_nz0.0 | 512 | 0.0 |
| 28 | B_mlp_topk_K1024_nz0.0 | 1024 | 0.0 |
| 29-32 | ... | ... | 1.0 |
| 33-36 | ... | ... | 2.0 |

#### Group C: CNN vs MLP (6 个)

| # | experiment_id | 模型 | 噪声 |
|---|---------------|------|------|
| 37 | C_cnn_topk_window_K256_nz0.0 | CNN | 0.0 |
| 38 | C_mlp_topk_window_K256_nz0.0 | MLP | 0.0 |
| 39 | C_cnn_topk_window_K256_nz1.0 | CNN | 1.0 |
| 40 | C_mlp_topk_window_K256_nz1.0 | MLP | 1.0 |
| 41 | C_cnn_topk_window_K256_nz2.0 | CNN | 2.0 |
| 42 | C_mlp_topk_window_K256_nz2.0 | MLP | 2.0 |

</details>

---

## 2.7 结果记录格式

### 2.7.1 统一结果表

**路径**: `results/nn_baselines/nn_vs_ml_results.csv`

每一行包含以下字段：

| 列名 | 说明 |
|------|------|
| `experiment_id` | 唯一标识 (如 `A_mlp_2x256_rand_nz0.0`) |
| `experiment_group` | A, B, 或 C |
| `model_family` | MLP 或 CNN |
| `model_name` | 模型名称 |
| `init_type` | random 或 ridge |
| `input_type` | full_spectrum, topk, topk_window |
| `K` | Top-K 值 (null = 全谱) |
| `noise_level` | 噪声水平 |
| `test_R2`, `test_MAE`, `test_RMSE` | **测试集指标** |
| `epochs_to_best` | 最佳 epoch |
| `training_time_sec` | 训练时间 |
| `num_params` | 参数量 |

### 2.7.2 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **$R^2$** | $1 - SS_{res}/SS_{tot}$ | 主要指标，越高越好 |
| **MAE** | $\text{mean}(\|y - \hat{y}\|)$ | 平均绝对误差 |
| **RMSE** | $\sqrt{\text{mean}((y - \hat{y})^2)}$ | 均方根误差 |

### 2.7.3 自动生成报告

```bash
python scripts/summarize_nn_results.py
```

输出: `results/nn_baselines/NN_BASELINE_REPORT.md`

内容包括:
- 按 noise_level 分组的结果表
- "NN 相对 Ridge/LightGBM 的 $\Delta R^2$" 可视化或表格
- 对应 Q1–Q5 的简短文字总结（✅/❌）

## 2.8 运行实验

### 2.8.1 环境准备

```bash
cd /home/swei20/VIT
source init.sh
```

### 2.8.2 预览实验（Dry Run）

```bash
python scripts/run_nn_baselines.py --dry-run
```

### 2.8.3 并行运行 (8 GPU 最快)

```bash
# 使用所有 GPU
python scripts/run_nn_baselines.py --parallel

# 指定 GPU
python scripts/run_nn_baselines.py --parallel --gpus 0,1,2,3,4,5,6,7
```

### 2.8.4 运行特定 Group

```bash
python scripts/run_nn_baselines.py --parallel -e A      # 只跑 Group A
python scripts/run_nn_baselines.py --parallel -e A,B    # 跑 A 和 B
```

### 2.8.5 快速测试

```bash
python scripts/run_nn_baselines.py --parallel --num-train 4000 --epochs 10 -e A
```

---

# 3. 📊 实验图表

> 实验完成后填写

### 图 1：[TODO]

### 图 2：[TODO]

---

# 4. 💡 关键洞见（Key Insights）

> 实验完成后填写

### 4.1 宏观层洞见

TODO

### 4.2 模型层洞见

TODO

### 4.3 实验层细节洞见

TODO

---

# 5. 📝 结论（Conclusion）

> 实验完成后填写

## 5.1 核心发现

TODO

## 5.2 关键结论

TODO

## 5.3 设计启示

TODO

## 5.4 物理解释

TODO

## 5.5 关键数字速查

TODO

## 5.6 下一步工作

TODO

---

# 6. 📎 附录

## 6.1 实验优先级排序

基于前期实验 insights 的置信度和预期收益，推荐以下实验优先级：

### 🔴 高优先级 (Must Do)

| 优先级 | 实验 | 预期收益 | 支撑 Insight |
|--------|------|----------|-------------|
| P0 | **Linear + Residual (架构 A)** | 验证"线性本质"核心假设 | Ridge: $R^2=0.999$ @ noise=0 |
| P0 | **Noise Augmentation** | 提升噪声鲁棒性 | Top-K: 高噪声训练效果是 Top-K 的 12 倍 |
| P0 | **Dual-Channel (架构 C)** | 利用 error 信息 | Error: $R^2=0.91$ from σ only |

### 🟡 中优先级 (Should Do)

| 优先级 | 实验 | 预期收益 | 支撑 Insight |
|--------|------|----------|-------------|
| P1 | **Attention-based (架构 B)** | 隐式特征选择 | Top-K: K=1000 (24%) 足够 |
| P1 | **Variance-Aware Normalization** | 保护低方差信号 | PCA: 信息在低方差 PC |
| P1 | **ViT (架构 D)** | 捕获局部谱线结构 | Small K: 关键谱线聚焦 |

### 🟢 低优先级 (Nice to Have)

| 优先级 | 实验 | 预期收益 | 支撑 Insight |
|--------|------|----------|-------------|
| P2 | **Noise-Adaptive (架构 E)** | 自适应正则化 | Ridge: 最优 α 随噪声变化 |
| P2 | **Physical Positional Encoding** | 物理可解释性 | Small K: 特定波长重要 |
| P2 | **Ensemble (Linear + NN)** | 稳定性提升 | - |

## 6.2 预期风险与缓解

| 风险 | 可能原因 | 缓解策略 |
|------|---------|----------|
| Linear shortcut 主导，残差无贡献 | 非线性成分确实很小 | 分析残差分布，如确实很小则接受 |
| Attention 权重不聚焦于物理谱线 | 数据驱动的"捷径" | 添加物理约束 (Ca II, Mg I 区域 prior) |
| 双通道没有额外提升 | Error 信息已被 flux 隐式编码 | 消融实验确认 |
| 无法超越 LightGBM | 树模型的组合优化更强 | 尝试更深的网络或集成方法 |

## 6.3 代码框架建议

```python
# 核心架构实现骨架

class LinearResidualNet(nn.Module):
    """架构 A: Linear + Residual MLP"""
    def __init__(self, input_dim=4096, hidden_dims=[512, 128, 32]):
        super().__init__()
        # Linear shortcut (可初始化为 Ridge 解)
        self.linear = nn.Linear(input_dim, 1)
        
        # Residual MLP
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.residual = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.linear(x) + self.residual(x)


class AttentionNet(nn.Module):
    """架构 B: Attention-based Feature Selection"""
    def __init__(self, input_dim=4096, hidden_dim=256):
        super().__init__()
        # Learnable attention weights
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        # Linear shortcut
        self.linear = nn.Linear(input_dim, 1)
        # MLP on weighted features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        attn = self.attention(x)  # (B, D)
        weighted = x * attn       # (B, D)
        return self.linear(x) + self.mlp(weighted)


class DualChannelNet(nn.Module):
    """架构 C: Dual-Channel (Flux + Error)"""
    def __init__(self, input_dim=4096, hidden_dim=256):
        super().__init__()
        # Flux encoder (浅层，因为线性为主)
        self.flux_enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Error encoder (深层，因为非线性)
        self.error_enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Linear shortcut (flux only)
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, flux, error):
        flux_feat = self.flux_enc(flux)
        error_feat = self.error_enc(error)
        fused = torch.cat([flux_feat, error_feat], dim=-1)
        return self.linear(flux) + self.fusion(fused)
```

## 6.4 代码文件结构

```
/home/swei20/VIT/
├── src/nn/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py            # MLP 模型 (支持 Ridge 初始化)
│   │   └── cnn1d.py          # CNN1D + TopKWindowCNN
│   ├── data_adapter.py       # 数据加载
│   ├── baseline_trainer.py   # train_and_evaluate
│   └── __init__.py
│
├── scripts/
│   ├── run_nn_baselines.py       # 主脚本 (支持多 GPU 并行)
│   └── summarize_nn_results.py   # 报告生成
│
├── results/nn_baselines/
│   ├── nn_vs_ml_results.csv      # 统一结果表
│   └── NN_BASELINE_REPORT.md     # 自动生成的报告
│
└── docs/
    └── NN_BASELINE_EXPERIMENTS.md # 实验设计文档
```

## 6.5 Baseline MLP/CNN 代码示例

```python
class MLP(nn.Module):
    """Baseline MLP for log g prediction"""
    def __init__(self, input_dim=4096, hidden_sizes=[256, 128], 
                 activation='relu', dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
    def init_from_ridge(self, ridge_weights, ridge_bias):
        """Initialize first layer from Ridge regression solution"""
        with torch.no_grad():
            # Expand ridge weights to first hidden layer
            first_linear = self.network[0]
            hidden_dim = first_linear.weight.shape[0]
            # Simple approach: tile the ridge solution
            first_linear.weight.data = ridge_weights.unsqueeze(0).repeat(hidden_dim, 1)
            first_linear.bias.data = ridge_bias.repeat(hidden_dim)


class CNN1D(nn.Module):
    """1D CNN for local spectral feature extraction"""
    def __init__(self, input_channels=1, seq_len=4096, 
                 channels=[32, 64], kernel_size=5, fc_dims=[128]):
        super().__init__()
        # Conv layers
        conv_layers = []
        prev_ch = input_channels
        for ch in channels:
            conv_layers.extend([
                nn.Conv1d(prev_ch, ch, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
            prev_ch = ch
        self.conv = nn.Sequential(*conv_layers)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, seq_len)
            conv_out = self.conv(dummy)
            flat_size = conv_out.view(1, -1).shape[1]
        
        # FC layers
        fc_layers = []
        prev_dim = flat_size
        for dim in fc_dims:
            fc_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        fc_layers.append(nn.Linear(prev_dim, 1))
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        # x: (batch, seq_len) -> (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        conv_out = self.conv(x)
        flat = conv_out.view(conv_out.shape[0], -1)
        return self.fc(flat)
```

## 6.6 Coding Agent Prompt（用于自动化实验）

<details>
<summary>📋 点击展开完整 Prompt</summary>

```text
你是一个熟悉 Python、PyTorch 和科学计算的 coding agent。  
现在请你基于我现有的 Ridge Regression / LightGBM 分析结果，搭建并运行一组神经网络 baseline 实验（MLP 和 1D CNN），用于预测光谱的 log_g，并系统对比 NN 与传统 ML 的表现。

请注意：  
- 目标是 **从简单 NN 开始，逐步复杂化（未来会扩展到 Vision Transformer）**。这一轮只做 MLP/CNN。  
- 实验设计要和我现有的 Ridge/LightGBM 实验高度对齐：相同的数据分割、噪声设定和评估指标。  
- 所有结果要结构化输出（例如 CSV/Parquet + Markdown 汇总表），便于后续分析。

--------------------------------
一、数据与已有结果（请复用）
--------------------------------

1. 数据格式
   - 输入：光谱 flux 向量，维度约为 4096
   - 输出：标量标签 log_g
   - 可能还存在每个像素的 error / noise 方差向量

2. 数据加载
   - 复用/封装现有的 data loader / preprocessing
   - 使用与 Ridge/LightGBM 相同的 train/valid/test 划分
   - 相同的标准化方式（按训练集统计对 flux 做标准化）

3. 传统 ML baseline
   - 加载 Ridge/LightGBM 的评估结果
   - 在 NN 的结果表中同时显示对应噪声水平下的 Ridge/LightGBM 指标

4. Top-K 特征信息
   - 加载 feature importance CSV，按 importance 排序后取 Top-K

--------------------------------
二、需要实现的模型
--------------------------------

1. MLP 模型（全连接网络）
   - 可配置：hidden_sizes, num_layers, activation, dropout
   - 支持两种初始化：Random init / Ridge 初始化第一层

2. 1D CNN 模型
   - 视光谱为一维序列 [batch_size, 1, seq_len]
   - 可配置：num_conv_layers, channels, kernel_size, stride/pooling

3. 统一接口
   - train_and_evaluate(model, train_loader, valid_loader, test_loader, config)
   - 使用 MSELoss，early stopping (patience=20)
   - 返回 R², MAE, RMSE 和训练日志

--------------------------------
三、实验设计
--------------------------------

noise_levels = [0.0, 0.5, 1.0, 2.0]

【实验 A：全谱 MLP vs Ridge】
- 输入：标准化后的全谱 flux
- MLP: num_layers ∈ {2, 3}, hidden_size ∈ {256, 512}
- 初始化：Random / Ridge-init
- 与 Ridge baseline 对比

【实验 B：Top-K MLP vs 全谱 MLP】
- Top-K = {128, 256, 512, 1024}
- 对照：Random-K（相同维度，随机选择）
- MLP 固定 3 层，hidden_size=256

【实验 C：Top-K+局部窗口 CNN vs MLP】
- 每个 Top-K 波长取 ±8 像素窗口
- CNN: 2 conv layers, channels=[32,64], kernel=5或7
- 对照：相近参数量的 MLP

【实验 D：Ridge 初始化效果】
- 对比 Random init vs Ridge-init
- 记录 epochs_to_best 和收敛曲线

--------------------------------
四、结果记录
--------------------------------

输出 nn_vs_ml_results.csv，字段包括：
model_family, model_name, init_type, input_type, K, noise_level,
train_R2, valid_R2, test_R2, train_MAE, valid_MAE, test_MAE,
train_RMSE, valid_RMSE, test_RMSE, epochs_to_best

自动生成 Markdown 报告片段
```

</details>

## 6.7 相关文件

| 类型 | 路径 |
|------|------|
| **NN 实验设计文档** | `/home/swei20/VIT/docs/NN_BASELINE_EXPERIMENTS.md` |
| Ridge 实验 | `logg/ridge/exp_ridge_alpha_sweep_20251127.md` |
| Ridge Top-K 实验 | `logg/ridge/exp_ridge_topk_20251129.md` |
| PCA 实验 | `logg/pca/exp_pca_linear_regression_20251128.md` |
| Top-K 实验 | `logg/noise/exp_noise_topk_feature_selection_20251128.md` |
| Small K 实验 | `logg/noise/exp_small_k_limit_20251129.md` |
| Error 实验 | `logg/ridge/exp_error_logg_20251127.md` |
| Feature Stability | `logg/ridge/exp_feature_importance_stability_20251128.md` |
| LightGBM Sweep | `logg/lightgbm/exp_lightgbm_hyperparam_sweep_20251129.md` |
| LGBM vs Ridge Top-K | `logg/noise/exp_topk_feature_selection_lgbm_vs_ridge_20251129.md` |

---

*报告创建时间: 2025-11-29*  
*更新时间: 2025-11-29 (整合详细实验设计 from VIT/docs/NN_BASELINE_EXPERIMENTS.md)*  
*基于 8 份前期实验报告的 insights 设计*  
*实验总数: 42 个 (Group A: 24, Group B: 12, Group C: 6)*  
*实验阶段: Phase 1 - Baseline MLP/CNN → Phase 2 - Physics-Informed Architectures → Phase 3 - Vision Transformer*

