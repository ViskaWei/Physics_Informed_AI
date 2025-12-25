# 🧠 Benchmark Hub
> **Name:** 跨模型 log_g Benchmark | **ID:** `VIT-20251205-benchmark-hub`  
> **Topic:** `benchmark` | **Layer:** L1 (Cross-Cutting) | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-05 | **Status:** 🔄 Exploring
```
💡 各模型在不同噪声/数据量下的性能对比  
决定：模型选择、资源分配优先级
```

---

## 🔗 Hub Dependencies

> **定义本 Hub 与其他 Hub 的引用关系，供自动传播使用**
> 
> 📋 完整依赖图见 [`../_hub_graph.md`](../_hub_graph.md)

### 📤 Parent Hubs (引用本 Hub 的上层)

| Parent Hub | 引用的数据 | 同步章节 |
|------------|-----------|---------|
| [`master_hub`](../master_hub.md) | Benchmark 战略结论, 最佳模型推荐 | §2 Strategic Questions, §3 Global Insights |

### 📥 Child Hubs (本 Hub 引用的下层)

| Child Hub | 引用的数据 | 来源章节 |
|-----------|-----------|---------|
| [`ridge_hub`](../ridge/ridge_hub_20251223.md) | R² @ all noise levels, 最优 α | §4.2 Key Numbers |
| [`lightgbm_hub`](../lightgbm/lightgbm_hub_20251130.md) | R² @ all noise levels, 最优配置 | §5.3 Key Numbers |
| [`NN_hub`](../NN/NN_main_20251130.md) | MLP R² @ all noise levels | §3 核心发现 |

---

## 🔗 相关文件

| 类型 | 文件 | 说明 |
|------|------|------|
| 📍 Roadmap | [`benchmark_roadmap_20251205.md`](./benchmark_roadmap_20251205.md) | 实验追踪与执行 |
| 📗 子实验 | `exp_*.md` | 单实验详情 |
| 📇 知识卡片 | `card_*.md` | 浓缩结论 |

### 📁 已有实验来源

| 模型 | Hub/Main 文件 | 说明 |
|------|--------------|------|
| Ridge | [`../ridge/ridge_main_20251130.md`](../ridge/ridge_main_20251130.md) | 32k baseline |
| Ridge 100k | [`exp_ridge_100k_noise_sweep_20251205.md`](./exp_ridge_100k_noise_sweep_20251205.md) | 100k 全 noise |
| LightGBM | [`../lightgbm/lightgbm_hub_20251130.md`](../lightgbm/lightgbm_hub_20251130.md) | 32k + 100k |
| LightGBM Summary | [`../lightgbm/exp_lightgbm_summary_20251205.md`](../lightgbm/exp_lightgbm_summary_20251205.md) | 32k n=1000 完整 benchmark |
| **LightGBM 100k SOTA** 🆕 | [`../lightgbm/exp_lightgbm_noise_config_consolidated_20251209.md`](../lightgbm/exp_lightgbm_noise_config_consolidated_20251209.md) | **100k n=2500, lr=0.05 全noise** |
| MoE | [`../moe/moe_main_20251203.md`](../moe/moe_main_20251203.md) | 分区 Ridge |
| MLP/NN | [`../NN/NN_main_20251130.md`](../NN/NN_main_20251130.md) | 32k + 100k |
| CNN | [`../cnn/cnn_main_20251201.md`](../cnn/cnn_main_20251201.md) | Dilated CNN |

---

# 📑 目录

- [1. 🌲 核心问题树](#1--核心问题树)
- [2. 🔺 假设金字塔](#2--假设金字塔)
- [3. 💡 洞见汇合站](#3--洞见汇合站)
- [4. 🧭 战略导航](#4--战略导航)
- [5. 📐 设计原则库](#5--设计原则库)
- [6. 📎 附录](#6--附录)

---

# 1. 🌲 核心问题树

## 1.1 顶层问题

> **在不同噪声水平和数据规模下，哪种模型是 $\log g$ 预测的最佳选择？各模型的 Scaling 规律是什么？**

## 1.2 问题分解

```
🎯 顶层问题: 跨模型 log_g 预测 Benchmark
│
├── Q1: 模型性能排序如何？
│   ├── Q1.1: 各 noise level 下哪个模型最强？ → ✅ 部分验证 [见 §5.3]
│   │         - noise=0: Ridge ≈ LightGBM ≈ 0.998-0.999
│   │         - noise=0.1-0.5: LightGBM > CNN > MLP > Ridge
│   │         - noise≥1.0: Ridge best > LightGBM (32k)
│   ├── Q1.2: 各 data size 下哪个模型最强？ → 🔄 进行中
│   └── Q1.3: 模型排序是否随 noise/data 变化？ → ✅ 是 [高噪声 Ridge 反超]
│
├── Q2: Data Scaling 规律
│   ├── Q2.1: 32k → 100k 各模型增益多少？ → ✅ LightGBM E03
│   │         - noise=0.1: +1.96%
│   │         - noise=0.5: +9.35%
│   │         - noise=1.0: +17.9%
│   ├── Q2.2: 哪个模型对数据量最敏感？ → 🔄 进行中 [需要 NN 100k 对比]
│   └── Q2.3: 是否存在 data ceiling？ → ⏳ 待验证 (需 1M 数据)
│
├── Q3: Noise Scaling 规律
│   ├── Q3.1: 各模型的噪声鲁棒性如何？ → ✅ 已量化 [见 §5.3]
│   ├── Q3.2: 高噪声下哪个模型衰减最慢？ → ✅ Ridge (调优 α 后)
│   └── Q3.3: 是否存在 noise ceiling？ → ✅ 部分验证 [noise=1.0 ≈ R²=0.5]
│
└── Q4: 实用指导
    ├── Q4.1: 什么场景用什么模型？ → ✅ 部分验证 [见 §5.1 设计原则]
    └── Q4.2: 性能 vs 训练成本 trade-off？ → ⏳ 待验证

状态图例:
✅ 已验证 | ❌ 已否定 | 🔄 进行中 | ⏳ 待验证
```

## 1.3 问题边界

| ✅ 本研究关注 | ❌ 本研究不关注 |
|-------------|---------------|
| Ridge, LightGBM, MoE, MLP, CNN 对比 | Transformer/ViT（后续专题） |
| $\log g$ 单目标预测 | 多目标预测 (Teff, [Fe/H]) |
| noise = 0, 0.1, 0.2, 0.5, 1, 2 | 其他噪声分布 |
| data size = 32k, 100k, (1M 待验证) | 更小/更大数据规模 |
| 最佳 R² 对比 | 训练时间/推理速度对比（可选） |
| 最优超参数配置 | 超参数敏感性分析（部分完成） |

### 已完成的主要维度

| 维度 | 已完成 | 待完成 |
|------|--------|--------|
| **模型** | Ridge ✅, LightGBM ✅, MoE ✅, MLP ✅, CNN ✅ | Transformer, ViT |
| **Noise** | 0, 0.1, 0.2, 0.5, 1.0, 2.0 ✅ | - |
| **Data Size** | 32k ✅, 100k (部分) | 1M |
| **MoE 分区** | [M/H] ✅, Teff ✅ | SNR, Latent gate |

---

# 2. 🔺 假设金字塔

## 2.1 L1 宏观假设（战略层）

| # | 宏观假设 | 验证状态 | 如果成立 | 如果不成立 |
|---|---------|---------|---------|-----------|
| **H1** | 非线性模型（LightGBM/MoE/NN）在大数据+低噪声下优于 Ridge | ✅ 部分 | NN 架构值得投入 | 专注 Ridge 优化 |
| **H2** | 数据量增加对所有模型都有益，但增益因模型而异 | ✅ 验证 | 按模型制定数据策略 | 统一数据策略 |
| **H3** | 高噪声下简单模型（Ridge）可能反超复杂模型 | ✅ 验证 | 噪声决定模型选择 | 始终用最强模型 |
| **H4** | $\log g$-flux 映射本质接近线性 | ✅ 验证 | Linear shortcut 是必须的 | NN 的主任务是非线性提取 |

## 2.2 L2 中观假设（战术层）

| # | 中观假设 | 上层假设 | 验证状态 | 关键实验 |
|---|---------|---------|---------|---------|
| **H1.1** | LightGBM 在 32k 数据+低噪声下是最强 baseline | H1 | ✅ 验证 | LightGBM E01/E02 |
| **H1.2** | MoE 能在 noise=0.2 超越全局 Ridge | H1 | ✅ 验证 | MoE MVP-1.1 (ΔR²=+0.050) |
| **H1.3** | MLP 在 32k 数据下能超越 Ridge | H1 | ✅ 验证 | NN E01 (0.498 vs 0.458) |
| **H1.4** | MLP 在 32k 数据下仍弱于 LightGBM | H1 | ✅ 验证 | NN E01 (0.498 vs 0.536) |
| **H1.5** | CNN 在全谱任务上可以工作 | H1 | ✅ 验证 | CNN k=9 达 R²=0.657 |
| **H2.1** | LightGBM 32k→100k 有显著增益（尤其高噪声） | H2 | ✅ 验证 | LightGBM E03 (+1.9%~+17.9%) |
| **H2.2** | **Ridge 32k→100k 增益有限 (<5%)** | H2 | ✅ **验证** | **MVP-1.0 (平均 +2.71%)** |
| **H2.3** | MLP 32k→100k 有显著增益 | H2 | ✅ 验证 | NN E01 (+10.6%) |
| **H3.1** | noise ≥ 1.0 时 Ridge 反超 LightGBM | H3 | ❌ **否定** | LightGBM Summary (32k n=1000: 0.536>0.458) |
| **H4.1** | noise=0 时 Ridge R² ≈ 1 | H4 | ✅ 验证 | Ridge E01 (R²=0.999) |

## 2.3 L3 微观假设（可验证层）

| # | 可验证假设 | 上层假设 | 验证标准 | 结果 | 来源 |
|---|-----------|---------|---------|------|------|
| **H1.1.1** | LightGBM(32k,noise=0) R² > 0.99 | H1.1 | R² ≥ 0.99 | ✅ 0.9982 | LightGBM E01 |
| **H1.1.2** | LightGBM > Ridge @ noise=0.1 | H1.1 | ΔLGB-Ridge > 0 | ✅ +0.045 | LightGBM E02 |
| **H1.2.1** | MoE(9专家) > Ridge(全局) @ noise=0.2 | H1.2 | ΔMoE-Ridge ≥ 0.03 | ✅ +0.050 | MoE MVP-1.1 |
| **H1.3.1** | MLP(Residual) > Ridge @ noise=1.0 | H1.3 | ΔMLP-Ridge > 0 | ✅ +0.04 | NN E01 |
| **H1.5.1** | CNN(k=9) > 0.3 @ noise=0.1 | H1.5 | R² > 0.3 | ✅ 0.657 | CNN E01 |
| **H2.1.1** | LGB 100k > LGB 32k @ all noise | H2.1 | Δ100k-32k > 0 | ✅ +1.9%~+17.9% | LightGBM E03 |
| **H2.2.1** | **Ridge 100k vs 32k 增益 < 5%** | H2.2 | Δ100k-32k < 5% | ✅ **+2.71%** | **MVP-1.0** |
| **H1.5.1** 🆕 | **最优 α 在 5000~1e8 之间存在倒 U 型曲线** | H1 | 峰值后 R² 下降 | ✅ **确认** | **SCALING-20251222** |
| **H2.3.1** | MLP 100k > MLP 32k @ noise=1.0 | H2.3 | Δ100k-32k > 0 | ✅ +10.6% | NN E01 |
| **H3.1.1** | Ridge(α=best) > LightGBM @ noise=1.0 | H3.1 | ΔRidge-LGB > 0 | ❌ **-0.078** | LightGBM Summary (32k n=1000) |
| **H4.1.1** | Ridge(noise=0) R² > 0.99 | H4.1 | R² > 0.99 | ✅ 0.999 | Ridge E01 |

---

# 3. 💡 洞见汇合站

## 3.1 汇合点列表

| # | 汇合主题 | 单点来源 | 汇合结论 | 置信度 |
|---|---------|---------|---------|--------|
| C1 | 低噪声模型排序 | Ridge, LightGBM | Ridge ≈ LightGBM @ noise=0 | 🟢 高 |
| C2 | 高噪声模型选择 | LightGBM Summary, Ridge E01 | **LightGBM > Ridge @ noise ≥ 1.0** (32k n=1000) | 🟢 高 ⚠️ 修正 |
| C3 | 数据量增益 | LightGBM E03 | 噪声越大，数据增益越大 | 🟢 高 |
| C4 | 映射本质线性 | Ridge E01, LightGBM E01 | $\log g$-flux 映射本质接近线性 | 🟢 高 |
| C5 | MLP vs 传统 ML | NN E01 | 32k 数据 MLP 介于 Ridge 和 LightGBM 之间 | 🟢 高 |
| C6 | CNN 小 kernel 最优 | CNN E01 | 小 kernel (k=9) >> 大 kernel (k=63) | 🟢 高 |
| C7 | MoE 分区有效 | MoE MVP-1.1 | 分区 Ridge ΔR²=+0.050，[M/H] 贡献 69% | 🟢 高 |
| C8 | Ridge 数据量增益有限 | MVP-1.0 | 100k vs 32k 平均 +2.71%，高噪声例外 (+14.8%) | 🟢 高 |
| **C9** | **LightGBM 100k SOTA** | **Consolidated** | **100k+n=2500+lr=0.05 是全 noise 最优，增益 +0.1%~+13.4%** | 🟢 高 |
| **C10** 🆕 | **Ridge α 倒 U 型曲线** | **SCALING-20251222** | **最优 α: 100k=3.16e4, 1M=1e5; 优化后 R²: 0.4856/0.46** | 🟢 高 |

## 3.2 汇合详情

### 汇合点 C1: 低噪声下模型近似等效

**单点发现汇总**：

| 来源实验 | 单点发现 | 关键数据 |
|---------|---------|---------|
| Ridge E01 | noise=0 时 Ridge R²=0.999 | α=0.001 |
| LightGBM E01 | noise=0 时 LightGBM R²=0.9982 | lr=0.05 |

**汇合结论**：
> **在 noise=0 时，Ridge 和 LightGBM 性能几乎相同（差异 < 0.1%）**。说明 $\log g$-flux 映射本质接近线性，非线性模型的优势有限。

---

### 汇合点 C2: 高噪声下模型选择 ⚠️ **已修正**

**单点发现汇总**：

| 来源实验 | 单点发现 | 关键数据 |
|---------|---------|---------|
| **LightGBM Summary** | noise=1.0 时 LightGBM R²=**0.5361** | 32k, n=1000, leaves=63 |
| Ridge E01 | noise=1.0 时 Ridge R²=0.458 | α=200 |
| **LightGBM Summary** | noise=2.0 时 LightGBM R²=**0.2679** | 32k, n=1000 |
| Ridge E01 | noise=2.0 时 Ridge R²=0.221 | α=1000 |

**汇合结论**：
> ⚠️ **修正**: 32k n=1000 配置的 LightGBM 在所有 noise level 下都优于 Ridge。
> - noise=1.0: LightGBM 0.5361 > Ridge 0.458 (+17%)
> - noise=2.0: LightGBM 0.2679 > Ridge 0.221 (+21%)

**设计启示**：
- LightGBM 正确配置（n=1000, early stopping）后是全场景最优传统 ML baseline
- 之前的结论基于 n=100 配置，n=1000 后差距消失

---

### 汇合点 C4: 映射本质线性

**单点发现汇总**：

| 来源实验 | 单点发现 | 关键数据 |
|---------|---------|---------|
| Ridge E01 | noise=0 时 Ridge R²=0.999 | 仅差 0.1% 于完美 |
| Ridge E01 | 即使 noise=0，α=0.001 也比 OLS 好 | OLS 仅 0.969 |

**汇合结论**：
> **$\log g$ 是 flux 的近线性函数**：$Y \approx w^\top X + \text{const}$。NN 的主要任务是"学会忽略哪些像素"（信息过滤），而非"非线性提取"。

**设计启示**：
- NN 需要包含 **Linear shortcut**
- Weight decay 需要与噪声挂钩

---

### 汇合点 C5: MLP vs 传统 ML (32k, noise=1.0)

**单点发现汇总**：

| 来源实验 | 模型 | R² | vs Ridge |
|---------|------|-----|---------|
| NN E01 | MLP (Residual) | 0.498 | +8.7% |
| NN E01 | Ridge | 0.458 | baseline |
| LightGBM E02 | LightGBM | 0.536 | - |

**汇合结论**：
> **32k 数据下 MLP 性能排序**：LightGBM (0.536) > MLP (0.498) > Ridge (0.458)。MLP 超越 Ridge 证明非线性有价值，但仍弱于 LightGBM。

---

### 汇合点 C6: CNN 小 kernel 最优

**单点发现汇总**：

| 来源实验 | 配置 | RF | R² |
|---------|------|-----|-----|
| CNN E01 | k=9, d=1, lr=3e-3 | 25 | **0.657** 🏆 |
| CNN E01 | k=63, d=1, lr=1e-3 | 187 | 0.02 |

**汇合结论**：
> **感受野假设被推翻**：大感受野反而导致性能下降。$\log g$ 信息通过局部特征 + 全局池化组合，不需要卷积本身有大感受野。

---

### 汇合点 C7: MoE 分区有效

**单点发现汇总**：

| 来源实验 | 配置 | R² | vs 全局 Ridge |
|---------|------|-----|-------------|
| MoE MVP-1.1 | 9 专家 (Teff×[M/H]) | 0.9116 | +0.050 |
| MoE MVP-1.1 | 全局 Ridge (mask-aligned) | 0.8616 | baseline |

**汇合结论**：
> **MoE 分区有真实收益**：ΔR²=+0.050，CI=[0.033,0.067]。**[M/H] 贡献 68.7%**，是主要分区维度。

---

### 汇合点 C8: Ridge 数据量增益有限

**单点发现汇总**：

| 来源实验 | 单点发现 | 关键数据 |
|---------|---------|---------|
| Ridge 100k MVP-1.0 | 100k vs 32k 平均增益仅 +2.71% | 最佳增益 noise=2.0 (+14.8%) |
| Ridge 100k MVP-1.0 | 高噪声场景受益最多 | noise=2.0: 0.2536 vs 0.221 |
| Ridge 100k MVP-1.0 | 中低噪声几乎饱和 | noise=0.2: -2.4%, noise=0.5: -0.4% |

**汇合结论**：
> **H2.2 成立**: Ridge 作为线性模型，对数据量增益有限。32k 数据已近饱和，100k 平均仅 +2.71%。**例外**：高噪声场景 (noise=2.0) 增益达 +14.8%，可能因噪声抵消需要更多样本。

**设计启示**：
- Ridge 不值得用更多数据训练，32k 已足够
- 高噪声场景若用 Ridge，可考虑更多数据
- 提升方向应放在模型复杂度（LightGBM、NN）而非数据量

---

### 汇合点 C9: LightGBM 100k SOTA

**单点发现汇总**：

| 来源实验 | 单点发现 | 关键数据 |
|---------|---------|---------|
| LightGBM 100k Consolidated | 100k + n=2500 + lr=0.05 是全 noise 最优配置 | 所有 noise level 都超越 32k |
| LightGBM 100k Consolidated | 增益随噪声递增 | noise=0.1: +1.04%, noise=2.0: +13.4% |
| LightGBM 100k Consolidated | tree 上限中位数 2179，推荐 n=2500 | lr=0.05 需要更多树 |
| LightGBM 100k Consolidated | lr=0.05 在低噪声/极高噪声最优，lr=0.1 在中高噪声更稳 | 见趋势表 |

**汇合结论**：
> **100k 数据 + n_estimators=2500 + lr=0.05 是各 noise level 的最优配置**；增益随噪声增大，低噪声 +1%，高噪声 +10%+。100k 数据必须配合 n≥2500 才能发挥优势。

**设计启示**：
- LightGBM 是全场景最强 baseline（优于 Ridge、MLP、CNN）
- 100k 配置推荐：`lr=0.05, n_estimators=2500, num_leaves=31`
- 高噪声场景从更多数据中获益最大

---

### 汇合点 C10: Ridge α 倒 U 型曲线 🆕

**单点发现汇总**：

| 来源实验 | 单点发现 | 关键数据 |
|---------|---------|---------|
| SCALING-20251222 | 100k 样本最优 α = 3.16e+04 | R² = 0.4856, +2.55% vs baseline |
| SCALING-20251222 | 1M 样本最优 α = 1.00e+05 | R² = 0.46, +0.42% vs baseline |
| SCALING-20251222 | 最优 α 比原 baseline (5000) 高 1-2 个数量级 | 100k: 6.3x, 1M: 20x |
| SCALING-20251222 | α > 1e6 时 R² 急剧下降 | 过度正则化危险区 |

**汇合结论**：
> **H1.5.1 成立**: Ridge 最优 α 在 5000~1e8 之间存在明显的**倒 U 型曲线**。
> - 100k: 峰值 α=3.16e+04，R²=0.4856
> - 1M: 峰值 α=1.00e+05，R²=0.46 (1k test)
> - 最优 α 随数据量增加而增加（约 3x per 10x data）

**设计启示**：
- Ridge α 应该更大：推荐 α ∈ [1e4, 1e5] 而非原来的 5000
- α 与数据量正相关：更多数据 → 更大的最优 α
- 过度正则化危险区：α > 1e6 时 R² 急剧下降
- 优化 α 带来的改进有限（<3%），说明 Ridge 的 ceiling 确实存在

---

# 4. 🧭 战略导航

## 4.1 方向状态总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Benchmark 研究方向状态图（更新版）                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   🟢 高信心方向（已有数据）               🟡 待验证方向               │
│   ├── Ridge 32k baseline ← 完整          ├── Ridge 100k, 1M         │
│   ├── LightGBM 32k, 100k ← 完整          ├── LightGBM 1M            │
│   ├── 高噪声用 Ridge ← E02               ├── MoE 在其他 noise        │
│   ├── MLP 32k/100k ← NN E01              ├── CNN 在其他 noise        │
│   ├── CNN k=9 最优 ← CNN E01             └── NN vs LGB @ 100k+       │
│   └── MoE 分区有效 (noise=0.2) ← MoE-1.1                            │
│                                                                     │
│   🔴 风险方向                            ⚫ 已关闭方向               │
│   └── MLP 在 32k 弱于 LightGBM           └── 大 kernel CNN           │
│                                                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 4.2 高信心方向（🟢 多实验支持）

| 方向 | 支持证据 | 下一步行动 | 优先级 |
|------|---------|-----------|--------|
| 低噪声用 LightGBM | LightGBM E01, E03 | 扩展到 1M | 🟡 P1 |
| 高噪声用 Ridge | LightGBM E02 | 扩展到 100k, 1M | 🟡 P1 |
| CNN 小 kernel | CNN E01 (k=9 最优) | 其他 noise level 验证 | 🟡 P1 |
| MoE 按 [M/H] 分区 | MoE MVP-1.1 (ΔR²=+0.05) | 尝试连续条件化 | 🟡 P1 |
| MLP Residual 策略 | NN E01 | 与 Ridge baseline 结合 | 🟢 P2 |

## 4.3 待验证方向（🟡 假设未验证）

| 方向 | 依赖假设 | 需要实验 | 预计收益 |
|------|---------|---------|---------|
| Ridge 100k/1M | H2.2 | MVP-1.x | 完善 baseline |
| LightGBM 1M | H1.1 | MVP-2.x | 确定 ceiling |
| NN vs LGB @ 100k | H1.3 | MVP-3.x | MLP 是否能超越 |
| MoE 全矩阵 | H1.2 | MVP-4.x | 最优条件 |
| CNN 不同 noise | H1.5 | MVP-5.x | 噪声鲁棒性 |

## 4.4 已关闭方向（⚫ 已否定）

| 方向 | 否定证据 | 关闭原因 | 教训 |
|------|---------|---------|------|
| 大 kernel CNN | CNN E01 | k=63 仅 R²=0.02，远差于 k=9 | 感受野不是 CNN 瓶颈 |
| ~~高噪声 Ridge > LightGBM~~ | **LightGBM Summary** | **32k n=1000: LGB 0.5361 > Ridge 0.458** | **正确配置的 LightGBM 全面领先** |

---

# 5. 📐 设计原则库

## 5.1 已确认原则

| # | 原则名称 | 具体建议 | 证据来源 | 适用范围 |
|---|---------|---------|---------|---------|
| P1 | **低噪声：模型不敏感** | noise=0 时 Ridge ≈ LightGBM | Ridge/LightGBM E01 | noise ≤ 0.1 |
| P2 | **~~高噪声：用 Ridge~~** | ❌ **已否定**: 32k n=1000 LightGBM (0.536) > Ridge (0.458) | LightGBM Summary | - |
| P3 | **数据量增益与噪声正相关** | 噪声越大，100k 增益越大 | LightGBM E03 | 所有模型 |
| P4 | **映射本质线性** | NN 需要 Linear shortcut | Ridge E01 | 所有 NN |
| P5 | **正则化总是有益** | 即使 noise=0 也需要正则化 | Ridge E01 | 所有模型 |
| P6 | **最优 α 随噪声单调增大** | Ridge α: 0.001(N=0)→1000(N=2) | Ridge E01 | Ridge |
| P7 | **CNN 用小 kernel** | k ∈ {7, 9, 11}，避免 k > 15 | CNN E01 | CNN |
| P8 | **lr 与 kernel 有交互** | 小 kernel 用 lr=3e-3 | CNN E01 | CNN |
| P9 | **MoE 优先按 [M/H] 分区** | [M/H] 贡献 68.7% | MoE MVP-1.1 | MoE |

## 5.2 待验证原则

| # | 原则名称 | 初步建议 | 需要验证 |
|---|---------|---------|---------|
| P10 | 1M 数据下 NN 超越 | 大数据 NN 可能最强 | MVP-4.x |
| P11 | MoE 最优噪声区间 | noise=0.2 附近 MoE 最有价值 | MVP-3.x |
| P12 | CNN + Position Encoding | 可能进一步提升 | CNN 后续 |

## 5.3 关键数字速查（已有数据）

### 📊 Ridge Baseline (32k)

| Noise | R² | 最优 α | vs OLS | 来源 |
|-------|-----|--------|--------|------|
| 0.0 | **0.999** | 0.001 | +3.1% | Ridge E01 |
| 0.1 | ~0.90 | 1.0 | - | Ridge E01 |
| 0.5 | ~0.67 | 50 | +10% | Ridge E01 |
| 1.0 | **0.458** | 200 | +19% | Ridge E01 |
| 2.0 | **0.221** | 1000 | **+68.7%** | Ridge E01 |

### 📊 Ridge 100k (BM-20251205-ridge-100k)

| Noise | 32k R² | 100k R² | 增益 | 最优 α | 来源 |
|-------|--------|---------|------|--------|------|
| 0.0 | 0.999 | **0.9994** | +0.04% | 0.001 | Ridge 100k E01 |
| 0.1 | 0.900 | **0.9174** | +1.9% | 1.0 | Ridge 100k E01 |
| 0.2 | 0.862 | 0.8413 | -2.4% | 10 | Ridge 100k E01 |
| 0.5 | 0.670 | 0.6674 | -0.4% | 100 | Ridge 100k E01 |
| 1.0 | 0.458 | **0.4687** | +2.3% | 100 | Ridge 100k E01 |
| 2.0 | 0.221 | **0.2536** | +14.8% | 1000 | Ridge 100k E01 |

**结论**: H2.2 成立 - Ridge 对数据量增益有限 (平均 +2.71%)

### 📊 Ridge α 优化 (SCALING-20251222-ridge-alpha-01) 🆕

| 数据规模 | Noise | 最优 α | 最优 R² | vs baseline α~3162 | 增益 |
|----------|-------|--------|---------|-------------------|------|
| 100k | 1.0 | **3.16e+04** | **0.4856** | 0.4735 | +2.55% |
| 1M | 1.0 | **1.00e+05** | **0.46** | 0.4997 | +0.42% |

**结论**: 
- 最优 α 随数据量增大（约 3x per 10x data）
- 存在明显倒 U 型曲线，α > 1e6 后 R² 急剧下降
- 推荐 α ∈ [1e4, 1e5]

### 📊 LightGBM (32k vs 100k SOTA)

| Noise | 32k R² | 100k R² (SOTA) | 增益 | 100k 配置 | 来源 |
|-------|--------|----------------|------|-----------|------|
| 0.0 | 0.9981 | **0.9991** | +0.10% | lr=0.05, n=5000 | LightGBM 100k Consolidated |
| 0.1 | 0.9616 | **0.9720** | +1.08% | lr=0.05, n=2218 | LightGBM 100k Consolidated |
| 0.2 | 0.9045 | **0.9318** | +3.02% | lr=0.05, n=3608 | LightGBM 100k Consolidated |
| 0.5 | 0.7393 | **0.7573** | +2.44% | lr=0.05, n=3855 | LightGBM 100k Consolidated |
| 1.0 | 0.5361 | **0.5582** | +4.12% | lr=0.05, n=2140 | LightGBM 100k Consolidated |
| 2.0 | 0.2679 | **0.3038** | +13.4% | lr=0.05, n=2000 | LightGBM 100k Consolidated |

> **32k 配置**: n=1000, leaves=63, lr=0.05
> **100k SOTA 配置**: n=2500 (推荐), leaves=31, lr=0.05
> **关键发现**: 100k 在所有 noise level 都优于 32k；增益随噪声增大 (+0.1% ~ +13.4%)

### 📊 MoE (32k, noise=0.2)

| 配置 | R² | ΔR² vs 全局 | 来源 |
|------|-----|------------|------|
| 分区 Ridge (9 专家) | **0.9116** | **+0.050** | MoE MVP-1.1 |
| 全局 Ridge (mask-aligned) | 0.8616 | baseline | MoE MVP-1.1 |
| Conditional Ridge (1st order) | 0.9018 | ~80% MoE | MoE MVP-3.2 |

### 📊 NN/MLP (noise=1.0)

| Data Size | MLP R² | Ridge | LightGBM | MLP vs Ridge | 来源 |
|-----------|--------|-------|----------|--------------|------|
| 32k | **0.498** | 0.458 | 0.536 | **+8.7%** | NN E01 |
| 100k | **0.551** | - | - | - | NN E01 |

### 📊 CNN (32k, noise=0.1)

| 配置 | RF | R² | 参数量 | 来源 |
|------|-----|-----|--------|------|
| k=9, d=1, lr=3e-3 | 25 | **0.657** 🏆 | 27K | CNN E01 |
| k=7, d=1, lr=3e-3 | 19 | 0.603 | 23K | CNN E01 |
| k=21, d=1, lr=3e-3 | 61 | 0.035 | 52K | CNN E01 |
| k=63, d=1, lr=1e-3 | 187 | 0.020 | 140K | CNN E01 |

---

# 6. 📎 附录

## 6.1 实验矩阵（已有数据整合）

### 目标：完整 Benchmark 矩阵

**模型**: Ridge, LightGBM, MoE, MLP, CNN
**Noise**: 0, 0.1, 0.2, 0.5, 1, 2
**Data**: 32k, 100k, 1M

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                       Data Size                              │
                    │           32k              100k              1M              │
┌───────────────────┼─────────────────────────────────────────────────────────────┤
│ Noise = 0         │ Ridge:  0.999 ✅        Ridge: 0.9994 ✅   Ridge: ?          │
│                   │ LGB:    0.9981 ✅       LGB:   0.9991 ✅🏆 LGB:   ?          │
│                   │ MoE:    ?               MoE:   ?          MoE:   ?          │
│                   │ MLP:    ?               MLP:   ?          MLP:   ?          │
│                   │ CNN:    ?               CNN:   ?          CNN:   ?          │
├───────────────────┼─────────────────────────────────────────────────────────────┤
│ Noise = 0.1       │ Ridge:  0.900 ✅        Ridge: 0.9174 ✅   Ridge: ?          │
│                   │ LGB:    0.9616 ✅       LGB:   0.9720 ✅🏆 LGB:   ?          │
│                   │ MoE:    ?               MoE:   ?          MoE:   ?          │
│                   │ MLP:    ?               MLP:   ?          MLP:   ?          │
│                   │ CNN:    0.657 ✅        CNN:   ?          CNN:   ?          │
├───────────────────┼─────────────────────────────────────────────────────────────┤
│ Noise = 0.2       │ Ridge:  0.862 ✅        Ridge: 0.8413 ✅   Ridge: ?          │
│ (MoE 主测点)       │ LGB:    0.9045 ✅       LGB:   0.9318 ✅🏆 LGB:   ?          │
│                   │ MoE:    0.912 ✅        MoE:   0.925 ✅   MoE:   ?          │
│                   │ MLP:    ?               MLP:   ?          MLP:   ?          │
│                   │ CNN:    ?               CNN:   ?          CNN:   ?          │
├───────────────────┼─────────────────────────────────────────────────────────────┤
│ Noise = 0.5       │ Ridge:  0.670 ✅        Ridge: 0.6674 ✅   Ridge: ?          │
│                   │ LGB:    0.7393 ✅       LGB:   0.7573 ✅🏆 LGB:   ?          │
│                   │ MoE:    0.772 ✅        MoE:   ?          MoE:   ?          │
│                   │ MLP:    ?               MLP:   ?          MLP:   ?          │
│                   │ CNN:    ?               CNN:   ?          CNN:   ?          │
├───────────────────┼─────────────────────────────────────────────────────────────┤
│ Noise = 1.0       │ Ridge:  0.458 ✅        Ridge: 0.4856 ✅🆕  Ridge: 0.46 ✅🆕 │
│ (NN 主测点)        │ LGB:    0.5361 ✅       LGB:   0.5582 ✅🏆 LGB:   ?          │
│                   │ MoE:    ?               MoE:   ?          MoE:   ?          │
│                   │ MLP:    0.498 ✅        MLP:   0.551 ✅   MLP:   ?          │
│                   │ CNN:    ?               CNN:   ?          CNN:   ?          │
├───────────────────┼─────────────────────────────────────────────────────────────┤
│ Noise = 2.0       │ Ridge:  0.221 ✅        Ridge: 0.2536 ✅   Ridge: ?          │
│                   │ LGB:    0.2679 ✅       LGB:   0.3038 ✅🏆 LGB:   ?          │
│                   │ MoE:    ?               MoE:   ?          MoE:   ?          │
│                   │ MLP:    ?               MLP:   ?          MLP:   ?          │
│                   │ CNN:    ?               CNN:   ?          CNN:   ?          │
└───────────────────┴─────────────────────────────────────────────────────────────┘

图例: ✅ = 已测量 | 🏆 = 该配置最佳 | ? = 待测量

> **LightGBM 100k SOTA 配置**: lr=0.05, n_estimators=2500, num_leaves=31
```

### 模型排序总结

#### 32k 数据

| Noise | 排序（高→低） | 最佳模型 | 备注 |
|-------|-------------|---------|------|
| 0.0 | Ridge ≈ LightGBM | 均可 (0.999 ≈ 0.998) | 映射本质线性 |
| 0.1 | **LightGBM > CNN** > Ridge | LightGBM (0.9616) | LGB n=1000 超越 CNN |
| 0.2 | MoE > **LightGBM** > Ridge | MoE (0.912) | LGB=0.9045, 分区带来 +0.01 |
| 0.5 | MoE > **LightGBM** > Ridge | MoE (0.772) | LGB=0.7393 |
| 1.0 | **LightGBM > MLP** > Ridge | LightGBM (0.5361) | LGB n=1000 反超 MLP |
| 2.0 | **LightGBM > Ridge** | LightGBM (0.2679) | LGB n=1000 > Ridge 0.221 |

#### 100k 数据 🆕

| Noise | 排序（高→低） | 最佳模型 | R² | 备注 |
|-------|-------------|---------|-----|------|
| 0.0 | **LightGBM** > Ridge | LightGBM | **0.9991** | 接近完美 |
| 0.1 | **LightGBM** > Ridge | LightGBM | **0.9720** | +1.04% vs 32k |
| 0.2 | **LightGBM** > MoE > Ridge | LightGBM | **0.9318** | +3.02% vs 32k |
| 0.5 | **LightGBM** > MoE > Ridge | LightGBM | **0.7573** | +2.44% vs 32k |
| 1.0 | **LightGBM** > MLP > Ridge | LightGBM | **0.5582** | +4.12% vs 32k |
| 2.0 | **LightGBM** > Ridge | LightGBM | **0.3038** | +13.4% vs 32k |

> **🆕 2025-12-22 新增**: Ridge α 优化实验 (noise=1.0)
> - 100k 最优 α=3.16e+04, R²=0.4856
> - 1M 最优 α=1.00e+05, R²=0.46 (1k test)

> **⚠️ 重要发现**: LightGBM 100k (n=2500, lr=0.05) 在所有 noise level 都是 SOTA，增益随噪声增大

## 6.2 术语表

| 术语 | 定义 | 备注 |
|------|------|------|
| Ridge | L2 正则化线性回归 | baseline，α 需调优 |
| LightGBM | 梯度提升决策树 | 非线性 baseline，lr 最敏感 |
| MoE | Mixture of Experts | 分专家模型，按 [M/H] 分区最有效 |
| MLP | Multi-Layer Perceptron | 全连接神经网络，Residual 策略有效 |
| CNN | 1D Convolutional NN | 小 kernel (k=9) 最优 |
| Noise | 光谱噪声水平 | σ 值，SNR ≈ 1/σ |
| RF | Receptive Field | 感受野，CNN 关键参数 |

## 6.3 实验来源索引

| 模型 | 实验 ID | 文件位置 | 主要发现 |
|------|--------|---------|---------|
| Ridge | Ridge E01 | `logg/ridge/exp_ridge_alpha_sweep_20251127.md` | noise=0 时 R²=0.999 |
| LightGBM | LightGBM E01/E02/E03 | `logg/lightgbm/exp_lightgbm_*.md` | lr 最敏感，100k +4~18% |
| MoE | MoE MVP-1.1 | `logg/moe/exp_moe_rigorous_validation_20251203.md` | ΔR²=+0.050，[M/H] 贡献 69% |
| MLP | NN E01 | `logg/NN/exp_nn_comprehensive_analysis_20251130.md` | Residual MLP 有效 |
| CNN | CNN E01 | `logg/cnn/exp_cnn_dilated_kernel_sweep_20251201.md` | k=9 最优，感受野假设被推翻 |

## 6.4 变更日志

| 日期 | 变更内容 | 影响章节 |
|------|---------|---------|
| 2025-12-05 | 创建 Hub，整合已有数据 | 全部 |
| 2025-12-05 | 整合 Ridge/LightGBM/MoE/MLP/CNN 实验结果 | §2, §3, §5, §6 |
| 2025-12-05 | 添加完整实验矩阵和模型排序 | §6.1 |
| 2025-12-05 | 更新假设验证状态 | §2 |
| 2025-12-05 | ⚠️ 重大修正: 更新 LightGBM 32k (n=1000) 完整 benchmark，否定 H3.1 | §2.2, §2.3, §3.1, §3.2 C2, §4.4, §5.1 P2, §5.3, §6.1 |
| **2025-12-09** | **🆕 添加 LightGBM 100k SOTA (n=2500, lr=0.05) 完整 benchmark** | §相关文件, §3.1 C9, §5.3, §6.1 |
| **2025-12-22** | **🆕 添加 Ridge α 扩展实验 (100k/1M, noise=1.0)：倒 U 型曲线验证** | §2.3 H1.5.1, §3.1 C10, §5.3, §6.1 |

---

*最后更新: 2025-12-22*
