# 🏔️ Master Knowledge Hub

> **Purpose:** 全局研究战略导航 — log_g 预测任务  
> **Author:** Viska Wei  
> **Created:** 2025-12-24  
> **Updated:** 2025-12-24  
> **Status:** 🔄 Active

---

## 🔗 Hub Directory

| Layer | Hub | Path | Focus | Status |
|-------|-----|------|-------|--------|
| **L1** | MoE | [`moe/moe_hub_20251203.md`](./moe/moe_hub_20251203.md) | 专家混合架构 | 🟢 Active |
| **L1** | Scaling | [`scaling/scaling_hub_20251222.md`](./scaling/scaling_hub_20251222.md) | 数据/模型容量 | 🟢 Active |
| **L1** | Benchmark | [`benchmark/benchmark_hub_20251205.md`](./benchmark/benchmark_hub_20251205.md) | 跨模型对比 | 🟢 Active |
| L2 | Ridge | [`ridge/ridge_hub_20251223.md`](./ridge/ridge_hub_20251223.md) | 岭回归专题 | ✅ Stable |
| L2 | LightGBM | [`lightgbm/lightgbm_hub_20251130.md`](./lightgbm/lightgbm_hub_20251130.md) | 树模型专题 | 🎯 Converging |
| L2 | NN | [`NN/NN_main_20251130.md`](./NN/NN_main_20251130.md) | 神经网络专题 | 🔄 Exploring |

> 📋 Hub 依赖图详见 [`_hub_graph.md`](./_hub_graph.md)

---

## 📑 Contents

- [1. 🧭 Strategic Questions Index](#1--strategic-questions-index)
- [2. 📊 Current Answers](#2--current-answers)
- [3. 💡 Global Insights](#3--global-insights)
- [4. 🎯 Recommended Routes](#4--recommended-routes)
- [5. 📐 Cross-Topic Principles](#5--cross-topic-principles)
- [6. 📎 Appendix](#6--appendix)

---

# 1. 🧭 Strategic Questions Index

> **从 L1 Hubs 汇总的核心战略问题**

| # | Strategic Question | Current Answer | Source Hub | Confidence |
|---|-------------------|----------------|------------|------------|
| **Q1** | 高噪声(σ=1)下最佳模型是什么？ | LightGBM 100k (R²=0.56) > Ridge (R²=0.46) | benchmark | 🟢 High |
| **Q2** | 数据量增加对模型有多大帮助？ | 模型相关：Ridge 边际收益小(+2%)，LightGBM 高噪声下受益大(+13%) | scaling | 🟢 High |
| **Q3** | MoE 结构值得做吗？ | ✅ 是! Oracle MoE ΔR²=+0.16 @ noise=1 | moe, scaling | 🟢 High |
| **Q4** | 理论上限(R²_max)是多少？ | ⏳ 待验证 (Fisher 分析待完成) | scaling | 🔴 Pending |
| **Q5** | NN 能超越传统 ML 吗？ | ⚠️ 32k 下 MLP < LightGBM；100k 待验证 | benchmark, NN | 🟡 Medium |
| **Q6** | 门控/条件化如何落地？ | ✅ Soft routing + 物理窗特征 ρ=1.00 | moe | 🟢 High |

---

# 2. 📊 Current Answers

> **逐一回答战略问题，给出决策含义**

## Q1: 高噪声(σ=1)下最佳模型是什么？

| Item | Content |
|------|---------|
| **Current Answer** | LightGBM 100k (R²=0.5582) > Ridge 1M (R²=0.46) > MLP 100k (R²=0.551) |
| **Implication** | LightGBM 是当前最强 baseline；Ridge 天花板明确 |
| **Confidence** | 🟢 High |
| **Evidence** | benchmark_hub §5.3, scaling_hub §5.3 |

## Q2: 数据量增加对模型有多大帮助？

| Item | Content |
|------|---------|
| **Current Answer** | Ridge: 100k→1M 仅 +2.44% (边际收益); LightGBM: 32k→100k +13% @ noise=2 |
| **Implication** | Ridge 是 model-limited，不是 data-limited；LightGBM 在高噪声下从更多数据受益 |
| **Confidence** | 🟢 High |
| **Evidence** | scaling_hub §2.1 Answer A, ridge_hub §2.1 |

## Q3: MoE 结构值得做吗？

| Item | Content |
|------|---------|
| **Current Answer** | ✅ **值得!** Oracle MoE ΔR²=+0.16 @ noise=1 (远超 0.03 阈值); Soft routing ρ=1.00 |
| **Implication** | 结构红利在高噪声下更大；可落地的物理窗 Gate 已验证 |
| **Confidence** | 🟢 High |
| **Evidence** | moe_hub §3 C10, scaling_hub §3 C5 |

## Q4: 理论上限(R²_max)是多少？

| Item | Content |
|------|---------|
| **Current Answer** | ⏳ **待验证** (Fisher/CRLB 方法有数值问题，需要替代方案) |
| **Implication** | 无法确定"提升空间有多大"；建议用经验上限 (noise=0 R²=0.999) 作为参考 |
| **Confidence** | 🔴 Low |
| **Evidence** | scaling_hub §3 C4 (Fisher ceiling 可能虚高) |

## Q5: NN 能超越传统 ML 吗？

| Item | Content |
|------|---------|
| **Current Answer** | 32k 下 MLP (0.498) < LightGBM (0.536)；100k 下 MLP (0.551) 接近但未超越 |
| **Implication** | 全谱 MLP 不是最佳架构；需要考虑 CNN 或局部特征 |
| **Confidence** | 🟡 Medium |
| **Evidence** | NN_hub §3.1, moe_hub §3 C7 |

## Q6: 门控/条件化如何落地？

| Item | Content |
|------|---------|
| **Current Answer** | ✅ Soft routing + 11 维物理窗特征 (Ca II triplet 为主) 达到 ρ=1.00 |
| **Implication** | MoE 门控落地问题已解决；回归 Gate 优于分类 Gate |
| **Confidence** | 🟢 High |
| **Evidence** | moe_hub §3 C6, C8 |

---

# 3. 💡 Global Insights

> **跨主题的核心洞见（从 L1 Hubs 汇合）**

## I1: 映射本质线性 (Ridge R²=0.999 @ noise=0)

> **Source:** ridge_hub §3 C1, benchmark_hub §3 C4

log_g 信息几乎完全编码在光谱的线性子空间中。NN 的主要任务不是"提取非线性特征"，而是"学会忽略无关像素"。

**Implications:**
- NN 架构应包含 Linear shortcut
- 非线性模型的优势在高噪声下有限

---

## I2: 高噪声下结构红利更大 (Oracle MoE ΔR²=+0.16 @ noise=1)

> **Source:** moe_hub §3 C5, scaling_hub §3.2 C5

在高噪声条件下，全局模型被噪声淹没，而分 bin 后每个 bin 内样本更相似，Oracle MoE 优势反而更明显。

**Implications:**
- MoE 在真实观测数据上可能比低噪声模拟更有价值
- 高噪声场景应优先考虑 MoE 架构

---

## I3: 传统 ML 存在明确天花板 (Ridge=0.46, LightGBM=0.57 @ 1M, noise=1)

> **Source:** scaling_hub §3 C1, benchmark_hub §3 C1

数据量从 100k 增加到 1M 仅带来 2-3% 提升，说明传统 ML 的瓶颈不在数据量。

**Implications:**
- 资源应投入模型架构改进而非增加数据
- 深度学习目标：超过 R²=0.70

---

## I4: Soft Routing 是成功的关键 (ρ=1.00 vs Hard ρ=0.72)

> **Source:** moe_hub §3 C6

即使 Gate 准确率只有 82%，Soft routing 也能保住 100% 的 Oracle 增益。Hard routing 损失 28%。

**Implications:**
- 永远用 Soft routing，不用 Hard routing
- Gate 准确率不是瓶颈

---

## I5: [M/H] 是 MoE 分区的首选维度 (贡献 68.7%)

> **Source:** moe_hub §3 C2

金属丰度决定谱线强度和可用特征分布。3 个 [M/H] 专家可获得近 70% 的 MoE 收益。

**Implications:**
- Gate 设计优先对齐 [M/H]
- Ca II triplet 是核心特征

---

# 4. 🎯 Recommended Routes

> **基于当前证据的战略推荐**

## 4.1 Overall Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                     当前推荐路线                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   📊 证据汇总:                                                   │
│   ├── Ridge 天花板 R²=0.46 @ noise=1 ────────► model-limited    │
│   ├── LightGBM R²=0.57 是最强 baseline ──────► 需要超越它       │
│   ├── Oracle MoE ΔR²=+0.16 ──────────────────► MoE 值得做       │
│   ├── Soft Gate ρ=1.00 ──────────────────────► 门控已落地       │
│   └── 全谱 MLP < LightGBM ───────────────────► 需要更好架构     │
│                                                                 │
│   🎯 推荐路线: MoE + CNN/局部特征                               │
│   ⚠️ 待验证: 理论上限 (Fisher) + 1D-CNN 效果                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 4.2 Priority Actions

| Priority | Action | Expected Outcome | Source |
|----------|--------|------------------|--------|
| 🔴 **P0** | 验证 1D-CNN @ noise=1 | 突破 R²=0.60+ | scaling Phase 16CNN |
| 🔴 **P0** | 完成 MoE 100% coverage | 可交付版本 | moe MVP-12B |
| 🟡 P1 | 验证 Whitening 输入 | 可能 +2-5% | scaling Phase 16W |
| 🟡 P1 | 验证理论上限 | 明确提升空间 | scaling Phase 16T |
| 🟢 P2 | MoE + CNN 组合 | 最大化性能 | moe + scaling |

---

# 5. 📐 Cross-Topic Principles

> **跨主题适用的设计原则**

| # | Principle | Recommendation | Evidence | Scope |
|---|-----------|----------------|----------|-------|
| **P1** | Linear Shortcut | NN 架构: $\hat{y} = w^\top x + g_\theta(x)$ | ridge_hub C1 | 所有 NN |
| **P2** | Soft Routing | 永远用 Soft，不用 Hard | moe_hub C6 | MoE 架构 |
| **P3** | [M/H] 优先门控 | Gate 特征优先 Ca II triplet | moe_hub C2 | MoE Gate |
| **P4** | 高噪声训练更鲁棒 | 训练噪声 ≥ 目标测试噪声 | ridge_hub C3 | 所有模型 |
| **P5** | LightGBM 用 Raw 输入 | ❌ 禁止 StandardScaler | benchmark_hub, lightgbm_hub | 树模型 |
| **P6** | Ridge α 随数据量增大 | 1M: α=1e5, 100k: α=3e4 | ridge_hub §4.2 | Ridge |

---

# 6. 📎 Appendix

## 6.1 Key Numbers Reference (从 L1/L2 Hubs 同步)

### 📊 High Noise (σ=1) Baseline Comparison

| Model | 32k R² | 100k R² | 1M R² | Source Hub |
|-------|--------|---------|-------|------------|
| Ridge | 0.458 | 0.486 | 0.46 | ridge_hub |
| LightGBM | 0.536 | 0.558 | - | lightgbm_hub |
| MLP | 0.498 | 0.551 | - | NN_hub |
| Oracle MoE (9 bin) | - | - | 0.625 | moe_hub, scaling_hub |

### 📊 MoE Key Metrics

| Metric | Value | Condition | Source |
|--------|-------|-----------|--------|
| Oracle MoE R² | 0.6249 | 1M, noise=1, 9 bins | moe_hub |
| Oracle MoE ΔR² | +0.1637 | vs Global Ridge | moe_hub |
| Soft Gate ρ | 1.00 | 保住 100% 增益 | moe_hub |
| [M/H] 贡献 | 68.7% | MoE 收益来源 | moe_hub |

### 📊 Scaling Key Metrics

| Metric | Value | Condition | Source |
|--------|-------|-----------|--------|
| Ridge 1M vs 100k | +2.44% | noise=1 | scaling_hub |
| LightGBM 32k→100k | +13.4% | noise=2 | scaling_hub |
| Ridge α (1M) | 100,000 | noise=1 | ridge_hub |

---

## 6.2 Changelog

| Date | Change | Sections |
|------|--------|----------|
| 2025-12-24 | 创建 Master Hub | All |
| 2025-12-24 | 从 L1 hubs 同步战略问题和答案 | §1, §2 |
| 2025-12-24 | 汇合全局洞见 I1-I5 | §3 |
| 2025-12-24 | 添加推荐路线和优先行动 | §4 |
| 2025-12-24 | 同步跨主题设计原则 | §5 |

---

> **Template Usage:**
> 
> ## Master Hub 职责
> - ✅ **Do:** 汇总 L1 战略结论，提供全局视角，推荐研究路线
> - ❌ **Don't:** 详细实验记录 (→ exp.md)，具体假设验证 (→ L1/L2 hubs)
> 
> ## Update Triggers
> - 当 L1 Hub 的 **§2 Answer Key** 战略结论改变时 → 更新 §1, §2
> - 当 L1 Hub 的 **§3 Insight Confluence** 有重大发现时 → 更新 §3
> - 每周 Review 时 → 更新 §4 Recommended Routes

---

*Last Updated: 2025-12-24*

