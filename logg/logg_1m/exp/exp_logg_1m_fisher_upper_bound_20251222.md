# 📘 Experiment Report: Fisher/Sensitivity Upper Bound

---
> **Name:** Fisher Information Upper Bound Analysis  
> **ID:** `VIT-20251222-logg_1m-fisher`  
> **Topic ｜ MVP:** `VIT` | `logg_1m` | MVP-1.1  
> **Author:** Viska Wei  
> **Date:** 2025-12-22  
> **Project:** `VIT`  
> **Status:** 🔄 In Progress
---

## 🔗 Upstream Links

| Type | Link | Description |
|------|------|-------------|
| 🧠 Hub | [`logg_1m_hub_20251222.md`](../logg_1m_hub_20251222.md) | Hypothesis H1.1 |
| 🗺️ Roadmap | [`logg_1m_roadmap_20251222.md`](../logg_1m_roadmap_20251222.md) | MVP-1.1 |
| 📋 Kanban | [`status/kanban.md`](../../../status/kanban.md) | Experiment queue |
| 📚 Prerequisite | [`exp_logg_1m_foundation_protocol_20251222.md`](./exp_logg_1m_foundation_protocol_20251222.md) | MVP-0.A (Low-noise 定义) |

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

## ⚡ 核心结论速览（供 hub 提取）

> **本节在实验完成后第一时间填写。**

### 一句话总结

> **[待实验完成后填写：模型误差是否远大于 Fisher 理论上限？]**

### 对假设的验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H1.1.1: 模型误差 > 2× Fisher σ？ | ⏳ | - |
| Q1.1: logg 在 low-noise 下还能提升多少？ | ⏳ | - |

### 设计启示（1-2 条）

| 启示 | 具体建议 |
|------|---------|
| [待填写] | [待填写] |

### 关键数字

| 指标 | 值 |
|------|-----|
| 理论 σ_logg (median) | [待计算] dex |
| 模型误差 (median) | [待测量] dex |
| 误差/理论σ 比值 | [待计算] |

---

# 1. 🎯 目标

## 1.1 实验目的

> **核心问题**：当前模型距离 Fisher 理论上限有多远？

**回答的问题**：
- Q1.1: Fisher 理论上限是多少？模型差距有多大？
- 是继续优化模型，还是转向物理先验/换波段？

**对应 hub.md 的**：
- 验证问题：Q1.1
- 子假设：H1.1（模型误差远大于 Fisher σ）

## 1.2 预期结果

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| 正常情况 | 误差/理论σ > 2 | 还有巨大优化空间 → 继续 Phase 1-2 |
| 异常情况 A | 误差/理论σ ≈ 1 | 已接近信息上限 → 转向物理先验/多臂/换波段 |
| 异常情况 B | Fisher 估计不稳定 | 需要更精细的估计方法 |

---

# 2. 🧪 实验设计

## 2.1 数据

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **数据来源** | mag205_225_lowT_1M (low-noise 子集) | 依赖 MVP-0.A |
| **样本数** | 5,000 - 20,000 条 | 用于 Fisher 估计 |
| **筛选条件** | Top 20% SNR | 来自 MVP-0.A |
| **波长范围** | 6500-9500 Å | PFS MR |
| **标签参数** | log_g, Teff, Fe_H | 需要三者做近邻查找 |

## 2.2 模型与算法

### Fisher Information 估计方法

**基本思路**：用有限差分估计谱对 logg 的敏感度

**步骤**：
1. 对每条谱 $F_i$，在 label-space 找近邻：Teff, Fe/H 相近但 logg 有差异
2. 用近邻差分估计偏导数：$\frac{\partial F}{\partial \log g} \approx \frac{F_j - F_i}{\log g_j - \log g_i}$
3. 用 error 数组估计 Fisher information：

$$
I(\log g) = \sum_\lambda \left( \frac{\partial F_\lambda}{\partial \log g} \right)^2 / \sigma_\lambda^2
$$

4. 理论 Cramér-Rao 下限：

$$
\sigma_{\log g}^{\text{theory}} = \frac{1}{\sqrt{I(\log g)}}
$$

### kNN 近邻查找

| 配置项 | 值 |
|--------|-----|
| **距离度量** | 在标准化 label-space 的欧氏距离 |
| **近邻条件** | |Δ Teff| < 50K, |Δ FeH| < 0.1, |Δ logg| > 0.2 |
| **k** | 3-5 个近邻取平均 |

## 2.3 超参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **样本量** | 5k, 10k, 20k | 检查收敛性 |
| **Teff 容差** | 50 K | 近邻查找 |
| **FeH 容差** | 0.1 dex | 近邻查找 |
| **logg 最小差异** | 0.2 dex | 保证导数可估 |

## 2.4 评价指标

| 指标 | 公式 | 用途 |
|------|------|------|
| 理论 σ_logg | $1/\sqrt{I(\log g)}$ | Fisher 下限 |
| 模型误差 | $\|y - \hat{y}\|$ | 实际模型表现 |
| 比值 | 误差 / 理论σ | 判断是否接近上限 |

---

# 3. 📊 实验图表

> [待实验完成后填充]

### 图 1：理论 σ_logg 分布 vs 模型误差分布

![图片](../img/fisher_vs_model_error.png)

**Figure 1. Fisher 理论 σ 分布（蓝）vs 模型误差分布（橙）**

**关键观察**：
- [待填写：两个分布的相对位置]

---

### 图 2：误差/理论σ 比值分布

![图片](../img/error_ratio_distribution.png)

**Figure 2. 模型误差与理论σ的比值分布**

**关键观察**：
- [待填写：比值的中位数和分布形状]

---

### 图 3：理论 σ 随 Teff/logg 的变化

![图片](../img/fisher_vs_params.png)

**Figure 3. Fisher 理论 σ 在参数空间的分布**

**关键观察**：
- [待填写：哪些区域可辨识性更好/更差]

---

# 4. 💡 关键洞见

> [待实验完成后填充]

## 4.1 宏观层洞见

> [待填写：关于 logg 信息上限的认识]

## 4.2 模型层洞见

> [待填写：关于模型与上限差距的认识]

## 4.3 实验层细节洞见

> [待填写：Fisher 估计的技术细节]

---

# 5. 📝 结论

## 5.1 核心发现

> [待实验完成后填写]

**假设验证**：
- ❌ 原假设：[待填写]
- ✅ 实验结果：[待填写]

## 5.2 关键结论（2-4 条）

| # | 结论 | 证据 |
|---|------|------|
| 1 | [待填写] | [待填写] |
| 2 | [待填写] | [待填写] |

## 5.3 设计启示

### 架构/方法原则

| 原则 | 建议 | 原因 |
|------|------|------|
| [待填写] | [待填写] | [待填写] |

## 5.4 物理解释

> [待填写：为什么某些区域可辨识性好/差]

## 5.5 关键数字速查

| 指标 | 值 | 配置/条件 |
|------|-----|----------|
| 理论 σ_logg (median) | [待计算] | low-noise |
| 模型误差 (median) | [待测量] | ViT |
| 误差/理论σ 比值 | [待计算] | 决定下一步方向 |

## 5.6 下一步工作

| 方向 | 具体任务 | 优先级 | 对应 MVP |
|------|----------|--------|---------|
| [取决于比值结果] | [待填写] | [待定] | [待定] |

---

# 6. 📎 附录

## 6.1 数值结果表

> [待实验完成后填充]

### Fisher σ 统计

| 分位数 | 理论 σ_logg (dex) |
|--------|------------------|
| 10% | [待计算] |
| 50% (median) | [待计算] |
| 90% | [待计算] |

### 模型误差统计

| 分位数 | 模型误差 (dex) |
|--------|---------------|
| 10% | [待测量] |
| 50% (median) | [待测量] |
| 90% | [待测量] |

---

## 6.2 实验流程记录

### 6.2.1 执行命令

```bash
# Step 1: 筛选 low-noise 子集
# TODO

# Step 2: 计算 Fisher information
# TODO

# Step 3: 对比模型误差
# TODO
```

---

## 6.3 相关文件

| 类型 | 路径 | 说明 |
|------|------|------|
| Hub | `logg/logg_1m/logg_1m_hub_20251222.md` | H1.1 假设 |
| 前置实验 | `exp_logg_1m_foundation_protocol_20251222.md` | MVP-0.A |

---

## 6.4 实验日志

| 时间 | 事件 | 处理 |
|------|------|------|
| 2025-12-22 | 创建实验框架 | - |

