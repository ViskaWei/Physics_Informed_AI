# 📘 Experiment Report: Sensitive Window vs Full Spectrum

---
> **Name:** Sensitive Window vs Full Spectrum  
> **ID:** `VIT-20251222-logg_1m-window`  
> **Topic ｜ MVP:** `VIT` | `logg_1m` | MVP-1.4  
> **Author:** Viska Wei  
> **Date:** 2025-12-22  
> **Project:** `VIT`  
> **Status:** 🔄 In Progress
---

## 🔗 Upstream Links

| Type | Link | Description |
|------|------|-------------|
| 🧠 Hub | [`logg_1m_hub_20251222.md`](../logg_1m_hub_20251222.md) | Hypothesis H2.1 |
| 🗺️ Roadmap | [`logg_1m_roadmap_20251222.md`](../logg_1m_roadmap_20251222.md) | MVP-1.4 |
| 📋 Kanban | [`status/kanban.md`](../../../status/kanban.md) | Experiment queue |
| 📚 Prerequisite | [`exp_logg_1m_fisher_upper_bound_20251222.md`](./exp_logg_1m_fisher_upper_bound_20251222.md) | MVP-1.1 (敏感窗口选择) |

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

> **[待实验完成后填写：敏感窗口是否优于/等于全谱？]**

### 对假设的验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H2.1.1: 敏感窗口-only logg R² ≥ 全谱 R²？ | ⏳ | - |
| Q1.2: logg 敏感窗口 vs 全谱，哪个更准？ | ⏳ | - |
| Q1.3: 无关波段是否在拖累优化？ | ⏳ | - |

### 设计启示（1-2 条）

| 启示 | 具体建议 |
|------|---------|
| [待填写] | [待填写] |

### 关键数字

| 指标 | 值 |
|------|-----|
| 全谱 logg R² | [baseline] |
| 敏感窗口 logg R² | [待实验] |
| Δ R² | [待计算] |

---

# 1. 🎯 目标

## 1.1 实验目的

> **核心问题**：全谱里的无关波段是否在拖累 logg 优化？

**回答的问题**：
- Q1.2: logg 敏感窗口 vs 全谱，哪个更准？
- Q1.3: 无关波段是否在拖累优化？
- 是否需要做自适应 token/窗口注意力 bias？

**对应 hub.md 的**：
- 验证问题：Q1.2, Q1.3
- 子假设：H2.1（物理敏感窗口比全谱更有效）

**物理背景**：
- logg 信息主要集中在少数特定谱线区域：
  - **Ca II triplet**: 8498, 8542, 8662 Å（压力敏感）
  - **H-α**: 6563 Å（线翼形状）
  - **Mg I**: 8807 Å
  - **分子带**：取决于 Teff
- 全谱大量区域对 logg 是噪声（信息≈0）
- 这些区域可能拖累优化、增加过拟合风险

## 1.2 预期结果

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| 正常情况 | 窗口 R² ≥ 全谱 R² | 验证干扰假设 |
| 窗口显著更好 | 全谱有严重干扰 | 优先做窗口相关优化 (Phase 2) |
| 差不多 | 干扰不严重 | 保持全谱，做其他优化 |
| 窗口更差 | 缺少上下文不行 | 需要多尺度方案 |

---

# 2. 🧪 实验设计

## 2.1 数据

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **数据来源** | mag205_225_lowT_1M | BOSZ→PFS MR 模拟 |
| **训练样本** | 50,000 - 200,000 | 方向判定用 |
| **测试集** | low-noise test (Top 20% SNR) | 来自 MVP-0.A |
| **波长范围** | 6500-9500 Å | PFS MR |
| **标签参数** | log_g | 主目标 |

### 敏感窗口定义

**方案 A：基于物理先验**

| 谱线 | 波长 (Å) | 窗口宽度 (Å) | 说明 |
|------|----------|-------------|------|
| H-α | 6563 | ±30 | 氢线，线翼对 logg 敏感 |
| Ca II 8498 | 8498 | ±15 | Ca II triplet |
| Ca II 8542 | 8542 | ±15 | Ca II triplet（最强） |
| Ca II 8662 | 8662 | ±15 | Ca II triplet |
| Mg I | 8807 | ±10 | 压力敏感 |

**方案 B：基于 Fisher 分析（来自 MVP-1.1）**
- 用 ∂F/∂logg 聚合 top-K 波段窗口
- K = 5, 10, 20 个峰值区域

### 两种数据版本

| Version | 处理方式 | 说明 |
|---------|---------|------|
| **A: Window-only** | 只保留敏感窗口，其他置零或 mask | 测试干扰假设 |
| **B: Full spectrum** | 全谱 | Baseline |

## 2.2 模型与算法

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **架构** | 同一小模型 (轻量 ViT 或 MLP) | 排除模型差异 |
| **训练配置** | 固定 | 只改数据 |

## 2.3 超参数配置

### 本实验扫描的参数

| 扫描参数 | 扫描值 | 固定参数 |
|---------|--------|---------|
| 数据版本 | Window-only, Full | 模型全固定 |
| 窗口定义 | 物理先验, Fisher top-10, Fisher top-20 | |

## 2.4 评价指标

| 指标 | 公式 | 用途 |
|------|------|------|
| $R^2$ (logg) | $1 - \frac{SS_{res}}{SS_{tot}}$ | 主要比较指标 |
| MAE (logg) | $\frac{1}{n}\sum\|y - \hat{y}\|$ | 辅助指标 |

**成功标准**：R²(window) ≥ R²(full)

**→ Hypothesis Impact:**
- If window ≥ full → 下一步做自适应 token/窗口注意力 bias (MVP-2.3, MVP-3.5)
- If window < full → 暂时不做窗口相关优化

---

# 3. 📊 实验图表

> [待实验完成后填充]

### 图 1：敏感窗口 vs 全谱 logg 对比

![图片](../img/window_vs_full.png)

**Figure 1. Window-only vs Full spectrum 的 logg R² 对比**

**关键观察**：
- [待填写]

---

### 图 2：敏感窗口位置可视化

![图片](../img/sensitive_windows.png)

**Figure 2. 选定的敏感窗口在光谱上的位置**

**关键观察**：
- [待填写]

---

### 图 3：Fisher ∂F/∂logg 热力图（如用方案 B）

![图片](../img/fisher_sensitivity_map.png)

**Figure 3. 波长 vs logg 敏感度热力图**

**关键观察**：
- [待填写：哪些区域对 logg 最敏感]

---

# 4. 💡 关键洞见

> [待实验完成后填充]

## 4.1 宏观层洞见

> [待填写]

## 4.2 模型层洞见

> [待填写]

## 4.3 实验层细节洞见

> [待填写]

---

# 5. 📝 结论

## 5.1 核心发现

> [待实验完成后填写]

## 5.2 关键结论（2-4 条）

| # | 结论 | 证据 |
|---|------|------|
| 1 | [待填写] | [待填写] |
| 2 | [待填写] | [待填写] |

## 5.3 设计启示

| 原则 | 建议 | 原因 |
|------|------|------|
| [待填写] | [待填写] | [待填写] |

## 5.4 物理解释

> [待填写：为什么某些窗口对 logg 特别敏感]

## 5.5 关键数字速查

| 指标 | 值 | 配置/条件 |
|------|-----|----------|
| Full spectrum R² | [baseline] | |
| Window-only R² (物理先验) | [待实验] | 5 窗口 |
| Window-only R² (Fisher top-10) | [待实验] | |
| Window-only R² (Fisher top-20) | [待实验] | |

## 5.6 下一步工作

| 方向 | 具体任务 | 优先级 | 对应 MVP |
|------|----------|--------|---------|
| 多尺度 token | 如果窗口有效 | 🔴 P0 | MVP-2.3 |
| 窗口注意力 bias | 软引导而非硬裁剪 | 🟡 P1 | MVP-3.5 |

---

# 6. 📎 附录

## 6.1 数值结果表

> [待实验完成后填充]

### 主要结果

| Version | 窗口定义 | logg R² | logg MAE | Δ R² vs Full |
|---------|---------|---------|----------|--------------|
| Full | - | [baseline] | [baseline] | - |
| Window | 物理先验 (5窗口) | [待实验] | [待实验] | [待计算] |
| Window | Fisher top-10 | [待实验] | [待实验] | [待计算] |
| Window | Fisher top-20 | [待实验] | [待实验] | [待计算] |

### 敏感窗口详细配置

| # | 谱线 | 中心波长 (Å) | 窗口范围 (Å) | 像素数 |
|---|------|-------------|-------------|--------|
| 1 | H-α | 6563 | 6533-6593 | [待计算] |
| 2 | Ca II | 8498 | 8483-8513 | [待计算] |
| 3 | Ca II | 8542 | 8527-8557 | [待计算] |
| 4 | Ca II | 8662 | 8647-8677 | [待计算] |
| 5 | Mg I | 8807 | 8797-8817 | [待计算] |

---

## 6.2 实验流程记录

### 6.2.1 执行命令

```bash
# Step 1: 定义敏感窗口
# TODO

# Step 2: 构造 window-only 数据版本
# TODO

# Step 3: 训练 full spectrum baseline
# TODO

# Step 4: 训练 window-only 模型
# TODO

# Step 5: 在 low-noise test 上评估
# TODO
```

---

## 6.3 相关文件

| 类型 | 路径 | 说明 |
|------|------|------|
| Hub | `logg/logg_1m/logg_1m_hub_20251222.md` | H2.1 假设 |
| Fisher 分析 | `exp_logg_1m_fisher_upper_bound_20251222.md` | MVP-1.1 |

---

## 6.4 实验日志

| 时间 | 事件 | 处理 |
|------|------|------|
| 2025-12-22 | 创建实验框架 | - |

