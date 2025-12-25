# 📘 Experiment Report: Foundation Protocol
> **Name:** TODO | **ID:** `VIT-20251222-logg_1m-foundation`  
> **Topic:** `VIT` | **MVP:** MVP-0.A | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-22 | **Status:** 🔄 In Progress
```
💡 实验目的  
决定：影响的决策
```

---

---

## 🔗 Upstream Links

| Type | Link | Description |
|------|------|-------------|
| 🧠 Hub | [`logg_1m_hub_20251222.md`](../logg_1m_hub_20251222.md) | Hypothesis pyramid |
| 🗺️ Roadmap | [`logg_1m_roadmap_20251222.md`](../logg_1m_roadmap_20251222.md) | MVP design |
| 📋 Kanban | [`status/kanban.md`](../../../status/kanban.md) | Experiment queue |

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

> **[待实验完成后填写]**

### 对假设的验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| Q0.1: 什么是"low-noise"？ | ⏳ | - |
| Q0.2: 模型距上下限多远？ | ⏳ | - |

### 设计启示（1-2 条）

| 启示 | 具体建议 |
|------|---------|
| [待填写] | [待填写] |

### 关键数字

| 指标 | 值 |
|------|-----|
| Low-noise SNR 阈值 | [待计算] |
| 下限 (Ridge/LGB) logg R² | [待实验] |
| 现状 (ViT) logg R² | [待实验] |
| 上限 proxy (Window) logg R² | [待实验] |

---

# 1. 🎯 目标

## 1.1 实验目的

> **核心问题**：建立可复现的实验条件和 baseline 上下限

**MVP-0.A 回答的问题**：
- Q0.1: 什么是"low-noise"？如何量化？
- 如何定义可复现的 SNR 筛选协议？

**MVP-0.B 回答的问题**：
- Q0.2: 当前模型距离上下限有多远？
- 简单模型 vs 复杂模型差距有多大？
- 敏感窗口 vs 全谱差距有多大？

**对应 hub.md 的**：
- 验证问题：Q0.1, Q0.2
- 子假设：H1（信息利用瓶颈）的前提验证

## 1.2 预期结果

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| MVP-0.A 正常 | 得到 Top 20% SNR 阈值 | 有明确数值协议 |
| MVP-0.B 正常 | 三条 baseline 都有指标 | 可以比较 |
| 下限 ≈ 现状 | 瓶颈在信息而非模型 | 重点做输入表示/归一化 |
| 上限 > 现状 | 全谱有干扰 | 优先做窗口相关实验 |

---

# 2. 🧪 实验设计

## 2.1 数据

### 数据来源与规模

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **数据来源** | mag205_225_lowT_1M | BOSZ→PFS MR 模拟 |
| **总样本数** | 1,000,000 | 93GB |
| **特征维度** | TODO | 光谱像素数 |
| **波长范围** | 6500-9500 Å | PFS MR |
| **标签参数** | log_g | 主目标 |
| **辅助参数** | Teff, Fe_H | 范围 Teff 3750-6000K, Fe/H -1~0 |

### 噪声配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **噪声类型** | 模拟观测噪声 | 12×900s, seeing/角度变化 |
| **背景** | 天空 + 月光 | PFS 模拟 |
| **error 数组** | 有 | 每像素方差 |

### 数据预处理

| 步骤 | 配置 |
|------|------|
| **归一化** | median (6500-9500Å) (现状) |
| **特征选择** | 全谱 (baseline) / 敏感窗口 (上限 proxy) |

## 2.2 模型与算法

### MVP-0.A: SNR 计算

$$
\text{SNR} = \frac{\|\text{flux}\|}{\|\text{error}\|}
$$

或用 pipeline 中已有的 `snr` 字段。

### MVP-0.B: 三条 Baseline

| Baseline | Model | Description |
|----------|-------|-------------|
| 下限 | Ridge / LightGBM | 输入 PCA 特征 (50-200 dim) 或 line index |
| 现状 | ViT | 当前最佳配置 |
| 上限 proxy | 小模型 | 只用敏感窗口 (Ca II, H-α 等) |

## 2.3 超参数配置

### MVP-0.A

| 参数 | 值 | 说明 |
|------|-----|------|
| **SNR 计算方式** | ||flux|| / ||error|| | 每条谱一个标量 |
| **分位数** | 10%, 20%, 50% | 待计算 |
| **Low-noise 定义** | Top 20% SNR | 或 Top 10% |

### MVP-0.B: Ridge

| 参数 | 值 | 说明 |
|------|-----|------|
| **alpha** | 0.01 ~ 100 | 网格搜索 |
| **PCA dim** | 50, 100, 200 | 特征降维 |

### MVP-0.B: LightGBM

| 参数 | 值 | 说明 |
|------|-----|------|
| **n_estimators** | 2500 | 基于之前实验 |
| **learning_rate** | 0.05 | 基于之前实验 |
| **num_leaves** | 31 | 默认 |

### MVP-0.B: ViT (现状)

| 参数 | 值 | 说明 |
|------|-----|------|
| **配置** | 当前最佳 | 从现有实验提取 |

## 2.4 评价指标

| 指标 | 公式 | 用途 |
|------|------|------|
| $R^2$ (logg) | $1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$ | 主要评价指标 |
| MAE (logg) | $\frac{1}{n}\sum\|y - \hat{y}\|$ | 绝对误差 |
| RMSE (logg) | $\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}$ | 与物理量纲一致 |

**评价协议**：
- low-noise test：Top 20% SNR 样本上的指标（主指标）
- 全分布 test：全部测试样本上的指标（观测 trade-off）

---

# 3. 📊 实验图表

> [待实验完成后填充]

### 图 1：SNR 分布直方图

![图片](../img/snr_distribution.png)

**Figure 1. 1M 样本的 SNR 分布，标注分位数阈值**

**关键观察**：
- [待填写]

---

### 图 2：Baseline 对比

![图片](../img/baseline_comparison.png)

**Figure 2. 三条 baseline 的 logg R² 对比**

**关键观察**：
- [待填写]

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

> [待填写]

## 5.4 物理解释

> [待填写]

## 5.5 关键数字速查

| 指标 | 值 | 配置/条件 |
|------|-----|----------|
| Low-noise SNR 阈值 | [待计算] | Top 20% |
| 下限 R² | [待实验] | Ridge/LGB |
| 现状 R² | [待实验] | ViT |
| 上限 proxy R² | [待实验] | Window-only |

## 5.6 下一步工作

| 方向 | 具体任务 | 优先级 | 对应 MVP |
|------|----------|--------|---------|
| Fisher 分析 | 估算理论上限 | 🔴 P0 | MVP-1.1 |
| 输入改进 | SNR/Error 作为输入 | 🔴 P0 | MVP-1.2 |
| 归一化 | 三种方式对照 | 🔴 P0 | MVP-1.3 |

---

# 6. 📎 附录

## 6.1 数值结果表

> [待实验完成后填充]

### SNR 分布

| 分位数 | SNR 阈值 |
|--------|---------|
| Top 10% | [待计算] |
| Top 20% | [待计算] |
| Top 50% | [待计算] |

### Baseline 对比

| Baseline | 配置 | logg R² (low-noise) | logg R² (全分布) |
|----------|------|---------------------|-----------------|
| 下限 | Ridge | [待实验] | [待实验] |
| 下限 | LightGBM | [待实验] | [待实验] |
| 现状 | ViT | [待实验] | [待实验] |
| 上限 proxy | Window-only | [待实验] | [待实验] |

---

## 6.2 实验流程记录

### 6.2.1 环境与配置

| 项目 | 值 |
|------|-----|
| **仓库** | `~/VIT` |
| **数据路径** | [待填写] |
| **Config 路径** | [待填写] |

### 6.2.2 执行命令

```bash
# Step 1: 计算 SNR 分布
# TODO

# Step 2: 训练 Ridge baseline
# TODO

# Step 3: 训练 LightGBM baseline
# TODO

# Step 4: 提取当前 ViT 指标
# TODO

# Step 5: 训练 Window-only baseline
# TODO
```

---

## 6.3 相关文件

| 类型 | 路径 | 说明 |
|------|------|------|
| Hub | `logg/logg_1m/logg_1m_hub_20251222.md` | 知识导航 |
| Roadmap | `logg/logg_1m/logg_1m_roadmap_20251222.md` | 实验追踪 |
| 本报告 | `logg/logg_1m/exp/exp_logg_1m_foundation_protocol_20251222.md` | 当前文件 |

---

## 6.4 实验日志

| 时间 | 事件 | 处理 |
|------|------|------|
| 2025-12-22 | 创建实验框架 | - |

