# 📘 Experiment Report: Conditional Ridge++

---
> **Name:** Conditional Ridge++ (Teff Interaction + Second Order)  
> **ID:** `VIT-20251203-moe-cond-pp-01`  
> **Topic ｜ MVP:** `VIT` | `moe` ｜ MVP-7.2  
> **Author:** Viska Wei  
> **Date:** 2025-12-03  
> **Project:** `VIT`  
> **Status:** 🔄 In Progress

---

## 🔗 Upstream Links

| 类型 | 链接 | 说明 |
|------|------|------|
| 🧠 Hub | `logg/moe/moe_hub_20251203.md` | 假设金字塔 |
| 📋 Kanban | `status/kanban.md` | 实验队列 |
| 📊 前置实验 | MVP-3.2 Conditional Ridge 1st-order | R²=0.9018, 达 MoE 80% |
| 💬 来源会话 | GPT 脑暴 2025-12-03 | 榨出剩余 20% MoE 差距 |

---

# 📑 Table of Contents

- [⚡ Key Findings](#-核心结论速览供-main-提取)
- [1. 🎯 Objective](#1--目标)
- [2. 🧪 Experiment Design](#2--实验设计)
- [3. 📊 Figures & Results](#3--实验图表)
- [4. 💡 Insights](#4--关键洞见)
- [5. 📝 Conclusions](#5--结论)
- [6. 📎 Appendix](#6--附录)

---

## ⚡ 核心结论速览（供 main 提取）

> **本节是给 main.md 提取用的摘要，实验完成后第一时间填写。**

### 一句话总结

> ⏳ 待实验完成后填写

### 对假设的验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| Teff 一阶交互项能提升多少？ | ⏳ | - |
| 交叉项 $(m \cdot t)$ 能再提升多少？ | ⏳ | - |
| 二阶项 $m^2$ 有必要吗？ | ⏳ | - |
| 最终能达到 MoE 的多少效果？ | ⏳ | - |

### 设计启示（1-2 条）

| 启示 | 具体建议 |
|------|---------|
| ⏳ | ⏳ |

### 关键数字

| 指标 | 值 |
|------|-----|
| Baseline (1st-order) R² | 0.9018 |
| + Teff 交互 R² | ⏳ |
| + 交叉项 R² | ⏳ |
| + 二阶项 R² | ⏳ |
| MoE Oracle R² | 0.9116 |
| 最终 Retention | ⏳ |

---

# 1. 🎯 目标

## 1.1 实验目的

> **核心问题**：你已经用 Conditional Ridge 1st-order 达到 MoE 80% 效果 (R²=0.9018)，剩下的 20% 差距能否通过添加 Teff 交互项/二阶项榨出来？

**回答的问题**：
- Q1: Teff 一阶交互项 $[x, m \cdot x, t \cdot x]$ 能带来多少提升？
- Q2: 交叉项 $[(m \cdot t) \cdot x]$ 能再提升多少？
- Q3: 二阶项 $[m^2 \cdot x]$ 有必要吗？
- Q4: 最终 Conditional Ridge++ 能否达到 MoE 的 ≥90% 效果？

**对应 main.md 的**：
- 验证问题：新增 Q11
- 子假设：验证 I2（Teff 贡献 42.9%）是否可通过交互项捕获

**消融思路：逐步加，不要一下子全上**
1. 先加 Teff 一阶交互
2. 再加交叉项
3. 最后考虑二阶（如果 1/2 不够）

## 1.2 预期结果

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| Teff 交互有效 | +Teff 后 ΔR² ≥ 0.005 | → 说明 Teff 确实有贡献 |
| 交叉项有效 | +交叉项后再 +0.003 | → 说明存在 m×t 交互 |
| 达到 90% MoE | 最终 R² ≥ 0.905 | → Conditional++ 可替代 MoE |
| Teff 交互无效 | +Teff 后 ΔR² < 0.003 | → Teff 真的不重要，只做 [M/H] 条件化 |

---

# 2. 🧪 实验设计

## 2.1 数据

| 配置项 | 值 |
|--------|-----|
| 训练样本数 | **100,000** |
| 验证样本数 | **10,000** |
| 测试样本数 | **10,000** |
| 特征维度 | 全谱 (4,096 dims) |
| 标签参数 | log g |
| 辅助参数 | [M/H] (m), Teff (t) |

**实验设定**：
- 基础噪声 noise = 0.2 (与 MVP-1.1 一致)
- 使用 **真值** [M/H] 和 Teff（这是 oracle 上限实验）

## 2.2 模型与算法

### 特征扩展方案（逐步消融）

**Step 0: Baseline (已有)**
$$
\phi_0(x, m) = [x, m \cdot x]
$$
- 维度：$2 \times 4096 = 8192$
- 已有结果：R² = 0.9018

**Step 1: + Teff 一阶交互**
$$
\phi_1(x, m, t) = [x, m \cdot x, t \cdot x]
$$
- 维度：$3 \times 4096 = 12288$
- 假设：Teff 也有线性调制效应

**Step 2: + 交叉项**
$$
\phi_2(x, m, t) = [x, m \cdot x, t \cdot x, (m \cdot t) \cdot x]
$$
- 维度：$4 \times 4096 = 16384$
- 假设：存在 [M/H] 和 Teff 的联合效应

**Step 3: + 二阶项（可选）**
$$
\phi_3(x, m, t) = [x, m \cdot x, t \cdot x, m^2 \cdot x, t^2 \cdot x]
$$
- 维度：$5 \times 4096 = 20480$
- 仅当 Step 1/2 不够时才考虑

### 模型

对每种特征扩展，训练单一 Ridge 回归：
$$
\hat{y} = w^T \phi(x, m, t)
$$

## 2.3 超参数配置

| 参数 | 范围/值 | 说明 |
|------|--------|------|
| Ridge α | [0.001, 0.01, 0.1, 1, 10, 100] | 每个 Step 单独调 |
| 归一化 | StandardScaler | 对 m, t 归一化到相似尺度 |
| 交叉验证 | 5-fold | 选择最优 α |

## 2.4 评价指标

| 指标 | 公式 | 用途 |
|------|------|------|
| $R^2$ | $1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$ | 主要评价指标 |
| $\Delta R^2$ vs Step 0 | $R^2_{\text{step}} - 0.9018$ | 相对基线提升 |
| MoE Retention | $\frac{R^2 - R^2_{\text{global}}}{R^2_{\text{MoE}} - R^2_{\text{global}}}$ | 达到 MoE 效果的比例 |

---

# 3. 📊 实验图表

> 实验完成后填写。

### 图 1：[预期] Step-wise Ablation Results

![图片](./img/moe_conditional_pp_ablation.png)

**Figure 1. Conditional Ridge++ 逐步消融结果**

**预期图表内容**：
- 条形图：Step 0/1/2/3 的 R²
- 水平参考线：Global Ridge baseline, MoE Oracle
- 标注每步的增量 ΔR²

**关键观察**：
- ⏳ 待填写

---

### 图 2：[预期] Feature Contribution Analysis

![图片](./img/moe_conditional_pp_contribution.png)

**Figure 2. 各交互项的贡献分析**

**关键观察**：
- ⏳ 待填写

---

# 4. 💡 关键洞见

> 实验完成后填写。

## 4.1 宏观层洞见

> ⏳

## 4.2 模型层洞见

> ⏳

## 4.3 实验层细节洞见

> ⏳

---

# 5. 📝 结论

## 5.1 核心发现

> ⏳

## 5.2 关键结论（2-4 条）

| # | 结论 | 证据 |
|---|------|------|
| 1 | ⏳ | ⏳ |

## 5.3 设计启示

### 架构/方法原则

| 原则 | 建议 | 原因 |
|------|------|------|
| ⏳ | ⏳ | ⏳ |

### ⚠️ 止损点

| 实验结果 | 下一步行动 |
|---------|-----------|
| 达到 ≥90% MoE 效果 | → 直接用 Conditional Ridge++，放弃 MoE |
| Teff 交互项 < 0.003 提升 | → Teff 真的不重要，只做 [M/H] 条件化 |
| 仍差 >5% | → 可能需要非线性（MVP-7.4）|

## 5.4 物理解释（可选）

> 如果 Teff 交互项有效，说明 log g 特征的提取确实受温度影响（不同温度的光谱结构不同）

## 5.5 关键数字速查

| 指标 | 值 | 配置/条件 |
|------|-----|----------|
| Step 0 (Baseline) | 0.9018 | $[x, m \cdot x]$ |
| Step 1 (+Teff) | ⏳ | ⏳ |
| Step 2 (+交叉项) | ⏳ | ⏳ |
| Step 3 (+二阶) | ⏳ | ⏳ |
| MoE Oracle | 0.9116 | 9 bin experts |
| MoE Retention | ⏳ | ⏳ |

## 5.6 下一步工作

| 方向 | 具体任务 | 优先级 | 对应 MVP |
|------|----------|--------|---------|
| 如果达到 ≥90% | 用 Conditional++ 替代 MoE | 🔴 | - |
| 如果仍差 | 结合 MVP-7.3 (Noise 条件化) | 🟡 | MVP-7.3 |
| 如果需要非线性 | 探索 MVP-7.4 物理窗门控 | 🟢 | MVP-7.4 |

---

# 6. 📎 附录

## 6.1 数值结果表

> 实验完成后填写。

### Step-wise Ablation Results

| Step | 特征扩展 | 维度 | R² | ΔR² vs Step 0 | MoE Retention |
|------|---------|------|-----|---------------|---------------|
| 0 (Baseline) | $[x, m \cdot x]$ | 8192 | 0.9018 | - | 80.3% |
| 1 (+Teff) | $[x, m \cdot x, t \cdot x]$ | 12288 | ⏳ | ⏳ | ⏳ |
| 2 (+交叉) | $[x, m \cdot x, t \cdot x, (mt) \cdot x]$ | 16384 | ⏳ | ⏳ | ⏳ |
| 3 (+二阶) | $[x, m \cdot x, t \cdot x, m^2 \cdot x, t^2 \cdot x]$ | 20480 | ⏳ | ⏳ | ⏳ |

### 参考基线

| 方法 | R² | 说明 |
|------|-----|------|
| Global Ridge | 0.8616 | noise=0.2, mask-aligned |
| MoE Oracle | 0.9116 | 9 bin experts |
| ΔR² (MoE - Global) | 0.050 | 目标：达到 ≥90% 即 0.045 |

---

## 6.2 实验流程记录

> 实验执行时填写。

### 6.2.1 环境与配置

| 项目 | 值 |
|------|-----|
| **仓库** | `~/VIT` |
| **Config 路径** | TBD |
| **输出路径** | TBD |
| **Python** | 3.x |
| **关键依赖** | scikit-learn, numpy |

### 6.2.2 执行命令

```bash
# TBD
```

---

## 6.3 相关文件

| 类型 | 路径 | 说明 |
|------|------|------|
| 主框架 | `logg/moe/moe_main_20251203.md` | main 文件 |
| 本报告 | `logg/moe/exp_moe_conditional_ridge_pp_20251203.md` | 当前文件 |
| 图表 | `logg/moe/img/` | 实验图表 |
| 前置实验 | `logg/moe/exp_moe_conditional_ridge_20251203.md` | 1st-order baseline |

---

## 🔗 Cross-Repo Metadata

| Field | Value |
|-------|-------|
| **experiment_id** | `VIT-20251203-moe-cond-pp-01` |
| **project** | `VIT` |
| **topic** | `moe` |
| **source_repo_path** | `~/VIT/results/moe_conditional_pp/` |
| **config_path** | TBD |
| **output_path** | TBD |

---

> **实验设计说明**：
> 
> 本实验目标是**榨出 Conditional Ridge 的极限**：
> - 如果达到 ≥90% MoE → 直接用条件线性，放弃复杂 MoE
> - 如果 Teff 交互项无效 → 简化为仅 [M/H] 条件化

