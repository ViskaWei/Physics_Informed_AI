# 📘 Experiment Report: Noise Continuous Conditioning
> **Name:** TODO | **ID:** `VIT-20251203-moe-noise-cont-01`  
> **Topic:** `VIT` | **MVP:** MVP-7.3 | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-03 | **Status:** 🔄 In Progress
```
💡 实验目的  
决定：影响的决策
```

---

---

## 🔗 Upstream Links

| 类型 | 链接 | 说明 |
|------|------|------|
| 🧠 Hub | `logg/moe/moe_hub_20251203.md` | 假设金字塔 |
| 📋 Kanban | `status/kanban.md` | 实验队列 |
| 📊 前置实验 | MVP-2.0 Noise-conditioned Expert | ΔR²=+0.080, 但 noise=0.5 翻车 |
| 💬 来源会话 | GPT 脑暴 2025-12-03 | H-C 假设设计 |

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
| H-C: 连续 noise 条件化能接近按档专家的平均收益？ | ⏳ | - |
| noise=0.5 的翻车是否被消除？ | ⏳ | - |
| 二阶项 $n^2$ 是否必要？ | ⏳ | - |

### 设计启示（1-2 条）

| 启示 | 具体建议 |
|------|---------|
| ⏳ | ⏳ |

### 关键数字

| 指标 | 值 |
|------|-----|
| Per-noise Expert 平均 R² | 0.7723 |
| Per-noise Expert ΔR² | +0.0797 |
| Mixed Ridge R² | 0.6926 |
| Continuous-cond R² | ⏳ |
| noise=0.5 差异 | ⏳ |

---

# 1. 🎯 目标

## 1.1 实验目的

> **核心问题**：你已证明 noise-matched expert 很强（平均 ΔR²≈+0.08），但离散分档在 noise=0.5 翻车（-0.031）。能否用连续 SNR/noise 条件化替代离散分档，消除异常点？

**回答的问题**：
- Q1: 连续 noise 条件化 $[x, n \cdot x]$ 能否接近 per-noise expert 的平均性能？
- Q2: 是否能消除 noise=0.5 的翻车现象？
- Q3: 二阶项 $[x, n \cdot x, n^2 \cdot x]$ 是否必要？
- Q4: 连续化方案在所有 noise level 上是否更稳定？

**对应 main.md 的**：
- 验证问题：新增 Q12
- 子假设：H-C

**核心洞察 (I6)**：
- Per-noise expert 平均 ΔR²=+0.080，很值钱
- 但 noise=0.5 专家反而比 mixed 差 (-0.031)
- 说明离散分档不稳定，应做"连续 SNR 条件化"

## 1.2 预期结果

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| 连续化有效 | 平均性能接近 expert（差 ≤0.01 R²） | → 用连续化替代分档 |
| noise=0.5 被修复 | noise=0.5 不再翻车（≥ mixed） | → 连续化更稳定 |
| 需要二阶项 | 二阶项 +0.005 以上提升 | → 使用二阶 |
| 连续化更差 | 效果 < mixed | → 说明 noise 确实需要离散化 |

---

# 2. 🧪 实验设计

## 2.1 数据

| 配置项 | 值 |
|--------|-----|
| 训练样本数 | **100,000** × 5 noise levels |
| 验证样本数 | **10,000** × 5 noise levels |
| 测试样本数 | **10,000** × 5 noise levels |
| 特征维度 | 全谱 (4,096 dims) |
| 标签参数 | log g |
| Noise levels | 0.0, 0.1, 0.2, 0.5, 1.0 |

**关键变量**：Noise level $n$ (作为连续条件输入)

## 2.2 模型与算法

### 对比方法

**方法 1: Per-noise Expert (已有基线)**
- 为每个 noise level 单独训练 Ridge
- 5 个独立模型
- 已有结果：平均 ΔR²=+0.0797

**方法 2: Mixed Ridge (已有基线)**
- 所有 noise level 混合训练
- 1 个模型
- 已有结果：平均 R²=0.6926

**方法 3: Continuous-conditioned (一阶)**
$$
\phi(x, n) = [x, n \cdot x]
$$
- 把 noise level 当连续变量
- 维度：$2 \times 4096 = 8192$

**方法 4: Continuous-conditioned (二阶)**
$$
\phi(x, n) = [x, n \cdot x, n^2 \cdot x]
$$
- 添加二阶项
- 维度：$3 \times 4096 = 12288$

### 训练方式

所有 noise level 的数据混合训练（像 Mixed），但特征中包含 noise level 信息：

```python
# 一阶条件化
X_cond = np.hstack([X, X * noise_level.reshape(-1, 1)])

# 二阶条件化
X_cond = np.hstack([X, X * noise_level.reshape(-1, 1), X * (noise_level**2).reshape(-1, 1)])
```

### 评估方式

在每个 noise level 上分别评估，然后计算平均

## 2.3 超参数配置

| 参数 | 范围/值 | 说明 |
|------|--------|------|
| Ridge α | [0.001, 0.01, 0.1, 1, 10, 100] | 自动选择 |
| Noise 归一化 | StandardScaler 或 [0,1] | 让 n 在合理范围 |
| 交叉验证 | 5-fold | 选择最优 α |

## 2.4 评价指标

| 指标 | 公式 | 用途 |
|------|------|------|
| $R^2_{\text{per-noise}}$ | 每个 noise level 的 R² | 分 noise 评估 |
| $R^2_{\text{avg}}$ | 5 个 noise level 的平均 | 综合性能 |
| $\Delta R^2$ | $R^2 - R^2_{\text{mixed}}$ | 相对 mixed 的提升 |
| Stability | std across noise levels | 稳定性评估 |

---

# 3. 📊 实验图表

> 实验完成后填写。

### 图 1：[预期] Per-noise R² Comparison

![图片](../img/moe_noise_continuous_comparison.png)

**Figure 1. 不同方法在各 noise level 上的 R² 对比**

**预期图表内容**：
- X 轴：Noise level (0.0, 0.1, 0.2, 0.5, 1.0)
- Y 轴：R²
- 4 条曲线：Expert, Mixed, Cond-1st, Cond-2nd
- 重点关注 noise=0.5 的表现

**关键观察**：
- ⏳ 待填写

---

### 图 2：[预期] ΔR² vs Mixed Baseline

![图片](../img/moe_noise_continuous_delta.png)

**Figure 2. 各方法相对 Mixed 的增益**

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
| 连续化接近 expert 且更稳定 | → 用连续化替代分档专家 |
| 连续化 < mixed | → 说明 noise 确实需要离散化 |
| noise=0.5 仍然翻车 | → 可能需要更复杂的建模 |

## 5.4 物理解释（可选）

> 如果连续化有效，说明 log g 特征随 SNR 的变化是平滑的（高噪声下某些特征逐渐失效）

## 5.5 关键数字速查

| 指标 | 值 | 配置/条件 |
|------|-----|----------|
| Expert 平均 R² | 0.7723 | per-noise |
| Mixed R² | 0.6926 | 混合训练 |
| Expert ΔR² | +0.0797 | vs mixed |
| Cond-1st R² | ⏳ | ⏳ |
| Cond-2nd R² | ⏳ | ⏳ |
| noise=0.5 Expert vs Mixed | -0.031 | 翻车点 |
| noise=0.5 Cond vs Mixed | ⏳ | ⏳ |

## 5.6 下一步工作

| 方向 | 具体任务 | 优先级 | 对应 MVP |
|------|----------|--------|---------|
| 结合 [M/H] 和 noise 条件化 | 多变量条件线性 | 🟡 | - |
| 如果连续化有效 | 与 MVP-7.2 合并 | 🔴 | - |

---

# 6. 📎 附录

## 6.1 数值结果表

> 实验完成后填写。

### Per-noise R² Results

| Noise | Expert R² | Mixed R² | Cond-1st R² | Cond-2nd R² |
|-------|-----------|----------|-------------|-------------|
| 0.0 | 0.9991 | 0.8207 | ⏳ | ⏳ |
| 0.1 | 0.9130 | 0.8168 | ⏳ | ⏳ |
| 0.2 | 0.8341 | 0.8001 | ⏳ | ⏳ |
| 0.5 | 0.6552 | **0.6863** | ⏳ | ⏳ |
| 1.0 | 0.4601 | 0.3393 | ⏳ | ⏳ |
| **平均** | **0.7723** | 0.6926 | ⏳ | ⏳ |

### ΔR² vs Mixed

| Noise | Expert ΔR² | Cond-1st ΔR² | Cond-2nd ΔR² |
|-------|-----------|--------------|--------------|
| 0.0 | +0.178 | ⏳ | ⏳ |
| 0.1 | +0.096 | ⏳ | ⏳ |
| 0.2 | +0.034 | ⏳ | ⏳ |
| 0.5 | **-0.031** | ⏳ | ⏳ |
| 1.0 | +0.121 | ⏳ | ⏳ |

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
| 本报告 | `logg/moe/exp_moe_noise_continuous_20251203.md` | 当前文件 |
| 图表 | `logg/moe/img/` | 实验图表 |
| 前置实验 | `logg/moe/exp_moe_noise_conditioned_20251203.md` | Expert 基线 |

---

## 🔗 Cross-Repo Metadata

| Field | Value |
|-------|-------|
| **experiment_id** | `VIT-20251203-moe-noise-cont-01` |
| **project** | `VIT` |
| **topic** | `moe` |
| **source_repo_path** | `~/VIT/results/moe_noise_continuous/` |
| **config_path** | TBD |
| **output_path** | TBD |

---

> **实验设计说明**：
> 
> 本实验目标是**修复离散分档的 noise=0.5 异常**：
> - 如果连续化更稳定 → 用连续条件化替代分档专家
> - 如果仍然翻车 → 可能需要更复杂的建模方式

