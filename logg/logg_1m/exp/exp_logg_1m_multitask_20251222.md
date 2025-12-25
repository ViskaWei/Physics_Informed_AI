# 📘 Experiment Report: Multi-task (Teff+FeH+logg)
> **Name:** TODO | **ID:** `VIT-20251222-logg_1m-multitask`  
> **Topic:** `VIT` | **MVP:** MVP-1.5 | **Project:** `VIT`  
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
| 🧠 Hub | [`logg_1m_hub_20251222.md`](../logg_1m_hub_20251222.md) | Hypothesis H4.1 |
| 🗺️ Roadmap | [`logg_1m_roadmap_20251222.md`](../logg_1m_roadmap_20251222.md) | MVP-1.5 |
| 📋 Kanban | [`status/kanban.md`](../../../status/kanban.md) | Experiment queue |
| 📚 Prerequisite | [`exp_logg_1m_foundation_protocol_20251222.md`](./exp_logg_1m_foundation_protocol_20251222.md) | MVP-0.A, 0.B |

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

> **[待实验完成后填写：多任务是否能解耦 logg 与 Teff/FeH？]**

### 对假设的验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H4.1.1: 多任务使 logg bias 减少？ | ⏳ | - |
| Q3.1: 多任务联合是否能拆分因素？ | ⏳ | - |

### 设计启示（1-2 条）

| 启示 | 具体建议 |
|------|---------|
| [待填写] | [待填写] |

### 关键数字

| 指标 | 值 |
|------|-----|
| 单任务 logg MAE | [baseline] |
| 多任务 logg MAE | [待实验] |
| Δ MAE | [待计算] |
| Bias 减少量 | [待计算] |

---

# 1. 🎯 目标

## 1.1 实验目的

> **核心问题**：logg 与 Teff/FeH 的耦合是否是精度瓶颈？

**回答的问题**：
- Q3.1: 多任务联合是否能拆分因素？
- logg 的系统偏差（随 Teff/FeH 漂移）能否减少？

**对应 hub.md 的**：
- 验证问题：Q3.1
- 子假设：H4.1（多任务联合减少 logg 的 Teff/FeH 混淆）

**物理背景**：
- logg、Teff、Fe/H 在光谱上强耦合
- 单任务回归可能把 Teff/FeH 变化误归因给 logg
- 多任务可以逼迫表示把不同因素拆开
- 如果有效，后续实验默认用多任务

## 1.2 预期结果

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| 正常情况 | logg MAE 下降 + bias 减少 | 多任务有效 |
| 多任务有效 | 后续默认用多任务 | 推荐 |
| 无明显改进 | 保持单任务 | 瓶颈不在耦合 |
| 多任务更差 | 需要更精细的权重设计 | 调整权重 |

---

# 2. 🧪 实验设计

## 2.1 数据

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **数据来源** | mag205_225_lowT_1M | BOSZ→PFS MR 模拟 |
| **训练样本** | 50,000 - 200,000 | 方向判定用 |
| **测试集** | low-noise test (Top 20% SNR) | 来自 MVP-0.A |
| **波长范围** | 6500-9500 Å | PFS MR |
| **标签参数** | log_g, Teff, Fe_H | 三个目标 |

### 标签范围

| 参数 | 范围 | 单位 |
|------|------|------|
| log g | 1-5 | dex |
| Teff | 3750-6000 | K |
| Fe/H | -1 ~ 0 | dex |

## 2.2 模型与算法

### 单任务 vs 多任务

| Version | Output | Head | Loss |
|---------|--------|------|------|
| **单任务** | logg | 1 head | MSE(logg) |
| **多任务** | logg, Teff, FeH | 3 heads (共享 backbone) | 加权 MSE |

### 多任务 Loss

$$
\mathcal{L} = w_1 \cdot MSE(\log g) + w_2 \cdot MSE(T_{eff}) + w_3 \cdot MSE([Fe/H])
$$

**权重设计**：
- 目标：让三者梯度量级接近
- 方法 1：按 target 方差反比设权重
- 方法 2：按首轮梯度量级调整
- 方法 3：使用 uncertainty weighting（学习权重）

### 模型修改

```python
class MultiTaskHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.logg_head = nn.Linear(hidden_dim, 1)
        self.teff_head = nn.Linear(hidden_dim, 1)
        self.feh_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        return {
            'logg': self.logg_head(x),
            'teff': self.teff_head(x),
            'feh': self.feh_head(x)
        }
```

## 2.3 超参数配置

### 权重配置

| 配置 | $w_{logg}$ | $w_{Teff}$ | $w_{FeH}$ | 说明 |
|------|-----------|------------|-----------|------|
| 均等 | 1.0 | 1.0 | 1.0 | baseline |
| 按方差 | [待计算] | [待计算] | [待计算] | 反比方差 |
| logg 优先 | 2.0 | 1.0 | 1.0 | 重点优化 logg |

### 本实验扫描的参数

| 扫描参数 | 扫描值 | 固定参数 |
|---------|--------|---------|
| 任务配置 | 单任务, 多任务 | 模型架构 |
| 权重配置 | 均等, 按方差, logg优先 | |

## 2.4 评价指标

| 指标 | 公式 | 用途 |
|------|------|------|
| MAE (logg) | $\frac{1}{n}\sum\|y - \hat{y}\|$ | 主要指标 |
| $R^2$ (logg) | $1 - \frac{SS_{res}}{SS_{tot}}$ | 辅助指标 |
| **Bias** | logg 误差随 Teff/FeH 的系统漂移 | 解耦效果 |

**Bias 评估方法**：
- 按 Teff 分箱（每 500K 一箱），计算各箱 logg 平均误差
- 按 FeH 分箱（每 0.25 dex 一箱），计算各箱 logg 平均误差
- 如果 bias 随 Teff/FeH 变化减少 → 解耦有效

**成功标准**：
- logg MAE 下降 ≥ 3%
- 或 bias 减少（各箱平均误差的标准差减小）

---

# 3. 📊 实验图表

> [待实验完成后填充]

### 图 1：单任务 vs 多任务 logg 指标对比

![图片](../img/multitask_comparison.png)

**Figure 1. 单任务 vs 多任务的 logg MAE/R² 对比**

**关键观察**：
- [待填写]

---

### 图 2：logg Bias 随 Teff 的变化

![图片](../img/logg_bias_vs_teff.png)

**Figure 2. logg 误差 vs Teff（单任务 vs 多任务）**

**关键观察**：
- [待填写：系统偏差是否减少]

---

### 图 3：logg Bias 随 FeH 的变化

![图片](../img/logg_bias_vs_feh.png)

**Figure 3. logg 误差 vs Fe/H（单任务 vs 多任务）**

**关键观察**：
- [待填写：系统偏差是否减少]

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

> [待填写：为什么多任务能/不能解耦]

## 5.5 关键数字速查

| 指标 | 值 | 配置/条件 |
|------|-----|----------|
| 单任务 logg MAE | [baseline] | |
| 多任务 logg MAE (均等权重) | [待实验] | |
| 多任务 logg MAE (最佳权重) | [待实验] | |
| Bias 标准差减少量 | [待计算] | |

## 5.6 下一步工作

| 方向 | 具体任务 | 优先级 | 对应 MVP |
|------|----------|--------|---------|
| 异方差回归 | 如果多任务有效 | 🟡 P1 | MVP-3.2 |
| 后续实验默认 | 如果多任务有效，后续默认用 | - | - |

---

# 6. 📎 附录

## 6.1 数值结果表

> [待实验完成后填充]

### 主要结果

| Version | 权重配置 | logg R² | logg MAE | Teff MAE | FeH MAE |
|---------|---------|---------|----------|----------|---------|
| 单任务 | - | [baseline] | [baseline] | - | - |
| 多任务 | 均等 | [待实验] | [待实验] | [待实验] | [待实验] |
| 多任务 | 按方差 | [待实验] | [待实验] | [待实验] | [待实验] |
| 多任务 | logg优先 | [待实验] | [待实验] | [待实验] | [待实验] |

### Bias 分析

| Teff 范围 (K) | 单任务 logg 误差 | 多任务 logg 误差 |
|--------------|-----------------|-----------------|
| 3750-4250 | [待计算] | [待计算] |
| 4250-4750 | [待计算] | [待计算] |
| 4750-5250 | [待计算] | [待计算] |
| 5250-5750 | [待计算] | [待计算] |
| 5750-6000 | [待计算] | [待计算] |
| **标准差** | [待计算] | [待计算] |

---

## 6.2 实验流程记录

### 6.2.1 执行命令

```bash
# Step 1: 修改模型增加多任务 head
# TODO

# Step 2: 训练单任务 baseline
# TODO

# Step 3: 训练多任务模型 (3种权重)
# TODO

# Step 4: 在 low-noise test 上评估
# TODO

# Step 5: 计算 bias 分析
# TODO
```

---

## 6.3 相关文件

| 类型 | 路径 | 说明 |
|------|------|------|
| Hub | `logg/logg_1m/logg_1m_hub_20251222.md` | H4.1 假设 |
| 前置实验 | `exp_logg_1m_foundation_protocol_20251222.md` | MVP-0.A, 0.B |

---

## 6.4 实验日志

| 时间 | 事件 | 处理 |
|------|------|------|
| 2025-12-22 | 创建实验框架 | - |

