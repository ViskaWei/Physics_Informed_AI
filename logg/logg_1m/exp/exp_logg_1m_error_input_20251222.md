# 📘 Experiment Report: SNR/Error as Input
> **Name:** TODO | **ID:** `VIT-20251222-logg_1m-error_input`  
> **Topic:** `VIT` | **MVP:** MVP-1.2 | **Project:** `VIT`  
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
| 🧠 Hub | [`logg_1m_hub_20251222.md`](../logg_1m_hub_20251222.md) | Hypothesis H1.2 |
| 🗺️ Roadmap | [`logg_1m_roadmap_20251222.md`](../logg_1m_roadmap_20251222.md) | MVP-1.2 |
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

> **[待实验完成后填写：让模型知道 error 是否提升 logg？]**

### 对假设的验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H1.2.1: SNR谱输入使 logg MAE 下降 ≥5%？ | ⏳ | - |
| H1.2.2: error 通道输入使 logg MAE 下降 ≥5%？ | ⏳ | - |

### 设计启示（1-2 条）

| 启示 | 具体建议 |
|------|---------|
| [待填写] | [待填写] |

### 关键数字

| 指标 | 值 |
|------|-----|
| Variant A (flux) logg MAE | [baseline] |
| Variant B (SNR谱) logg MAE | [待实验] |
| Variant C (flux+error) logg MAE | [待实验] |
| ΔMAE vs baseline | [待计算] |

---

# 1. 🎯 目标

## 1.1 实验目的

> **核心问题**：让模型知道哪些像素可信是否能提升 logg？

**回答的问题**：
- Q2.1: 把 error/ivar 作为输入能否提升？
- 哪种输入方式最有效？（SNR谱 vs error 通道）

**对应 hub.md 的**：
- 验证问题：Q2.1
- 子假设：H1.2（让模型知道 error 可以提升精度）

**物理背景**：
- logg 信号集中在窄特征（线翼形状）
- 噪声随波长变化（天空线、吞吐等）
- 即使整体 SNR 高，某些波段可能是"灾区"
- 不知道 error 会被局部高噪声/天空线污染

## 1.2 预期结果

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| 正常情况 | ΔMAE ≥ 5% | 至少一种 variant 有效 |
| Variant B 有效 | SNR谱输入简单有效 | 推荐作为默认输入 |
| Variant C 有效 | 2-channel 最有效 | 值得改架构 |
| 都不有效 | ΔMAE < 2% | 瓶颈不在"不知 error" |

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

### 三种输入变体

| Variant | 输入 | 公式 | 说明 |
|---------|------|------|------|
| **A** | flux | $F_\lambda$ | 现状 baseline |
| **B** | SNR谱 | $F_\lambda / (\sigma_\lambda + \epsilon)$ | 近似像素级 SNR |
| **C** | 双通道 | concat$(F_\lambda, \log \sigma_\lambda)$ | 需改 patch embedding 为 2 channel |

## 2.2 模型与算法

### ViT 配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **架构** | 当前最佳 ViT | 固定，只改输入 |
| **Variant A/B** | 1 channel | 不改架构 |
| **Variant C** | 2 channel | patch embedding 改为 2 channel |

### 修改点

**Variant B**：
```python
# 数据预处理
input = flux / (error + 1e-8)
```

**Variant C**：
```python
# 数据预处理
input = np.stack([flux, np.log(error + 1e-8)], axis=-1)  # [seq_len, 2]
# 模型：patch embedding in_channels=2
```

## 2.3 超参数配置

### 训练超参数（固定）

| 参数 | 值 | 说明 |
|------|-----|------|
| **epochs** | [与 baseline 一致] | |
| **batch_size** | [与 baseline 一致] | |
| **learning_rate** | [与 baseline 一致] | |
| **optimizer** | AdamW | |

### 本实验扫描的参数

| 扫描参数 | 扫描值 | 固定参数 |
|---------|--------|---------|
| 输入方式 | A, B, C | 其他全固定 |

## 2.4 评价指标

| 指标 | 公式 | 用途 |
|------|------|------|
| MAE (logg) | $\frac{1}{n}\sum\|y - \hat{y}\|$ | 主要比较指标 |
| $R^2$ (logg) | $1 - \frac{SS_{res}}{SS_{tot}}$ | 辅助指标 |
| ΔMAE | $(MAE_A - MAE_{B/C}) / MAE_A$ | 相对提升 |

**成功标准**：ΔMAE ≥ 5%

**可选解释性验证**：
- 可视化注意力/梯度在天空线区域的变化
- 对比 Variant A vs B/C 在高 error 波段的预测差异

---

# 3. 📊 实验图表

> [待实验完成后填充]

### 图 1：三种输入方式的 logg 指标对比

![图片](../img/error_input_comparison.png)

**Figure 1. Variant A/B/C 的 logg MAE 对比**

**关键观察**：
- [待填写]

---

### 图 2：注意力变化（可选）

![图片](../img/attention_vs_error.png)

**Figure 2. 不同输入方式下模型对天空线区域的注意力变化**

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

| 原则 | 建议 | 原因 |
|------|------|------|
| [待填写] | [待填写] | [待填写] |

## 5.4 物理解释

> [待填写：为什么让模型知道 error 有效/无效]

## 5.5 关键数字速查

| 指标 | 值 | 配置/条件 |
|------|-----|----------|
| Variant A MAE | [baseline] | flux 输入 |
| Variant B MAE | [待实验] | SNR谱 |
| Variant C MAE | [待实验] | flux+error |
| 最佳 ΔMAE | [待计算] | |

## 5.6 下一步工作

| 方向 | 具体任务 | 优先级 | 对应 MVP |
|------|----------|--------|---------|
| [取决于结果] | [待填写] | [待定] | [待定] |

---

# 6. 📎 附录

## 6.1 数值结果表

> [待实验完成后填充]

### 主要结果

| Variant | 输入 | logg R² | logg MAE | ΔMAE |
|---------|------|---------|----------|------|
| A | flux | [baseline] | [baseline] | - |
| B | SNR谱 | [待实验] | [待实验] | [待计算] |
| C | flux+error | [待实验] | [待实验] | [待计算] |

---

## 6.2 实验流程记录

### 6.2.1 执行命令

```bash
# Step 1: 准备三种数据版本
# TODO

# Step 2: 训练 Variant A (baseline)
# TODO

# Step 3: 训练 Variant B (SNR谱)
# TODO

# Step 4: 训练 Variant C (2-channel)
# TODO

# Step 5: 在 low-noise test 上评估
# TODO
```

---

## 6.3 相关文件

| 类型 | 路径 | 说明 |
|------|------|------|
| Hub | `logg/logg_1m/logg_1m_hub_20251222.md` | H1.2 假设 |
| 前置实验 | `exp_logg_1m_foundation_protocol_20251222.md` | MVP-0.A, 0.B |

---

## 6.4 实验日志

| 时间 | 事件 | 处理 |
|------|------|------|
| 2025-12-22 | 创建实验框架 | - |

