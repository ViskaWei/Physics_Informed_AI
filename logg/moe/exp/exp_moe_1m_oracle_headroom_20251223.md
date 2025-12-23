# 📘 Experiment Report: Oracle MoE Headroom @ noise=1

---
> **Name:** Oracle MoE 结构 Headroom @ 1M + noise=1  
> **ID:**  `MOE-20251223-oracle-headroom-01`  
> **Topic ｜ MVP:** `moe` ｜ MVP-16O (P0 最高优先级)  
> **Author:** Viska Wei  
> **Date:** 2025-12-23  
> **Project:** `VIT`  
> **Status:** 🔴 Ready
---
## 🔗 Upstream Links

| Type | Link | Description |
|------|------|-------------|
| 🧠 Hub | [`moe_hub_20251203.md`](../moe_hub_20251203.md) | H-16O.1~3, Q10.3 |
| 🗺️ Roadmap | [`moe_roadmap_20251203.md`](../moe_roadmap_20251203.md) | Phase 16 |
| 📋 Kanban | `status/kanban.md` | Phase 16 P0 三件套 |
| 📚 Scaling Hub | [`scaling_hub_20251222.md`](../../scaling/scaling_hub_20251222.md) | H4.1 结构上限 |

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


## ⚡ 核心结论速览（供 main 提取）

> ⏳ **实验完成后填写**

### 一句话总结

> TODO

### 对假设的验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H-16O.1: Oracle MoE R² > 0.55? | ⏳ | TODO |
| H-16O.2: ΔR² (Oracle - Global) ≥ 0.05? | ⏳ | TODO |
| H-16O.3: Metal-poor bins 受损更严重? | ⏳ | TODO |

### 设计启示（1-2 条）

| 启示 | 具体建议 |
|------|---------|
| TODO | TODO |

### 关键数字

| 指标 | 值 |
|------|-----|
| R²_oracle (9 bin) | TODO |
| R²_global | TODO |
| ΔR² (Oracle - Global) | TODO |
| 最差 bin | TODO |

---

# 1. 🎯 目标

## 1.1 实验目的

> **核心问题**：MoE 这条路在 noise=1 还有没有结构红利？

**回答的问题**：
1. Oracle MoE (用真值路由) 能达到什么 R²？
2. Oracle MoE - Global Ridge 的差值有多少？ ≥0.05 说明结构红利明显
3. 哪些 bins 在高噪声下受损最严重？

**对应 hub.md 的**：
- 验证假设：H4.1, H-16O.1, H-16O.2, H-16O.3
- 问题树：Q10.3

**决策规则**：
- 若 ΔR² < 0.02 → MoE 不是主要杠杆，应直接上 CNN/表示学习
- 若 ΔR² ≥ 0.05 → 结构红利仍在，MoE 值得做（保底收益）

## 1.2 预期结果

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| 结构红利明显 | ΔR² ≥ 0.05 | MoE 值得做 |
| 结构红利小 | ΔR² < 0.02 | 放弃 MoE，直接上 CNN |
| Metal-poor 更差 | Bin0/3/6 ΔR² < 平均 | 与预期一致 |

---

# 2. 🧪 实验设计

## 2.1 数据

### 数据来源与规模

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **数据来源** | BOSZ 合成光谱 | mag205_225_lowT_1M |
| **训练样本数** | 100k~1M | 可先 100k 快速跑，再上 1M |
| **测试样本数** | 5k~20k | 扩大 test 提高统计可信度 |
| **特征维度** | ~7200 | 全波段光谱 |
| **目标参数** | log_g | 主目标 |
| **分 bin 参数** | Teff, [M/H] | 9 bins = 3×3 |

### 噪声配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **噪声类型** | gaussian | 高斯噪声 |
| **噪声水平 σ** | 1.0 | 高噪声场景 |
| **应用范围** | all | 训练和测试都加噪声 |

### 9 Bin 定义（沿用 noise=0.2 的成熟配置）

| Bin | Teff 区间 | [M/H] 区间 | 描述 |
|-----|-----------|------------|------|
| Bin0 | Cool | Metal-poor | 最难 |
| Bin1 | Cool | Solar | |
| Bin2 | Cool | Metal-rich | |
| Bin3 | Mid | Metal-poor | 最难 |
| Bin4 | Mid | Solar | |
| Bin5 | Mid | Metal-rich | |
| Bin6 | Hot | Metal-poor | 最难 |
| Bin7 | Hot | Solar | |
| Bin8 | Hot | Metal-rich | |

## 2.2 模型与算法

### Oracle MoE

Oracle MoE 使用真值 (Teff, [M/H]) 进行路由，是 MoE 的理论上限：

$$
\hat{y} = \sum_{k=1}^{9} \mathbf{1}[z \in \text{Bin}_k] \cdot f_k(x)
$$

其中 $f_k$ 是第 k 个 Ridge expert，$z$ 是真值参数。

### Ridge Expert

每个 bin 独立训练一个 Ridge Regressor：
$$
\hat{y}_k = \mathbf{w}_k^\top \mathbf{x}
$$

**超参数**：
- alpha：沿用 MVP-1.4 的最优 α = 1e5（或每个 bin 独立扫描）

### Global Ridge Baseline

全局 Ridge（不分 bin）作为 baseline：
$$
\hat{y} = \mathbf{w}^\top \mathbf{x}
$$

## 2.3 超参数配置

### 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **Ridge α** | 1e5 | 沿用 MVP-1.4 最优值 |
| **训练数据** | 100k 或 1M | 先 100k 快速验证 |
| **测试数据** | 5k~20k | 大 test set |
| **random_seed** | 42, 123, 456, 789, 2024 | 多 seed |

### 本实验扫描的参数

| 扫描参数 | 扫描范围 | 固定参数 |
|---------|---------|---------|
| 无（固定配置验证） | - | α=1e5, 9 bins |

## 2.4 评价指标

| 指标 | 公式 | 用途 |
|------|------|------|
| **R²_oracle** | 9-expert MoE with true routing | Oracle 上限 |
| **R²_global** | 全局 Ridge | Baseline |
| **ΔR²** | R²_oracle - R²_global | 结构 headroom |
| **per-bin R²** | 每个 bin 的局部 R² | 定位困难区域 |

---

# 3. 📊 实验图表

> ⏳ **实验完成后填写**

### 图 1：Oracle MoE vs Global Ridge

![图片](../img/TODO.png)

**Figure 1. Oracle MoE 与 Global Ridge 的性能对比**

**关键观察**：
- TODO

---

### 图 2：Per-bin R² 分布

![图片](../img/TODO.png)

**Figure 2. 各 bin 的 R² 分布（热图或柱状图）**

**关键观察**：
- TODO

---

### 图 3：noise=0.2 vs noise=1 对比

![图片](../img/TODO.png)

**Figure 3. 高噪声下各 bin 的 R² 衰减**

**关键观察**：
- TODO

---

# 4. 💡 关键洞见

> ⏳ **实验完成后填写**

## 4.1 宏观层洞见

TODO

## 4.2 模型层洞见

TODO

## 4.3 实验层细节洞见

TODO

---

# 5. 📝 结论

> ⏳ **实验完成后填写**

## 5.1 核心发现

TODO

## 5.2 关键结论（2-4 条）

TODO

## 5.3 设计启示

TODO

## 5.4 物理解释

> 预期 Metal-poor bins 更差的原因：
> - 低金属丰度时谱线更弱，SNR 更低
> - 可用的特征更少（金属线稀疏）
> - 对噪声更敏感

## 5.5 关键数字速查

| 指标 | 值 | 配置/条件 |
|------|-----|----------|
| R²_oracle | TODO | noise=1 |
| R²_global | TODO | noise=1 |
| ΔR² | TODO | headroom |
| 最差 bin | TODO | |

## 5.6 下一步工作

| 方向 | 具体任务 | 优先级 | 对应 MVP |
|------|----------|--------|---------|
| 如果 ΔR² ≥ 0.05 | 验证可落地 Gate | 🔴 | MVP-16G |
| 如果 ΔR² < 0.02 | 放弃 MoE，直接上 CNN | 🔴 | MVP-16CNN |
| 对比 noise=0.2 | 量化高噪声衰减 | 🟡 | 分析 |

---

# 6. 📎 附录

## 6.1 数值结果表

> ⏳ **实验完成后填写**

### Oracle MoE 总体结果

| 配置 | R²_oracle | R²_global | ΔR² |
|------|-----------|-----------|-----|
| 100k train, noise=1 | TODO | TODO | TODO |
| 1M train, noise=1 | TODO | TODO | TODO |

### Per-bin R² 结果

| Bin | Teff | [M/H] | R²_oracle | noise=0.2 参考 | 衰减 |
|-----|------|-------|-----------|----------------|------|
| Bin0 | Cool | Poor | TODO | ~0.90 | TODO |
| Bin1 | Cool | Solar | TODO | ~0.95 | TODO |
| Bin2 | Cool | Rich | TODO | ~0.98 | TODO |
| Bin3 | Mid | Poor | TODO | ~0.85 | TODO |
| Bin4 | Mid | Solar | TODO | ~0.93 | TODO |
| Bin5 | Mid | Rich | TODO | ~0.97 | TODO |
| Bin6 | Hot | Poor | TODO | ~0.80 | TODO |
| Bin7 | Hot | Solar | TODO | ~0.92 | TODO |
| Bin8 | Hot | Rich | TODO | ~0.96 | TODO |

---

## 6.2 实验流程记录

> ⏳ **实验完成后填写**

### 6.2.1 环境与配置

| 项目 | 值 |
|------|-----|
| **仓库** | `~/VIT` |
| **脚本路径** | `scripts/moe_oracle_headroom.py` |
| **输出路径** | `results/moe_oracle_headroom/` |

### 6.2.2 执行命令

```bash
# TODO: 实验执行命令
```

---

## 6.3 相关文件

| 类型 | 路径 | 说明 |
|------|------|------|
| MoE Hub | `logg/moe/moe_hub_20251203.md` | H-16O, Q10.3 |
| Scaling Hub | `logg/scaling/scaling_hub_20251222.md` | H4.1 |
| 本报告 | `logg/moe/exp/exp_moe_1m_oracle_headroom_20251223.md` | 当前文件 |
| 参考：noise=0.2 | `logg/moe/exp/exp_moe_9expert_phys_gate_20251204.md` | 低噪声结果 |

---

## 6.4 与 noise=0.2 结果对比

> 在 noise=0.2 时，已验证的结论：
> - Oracle MoE ΔR² ≈ +0.050
> - Soft routing ρ ≈ 1.00
> - Metal-poor bins (Bin3/Bin6) 是困难区域

本实验需要验证这些结论在 noise=1 下是否仍然成立。

---

> **模板使用说明**：
> 
> - ⏳ 标记的部分在实验完成后填写
> - 核心结论速览在实验完成后第一时间填写
> - 同步到 moe_hub.md §3 洞见汇合站（如有重要发现）


