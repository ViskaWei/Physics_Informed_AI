# 📘 Experiment Report: Expert 校准

---
> **Name:** Expert 校准  
> **ID:** `VIT-20251204-moe-calibration-01`  
> **Topic ｜ MVP:** `VIT` / `moe` ｜ MVP-Next-C (from moe_roadmap)   
> **Author:** Viska Wei  
> **Date:** 2025-12-04  
> **Project:** `VIT`  
> **Status:** 🔄 立项中
> **验证假设:** H-C
---

## 🔗 Upstream Links

| Type | Link | Description |
|------|------|-------------|
| 🧠 Hub | [`moe_hub_20251203.md`](./moe_hub_20251203.md) | Hypothesis: H-C (系统性 bias 校准) |
| 🗺️ Roadmap | [`moe_roadmap_20251203.md`](./moe_roadmap_20251203.md) | MVP-Next-C detailed design |
| 📋 Kanban | [`../../status/kanban.md`](../../status/kanban.md) | Experiment ID |
| 📚 Prerequisite | [`exp_moe_9expert_phys_gate_20251204.md`](./exp_moe_9expert_phys_gate_20251204.md) | MVP-9E1: Baseline (Bin3/Bin6 较弱) |

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

### 一句话总结

> **TODO**: 实验完成后填写

### 对假设的验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H-C: 最弱 bins 的误差主要是系统性 bias，affine 校准能改善？ | ⏳ | TODO |

### 设计启示（1-2 条）

| 启示 | 具体建议 |
|------|---------|
| TODO | TODO |

### 关键数字

| 指标 | 值 |
|------|-----|
| **R²_uncalibrated (baseline)** | 0.9213 |
| **R²_calibrated (new)** | ⏳ TODO |
| **ΔR²** | ⏳ TODO |
| **Bin3 R² (uncalib → calib)** | ⏳ TODO |
| **Bin6 R² (uncalib → calib)** | ⏳ TODO |

---

# 1. 🎯 目标

## 1.1 实验目的

**核心问题**：最弱 bins (Bin3/Bin6) 的误差是否主要是系统性 bias？

**回答的问题**：
- Bin3 (Mid Metal-poor) 和 Bin6 (Hot Metal-poor) 的 expert 输出是否有 bias/scale mismatch？
- 对每个 expert 做 affine 校准 $\hat{y}'_k = a_k \hat{y}_k + b_k$ 能提升多少？
- 校准后再做 soft mixing 能否提升整体 R²？

**对应 hub.md 的**：
- 验证问题：Q8.3
- 假设：H-C

**设计原则**：保持线性、可解释、低风险。

## 1.2 预期结果

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| 正常情况 | Bin3/Bin6 ΔR² ≥ 0.02，整体 R² 提升 | 验收通过 ✅ |
| 部分成功 | Bin3/Bin6 有改善但 < 0.02 | 有效但增益有限 |
| 异常情况 | 校准后 R² 下降或无变化 | 偏差不是主因，需要看特征 |

---

# 2. 🧪 实验设计

## 2.1 数据

| 配置项 | 值 |
|--------|-----|
| 训练样本数 | 32,000 (专家已训好，冻结) |
| 验证样本数 | ~5,000 (用于学习校准参数) |
| 测试样本数 | 816 (covered) |
| 重点 bins | Bin3 (Mid Metal-poor), Bin6 (Hot Metal-poor) |

**噪声模型**：noise_level = 0.2

## 2.2 校准方法

### Affine 校准

对每个 expert k，学习校准参数 $(a_k, b_k)$：

$$\hat{y}'_k = a_k \hat{y}_k + b_k$$

然后再做 soft mixing：

$$\hat{y} = \sum_k p_k \hat{y}'_k$$

### 参数学习

在 validation set 上，对每个 expert k：
1. 取该 expert 负责的样本
2. 简单线性回归：$y \sim a_k \hat{y}_k + b_k$
3. 共 9×2=18 个参数

### 对照

| 方法 | 描述 |
|------|------|
| 无校准 (baseline) | 直接 soft mixing，R²=0.9213 |
| 有校准 (new) | 先 affine 变换再 soft mixing |

## 2.3 评价指标

| 指标 | 公式 | 用途 |
|------|------|------|
| $R^2$ | $1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$ | 主要评价指标 |
| ΔR² | R²_calibrated - R²_uncalibrated | 核心改进 |
| Per-bin R² | 每个 bin 的局部 R² | **重点看 Bin3/Bin6** |
| 校准参数 | $(a_k, b_k)$ for k=0..8 | 理解 bias 性质 |

---

# 3. 📊 实验图表

> **TODO**: 实验完成后添加图表

### 图 1：Per-Expert Pred vs True (Before Calibration)

**TODO**: 9 个子图，检查是否有 bias/scale mismatch，重点看 Bin3/Bin6

### 图 2：校准参数分布

**TODO**: 9 个 expert 的 $(a_k, b_k)$ 分布，看哪些 expert 需要大校准

### 图 3：Per-Bin R² 对比 (Before vs After Calibration)

**TODO**: 柱状图对比校准前后的 per-bin R²

### 图 4：整体 R² 对比

**TODO**: Uncalibrated vs Calibrated

---

# 4. 💡 关键洞见

> **TODO**: 实验完成后填写

## 4.1 宏观层洞见

TODO

## 4.2 模型层洞见

TODO

## 4.3 实验层细节洞见

TODO

---

# 5. 📝 结论

> **TODO**: 实验完成后填写

## 5.1 核心发现

TODO

## 5.2 关键结论（3 条）

| # | 结论 | 证据 |
|---|------|------|
| 1 | TODO | TODO |
| 2 | TODO | TODO |
| 3 | TODO | TODO |

## 5.3 设计启示

TODO

## 5.4 物理解释

TODO: 为什么 Bin3/Bin6 (metal-poor) 有 bias？可能与金属线稀疏有关。

## 5.5 关键数字速查

| 指标 | 值 | 配置/条件 |
|------|-----|----------|
| Uncalibrated (baseline) | R² = 0.9213 | MVP-9E1 |
| Calibrated | TODO | TODO |
| ΔR² | TODO | TODO |
| Bin3 改善 | TODO | TODO |
| Bin6 改善 | TODO | TODO |

## 5.6 下一步工作

| 方向 | 具体任务 | 优先级 | 对应 MVP |
|------|----------|--------|---------|
| TODO | TODO | TODO | TODO |

---

# 6. 📎 附录

## 6.1 数值结果表

> **TODO**: 实验完成后填写

### Per-Expert 校准参数

| Expert k | Bin | $a_k$ | $b_k$ | R²_before | R²_after |
|----------|-----|-------|-------|-----------|----------|
| 0 | Cool Metal-poor | TODO | TODO | TODO | TODO |
| 1 | Cool Solar | TODO | TODO | TODO | TODO |
| 2 | Cool Metal-rich | TODO | TODO | TODO | TODO |
| **3** | **Mid Metal-poor** | TODO | TODO | TODO | TODO |
| 4 | Mid Solar | TODO | TODO | TODO | TODO |
| 5 | Mid Metal-rich | TODO | TODO | TODO | TODO |
| **6** | **Hot Metal-poor** | TODO | TODO | TODO | TODO |
| 7 | Hot Solar | TODO | TODO | TODO | TODO |
| 8 | Hot Metal-rich | TODO | TODO | TODO | TODO |

## 6.2 实验流程记录

### 6.2.1 环境与配置

| 项目 | 值 |
|------|-----|
| **仓库** | `~/VIT` |
| **脚本路径** | `scripts/moe_expert_calibration.py` (待创建) |
| **输出路径** | `results/moe/expert_calibration/` |
| **Python** | 3.13 |
| **关键依赖** | PyTorch, scikit-learn |

### 6.2.2 执行命令

```bash
# TODO: 实验执行时记录
cd /home/swei20/VIT
source init.sh
python -u scripts/moe_expert_calibration.py
```

### 6.2.3 运行日志摘要

> **TODO**: 实验完成后填写

---

## 6.3 相关文件

| 类型 | 路径 | 说明 |
|------|------|------|
| 主框架 | `logg/moe/moe_roadmap_20251203.md` | MoE roadmap |
| 本报告 | `logg/moe/exp_moe_expert_calibration_20251204.md` | 当前文件 |
| 图表 | `logg/moe/img/moe_calibration_*.png` | 待生成 |
| 实验代码 | `~/VIT/scripts/moe_expert_calibration.py` | 待创建 |

---

## 🔗 Cross-Repo Metadata

| Field | Value |
|-------|-------|
| **source_repo_path** | `~/VIT/results/moe/expert_calibration` |
| **script_path** | `~/VIT/scripts/moe_expert_calibration.py` |
| **output_path** | `~/VIT/results/moe/expert_calibration/` |

---

> **实验状态**：🔄 立项中，待执行

