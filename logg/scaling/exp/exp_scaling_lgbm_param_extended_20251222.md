# 📘 Experiment Report: LightGBM Parameter Extended Sweep
> **Name:** TODO | **ID:** `SCALING-20251222-lgbm-param-01`  
> **Topic:** `scaling` | **MVP:** MVP-1.5 | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-22 | **Status:** ⏳ Planned
```
💡 实验目的  
决定：影响的决策
```

---

---

## 🔗 Upstream Links

| Type | Link | Description |
|------|------|-------------|
| 🧠 Hub | [`scaling_hub_20251222.md`](../scaling_hub_20251222.md) | Hypothesis pyramid |
| 🗺️ Roadmap | [`scaling_roadmap_20251222.md`](../scaling_roadmap_20251222.md) | MVP design |
| 📋 Kanban | [`kanban.md`](../../../status/kanban.md) | Experiment queue |
| 📗 Previous | [`exp_scaling_ml_ceiling_20251222.md`](./exp_scaling_ml_ceiling_20251222.md) | Baseline results |

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

### 一句话总结

> **TODO：实验完成后填写**

### 对假设的验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H1.6.1: num_leaves=127/255 能提升 R² > 0.01？ | TODO | TODO |
| H1.6.2: lr=0.01/0.02 能提升 R² > 0.01？ | TODO | TODO |

### 设计启示

| 启示 | 具体建议 |
|------|---------|
| TODO | TODO |

### 关键数字

| 指标 | 值 |
|------|-----|
| 最优 num_leaves | TODO |
| 最优 learning_rate | TODO |
| 最优 min_data_in_leaf | TODO |
| 最优 R² @ 1M | TODO |
| vs 原配置的提升 | TODO |

---

# 1. 🎯 目标

## 1.1 实验目的

> **验证 LightGBM 参数空间是否探索完全，是否还能抬高 R² 上限**

**背景观察**：
- 原配置：lr=0.05, num_leaves=63, early stopping @ 50 rounds
- 1M 时实际只用了 ~1293 棵树（vs max 5000）
- 这说明"不是树不够"，而是"继续加树在验证集上不再带来泛化增益"
- 但这可能是参数配置问题，而非模型极限

**核心问题**：
1. 更大的树复杂度 (num_leaves↑) 能否提升？
2. 更小的学习率 (lr↓) + 更多树能否更精细拟合？
3. early stopping 是否"过早停"了？

## 1.2 预期 vs 实际结果

| 场景 | 预期结果 | 实际结果 | 判断 |
|------|---------|---------|------|
| num_leaves=127/255 | R² 轻微提升 0-2% | TODO | TODO |
| lr=0.02/0.01 | R² 轻微提升 0-2% | TODO | TODO |
| 固定轮数 vs early stopping | 差异 < 0.01 | TODO | TODO |

---

# 2. 🧪 实验设计

## 2.1 数据

| 配置项 | 值 |
|--------|-----|
| **数据来源** | BOSZ 模拟光谱 (mag205_225_lowT_1M) |
| **训练样本数** | 1,000,000 |
| **验证样本数** | 从训练集划分 10% 或使用更大验证集 |
| **测试样本数** | 500-1000 |
| **噪声水平** | σ = 1.0 |
| **目标变量** | log_g |

## 2.2 实验设计：3 组小网格

### Sweep 1: 树复杂度

| 参数 | 值 |
|------|-----|
| **num_leaves** | 63 (baseline), 127, 255 |
| **max_depth** | -1 (默认), 10, 12 |
| **其他参数** | 保持 baseline |

### Sweep 2: 学习率 + 树数量

| 参数 | 值 |
|------|-----|
| **learning_rate** | 0.05 (baseline), 0.02, 0.01 |
| **n_estimators** | 5000, 10000, 20000 |
| **early_stopping_rounds** | 50, 100 |

### Sweep 3: 正则化 / 防过拟合

| 参数 | 值 |
|------|-----|
| **min_data_in_leaf** | 20 (默认), 100, 500 |
| **subsample** | 0.8 (baseline), 0.6 |
| **reg_alpha** | 0, 0.1, 1.0 |
| **reg_lambda** | 0, 0.1, 1.0 |

### Sanity Check: 固定轮数

| 配置 | 目的 |
|------|------|
| 固定 n_estimators=2000, 无 early stopping | 验证 early stopping 是否"过早停" |

## 2.3 Baseline 配置（来自 MVP-1.1）

```python
baseline_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 5000,
    'early_stopping_rounds': 50,
    'verbose': -1
}
# Baseline Result: R² = 0.5709 @ 1M, trees = 1293
```

## 2.4 评价指标

| 指标 | 定义 | 用途 |
|------|------|------|
| R² | 决定系数 | 主指标 |
| 实际使用树数 | early stopping 后的树数量 | 模型复杂度 |
| 训练时间 | 秒 | 效率参考 |
| 验证曲线 | train/valid loss 随 epoch 变化 | 过拟合诊断 |

---

# 3. 📊 实验图表

### 图 1：num_leaves vs R²

*TODO: 实验完成后添加图表*

**关键观察**：
- TODO

---

### 图 2：learning_rate vs R² (with different n_estimators)

*TODO: 实验完成后添加图表*

**关键观察**：
- TODO

---

### 图 3：Training Curves (Best Config)

*TODO: 实验完成后添加图表*

**关键观察**：
- TODO

---

### 图 4：Parameter Sensitivity Heatmap

*TODO: 实验完成后添加图表*

**关键观察**：
- TODO

---

# 4. 💡 关键洞见

## 4.1 宏观层洞见

*TODO: 实验完成后填写*

## 4.2 模型层洞见

**预期可能的发现**：

1. **如果 num_leaves↑ 有效**：说明原配置欠拟合，树容量不够
2. **如果 lr↓ 有效**：说明需要更细粒度的梯度更新
3. **如果都无效**：说明 LightGBM 确实达到了极限

## 4.3 物理解释

- 高噪声下，模型容易过拟合噪声
- 但如果正则化太强，又会欠拟合真实信号
- 最优配置需要在两者之间找平衡

---

# 5. 📝 结论

## 5.1 核心发现

> **TODO: 实验完成后填写**

## 5.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | TODO | TODO |

## 5.3 设计启示

*TODO: 实验完成后填写*

## 5.4 下一步工作

| 方向 | 具体任务 | 优先级 | 对应 MVP |
|------|----------|--------|---------|
| 如果找到更优配置 | 更新 LightGBM baseline | 🔴 P0 | - |
| 如果无提升 | 确认 LightGBM 达极限 | - | 转向 MVP-2.x |

---

# 6. 📎 附录

## 6.1 数值结果表

### Sweep 1: num_leaves

| num_leaves | max_depth | R² | MAE | Trees | Train Time (s) |
|------------|-----------|-----|-----|-------|----------------|
| 63 (baseline) | -1 | 0.5709 | 0.5845 | 1293 | 1643 |
| 127 | -1 | - | - | - | - |
| 255 | -1 | - | - | - | - |
| 127 | 10 | - | - | - | - |
| 127 | 12 | - | - | - | - |

### Sweep 2: learning_rate

| lr | n_estimators | R² | MAE | Trees | Train Time (s) |
|----|--------------|-----|-----|-------|----------------|
| 0.05 (baseline) | 5000 | 0.5709 | 0.5845 | 1293 | 1643 |
| 0.02 | 5000 | - | - | - | - |
| 0.02 | 10000 | - | - | - | - |
| 0.01 | 10000 | - | - | - | - |
| 0.01 | 20000 | - | - | - | - |

### Sweep 3: 正则化

| min_data_in_leaf | subsample | R² | MAE | Trees |
|------------------|-----------|-----|-----|-------|
| 20 (baseline) | 0.8 | 0.5709 | 0.5845 | 1293 |
| 100 | 0.8 | - | - | - |
| 500 | 0.8 | - | - | - |
| 20 | 0.6 | - | - | - |

### Sanity Check: 固定轮数

| Config | R² | vs Early Stopping |
|--------|-----|-------------------|
| n=2000, no early stop | - | - |
| n=1293, early stop (baseline) | 0.5709 | baseline |

---

## 6.2 实验流程记录

### 6.2.1 环境与配置

| 项目 | 值 |
|------|-----|
| **仓库** | `~/VIT` |
| **Python** | 3.13 |
| **关键依赖** | lightgbm, scikit-learn |

### 6.2.2 执行命令

```bash
# TODO: 实验执行时填写
cd ~/VIT && source init.sh
python scripts/scaling_lgbm_param_extended.py \
    --sweep-type all \
    --output ./results/scaling_lgbm_param \
    --img-dir /home/swei20/Physics_Informed_AI/logg/scaling/img
```

### 6.2.3 运行日志摘要

```
# TODO: 实验执行时填写
```

---

## 6.3 相关文件

| 类型 | 路径 |
|------|------|
| Hub | `logg/scaling/scaling_hub_20251222.md` |
| Roadmap | `logg/scaling/scaling_roadmap_20251222.md` |
| 本报告 | `logg/scaling/exp/exp_scaling_lgbm_param_extended_20251222.md` |
| Baseline 报告 | `logg/scaling/exp/exp_scaling_ml_ceiling_20251222.md` |

---

## 🔗 Cross-Repo Metadata

| Field | Value |
|-------|-------|
| **experiment_id** | SCALING-20251222-lgbm-param-01 |
| **priority** | 🔴 P0 |
| **depends_on** | MVP-1.3 (可选，验证 plateau 后再做更有意义) |
| **blocks** | MVP-1.6, 1.7 (需要最优配置作为 baseline) |

