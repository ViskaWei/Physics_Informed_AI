# 📘 Experiment Report: LightGBM Noise Sweep (Learning Rate as Main Axis)

---
> **Name:** LightGBM Noise Sweep (lr 主轴)  
> **ID:** `VIT-20251204-lightgbm-noise-sweep-01`  
> **Topic ｜ MVP:** `VIT` / `lightgbm` ｜ 衍生自 main.md §5 P0 噪声鲁棒性   
> **Author:** Viska Wei  
> **Date:** 2025-12-04  
> **Project:** `VIT`  
> **Status:** 🔄 立项中
---

## 🔗 Upstream Links

| Type | Link | Description |
|------|------|-------------|
| 📊 Main | [`lightgbm_main_20251130.md`](./lightgbm_main_20251130.md) | LightGBM 主笔记 |
| 📚 Prerequisite | [`exp_lightgbm_hyperparam_sweep_20251129.md`](./exp_lightgbm_hyperparam_sweep_20251129.md) | Noiseless sweep baseline |
| 📋 Kanban | [`../../status/kanban.md`](../../status/kanban.md) | 实验队列 |

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

> **待实验完成后填写**

### 一句话总结

> **TODO**: 实验完成后填写

### 对假设的验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| Q1: 各 noise level 的 R² 上限是多少？ | ⏳ | TODO |
| Q2: 最优 lr 随 noise 变化趋势？ | ⏳ | TODO（变大=快速平均？变小=保守拟合？） |
| Q3: LightGBM vs Ridge 在小模型约束下差多少？ | ⏳ | TODO |

### 设计启示（1-2 条）

| 启示 | 具体建议 |
|------|---------|
| TODO | TODO |

### 关键数字

| 指标 | 值 |
|------|-----|
| **Baseline (noiseless)** | R² = 0.9982 |
| **R² @ noise=0.1** | ⏳ TODO |
| **R² @ noise=0.2** | ⏳ TODO |
| **R² @ noise=0.5** | ⏳ TODO |
| **R² @ noise=1.0** | ⏳ TODO |
| **最优 lr 趋势** | ⏳ TODO |

---

# 1. 🎯 目标

## 1.1 实验目的

**核心问题**：在 n_estimators ≤ 100、max_depth=-1 约束下，LightGBM 在各噪声水平的 R² 上限是多少？最优 learning_rate 如何随噪声变化？

**回答的问题**：
1. **R² 上限**：在每个 noise level 下（0.1 / 0.2 / 0.5 / 1.0），LightGBM 能做到的最高 R² 大概是多少？
2. **超参数漂移**：这些最佳配置和 noiseless 最优配置在超参数（尤其是 lr）上有什么系统性变化？
3. **Cross-noise 鲁棒性**（Bonus）：同一套超参数在不同噪声下的性能变化有多大？

**对应 main.md 的**：
- §5 下一步实验计划 → P0 噪声鲁棒性

**设计原则**：以 `learning_rate` 为主轴，`num_leaves` 和 `n_estimators` 为配角，快速扫描噪声影响。

## 1.2 预期结果

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| 正常情况 | noise=0.1 R² > 0.95, noise=1.0 R² > 0.50 | 与之前 32k SOTA (0.536) 对齐 |
| 发现 1 | 高噪声时最优 lr 变大 | 验证"快速平均噪声"假说 |
| 发现 2 | 高噪声时最优 lr 变小 | 验证"保守拟合"假说 |
| 异常情况 | noise=1.0 R² < 0.40 | 需要检查数据/代码 |

---

# 2. 🧪 实验设计

## 2.1 数据

| 配置项 | 值 |
|--------|-----|
| 训练样本数 | 32,000 |
| 验证样本数 | ~5,000 |
| 测试样本数 | ~5,000 |
| 特征维度 | 7514 (光谱) |
| 标签参数 | log_g |
| 数据划分 seed | 与 noiseless sweep 一致 |

**噪声模型**：

$$
\text{noisy\_flux} = \text{flux} + \mathcal{N}(0, \sigma^2)
$$

**Noise levels**: $\sigma \in \{0.1, 0.2, 0.5, 1.0\}$

- 训练 / 验证 / 测试都注入**同一个** noise level
- 每个 noise level 独立跑完整 sweep

## 2.2 模型与算法

### LightGBM

```python
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'max_depth': -1,  # 固定
    'learning_rate': [0.02, 0.05, 0.1],  # 主调参数
    'num_leaves': [15, 31, 63],  # 次要参数
    'n_estimators': [50, 100],  # 次要参数
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1
}
```

## 2.3 超参数配置

| 参数 | 范围/值 | 说明 |
|------|--------|------|
| **learning_rate** | {0.02, 0.05, 0.1} | **主调参数**：0.01 已证明欠拟合，0.05 是 sweet spot，0.1 看高噪声 regime |
| num_leaves | {15, 31, 63} | 次要参数：控制树复杂度 |
| n_estimators | {50, 100} | 次要参数：满足 ≤100 约束 |
| max_depth | -1 (固定) | 无限深度，由 num_leaves 控制复杂度 |

**每个 noise level 组合数**：`3 (lr) × 3 (leaves) × 2 (n_est) = 18`

**总组合数**：`18 × 4 (noise levels) = 72`

## 2.4 评价指标

| 指标 | 公式 | 用途 |
|------|------|------|
| $R^2$ | $1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$ | 主要评价指标 |
| MAE | $\frac{1}{n}\sum\|y - \hat{y}\|$ | 绝对误差参考 |
| train_time_s | - | 训练耗时 |

## 2.5 输出文件结构

```
~/VIT/results/lightgbm_noise_sweep/
├── nz0.1/
│   └── sweep_results.csv
├── nz0.2/
│   └── sweep_results.csv
├── nz0.5/
│   └── sweep_results.csv
├── nz1.0/
│   └── sweep_results.csv
├── lightgbm_noise_sweep_best_per_noise.csv  # Summary A
└── lightgbm_noise_sweep_lr_grid.csv         # Summary B
```

## 2.6 两个 Summary 视图

### Summary A: 按 noise 找全局最优配置

```
noise_level, best_val_R2, best_test_R2, learning_rate, n_estimators, num_leaves, train_time_s
0.1, ...
0.2, ...
0.5, ...
1.0, ...
```

### Summary B: 按 (noise, lr) 看 lr 的作用

```
noise_level, learning_rate, best_val_R2, best_test_R2, best_num_leaves, best_n_estimators
0.1, 0.02, ...
0.1, 0.05, ...
0.1, 0.1, ...
0.2, 0.02, ...
...
```

用途：
- 看噪声变大时，最优 lr 往上还是往下漂
- 验证"高噪声 regime 是否更需要大 lr 来快速平均噪声"

## 2.7 实验后分析重点（⚠️ 必须回答）

跑完后需要贴出 `*_best_per_noise.csv` 和 `*_lr_grid.csv` 的结果，并回答：

### 分析 1：最优 lr 随噪声的漂移趋势

| noise_level | best_lr | 趋势判断 |
|-------------|---------|---------|
| 0.1 | ⏳ | - |
| 0.2 | ⏳ | - |
| 0.5 | ⏳ | - |
| 1.0 | ⏳ | - |

**核心问题**：高噪声时最优 lr 是变大（快速平均噪声）还是变小（保守拟合）？

### 分析 2：LightGBM vs Ridge 在小模型约束下的 noise-R² 对比

| noise_level | LightGBM best R² | Ridge R² (参考) | ΔR² | 优势方 |
|-------------|------------------|-----------------|-----|--------|
| 0.1 | ⏳ | ⏳ | - | - |
| 0.2 | ⏳ | ⏳ | - | - |
| 0.5 | ⏳ | ⏳ | - | - |
| 1.0 | ⏳ | ⏳ | - | - |

**核心问题**：在 `n_estimators ≤ 100` 约束下，LightGBM 相比 Ridge 的优势有多大？高噪声时差距是收窄还是拉大？

> **注意**：Ridge R² 需要从 `logg/ridge/` 或已有实验中查找对应 noise level 的结果。

---

# 3. 📊 实验图表

> **TODO**: 实验完成后添加图表

### 图 1：R² vs Noise Level (Best Config per Noise)

**TODO**: 折线图，x 轴 noise level，y 轴 best R²

### 图 2：R² vs Learning Rate (per Noise Level)

**TODO**: 4 条线（每个 noise），x 轴 lr，y 轴 best R² for that lr

### 图 3：最优 lr 随 Noise 变化

**TODO**: 观察最优 lr 是否从 0.05 向 0.1 或 0.02 漂移

### 图 4：Cross-Noise Robustness (Optional)

**TODO**: 如果时间允许，把 noiseless 最优配置直接在各 noise 上测试

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

TODO

## 5.5 关键数字速查

| 指标 | 值 | 配置/条件 |
|------|-----|----------|
| Noiseless best | R² = 0.9982 | lr=0.05, n=2000, leaves=31, depth=7 |
| noise=0.1 best | TODO | TODO |
| noise=0.2 best | TODO | TODO |
| noise=0.5 best | TODO | TODO |
| noise=1.0 best | TODO | TODO |
| 最优 lr 趋势 | TODO | - |

## 5.6 下一步工作

| 方向 | 具体任务 | 优先级 | 对应 MVP |
|------|----------|--------|---------|
| TODO | TODO | TODO | TODO |

---

# 6. 📎 附录

## 6.1 数值结果表

> **TODO**: 实验完成后填写

### Summary A: Best Config per Noise

| noise_level | best_val_R² | best_test_R² | learning_rate | n_estimators | num_leaves | train_time_s |
|-------------|-------------|--------------|---------------|--------------|------------|--------------|
| 0.1 | | | | | | |
| 0.2 | | | | | | |
| 0.5 | | | | | | |
| 1.0 | | | | | | |

### Summary B: R²(noise, lr) Grid

| noise_level | learning_rate | best_val_R² | best_test_R² | best_num_leaves | best_n_estimators |
|-------------|---------------|-------------|--------------|-----------------|-------------------|
| 0.1 | 0.02 | | | | |
| 0.1 | 0.05 | | | | |
| 0.1 | 0.1 | | | | |
| 0.2 | 0.02 | | | | |
| 0.2 | 0.05 | | | | |
| 0.2 | 0.1 | | | | |
| 0.5 | 0.02 | | | | |
| 0.5 | 0.05 | | | | |
| 0.5 | 0.1 | | | | |
| 1.0 | 0.02 | | | | |
| 1.0 | 0.05 | | | | |
| 1.0 | 0.1 | | | | |

---

## 6.2 实验流程记录

### 6.2.1 环境与配置

| 项目 | 值 |
|------|-----|
| **仓库** | `~/VIT` |
| **脚本路径** | `scripts/lightgbm_noise_sweep_lr_main.py` (待创建) |
| **输出路径** | `results/lightgbm_noise_sweep/` |
| **Python** | 3.x |
| **关键依赖** | LightGBM, numpy, pandas, sklearn |

### 6.2.2 执行命令

```bash
# TODO: 实验执行时记录
cd /home/swei20/VIT
source init.sh
python -u scripts/lightgbm_noise_sweep_lr_main.py
```

### 6.2.3 运行日志摘要

> **TODO**: 实验完成后填写

---

## 6.3 相关文件

| 类型 | 路径 | 说明 |
|------|------|------|
| 主笔记 | `logg/lightgbm/lightgbm_main_20251130.md` | main 文件 |
| 本报告 | `logg/lightgbm/exp_lightgbm_noise_sweep_lr_20251204.md` | 当前文件 |
| 前置实验 | `logg/lightgbm/exp_lightgbm_hyperparam_sweep_20251129.md` | Noiseless sweep |
| 图表 | `logg/lightgbm/img/` | 实验图表 |
| 实验代码 | `~/VIT/scripts/lightgbm_noise_sweep_lr_main.py` | 待创建 |

---

## 🔗 Cross-Repo Metadata

| Field | Value |
|-------|-------|
| **source_repo_path** | `~/VIT/results/lightgbm_noise_sweep/` |
| **script_path** | `~/VIT/scripts/lightgbm_noise_sweep_lr_main.py` |

---

> **实验状态**：🔄 立项中，待执行

