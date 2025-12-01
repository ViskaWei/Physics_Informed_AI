# LightGBM · log_g 实验主笔记（截至 2025-11-30）

- 本目录：`logg/lightgbm/`
- 最近更新时间：2025-11-30
- 写作风格参考：`/home/swei20/VIT/docs`
- 说明：此文件是该子目录所有实验的主索引、核心结论与未来计划。

---

# 目录

0. [问题重述](#0-问题重述)
1. [研究目标](#1-研究目标)
2. [实验地图（含超链接）](#2-实验地图含超链接)
3. [核心发现（中文·专业·压缩）](#3-核心发现中文专业压缩)
4. [关键图表](#4-关键图表)
5. [下一步实验计划](#5-下一步实验计划)
6. [附录：所有子实验索引](#6-附录所有子实验索引)

---

# 0. 问题重述

## 0.1 核心研究问题

> **LightGBM 作为非线性 baseline，能否为 $\log g$ 预测设定一个"黄金标准"？其最优配置是什么？**

LightGBM 在 $\log g$ 预测中的定位：
- **noise=0 时的性能上界**：$R^2 = 0.998$，证明非线性模型能完美学习
- **noise=1.0 时的 32k SOTA**：$R^2 = 0.536$，是所有 NN 必须超越的目标

## 0.2 对 NN 设计的直接影响

| 如果 NN... | 则说明... |
|-----------|----------|
| 在 32k 数据下超过 LightGBM | NN 有明确优势，可继续深入 |
| 在 32k 数据下低于 LightGBM | 需要更多数据或更好的架构 |
| 与 LightGBM 持平 | 可以考虑集成或知识蒸馏 |

## 0.3 超参数优化的价值

理解 LightGBM 的最优配置，可以：
- 为 NN 的 **learning rate schedule** 提供参考
- 揭示 **模型复杂度** 与性能的关系
- 识别 **Pareto 最优** 的效率-性能权衡点

---

# 1. 研究目标

本目录聚焦于 **LightGBM（梯度提升树）在 $\log g$ 回归任务上的超参数优化**，主要回答：

- LightGBM 的最优超参数配置是什么？
- `learning_rate`、`num_leaves`、`max_depth`、`n_estimators` 各自对性能的影响有多大？
- 是否存在明确的效率-性能 Pareto 最优前沿？
- LightGBM 能否作为 NN 的强 baseline？

与 $\log g$ 总目标的关系：LightGBM 在所有噪声水平下表现最优（$R^2=0.998$ @ noise=0），是 NN 必须超越的**黄金 baseline**。

---

# 2. 实验地图（含超链接）

| 实验ID | 文件/目录 | 类型 | 关键配置 | 主要指标 | 链接 |
|-------|-----------|------|----------|----------|------|
| E01 | exp_lightgbm_hyperparam_sweep_20251129.md | 超参数网格搜索 | 180 配置组合, 32k 数据 | $R^2_{\text{max}}=0.9982$ | [详情](exp_lightgbm_hyperparam_sweep_20251129.md) |
| img/ | img/ | 图表目录 | 6 张核心图表 | - | [目录](img/) |

> 表格按重要性排序。

---

# 3. 核心发现（中文·专业·压缩）

## 3.1 宏观结论

| 结论 | 数值证据 | 设计启示 |
|------|---------|---------|
| **`learning_rate` 最关键** | 与 $R^2$ 相关系数 +0.491 | 必须使用 0.05 或 0.1 |
| **存在最优复杂度区间** | `num_leaves=31`, `depth=7` 最佳 | 超过 128 叶子会过拟合 |
| **收益递减明显** | 超过 60K 总叶子数后提升有限 | 不必追求极致复杂度 |
| **`n_estimators` 影响最小** | 1000→2000 仅 +0.11%，时间 +43% | 1000 棵树足够 |

## 3.2 推荐配置

| 场景 | 配置 | 预期 $R^2$ | 训练时间 |
|------|------|-----------|----------|
| **默认推荐** | n=1000, leaves=31, depth=7, lr=0.05 | 0.9976 | 37s |
| **最优性能** | n=2000, leaves=31, depth=7, lr=0.05 | 0.9982 | 68s |
| **快速原型** | n=500, leaves=31, depth=5, lr=0.1 | 0.9970 | 15s |
| **极速配置** | n=1000, leaves=128, depth=3, lr=0.1 | 0.9955 | 11s |

## 3.3 超参数重要性排序

| 排名 | 超参数 | 与 $R^2$ 相关系数 | 影响程度 |
|:----:|--------|:-----------------:|:--------:|
| 1 | `learning_rate` | **+0.491** | 最关键 |
| 2 | `num_leaves` | +0.141 | 中等 |
| 3 | `max_depth` | +0.124 | 中等 |
| 4 | `n_estimators` | +0.080 | 较低 |

## 3.4 关键数字速查

| 指标 | 值 |
|------|-----|
| 最优 $R^2$ | **0.9982** |
| 最优 MAE | **0.0075** |
| 性价比最优配置 $R^2$ | 0.9976 @ 37s |
| 最快配置 $R^2$ | 0.9955 @ 11s |
| 搜索空间 | 180 配置 |
| 数据量 | 32k |

---

# 4. 关键图表

> 所有图表位于 `img/` 子目录：

| 图表 | 描述 | 文件 |
|------|------|------|
| 误差 vs 模型大小 | 总叶子数与测试误差关系 | [1_error_vs_model_size.png](img/1_error_vs_model_size.png) |
| 误差 vs 超参数 | 各超参数的箱线图 | [2_error_vs_hyperparams.png](img/2_error_vs_hyperparams.png) |
| 超参数热力图 | 组合配置的 $R^2$ 热力图 | [3_hyperparameter_heatmap.png](img/3_hyperparameter_heatmap.png) |
| 效率 vs 性能 | Pareto 前沿分析 | [4_efficiency_vs_performance.png](img/4_efficiency_vs_performance.png) |
| 模型大小 vs 时间 | 复杂度与训练时间关系 | [5_model_size_vs_time.png](img/5_model_size_vs_time.png) |
| 收益递减曲线 | 时间投入的边际收益 | [6_time_vs_r2_diminishing_returns.png](img/6_time_vs_r2_diminishing_returns.png) |

---

# 5. 下一步实验计划

| 优先级 | 方向 | 具体任务 | 预期收益 |
|--------|------|----------|----------|
| **P0** | 噪声鲁棒性 | 最优配置在 $\sigma=0.5, 1.0, 2.0$ 下的性能衰减 | 量化抗噪能力 |
| **P0** | 100k 公平比较 | 用 100k 数据重新训练 | 与 NN 公平对比 |
| **P1** | 特征选择 | 结合 Top-K 测试是否能减少过拟合 | 可能提升高噪声性能 |
| **P1** | Early Stopping | 使用验证集早停 | 进一步减少训练时间 |

---

# 6. 附录：所有子实验索引

## 6.1 完整实验列表

| 文件 | 日期 | 主题 | 配置数 | 状态 |
|------|------|------|--------|------|
| [exp_lightgbm_hyperparam_sweep_20251129.md](exp_lightgbm_hyperparam_sweep_20251129.md) | 2025-11-29 | 超参数网格搜索 | 180 | ✅ 完成 |

## 6.2 相关外部文件

| 类型 | 路径 |
|------|------|
| 原始数据 | `/home/swei20/VIT/results/lightgbm_sweep/sweep_results.csv` |
| 可视化图 | `/home/swei20/VIT/results/lightgbm_sweep/professional_viz/` |
| Sweet Spot 分析 | `/home/swei20/VIT/results/lightgbm_sweep/sweetspot_config.csv` |

## 6.3 与其他目录的关联

| 目录 | 关联主题 | 链接 |
|------|----------|------|
| `ridge/` | 线性 baseline 对比 | [ridge_main](../ridge/ridge_main_20251130.md) |
| `NN/` | NN vs LightGBM 对比 | [NN_main](../NN/NN_main_20251130.md) |
| `noise/` | TopK 特征选择 + LightGBM | [noise_main](../noise/noise_main_20251130.md) |

## 6.4 推荐代码配置

```python
# ⭐ 推荐默认配置
LIGHTGBM_DEFAULT = {
    'n_estimators': 1000,
    'num_leaves': 31,
    'max_depth': 7,
    'learning_rate': 0.05,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42
}
# 预期: R² ≈ 0.9976, MAE ≈ 0.0099, Time ≈ 37s
```

---

*最后更新: 2025-11-30*  
*总配置数: 180 个超参数组合*  
*核心发现: learning_rate 是最关键超参数，最优配置可达 $R^2=0.9982$*

