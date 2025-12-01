# Ridge · log_g 实验主笔记（截至 2025-11-30）

- 本目录：`logg/ridge/`
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

> **$\log g$-flux 映射是否本质线性？正则化在其中扮演什么角色？**

这个问题决定了：
- **NN 架构选择**：如果映射本质线性，NN 的主要任务是"信息过滤"而非"非线性提取"
- **正则化策略**：最优 $\alpha$ 与噪声的关系揭示了"需要删除多少无关信息"
- **Linear shortcut 的必要性**：如果线性模型已接近上限，NN 应包含 linear shortcut

## 0.2 统计学意义

如果 noise=0 时 Ridge 达到 $R^2 \approx 1$：
$$Y \approx w^\top X + \text{const}$$

则说明 $\log g$ 是 flux 的**近线性函数**，NN 只需学习残差。

## 0.3 对 NN 设计的直接影响

| 如果... | 则... |
|---------|-------|
| 最优 $\alpha$ 随噪声增大 | Weight decay 需要与噪声挂钩 |
| noise=0 时 $R^2 \approx 1$ | Linear shortcut 是必须的 |
| 正则化总是有益 | NN 也需要强正则化 |

---

# 1. 研究目标

本目录聚焦于 **Ridge 回归作为线性 baseline** 的系统分析，核心问题是：

- $\log g$-flux 映射是否本质线性？正则化的必要性如何？
- 最优 $\alpha$ 如何随噪声水平变化？
- 特征重要性（Ridge 系数）在不同 $\alpha$ 和噪声下是否稳定？
- Error 通道 $\sigma$ 是否携带额外物理信息？

与 $\log g$ 总目标的关系：Ridge 是所有模型必须超越的**线性 baseline**，其行为揭示了 $\log g$ 信息的本质特性。

---

# 2. 实验地图（含超链接）

| 实验ID | 文件/目录 | 类型 | 关键配置 | 主要指标 | 链接 |
|-------|-----------|------|----------|----------|------|
| E01 | exp_ridge_alpha_sweep_20251127.md | α 参数扫描 | α∈[0.001,1000], noise∈[0,2] | $R^2$=0.999 @ noise=0 | [详情](exp_ridge_alpha_sweep_20251127.md) |
| E02 | exp_feature_importance_stability_20251128.md | 特征稳定性 | α×noise 矩阵 | 高噪声下高度稳定 | [详情](exp_feature_importance_stability_20251128.md) |
| E03 | exp_error_logg_20251127.md | Error 通道分析 | σ-only 输入 | $R^2$=0.91 | [详情](exp_error_logg_20251127.md) |
| E04 | exp_ridge_topk_20251129.md | Ridge TopK | K∈[50,4096] | TopK 对树模型提升有限 | [详情](exp_ridge_topk_20251129.md) |
| img/ | img/ | 图表目录 | 5 张核心图表 | - | [目录](img/) |

> 表格按重要性排序。

---

# 3. 核心发现（中文·专业·压缩）

## 3.1 宏观结论

| 结论 | 数值证据 | 设计启示 |
|------|---------|---------|
| **映射本质线性** | noise=0 时 $R^2=0.999$ | Linear shortcut 是必须的 |
| **正则化总是有益** | 即使 noise=0，α=0.001 也比 OLS 好 | 光谱含无关信息需压制 |
| **最优 α 随噪声单调增大** | 0.001 (N=0) → 1000 (N=2.0) | Weight decay 需与噪声挂钩 |
| **Error σ 携带物理信息** | σ-only 达 $R^2=0.91$ | 双通道输入 [flux, σ] |

## 3.2 最优 α 参数表

| 噪声水平 | 最优 α | 最优 $R^2$ | OLS $R^2$ | 提升 |
|----------|--------|-----------|----------|------|
| 0.0 | **0.001** | 0.999 | 0.969 | +3.1% |
| 0.1 | **1.0** | ~0.90 | 0.901 | - |
| 0.5 | **50** | ~0.67 | 0.608 | +10% |
| 1.0 | **200** | 0.458 | 0.385 | +19% |
| 2.0 | **1000** | 0.221 | 0.131 | **+68.7%** |

## 3.3 特征稳定性发现

| 发现 | 数值证据 | 设计启示 |
|------|---------|---------|
| **noise=0 是"孤岛"** | 与其他噪声相关性 <0.5 | 避免在无噪声数据上训练 |
| **高噪声下高度稳定** | noise≥0.5 相关性 >0.95 | 高噪声训练更鲁棒 |
| **α 影响较小** | 相关性 >0.85 跨 α | 特征选择不依赖 α |

## 3.4 Error 通道分析

| 结论 | 数值证据 |
|------|---------|
| σ-only 预测能力 | $R^2 = 0.91$ |
| 关系类型 | 非线性（需 MLP） |
| 物理解释 | σ 编码了恒星亮度/距离等信息 |

## 3.5 关键数字速查

| 指标 | 值 |
|------|-----|
| noise=0 最佳 $R^2$ | **0.999** (α=0.001) |
| noise=1.0 最佳 $R^2$ | **0.458** (α=200) |
| noise=2.0 最佳 $R^2$ | **0.221** (α=1000) |
| OLS vs Ridge 最大差距 | **+68.7%** @ noise=2.0 |
| σ-only $R^2$ | **0.91** |

---

# 4. 关键图表

> 所有图表位于 `img/` 子目录：

| 图表 | 描述 | 文件 |
|------|------|------|
| $R^2$ vs α (by noise) | 不同噪声下的 α 曲线 | [ridge_r2_vs_alpha_by_noise.png](img/ridge_r2_vs_alpha_by_noise.png) |
| 综合指标 vs α | MAE/RMSE/$R^2$ 三指标 | [ridge_metrics_vs_alpha_by_noise.png](img/ridge_metrics_vs_alpha_by_noise.png) |
| 特征相关性 vs α | 不同 α 下特征重要性相关性 | [feature_importance_correlation_vs_alpha.png](img/feature_importance_correlation_vs_alpha.png) |
| 特征相关性详细版 | 带误差条的详细分析 | [feature_importance_correlation_vs_alpha_detailed.png](img/feature_importance_correlation_vs_alpha_detailed.png) |
| 最优 α 分析 | 最优 α 的稳定性分析 | [feature_importance_optimal_alpha_analysis.png](img/feature_importance_optimal_alpha_analysis.png) |

---

# 5. 下一步实验计划

| 优先级 | 方向 | 具体任务 | 预期收益 |
|--------|------|----------|----------|
| **P0** | 100k 数据 | 用 100k 数据重新训练 Ridge | 与 NN 公平比较 |
| **P1** | Ridge + Error | [flux, σ] 双输入 Ridge | 可能提升高噪声性能 |
| **P1** | Elastic Net | L1+L2 混合正则化 | 稀疏性+稳定性 |
| **P2** | 分区间分析 | 按 $\log g$ 区间分析误差分布 | 诊断模型盲点 |

---

# 6. 附录：所有子实验索引

## 6.1 完整实验列表

| 文件 | 日期 | 主题 | 状态 |
|------|------|------|------|
| [exp_ridge_alpha_sweep_20251127.md](exp_ridge_alpha_sweep_20251127.md) | 2025-11-27 | α 参数扫描 | ✅ 完成 |
| [exp_feature_importance_stability_20251128.md](exp_feature_importance_stability_20251128.md) | 2025-11-28 | 特征稳定性 | ✅ 完成 |
| [exp_error_logg_20251127.md](exp_error_logg_20251127.md) | 2025-11-27 | Error 通道分析 | ✅ 完成 |
| [exp_ridge_topk_20251129.md](exp_ridge_topk_20251129.md) | 2025-11-29 | Ridge TopK | ✅ 完成 |

## 6.2 相关外部文件

| 类型 | 路径 |
|------|------|
| VIT 总结 | `/home/swei20/VIT/docs/summaries/ridge_comprehensive.md` |
| 线性回归总结 | `/home/swei20/VIT/docs/summaries/linear_regression_summary.md` |

## 6.3 与其他目录的关联

| 目录 | 关联主题 | 链接 |
|------|----------|------|
| `NN/` | Residual MLP (学习 Ridge 残差) | [NN_main](../NN/NN_main_20251130.md) |
| `lightgbm/` | 树模型对比 | [lightgbm_main](../lightgbm/lightgbm_main_20251130.md) |
| `pca/` | PCA+Ridge 组合 | [pca_main](../pca/pca_main_20251130.md) |
| `noise/` | Ridge 特征选择器 | [noise_main](../noise/noise_main_20251130.md) |

## 6.4 对 NN 设计的启示

| 设计原则 | 具体建议 | 来源 |
|---------|---------|------|
| **信息过滤优先** | NN 主要任务是"学会忽略哪些像素" | Ridge α 实验 |
| **Linear shortcut** | 先用 Ridge 预测，再学习残差 | 线性本质 |
| **Weight decay 与噪声挂钩** | 噪声越大，正则化越强 | 最优 α 规律 |
| **双通道输入** | [flux, σ] 双通道 | Error 通道实验 |

---

*最后更新: 2025-11-30*  
*总实验数: 52+ 配置组合 (α × noise)*  
*核心发现: $\log g$-flux 映射本质线性，NN 的主要任务是"学会忽略无关像素"*

