# Noise · log_g 实验主笔记（截至 2025-11-30）

- 本目录：`logg/noise/`
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

> **$\log g$ 信息在光谱中的分布有多稀疏？Top-K 特征选择能保留多少信息？噪声增强训练的效果如何？**

这个问题决定了：
- **Attention 机制设计**：如果信息高度稀疏，可以用 learnable attention 替代显式 TopK
- **Patch/Token 设计**：约需要多少 Token 才能覆盖主要信息
- **训练策略**：噪声增强训练 vs 显式特征选择的权衡

## 0.2 稀疏信息假设的验证

如果 K=1000 (24%) 即可匹配全谱性能：
$$R^2(\text{TopK}) \approx R^2(\text{Full})$$

则说明 $\log g$ 信息**高度稀疏**，可以设计更高效的特征提取策略。

## 0.3 对 NN 设计的直接影响

| 如果... | 则... |
|---------|-------|
| K=50-100 即可达到 $R^2 > 0.4$ | ViT Patch 可以聚焦于少数关键区域 |
| 噪声训练效果 >> TopK | 优先使用噪声增强训练，而非显式特征选择 |
| LGBM importance 优于 Ridge | NN 的 attention 可以学习类似的权重分布 |

---

# 1. 研究目标

本目录聚焦于 **噪声鲁棒性与 Top-K 特征选择** 两大主题，核心问题是：

- $\log g$ 信息在光谱中的分布有多稀疏？Top-K 特征能覆盖多少信息？
- 噪声增强训练（train_noise > test_noise）是否能提升泛化性能？
- 极小 K 值（K < 50）下的性能极限是什么？
- LightGBM 的 feature importance 与 Ridge 系数作为特征选择器有何差异？

与 $\log g$ 总目标的关系：为 NN 的**注意力机制/特征门控**设计提供理论依据，指导如何实现"物理先验驱动的稀疏化"。

---

# 2. 实验地图（含超链接）

| 实验ID | 文件/目录 | 类型 | 关键配置 | 主要指标 | 链接 |
|-------|-----------|------|----------|----------|------|
| E01 | exp_noise_topk_feature_selection_20251128.md | 噪声+TopK 综合 | 全面噪声×TopK 扫描 | TopK 优于 Full +0.58 | [详情](exp_noise_topk_feature_selection_20251128.md) |
| E02 | exp_topk_feature_selection_lgbm_vs_ridge_20251129.md | LGBM vs Ridge TopK | 104 配置组合 | LGBM importance 略优 | [详情](exp_topk_feature_selection_lgbm_vs_ridge_20251129.md) |
| E03 | exp_small_k_limit_20251129.md | 极小 K 极限 | K=10-200, noise=1.0 | K=50 达 $R^2=0.39$ | [详情](exp_small_k_limit_20251129.md) |
| img/ | img/ | 图表目录 | - | - | [目录](img/) |

> 表格按重要性排序。

---

# 3. 核心发现（中文·专业·压缩）

## 3.1 稀疏信息假设验证

| 结论 | 数值证据 | 设计启示 |
|------|---------|---------|
| **信息高度稀疏** | K=1000 (24%) 即可匹配全谱 | Learnable attention 可行 |
| **TopK 在高噪声下优于 Full** | noise=1.0: $\Delta R^2 = +0.58$ | 显式稀疏化有效 |
| **K=50 是实用阈值** | LGBM $R^2=0.39$, Ridge $R^2=0.24$ | 约 50 个 Token 足够 |
| **K=100 保留 92.5% 性能** | $R^2=0.49$ vs full $R^2=0.53$ | Patch 设计可聚焦少数区域 |

## 3.2 噪声增强训练验证

| 结论 | 数值证据 | 设计启示 |
|------|---------|---------|
| **噪声训练是 20 倍于 TopK** | train_noise 0→1: $\Delta R^2 \approx 0.49$ | Noise augmentation 是核心策略 |
| **最优 train/test 比例** | ~1.0-1.2 | 略高于 test noise 训练最佳 |
| **高噪声训练等效隐式 TopK** | 高噪声模型自动忽略噪声敏感像素 | 可省去显式特征选择 |

## 3.3 特征选择器对比

| 选择器 | test_noise=1.0 最佳 $R^2$ | 最优 K | 结论 |
|--------|-------------------------|--------|------|
| **LGBM importance** | 0.5658 | K=500 | 略优于 Ridge |
| **Ridge |coef|** | 0.5497 | K=2000 | 需要更多特征 |

## 3.4 关键谱线（Top-10 by LGBM importance）

| 排名 | 波长 (Å) | 元素 | 类型 | 物理意义 |
|------|----------|------|------|----------|
| 1 | 8544.3 | **Ca II** | Ca II Triplet 核心 ⭐ | 压力敏感线 |
| 2 | 8542.0 | **Ca II** | Ca II Triplet | 压力敏感线 |
| 3 | 8809.2 | **Mg I** | 重力敏感线 | 高丰度元素 |
| 4 | 8753.2 | **Fe I** | Fe I multiplet | 金属线 |
| 5 | 8185.8 | **Na I** | Na I doublet | 压力敏感线 |

## 3.5 关键数字速查

| 指标 | 值 |
|------|-----|
| TopK vs Full (noise=1.0) | **+0.58 $R^2$** |
| 最优 K (LGBM) | **500** |
| 最小 K 达 $R^2 > 0.3$ | **50** (LGBM) |
| 最小 K 达 $R^2 > 0.4$ | **100** (LGBM) |
| 非线性组合增益 (LGBM vs Ridge) | **+0.15-0.19 $R^2$** |
| 噪声训练增益 (0→1.0) | **~0.49 $R^2$** |

---

# 4. 关键图表

> 所有图表位于 `img/` 子目录：

| 图表 | 描述 | 状态 |
|------|------|------|
| TopK vs K 曲线 | 不同 K 值的 $R^2$ 变化 | TODO |
| 噪声×TopK 热力图 | train_noise × K 的性能矩阵 | TODO |
| LGBM vs Ridge 特征重叠 | Top-K 重叠度分析 | TODO |
| 关键谱线波长分布 | Top-50 波长在光谱上的分布 | TODO |

---

# 5. 下一步实验计划

| 优先级 | 方向 | 具体任务 | 预期收益 |
|--------|------|----------|----------|
| **P1** | TopK + window | 取 TopK 索引的邻域窗口 (±W=8) | 保留局部上下文 |
| **P1** | Learnable attention | 用 NN 学习 soft attention 权重 | 替代显式 TopK |
| **P1** | 物理谱线分析 | 对 Top-50 波长做天体物理解读 | 验证物理可解释性 |
| **P2** | 跨噪声泛化 | 训练噪声与测试噪声不匹配的系统分析 | 指导 noise augmentation |

---

# 6. 附录：所有子实验索引

## 6.1 完整实验列表

| 文件 | 日期 | 主题 | 配置数 | 状态 |
|------|------|------|--------|------|
| [exp_noise_topk_feature_selection_20251128.md](exp_noise_topk_feature_selection_20251128.md) | 2025-11-28 | 噪声+TopK 综合分析 | - | ✅ 完成 |
| [exp_topk_feature_selection_lgbm_vs_ridge_20251129.md](exp_topk_feature_selection_lgbm_vs_ridge_20251129.md) | 2025-11-29 | LGBM vs Ridge TopK | 104 | ✅ 完成 |
| [exp_small_k_limit_20251129.md](exp_small_k_limit_20251129.md) | 2025-11-29 | 极小 K 极限探索 | - | ✅ 完成 |

## 6.2 相关外部文件

| 类型 | 路径 |
|------|------|
| TopK LGBM 结果 | `/home/swei20/VIT/results/topk_lgbm_importance/` |
| Small K 数据 | `/home/swei20/VIT/gpt/noise/small_k_limit/` |
| 特征重要性 CSV | `/home/swei20/VIT/results/topk_lightgbm/` |

## 6.3 与其他目录的关联

| 目录 | 关联主题 | 链接 |
|------|----------|------|
| `ridge/` | Ridge 系数作为特征选择器 | [ridge_main](../ridge/ridge_main_20251130.md) |
| `lightgbm/` | LGBM importance 提取 | [lightgbm_main](../lightgbm/lightgbm_main_20251130.md) |
| `NN/` | TopK 特征输入 NN | [NN_main](../NN/NN_main_20251130.md) |
| `gta/` | EW/颜色特征设计 | [gta_main](../gta/gta_main_20251130.md) |

---

*最后更新: 2025-11-30*  
*核心发现: $\log g$ 信息高度稀疏，K=1000 (24%) 即可匹配全谱；噪声训练效果是 TopK 的 20 倍*

