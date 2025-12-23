# 📗 实验报告：1M + noise=1 大规模低 SNR MoE 验证

---
> **实验名称：** MVP-16A: 1M + noise=1 大规模低 SNR MoE 验证  
> **对应 MVP：** MVP-16A  
> **作者：** Viska Wei  
> **日期：** 2025-12-23  
> **数据版本：** 1M train / test TBD  
> **模型版本：** 9× Ridge + 回归 gate (Next-A 配置)  
> **状态：** 🔄 立项中  
> **验证假设：** H-16A, H-16B, H-16C, H-16D

---

## 🔗 上游追溯链接

| 类型 | 链接 | 说明 |
|------|------|------|
| 📍 Roadmap | [`moe_roadmap.md`](../moe_roadmap_20251203.md) | Phase 16 |
| 🧠 Hub | [`moe_hub.md`](../moe_hub_20251203.md) | Q10, H-16A~D |
| 📋 Kanban | [`kanban.md`](../../../status/kanban.md) | VIT-20251223-moe-1m-highnoise-01 |
| 📗 前序实验 | [`exp_moe_100k_replication`](./exp_moe_100k_replication_20251205.md) | MVP-12A R²=0.9400 (noise=0.2) |

---

## ⚡ 核心结论速览

> **一句话总结**：⏳ 实验进行中
>
> **假设验证**：
> - H-16A: 1M + noise=1 下 MoE R²仍显著优于 Global → ⏳
> - H-16B: 高噪声下 Gate 准确率仍 >60% → ⏳
> - H-16C: Metal-poor bins 在高噪声下受损更严重 → ⏳
> - H-16D: Soft routing 在高噪声下仍能保住 ≥70% oracle 增益 → ⏳
>
> **关键数字**：
> - MoE R²: ⏳
> - Global R²: ⏳
> - ρ (Soft routing): ⏳
> - Gate 准确率: ⏳

---

# 1. 🎯 目标

## 1.1 实验目的

验证 MoE 在**大规模 + 高噪声**场景下的鲁棒性。

**为什么做**：
- 当前最佳结果 (R²=0.9400) 是在 100k + noise=0.2 下获得的
- 实际应用场景可能面临：
  - **更大规模数据** (1M 级别)
  - **更低 SNR** (noise=1.0，对应 SNR~1)
- 需要验证 MoE 方法在极端条件下是否仍有价值

**核心问题**：
1. **规模效应**：1M 数据是否带来额外提升？
2. **噪声鲁棒性**：noise=1 下 MoE 是否仍优于 Global？
3. **Gate 稳定性**：高噪声下 Gate 准确率下降多少？
4. **Bin 差异**：哪些 bins 在高噪声下受损最严重？

## 1.2 预期结果

| 指标 | 预期值 | 最低可接受 | 说明 |
|------|--------|----------|------|
| **ΔR² (MoE - Global)** | > 0.03 | > 0.02 | 核心验收标准 |
| **ρ (Soft routing)** | ≥ 0.7 | ≥ 0.5 | 保住 ≥50% oracle 增益 |
| **Gate 准确率** | > 70% | > 60% | 9-class 分类 |
| **Metal-poor bins R²** | 相对下降 | - | 预期困难区域更脆弱 |

---

# 2. 🧪 实验设计

## 2.1 数据配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **训练数据规模** | 1,000,000 | 10× 100k |
| **测试数据规模** | TBD | 建议 5k-10k |
| **噪声水平** | **1.0** | 🔴 高噪声（SNR~1） |
| **分 bin 方案** | 9 bins (3×Teff × 3×[M/H]) | 沿用 MVP-9E1 |

### 噪声模型

$$
\text{noisy\_flux} = \text{flux} + \text{error} \times \mathcal{N}(0, 1.0^2)
$$

**物理意义**：
- noise=1.0 对应 SNR ≈ 1 的极端低质量光谱
- 这模拟了观测条件极差或曝光时间极短的场景

## 2.2 模型与算法

### 专家设置
- **专家数量**：9× Ridge（沿用 MVP-9E1 架构）
- **每个专家**：用 1M train 的对应子集训练
- **正则化**：α 通过 5-fold CV 选择（预期 α 会增大以应对高噪声）

### Gate 设置
- **架构**：回归 gate MLP (13 → 64 → 9) + softmax
- **损失函数**：MSE（回归）
- **特征**：13 维物理窗特征（Ca II + Na + PCA1-4）

### 对照组

| 方法 | 说明 |
|------|------|
| Global Ridge (noise=1) | 全局 baseline |
| Oracle MoE (noise=1) | 理论上限（真值路由） |
| Soft Gate MoE (noise=1) | 本实验主体 |
| 参考：MoE (noise=0.2) | MVP-12A 结果 |

## 2.3 超参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Ridge α | CV 选择 | 预期高噪声下 α 更大 |
| Gate MLP hidden | 64 | 沿用 MVP-Next-A |
| Gate epochs | 100 | Early stopping |
| Bootstrap iterations | 1000 | CI 估计 |

## 2.4 评价指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **R²** | 主指标 | 预测性能 |
| **ρ** | $(R^2_{soft} - R^2_{global}) / (R^2_{oracle} - R^2_{global})$ | oracle 增益保留率 |
| **ΔR²** | MoE - Global | 核心改善 |
| **Gate 准确率** | 9-class 分类 | Gate 稳定性 |
| **Per-bin R²** | 各 bin 独立 | 定位脆弱区域 |

---

# 3. 📊 实验图表

> ⏳ 实验进行中，待填充

### 预期图表

1. **R² 对比图**：Global vs Oracle vs Soft (noise=1)
2. **Per-bin R² 对比**：9 个 bin 的独立性能
3. **Gate 混淆矩阵**：高噪声下的分类准确率
4. **ρ 对比图**：noise=0.2 vs noise=1.0
5. **噪声敏感性曲线**：R² vs noise level

---

# 4. 💡 关键洞见

> ⏳ 实验进行中，待填充

### 预期关注点

1. **噪声对 Gate 的影响**：高噪声是否严重干扰物理窗特征提取？
2. **Soft routing 的鲁棒性**：是否仍能平滑边界误差？
3. **Metal-poor bins 的脆弱性**：谱线更弱的区域是否更难预测？
4. **规模效应**：1M 数据是否能部分抵消噪声影响？

---

# 5. 📝 结论

> ⏳ 实验进行中，待填充

---

# 6. 📎 附录

## 6.1 数值结果表

> ⏳ 实验进行中，待填充

## 6.2 实验流程记录

### 执行命令

```bash
cd ~/VIT
source init.sh
python scripts/moe_1m_high_noise.py --noise_level 1.0 --train_size 1000000
```

### 关键日志

> ⏳ 实验进行中，待填充

## 6.3 相关文件

| 文件类型 | 路径 | 说明 |
|---------|------|------|
| 训练脚本 | `~/VIT/scripts/moe_1m_high_noise.py` | TBD |
| 结果目录 | `~/VIT/results/moe/1m_high_noise/` | TBD |
| 图表目录 | `logg/moe/img/` | TBD |

---

## 6.4 与 noise=0.2 baseline 的对比预期

| 指标 | noise=0.2 (MVP-12A) | noise=1.0 预期 | 变化预测 |
|------|---------------------|----------------|----------|
| Global R² | ~0.86 | ~0.3-0.5 | 大幅下降 |
| Oracle MoE R² | 0.9400 | ~0.4-0.6 | 大幅下降 |
| ΔR² (MoE - Global) | +0.08 | +0.02~0.05 | 预期仍有增益 |
| ρ | ~1.0 | 0.5-0.8 | 预期下降 |
| Gate 准确率 | 94% | 60-80% | 预期下降 |

---

*实验 ID: VIT-20251223-moe-1m-highnoise-01*
*创建日期: 2025-12-23*

