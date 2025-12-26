# 🍃 Fisher Efficiency 分桶评估
> **Name:** Fisher Efficiency Binned Evaluation  
> **ID:** `SCALING-20251225-fisher-efficiency-01`  
> **Topic:** `scaling` | **MVP:** MVP-F-EFF | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-25 | **Status:** 🔄 进行中  
> **Root:** `scaling` | **Parent:** `Phase 4` | **Child:** -

> 🎯 **Target:** 各模型按 mag/SNR 分桶的 efficiency (R²/R²_max)，量化不同 SNR 区间的 headroom，决定投模型还是投结构  
> 🚀 **Next:** efficiency < 80% @ 高SNR → 继续投模型；efficiency ≥ 80% @ 高SNR → 转结构化

---

## ⚡ 核心结论速览

> **一句话**: 基于 Fisher Multi-Mag 的 R²_max 和各模型预测结果，计算按 magnitude/SNR 分桶的 efficiency (R²_model/R²_max)，量化不同 SNR 区间的 headroom，为"投模型 vs 投结构"决策提供依据。

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H-F-EFF.1: 高SNR区间 efficiency < 80%? | ⏳ 待验证 | 如果通过 → 继续投模型；如果失败 → 转结构化 |
| H-F-EFF.2: 不同模型在不同 SNR 区间的 efficiency 差异显著? | ⏳ 待验证 | 识别模型优势区间 |

| 指标 | 值 | 启示 |
|------|-----|------|
| Efficiency (高SNR, median) | ⏳ 待计算 | 决定投模型 vs 投结构 |
| Efficiency (低SNR, median) | ⏳ 待计算 | 识别信息悬崖区域 |

| Type | Link |
|------|------|
| 🧠 **Fisher Hub** | [`../fisher_hub_20251225.md`](../fisher_hub_20251225.md) § DG1, P7 |
| 🗺️ **Fisher Roadmap** | [`../fisher_roadmap_20251225.md`](../fisher_roadmap_20251225.md) § MVP-F-EFF |
| 🗺️ Scaling Roadmap | [`../scaling_roadmap_20251222.md`](../scaling_roadmap_20251222.md) |
| 📋 Kanban | `status/kanban.md` |

---

# 1. 🎯 目标

**核心问题**: 全局 R² 掩盖区域差异，必须按 mag/SNR 分桶评估。各模型在不同 SNR 区间的 efficiency (R²/R²_max) 是多少？这决定了是继续投模型还是转结构化。

**背景**:
- Fisher Multi-Mag 显示 R²_max 从 0.99（mag=20, SNR≈24）到 0（mag=23, SNR<2）阶梯式下降
- 全局 R²=0.62 (Oracle MoE) 无法判断是"高 SNR 区仍有大量 headroom"还是"已接近混合分布上限"
- 需要分桶评估以识别模型优势区间和瓶颈区域

**验证假设**:
- H-F-EFF.1: 高SNR区间 (mag≤21.5, SNR≥7) efficiency < 80%
- H-F-EFF.2: 不同模型在不同 SNR 区间的 efficiency 差异显著

| 预期 | 判断标准 |
|------|---------|
| efficiency < 80% @ 高SNR | 通过 → 继续投模型（算法仍有 headroom） |
| efficiency ≥ 80% @ 高SNR | 失败 → 转结构化（已接近上限） |

**Gate**: Gate-1 (Efficiency 分桶评估)

---

# 2. 🦾 算法

**Efficiency 定义**：

$$\text{efficiency} = \frac{R^2_{\text{model}}}{R^2_{\max}}$$

其中：
- $R^2_{\text{model}}$ 是模型在某个 mag/SNR 桶内的 R²
- $R^2_{\max}$ 是 Fisher 理论上限在该桶内的 R²_max

**分桶策略**：

按 magnitude 或 SNR 分桶：
- **Magnitude 桶**: [18.0, 20.0], [20.0, 21.5], [21.5, 22.0], [22.0, 22.5], [22.5, 23.0], [23.0, +∞]
- **SNR 桶**: [0, 2], [2, 4], [4, 7], [7, 20], [20, +∞]

**关键步骤**：
1. **数据准备**: 加载各模型的预测结果（需包含 mag 信息或按 mag 数据分别评估）
2. **分桶**: 按 magnitude 或 SNR 将测试样本分桶（与 Fisher Multi-Mag 对齐）
3. **计算 R²_model**: 对每个模型、每个桶，计算 R²_model（仅 log_g 预测）
4. **获取 R²_max**: 从 Fisher Multi-Mag 结果中获取对应桶的 R²_max（median 值）
5. **计算 efficiency**: efficiency = R²_model / R²_max（处理 R²_max=0 的情况）
6. **可视化**: 
   - Efficiency heatmap（模型 × mag 桶）
   - Efficiency vs SNR 曲线
   - Headroom 分析图
   - Efficiency 分布图
7. **分析**: 识别高/低 efficiency 区域，验证假设

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | 所有已训练模型的预测结果 |
| 数据版本 | 与 Fisher Multi-Mag 一致 |
| 分桶依据 | Magnitude 或 SNR（与 Fisher Multi-Mag 对齐） |
| 目标 | logg |

**需要评估的模型**:
- Ridge (全局 R²=0.50 @ mag=21.5, noise=1, 1M data)
- LightGBM (全局 R²=0.57 @ mag=21.5, noise=1, 1M data)
- MLP (全局 R²≈0.47 @ mag=21.5, noise=1, 需确认)
- CNN (全局 R²≈0.41 @ mag=21.5, noise=1, 需确认)
- Oracle MoE (全局 R²=0.62 @ mag=21.5, noise=1, 1M data)
- Soft-gate MoE (如有)

**Fisher Ceiling 参考**:
- 使用 Fisher Multi-Mag 的 R²_max 结果（6 个 mag 点）：
  - mag=18.0: R²_max (median) = 0.9994
  - mag=20.0: R²_max (median) = 0.9906
  - mag=21.5: R²_max (median) = 0.8914
  - mag=22.0: R²_max (median) = 0.7396
  - mag=22.5: R²_max (median) = 0.3658
  - mag=23.0: R²_max (median) = 0.0000

## 3.2 模型预测结果

| 模型 | 数据来源 | 预测结果路径 | 备注 |
|------|---------|------------|------|
| Ridge | MVP-1.0 | `~/VIT/results/scaling_ml_ceiling/ridge_results.csv` | 需按 mag 分桶，当前为全局 R² |
| LightGBM | MVP-1.1 | `~/VIT/results/scaling_ml_ceiling/lightgbm_results.csv` | 需按 mag 分桶，当前为全局 R² |
| MLP | MVP-NN-0 | `~/VIT/results/scaling_nn_baselines/` | 需确认路径和格式 |
| CNN | MVP-NN-0 | `~/VIT/results/scaling_nn_baselines/` | 需确认路径和格式 |
| Oracle MoE | MVP-16A-0 | `~/VIT/results/scaling_oracle_moe/` | 需按 mag 分桶，当前为全局 R² |

**注意**: 
- 当前模型预测结果多为全局 R²（在 mag=21.5 数据上训练/测试）
- **需要重新评估**: 在多个 magnitude 数据上分别评估，或使用包含 mag 信息的测试集按 mag 分桶
- **数据对齐**: 确保测试数据与 Fisher Multi-Mag 使用的数据一致（grid_mag*_lowT）

## 3.3 分桶配置

| 桶类型 | 桶边界 | 对应 Fisher Multi-Mag | R²_max (median) |
|--------|--------|----------------------|-----------------|
| Magnitude | [18.0, 20.0] | grid_mag18_lowT, grid_mag20_lowT | 0.9994 (mag18), 0.9906 (mag20) |
| | [20.0, 21.5] | grid_mag20_lowT, grid_mag215_lowT | 0.9906 (mag20), 0.8914 (mag215) |
| | [21.5, 22.0] | grid_mag215_lowT, grid_mag22_lowT | 0.8914 (mag215), 0.7396 (mag22) |
| | [22.0, 22.5] | grid_mag22_lowT, grid_mag225_lowT | 0.7396 (mag22), 0.3658 (mag225) |
| | [22.5, 23.0] | grid_mag225_lowT, grid_mag23_lowT | 0.3658 (mag225), 0.0000 (mag23) |
| | [23.0, +∞] | grid_mag23_lowT | 0.0000 (mag23) |

**SNR 桶**（备选，用于交叉验证）:
- 使用 Fisher Multi-Mag 的 SNR 中位数作为桶边界
- [0, 2]: mag23 (SNR~1.9)
- [2, 4]: mag225 (SNR~2.8), mag22 (SNR~4.0)
- [4, 7]: mag22 (SNR~4.0), mag215 (SNR~7.1)
- [7, 20]: mag215 (SNR~7.1), mag20 (SNR~24.0)
- [20, +∞]: mag20 (SNR~24.0), mag18 (SNR~87.4)

**分桶策略**:
- **主要方法**: 按 magnitude 分桶（与 Fisher Multi-Mag 对齐）
- **备选方法**: 按 SNR 分桶（用于验证 SNR 阈值效应）
- **R²_max 插值**: 对于跨 mag 的桶，使用线性插值或取较小值（保守估计）

## 3.4 评价指标

| 指标 | 说明 |
|------|------|
| Efficiency (median) | 每个桶内的 efficiency 中位数 |
| Efficiency (mean) | 每个桶内的 efficiency 均值 |
| Efficiency (std) | 每个桶内的 efficiency 标准差 |
| Headroom | 1 - efficiency = (R²_max - R²_model) / R²_max |

---

# 4. 📊 实验图表

> ⚠️ 图表文字必须全英文！

### Fig 1: Efficiency Heatmap (Model × Magnitude)
![](../img/fisher_efficiency_heatmap_mag.png)

**预期观察**:
- 高 SNR 区域 (mag≤20): efficiency 可能接近 1.0（模型接近理论上限）
- 中 SNR 区域 (mag=21.5): efficiency 约 0.5-0.7（LightGBM/Oracle MoE 相对较高）
- 低 SNR 区域 (mag≥22.5): efficiency 显著下降，可能接近 0（信息悬崖）
- **关键**: 识别高 SNR 区 efficiency < 80% 的区域，决定是否继续投模型

### Fig 2: Efficiency vs SNR
![](../img/fisher_efficiency_vs_snr.png)

**预期观察**:
- SNR > 20: efficiency 接近 1.0
- SNR = 7-20: efficiency 约 0.6-0.8
- SNR = 4-7: efficiency 约 0.5-0.7（临界区域）
- SNR < 4: efficiency 急剧下降（信息悬崖）
- **验证**: SNR≈4 是否为 efficiency 的临界点

### Fig 3: Headroom by Model and Magnitude
![](../img/fisher_headroom_by_model_mag.png)

**预期观察**:
- Headroom = 1 - efficiency = (R²_max - R²_model) / R²_max
- 高 SNR: headroom 小（接近上限）
- 中 SNR: headroom 中等（仍有改进空间）
- 低 SNR: headroom 大但 R²_max 本身很小（信息不足）
- **量化**: 各模型在不同 mag 的剩余 headroom，指导资源分配

### Fig 4: Efficiency Distribution by Bucket
![](../img/fisher_efficiency_distribution.png)

**预期观察**:
- 展示每个 magnitude 桶内 efficiency 的分布（箱线图或 violin plot）
- 识别 efficiency 分布偏态（某些参数区域更容易达到高 efficiency）
- **分析**: efficiency 的方差和偏度，识别模型优势区间

---

# 5. 💡 洞见

## 5.1 宏观

**预期发现**:
- **SNR 阈值效应**: 不同 SNR 区间的 efficiency 差异显著，验证 Fisher Multi-Mag 发现的临界 SNR≈4
- **信息悬崖**: mag≥22.5 (SNR<3) 时，即使理论上限也很低，模型改进空间有限
- **高 SNR 区 headroom**: mag≤20 (SNR>20) 时，如果 efficiency < 80%，说明模型仍有改进空间

## 5.2 模型层

**预期发现**:
- **Ridge vs LightGBM**: LightGBM 在中等 SNR 区可能效率更高（非线性特征交互）
- **Oracle MoE**: 在所有 SNR 区效率最高，验证结构化方法的优势
- **模型优势区间**: 不同模型在不同 SNR 区间的相对表现可能不同

## 5.3 细节

**预期发现**:
- **Efficiency 分布**: 同一 magnitude 桶内，不同参数区域的 efficiency 可能有显著差异
- **参数纠缠影响**: Schur decay 恒定的情况下，efficiency 主要受 SNR 影响
- **模型瓶颈**: 如果高 SNR 区 efficiency < 80%，瓶颈可能在特征提取而非信息量

---

# 6. 📝 结论

## 6.1 核心发现
> **待实验完成后填写实际结果**

**预期结论**:
- 如果高 SNR 区 (mag≤21.5, SNR≥7) efficiency < 80% → **继续投模型**（算法仍有 headroom）
- 如果高 SNR 区 efficiency ≥ 80% → **转结构化**（已接近理论上限，需结构化方法）

- ⏳ H-F-EFF.1: [待验证] - 高SNR区间 efficiency < 80%?
- ⏳ H-F-EFF.2: [待验证] - 不同模型在不同 SNR 区间的 efficiency 差异显著?

## 6.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | **[待填写]** | [数据] - 例如：高 SNR 区 efficiency = X% |
| 2 | **[待填写]** | [数据] - 例如：Oracle MoE 在所有区间效率最高 |
| 3 | **[待填写]** | [数据] - 例如：SNR≈4 是 efficiency 的临界点 |

## 6.3 设计启示

| 原则 | 建议 |
|------|------|
| **高 SNR 区策略** | 如果 efficiency < 80% → 继续改进模型（特征提取、架构优化） |
| **低 SNR 区策略** | 如果 efficiency 接近 R²_max 但 R²_max 本身很小 → 考虑多曝光、先验、ensemble |
| **模型选择** | 根据目标 SNR 区间选择最优模型（Ridge/LightGBM/MoE） |
| **资源分配** | 优先投入 efficiency 低但 R²_max 高的区域（最大 ROI） |

## 6.4 关键数字速查

| 指标 | 值 | 条件 |
|------|-----|------|
| Efficiency (高SNR, median) | ⏳ | mag≤21.5, SNR≥7 |
| Efficiency (低SNR, median) | ⏳ | mag≥22.5, SNR<4 |
| Headroom (高SNR, median) | ⏳ | mag≤21.5, SNR≥7 |

## 6.5 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| 决策 | 如果 efficiency < 80% @ 高SNR → 继续投模型 | 🔴 |
| 决策 | 如果 efficiency ≥ 80% @ 高SNR → 转结构化 | 🔴 |
| 后续 | 根据 efficiency 结果决定是否启动 MVP-F-WGT | 🟡 |

---

# 7. 📎 附录

## 7.1 数值结果表

| 模型 | Magnitude 桶 | SNR 范围 | R²_model | R²_max | Efficiency | Headroom |
|------|-------------|----------|----------|--------|------------|----------|
| Ridge | [18.0, 20.0] | [20, +∞] | ⏳ | 0.9994* | ⏳ | ⏳ |
| Ridge | [20.0, 21.5] | [7, 20] | ⏳ | 0.8914* | ⏳ | ⏳ |
| Ridge | [21.5, 22.0] | [4, 7] | ⏳ | 0.7396* | ⏳ | ⏳ |
| Ridge | [22.0, 22.5] | [3, 4] | ⏳ | 0.3658* | ⏳ | ⏳ |
| Ridge | [22.5, 23.0] | [2, 3] | ⏳ | 0.0000* | ⏳ | ⏳ |
| LightGBM | [18.0, 20.0] | [20, +∞] | ⏳ | 0.9994* | ⏳ | ⏳ |
| LightGBM | [20.0, 21.5] | [7, 20] | ⏳ | 0.8914* | ⏳ | ⏳ |
| LightGBM | [21.5, 22.0] | [4, 7] | ⏳ | 0.7396* | ⏳ | ⏳ |
| LightGBM | [22.0, 22.5] | [3, 4] | ⏳ | 0.3658* | ⏳ | ⏳ |
| LightGBM | [22.5, 23.0] | [2, 3] | ⏳ | 0.0000* | ⏳ | ⏳ |
| Oracle MoE | [18.0, 20.0] | [20, +∞] | ⏳ | 0.9994* | ⏳ | ⏳ |
| Oracle MoE | [20.0, 21.5] | [7, 20] | ⏳ | 0.8914* | ⏳ | ⏳ |
| Oracle MoE | [21.5, 22.0] | [4, 7] | ⏳ | 0.7396* | ⏳ | ⏳ |
| Oracle MoE | [22.0, 22.5] | [3, 4] | ⏳ | 0.3658* | ⏳ | ⏳ |
| Oracle MoE | [22.5, 23.0] | [2, 3] | ⏳ | 0.0000* | ⏳ | ⏳ |
| MLP | [18.0, 20.0] | [20, +∞] | ⏳ | 0.9994* | ⏳ | ⏳ |
| MLP | [20.0, 21.5] | [7, 20] | ⏳ | 0.8914* | ⏳ | ⏳ |
| CNN | [18.0, 20.0] | [20, +∞] | ⏳ | 0.9994* | ⏳ | ⏳ |
| CNN | [20.0, 21.5] | [7, 20] | ⏳ | 0.8914* | ⏳ | ⏳ |

*注: R²_max 来自 Fisher Multi-Mag 结果（median 值）。对于跨 mag 的桶，使用较小值或插值。

## 7.2 实验流程记录

| 项 | 值 |
|----|-----|
| 仓库 | `~/VIT` |
| 脚本 | `scripts/scaling_fisher_efficiency_binned.py` (待创建) |
| 参考脚本 | `scripts/scaling_fisher_ceiling_v2_multi_mag.py` (Fisher Multi-Mag) |
| 数据路径 | 各模型的预测结果 + Fisher Multi-Mag 的 R²_max |
| 输出路径 | `results/fisher_efficiency_binned/` |

```bash
# 执行命令（参考 scaling_fisher_ceiling_v2_multi_mag.py 的结构）
python scripts/scaling_fisher_efficiency_binned.py \
    --models ridge lgbm mlp cnn oracle_moe \
    --mag_bins 18.0 20.0 21.5 22.0 22.5 23.0 \
    --fisher_results results/SCALING-20251224-fisher-multi-mag/combined_summary.json \
    --model_results_dir results/scaling_ml_ceiling \
    --output results/fisher_efficiency_binned/ \
    --img-dir /home/swei20/Physics_Informed_AI/logg/scaling/img
```

**脚本设计要点**（参考 `scaling_fisher_ceiling_v2_multi_mag.py`）:
1. 加载各模型的预测结果（需包含 mag 信息或按 mag 数据分别评估）
2. 加载 Fisher Multi-Mag 的 R²_max 结果
3. 按 magnitude 分桶计算 R²_model
4. 计算 efficiency = R²_model / R²_max
5. 生成 heatmap、分布图等可视化
6. 输出数值结果表（CSV/JSON）

## 7.3 调试记录

| 问题 | 解决 |
|------|------|
| 模型预测结果缺少 mag 信息 | 方案 A: 在多个 mag 数据上重新评估；方案 B: 使用包含 mag 的测试集 |
| 跨 mag 桶的 R²_max 选择 | 使用较小值（保守估计）或线性插值 |
| 数据对齐问题 | 确保测试数据与 Fisher Multi-Mag 使用的数据一致（grid_mag*_lowT） |

---

> **实验完成时间**: [待填写]

