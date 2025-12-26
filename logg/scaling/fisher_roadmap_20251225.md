# 🗺️ Fisher Roadmap: 理论上限与信息诊断
> **Name:** Fisher Information & CRLB Ceiling | **ID:** `SCALING-20251225-fisher-roadmap`  
> **Topic:** `fisher` | **Phase:** Phase 2 (V2 完成，V3-A 进行中) | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-25 | **Status:** 🔄 Active

```
💡 当前阶段目标  
验证 Fisher 上限的"世界定义"：conditional vs marginal ceiling
Gate：V3-A 验证化学丰度 nuisance 的影响
```

---

## 🔗 Related Files

| Type | File | Description |
|------|------|-------------|
| 🧠 **Fisher Hub** | [`fisher_hub_20251225.md`](./fisher_hub_20251225.md) | 问题树、假设、洞见汇合、战略导航 |
| 🧠 Scaling Hub | [`scaling_hub_20251222.md`](./scaling_hub_20251222.md) | 上层战略 |
| 🗺️ Scaling Roadmap | [`scaling_roadmap_20251222.md`](./scaling_roadmap_20251222.md) | 完整实验追踪 |
| 📋 Kanban | `../../status/kanban.md` | 全局任务看板 |
| 📗 Experiments | `exp/exp_scaling_fisher_*.md` | 详细实验报告 |

---

# 1. 🚦 Decision Gates

> Roadmap 定义怎么验证，Hub 做战略分析

## 1.1 战略路线 (来自 Hub)

| Route | 名称 | Hub推荐 | 验证Gate |
|-------|------|---------|----------|
| **Route A** | 继续投模型 | 🟡 | Gate-1 (efficiency 分桶评估) |
| **Route B** | 结构化 (MoE + Error-aware) | 🟢 **推荐** | Gate-2 (weighted loss 验证) |
| **Route C** | 改任务 (分类/先验/多曝光) | 🔴 | 仅在 mag>22.5 考虑 |

## 1.2 Gate 定义

### Gate-1: Efficiency 分桶评估

| 项 | 内容 |
|----|------|
| **验证** | 各模型按 mag/SNR 分桶的 efficiency (R²/R²_max) |
| **MVP** | MVP-F-EFF (待规划) |
| **若A** | efficiency < 80% @ 高SNR → 继续投模型 |
| **若B** | efficiency ≥ 80% @ 高SNR → 转结构化 |
| **状态** | ⏳ 待启动 |

### Gate-2: Weighted Loss 验证

| 项 | 内容 |
|----|------|
| **验证** | Error-aware 输入 + weighted loss 能否提升性能 |
| **MVP** | MVP-F-WGT (待规划) |
| **若A** | CNN/MLP ≥ Ridge → 误差是瓶颈 |
| **若B** | CNN/MLP < Ridge → 结构是瓶颈 |
| **状态** | ⏳ 待启动 |

### Gate-3: V3-A Ceiling 下降幅度

| 项 | 内容 |
|----|------|
| **验证** | 加入化学丰度 nuisance 后 ceiling 下降幅度 |
| **MVP** | MVP-F-V3A |
| **若A** | Δceiling < 10% (R²_max ≥ 0.80) → V2 结论稳健，继续投模型 |
| **若B** | Δceiling 10-20% (R²_max 0.70-0.80) → 需重新评估 |
| **若C** | Δceiling > 20% (R²_max < 0.70) → 可能已接近真实上限 |
| **状态** | 🔄 进行中 |

## 1.3 本周重点

| 优先级 | MVP | Gate | 状态 |
|--------|-----|------|------|
| 🔴 P0 | MVP-F-V3A | Gate-3 | 🔄 进行中 |
| 🟡 P1 | MVP-F-EFF | Gate-1 | ⏳ 待启动 |
| 🟡 P1 | MVP-F-WGT | Gate-2 | ⏳ 待启动 |

---

# 2. 📋 MVP 列表

## 2.1 总览

| MVP | 名称 | Phase | Gate | 状态 | exp_id | 报告 |
|-----|------|-------|------|------|--------|------|
| **MVP-F-V1** | Fisher/CRLB V1 (失败) | 0 | - | ❌ | `SCALING-20251223-fisher-ceiling-01` | [Link](./exp/exp_scaling_fisher_ceiling_20251223.md) |
| **MVP-F-V2** | Fisher/CRLB V2 (规则网格) | 1 | - | ✅ | `SCALING-20251224-fisher-ceiling-02` | [Link](./exp/exp_scaling_fisher_ceiling_v2_20251224.md) |
| **MVP-F-MM** | Multi-Magnitude Sweep | 1 | - | ✅ | `SCALING-20251224-fisher-multi-mag` | [Link](./exp/exp_scaling_fisher_multi_mag_20251224.md) |
| **MVP-F-V3A** | V3-A: 化学丰度 Nuisance | 2 | Gate-3 | ✅ | `SCALING-20251225-fisher-ceiling-03` | [Link](./exp/exp_scaling_fisher_ceiling_v3_chemical_20251225.md) |
| **MVP-F-V3B** | V3-B: Redshift/RV Nuisance | 3 | - | ⏳ | - | - |
| **MVP-F-V3C** | V3-C: Moon/Sky 条件扫描 | 3 | - | ⏳ | - | - |
| **MVP-F-EFF** | Efficiency 分桶评估 | 4 | Gate-1 | 🔴 | `SCALING-20251225-fisher-efficiency-01` | [Link](./exp/exp_scaling_fisher_efficiency_binned_20251225.md) |
| **MVP-F-WGT** | Weighted Loss 验证 | 4 | Gate-2 | ⏳ | - | - |

**状态**: ⏳计划 | 🔴就绪 | 🚀运行 | ✅完成 | ❌取消

## 2.2 配置速查

| MVP | 数据 | 网格结构 | 参数维度 | 关键变量 |
|-----|------|---------|---------|---------|
| V1 | BOSZ 连续采样 | 非规则网格 | 3D (T_eff, logg, [M/H]) | - |
| V2 | grid_mag215_lowT | 规则网格 (10×9×14) | 3D (T_eff, logg, [M/H]) | noise=1 |
| Multi-Mag | grid_mag{18,20,215,22,225,23}_lowT | 规则网格 | 3D | mag sweep |
| V3-A | 规则网格（需含化学丰度轴） | 规则网格 | 5/6D (+C_M, O_M, a_M) | 化学丰度 nuisance |
| V3-B | 规则网格 | 规则网格 | 4D (+redshift/RV) | redshift/RV nuisance |
| V3-C | 规则网格 | 规则网格 | 3D | moon_phase/sky_level sweep |

---

# 3. 🔧 MVP 规格

## Phase 0: 初始尝试（失败）

### MVP-F-V1: Fisher/CRLB V1（❌ 失败）

| 项 | 配置 |
|----|------|
| **目标** | 使用 BOSZ 连续采样数据计算 Fisher/CRLB 理论上限 |
| **数据** | BOSZ 50000 连续采样（~40k 唯一值/参数） |
| **方法** | 邻近点差分法 |
| **失败原因** | 连续采样数据导致偏导估计混参，CRLB 跨 20 数量级 |
| **教训** | **必须使用规则网格数据** |

---

## Phase 1: 基线建立（✅ 完成）

### MVP-F-V2: Fisher/CRLB V2（✅ 完成）

| 项 | 配置 |
|----|------|
| **目标** | 使用规则网格数据计算 noise=1 时的理论可达上限 R²_max |
| **数据** | `/datascope/.../grid/grid_mag215_lowT/dataset.h5` (30,182 samples) |
| **Grid** | T_eff: 250K step, logg: 0.5 step, [M/H]: 0.25 step |
| **方法** | 沿网格轴有限差分 → Fisher → Schur complement → CRLB |
| **验收** | R²_max ≥ 0.75 → 通过 ✅ |
| **结果** | R²_max = **0.8914** (median), Schur = 0.6906 |

**关键输出**:
- R²_max = 0.8914 (median), 0.9804 (90%)
- Schur decay = 0.6906 (69% 信息保留)
- CRLB range = 2.9 orders（数值稳定）
- Gap vs LightGBM = +0.32

**决策影响**: 理论上限高，继续投入 CNN/MoE 值得

---

### MVP-F-MM: Multi-Magnitude Sweep（✅ 完成）

| 项 | 配置 |
|----|------|
| **目标** | 扩展到 6 个不同 magnitude，验证 SNR 阈值效应 |
| **数据** | grid_mag{18,20,215,22,225,23}_lowT |
| **方法** | 与 V2 一致，在不同 mag 上重复计算 |
| **验收** | 发现 SNR 阈值效应 → 通过 ✅ |
| **结果** | 临界 SNR≈4, 信息悬崖 SNR<2 |

**关键发现**:
- **信息阶梯**: SNR↓ → R²_max 阶梯式下降
- **临界 SNR**: SNR≈4 (mag≈22) 是临界点
- **信息悬崖**: SNR<2 (mag>23) 时 median R²_max=0
- **Schur 恒定**: Schur≈0.69 across all SNR（由光谱物理决定）

**决策影响**: 按 mag/SNR 分层评估，mag≥22.5 需改变策略

---

## Phase 2: Nuisance 参数扩展（✅ 完成 V3-A）

### MVP-F-V3A: 化学丰度 Nuisance（✅ 完成）

| 项 | 配置 |
|----|------|
| **目标** | 将化学丰度参数 (C_M, O_M, a_M) 作为 nuisance 加入 Fisher 计算 |
| **核心问题** | Fisher 上限到底是在给"哪种世界"算上限？ |
| **数据** | 规则网格数据，需包含 (C_M, O_M, a_M) 轴（固定间隔） |
| **Grid** | 参数从 3 维扩展到 5/6 维：$(T_{\rm eff}, \log g, [M/H], C_M, a_M, O_M)$ |
| **方法** | 沿网格轴有限差分 → Fisher → Schur complement → CRLB（与 V2 一致） |
| **计算范围** | 2-3 个关键 mag（21.5, 22.0, 22.5） |
| **验收** | H-16T-V3A.1: Δceiling < 10% (R²_max ≥ 0.80) |

**实际结果**:
- R²_max = 0.8742 (median)
- Δceiling = 1.93% < 10%
- ✅ Gate-3 通过验证
| **Gate** | Gate-3 |

**关键输出**:
- R²_max (V3-A, median + 分位数)
- Schur decay (V3-A)
- **Δceiling = V3-A vs V2 的下降幅度**

**决策规则**:
- Δceiling < 10% → V2 结论稳健，继续投模型
- Δceiling 10-20% → 需重新评估
- Δceiling > 20% → 可能已接近真实上限

**论文影响**:
- 若新 ceiling 仍高：主张"算法还有大量可挖掘信息"
- 若新 ceiling 明显下降：主张"已接近物理极限"

---

## Phase 3: 进一步 Nuisance 扩展（⏳ 待规划）

### MVP-F-V3B: Redshift/RV Nuisance（⏳ 待规划）

| 项 | 配置 |
|----|------|
| **目标** | 将 redshift/RV 作为 nuisance 加入 Fisher 计算 |
| **方法** | 两种 ceiling：Pipeline-corrected (z 已知) vs End-to-end (z 未知) |
| **数据** | 当前数据 z=0，Pipeline-corrected ceiling 已匹配 |
| **优先级** | 🟡 P1（可作为 follow-up） |

**说明**: redshift 不需要网格，可通过数值微分计算 $\partial\mu/\partial z$

---

### MVP-F-V3C: Moon/Sky 条件扫描（⏳ 待规划）

| 项 | 配置 |
|----|------|
| **目标** | 不同 moon_phase/sky_level 下的 ceiling 扫描 |
| **方法** | 条件 sweep（非参数进 Fisher），改变噪声结构 Σ |
| **数据** | 当前数据 moon_phase=0（新月），可解释为"dark-time upper bound" |
| **优先级** | 🟢 P2（未来推广到真实 survey 条件） |

**说明**: moon/sky 主要改变噪声结构，而非稳定的 mean shift，更适合条件扫描而非参数进 Fisher

---

## Phase 4: 应用导向验证（⏳ 待规划）

### MVP-F-EFF: Efficiency 分桶评估（🔴 就绪）

| 项 | 配置 |
|----|------|
| **目标** | 各模型按 mag/SNR 分桶的 efficiency (R²/R²_max) |
| **Gate** | Gate-1 |
| **数据** | 所有已训练模型（Ridge, LightGBM, MLP, CNN, Oracle MoE） |
| **方法** | 按 mag/SNR 分桶，计算每桶的 efficiency = R²_model / R²_max |
| **验收** | efficiency 图 → 决定投模型还是投结构 |
| **状态** | 🔴 就绪（实验框架已创建） |

**决策规则**:
- efficiency < 80% @ 高SNR → 继续投模型
- efficiency ≥ 80% @ 高SNR → 转结构化

**关键输出**:
- Efficiency heatmap (模型 × mag/SNR 桶)
- Headroom 分析
- 模型优势区间识别

**参考**:
- Fisher Multi-Mag 的 R²_max 结果（6 个 mag 点）
- 各模型的预测结果（需按 mag 分桶）

---

### MVP-F-WGT: Weighted Loss 验证（⏳ 待规划）

| 项 | 配置 |
|----|------|
| **目标** | Error-aware 输入 + weighted loss 能否提升性能 |
| **Gate** | Gate-2 |
| **方法** | 对比 unweighted vs weighted loss (Σ⁻¹ 加权) |
| **模型** | Ridge, MLP, CNN |
| **验收** | CNN/MLP ≥ Ridge → 误差是瓶颈；Else → 结构是瓶颈 |

**理论依据**: Fisher 最优估计用 Σ⁻¹ 加权，当前 ML 多数未利用

---

# 4. 📊 进度追踪

## 4.1 看板

```
⏳计划          🔴就绪          🚀运行          ✅完成
MVP-F-V3B       MVP-F-EFF                       MVP-F-V2
MVP-F-V3C       MVP-F-WGT                       MVP-F-MM
                                                                 MVP-F-V3A
                                                
❌取消
MVP-F-V1
```

## 4.2 Gate 进度

| Gate | MVP | 状态 | 结果 |
|------|-----|------|------|
| Gate-1 | MVP-F-EFF | ⏳ | - |
| Gate-2 | MVP-F-WGT | ⏳ | - |
| Gate-3 | MVP-F-V3A | ✅ | Δceiling=1.93% < 10%, V2 结论稳健 |

## 4.3 结论快照

| MVP | 结论 | 关键指标 | 同步Hub |
|-----|------|---------|---------|
| **MVP-F-V2** | ✅ 理论上限 R²_max=0.89，headroom +32% vs LightGBM | R²_max=0.8914, Schur=0.6906 | ✅ §2.1 |
| **MVP-F-MM** | ✅ 临界 SNR≈4，信息悬崖 SNR<2，Schur 恒定 | SNR_threshold=4, Schur=0.69 | ✅ §2.1 |
| **MVP-F-V3A** | ✅ 化学丰度 nuisance 仅使 ceiling 下降 1.93%，V2 结论稳健 | R²_max=0.8742, Δceiling=1.93% | ✅ §2.1 |
| **MVP-F-EFF** | 🔄 进行中 | ⏳ | ⏳ |

## 4.4 时间线

| 日期 | 事件 | 关键结果 |
|------|------|---------|
| 2025-12-23 | MVP-F-V1 失败 | 方法论缺陷（非规则网格） |
| 2025-12-24 | MVP-F-V2 完成 | R²_max=0.8914, Schur=0.6906 |
| 2025-12-24 | MVP-F-MM 完成 | 临界 SNR=4, 信息悬崖 SNR<2 |
| 2025-12-25 | MVP-F-V3A 立项 | 化学丰度 nuisance 实验框架创建 |
| 2025-12-25 | MVP-F-V3A 完成 | Δceiling=1.93%, Gate-3 通过验证 |
| 2025-12-25 | MVP-F-EFF 立项 | Efficiency 分桶评估实验框架创建 |

---

# 5. 🔗 跨仓库集成

## 5.1 实验索引

| exp_id | project | topic | 状态 | MVP |
|--------|---------|-------|------|-----|
| `SCALING-20251223-fisher-ceiling-01` | VIT | fisher | ❌ | MVP-F-V1 |
| `SCALING-20251224-fisher-ceiling-02` | VIT | fisher | ✅ | MVP-F-V2 |
| `SCALING-20251224-fisher-multi-mag` | VIT | fisher | ✅ | MVP-F-MM |
| `SCALING-20251225-fisher-ceiling-03` | VIT | fisher | ✅ | MVP-F-V3A |

## 5.2 仓库链接

| 仓库 | 路径 | 用途 |
|------|------|------|
| VIT | `~/VIT/scripts/scaling_fisher_*.py` | Fisher 计算脚本 |
| VIT | `~/VIT/results/fisher_*/` | 计算结果 |
| 本仓库 | `logg/scaling/exp/exp_scaling_fisher_*.md` | 实验报告 |
| 本仓库 | `logg/scaling/img/` | 图表 |

## 5.3 运行路径

| MVP | 脚本 | 配置 | 输出 |
|-----|------|------|------|
| V2 | `scripts/scaling_fisher_ceiling_v2.py` | - | `results/fisher_v2/` |
| Multi-Mag | `scripts/scaling_fisher_ceiling_v2_multi_mag.py` | - | `results/fisher_multi_mag/` |
| V3-A | `scripts/scaling_fisher_ceiling_v3_chemical.py` (待创建) | - | `results/fisher_v3_chemical/` |

---

# 6. 📎 附录

## 6.1 数值汇总

| MVP | Magnitude | SNR | R²_max (median) | R²_max (90%) | Schur Decay |
|-----|-----------|-----|-----------------|--------------|-------------|
| **V2** | 21.5 | 7.1 | **0.8914** | 0.9804 | 0.6906 |
| **Multi-Mag** | 18.0 | 87.4 | 0.9994 | 0.9999 | 0.6641 |
| **Multi-Mag** | 20.0 | 24.0 | 0.9906 | 0.9983 | 0.6842 |
| **Multi-Mag** | 21.5 | 7.1 | 0.8914 | 0.9804 | 0.6906 |
| **Multi-Mag** | 22.0 | 4.6 | 0.7396 | 0.9530 | 0.6921 |
| **Multi-Mag** | 22.5 | 3.0 | 0.3658 | 0.8854 | 0.6922 |
| **Multi-Mag** | 23.0 | 1.9 | 0.0000 | 0.7180 | 0.6923 |
| **V3-A** | 21.5 | 7.1 | **0.8742** | 0.9768 | 0.5778 |

## 6.2 关键数字速查

| 指标 | 值 | 条件 | 来源 |
|------|-----|------|------|
| **R²_max (median)** | **0.8914** | noise=1, mag=21.5 | V2 |
| **R²_max (90%)** | 0.9804 | noise=1, mag=21.5 | V2 |
| **Gap vs LightGBM** | **+0.32** | - | V2 |
| **Schur decay** | **0.6906** | 恒定 across SNR | Multi-Mag |
| **临界 SNR** | **~4** | R²_max>0.5 边界 | Multi-Mag |
| **信息悬崖** | **SNR<2** | median=0 | Multi-Mag |
| **V3-A R²_max (median)** | **0.8742** | noise=1, mag=21.5, 5D | V3-A |
| **Δceiling (V3-A vs V2)** | **-1.93%** | 化学丰度 nuisance 影响 | V3-A |
| **V3-A Schur decay** | **0.5778** | 5D with chemical | V3-A |

## 6.3 文件索引

| 类型 | 路径 |
|------|------|
| Roadmap | `logg/scaling/fisher_roadmap_20251225.md` |
| Hub | `logg/scaling/fisher_hub_20251225.md` |
| Exp V1 | `logg/scaling/exp/exp_scaling_fisher_ceiling_20251223.md` |
| Exp V2 | `logg/scaling/exp/exp_scaling_fisher_ceiling_v2_20251224.md` |
| Exp Multi-Mag | `logg/scaling/exp/exp_scaling_fisher_multi_mag_20251224.md` |
| Exp V3-A | `logg/scaling/exp/exp_scaling_fisher_ceiling_v3_chemical_20251225.md` |
| 图表 | `logg/scaling/img/` |

## 6.4 更新日志

| 日期 | 变更 | 章节 |
|------|------|------|
| 2025-12-25 | 创建 Fisher Roadmap | - |
| 2025-12-25 | 整合 V1/V2/Multi-Mag/V3-A 所有实验 | §2.1, §3, §4.3 |
| 2025-12-25 | 规划 Phase 3-4 后续实验 | §3 |
| 2025-12-25 | MVP-F-V3A 完成：Δceiling=1.93%，Gate-3 通过验证 | §2.1, §3, §4.2, §4.3, §6.1 |

---

## 📌 核心决策树

```
V2 完成 (R²_max=0.89)
    │
    ├─ Multi-Mag 完成 (SNR阈值效应)
    │
    ├─ V3-A 进行中 (化学丰度 nuisance)
    │   │
    │   ├─ Δceiling < 10% → V2 结论稳健，继续投模型
    │   ├─ Δceiling 10-20% → 需重新评估
    │   └─ Δceiling > 20% → 可能已接近真实上限
    │
    ├─ Phase 3: 进一步 Nuisance (V3-B, V3-C)
    │
    └─ Phase 4: 应用导向 (Efficiency, Weighted Loss)
        │
        ├─ Gate-1: Efficiency 分桶 → 决定投模型 vs 投结构
        └─ Gate-2: Weighted Loss → 决定误差瓶颈 vs 结构瓶颈
```


**关键输出**:
- R²_max = 0.8742 (median), 0.9768 (90%)
- Schur decay = 0.5778 (比 V2 的 0.6906 更低)
- CRLB range = 3.56 orders（数值稳定）
- Δceiling = **1.93%** (远小于 10% 阈值)

**Gate-3 决策**: ✅ **通过验证** - Δceiling < 10% → V2 结论稳健，继续模型部署

**决策影响**: V2 的 R²_max=0.89 对化学丰度 nuisance 高度稳健
