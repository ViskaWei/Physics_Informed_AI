# 🗺️ MoE 实验路线图（Roadmap）

---
> **主题名称：** Mixture of Experts（MoE）要不要上？什么时候有用？  
> **作者：** Viska Wei  
> **创建日期：** 2025-12-03  
> **最后更新：** 2025-12-07  
> **当前 Phase：** Phase 11 ✅ M2 里程碑达成 → 执行 Phase 12-13（大规模验证 + 特征增强）

---

## 🔗 相关文件

| 类型 | 文件 | 说明 |
|------|------|------|
| 🧠 Hub | [`moe_hub.md`](./moe_hub_20251203.md) | 智库导航 |
| 📋 Kanban | [`kanban.md`](../../status/kanban.md) | 全局看板 |
| 📗 子实验 | `exp_moe_*.md` | 单实验详情 |

---

# 📑 目录

- [1. 🎯 Phase 总览](#1--phase-总览)
- [2. 📋 MVP 实验列表](#2--mvp-实验列表)
- [3. 🔧 MVP 详细设计](#3--mvp-详细设计)
- [4. 📊 进度追踪](#4--进度追踪)
- [5. 🔗 跨仓库集成](#5--跨仓库集成)
- [6. 📎 附录](#6--附录)

---

# 1. 🎯 Phase 总览

## 1.1 Phase 列表

| Phase | 目的 | 包含 MVP | 状态 | 关键产出 |
|-------|------|---------|------|---------|
| **Phase 0: Baseline** | 建立全局 Ridge 基准 | MVP-0 | ✅ 完成 | baseline R²=0.8616 |
| **Phase 1: 分区间 Ridge** | 验证"分段简单"假设 | MVP-1.x | ✅ 完成 | ΔR²=+0.050，[M/H] 68.7% |
| **Phase 2: 按 SNR 分专家** | 验证 noise-conditioned MoE | MVP-2.0 | ✅ 完成 | ΔR²=+0.080 |
| **Phase 3: 可落地性验证** | Quantile + Pseudo + Conditional | MVP-3.x | ✅ 完成（负面+正面） | Quantile ❌，Cond ✅ 80% |
| **Phase 4: log g 门控上限** | Oracle/Pseudo/Learned gate | MVP-4.0 | ⏳ 计划中 | - |
| **Phase 5: 系数解释** | Ridge 系数波段分析 | MVP-5.0 | ✅ 完成 | Ca II 1.65× |
| **Phase 6: NN-MoE** | 非线性 MoE（条件满足才做） | MVP-6.x | ⏳ 计划中 | - |
| **Phase 7: 连续条件化** | 从离散 → 连续条件模型 | MVP-7.x | ❌ 取消 | Gate 已解决，不需要 |
| **🟢 Phase 8: 物理窗 Gate** | 用物理窗特征验证 gate 可落地性 | MVP-PG1~PG3 | ✅ **MVP-PG1 完成** | **ρ=1.00 超预期！** |
| **🟢 Phase 9: 9 专家扩展** | 物理窗 gate → 9 专家 (Teff×[M/H]) | MVP-9E1 | ✅ **完成！** | **R²=0.9213, ρ=1.13** |
| **⚠️ Phase 10: NN Expert** | 固定 gate + NN expert | MVP-NN1 | ✅ 完成 | ⚠️ NN<<Ridge，暂停 |
| **🟢 Phase 11: 优化 & 工程化** | 回归最优 gate + coverage + 校准 | MVP-Next-A/B/C | **MVP-Next-A ✅, MVP-Next-B ✅** | M2 里程碑 |
| **🔴 Phase 12: 大规模验证** | 100k 复刻 + Coverage++ | MVP-12A/12B | ⏳ 立项中 | 稳态结论 + full>0.91 |
| **🟡 Phase 13: 特征增强 & 小模型** | Feature mining + embedding + LGBM expert | MVP-13/14/15 | ⏳ 立项中 | Bin3/Bin6 增量改进 |

## 1.2 依赖关系图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MVP 实验依赖图 (Phase 11-13 当前重点)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ════════════════════ 已完成路径 ════════════════════                  │
│                                                                         │
│   Phase 0-2: ✅ MoE 有结构 (ΔR²=0.050)                                  │
│         │                                                               │
│         ▼                                                               │
│   Phase 3: ✅ 可落地性验证 → Pseudo ❌, Cond ✅ 80%                       │
│         │                                                               │
│         ▼                                                               │
│   Phase 5: ✅ 系数解释 → Ca II 重要                                      │
│         │                                                               │
│         ▼                                                               │
│   Phase 8: ✅ 物理窗 Gate → ρ=1.00 超预期！                              │
│         │                                                               │
│         ▼                                                               │
│   Phase 9: ✅ 9 专家扩展 → R²=0.9213, ρ=1.13                            │
│         │                                                               │
│         ▼                                                               │
│   Phase 10: ⚠️ NN Expert → NN<<Ridge，暂停                              │
│         │                                                               │
│         ▼                                                               │
│   Phase 11: ✅ M2 里程碑达成                                             │
│   ├─ MVP-Next-A ✅ 回归 Gate → R²=0.9310                                │
│   ├─ MVP-Next-B ✅ Full Coverage → R²=0.8957                            │
│   └─ MVP-Next-C ❌ 校准假设被否定                                        │
│                                                                         │
│   ════════════════════ 当前执行 ════════════════════                    │
│                                                                         │
│   🔴 Phase 12: 大规模验证                                                │
│   ├─ [MVP-12A: 100k 复刻] 🔴 P0                                         │
│   │   → 验证 0.9310 在 100k 规模可复现                                   │
│   │   → 同 split 对比 LGBM=0.91                                         │
│   │                                                                     │
│   └─ [MVP-12B: Coverage++] 🔴 P0                                        │
│       → 第 10 个 oor expert                                              │
│       → full R² > 0.91                                                  │
│                                                                         │
│   🟡 Phase 13: 特征增强 & 小模型 (P1 待验证)                             │
│   ├─ [MVP-13: Feature Mining] → Bin3/Bin6 窗口特征                      │
│   ├─ [MVP-14: Embedding Gate] → 1M 参数 CNN/AE                          │
│   └─ [MVP-15: LGBM Expert] → 替换困难 bin expert                        │
│                                                                         │
│   ════════════════════ 决策点 ════════════════════                      │
│                                                                         │
│   MVP-12A 完成后：                                                       │
│   ├─ 如果 100k R² ≥ 0.93 且 > LGBM → ✅ MoE 稳态结论                    │
│   └─ 如果 R² 下降或 CI 包含 0 → 分析难样本                               │
│                                                                         │
│   MVP-12B 完成后：                                                       │
│   ├─ 如果 full R² > 0.91 → ✅ 可交付                                    │
│   └─ 如果 full R² < 0.88 → 转 Phase 13 特征增强                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1.3 决策点

| 决策点 | 触发条件 | 选项 A | 选项 B |
|--------|---------|--------|--------|
| D1 | MVP-1.0 完成后 | ✅ ΔR² ≥ 0.03 → 继续 Phase 2+ | 如果 < 0.03 → 终止 |
| D2 | MVP-3.1 完成后 | 如果 Pseudo ≥ 70% → Phase 6 | ❌ < 50% → 转 Phase 7 |
| D3 | MVP-7.1 完成后 | 如果 Cond 保住 ≥60% → 主线用 Cond | 如果两者都崩 → 需要 latent gate |
| D4 | MVP-7.2 完成后 | 如果 ≥90% MoE → 用 Cond，放弃 MoE | 如果 <80% → 继续优化门控 |

---

# 2. 📋 MVP 实验列表

## 2.1 实验总览

| MVP | 实验名称 | Phase | 状态 | experiment_id | 报告链接 |
|-----|---------|-------|------|---------------|---------|
| MVP-0 | 全局 Ridge Baseline | 0 | ✅ | - | (包含在 MVP-1.0) |
| MVP-1.0 | Piecewise Ridge | 1 | ✅ | `VIT-20251203-moe-piecewise-01` | [exp](./exp_moe_piecewise_ridge_20251203.md) |
| MVP-1.1 | 严谨验证 | 1 | ✅ | `VIT-20251203-moe-rigorous-01` | [exp](./exp_moe_rigorous_validation_20251203.md) |
| MVP-2.0 | Noise-conditioned Ridge | 2 | ✅ | `VIT-20251203-moe-snr-02` | [exp](./exp_moe_noise_conditioned_20251203.md) |
| MVP-3.0 | Quantile Bins Sweep | 3 | ✅ | `VIT-20251203-moe-quantile-01` | [exp](./exp_moe_quantile_bins_sweep_20251203.md) |
| MVP-3.1 | Pseudo Gating | 3 | ✅ | `VIT-20251203-moe-pseudo-01` | - |
| MVP-3.2 | Conditional Ridge | 3 | ✅ | `VIT-20251203-moe-conditional-01` | [exp](./exp_moe_conditional_ridge_20251203.md) |
| MVP-4.0 | log g Gate Analysis | 4 | ⏳ | `VIT-20251203-moe-logg-gate-01` | - |
| MVP-5.0 | Ridge 系数解释 | 5 | ✅ | `VIT-20251203-moe-coef-01` | [exp](./exp_moe_coefficient_analysis_20251203.md) |
| MVP-6.0 | Learned Gate | 6 | ⏳ | - | - |
| MVP-6.1 | NN-MoE | 6 | ⏳ | `VIT-20251203-moe-nn-03` | - |
| MVP-7.1 | Gate 噪声敏感性 | 7 | ⏳ 暂缓 | - | - |
| MVP-7.2 | Conditional Ridge++ | 7 | ⏳ 暂缓 | - | - |
| MVP-7.3 | Noise 连续条件化 | 7 | ⏳ 暂缓 | - | - |
| MVP-7.4 | 物理窗门控 | 7 | → Phase 8 | - | - |
| **MVP-PG1** | **🟢 物理窗 Gate Baseline** | 8 | ✅ **完成 ρ=1.00！** | `VIT-20251204-moe-phys-gate-01` | [exp](./exp_moe_phys_gate_baseline_20251204.md) |
| ~~MVP-PG2~~ | ~~窗口形状 PCA Gate~~ | 8 | ❌ 不需要 | - | ρ≈1.00 已足够 |
| MVP-PG3 | 小 CNN Gate | 8 | ⏳ 可选 | - | 锦上添花 |
| **🟢 MVP-9E1** | **物理窗 gate → 9 专家** | **9** | **✅ 完成！** | `VIT-20251204-moe-9expert-01` | [exp](./exp_moe_9expert_phys_gate_20251204.md) |
| **⚠️ MVP-NN1** | **固定 gate + NN expert** | **10** | **✅ 完成** | `VIT-20251204-moe-nn-expert-01` | [exp](./exp_moe_nn_experts_20251204.md) |
| **✅ MVP-Next-A** | **回归最优 soft mixing** | **11** | **✅ R²=0.9310 完成！** | `VIT-20251204-moe-regress-gate-01` | [exp](./exp_moe_regression_gate_20251204.md) |
| **✅ MVP-Next-B** | **100% coverage** | **11** | **✅ R²=0.8957 完成！** | `VIT-20251204-moe-full-coverage-01` | [exp](./exp_moe_full_coverage_20251204.md) |
| **❌ MVP-Next-C** | **Expert 校准** | **11** | **❌ 完成 (Negative)** | `VIT-20251204-moe-calibration-01` | [exp](./exp_moe_expert_calibration_20251204.md) |
| **🔴 MVP-12A** | **100k 规模复刻 Next-A** | **12** | **⏳ 立项中** | `VIT-20251205-moe-100k-01` | (待创建) |
| **🔴 MVP-12B** | **Coverage++** | **12** | **⏳ 立项中** | `VIT-20251205-moe-coverage-plus-01` | (待创建) |
| **🟡 MVP-13** | **Feature mining Bin3/Bin6** | **13** | **⏳ 立项中** | `VIT-20251205-moe-feature-mining-01` | (待创建) |
| **🟡 MVP-14** | **1M embedding for gate** | **13** | **⏳ 立项中** | `VIT-20251205-moe-embedding-01` | (待创建) |
| **🟡 MVP-15** | **Hard bins 小 LightGBM expert** | **13** | **⏳ 立项中** | `VIT-20251205-moe-lgbm-expert-01` | (待创建) |

## 2.2 配置速查表

| MVP | 数据规模 | 特征维度 | 模型 | 关键变量 | 验收标准 |
|-----|---------|---------|------|---------|---------|
| MVP-0 | 全量 | 全谱 | Ridge | $\alpha$ | 获得 baseline |
| MVP-1.0 | 分 9 bin | 全谱 | 9× Ridge | $(T_{\text{eff}}, [\text{M/H}])$ bin | ΔR² ≥ 0.03 |
| MVP-1.1 | 分 9 bin | 全谱 | 9× Ridge | mask-aligned | CI 验证 |
| MVP-2.0 | 按 noise 分 | 全谱 | 5× Ridge | noise level | 专家 vs 混合 |
| MVP-3.0 | quantile bins | 全谱 | K× Ridge | K=2~6 | 最佳 K |
| MVP-3.1 | quantile bins | 全谱 | K× Ridge | pseudo vs oracle | ≥70% |
| MVP-3.2 | 全量 | $3\times$全谱 | 条件 Ridge | $(x, m\cdot x, m^2\cdot x)$ | ΔR² ≥ 0.04 |
| MVP-5.0 | 全量 | 系数分析 | - | 波段差分 | 物理解释 |
| MVP-7.1 | 全量 | 全谱 | Hard MoE vs Cond | gate noise $\sigma$ | Cond 保住 ≥60-80% |
| MVP-7.2 | 全量 | 扩展特征 | Cond Ridge++ | Teff 交互项 | ≥90% MoE 增益 |
| MVP-7.3 | 全量 | 全谱 | SNR-cond Ridge | 连续 noise | 消除 noise=0.5 翻车 |
| **MVP-PG1** | 全量 | **11 维物理窗** | **LogReg (C=10)** | Ca II/Na/PCA1-2 | ✅ **ρ=1.00！** |
| ~~MVP-PG2~~ | ~~全量~~ | ~~窗口标量+PCA~~ | ~~LogReg/MLP~~ | ~~窗口形状 PCA~~ | ❌ 不需要 |
| MVP-PG3 | 全量 | 窗口原始值 | 小 CNN gate | learned features | ⏳ 可选 |
| **MVP-9E1** | 全量 | **~15 维物理窗** | **LogReg 9-class** | Ca II/Na/PCA1-4 | **R² ≥ 0.90, ρ ≥ 0.85** |
| **MVP-NN1** | 全量 | 全谱 | **3× MLP expert** | 固定 LogReg gate | **ΔR² ≥ 0.02 vs global NN** |
| **MVP-Next-A** | 全量 | 13 维 gate 特征 | **小 MLP/线性 gate** | 回归 MSE 损失 | **R² > 0.9213 + 0.003** |
| **MVP-Next-B** | full-1000 | 13 维 gate 特征 | 9-expert + fallback | edge-clamp/extra expert | **full R² > global + 0.05** |
| **MVP-Next-C** | 全量 | 9-expert 输出 | **affine 校准** | per-expert $a_k, b_k$ | **Bin3/Bin6 ΔR² ≥ 0.02** |

| **MVP-12A** | 100k train / 更大 test | 全谱 | **9× Ridge + 回归 gate** | 100k 规模 | **covered R² ≥ 0.93, CI_low > 0, MoE > LGBM** |
| **MVP-12B** | full coverage | 全谱 + window 特征 | **第 10 个 oor expert / cond fallback** | out-of-range 专家 | **full R² ≥ max(LGBM, global+0.05)** |
| **MVP-13** | 全量 | **新增窗口特征** | Ridge + 新特征 | 选线窗口 depth/EW/shape | **Bin3 或 Bin6 ΔR² ≥ +0.02** |
| **MVP-14** | 全量 | **候选窗口 + 上下文** | **小 CNN/AE ~1M 参数** | 8~32 维 embedding | **R² +0.003 或 Bin3/Bin6 改善** |
| **MVP-15** | 全量 | **选线窗口特征** | **Bin3/Bin6 用 LGBM expert** | stacking-safe OOF | **full R² > 0.91, Bin3/Bin6 不拖后腿** |
---

# 3. 🔧 MVP 详细设计

> **导航说明**：
> - 🔴 **当前执行**：Phase 12-13（大规模验证 + 特征增强）
> - ✅ **已完成**：Phase 7(取消), 8, 9, 10, 11
> - ⏳ **计划中**：Phase 4, 6

---

## 🔴 当前执行：Phase 12-13

<details>
<summary><b>❌ Phase 7: 连续条件化（已取消 - 物理窗 Gate 已解决问题）</b></summary>

## Phase 7: 连续条件化（❌ 已取消 - 物理窗 Gate 已解决问题）

### MVP-7.1: Gate 噪声敏感性曲线 ❌ 已取消

| 项目 | 配置 |
|------|------|
| **目标** | 量化 gate 噪声对 Hard MoE vs Conditional Ridge 的影响差异 |
| **验证假设** | H-A1: 连续条件模型对 gate 噪声更耐受 |
| **方法** | 用真值 $m=[\text{M/H}]$ 作为条件输入时的性能作为上限；然后对 $m$ 注入逐级噪声 $\tilde{m} = m + \epsilon,\ \epsilon \sim \mathcal{N}(0, \sigma^2)$，$\sigma \in \{0.0, 0.1, 0.2, 0.5, 1.0\}$ |
| **对比** | (1) Hard MoE（按 $\tilde{m}$ 分桶）vs (2) Conditional Ridge（用 $\tilde{m}$ 做连续条件项） |
| **验收标准** | 如果 Conditional 在较大 $\sigma$ 下仍能保住 ≥60-80% 增益，而 Hard MoE 快速崩：**主线转向条件化，放弃离散 routing** |
| **止损点** | 如果两者都快速崩 → 说明 gate 本身就不靠谱，需要 latent gate |

**→ 对假设的影响**：这一步会直接解释 pseudo gating 7.3% 的根因（hard routing 太脆 or gate 太差）

**实验步骤**：
1. 准备 test set 的真值 $m=[\text{M/H}]$
2. 生成 5 个噪声水平的 $\tilde{m}$
3. 对每个噪声水平：
   - Hard MoE：用 $\tilde{m}$ quantile 分桶，选专家
   - Conditional Ridge：用 $\tilde{m}$ 做条件项 $(x, \tilde{m}\cdot x)$
4. 画曲线：$\sigma$ vs R² for both methods
5. 分析交叉点和崩溃速度

---

### MVP-7.2: Conditional Ridge++ ❌ 已取消

| 项目 | 配置 |
|------|------|
| **目标** | 把 Conditional Ridge 从 80% → 90%+ MoE 效果 |
| **验证假设** | 剩余差距来自 Teff 交互项/二阶项/更合理的缩放 |
| **方法** | 逐步加特征：<br>(1) 加 Teff 一阶交互：$[x,\ m\cdot x,\ t\cdot x]$<br>(2) 再加交叉项：$[x,\ m\cdot x,\ t\cdot x,\ (m\cdot t)\cdot x]$<br>(3) 最后考虑二阶（如果 1/2 不够）：$[m^2\cdot x]$ |
| **验收标准** | 达到 MoE 的 ≥90% 增益（即 $\Delta R^2 \geq 0.045$），并保持 100% coverage |
| **止损点** | 如果 Teff 交互项带来 <0.01 提升 → 说明 Teff 真的不重要，只做 [M/H] 条件化 |

---

### MVP-7.3: Noise 连续条件化 ❌ 已取消

| 项目 | 配置 |
|------|------|
| **目标** | 用连续 SNR/noise 条件化替代离散分档专家 |
| **验证假设** | H-C: 连续条件化能接近甚至超过按档专家的平均收益，并消除 noise=0.5 异常 |
| **方法** | 把 noise level / SNR 当连续变量 $n$，做条件线性：<br>(1) $[x,\ n\cdot x]$<br>(2) $[x,\ n\cdot x,\ n^2\cdot x]$ |
| **对比** | per-noise expert、mixed、continuous-conditioned |
| **验收标准** | 平均性能接近 expert（差 ≤0.01 R²），并且在 noise=0.5 上不掉队 |
| **止损点** | 如果连续化效果 < mixed → 说明 noise 确实需要离散化处理 |

---

### MVP-7.4: "物理窗"诱导的轻量门控 → 已移至 Phase 8

| 项目 | 配置 |
|------|------|
| **目标** | 用物理先验设计轻量 gate，验证 learned gate 是否比 pseudo gate 更好 |
| **验证假设** | H-B: 存在可学习的物理边界/soft gating |
| **方法** | Gate 输入只用几个"物理窗"的统计特征：<br>- Ca II triplet window 的 line depth/等效宽度 proxy<br>- 若干窗的 PCA 分量 |
| **验收标准** | Learned gate ≥ pseudo gate，且能明显超过 conditional ridge++ |
| **止损点** | 如果 learned gate ≈ conditional ridge++ → 直接用 conditional，放弃 NN-MoE |

</details>

---

<details>
<summary><b>✅ Phase 8: 物理窗 Gate（已完成 - ρ=1.00 超预期！）</b></summary>

## ✅ Phase 8: 物理窗 Gate（已完成）

> **核心目标**：用最小成本验证"物理窗特征能否把 gate 做对（至少比 pseudo 好）"
> 
> **验证指标**：$\rho = \frac{R^2_{\text{phys-gate}} - R^2_{\text{global}}}{R^2_{\text{oracle}} - R^2_{\text{global}}}$
> - 最低可用：$\rho \ge 0.5$
> - 很有戏：$\rho \ge 0.7$
> 
> **✅ 实际结果：ρ=1.00，远超预期！**

### MVP-PG1: 物理窗 Gate Baseline ✅ 完成！

| 项目 | 配置 | **实际结果** |
|------|------|------------|
| **目标** | 验证物理窗特征 gate 能否显著优于 pseudo gating | ✅ **ρ=1.00 超预期！** |
| **验证假设** | H-PG1: 物理窗特征区分专家域<br>H-PG2: Soft > Hard<br>H-PG3: Teff proxy 必要性 | ✅ Acc=82%<br>✅ Soft ρ=1.0 >> Hard ρ=0.72<br>⚠️ 有帮助但非必需 |
| **专家设置** | 固定 [M/H]-only 3 专家 Ridge | α=1.0 per expert |
| **Gate 特征** | 11 维：Ca II depth/EW×3 + EW_CaT + Na×2 + PCA1/2 | EW_8542 最重要 |
| **Gate 模型** | LogReg (L2, C=10) | 准确率 82.10% |
| **路由方式** | Hard / Soft / Soft+fallback | **Soft ρ=0.997 推荐** |
| **验收标准** | ρ mean ≥ 0.5 且 CI_low > 0.3 | ✅ **ρ=1.00, CI=[0.73,1.44]** |
| **止损点** | ρ 接近 0 → 转 Conditional | ✅ **无需转向！** |

**Gate 特征设计（最小集，先别上 PCA）**：

| 窗口 | 特征 | 说明 |
|------|------|------|
| **Ca II triplet** | depth_8498, depth_8542, depth_8662 | 各条线的 depth proxy: $d = 1 - \min(f/\hat{c})$ |
| | EW_8498, EW_8542, EW_8662 | 各条线的 EW proxy: $\sum_{core}(1 - f/\hat{c})$ |
| | **EW_CaT** | 总和特征: EW_8498 + EW_8542 + EW_8662 |
| **Na I** | depth_Na, EW_Na | 8183/8195 或 5890/5896（视波段覆盖） |
| **Teff proxy** | **PCA1, PCA2** | 全谱 PCA 前 2 维（无额外监督，最省事）|
| **可选** | noise_proxy | 局部 MAD（抗噪特征） |

**三种路由方式对比**：

| 方式 | 实现 | 预期特点 |
|------|------|---------|
| **Hard** | argmax 类别 → 选专家 | 简单但脆弱 |
| **Soft** | softmax 概率权重混合 $\hat{y} = \sum_k p_k \hat{y}_k$ | 更稳、抗噪 |
| **Soft + fallback** | max(p) < τ (如 0.5) 时回退 global Ridge | 避免"错路由灾难" |

**实验步骤**：

1. **准备专家**：用真值 [M/H] 在 train 上训练 3 个 Ridge 专家（low/mid/high）
2. **计算 oracle 性能**：test 时用真值 [M/H] 路由 → $R^2_{\text{oracle}}$
3. **提取物理窗特征**：对每个样本提取 Ca II/Na/PCA 特征
4. **训练 gate**：用特征预测 expert id（3 分类 LogReg）
5. **测试三种路由**：Hard / Soft / Soft+fallback
6. **计算 ρ**：对每种路由方式计算 ρ 和 bootstrap CI

---

### MVP-PG2: 窗口形状 PCA Gate

| 项目 | 配置 |
|------|------|
| **目标** | 验证"窗口形状信息"能否把 gate 再推一截 |
| **前提** | MVP-PG1 的 ρ 接近但未达 0.7 |
| **验证假设** | H-PG4: 标量特征不够时，窗口 PCA 能提供增量 |
| **新增特征** | 对 Ca II 窗口（或 Na 窗口）提取 **PCA 前 2-5 维** |
| **对比** | 标量特征 vs 标量+窗口PCA 的 ρ 提升幅度 |
| **验收标准** | Δρ > 0.1 |
| **止损点** | Δρ < 0.05 → 形状信息不值得，用标量足够 |

---

### MVP-PG3: 小 CNN Gate

| 项目 | 配置 |
|------|------|
| **目标** | 用小 CNN gate 吃窗口形状，输出 soft weights |
| **前提** | MVP-PG1/PG2 已验证物理窗方向可行（ρ≥0.5） |
| **验证假设** | Learned gate 能接近 oracle |
| **架构** | 小 CNN (3-5 层 Conv1D) 吃窗口区域，输出 3-class softmax |
| **专家** | 仍用 Ridge（最小 learned gate，不动专家） |
| **验收标准** | ρ ≥ 0.8 |
| **意义** | 这是"最小 learned gate"，比全谱 gate 更物理、更可解释，可连接 NN-MoE |

---

</details>

---

<details>
<summary><b>✅ Phase 9: 9 专家扩展（已完成 - R²=0.9213, ρ=1.13 大成功！）</b></summary>

## ✅ Phase 9: 9 专家扩展（已完成）

### MVP-9E1: 物理窗 Soft Gate → 9 专家 ✅ 已完成

| 项目 | 配置 |
|------|------|
| **目标** | 把物理窗 gate 从 3 专家扩展到 9 专家 (Teff×[M/H])，直接抬可落地 R² 到 ~0.91 |
| **验证假设** | H-9E1: 物理窗特征（CaT/Na + PCA 作为 Teff proxy）能在 9 类上用 Soft routing 保住大部分 oracle 增益 |
| **专家设置** | 9 个 Ridge 专家，沿用 `moe_piecewise_ridge.py` 的 bin 定义 |
| **Gate 特征** | ~15 维：MVP-PG1 的 11 维 + PCA3/4（可能需要更多 Teff 信息） |
| **Gate 模型** | LogReg 9-class（先行），如不够再试 MLP |
| **路由方式** | **只做 Soft routing + fallback**（Hard 已证明损失 28%，不值得试） |
| **验收标准** | **目标 1**: R²_phys-gate(9) ≥ 0.90 / **目标 2**: ρ ≥ 0.85 / **目标 3**: 100% coverage (fallback) |
| **止损点** | 如果 9-class gate 准确率 <40% → 考虑 hierarchical gate (先 [M/H] 再 Teff) |

**复用代码策略**：
1. 专家训练：`scripts/moe_piecewise_ridge.py` 的 bin 定义和专家训练逻辑
2. 特征提取：`scripts/moe_phys_gate_baseline.py` 的 `extract_physical_features()` + PCA (noisy fit/transform)
3. 评估框架：沿用 ρ 定义和 bootstrap CI

**实验步骤**：
```
Step 1: 准备 9 专家
├── 用真值 (Teff, [M/H]) 把 train 分成 9 组
├── 每组训练一个 Ridge 专家
└── 记录 α_optimal

Step 2: 计算 Oracle 性能
├── Test 时用真值 (Teff, [M/H]) 路由到对应专家
└── 得到 R²_oracle(9) ≈ 0.9116

Step 3: 扩展物理窗特征
├── 复用 MVP-PG1 的 11 维特征
├── 可能加 PCA3/4（更多 Teff proxy）
└── 可能加 H 线相关窗（帮助区分 Teff）

Step 4: 训练 9-class Gate
├── LogReg 9-class (L2, C=CV)
├── 如果准确率 <50%，尝试小 MLP
└── 记录分类准确率和混淆矩阵

Step 5: 只做 Soft routing
├── soft = sum(p_k * y_hat_k)
├── soft+fallback = max(p)<0.4 时用 global
└── 不试 Hard（已证明损失大）

Step 6: 计算 ρ 和 CI
├── ρ = (R²_phys-gate(9) - R²_global) / (R²_oracle(9) - R²_global)
└── Bootstrap CI
```

---

</details>

---

<details>
<summary><b>⚠️ Phase 10: NN Expert（已完成 - NN<<Ridge，暂停）</b></summary>

## ⚠️ Phase 10: NN Expert（已完成 - 结论：全谱 MLP 不适合）

### MVP-NN1: 固定物理 gate + NN experts ⚠️ 已完成

| 项目 | 配置 |
|------|------|
| **目标** | Gate 已不是瓶颈，验证把 Ridge expert 换成 NN expert 能否再涨 |
| **验证假设** | H-NN1: MoE 的 R² 上限被 expert 表达能力限制；NN expert 能改善困难子域 |
| **Gate 设置** | **固定 MVP-PG1 的 LogReg gate**（不 end2end），保证因果清晰——涨了就是 expert 变强 |
| **专家设置** | 先 K=3 [M/H] 专家（因为 [M/H] 是主贡献维度，最稳配置） |
| **Expert 模型** | 小 MLP (2-3 层, 128-256 hidden, ReLU, Dropout=0.2) |
| **对照组** | global NN baseline vs "3-expert NN + soft gate"；参数量/训练预算对齐 |
| **验收标准** | ΔR² ≥ 0.02 相对 global NN；困难子域 (Bin4/Bin7) 误差明显下降 |
| **止损点** | 如果 ΔR² < 0.01 → expert 变强不是主要增益来源，需要更强 gate/更多专家 |

**关键设计原则**：
1. **Gate 固定**：用 MVP-PG1 训好的 LogReg，不参与 NN 训练
2. **对照公平**：global NN 和 expert NN 参数量/训练 epoch 尽量对齐
3. **定位增益**：按子域看误差，重点盯 Bin4/Bin7

**实验步骤**：
```
Step 1: 准备数据
├── 复用 MVP-PG1 的数据和 [M/H] 分组
└── train/test split 保持一致

Step 2: 训练 Global NN Baseline
├── 小 MLP (2 层, 256 hidden)
├── 记录 R²_global_nn
└── 记录各子域性能（特别是 Bin4/Bin7）

Step 3: 训练 3× NN Expert
├── 每个 expert 相同架构 (2 层, 256 hidden)
├── 用真值 [M/H] 分组训练
└── 参数量 ≈ global NN

Step 4: 用固定 gate + soft routing 测试
├── 复用 MVP-PG1 的 LogReg gate
├── 只用 Soft routing
└── 得到 R²_moe_nn

Step 5: 分析增益来源
├── 总体 ΔR² = R²_moe_nn - R²_global_nn
├── 按子域看：哪些 bin 改善最多？
└── 特别关注 Bin4/Bin7（困难子域）
```

**后续可选扩展**：
- 如果 K=3 有效 → 扩展到 K=9
- 如果困难子域改善明显 → 给 Bin4/Bin7 专家加 H 线相关窗特征

---

</details>

---

<details>
<summary><b>✅ Phase 11: 优化 & 工程化（M2 里程碑达成！）</b></summary>

## ✅ Phase 11: 优化 & 工程化（M2 里程碑达成）

> **核心目标**：将 9 专家物理 MoE 升级为"回归最优 soft mixing + 100% coverage"，目标是在 full-test 上稳定超过当前分类 gate 的 R²。
> 
> **里程碑 M2**：✅ 已达成！回归 gate R²=0.9310 + full coverage R²=0.8957

### MVP-Next-A: 回归最优 Soft Mixing ✅ 已完成 (R²=0.9310)

| 项目 | 配置 |
|------|------|
| **目标** | 把 gate 从"分类最优"升级为"回归最优"的 soft mixing |
| **验证假设** | H-A: 当前用 LogReg 做 9-class 分类得到的概率 $p_k$ 不是让 log g MSE 最小的权重；如果直接用验证集最小化 log g 误差来学习权重映射，R² 还能涨一截（尤其在 bin 边界与 metal-poor bins） |
| **专家设置** | **冻结 9 个 ridge experts**（完全复用 MVP-9E1 的专家） |
| **Gate 设置** | 不再用 class CE loss，而是学一个小 gate：输入 13D gate features → 输出 9 个权重（softmax）→ 损失：**validation MSE（回归）** |
| **对照组** | (1) 现有分类 LogReg + soft routing（baseline = 0.9213）; (2) 回归训练的 soft weights（新） |
| **验收标准** | 整体 R² 明显高于 0.9213（哪怕 +0.003~0.01 都是很真实的提升）；重点看 Bin3/Bin6 |
| **止损点** | 如果 ΔR² < 0.001 → 说明分类权重已经接近最优 |

**实验步骤**：
```
Step 1: 准备
├── 复用 MVP-9E1 的 9 个 Ridge expert（完全冻结）
├── 复用 13 维 gate 特征
└── 分出 held-out validation set (用于训练 gate)

Step 2: 训练回归 Gate
├── 方法 A: 小 MLP (13 → 64 → 9) + softmax + MSE loss
├── 方法 B: 线性层 (13 → 9) + softmax + MSE loss
├── 在 validation set 上最小化 MSE
└── 用 test set 评估

Step 3: 对比
├── 分类 gate (LogReg): R²_classify
├── 回归 gate (小 MLP/线性): R²_regress
└── 重点看 per-bin 改善，尤其 Bin3/Bin6

Step 4: 统计验证
├── Bootstrap CI 验证 ΔR² 的显著性
└── 如果 ΔR² > 0.003 且 CI_low > 0 → 采用回归 gate
```

**好处**：不需要 NN experts，也不需要换特征；纯粹把"权重学习目标"从分类改成回归，信息增量最大。

---

### MVP-Next-B: 100% Coverage ✅ 完成

| 项目 | 配置 |
|------|------|
| **目标** | 做"100% coverage 的最终指标"，确保增益不依赖于筛样本 |
| **验证假设** | H-B: 把 out-of-range 样本纳入统一流程后，整体 R² 仍维持在一个"实际可用"的高水平 |
| **背景** | MVP-9E1 test 1000 里 816 covered（教授一定会问！） |
| **策略选项** | (1) **Edge-clamp**: 落到最近的 Teff/[M/H] bin; (2) **Extra "other" expert**: 第 10 个专家训练 out-of-range train; (3) **Global fallback**: 最稳但通常会稍降 |
| **汇报方式** | 两套：covered-only（便于对比历史）+ full-1000（可展示/可交付版本） |
| **验收标准** | full-1000 的 R² 仍显著高于 global，并且"差距原因可解释" |
| **止损点** | 如果 full-1000 R² < global R² + 0.03 → 需要分析 out-of-range 样本特性 |

**实验步骤**：
```
Step 1: 分析 out-of-range 样本
├── 识别 184 个 out-of-range 样本
├── 分析其 (Teff, [M/H]) 分布
└── 理解为什么落在 9 bin 之外

Step 2: 实现三种策略
├── (A) Edge-clamp: 映射到最近的边界 bin
├── (B) Extra expert: 训练第 10 个专家
└── (C) Global fallback: 直接用全局 Ridge

Step 3: 评估
├── covered-only (816): R²_covered
├── out-of-range (184): R²_oor
├── full-1000: R²_full = weighted average
└── 对比三种策略的 R²_full

Step 4: 选择最优策略
├── 如果 edge-clamp 效果好 → 最简单
├── 如果 extra expert 更好 → 更精确
└── 输出可交付版本的最终指标
```

---

### MVP-Next-C: Expert 校准 🔴 冲 R² 主战场

| 项目 | 配置 |
|------|------|
| **目标** | 专打最弱 bins 的"偏差项/校准"，提升总体 R² |
| **验证假设** | H-C: 最弱 bins (Bin3/Bin6) 的误差主要是"系统性 bias/尺度不匹配"，先做轻量校准就能提升 |
| **背景** | Bin3 (Mid Metal-poor) 和 Bin6 (Hot Metal-poor) 虽然 soft 已经救了很多，但 R² 仍明显低于其他 bin |
| **方法** | 在验证集上对每个 expert 输出做 **affine 校准**: $\hat{y}'_k = a_k \hat{y}_k + b_k$，然后再做 soft mixing: $\hat{y} = \sum_k p_k \hat{y}'_k$ |
| **对照** | 无校准 vs 校准 |
| **验收标准** | Bin3/Bin6 的 R² 有明确提升，同时整体 R² 增加 |
| **止损点** | 如果校准后 ΔR² < 0.005 → 说明偏差不是主因，需要看特征 |

**实验步骤**：
```
Step 1: 分析 per-expert 偏差
├── 对每个 expert k，在 validation 上画 pred vs true
├── 检查是否有系统性 bias (截距偏移) 或 scale mismatch (斜率 ≠ 1)
└── 重点看 Bin3 和 Bin6

Step 2: 学习校准参数
├── 对每个 expert k，用 validation 拟合 a_k, b_k
├── 方法: 简单线性回归 y ~ a_k * y_hat_k + b_k
└── 共 9×2=18 个参数

Step 3: 应用校准
├── 对每个 expert 输出做 affine 变换
├── 然后用 soft gate 权重混合
└── 在 test 上评估

Step 4: 分析增益来源
├── 整体 ΔR² = R²_calibrated - R²_uncalibrated
├── Per-bin ΔR²，重点看 Bin3/Bin6
└── 如果有效 → 纳入最终流程
```

**保持可解释性**：这个方法是线性、透明的，每个专家的校准参数 $(a_k, b_k)$ 都有明确的物理解释。

</details>

## 🔴 当前执行：Phase 12-13
</details>

---

## 🔴 当前执行：Phase 12-13

---

## 🔴 Phase 12: 大规模验证（2025-12-05 新立项）

> **核心目标**：把 0.9310 从 32k/816 test 升级为 100k 规模的"稳态结论"，并把 full coverage 拉回 >0.91
>
> **总策略**：先巩固稳态，再超越 LGBM=0.91

### MVP-12A: 100k 规模复刻 Next-A 🔴 P0 最高优先级

| 项目 | 配置 |
|------|------|
| **目标** | 把 0.9310 从"32k/816 test"升级成"100k/更大 test 的稳定结论"，并给出更硬的 CI |
| **验证假设** | H-12A: 100k 规模下 MoE R²≥0.93 可复现，CI_low>0，且明显优于 LGBM=0.91 |
| **专家设置** | 仍然 9×Ridge（和现在一致），但用 100k train 重新训练 |
| **Gate 设置** | 仍然回归 gate(MLP) soft mixing（Next-A 配置） |
| **Baseline 对照** | 同一 split 下同时跑 **LGBM baseline**，确保公平 |
| **验收标准** | covered-test R² ≥ 0.93（或更稳的 CI_low > 0）；同 split 下 MoE 明显高于 LGBM 0.91 |
| **止损点** | 如果 R² 下降或 CI 包含 0 → 需要分析是否 100k 带来更多难样本 |

**实验步骤**：
```
Step 1: 准备 100k 数据
├── 重新 split train/val/test
├── 确保 test 更大、分布更接近真实
└── 记录各 bin 样本数

Step 2: 训练 9 专家
├── 用真值 (Teff, [M/H]) 分组
├── 每组用 100k 子集训练 Ridge
└── 记录 α_optimal

Step 3: 训练回归 Gate
├── 复用 Next-A 的 MLP gate 配置
├── 在 val 上最小化 MSE
└── 输出 R²_moe

Step 4: LGBM Baseline 对照
├── 同 split 训练 LGBM
├── 使用相同特征
└── 输出 R²_lgbm

Step 5: 统计验证
├── Bootstrap CI for R²_moe
├── 公平对比 R²_moe vs R²_lgbm
└── 确认 MoE > LGBM 的显著性
```

---

### MVP-12B: Coverage++ 🔴 P0

| 项目 | 配置 |
|------|------|
| **目标** | 让 full-test R² 接近 covered-test，使 MoE 可交付 |
| **验证假设** | H-12B: 第 10 个 out-of-range expert 能把 full-test R² 拉回 >0.91 |
| **背景** | 当前最大短板不是 0.93，而是 full coverage 版本拉胯（会直接输给 LGBM） |
| **策略 1** | **第 10 个 "out-of-range expert"**：专门用 out-of-range 的 train 样本训练 |
| **策略 2** | fallback 不是 global ridge，而是 **conditional ridge / 或者 lightgbm-lite** |
| **输出指标** | 同时输出 covered & full 两套指标，保留可比性 |
| **验收标准** | full-test R² ≥ max(LGBM, global+0.05)；至少要把 full 拉回到 >0.91 |
| **止损点** | 如果 full R² < 0.88 → 分析 out-of-range 样本特性 |

**实验步骤**：
```
Step 1: 分析 out-of-range 样本
├── 识别落在 9 bin 之外的样本
├── 分析其 (Teff, [M/H]) 分布
└── 理解为什么落在 9 bin 之外

Step 2: 训练第 10 个专家
├── 只用 out-of-range train 样本
├── 使用 Ridge 或 LGBM-lite
└── 记录该专家的单独性能

Step 3: 扩展 Gate
├── 方案 A: 10-class gate
├── 方案 B: 9-class + "其他"检测器
└── 方案 C: 概率低于阈值时 fallback

Step 4: 评估
├── covered-only (原 816)
├── out-of-range only
├── full coverage (1000)
└── 对比三种策略
```

---

## 🟡 Phase 13: 特征增强 & 小模型（P1 待验证）

> **核心目标**：所有创新都只允许针对 Bin3/Bin6 做增量
>
> **前提**：H-C 校准失败表明 Metal-poor 误差不是简单 bias，需要特征/容量/分布层面改进

### MVP-13: Feature Mining Bin3/Bin6 🟡 P1

| 项目 | 配置 |
|------|------|
| **目标** | 只改善 Bin3/Bin6（Metal-poor bins） |
| **验证假设** | H-13: 用 ridge 系数/残差选线添加窗口特征能改善 Bin3/Bin6 |
| **背景** | H-C 校准失败说明 Metal-poor 误差不是简单 bias，是"缺信息/异质性" |
| **方法** | 用 ridge expert 系数/残差做"选线"：找对 log g 或 [M/H]/Teff 最敏感的局部窗 |
| **新增特征** | 把这些窗的 depth/EW/shape 特征写进 gate features 或 Bin3/Bin6 专用 expert features |
| **注意** | 每次只加一类窗口（比如新增 1 组线），做 ablation，避免"堆一堆不知道谁有用" |
| **验收标准** | Bin3 或 Bin6 至少一个 **ΔR² ≥ +0.02**，否则立刻止损 |
| **止损点** | 如果 ΔR² < 0.01 → 停止该方向，转 MVP-15 |

**候选窗口**：
- Ca II 以外的金属线（Na, Mg, Fe...）
- H 线 (Hα/Hβ) — 可能帮助区分 Teff
- 弱金属线特征（在 Metal-poor 区域可能更敏感）

---

### MVP-14: 1M 参数 Embedding for Gate 🟡 P1

| 项目 | 配置 |
|------|------|
| **目标** | 学一个更强的低维表征（embedding）当 gate 的 Teff/[M/H] proxy |
| **验证假设** | H-14: 小模型 (1M 参数) 学习的 embedding 能改善 gate 质量 |
| **定位** | **不是替代专家**，只是给 gate 提供更好的输入（专家换 NN 已踩过坑） |
| **架构** | 小的 1D-CNN/autoencoder/supervised proxy（参数量 ~1M） |
| **输入** | 只用"候选窗口 + 少量上下文"（不需要全谱） |
| **输出** | 8~32 维 embedding，拼到现有 13D gate features 里 |
| **Gate** | 仍然用回归 gate(MLP) 学 soft weights（Next-A 主线不变） |
| **验收标准** | 总体 R² +0.003 以上 或 Bin3/Bin6 明显改善 |
| **止损点** | 如果 ΔR² < 0.001 → embedding 不值得，维持现有 gate features |

---

### MVP-15: Hard Bins 小 LightGBM Expert 🟡 P1

| 项目 | 配置 |
|------|------|
| **目标** | 用小 LightGBM 替换/增强 Bin3/Bin6 的 expert |
| **验证假设** | H-15: 用小 LightGBM 替换困难 bin expert 能改善该区域性能 |
| **定位** | **只替换 Bin3、Bin6 的 expert**（或作为 residual corrector），其余 bin 继续 Ridge |
| **输入** | 不要全谱，优先用"MVP-13 选出来的窗口特征 + 少量 summary" |
| **关键防翻车** | **stacking-safe** 版本：对 gate/二级模型用 **out-of-fold 的 expert 预测** 训练（避免 meta-model 过拟合） |
| **验收标准** | full coverage 稳定超过 0.91，Bin3/Bin6 不再拖后腿 |
| **止损点** | 如果 LGBM expert 只提升 <0.01 → 说明特征才是瓶颈，回到 MVP-13 |

**实验步骤**：
```
Step 1: 选择窗口特征
├── 复用 MVP-13 选出的窗口
├── 提取 depth/EW/shape 特征
└── 加少量 summary（如 PCA 前几维）

Step 2: 训练 LGBM Expert
├── 只为 Bin3 训练一个 LGBM
├── 只为 Bin6 训练一个 LGBM
└── 使用 OOF 预测避免过拟合

Step 3: 集成
├── Bin3/Bin6 用 LGBM expert
├── 其他 bin 继续用 Ridge expert
└── Gate 权重保持不变

Step 4: 评估
├── per-bin R² 对比
├── full coverage R²
└── 确认 Bin3/Bin6 不再拖后腿
```

## 已完成 Phase 详细设计

### Phase 1: 分区间 Ridge

#### MVP-1.0: Piecewise Ridge ✅

| 项目 | 配置 |
|------|------|
| **目标** | 验证按 $(T_{\text{eff}}, [\text{M/H}])$ 分 bin 后，分区 Ridge 是否比全局 Ridge 更好 |
| **分 bin 方案** | $T_{\text{eff}} \in [3750, 4500), [4500, 5250), [5250, 6000]$; $[\text{M/H}] \in [-2, -1), [-1, 0.0], (0.0, 0.5]$ → 3×3=9 bins |
| **结果** | ⚠️ 原 ΔR²=0.078 被高估，见 MVP-1.1 |

#### MVP-1.1: 严谨验证 ✅

| 项目 | 配置 |
|------|------|
| **目标** | mask-aligned 公平比较 + Bootstrap CI + 消融 |
| **结果** | **ΔR²=0.050**, CI=[0.033, 0.067]；[M/H] 贡献 68.7% |

---

### Phase 2: 按 SNR 分专家

#### MVP-2.0: Noise-conditioned Ridge ✅

| 项目 | 配置 |
|------|------|
| **目标** | 验证为每个 noise level 单独训练模型是否比混合训练更好 |
| **结果** | **平均 ΔR²=+0.080**；但 noise=0.5 翻车 |

---

# 4. 📊 进度追踪

## 4.1 看板视图

```
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│   ⏳ 计划中   │  🔴 待执行   │  🚀 运行中   │   ✅ 已完成   │   ❌ 已取消   │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ MVP-4.0      │ MVP-12A      │              │ MVP-0        │ MVP-7.1~7.3  │
│ MVP-6.0      │ MVP-12B      │              │ MVP-1.0      │ MVP-PG2      │
│ MVP-PG3      │ MVP-13~15    │              │ MVP-1.1 ✓    │ MVP-Next-C ❌│
│              │              │              │ MVP-2.0      │              │
│              │              │              │ MVP-3.0 ✓    │              │
│              │              │              │ MVP-3.1 ❌   │              │
│              │              │              │ MVP-3.2 ✓    │              │
│              │              │              │ MVP-5.0 ✓    │              │
│              │              │              │ MVP-PG1 ✓ 🟢 │              │
│              │              │              │ MVP-NN1 ✓ ⚠️ │              │
│              │              │              │ MVP-9E1 ✓ 🟢 │              │
│              │              │              │ MVP-Next-A ✓ │              │
│              │              │              │ MVP-Next-B ✓ │              │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

✅ Phase 11 完成（2025-12-04）：
- MVP-Next-A ✅: 回归最优 soft mixing → R²=0.9310
- MVP-Next-B ✅: 100% coverage → R²=0.8957
- MVP-Next-C ❌: Expert 校准 → 假设被否定
- M2 里程碑：✅ 达成！回归 gate + full coverage 验证完成

🔴 Phase 12-13 立项（2025-12-05）：
- MVP-12A: 100k 规模复刻（最高优先级）
- MVP-12B: Coverage++（可交付性）
- MVP-13~15: 特征增强 & 小模型（P1 待验证）

🟢 Phase 9 完成（2025-12-04）：
- MVP-9E1 ✅ 大成功！R²=0.9213, ρ=1.13, Gate 准确率 94.6%

⚠️ Phase 10 MVP-NN1 完成（2025-12-04）：
- ΔR²=+0.257（MoE vs Global）✅ 达标
- 但 NN R²=0.38 << Ridge R²=0.87 ❌ 全谱 MLP 架构不适合

🟢 Phase 8 重大突破（2025-12-04）：
- MVP-PG1 ✅ 完成！ρ=1.00，Soft routing 保住 100% oracle 增益
- Gate 问题已解决！瓶颈转移到专家数量和表达能力

❌ Phase 7 取消（不再需要）：
- MVP-7.1~7.3 取消，物理窗 Gate 已解决 gating 问题
- 无需转向 Conditional 路线
```

## 4.2 核心结论快照

| MVP | 核心结论（一句话） | 关键数字 | 同步到 Hub |
|-----|------------------|---------|-----------|
| MVP-0 | baseline | R²=0.8616 | ✅ §5.3 |
| MVP-1.0 | ⚠️ 原 ΔR²=0.078 被高估 | - | ✅ §3.1 C1 |
| MVP-1.1 | MoE 效果稳健，[M/H] 主导 | ΔR²=+0.050, CI=[0.033,0.067] | ✅ §3.2 C1, C2 |
| MVP-2.0 | Noise-matched 有价值，但离散分档不稳定 | ΔR²=+0.080 avg | ✅ §3.2 C5 |
| MVP-3.0 | Quantile bins 失败 | K=2 唯一正增益 | ✅ §3.2 C3 |
| MVP-3.1 | Pseudo gating 几乎无用 | 仅 7.3% Oracle | ✅ §3.2 C3 |
| MVP-3.2 | 条件线性达 80% MoE | R²=0.9018 | ✅ §3.2 C4 |
| MVP-5.0 | Ca II triplet 重要，高[M/H]分散 | 1.65× | ✅ §3.2 C2 |
| **MVP-PG1** | **🟢 物理窗 Gate 超预期！Soft routing 保住 100% 增益** | **ρ=1.00, Acc=82%** | ✅ **§3.2 C6** |
| **MVP-NN1** | **⚠️ MoE NN 提升明显但 NN<<Ridge；全谱 MLP 不适合** | **ΔR²=+0.257, R²_NN=0.38** | ✅ **§3.2 C7** |
| **MVP-9E1** | **🟢 9专家扩展大成功！ρ=1.13超越Oracle，R²=0.9213突破0.90** | **ρ=1.13, R²=0.9213, Acc=94.6%** | ✅ **§3.2 C8** |
| **MVP-Next-A** | **🟢 回归Gate优于分类Gate，R²=0.9310** | **ΔR²=+0.0097** | ✅ **§3.2 C8** |
| **MVP-Next-C** | **❌ H-C 校准假设被否定，Metal-poor 误差非系统性 bias** | **ΔR²=-0.0013, Bin3 ΔR²=-0.0083** | ✅ **§3.2 C9** |

## 4.3 时间线

| 日期 | 事件 | 备注 |
|------|------|------|
| 2025-12-03 | 创建 MoE 实验框架 | Phase 0-2 启动 |
| 2025-12-03 | MVP-1.0 完成 | ⚠️ 原报告 ΔR²=0.078 被高估 |
| 2025-12-03 | MVP-2.0 完成 | ΔR²=+0.080 |
| 2025-12-03 | MVP-1.1 完成 | **修正 ΔR²=0.050**，[M/H] 68.7% |
| 2025-12-03 | 大数据集验证 | ΔR²=0.050 在 100k/10k 复现 |
| 2025-12-03 | MVP-3.0~3.2 完成 | Quantile ❌，Pseudo ❌，Cond ✅ |
| 2025-12-03 | MVP-5.0 完成 | Ca II 1.65×，系数分析 |
| 2025-12-03 | 决策点 D2 触发 | Pseudo <50%，转 Phase 7 |
| 2025-12-04 | 拆分 main → hub + roadmap | 架构重构 |
| 2025-12-04 | **立项 Phase 8: 物理窗 Gate** | MVP-PG1/PG2/PG3，验证 gate 可落地性 |
| **2025-12-04** | **🟢 MVP-PG1 完成！** | **ρ=1.00，Soft routing 超预期成功** |

---

# 5. 🔗 跨仓库集成

## 5.1 实验索引

| experiment_id | project | topic | 状态 | 对应 MVP |
|---------------|---------|-------|------|---------|
| `VIT-20251203-moe-piecewise-01` | VIT | moe | ✅ | MVP-1.0 |
| `VIT-20251203-moe-snr-02` | VIT | moe | ✅ | MVP-2.0 |
| `VIT-20251203-moe-rigorous-01` | VIT | moe | ✅ | MVP-1.1 |
| `VIT-20251203-moe-quantile-01` | VIT | moe | ✅ | MVP-3.0 |
| `VIT-20251203-moe-pseudo-01` | VIT | moe | ⏳ | MVP-3.1 |
| `VIT-20251203-moe-conditional-01` | VIT | moe | ✅ | MVP-3.2 |
| `VIT-20251203-moe-logg-gate-01` | VIT | moe | ⏳ | MVP-4.0 |
| `VIT-20251203-moe-coef-01` | VIT | moe | ✅ | MVP-5.0 |
| `VIT-20251203-moe-nn-03` | VIT | moe | ⏳ | MVP-6.1 |
| `VIT-20251204-moe-phys-gate-01` | VIT | moe | ✅ **完成** | MVP-PG1 |
| **`VIT-20251204-moe-9expert-01`** | VIT | moe | ✅ **完成！** | **MVP-9E1** |
| `VIT-20251204-moe-nn-expert-01` | VIT | moe | ✅ **完成** | MVP-NN1 |
| **`VIT-20251204-moe-regress-gate-01`** | VIT | moe | ✅ **R²=0.9310** | **MVP-Next-A** |
| **`VIT-20251204-moe-full-coverage-01`** | VIT | moe | ✅ **完成** | **MVP-Next-B** |
| **`VIT-20251204-moe-calibration-01`** | VIT | moe | ⏳ **立项** | **MVP-Next-C** |
| **`VIT-20251205-moe-100k-01`** | VIT | moe | ⏳ **立项中** | **MVP-12A** |
| **`VIT-20251205-moe-coverage-plus-01`** | VIT | moe | ⏳ **立项中** | **MVP-12B** |
| **`VIT-20251205-moe-feature-mining-01`** | VIT | moe | ⏳ **立项中** | **MVP-13** |
| **`VIT-20251205-moe-embedding-01`** | VIT | moe | ⏳ **立项中** | **MVP-14** |
| **`VIT-20251205-moe-lgbm-expert-01`** | VIT | moe | ⏳ **立项中** | **MVP-15** |

## 5.2 仓库关联

| 仓库 | 相关目录 | 说明 |
|------|---------|------|
| VIT | `~/VIT/results/moe/` | 训练结果 |
| 本仓库 | `logg/moe/` | 知识沉淀 |

---

# 6. 📎 附录

## 6.1 数值结果汇总表

### 主要指标对比 (noise=0.2, mask-aligned)

| MVP | 配置 | $R^2$ | ΔR² | 95% CI | 备注 |
|-----|------|-------|-----|--------|------|
| MVP-0 | 全局 Ridge | 0.8616 | - | - | baseline |
| MVP-1.0 | 分区 Ridge | **0.9116** | **+0.050** | [0.033, 0.067] | ⚠️ 原 +0.078 被高估 |
| MVP-1.1 消融 ([M/H] only) | 3 专家 | - | +0.034 | - | 贡献 68.7% |
| MVP-1.1 消融 (Teff only) | 3 专家 | - | +0.021 | - | 贡献 42.9% |
| MVP-2.0 | Noise-cond Expert (avg) | - | **+0.0797** | - | ✅ 有价值 |
| MVP-3.2 | Conditional Ridge 1st | **0.9018** | - | - | 达 MoE 80% |
| **MVP-PG1** | **Phys Gate (Soft)** | **0.8724** | - | - | 🟢 **ρ=1.00！** |

### 🟢 物理窗 Gate 结果 (MVP-PG1)

| 路由方式 | R² | ρ | 备注 |
|---------|-----|---|------|
| Oracle | 0.8725 | 1.000 | 上限 |
| **Soft** | **0.8724** | **0.997** | ✅ **推荐** |
| Soft+fallback | 0.8727 | 1.006 | 最佳但差异小 |
| Hard | 0.8619 | 0.722 | 损失 28% |
| Global Ridge | 0.8341 | 0.000 | baseline |

**消融实验**：
| 特征配置 | ρ | 说明 |
|---------|---|------|
| All (11 维) | 0.997 | 完整 |
| No Teff proxy | 0.887 | -11% |
| Only Ca II | 0.629 | 核心特征 |
| Only EW_CaT | 0.548 | 最小可用 |

### 大数据集验证 (100k train / 10k test)

| Noise | Global R² | MoE R² | ΔR² | 95% CI | Verdict |
|-------|-----------|--------|-----|--------|---------|
| **0.2** | 0.8745 | **0.9246** | **+0.0501** | **[0.0451, 0.0552]** | ✅ 稳健 |
| **0.5** | 0.7198 | **0.7719** | **+0.0521** | **[0.0421, 0.0611]** | ✅ 稳健 |

### 分 bin 统计 (noise=0.2)

| Bin | $T_{\text{eff}}$ 范围 | $[\text{M/H}]$ 范围 | Train | Test | 局部 $R^2$ | vs 全局 |
|-----|---------------------|-------------------|-------|------|-----------|--------|
| 1 | [3750, 4500) | [-2, -1) | 2,910 | 90 | 0.8971 | +0.063 |
| 2 | [3750, 4500) | [-1, 0.0] | 2,872 | 76 | 0.9569 | +0.123 |
| 3 | [3750, 4500) | (0.0, 0.5] | 2,064 | 59 | **0.9803** | **+0.146** |
| 4 | [4500, 5250) | [-2, -1) | 3,278 | 105 | **0.7726** | **-0.062** |
| 5 | [4500, 5250) | [-1, 0.0] | 3,224 | 83 | 0.9159 | +0.082 |
| 6 | [4500, 5250) | (0.0, 0.5] | 2,308 | 65 | 0.9789 | +0.145 |
| 7 | [5250, 6000] | [-2, -1) | 3,658 | 107 | **0.7869** | **-0.047** |
| 8 | [5250, 6000] | [-1, 0.0] | 3,769 | 129 | 0.9277 | +0.094 |
| 9 | [5250, 6000] | (0.0, 0.5] | 2,526 | 102 | 0.9691 | +0.135 |

### Noise-conditioned Expert 对比

| Test Noise | Expert R² | Mixed R² | ΔR² | Expert 最优 α |
|------------|-----------|----------|-----|--------------|
| 0.0 | 0.9991 | 0.8207 | **+0.178** | 0.001 |
| 0.1 | 0.9130 | 0.8168 | +0.096 | 1.0 |
| 0.2 | 0.8341 | 0.8001 | +0.034 | 10.0 |
| 0.5 | 0.6552 | 0.6863 | **-0.031** | 100.0 |
| 1.0 | 0.4601 | 0.3393 | **+0.121** | 100.0 |

---

## 6.2 相关文件索引

| 类型 | 文件路径 | 说明 |
|------|---------|------|
| Roadmap | `logg/moe/moe_roadmap_20251203.md` | 当前文件 |
| Hub | `logg/moe/moe_hub_20251203.md` | 智库导航 |
| MVP-1.0 报告 | `logg/moe/exp_moe_piecewise_ridge_20251203.md` | Piecewise Ridge |
| MVP-1.1 报告 | `logg/moe/exp_moe_rigorous_validation_20251203.md` | 严谨验证 |
| MVP-2.0 报告 | `logg/moe/exp_moe_noise_conditioned_20251203.md` | Noise-conditioned |
| MVP-3.0 报告 | `logg/moe/exp_moe_quantile_bins_sweep_20251203.md` | Quantile Bins |
| MVP-3.2 报告 | `logg/moe/exp_moe_conditional_ridge_20251203.md` | Conditional Ridge |
| MVP-5.0 报告 | `logg/moe/exp_moe_coefficient_analysis_20251203.md` | 系数分析 |
| **MVP-PG1 报告** | `logg/moe/exp_moe_phys_gate_baseline_20251204.md` | 🟢 物理窗 Gate |
| 图表目录 | `logg/moe/img/` | 实验图表 |

---

## 6.3 变更日志

| 日期 | 变更内容 | 影响 |
|------|---------|------|
| 2025-12-03 | 从 moe_main.md 拆分创建 Roadmap | 全部 |
| 2025-12-03 | Phase 1-5 完成状态同步 | §2, §4 |
| 2025-12-03 | 添加 Phase 7 详细设计 | §3 |
| 2025-12-04 | 架构重构：main → hub + roadmap | 全部 |
| **2025-12-04** | **🟢 MVP-PG1 完成！ρ=1.00** | §1.1, §2.1, §3, §4, §5.1 |
| 2025-12-04 | 更新 MVP-PG1 详细设计加入实际结果 | §3 |
| 2025-12-04 | Phase 7 取消（不再需要） | §4.1 |
| 2025-12-04 | 推荐下一步：Phase 6 NN-MoE 集成 | §1.2 |
| **2025-12-04** | **🔴 立项 Phase 9-10: MVP-9E1, MVP-NN1** | §1.1, §2.1, §2.2, §3, §4.1, §5.1 |
| 2025-12-04 | 添加 Phase 9 (9专家扩展) 详细设计 | §3 |
| 2025-12-04 | 添加 Phase 10 (NN Expert) 详细设计 | §3 |
| 2025-12-04 | 更新看板视图和实验索引 | §4.1, §5.1 |
| **2025-12-04** | **⚠️ MVP-NN1 完成！ΔR²=+0.257 但 NN<<Ridge** | §2.1, §4.1, §4.2, §5.1 |
| **2025-12-04** | **🟢 MVP-9E1 大成功！ρ=1.13, R²=0.9213, Acc=94.6%** | §1.1, §2.1, §4.1, §4.2, §5.1 |
| **2025-12-04** | **🔴 立项 Phase 11: MVP-Next-A/B/C** | §1.1, §2.1, §2.2, §3, §4.1, §5.1 |
| 2025-12-04 | 添加 Phase 11 详细设计：回归 gate + coverage + 校准 | §3 |
| 2025-12-04 | 更新看板视图和实验索引 | §4.1, §5.1 |
| 2025-12-04 | 设定 M2 里程碑目标 | §3 |
| **2025-12-04** | **❌ MVP-Next-C 完成！H-C 校准假设被否定** | §2.1, §4.1, §4.2, §5.1 |
| 2025-12-04 | 更新看板视图：MVP-Next-C 移至已取消，MVP-Next-A 移至已完成 | §4.1 |
| **2025-12-05** | **🔴 立项 Phase 12-13: MVP-12A/12B/13/14/15** | §1.1, §2.1, §2.2, §3, §5.1 |
| 2025-12-05 | 添加 Phase 12 (大规模验证) 详细设计 | §3 |
| 2025-12-05 | 添加 Phase 13 (特征增强 & 小模型) 详细设计 | §3 |
| 2025-12-05 | 更新实验索引：添加 MVP-12A~15 | §5.1 |
| **2025-12-07** | **Review & Fix：状态一致性修复** | §1.1, §1.2, §2.1, §3, §4.1 |
| 2025-12-07 | 更新依赖图为 Phase 11-13 重点 | §1.2 |
| 2025-12-07 | Phase 7 详细设计标记为已取消 | §3 |
| 2025-12-07 | 已完成 Phase 折叠到 `<details>` 区块 | §3 |
| 2025-12-07 | MVP-Next-B 移至看板已完成区域 | §4.1 |
| 2025-12-07 | 待创建实验报告链接标注为 (待创建) | §2.1 |
