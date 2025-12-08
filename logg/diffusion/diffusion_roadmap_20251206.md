# 🗺️ Diffusion 实验路线图（Roadmap）

---
> **主题名称：** Diffusion 1D 恒星光谱降噪与参数推断  
> **作者：** Viska Wei  
> **创建日期：** 2025-12-06  
> **最后更新：** 2025-12-06  
> **当前 Phase：** Phase 0 (Sanity Check) 已完成，准备进入 Phase 1

---

## 🔗 相关文件

| 类型 | 文件 | 说明 |
|------|------|------|
| 🧠 Hub | [`diffusion_hub_20251206.md`](./diffusion_hub_20251206.md) | 智库导航 |
| 📋 Kanban | [`kanban.md`](../../status/kanban.md) | 全局看板 |
| 📗 子实验 | `exp_diffusion_*.md` | 单实验详情 |

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

> **按阶段组织实验，每个 Phase 有明确目标**

## 1.1 Phase 列表

| Phase | 目的 | 包含 MVP | 状态 | 关键产出 |
|-------|------|---------|------|---------|
| **Phase 0: Sanity Check** | 验证 1D 光谱降噪可行性 | MVP-0.0, **MVP-0.5**, **MVP-0.6** | ✅ 完成 | 有限噪声 denoiser 有效 |
| **Phase 1: 降噪路线对比** | 监督式 vs DPS 后验采样 | MVP-1.0, MVP-1.1, MVP-1.2 | ⏳ 计划中 | 最佳降噪管线 |
| **Phase 2: 参数推断** | 降噪谱 → 参数 + 不确定性 | MVP-2.0, MVP-2.1 | ⏳ 计划中 | 端到端参数后验 |
| **Phase 3: 评价与校准** | 谱线级评价 + 覆盖率测试 | MVP-3.0, MVP-3.1 | ⏳ 计划中 | 科学可信度验证 |

## 1.2 依赖关系图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Diffusion 实验依赖图                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   [Phase 0: Sanity Check] ✅                                                │
│   ├── MVP-0.0: DDPM Baseline ❌                                             │
│   ├── MVP-0.5: Bounded Noise ✅ (59.5%)                                     │
│   └── MVP-0.6: wMAE Residual ✅ (46.5%)                                     │
│         │                                                                   │
│         ├──────────────────────┬───────────────────────────────┐            │
│         ▼                      ▼                               ▼            │
│   [MVP-1.0: 监督式 DDPM]  [MVP-1.1: DPS 后验采样]     [MVP-1.2: +ivar]      │
│   (spec-DDPM 复现)        (先验+likelihood)          (异方差条件化)          │
│   ⏸️ 暂停                  ⏳ 待执行                 ⏳ 待执行               │
│         │                      │                               │            │
│         └──────────────────────┼───────────────────────────────┘            │
│                                ▼                                            │
│                    [MVP-2.0: 采样谱 → 参数后验]                              │
│                    (样本传播不确定性) ⏳                                      │
│                                │                                            │
│                                ├───────────────┐                            │
│                                ▼               ▼                            │
│              [MVP-3.0: 谱线级评价]    [MVP-3.1: 覆盖率测试]                  │
│              (EW/线深/RV 偏置) ⏳      (PIT/可信区间校准) ⏳                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1.3 决策点

> **关键分叉点：根据实验结果决定下一步方向**

| 决策点 | 触发条件 | 选项 A | 选项 B |
|--------|---------|--------|--------|
| D1 | Phase 0 完成后 | ✅ 有限噪声有效 → 进入 Phase 1 | ❌ 若全失败 → 转向判别式 |
| D2 | MVP-1.1 完成后 | 如果 DPS 偏置↓ → 推荐 DPS 路线 | 如果偏置相近 → 用更简单的监督式 |
| D3 | MVP-2.0 完成后 | 如果覆盖率 60-75% → Phase 3 | 如果 <50% 或 >90% → 后校准 |

---

# 2. 📋 MVP 实验列表

> **所有 MVP 的一览表，便于快速查找和追踪**

## 2.1 实验总览

| MVP | 实验名称 | Phase | 状态 | experiment_id | 报告链接 |
|-----|---------|-------|------|---------------|---------|
| MVP-0.0 | 1D U-Net DDPM Baseline | 0 | ❌ 失败 | `SD-20251203-diff-baseline-01` | [报告](./exp_diffusion_baseline_20251203.md) |
| **MVP-0.5** | **有限噪声多级 Denoiser** | 0 | ✅ 完成 | `SD-20251204-diff-bounded-01` | [报告](./exp_diffusion_bounded_noise_denoiser_20251204.md) |
| **MVP-0.6** | **wMAE Residual Denoiser** | 0 | ✅ 完成 | `SD-20251204-diff-wmae-01` | [报告](./exp_diffusion_wmae_residual_denoiser_20251204.md) |
| MVP-1.0 | 监督式条件 DDPM | 1 | ⏸️ 暂停 | `SD-20251203-diff-supervised-01` | [报告](./exp_diffusion_supervised_20251203.md) |
| MVP-1.1 | DPS 后验采样 | 1 | ⏳ 计划中 | - | - |
| MVP-1.2 | +ivar 条件化 | 1 | ⏳ 计划中 | - | - |
| MVP-2.0 | 采样谱 → 参数后验 | 2 | ⏳ 计划中 | - | - |
| MVP-2.1 | 摊销后验 p(θ\|y) | 2 | ⏳ 低优先 | - | - |
| MVP-3.0 | 谱线级评价 | 3 | ⏳ 计划中 | - | - |
| MVP-3.1 | 覆盖率测试 | 3 | ⏳ 计划中 | - | - |

**状态图例**：
- ⏳ 计划中（Planned）
- 🔴 待执行（Ready）
- 🚀 运行中（Running）
- ✅ 已完成（Done）
- ❌ 已失败（Failed）
- ⏸️ 暂停（Paused）

## 2.2 配置速查表

> **所有 MVP 的关键配置对比**

| MVP | 数据规模 | 特征维度 | 模型 | 关键变量 | 验收标准 |
|-----|---------|---------|------|---------|---------|
| MVP-0.0 | 10K | 4096 | UNet1D (5.9M) | T=1000 | 能生成谱 ❌ |
| MVP-0.5 | 10K | 4096 | ConditionalUNet1D (6.3M) | λ∈[0.1,0.5] | MSE↓>30% ✅ |
| MVP-0.6 | 5K | 4096 | ConditionalResidualNet1D (7.5M) | s∈[0,0.2] | wMAE↓≥10% ✅ |
| MVP-1.0 | 10K pairs | 4096 | ConditionalUNet1D (6.3M) | SNR∈[5,50] | MSE < baseline |
| MVP-1.1 | 10K clean | 4096 | UNet1D + DPS | guidance scale | 偏置↓ |
| MVP-1.2 | +ivar | 4096 | 2-channel input | ivar channel | 低SNR改善 |
| MVP-2.0 | - | - | Ridge/MLP | N_samples=100 | 覆盖率~68% |
| MVP-3.0 | - | - | - | 谱线窗 | EW bias<5% |
| MVP-3.1 | - | - | - | PIT | 覆盖率校准 |

---

# 3. 🔧 MVP 详细设计

> **每个 MVP 的详细规格，便于快速执行**

## Phase 0: Sanity Check ✅ 完成

### MVP-0.0: 1D U-Net DDPM Baseline ❌ 失败

| 项目 | 配置 |
|------|------|
| **目标** | 验证 1D U-Net 能在恒星光谱上训练 DDPM |
| **数据** | BOSZ 50000 z0 高 SNR 谱 (~10,000) |
| **模型** | 1D U-Net (4 blocks, 64→256 channels), T=1000 steps |
| **训练** | Noise prediction (ε-prediction), 50 epochs |
| **验收标准** | 能生成视觉上像光谱的样本 |
| **结果** | ❌ **失败**：loss=0.0072 但采样生成高斯噪声 |
| **失败原因** | eps vs x0 预测类型不一致 |

---

### MVP-0.5: 有限噪声多级 Denoiser ✅ 通过

| 项目 | 配置 |
|------|------|
| **目标** | 验证 denoiser 能在已知噪声模型下降噪，不依赖全 diffusion |
| **噪声模型** | $y = x_0 + \lambda \cdot \sigma \odot \epsilon$ |
| **噪声范围** | $\lambda \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$ |
| **网络输入** | $[x_t, \sigma]$ (2 channel) |
| **训练目标** | x0-MSE (直接预测干净谱) |
| **验收标准** | λ=0.5 时，MSE 降低 >30% |
| **结果** | ✅ **通过**：59.5% MSE 降低 @ λ=0.5 |

**关键发现**：
- 高噪声（λ≥0.4）效果显著
- 低噪声（λ<0.3）会过度平滑

---

### MVP-0.6: wMAE Residual Denoiser ✅ 通过

| 项目 | 配置 |
|------|------|
| **目标** | 验证 residual 结构 + wMAE 损失在弱噪声区间可控 |
| **噪声模型** | $y = x_0 + s \cdot \sigma \odot \epsilon$ |
| **噪声范围** | $s \in \{0.0, 0.05, 0.1, 0.2\}$ |
| **核心公式** | $\hat{x}_0 = y + s \cdot g_\theta(y, s, \sigma)$ |
| **损失函数** | wMAE = $\frac{1}{N}\sum\frac{\|\hat{x}_0 - x_0\|}{\sigma}$ |
| **验收标准** | s=0 identity; s=0.2 wMAE↓≥10% |
| **结果** | ✅ **通过**：s=0 wMAE=0; s=0.2 wMAE↓46.5% |

**关键发现**：
- Residual 结构保证 s=0 时严格 identity
- 所有 s>0 都有改善（意外收获）
- wMAE 损失保护高 SNR 区域

---

## Phase 1: 降噪路线对比

### MVP-1.0: 监督式条件 DDPM ⏸️ 暂停

| 项目 | 配置 |
|------|------|
| **目标** | 复现 spec-DDPM 思路，训练 $p(x_{clean} \| y_{noisy})$ |
| **数据** | 成对低/高 SNR 谱 (人工加噪) |
| **模型** | 条件 1D U-Net，$y_{noisy}$ 作为 conditioning |
| **指标** | MSE, SSIM-1D, 后续参数偏置 |
| **验收标准** | MSE 优于 DnCNN/PCA baseline |
| **状态** | ⏸️ 暂停：依赖 baseline，待 MVP-0.5/0.6 稳定后再继续 |

---

### MVP-1.1: DPS 后验采样 ⏳ 计划中

| 项目 | 配置 |
|------|------|
| **目标** | 训练无条件先验 $p(x)$，推理时用 DPS 做后验 |
| **数据** | 只需高 SNR 谱训练先验 |
| **模型** | 无条件 1D U-Net + DPS guidance |
| **Forward Operator** | $A(x) = x$ (或含 LSF/mask) |
| **噪声模型** | 异方差高斯，$\sigma(\lambda)$ from ivar |
| **验收标准** | 偏置 < 监督式 DDPM |

**→ 对假设的影响**：若偏置显著降低，则 H2.1 成立

---

### MVP-1.2: +ivar 条件化 ⏳ 计划中

| 项目 | 配置 |
|------|------|
| **目标** | 测试将 per-pixel ivar 作为额外 conditioning 的效果 |
| **依赖** | MVP-1.1 |
| **数据** | 同上，额外提供 ivar channel |
| **模型** | 2-channel input (spectrum + ivar) |
| **验收标准** | 低 SNR 区域改善 |

---

## Phase 2: 参数推断

### MVP-2.0: 采样谱 → 参数后验 ⏳ 计划中

| 项目 | 配置 |
|------|------|
| **目标** | 从后验谱样本传播不确定性到参数 |
| **方法** | $x^{(s)} \sim p(x\|y)$ → 参数估计器 → $\theta^{(s)}$ |
| **参数估计器** | Ridge/MLP/The Payne |
| **输出** | $\theta$ 后验样本集合 |
| **验收标准** | 覆盖率接近名义值 (60-75%) |

---

### MVP-2.1: 摊销后验 p(θ|y) ⏳ 低优先级

| 项目 | 配置 |
|------|------|
| **目标** | 直接学习 $p(\theta\|y)$ 的条件 score |
| **优点** | 推理快，一次 forward 得参数分布 |
| **风险** | 标签噪声直接进后验，跨巡天 shift |
| **验收标准** | 推理速度 <1s，覆盖率可接受 |

---

## Phase 3: 评价与校准

### MVP-3.0: 谱线级评价 ⏳ 计划中

| 项目 | 配置 |
|------|------|
| **目标** | 评估关键谱线的物理量偏置 |
| **指标** | 等效宽度 (EW) 偏差、线深偏差、RV 偏差、continuum 偏差 |
| **谱线窗** | H$\alpha$, H$\beta$, Ca II triplet, Mg b |
| **验收标准** | EW bias < 5%, RV bias < 1 km/s |

---

### MVP-3.1: 覆盖率测试 ⏳ 计划中

| 项目 | 配置 |
|------|------|
| **目标** | 验证后验不确定性是否校准 |
| **方法** | PIT histogram, 68%/95% 覆盖率统计 |
| **验收标准** | 覆盖率偏差 < 10% (e.g., 68% CI 实际覆盖 60-75%) |

---

# 4. 📊 进度追踪

## 4.1 看板视图

```
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│   ⏳ 计划中   │  ⏸️ 暂停     │  🚀 运行中   │   ✅ 已完成   │   ❌ 已失败   │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ MVP-1.1      │ MVP-1.0      │              │ MVP-0.5      │ MVP-0.0      │
│ MVP-1.2      │              │              │ MVP-0.6      │              │
│ MVP-2.0      │              │              │              │              │
│ MVP-2.1      │              │              │              │              │
│ MVP-3.0      │              │              │              │              │
│ MVP-3.1      │              │              │              │              │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

## 4.2 核心结论快照

> **每个完成的 MVP 的一句话结论，同步到 Hub**

| MVP | 核心结论（一句话） | 关键数字 | 同步到 Hub |
|-----|------------------|---------|-----------|
| MVP-0.0 | ❌ loss 收敛但采样生成高斯噪声，eps/x0 不一致 | loss=0.0072 | ✅ §3.1 C1 |
| MVP-0.5 | ✅ 有限噪声 denoiser 在 λ=0.5 时 59.5% MSE↓ | MSE: 0.0846→0.0342 | ✅ §3.1 C2 |
| MVP-0.6 | ✅ Residual+wMAE 在 s=0.2 时 46.5% wMAE↓，s=0 严格 identity | wMAE: 0.1596→0.0854 | ✅ §3.1 C3, C4 |
| MVP-1.0 | ⏸️ 训练收敛(loss=0.0025)，暂停等待 baseline | loss=0.0025 | ✅ §4.4 |

## 4.3 时间线

| 日期 | 事件 | 备注 |
|------|------|------|
| 2025-12-03 | MVP-0.0 启动 | DDPM baseline |
| 2025-12-03 | MVP-0.0 完成 | ❌ 失败：采样是高斯噪声 |
| 2025-12-03 | MVP-1.0 启动 | 监督式 DDPM |
| 2025-12-03 | MVP-1.0 暂停 | 依赖 baseline |
| 2025-12-04 | MVP-0.5 立项 | 有限噪声方案 |
| 2025-12-04 | MVP-0.5 完成 | ✅ 通过：59.5% MSE↓ |
| 2025-12-04 | MVP-0.6 立项 | wMAE Residual |
| 2025-12-04 | MVP-0.6 完成 | ✅ 通过：46.5% wMAE↓ |
| 2025-12-06 | Phase 0 完成 | 进入 Phase 1 规划 |

---

# 5. 🔗 跨仓库集成

## 5.1 实验索引

> **链接到 experiments_index/index.csv**

| experiment_id | project | topic | 状态 | 对应 MVP |
|---------------|---------|-------|------|---------|
| `SD-20251203-diff-baseline-01` | SpecDiffusion | diffusion | ❌ | MVP-0.0 |
| `SD-20251203-diff-supervised-01` | SpecDiffusion | diffusion | ⏸️ | MVP-1.0 |
| `SD-20251204-diff-bounded-01` | SpecDiffusion | diffusion | ✅ | MVP-0.5 |
| `SD-20251204-diff-wmae-01` | SpecDiffusion | diffusion | ✅ | MVP-0.6 |

## 5.2 仓库关联

| 仓库 | 相关目录 | 说明 |
|------|---------|------|
| **SpecDiffusion** | `~/SpecDiffusion/` | ⚠️ Diffusion 专用代码仓库 |
| 本仓库 | `logg/diffusion/` | 知识沉淀 |

> ⚠️ **重要**：所有 diffusion 实验必须在 `~/SpecDiffusion` 执行，Experiment ID 使用 `SD-` 前缀

## 5.3 运行路径记录

> **记录实验的实际运行路径，便于复现**

| MVP | 仓库 | 运行路径 | 配置文件 | 输出路径 |
|-----|------|---------|---------|---------|
| MVP-0.0 | SpecDiffusion | `scripts/train_diffusion.py` | `configs/diffusion/baseline.yaml` | `lightning_logs/diffusion/baseline` |
| MVP-0.5 | SpecDiffusion | `scripts/train_bounded_denoiser.py` | inline | `lightning_logs/diffusion/bounded_noise` |
| MVP-0.6 | SpecDiffusion | `scripts/train_wmae_residual_denoiser.py` | inline | `lightning_logs/diffusion/wmae_residual` |
| MVP-1.0 | SpecDiffusion | `scripts/train_supervised.py` | `configs/supervised.yaml` | `lightning_logs/supervised` |

---

# 6. 📎 附录

## 6.1 数值结果汇总表

> **所有 MVP 的核心数值结果**

### 主要指标对比

| MVP | 配置 | 主要指标 | 值 | vs Baseline |
|-----|------|---------|-----|-------------|
| MVP-0.0 | T=1000 DDPM | Final Loss | 0.0072 | ❌ 采样失败 |
| MVP-0.5 | λ=0.5 | MSE | 0.0342 | **-59.5%** ✅ |
| MVP-0.5 | λ=0.4 | MSE | 0.0348 | -28.5% |
| MVP-0.6 | s=0.2 | wMAE | 0.0854 | **-46.5%** ✅ |
| MVP-0.6 | s=0.1 | wMAE | 0.0527 | -33.9% |
| MVP-0.6 | s=0.05 | wMAE | 0.0320 | -19.8% |
| MVP-0.6 | s=0 | wMAE | 0.0000 | identity ✅ |
| MVP-1.0 | SNR=5-50 | MSE | ~0.48 | ⏸️ 暂停 |

### MVP-0.5 噪声级别扫描

| λ | MSE(noisy) | MSE(denoised) | 提升 |
|---|------------|---------------|------|
| 0.1 | 0.0033 | 0.0303 | -828% ❌ |
| 0.2 | 0.0139 | 0.0343 | -147% ❌ |
| 0.3 | 0.0284 | 0.0313 | -10.5% |
| 0.4 | 0.0487 | 0.0348 | **28.5%** |
| 0.5 | 0.0846 | 0.0342 | **59.5%** ✅ |

### MVP-0.6 噪声级别扫描

| s | wMAE(noisy) | wMAE(denoised) | 提升 |
|---|-------------|----------------|------|
| 0.00 | 0.0000 | 0.0000 | 0.0% (identity) |
| 0.05 | 0.0399 | 0.0320 | **19.8%** |
| 0.10 | 0.0798 | 0.0527 | **33.9%** |
| 0.20 | 0.1596 | 0.0854 | **46.5%** ✅ |

---

## 6.2 相关文件索引

| 类型 | 文件路径 | 说明 |
|------|---------|------|
| Roadmap | `logg/diffusion/diffusion_roadmap_20251206.md` | 当前文件 |
| Hub | `logg/diffusion/diffusion_hub_20251206.md` | 智库导航 |
| MVP-0.0 报告 | `logg/diffusion/exp_diffusion_baseline_20251203.md` | DDPM Baseline (失败) |
| MVP-0.5 报告 | `logg/diffusion/exp_diffusion_bounded_noise_denoiser_20251204.md` | Bounded Noise (通过) |
| MVP-0.6 报告 | `logg/diffusion/exp_diffusion_wmae_residual_denoiser_20251204.md` | wMAE Residual (通过) |
| MVP-1.0 报告 | `logg/diffusion/exp_diffusion_supervised_20251203.md` | 监督式 DDPM (暂停) |
| 会话归档 | `logg/diffusion/sessions/session_20251203_diffusion_init.md` | 立项讨论 |
| 图表目录 | `logg/diffusion/img/` | 实验图表 |

---

## 6.3 变更日志

| 日期 | 变更内容 | 影响 |
|------|---------|------|
| 2025-12-06 | 从 diffusion_main.md 拆分创建 Roadmap | - |
| 2025-12-06 | 整合 MVP-0.0/0.5/0.6/1.0 执行记录 | §2, §3, §4 |
| 2025-12-06 | 建立跨仓库集成 | §5 |
| 2025-12-06 | 整理数值结果汇总表 | §6.1 |

---

> **Roadmap 使用说明**：
> 
> **Roadmap 的定位**：
> - ✅ **做**：MVP 规格、执行追踪、进度看板、跨仓库集成、数值结果
> - ❌ **不做**：假设管理（→ hub.md）、洞见汇合（→ hub.md）、战略导航（→ hub.md）
> 
> **更新时机**：
> - 规划新 MVP 时，更新 §2, §3
> - MVP 状态变更时，更新 §4
> - 实验完成后，记录核心结论到 §4.2，同步到 Hub
> 
> **与 hub.md 的分工**：
> - Hub = 「我们知道了什么？下一步该往哪走？」
> - Roadmap = 「我们计划跑哪些实验？进度如何？」

---

*最后更新: 2025-12-06*

