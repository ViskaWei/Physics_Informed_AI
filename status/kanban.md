# 📌 实验看板（Experiment Kanban）

---
> **最后更新：** 2025-12-22  
> **活跃项目：** VIT / BlindSpot  
> **本周重点：** **🔴 Scaling Law: 验证传统 ML 在 1M 数据+高噪声下的性能瓶颈** — 目标：证明 ML 存在天花板，NN 能突破

---

# 📊 状态统计

| 状态 | 数量 | 说明 |
|------|------|------|
| 💡 Inbox | 11 | 待结构化的 idea |
| ⏳ TODO | **22** | 已分配 ID，待启动 **(+7 logg_1m Phase 0-1)** |
| 🚀 Running | 0 | 正在运行 |
| ✅ Done | 1 | 完成待写 exp.md |
| 📚 Archived | 19 | 已归档 |

---

# 💡 Inbox / Idea

> 来自 `sessions/` 和日常灵感，尚未分配 experiment_id

| idea | 来源 | topic | 添加日期 | 备注 |
|------|------|-------|---------|------|
| **🆕 Diffusion MVP-1.1: DPS 后验采样** | session_diffusion_init | diffusion | 2025-12-03 | 依赖 MVP-0.0, MVP-1.0 |
| **🆕 Diffusion MVP-1.2: +ivar 条件化** | session_diffusion_init | diffusion | 2025-12-03 | 异方差噪声处理 |
| **🆕 Diffusion MVP-2.0: 采样谱 → 参数后验** | session_diffusion_init | diffusion | 2025-12-03 | 不确定性传播，等降噪完成 |
| **🆕 Diffusion MVP-3.0: 谱线级评价** | session_diffusion_init | diffusion | 2025-12-03 | EW/RV 偏置分析 |
| **🆕 Diffusion MVP-3.1: 覆盖率测试** | session_diffusion_init | diffusion | 2025-12-03 | PIT/CI 校准 |
| MoE-2: 按 SNR/noise level 分专家 | moe_main | moe | 2025-12-03 | 依赖 MoE-1 结果 |
| MoE-3: NN-MoE 架构（仅当 MoE-1/2 有收益） | moe_main | moe | 2025-12-03 | K=2~4 专家 + gating 网络 |
| 用 Swin attention 替换 CNN 看 noise 鲁棒性 | 灵感 | swin | 2025-12-01 | 基于 CNN vs Transformer 结论 |
| 测试 multi-scale dilation 架构 | exp_cnn_dilated | cnn | 2025-12-01 | dilation=2 最优，尝试组合 |
| 加入 BlindSpot Latent 特征到 Global Tower | gta_main | gta | 2025-12-01 | 提升 noise=1.0 性能 |
| Error 捷径问题分析（Stage A） | distill_main | distill | 2025-12-01 | Error 单独 $R^2$≈0.91，需分离 |

---

# ⏳ TODO - 待跑实验

> 已分配 experiment_id，等待启动

### 🆕🆕 Phase NN: NN Baseline 实验系列（2025-12-24 大立项）

> **📍 智库导航**: [`logg/scaling/scaling_hub_20251222.md`](../logg/scaling/scaling_hub_20251222.md) §2.3 H-NN  
> **🗺️ 实验追踪**: [`logg/scaling/scaling_roadmap_20251222.md`](../logg/scaling/scaling_roadmap_20251222.md) Phase NN  
> **目标**: 快速判断 NN 能否接近/超过 Oracle MoE (0.62)，如果不能，是结构不对还是输入/训练不对

| experiment_id | MVP | project | topic | 优先级 | 预估时间 | 备注 |
|---------------|-----|---------|-------|--------|---------|------|
| **🆕 `SCALING-20251224-nn-baseline-framework-01`** | **MVP-NN-0** | VIT | **scaling** | **🔴🔴 P0** | ~半天 | **🚀 可靠基线框架：验证输入/评估没问题** |
| **🆕 `SCALING-20251224-mlp-baseline-01`** | **MVP-MLP-1** | VIT | **scaling** | **🔴🔴 P0** | ~1天 | **MLP 100k+1M 止损判断：归纳偏置对不对** |
| **🆕 `SCALING-20251224-cnn-baseline-01`** | **MVP-CNN-1** | VIT | **scaling** | **🔴 P0** | ~1.5天 | **CNN 100k+1M：验证局部结构带来质变** |
| `SCALING-20251224-cnn-multiscale-01` | MVP-CNN-2 | VIT | scaling | 🟡 P1 | ~1天 | 多尺度 CNN（仅当 CNN-1 < 0.60） |
| `SCALING-20251224-nn-compare-01` | MVP-Compare | VIT | scaling | 🔴 P0 | ~2h | 三件套同评估：Ridge/LGB/CNN/Oracle |
| `SCALING-20251224-moe-cnn-oracle-01` | MVP-MoE-CNN-0 | VIT | scaling | 🟢 P2 | 视情况 | MoE-CNN（仅当 global CNN < 0.60 明显） |

**执行顺序**:
1. MVP-NN-0 (框架搭建) → 2. MVP-MLP-1 @100k+1M (止损判断) → 3. MVP-CNN-1 @100k → 4. MVP-CNN-1 @1M → 5. MVP-Compare
6. 仅当 global CNN < 0.60: MVP-CNN-2 或 MVP-MoE-CNN-0

---

### 其他 TODO

| experiment_id | MVP | project | topic | 优先级 | 预估时间 | session 来源 | 备注 |
|---------------|-----|---------|-------|--------|---------|-------------|------|
|| ~~**🆕 `SCALING-20251222-ridge-1m-01`**~~ | ~~**MVP-1.0**~~ | ~~VIT~~ | ~~**scaling**~~ | ~~**🔴🔴 P0**~~ | ~~~4h~~ | ~~立项 2025-12-22~~ | ✅ Done |
|| ~~**🆕 `SCALING-20251222-lgbm-1m-01`**~~ | ~~**MVP-1.1**~~ | ~~VIT~~ | ~~**scaling**~~ | ~~**🔴🔴 P0**~~ | ~~~6h~~ | ~~立项 2025-12-22~~ | ✅ Done |
|| ~~**🆕 `SCALING-20251222-mlp-1m-01`**~~ | ~~**MVP-2.0**~~ | ~~VIT~~ | ~~**scaling**~~ | ~~**🔴 P0**~~ | ~~~8h~~ | ~~立项 2025-12-22~~ | → 替换为 MVP-MLP-1 |
| **🆕 `SD-20251204-diff-wmae-01`** | **MVP-0.6** | SpecDiffusion | diffusion | **🔴 P0** | ~3h | MVP-0.5 后续 | **wMAE + residual 结构，s≤0.2 弱噪声降噪** |
| **🆕 `VIT-20251203-moe-gate-noise-01`** | **MVP-7.1** | VIT | moe | **🔴🔴 P0** | ~2h | GPT 脑暴 2025-12-03 | **🆕 Gate 噪声敏感性曲线 → 决定"硬 MoE 还能不能救"** |
| **🆕 `VIT-20251203-moe-cond-pp-01`** | **MVP-7.2** | VIT | moe | **🔴 P0** | ~2h | GPT 脑暴 2025-12-03 | **🆕 Conditional Ridge++ → 榨出剩余 20% MoE 差距** |
| **🆕 `VIT-20251203-moe-noise-cont-01`** | **MVP-7.3** | VIT | moe | **🔴 P0** | ~2h | GPT 脑暴 2025-12-03 | **🆕 Noise 连续条件化 → 修复 noise=0.5 翻车** |
| ~~`VIT-20251203-diff-baseline-01`~~ | ~~MVP-0.0~~ | ~~VIT~~ | ~~diffusion~~ | ~~🔴 P0~~ | ~~3h~~ | - | ❌ 失败 → Archived |
| ~~`VIT-20251203-diff-supervised-01`~~ | ~~MVP-1.0~~ | ~~VIT~~ | ~~diffusion~~ | ~~🔴 P0~~ | ~~4h~~ | - | ⚠️ 待验证 |
| ~~`VIT-20251203-moe-piecewise-01`~~ | ~~MVP-1.0~~ | ~~VIT~~ | ~~moe~~ | ~~🔴 P0~~ | ~~2h~~ | - | ✅ **已完成** → Archived |
| `VIT-20251203-moe-pseudo-01` | MVP-3.1 | VIT | moe | 🟡 P1 | ~2h | MoE-1.1 后续规划 | Pseudo Gating：用 $\widehat{[M/H]}$ 做 gate |
| `VIT-20251203-moe-logg-gate-01` | MVP-4.0 | VIT | moe | 🟡 P1 | ~2h | MoE-1.1 后续规划 | log g Oracle/Pseudo Gate 三件套 |
| `VIT-20251201-gta-fusion-01` | MVP-Global-2 | VIT | gta | 🔴 P0 | ~3h | [session_gta_fusion](../logg/gta/sessions/session_20251201_gta_fusion.md) | 双塔融合 (Global + Local) |
| `BS-20251201-latent-gta-01` | MVP-Global-2 | BlindSpot | distill | 🔴 P0 | ~2h | - | Latent 特征提取给 GTA |
| `BS-20251201-distill-finetune-01` | MVP-2.3 | BlindSpot | distill | 🟡 P1 | ~4h | [session_distill](../logg/distill/sessions/session_20251130_distill_latent_probe.md) | Fine-tune encoder 测试 |
| **🆕 `VIT-20251205-moe-100k-01`** | **MVP-12A** | VIT | moe | **🔴🔴 P0** | ~4h | Phase 12 | **100k 规模复刻 Next-A → 稳态结论** |
| **🆕 `VIT-20251205-moe-coverage-plus-01`** | **MVP-12B** | VIT | moe | **🔴🔴 P0** | ~3h | Phase 12 | **Coverage++ → full-test > 0.91** |
| **🆕 `VIT-20251205-moe-feature-mining-01`** | **MVP-13** | VIT | moe | **🟡 P1** | ~3h | Phase 13 | **Feature mining Bin3/Bin6 → ΔR² ≥ 0.02** |
| **🆕 `VIT-20251205-moe-embedding-01`** | **MVP-14** | VIT | moe | **🟡 P1** | ~4h | Phase 13 | **1M embedding for gate** |
| **🆕 `VIT-20251205-moe-lgbm-expert-01`** | **MVP-15** | VIT | moe | **🟡 P1** | ~3h | Phase 13 | **小 LGBM 替换 Bin3/Bin6 expert** |

### 🆕🆕 logg 1M Breakthrough 实验系列（2025-12-22 立项）

> **📍 智库导航**: [`logg/logg_1m/logg_1m_hub_20251222.md`](../logg/logg_1m/logg_1m_hub_20251222.md)  
> **🗺️ 实验追踪**: [`logg/logg_1m/logg_1m_roadmap_20251222.md`](../logg/logg_1m/logg_1m_roadmap_20251222.md)  
> **目标**: 在 low-noise 条件下突破 log g 预测精度，验证信息瓶颈假设

| experiment_id | MVP | project | topic | 优先级 | 预估时间 | 备注 |
|---------------|-----|---------|-------|--------|---------|------|
| **🆕 `VIT-20251222-logg_1m-baseline-scaling-01`** | **MVP-0.B** | VIT | **logg_1m** | **🔴🔴 P0** | ~4h | **🚀 Ridge+LightGBM @ noise=1.0, 10k→1M scaling** |
| `VIT-20251222-logg_1m-foundation` | MVP-0.A | VIT | logg_1m | 🔴 P0 | ~2h | Low-noise 定义 |
| **🆕 `VIT-20251222-logg_1m-fisher`** | **MVP-1.1** | VIT | **logg_1m** | **🔴🔴 P0** | ~3h | **Fisher 理论上限分析 → 决定是否继续优化模型** |
| **🆕 `VIT-20251222-logg_1m-error_input`** | **MVP-1.2** | VIT | **logg_1m** | **🔴 P0** | ~4h | **SNR/Error 作为输入 → 让模型知道哪些像素可信** |
| **🆕 `VIT-20251222-logg_1m-normalization`** | **MVP-1.3** | VIT | **logg_1m** | **🔴 P0** | ~4h | **归一化三连对照 → median vs chunk-zscore vs continuum** |
| **🆕 `VIT-20251222-logg_1m-window`** | **MVP-1.4** | VIT | **logg_1m** | **🔴 P0** | ~4h | **敏感窗口 vs 全谱 → 验证干扰假设** |
| **🆕 `VIT-20251222-logg_1m-multitask`** | **MVP-1.5** | VIT | **logg_1m** | **🔴 P0** | ~4h | **多任务联合 Teff+FeH+logg → 解耦因素** |
| `VIT-20251222-logg_1m-msm` | MVP-2.1 | VIT | logg_1m | 🟡 P1 | ~6h | MSM 预训练 → 自监督突破（待 Phase 1 完成） |

---

# 🚀 Running - 已启动未归档

> 实验正在运行或已完成但未写报告

| experiment_id | 运行路径 | 开始时间 | 预期结束 | 状态 | 备注 |
|---------------|----------|----------|----------|------|------|
| - | - | - | - | - | 当前无运行中实验 |

---

# ✅ Done - 已完成待写 exp.md

> 实验完成，等待写 exp.md 报告

| experiment_id | 完成时间 | 主指标 | raw log 路径 | exp.md 状态 | 下一步 |
|---------------|----------|--------|--------------|------------|--------|
| **`VIT-20251205-lightgbm-100k-noise-01`** | **2025-12-05 19:37** | **🟢 R²↑1.85%~8.05%** | `results/lightgbm_100k/` | ✅ 已完成 | → Archived |
| `VIT-20251203-moe-conditional-01` | 2025-12-03 18:09 | **R²=0.9018 (1st order)** | `results/moe_conditional_ridge/` | ✅ 已完成 | → Archived |
| **`VIT-20251204-moe-phys-gate-01`** | **2025-12-04 23:15** | **🟢 ρ=1.00 (Soft)** | `results/moe/phys_gate_baseline/` | ✅ 已完成 | → Archived |
| **`VIT-20251204-moe-nn-expert-01`** | **2025-12-04 01:32** | **ΔR²=+0.257, NN<Ridge** | `results/moe/nn_experts/` | ✅ 已完成 | → Archived |
| **`VIT-20251204-moe-regress-gate-01`** | **2025-12-04 13:27** | **🟢 R²=0.9310 (+0.0097)** | `results/moe/regression_gate/` | ✅ 已完成 | → Archived |
| **`VIT-20251204-moe-full-coverage-01`** | **2025-12-04 13:41** | **🟢 R²_full=0.8957 (Edge-Clamp)** | `results/moe/full_coverage/` | ✅ 已完成 | → Archived |

---

# 📚 Archived - 已写 exp + card

> 实验已完全归档，有完整文档

### MoE 实验系列（Phase 1-5 已完成 ✅）

| experiment_id | 完成日期 | topic | 主指标 | exp.md | 同步到 main |
|---------------|---------|-------|--------|--------|-------------|
| `VIT-20251203-moe-quantile-01` | 2025-12-03 | moe | ❌ ΔR²=+0.004 (负面) | [✅ exp_moe_quantile_bins_sweep](../logg/moe/exp_moe_quantile_bins_sweep_20251203.md) | ✅ |
| `VIT-20251203-moe-conditional-01` | 2025-12-03 | moe | ✅ R²=0.9018 (80% MoE) | [✅ exp_moe_conditional_ridge](../logg/moe/exp_moe_conditional_ridge_20251203.md) | ✅ |
| `VIT-20251203-moe-rigorous-01` | 2025-12-03 | moe | ✅ ΔR²=0.050 | [✅ exp_moe_rigorous_validation](../logg/moe/exp_moe_rigorous_validation_20251203.md) | ✅ |
| `VIT-20251203-moe-piecewise-01` | 2025-12-03 | moe | ✅ ΔR²=0.050 | [✅ exp_moe_piecewise_ridge](../logg/moe/exp_moe_piecewise_ridge_20251203.md) | ✅ |
| `VIT-20251203-moe-snr-02` | 2025-12-03 | moe | ✅ ΔR²=0.080 | [✅ exp_moe_noise_conditioned](../logg/moe/exp_moe_noise_conditioned_20251203.md) | ✅ |
| `VIT-20251203-moe-coef-01` | 2025-12-03 | moe | ✅ Ca II 1.65× | [✅ exp_moe_coefficient_analysis](../logg/moe/exp_moe_coefficient_analysis_20251203.md) | ✅ |

**MoE 已验证 Insights (I1-I7)**：见 `logg/moe/moe_main_20251203.md` §1.4.1

### 其他实验
| `VIT-20251201-gta-local-01` | 2025-12-01 | gta | $R^2$=0.9313 | [✅ exp_topk_window_cnn](../logg/gta/exp_topk_window_cnn_transformer_20251201.md) | ✅ |
| `VIT-20251201-gta-global-01` | 2025-12-01 | gta | $R^2$=0.9588 | [✅ exp_global_feature_tower](../logg/gta/exp_global_feature_tower_mlp_20251201.md) | ✅ |
| `VIT-20251130-gta-baseline-01` | 2025-11-30 | gta | $R^2$≈0 | [✅ exp_gta_f0f1_metadata](../logg/gta/exp_gta_f0f1_metadata_baseline_20251130.md) | ✅ |
| `VIT-20251201-cnn-dilated-01` | 2025-12-01 | cnn | $R^2$=0.992 | [✅ exp_cnn_dilated_kernel](../logg/cnn/exp_cnn_dilated_kernel_sweep_20251201.md) | ✅ |
| `BS-20251201-distill-latent-01` | 2025-12-01 | distill | $R^2$=0.5516 | [✅ exp_latent_extraction](../logg/distill/exp_latent_extraction_logg_20251201.md) | ✅ |
| `BS-20251130-distill-probe-01` | 2025-11-30 | distill | $R^2$=0.28 | [✅ exp_linear_probe](../logg/distill/exp_linear_probe_latent_20251130.md) | ✅ |
| `BS-20251201-encoder-logg-01` | 2025-12-01 | distill | $R^2$=0.6117 | [✅ exp_encoder_nn](../logg/distill/exp_encoder_nn_logg_20251201.md) | ✅ |
| `VIT-20251130-train-val-01` | 2025-11-30 | train | - | [✅ exp_val_size_sweep](../logg/train/exp_val_size_sweep_20251130.md) | ✅ |
| `VIT-20251129-lightgbm-01` | 2025-11-29 | lightgbm | $R^2$=0.536 | [✅ exp_lightgbm_hyperparam](../logg/lightgbm/exp_lightgbm_hyperparam_sweep_20251129.md) | ✅ |

---

# 🔄 本周回顾 (2025-12-01)

## 完成的重要实验

| 实验 | 核心结论 | 影响 |
|------|---------|------|
| GTA Global Tower | 126维特征 $R^2$=0.9588 @ noise=0.1 | 证明全局特征高效 |
| GTA Local Tower | TopK CNN K=256 $R^2$=0.9313 | CNN >> Transformer |
| CNN Dilated | dilation=2 最优 | 感受野匹配吸收线 |
| **Distill Latent 提取** | $R^2$: 0.22→0.55 (+150%) | `seg_mean_K8` 保留空间信息 |
| **Distill Encoder+MLP** | $R^2$=0.6117 (+10.9% vs Ridge) | MLP 捕捉非线性关系 |

## 下一步方向

| 方向 | 优先级 | 对应 TODO |
|------|--------|----------|
| **🆕🆕 MoE Gate 噪声敏感性** | **🔴🔴 P0** | **VIT-20251203-moe-gate-noise-01** → 决定技术路线！ |
| **🆕 MoE Conditional Ridge++** | 🔴 P0 | VIT-20251203-moe-cond-pp-01 |
| **🆕 MoE Noise 连续条件化** | 🔴 P0 | VIT-20251203-moe-noise-cont-01 |
| Diffusion Baseline | 🔴 P0 | VIT-20251203-diff-baseline-01 |
| Diffusion 监督式降噪 | 🔴 P0 | VIT-20251203-diff-supervised-01 |
| 双塔融合 | 🔴 P0 | VIT-20251201-gta-fusion-01 |
| Latent 增强 | 🔴 P0 | BS-20251201-latent-gta-01 |
| MoE Pseudo Gating | 🟡 P1 | VIT-20251203-moe-pseudo-01 |
| MoE log g Gate 分析 | 🟡 P1 | VIT-20251203-moe-logg-gate-01 |
| Distill Fine-tune | 🟡 P1 | BS-20251201-distill-finetune-01 |
| Diffusion DPS 后验采样 | 🟡 P1 | (Inbox, 依赖 MVP-0.0, MVP-1.0) |

### ~~MoE Phase 7 执行顺序~~ (已取消，Gate 已解决)

~~Phase 7 已不需要，MVP-PG1 物理窗 Gate 已解决门控问题~~

### 🔴🔴 MoE Phase 12-13 执行顺序（2025-12-05 立项）

**总策略**：先把 0.9310 变成 100k 稳态结论 → 再拉 full > 0.91 → 所有创新只针对 Bin3/Bin6 做增量

```
🔴 P0 (先做，最稳、最能对齐 LGBM=0.91):
    │
    ├── MVP-12A: 100k 规模复刻 Next-A
    │   └── 验收: covered R² ≥ 0.93, CI_low > 0, MoE > LGBM
    │
    └── MVP-12B: Coverage++ (第 10 个 oor expert)
        └── 验收: full R² ≥ max(LGBM, global+0.05)

🟡 P1 (之后做，特征/容量/分布改进):
    │
    ├── MVP-13: Feature mining Bin3/Bin6
    │   └── 验收: Bin3 或 Bin6 ΔR² ≥ +0.02，否则止损
    │
    ├── MVP-14: 1M embedding for gate (只喂 gate，不动专家)
    │   └── 验收: R² +0.003 或 Bin3/Bin6 改善
    │
    └── MVP-15: 小 LGBM 替换 Bin3/Bin6 expert (stacking-safe OOF)
        └── 验收: full R² > 0.91, Bin3/Bin6 不拖后腿
```

---

# 📎 快捷命令

| 命令 | 作用 |
|------|------|
| `?` / `status` | 查看整体进度 |
| `a` / `归档` | 进入归档流程 |
| `n [描述]` | 新建实验计划 |
| `sync` | 同步实验索引 |
| `kb` / `kanban` | 查看/更新看板 |

---

*最后更新: 2025-12-05*


### 🆕 新增完成 (2025-12-04)

| experiment_id | 完成时间 | 主指标 | raw log 路径 | exp.md 状态 | 下一步 |
|---------------|----------|--------|--------------|------------|--------|
| **`VIT-20251205-lightgbm-100k-noise-01`** | **2025-12-05 19:37** | **🟢 R²↑1.85%~8.05%** | `results/lightgbm_100k/` | ✅ 已完成 | → Archived |
|---------------|----------|--------|--------------|------------|--------|
| **`VIT-20251204-moe-9expert-01`** | **2025-12-04 01:35** | **🟢 ρ=1.13, R²=0.9213** | `results/moe/9expert_phys_gate/` | ✅ 已完成 | → Archived |

- [x] VIT-20251204-moe-calibration-01: Expert 校准 [H-C ❌ 偏差非主因]

### 🔄 进行中 (2025-12-04 ~ 12-05)

| experiment_id | 立项时间 | 主题 | 配置数 | exp.md 状态 |
|---------------|----------|------|--------|-------------|
| ~~`VIT-20251204-lightgbm-noise-sweep-01`~~ | 2025-12-04 | ~~LightGBM Noise Sweep (lr 主轴)~~ | ~~72~~ | ✅ 完成 |
| **`VIT-20251205-lightgbm-100k-noise-01`** | **2025-12-05** | **LightGBM 100k Noise Sweep (n=500)** | **12** | 🔄 [立项中](../logg/lightgbm/exp_lightgbm_100k_noise_sweep_20251205.md) |


### 🆕 新增完成 (2025-12-05)

| experiment_id | 完成时间 | 主指标 | raw log 路径 | exp.md 状态 | 下一步 |
|---------------|----------|--------|--------------|------------|--------|
| **`BM-20251205-ridge-100k`** | **2025-12-05 20:12** | **🟢 H2.2 成立 (+2.71%平均增益)** | `results/benchmark_ridge_100k/` | ✅ 已完成 | → Archived |


### VIT-20251205-moe-100k-01 ✅ (2025-12-07)
- MoE R² = 0.9400 (目标 ≥0.93)
- ΔR² CI = [0.0045, 0.0175] (显著 > 0)
- 100k 规模验证通过


- [x] VIT-20251207-lgb-100k-tree-01: 100k tree 上限确认，best_iter中位数=2179，推荐n=2500，100k全面反超32k ✅

## ✅ Done (2025-12-22)

- [x] **SCALING-20251222-ml-ceiling-01**: Traditional ML Ceiling @ 1M
  - Ridge R²=0.50, LightGBM R²=0.57 @ noise=1
  - 确认传统 ML 性能天花板，100k→1M 增益 <3%
  - 详见: `logg/scaling/exp/exp_scaling_ml_ceiling_20251222.md`

## ✅ Done (2025-12-22 继续)

- [x] **SCALING-20251222-ridge-alpha-01**: Ridge Alpha Extended Sweep
  - 100k: 最优 α=3.16e+04, R²=0.4856 (+2.55% vs baseline)
  - 1M: 最优 α=1.00e+05, R²=0.5017 (+0.42% vs baseline)
  - ✅ H1.5.1 验证：观察到倒 U 型曲线，峰值后明显下降
  - 详见: `logg/scaling/exp/exp_scaling_ridge_alpha_extended_20251222.md`

- [x] **SCALING-20251222-whitening-01**: Whitening/SNR Input Experiment
  - H1.7.1 ❌ REJECTED: SNR vs standardized ΔR² = +0.0146 (Ridge), -0.19 (LightGBM)
  - ⚠️ 重要发现: LightGBM 必须用 raw 输入，StandardScaler 严重损害性能 (R² 0.55→0.20)
  - 详见: `logg/scaling/exp/exp_scaling_whitening_snr_20251222.md`

## ✅ Done (2025-12-23)

- [x] **SCALING-20251223-fisher-ceiling-01**: Fisher/CRLB Theoretical Upper Bound
  - R²_max (median) = **0.9661** (理论上限极高)
  - Schur decay = **0.2366** (degeneracy 极强，仅保留 24% 信息)
  - ✅ H-16T.1 验证：R²_max = 0.966 ≥ 0.75 → 存在巨大 headroom
  - ✅ H-16T.2 验证：Schur decay = 0.24 < 0.9 → degeneracy 显著
  - Gap vs Ridge (0.50): **+0.47** | Gap vs LightGBM (0.57): **+0.40**
  - 详见: \`logg/scaling/exp/exp_scaling_fisher_ceiling_20251223.md\`

## ❌ Failed (2025-12-23)

- [x] **SCALING-20251223-fisher-ceiling-01**: Fisher/CRLB Theoretical Upper Bound
  - ❌ **实验失败**：偏导数估计方法存在根本性缺陷
  - **根因**：BOSZ 数据为连续采样（~40k 唯一参数值），不是规则网格
  - 邻近点差分法无法正确估计 ∂μ/∂θ，导致 Fisher 矩阵计算不可靠
  - R²_max = 0.97 的结果**不可信**
  - 下一步：等待方法论改进（数值微分/局部回归）
  - 详见: \`logg/scaling/exp/exp_scaling_fisher_ceiling_20251223.md\`

## ✅ Done (2025-12-24)

- [x] **SCALING-20251224-fisher-ceiling-02**: Fisher/CRLB V2 (Grid-based) ✅
  - V2 成功修复 V1 的数值问题，结果可信
  - R²_max (median) = **0.8914** (理论上限高)
  - Schur decay = **0.6906** (degeneracy 显著但非极端)
  - CRLB 范围仅 **2.9** 数量级 (V1 是 20！)
  - Gap vs Ridge: +0.43 | Gap vs LightGBM: +0.32
  - ✅ H-16T.1 (V2) 验证通过 | ✅ H-16T.2 (V2) 验证通过
  - 详见: \`logg/scaling/exp/exp_scaling_fisher_ceiling_v2_20251224.md\`

- [x] SCALING-20251224-nn-baseline-framework-01: MLP=0.47, CNN=0.43, vs Oracle gap=0.15~0.19
