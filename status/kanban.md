# 📌 实验看板（Experiment Kanban）

---
> **最后更新：** 2025-12-03  
> **活跃项目：** VIT / BlindSpot  
> **本周重点：** **🆕 MoE Phase 7: 连续条件化（Gate 噪声敏感性 + Conditional++ + Noise 连续化）** + Diffusion 1D 光谱降噪

---

# 📊 状态统计

| 状态 | 数量 | 说明 |
|------|------|------|
| 💡 Inbox | 11 | 待结构化的 idea |
| ⏳ TODO | 10 | 已分配 ID，待启动 |
| 🚀 Running | 0 | 正在运行 |
| ✅ Done | 0 | 完成待写 exp.md |
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

| experiment_id | MVP | project | topic | 优先级 | 预估时间 | session 来源 | 备注 |
|---------------|-----|---------|-------|--------|---------|-------------|------|
| **🆕 `VIT-20251203-moe-gate-noise-01`** | **MVP-7.1** | VIT | moe | **🔴🔴 P0** | ~2h | GPT 脑暴 2025-12-03 | **🆕 Gate 噪声敏感性曲线 → 决定"硬 MoE 还能不能救"** |
| **🆕 `VIT-20251203-moe-cond-pp-01`** | **MVP-7.2** | VIT | moe | **🔴 P0** | ~2h | GPT 脑暴 2025-12-03 | **🆕 Conditional Ridge++ → 榨出剩余 20% MoE 差距** |
| **🆕 `VIT-20251203-moe-noise-cont-01`** | **MVP-7.3** | VIT | moe | **🔴 P0** | ~2h | GPT 脑暴 2025-12-03 | **🆕 Noise 连续条件化 → 修复 noise=0.5 翻车** |
| `VIT-20251203-diff-baseline-01` | MVP-0.0 | VIT | diffusion | 🔴 P0 | ~3h | [session_diffusion_init](../logg/diffusion/sessions/session_20251203_diffusion_init.md) | 1D U-Net DDPM sanity check |
| `VIT-20251203-diff-supervised-01` | MVP-1.0 | VIT | diffusion | 🔴 P0 | ~4h | [session_diffusion_init](../logg/diffusion/sessions/session_20251203_diffusion_init.md) | spec-DDPM 复现（监督式降噪） |
| ~~`VIT-20251203-moe-piecewise-01`~~ | ~~MVP-1.0~~ | ~~VIT~~ | ~~moe~~ | ~~🔴 P0~~ | ~~2h~~ | - | ✅ **已完成** → Archived |
| `VIT-20251203-moe-pseudo-01` | MVP-3.1 | VIT | moe | 🟡 P1 | ~2h | MoE-1.1 后续规划 | Pseudo Gating：用 $\widehat{[M/H]}$ 做 gate |
| `VIT-20251203-moe-logg-gate-01` | MVP-4.0 | VIT | moe | 🟡 P1 | ~2h | MoE-1.1 后续规划 | log g Oracle/Pseudo Gate 三件套 |
| `VIT-20251201-gta-fusion-01` | MVP-Global-2 | VIT | gta | 🔴 P0 | ~3h | [session_gta_fusion](../logg/gta/sessions/session_20251201_gta_fusion.md) | 双塔融合 (Global + Local) |
| `BS-20251201-latent-gta-01` | MVP-Global-2 | BlindSpot | distill | 🔴 P0 | ~2h | - | Latent 特征提取给 GTA |
| `BS-20251201-distill-finetune-01` | MVP-2.3 | BlindSpot | distill | 🟡 P1 | ~4h | [session_distill](../logg/distill/sessions/session_20251130_distill_latent_probe.md) | Fine-tune encoder 测试 |

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

### 🆕 MoE Phase 7 执行顺序（决策树）

```
MVP-7.1 Gate 噪声敏感性 🔴
    │
    ├─ Conditional 明显优于 Hard MoE → MVP-7.2 Conditional++ 🔴
    │                                   │
    │                                   └─ MVP-7.3 Noise 连续化
    │
    └─ 两者都崩 → MVP-7.4 物理窗门控（探索 latent gate）
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

*最后更新: 2025-12-03*


### 🆕 新增完成 (2025-12-04)

| experiment_id | 完成时间 | 主指标 | raw log 路径 | exp.md 状态 | 下一步 |
|---------------|----------|--------|--------------|------------|--------|
| **`VIT-20251204-moe-9expert-01`** | **2025-12-04 01:35** | **🟢 ρ=1.13, R²=0.9213** | `results/moe/9expert_phys_gate/` | ✅ 已完成 | → Archived |

