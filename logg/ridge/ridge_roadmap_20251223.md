# 🗺️ Ridge Experiment Roadmap

> **Topic:** Ridge Regression for log_g Prediction  
> **Author:** Viska Wei  
> **Created:** 2025-11-27 | **Updated:** 2025-12-23  
> **Current Phase:** Phase 1 Complete → Phase 2 (Gate Verification)

<!-- 
📝 Language Convention:
- Headers & section titles: English (keep as-is)
- Content (objectives, conclusions, notes): Chinese OK
- Table column headers: English (keep as-is)
- Table cell content: Chinese OK
-->

## 🔗 Related Files

| Type | File | Description |
|------|------|-------------|
| 🧠 Hub | [`ridge_hub_20251223.md`](./ridge_hub_20251223.md) | Knowledge & strategy |
| 📋 Kanban | [`kanban.md`](../../status/kanban.md) | Global task board |
| 📗 Experiments | `exp/*.md` | Detailed reports |
| 📄 Main (Legacy) | [`ridge_main_20251130.md`](./ridge_main_20251130.md) | Original summary |

## 📑 Contents

- [1. 🚦 Decision Gates](#1--decision-gates)
- [2. 📋 MVP List](#2--mvp-list)
- [3. 🔧 MVP Specifications](#3--mvp-specifications)
- [4. 📊 Progress Tracking](#4--progress-tracking)
- [5. 🔗 Cross-Repo Integration](#5--cross-repo-integration)
- [6. 📎 Appendix](#6--appendix)

---

# 1. 🚦 Decision Gates

> **Hub 推荐战略方向，Roadmap 定义怎么验证**
>
> ⚠️ **职责边界**: 只做验证计划，战略理由见 [Hub §2](./ridge_hub_20251223.md#2--answer-key--strategic-route)

## 1.1 Current Strategic Route (from Hub)

> **来自 Hub §2 的战略推荐**

| Route | 路线名称 | Hub 推荐 | 需要验证 |
|-------|---------|---------|---------|
| Route I | Information Ceiling | 🟡 待验证 | Gate-1 |
| **Route M** | Representation / Model | 🟢 **推荐** | Gate-2 |
| Route S | Sigma Channel | 🟡 高风险 | Gate-3 |

> 📖 **战略推荐理由**见 [Hub §2 Answer Key](./ridge_hub_20251223.md#21-answer-key-to-question-tree)

---

## 1.2 Gate Definitions

### Gate-1: 信息论上限门 (Fisher Sanity Check)

| Item | Content |
|------|---------|
| **验证什么** | 信息论上限到底是多少？Ridge 的 0.50 天花板是信息上限还是模型上限？ |
| **对应 MVP** | MVP-2.0 |
| **Outcome A** | If Upper bound **≤ ~0.6** → 信息上限主导 → 优先 **Route I**，MoE/NN 投入谨慎 |
| **Outcome B** | If Upper bound **≥ ~0.8** → 模型/表征主导 → 直接 **Route M** |
| **Status** | ⏳ Pending |

### Gate-2: 表征跳变门 (Representation Jump)

| Item | Content |
|------|---------|
| **验证什么** | 轻量表征改进（形状特征/SNR-aware）能否带来跳变？ |
| **对应 MVP** | MVP-2.1 (E2 形状特征), MVP-2.2 (E3 SNR-aware) |
| **Outcome A** | If R² 跳变 **>10%** → 瓶颈在"形状/选择性过滤" → **Route M** 确认 |
| **Outcome B** | If R² 跳变 **<5%** → 更可能是信息退化 → 回到 Gate-1 深挖 |
| **Status** | ⏳ Pending |

### Gate-3: σ 审计门 (Sigma Audit)

| Item | Content |
|------|---------|
| **验证什么** | σ 通道的强信号是物理信息还是数据捷径（selection effect）？ |
| **对应 MVP** | MVP-2.3 (E5 σ 审计) |
| **Outcome A** | If Shuffle σ 后掉分 **>50%** → σ 是捷径 → 只用于诊断/分层，不作为主输入 |
| **Outcome B** | If Shuffle σ 后掉分 **<20%** → σ 增益更像物理稳健信号 → σ 进入主模型 |
| **Status** | ⏳ Pending |

---

## 1.3 This Week's Focus

> **本周要做的 2-3 个 MVP（对应 Gate 验证）**

| Priority | MVP | 对应 Gate | Why First | Status |
|----------|-----|-----------|-----------|--------|
| 🔴 P0 | MVP-2.2: E3 SNR-aware Ridge | Gate-2 | 最快、最像"开灯"——若成功直接确认 Route M | ⏳ |
| 🔴 P0 | MVP-2.0: Fisher sanity check | Gate-1 | 一锤定音"该不该重投入复杂模型" | ⏳ |
| 🟡 P1 | MVP-2.3: E5 σ 审计 | Gate-3 | 决定 σ 路线能不能押 | ⏳ |

---

## 1.4 Gate Progress Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Gate Progress Flow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Hub 推荐: Route M (Representation/Model)                      │
│                    ↓                                            │
│   ┌─────────────────────────────────────┐                       │
│   │ Gate-1: Fisher 上限                  │ Status: ⏳            │
│   │ MVP: MVP-2.0                         │                       │
│   └─────────────────────────────────────┘                       │
│          ↓ ≤0.6              ↓ ≥0.8                             │
│    Route I               Route M                                │
│    (信息上限)             (表征/模型)                            │
│                              │                                  │
│                    ┌─────────┴─────────┐                        │
│                    ↓                   ↓                        │
│   ┌────────────────────┐   ┌────────────────────┐              │
│   │ Gate-2: 表征跳变    │   │ Gate-3: σ 审计     │              │
│   │ MVP: 2.1, 2.2       │   │ MVP: 2.3           │              │
│   │ Status: ⏳          │   │ Status: ⏳          │              │
│   └────────────────────┘   └────────────────────┘              │
│          ↓                         ↓                            │
│   SNR-aware Attention        Route S 可否押？                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# 2. 📋 MVP List

> **Overview of all MVPs for quick lookup and tracking**

## 2.1 Experiment Summary

| MVP | Name | Phase | Gate | Status | experiment_id | Report |
|-----|------|-------|------|--------|---------------|--------|
| MVP-0.1 | Ridge α Sweep | 0 | - | ✅ Done | `VIT-20251127-ridge-alpha-01` | [exp_alpha_sweep](./exp/exp_ridge_alpha_sweep_20251127.md) |
| MVP-0.2 | Error Channel Analysis | 0 | - | ✅ Done | `VIT-20251127-ridge-error-01` | [exp_error](./exp/exp_error_logg_20251127.md) |
| MVP-0.3 | Feature Stability | 0 | - | ✅ Done | `VIT-20251128-ridge-stability-01` | [exp_stability](./exp/exp_feature_importance_stability_20251128.md) |
| MVP-0.4 | Top-K Selection | 0 | - | ✅ Done | `VIT-20251129-ridge-topk-01` | [exp_topk](./exp/exp_ridge_topk_20251129.md) |
| MVP-1.0 | Ridge 100k Noise Sweep | 1 | - | ✅ Done | `BM-20251205-ridge-100k` | [exp_100k](./exp/exp_ridge_100k_noise_sweep_20251205.md) |
| MVP-1.1 | Extended α Sweep (100k/1M) | 1 | - | ✅ Done | `VIT-20251222-scaling-ridge-01` | [exp_scaling](../scaling/exp/) |
| **MVP-2.0** | **Fisher Sanity Check** | 2 | Gate-1 | ⏳ Planned | - | - |
| **MVP-2.1** | **E2 形状特征显式化** | 2 | Gate-2 | ⏳ Planned | - | - |
| **MVP-2.2** | **E3 SNR-aware Ridge** | 2 | Gate-2 | ⏳ Planned | - | - |
| **MVP-2.3** | **E5 σ 泄漏审计** | 2 | Gate-3 | ⏳ Planned | - | - |
| MVP-3.0 | E4 Mixture-of-Linear | 3 | - | ⏳ Planned | - | - |

**Status Legend:**
- ⏳ Planned | 🔴 Ready | 🚀 Running | ✅ Done | ❌ Cancelled | ⏸️ Paused

## 2.2 Configuration Reference

| MVP | Data Size | Features | Model | Key Variable | Acceptance |
|-----|-----------|----------|-------|--------------|------------|
| MVP-0.1 | 32k/512 | 4096 flux | Ridge | α ∈ [0.001, 1000] | R²(noise=0) ≥ 0.99 |
| MVP-0.2 | 32k/512 | 4096 error | Ridge + LightGBM | 特征类型 | R² > 0.3 |
| MVP-0.3 | 32k/512 | 4096 flux | Ridge | α × noise 矩阵 | 相关性分析 |
| MVP-0.4 | 32k/512 | Top-K | Ridge | K ∈ [10, 2000] | K vs R² 曲线 |
| MVP-1.0 | 100k/1k | 4096 flux | Ridge | noise ∈ [0, 2] | vs 32k 对比 |
| MVP-1.1 | 100k/1M | 4096 flux | Ridge | α ∈ [1e2, 1e8] | 倒 U 型验证 |
| MVP-2.0 | 32k | 4096 flux | Fisher 分析 | - | Upper bound 估计 |
| MVP-2.1 | 32k | 一阶导/二阶导 | Ridge | 特征类型 | ΔR² > 10% |
| MVP-2.2 | 32k | [flux, σ, flux/σ] | Weighted Ridge | 权重方式 | ΔR² > 0 |
| MVP-2.3 | 32k | 4096 error | LightGBM | shuffle σ | 掉分幅度 |

---

# 3. 🔧 MVP Specifications

## Phase 0: Baseline (✅ Done)

<details>
<summary><b>MVP-0.1 ~ MVP-0.4</b> (已完成)</summary>

### MVP-0.1: Ridge α Sweep

| Item | Config |
|------|--------|
| **Objective** | 确定不同噪声下最优 α，验证线性假设 |
| **Data** | 32k train / 512 test, 4096 flux |
| **Model** | sklearn.linear_model.Ridge |
| **α Range** | [0.001, 0.01, 0.1, 1, 10, 100, 1000] |
| **Noise Range** | [0.0, 0.1, 0.2, 0.5, 1.0, 2.0] |
| **Result** | ✅ R²=0.999, 最优 α 规律确认 |

### MVP-0.2: Error Channel Analysis

| Item | Config |
|------|--------|
| **Objective** | 验证 Error σ 是否包含 log_g 信息 |
| **Data** | 32k train / 512 test, 4096 error |
| **Model** | Ridge (α=100) + LightGBM |
| **Result** | ✅ LightGBM R²=0.91, Linear R²≈0 |

### MVP-0.3: Feature Importance Stability

| Item | Config |
|------|--------|
| **Objective** | 分析特征重要性对 α 和噪声的稳定性 |
| **Data** | 32k train / 512 test, 4096 flux |
| **Model** | Ridge, 分析 \|w_i\| 相关性 |
| **Result** | ✅ noise=0 是"孤岛", 高噪声稳定 |

### MVP-0.4: Top-K Feature Selection

| Item | Config |
|------|--------|
| **Objective** | 测试基于 Ridge 系数的 Top-K 特征选择 |
| **Data** | 32k, Top-K features |
| **Result** | ✅ nz1.0 selector 在噪声测试下更优 |

</details>

---

## Phase 1: Scaling (✅ Done)

<details>
<summary><b>MVP-1.0 ~ MVP-1.1</b> (已完成)</summary>

### MVP-1.0: Ridge 100k Noise Sweep

| Item | Config |
|------|--------|
| **Objective** | 验证 Ridge 对数据量增益 |
| **Data** | 100k train / 1k test |
| **Model** | Ridge, 最优 α |
| **Result** | ✅ 平均增益 +2.71%, H1.3 成立 |

### MVP-1.1: Extended α Sweep (100k/1M)

| Item | Config |
|------|--------|
| **Objective** | 大样本下最优 α 搜索 |
| **Data** | 100k, 1M |
| **α Range** | [1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 1e7, 1e8] |
| **Result** | ✅ 100k: α=3e4, 1M: α=1e5 |

</details>

---

## Phase 2: Gate Verification (⏳ Current)

> **用于验证 Decision Gates 的实验**

### MVP-2.0: Fisher Sanity Check (Gate-1)

| Item | Config |
|------|--------|
| **Objective** | 估计 noise=1 下的信息论上限 |
| **Gate** | Gate-1: 信息上限门 |
| **Data** | 32k, noise=1 |
| **Method** | Fisher Information Matrix 估计 / Cramer-Rao bound / Posterior variance |
| **Acceptance** | 得出可解释的上限估计 |

**→ Gate Impact:** 
- If Upper bound ≤ 0.6 → Route I（信息上限主导）
- If Upper bound ≥ 0.8 → Route M（模型上限主导）

---

### MVP-2.1: E2 形状特征显式化 (Gate-2)

| Item | Config |
|------|--------|
| **Objective** | 测试显式形状特征能否带来跳变 |
| **Gate** | Gate-2: 表征跳变门 |
| **Data** | 32k, noise=1 |
| **Features** | 一阶导、二阶导、局部平滑差分、线系窗口等效宽度 |
| **Model** | Ridge |
| **Acceptance** | ΔR² > 10% (相对于 flux-only Ridge) |

**→ Gate Impact:** 
- If ΔR² > 10% → 瓶颈在 representation → Route M 确认
- If ΔR² < 5% → 形状特征无效 → 可能是信息退化

---

### MVP-2.2: E3 SNR-aware Ridge (Gate-2)

| Item | Config |
|------|--------|
| **Objective** | 测试 SNR-aware 的选择性过滤 |
| **Gate** | Gate-2: 表征跳变门 |
| **Data** | 32k, noise=1 |
| **Features** | [flux, σ, flux/σ] 或 weighted features |
| **Model** | Weighted Ridge（按 1/σ² 加权）或 augmented Ridge |
| **Acceptance** | ΔR² > 0 vs 标准 Ridge |

**→ Gate Impact:** 
- If 提升明显 → 选择性过滤是关键 → NN/MoE 应围绕 SNR 做门控
- If 无提升 → SNR-aware 不是关键

---

### MVP-2.3: E5 σ 泄漏审计 (Gate-3)

| Item | Config |
|------|--------|
| **Objective** | 审计 σ 通道是物理信号还是数据捷径 |
| **Gate** | Gate-3: σ 审计门 |
| **Data** | 32k, noise=0 |
| **Method** | (1) Shuffle σ 保持边际分布 (2) 只保留与 flux 同步的 Poisson 部分 |
| **Model** | LightGBM (error-only) |
| **Acceptance** | 观察性能下降幅度 |

**→ Gate Impact:** 
- If 掉分 > 50% → σ 走捷径（可能是 selection effect）
- If 掉分 < 20% → σ 增益更像物理稳健信号

---

## Phase 3: Extensions (⏳ Planned)

### MVP-3.0: E4 Mixture-of-Linear

| Item | Config |
|------|--------|
| **Objective** | 测试分区专家是否能超越全局 Ridge |
| **Data** | 32k, noise=1 |
| **Method** | 用 Teff/[M/H] 或无监督聚类做 gating，分簇后每簇 Ridge |
| **Acceptance** | 超过全局 Ridge |

---

# 4. 📊 Progress Tracking

## 4.1 Kanban View

```
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│  ⏳ Planned  │   🔴 Ready   │  🚀 Running  │    ✅ Done   │  ❌ Cancelled │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ MVP-2.0      │              │              │ MVP-0.1      │              │
│ MVP-2.1      │              │              │ MVP-0.2      │              │
│ MVP-2.2      │              │              │ MVP-0.3      │              │
│ MVP-2.3      │              │              │ MVP-0.4      │              │
│ MVP-3.0      │              │              │ MVP-1.0      │              │
│              │              │              │ MVP-1.1      │              │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

## 4.2 Gate Progress

| Gate | MVP | Status | Result | Outcome |
|------|-----|--------|--------|---------|
| Gate-1 | MVP-2.0 | ⏳ | - | - |
| Gate-2 | MVP-2.1, MVP-2.2 | ⏳ | - | - |
| Gate-3 | MVP-2.3 | ⏳ | - | - |

## 4.3 Key Conclusions Snapshot

> **One-line conclusion per completed MVP, synced to Hub**

| MVP | Conclusion | Key Metric | Synced to Hub |
|-----|------------|------------|---------------|
| MVP-0.1 | log_g-flux 映射本质线性，最优 α 随噪声单调增大 | R²=0.999 @ noise=0, α: 0.001→1000 | ✅ §2.1 B), E) |
| MVP-0.2 | Error σ 包含物理信息，关系非线性 | LightGBM R²=0.91 | ✅ §2.1 D) |
| MVP-0.3 | noise=0 是"孤岛"，高噪声下特征稳定 | ρ>0.95 @ noise≥0.5 | ✅ §2.1 C) |
| MVP-0.4 | Selector 需匹配测试噪声 | nz1.0 优于 nz0.0 | ✅ §2.1 C) |
| MVP-1.0 | Ridge 对数据量增益有限 | +2.71% avg (100k vs 32k) | ✅ §2.1 A) |
| MVP-1.1 | 大样本下最优 α 更大，存在倒 U 型曲线 | α=3e4 (100k), α=1e5 (1M) | ✅ §2.1 A), B) |

## 4.4 Timeline

| Date | Event | Notes |
|------|-------|-------|
| 2025-11-27 | MVP-0.1 完成 | α sweep baseline |
| 2025-11-27 | MVP-0.2 完成 | Error 通道分析 |
| 2025-11-28 | MVP-0.3 完成 | 特征稳定性 |
| 2025-11-29 | MVP-0.4 完成 | Top-K 选择 |
| 2025-11-30 | Phase 0 总结 | ridge_main 创建 |
| 2025-12-05 | MVP-1.0 完成 | 100k noise sweep |
| 2025-12-22 | MVP-1.1 完成 | 大样本 α 扩展 |
| 2025-12-23 | Hub/Roadmap 重构 | 新模板，Decision Gates |
| TBD | Phase 2 开始 | Gate 验证 |

---

# 5. 🔗 Cross-Repo Integration

## 5.1 Experiment Index

| experiment_id | project | topic | status | MVP |
|---------------|---------|-------|--------|-----|
| `VIT-20251127-ridge-alpha-01` | VIT | ridge | ✅ | MVP-0.1 |
| `VIT-20251127-ridge-error-01` | VIT | ridge | ✅ | MVP-0.2 |
| `VIT-20251128-ridge-stability-01` | VIT | ridge | ✅ | MVP-0.3 |
| `VIT-20251129-ridge-topk-01` | VIT | ridge | ✅ | MVP-0.4 |
| `BM-20251205-ridge-100k` | VIT | benchmark | ✅ | MVP-1.0 |
| `VIT-20251222-scaling-ridge-01` | VIT | scaling | ✅ | MVP-1.1 |

## 5.2 Repository Links

| Repo | Directory | Purpose |
|------|-----------|---------|
| VIT | `~/VIT/results/linear_alpha_search/` | α sweep 结果 |
| VIT | `~/VIT/results/benchmark_ridge_100k/` | 100k 结果 |
| This repo | `logg/ridge/` | Knowledge base |
| This repo | `logg/ridge/img/` | 图表 |

## 5.3 Run Path Records

| MVP | Repo | Script | Config | Output |
|-----|------|--------|--------|--------|
| MVP-0.1 | VIT | `scripts/alpha_sweep.sh` | `configs/exp/logg/linear_ridge.yaml` | `results/linear_alpha_search/` |
| MVP-0.2 | VIT | - | - | `models/lgbm_error_test/` |
| MVP-1.0 | VIT | `scripts/ridge_100k_noise_sweep.py` | - | `results/benchmark_ridge_100k/` |

---

# 6. 📎 Appendix

## 6.1 Results Summary

### Main Metrics Comparison

| MVP | Config | $R^2$ | MAE | RMSE | Key Finding |
|-----|--------|-------|-----|------|-------------|
| MVP-0.1 | noise=0, α=0.001 | 0.999 | 0.006 | 0.009 | 映射本质线性 |
| MVP-0.1 | noise=1.0, α=200 | 0.458 | 0.171 | 0.215 | 最优 α 增大 |
| MVP-0.1 | noise=2.0, α=1000 | 0.221 | 0.218 | 0.258 | 正则化收益 +68% |
| MVP-0.2 | Error-only, LightGBM | 0.910 | 0.187 | 0.228 | Error 包含物理信息 |
| MVP-1.0 | 100k, noise=0 | 0.9994 | - | - | vs 32k +0.04% |
| MVP-1.0 | 100k, noise=2 | 0.2536 | - | - | vs 32k +14.8% |
| MVP-1.1 | 100k, α=3e4 | 0.4856 | - | - | 倒 U 型验证 |
| MVP-1.1 | 1M, α=5000 | 0.4997 | - | - | Ridge 天花板 |

### Optimal α vs Noise (32k)

| Noise | Best α | Best R² | OLS R² | ΔR² |
|-------|--------|---------|--------|-----|
| 0.0 | 0.001 | 0.999 | 0.969 | +3.1% |
| 0.1 | 1.0 | 0.909 | 0.901 | +0.9% |
| 0.2 | 10.0 | 0.826 | 0.811 | +1.9% |
| 0.5 | 50.0 | 0.655 | 0.608 | +7.8% |
| 1.0 | 200.0 | 0.458 | 0.385 | +18.9% |
| 2.0 | 1000.0 | 0.221 | 0.131 | +68.4% |

### Optimal α vs Data Size (noise=1.0)

| Data Size | Best α | Best R² | vs baseline |
|-----------|--------|---------|-------------|
| 32k | 200 | 0.458 | - |
| 100k | 3.16e4 | 0.4856 | +2.55% |
| 1M | 5000 | 0.4997 | +0.42% |

---

## 6.2 File Index

| Type | Path | Description |
|------|------|-------------|
| Roadmap | `logg/ridge/ridge_roadmap_20251223.md` | This file |
| Hub | `logg/ridge/ridge_hub_20251223.md` | Knowledge navigation |
| MVP-0.1 | `logg/ridge/exp/exp_ridge_alpha_sweep_20251127.md` | α sweep 实验 |
| MVP-0.2 | `logg/ridge/exp/exp_error_logg_20251127.md` | Error 通道实验 |
| MVP-0.3 | `logg/ridge/exp/exp_feature_importance_stability_20251128.md` | 特征稳定性 |
| MVP-0.4 | `logg/ridge/exp/exp_ridge_topk_20251129.md` | Top-K 选择 |
| MVP-1.0 | `logg/ridge/exp/exp_ridge_100k_noise_sweep_20251205.md` | 100k 实验 |
| Images | `logg/ridge/img/` | 实验图表 |

---

## 6.3 Changelog

| Date | Change | Sections |
|------|--------|----------|
| 2025-12-23 | 🚦 **新增 §1 Decision Gates**：Gate-1/2/3 + This Week's Focus + Progress Flow | §1 全面新增 |
| 2025-12-23 | 📋 MVP 列表增加 Phase 2 Gate 验证实验 (MVP-2.0 ~ 2.3) | §2, §3 |
| 2025-12-23 | Created Roadmap | All |
| 2025-12-22 | MVP-1.1 completed | §4 |
| 2025-12-05 | MVP-1.0 completed | §4 |
| 2025-11-30 | Phase 0 completed | §4 |
| 2025-11-27 | Started Phase 0 | §3 |

---

> **Template Usage:**
> 
> ## Hub vs Roadmap 职责分工
> 
> | 问题 | Hub | Roadmap |
> |------|-----|---------|
> | 我们知道什么？ | ✅ §2 Answer Key | |
> | 该往哪走？ | ✅ §2 Strategic Route | |
> | 怎么验证？（Decision Gates） | | ✅ §1 |
> | 做哪些实验？ | | ✅ §2, §3 |
> | 本周做什么？ | | ✅ §1.3 This Week's Focus |
> | 进度如何？ | | ✅ §4 |
> | 学到了什么洞见？ | ✅ §3 Confluence | |
> | 设计原则是什么？ | ✅ §4 Principles | |
> 
> ## Roadmap Scope
> - ✅ **Do:** Decision Gates, MVP specs, execution tracking, progress, cross-repo integration
> - ❌ **Don't:** Insight synthesis (→ hub.md), strategic reasoning (→ hub.md)


---

## 🆕 Ridge Baseline Consolidation (2025-12-24)

### 实验总结

| Experiment | Config | Best α | R² | Status |
|------------|--------|:------:|---:|:------:|
| Alpha Sweep (500 test) | 1M, StandardScaler | 100,000 | 0.5017 | ✅ |
| Alpha Sweep (1k test) | 1M, StandardScaler | 100,000 | **0.4551** | ✅ 标准 |
| Y-Scaling | 1M, +MinMaxScaler(y) | 100,000 | 0.4551 | ✅ 无效 |

### 最终基准线

| 指标 | 值 | 备注 |
|------|-----|------|
| **Standard R²** | **0.4551** | 1M train, 1k test, α=100k |
| MAE | 0.6605 | |
| RMSE | ~0.80 | |

### Alpha 选择指南

| Data Size | Optimal α | R² |
|-----------|:---------:|---:|
| 100k | 31,623 | 0.4856 |
| 1M | **100,000** | **0.4551** |

### 与 Oracle MoE 对比

| Model | R² | ΔR² |
|-------|---:|----:|
| Global Ridge | 0.4551~0.4611 | baseline |
| Oracle MoE (9 bins) | 0.6249 | **+0.16** |

**结论**: Ridge 有明确上限，MoE 分层建模可突破 ~16%。

---

*Updated: 2025-12-24*
