# 🧠 Scaling Hub (v3 Audited) — 数据规模与模型容量
> **ID:** VIT-20251222-scaling-hub | **Version:** v3 (Audited 2025-12-27)  
> **Scope:** BOSZ 合成光谱 → log_g 回归  
> **Focus:** noise=1 (高噪声) + 大数据 (≥100k, up to 1M)  
> **Roadmap:** [scaling_roadmap_20251222.md](./scaling_roadmap_20251222.md)  
> **Audit:** [contradiction_audit.md](./contradiction_audit.md)

---

## 0. TL;DR (≤10 Lines)

### Current Consensus (≤3 Statements)

| # | Consensus | So What | Evidence |
|---|-----------|---------|----------|
| **C1** | 传统 ML 在 noise=1 下存在明确 ceiling: Ridge≈0.46, LightGBM≈0.57 | 继续堆数据对传统 ML 无意义 | `oracle-moe-noise1-01`, `ml-ceiling-01` |
| **C2** | 理论上限 R²_max=0.89 >> 当前最佳，存在 +32% headroom | 继续投入 CNN/MoE 值得 | `fisher-ceiling-v2` |
| **C3** | Soft-gate MoE 保留 80.5% Oracle 收益 (ρ=0.805) | MoE 是 noise=1 场景可落地主线方案 | `soft-moe-noise1-01` |

### Decision Ready
- ✅ **Route B (MoE + Structure)** 验证通过，可进入生产化阶段
- 🟡 **Route A (CNN)** 待 MLP@1M flux_only 实验后裁决

---

## 1. Consensus (Stable)

> 仅放 Verified 结论，每条带明确 Scope + Evidence

| # | Statement | Scope | Evidence | Confidence |
|---|-----------|-------|----------|------------|
| **K1** | Ridge R² = **0.46** | train=1M, test=1k, noise=1, α=100k, pre-noised | `oracle-moe-noise1-01` | ✅ Verified |
| **K2** | LightGBM R² = **0.57** | train=1M, test=500, noise=1, raw input | `ml-ceiling-01` | ✅ Verified |
| **K3** | Oracle MoE R² = **0.62** | train=1M, test=1k, noise=1, 9-bin | `oracle-moe-noise1-01` | ✅ Verified |
| **K4** | Soft-gate MoE R² = **0.59**, ρ=0.805 | train=1M, test=1k, noise=1, 37-dim gate | `soft-moe-noise1-01` | ✅ Verified |
| **K5** | Fisher ceiling R²_max = **0.89** | mag=21.5, grid data, 3D | `fisher-ceiling-v2` | ✅ Verified |
| **K6** | 5D Fisher ceiling R²_max = **0.87** | mag=21.5, grid data, 5D+chemical | `fisher-ceiling-v3a` | ✅ Verified |
| **K7** | 100k→1M 对 Ridge 增益 <3% | noise=1 | `ml-ceiling-01` | ✅ Verified |
| **K8** | Gate 特征 Acc=88% @ noise=1 | 37-dim (PCA+physical) | `gate-feat-sanity-01` | ✅ Verified |

---

## 2. Conditional Insights (Slice-Dependent)

> 按 mag/SNR/协议分层的结论

### 2.1 By Test Size

| Condition | Ridge R² | Notes |
|-----------|----------|-------|
| test=500 (deprecated) | 0.50 | Historical, superseded |
| **test=1k (canonical)** | **0.46** | Current standard |

### 2.2 By Magnitude/SNR

| Magnitude | SNR | R²_max (median) | Model Efficiency | Notes |
|-----------|-----|-----------------|------------------|-------|
| 18.0 | 87.4 | 0.9994 | ~60% | 信息饱和 |
| 20.0 | 24.0 | 0.9906 | ~60% | 信息饱和 |
| **21.5** | **7.1** | **0.89** | **64%** | **Canonical (noise=1)** |
| 22.0 | 4.6 | 0.74 | - | 临界区 |
| 22.5 | 3.0 | 0.37 | - | 信息悬崖边缘 |
| 23.0 | 1.9 | 0.00 | - | 信息悬崖 |

### 2.3 By Input Preprocessing

| Model | Best Input | Δ vs Alternative | Evidence |
|-------|------------|------------------|----------|
| Ridge | StandardScaler or raw | ≈0 | `whitening-01` |
| LightGBM | **raw only** | -0.36 vs StandardScaler | `whitening-01` |
| NN (MLP/CNN) | **flux_only** | Whitening causes R²≈0 | `nn-baseline-01` |

---

## 3. Open Contradictions / Unknowns

### 3.1 Unresolved

| Issue | What We Don't Know | Minimal Fix | Priority |
|-------|-------------------|-------------|----------|
| **MLP 1M flux_only** | Does MLP benefit from 1M scale? | Run MLP 3L_1024 @ 1M flux_only | 🔴 P0 |
| **CNN hyperparams** | Why CNN < MLP @ 100k? | Tune lr, warmup, bn | 🟡 P1 |

### 3.2 Pending Verification

| Claim | Current Status | What Would Close It |
|-------|----------------|---------------------|
| "NN can match Oracle MoE" | ❌ Current best NN=0.47 << 0.62 | NN (MLP/CNN) R² ≥ 0.60 |
| "Weighted loss helps" | ⏳ Not tested | MVP-F-WGT |

---

## 4. Rejected / Invalidated Claims

> 已否定的结论，防止未来误用

| Claim | Reason | Evidence | Date |
|-------|--------|----------|------|
| Fisher V1 R²_max = 0.97 | **Method failed**: BOSZ continuous sampling caused CRLB to span 20 orders | `fisher-ceiling-01` | 2025-12-23 |
| MLP whitening R² = -0.0003 | **Implementation failed**: whitening preprocessing causes training collapse | `nn-baseline-01` | 2025-12-24 |
| "100k→1M 不提升 Ridge" | **Partially incorrect**: 提升 2-3%, 虽小但非零 | `ml-ceiling-01` | 2025-12-22 |
| Ridge R² = 0.50 @ 500 test | **Superseded**: 已被 1k test 协议取代 | `ridge-1ktest-01` | 2025-12-24 |

---

## 5. Decision Hooks

> Hub 洞见 → Roadmap 决策门

### 5.1 Route Selection

| Condition | Route | Action |
|-----------|-------|--------|
| If Soft MoE ρ ≥ 0.70 | **Route B: MoE** ✅ | 进入生产化 |
| If efficiency@highSNR < 80% | Route A: 投模型 | 继续 CNN/Transformer |
| If efficiency@highSNR ≥ 80% | Route B: 投结构 | MoE/分域 |
| If SNR < 2 (mag > 23) | Route C: 改任务 | 多曝光/先验/分类 |

### 5.2 Active Decision

**Current State**: Route B (MoE) 验证通过，ρ=0.805 ≥ 0.70

**Next Gate**: MVP-MLP-1M (决定是否 Route A 也可行)

---

## 6. Canonical Evaluation Protocol (Frozen)

| Item | Specification |
|------|---------------|
| Dataset | BOSZ 50000, mag205_225_lowT_1M |
| Train | 1M (5 shards × 200k) |
| **Test** | **1k (full test_1k_0, pre-noised)** ← Canonical |
| Features | 4096 (MR arm) |
| Target | log_g ∈ [1.0, 5.0] |
| Noise | σ=1.0 (heteroscedastic Gaussian, pre-stored) |
| Metric | R² over test set |
| Ridge α | 100000 (for 1M train) |
| LightGBM input | raw (never standardized) |
| NN input | flux_only (never whitening) |

> **Rule**: 任何口径变更必须写入 §9 Changelog

---

## 7. Design Principles (Portable)

| # | Principle | Recommendation | Scope | Evidence |
|---|-----------|----------------|-------|----------|
| P1 | Ridge α 应更大 | α ∈ [1e4, 1e5] | noise=1, 1M | `ridge-alpha-01` |
| P2 | LightGBM 必须用 raw | ❌ 禁止 StandardScaler | 所有 LightGBM | `whitening-01` |
| P3 | NN 必须用 flux_only | ❌ 禁止 whitening | 所有 NN | `nn-baseline-01` |
| P4 | 高噪声优先分域 | MoE/分域比堆数据划算 | noise≥1 | `oracle-moe-noise1-01` |
| P5 | Fisher 必须用规则网格 | ❌ 禁止连续采样差分 | 理论分析 | `fisher-v1 vs v2` |
| P6 | Gate 特征 PCA+物理窗口 | 37 维足够 | MoE routing | `gate-feat-sanity-01` |
| P7 | Test 用 pre-noised | ❌ 禁止 on-fly 加噪 | 所有实验 | Canonical protocol |

---

## 8. Pointers

| Type | File | Description |
|------|------|-------------|
| 📍 Roadmap | [`scaling_roadmap_20251222.md`](./scaling_roadmap_20251222.md) | 实验规划与执行 |
| 🧠 Fisher Hub | [`fisher_hub_20251225.md`](./fisher_hub_20251225.md) | Fisher/CRLB 专题 |
| 📗 Experiments | `exp/exp_*.md` | 详细实验报告 |
| 🔍 Audit | [`contradiction_audit.md`](./contradiction_audit.md) | 矛盾审计 |

---

## 9. Changelog

| Date | Change | Impact |
|------|--------|--------|
| 2025-12-22 | 创建 Hub v1 | - |
| 2025-12-24 | Hub v2 重构 | 精简假设→结论账本 |
| **2025-12-27** | **Hub v3 Audited** | **矛盾消除，协议标准化** |
| - | 标准化 test=1k 协议 | Ridge canonical = 0.46 |
| - | 隔离 Fisher V1 到 §Rejected | 防止误用 |
| - | 隔离 whitening 失败到 §Rejected | 防止误用 |
| - | 新增 §Decision Hooks | 明确决策规则 |

---

## 📎 Appendix: Canonical Scoreboard

> **唯一权威口径**: train=1M, test=1k, noise=1.0, metric=R²

| Model | R² | Config | Status |
|-------|-----|--------|--------|
| Ridge | **0.46** | α=100k | ✅ Verified |
| LightGBM | **0.57** | raw input | ✅ Verified |
| MLP (100k) | 0.47 | flux_only, 3L_1024 | ✅ Verified |
| CNN (100k) | 0.41 | flux_only, 4L_k5_bn | ✅ Verified |
| Oracle MoE | **0.62** | 9-bin 真值 routing | ✅ Verified |
| Soft-gate MoE | **0.59** | 37-dim gate | ✅ Verified |
| Fisher ceiling | **0.89** | V2 规则网格 3D | ✅ Verified |
| Fisher ceiling | **0.87** | V3-A 5D+chemical | ✅ Verified |

---

*Audited: 2025-12-27 | Source: contradiction_audit.md*
