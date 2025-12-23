# 🧠 Knowledge Hub: Data Scaling

> **Topic:** Data Scaling & Model Capacity  
> **Author:** Viska Wei  
> **Created:** 2025-12-22 | **Updated:** 2025-12-22  
> **Status:** 🔄 Exploring

## 🔗 Related Files

| Type | File | Description |
|------|------|-------------|
| 📍 Roadmap | [`scaling_roadmap_20251222.md`](./scaling_roadmap_20251222.md) | Experiment tracking |
| 📗 Experiments | `exp/exp_*.md` | Detailed reports |
| 📇 Cards | `card_*.md` | Condensed insights |

## 📑 Contents

- [1. 🌲 Question Tree](#1--question-tree)
- [2. 🔺 Hypothesis Pyramid](#2--hypothesis-pyramid)
- [3. 💡 Insight Confluence](#3--insight-confluence)
- [4. 🧭 Strategic Navigation](#4--strategic-navigation)
- [5. 📐 Design Principles](#5--design-principles)
- [6. 📎 Appendix](#6--appendix)

---

# 1. 🌲 Question Tree

> **Hierarchical structure of research questions and boundaries**

## 1.1 Top-Level Question

> **在高噪声环境下（noise_level=1），传统机器学习方法（Linear Regression, LightGBM）是否存在性能天花板？大规模数据 + 神经网络能否突破这个瓶颈？**

## 1.2 Question Decomposition

```
🎯 Top-Level: 数据规模与模型容量的 Scaling Law
│
├── Q1: 传统 ML 的数据 Scaling 瓶颈
│   ├── Q1.1: Linear Regression 在 1M 数据下能达到什么性能？ → ✅ MVP-1.0 (R²=0.50)
│   ├── Q1.2: LightGBM 在 1M 数据下能达到什么性能？ → ✅ MVP-1.1 (R²=0.57)
│   ├── Q1.3: 增加数据从 100k → 1M 能带来多少提升？ → ✅ MVP-1.2 (ΔR²<0.03)
│   │
│   ├── Q1.4: "没 plateau" 是真实还是统计假象？ → ⏳ MVP-1.3 (P0)
│   │   ├── Q1.4.1: 多 seed 重复时 R² 的方差有多大？
│   │   └── Q1.4.2: test=500 的评估噪声是否掩盖真实趋势？
│   │
│   ├── Q1.5: Ridge α 是否还没扫到最优？ → ✅ MVP-1.4 (Done)
│   │   ├── Q1.5.1: α=5000 是否只是 sweep 边界而非真正最优？
│   │   └── Q1.5.2: 更大 α (1e6, 1e8) 能否继续提升？
│   │
│   ├── Q1.6: LightGBM 参数空间是否探索完全？ → ⏳ MVP-1.5 (P0)
│   │   ├── Q1.6.1: num_leaves 增大能否抬高上限？
│   │   ├── Q1.6.2: lr 减小是否能更精细拟合？
│   │   └── Q1.6.3: early stopping 是否"过早停"了？
│   │
│   ├── Q1.7: 输入表示方式是否最优？ → ⏳ MVP-1.6/1.7 (P1)
│   │   ├── Q1.7.1: Whitening (flux/error) 能否提升？
│   │   ├── Q1.7.2: PLS (监督降维) vs PCA (无监督) 哪个更好？ → 🔴 MVP-1.7
│   │   ├── Q1.7.3: PCA 会不会误伤低方差高信息的细谱线特征？ → 🔴 MVP-1.7
│   │   └── Q1.7.4: PCA 建在什么空间最稳健（noisy/whitened/denoised）？ → 🔴 MVP-1.7
│   │
│   └── Q1.8: 分段建模能否提升极值区域？ → ⏳ MVP-1.8 (P2)
│       └── Q1.8.1: 按 Teff/log_g bin 分模型效果如何？
│
├── Q2: 神经网络的大数据优势
│   ├── Q2.1: MLP 在 1M 数据下的性能？ → ⏳ MVP-2.0
│   ├── Q2.2: CNN 在 1M 数据下的性能？ → ⏳ MVP-2.1
│   └── Q2.3: 数据量翻倍时 NN 的性能提升是否持续？ → ⏳ MVP-2.2
│
└── Q3: 瓶颈的本质是什么？
    ├── Q3.1: 是噪声导致的信息上限？ → ⏳ MVP-3.0
    ├── Q3.2: 是模型容量不足？ → ⏳ MVP-3.1
    └── Q3.3: 是特征表达能力限制？ → ⏳ MVP-3.2
│
├── Q4: 🔴 理论上限是多少？（Phase 16T/L 新增）
│   ├── Q4.1: Fisher/CRLB 理论上限 R²_max 是多少？ → ✅ MVP-16T (R²=0.97，但需校准)
│   │   ├── Q4.1.1: 边缘化后 log_g 的 CRLB 下界是多少？ → ✅ 0.2366 Schur decay
│   │   ├── Q4.1.2: degeneracy (log_g 与 Teff/[M/H] 的信息纠缠) 有多强？ → ✅ 极强
│   │   │
│   │   └── 🆕 Q4.1.3: Fisher ceiling 是否被高估（偏导混参污染）？ → ⏳ MVP-T 系列
│   │       ├── Q4.1.3.1: 收紧邻居约束后 R²_max 是否下降？ → ⏳ MVP-T1
│   │       ├── Q4.1.3.2: R²_max 随 noise_level 是否单调下降？ → ⏳ MVP-T0
│   │       ├── Q4.1.3.3: 用局部线性回归估 Jacobian 结果如何？ → ⏳ MVP-T2
│   │       └── Q4.1.3.4: noise=1 实际 SNR 是多少？ → ⏳ MVP-T3
│   │
│   ├── Q4.2: 线性模型族的上限 (LMMSE) 是多少？ → ⏳ MVP-16L
│   │   └── Q4.2.1: Ridge 与 LMMSE 差多少？差 < 1% 则线性已到极限
│   │
│   └── Q4.3: 结构上限 (Oracle MoE headroom) 是多少？ → ⏳ MVP-16A-0
│       └── Q4.3.1: Oracle MoE - Global Ridge ≥ 0.03 则 MoE 值得做 @ noise=1
│
├── Q5: 🔴 现有 ceiling 结论可信吗？（Phase 16B 新增）
│   ├── Q5.1: 多 seed 重复时 R² 的方差有多大？ → ⏳ MVP-16B
│   ├── Q5.2: 扩大 test set (500→5k) 后趋势是否改变？ → ⏳ MVP-16B
│   └── Q5.3: LightGBM 参数空间是否覆盖完全？ → ⏳ MVP-16B
│
└── Q6: 🟡 表示/模型方向哪个更值得投入？（Phase 16W/CNN 新增）
    ├── Q6.1: Whitening (flux/error) 能带来多少提升？ → ⏳ MVP-16W
    ├── Q6.2: 1D-CNN 能比 Ridge 提升多少？ → ⏳ MVP-16CNN
    └── Q6.3: 表示改进 + CNN 组合能接近理论上限吗？ → ⏳ MVP-16CNN

Legend: ✅ Verified | ❌ Rejected | 🔄 In Progress | ⏳ Pending | 🚫 Closed
```

## 1.3 Scope Boundaries

> **Define what is and isn't within research scope**

| ✅ In Scope | ❌ Out of Scope |
|------------|----------------|
| noise_level = 1.0 (高噪声) | noise_level < 0.5 (低噪声场景) |
| BOSZ 模拟光谱数据 | 真实观测数据 |
| log_g 预测任务 | 多目标预测 (T_eff, Fe_H) |
| 1M 规模数据 | 分布式训练 / 10M+ 数据 |
| Ridge, LightGBM, MLP, CNN | Transformer, Diffusion 等复杂模型 |

---

# 2. 🔺 Hypothesis Pyramid

> **Strategic → Tactical → Testable hypotheses, progressively refined**

## 2.1 L1 Strategic Hypotheses

> **Core beliefs that determine research direction**

| # | Hypothesis | Status | If True | If False |
|---|------------|--------|---------|----------|
| **H1** | 传统 ML 在高噪声 + 大数据下存在**不可逾越的性能瓶颈** | ⏳ | NN 是唯一出路 | 继续优化 ML 方法 |
| **H2** | 神经网络能从**大规模数据**中学到传统 ML 无法捕获的模式 | ⏳ | 证明 data-driven 路线正确 | 重新审视问题 |
| **H3** | 🔴 noise=1 场景存在**可计算的理论上限** R²_max ≤ 某值 | ⏳ | 用 Fisher/CRLB 量化 degeneracy | 理论分析不适用 |
| **H4** | 🔴 **MoE 结构红利在高噪声下仍存在** (Oracle headroom > 0.02) | ⏳ | MoE 值得做 | 放弃 MoE，直接上 CNN |

## 2.2 L2 Tactical Hypotheses

> **Concrete implementation paths for strategic hypotheses**

| # | Hypothesis | Parent | Status | Key MVP |
|---|------------|--------|--------|---------|
| **H1.1** | Ridge 的瓶颈源于**线性假设**无法建模非线性噪声-信号交互 | H1 | ⏳ | MVP-1.0 |
| **H1.2** | LightGBM 的瓶颈源于**树模型的局部性**，无法做全局模式提取 | H1 | ⏳ | MVP-1.1 |
| **H1.3** | 数据量从 100k → 1M 对传统 ML 的边际收益**递减至零** | H1 | ⏳ | MVP-1.2 |
| **H2.1** | MLP 能学习到**非线性特征组合**，突破线性瓶颈 | H2 | ⏳ | MVP-2.0 |
| **H2.2** | CNN 能学习到**局部-全局特征层次**，优于 MLP | H2 | ⏳ | MVP-2.1 |
| **H3.1** | 🔴 Fisher/CRLB 给出的 R²_max ≥ 0.75 (存在大 headroom) | H3 | ✅ | MVP-16T |
| **H3.2** | 🔴 LMMSE (最优线性预测器) 与 Ridge 差 < 1% | H3 | ⏳ | MVP-16L |
| **H4.1** | 🔴 Oracle MoE - Global Ridge ≥ 0.05 (结构红利明显) | H4 | ⏳ | MVP-16O |
| **H4.2** | 🔴 可落地 Gate 在 noise=1 下仍能保住 ρ ≥ 0.7 | H4 | ⏳ | MVP-16G |

## 2.3 L3 Testable Hypotheses

> **Each hypothesis maps to a specific experiment with clear acceptance criteria**

| # | Testable Hypothesis | Parent | Criteria | Result | Source |
|---|---------------------|--------|----------|--------|--------|
| **H1.1.1** | Ridge 在 1M 数据、noise=1 下，R² < 0.6 | H1.1 | R² < 0.6 | ✅ 0.4997 | MVP-1.0 |
| **H1.2.1** | LightGBM 在 1M 数据、noise=1 下，R² < 0.65 | H1.2 | R² < 0.65 | ✅ 0.5709 | MVP-1.1 |
| **H1.3.1** | Ridge: 1M vs 100k 的 ΔR² < 0.02 | H1.3 | ΔR² < 0.02 | ❌ 0.0244 | MVP-1.2 |
| **H1.3.2** | LightGBM: 1M vs 100k 的 ΔR² < 0.03 | H1.3 | ΔR² < 0.03 | ✅ 0.0176 | MVP-1.2 |
| **H1.4.1** | 多 seed 重复时，1M vs 500k 差异在误差棒内 | H1.4 | 差异 < σ | ⏳ | MVP-1.3 |
| **H1.4.2** | 扩大 test set (500→1k) 后趋势判断改变 | H1.4 | test=1k+ | ⏳ | MVP-1.3 |
| **H1.5.1** | Ridge 最优 α 在 5000~1e8 之间存在峰值 | H1.5 | 峰值后下降 | ✅ | MVP-1.4 |
| **H1.6.1** | num_leaves=127/255 能提升 R² > 0.01 | H1.6 | ΔR² > 0.01 | ⏳ | MVP-1.5 |
| **H1.6.2** | lr=0.01/0.02 能提升 R² > 0.01 | H1.6 | ΔR² > 0.01 | ⏳ | MVP-1.5 |
| **H1.7.1** | Whitening (flux/error) 能提升 R² > 0.02 | H1.7 | ΔR² > 0.02 | ❌ +0.0146 | MVP-1.6 |
| **H1.7.2** | PLS 优于 PCA（相同维度下） | H1.7 | PLS > PCA | 🔴 | MVP-1.7 |
| **H1.7.3** | PCA 可能误伤低方差高信息特征（细谱线） | H1.7 | PCA 降维后 R² < 全特征 Ridge | 🔴 | MVP-1.7 |
| **H1.7.4** | Whitened/Denoised space 建 PCA 比 noisy space 更稳健 | H1.7 | R²(whitened) > R²(noisy) | 🔴 | MVP-1.7 |
| **H2.1.1** | MLP 在 1M 数据、noise=1 下，R² > 0.70 | H2.1 | R² > 0.70 | ⏳ | MVP-2.0 |
| **H2.2.1** | CNN 在 1M 数据、noise=1 下，R² > Ridge + 0.15 | H2.2 | ΔR² > 0.15 | ⏳ | MVP-2.1 |
| **H3.1.1** | 🔴 Fisher CRLB 转换的 R²_max ≥ 0.75 (大 headroom 存在) | H3.1 | R²_max ≥ 0.75 | ✅ **0.9661** | MVP-16T |
| **H3.1.2** | 🔴 degeneracy 指标显著 (log_g 与 Teff/[M/H] 信息纠缠) | H3.1 | Fisher 条件数 > 100 | ✅ **8.65×10⁵** | MVP-16T |
| **H3.2.1** | 🔴 LMMSE R² - Ridge R² < 0.01 (线性模型已到极限) | H3.2 | 差值 < 0.01 | ⏳ | MVP-16L |
| **H4.1.1** | 🔴 Oracle MoE (9 bin) R² > 0.55 @ noise=1 | H4.1 | R² > 0.55 | ⏳ | MVP-16O |
| **H4.1.2** | 🔴 Oracle MoE - Global Ridge ≥ 0.05 @ noise=1 | H4.1 | ΔR² ≥ 0.05 | ⏳ | MVP-16O |
| **H4.2.1** | 🔴 Gate 9-class 准确率 > 60% @ noise=1 | H4.2 | Acc > 60% | ⏳ | MVP-16G |
| **H4.2.2** | 🔴 Soft routing ρ ≥ 0.7 @ noise=1 | H4.2 | ρ ≥ 0.7 | ⏳ | MVP-16G |

### 🔴 Phase 16 信息论上限假设（2025-12-23 新增）

| # | 可验证假设 | 上层 | 验证标准 | 结果 | 来源 |
|---|-----------|------|---------|------|------|
| **H-16T.1** | CRLB 给出的 MSE_min 可转换为 R²_max 上界 | H3 | R²_max ≥ 0.75 | ✅ **0.9661** ⚠️ 待校准 | MVP-16T |
| **H-16T.2** | degeneracy 显著 (Schur decay < 0.9) | H3 | Schur < 0.9 | ✅ **0.2366** | MVP-16T |
| **H-16L.1** | LMMSE (Σ_xx^-1 Σ_xy) 是线性模型族上限 | H3.2 | Ridge ≈ LMMSE | ⏳ | MVP-16L |
| **H-16B.1** | 多 seed 实验确认 Ridge=0.50, LGB=0.57 的统计置信度 | H1 | std < 0.01 | ⏳ | MVP-16B |
| **H-16B.2** | 扩大 test set (500→5k) 后 ceiling 结论不变 | H1 | ΔR² < 0.01 | ⏳ | MVP-16B |

### ❌ Phase T: Fisher Ceiling 校准假设（方法失败）

> **失败原因**：BOSZ 数据是连续采样（~40k 唯一值/参数），不是规则网格
> **根本问题**：邻近点差分法在非网格数据上完全失效

| # | 可验证假设 | 上层 | 验证标准 | 结果 | 来源 |
|---|-----------|------|---------|------|------|
| ~~H-T0.1~~ | ~~R²_max 随 noise_level 单调下降~~ | H-16T | - | ❌ **取消** | 方法失败 |
| ~~H-T1.1~~ | ~~收紧邻居约束后 R²_max 显著下降~~ | H-16T | - | ❌ **取消** | 方法失败 |
| **H-T2.1** | 局部线性回归 Jacobian 给出更稳定 ceiling | H-16T | CRLB 分布合理 | ⏳ 降级 | MVP-T2 |
| **H-T3.1** | noise=1 实际 SNR ≈ 1（非虚高 SNR） | H-16T | median(\|flux\|)/median(error×σ) ≈ 1 | ⏳ | MVP-T3 |

### 🆕 Phase D: 经验上限假设（替代 Fisher）

> **核心思路**：用 noise=0 的经验上限替代理论 CRLB

| # | 可验证假设 | 上层 | 验证标准 | 结果 | 来源 |
|---|-----------|------|---------|------|------|
| **H-D0.1** | noise=0 时 Ridge R² > 0.95 | H3 | R² > 0.95 | ⏳ | MVP-D0 |
| **H-D0.2** | headroom = R²(noise=0) - R²(noise=1) > 0.40 | H3 | > 0.40 | ⏳ | MVP-D0 |

### 🆕 Phase A: noise=1 MoE 结构红利假设（2025-12-23 新增）

> **核心问题**：noise=1 下 MoE 的结构红利是否还存在？

| # | 可验证假设 | 上层 | 验证标准 | 结果 | 来源 |
|---|-----------|------|---------|------|------|
| **H-A0.1** | Oracle MoE @ noise=1 有结构红利 | H4 | ΔR² ≥ 0.03 vs Global Ridge | ⏳ | MVP-16A-0 |
| **H-A1.1** | Gate 特征 @ noise=1 仍有分类信号 | H4.2 | Ca II triplet 等特征可区分 bins | ⏳ | MVP-16A-1 |
| **H-A2.1** | Soft-gate MoE @ noise=1 能保持 ≥70% oracle 收益 | H4.2 | ρ ≥ 0.7 | ⏳ | MVP-16A-2 |

## 2.4 Dependency Graph

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                   Hypothesis Pyramid Dependencies (Phase 16 扩展)              │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   L1:  [H1 ML 瓶颈]      [H2 NN 优势]     [H3 理论上限]     [H4 MoE 结构红利] │
│            │                  │                │                  │           │
│      ┌─────┼─────┐      ┌────┴────┐      ┌────┴────┐      ┌─────┴─────┐     │
│      ▼     ▼     ▼      ▼         ▼      ▼         ▼      ▼           ▼     │
│   L2: H1.1 H1.2 H1.3  H2.1      H2.2   H3.1      H3.2   H4.1        H4.2   │
│      Ridge LGB  边际  MLP       CNN   Fisher   LMMSE  Oracle      Gate    │
│       │     │     │     │         │      │         │      │           │     │
│       ▼     ▼     ▼     ▼         ▼      ▼         ▼      ▼           ▼     │
│   L3: ✅    ✅    ⏳    ⏳        ⏳     ⏳        ⏳     ⏳          ⏳    │
│                                                                               │
│   Phase 16 三件套（性价比最高）：                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                               │
│   │ MVP-16T  │ → │ MVP-16O  │ → │ MVP-16B  │                                │
│   │ 理论上限  │    │ 结构上限  │    │ 可信度   │                                │
│   └──────────┘    └──────────┘    └──────────┘                               │
│         ↓               ↓               ↓                                     │
│   决定：上限多高？  决定：MoE 值不值？  决定：baseline 可信？                   │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

# 3. 💡 Insight Confluence

> **Aggregate findings from multiple experiments → high-level conclusions**

## 3.1 Confluence Index

| # | Theme | Sources | Conclusion | Confidence |
|---|-------|---------|------------|------------|
| C1 | 传统 ML 性能天花板 | MVP-1.0, 1.1, 1.2 | Ridge=0.50, LGB=0.57 @ 1M, noise=1 | 🟢 高 |
| C2 | Ridge α 优化空间 | MVP-1.4 | 最优 α=1e4~1e5，倒 U 型曲线，优化提升仅 0.4%~2.5% | 🟢 高 |
| C3 | SNR 输入表示 | MVP-1.6 | H1.7.1 ❌: SNR ΔR²=+0.015 未达阈值; StandardScaler 严重损害 LightGBM (-0.36) | 🟢 高 |
| **C4** | **⚠️ Fisher ceiling 可能虚高** | MVP-16T + 理论分析 | R²_max=0.97 可能因"偏导混参污染"被高估，需 MVP-T 系列校准 | 🟡 待验证 |
| C3 | Whitening/SNR 输入影响 | MVP-1.6 | SNR 对 Ridge +1.5%（未达阈值），StandardScaler 严重损害 LightGBM（-0.36） | 🟢 高 |

## 3.2 Confluence Details

### C3: Whitening/SNR 输入方式对模型影响（MVP-1.6）

**来源**: MVP-1.6 (Whitening/SNR Input Experiment)

**发现汇合**:
1. **SNR/Whitening 对 Ridge 微小提升但未达阈值**: snr_centered ΔR²=+0.0146 (< 0.02)
2. **StandardScaler 对 Ridge 无害**: raw ≈ standardized
3. **⚠️ 重大发现: StandardScaler 严重损害 LightGBM**: raw=0.5533 vs standardized=0.1966 (-0.36!)
4. **SNR 化对 LightGBM 有害**: SNR 输入导致 LightGBM 性能暴跌至 ~0.01

**汇合结论**: 
> - Ridge 对 input scaling 几乎不敏感（线性可逆变换）
> - LightGBM 必须使用 raw 输入，standardization 会破坏树模型的分裂点最优性
> - Whitening/SNR 化不是改进信号质量的银弹

---

### C1: 传统 ML 性能天花板

**来源**: MVP-1.0 (Ridge), MVP-1.1 (LightGBM), MVP-1.2 (Scaling Law)

**发现汇合**:
1. Ridge 在 1M 数据下 R²=0.50，LightGBM R²=0.57
2. 100k → 1M 仅提升 2-3%，边际收益递减明显
3. 高噪声下非线性优势有限（LGB 仅比 Ridge 好 ~7%）

**汇合结论**: 传统 ML 在高噪声场景下存在明确的表达能力瓶颈，深度学习目标应突破 R²=0.70

---

### C2: Ridge α 优化空间有限

**来源**: MVP-1.4 (Ridge Alpha Extended Sweep)

**发现汇合**:
1. 倒 U 型曲线明确存在：100k 峰值 α=3.16e+04，1M 峰值 α=1.00e+05
2. 最优 α 比原 baseline (5000) 高 1-2 个数量级
3. 优化 α 仅带来 0.4%~2.5% 提升，说明 Ridge ceiling 确实存在
4. 更大的数据量需要更大的最优 α（正相关）

**汇合结论**: Ridge 的瓶颈不是 α 调参问题，而是线性模型本身的表达能力限制。应转向 NN 方法。

---

### C4: ⚠️ Fisher Ceiling 可能虚高（需校准）

**来源**: MVP-16T 结果 + 理论分析 (2025-12-23)

**问题发现**:
MVP-16T 计算 R²_max ≈ 0.97，第一反应应该是"哪里把信息算多了"。

**最可疑根因：偏导估计的"混参污染"（confounding）**：
1. KDTree 找邻居做有限差分时，"其它参数差不多"的容许区间偏大
2. flux 的变化里混入了其它参数的影响（Teff ↔ logg ↔ [M/H] 强耦合）
3. 这部分变化被"归因"给当前参数的偏导 → |∂μ/∂θ| 被系统性放大
4. Fisher I = J^T Σ^-1 J 随梯度平方放大 → **CRLB 异常小** → **R²_max 虚高**

**这同时解释了**：
- "上限很高"（0.97）：梯度被放大
- "degeneracy 很强"（Schur=0.24）：cross-term 也很大

**验证方法（MVP-T 系列）**：
1. **T0 (Monotonicity)**: noise 0.2→1.0→2.0，R²_max 应单调下降
2. **T1 (Confounding)**: 收紧邻居约束 5-10 倍，看 R²_max 是否大幅下降
3. **T2 (LLR Jacobian)**: 用局部线性回归估计 Jacobian，天然控制混参
4. **T3 (Scale audit)**: 确认 noise=1 实际 SNR

**物理可能性**：
- 即使 noise=1 真的很吵，**7200 波长点**的信息会累积
- 信息按 I ~ Σ (∂μ/∂θ)² / σ² 求和
- 单点 SNR 不高 ≠ 参数估计一定差
- 但 R²≈0.97 意味着 CRLB RMSE 远小于 log_g 分布 std，仍需验证

**决策规则**：
- 如果 T1 后 R²_max 从 0.97 降到 0.7-0.85 → 坐实"混参污染"，使用校准后的值
- 如果 T1 后 R²_max 仍 >0.9 → ceiling 可信，headroom 确实很大

---

# 4. 🧭 Strategic Navigation

> **Recommended research directions based on accumulated insights**

## 4.1 Direction Status Overview

```
┌───────────────────────────────────────────────────────────────┐
│                    Research Direction Status                  │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   🟢 High Confidence              🟡 Pending                  │
│   └── (待验证)                    ├── 传统 ML 瓶颈验证        │
│                                   └── NN scaling 验证         │
│                                                               │
│   🔴 Risky                        ⚫ Closed                   │
│   └── (待验证)                    └── (无)                    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## 4.2 High Confidence Directions (🟢)

| Direction | Evidence | Next Action | Priority |
|-----------|----------|-------------|----------|
| 传统 ML 存在性能天花板 | Ridge=0.50, LGB=0.57 @ 1M | 确认统计可信度 | 🟢 已确认 |
| 数据量非瓶颈 | 100k→1M 仅 +2-3% | 转向模型改进 | 🟢 已确认 |

## 4.3 Pending Directions (🟡)

| Direction | Depends On | Required MVP | Expected Gain |
|-----------|------------|--------------|---------------|
| 确认 plateau 真实性 | 多 seed + 大 test set | MVP-1.3 | 验证趋势可信 |
| ~~扩展 Ridge α 搜索~~ | ~~MVP-1.3 完成~~ | ~~MVP-1.4~~ | ✅ **已完成**: 最优 α=1e4~1e5，仅提升 0.4%~2.5% |
| 扩展 LightGBM 参数 | MVP-1.3 完成 | MVP-1.5 | 验证是否达上限 |
| Whitening/PLS 输入表示 | MVP-1.4/1.5 完成 | MVP-1.6, 1.7 | 可能 +2-5% R² |
| 验证 NN 优势 | Phase 1.x 完成 | MVP-2.0, 2.1 | 突破 0.70 目标 |

## 4.4 Risky/Low Priority Directions (🔴)

| Direction | Risk | Required MVP | Expected Gain |
|-----------|------|--------------|---------------|
| 分段建模 (MoE) | 实现复杂，可能过拟合 | MVP-1.8 | 极值区域改进 |
| 物理特征工程 | 领域知识依赖重 | MVP-1.9 | 不确定 |
| 继续堆数据到 2M+ | 边际收益极小 | - | 不推荐 |

---

# 5. 📐 Design Principles

> **Reusable principles distilled from experiments**

## 5.1 Confirmed Principles

| # | Principle | Recommendation | Evidence | Scope |
|---|-----------|----------------|----------|-------|
| P1 | Ridge α 应更大 | α ∈ [1e4, 1e5] 而非 5000 | MVP-1.4 倒 U 型曲线 | noise=1 场景 |
| P2 | α 与数据量正相关 | 更多数据 → 更大的最优 α | 100k: α=3e4, 1M: α=1e5 | 线性模型 |
| P3 | 避免过度正则化 | α > 1e6 时 R² 急剧下降 | MVP-1.4 扫描 | Ridge |
| **P4** | **LightGBM 必须用 raw 输入** | ❌ 禁止 StandardScaler | MVP-1.6: standardized ΔR²=-0.36 | 树模型 |
| **P5** | **Ridge 对 scaling 不敏感** | StandardScaler 可用可不用 | MVP-1.6: raw ≈ standardized | 线性模型 |
| **P6** | **SNR 化效果有限** | 不推荐作为默认输入 | MVP-1.6: ΔR²=+0.015 < 0.02 | 全模型 |

## 5.2 Pending Principles

| # | Principle | Initial Suggestion | Needs Verification |
|---|-----------|--------------------|--------------------|
| P1 | 高噪声场景选模型 | 优先考虑 NN 而非 ML | MVP-2.x |
| P2 | 数据量收益 | 100k → 1M 对 NN 有效，对 ML 无效 | MVP-1.2, 2.2 |

## 5.3 Key Numbers Reference

> **Quick reference for important values (from previous experiments)**

| Metric | Value | Condition | Source |
|--------|-------|-----------|--------|
| Ridge R² (100k, σ=1) | 0.4856 | α=3.16e+04 (最优) | MVP-1.4 |
| Ridge R² (1M, σ=1) | 0.5017 | α=1.00e+05 (最优) | MVP-1.4 |
| Ridge 最优 α (100k) | 3.16e+04 | noise=1 | MVP-1.4 |
| Ridge 最优 α (1M) | 1.00e+05 | noise=1 | MVP-1.4 |
| LightGBM R² (1M, σ=1) | 0.5709 | 1M train | MVP-1.1 |
| α 调优提升幅度 | 0.4%~2.5% | vs baseline α=5000 | MVP-1.4 |
| **LightGBM raw vs std ΔR²** | **-0.3567** | StandardScaler 严重损害 | MVP-1.6 |
| **SNR_centered vs std ΔR² (Ridge)** | +0.0146 | 微小提升，未达阈值 | MVP-1.6 |
| **最优输入 (Ridge)** | snr_centered | R²=0.5222 | MVP-1.6 |
| **最优输入 (LightGBM)** | raw | R²=0.5533 | MVP-1.6 |

---

# 6. 📎 Appendix

## 6.1 Domain Background

### 6.1.1 BOSZ 光谱模拟数据

- **来源**: BOSZ (Bohlin, Rauch, & Sah) 恒星大气模型
- **分辨率**: R = 50,000
- **波长范围**: MR 臂 (6500-9500 Å)
- **参数范围**:
  - T_eff: 3750-6000 K
  - log_g: 1.0-5.0 dex
  - [Fe/H]: -1.0-0.0 dex

### 6.1.2 噪声模型

- **noise_level = σ**: 高斯噪声标准差
- **σ = 1.0** 对应约 SNR ≈ 1（极低信噪比）
- 物理意义：模拟暗天体 / 恶劣观测条件

### 6.1.3 1M 数据集规格

| 项目 | 值 |
|------|-----|
| 数据集名称 | mag205_225_lowT_1M |
| 总样本数 | 1,000,000 |
| 总数据量 | 93 GB |
| 分片 | 5 × 200k |
| 路径 | `/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/` |

---

## 6.2 Glossary

| Term | Definition | Notes |
|------|------------|-------|
| Scaling Law | 模型性能随数据/参数规模的变化规律 | 本研究关注数据 scaling |
| 性能瓶颈 | 增加资源无法继续提升性能的临界点 | 本假设认为 ML 存在此瓶颈 |
| noise_level (σ) | 高斯噪声标准差 | σ=1 为高噪声 |

---

## 6.3 Changelog

| Date | Change | Sections |
|------|--------|----------|
| 2025-12-22 | Created Hub | - |
| 2025-12-22 | Phase 1 结果填充 | §2.3, §3, §5.3 |
| 2025-12-22 | Phase 1.x 规划 | §1.2, §2.3, §4 |
| 2025-12-23 | MVP-1.7 PCA vs PLS 立项，添加 H1.7.3, H1.7.4 | §1.2, §2.3 |
| 2025-12-23 | MVP-1.4 完成，H1.5.1 验证通过，洞见 C2 汇合 | §2.3, §3, §4.3, §5 |
| **2025-12-23** | **🔴 Phase 16 大立项：信息论上限 + 结构上限 + 可信度验证** | §1.2, §2, §2.4 |
| 2025-12-23 | 添加 Q4-Q6 问题树（理论上限、可信度、表示方向） | §1.2 |
| 2025-12-23 | 添加 H3, H4 战略假设 + H3.x, H4.x 假设链 | §2.1, §2.2, §2.3 |
| 2025-12-23 | 添加 Phase 16 假设组（H-16T, H-16L, H-16B） | §2.3 |
| 2025-12-23 | 更新依赖图：展示 Phase 16 三件套 | §2.4 |
| **2025-12-23** | **MVP-16T 完成：H-16T.1, H-16T.2, H3.1 验证通过** | §2.2, §2.3, §3 |
| **2025-12-23** | **🆕 Phase T/A/NN 大立项：Fisher 校准 + MoE 验证 + NN baseline** | §1.2, §2.3, §3 |
| 2025-12-23 | 添加 Q4.1.3 问题树（Fisher 可信度）| §1.2 |
| 2025-12-23 | 添加 H-T/A 假设系列 | §2.3 |
| 2025-12-23 | 添加 C4 洞见（Fisher ceiling 可能虚高）| §3 |
| **2025-12-23** | **❌ MVP-16T 失败：方法论缺陷（非规则网格）** | §2.3, §3 |
| 2025-12-23 | 取消 H-T0.1, H-T1.1；降级 H-T2.1 | §2.3 |
| 2025-12-23 | 新增 Phase D 经验上限假设 (H-D0.1, H-D0.2) | §2.3 |
| **2025-12-23** | **MVP-1.6 完成：H1.7.1 ❌ REJECTED，洞见 C3 汇合，P4-P6 原则** | §2.3, §3.1, §3.2, §5.1, §5.3 |

---

> **Template Usage:**
> 
> **Hub Scope:**
> - ✅ **Do:** Question mapping, hypothesis management, insight synthesis, strategic navigation, design principles
> - ❌ **Don't:** Experiment tracking (→ roadmap.md), daily backlog (→ kanban.md)
> 
> **Hub vs Roadmap:**
> - Hub = "What do we know? Where should we go?"
> - Roadmap = "What experiments are planned? What's the progress?"

---

## 🔬 SCALING-20251222-ml-ceiling-01 洞见汇合

### 假设验证结果

| 假设 | 预期 | 实际 | 验证 |
|------|------|------|------|
| H1.1.1: Ridge R² < 0.6 @ 1M, σ=1 | < 0.6 | 0.4997 | ✅ |
| H1.2.1: LightGBM R² < 0.65 @ 1M, σ=1 | < 0.65 | 0.5709 | ✅ |
| H1.3.1: Ridge ΔR² < 0.02 (1M vs 100k) | < 0.02 | 0.0244 | ❌ (略超) |
| H1.3.2: LightGBM ΔR² < 0.03 (1M vs 100k) | < 0.03 | 0.0176 | ✅ |

### 关键发现

1. **性能天花板确认**
   - Ridge: R² = 0.4997 ≈ 0.50
   - LightGBM: R² = 0.5709 ≈ 0.57
   - 两者均远低于低噪声时的 0.90+

2. **数据增益边际递减**
   - 10k → 100k: +0.10~0.14 R² (显著提升)
   - 100k → 1M: +0.02~0.03 R² (边际收益)
   - Scaling law 在传统 ML 上失效

3. **非线性优势有限**
   - LightGBM 仅比 Ridge 好 ~7%
   - 高噪声下，树模型的特征交互价值有限

### 汇合结论

> 传统 ML 在高噪声场景下存在明确的表达能力瓶颈，R² ≈ 0.50-0.57。
> 这为深度学习方法设定了清晰的改进目标：突破 0.70。
> 后续实验应聚焦于模型架构创新，而非数据量扩展。

### 下游影响

| 影响 | 具体行动 |
|------|---------|
| MVP-2.0 (MLP) 目标 | R² > 0.60 才有意义 |
| MVP-2.1 (CNN) 目标 | R² > 0.65 才算突破 |
| 资源分配 | 模型研发 > 数据采集 |

---

# 🆕 Phase 16T 洞见更新 (2025-12-23)

## 新增已验证假设

| # | Hypothesis | Parent | Criteria | Result | Source |
|---|-----------|--------|----------|--------|--------|
| **H-16T.1** | R²_max (CRLB) ≥ 0.75 | H3.1 | ≥ 0.75 | ✅ **0.9661** | MVP-16T |
| **H-16T.2** | degeneracy 显著 (Schur < 0.9) | H3.1 | < 0.9 | ✅ **0.2366** | MVP-16T |

## 洞见汇合站更新

### I-16T.1: 理论上限极高，巨大 headroom 存在

> **核心发现**：noise=1 下 Fisher/CRLB 理论上限 R²_max = 0.97 (median)，远超当前最佳 LightGBM (0.57)。
> 
> **含义**：投入更复杂模型（CNN/Transformer）是值得的，提升空间约 40%。

### I-16T.2: degeneracy 是主要信息瓶颈

> **核心发现**：Schur decay = 0.24，表明边缘化 Teff/[M/H] 后，log_g 的 Fisher 信息损失了 76%。
> 
> **含义**：
> - log_g 与 Teff/[M/H] 高度纠缠
> - Multi-task 联合估计可能是解纠缠的关键
> - 单独估计 log_g 本质上很困难

### I-16T.3: Fisher 条件数极高，参数耦合强

> **核心发现**：Fisher 矩阵条件数 median = 8.65×10⁵，表明参数之间存在强耦合。
> 
> **物理解释**：
> - 压力敏感线同时也对温度敏感
> - 金属丰度影响谱线强度，与 log_g 效应混淆
> - 高噪声下这些效应更难分离

## 设计原则更新

| Principle | From | Description |
|-----------|------|-------------|
| **P-16T.1** | MVP-16T | 理论上限 R²_max ≈ 0.97 证明继续投入更强模型是合理的 |
| **P-16T.2** | MVP-16T | degeneracy 强 → 考虑 multi-task 联合预测 Teff/log_g/[M/H] |
| **P-16T.3** | MVP-16T | 不同 Teff 区间的 degeneracy 程度不同 → 可能需要区域特化模型 |

## 关键数字速查

| Metric | Value | Description |
|--------|-------|-------------|
| R²_max (CRLB median) | **0.9661** | 理论上限 |
| R²_max (CRLB 90%) | 0.9995 | 高分位上限 |
| Schur decay | 0.2366 | 边缘化后仅保留 24% 信息 |
| Fisher 条件数 | 8.65×10⁵ | 参数耦合强度 |
| Gap vs Ridge | +0.47 | 相对 0.50 的提升空间 |
| Gap vs LightGBM | +0.40 | 相对 0.57 的提升空间 |

---

# ❌ Phase 16T 失败总结 (2025-12-23 更新)

## MVP-16T 失败原因

| 问题 | 详情 |
|------|------|
| **核心失败** | 偏导数估计方法存在根本性缺陷 |
| **数据问题** | BOSZ 为连续采样（~40k 唯一参数值），不是规则网格 |
| **方法问题** | 邻近点差分法在非规则网格上完全失效 |
| **数值表现** | CRLB 跨越 20 个数量级（2.77e-10 到 1.00e+10） |
| **结果可靠性** | R²_max = 0.97 **不可信** |

## 假设验证状态

| Hypothesis | Status | 说明 |
|------------|--------|------|
| H-16T.1: R²_max ≥ 0.75 | ❓ 未验证 | 方法失败 |
| H-16T.2: degeneracy 显著 | ❓ 未验证 | 方法失败 |

## 经验教训

1. **数据结构检查**：设计算法前应先验证数据假设
2. **数值稳定性**：CRLB 跨 20 个数量级是明显红旗
3. **Sanity check**：R²_max=0.97 vs 实际 R²=0.57 的差距本身值得怀疑

## 下一步方向

| 方向 | 优先级 | 说明 |
|------|--------|------|
| 暂停 MVP-16T | - | 等待方法论改进 |
| 调研 BOSZ 前向模型 | 🟡 | 是否可数值微分 |
| 尝试局部多项式回归 | 🟡 | 在现有数据上验证 |
| 改用经验上限（noise=0 Oracle） | 🟢 | 替代方案 |
