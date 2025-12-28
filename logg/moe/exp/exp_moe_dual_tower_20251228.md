# 🍃 MoE Dual-Tower Gate: Physics + Quality
> **Name:** Dual-Tower MoE Gate  
> **ID:** `VIT-20251228-moe-dual-tower-01`  
> **Topic:** `moe_dual_tower` | **MVP:** MVP-4.0/4.1/4.2 | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-28 | **Status:** ⏳ 立项  
> **Root:** `moe_hub` | **Parent:** `moe_snr_hub` | **Child:** -

> 🎯 **Target:** 验证"物理 9-gate (Teff×[M/H])"和"SNR/quality gate"双塔融合能否叠加增益  
> 🚀 **Next:** 若成功 → 统一 MoE 架构；若失败 → 选择单塔最优

---

## ⚡ 核心结论速览

> **一句话**: 立项中，待执行

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| Q6.1: 双塔 gate > 单塔 phys gate? | ⏳ | 待验证 |
| Q6.2: quality gate 控制温度有效? | ⏳ | 待验证 |
| Q6.3: 因子分解 gate 可解释? | ⏳ | 待验证 |

| Type | Link |
|------|------|
| 🧠 Hub | `logg/moe/moe_hub_20251203.md` § Q6 |
| 🧠 SNR Hub | `logg/moe/moe_snr_hub.md` |
| 🗺️ Roadmap | `logg/moe/moe_snr_roadmap.md` § MVP-4.x |

---

# 1. 🎯 目标

## 1.1 核心问题

**物理 9-gate (Teff×[M/H])** 和 **SNR/quality gate** 是**两条正交的条件轴**：
- **物理轴**：描述"物理子域的函数分段"
- **质量轴**：描述"观测质量导致的可用信息量变化"

**已有证据支持两条线各自都能跑通**：
- 物理窗 gate 在 9 expert 上能做到 **ρ ≈ 1.13**（超 Oracle）
- SNR-MoE 也能做到 **ρ ≈ 1.04**（超 Oracle）

**核心假设**：Soft routing 是关键，边界误分不会致命，反而能吃到"跨边界混合更优"的收益。

## 1.2 验证目标

| 预期 | 判断标准 |
|------|---------|
| 双塔叠加增益 | ΔR² ≥ +0.01 (相对单塔最优) → 继续开发 |
| 低 SNR 子集改善 | per-SNR bin ΔR² ≥ +0.02 → quality gate 有效 |
| 无叠加增益 | 选择更简单的单塔方案 |

---

# 2. 🦾 设计方案（从最小改动到最大容量）

## 2.1 方案 A: 最小改动（推荐先做）✅ MVP-4.0

**思路**：9 个物理 experts 不变，把 SNR/quality 特征当作 gate 的"辅助条件"

```
Gate 输入:
├── phys_features:  CaT/Na/PCA1-4 等物理窗特征（已验证）
└── quality_features: 去泄露版 10D aggregate stats（已冻结）

Gate 输出: 9 维 soft weights
Experts: 9× Ridge（按 Teff×[M/H] 分训练，不变）
```

**Gate 训练目标**：用 **回归最优 gate（MSE）** 而不是分类 CE

**优点**：
- 不增加 experts 数量，工程最稳
- 风险低：quality feature 用的是"去泄露版本"
- 预期提升"低 SNR 切片"与"边界样本"的稳定性

### 变体：Quality 控制"温度/熵"

让 quality tower 输出温度 $\tau(q)$（$\tau \geq 1$）：

$$w = \mathrm{softmax}\left(\frac{\text{logits}_{\text{phys}}}{\tau(q)}\right)$$

- **SNR 低** → τ 大 → 权重更均匀（更像 ensemble / 更鲁棒）
- **SNR 高** → τ 小 → 权重更尖锐（更像 routing / 更专精）

**物理直觉**：噪声大就别太自信。

---

## 2.2 方案 B: 因子分解/双 gate 融合 ✅ MVP-4.1

保留"两个 gate 各自可解释"，做 **factorized gating**：

### B1: Logit 相加融合（最稳、最像双塔）

$$
a = f_{\text{phys}}(\text{phys\_feat}) \quad \text{(9 维 logits)}
$$
$$
b = f_{\text{snr}}(\text{quality\_feat}) \quad \text{(9 维 logits/bias)}
$$
$$
w = \mathrm{softmax}(a + b)
$$

**解释**：SNR gate 不负责"判物理 bin"，它负责"给每个 expert 加一个质量相关的偏置/先验"。

### B2: 概率乘积融合（Product-of-Experts）

$$
p = \mathrm{softmax}(a), \quad q = \mathrm{softmax}(b)
$$
$$
w \propto p \odot q \quad \text{(归一化)}
$$

**解释**：两个 gate 都"同意"的专家权重会上升；任意一个 gate 觉得不该用的专家会被压下去。

---

## 2.3 方案 C: 3D 联合分域（36 Experts）🟡 MVP-4.2 (谨慎)

把条件维度做成 3D（Teff×[M/H]×SNR）：

$$
\text{物理 9 bin} \times \text{SNR 4 bin} \Rightarrow \textbf{36 experts}
$$

**风险**：数据被切得太碎，某些 (phys_bin, snr_bin) cell 可能样本很少。

**降低碎片化的办法**：
1. **层级共享**：$\text{expert}_{k,s} = \text{expert}_k + \Delta_s$（或用 FiLM / 线性缩放）
2. **只在 SNR 轴上做正则化调整**：每个 SNR bin 只调整 α，不完全独立训练

---

## 2.4 双塔抽象架构

```
Tower_phys(物理窗/局部CNN) ────→ h_phys
Tower_qual(10D质量统计)   ────→ h_qual
         │                           │
         └───────→ Fusion ←──────────┘
                     │
                     ▼
              gate logits (9-dim)
                     │
                     ▼
         softmax → weights w (9-dim)
                     │
                     ▼
         prediction = Σ w_k * expert_k(flux)
```

**设计建议**：
- qual tower 输入**必须用"去泄露版本"** 的 quality_features（Gate-1 已冻结）
- gate head 用**回归损失（MSE）**训练更贴近最终目标（Next-A 的经验）
- `h_qual` 可同时输出一个"温度/熵惩罚系数"，用于控制 gate 的 sharpness

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | BOSZ 1M |
| 路径 | `~/VIT/data/bosz50000/z0/mag205_225_lowT_1M/` |
| Train/Val/Test | 800k / 100k / 100k |
| 特征维度 | 4096 (flux) + 10 (quality) + N (phys) |
| 目标 | log_g |

## 3.2 噪声

| 项 | 值 |
|----|-----|
| 类型 | heteroscedastic |
| noise_level | 1.0 |
| 范围 | all |

## 3.3 模型

| 参数 | 值 |
|------|-----|
| Experts | 9× Ridge（按 Teff×[M/H] 分训练） |
| Gate (MVP-4.0) | MLP([phys_feat, quality_feat]) → 9 logits |
| Gate (MVP-4.1) | f_phys + f_snr → softmax |
| Gate (MVP-4.2) | 36 experts + 层级共享 |

## 3.4 训练

| 参数 | 值 |
|------|-----|
| Gate 训练 | 回归最优（MSE on final logg） |
| Experts | 预训练 Ridge（沿用已有） |
| seed | 42, 123, 456 |

## 3.5 扫描参数

| 扫描 | 范围 | 固定 |
|------|------|------|
| 融合方式 | logit-add, prob-prod, temp-control | experts=9 |
| quality feature 维度 | 10 (frozen), 5 (subset) | - |

---

# 4. 📊 评估指标

## 4.1 主要指标

| 指标 | 定义 | 目标 |
|------|------|------|
| ΔR² | R²_dual - R²_single_best | ≥ +0.01 |
| ρ | (R²_deploy - R²_global) / (R²_oracle - R²_global) | ≥ 0.8 |

## 4.2 切片分析（必须）

| 切片 | 说明 |
|------|------|
| 按 SNR bins | >7 / 4-7 / 2-4 / ≤2 |
| 按物理 bin | 9 个 Teff×[M/H] 组合 |
| 边界样本 | 多 bin 重叠区域 |

## 4.3 可解释性分析

- Gate 熵/置信度随 SNR 的关系（低 SNR 是否更"平"）
- 双 gate 权重相关性
- 哪个 tower 贡献更大

---

# 5. 📋 MVP 执行计划

## 5.1 MVP 总览

| MVP | 名称 | 依赖 | 工作量 | 状态 |
|-----|------|------|--------|------|
| **4.0** | 最小改动：9 experts + gate concat phys+qual + MSE | - | ~2h | ⏳ |
| **4.1** | 因子分解：logit-add / prob-prod | 4.0 | ~1h | ⏳ |
| **4.2** | 温度控制：quality → τ | 4.0 | ~1h | ⏳ |
| **4.3** | 36 experts (可选) | 4.0/4.1 通过后 | ~4h | ⏳ |

## 5.2 MVP-4.0 详细规格

| 项 | 配置 |
|----|------|
| 目标 | 验证 phys+qual 联合 gate 能否超过单塔 |
| Experts | 9× Ridge（沿用已有） |
| Gate 输入 | `[phys_features, quality_features]` |
| Gate 架构 | MLP [32, 16] → 9 logits |
| 训练 | 回归最优（直接最小化 logg MSE） |
| 验收 | ΔR² ≥ +0.005 → 继续；< 0 → fallback 单塔 |

## 5.3 关键参考代码

| 脚本 | 功能 | 需修改 |
|------|------|--------|
| `~/VIT/scripts/logg_snr_moe.py` | SNR-MoE 实现 | 添加 phys_features |
| `~/VIT/scripts/physical_feature_gate.py` | 物理窗 gate | 添加 quality_features |
| `~/VIT/lib/quality_features.py` | quality_features() | 无（已冻结） |

---

# 6. ⚠️ 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| quality feature 与 phys feature 冗余 | 无叠加增益 | 先做消融：只加 quality |
| 训练不稳定 | Gate 震荡 | 用 warmup + 低 lr |
| 36 experts 数据碎片 | 方差大 | 层级共享 / 早停 |

---

# 7. 📎 附录

## 7.1 理论依据

**为什么两条轴正交**：

1. **物理轴 (Teff×[M/H])**：决定谱线的"存在性"与"分布"
   - [M/H] 决定金属线强度
   - Teff 决定电离平衡、谱线激发

2. **质量轴 (SNR)**：决定"可用信息量"
   - 高 SNR：细节可用，精调有效
   - 低 SNR：只有大信号可用，需要更保守

**物理直觉**：一个解释"光谱长什么样"，一个解释"我们能看清多少"。

## 7.2 冻结的 quality_features() 实现

```python
def quality_features(error: np.ndarray) -> np.ndarray:
    """10 aggregate statistics - de-leaked, SNR-preserving."""
    from scipy import stats
    return np.column_stack([
        np.mean(error, axis=-1),      # 0: mean
        np.std(error, axis=-1),       # 1: std
        np.min(error, axis=-1),       # 2: min
        np.max(error, axis=-1),       # 3: max
        np.median(error, axis=-1),    # 4: median
        np.sum(error, axis=-1),       # 5: sum
        np.percentile(error, 25, axis=-1),  # 6: q25
        np.percentile(error, 75, axis=-1),  # 7: q75
        stats.skew(error, axis=-1),   # 8: skew
        stats.kurtosis(error, axis=-1),     # 9: kurtosis
    ])
```

## 7.3 上游文档链接

| 文档 | 关键内容 |
|------|---------|
| `moe_hub_20251203.md` | 物理 9-gate 验证：ρ=1.00~1.13 |
| `moe_snr_hub.md` | SNR-MoE 验证：ρ=1.04，超越 Oracle |
| `exp_logg_snr_gate_01_20251226.md` | quality_features 去泄露验证 |

---

> **立项时间**: 2025-12-28
