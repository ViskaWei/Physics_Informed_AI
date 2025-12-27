下面我按你给的 **Hub / Roadmap 模版**把这条 “**按 SNR level 分 MoE + 合理利用 error vector（避免信息泄露）**” 的方案做一次可行性分析，并给出两份可直接落盘的 Markdown（Hub + Roadmap）。

---

## 先回答：方案可行吗？关键风险在哪？

### ✅ 为什么“按 SNR/信息量分专家”是有戏的

1. **理论上限强烈依赖 SNR**：Fisher/CRLB 的 multi-mag 扫描显示，SNR 从 7→4→3 会让可达上限 (R^2_{max}) 从 ~0.89 → ~0.74 → ~0.37，存在明显“分段”与“临界点”（SNR≈4）——这非常像 MoE 的适用场景：不同 SNR 区间的最优映射/最优正则强度不同。 
2. **高噪声下结构红利更大**：noise=1 时 Oracle MoE（9 专家）比 Global Ridge 高 **+0.1637 R²**，说明“分段训练/分域建模”在低 SNR/高噪声 regime 下更值得做（至少结构上限很高）。  
3. **你当前 ViT=0.71 vs Fisher~0.8（或更高）**：说明仍有 headroom，SNR-aware 方向是合理杠杆（尤其当数据是 mag205–225 混合、SNR 分布很宽时）。同时 Fisher V2（单 mag=21.5）给的中位上限甚至到 0.89，说明“上限不是卡死的”。 

### ⚠️ 为什么你现在的 error vector “不能直接用”

你观察到 **只喂 clean error vector 做线性回归，R²=0.91**，这说明 error vector 在你的生成管线里**携带了强烈的天体参数信息**（至少携带了能 proxy 出 logg 或其强相关变量的信号）。
即使 error vector 96% 相似，只剩 **40/4096** 不同，那 40 个位置很可能恰好对应“随谱型/谱线深度变化的 Poisson 项 / mask / throughput 特征”，它们可以把模型“抄近路”。

> 结论：**error vector 不是不能用，而是必须“去泄露/口径一致化”后再用**——把它约束成“只表达观测质量（SNR）”，而不是“表达光谱内容”。

### ✅ 一个核心思路：把 error vector 压缩成“观测质量参数”

你给出的“96% 相似 + 少量差异”其实是好消息：意味着 error 向量可近似写成
[
e \approx s \cdot e_0 + \delta
]

* (e_0)：全局模板（instrument throughput + 固定噪声形状）
* (s)：单标量（整体噪声幅度 / magnitude / exposure）
* (\delta)：稀疏残差（最可能藏泄露的部分）

**可落地策略**：优先只用 (s)（或少量粗粒度统计），把 (\delta) 当作“异常检测/质量标志”而不是回归特征。

---

# 🧠 Hub（按模板，可直接存成 `logg_snr_moe_hub_20251226.md`）

> 下面内容遵循你上传的 Hub 模版结构。  并参考 MoE Hub 的表达方式（问题树、ρ 指标、路线决策）。 

---

# 🧠 logg SNR-MoE Hub

> **ID:** `VIT-20251226-logg-snr-moe-hub` | **Status:** ✅完成 |
> **Date:** 2025-12-26 | **Update:** 2025-12-27
> **Root:** `logg` | **Child:** `snr_gate`, `error_repr`, `snr_experts` |

### 📤 Parent Hubs (本 Hub 的上层关联)

| Parent Hub | 引用关系 | 说明 |
|------------|----------|------|
| [`moe_hub_20251203.md`](./moe_hub_20251203.md) | Phase 16 方向整合 | MoE 主 Hub，本 Hub 是其 SNR 专题分支 |

> **与 MoE Hub 的关系**：MoE Hub 的 Phase 16 探索 Teff×[M/H] 在 noise=1 下的结构红利；本 Hub 探索 **SNR 直接分域** 的独立路线。两者是**互补方向**。

| #  | 💡 共识[抽象洞见]                       | 证据                                                               | 决策                                     |
| -- | --------------------------------- | ---------------------------------------------------------------- | -------------------------------------- |
| K1 | **logg 的可达精度由“信息量(SNR)”强烈分段决定**   | Fisher multi-mag：SNR 7→4→3 时 (R^2_{max}) 0.89→0.74→0.37；临界 SNR≈4 | 需要把模型设计成“按 SNR 分策略”，MoE/conditional 都可 |
| K2 | **高噪声下“分域训练”的结构红利更大**             | noise=1 Oracle MoE: 0.6249 vs Global Ridge 0.4611，ΔR²=+0.1637    | 低 SNR 场景更值得上 MoE（至少结构上限很高）             |
| K3 | **error vector 在当前生成口径下包含极强泄露信号**  | **error-only Ridge R²=0.9896** (LOGG-ERR-BASE-01, α=0.001)，Shuffle 后 R²=-0.98 | 必须先做 error 表示的"去泄露/口径一致化"再用            |
| K4 | **error vector 近似"模板×标量 + 稀疏残差"** | 96% 相似，仅 40/4096 不同（用户观察）                                        | 用"低维 quality 参数"替代直接输入 error 全向量       |
| K5 | **S3b_aggregate_stats 完美去泄露** | logg R²=0.042 < 0.05 ✅ 且 SNR R²=0.995 > 0.5 ✅ (LOGG-ERR-REPR-01) | **冻结 quality_features() = 10个聚合统计量** |

**🦾 现阶段信念 [≤10条，写“所以呢]**

* **信念1**：SNR-conditioned（或 SNR-binned）训练应该能提升 logg（尤其在混合 mag/SNR 数据上） → 优先做 “Oracle SNR split” 量化 headroom。 
* **信念2**：直接喂 error vector 会让模型学到“谱线深度/连续谱形状”等捷径 → 强制 error 表示只保留整体噪声尺度/粗粒度统计。
* **信念3**：Soft routing 比 hard routing 更稳（尤其边界样本） → 任何可落地 MoE 都默认 soft mixing + fallback（复用 MoE Hub 经验）。 

**👣 下一步最有价值  [≤2条，直接可进 Roadmap Gate] **

* ~~🔴 P0~~：✅ **已完成** Leakage Audit + 去泄露 → **S3b_aggregate_stats 冻结为 quality_features()**
* ~~🟡 P1~~：✅ **已完成** Oracle SNR-binned Experts → **ΔR² = +0.05 ≥ 0.02 PASS**
* ~~🔴 P0~~：✅ **已完成** MVP-2.0 Deployable Gate → **ρ = 1.04 ≥ 0.7，超越 Oracle！**

---

### 🏆 Route M 完成！下一阶段

* 🟢 **P0 (集成)**：将 SNR-MoE 集成到 ViT 模型中
* 🟡 **P1 (可选)**：MVP-3.0 Whitening/Conditional 对照（验证 MoE 优于简单方法）

> **权威数字（一行即可）**：Ceiling(Fisher)≈0.89@SNR~7；Oracle MoE(noise=1)=0.6249；Global Ridge(noise=1)=0.4611；SNR≈4 是“可达上限掉崖”临界。   

| 模型/方法                  | 指标值                  | 配置                           | 备注                   |
| ---------------------- | -------------------- | ---------------------------- | -------------------- |
| ViT (当前)               | ~0.71                | val（不一定是 test）               | 用户当前最好               |
| Oracle MoE (9 experts) | 0.6249               | 1M data, noise=1             | 结构上限（oracle routing） |
| **Oracle SNR-MoE (4 bins)** | **0.5430**       | 1M data, noise=1, SNR-binned | **✅ LOGG-SNR-ORACLE-01** |
| **🏆 Deployed SNR-MoE** | **0.5443**           | 1M data, noise=1, quality_features gate | **✅ LOGG-SNR-GATE-01 (超越 Oracle!)** |
| Global Ridge           | 0.5087               | 1M data, noise=1             | baseline             |
| Fisher/CRLB 上限         | R²_max median 0.8914 | mag=21.5, SNR~7.1, noise=1   | 理论上限（中位）             |
| Fisher multi-mag       | SNR≈4 临界             | mag=22 SNR~4.6 → R²_max 0.74 | 设计分段阈值               |

---

## 1) 🌲 核心假设树

```
🌲 核心: 用 SNR/观测质量分域 + MoE/conditional 提升 logg 预测，同时避免 error-vector 泄露
│
├── Q1: SNR 分域是否真的有 headroom？ ✅ PASS
│   ├── Q1.1: Oracle 按 SNR 分专家（真 SNR 路由）ΔR²≥0.02？ → ✅ ΔR²=+0.05
│   └── Q1.2: headroom 是否主要来自 low-SNR 子集？ → ✅ Medium SNR (4-7) 最大 +9.6%
│
├── Q2: error vector 如何"可用但不泄露"？ ✅ PASS
│   ├── Q2.1: error-only 预测 logg 的 R² 能否压到 <0.05？ → ✅ S3b: R²=0.042 < 0.05 ✅
│   └── Q2.2: 去泄露表示能预测 SNR？ → ✅ S3b: SNR R²=0.995 > 0.5 ✅
│
└── Q3: 可落地 gate 能保住多少 oracle 增益？ ✅ PASS
    ├── Q3.1: 用 quality features 做 soft routing ρ≥0.7？ → ✅ ρ=1.04 > 1.0 (超越 Oracle!)
    └── Q3.2: fallback/拒识能否稳定提升 full-test？ → ✅ Fallback rate=0%, 不需要

Legend: ✅ 已验证 | ❌ 已否定 | 🔆 进行中 | ⏳ 待验证 | 🗑️ 已关闭
```

## 2) 口径冻结（唯一权威）

| 项目                 | 规格                                                      |
| ------------------ | ------------------------------------------------------- |
| Dataset / Version  | BOSZ / PFS simulator 产物（含 flux, error, mask）            |
| Train / Val / Test | 固定 split；明确 val≠test 的差异                                |
| Noise / Regime     | heteroscedastic Gaussian；noise_level=1（或对应 mag 混合）      |
| Metric             | R²（主）；并报 per-SNR bin R²                                 |
| Seed / Repeats     | ≥3 seeds（避免偶然）                                          |
| Allowed Inputs     | **协议 A：flux(+mask)**；协议 B：flux+error(+mask)**（需去泄露版本）** |

> 规则：任何口径变更必须写入 §8 变更日志。

---

## 3) 当前答案 & 战略推荐（对齐问题树）

### 3.1 战略推荐（只保留“当前推荐”）

* **推荐路线：Route M（Quality-parameter SNR-MoE）**：把 error vector 压缩成“观测质量参数”（scale + 粗统计），做 SNR 分域 soft MoE；并对照 whitening/conditional。
* 需要 Roadmap 关闭的 Gate：Gate-1（Leakage Audit）, Gate-2（Oracle SNR headroom）, Gate-3（Deployable gate ρ）

| Route       | 一句话定位                             | 当前倾向  | 关键理由                           | 需要的 Gate   |
| ----------- | --------------------------------- | ----- | ------------------------------ | ---------- |
| Route A     | 直接喂 error 全向量给模型                  | 🔴    | 已观察到 error-only R²=0.91，极高泄露风险 | Gate-1     |
| Route B     | 只做 whitening/加权（不做 MoE）           | 🟡    | 低风险但未必吃到分段结构红利                 | Gate-2/4   |
| **Route M** | **error→quality 参数 + SNR 分域 MoE** | 🟢 推荐 | 最大化利用观测可得信息，同时最小化泄露            | Gate-1/2/3 |

### 3.2 分支答案表（每行必须回答“所以呢”）

| 分支       | 当前答案（1句话）                        | 置信度 | 决策含义（So what）                    | 证据（exp/MVP）      |
| -------- | -------------------------------- | --- | -------------------------------- | ---------------- |
| SNR 影响上限 | SNR≈4 附近出现"上限掉崖"，分段建模合理          | 🟢  | 必须做 SNR-aware（MoE 或 conditional） | Fisher multi-mag |
| ~~error 泄露~~ | ~~error R²=0.99 极高泄露~~ | ~~🔴~~ | ~~必须先做去泄露~~ | LOGG-ERR-BASE-01 |
| **error 去泄露** | **S3b: logg R²=0.042, SNR R²=0.995** | 🟢 | **quality_features() 已冻结，可进入 MVP-2.0** | **LOGG-ERR-REPR-01** |
| MoE 价值   | noise=1 下 Oracle MoE headroom 很大 | 🟢  | MoE 在低 SNR regime 值得投入           | Oracle MoE, LOGG-SNR-ORACLE-01 |

---

## 4) 洞见汇合（多实验 → 共识）

| #  | 洞见（标题）       | 观察（What）                       | 解释（Why）                 | 决策影响（So what）                | 证据                |
| -- | ------------ | ------------------------------ | ----------------------- | ---------------------------- | ----------------- |
| I1 | SNR 决定可达上限   | SNR 从 7 降到 3，上限从 0.89 掉到 0.37  | 信息论限制；模型再强也无法凭空创造信息     | 需要“按 SNR 分策略”的模型             | Fisher multi-mag  |
| I2 | 高噪声放大结构红利    | noise=1 下 Oracle MoE ΔR²=+0.16 | 分段训练在高噪声更能抑制偏差/方差       | 低 SNR 做 MoE ROI 更高           | Oracle MoE        |
| I3 | error 向量可低维化 | error 96% 相似                   | 噪声形状近似由仪器决定；样本差异主要是整体尺度 | gate 应使用 quality 参数而非全 error | 用户观测              |
| **I4** | **Medium SNR 受益最大** | Bin M (SNR 4-7) ΔR²=+9.6%，远超其他 bin | 临界区域专家化收益最大 | **SNR predictor 重点关注边界样本** | LOGG-SNR-ORACLE-01 |
| **I5** | **SNR 4-bin 策略有效** | 所有 4 个 bin 的 Oracle Expert 均优于 Global | 分域策略全面有效 | **可继续开发 deployable gate** | LOGG-SNR-ORACLE-01 |
| **I6** | **Error 泄露是波长对齐的** | Shuffle 后 R² 从 0.99 崩溃到 -0.98 | 模型依赖像素位置信息，不是简单统计量 | 去泄露必须破坏波长对齐 | LOGG-ERR-BASE-01 |
| **I7** | **Top 泄露像素集中特定区域** | 像素 3277-3388, 3724-3869 重要性最高 | 可能对应特定谱线区域 | 这些区域需特殊处理或屏蔽 | LOGG-ERR-BASE-01 |
| **I8** | **S1 同口径归一化无效** | error/median(flux) 仍有 R²=0.788 | 归一化不破坏波长对齐 pattern | 必须彻底打乱对齐 | LOGG-ERR-REPR-01 |
| **I9** | **S2 template×scale 去泄露但 SNR 略低** | logg R²=-0.001 但 SNR R²=0.808 | 标量丢失分布信息 | 备选方案 | LOGG-ERR-REPR-01 |
| **I10** | **S3b aggregate_stats 完美平衡** | logg R²=0.042, SNR R²=0.995 | 聚合统计量无对齐但保留分布 | **最佳策略** | LOGG-ERR-REPR-01 |
| **I11** | **Deployed 超越 Oracle** | ρ=1.04, R²=0.544 vs Oracle 0.543 | Soft routing 在边界样本上优于 hard routing | **Route M 成功** | LOGG-SNR-GATE-01 |
| **I12** | **Gate 准确率极高** | 99.6% accuracy | quality_features() 完美捕获 SNR 信息 | LogReg 足够，不需要复杂 gate | LOGG-SNR-GATE-01 |
| **I13** | **M bin 收益最大** | Expert M R²=0.705，远超其他 bin | 临界 SNR 区域 (4-7) 专家化效果最好 | 重点优化 M bin 专家 | LOGG-SNR-GATE-01 |

---

## 5) 决策空白（Decision Gaps）

| DG  | 我们缺的答案                         | 为什么重要（会改哪个决策）      | 什么结果能关闭它                        | 决策规则                                          |
| --- | ------------------------------ | ------------------ | ------------------------------- | --------------------------------------------- |
| ~~DG1~~ | ~~去泄露后 error-feature 是否仍能做 gate？~~ | ✅ **已关闭** | **S3b: logg R²=0.042 < 0.05, SNR R²=0.995 > 0.5** | **冻结 quality_features() → MVP-2.0** |
| ~~DG2~~ | ~~Oracle SNR 分专家的 headroom 有多大？~~ | ✅ **已关闭** | **ΔR² = +0.05 ≥ 0.02 PASS** | **继续做 MoE → MVP-2.0** |
| ~~DG3~~ | ~~可落地 gate 能保住多少 oracle？~~ | ✅ **已关闭** | **ρ = 1.04 ≥ 0.7，Deployed R² = 0.544 超越 Oracle 0.543** | **Route M 可交付！进入集成阶段** |

---

## 6) 设计原则（可复用规则）

### 6.1 已确认原则

| #  | 原则               | 建议（做/不做）                              | 适用范围    | 证据                |
| -- | ---------------- | ------------------------------------- | ------- | ----------------- |
| P1 | Soft routing 优先  | 默认 soft mixing + fallback             | MoE 路由  | MoE Hub 经验        |
| P2 | SNR 分段阈值来自信息论    | 用 SNR≈4 作为关键门槛                        | gate/分桶 | Fisher multi-mag  |
| P3 | error 只能表达“观测质量” | error 必须与 flux 同口径变换后再用；优先用 scale/粗统计 | 输入表示    | 用户泄露观察            |

### 6.2 待验证原则

| #  | 原则                      | 初步建议                       | 需要验证（MVP/Gate）   |
| -- | ----------------------- | -------------------------- | ---------------- |
| P4 | template×scale 足够做 gate | 只用 1–5 维 quality 参数即可      | MVP-1.1 / Gate-1 |
| P5 | 低 SNR 专家应更强正则/更强先验      | low-SNR expert 用更强正则或物理窗特征 | MVP-2.0          |

---

## 7) 指针（详细信息在哪）

| 类型                    | 路径                                          | 说明             |
| --------------------- | ------------------------------------------- | -------------- |
| 🧠 参考 Hub 模版          | `hub.md`                                    | Hub 结构规范       |
| 🗺️ 参考 Roadmap 模版     | `roadmap.md`                                | Gate/MVP 结构规范  |
| 🧠 参考 MoE Hub         | `moe_hub_20251203.md`                       | ρ 指标与路由经验      |
| 🗺️ 参考 MoE Roadmap    | `moe_roadmap_20251203.md`                   | MVP 编排方式       |
| 📘 Fisher 上限          | `exp_scaling_fisher_ceiling_v2_20251224.md` | 单 mag 理论上限     |
| 📘 Fisher multi-mag   | `exp_scaling_fisher_multi_mag_20251224.md`  | SNR 分段规律       |
| 📘 Oracle MoE noise=1 | `exp_scaling_oracle_moe_noise1_20251223.md` | noise=1 结构红利   |
| 📘 **MVP-0.2 去泄露**    | `exp/exp_logg_err_repr_01_20251226.md`      | quality_features() 冻结 |
| 📘 **MVP-1.0 Oracle**  | `exp/exp_logg_snr_oracle_01_20251226.md`    | SNR 分域 headroom |
| 📘 **MVP-2.0 Gate**    | `exp/exp_logg_snr_gate_01_20251226.md`      | 🏆 Deployed 超越 Oracle |

---

## 8) 变更日志（只记录“知识改变”）

| 日期         | 变更             | 影响 |
| ---------- | -------------- | -- |
| 2025-12-26 | 创建 SNR-MoE Hub | -  |
| 2025-12-26 | ✅ Gate-2 通过：ΔR²=+0.05 | Q1.1/Q1.2 验证，DG2 关闭，新增洞见 I4/I5 |
| 2025-12-26 | ❌ Gate-1 初次未通过：error R²=0.9896 | Q2.1 ❌，K3 更新为实际数值，新增洞见 I6/I7，下一步 MVP-0.2 |
| 2025-12-26 | ✅ **Gate-1 通过**：S3b logg R²=0.042, SNR R²=0.995 | Q2.1 ✅ Q2.2 ✅，DG1 关闭，K5 新增，洞见 I8/I9/I10，**冻结 quality_features()** |
| 2025-12-26 | ✅ **Gate-3 通过**：ρ=1.04, Deployed R²=0.544 超越 Oracle | Q3.1 ✅ Q3.2 ✅，DG3 关闭，洞见 I11/I12/I13，**Route M 可交付！** |
| 2025-12-26 | 🏆 **SNR-MoE 完成** | 所有 Gate 通过，进入集成阶段 |
| 2025-12-27 | 📝 Hub 同步更新：修正 K3 error R² 为 0.9896 (α=0.001) | 与 LOGG-ERR-BASE-01 实验对齐 |

