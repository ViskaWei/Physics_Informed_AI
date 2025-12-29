# 🗺️ MoE → Theoretical Ceiling (~0.8) Roadmap
> **Name:** MoE to Ceiling | **ID:** `VIT-20251228-moe-to-ceiling-roadmap`  
> **Topic:** `moe` | **Phase:** Ceiling | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-28 | **Status:** 🔄 进行中
```
💡 核心目标  
noise=1 regime 下，把 MoE 推到 R²≈0.78~0.80（接近理论上线）
策略：先抬 Oracle（专家变强）→ 再用 soft 回归 gate 保住 ρ
```

| 当前进度 | 当前最佳 | 目标 |
|---------|----------|------|
| Phys-only Gate | R²=0.601 (ρ=0.84) | R²≥0.78 (逼近 ceiling 0.87) |

---

# 0. 🔍 口径澄清

## 0.1 两种"Oracle"的区分

| 类型 | 定义 | 当前数值 | 问题 |
|------|------|---------|------|
| **物理 Oracle** | Teff×[M/H] 9-bin, Ridge experts | 0.627 (test=10k) | "按物理域分段，结构红利有多大？" |
| **SNR Oracle** | 4-bin SNR experts | ΔR²≈+0.05 | "按信息量分段，headroom 有多大？" |

> **不矛盾**：物理 Oracle 解释"光谱长什么样"，SNR Oracle 解释"能看清多少"

## 0.2 test=1k vs test=10k 差异

| 指标 | test=1k | test=10k | 差异原因 |
|------|---------|----------|---------|
| Global Ridge | 0.46 | **0.4957** | 抽样抖动 |
| Oracle MoE | 0.62 | **0.627** | 更稳定估计 |

> **结论**：冲 0.8 路线图，主评估集固定 **test=10k**（减少抽样抖动误判）

## 0.3 主矛盾识别

| 指标 | 数值 | 分析 |
|------|------|------|
| Global Ridge | 0.4957 | baseline |
| Phys-only Gate | 0.601 | deployable |
| Oracle (9×Ridge) | 0.627 | 当前 oracle 上限 |
| Gap (Gate vs Oracle) | 0.024 | **gate 已经很接近 oracle！** |
| 5D Fisher Ceiling | 0.87 | 理论上限 |

> **核心判断**：主矛盾不在 gate，而在 **Oracle 太低 (0.627 << 0.87)**  
> **战略**：把 Oracle 从 "Ridge-Oracle ~0.62" 抬到 "强专家 Oracle ~0.78/0.80"

---

# 1. 🚦 Decision Gates

## 1.1 战略路线

| Route | 名称 | 核心思路 | 状态 |
|-------|------|---------|------|
| **C0** | 口径冻结 | 固定 test=10k + 统一 eval 协议 | 🔴 首先 |
| **C1** | Oracle Uplift | 把专家上限抬到 ≥0.75 | 🔴 P0 |
| **C2** | Deployable ρ | oracle 抬起来后 gate 跟上 | 🟡 P1 |
| **C3** | Shared-Trunk MoE | ViT-trunk + K heads | ⏳ 视 C1 结果 |
| **C4** | 冲刺 0.8 | Residual-MoE + 极端区域处理 | ⏳ 视 C3 结果 |

## 1.2 Gate 定义

### Gate-C0: 评估口径冻结

| 项 | 内容 |
|----|------|
| 验证 | 把评估尺子焊死，避免数字漂移误导路线 |
| MVP | MVP-C0.0 |
| Pass | Ridge 在 10k 上稳定 ~0.50；所有后续用同一 eval 入口 |
| 状态 | 🔴 就绪 |

### Gate-C1: Oracle Uplift (≥0.75)

| 项 | 内容 |
|----|------|
| 验证 | 能否把 oracle 从 0.627 抬到 ≥0.75？ |
| MVP | MVP-C1.0, MVP-C1.1 |
| 若 A | Oracle-Hybrid ≥ 0.70 → 继续 C2 保 ρ |
| 若 B | Oracle-Hybrid < 0.70 → 进入 C3 (共享 trunk) |
| 状态 | ⏳ |

### Gate-C2: Deployable ρ ≥ 0.85

| 项 | 内容 |
|----|------|
| 验证 | Hybrid experts + Soft gate 能保住多少 oracle 增益？ |
| MVP | MVP-C2.0, MVP-C2.1 |
| 若 A | ρ ≥ 0.85 → 继续 C4 冲刺 |
| 若 B | ρ < 0.70 → 回到 C1 优化 gate 输入 |
| 状态 | ⏳ |

### Gate-C3: Shared-Trunk MoE ≥ ViT

| 项 | 内容 |
|----|------|
| 验证 | 共享 ViT trunk + K heads 能否超过单 ViT？ |
| MVP | MVP-C3.0 |
| Level-1 | ≥ ViT (≥0.70) |
| Level-2 | ≥ 0.78 (明显逼近 ceiling) |
| 状态 | ⏳ 依赖 C1 结果 |

### Gate-C4: 最后冲刺 ≥0.78

| 项 | 内容 |
|----|------|
| 验证 | 最后 0.02~0.03 的冲刺能否到达？ |
| MVP | MVP-C4.0 |
| Pass | overall ≥ 0.78 且 metal-poor / SNR≤4 不退步 |
| 状态 | ⏳ 依赖 C3 结果 |

## 1.3 本周重点

| 优先级 | MVP | Gate | 状态 |
|--------|-----|------|------|
| 🔴 P0 | MVP-C0.0 | Gate-C0 | 🔴 就绪 |
| 🔴 P0 | MVP-C1.0 | Gate-C1 | ⏳ |
| 🟡 P1 | MVP-C1.1 | Gate-C1 | ⏳ |
| 🟡 P1 | MVP-C2.0 | Gate-C2 | ⏳ |

---

# 2. 📋 MVP 列表

## 2.1 总览

| MVP | 名称 | Gate | 状态 | exp_id | 报告 |
|-----|------|------|------|--------|------|
| C0.0 | 评估协议冻结 | Gate-C0 | 🔴 就绪 | - | - |
| C1.0 | Per-bin Expert Sweep | Gate-C1 | ❌ FAIL | `MOE-CEILING-C1-01` | [Report](./exp/exp_moe_ceiling_expert_sweep_20251228.md) |
| C1.1 | Metal-poor Rescue | Gate-C1 | ⏳ | - | - |
| C2.0 | Regress-Gate Hybrid | Gate-C2 | ⏳ | - | - |
| C2.1 | SNR 条件性融合 | Gate-C2 | ⏳ | - | - |
| C3.0 | ViT-trunk + K heads | Gate-C3 | ⏳ | - | - |
| C4.0 | Residual-MoE | Gate-C4 | ⏳ | - | - |

**状态**: ⏳计划 | 🔴就绪 | 🚀运行 | ✅完成 | ❌取消

## 2.2 配置速查

| MVP | 数据量 | 噪声 | 专家类型 | 关键变量 |
|-----|--------|------|----------|---------|
| C0.0 | 1M | 1.0 | Ridge | test=10k 固定 |
| C1.0 | 1M | 1.0 | Ridge/LGBM/CNN | 每 bin 最优专家 |
| C1.1 | 1M | 1.0 | LGBM/小 CNN | Bin3/Bin6 专项 |
| C2.0 | 1M | 1.0 | Hybrid | MLP regress gate |
| C2.1 | 1M | 1.0 | Hybrid | 条件性 quality 融合 |
| C3.0 | 1M | 1.0 | ViT-trunk + heads | 共享表示 |
| C4.0 | 1M | 1.0 | ViT + MoE residual | 残差学习 |

---

# 3. 🔧 MVP 规格

## Phase C0: 评估口径冻结

### MVP-C0.0: 评估协议固定

| 项 | 配置 |
|----|------|
| 目标 | 冻结评估尺子，避免数字漂移误导路线选择 |
| 数据 | 1M train, noise=1.0 |
| 测试集 | **10k**（主） + 1k（兼容 canonical，可选） |
| 输出指标 | overall R² + per-(SNR bin) + per-(Teff×[M/H] bin) |
| 额外指标 | efficiency = R² / R²_ceiling(SNR)（见 Fisher Hub） |
| Pass | Ridge baseline 在 10k 上稳定 ~0.50 |

**产物**：
- [ ] `logg/moe/eval_protocol_frozen.md` — 评估协议文档
- [ ] 统一 eval 入口脚本

---

## Phase C1: Oracle Uplift (目标 ≥0.75)

> **核心任务**：把"专家上限"从 Ridge-Oracle 0.627 抬到 ≥0.75

### MVP-C1.0: Per-bin Expert Sweep（离线 oracle）

| 项 | 配置 |
|----|------|
| 目标 | 找每个 bin 的最优专家，组合成 Oracle-Hybrid |
| Binning | 9 物理 bin (Teff×[M/H]) |
| 专家候选 | Ridge（α per-bin sweep）、LightGBM、Whitened Ridge、小 1D-CNN |
| 路由 | Oracle (真值分配) |
| 重点盯 | metal-poor bins (Bin3/Bin6) |

**输出**：
- 每个 bin 的 best expert & R²
- Oracle-Hybrid 整体 R²
- metal-poor vs 其他区域对比

**Pass 条件**：
- Oracle-Hybrid 比 Ridge-Oracle 提升 **≥ +0.05**
- 或 Oracle-Hybrid **≥ 0.70**（追平 ViT）

### MVP-C1.1: Metal-poor Rescue Package

| 项 | 配置 |
|----|------|
| 目标 | 专门抬起 Bin3/Bin6（最差 bins） |
| 专家 | 非线性（LGBM / 小 CNN） |
| 输入方案 1 | raw flux（保持一致） |
| 输入方案 2 | 物理窗/线指数增强版（CaT/Na/PCA 作为 stacking feature） |

**Pass 条件**：
- 最差 bin 的 R² 提升 **≥ +0.05**
- overall oracle 被抬高

**已有证据**（可复用）：
- MVP-15：Bin3 LGBM +0.056 ✅，Bin6 LGBM -0.032 ❌
- 结论：Bin3 可用 LGBM，Bin6 需要其他方案

---

## Phase C2: Deployable 保 ρ

> **核心任务**：oracle 抬起来后，gate 能不能跟上？

### MVP-C2.0: Regress-Gate on Hybrid Experts

| 项 | 配置 |
|----|------|
| 目标 | 验证 Hybrid experts + Soft gate 的 ρ |
| Experts | Gate-C1 选出的 Hybrid（不同 bin 允许不同模型） |
| Gate | MLP regress gate（直接最小化 final MSE） |
| Routing | Soft mixing（必选） |

**验收指标**：
- ρ_hybrid = (R²_deploy − R²_global) / (R²_oracle-hybrid − R²_global)
- **Pass: ρ ≥ 0.85**

**依据**：当前 gate 表现已经接近 oracle (0.601 vs 0.627)

### MVP-C2.1: SNR 条件性融合

| 项 | 配置 |
|----|------|
| 目标 | 只在低 SNR 区域引入 quality 信息 |
| 策略 | 仅当 gate 判断落在 X bin (SNR≤2) 时，引入 quality tower 或温度控制 |
| 其他区间 | 保持 phys gate |

**依据**：
- 双塔实验：整体 ΔR²=+0.0017 微弱
- 但 X-bin (SNR≤2) 确实 +0.011
- **不做全局双塔，只吃 X bin 正收益**

---

## Phase C3: Shared-Trunk MoE Heads

> **触发条件**：如果 Gate-C1 抬 oracle 还不够（卡在 0.72~0.75）

### MVP-C3.0: ViT-trunk + K heads（Mixture-of-Heads）

| 项 | 配置 |
|----|------|
| 目标 | 用共享 trunk 代替 K 个独立 ViT（省资源） |
| Trunk | 现有 ViT encoder（或 1D-CNN encoder） |
| Experts | K 个小 head（线性/两层 MLP），每个专注一个物理 bin |
| 特殊 head | "metal-poor 专家"（功能性 bin） |
| Gate | 物理窗特征（可拼 trunk embedding，先从稳的开始） |
| 辅助任务 | Teff、[M/H] 预测（帮助表示更物理一致） |

**Pass 条件**：
- Level-1：≥ ViT（≥ 0.70）
- Level-2：明显逼近 ceiling（≥ 0.78）

**核心 insight**：不训练 9 个 ViT 专家（太贵），而是 1 trunk + 多 heads

---

## Phase C4: 冲刺 0.8

> **触发条件**：已经 0.75+ 时

### MVP-C4.0: Residual-MoE

| 项 | 配置 |
|----|------|
| 目标 | 最后 0.02~0.03 的冲刺 |
| 主干 | 强模型（ViT 或 ViT-MoE） |
| MoE 角色 | 只学残差修正项 y = y_strong + Σ w_k · r_k |
| 残差专家 | 轻模型（Ridge/LGBM），输入用"物理/线指数" |
| 专注区域 | "强模型没学好的那 20%"（metal-poor, SNR≤4） |

**Pass 条件**：
- overall ≥ 0.78
- metal-poor / SNR≤4 子集不退步

---

# 4. 📊 进度追踪

## 4.1 看板

```
⏳计划      🔴就绪      🚀运行      ❌失败      ✅完成
MVP-C1.1    MVP-C0.0                MVP-C1.0
MVP-C2.0    MVP-C3.0                            
MVP-C2.1                            
MVP-C4.0                            
```

## 4.2 Gate 进度

| Gate | MVP | 状态 | 结果 |
|------|-----|------|------|
| Gate-C0 | MVP-C0.0 | 🔴 就绪 | - |
| Gate-C1 | MVP-C1.0, C1.1 | ❌ C1.0 FAIL | Oracle-Hybrid=0.6436 < Ridge-Oracle=0.6666 |
| Gate-C2 | MVP-C2.0, C2.1 | ⏳ | - |
| Gate-C3 | MVP-C3.0 | 🔴 就绪 | 依赖 C1 结果 → 直接进入 |
| Gate-C4 | MVP-C4.0 | ⏳ | - |

## 4.3 结论快照

| MVP | 结论 | 关键指标 | 同步Hub |
|-----|------|---------|---------|
| C1.0 | ❌ Hybrid 反而更差 | Oracle-Hybrid=0.6436 < Ridge-Oracle=0.6666 | ✅ |
| C1.0 | Ridge 全面碾压 LGBM | 8/9 bins Ridge 最优 | ✅ |
| C1.0 | 唯一亮点 Bin3 | LGBM +0.059 vs Ridge | ✅ |
| C1.0 | 1D-CNN 失败 | Bin3=0.13, Bin6=0.36 | ✅ |

## 4.4 时间线

| 日期 | 事件 |
|------|------|
| 2025-12-28 | Roadmap 创建 |
| 2025-12-28 | MVP-C1.0 完成 ❌ FAIL: Oracle-Hybrid < Ridge-Oracle |
| 2025-12-28 | 决策: 跳过 C1.1/C2，直接进入 MVP-C3.0 (共享 trunk) |

---

# 5. 🎯 关键取舍

## 5.1 不做什么

| 不做 | 原因 | 证据 |
|------|------|------|
| ❌ 继续投复杂 gate | gate 已接近 oracle (0.601 vs 0.627) | LOGG-DUAL-TOWER-01 |
| ❌ 全局双塔融合 | 增益微弱 ΔR²=+0.0017 | LOGG-DUAL-TOWER-01 |
| ❌ 训练 9 个独立 ViT | 太贵，用共享 trunk 替代 | 资源约束 |

## 5.2 集中做什么

| 做 | 原因 | 预期收益 |
|----|------|---------|
| ✅ 抬 Oracle（bin 内专家能力） | 主矛盾在专家太弱 | +0.10~0.15 |
| ✅ Metal-poor rescue | 拖累 overall 的瓶颈 | +0.05 per bin |
| ✅ 共享 trunk + 多 heads | 省资源逼近 ViT 天花板 | ≥0.70 |
| ✅ SNR 条件性使用 quality | 只吃 X bin 正收益 | +0.01 in X bin |

## 5.3 Hub 共识复用

| 共识 | 应用到本 Roadmap |
|------|-----------------|
| K3: Soft routing 是落地关键 | C2 必须用 Soft |
| K5: Metal-poor 是瓶颈 | C1.1 专项救援 |
| I7: Hybrid MoE 可行 | C1.0 不同 bin 用不同专家 |
| I9: SNR≈4 临界点 | C4 极端区域特殊处理 |
| I12: 极低 SNR 是唯一受益区 | C2.1 条件性融合 |

---

# 6. 📎 附录

## 6.1 数值汇总

| 模型 | R² | 配置 | 备注 |
|------|-----|------|------|
| Global Ridge | 0.4957 | noise=1, 1M, test=10k | baseline |
| Phys-only Gate | 0.601 | noise=1, 1M, ρ=0.84 | 当前 deployable |
| Oracle (9×Ridge) | **0.6666** | noise=1, 1M, test=10k | C1.0 测量值 |
| Oracle-Hybrid | 0.6436 | noise=1, 1M, test=10k | C1.0 ❌ 反而更差 |
| ViT (参考) | ~0.70 | noise=1 | 目标超越 |
| 5D Fisher Ceiling | 0.87 | 理论上限 | 目标逼近 |

## 6.2 目标里程碑

| 里程碑 | R² | 验证 Gate |
|--------|-----|----------|
| Oracle-Hybrid | ≥ 0.70 | Gate-C1 |
| Deployable-Hybrid | ≥ 0.65 (ρ≥0.85) | Gate-C2 |
| MoE-Heads | ≥ 0.75 | Gate-C3 |
| Final | ≥ 0.78 | Gate-C4 |

## 6.3 文件索引

| 类型 | 路径 |
|------|------|
| 本 Roadmap | `moe_to_ceiling_roadmap_20251228.md` |
| MoE Hub | `moe_hub_20251203.md` |
| SNR-MoE Hub | `moe_snr_hub.md` |
| Fisher Hub | `../scaling/fisher_hub_20251225.md` |
| 双塔实验 | `./exp/exp_moe_dual_tower_20251228.md` |

## 6.4 更新日志

| 日期 | 变更 | 章节 |
|------|------|------|
| 2025-12-28 | 创建：口径澄清、5 Gate 体系、7 MVP 规格 | 全文 |
| 2025-12-28 | MVP-C1.0 完成 ❌ FAIL | §2.1, §4, §6.1 |

---

*Created: 2025-12-28*
*Goal: MoE R² 0.6 → 0.7 → 0.8 (逼近理论上限 0.87)*
