# 🧠 Fisher Hub: 理论上限与信息诊断
> **Topic:** `scaling/fisher` | **Type:** Mini-Hub  
> **Author:** Viska Wei | **Created:** 2025-12-25 | **Last Updated:** 2025-12-25  
> **Status:** 🟢 Stable | **Confidence:** ✅ High (V2 + Multi-Mag 验证)

---

## ⚡ Answer Key (前 30 行核心)

> **一句话**: Fisher/CRLB 分析确认 noise=1 时理论上限 R²_max = **0.89**，与 LightGBM 0.57 存在 **+32% headroom**；SNR=4 是临界点，Schur≈0.69 表明 degeneracy 中等。

| 核心问题 | 答案 | 置信度 |
|---------|------|--------|
| Q1: 理论上限多高？ | R²_max = 0.89 (median @ noise=1) | 🟢 高 |
| Q2: 与实际差距多大？ | +32% vs LightGBM, +43% vs Ridge | 🟢 高 |
| Q3: Degeneracy 多严重？ | Schur = 0.69 (69% 信息保留) | 🟢 高 |
| Q4: SNR 临界点在哪？ | SNR ≈ 4 (mag ≈ 22) | 🟢 高 |

### 决策含义

| 如果... | 则... |
|---------|-------|
| 上限高 + 实际差距大 (✅ 现状) | **继续投入 CNN/Transformer 值得** |
| 上限接近实际 | 模型已接近极限，需改变目标 |
| Schur < 0.5 | Multi-task 解纠缠优先 |
| Schur > 0.7 (✅ 现状) | Multi-task 可选但非必须 |

### 关键数字速查

| 指标 | 值 | 条件 | 来源 |
|------|-----|------|------|
| R²_max (median) | **0.8914** | noise=1, mag=21.5 | V2 |
| R²_max (90%) | 0.9804 | 高分位 | V2 |
| Gap vs LightGBM | **+0.32** | - | V2 |
| Gap vs Ridge | +0.43 | - | V2 |
| Schur decay | **0.6906** | 恒定 across SNR | Multi-Mag |
| 临界 SNR | **~4** | R²_max > 0.5 边界 | Multi-Mag |
| 信息悬崖 | SNR < 2 | median R²_max = 0 | Multi-Mag |

---

## 📊 Experiment Graph

```
📦 Fisher 实验关系图
═══════════════════════════════════════════════════════════════

exp_fisher_ceiling_20251223 (V1: 失败)
│   ⚠️ 连续采样数据 + KDTree 邻居差分
│   ❌ CRLB 跨 20 数量级，偏导混参污染
│
├─── [fix: 改用规则网格 + 沿轴精确差分] ──►
│
exp_fisher_ceiling_v2_20251224 (V2: 基线成功)
│   ✅ 规则网格 (10×9×14), CRLB 仅 2.9 数量级
│   📊 R²_max = 0.89, Schur = 0.69
│
└─── [extend: 6 个 magnitude sweep] ──►
    │
    exp_fisher_multi_mag_20251224 (扩展: 完成)
        ✅ SNR 从 87 到 1.9 的完整趋势
        📊 临界 SNR ≈ 4, Schur 恒定

        ═══════════════════════════════════════
                     ▼
              fisher_hub (本文档)
              └── 汇合: 理论上限 + SNR 边界 + 设计原则
```

---

## 🔗 Confluence Index (洞见汇合)

| ID | 洞见 | 来源 | 状态 |
|----|------|------|------|
| F-1 | R²_max = 0.89 证明理论上限高，+32% headroom 值得投入 | V2 | ✅ 稳定 |
| F-2 | Schur = 0.69 表明 degeneracy 中等，multi-task 可选但非必须 | V2, Multi-Mag | ✅ 稳定 |
| F-3 | SNR=4 是临界点，mag≥22.5 需要额外策略 | Multi-Mag | ✅ 稳定 |
| F-4 | SNR < 2 存在信息悬崖，但 top 10% 样本仍可估计 | Multi-Mag | ✅ 稳定 |
| F-5 | Schur decay 与 SNR 无关，由光谱物理决定 | Multi-Mag | ✅ 稳定 |
| F-6 | 高 T_eff + 高 log_g 区域 R²_max 更高 | V2 | ⚠️ 待进一步验证 |

---

## ⚠️ 冲突与修正记录

| V1 结论 | V2 修正 | 原因 |
|---------|---------|------|
| R²_max (median) = 0.97 | 0.89 | V1 偏导混参，过高估计 |
| Schur decay = 0.24 | 0.69 | V1 数值不稳定 |
| CRLB range = 20 orders | 2.9 orders | V1 邻近点差分失败 |
| Condition max = 5e+16 | 3.78e+06 | 改善 10 数量级 |

**V1→V2 教训**: 必须检查数据结构假设（规则网格 vs 连续采样），Fisher 计算对偏导估计方法敏感。

---

## 📐 Design Principles (设计原则库)

### 已确认原则

| # | 原则 | 证据 | 适用范围 |
|---|------|------|---------|
| P-F1 | Fisher/CRLB 计算必须使用规则网格数据 | V1 失败 vs V2 成功 | 理论分析 |
| P-F2 | CRLB range 应 < 3 数量级，否则方法有问题 | V1 20 vs V2 2.9 | 数值诊断 |
| P-F3 | 理论上限高 (R²_max ≥ 0.75) 时继续投入值得 | V2: 0.89 | 决策门控 |
| P-F4 | SNR < 4 需要额外策略（多曝光/先验/ensemble） | Multi-Mag | 观测规划 |
| P-F5 | Schur > 0.7 时 Multi-task 非必须 | Schur = 0.69 | 架构选择 |

### 关键数字约束

| 条件 | 阈值 | 决策 |
|------|------|------|
| R²_max (median) | ≥ 0.75 | ✅ 继续投入 |
| Schur decay | > 0.7 | Multi-task 可选 |
| Schur decay | < 0.5 | Multi-task 优先 |
| SNR | ≥ 4 | 正常估计 |
| SNR | < 2 | 信息悬崖警告 |

---

## ➡️ Next Actions (战略导航)

### 基于 Fisher 结论的路线

| 方向 | 任务 | 优先级 | 理由 |
|------|------|--------|------|
| 继续 CNN/Transformer | Phase 2+ | 🔴 P0 | +32% headroom 证明值得 |
| Multi-task 尝试 | 可选 | 🟡 P1 | Schur=0.69，可能有帮助 |
| 区域特化模型 | 高 T_eff + 高 log_g | 🟢 P2 | 参数空间异质性 |
| 暗目标策略 | mag ≥ 22.5 | 🟢 P2 | 临界 SNR=4 边界 |

### 已关闭方向

| 方向 | 原因 |
|------|------|
| 放弃 CNN 投入 | ❌ headroom 大，值得继续 |
| 强制 Multi-task | ❌ Schur=0.69 非必须 |

---

## 📎 附录

### A. 子实验索引

| # | Experiment ID | 文件 | 状态 | 角色 |
|---|--------------|------|------|------|
| 1 | SCALING-20251223-fisher-ceiling-01 | [V1](../exp/exp_scaling_fisher_ceiling_20251223.md) | ❌ Failed | 首次尝试 |
| 2 | SCALING-20251224-fisher-ceiling-02 | [V2](../exp/exp_scaling_fisher_ceiling_v2_20251224.md) | ✅ Done | 修正基线 |
| 3 | SCALING-20251224-fisher-multi-mag | [Multi-Mag](../exp/exp_scaling_fisher_multi_mag_20251224.md) | ✅ Done | 扩展验证 |

### B. 图表索引

| 图表 | 路径 | 说明 |
|------|------|------|
| R²_max Distribution (V2) | `img/fisher_ceiling_v2_r2max_dist.png` | 单 mag 分布 |
| R²_max vs Magnitude | `img/fisher_multi_mag_r2max.png` | SNR 趋势 |
| Schur Decay | `img/fisher_multi_mag_schur.png` | 恒定性验证 |
| Ceiling vs Baseline | `img/fisher_ceiling_v2_vs_baseline.png` | Headroom 对比 |

### C. 代码索引

| 脚本 | 路径 | 说明 |
|------|------|------|
| V2 脚本 | `~/VIT/scripts/scaling_fisher_ceiling_v2.py` | 规则网格版 |
| Multi-Mag 脚本 | `~/VIT/scripts/scaling_fisher_ceiling_v2_multi_mag.py` | 多 mag 扫描 |

### D. 数据索引

| 数据集 | 路径 | 用途 |
|--------|------|------|
| 规则网格 | `/datascope/.../grid/grid_mag215_lowT/` | V2 基线 |
| 多 Mag 网格 | `/datascope/.../grid/grid_mag{18,20,215,22,225,23}_lowT/` | Multi-Mag |
| 数据索引文档 | `data/bosz50k/z0/grid_fisher_datasets.md` | 完整列表 |

---

> **Hub 创建时间**: 2025-12-25  
> **基于实验**: V1 (失败) → V2 (基线) → Multi-Mag (扩展)  
> **置信度**: 🟢 High (规则网格+精确差分方法验证)

