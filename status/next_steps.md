# 📋 下一步计划（Next Steps）

---
> **最后更新：** 2025-12-28  
> **当前焦点：** MoE 双塔融合 (Phase 4)

---

# 🔴 P0 — 高优先级（今天/明天）

| # | 任务 | 对应实验 | 状态 | 备注 |
|---|------|---------|------|------|
| 🆕 | **MoE 双塔融合 MVP-4.0** | `VIT-20251228-moe-dual-tower-01` | 🔴 就绪 | 9 experts + gate concat [phys,qual] + MSE |
| 1 | 完成 GTA 双塔融合实验 | VIT-20251201-gta-fusion-01 | ⬜ 待做 | Global + Local concat/FiLM |
| 2 | 提取 BlindSpot Latent 特征给 GTA | BS-20251201-latent-gta-01 | ⬜ 待做 | 增强 noise=1.0 性能 |

---

# 🟡 P1 — 中优先级（本周内）

| # | 任务 | 对应实验 | 状态 | 备注 |
|---|------|---------|------|------|
| 1 | 测试 multi-scale dilation CNN | - | ⬜ 待做 | 基于 dilation=2 最优结论 |
| 2 | F2/F3 特征验证 (统计量/EW) | - | ⬜ 待做 | 测试贡献度 |
| 3 | 为 CNN dilated 实验写 knowledge card | - | ⬜ 待做 | 提炼核心结论 |

---

# 🟢 P2 — 低优先级（后续考虑）

| # | 任务 | 对应实验 | 状态 | 备注 |
|---|------|---------|------|------|
| 1 | 扩展到 100k 数据集 | - | ⬜ 待做 | 测试 scaling |
| 2 | Swin Transformer vs CNN 系统对比 | - | ⬜ 待做 | 不同噪声水平 |

---

# ✅ 已完成

| # | 任务 | 完成日期 | 成果 |
|---|------|---------|------|
| 1 | GTA Phase 1-3 完成 | 2025-12-01 | Global $R^2$=0.96, Local $R^2$=0.93 |
| 2 | CNN dilated kernel sweep | 2025-12-01 | dilation=2 最优 |
| 3 | Latent extraction 实验 | 2025-12-01 | $R^2$=0.598 |

---


