# 📇 Knowledge Card: Fisher/CRLB Theoretical Ceiling
> **Name:** Fisher Ceiling V2 | **ID:** `VIT-20251224-scaling-fisher-card`  
> **Topic:** `scaling` | **Source:** `exp_scaling_fisher_ceiling_v2_20251224.md` | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-24
```
💡 理论上限 R²_max = 0.89 (median)，与 LightGBM 0.57 存在 +32% headroom，证明深度学习投入值得  
适用：评估模型改进潜力
```

---

## 🎯 问题与设置

**问题**: noise=1 时，任何模型能达到的理论上限 R²_max 是多少？

**设置**: 
- 数据: BOSZ 规则网格 (10×9×14), 1,260 samples
- 方法: Fisher Information / CRLB 理论分析
- 关键变量: 沿网格轴精确有限差分

---

## 📊 关键结果

| # | 结果 | 数值 | 配置 |
|---|------|------|------|
| 1 | R²_max (median) | **0.8914** | 规则网格, noise=1 |
| 2 | R²_max (90%) | 0.9804 | 高分位上限 |
| 3 | Gap vs Ridge | **+0.43** | 相对 0.46 |
| 4 | Gap vs LightGBM | **+0.32** | 相对 0.57 |
| 5 | Schur decay | 0.6906 | 69% 信息保留 |

**V1 失败教训**: 非规则网格数据用邻近点差分法会导致 CRLB 跨越 20 个数量级，结果不可靠

---

## 💡 核心洞见

### 🏗️ 宏观层（架构设计）

- **理论上限高**: R²_max = 0.89 表明理论可解释 89% 的 log_g 方差
- **当前模型仅利用 64% 理论信息**: LightGBM 0.57 / 理论 0.89

### 🔧 模型层（调参优化）

- **Degeneracy 中等** (Schur=0.69): Multi-task 解纠缠可考虑但非必须
- **参数空间异质性**: 高 log_g (>4) + 高 T_eff (>5000K) 区域更容易

### ⚙️ 工程层（实现细节）

- ⚠️ Fisher 计算必须使用规则网格数据
- CRLB 范围应在 ~3 个数量级内，否则方法有问题

---

## ➡️ 下一步

| 优先级 | 任务 | 相关 experiment_id |
|--------|------|-------------------|
| 🔴 P0 | 继续 CNN/Transformer，理论上限高 | MVP-2.x |
| 🟡 P1 | 尝试 Multi-task (Schur=0.69) | - |
| 🟢 P2 | 区域特化模型（高 T_eff + 高 log_g） | - |

---

## 🔗 相关链接

| 类型 | 路径 |
|------|------|
| 训练仓库 | `~/VIT/` |
| 脚本 | `~/VIT/scripts/scaling_fisher_ceiling_v2.py` |
| 完整报告 | `logg/scaling/exp/exp_scaling_fisher_ceiling_v2_20251224.md` |
| V1 失败报告 | `logg/scaling/exp/exp_scaling_fisher_ceiling_20251223.md` |

