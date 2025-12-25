# 📇 Knowledge Card: Oracle MoE Structure Bonus
> **Name:** Oracle MoE @ noise=1 | **ID:** `VIT-20251223-scaling-oracle-moe-card`  
> **Topic:** `scaling` | **Source:** `exp_scaling_oracle_moe_noise1_20251223.md` | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-23
```
💡 Oracle MoE R²=0.625，比 Global Ridge +16%，结构化先验在高噪声下收益巨大  
适用：MoE 架构设计决策
```

---

## 🎯 问题与设置

**问题**: Oracle MoE (per-bin 专家) 在高噪声下能带来多少结构性收益？

**设置**: 
- 数据: BOSZ 1M train, 1k test, noise σ=1.0
- 模型: 9-bin Oracle MoE (3 Teff × 3 [M/H])
- 关键变量: Global Ridge vs Oracle per-bin Ridge

---

## 📊 关键结果

| # | 结果 | 数值 | 配置 |
|---|------|------|------|
| 1 | Oracle MoE R² | **0.6249** | 1M, noise=1, α=1e5 |
| 2 | Global Ridge R² | 0.46 | 同配置 |
| 3 | Structure Bonus ΔR² | **+0.1637** | 5.5× 阈值 0.03 |
| 4 | Metal-poor bins ΔR² | +0.17~0.19 | 收益最大 |
| 5 | noise=0.2 vs noise=1.0 | 3.3× 放大 | 高噪声收益更大 |

---

## 💡 核心洞见

### 🏗️ 宏观层（架构设计）

- **MoE 结构在高噪声下价值巨大**: ΔR² = +16%
- **噪声放大效应**: noise=0.2 时 ΔR²=+5%, noise=1.0 时 ΔR²=+16%

### 🔧 模型层（调参优化）

- **Metal-poor bins 收益最大**: 可能因为这些区域信号更弱
- **所有 9 bins 都有提升**: 结构先验普遍有效

### ⚙️ 工程层（实现细节）

- 9-bin 划分: Teff [3750,4500,5250,6000] × [M/H] [-2,-1,0,0.5]
- 样本分布: 63k-117k per bin (train), 62-126 (test)

---

## ➡️ 下一步

| 优先级 | 任务 | 相关 experiment_id |
|--------|------|-------------------|
| 🔴 P0 | 开发 trainable soft-gate MoE | MVP-16A-2 |
| 🟡 P1 | 验证 soft routing 接近 oracle 性能 | - |

---

## 🔗 相关链接

| 类型 | 路径 |
|------|------|
| 训练仓库 | `~/VIT/` |
| 脚本 | `~/VIT/scripts/scaling_oracle_moe_noise1.py` |
| 完整报告 | `logg/scaling/exp/exp_scaling_oracle_moe_noise1_20251223.md` |

