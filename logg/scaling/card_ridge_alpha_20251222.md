# 📇 Knowledge Card: Ridge Alpha Optimization
> **Name:** Ridge Alpha Extended Sweep | **ID:** `VIT-20251222-scaling-ridge-alpha-card`  
> **Topic:** `scaling` | **Source:** `exp_scaling_ridge_alpha_extended_20251222.md` | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-22
```
💡 最优 α 呈倒 U 曲线：100k→3.16e4, 1M→1e5；比原 baseline (5000) 高 1-2 个数量级  
适用：Ridge 超参数调优
```

---

## 🎯 问题与设置

**问题**: Ridge α 的最优值是多少？是否存在倒 U 型曲线？

**设置**: 
- 数据: BOSZ 100k/1M train, 500 test, noise σ=1.0
- 模型: Ridge Regression
- 关键变量: α ∈ [1e2, 1e8] (13 points, logspace)

---

## 📊 关键结果

| # | 结果 | 数值 | 配置 |
|---|------|------|------|
| 1 | 最优 α @ 100k | **3.16e+04** | R²=0.4856 |
| 2 | 最优 α @ 1M | **1.00e+05** | R²=0.5017 |
| 3 | vs baseline α=5000 @ 100k | +2.55% | 改进有限 |
| 4 | vs baseline α=5000 @ 1M | +0.42% | 改进有限 |
| 5 | 过正则化拐点 | α > 1e6 | R² 急剧下降 |

---

## 💡 核心洞见

### 🏗️ 宏观层（架构设计）

- **倒 U 型曲线确认**: α 太小欠正则化，太大过正则化
- **优化 α 改进有限 (<3%)**: Ridge ceiling 确实存在

### 🔧 模型层（调参优化）

- **推荐 α ∈ [1e4, 1e5]**: 而非原来的 5000
- **α 与数据量正相关**: 更多数据 → 更大的最优 α

### ⚙️ 工程层（实现细节）

- α > 1e6 时 R² 急剧下降，避免盲目增大
- 1k test vs 500 test: R² 差异约 0.05（更大 test 更准确）

---

## ➡️ 下一步

| 优先级 | 任务 | 相关 experiment_id |
|--------|------|-------------------|
| ✅ Done | 将最优 α=1e5 应用到后续实验 | - |
| - | Ridge 调参空间已饱和 | - |

---

## 🔗 相关链接

| 类型 | 路径 |
|------|------|
| 训练仓库 | `~/VIT/` |
| 脚本 | `~/VIT/scripts/scaling_ridge_alpha_extended.py` |
| 完整报告 | `logg/scaling/exp/exp_scaling_ridge_alpha_extended_20251222.md` |

