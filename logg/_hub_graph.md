# 🔗 Hub Dependency Graph

> **Purpose:** 定义 hub 之间的引用关系，供自动更新使用  
> **Updated:** 2025-12-24  
> **Status:** 🟢 Pilot (ridge, lightgbm, NN → moe, scaling, benchmark → master)

---

## 📊 三层金字塔架构

```
┌─────────────────────────────────────────────────────────────────┐
│                   L0 Master Hub (全局战略)                       │
│                   logg/master_hub.md                            │
│   ────────────────────────────────────────────────────────────  │
│   • 跨主题战略问题索引                                           │
│   • 全局 insights 汇合                                          │
│   • 研究路线图                                                   │
├─────────────────────────────────────────────────────────────────┤
│            L1 Cross-Cutting Hubs (横向研究问题)                  │
│   ─────────────────────────────────────────────────────────── │
│   moe/           scaling/        benchmark/                     │
│   (专家混合)      (数据规模)      (模型对比)                      │
│                                                                 │
│   📌 职责：跨模型的特定研究问题                                   │
│   📌 引用：L2 Topic Hubs 的关键数字                              │
├─────────────────────────────────────────────────────────────────┤
│            L2 Topic Hubs (纵向模型专题)                          │
│   ────────────────────────────────────────────────────────────  │
│   ridge/         lightgbm/       NN/                            │
│   (岭回归)       (树模型)        (神经网络)                       │
│                                                                 │
│   📌 职责：单模型的深度探索、超参优化、设计原则                    │
│   📌 产出：关键数字供 L1 引用                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔗 依赖关系定义

```yaml
# ═══════════════════════════════════════════════════════════════
# L0 Master Hub
# ═══════════════════════════════════════════════════════════════
master_hub:
  path: logg/master_hub.md
  layer: L0
  imports_from:
    - moe_hub         # MoE 战略结论
    - scaling_hub     # Scaling 战略结论
    - benchmark_hub   # Benchmark 战略结论

# ═══════════════════════════════════════════════════════════════
# L1 Cross-Cutting Hubs (横向问题)
# ═══════════════════════════════════════════════════════════════
moe_hub:
  path: logg/moe/moe_hub_20251203.md
  layer: L1
  imports_from:
    - ridge_hub       # Ridge baseline, Oracle Expert R²
    - lightgbm_hub    # LightGBM Expert 对比
    - NN_hub          # NN Expert 对比
  exports_to:
    - master_hub      # MoE 战略结论
    - scaling_hub     # MoE @ noise=1 结果

scaling_hub:
  path: logg/scaling/scaling_hub_20251222.md
  layer: L1
  imports_from:
    - ridge_hub       # Ridge α sweep, 1M ceiling
    - lightgbm_hub    # LightGBM scaling
    - moe_hub         # Oracle MoE headroom
  exports_to:
    - master_hub      # Scaling 战略结论

benchmark_hub:
  path: logg/benchmark/benchmark_hub_20251205.md
  layer: L1
  imports_from:
    - ridge_hub       # Ridge R² @ all noise
    - lightgbm_hub    # LightGBM R² @ all noise
    - NN_hub          # MLP R² @ all noise
  exports_to:
    - master_hub      # Benchmark 战略结论

# ═══════════════════════════════════════════════════════════════
# L2 Topic Hubs (纵向专题) - 叶子节点
# ═══════════════════════════════════════════════════════════════
ridge_hub:
  path: logg/ridge/ridge_hub_20251223.md
  layer: L2
  imports_from: []    # 叶子节点无下层依赖
  exports_to:
    - moe_hub         # Ridge baseline, Oracle Expert
    - scaling_hub     # α sweep, 1M ceiling
    - benchmark_hub   # R² @ all noise

lightgbm_hub:
  path: logg/lightgbm/lightgbm_hub_20251130.md
  layer: L2
  imports_from: []    # 叶子节点无下层依赖
  exports_to:
    - moe_hub         # LightGBM Expert 对比
    - scaling_hub     # LightGBM scaling
    - benchmark_hub   # R² @ all noise

NN_hub:
  path: logg/NN/NN_main_20251130.md  # TODO: 升级为 NN_hub
  layer: L2
  imports_from: []    # 叶子节点无下层依赖
  exports_to:
    - moe_hub         # NN Expert 对比
    - benchmark_hub   # MLP R² @ all noise
```

---

## 🔄 更新触发规则

### 当 L2 Hub 更新时

| 更新的内容 | 自动同步到 | 同步的章节 |
|-----------|-----------|-----------|
| `§4.2 Key Numbers` | 所有 `exports_to` hubs | `§5.3 Key Numbers` (L1) 或 `§2 Strategic Questions` (L0) |
| `§3 Insight Confluence` 新增洞见 | 相关 L1 hubs | `§3 Insight Confluence` |
| `§4.1 Confirmed Principles` 新增原则 | 相关 L1 hubs | `§5.1 Design Principles` |

### 当 L1 Hub 更新时

| 更新的内容 | 自动同步到 | 同步的章节 |
|-----------|-----------|-----------|
| `§2 Answer Key` 战略结论改变 | `master_hub` | `§2 Strategic Questions` |
| `§3 Insight Confluence` 重大发现 | `master_hub` | `§3 Global Insights` |

### 传播深度规则

```
L2 更新 → 总是传播到 L1
L1 更新 → 仅当「战略结论改变」时传播到 L0
```

---

## 📋 关键数字同步映射

### Ridge → Parent Hubs

| Ridge Hub 指标 | 同步到 | 目标章节 |
|---------------|-------|---------|
| R² @ noise=1, 32k | benchmark, scaling | §5.3 Key Numbers |
| R² @ noise=1, 100k | benchmark, scaling | §5.3 Key Numbers |
| R² @ noise=1, 1M | benchmark, scaling | §5.3 Key Numbers |
| 最优 α (32k/100k/1M) | scaling | §5.3 Key Numbers |
| Ridge 天花板结论 | moe, scaling | §3 Confluence |

### LightGBM → Parent Hubs

| LightGBM Hub 指标 | 同步到 | 目标章节 |
|------------------|-------|---------|
| R² @ all noise (32k) | benchmark | §5.3 Key Numbers |
| R² @ all noise (100k) | benchmark, scaling | §5.3 Key Numbers |
| 最优配置 (lr, n_estimators) | benchmark | §5.1 Design Principles |
| LightGBM > Ridge 结论 | benchmark | §3 Confluence |

### NN → Parent Hubs

| NN Hub 指标 | 同步到 | 目标章节 |
|------------|-------|---------|
| MLP R² @ noise=1 (32k/100k) | benchmark, moe | §5.3 Key Numbers |
| MLP vs Ridge/LightGBM 对比 | benchmark | §3 Confluence |
| Residual 策略有效性 | moe | §5.1 Design Principles |

---

## 🛠️ `u` 命令传播流程

```
用户: u VIT-20251224-ridge-xxx
    │
    ├─ Step 1: 更新 exp.md
    │
    ├─ Step 2: 更新 ridge_hub.md §4.2 Key Numbers
    │
    ├─ Step 3: 读取本文件，找到 ridge_hub.exports_to
    │   → [moe_hub, scaling_hub, benchmark_hub]
    │
    ├─ Step 4: 传播到 L1 hubs
    │   ├─ moe_hub.md §5.3: 更新 Ridge 相关行
    │   ├─ scaling_hub.md §5.3: 更新 Ridge 相关行
    │   └─ benchmark_hub.md §5.3: 更新 Ridge 相关行
    │
    ├─ Step 5: 检查 L1 战略结论是否改变
    │   如果改变 → 传播到 master_hub.md §2
    │
    └─ Step 6: Git commit + push
        "update: ridge-xxx + propagate to moe, scaling, benchmark"
```

---

## 📌 快捷命令

| 命令 | 作用 |
|------|------|
| `u [experiment_id]` | 更新实验 + 自动传播到 parent hubs |
| `propagate [hub]` | 手动触发某个 hub 的传播 |
| `propagate all` | 全量刷新所有 hub 依赖 |
| `hub status` | 查看 hub 依赖图状态 |

---

## 📎 Changelog

| Date | Change |
|------|--------|
| 2025-12-24 | 创建 Hub Dependency Graph (试点版) |
| 2025-12-24 | 定义 L0/L1/L2 三层架构 |
| 2025-12-24 | 添加 ridge, lightgbm, NN 作为 L2 叶子节点 |
| 2025-12-24 | 添加 moe, scaling, benchmark 作为 L1 横向 hubs |
| 2025-12-24 | 定义更新触发规则和传播流程 |

---

*Last Updated: 2025-12-24*

