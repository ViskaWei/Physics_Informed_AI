# 📘 📗 实验报告：1M 参数 Embedding for Gate
> **Name:** TODO | **ID:** `VIT-20251205-moe-01`  
> **Topic:** `moe` | **MVP:** MVP-14 | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-05 | **Status:** 🔄
```
💡 实验目的  
决定：影响的决策
```

---


## 🔗 Upstream Links
| Type | Link |
|------|------|
| 🧠 Hub | `logg/moe/moe_hub.md` |
| 🗺️ Roadmap | `logg/moe/moe_roadmap.md` |

---

---

## 🔗 上游追溯链接

| 类型 | 链接 | 说明 |
|------|------|------|
| 📍 Roadmap | [`moe_roadmap.md`](./moe_roadmap_20251203.md) | Phase 13 |
| 🧠 Hub | [`moe_hub.md`](./moe_hub_20251203.md) | Q9.4, H-14 |
| 📋 Kanban | [`kanban.md`](../../status/kanban.md) | VIT-20251205-moe-embedding-01 |
| 📗 相关实验 | [`exp_moe_nn_experts`](./exp_moe_nn_experts_20251204.md) | NN expert 失败，但架构思路 |

---

## ⚡ 核心结论速览

> **一句话总结**：⏳ 实验进行中
>
> **假设验证**：
> - H-14: 小模型 (1M 参数) 学习的 embedding 能改善 gate 质量 → ⏳
>
> **关键数字**：
> - 总体 ΔR²: ⏳
> - Bin3/Bin6 改善: ⏳

---

# 1. 🎯 目标

## 1.1 实验目的

学一个更强的低维表征（embedding）当 gate 的 Teff/[M/H] proxy。

**为什么做**：
- 当前 gate 用 13 维特征（Ca II/Na/PCA）
- 这些是手工设计的 proxy，可能不是最优的
- 用小 NN 学习的 embedding 可能捕捉更多信息

**关键定位**：
- **不是替代专家**，只是给 gate 提供更好的输入
- 专家换 NN 已踩过坑（MVP-NN1：NN R²=0.38 << Ridge R²=0.87）

**做完能不能停**：
- ✅ 如果总体 R² +0.003 或 Bin3/Bin6 明显改善 → 继续优化 embedding 设计
- ❌ 如果 ΔR² < 0.001 → embedding 不值得，维持现有 gate features

## 1.2 预期结果

| 指标 | 预期值 | 最低可接受 |
|------|--------|----------|
| 总体 ΔR² | ≥ +0.003 | ≥ +0.001 |
| Bin3/Bin6 | 明显改善 | 不下降 |
| 训练稳定性 | 收敛 | - |

---

# 2. 🧪 实验设计

## 2.1 数据配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **输入** | 候选窗口 + 少量上下文 | 不需要全谱 |
| **窗口选择** | Ca II triplet + Na + H 线 | 物理先验 |
| **训练数据** | 全量 | - |
| **噪声水平** | 0.2 | 与主线一致 |

## 2.2 模型与算法

### Embedding 模型架构

| 架构选项 | 说明 | 参数量 |
|---------|------|--------|
| **1D-CNN** | 捕捉局部谱线形状 | ~1M |
| **Autoencoder** | 无监督特征学习 | ~1M |
| **Supervised Proxy** | 直接预测 [M/H]/Teff | ~1M |

### 推荐架构: 小 1D-CNN

```
Input: [窗口光谱] (e.g., 200 维)
    ↓
Conv1D (32 filters, kernel=5)
    ↓
MaxPool1D (2)
    ↓
Conv1D (64 filters, kernel=3)
    ↓
GlobalAvgPool
    ↓
Dense (64)
    ↓
Output: embedding (8~32 维)
```

### Gate 集成

```
现有 gate features (13 维)
    +
新 embedding (8~32 维)
    ↓
拼接 (21~45 维)
    ↓
回归 gate(MLP)
    ↓
soft weights (9 维)
```

## 2.3 超参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Embedding 维度 | 8, 16, 32 | 消融 |
| CNN filters | 32, 64 | - |
| Dropout | 0.2 | 防过拟合 |
| 学习率 | 1e-3 | Adam |
| Epochs | 50 | Early stopping |

## 2.4 评价指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **总体 R²** | 主指标 | - |
| **Bin3/Bin6 R²** | 困难区域 | 重点关注 |
| **Embedding 可解释性** | t-SNE 可视化 | 是否按物理参数聚类 |

---

# 3. 📊 实验图表

> ⏳ 实验进行中，待填充

---

# 4. 💡 关键洞见

> ⏳ 实验进行中，待填充

---

# 5. 📝 结论

> ⏳ 实验进行中，待填充

---

# 6. 📎 附录

## 6.1 数值结果表

> ⏳ 实验进行中，待填充

## 6.2 实验流程记录

### 执行命令

```bash
# TBD
```

### 关键日志

> ⏳ 实验进行中，待填充

## 6.3 相关文件

| 文件类型 | 路径 | 说明 |
|---------|------|------|
| 训练脚本 | `~/VIT/scripts/moe_embedding_gate.py` | TBD |
| 结果目录 | `~/VIT/results/moe/embedding_gate/` | TBD |
| 图表目录 | `logg/moe/img/` | TBD |

---

*实验 ID: VIT-20251205-moe-embedding-01*

