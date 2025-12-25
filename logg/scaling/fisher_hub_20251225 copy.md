# 🧠 Fisher Hub: 理论上限与信息诊断
> **Topic:** `scaling/fisher` | **Type:** Mini-Hub  
> **Author:** Viska Wei | **Created:** 2025-12-25 | **Last Updated:** 2025-12-25  
> **Status:** 🟢 Stable | **Confidence:** ✅ High (V2 + Multi-Mag 验证 + 框架审核)

---

## ⚡ Answer Key (前 30 行核心)

> **一句话**: Fisher/CRLB V2 框架已验证正确，noise=1 时理论上限 R²_max = **0.89**，与 LightGBM 0.57 存在 **+32% headroom**；SNR=4 是临界点，Schur≈0.69 表明 degeneracy 中等。

| 核心问题 | 答案 | 置信度 |
|---------|------|--------|
| Q1: 理论上限多高？ | R²_max = 0.89 (median @ noise=1) | 🟢 高 |
| Q2: 与实际差距多大？ | +32% vs LightGBM, +43% vs Ridge | 🟢 高 |
| Q3: Degeneracy 多严重？ | Schur = 0.69 (69% 信息保留) | 🟢 高 |
| Q4: SNR 临界点在哪？ | SNR ≈ 4 (mag ≈ 22) | 🟢 高 |
| Q5: 计算方法是否正确？ | ✅ V2 框架已验证，数值自洽 | 🟢 高 |

### 决策规则（按 mag/SNR 分层）

| SNR 范围 | Magnitude | R²_max (median) | 决策含义 |
|----------|-----------|-----------------|----------|
| > 20 | ≤ 20.0 | > 0.99 | 瓶颈在模型/特征提取，不在信息量 |
| 7-20 | 20.5-21.0 | 0.89 | 优秀，仍有大量 headroom 可挖 |
| **4-7** | **21.5-22.0** | **0.74** | 临界区域，模型 + 结构都需投入 |
| 2-4 | 22.0-22.5 | 0.37 | 困难，需结构化/先验策略 |
| < 2 | > 23.0 | 0.00 | 信息悬崖，需改变任务/加信息 |

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

## 📐 计算方法验证（2025-12-25 审核）

### 1. 框架验证：V2 实现正确

V2 的关键点是：**用规则网格沿网格轴做有限差分，拿到偏导数 Jacobian**，再用 heteroscedastic noise 的 Σ 进入 Fisher，最后 Schur complement 边缘化 nuisance 参数得到 logg 的 CRLB，再换算成 R²_max。这个链条在统计学/参数估计里是标准做法。

#### 公式层面：三个核心步骤都正确

**(A) Fisher 信息矩阵**

$$I(\theta)= J^{\top}\Sigma^{-1}J$$

其中 $\mu(\theta)$ 是每个像素的期望光谱，$\Sigma$ 是噪声协方差（对角、每像素方差为 error²）。

**(B) marginal logg 的 CRLB（Schur complement）**

$$\mathrm{Var}(\hat g)\ge \left(I_{gg}-I_{g\eta}I_{\eta\eta}^{-1}I_{\eta g}\right)^{-1}$$

nuisance 参数 $\eta=(T_{\rm eff},[M/H])$，目标参数 $g=\log g$。

**(C) CRLB → R²_max**

$$R^2_{\max}=1-\frac{\mathrm{CRLB}_{g,\mathrm{marg}}}{\mathrm{Var}(g)}$$

这是"unbiased + efficient estimator"意义下的上界。

> ⚠️ **重要提醒**：如果后续引入强先验/后验（Bayes）、或者把任务改成分类（dwarf/giant），这个上界要重新定义（但 Fisher 仍能用：Bayesian CRLB / posterior Fisher）。

### 2. 数值自洽性验证

V2 的几个现象是计算正确的最好 sanity check：

| 指标 | V1 (失败) | V2 (成功) | 改善 | 含义 |
|------|----------|----------|------|------|
| CRLB range | 20 orders | **2.9 orders** | ↓17 | 进入可信数值区 |
| Condition max | 5e+16 | **3.78e+06** | ↓10¹⁰ | 矩阵不再病态 |
| R²_max median | 0.97 (过高) | **0.89** | 合理 | 不再过度乐观 |
| Schur decay | 0.24 (不稳) | **0.69** | 稳定 | 与物理直觉一致 |

Multi-mag 扫描呈现的**单调"信息阶梯/悬崖"**非常符合直觉：
- 亮星（SNR>20）R²_max≈1
- SNR~4 附近明显转折
- SNR<2 median 直接掉到 0

**Schur decay 基本不随 SNR 变**（≈0.68~0.69）也合理：纠缠主要由谱线物理/参数相关性决定，而不是由噪声幅度决定。

### 3. 建议补充的 3 个最终确认（论文写 "validated" 前）

| # | 检查项 | 目的 | 方法 |
|---|--------|------|------|
| 1 | **步长敏感性** | 确认差分方案稳定 | 对比不同 ΔTeff/Δlogg/Δ[M/H] 或用二次拟合，差异<几% |
| 2 | **噪声放大扫描** | 验证 R²_max 随噪声下降的函数形态稳定 | 人为放大 error（×0.5, ×1.0, ×2.0） |
| 3 | **Schur vs Full inverse 一致性** | 数值一致性检验 | 同一批网格点，比较两种方法得到的 Var(logg) |

---

## 📝 论文叙事框架：Ceiling–Gap–Structure 三段论

### 论文结构建议

**(1) Ceiling：Fisher 给出可达上限与信息悬崖**

核心贡献：
- 在 PFS-like heteroscedastic 噪声下，logg 的 Fisher/CRLB 上限随 SNR 发生**阶梯式坍塌**
- 临界 SNR≈4，SNR<2 出现信息悬崖
- degeneracy（Schur decay≈0.69）基本与 SNR 无关，是物理本征

**(2) Gap：现有 ML 到上限还差多少（按 mag/SNR 分层）**

核心图表：把模型按 mag/SNR 分桶，并叠加 R²_max：

$$\text{efficiency} = \frac{R^2_{\text{model}}}{R^2_{\max}}$$

这张"信息论效率图"是论文核心贡献之一。

| 模型 | 全局 R² | efficiency (@ mag=21.5) |
|------|---------|------------------------|
| Template | 0.40 | 45% |
| Ridge | 0.46 | 52% |
| LightGBM | 0.57 | 64% |
| Oracle MoE | 0.62 | 70% |
| **Fisher ceiling** | **0.89** | **100%** |

**(3) Structure：MoE 结构红利在高噪声更显著**

核心结论：
- Oracle MoE 在 noise=1 时 ΔR²≈+0.16，且所有 bins 都提升
- 金属贫区域提升最大
- 这可以作为"结构先验（分区/分专家）"在低 SNR 下更关键的证据

> **论文叙事核心**：
> - "**观测上到底能做到多好？**"（Fisher ceiling + SNR cliff）
> - "**算法上怎么接近这个上限？**"（MoE/结构化 + error-aware 学习）

### 论文核心 4 张图

| # | 图表 | 内容 | 作用 |
|---|------|------|------|
| 1 | R²_max vs SNR/mag | 理论上限的阶梯式下降 + 信息悬崖 | 定义物理边界 |
| 2 | Model efficiency heatmap | 各模型 R²/R²_max 按 mag 分层 | 量化 headroom |
| 3 | MoE structure bonus | Oracle MoE ΔR² 按 bin 分解 | 证明结构红利 |
| 4 | Schur decay vs SNR | 恒定的 degeneracy | 证明纠缠是物理本征 |

---

## 🎯 P0/P1 路线图（基于 Fisher 结论）

### P0：最优先（1-2 个迭代决定方向）

#### P0-1: 按 mag/SNR 分桶评估所有模型（必须做）

**目标**：回答"你离天花板还有多远？"

**方法**：
- 把 template/linear/LGBM/MoE/CNN 按 mag 分桶
- 叠加 multi-mag 的 R²_max 曲线
- 画出"信息论效率图"

**决策规则**：
- 若高 SNR 区域 efficiency < 80% → 继续投模型/表示学习
- 若低 SNR 区域 efficiency > 80% → 转向结构化/先验

#### P0-2: 按 (Teff, [M/H]) bins 做 gap heatmap

**目标**：定位"哪里该投入"

**方法**：3×3 heatmap，每个 bin 显示：
- R²_max（理论上限）
- R²_model（当前最佳）
- gap = R²_max - R²_model
- efficiency = R²_model / R²_max

#### P0-3: 加权损失 / Error-aware 输入

**目标**：让模型正确利用 heteroscedastic noise 信息

**方法**：
- weighted MSE：$\sum_i \frac{(y-\hat y)^2}{\sigma_i^2}$
- 或把 error 作为输入通道：$(x, \sigma)$ 双通道输入
- 或输入归一化：$x = \text{flux}/\text{error}$

**决策规则**：
- 若 CNN/MLP 从"乱学"变成"≥Ridge" → 误差建模是关键瓶颈
- 否则 → 瓶颈更多来自结构/非线性

#### P0-4: 加权线性基线对齐 Fisher

**目标**：验证 weighted ridge 能否超过 0.46

**方法**：广义最小二乘 / weighted ridge

**意义**：若显著提升说明"误差建模"是主要瓶颈之一

#### P0-5: Trainable gate MoE

**目标**：把 Oracle MoE 的结构红利变成可落地收益

**方法**：
- Experts：继续用 ridge/lightgbm（稳定、可解释）
- Gate：小模型（logistic / 小 MLP），输入使用：
  - SNR 统计（median SNR、分波段 SNR）
  - logg 敏感线的 index（等效宽度/线深）
  - PCA 前 10-50 维
  - template-fitting 输出作为物理 summary feature

**目标**：trainable gate R² ≥ Oracle 的一部分（先追到 0.60-0.62）

### P1：在 P0 做完后再决定

#### P1-1: 1D ResNet + (flux,error) 双通道 + weighted loss

**触发条件**：mag≤21.5 或 SNR≥7 区域 R²_max 仍很高（~0.9），但 MoE/LGBM 仍远低于上限

**方法**：
- 输入：2 通道（flux, error）
- loss：weighted MSE 或 Gaussian NLL
- 归一化：每条谱做鲁棒归一化（除以中位数/continuum），再做全局标准化
- 输出：logg + 不确定度（可选）

#### P1-2: 多任务（logg + Teff + [M/H]）

**理由**：Schur decay≈0.69 表明纠缠中等，多任务可能帮你"学到解纠缠的中间表征"

**方法**：同时预测三个参数，观察是否有正则化效应

### P2：如果关心 mag>22.5 的科学目标

| 策略 | 方法 |
|------|------|
| 改任务 | 连续 logg → dwarf/giant 分类、或输出 credible interval |
| 加信息 | 多曝光叠加、加入 photometry、加入先验（银河系恒星族群分布） |

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
              └── 汇合: 理论上限 + SNR 边界 + 设计原则 + P0/P1 路线图
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
| **F-7** | **V2 框架已验证：Fisher→CRLB→R²_max 计算链正确** | 框架审核 | ✅ 稳定 |
| **F-8** | **最优估计应用 Σ⁻¹ 像素加权，当前 ML 可能未利用** | 框架审核 | ⚠️ 待验证 P0-3 |
| **F-9** | **按 mag/SNR 分桶评估是决策的前提** | 战略审核 | 🔴 P0 优先 |
| **F-10** | **Trainable gate MoE 是最像"能发论文"的方向** | 战略审核 | 🟡 P0-5 |

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
| **P-F6** | **最优估计器应使用 Σ⁻¹ 加权（weighted MSE/NLL）** | Fisher 框架 | 损失函数设计 |
| **P-F7** | **不应用全局 R² 指导策略，应按 mag/SNR 分桶** | Multi-Mag 趋势 | 评估方法 |
| **P-F8** | **高噪声下结构化建模（MoE）比堆数据更有价值** | Oracle MoE ΔR²=+0.16 | 架构选择 |

### 关键数字约束

| 条件 | 阈值 | 决策 |
|------|------|------|
| R²_max (median) | ≥ 0.75 | ✅ 继续投入 |
| Schur decay | > 0.7 | Multi-task 可选 |
| Schur decay | < 0.5 | Multi-task 优先 |
| SNR | ≥ 4 | 正常估计 |
| SNR | < 2 | 信息悬崖警告 |
| efficiency | < 80% 在高 SNR 区 | 继续投模型/表示 |

---

## ➡️ Next Actions (战略导航)

### 基于 Fisher 结论的优先路线

| # | 任务 | 优先级 | 理由 | 决策门控 |
|---|------|--------|------|---------|
| P0-1 | 按 mag/SNR 分桶评估所有模型 | 🔴 P0 | 定量回答"离天花板多远" | 必须做 |
| P0-2 | (Teff, [M/H]) gap heatmap | 🔴 P0 | 定位投入区域 | 必须做 |
| P0-3 | Error-aware 输入 / weighted loss | 🔴 P0 | 修复 CNN/MLP 训练 | 若显著提升→误差建模关键 |
| P0-4 | 加权线性基线 | 🔴 P0 | 对齐 Fisher 框架 | 若超0.46→误差是瓶颈 |
| P0-5 | Trainable gate MoE | 🔴 P0 | 把 Oracle 红利落地 | 最像发论文的方向 |
| P1-1 | 1D ResNet + 双通道 | 🟡 P1 | 在高 SNR 区继续挖 headroom | 仅当 P0 验证 gap 大 |
| P1-2 | Multi-task (logg+Teff+[M/H]) | 🟡 P1 | Schur=0.69 可能有帮助 | 可选实验 |
| P2 | 改任务/加信息 (mag>22.5) | 🟢 P2 | 临界区域策略 | 待 P0 图表后决定 |

### 最短闭环清单

**立刻做（P0）**：
1. ✅ 按 mag/SNR 分桶评估：template/linear/LGBM/MoE/CNN 画成 R²(mag) 曲线，叠加 R²_max(mag)
2. ✅ 做 (Teff,[M/H]) 3×3 bin 的 gap heatmap：把 oracle MoE 结论也一起放进去
3. ✅ 加上 error-aware（weighted loss 或把 error 当输入通道）：先把 CNN/MLP 修到"至少不差于线性"

**接着做（仍是 P0，更偏方法贡献/发论文）**：
4. Trainable gate MoE：用低维物理特征/PCA/SNR 做 gating，expert 用 ridge 或 lightgbm

**最后再决定（P1）**：
5. 如果在高 SNR 区域仍有大 headroom，再投 1D ResNet/Transformer；否则把精力放在结构化/先验上

### 已关闭方向

| 方向 | 原因 |
|------|------|
| 放弃 CNN 投入 | ❌ headroom 大，值得继续 |
| 强制 Multi-task | ❌ Schur=0.69 非必须 |
| 在 mag>23 堆模型 | ❌ 信息悬崖，需改变策略 |

---

## 📎 附录

### A. 子实验索引

| # | Experiment ID | 文件 | 状态 | 角色 |
|---|--------------|------|------|------|
| 1 | SCALING-20251223-fisher-ceiling-01 | [V1](./exp/exp_scaling_fisher_ceiling_20251223.md) | ❌ Failed | 首次尝试 |
| 2 | SCALING-20251224-fisher-ceiling-02 | [V2](./exp/exp_scaling_fisher_ceiling_v2_20251224.md) | ✅ Done | 修正基线 |
| 3 | SCALING-20251224-fisher-multi-mag | [Multi-Mag](./exp/exp_scaling_fisher_multi_mag_20251224.md) | ✅ Done | 扩展验证 |

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

### E. Multi-Magnitude 完整结果

| Magnitude | SNR | R²_max (median) | R²_max (90%) | Schur Decay | CRLB Orders |
|-----------|-----|-----------------|--------------|-------------|-------------|
| **18.0** | 87.4 | **0.9994** | 0.9999 | 0.6641 | 2.9 |
| **20.0** | 24.0 | **0.9906** | 0.9983 | 0.6842 | 2.9 |
| **21.5** | 7.1 | **0.8914** | 0.9804 | 0.6906 | 2.9 |
| **22.0** | 4.6 | **0.7396** | 0.9530 | 0.6921 | 2.9 |
| **22.5** | 3.0 | **0.3658** | 0.8854 | 0.6922 | 2.9 |
| **23.0** | 1.9 | **0.0000** | 0.7180 | 0.6923 | 2.9 |

---

> **Hub 创建时间**: 2025-12-25  
> **最后更新**: 2025-12-25 (添加计算验证 + 论文框架 + P0/P1 路线图)  
> **基于实验**: V1 (失败) → V2 (基线) → Multi-Mag (扩展)  
> **置信度**: 🟢 High (规则网格+精确差分方法验证 + 框架审核)
