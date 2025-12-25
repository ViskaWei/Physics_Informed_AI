# 🧠 Card F2｜Schur Complement = "扣掉可被 nuisance 解释的信息量"

> **结论（可指导决策）**
> 1) Schur decay ≈ 0.69 **恒定**（与 SNR 无关）：**degeneracy 是结构性的物理现象，不是噪声造成的**
> 2) 想靠"只提升 SNR"来解决 degeneracy → **不现实**，只是让同样的纠缠变得更清晰
> 3) 真正减少 degeneracy 需要：换波段/加观测信息、引入先验/多任务约束 nuisance

---

## 1️⃣ 数学 / 理论依据

### 符号定义

目标参数：$g = \log g$  
Nuisance 参数：$\eta = (T_{\text{eff}}, [\text{M/H}])$

Fisher 信息矩阵按参数分块：

$$
I = \begin{pmatrix} I_{gg} & I_{g\eta} \\ I_{\eta g} & I_{\eta\eta} \end{pmatrix}
$$

### Schur Complement 定义

$$
I_{gg}^{\text{eff}} := I_{gg} - I_{g\eta} I_{\eta\eta}^{-1} I_{\eta g}
$$

这就是边缘化 log_g 的 **有效 Fisher 信息**。

### 直观含义：回归残差类比

这个结构非常像"回归里的残差方差"：

| 项 | 含义 |
|----|------|
| $I_{gg}$ | log_g 自己的信息强度（假装 Teff、[M/H] 已知） |
| $I_{g\eta}$ | log_g 和 nuisance 的"纠缠/相关"（谁影响的谱形很像） |
| $I_{g\eta} I_{\eta\eta}^{-1} I_{\eta g}$ | log_g 信息里，**可被 nuisance 拟合解释掉的部分** |
| $I_{gg}^{\text{eff}}$ | **投影掉 nuisance 子空间后，log_g 独有的信息** |

> **一句话**：Schur complement = "在允许 Teff、[M/H] 一起自由变化的前提下，log_g 还能留下多少可辨识信息"

### CRLB 边缘化公式

$$
\text{CRLB}_{g,\text{marg}} = \frac{1}{I_{gg}^{\text{eff}}} = \frac{1}{I_{gg} - I_{g\eta} I_{\eta\eta}^{-1} I_{\eta g}}
$$

### Schur Decay（信息保留比例）

$$
\text{Schur Decay} = \frac{I_{gg}^{\text{eff}}}{I_{gg}} = 1 - \frac{I_{g\eta} I_{\eta\eta}^{-1} I_{\eta g}}{I_{gg}}
$$

---

## 2️⃣ 几何直觉：导数向量夹角

想象只有两参数 $g$ 和 $\eta$，Jacobian 的两条"导数向量"分别是 $j_g, j_\eta$（都是 4096 维的谱形变化方向）：

| 情况 | 几何 | 物理 | Schur 结果 |
|------|------|------|-----------|
| $j_g \parallel j_\eta$ | 几乎平行 | 改一点 log_g 造成的谱形变化，可被改 Teff 抵消 | Schur ≈ 0，**强 degeneracy** |
| $j_g \perp j_\eta$ | 几乎正交 | log_g 变化方向跟 nuisance 完全不一样 | Schur ≈ 1，**弱 degeneracy** |

> **Schur decay 就是"导数方向重叠程度"的量化**

---

## 3️⃣ 关键发现：Schur Decay 恒定

### Multi-Mag Sweep 结果

| Magnitude | SNR | R²_max (median) | Schur Decay |
|-----------|-----|-----------------|-------------|
| 18.0 | 87.4 | 0.9994 | **0.6641** |
| 20.0 | 24.0 | 0.9906 | **0.6842** |
| 21.5 | 7.1 | 0.8914 | **0.6906** |
| 22.0 | 4.6 | 0.7396 | **0.6921** |
| 22.5 | 3.0 | 0.3658 | **0.6922** |
| 23.0 | 1.9 | 0.0000 | **0.6923** |

> 从 mag=18 到 mag=23，SNR 变化 50 倍，但 **Schur decay 恒定在 ~0.68–0.69**

### 为什么恒定？

噪声（$\Sigma$）会把**所有导数向量**按像素权重一起缩放（做同一个 whitening）：
- 它会让整体信息量变大/变小（所以 $R^2_{\max}$ 随 SNR 剧烈变化）
- 但它**不改变**"log_g 导数方向 vs Teff/[M/H] 导数方向"的夹角结构

| 随 SNR 变化 | 随 SNR 不变 |
|------------|------------|
| $I_{gg}$ 绝对值 | 导数向量夹角 |
| $I_{gg}^{\text{eff}}$ 绝对值 | Schur ratio |
| $R^2_{\max}$ | **degeneracy 比例** |

> 噪声把"信号强度"调大调小，但不改变"哪些参数像彼此"的谱形相似性结构

---

## 4️⃣ 设计启示

### 关于 Schur ≈ 0.69 的解读

> **约 31% 的 log_g 信息会被 Teff/[M/H] 的可变性吃掉**（因为谱形相似/可替代）

| 解读 | 建议 |
|------|------|
| Schur = 0.69 = "中等纠缠" | 有损失但不极端 |
| 靠提升 SNR 解决 degeneracy | ❌ 不现实（比例不变） |
| 真正解纠缠 | ✅ 换波段/加观测、引入先验/多任务 |

### Multi-task 是否值得？

Schur ≈ 0.69 表明：
- Multi-task 可能帮助，但**非必须**
- 更优先的是：error-aware loss / whitening（让模型更像 Fisher 最优）
- MoE 分区已验证结构红利很大

### 两个反直觉的点

| 误解 | 正解 |
|------|------|
| Schur 稳定 = 估计难度稳定 | ❌ R²_max 仍随 SNR 骤降（总信息量变小） |
| Schur 小 = 必须 multi-task | ❌ 0.69 是中等纠缠，多种策略可选 |

---

## 5️⃣ 关键数字速查

| 指标 | 值 | 条件 |
|------|-----|------|
| **Schur decay** | **0.69 ± 0.01** | 恒定，与 SNR 无关 |
| 信息损失比例 | ~31% | 被 nuisance 解释掉的 |
| Schur 稳定性 | < 0.03 变异 | 跨 mag=18-23 (SNR 50x) |

---

## 6️⃣ 实验链接

| 来源 | 路径 | 说明 |
|------|------|------|
| exp_1 (V2 基准) | `logg/scaling/exp/exp_scaling_fisher_ceiling_v2_20251224.md` | 单 mag=21.5 |
| **exp_2 (Multi-Mag)** | `logg/scaling/exp/exp_scaling_fisher_multi_mag_20251224.md` | **6 个 magnitude sweep** |
| Fisher Hub | `logg/scaling/fisher_hub_20251225.md` | 专题汇合 |
| Card F1 | `logg/scaling/card/card_fisher_ceiling_20251224.md` | R²_max 与 SNR 关系 |
| 脚本 | `~/VIT/scripts/scaling_fisher_ceiling_v2_multi_mag.py` | 多 mag |

---

<!--
📌 与 Card F1 的区别：
- Card F1: 讲 R²_max 天花板和 SNR-R²_max 关系
- Card F2: 讲 Schur complement 的数学直觉和 degeneracy 恒定性

📌 使用场景：
- 当需要解释"为什么 multi-task 帮助有限"时引用
- 当需要理解"degeneracy 是物理结构不是噪声"时引用
- 当讨论"SNR 提升无法解决纠缠"时引用
-->

