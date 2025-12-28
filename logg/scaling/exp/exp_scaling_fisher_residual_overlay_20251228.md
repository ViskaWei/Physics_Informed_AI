<!--
📝 Agent 书写规范（不出现在正文）:
- Header 全英文
- 正文中文
- 图表文字全英文（中文会乱码）
- 公式用 LaTeX: $inline$ 或 $$block$$
-->

# 🍃 Fisher CRLB Residual Overlay
> **Name:** Fisher CRLB Residual Overlay  
> **ID:** `SCALING-20251228-fisher-residual-overlay`  
> **Topic:** `fisher` | **MVP:** MVP-FU-3 | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-28 | **Status:** ⏳ 立项  
> **Root:** `fisher` | **Parent:** `MVP-FU-1` (Upper-Bound Curves) | **Child:** -

> 🎯 **Target:** 在现有模型的 parity/residual 图上叠加 Fisher CRLB 理论下界，可视化"理论最小误差"vs"实际模型误差"  
> 🚀 **Next:** 产出论文级图表 → Ceiling–Gap–Structure 叙事的直观落地

## ⚡ 核心结论速览

> **一句话**: [待完成] 在 residual 图上叠加 σ_fisher(x) 包络，直观展示模型 vs 理论极限的差距

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| Q7.1: σ_fisher(x) 按 logg 分箱是否揭示结构？ | ⏳ | [待实验] |
| Q7.2: 模型 residual 能否系统性低于理论下界？ | ⏳ | [待实验] |

| 指标 | 值 | 启示 |
|------|-----|------|
| Best R² | ⏳ | ⏳ |
| vs Fisher ceiling | ⏳ | ⏳ |

| Type | Link |
|------|------|
| 🧠 Hub | `logg/scaling/fisher_hub_20251225.md` § Q7 |
| 🗺️ Roadmap | `logg/scaling/fisher_roadmap_20251225.md` § MVP-FU-3 |
| 📋 Kanban | `status/kanban.md` |

---
# 1. 🎯 目标

**问题**: 如何在模型诊断图（parity/residual）上直观展示 Fisher 理论下界？

**验证**: Q7 / MVP-FU-3

| 预期 | 判断标准 |
|------|---------|
| 理论下界清晰可见 | 包络线/带状区域与 residual 形成对比 |
| 模型 residual 高于理论下界 | 若系统性低于 → 模型过拟合或 Fisher 假设有问题 |
| 按 logg 分箱存在结构 | 若某些 logg 区域 gap 更大 → 指导后续优化方向 |

**核心动机**:
- Fisher/CRLB 给出的是 **局部（per-sample/per-θ）下界**：依赖该样本的真实参数 θ=(Teff, logg, [M/H]) 以及误差协方差 Σ
- 因为它依赖 **Teff、[M/H]、error 向量**，所以只按 true logg 一维画一条线需要做 **分箱聚合**
- 这个实验把 MVP-FU-1 的全局 R²_max(SNR) 曲线落地到每个样本的可视化

---

# 2. 🦾 算法

> 📌 理论分析类实验必填

**每个样本的理论最小误差：σ_fisher(x)**

对每个样本（或每个网格点）计算：

1. **前向模型均值光谱**：$\mu(\theta)$（用 clean flux / 模拟器输出；不需要 noisy）

2. **Jacobian**：$J=\frac{\partial \mu}{\partial \theta}$（用规则网格做精确有限差分）

3. **噪声协方差**：$\Sigma = \mathrm{diag}((\text{noise\_level}\cdot \text{error})^2)$

4. **Fisher 信息矩阵**：
$$
I(\theta)=J^\top \Sigma^{-1} J
$$

5. **边缘化 logg 的 CRLB**（把 Teff、[M/H] 当 nuisance，用 Schur complement）：
$$
\mathrm{CRLB}_{g,\text{marg}}=\frac{1}{I_{gg}-I_{g\eta}I_{\eta\eta}^{-1}I_{\eta g}}
$$

6. **该样本的理论最小 1σ 误差**：
$$
\sigma_{\text{fisher}}(\theta)=\sqrt{\mathrm{CRLB}_{g,\text{marg}}}
$$

**分箱聚合方法**：

把样本按 **true logg** 分箱（0.1 或 0.2 dex 一箱），每箱计算：
$$
\tilde\sigma(x) = \text{median}(\sigma_{\text{fisher}}(\theta))
$$

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | BOSZ / 规则网格 |
| 路径 | `/datascope/.../grid/grid_mag215_lowT/dataset.h5` |
| Train/Val/Test | N/A（使用 Fisher 计算结果） |
| 特征维度 | 50000（光谱） |
| 目标 | log_g |

## 3.2 噪声

| 项 | 值 |
|----|-----|
| 类型 | heteroscedastic Gaussian (PFS 模拟器) |
| σ | noise_level × error vector |
| 范围 | per-sample |

## 3.3 模型

| 参数 | 值 |
|------|-----|
| 理论模型 | Fisher/CRLB (已有 V2 结果) |
| 对比模型 | Ridge / LightGBM / CNN / Oracle MoE |

## 3.4 训练

> 理论分析类实验可填 N/A

| 参数 | 值 |
|------|-----|
| epochs | N/A |
| batch | N/A |
| lr | N/A |
| optimizer | N/A |
| seed | 42 |

## 3.5 扫描参数

| 扫描 | 范围 | 固定 |
|------|------|------|
| Magnitude/SNR | [18, 20, 21.5, 22, 22.5, 23] | - |
| 分箱宽度 | 0.1 / 0.2 dex | - |
| 统计方法 | median / mean | - |

---

# 4. 📊 图表

> ⚠️ 图表文字必须全英文！

## 4.1 必须产出的图表

### Fig 1: Residual vs True logg with Fisher Envelope (核心图)

**内容**：
- 散点：模型预测 residual = pred_logg - true_logg
- 水平虚线：当前的 ±1σ 全局 std（如 ±0.63）
- **新增**：两条曲线作为理论下界包络
  - 上包络：$+\tilde\sigma(x)$
  - 下包络：$-\tilde\sigma(x)$

**坐标轴**：
- x: true_logg (dex)
- y: residual (dex)

**解释口径**：这不是"模型应该落在里面"，而是"任何方法的 residual 标准差不可能系统性低于这条曲线（在无偏、模型正确条件下）"

![](./img/fisher_residual_overlay.png)

**观察**:
- [待完成]
- [待完成]

---

### Fig 2: Parity Plot with Fisher Band

**内容**：
- 散点：pred_logg vs true_logg
- 红色虚线：y = x (Perfect)
- **新增**：带状区域
  - 带上边界：$y=x+\tilde\sigma(x)$
  - 带下边界：$y=x-\tilde\sigma(x)$

**坐标轴**：
- x: true_logg (dex)
- y: pred_logg (dex)

**视觉效果**：模型点云的厚度 vs Fisher 给的"理论最窄厚度"

![](./img/fisher_parity_overlay.png)

**观察**:
- [待完成]
- [待完成]

---

### Fig 3: Residual Histogram with Fisher RMSE

**内容**：
- 直方图：residual 分布
- **新增**：竖线标注数据集级别的理论最小 RMSE：
$$
\mathrm{RMSE}_{\min} \approx \sqrt{\mathbb{E}[\mathrm{CRLB}_{g,\text{marg}}]}
$$

**坐标轴**：
- x: residual (dex)
- y: count

![](./img/fisher_histogram_overlay.png)

**观察**:
- [待完成]
- [待完成]

---

## 4.2 可选扩展图表

### Fig 4: σ_fisher Distribution per logg Bin (P1)

**内容**：
- 箱线图：每个 logg bin 的 σ_fisher 分布
- 显示异质性：某些 logg 区域的理论误差更大

---

### Fig 5: Multi-Model Comparison (P1)

**内容**：
- 多条 residual 曲线（Ridge / LightGBM / CNN / MoE）与同一条 Fisher 包络对比
- 量化各模型的 efficiency = |residual| / σ_fisher

---

### Fig 6: 2D Heatmap - σ_fisher vs (Teff, logg) (P2)

**内容**：
- Fisher 热力图按 (Teff, logg) 二维展示
- 更强的表达方式：不把异质性抹平

---

# 5. 💡 洞见

## 5.1 宏观
- Fisher CRLB 是在以下假设下的"上限/下界"：
  - 前向模型 μ(θ) 正确
  - 噪声协方差 Σ 正确
  - 估计器无偏（或近似无偏）
  - 是局部（在该 θ 附近）信息量的界
- 图上最好标注成：**"Fisher CRLB (marginal) lower bound"**，避免读者误解为"保证能达到"

## 5.2 模型层
- σ_fisher 其实强依赖 (Teff, [M/H], SNR/error)
- 只按 logg 一维画会把异质性抹平
- 更强的表达方式：画两条理论线 (median 和 90%)，或做 (Teff, logg) 热力图

## 5.3 细节
- 分箱聚合用 median 更稳定，mean 易受离群点影响
- ±1σ 画单条线，±1.96σ 画 95% 理论带

---

# 6. 📝 结论

## 6.1 核心发现
> **[待完成]**

- ⏳ Q7.1: [待实验]
- ⏳ Q7.2: [待实验]

## 6.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | **[待完成]** | [待完成] |
| 2 | **[待完成]** | [待完成] |

## 6.3 设计启示

| 原则 | 建议 |
|------|------|
| 标注规范 | 图上标注 "Fisher CRLB (marginal) lower bound" |
| 分箱宽度 | 0.1~0.2 dex 为宜 |

| ⚠️ 陷阱 | 原因 |
|---------|------|
| 误解为"保证能达到" | CRLB 只是下界，不是保证 |
| 按 logg 一维画线抹平异质性 | 信息依赖 Teff/[M/H]/error |

## 6.4 关键数字

| 指标 | 值 | 条件 |
|------|-----|------|
| σ_fisher (median) | ⏳ | mag=21.5 |
| RMSE_min (dataset) | ⏳ | - |
| vs model RMSE | ⏳ | - |

## 6.5 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| 执行实验 | 生成 Fig 1-3 | 🔴 P0 |
| 扩展 | 多模型对比 | 🟡 P1 |
| 扩展 | (Teff, logg) 2D 热力图 | 🟢 P2 |

---

# 7. 📎 附录

## 7.1 数值结果

| 配置 | R² | MAE | RMSE |
|------|-----|-----|------|
| [待完成] | | | |

## 7.2 执行记录

| 项 | 值 |
|----|-----|
| 仓库 | `~/VIT` |
| 脚本 | `scripts/scaling_fisher_residual_overlay.py` (待创建) |
| Config | - |
| Output | `results/fisher_residual_overlay/` |
| 已有数据 | V2 的 `crlb_logg_marg` 数组可直接复用 |

```bash
# 执行
python scripts/scaling_fisher_residual_overlay.py --mag 21.5

# 图表保存
# → logg/scaling/exp/img/fisher_residual_overlay.png
# → logg/scaling/exp/img/fisher_parity_overlay.png
# → logg/scaling/exp/img/fisher_histogram_overlay.png
```

## 7.3 参考代码

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `~/VIT/scripts/scaling_fisher_ceiling_v2.py` | Fisher/CRLB 计算逻辑 | 添加 per-sample 输出 |
| `~/VIT/scripts/scaling_fisher_ceiling_v2_multi_mag.py` | multi-mag 循环 | 添加分箱聚合 |
| `~/VIT/utils/plotting.py` | 基础绑图框架 | 添加 overlay 逻辑 |

## 7.4 关键实现细节

### 分箱方案

```
logg_bins = np.arange(0.5, 5.5, 0.2)  # 0.2 dex 分箱
for bin_center in logg_bins:
    mask = (logg >= bin_center - 0.1) & (logg < bin_center + 0.1)
    sigma_median[bin_center] = np.median(sigma_fisher[mask])
```

### 曲线插值

```
# 在 residual 图上画包络
logg_smooth = np.linspace(0.5, 5.0, 100)
sigma_interp = np.interp(logg_smooth, logg_bins, sigma_median)
ax.plot(logg_smooth, +sigma_interp, 'r--', label='Fisher CRLB (marginal) +1σ')
ax.plot(logg_smooth, -sigma_interp, 'r--', label='Fisher CRLB (marginal) -1σ')
```

### 95% 理论带

```
# 如果想画 95% 理论带
ax.fill_between(logg_smooth, -1.96*sigma_interp, +1.96*sigma_interp, 
                alpha=0.2, color='red', label='Fisher 95% theoretical band')
```

---

> **实验完成时间**: [待完成]
