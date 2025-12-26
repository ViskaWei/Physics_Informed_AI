# 🍃 Fisher/CRLB 理论上限 V3-A: 化学丰度 Nuisance
> **Name:** Fisher Ceiling V3-A (Chemical Abundance Nuisance)  
> **ID:** `SCALING-20251225-fisher-ceiling-03`  
> **Topic:** `scaling` | **MVP:** MVP-F-V3A | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-25 | **Status:** ✅  
> **Root:** `scaling` | **Parent:** `fisher` | **Child**: -

> 🎯 **Target:** 将化学丰度参数 (C_M, a_M, O_M) 作为 nuisance 加入 Fisher 计算，验证 V2 ceiling 的稳健性  
> 🚀 **Decide:** Δceiling < 10% → V2 结论稳健，继续投模型；Δceiling > 20% → 可能已接近真实上限

---
## ⚡ 核心结论速览
> **一句话**: V3-A 扩展参数空间至 5D（加入 C_M, a_M 作为 nuisance）后，Fisher ceiling 仅下降 **1.93%**，远小于 10% 阈值，**V2 结论高度稳健**。

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H-16T-V3A.1: Δceiling < 10% (R²_max ≥ 0.80)? | ✅ **1.93%** | **通过验证，V2 结论稳健** |
| V3-A R²_max (median) | **0.8742** | 仍远高于 baseline (Ridge: 0.46, LightGBM: 0.57) |
| Schur decay (V3-A) | **0.5778** | 比 V2 (0.6906) 更低，说明化学丰度 nuisance 带来额外退化 |

| 指标 | V2 值 | V3-A 值 | Δ | 启示 |
|------|-------|---------|---|------|
| R²_max (median) | 0.8914 | **0.8742** | **-1.93%** | ✅ **下降幅度极小，V2 结论稳健** |
| R²_max (90%) | 0.9804 | **0.9768** | **-0.37%** | 高置信度样本几乎无影响 |
| Schur decay | 0.6906 | **0.5778** | **-16.3%** | 化学丰度 nuisance 确实带来额外退化，但 ceiling 下降有限 |
| CRLB range (orders) | 2.88 | **3.56** | +0.68 | 数值稳定性仍然良好 |

**Gate-3 决策**: ✅ **Δceiling = 1.93% < 10%** → **V2 结论稳健，继续模型部署**

---
## 1. 🎯 目标

### 1.1 实验目的

> **核心问题**：Fisher 上限到底是在给"哪种世界"算上限？V2 固定了化学丰度 (C_M=0, a_M=0, O_M=0)，实际观测中这些是未知的 nuisance 参数。

**回答的问题**：
1. ✅ 将化学丰度作为 nuisance 参数后，Fisher ceiling 下降多少？ → **仅下降 1.93%**
2. ✅ V2 的结论（R²_max=0.89）是否仍然稳健？ → **高度稳健，R²_max=0.87 仍远高于 baseline**
3. ✅ 参数维度从 3D (T_eff, logg, [M/H]) 扩展到 5D 后，计算是否仍然稳定？ → **稳定，CRLB range 3.56 orders（与 V2 的 2.88 orders 接近）**

**验证假设**: H-16T-V3A.1 (来自 fisher_hub.md)

### 1.2 预期 vs 实际结果

| 场景 | 预期结果 | 实际结果 | 判定 |
|------|---------|---------|------|
| 理想情况 | Δceiling < 10% (R²_max ≥ 0.80) | **Δceiling = 1.93%, R²_max = 0.87** | ✅ **通过** |
| 中等情况 | Δceiling 10-20% (R²_max 0.70-0.80) | - | - |
| 悲观情况 | Δceiling > 20% (R²_max < 0.70) | - | - |

**结论**: V3-A 完全符合理想情况，V2 结论高度稳健。

---

## 2. 🦾 算法

### 2.1 Fisher Information 矩阵扩展

**V2 方法（3D）**：
- 参数: $\theta = (T_{\rm eff}, \log g, [M/H])$
- Nuisance: $\eta = (T_{\rm eff}, [M/H])$ (边缘化后只保留 log_g)

**V3-A 方法（5D）**：
- 参数: $\theta = (T_{\rm eff}, \log g, [M/H], C_M, a_M)$
- Nuisance: $\eta = (T_{\rm eff}, [M/H], C_M, a_M)$ (边缘化后只保留 log_g)
- **注意**: 数据集只包含 C_M 和 a_M，不包含 O_M，因此是 5D 而非 6D

**Fisher 矩阵**：

$$
I(\theta) = J^{\top} \Sigma^{-1} J
$$

其中 Jacobian $J$ 现在包含 5 个维度，沿网格轴计算偏导数。

**CRLB（边缘化 log_g）**：

$$
\text{CRLB}_{\log g, \text{marg}} = \frac{1}{I_{gg} - I_{g\eta} I_{\eta\eta}^{-1} I_{\eta g}}
$$

**R²_max 转换**：

$$
R^2_{\max} = 1 - \frac{\text{CRLB}_{\log g, \text{marg}}}{\text{Var}(\log g)}
$$

### 2.2 化学丰度参数检测

**自动检测逻辑**：
```python
def detect_chemical_parameters(df):
    """自动检测数据集包含的化学丰度参数"""
    detected = []
    for param in ['C_M', 'a_M', 'O_M']:
        if param in df.columns:
            unique_vals = df[param].unique()
            if len(unique_vals) >= 2:
                # 检查是否为规则网格
                diffs = np.diff(np.sort(unique_vals))
                if np.std(diffs) / np.median(diffs) < 0.1:
                    detected.append(param)
    return detected
```

**检测结果**：
- ✅ **C_M**: 6 个值，步长 0.25，范围 [-0.75, 0.50]
- ✅ **a_M**: 4 个值，步长 0.25，范围 [-0.25, 0.50]
- ❌ **O_M**: 只有 1 个唯一值，跳过

**实际参数维度**: 3 (基础) + 2 (化学丰度) = **5D**

---

## 3. 🧪 实验设计

### 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | BOSZ 规则网格合成光谱 |
| 路径 | `/datascope/subaru/user/swei20/data/bosz50000/grid/grid_mag215_lowT/dataset.h5` |
| 采样模式 | 规则网格 |
| 特征维度 | 4,096 (MR arm) |
| 目标 | log_g |

**参数维度**：
- **基础参数（3D）**: T_eff, log_g, [M/H]（与 V2 一致）
- **化学丰度参数（2D）**: C_M, a_M（O_M 不在数据集中）

**数据规模**：
- 总样本数: 30,182
- 成功计算: 30,155
- 失败（边界）: 27

### 3.2 扫描范围

| Magnitude | SNR (approx) | 状态 |
|-----------|--------------|--------|
| 21.5 | 7.1 | ✅ **已完成**（与 V2 对比基准） |

### 3.3 噪声

| 项 | 值 |
|----|-----|
| 类型 | heteroscedastic gaussian |
| σ | 1.0（与 V2 一致） |
| 范围 | 理论分析 (CRLB 计算) |

### 3.4 模型

| 参数 | 值 |
|------|-----|
| 模型 | Fisher Information / CRLB 理论分析 |
| 方法 | 沿网格轴精确有限差分（扩展到 5 维） |

---

## 4. 📊 图表

### 图 1: V3-A vs V2 R²_max 对比

![R²_max Comparison](img/fisher_v3_chemical_r2max_vs_v2.png)

**观察**：
- V3-A (红色) 与 V2 (蓝色) 曲线几乎重叠
- 在 mag=21.5 处，Δ = 1.93%（标注在图中）
- 90% 分位数几乎无差异（V2: 0.9804, V3-A: 0.9768）

### 图 2: Δceiling 下降幅度

![Delta Ceiling](img/fisher_v3_chemical_delta_ceiling.png)

**观察**：
- mag=21.5 处，Δceiling = **1.93%**
- 远低于 10% 阈值线（绿色虚线）
- 远低于 20% 警告线（橙色虚线）

### 图 3: Schur Decay 对比

![Schur Comparison](img/fisher_v3_chemical_schur_comparison.png)

**观察**：
- V3-A (0.5778) < V2 (0.6906)，说明化学丰度 nuisance 带来额外退化
- 但 ceiling 下降有限（仅 1.93%），说明退化主要集中在低置信度样本

### 图 4: CRLB 分布对比

![CRLB Distribution](img/fisher_v3_chemical_crlb_dist.png)

**观察**：
- V2 和 V3-A 的 CRLB 分布形状相似
- V3-A 分布略向右移（CRLB 略大），与 R²_max 下降一致
- 数值稳定性良好（范围 3.56 orders，与 V2 的 2.88 orders 接近）

### 图 5: 检测到的化学丰度参数

![Parameters Detected](img/fisher_v3_chemical_params_detected.png)

**信息**：
- 检测到 2 个化学丰度参数：C_M, a_M
- O_M 不在数据集中
- 总维度：5D（3 基础 + 2 化学丰度）

---

## 5. 💡 关键洞见

### 5.1 Δceiling 的物理含义

**1.93% 的下降意味着什么？**

- **Fisher ceiling 几乎不变**：说明化学丰度参数（C_M, a_M）对 log_g 的 Fisher 信息贡献很小
- **实际观测中，即使不知道 C_M 和 a_M，模型仍能达到接近 V2 ceiling 的性能**
- **V2 的结论（R²_max ≈ 0.89）对未知化学丰度是稳健的**

### 5.2 化学丰度 nuisance 的影响

**Schur decay 的下降（0.6906 → 0.5778）说明**：
- 化学丰度参数确实与 log_g 存在退化关系
- 但这种退化主要集中在低置信度样本（导致 R²_max median 几乎不变）

**为什么 ceiling 下降有限？**
- 可能原因：C_M 和 a_M 对光谱的影响相对较小（相对于 T_eff, log_g, [M/H]）
- 或者：C_M 和 a_M 的网格覆盖范围较小（C_M: 6 个值，a_M: 4 个值），不足以造成显著退化

### 5.3 与 V2 结论的对比

| 维度 | V2 结论 | V3-A 验证 | 一致性 |
|------|---------|-----------|--------|
| R²_max ceiling | 0.89 | 0.87 | ✅ **高度一致** |
| 数值稳定性 | CRLB range 2.88 orders | 3.56 orders | ✅ **相近** |
| 对 baseline 优势 | +0.43 vs Ridge, +0.32 vs LightGBM | +0.41 vs Ridge, +0.30 vs LightGBM | ✅ **几乎一致** |

**结论**: V2 的结论对化学丰度 nuisance 高度稳健。

---

## 6. 📝 结论

### 6.1 Gate-3 决策

**判定标准**: Δceiling < 10% (R²_max ≥ 0.80)

**实际结果**:
- ✅ Δceiling = **1.93%** < 10%
- ✅ R²_max = **0.8742** ≥ 0.80
- ✅ 数值稳定性良好（CRLB range 3.56 orders）

**Gate-3 决策**: ✅ **通过验证，V2 结论稳健，继续模型部署**

### 6.2 对后续实验的启示

1. **V2 结论稳健**: 化学丰度参数作为 nuisance 几乎不影响 Fisher ceiling
2. **模型部署建议**: 可以基于 V2 的 R²_max = 0.89 设定目标，即使实际观测中化学丰度未知
3. **进一步验证**: 可考虑 V3-B（加入更多 nuisance，如红化、速度弥散等）

### 6.3 假设验证

| 假设 | 验证结果 |
|------|---------|
| H-16T-V3A.1: Δceiling < 10% (R²_max ≥ 0.80)? | ✅ **通过** (Δ=1.93%, R²_max=0.87) |

---

## 7. 📎 附录

### 7.1 完整数值结果表

| Magnitude | SNR | R²_max (median) | R²_max (90%) | Schur Decay | CRLB Orders |
|-----------|-----|-----------------|--------------|-------------|-------------|
| 21.5 | 7.1 | 0.8742 | 0.9768 | 0.5778 | 3.56 |

**对比 V2**:
| 指标 | V2 | V3-A | Δ |
|------|----|----|---|
| R²_max (median) | 0.8914 | 0.8742 | -1.93% |
| R²_max (90%) | 0.9804 | 0.9768 | -0.37% |
| Schur decay | 0.6906 | 0.5778 | -16.3% |
| CRLB range (orders) | 2.88 | 3.56 | +0.68 |

### 7.2 实验流程记录

**执行命令**:
```bash
cd ~/VIT
source init.sh
python scripts/scaling_fisher_ceiling_v3_chemical.py \
    --data_path /datascope/subaru/user/swei20/data/bosz50000/grid/ \
    --magnitudes 21.5 \
    --output_dir results/fisher_v3_chemical
```

**参数检测输出**:
```
Detecting chemical abundance parameters...
  ✓ Detected C_M: 6 values, step=0.250, range=[-0.750, 0.500]
  ✓ Detected a_M: 4 values, step=0.250, range=[-0.250, 0.500]
  Warning: O_M has only 1 unique values, skipping

Total parameter dimensions: 5D
  Base parameters (3D): ['T_eff', 'log_g', 'M_H']
  Chemical parameters (2D): ['C_M', 'a_M']
```

**计算统计**:
- 总样本数: 30,182
- 成功计算: 30,155 (99.91%)
- 失败（边界）: 27 (0.09%)
- 计算时间: ~13 秒

**数值稳定性检查**:
- CRLB range: 3.56 orders (与 V2 的 2.88 orders 相近，✅ 通过)
- Condition number: median ~2e5 (与 V2 的 ~2e5 相近，✅ 通过)

### 7.3 代码关键片段

**化学丰度参数检测逻辑** (from `scaling_fisher_ceiling_v3_chemical.py`):

```python
def detect_chemical_parameters(df: pd.DataFrame) -> Tuple[List[str], Dict[str, float]]:
    """自动检测数据集包含的化学丰度参数及其网格结构"""
    detected_params = []
    grid_steps = {}
    
    for param in CHEMICAL_PARAMS:
        if param not in df.columns:
            continue
        
        values = df[param].values
        unique_values = np.unique(values)
        
        if len(unique_values) < 2:
            continue
        
        # 检测网格步长
        diffs = np.diff(np.sort(unique_values))
        diff_median = np.median(diffs)
        diff_std = np.std(diffs)
        
        # 如果标准差/中位数 < 0.1，认为是规则网格
        if diff_std / diff_median < 0.1:
            grid_steps[param] = float(diff_median)
            detected_params.append(param)
    
    return detected_params, grid_steps
```

**5D Fisher 矩阵计算**:

```python
def compute_crlb_from_fisher(I: np.ndarray, param_names: List[str]) -> Dict:
    """计算 CRLB，边缘化所有 nuisance 参数（T_eff, [M/H], C_M, a_M）"""
    logg_idx = param_names.index('log_g')
    nuisance_indices = [i for i in range(len(param_names)) if i != logg_idx]
    
    # Schur complement
    I_nuisance = I[np.ix_(nuisance_indices, nuisance_indices)]
    I_g_nuisance = I[logg_idx, nuisance_indices]
    I_gg_eff = I[logg_idx, logg_idx] - I_g_nuisance @ I_nuisance_inv @ I_g_nuisance.T
    
    crlb_logg_marginalized = 1.0 / max(I_gg_eff, REGULARIZATION)
    return crlb_logg_marginalized
```

---

**实验完成时间**: 2025-12-25  
**下一步**: 更新 kanban.md, roadmap.md, hub.md
