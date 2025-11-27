# 📘 Error 特征预测 log_g 实验报告

---

# 0. 元信息（Meta Information）
- **实验名称：** Error-based log_g Prediction Experiment  
- **作者：** TODO  
- **日期：** 2025-11-27  
- **数据版本：** HDF5 光谱数据（含 flux 和 error 列）  
- **模型版本：** LightGBM (lgbm_error_nz0.pkl), Ridge Regression (lnreg_e_n32k_nz0.pkl)

---

# 1. 目标

## 1.1 大目标

> 探索光谱数据中 **已知高斯噪声方差（error）** 是否包含恒星物理参数（log_g）的信息，从而：

- 理解光谱测量误差与恒星物理参数之间的潜在关联
- 评估 error 特征在神经网络架构设计中的价值
- 揭示数据生成过程中可能存在的隐式信息结构或数据泄漏

**最终目标：** 通过对比 flux-based 和 error-based 模型，理解不同特征类型对 log_g 估计的贡献，为后续神经网络设计提供特征工程依据。

## 1.2 实验目标

本次实验属于以下中层方向：

- **评估异方差误差（error σ）作为独立特征的预测能力**
- 对比线性模型与非线性模型（LightGBM）从 error 中提取信息的能力差异
- 探究 error 与 log_g 之间是否存在物理或统计上的相关性

## 1.3 子目标

本次实验要验证的具体问题：

1. **验证 "error 单独作为特征能否预测 log_g"**
2. **检验线性模型（Ridge）vs 非线性模型（LightGBM）在 error 特征上的表现差异**
3. **对比 error-based 与 flux-based 模型的性能差距**
4. **分析 error 中 log_g 信息的来源（物理关联 vs 数据偏差）**

---

# 2. 实验设计（Experiment Design）

## 2.1 数据（Data）

| 配置项 | 值 |
|--------|-----|
| 训练样本数 | 32,000 |
| 测试样本数 | 512 |
| 特征维度 | 4,096 |
| 标签参数 | log_g |
| 噪声水平 | 0 (noiseless) |

**噪声模型：**
$$
\text{noisy\_flux} = \text{flux} + \mathcal{N}(0, 1) \times \text{error} \times \text{noise\_level}
$$

其中：
- **flux**: 光谱流量值（每个波长点的辐射强度）
- **error**: 每个波长点的已知高斯噪声标准差 σ（先验已知，由仪器模型/光子计数计算得出）
- 本实验 noise_level = 0，即未实际添加噪声，但 error 本身仍为特征

## 2.2 使用的特征类型

| 实验组 | 特征类型 | 特征维度 |
|--------|----------|----------|
| 本实验 | error (噪声标准差) | 4,096 |
| 对照组 | flux (光谱流量) | 4,096 |

## 2.3 模型与算法（Model & Algorithm）

### 线性回归 / 岭回归（Ridge Regression）
$$
\hat{y} = X w + b
$$
$$
w = (X^\top X + \alpha I)^{-1} X^\top y
$$

其中 $\alpha = 100$（强正则化）

### LightGBM
- **n_estimators**: 1000 trees
- **训练时间**: ~18 min
- 其他参数: 默认配置

## 2.4 超参数（Hyperparameters）

| 参数 | 值 |
|------|-----|
| Ridge α | 100 |
| LightGBM n_estimators | 1000 |
| noise_level | 0 |
| 特征维度 | 4096 |

---

# 3. 实验结果表（Results）

## 3.1 Error-Based Models（本实验）

| Model | test_R² | test_MAE | test_RMSE | train_R² | Notes |
|-------|---------|----------|-----------|----------|-------|
| **LightGBM** | **0.3920** | **0.1872** | **0.2277** | 0.7896 | 1000 trees, 18min |
| Linear (Ridge α=100) | -0.0009 | 0.2509 | 0.2921 | 0.0013 | 无预测能力 |

## 3.2 Flux-Based Models（对照组）

| Model | test_R² | test_MAE | test_RMSE | Notes |
|-------|---------|----------|-----------|-------|
| **LightGBM** | **0.9981** | **0.0082** | **0.0128** | Best performance |
| LinearRegression (OLS) | 0.9694 | 0.0380 | 0.0511 | 无正则化 |
| Ridge α=100 | 0.7943 | 0.1014 | 0.1324 | 过度正则化 |

## 3.3 对比分析

| 对比项 | Flux-based | Error-based | Δ (差异) |
|--------|------------|-------------|----------|
| LightGBM R² | 0.998 | 0.392 | **-0.606** |
| Linear R² | 0.969 | ~0 | **-0.969** |

---

# 4. 关键洞见（Key Insights）

## 4.1 宏观层洞见（用于指导 Neural Network 架构设计）

### 核心发现：Error 确实包含 log_g 信息！

1. **Error 包含约 39% 的 log_g 可解释方差**
   - LightGBM 使用 error 特征达到 R² = 0.392，远高于随机水平
   - 这表明 error 中存在与 log_g 相关的结构性信息

2. **Error 信息是高度非线性的**
   - 线性模型（Ridge）完全无法提取（R² ≈ 0）
   - 只有非线性模型（树模型）能够捕捉这种关系

3. **对 NN 设计的启示**
   - 如果使用 error 作为辅助特征，需要非线性层来提取信息
   - 简单的线性组合 (flux + error) 可能无法充分利用 error 信息
   - 建议考虑 **SNR = flux / error** 或其他非线性组合方式

## 4.2 模型层洞见（用于优化模型）

### 线性模型失效原因

线性模型的预测形式：
$$
\hat{y} = w_0 + \sum_{i=1}^{4096} w_i \cdot \text{error}\_i
$$

线性模型无法捕捉非线性关系如：
$$
\text{error}\_i \propto \sqrt{F\_i} \quad \Rightarrow \quad F\_i \propto \text{error}\_i^2
$$

因此，即使 error 与 log_g 存在间接关联，线性模型也无法"自动平方"各维度再组合。

### LightGBM 有效原因

- 树模型通过分裂阈值组合，可以近似任意非线性函数
- 能够隐式学习 $\text{error}_i^2$、$\text{error}_i \cdot \text{error}_j$ 等非线性项

### 过拟合警示

- train_R² (0.79) >> test_R² (0.39)
- 存在明显过拟合，可能需要更强正则化或更多训练数据

## 4.3 物理层洞见

### Error 为什么会"知道" log_g？

**物理解释链：**
$$
\log g \longrightarrow \text{恒星亮度 } L\_\star \longrightarrow \text{光子数 } F\_i(\lambda) \longrightarrow \text{error}\_i(\lambda)
$$

**噪声模型的物理依赖：**

在典型光子计数系统中：
$$
\sigma\_i \approx \sqrt{F\_i + N\_{\text{sky},i} + N\_{\text{read}}^2}
$$

因此 error 是 flux 的非线性函数，而 flux 与 log_g 强相关。

**天体物理关联：**
- **巨星（低 log_g）** → 本征更亮 → flux 更大 → Poisson 噪声绝对值更大
- **矮星（高 log_g）** → 本征更暗 → flux 更小 → readnoise/sky noise 比例更大

### ⚠️ 数据偏差警示

如果不同 log_g 区间的星在模拟/观测时被赋予了不同的噪声策略或 SNR 分布，error 就会成为"隐式编码 log_g 的 side channel"，这是需要警惕的数据泄漏风险。

## 4.4 统计自洽性验证

**R² 与 RMSE 的一致性检验：**

从 flux 模型估计 log_g 总体标准差：
$$
\sigma\_y \approx \frac{\text{RMSE}\_\text{flux}}{\sqrt{1-R^2\_\text{flux}}} = \frac{0.0128}{\sqrt{1-0.9981}} \approx 0.293
$$

验证 error 模型的 RMSE：
$$
\text{RMSE}\_\text{error} \approx \sqrt{(1-0.392) \cdot 0.293^2} \approx 0.228
$$

实际值 0.2277，与理论计算一致 ✓

---

# 5. 建议绘图（Plot Suggestions）

### 5.1 LightGBM 特征重要性 vs 波长

- **内容**: 横轴为波长（4096 个点），纵轴为 LightGBM feature_importance
- **标注**: Top 50 最重要的波长位置
- **目的**: 验证重要的 error 像素是否对应 log_g 敏感的谱线区域（Balmer wings、Ca II triplet 等）

```python
# 代码建议（不执行）
importance = model.feature_importances_
top_k_indices = np.argsort(importance)[-50:]
plt.stem(wavelengths, importance)
```

### 5.2 Error vs log_g 标量相关性

- **内容**: 计算 mean(error)、std(error)、norm(error) 等标量指标
- **展示**: 散点图 + Pearson/Spearman 相关系数
- **目的**: 验证 error 的全局统计量是否与 log_g 有简单相关性

### 5.3 模型性能对比柱状图

- **内容**: 四组柱状图 (Flux+LightGBM, Flux+Linear, Error+LightGBM, Error+Linear)
- **指标**: R², MAE, RMSE
- **目的**: 直观展示特征类型和模型类型的交互效应

### 5.4 Train vs Test R² 对比（过拟合诊断）

- **内容**: Error-based LightGBM 的 train_R²=0.79 vs test_R²=0.39
- **展示**: 双柱状图或差异指标
- **目的**: 量化过拟合程度

### 5.5 Error 热力图（按 log_g 分组）

- **内容**: 横轴为波长，纵轴为 log_g 区间，颜色为平均 error 值
- **目的**: 可视化不同 log_g 区间的 error 模式差异

---

# 6. 结论（Conclusion）

## 6.1 本实验验证了什么？

| 结论 | 证据 |
|------|------|
| ✅ Error 包含 log_g 的可利用信息 | LightGBM test_R² = 0.392 |
| ✅ Error 与 log_g 的关系是非线性的 | Linear R² ≈ 0, LightGBM R² = 0.39 |
| ⚠️ Error 信息量远低于 Flux | Error R² = 0.39 vs Flux R² = 0.998 |
| ⚠️ 存在过拟合风险 | train_R² = 0.79 >> test_R² = 0.39 |

## 6.2 对模型设计的启示

1. **Error 可作为辅助特征**，但需要非线性模型来提取信息
2. **推荐组合方式**：
   - 直接拼接：$X = [\text{flux}, \text{error}]$（8192 维）
   - SNR 特征：$X_\text{SNR} = \text{flux} / \text{error}$
   - 加权特征：$X_\text{weighted} = \text{flux} / \text{error}^2$（逆方差加权）

3. **神经网络设计**：如果使用 error 特征，需要足够的非线性层

## 6.3 对光谱物理的贡献

- 证实了 **观测噪声模式与恒星演化阶段存在统计关联**
- 这种关联源于：恒星亮度 → 光子数 → 噪声特性的物理链条
- 为理解天文数据中的"隐式信息结构"提供了实证

## 6.4 仍不确定的问题

- Error 中的 log_g 信息是"真物理"还是"数据偏差/采样偏差"？
- 在更多样化的数据集上，R² = 0.39 是否可复现？
- 组合 flux + error 能否超越单独使用 flux 的性能？

---

# 7. 下一步（Next Steps）

## 7.1 验证信息来源（Sanity Checks）

| 实验 | 目的 |
|------|------|
| 计算 error 标量统计量与 log_g 的相关系数 | 验证是否存在简单线性相关 |
| 随机打乱 error（shuffle test） | 验证 R²=0.39 是否来自真实相关 |
| 检查 train/test 划分是否按物理对象划分 | 排除数据泄漏 |

## 7.2 组合特征实验

```python
# 方案 1: 拼接
X_combined = np.concatenate([flux, error], axis=1)  # 8192 维

# 方案 2: SNR
X_snr = flux / error

# 方案 3: 逆方差加权
X_weighted = flux / (error ** 2)
```

## 7.3 特征重要性分析

```python
# 分析 LightGBM 的特征重要性
model = pickle.load(open('models/lgbm_error_test/lgbm_error_nz0.pkl', 'rb'))['model']
importance = model.feature_importances_
top_k_indices = np.argsort(importance)[-50:]  # Top 50 features
```

## 7.4 物理验证

- 检查最重要的 error 波长位置是否对应 log_g 敏感的谱线区域
- 对比 error 热力图与已知的 log_g 诊断谱线位置

---

# 8. 模型文件索引

| Model | Path |
|-------|------|
| LightGBM (error) | `models/lgbm_error_test/lgbm_error_nz0.pkl` |
| Linear (error) | `models/lnreg_e_n32k_test/lnreg_e_n32k/lnreg_e_n32k_nz0.pkl` |

---

*Generated: 2025-11-27*

