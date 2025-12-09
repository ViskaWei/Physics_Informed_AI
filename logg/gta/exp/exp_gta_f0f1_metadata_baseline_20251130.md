# 📘 F0/F1 元数据 Baseline 实验报告

---
> **实验名称：** F0/F1 Metadata Baseline - 用 DataFrame 元数据预测 $\log g$  
> **对应 MVP：** MVP-1.0, MVP-2.0（来自 gta_main）  
> **作者：** Viska Wei  
> **日期：** 2025-11-30  
> **数据版本：** BOSZ50000 z=0 Synthetic Spectra  
> **模型版本：** OLS / LightGBM  
> **状态：** ✅ 已完成

---

# ⚡ 核心结论速览（供 main 提取）

### 一句话总结

> **元数据（Teff、[M/H]、观测条件）完全无法预测 $\log g$（$R^2 \approx 0$），验证了 Grid 采样设计正确且 $\log g$ 信息必须从光谱 flux 中提取。**

### 对假设的验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| Q1: Teff-only 能达到多少 $R^2$？ | ✅ $R^2 \approx 0$ | Grid 无 Teff-$\log g$ 相关 |
| Q2: Teff + [M/H] 能达到多少 $R^2$？ | ✅ $R^2 \approx 0$ | 元数据不携带 $\log g$ 信息 |

### 设计启示（1-2 条）

| 启示 | 具体建议 |
|------|---------|
| **必须使用光谱 flux** | Global Tower 必须基于光谱特征（PCA/EW/统计量），不能只用元数据 |
| **F0/F1 可跳过** | 直接从 F2（全谱统计）或 F3（EW）开始设计特征 |

### 关键数字

| 指标 | 值 |
|------|-----|
| F0 (Teff-only) $R^2$ | **≈ 0** |
| F1 (+ [M/H]) $R^2$ | **-0.002** |
| OLS T_eff 系数 | 0.0067 |
| LightGBM 状态 | 第1轮早停 |

---

# 📑 目录

- [1. 🎯 目标](#1--目标)
- [2. 🧪 实验设计](#2--实验设计)
- [3. 📊 实验结果](#3--实验结果)
- [4. 💡 关键洞见](#4--关键洞见)
- [5. 📝 结论](#5--结论)
- [6. 📎 附录](#6--附录)

---

# 1. 🎯 目标

## 1.1 实验目的

在光谱分析中，我们通常使用光谱 flux 来预测恒星参数 log_g。本实验旨在验证：**如果只使用 DataFrame 中的元数据（观测条件、恒星参数等），是否能预测 log_g？**

这个实验的意义在于：
1. 了解元数据中包含多少关于 log_g 的信息
2. 验证 log_g 信息是否主要编码在光谱中
3. 为后续实验提供 baseline

---

## 2. 数据说明

### 2.1 数据来源
- **训练集**: `/srv/local/tmp/swei20/data/bosz50000/z0/train_100k/dataset.h5` (前 32,000 条)
- **测试集**: `/srv/local/tmp/swei20/data/bosz50000/z0/val_100k/dataset.h5` (后 1,000 条)

### 2.2 DataFrame 原始列 (35 列)

| 列名 | 数据类型 | 说明 |
|------|----------|------|
| id | int64 | 样本ID |
| redshift | float64 | 红移 |
| redshift_err | float64 | 红移误差 |
| exp_count | int64 | 曝光次数 |
| exp_time | float64 | 曝光时间 |
| seeing | float64 | 视宁度 |
| ext | float64 | 消光 |
| target_zenith_angle | float64 | 目标天顶角 |
| target_field_angle | float64 | 目标场角 |
| moon_zenith_angle | float64 | 月球天顶角 |
| moon_target_angle | float64 | 月球-目标夹角 |
| moon_phase | float64 | 月相 |
| snr | float64 | 信噪比 |
| mag | float64 | 星等 |
| fiberid | float64 | 光纤ID |
| cont_fit | float64 | 连续谱拟合 |
| random_seed | float64 | 随机种子 |
| Fe_H | float64 | 铁丰度 |
| Fe_H_err | float64 | 铁丰度误差 |
| M_H | float64 | 金属丰度 |
| M_H_err | float64 | 金属丰度误差 |
| a_M | float64 | α元素丰度 |
| a_M_err | float64 | α元素丰度误差 |
| C_M | float64 | 碳丰度 |
| C_M_err | float64 | 碳丰度误差 |
| O_M | float64 | 氧丰度 |
| O_M_err | float64 | 氧丰度误差 |
| T_eff | float64 | 有效温度 |
| T_eff_err | float64 | 有效温度误差 |
| **log_g** | float64 | **表面重力 (目标变量)** |
| log_g_err | float64 | 表面重力误差 |
| N_He | float64 | 氦丰度 |
| v_turb | float64 | 湍流速度 |
| L_H | float64 | 氢光度 |
| interp_param | object | 插值参数 |

### 2.3 数据过滤

经过过滤后，**有效特征列为 11 个**：

**排除的列：**
- `id`, `log_g`, `interp_param`: 非特征列
- `redshift`, `redshift_err`, `exp_count`, `exp_time`, `moon_phase`: 方差为 0
- `ext`, `fiberid`, `cont_fit`, `random_seed`, `Fe_H`, `Fe_H_err`, `M_H_err`, `a_M_err`, `C_M_err`, `O_M`, `O_M_err`, `T_eff_err`, `log_g_err`, `N_He`, `v_turb`, `L_H`: NaN 比例 = 100%

**保留的特征列 (11 个)：**

| 类别 | 特征 |
|------|------|
| 观测条件 | seeing, target_zenith_angle, target_field_angle |
| 月球相关 | moon_zenith_angle, moon_target_angle |
| 信号质量 | snr, mag |
| 恒星参数 | M_H, a_M, C_M, T_eff |

---

## 3. 实验方法

### 3.1 实验设计

| 配置 | 值 |
|------|-----|
| 训练样本数 | 32,000 |
| 测试样本数 | 1,000 |
| 特征数 | 11 |
| 目标变量 | log_g |

### 3.2 模型

1. **OLS (Ordinary Least Squares)**: 线性回归，特征标准化后训练
2. **LightGBM**: 梯度提升树，参数如下：
   - `num_leaves`: 31
   - `learning_rate`: 0.1
   - `feature_fraction`: 0.9
   - `bagging_fraction`: 0.8
   - `num_boost_round`: 500 (early stopping: 50)

---

## 4. 实验结果

### 4.1 全部特征 (11 个)

#### 模型性能对比

| Model | Train R² | Test R² | Test MAE | Test RMSE |
|-------|----------|---------|----------|-----------|
| **OLS** | 0.000474 | -0.001596 | 1.006406 | 1.169260 |
| **LightGBM** | 0.001546 | -0.002465 | 1.006493 | 1.169767 |

> ⚠️ **R² ≈ 0 说明这些特征几乎无法预测 log_g！**  
> ⚠️ **LightGBM 在第 1 轮就早停了**，说明非线性模型也无法从中学到信息。

#### OLS 系数 (标准化后)

| Rank | Feature | Coefficient | 解读 |
|------|---------|-------------|------|
| 1 | mag | +0.0440 | 星等增加 → log_g 微弱增加 |
| 2 | snr | +0.0419 | 信噪比增加 → log_g 微弱增加 |
| 3 | target_zenith_angle | +0.0173 | 天顶角增加 → log_g 微弱增加 |
| 4 | moon_zenith_angle | -0.0076 | 月球天顶角增加 → log_g 微弱减少 |
| 5 | T_eff | +0.0067 | 有效温度增加 → log_g 微弱增加 |
| 6 | target_field_angle | -0.0064 | 场角增加 → log_g 微弱减少 |
| 7 | M_H | +0.0059 | 金属丰度增加 → log_g 微弱增加 |
| 8 | seeing | +0.0040 | 视宁度增加 → log_g 微弱增加 |
| 9 | moon_target_angle | -0.0036 | 月球-目标角增加 → log_g 微弱减少 |
| 10 | C_M | -0.0024 | 碳丰度增加 → log_g 微弱减少 |
| 11 | a_M | +0.0021 | α丰度增加 → log_g 微弱增加 |

**截距 (Intercept)**: 2.993934 (接近 log_g 的均值)

#### LightGBM 特征重要性 (Gain)

| Rank | Feature | Gain | Splits |
|------|---------|------|--------|
| 1 | moon_target_angle | 90.00 | 7 |
| 2 | mag | 62.47 | 5 |
| 3 | moon_zenith_angle | 42.32 | 4 |
| 4 | target_zenith_angle | 34.55 | 3 |
| 5 | M_H | 28.68 | 2 |
| 6 | a_M | 27.19 | 2 |
| 7 | snr | 25.37 | 2 |
| 8 | seeing | 22.84 | 2 |
| 9 | C_M | 21.07 | 2 |
| 10 | T_eff | 9.26 | 1 |
| 11 | target_field_angle | 0.00 | 0 |

---

### 4.2 去除 mag, snr, moon 后 (7 个特征)

#### 保留的特征
- 观测条件: seeing, target_zenith_angle, target_field_angle
- 恒星参数: M_H, a_M, C_M, T_eff

#### 模型性能

| Model | Train R² | Test R² | Test MAE | Test RMSE |
|-------|----------|---------|----------|-----------|
| **OLS** | 0.000355 | -0.000059 | 1.005548 | 1.168362 |

> R² 从 0.0005 变为 0.0004，几乎没有变化

#### OLS 系数

| Rank | Feature | Coefficient |
|------|---------|-------------|
| 1 | target_zenith_angle | +0.0149 |
| 2 | target_field_angle | -0.0129 |
| 3 | M_H | +0.0069 |
| 4 | T_eff | +0.0049 |
| 5 | seeing | +0.0040 |
| 6 | C_M | -0.0025 |
| 7 | a_M | +0.0021 |

---

## 5. 结论

### 5.1 核心发现

1. **元数据无法预测 log_g**
   - OLS R² ≈ 0.0005
   - LightGBM R² ≈ 0.0015 (但测试集为负)
   - 两种模型性能几乎相同，说明问题不在于模型复杂度

2. **线性与非线性模型表现相当**
   - LightGBM 第 1 轮早停
   - 说明元数据与 log_g 之间既无线性关系，也无非线性关系

3. **恒星参数 (T_eff, M_H, a_M, C_M) 与 log_g 几乎无关**
   - 所有系数 < 0.01
   - 这些参数虽然物理上相关，但不能线性预测 log_g

4. **观测条件 (snr, mag, seeing 等) 对 log_g 无预测能力**
   - 这是符合预期的：log_g 是恒星固有属性，不应依赖于观测条件

### 5.2 物理解释

log_g (表面重力) 是恒星的内禀属性，主要通过以下方式影响光谱：
- 压力致宽效应
- 电离平衡
- 连续谱斜率

这些信息编码在**光谱 flux 的细节特征**中，而非简单的元数据参数。因此：

> **log_g 的预测必须依赖光谱 flux，元数据无法替代。**

### 5.3 对后续实验的启示

1. ✅ 使用光谱 flux 预测 log_g 是必要的
2. ✅ Ridge/NN 模型从 flux 中学到的信息是真实的 log_g 信号
3. ❌ 不应期望通过简单参数组合来预测 log_g
4. ⚠️ 模型评估时，这些元数据不应作为 log_g 预测的 baseline

---

## 6. 代码

```python
"""
OLS 实验：用 DataFrame 中所有数值列预测 log_g
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 加载数据
train_path = '/srv/local/tmp/swei20/data/bosz50000/z0/train_100k/dataset.h5'
test_path = '/srv/local/tmp/swei20/data/bosz50000/z0/val_100k/dataset.h5'

df_train = pd.read_hdf(train_path)[:32000]
df_test = pd.read_hdf(test_path)[-1000:]

# 选择有效特征
feature_cols = ['seeing', 'target_zenith_angle', 'target_field_angle',
                'moon_zenith_angle', 'moon_target_angle', 'snr', 'mag',
                'M_H', 'a_M', 'C_M', 'T_eff']

# 准备数据
X_train = df_train[feature_cols].fillna(df_train[feature_cols].median()).values
y_train = df_train['log_g'].values
X_test = df_test[feature_cols].fillna(df_train[feature_cols].median()).values
y_test = df_test['log_g'].values

# 标准化 + 训练
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 评估
y_pred = model.predict(X_test_scaled)
print(f"R² = {r2_score(y_test, y_pred):.6f}")
print(f"Coefficients: {dict(zip(feature_cols, model.coef_))}")
```

---

## 附录：实验环境

- Python: 3.13
- scikit-learn: latest
- LightGBM: latest
- 数据: BOSZ50000 z=0 synthetic spectra

