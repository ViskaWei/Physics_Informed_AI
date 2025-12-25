# 📘 子实验报告：Error 信息分解与 log g 残差分析
> **Name:** TODO | **ID:** `VIT-20251201-error-01`  
> **Topic:** `error` | **MVP:** MVP-X.X | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-01 | **Status:** 🔄
```
💡 实验目的  
决定：影响的决策
```

---


## 🔗 Upstream Links
| Type | Link |
|------|------|
| 🧠 Hub | `logg/error/error_hub.md` |
| 🗺️ Roadmap | `logg/error/error_roadmap.md` |

---

---

# ⚡ 核心结论速览（供 main 提取）

> **本节是给 main.md 提取用的摘要，实验完成后第一时间填写。**

### 一句话总结

> **TODO：实验完成后填写**

### 对假设的验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| Error-only 对 log g 的预测能力？ | ⏳ 待验证 | - |
| Flux 在去除 error 贡献后还能提供多少信息？ | ⏳ 待验证 | - |
| BlindSpot latent 的 log g 信息来自 error 还是 flux？ | ⏳ 待验证 | - |

### 设计启示（1-2 条）

| 启示 | 具体建议 |
|------|---------|
| TODO | TODO |

### 关键数字

| 指标 | 值 |
|------|-----|
| Error-only $R^2$ | ⏳ |
| Flux-only 残差 $R^2$ | ⏳ |
| Error → Latent → log g 路径贡献 | ⏳ |

---

# 0. 🔥 问题背景与动机

## 0.1 核心问题：Error 里的 "捷径" 信息

### 发现

在之前的实验中发现：

> **用 CleanError 单独做线性回归预测 $\log g$，$R^2 \approx 0.91$**

这意味着 **error 本身就强烈编码了 $\log g$ 信息**。

### 问题

BlindSpot Denoiser 的 latent/激活里看到的 $\log g$，很可能是沿着 **"error → 内部表征 → log g"** 这条捷径来的，而不是 flux 真正提供的。

```
                     ┌─────────────────────────────────────────┐
                     │           可能的信息流动路径             │
                     └─────────────────────────────────────────┘

    实际路径（我们希望的）：
    ┌───────┐                                      ┌───────┐
    │ Flux  │ ──────────► [物理特征] ─────────────►│ log g │
    └───────┘                                      └───────┘

    捷径路径（我们担心的）：
    ┌───────┐                                      ┌───────┐
    │ Error │ ──────────► [BlindSpot Latent] ─────►│ log g │
    └───────┘                 ↑                    └───────┘
                              │
                     (error 本身就编码了 log g)
```

## 0.2 需要纠正的一个直觉

> ❌ **错误直觉**：网络要还原光谱，所以"应该有些层能 100% 还原 log g，不然无法还原光谱"

> ✅ **正确理解**：
> - 网络可以完美还原光谱，而内部**从来没有一个"显式的 log g 标量"**
> - 它只需要学到一个从 `noisy flux + error → clean flux` 的函数
> - 不必在某个地方"算出 log g 再去生成"

### 核心思维转变

我们现在要做的**不是**"找到那个 100% log g 的 neuron"。

而是要系统地回答：

> **在 error 已经这么强的前提下，flux 还能贡献多少 $\log g$ 信息？网络有没有学到 flux → log g 的那一部分？**

## 0.3 两个核心问题

| # | 核心问题 | 说明 |
|---|---------|------|
| **Q1** | 信息来源分解 | $\log g$ 的信息到底来自哪儿？error？flux？两者叠加？ |
| **Q2** | 可靠预测建模 | 在不依赖"错误的特权信息"的前提下，怎样设计一个真正可靠的 $\log g$ 网络？ |

---

# 1. 🎯 目标

## 1.1 实验目的

> **Stage A：把 "error 本身的 log g 信息" 定量拆干净**

**回答的问题**：

1. Error-only 对 $\log g$ 的预测能力（建立 baseline）
2. 定义并计算 $\log g$ 的"残差"（去除 error 贡献后的剩余信息）
3. 用残差分析 BlindSpot latent 是否真的学到了 flux → log g 的信息

**对应的设计目标**：

- 为后续所有 probe/蒸馏实验提供一个**干净的 baseline**
- 区分"利用 error 捷径"和"真正利用 flux 信息"

## 1.2 Stage A 实验路线图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Stage A: 信息来源分解                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   A1. Error-only Baseline                                           │
│   ├── 输入: error 向量 (4096 维)                                    │
│   ├── 输出: log g 预测                                              │
│   ├── 模型: Ridge / LightGBM                                        │
│   └── 得到: R²_error_only, ŷ_logg,error-only                        │
│         │                                                           │
│         ▼                                                           │
│   A2. 残差定义                                                       │
│   ├── r = y_logg - ŷ_logg,error-only                                │
│   ├── 这个残差 r 是 error 信息贡献之外的部分                          │
│   └── 后续实验可以用"预测残差 r"替代"预测 log g"                      │
│         │                                                           │
│         ▼                                                           │
│   A3. Flux-only 残差预测                                             │
│   ├── 输入: flux 向量 (4096 维)                                     │
│   ├── 输出: 残差 r                                                   │
│   ├── 模型: Ridge / LightGBM                                        │
│   └── 验证: flux 是否能预测 error 未能解释的 log g 变异              │
│         │                                                           │
│         ▼                                                           │
│   A4. BlindSpot Latent 残差预测                                      │
│   ├── 输入: BlindSpot latent (最佳配置)                              │
│   ├── 输出: 残差 r                                                   │
│   ├── 对比: 直接预测 log g vs 预测残差 r                              │
│   └── 判断: latent 的信息来自 error 捷径还是 flux 真实贡献            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 1.3 预期结果

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| A1 正常 | Error-only $R^2 \approx 0.91$ | 验证已知结果 |
| A3 flux 有贡献 | Flux → 残差 $R^2 > 0.1$ | flux 确实提供了 error 之外的信息 |
| A3 flux 无贡献 | Flux → 残差 $R^2 \approx 0$ | flux 不提供额外信息，或 error 已包含全部 |
| A4 latent 走捷径 | Latent → 残差 $R^2 \approx 0$ | latent 只学到了 error → log g |
| A4 latent 有真实贡献 | Latent → 残差 $R^2 > 0$ | latent 学到了 flux → log g 的信息 |

---

# 2. 🧪 实验设计

## 2.1 数据

| 配置项 | 值 |
|--------|-----|
| 训练样本数 | 100k |
| 测试样本数 | 1k（独立） |
| 特征维度 | 4096（波长点数） |
| 标签参数 | $\log g$ |
| 数据来源 | BOSZ mag215 |

**保持与之前实验一致的 train/test 划分，确保可比性。**

## 2.2 特征设计

| 特征类型 | 维度 | 说明 |
|---------|------|------|
| **CleanError** | 4096 | per-pixel 误差向量 |
| **Noisy Flux** | 4096 | 带噪光谱 |
| **Clean Flux** | 4096 | 无噪声光谱（如可用） |
| **BlindSpot Latent** | 384 | `enc_pre_latent + seg_mean_K8` 配置 |

### 2.2.1 残差的数学定义

$$
r_i = y_{\log g, i} - \hat{y}_{\log g, i}^{\text{error-only}}
$$

其中：
- $y_{\log g, i}$：第 $i$ 个样本的真实 $\log g$ 值
- $\hat{y}_{\log g, i}^{\text{error-only}}$：仅使用 error 预测的 $\log g$ 值

**物理含义**：残差 $r$ 代表 **error 信息无法解释的 $\log g$ 变异**。如果 flux 或 latent 能预测这个残差，说明它们提供了 error 之外的信息。

### 2.2.2 方差分解视角

$$
\text{Var}(\log g) = \text{Var}(\hat{y}^{\text{error}}) + \text{Var}(r) + 2\text{Cov}(\hat{y}^{\text{error}}, r)
$$

如果 error-only 模型拟合良好，则 $\text{Cov}(\hat{y}^{\text{error}}, r) \approx 0$，因此：

$$
\text{Var}(\log g) \approx \text{Var}(\hat{y}^{\text{error}}) + \text{Var}(r)
$$

- $R^2_{\text{error-only}} \approx 0.91$ → error 解释了 91% 的方差
- 剩余 9% 的方差在残差 $r$ 中

## 2.3 模型与算法

### A1: Error-only Baseline

$$
\hat{y}_{\log g}^{\text{error}} = f_{\text{error}}(\text{CleanError})
$$

**模型选择**：
- Ridge Regression（$\alpha = 0.001$）
- LightGBM（num_leaves=15, n_estimators=50）

**选择泛化最好的模型作为 baseline**。

### A2: 残差计算

```python
# 在 test 集上
y_pred_error = model_error.predict(X_error_test)
residual = y_logg_test - y_pred_error
```

### A3: Flux-only 残差预测

$$
\hat{r} = f_{\text{flux}}(\text{Flux})
$$

用 flux 预测残差 $r$，而不是直接预测 $\log g$。

### A4: Latent 残差预测

$$
\hat{r} = f_{\text{latent}}(\text{BlindSpot Latent})
$$

对比：
- **直接预测**：Latent → $\log g$
- **残差预测**：Latent → 残差 $r$

## 2.4 超参数配置

| 参数 | 范围/值 | 说明 |
|------|--------|------|
| Ridge $\alpha$ | 0.001 | 保持一致性 |
| LightGBM num_leaves | 15 | 保守配置 |
| LightGBM n_estimators | 50 | 保守配置 |
| Train size | 100k | 避免小样本过拟合 |

## 2.5 评价指标

| 指标 | 公式 | 用途 |
|------|------|------|
| $R^2$ | $1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$ | 主要评价指标 |
| MAE | $\frac{1}{n}\sum\|y - \hat{y}\|$ | 绝对误差参考 |
| 残差 $R^2$ | $1 - \frac{\sum(r - \hat{r})^2}{\sum(r - \bar{r})^2}$ | 残差预测能力 |

### 2.5.1 解读残差 $R^2$ 的意义

| 残差 $R^2$ | 含义 |
|-----------|------|
| $\approx 0$ | 模型无法预测 error 之外的信息 |
| $> 0.3$ | 模型能捕获显著的 error 之外信息 |
| $> 0.5$ | 模型在 error 之外有强预测能力 |

---

# 3. 📊 实验图表

> 实验完成后添加图表。

### 图 1：Error-only vs Full 预测对比

TODO: `logg/distill/img/error_only_vs_full.png`

**Figure 1. Error-only 预测 vs 真实 $\log g$ 散点图**

**关键观察**：
- TODO

---

### 图 2：残差分布

TODO: `logg/distill/img/residual_distribution.png`

**Figure 2. 残差 $r = y - \hat{y}^{\text{error}}$ 的分布**

**关键观察**：
- TODO

---

### 图 3：Flux/Latent 残差预测对比

TODO: `logg/distill/img/residual_prediction_comparison.png`

**Figure 3. 不同输入对残差的预测能力对比**

**关键观察**：
- TODO

---

### 图 4：信息来源分解饼图

TODO: `logg/distill/img/info_source_decomposition.png`

**Figure 4. $\log g$ 信息来源分解**

**关键观察**：
- TODO

---

# 4. 💡 关键洞见

> 实验完成后填写。

## 4.1 宏观层洞见

> 用于指导架构设计、理解问题本质的高层次发现。

- TODO

## 4.2 模型层洞见

> 用于优化模型、调参的中层次发现。

- TODO

## 4.3 实验层细节洞见

> 具体的实验观察和技术细节。

- TODO

---

# 5. 📝 结论

> 实验完成后填写。

## 5.1 核心发现

> TODO

## 5.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | TODO | TODO |
| 2 | TODO | TODO |
| 3 | TODO | TODO |

## 5.3 设计启示

### 架构/方法原则

| 原则 | 建议 | 原因 |
|------|------|------|
| TODO | TODO | TODO |

### ⚠️ 常见陷阱

| 常见做法 | 实验证据 |
|----------|----------|
| "直接用 latent 预测 log g 就是在用 flux 信息" | TODO: 需要验证是否走了 error 捷径 |

## 5.4 物理解释

> TODO: 为什么 error 会编码 log g 信息？

**可能的物理机制**：

1. **信噪比与恒星参数的相关性**：
   - 表面重力 $\log g$ 影响光谱线的压力增宽
   - 不同 $\log g$ 的恒星可能有系统性的信噪比差异

2. **观测选择效应**：
   - 某些 $\log g$ 范围的恒星更难/更容易观测
   - 导致 error 分布与 $\log g$ 相关

3. **误差估计方法**：
   - 如果误差估计依赖于光谱本身的某些特征
   - 而这些特征又与 $\log g$ 相关

## 5.5 关键数字速查

| 指标 | 值 | 配置/条件 |
|------|-----|----------|
| Error-only $R^2$ | ⏳ | Ridge/LightGBM |
| 残差方差占比 | ⏳ | $1 - R^2_{\text{error}}$ |
| Flux → 残差 $R^2$ | ⏳ | - |
| Latent → 残差 $R^2$ | ⏳ | - |
| Latent → log g $R^2$ | ~0.55 | 已知 baseline |

## 5.6 下一步工作

| 方向 | 具体任务 | 优先级 | 对应 Stage |
|------|----------|--------|---------|
| Stage B | 蒸馏实验中用残差替代 log g | 🔴 高 | B |
| Stage B | 验证 Student 是否学到 flux→log g | 🔴 高 | B |
| Ablation | 打乱 error 后的 latent 表现 | 🟡 中 | A.5 |

---

# 6. 📎 附录

## 6.1 数值结果表

> 实验完成后填写。

### A1: Error-only Baseline

| 模型 | $R^2$ | MAE | RMSE | 备注 |
|------|-------|-----|------|------|
| Ridge | ⏳ | ⏳ | ⏳ | |
| LightGBM | ⏳ | ⏳ | ⏳ | |

### A2: 残差统计

| 统计量 | 值 |
|--------|-----|
| Mean(r) | ⏳ |
| Std(r) | ⏳ |
| 残差方差占总方差比例 | ⏳ |

### A3: Flux → 残差

| 模型 | 残差 $R^2$ | 残差 MAE | 备注 |
|------|-----------|----------|------|
| Ridge | ⏳ | ⏳ | |
| LightGBM | ⏳ | ⏳ | |

### A4: Latent → 残差 vs Latent → log g

| 预测目标 | 模型 | $R^2$ | 备注 |
|---------|------|-------|------|
| log g（直接） | Ridge | ~0.55 | 已知 baseline |
| 残差 r | Ridge | ⏳ | 新实验 |
| log g（直接） | LightGBM | ⏳ | |
| 残差 r | LightGBM | ⏳ | 新实验 |

---

## 6.2 实验代码框架

### A1: Error-only Baseline

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
import lightgbm as lgb

# 加载数据
train_data = torch.load('evals/latent_probe_train_100k.pt')
test_data = torch.load('evals/latent_probe_test_1k.pt')

# 提取 error
X_error_train = train_data['error'].numpy()  # (100k, 4096)
X_error_test = test_data['error'].numpy()    # (1k, 4096)
y_train = train_data['logg'].numpy()
y_test = test_data['logg'].numpy()

# Ridge
ridge = Ridge(alpha=0.001)
ridge.fit(X_error_train, y_train)
y_pred_ridge = ridge.predict(X_error_test)
r2_ridge = r2_score(y_test, y_pred_ridge)

# LightGBM
lgbm = lgb.LGBMRegressor(num_leaves=15, n_estimators=50, random_state=42)
lgbm.fit(X_error_train, y_train)
y_pred_lgbm = lgbm.predict(X_error_test)
r2_lgbm = r2_score(y_test, y_pred_lgbm)

print(f"Error-only Ridge R²: {r2_ridge:.4f}")
print(f"Error-only LightGBM R²: {r2_lgbm:.4f}")

# 选择更好的模型作为 baseline
if r2_lgbm > r2_ridge:
    y_pred_error = y_pred_lgbm
    print("Using LightGBM as error baseline")
else:
    y_pred_error = y_pred_ridge
    print("Using Ridge as error baseline")
```

### A2: 残差计算

```python
# 计算残差
residual_test = y_test - y_pred_error

# 残差统计
print(f"Residual mean: {residual_test.mean():.4f}")
print(f"Residual std: {residual_test.std():.4f}")
print(f"Residual var / Total var: {residual_test.var() / y_test.var():.4f}")

# 也需要在训练集上计算残差（用于后续模型训练）
y_pred_error_train = model_error.predict(X_error_train)
residual_train = y_train - y_pred_error_train
```

### A3: Flux → 残差

```python
# 提取 flux
X_flux_train = train_data['noisy_flux'].numpy()
X_flux_test = test_data['noisy_flux'].numpy()

# Ridge: Flux → 残差
ridge_flux = Ridge(alpha=0.001)
ridge_flux.fit(X_flux_train, residual_train)
r_pred_ridge = ridge_flux.predict(X_flux_test)
r2_flux_ridge = r2_score(residual_test, r_pred_ridge)

# LightGBM: Flux → 残差
lgbm_flux = lgb.LGBMRegressor(num_leaves=15, n_estimators=50)
lgbm_flux.fit(X_flux_train, residual_train)
r_pred_lgbm = lgbm_flux.predict(X_flux_test)
r2_flux_lgbm = r2_score(residual_test, r_pred_lgbm)

print(f"Flux → Residual Ridge R²: {r2_flux_ridge:.4f}")
print(f"Flux → Residual LightGBM R²: {r2_flux_lgbm:.4f}")
```

### A4: Latent → 残差

```python
# 提取 latent (最佳配置: enc_pre_latent + seg_mean_K8)
# 假设已经有提取好的 latent 特征
latent_train = train_data['latent_best'].numpy()  # (100k, 384)
latent_test = test_data['latent_best'].numpy()    # (1k, 384)

# Latent → log g (直接)
ridge_latent_direct = Ridge(alpha=0.001)
ridge_latent_direct.fit(latent_train, y_train)
y_pred_direct = ridge_latent_direct.predict(latent_test)
r2_direct = r2_score(y_test, y_pred_direct)

# Latent → 残差
ridge_latent_residual = Ridge(alpha=0.001)
ridge_latent_residual.fit(latent_train, residual_train)
r_pred_latent = ridge_latent_residual.predict(latent_test)
r2_residual = r2_score(residual_test, r_pred_latent)

print(f"Latent → log g (direct) R²: {r2_direct:.4f}")
print(f"Latent → Residual R²: {r2_residual:.4f}")

# 判断信息来源
if r2_residual > 0.1:
    print("✅ Latent 学到了 error 之外的 flux 信息")
else:
    print("⚠️ Latent 可能主要在利用 error 捷径")
```

---

## 6.3 判断逻辑总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                        判断 Latent 信息来源                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   情况 1: Latent → 残差 R² ≈ 0                                      │
│   └── 结论: Latent 的 log g 信息主要来自 error 捷径                   │
│   └── 启示: 蒸馏 Student 时需要特别小心                               │
│                                                                     │
│   情况 2: Latent → 残差 R² > 0 且 ≈ Flux → 残差 R²                   │
│   └── 结论: Latent 保留了 flux 中的 log g 信息                        │
│   └── 启示: 蒸馏是有价值的，能迁移 flux 信息                          │
│                                                                     │
│   情况 3: Latent → 残差 R² > Flux → 残差 R²                          │
│   └── 结论: Latent 通过非线性变换增强了 flux 的 log g 信号             │
│   └── 启示: 表示学习提供了额外价值                                    │
│                                                                     │
│   情况 4: Latent → 残差 R² < Flux → 残差 R²                          │
│   └── 结论: Latent 编码过程丢失了 flux 的部分 log g 信息               │
│   └── 启示: 考虑直接用 flux 或改进 latent 提取方式                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6.4 相关文件

| 类型 | 路径 | 说明 |
|------|------|------|
| 主框架 | `logg/distill/distill_main_20251130.md` | main 文件 |
| 本报告 | `logg/distill/exp_error_info_decomposition_20251201.md` | 当前文件 |
| 图表 | `logg/distill/img/` | 实验图表 |
| Latent 特征 (100k train) | `evals/latent_probe_train_100k.pt` | 训练数据 |
| Latent 特征 (1k test) | `evals/latent_probe_test_1k.pt` | 测试数据 |
| 层激活 (train) | `evals/layer_features_train_100k.pt` | 多层特征 |
| 层激活 (test) | `evals/layer_features_test_1k.pt` | 多层特征 |

---

## 6.5 后续 Stage B 预览

在 Stage A 完成后，Stage B 将：

1. **用残差替代 log g 作为蒸馏目标**：
   - Student 不是学预测 log g，而是学预测残差 $r$
   - 这样可以强制 Student 学习 flux → log g 的那部分信息

2. **对比实验**：
   - Student（预测 log g）vs Student（预测残差）
   - 看哪种方式能更好地迁移 flux 信息

3. **设计新的 loss function**：
   $$
   \mathcal{L} = \lambda_1 \| \hat{r} - r \|^2 + \lambda_2 \| z_{\text{student}} - z_{\text{teacher}} \|^2
   $$

---

> **模板使用说明**：
> 
> **工作流程**：
> 1. **实验前**：已填写 §0（动机）、§1（目标）、§2（实验设计）
> 2. **实验中**：记录结果到 §6.1，添加图表到 §3
> 3. **实验后**：填写 §4（洞见）、§5（结论）
> 4. **最后**：填写"核心结论速览"，同步到 main.md

