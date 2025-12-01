# MVP-Global-1: Global Feature Tower + MLP 实验

- **实验目录**: `/home/swei20/VIT/scripts/global_feature_experiments.py`
- **结果目录**: `/home/swei20/VIT/results/global_features/`
- **创建日期**: 2025-12-01
- **状态**: 🔄 代码已实现，待运行

---

## 1. 实验目标

组装一套 **global feature 向量 g(x)**（约 158 维），在此之上实现 Residual MLP baseline：

- 主任务噪声：`noise=1.0`（附带 `noise=0.1`）
- 预测形式：`y_hat = y_ridge + g_theta(g(x))`

---

## 2. 实现的模块

### 2.1 新增文件

| 文件 | 用途 |
|------|------|
| `src/nn/global_features.py` | Global Feature 构建函数 + GlobalFeatureMLP 模型 |
| `scripts/global_feature_experiments.py` | 实验运行脚本 |

### 2.2 复用的模块

| 模块 | 来源 |
|------|------|
| DataModule | `src/nn/data_adapter.py` |
| Ridge model loader | `src/utils/model_loader.py` |
| PCA | scikit-learn |

---

## 3. Feature Families 设计

### 3.1 特征维度汇总

| Family | 描述 | 维度 | 状态 |
|--------|------|------|------|
| **F1** | PCA Global Shape | 96 | ✅ |
| **F2** | Ridge View | 1 | ✅ |
| **F3** | Top-K Segment Summary | 24 | ✅ |
| **F4** | Error Summary | 5 | ✅ |
| **F5** | Latent Segmented Features | 32 | ⏳ 需要预计算 |
| **总计** | | **158** | |

### 3.2 各 Family 详细说明

#### F1: PCA Global Shape (~96 维)

```python
# 使用 PCA 对 flux 做变换，保留前 96 个主成分
pca = PCA(n_components=96)
f1 = pca.fit_transform(flux)  # (N, 96)
```

#### F2: Linear/Ridge View (1 维)

```python
# Ridge 模型预测值作为特征
# 同时在 Residual 模式下作为 shortcut
f2 = ridge_model.predict(flux)  # (N, 1)
```

#### F3: Top-K Segment Summary (~24 维)

```python
# Top-K wavelengths (K=512) 按波长排序后分成 24 段
# 每段计算 flux mean
topk_sorted = np.sort(topk_indices)
for i in range(24):
    segment = flux[:, topk_sorted[start:end]]
    f3[:, i] = segment.mean(axis=1)
```

#### F4: Noise/Error Summary (~5 维)

```python
# 如果有 per-pixel error σ:
f4 = [mean(σ), std(σ), max(σ), p25(σ), p75(σ)]

# 如果只有统一 noise level N:
f4 = [N, N², 1]  # (3 维)
```

#### F5: Latent Segmented Features (~32 维)

```python
# 复用 distill/ 中的最佳 latent 表示
# enc_pre_latent + seg_mean_K8 (384 维) → 截取前 32 维
latent = load_layer_features("enc_pre_latent")  # (N, 48, 13)
pooled = segment_mean_pool(latent, K=8)  # (N, 384)
f5 = pooled[:, :32]  # (N, 32)
```

---

## 4. 模型架构

### GlobalFeatureMLP

```
输入: features (B, D), ridge_pred (B, 1)
      D ≈ 158 (或根据配置)

1. MLP:
   - Linear(D, 256) + ReLU + Dropout(0.3)
   - Linear(256, 64) + ReLU + Dropout(0.3)
   - Linear(64, 1) → Δy

2. Residual:
   y_pred = ridge_pred + Δy

参数量: ~49K (D=158)
```

---

## 5. 实验配置

### 5.1 MVP 实验列表

| 实验 ID | Features | noise | 目的 |
|---------|----------|-------|------|
| MVP_Full_nz1p0 | F1+F2+F3+F4 | 1.0 | 主实验 |
| MVP_F1F2_nz1p0 | F1+F2 | 1.0 | Ablation: baseline |
| MVP_F1F2F3_nz1p0 | F1+F2+F3 | 1.0 | Ablation: +TopK |
| MVP_Full_nz0p1 | F1+F2+F3+F4 | 0.1 | 低噪声测试 |

### 5.2 训练超参

| 参数 | 值 |
|------|-----|
| Learning rate | 1e-3 |
| Weight decay | 0 |
| Batch size | 2048 |
| Epochs | 100 |
| Early stopping | 20 |
| Optimizer | Adam |
| Dropout | 0.3 |

---

## 6. 运行方法

```bash
# 激活环境
cd /home/swei20/VIT
source init.sh

# Sanity check
python scripts/global_feature_experiments.py --sanity --gpu 0

# 运行所有 MVP 实验
python scripts/global_feature_experiments.py --gpu 0

# 只运行 ablation
python scripts/global_feature_experiments.py --ablation f1f2 --gpu 0
```

---

## 7. 结果存储

- **CSV 结果**: `/home/swei20/VIT/results/global_features/mvp_results.csv`
- **总结报告**: `/home/swei20/VIT/results/global_features/mvp_summary.md`

---

## 8. 与 Baseline 对比

| 模型 | Test R² | 噪声 | 备注 |
|------|---------|------|------|
| LightGBM | 0.536 | 1.0 | 32k 数据 SOTA |
| Residual MLP | 0.498 | 1.0 | - |
| Ridge | 0.458 | 1.0 | 线性 baseline |
| **GlobalFeatureMLP (Full)** | 待测 | 1.0 | 目标 ≥0.50 |

---

## 9. 关键设计决策

1. **Residual on Ridge**: 所有模型都使用 `y_pred = y_ridge + Δy`

2. **PCA 在线拟合**: PCA 模型在训练数据上拟合，然后应用于 val/test

3. **Top-K 来自 Ridge**: 使用 Ridge 系数绝对值排序获取 Top-K 索引

4. **Error Summary 自适应**: 根据是否有 per-pixel error 选择不同的特征构建方式

5. **Latent 可选**: 如果预计算的 latent 特征不可用，可以跳过 F5

---

## 10. 下一步

1. 运行 MVP 实验，获取 baseline 结果
2. 如果 F5 (Latent) 可用，补充实验
3. 根据 ablation 结果，分析各 feature family 的贡献
4. 如果性能达标，考虑集成到双塔架构

---

*最后更新: 2025-12-01*
