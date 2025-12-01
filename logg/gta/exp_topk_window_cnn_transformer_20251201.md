# MVP-Local-1: Top-K Window + CNN / Transformer 实验

- **实验目录**: `/home/swei20/VIT/scripts/topk_window_experiments.py`
- **结果目录**: `/home/swei20/VIT/results/topk_window/`
- **创建日期**: 2025-12-01
- **状态**: 🔄 代码已实现，待运行

---

## 1. 实验目标

在现有 log_g pipeline 上，实现 **Top-K window 模型**：

- 从 Ridge 权重中选取 Top-K 重要波长
- 在每个重要波长周围提取 window (±8 像素，共 17 像素宽)
- 使用 CNN 或 Transformer 编码这些 windows
- 预测形式：**Residual on Ridge**: `y_hat = y_ridge + f_theta(TopKWindows(x))`

---

## 2. 实现的模块

### 2.1 新增文件

| 文件 | 用途 |
|------|------|
| `src/nn/models/topk_window.py` | Top-K Window CNN 和 Transformer 模型定义 |
| `scripts/topk_window_experiments.py` | 实验运行脚本 |

### 2.2 复用的模块

| 模块 | 来源 |
|------|------|
| DataModule | `src/nn/data_adapter.py` |
| Ridge model loader | `src/utils/model_loader.py` |
| Training utilities | `src/nn/baseline_trainer.py` |

---

## 3. 模型架构

### 3.1 TopKWindowCNN

```
输入: flux (B, 4096), ridge_pred (B, 1)

1. 提取 Top-K windows: (B, 4096) → (B, K, W)
   - K = 256 或 512
   - W = 17 (window_radius=8)

2. 局部 window CNN (共享权重):
   - Conv1d(1, 16, kernel=3) + BN + ReLU
   - Conv1d(16, 32, kernel=3) + BN + ReLU
   - AdaptiveAvgPool1d(1)
   → 每个 window 得到 32-d embedding
   → reshape: (B, K, 32)

3. 全局 aggregator:
   - reshape: (B, 32, K)
   - Conv1d(32, 64, kernel=3) + BN + ReLU
   - Conv1d(64, 64, kernel=3) + BN + ReLU
   - AdaptiveAvgPool1d(1)
   → (B, 64)

4. MLP head:
   - Linear(64, 32) + ReLU + Dropout(0.2)
   - Linear(32, 1) → Δy

输出: y_pred = ridge_pred + Δy

参数量: ~28K
```

### 3.2 TopKWindowTransformer

```
输入: flux (B, 4096), ridge_pred (B, 1)

1. 提取 Top-K windows: (B, 4096) → (B, K, W)

2. 局部 window CNN (同上):
   → (B, K, 32)

3. Transformer:
   - Linear(32, 64) 升维
   - 波长位置编码
   - 2 层 TransformerEncoder (d_model=64, nhead=4)
   - Mean pooling: (B, K, 64) → (B, 64)

4. MLP head (同上):
   → Δy

输出: y_pred = ridge_pred + Δy

参数量: ~74K
```

---

## 4. 实验配置

### 4.1 MVP 实验列表

| 实验 ID | 模型 | K | noise | lr | 预期目标 |
|---------|------|---|-------|-----|---------|
| MVP_CNN_K256_nz0 | TopKWindowCNN | 256 | 0.0 | 3e-3 | R² ≥ 0.99 |
| MVP_CNN_K512_nz0 | TopKWindowCNN | 512 | 0.0 | 3e-3 | R² ≥ 0.99 |
| MVP_CNN_K256_nz0p1 | TopKWindowCNN | 256 | 0.1 | 3e-3 | R² ≥ 0.70 |
| MVP_Transformer_K256_nz0 | TopKWindowTransformer | 256 | 0.0 | 3e-4 | R² ≥ 0.99 |
| MVP_Transformer_K256_nz0p1 | TopKWindowTransformer | 256 | 0.1 | 3e-4 | R² ≥ 0.65 |

### 4.2 训练超参

| 参数 | CNN | Transformer |
|------|-----|-------------|
| Learning rate | 3e-3 | 3e-4 |
| Weight decay | 0 | 0 |
| Batch size | 2048 | 2048 |
| Epochs | 100 | 100 |
| Early stopping | 20 | 20 |
| Optimizer | AdamW | AdamW |

---

## 5. 运行方法

```bash
# 激活环境
cd /home/swei20/VIT
source init.sh

# Sanity check (快速验证)
python scripts/topk_window_experiments.py --sanity --gpu 0

# 运行所有 MVP 实验
python scripts/topk_window_experiments.py --gpu 0

# 只运行 CNN 实验
python scripts/topk_window_experiments.py --model cnn --gpu 0

# 只运行 Transformer 实验
python scripts/topk_window_experiments.py --model transformer --gpu 0
```

---

## 6. 结果存储

- **CSV 结果**: `/home/swei20/VIT/results/topk_window/mvp_results.csv`
- **总结报告**: `/home/swei20/VIT/results/topk_window/mvp_summary.md`

---

## 7. 与 Baseline 对比

| 模型 | Test R² | 噪声 | 备注 |
|------|---------|------|------|
| Ridge | 0.909 | 0.1 | 线性 baseline |
| 小 kernel CNN (k=9) | 0.657 | 0.1 | 当前最优 NN |
| Residual MLP | 0.498 | 1.0 | - |
| **TopKWindowCNN** | 待测 | 0.1 | 目标 ≥0.70 |
| **TopKWindowTransformer** | 待测 | 0.1 | 目标 ≥0.65 |

---

## 8. 关键设计决策

1. **Residual on Ridge**: 所有模型都使用 `y_pred = y_ridge + Δy`，复用线性 baseline 的信息

2. **共享权重 local encoder**: 所有 K 个 window 共享同一个 CNN 编码器，减少参数

3. **波长位置编码**: Transformer 使用基于实际波长位置的 sinusoidal encoding

4. **Window 边界处理**: 使用 zero-padding 处理边界 window

---

*最后更新: 2025-12-01*
