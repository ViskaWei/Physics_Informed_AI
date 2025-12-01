# MVP 实验实现总结

**日期**: 2025-12-01  
**任务**: Top-K Window + CNN/Transformer & Global Feature Tower + MLP

---

## 1. 完成状态

| 任务 | 状态 | 说明 |
|------|------|------|
| 代码分析 | ✅ 完成 | 确认可复用模块，制定最小改动方案 |
| Top-K Window 模型 | ✅ 完成 | TopKWindowCNN + TopKWindowTransformer |
| Global Feature Tower | ✅ 完成 | GlobalFeatureBuilder + GlobalFeatureMLP |
| 实验脚本 | ✅ 完成 | topk_window_experiments.py + global_feature_experiments.py |
| Sanity Check | ✅ 通过 | 两个实验 pipeline 均验证通过 |
| MVP 实验 | 🔄 运行中 | tmux sessions: `topk_mvp`, `global_mvp` |

---

## 2. 新增文件

### 2.1 模型代码

| 文件 | 用途 | 行数 |
|------|------|------|
| `src/nn/models/topk_window.py` | Top-K Window CNN + Transformer 模型 | ~650 |
| `src/nn/global_features.py` | Global Feature 构建 + MLP 模型 | ~550 |

### 2.2 实验脚本

| 文件 | 用途 |
|------|------|
| `scripts/topk_window_experiments.py` | Top-K Window MVP 实验 |
| `scripts/global_feature_experiments.py` | Global Feature MVP 实验 |

### 2.3 文档

| 文件 | 用途 |
|------|------|
| `logg/gta/exp_topk_window_cnn_transformer_20251201.md` | Top-K Window 实验计划 |
| `logg/gta/exp_global_feature_tower_mlp_20251201.md` | Global Feature 实验计划 |

---

## 3. 复用的模块

| 模块 | 来源 | 用途 |
|------|------|------|
| DataModule | `src/nn/data_adapter.py` | 数据加载，支持 noise 和 residual 模式 |
| CNN1D/MLP | `src/nn/models/` | 基础模型架构参考 |
| train_and_evaluate | `src/nn/baseline_trainer.py` | 训练循环（部分复用） |
| load_best_ridge_model | `src/utils/model_loader.py` | 加载 Ridge 模型 |
| get_model_importance | `src/utils/model_loader.py` | 获取 Top-K indices |

---

## 4. Sanity Check 结果

### 4.1 Top-K Window (10 epochs, noise=0.1)

| 模型 | Test R² | 参数量 | 训练时间 |
|------|---------|--------|----------|
| TopKWindowCNN (K=128) | **0.8382** | 27,873 | 47s |
| TopKWindowTransformer (K=128) | **0.7354** | 17,633 | 51s |

### 4.2 Global Feature (10 epochs, noise=1.0)

| 模型 | Test R² | 参数量 | 训练时间 |
|------|---------|--------|----------|
| GlobalFeatureMLP (PCA+Ridge) | **0.9710** | 4,289 | 14s |

---

## 5. MVP 实验配置

### 5.1 Top-K Window 实验

| 实验 | 模型 | K | noise | lr |
|------|------|---|-------|-----|
| MVP_CNN_K256_nz0 | CNN | 256 | 0.0 | 3e-3 |
| MVP_CNN_K512_nz0 | CNN | 512 | 0.0 | 3e-3 |
| MVP_CNN_K256_nz0p1 | CNN | 256 | 0.1 | 3e-3 |
| MVP_Transformer_K256_nz0 | Transformer | 256 | 0.0 | 3e-4 |
| MVP_Transformer_K256_nz0p1 | Transformer | 256 | 0.1 | 3e-4 |

### 5.2 Global Feature 实验

| 实验 | Features | noise |
|------|----------|-------|
| MVP_Full_nz1p0 | PCA+Ridge+TopK+Error | 1.0 |
| MVP_F1F2_nz1p0 | PCA+Ridge | 1.0 |
| MVP_F1F2F3_nz1p0 | PCA+Ridge+TopK | 1.0 |
| MVP_Full_nz0p1 | PCA+Ridge+TopK+Error | 0.1 |

---

## 6. 运行方法

```bash
# 激活环境
cd /home/swei20/VIT
source init.sh

# === Top-K Window 实验 ===
# Sanity check
python scripts/topk_window_experiments.py --sanity --gpu 0

# 完整 MVP
python scripts/topk_window_experiments.py --gpu 0

# === Global Feature 实验 ===
# Sanity check
python scripts/global_feature_experiments.py --sanity --gpu 0

# 完整 MVP
python scripts/global_feature_experiments.py --gpu 0
```

---

## 7. 结果存储

| 实验 | CSV | 总结 |
|------|-----|------|
| Top-K Window | `results/topk_window/mvp_results.csv` | `results/topk_window/mvp_summary.md` |
| Global Feature | `results/global_features/mvp_results.csv` | `results/global_features/mvp_summary.md` |

---

## 8. 设计亮点

### 8.1 最小改动原则

- **不修改** 现有 DataModule、训练循环
- **新增** 模型类而非修改现有类
- **复用** Ridge 模型加载、importance 提取

### 8.2 Residual on Ridge

所有模型都使用：
```
y_pred = y_ridge + f_theta(features)
```
- 复用线性 baseline 的信息
- 神经网络只需学习残差

### 8.3 模块化设计

- TopK 索引提取可复用于多个模型
- GlobalFeatureBuilder 支持灵活的 feature family 组合
- 实验脚本支持 sanity check、ablation 等多种模式

---

## 9. 下一步

1. **等待 MVP 实验完成**
   - 查看 tmux: `tmux a -t topk_mvp` / `tmux a -t global_mvp`

2. **分析结果**
   - 对比不同 K 值、噪声水平的影响
   - 分析 feature family 的贡献 (ablation)

3. **如果达标**
   - 集成到双塔架构
   - 考虑更多 K 值和 window 大小

---

*最后更新: 2025-12-01*

