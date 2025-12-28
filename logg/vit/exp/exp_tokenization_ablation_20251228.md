# 🔬 Tokenization Ablation Study - 调查报告

> **Name:** Tokenization-Ablation-Investigation  
> **ID:** `VIT-20251228-tokenization-ablation-02`  
> **Topic:** `vit` | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-28 | **Status:** ✅ 调查完成
> **Root:** `logg/vit` | **Parent:** `exp_vit_sweep_hlshu8vl_20251227.md`

---

## 📊 问题摘要

Sweep `hlshu8vl` 中 **SW (Sliding Window) 的 15 个 runs 全部失败**，而 C1D 有 29% 的成功率。

| Method | Total | Finished | Failed | Success Rate |
|--------|-------|----------|--------|--------------|
| **C1D** | 79 | 23 | 52 | **29%** |
| **SW** | 15 | 0 | 15 | **0%** |

---

## 🔍 调查结果

### 1. SW 失败模式

```
- SW runs 在 epoch 10-13 就失败了（目标是 50 epochs）
- mse_loss 始终 ≈ 1.0（等于标准化标签的方差）
- val_r2 ≈ -0.01（模型输出常数，完全没有学习）
- 平均运行时间 344s（C1D 成功的平均 1328s）
```

### 2. Tokenizer 实现验证

**单独测试 tokenizer 功能：✅ 正常**

```python
# 测试结果
- SW 和 C1D 输出形状相同
- 梯度正常传播
- 初始化相似
```

**完整模型训练测试：✅ 两者都能学习**

```
# 小规模测试 (256 input, 50 epochs)
C1D Final R²: 0.92
SW Final R²:  0.86
差异: 0.06 (可接受)
```

### 3. 可能原因分析

| 原因 | 可能性 | 说明 |
|------|--------|------|
| SW 实现 bug | ❌ 低 | 单独测试通过 |
| 梯度爆炸 | ⚠️ 中 | 初始梯度 SW 略大 |
| FP16 兼容性 | ⚠️ 中 | sweep 使用 16-mixed |
| 超参数不适配 | ✅ 高 | lr=0.0003 可能对 SW 太大 |
| 数据规模问题 | ⚠️ 中 | 4096 input + 256 patches |

### 4. 关键差异

**C1D (Conv1d) vs SW (Linear)**

```
C1D: x.reshape(-1, 1, 4096) → Conv1d → (batch, 256, hidden)
SW:  x.unfold(1, 16, 16) → Linear → (batch, 256, hidden)

主要差异：
1. Conv1d 有共享权重（同一个 kernel 扫描所有位置）
2. Linear 每个位置独立但参数相同
3. 梯度流动路径不同
```

---

## ✅ 结论

**SW tokenizer 实现本身没有问题**，但在当前 sweep 配置下失败。

可能的解决方案：
1. **降低学习率**: lr=0.0001 或更低
2. **使用 FP32**: 避免 16-mixed 精度问题
3. **添加 LayerNorm**: 在 tokenizer 输出后添加 normalization
4. **梯度裁剪**: 防止梯度爆炸

---

## 🎯 建议的后续实验

```yaml
# 建议配置
model:
  proj_fn: SW
  patch_size: 16
  hidden_size: 256
opt:
  lr: 0.0001  # 降低 3x
train:
  precision: 32  # 使用 FP32
  gradient_clip: 1.0  # 添加梯度裁剪
```

---

## 📁 生成的图表

- `results/tokenization_ablation/tokenization_ablation_combined.png`
- `results/tokenization_ablation/ablation_c1d_vs_sw.png`
- `results/tokenization_ablation/ablation_patch_size.png`

---

*Generated: 2025-12-28*
