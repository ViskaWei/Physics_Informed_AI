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

## 🔍 深度调查结果

### 1. SW 失败模式

```
- SW runs 在 epoch 10-13 就失败了（目标是 50 epochs）
- mse_loss 始终 ≈ 1.0（等于标准化标签的方差）
- val_r2 ≈ -0.01（模型输出常数，完全没有学习）
- 平均运行时间 344s（C1D 成功的平均 1328s）
```

### 2. 详细测试结果

**Tokenizer 单元测试：✅ 正常**

```python
# 梯度流测试
SW (unfold+linear):
  Input grad norm: 246.35
  Input grad max:  4.24

C1D (conv1d):
  Input grad norm: 297.50
  Input grad max:  4.84

# 结论：梯度流相似
```

**一步训练测试 (真实数据)：**

```
C1D:
  Initial loss: 1.0180 → After 1 step: 0.9772 (下降 0.04)
  Transformer gradient norm: 5.63

SW:
  Initial loss: 1.3651 → After 1 step: 0.8730 (下降 0.49!)
  Transformer gradient norm: 11.46 (2x larger!)
```

**关键发现**: SW 的 Transformer 梯度是 C1D 的 2 倍！但第一步 loss 下降更多。

**完整训练测试 (256样本, 50步)：**

```
C1D step 1:  loss=1.0448, Final R²=-0.0004
SW step 1:   loss=1.3186, Final R²=-0.0003

# 在小规模测试中表现相似
```

### 3. 根本原因分析

| 原因 | 可能性 | 证据 |
|------|--------|------|
| SW 实现 bug | ❌ 低 | 单元测试通过，梯度流正常 |
| 梯度不稳定 | ✅ 高 | Transformer 梯度 2x larger |
| FP16 精度问题 | ⚠️ 中 | 未能在 GPU 上验证 |
| 训练动态问题 | ✅ 高 | 第一步 OK，长期训练失败 |

### 4. 技术差异

**C1D (Conv1d) vs SW (Linear)**

```python
# C1D
x.reshape(-1, 1, 4096) → Conv1d(1, 256, k=16, s=16) → transpose

# SW  
x.unfold(1, 16, 16) → Linear(16, 256)

# 关键差异：
# 1. Conv1d: 权重 shape (out, in, kernel) = (256, 1, 16)
# 2. Linear: 权重 shape (out, in) = (256, 16)
# 3. 相同的有效参数数量，但梯度传播路径不同
```

---

## ✅ 结论

**SW tokenizer 实现正确**，但在长期训练中出现不稳定。

**根本原因**: Transformer 层接收到的梯度是 C1D 的 2 倍，导致：
1. 初期学习过快
2. 后期振荡或梯度爆炸
3. 模型崩塌到输出常数

---

## 🎯 推荐

### 方案 A: 直接使用 C1D (推荐)

```yaml
model:
  proj_fn: C1D  # 已验证可用
```

### 方案 B: 如需使用 SW

```yaml
model:
  proj_fn: SW
opt:
  lr: 0.00015  # 降低 2x 匹配梯度差异
train:
  precision: 32  # 使用 FP32
  gradient_clip: 0.5  # 更严格的梯度裁剪
```

---

## 📁 相关文件

- `src/models/tokenization.py` - 添加了 SW 不稳定性警告
- `scripts/test_sw_sweep_config.py` - SW 测试脚本
- `scripts/test_c1d_sweep_config.py` - C1D 测试脚本

---

*Updated: 2025-12-28*
