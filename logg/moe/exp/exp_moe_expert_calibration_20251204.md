# 📘 📋 Experiment Report: MVP-Next-C Expert Calibration
> **Name:** TODO | **ID:** `VIT-20251204-moe-01`  
> **Topic:** `moe` | **MVP:** MVP-9E | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-04 | **Status:** ✅ Completed
```
💡 实验目的  
决定：影响的决策
```

---


## 🔗 Upstream Links
| Type | Link |
|------|------|
| 🧠 Hub | `logg/moe/moe_hub.md` |
| 🗺️ Roadmap | `logg/moe/moe_roadmap.md` |

---

## ⚡ 核心结论速览

> **一句话总结:** Affine 校准无效甚至略微降低性能 (ΔR² = -0.0013)，表明 metal-poor bins 的误差**不是**简单的系统性 bias/scale 偏移

**假设验证结果:**
| 假设 | 结果 | 说明 |
|------|------|------|
| H-C: Metal-poor 误差来自系统性偏差 | ❌ | 校准后性能下降 |

**关键数字:**
- 整体 ΔR² (cal - uncal): **-0.0013**
- Bin3 (Mid MP) ΔR²: **-0.0083**
- Bin6 (Hot MP) ΔR²: **+0.0015** (微弱)

---

## 🎯 实验设计

### 验证假设 (H-C)
> 最弱 bins (metal-poor) 的误差主要是系统性 bias/scale 不匹配，affine 校准 `y'_k = a_k * y_k + b_k` 能改善

### 方法
1. **复用 MVP-9E1** 的 9 个 Ridge expert 和 9-class LogReg gate
2. **Validation Set 拟合:** 对每个 expert，用 validation 数据拟合 (a, b)
3. **Test Set 评估:** 应用校准，比较 R² 变化

### 验收标准
- 整体 ΔR² > 0.005
- Bin3/Bin6 ΔR² ≥ 0.02

---

## 📊 实验结果

### 整体性能
| 模型 | R² | 备注 |
|------|-----|------|
| Global Ridge | 0.8613 | Baseline |
| MoE (uncalibrated) | **0.9214** | MVP-9E1 结果 |
| MoE (calibrated) | **0.9201** | 校准后反而下降! |
| ΔR² | **-0.0013** | ❌ 不达标 |

### Per-Bin R² 变化
| Bin | 描述 | R² (uncal) | R² (cal) | ΔR² | 状态 |
|-----|------|-----------|----------|-----|------|
| 0 | Cool MP | 0.9225 | 0.9210 | -0.0015 | ❌ |
| 1 | Cool Solar | 0.9674 | 0.9654 | -0.0020 | ❌ |
| 2 | Cool MR | 0.9840 | 0.9842 | +0.0002 | 🟡 |
| **3** | **Mid MP** | **0.7884** | **0.7802** | **-0.0083** | **❌** |
| 4 | Mid Solar | 0.9209 | 0.9221 | +0.0012 | 🟡 |
| 5 | Mid MR | 0.9812 | 0.9806 | -0.0005 | 🟡 |
| **6** | **Hot MP** | **0.8263** | **0.8278** | **+0.0015** | **🟡** |
| 7 | Hot Solar | 0.9427 | 0.9420 | -0.0007 | 🟡 |
| 8 | Hot MR | 0.9678 | 0.9676 | -0.0002 | 🟡 |

### 校准参数分析
| Expert | a (scale) | b (bias) | 偏离程度 |
|--------|-----------|----------|----------|
| 0 (Cool MP) | 1.0520 | -0.1740 | ⚠️ 高偏差 |
| 3 (Mid MP) | 1.0690 | -0.1716 | ⚠️ 高偏差 |
| 6 (Hot MP) | **1.0746** | -0.1665 | ⚠️ **最大 scale 偏差** |
| 其他 | ~1.0 | ~0 | 正常 |

---

## 🔬 深度分析

### 为什么校准失败？

**观察到的现象:**
1. Validation set 上确实观察到 scale 偏差 (a=1.05-1.07) 和 bias (b=-0.17)
2. 但将这些校准应用到 test set 后性能**下降**
3. 这说明偏差在 validation 和 test 之间**不稳定**

**根本原因:**
1. **异质性问题:** Metal-poor bins 内部的误差分布不均匀，不是简单的全局偏移
2. **过拟合风险:** 用少量 validation 样本拟合的校准参数不能泛化
3. **问题本质不同:** 误差可能来自**缺失特征/物理信息**，而非简单的 bias

### Metal-poor Bins 为何难以校准？

| 因素 | 解释 |
|------|------|
| **样本稀疏** | Metal-poor 区域训练样本较少 |
| **物理复杂** | 低金属丰度恒星的光谱特征更弱、更难提取 |
| **特征不足** | 可能需要额外的物理特征来区分 |

---

## 📈 图表

生成了 5 张图表:
1. `moe_calibration_pred_vs_true_per_expert.png` - 9 个 expert 的预测 vs 真值
2. `moe_calibration_params.png` - 校准参数 (a, b) 分布
3. `moe_calibration_per_bin_r2.png` - Per-bin R² 对比
4. `moe_calibration_overall_comparison.png` - 整体 R² 对比
5. `moe_calibration_residual_dist.png` - 残差分布

---

## 💡 设计启示

### 止损判断
- ✅ **达到止损点:** ΔR² < 0.005 → 偏差不是主因

### 下一步方向
| 优先级 | 方向 | 理由 |
|--------|------|------|
| 🔴 P0 | 增强 Metal-poor 特征 | 问题可能在特征不足 |
| 🟡 P1 | 尝试更复杂的 expert | Ridge 可能容量不足 |
| 🟡 P1 | 增加 Metal-poor 训练样本 | 数据稀疏可能是根因 |
| 🟢 P2 | 探索 ensemble 方法 | 降低单一 expert 风险 |

### 对 Roadmap 的影响
- H-C 假设被否定 → **应转向 H-D (特征增强) 或 H-E (模型增强)**
- 简单校准无法解决问题 → 需要更根本的方法改进

---

## 📁 产出文件

| 类型 | 路径 |
|------|------|
| 结果 CSV | `~/VIT/results/moe/calibration/results.csv` |
| Per-bin 结果 | `~/VIT/results/moe/calibration/per_bin_results.csv` |
| 校准参数 | `~/VIT/results/moe/calibration/calibration_params.csv` |
| 图表目录 | `~/Physics_Informed_AI/logg/moe/img/` |

---

## ✅ 成功标准检查

| 检查项 | 状态 |
|--------|------|
| 9 个 expert 的 pred vs true 图已生成 | ✅ |
| 校准参数 (a_k, b_k) 已拟合 | ✅ |
| Per-bin R² 对比图已生成 | ✅ |
| 整体 R² 对比图已生成 | ✅ |
| 报告已写入知识中心 | ✅ |

---

**实验耗时:** 2 分 28 秒
