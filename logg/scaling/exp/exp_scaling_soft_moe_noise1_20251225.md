# Soft-gate MoE @ noise=1

> experiment_id: SCALING-20251223-soft-moe-noise1-01 | date: 2025-12-25 | status: ✅ 完成

## ⚡ 核心结论速览
| 项目 | 内容 |
|------|------|
| **一句话总结** | Soft-gate MoE 在 noise=1 下成功保留 80.5% Oracle 收益，ρ=0.8052 远超 0.70 阈值 |
| **假设验证** | ✅ H-A2.1: ρ ≥ 0.70 已验证 |
| **关键数字** | ρ=0.8052, R²_soft=0.5930, ΔR²=+0.1318 vs Global |
| **设计启示** | 37维gate特征+LogReg足够实现有效路由，Soft MoE 可作为 noise=1 主线方案 |

## 1. 目标
验证 Soft-gate MoE 在 noise=1 高噪声条件下能保留多少 Oracle 收益。

**核心问题**：
- 用可训练的 gate (LogReg) 替代真值路由，评估可落地性
- ρ = (R²_soft - R²_global) / (R²_oracle - R²_global) ≥ 0.70 → 成功

## 2. 实验设计
### 2.1 数据
| 配置项 | 值 |
|--------|-----|
| Source | BOSZ simulated spectra (mag205_225_lowT_1M) |
| Train | 1M samples (5 shards × 200k) |
| Test | 1k samples (**使用 pre-stored noisy**) |
| Feature dim | 4096 (MR arm) |
| Target | log_g ∈ [1.0, 5.0] |
| Noise | gaussian, level=1.0 |

### 2.2 模型架构
| 组件 | 配置 |
|------|------|
| Experts | 9 × Ridge (α=100000), 每个bin单独训练后冻结 |
| Gate | LogisticRegression 9-class (multinomial) |
| Gate Features | 37 dim (10 PCA + 27 physical windows) |
| Mixing | Soft: y = Σ p_k * y_k |

### 2.3 Gate 特征 (37维)
```
PCA: 10 维 (全光谱 shape)
Ca II Triplet: 9 维 (8498, 8542, 8662 Å: mean/depth/ew × 3)
Na I Doublet: 6 维 (8183, 8195 Å: mean/depth/ew × 2)
Mg I Triplet: 9 维 (8806, 8807, 8808 Å: mean/depth/ew × 3)
Combined EW: 3 维 (CaII_EW_total, NaI_EW_total, MgI_EW_total)
```

## 3. 图表
### 3.1 R² Comparison
![soft_moe_comparison.png](../img/soft_moe_comparison.png)
**观察**: Soft-gate MoE (R²=0.5930) 显著优于 Global Ridge (R²=0.4611)，接近 Oracle MoE (R²=0.6249)

### 3.2 ρ Bar Chart
![soft_moe_rho_bar.png](../img/soft_moe_rho_bar.png)
**观察**: ρ=0.8052 >> 0.70 阈值，成功保留 80.5% Oracle 收益

### 3.3 Gate Probability Heatmap
![soft_moe_gate_probs.png](../img/soft_moe_gate_probs.png)
**观察**: Gate 对角线概率高，证明 37 维特征有效区分 9 个 bin

### 3.4 Per-bin R² Comparison
![soft_moe_perbin_r2.png](../img/soft_moe_perbin_r2.png)
**观察**: 各 bin 的 Soft MoE 性能接近 Oracle，Metal-poor bins 略差

### 3.5 Pred vs True (colored by bin)
![soft_moe_pred_vs_true.png](../img/soft_moe_pred_vs_true.png)
**观察**: 预测分布均匀，无明显 bias

### 3.6 Routing Confusion Matrix
![soft_moe_routing_confusion.png](../img/soft_moe_routing_confusion.png)
**观察**: 86.0% routing accuracy，混淆主要在相邻 bins 之间

## 4. 洞见
1. **Soft routing 优于 Hard routing**: ρ_soft=0.8052 > ρ_hard=0.7696
   - Soft mixing 的概率加权平滑了路由误差
2. **37 维 gate 特征足够**: 86.0% routing accuracy 在高噪声下仍表现良好
3. **保留 80% Oracle 收益**: 远超 70% 阈值，说明 soft-gate 方案可行性很高

## 5. 结论
### 5.1 假设验证
| Hypothesis | Threshold | Actual | Result |
|------------|-----------|--------|--------|
| H-A2.1 | ρ ≥ 0.70 | **0.8052** | ✅ PASS |

### 5.2 与已知 baseline 对比
| Model | R² | ΔR² vs Global |
|-------|-----|---------------|
| Global Ridge | 0.4611 | - |
| LightGBM (estimated) | ~0.57 | +0.11 |
| **Soft-gate MoE** | **0.5930** | **+0.1318** |
| Oracle MoE | 0.6249 | +0.1637 |

### 5.3 决策
✅ **Soft MoE 可作为 noise=1 主线方案**

**下一步**: 
- 可选：尝试 Regression Gate (MSE loss 端到端训练) 进一步优化
- 主线：将 Soft-gate MoE 纳入 scaling 曲线分析

## 6. 附录
### 6.1 数值结果
| Metric | Value |
|--------|-------|
| R²_global | 0.4611 |
| R²_soft | 0.5930 |
| R²_hard | 0.5871 |
| R²_oracle | 0.6249 |
| ρ_soft | **0.8052** |
| ρ_hard | 0.7696 |
| Routing Acc | 86.0% |

### 6.2 执行日志
```
cd ~/VIT && source init.sh && python scripts/scaling_soft_moe_noise1.py
# 执行时间: 19.7 分钟
# 输出: results/scaling_soft_moe/
```

### 6.3 训练样本分布
| Bin | Teff | [M/H] | Train | Test |
|-----|------|-------|-------|------|
| 0 | Cool | Metal-poor | 88666 | 93 |
| 1 | Cool | Solar | 89050 | 85 |
| 2 | Cool | Metal-rich | 62702 | 71 |
| 3 | Mid | Metal-poor | 102801 | 89 |
| 4 | Mid | Solar | 102864 | 111 |
| 5 | Mid | Metal-rich | 71516 | 62 |
| 6 | Hot | Metal-poor | 116727 | 117 |
| 7 | Hot | Solar | 115778 | 126 |
| 8 | Hot | Metal-rich | 80717 | 81 |

### 6.4 数据一致性说明
⚠️ **重要**: 本实验使用 test_1k_0 文件中预存的 `noisy` 数据作为测试输入，
与 Oracle MoE 实验 (MVP-16A-0) 保持一致。这确保了 baseline 数字可比:
- Global Ridge R² = 0.4611 ✓ (与 Oracle MoE 实验一致)
- Oracle MoE R² = 0.6249 ✓ (与 Oracle MoE 实验一致)
