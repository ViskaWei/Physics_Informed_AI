# 🧠 GPT 脑暴报告 - $\log g$ 预测实验汇总

---
> **生成日期：** 2025-12-01  
> **覆盖时间：** 2025-11-28 ~ 2025-12-01  
> **目的：** 汇总所有实验结论，进行深度分析和下一步规划

---

# 📌 核心结论汇总

## 🏗️ [GTA] Global Tower Architecture

### 实验 1: F0/F1 元数据 Baseline
- **一句话**: **元数据（Teff + [M/H]）完全无法预测 $\log g$**，$R^2 \approx 0$
- **关键数字**: $R^2 \approx 0$
- **设计启示**: 必须使用光谱 flux，不能只靠 metadata

### 实验 2: Global Feature MLP (126 维)
- **一句话**: **126 维全局特征在 noise=0.1 下达到 $R^2=0.9588$**，证明全局特征高效有效
- **关键数字**: 
  - noise=0.1: $R^2=0.9588$, MAE=0.162
  - noise=1.0: $R^2=0.4883$, MAE=0.656
- **设计启示**: 
  - 低噪声下 Global Tower 即可达到极高性能
  - 高噪声下需要增强（Latent 或双塔融合）

### 实验 3: TopK Window CNN (Local Tower)
- **一句话**: **K=256 + Residual on Ridge 达到 $R^2=0.9313$**，超越所有之前 NN baseline
- **关键数字**: 
  - K=256: $R^2=0.9313$
  - K=512: $R^2=0.72$（性能反而下降）
- **设计启示**: 
  - CNN >> Transformer（小数据+有噪声场景）
  - K=256 最优，更多特征引入噪声
  - Residual on Ridge 降低学习难度

---

## 🧬 [CNN] Dilated Kernel 实验

### 实验: CNN Dilated Kernel Sweep
- **一句话**: **dilation=2 时感受野匹配吸收线宽度，达到最佳 $R^2=0.992$**
- **关键数字**: 
  - dilation=2: $R^2=0.992$ ⭐
  - dilation=1: $R^2=0.985$
  - dilation=4: $R^2=0.978$
- **设计启示**: 感受野 = dilation × (kernel_size - 1) + 1，典型吸收线宽度 ~10-20 像素

---

## 🔮 [Distill] BlindSpot Latent 蒸馏

### 实验 1: Linear Probe (MVP 1.1)
- **一句话**: **Latent 强编码 $T_{\text{eff}}$ ($R^2=0.98$) 和 [M/H] ($R^2=0.96$)，但 $\log g$ 线性信号弱**
- **关键数字**: 
  - $T_{\text{eff}}$: Ridge $R^2=0.9702$, LightGBM $R^2=0.9775$
  - [M/H]: Ridge $R^2=0.9444$, LightGBM $R^2=0.9599$
  - $\log g$: Ridge $R^2=0.2192$, LightGBM $R^2=0.2830$ (+29%)
- **设计启示**: 
  - 数据量是关键：1k 样本 LightGBM 过拟合
  - $\log g$ 需要非线性方法

### 实验 2: Latent 提取优化 (MVP 1.4)
- **一句话**: **`enc_pre_latent + seg_mean_K8` 配置将 $\log g$ 从 $R^2=0.22$ 提升到 $R^2=0.5516$ (+150%)**
- **关键数字**: 
  - Baseline (enc_last + global_mean): $R^2=0.2202$
  - 最佳 (enc_pre_latent + seg_mean_K8): $R^2=0.5516$ (+150%)
- **设计启示**: 
  - **Pooling 是主要瓶颈**：global mean 抹掉了空间信息
  - **波长局部性对 $\log g$ 至关重要**：压力增宽特征集中在特定波段

### 实验 3: Encoder + MLP (MVP 2.2)
- **一句话**: **冻结 Encoder + MLP Head 达到 $R^2=0.6117$**，比 Ridge 提升 10.9%
- **关键数字**: 
  - Ridge baseline: $R^2=0.5516$
  - MLP head: $R^2=0.6117$ (+10.9%)
- **设计启示**: 
  - MLP 能捕捉非线性关系
  - 冻结 encoder 有效，无需 fine-tune

### ⚠️ 待解决: Error 捷径问题
- **发现**: CleanError 单独预测 $\log g$ 可达 $R^2 \approx 0.91$
- **问题**: Latent 的 $\log g$ 信息是来自 error 还是 flux？
- **风险**: Teacher latent 可能走了 "error → latent → log g" 捷径

---

## 📈 [LightGBM] 基础实验

### 实验: Hyperparam Sweep
- **一句话**: **LightGBM 在 noise=1.0 下达到 $R^2=0.536$，是当前高噪声 SOTA**
- **关键数字**: noise=1.0: $R^2=0.536$
- **设计启示**: GBDT 在高噪声下比 MLP 更稳健

---

# 📊 性能对比矩阵

## noise=0.1 场景

| 方法 | 输入 | Test $R^2$ | 参数量 | 备注 |
|------|------|-----------|--------|------|
| **GlobalFeatureMLP** | 126维全局特征 | **0.9588** ⭐ | 49K | 最高 |
| **TopKWindowCNN** | K=256 窗口 | **0.9313** | 28K | 第二 |
| **CNN Dilated (d=2)** | 4096 全谱 | **0.992** | ~30K | 全谱 NN |
| Ridge | 4096 全谱 | 0.909 | - | 线性 baseline |
| Distill MLP | 384 Latent | 0.6117 | - | Encoder 特征 |

## noise=1.0 场景

| 方法 | 输入 | Test $R^2$ | 备注 |
|------|------|-----------|------|
| **LightGBM** | 4096 全谱 | **0.536** | 当前 SOTA |
| Residual MLP | 4096 全谱 | 0.498 | - |
| GlobalFeatureMLP | 126维全局特征 | 0.4883 | 下降明显 |
| Ridge | 4096 全谱 | 0.458 | 线性 baseline |

---

# ✅ 已验证的假设

| 假设 | 结论 | 证据 |
|------|------|------|
| 元数据无法预测 $\log g$ | ✅ 验证 | Teff+[M/H] 给 $R^2 \approx 0$ |
| Grid 采样设计正确 | ✅ 验证 | Teff-$\log g$ 无伪相关 |
| 全局特征在低噪下有效 | ✅ 验证 | $R^2=0.9588$ @ noise=0.1 |
| Latent 强编码全局参数 | ✅ 验证 | $T_{\text{eff}}$ $R^2=0.98$ |
| Mean pooling 丢失空间信息 | ✅ 验证 | seg_mean_K8 提升 +77.6% |
| CNN 优于 Transformer | ✅ 验证 | CNN $R^2$=0.93 >> Transformer |
| K=256 优于 K=512 | ✅ 验证 | 更多特征引入噪声 |
| Residual on Ridge 有效 | ✅ 验证 | 降低学习难度 |

---

# ❓ 待解答的问题

| 问题 | 优先级 | 可能的实验方向 |
|------|--------|---------------|
| **双塔融合效果如何？** | 🔴 高 | Global + Local concat/FiLM |
| **Latent 信息来自 error 还是 flux？** | 🔴 高 | Stage A: Error-only baseline + 残差分析 |
| **noise=1.0 下如何达到 $R^2 \geq 0.50$？** | 🔴 高 | 双塔融合 / Latent 增强 |
| **Fine-tune encoder 能提升多少？** | 🟡 中 | MVP-2.3 |
| **Multi-scale dilation 架构效果？** | 🟡 中 | 组合 dilation=1,2,4 |
| **F2/F3 特征贡献多少？** | 🟡 中 | 测试 mean/std/颜色、EW |

---

# 💡 洞见与模式

## 跨实验发现的规律

1. **空间局部性是关键**：
   - CNN dilation=2 匹配吸收线宽度 → 最优
   - seg_mean_K8 保留空间结构 → +150% 提升
   - K=256 优于 K=512 → 聚焦关键波段

2. **信息冗余现象**：
   - 126 维全局特征 vs 4096 全谱：$R^2$ 差距 <5%
   - 更多特征反而引入噪声（K=512 性能下降）

3. **非线性关系存在**：
   - LightGBM vs Ridge: +29% ($\log g$)
   - MLP vs Ridge: +10.9% (Latent → $\log g$)

4. **噪声鲁棒性差异**：
   - noise 0.1→1.0: Global Feature $R^2$ 从 0.96 降到 0.49
   - GBDT 在高噪声下更稳健

5. **物理解释一致性**：
   - $\log g$ 信息来自压力增宽（Balmer wings）
   - 集中在特定波段，需要保留空间结构

---

# 🤖 给 GPT 的思考提示

基于以上进展，请帮我思考：

1. **结论验证**: 
   - 上述核心结论是否合理？
   - "K=256 优于 K=512" 这个结论是否说明信息冗余？还是可能有其他解释？
   - Error 捷径问题有多严重？如何彻底排除？

2. **下一步建议**: 
   - 最值得尝试的下一个实验是什么？
   - 双塔融合应该用什么策略（concat / FiLM / cross-attention）？
   - 如何设计实验来证明 Latent 的 $\log g$ 信息来自 flux 而非 error？

3. **潜在问题**: 
   - noise=1.0 下 Global Feature 性能大幅下降，可能的原因是什么？
   - PCA 分量在高噪声下是否被污染？
   - 有没有遗漏的 confounding factor？

4. **架构改进**: 
   - Global Tower 应该增加什么特征？
   - Local Tower 的 K 值选择有什么理论指导？
   - 双塔应该如何分工？

5. **物理解释**: 
   - $\log g$ 信息为什么集中在特定波段？
   - dilation=2 最优的物理解释是什么？
   - 为什么 GBDT 在高噪声下比 MLP 更稳健？

---

# 📎 附录

## 实验索引

| experiment_id | topic | 状态 | exp.md |
|---------------|-------|------|--------|
| `VIT-20251201-gta-global-01` | gta | ✅ | [exp_global_feature_tower_mlp](../logg/gta/exp_global_feature_tower_mlp_20251201.md) |
| `VIT-20251201-gta-local-01` | gta | ✅ | [exp_topk_window_cnn_transformer](../logg/gta/exp_topk_window_cnn_transformer_20251201.md) |
| `VIT-20251130-gta-baseline-01` | gta | ✅ | [exp_gta_f0f1_metadata_baseline](../logg/gta/exp_gta_f0f1_metadata_baseline_20251130.md) |
| `VIT-20251201-cnn-dilated-01` | cnn | ✅ | [exp_cnn_dilated_kernel_sweep](../logg/cnn/exp_cnn_dilated_kernel_sweep_20251201.md) |
| `BS-20251130-distill-probe-01` | distill | ✅ | [exp_linear_probe_latent](../logg/distill/exp_linear_probe_latent_20251130.md) |
| `BS-20251201-distill-latent-01` | distill | ✅ | [exp_latent_extraction_logg](../logg/distill/exp_latent_extraction_logg_20251201.md) |
| `BS-20251201-encoder-logg-01` | distill | ✅ | [exp_encoder_nn_logg](../logg/distill/exp_encoder_nn_logg_20251201.md) |
| `VIT-20251129-lightgbm-01` | lightgbm | ✅ | [exp_lightgbm_hyperparam_sweep](../logg/lightgbm/exp_lightgbm_hyperparam_sweep_20251129.md) |

## 待做实验

| experiment_id | topic | 优先级 | 描述 |
|---------------|-------|--------|------|
| `VIT-20251201-gta-fusion-01` | gta | 🔴 P0 | 双塔融合 |
| `BS-20251201-latent-gta-01` | distill | 🔴 P0 | Latent 给 GTA |
| `BS-20251201-distill-finetune-01` | distill | 🟡 P1 | Fine-tune encoder |
| Stage A.1-A.4 | distill | 🟡 P1 | Error 捷径分析 |

---

*生成时间: 2025-12-01*  
*使用方法: 复制全文到 ChatGPT/Claude 进行深度讨论*


