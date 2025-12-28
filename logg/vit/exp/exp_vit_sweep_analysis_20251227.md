<!--
📝 Agent 书写规范（不出现在正文）:
- Header 全英文
- 正文中文
- 图表文字全英文（中文会乱码）
- 公式用 LaTeX: $inline$ 或 $$block$$
-->

# 🍃 ViT 1M Scaling Sweep 分析
> **Name:** ViT-1M-Sweep-Analysis  
> **ID:** `VIT-20251227-vit-sweep-01`  
> **Topic:** `vit` | **MVP:** MVP-1.0 | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-27 | **Status:** 🔄  
> **Root:** `logg/vit` | **Parent:** `exp_vit_1m_scaling_20251226` | **Child:** -

> 🎯 **Target:** 分析 wandb sweep 结果，选择最优架构进行 200 epoch 训练  
> 🚀 **Next:** 完成 200 epoch 训练 → 与 LightGBM baseline 比较

## ⚡ 核心结论速览

> **一句话**: Sweep 分析确定最优架构 p16_h256_L6_a8，采用**架构多样性策略**启动 Top 3 配置的 200 epoch 训练

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H1: 最优架构配置? | ✅ p16_h256_L6_a8 | val_R²=0.6619，Best |
| H2: 小 patch vs 大 patch? | ⚖️ 各有优势 | p8 深度好，p16 综合最佳 |
| H3: 架构参数敏感性? | ✅ hidden_size > num_layers | 256 hidden 比 384 更优 |
| H4: 选择策略? | 🎯 架构多样性 | 探索 scaling 行为 > 严格 R² 排名 |

| 指标 | 值 | 启示 |
|------|-----|------|
| Best val_R² | 0.6619 | p16_h256_L6_a8, 10 epochs |
| Sweep 均值 | 0.4731 | 21 runs |
| R² Range | -0.03 ~ 0.66 | 架构选择关键 |

| Type | Link |
|------|------|
| 🧪 WandB Sweep | https://wandb.ai/viskawei-johns-hopkins-university/vit-1m-scaling |
| 📁 Top3 Configs | `~/VIT/configs/exp/vit_200ep_top*.yaml` |
| 📜 Launch Script | `~/VIT/scripts/launch_top3_200ep.sh` |

---

# 1. 🎯 目标

**问题**: 从 21 个 sweep runs 中分析最优 ViT 架构配置

**验证**: 
- H1: 哪种 patch_size/hidden_size/num_layers 组合最优?
- H2: 更大模型是否更好?

| 预期 | 判断标准 |
|------|---------|
| 找到 R² > 0.60 配置 | 通过 → 选择 Top 3 进行 200ep 训练 |

---

# 2. 📊 Sweep 结果分析

## 2.1 Top 10 Runs

| Rank | ID | 配置 | val_R² | val_MAE | Epochs | Params |
|------|----|----|--------|---------|--------|--------|
| 1 | w3a8zlrh | p16_h256_L6_a8_s16 | **0.6619** | 0.4298 | 10 | ~4.9M |
| 2 | c8lw0xbr | p16_h256_L6_a8_s16 | 0.6619 | 0.4298 | 10 | ~4.9M |
| 3 | 6i1vob7q | p16_h256_L4_a8_s16 | 0.6139 | 0.4862 | 50 | 3.296M |
| 4 | 1um1fp6i | p8_h128_L8_a4_s8 | 0.6117 | 0.1370 | 150 | 1.67M |
| 5 | f88vvqsp | p32_h256_L4_a4_s8 | 0.6016 | 0.1415 | 150 | 3.365M |
| 6 | cbu7lkv9 | p16_h256_L4_a8_s16 | 0.5972 | 0.4864 | 50 | 3.296M |
| 7 | gkteocmp | p16_h256_L8_a8_s16 | 0.5956 | 0.4874 | 50 | 6.455M |
| 8 | cgf2ghc0 | p16_h384_L6_a8_s16 | 0.5889 | 0.4948 | 50 | 10.901M |
| 9 | z3y3x9xm | p16_h256_L6_a8_s16 | 0.5814 | 0.4995 | 50 | 4.876M |
| 10 | zc3n2rs3 | p16_h256_L6_a8_s16 | 0.5814 | 0.4995 | 50 | 4.876M |

## 2.2 架构分析

### By Patch Size

| patch_size | n | Best R² | Mean R² | 分析 |
|------------|---|---------|---------|------|
| **8** | 1 | 0.6117 | 0.6117 | 最细粒度，需更深网络 |
| **16** | 13 | **0.6619** | 0.5347 | **综合最优** |
| 32 | 6 | 0.6016 | 0.3066 | 太粗，需重叠 stride |
| 64 | 1 | 0.5335 | 0.5335 | 过粗，信息丢失 |

**结论**: patch_size=16 是最佳选择，平衡了 token 数量和计算效率

### By Hidden Size

| hidden_size | n | Best R² | Mean R² | 分析 |
|-------------|---|---------|---------|------|
| 128 | 3 | 0.6117 | 0.5191 | 小模型，需深网络补偿 |
| **256** | 13 | **0.6619** | 0.5290 | **最优容量** |
| 384 | 5 | 0.5889 | 0.3002 | 过大，可能过拟合 |

**结论**: hidden_size=256 比 384 更优，更大不一定更好

### By Number of Layers

| num_layers | n | Best R² | Mean R² | 分析 |
|------------|---|---------|---------|------|
| 4 | 6 | 0.6139 | 0.5263 | 浅层，适合小 patch |
| **6** | 10 | **0.6619** | 0.4734 | **最优深度** |
| 8 | 5 | 0.6117 | 0.4088 | 更深需要更多 epochs |

**结论**: 6 层是最佳平衡点，4 层过浅，8 层需更长训练

### By Learning Rate

| lr | n | Best R² | Mean R² |
|----|---|---------|---------|
| 0.0001 | 3 | 0.5889 | 0.5330 |
| 0.0002 | 1 | 0.6016 | 0.6016 |
| **0.0003** | 16 | **0.6619** | 0.4452 |
| 0.00045 | 1 | 0.6117 | 0.6117 |

**结论**: lr=0.0003 是最常用且效果最好的选择

---

# 3. 🏆 Top 3 配置选择

## 3.0 选择策略说明

> ⚠️ **重要**: 本次选择采用**架构多样性策略**，而非严格按 val_R² 排名

### 为什么不选严格 Top 3？

按 val_R² 排名，前 3 名是：

| 真实排名 | ID | 配置 | val_R² | 问题 |
|---------|----|----|--------|------|
| #1 | w3a8zlrh | p16_h256_L6 | 0.6619 | ✅ 选入 |
| #2 | c8lw0xbr | p16_h256_L6 | 0.6619 | ❌ 与 #1 重复 |
| #3 | [6i1vob7q](https://wandb.ai/viskawei-johns-hopkins-university/vit-1m-scaling/sweeps/hlshu8vl/runs/6i1vob7q) | p16_h256_L4 | 0.6139 | ⚠️ 与 #1 仅差 2 层 |

如果选严格 Top 3，我们会得到：
- p16_h256_L6 (两次)
- p16_h256_L4

这样的选择**缺乏架构多样性**，无法回答以下问题：
1. 小 patch (p8) 在 200 epochs 后能否赶上 p16？
2. 大 patch (p32) + overlap stride 的 scaling 行为如何？
3. 深度 (L8) vs 宽度 (h256) 的 trade-off？

### 架构多样性策略

| 选择 | 配置 | val_R² | 选择理由 |
|------|------|--------|---------|
| **Top 1** | p16_h256_L6 | 0.6619 | 严格最优，作为 baseline |
| **Top 2** | p8_h128_L8 | 0.6117 | **探索小 patch + 深网络** |
| **Top 3** | p32_h256_L4 | 0.6016 | **探索大 patch + overlap** |

### 科学价值

| 实验对比 | 回答的问题 |
|---------|-----------|
| Top 1 vs Top 2 | patch_size 对特征提取的影响 |
| Top 1 vs Top 3 | stride overlap 能否弥补 large patch 的信息损失 |
| Top 2 vs Top 3 | 深度 vs 宽度 vs token 数量的 trade-off |
| 全部 | 不同架构的 200 epoch scaling 曲线 |

### 被跳过的 6i1vob7q

[6i1vob7q](https://wandb.ai/viskawei-johns-hopkins-university/vit-1m-scaling/sweeps/hlshu8vl/runs/6i1vob7q) (p16_h256_L4, R²=0.6139) 虽然排名 #3，但：
- 与 Top 1 仅差 `num_layers` (4 vs 6)
- 可通过 Top 1 的训练曲线推断其行为
- 不提供额外的架构洞见

**结论**: 选择架构多样性 > 严格 R² 排名，以获得更多 scaling 规律的洞见

---

## 3.1 最优配置选择

基于上述策略，选择以下 3 个**架构多样化**的配置进行 200 epoch 训练：

### Top 1: p16_h256_L6_a8 (Best Overall)

```yaml
# 最佳配置：平衡的中等 patch + 6 层深度
model:
  patch_size: 16
  hidden_size: 256
  num_hidden_layers: 6
  num_attention_heads: 8
  stride_size: 16
opt:
  lr: 0.0003
```

| 指标 | 值 |
|------|-----|
| Sweep val_R² | 0.6619 |
| 参数量 | ~4.9M |
| Sweep Epochs | 10 |
| 预期 200ep R² | 0.72+ |

### Top 2: p8_h128_L8_a4 (Small Patch + Deep)

```yaml
# 小 patch + 深网络：细粒度特征
model:
  patch_size: 8
  hidden_size: 128
  num_hidden_layers: 8
  num_attention_heads: 4
  stride_size: 8
opt:
  lr: 0.00045
```

| 指标 | 值 |
|------|-----|
| Sweep val_R² | 0.6117 |
| 参数量 | 1.67M |
| Sweep Epochs | 150 |
| 特点 | 更多 tokens (512 vs 256) |

### Top 3: p32_h256_L4_a4 (Large Patch + Overlap)

```yaml
# 大 patch + 重叠：计算高效
model:
  patch_size: 32
  hidden_size: 256
  num_hidden_layers: 4
  num_attention_heads: 4
  stride_size: 8  # 重叠 stride
opt:
  lr: 0.0002
```

| 指标 | 值 |
|------|-----|
| Sweep val_R² | 0.6016 |
| 参数量 | 3.365M |
| Sweep Epochs | 150 |
| 特点 | 重叠 patches 增加上下文 |

---

# 4. 🚀 200 Epoch 训练

## 4.1 启动信息

| 配置 | GPU | Config File | Log |
|------|-----|-------------|-----|
| Top 1 (p16_h256_L6) | GPU 4 | `configs/exp/vit_200ep_top1_p16h256L6.yaml` | `results/vit_200ep_top1.log` |
| Top 2 (p8_h128_L8) | GPU 5 | `configs/exp/vit_200ep_top2_p8h128L8.yaml` | `results/vit_200ep_top2.log` |
| Top 3 (p32_h256_L4) | GPU 6 | `configs/exp/vit_200ep_top3_p32h256L4.yaml` | `results/vit_200ep_top3.log` |

**启动时间**: 2025-12-27 16:07:39 EST

```bash
# 启动命令
./scripts/launch_top3_200ep.sh

# 监控
tail -f results/vit_200ep_top*.log
```

## 4.2 训练配置

| 参数 | 值 |
|------|-----|
| Epochs | 200 |
| Batch Size | 256 |
| Optimizer | AdamW |
| LR Scheduler | Cosine |
| Precision | 16-mixed |
| Data | 200k training samples |
| Noise Level | 1.0 |

---

# 5. 🔍 特定 Run 分析: ix08tgwq

此 run 是用户指定的分析对象：

| 属性 | 值 |
|------|-----|
| Run ID | ix08tgwq |
| Name | ViT_p32_h384_l4_a8_s16_pC1D_nz1 |
| State | **failed** (在 epoch 11 后) |
| Runtime | 287 秒 |

## 配置

```yaml
model:
  patch_size: 32
  hidden_size: 384
  num_hidden_layers: 4
  num_attention_heads: 8
  stride_size: 16
  proj_fn: C1D
opt:
  lr: 0.0001
  weight_decay: 0.01
train:
  epochs: 50
  batch_size: 256
loss:
  name: mse
data:
  num_samples: 50000
  label_norm: standard
```

## 结果 (Epoch 11)

| 指标 | 值 |
|------|-----|
| val_R² | 0.4428 |
| val_MAE | 0.6057 |
| val_MSE | 0.5754 |
| num_params | 7.358M |

## 分析

**失败原因可能**:
1. **hidden_size=384 过大** - 分析显示 h256 比 h384 更优
2. **patch_size=32 without overlap stride** - stride=16 (overlap) 但 patch=32 可能信息不足
3. **lr=0.0001 可能过小** - 最优 runs 使用 lr=0.0003

**对比 Top 配置**:
| 对比项 | ix08tgwq | Top 1 (w3a8zlrh) |
|--------|----------|------------------|
| patch_size | 32 | 16 |
| hidden_size | 384 | 256 |
| num_layers | 4 | 6 |
| lr | 0.0001 | 0.0003 |
| val_R² | 0.4428 | **0.6619** |

**结论**: ix08tgwq 的配置不在最优区域，p32_h384 组合表现较差

---

# 6. 💡 洞见

## 6.1 架构设计

| 发现 | 结论 |
|------|------|
| **patch_size=16 最优** | 平衡 token 数量和计算效率 |
| **hidden_size=256 > 384** | 更大容量不一定更好 |
| **6 层 Transformer 最佳** | 4 层不足，8 层需更多 epochs |
| **lr=0.0003 效果最好** | 比 0.0001 更快收敛 |

## 6.2 Scaling 规律

```
更大模型 ≠ 更好性能
h384 (10.9M params) < h256 (4.9M params)
```

**原因分析**:
1. 数据量 (50k-200k) 可能不足以支撑 10M+ 参数
2. 过大模型容易过拟合或需更多 epochs
3. 架构匹配 > 单纯增大参数

## 6.3 Patch Size 影响

| patch_size | Token 数 | 感受野 | 适用场景 |
|------------|----------|--------|----------|
| 8 | 512 | 局部 | 细粒度特征，需深网络 |
| 16 | 256 | 中等 | **通用最优** |
| 32 | 128 | 全局 | 需 stride overlap |
| 64 | 64 | 超全局 | 信息丢失 |

---

# 7. 📝 结论

## 7.1 核心发现

> **最优 ViT 配置: p16_h256_L6_a8, val_R²=0.6619**

| # | 结论 | 证据 |
|---|------|------|
| 1 | **patch_size=16 最优** | Best R²=0.6619 vs p8/p32 |
| 2 | **hidden_size=256 > 384** | Mean 0.53 vs 0.30 |
| 3 | **lr=0.0003 最佳** | 16/21 runs 使用 |
| 4 | **6 层深度平衡** | Best in num_layers |
| 5 | **架构多样性策略** | 选择 p8/p16/p32 三种 patch 探索 scaling |

## 7.2 关键数字

| 指标 | 值 |
|------|-----|
| Sweep Best R² | **0.6619** |
| Sweep Mean R² | 0.4731 |
| 最优配置 | p16_h256_L6_a8 |
| 最优 lr | 0.0003 |
| 最优参数量 | ~4.9M |
| 被跳过的 6i1vob7q | R²=0.6139 (真实 #3，与 Top 1 架构相似) |
| Top 2 vs 6i1vob7q | Δ=-0.0022 (架构多样性换取) |
| Top 3 vs 6i1vob7q | Δ=-0.0123 (架构多样性换取) |

## 7.3 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| 200ep 训练完成 | 等待 Top 3 configs 完成 | 🔴 |
| 结果对比 | 分析 3 个配置的最终 R² | 🔴 |
| vs LightGBM | 与 baseline 比较 | 🟡 |
| 更多数据 | 扩展到 1M 样本 | 🟢 |

---

# 8. 📎 附录

## 8.1 Sweep 原始数据

| ID | Name | val_R² | patch | hidden | layers | lr |
|----|------|--------|-------|--------|--------|-----|
| w3a8zlrh | ViT-1M-L6-H256 | 0.6619 | 16 | 256 | 6 | 0.0003 |
| c8lw0xbr | ViT-1M-L6-H256 | 0.6619 | 16 | 256 | 6 | 0.0003 |
| 6i1vob7q | ViT_p16_h256_l4 | 0.6139 | 16 | 256 | 4 | 0.0003 |
| 1um1fp6i | ViT_p8_h128_l8 | 0.6117 | 8 | 128 | 8 | 0.00045 |
| f88vvqsp | ViT_p32_h256_l4 | 0.6016 | 32 | 256 | 4 | 0.0002 |
| cbu7lkv9 | ViT_p16_h256_l4 | 0.5972 | 16 | 256 | 4 | 0.0003 |
| gkteocmp | ViT_p16_h256_l8 | 0.5956 | 16 | 256 | 8 | 0.0003 |
| cgf2ghc0 | ViT_p16_h384_l6 | 0.5889 | 16 | 384 | 6 | 0.0001 |
| z3y3x9xm | ViT_p16_h256_l6 | 0.5814 | 16 | 256 | 6 | 0.0003 |
| zc3n2rs3 | ViT_p16_h256_l6 | 0.5814 | 16 | 256 | 6 | 0.0003 |

## 8.2 执行记录

```bash
# 分析 sweep 结果
python -c "import wandb; api = wandb.Api(); ..."

# 创建配置文件
vim configs/exp/vit_200ep_top1_p16h256L6.yaml
vim configs/exp/vit_200ep_top2_p8h128L8.yaml
vim configs/exp/vit_200ep_top3_p32h256L4.yaml

# 启动训练
./scripts/launch_top3_200ep.sh
# PIDs: 857138, 857140, 857142
# GPUs: 4, 5, 6
# Start: 2025-12-27 16:07:39 EST
```

## 8.3 配置文件位置

| File | Description |
|------|-------------|
| `configs/exp/vit_200ep_top1_p16h256L6.yaml` | Top 1 配置 |
| `configs/exp/vit_200ep_top2_p8h128L8.yaml` | Top 2 配置 |
| `configs/exp/vit_200ep_top3_p32h256L4.yaml` | Top 3 配置 |
| `scripts/launch_top3_200ep.sh` | 启动脚本 |

---

> **实验完成时间**: 2025-12-27 (训练进行中)
