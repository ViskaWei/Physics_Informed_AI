# 📘 子实验报告：Encoder + NN for log_g 预测

---
> **实验名称：** Encoder + NN End-to-End Training for log_g  
> **对应 MVP：** MVP-2.2 (Student latent → log_g)  
> **作者：** Viska Wei  
> **日期：** 2025-12-01  
> **数据版本：** mag215 100k train / 1k val  
> **模型版本：** BlindSpot m215l9e48k25s1bn1d1ep5000  
> **状态：** 🔄 进行中

---

# 📑 目录

- [1. 🎯 目标](#1--目标)
- [2. 🧪 实验设计](#2--实验设计)
- [3. 📊 实验图表](#3--实验图表)
- [4. 💡 关键洞见](#4--关键洞见)
- [5. 📝 结论](#5--结论)
- [6. 📎 附录](#6--附录)

---

# ⚡ 核心结论速览（供 main 提取）

> **本节是给 main.md 提取用的摘要，实验完成后第一时间填写。**

### 一句话总结

> **完成了从预训练 BlindSpot encoder 到 log_g 预测的完整 end-to-end 训练框架实现，冻结 encoder + MLP head 初步达到 val $R^2 = 0.47$，待进一步调优可逼近 Ridge baseline ($R^2 = 0.55$)。**

### 对假设的验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| End-to-end NN 能否利用 encoder 特征预测 log_g？ | ✅ val $R^2 = 0.47$ | 可行，MLP head 学到有意义的映射 |
| 冻结 encoder 是否足够？ | ⏳ 待验证 | 当前结果略低于 Ridge probe，需测试 fine-tune |
| NN head vs Ridge 对比如何？ | ⚠️ 差距 0.08 | MLP 可能需更多训练或调参 |

### 设计启示（1-2 条）

| 启示 | 具体建议 |
|------|---------|
| **框架完整可用** | 已有完整的训练 pipeline，可直接复用进行实验 |
| **配置灵活** | 支持 encoder 层选择、pooling 策略、head 架构的组合扫描 |

### 关键数字

| 指标 | 值 |
|------|-----|
| **最佳 val $R^2$** | 0.4699 (epoch 3) |
| **Ridge baseline** | 0.5516 |
| **特征维度** | 384 (enc_pre_latent + seg_mean_K8) |
| **冻结参数量** | ~500K |

---

# 1. 🎯 目标

## 1.1 实验目的

> 从 main.md 的 MVP 设计中提取，明确本实验要回答的具体问题。

**回答的问题**：
- 能否用 end-to-end 训练的 NN head 替代离线 Ridge 回归？
- 冻结 encoder + 可训练 MLP head 的性能上限是多少？
- 训练框架是否正确实现、可以复现？

**对应 main.md 的**：
- 验证问题：Q5, Q6
- 子假设：H3（Student latent 可学习性）

**核心动机**：
之前的离线 probe 实验（Ridge 回归）已验证 `enc_pre_latent + seg_mean_K8` 配置可达 $R^2 = 0.55$。本实验目标是实现一个完整的端到端训练框架，为后续 fine-tuning 和更复杂的 head 设计奠定基础。

## 1.2 预期结果

> 在实验前写下预期，实验后对照。

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| 正常情况 | val $R^2 \geq 0.50$ | 接近 Ridge baseline (0.55) |
| 可接受情况 | val $R^2 \in [0.40, 0.50)$ | 框架正确但需调优 |
| 异常情况 | val $R^2 < 0.30$ | 需检查实现 bug |

---

# 2. 🧪 实验设计

## 2.1 数据

| 配置项 | 值 |
|--------|-----|
| 训练样本数 | 100,000 |
| 验证样本数 | 1,000 |
| 测试样本数 | 1,000 |
| 特征维度 | 4,096（波长点） |
| Encoder 输出 | 48 channels × 8 segments = 384 维 |
| 标签参数 | $\log g$ |

**数据路径**：
- Train: `/datascope/subaru/user/swei20/data/bosz50000/test/mag215/train_100k/dataset.h5`
- Val: `/datascope/subaru/user/swei20/data/bosz50000/mag215/train_1k/dataset.h5`
- Test: `/datascope/subaru/user/swei20/data/bosz50000/mag215/val_1k/dataset.h5`

**噪声模型**：

$$
\text{noisy\_flux} = \text{flux} + \mathcal{N}(0, \sigma^2 \cdot \text{noise\_level}^2)
$$

**Noise level**: $\sigma = 1.0$

## 2.2 特征设计

| 特征类型 | 维度 | 说明 |
|---------|------|------|
| Encoder feature map | (B, 48, L') | `enc_pre_latent` 层输出 |
| Pooled features | (B, 384) | `seg_mean_K8` pooling |

**特征提取细节**：
1. 输入 noisy flux + error 到 BlindSpot encoder
2. 使用 `encode_flux()` 接口提取 `enc_pre_latent` 层
3. 应用 `seg_mean_K8` pooling 转换为固定维度向量

## 2.3 模型与算法

### Encoder（预训练，可选冻结）

```
BlindspotModel1D:
  - num_layers: 9
  - embed_dim: 48
  - kernel_size: 25
  - input_sigma: true
  - blindspot: true
  - use_bn: true
```

**Checkpoint**: `evals/m215l9e48k25s1bn1d1ep5000.ckpt`

### Log_g Head（MLP）

```python
class LogGHead(nn.Module):
    # architecture = 'mlp_1'
    net = nn.Sequential(
        nn.Linear(384, 256),      # input_dim -> hidden_dim
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, 1),        # hidden_dim -> output
    )
```

### 训练损失

$$
\mathcal{L} = \text{MSE}(\hat{y}_{\log g}, y_{\log g}) = \frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2
$$

## 2.4 超参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Batch size | 256 | |
| Learning rate | 1e-3 | AdamW |
| Weight decay | 1e-4 | |
| Max epochs | 100 | |
| Early stopping patience | 15 | monitor: val/r2 |
| LR scheduler | ReduceLROnPlateau | factor=0.5, patience=5 |
| Gradient clip | 0.5 | |
| Dropout | 0.1 | |
| Hidden dim | 256 | MLP head |

## 2.5 评价指标

| 指标 | 公式 | 用途 |
|------|------|------|
| $R^2$ | $1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$ | 主要评价指标 |
| RMSE | $\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}$ | 绝对误差 |
| MAE | $\frac{1}{n}\sum\|y - \hat{y}\|$ | 鲁棒误差 |

---

# 3. 📊 实验图表

### 表 1：训练进度（version_2 运行）

| Epoch | Train Loss | Train RMSE | Val Loss | Val RMSE | Val MAE | **Val R²** |
|-------|------------|------------|----------|----------|---------|-----------|
| 0 | 1.10 | 1.05 | 0.88 | 0.94 | 0.78 | 0.334 |
| 1 | 0.82 | 0.90 | 0.78 | 0.88 | 0.72 | 0.409 |
| 2 | 0.77 | 0.88 | 0.74 | 0.86 | 0.69 | 0.442 |
| 3 | 0.73 | 0.85 | 0.70 | 0.84 | 0.67 | **0.470** |
| 4 | 0.71 | 0.84 | - | - | - | - |

**关键观察**：
- 训练稳定收敛，loss 持续下降
- Val $R^2$ 从 0.33 提升到 0.47 (+40%)
- 尚未达到 early stopping 条件

---

# 4. 💡 关键洞见

## 4.1 宏观层洞见

> 用于指导架构设计、理解问题本质的高层次发现。

- **端到端框架可行**：已完成从预训练 encoder 到 log_g 预测的完整链路
- **冻结 encoder 有效**：MLP head 可以从冻结特征中学习到 log_g 信息
- **与 Ridge baseline 差距**：当前 MLP ($R^2=0.47$) vs Ridge ($R^2=0.55$) 差距约 0.08

## 4.2 模型层洞见

> 用于优化模型、调参的中层次发现。

- **特征提取正确**：`encode_flux()` 接口正常工作，返回 (B, 48, L') feature map
- **Pooling 策略有效**：`seg_mean_K8` 保留波长局部性，384 维特征可学习
- **MLP head 足够简单**：单隐藏层已能学习，但可能需要更深/更宽

## 4.3 实验层细节洞见

> 具体的实验观察和技术细节。

- **数据加载正确**：`LogGDataModule` 正确加载 log_g 标签
- **Lightning 集成完善**：checkpoint、early stopping、lr scheduler 正常工作
- **训练速度**：100k 样本，约 400 step/epoch，单 GPU 可接受

---

# 5. 📝 结论

## 5.1 核心发现

> 用一句话总结本实验最重要的发现（punch line），使用引用格式。

> **完整的 Encoder + NN 端到端训练框架已实现并验证可用，冻结 encoder + MLP head 初步达到 val $R^2 = 0.47$，为后续 fine-tuning 实验提供基础。**

**假设验证**：
- ✅ 原假设：end-to-end NN 可以利用 encoder 特征预测 log_g
- ⚠️ 实验结果：val $R^2 = 0.47$，略低于 Ridge baseline (0.55)

## 5.2 关键结论（2-4 条）

| # | 结论 | 证据 |
|---|------|------|
| 1 | **框架完整实现** | 包含 encode_flux 接口、LogGFromEncoderLightning、LogGDataModule |
| 2 | **冻结 encoder 有效** | val $R^2$ 从 0.33 提升至 0.47 (+40%) |
| 3 | **与 Ridge 差距可接受** | 差距 0.08，可能通过 fine-tune 或更深 head 弥补 |
| 4 | **训练稳定** | Loss 平稳下降，无过拟合迹象 |

## 5.3 设计启示

### 架构/方法原则

| 原则 | 建议 | 原因 |
|------|------|------|
| **先冻结再 fine-tune** | 验证框架后再开放 encoder | 避免破坏预训练表示 |
| **特征提取复用** | 使用 `encode_flux()` 接口 | 保持代码一致性 |

### ⚠️ 常见陷阱

| 常见做法 | 实验证据 |
|----------|----------|
| "直接 fine-tune encoder" | 应先验证冻结版性能，建立 baseline |
| "忽略 noise_level" | DataModule 中需正确传递 noise_level 到 batch |

## 5.4 物理解释（可选）

> 用领域知识解释为什么会有这样的结果。

- MLP head 需要学习从 encoder 特征到 log_g 的非线性映射
- Ridge 直接在高维特征上做线性回归，可能更适合当前特征结构
- 后续可尝试更深的 head 或 attention-based pooling

## 5.5 关键数字速查

| 指标 | 值 | 配置/条件 |
|------|-----|----------|
| 最佳 val $R^2$ | **0.4699** | epoch 3, frozen encoder |
| Ridge baseline | 0.5516 | enc_pre_latent + seg_mean_K8 |
| 特征维度 | 384 | 48 channels × 8 segments |
| 冻结参数量 | ~500K | BlindSpot encoder |
| 可训练参数 | ~100K | MLP head (384→256→1) |

## 5.6 下一步工作

> 基于本实验结果，建议的后续方向。

| 方向 | 具体任务 | 优先级 | 对应 MVP |
|------|----------|--------|---------|
| **Fine-tune encoder** | 开放 encoder 训练，测试性能提升 | 🔴 高 | MVP-2.3 |
| **更深 head** | 测试 mlp_2（2 hidden layers）架构 | 🟡 中 | - |
| **超参扫描** | lr, hidden_dim, dropout 组合 | 🟡 中 | - |
| **对比 Ridge** | 在完全相同设置下对比 | 🟢 低 | - |

---

# 6. 📎 附录

## 6.1 代码结构

### 核心文件（BlindSpotDenoiser 仓库）

| 文件 | 说明 |
|------|------|
| `src/blindspot.py` | BlindspotModel1D + `encode_flux()` 接口 |
| `src/logg_from_encoder.py` | LogGFromEncoderLightning + LogGDataModule |
| `experiments/train_logg_from_encoder.py` | 训练脚本 |
| `configs/logg_from_encoder.yaml` | 配置文件 |
| `utils/activation_extractor.py` | Pooling 函数 + ActivationExtractor |

### `encode_flux()` 接口设计

```python
class BlindspotModel1D(BaseModel):
    def encode_flux(self, x_noisy, sigma=None, layer='enc_pre_latent'):
        """
        Extract encoder features from noisy flux input.
        
        Args:
            x_noisy: Noisy flux tensor (B, L) or (B, 1, L)
            sigma: Error/noise tensor (B, L) or (B, 1, L)
            layer: Which encoder layer ('enc_pre_latent' or 'enc_last')
            
        Returns:
            Feature map tensor (B, C, L')
        """
        # Forward pass with return_encoder_features=True
        _, encoder_features = self.forward(x, return_encoder_features=True)
        return encoder_features[layer]
```

### LogGFromEncoderLightning 类

```python
class LogGFromEncoderLightning(L.LightningModule):
    def __init__(
        self,
        encoder_ckpt_path: str,
        config: Dict,
        freeze_encoder: bool = True,
        encoder_layer: str = 'enc_pre_latent',
        pooling: str = 'seg_mean_K8',
        head_architecture: str = 'mlp_1',
        ...
    ):
        # 1. Load pre-trained encoder from checkpoint
        # 2. Optionally freeze encoder
        # 3. Initialize MLP head for log_g prediction
        
    def forward(self, noisy, error):
        # 1. encoder.encode_flux() -> feature map
        # 2. pooling -> fixed-dim vector
        # 3. head -> log_g prediction
```

## 6.2 使用方法

### 训练命令

```bash
# 基础训练（冻结 encoder）
python experiments/train_logg_from_encoder.py \
    --config configs/logg_from_encoder.yaml \
    --encoder-ckpt evals/m215l9e48k25s1bn1d1ep5000.ckpt \
    --freeze-encoder

# Fine-tune encoder
python experiments/train_logg_from_encoder.py \
    --config configs/logg_from_encoder.yaml \
    --encoder-ckpt evals/m215l9e48k25s1bn1d1ep5000.ckpt \
    --no-freeze-encoder

# 使用 wandb 日志
python experiments/train_logg_from_encoder.py \
    --config configs/logg_from_encoder.yaml \
    --encoder-ckpt evals/m215l9e48k25s1bn1d1ep5000.ckpt \
    --freeze-encoder \
    --wandb
```

### 配置文件要点

```yaml
# configs/logg_from_encoder.yaml

# Encoder 必须匹配预训练 checkpoint
model:
  num_layers: 9
  embed_dim: 48
  kernel_size: 25
  input_sigma: true

# Log_g head 配置
logg_head:
  encoder_layer: 'enc_pre_latent'  # 最佳层
  pooling: 'seg_mean_K8'            # 最佳 pooling
  architecture: 'mlp_1'             # 1 hidden layer
  hidden_dim: 256
  dropout: 0.1
```

## 6.3 数值结果表

### 训练指标（Version 2）

| Epoch | Train Loss | Train RMSE | Val Loss | Val RMSE | Val MAE | Val R² |
|-------|------------|------------|----------|----------|---------|--------|
| 0 | 1.104 | 1.051 | 0.880 | 0.938 | 0.780 | 0.334 |
| 1 | 0.818 | 0.904 | 0.781 | 0.884 | 0.719 | 0.409 |
| 2 | 0.772 | 0.878 | 0.738 | 0.859 | 0.691 | 0.442 |
| 3 | 0.738 | 0.859 | 0.701 | 0.837 | 0.671 | **0.470** |

### 与 Baseline 对比

| 方法 | 配置 | Test $R^2$ | 备注 |
|------|------|-----------|------|
| Ridge (offline) | enc_pre_latent + seg_mean_K8 | **0.5516** | MVP 1.4 结果 |
| LightGBM (offline) | enc_last + global_mean | 0.2830 | MVP 1.1 结果 |
| **MLP (end-to-end)** | enc_pre_latent + seg_mean_K8 | 0.4699 | 本实验 (val) |

## 6.4 相关文件

| 类型 | 路径 | 说明 |
|------|------|------|
| 主框架 | `logg/distill/distill_main_20251130.md` | main 文件 |
| 本报告 | `logg/distill/exp_encoder_nn_logg_20251201.md` | 当前文件 |
| Baseline 报告 | `logg/distill/exp_latent_extraction_logg_20251201.md` | Ridge probe |
| 代码仓库 | `/home/swei20/BlindSpotDenoiser/` | 实现代码 |
| Checkpoint | `evals/m215l9e48k25s1bn1d1ep5000.ckpt` | 预训练 encoder |
| 训练日志 | `logs/logg_from_encoder/version_2/` | Lightning 日志 |

## 6.5 实验日志

| 时间 | 事件 | 处理 |
|------|------|------|
| 2025-12-01 | 完成框架实现 | 包含 encode_flux 接口、LogGFromEncoderLightning、训练脚本 |
| 2025-12-01 | 首次训练 v1 | 发现 R² 为负数，检查数据加载 |
| 2025-12-01 | 修复 DataModule | 添加 log_g 到 batch 返回 |
| 2025-12-01 | 训练 v2 | val R² = 0.47，框架验证通过 |

---

> **模板使用说明**：
> 
> **与 main.md 的关系**：
> - 本实验对应 distill_main.md 的 MVP-2.2
> - 验证了 Student latent → log_g 的端到端训练可行性
> - 为 MVP-2.3 (fine-tune encoder) 提供基础

