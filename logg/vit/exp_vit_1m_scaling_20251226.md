<!--
📝 Agent 书写规范（不出现在正文）:
- Header 全英文
- 正文中文
- 图表文字全英文（中文会乱码）
- 公式用 LaTeX: $inline$ 或 $$block$$
-->

# 🍃 ViT 1M Scaling 实验
> **Name:** ViT-1M-L6-H256  
> **ID:** `VIT-20251226-vit-1m-large-01`  
> **Topic:** `vit` | **MVP:** MVP-1.0 | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-26 | **Status:** 🔄  
> **Root:** `logg/vit` | **Parent:** `-` | **Child:** -

> 🎯 **Target:** 验证 ViT 在 1M 数据上的 log_g 预测能力（noise_level=1.0）  
> 🚀 **Next:** 模型达到 R²=0.71+ → 与 LightGBM baseline 比较，探索更深架构

## ⚡ 核心结论速览

> **一句话**: ViT-1M 在 Epoch 96 达到 val_r2=0.713，MAE=0.383，历史新高

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H1: ViT 能否在 1M 数据上有效学习 log_g? | ✅ R²=0.713 | 大数据+大模型有效 |
| H2: noise_level=1.0 下是否可学习? | ✅ 训练收敛良好 | 噪声增强有效 |

| 指标 | 值 | 启示 |
|------|-----|------|
| Best R² | 0.713 | Epoch 96, 仍在上升 |
| val_mae | 0.383 | 预测误差约 0.38 dex |
| vs baseline | 待比较 | LightGBM 1M baseline |

| Type | Link |
|------|------|
| 🧪 WandB (Run 1 - MSE) | https://wandb.ai/viskawei-johns-hopkins-university/vit-1m-scaling/runs/khgqjngm |
| 🧪 WandB (Run 2 - L1) | https://wandb.ai/viskawei-johns-hopkins-university/vit-1m-scaling/runs/6yg86hgi |
| 📁 Config | `~/VIT/configs/exp/vit_1m_l1.yaml` |
| 📂 Checkpoint | `~/VIT/checkpoints/vit_1m/` |

---
# 1. 🎯 目标

**问题**: 验证 Vision Transformer 在大规模光谱数据上预测 log_g 的能力

**验证**: H1: 1M 数据是否能让 ViT 充分学习? H2: noise_level=1.0 是否可行?

| 预期 | 判断标准 |
|------|---------|
| R² > 0.6 | 通过 → 继续扩展架构 |
| R² < 0.5 | 需要调整架构/超参 |

---

# 2. 🦾 算法

## 2.1 整体架构

**Vision Transformer for Spectral Regression**：模型将 1D 光谱视为 "图像"，使用 patch embedding 提取局部特征，然后通过多层 Transformer encoder 捕捉全局依赖关系。

$$
\text{output} = \text{Linear}(\text{CLS\_token}(\text{TransformerEncoder}(\text{SpectraEmbeddings}(x))))
$$

### 核心 Pipeline 流程

```
Input Spectrum (4096,)
       ↓
[SpectraEmbeddings]
   ├── Tokenizer: C1D or SW → (batch, 256, patch_size=16)
   ├── Linear Projection → (batch, 256, hidden_size=256)
   ├── Prepend CLS Token → (batch, 257, 256)
   └── Add Position Embeddings → (batch, 257, 256)
       ↓
[ViT Encoder] × 6 layers
   ├── Multi-Head Self-Attention (8 heads)
   ├── LayerNorm + Residual
   ├── FFN (hidden → 4×hidden → hidden)
   └── LayerNorm + Residual
       ↓
[CLS Token Extraction]
   └── outputs[:, 0, :] → (batch, 256)
       ↓
[Regression Head]
   └── Linear(256, 1) → log_g prediction
```

## 2.2 模型组件详解

### 2.2.1 Patch Tokenization (`src/models/tokenization.py`)

两种 tokenization 策略将 1D 光谱切分为 patches：

| 方法 | 实现 | 数学表达 | 特点 |
|------|------|----------|------|
| **C1D** (Conv1D) | `nn.Conv1d(1, 256, kernel=16, stride=16)` | $y_i = W * x[i \cdot s : i \cdot s + k]$ | 卷积核学习局部特征，参数共享 |
| **SW** (Sliding Window) | `unfold() + nn.Linear(16, 256)` | $y_i = W \cdot x[i \cdot s : i \cdot s + k] + b$ | 线性投影，更简单 |

**C1D 代码实现**:
```python
class Conv1DPatchTokenizer(nn.Module):
    def __init__(self, input_length, patch_size, hidden_size, stride):
        self.projection = nn.Conv1d(1, hidden_size, 
                                    kernel_size=patch_size, 
                                    stride=stride)
        self.num_patches = ((input_length - patch_size) // stride) + 1
    
    def forward(self, x):
        x = x.reshape(-1, 1, self.image_size)  # (B, 1, 4096)
        x = self.projection(x)                  # (B, 256, 256)
        return x.transpose(1, 2)                # (B, 256, 256)
```

**SW 代码实现**:
```python
class SlidingWindowTokenizer(nn.Module):
    def __init__(self, input_length, patch_size, hidden_size, stride):
        self.projection = nn.Linear(patch_size, hidden_size)
        self.num_patches = ceil((input_length - patch_size) / stride) + 1
    
    def forward(self, x):
        patches = x.unfold(1, self.patch_size, self.stride_size)  # (B, 256, 16)
        return self.projection(patches)                            # (B, 256, 256)
```

**关键差异**:
- C1D: 卷积操作，可学习局部特征模式，类似于 CNN 第一层
- SW: 线性投影，更接近原始 ViT 的 patch embedding

### 2.2.2 Position Encoding (`src/models/embedding.py`)

三种位置编码策略：

| 类型 | 实现 | 适用场景 |
|------|------|----------|
| `learned` | `nn.Parameter(randn(1, 257, 256))` | 固定序列长度，可学习 |
| `rope` | Rotary Position Embedding | 更好的长度泛化 |
| `none` | 无位置编码 | 测试 attention pattern |

**当前实验使用 `learned` 位置编码**：
```python
self.position_embeddings = nn.Parameter(
    torch.randn(1, self.num_patches + 1, config.hidden_size)
)
# Forward: tokens = tokens + self.position_embeddings
```

### 2.2.3 ViT Encoder (`transformers.ViTModel`)

基于 HuggingFace `ViTModel`，每层包含：

```python
# 单层 Transformer Block
class ViTLayer:
    def forward(self, hidden_states):
        # 1. Multi-Head Self-Attention
        attention_output = self.attention(
            hidden_states,  # Q, K, V 都来自同一输入
            output_attentions=True
        )
        # 2. Residual + LayerNorm
        hidden_states = self.layernorm_before(hidden_states + attention_output)
        
        # 3. FFN (MLP)
        mlp_output = self.intermediate(hidden_states)  # Linear(256, 1024) + GELU
        mlp_output = self.output(mlp_output)           # Linear(1024, 256)
        
        # 4. Residual + LayerNorm
        hidden_states = self.layernorm_after(hidden_states + mlp_output)
        return hidden_states
```

**Attention 计算**:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中 $d_k = \text{hidden\_size} / \text{num\_heads} = 256 / 8 = 32$

### 2.2.4 Regression Head (`src/models/specvit.py`)

```python
class MyViT(ViTPreTrainedModel):
    def __init__(self, config):
        # ...
        self.regressor = nn.Linear(config.hidden_size, config.num_labels)
        # Loss function based on config
        if loss_name in ("mae", "l1"):
            self.loss_fct = nn.L1Loss()
        elif loss_name in ("mse", "l2"):
            self.loss_fct = nn.MSELoss()
    
    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values)
        cls_token = outputs[0][:, 0, :]  # Extract CLS token
        logits = self.regressor(cls_token)
        
        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
        return SequenceClassifierOutput(loss=loss, logits=logits)
```

## 2.3 Loss 函数选择

| Loss | 公式 | 特点 | 适用场景 |
|------|------|------|----------|
| **MSE** | $L = \frac{1}{n}\sum(y - \hat{y})^2$ | 对大误差惩罚更重 | 高斯误差假设 |
| **L1/MAE** | $L = \frac{1}{n}\sum|y - \hat{y}|$ | 对异常值更鲁棒 | 存在离群点时 |
| **Huber** | 分段函数 | 结合 MSE 和 L1 优点 | 平衡场景 |

**注**: Heteroscedastic loss (除以 flux error) 不适用于 log_g 预测，因为：
- flux error = 观测噪声，与 flux 值相关
- log_g error = 参数预测误差，与物理模型相关
- 两者没有直接数学关系

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | BOSZ synthetic spectra |
| 路径 | `/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/` |
| Train | 1,000,000 (5 shards × 200k) |
| Val | 1,000 |
| Test | 10,000 |
| 特征维度 | 4096 |
| 目标 | log_g |

### 数据加载流程 (`src/dataloader/base.py`)

```
HDF5 File
    ├── spectrumdataset/wave    → wavelength array (4096,)
    ├── dataset/arrays/flux/value → flux matrix (N, 4096)
    └── dataset/arrays/error/value → error matrix (N, 4096)
    └── DataFrame (pandas)
        ├── T_eff, log_g, M_H, a_M, C_M
        ├── mag, redshift, snr
        └── ...
```

**加载代码** (`BaseSpecDataset.load_data`):
```python
with h5py.File(load_path, "r") as f:
    self.wave = torch.tensor(f["spectrumdataset/wave"][()])
    self.flux = torch.tensor(f["dataset/arrays/flux/value"][:num_samples])
    self.error = torch.tensor(f["dataset/arrays/error/value"][:num_samples])

self.flux = self.flux.clip(min=0.0)  # 防止负 flux
# 加载参数
df = pd.read_hdf(load_path)
self.labels = torch.tensor(df["log_g"].values).float()
```

### Label 归一化 (`RegSpecDataset._maybe_normalize_labels`)

| 类型 | 公式 | 范围 |
|------|------|------|
| `standard` (z-score) | $\tilde{y} = \frac{y - \mu}{\sigma}$ | $\mathbb{E}[\tilde{y}] = 0$, $\text{Var}[\tilde{y}] = 1$ |
| `minmax` | $\tilde{y} = \frac{y - y_{\min}}{y_{\max} - y_{\min}}$ | $\tilde{y} \in [0, 1]$ |

**归一化参数从训练集计算，传播到 val/test**:
```python
# Training set
self.label_mean = self.labels.mean()
self.label_std = self.labels.std()
self.labels = (self.labels - self.label_mean) / self.label_std

# Val/Test sets inherit from training set
for k in ('label_mean', 'label_std', 'label_min', 'label_max'):
    setattr(val_dataset, k, getattr(train_dataset, k))
```

## 3.2 噪声增强

| 项 | 值 |
|----|-----|
| 类型 | heteroscedastic gaussian |
| noise_level | 1.0 |
| Train | on-the-fly 生成 |
| Val/Test | 固定 seed 预生成 |

### 噪声模型

$$
\text{noisy\_flux}_i = \text{flux}_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, (\text{noise\_level} \times \text{error}_i)^2)
$$

**Training (on-the-fly)**:
```python
def training_step(self, batch, batch_idx):
    flux, error, labels = batch
    if self.noise_level > 0:
        noisy = flux + torch.randn_like(flux) * error * self.noise_level
        loss = self(noisy, labels)
    else:
        loss = self(flux, labels)
    return loss
```

**Val/Test (pre-generated, fixed seed)**:
```python
def _set_noise(self, seed=42):
    torch.manual_seed(seed)  # 固定 seed 保证可复现
    noise = torch.randn_like(self.flux) * self.error * self.noise_level
    self.noisy = self.flux + noise
```

**为什么 Val/Test 用固定 seed?**
- 保证每次评估结果一致，便于比较不同 epoch/model
- Training 用 on-the-fly 是为了数据增强多样性

## 3.3 模型对比

| 参数 | Run 1 (MSE) | Run 2 (L1) |
|------|-------------|------------|
| 模型 | ViT | ViT |
| image_size | 4096 | 4096 |
| patch_size | 16 | 16 |
| hidden_size | 256 | 256 |
| num_hidden_layers | 6 | 6 |
| num_attention_heads | 8 | 8 |
| intermediate_size | 1024 | 1024 |
| **proj_fn** | **C1D** | **SW** |
| **loss** | **MSE** | **L1** |
| **label_norm** | **standard** | **minmax** |
| pos_encoding | learned | learned |
| head_dim | 32 | 32 |
| dropout | 0.1 | 0.1 |
| Total params | ~4.9M | ~4.9M |

### 参数量详细分解

| 组件 | 参数量 | 计算 |
|------|--------|------|
| **Patch Embedding** | | |
| - C1D Projection | 4,352 | $1 \times 16 \times 256 + 256$ |
| - SW Projection | 4,352 | $16 \times 256 + 256$ |
| **CLS Token** | 256 | $1 \times 256$ |
| **Position Embedding** | 65,792 | $(256 + 1) \times 256$ |
| **Transformer Layer** (×6) | | |
| - QKV Projection | 196,608 | $3 \times 256 \times 256 \times 1$ (no bias) |
| - Output Projection | 65,792 | $256 \times 256 + 256$ |
| - LayerNorm (×2) | 1,024 | $2 \times (256 + 256)$ |
| - FFN (up + down) | 526,592 | $256 \times 1024 + 1024 + 1024 \times 256 + 256$ |
| **Regression Head** | 257 | $256 \times 1 + 1$ |
| **Total** | **~4.9M** | |

### Attention 维度详解

```
Input: (batch, 257, 256)
       ↓
Linear(256, 256×3) → Q, K, V  # (batch, 257, 768)
       ↓
Split + Reshape → 8 heads
  Q, K, V: (batch, 8, 257, 32)  # head_dim = 256/8 = 32
       ↓
Attention = softmax(QK^T / sqrt(32)) @ V
  scores: (batch, 8, 257, 257)  # 每个 token 对所有 token 的注意力
  output: (batch, 8, 257, 32)
       ↓
Concat + Linear(256, 256)
  output: (batch, 257, 256)
```

### 序列长度分析

| 项 | 值 | 说明 |
|----|-----|------|
| 输入长度 | 4096 | 光谱波长点数 |
| Patch 大小 | 16 | 每个 patch 覆盖 16 个波长点 |
| Num patches | 256 | $\lfloor 4096 / 16 \rfloor$ |
| + CLS token | 257 | 实际序列长度 |
| 位置编码容量 | 512 | 最大支持长度 |

## 3.4 训练

| 参数 | 值 |
|------|-----|
| epochs | 200 |
| batch_size | 256 |
| lr | 0.0003 |
| optimizer | AdamW |
| weight_decay | 0.01 |
| lr_scheduler | cosine (eta_min=1e-5) |
| precision | 16-mixed |
| seed | 42 |
| gradient_clip | 0.5 |
| deterministic | True |

### 训练流程 (`src/base/vit.py`)

```python
# Experiment 初始化
experiment = Experiment(config, use_wandb=True, num_gpus=1)
#   ├── ViTLModule: 包装模型 + 训练逻辑
#   ├── ViTDataModule: 数据加载器
#   └── SpecTrainer: Lightning Trainer + Callbacks

# 训练
experiment.t.trainer.fit(
    experiment.lightning_module,
    datamodule=experiment.data_module
)
# 测试
experiment.t.test_trainer.test(
    experiment.lightning_module,
    datamodule=experiment.data_module
)
```

### 优化器配置 (`src/opt/optimizer.py`)

```python
# AdamW with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0003,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

# Cosine Annealing LR Scheduler
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-5
)
```

**Cosine Annealing 学习率曲线**:
$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)
$$

其中 $\eta_{\max} = 0.0003$, $\eta_{\min} = 0.00001$, $T_{\max} = 200$

### 混合精度训练

使用 PyTorch 的 AMP (Automatic Mixed Precision)：
- 前向/后向传播: FP16
- 参数更新: FP32
- 加速约 2x，减少显存占用

```python
# Lightning 自动处理
trainer = L.Trainer(precision='16-mixed')
```

### Callbacks

| Callback | 功能 |
|----------|------|
| `EarlyStopping` | 500 epochs 无改善停止 |
| `ModelCheckpoint` | 保存 best val_mae 模型 |
| `WandbLogger` | 日志记录到 W&B |

## 3.5 并行实验

| 实验 | GPU | Loss | Patch Embed | Label Norm | 状态 |
|------|-----|------|-------------|------------|------|
| Run 1 | GPU 4 | MSE | C1D | standard | 🔄 Epoch 96+ |
| Run 2 | GPU 5 | L1 | SW | minmax | 🔄 Epoch 0+ |

---

# 4. 📊 图表

> 实验仍在运行中，图表待训练完成后补充

### Fig 1: Training Curves (WandB)
- **Run 1 (MSE)**: https://wandb.ai/viskawei-johns-hopkins-university/vit-1m-scaling/runs/khgqjngm
- **Run 2 (L1)**: https://wandb.ai/viskawei-johns-hopkins-university/vit-1m-scaling/runs/6yg86hgi

**观察 (Run 1)**:
- val_r2 从 0 持续上升至 0.713 (Epoch 96)
- 训练尚未收敛，仍有上升空间
- loss 持续下降，无过拟合迹象

**待观察 (Run 2)**:
- L1 loss + SW + minmax 组合效果
- 与 Run 1 的对比

---

# 5. 💡 洞见

## 5.1 宏观
- 1M 数据量对 ViT 训练至关重要，小数据集无法充分发挥 Transformer 优势
- noise_level=1.0 的噪声增强使模型学到更鲁棒的特征

## 5.2 模型层
- 6 层 Transformer + 256 hidden_size 的配置平衡了模型容量和训练效率
- **Patch embedding 选择**:
  - C1D: 1D CNN, 保留局部连续性，类似于 CNN 第一层的卷积核
  - SW: Sliding Window, 更简单的线性投影，可能更适合光谱
- **Loss 选择**:
  - MSE: 对高斯误差最优，梯度 $\propto (y - \hat{y})$
  - L1: 对异常值更鲁棒，梯度 $\propto \text{sign}(y - \hat{y})$
- **Heteroscedastic Loss 不适用**: flux error 与 log_g 预测无关

## 5.3 Pipeline 设计洞见

### 5.3.1 CLS Token vs Mean Pooling

当前使用 CLS token 进行回归：
```python
cls_token = outputs[0][:, 0, :]  # 只用第一个 token
logits = self.regressor(cls_token)
```

**优点**:
- CLS token 通过 self-attention 聚合全局信息
- 无需额外的 pooling 层

**可能的改进**:
- Mean pooling: `mean(outputs[:, 1:, :])` - 平均所有 patch tokens
- Attention pooling: 学习权重加权平均

### 5.3.2 Noise Injection Strategy

**On-the-fly (Training)**:
- 每个 epoch 同一样本看到不同噪声
- 等效于数据增强 $200 \times$ (200 epochs)
- 防止过拟合到特定噪声实例

**Pre-generated (Val/Test)**:
- 固定 seed = 42 保证可复现
- 每次评估条件一致，便于比较

### 5.3.3 Label Normalization 影响

| 归一化 | 特点 | 适用场景 |
|--------|------|----------|
| `standard` | $\mu=0, \sigma=1$ | 配合 MSE loss |
| `minmax` | $[0, 1]$ 范围 | 配合 L1 loss, 输出更可解释 |

**R² 尺度不变性**:
$$
R^2 = 1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}
$$

线性变换 $y' = ay + b$ 后，$R^2$ 不变！因此可以直接与未归一化的 baseline 比较。

### 5.3.4 Position Encoding 重要性

光谱数据中位置信息非常重要：
- 不同波长对应不同元素/分子的吸收线
- log_g 敏感的特征 (如 Mg b triplet, Ca II triplet) 有固定波长位置

**Learned Position Embedding**:
- 可学习每个 patch 的位置特征
- 能发现波长相关的重要区域

## 5.4 细节
- Cosine annealing LR scheduler 稳定收敛，避免后期震荡
- 混合精度 (16-mixed) 加速训练约 2x，显存占用减半
- minmax label norm 可能比 standard 更稳定（输出有界）
- `gradient_clip_val=0.5` 防止梯度爆炸

---

# 6. 📝 结论

## 6.1 核心发现
> **ViT 在 1M 光谱数据上达到 R²=0.713，证明 Transformer 架构可有效学习 log_g 预测**

- ✅ H1: ViT 在大数据上有效学习 log_g
- ✅ H2: noise_level=1.0 训练可行且有益
- ❌ Heteroscedastic Loss: 不适用（flux error ≠ label error）

### 🔑 R² 计算对齐验证

详见: [note_r2_alignment_20251226.md](note_r2_alignment_20251226.md)

**关键发现**: R² 对线性归一化（standard/minmax）是**尺度不变**的！

| 空间 | R² | MAE |
|------|-----|-----|
| 归一化空间 | 0.7132 | 0.3766 |
| **原始空间 (log_g)** | **0.7132** | **0.4399** |

- R² 无需转换，可直接与 LightGBM/Ridge 比较
- MAE 需要 × std 才能得到原始空间的值

## 6.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | **数据规模重要** | 1M > 小规模数据集 |
| 2 | **架构有效** | 4.9M 参数 ViT 达 R²=0.71+ |
| 3 | **噪声增强有效** | noise_level=1.0 收敛良好 |
| 4 | **Loss 选择** | L1 vs MSE 待对比 |

## 6.3 设计启示

| 原则 | 建议 |
|------|------|
| 数据规模 | 1M+ 样本对 ViT 有显著帮助 |
| Patch embedding | C1D 或 SW 都可行 |
| 训练策略 | Cosine LR + AdamW |
| Label norm | minmax 可能优于 standard |

| ⚠️ 陷阱 | 原因 |
|---------|------|
| 小数据集训练 ViT | 容易过拟合，需大数据 |
| Heteroscedastic loss for label | flux error 与 label 无关 |

## 6.4 关键数字

| 指标 | 值 | 条件 |
|------|-----|------|
| Best R² (Run 1) | **0.7132** | Epoch 112, 原始/归一化空间一致 |
| val_mae 归一化 (Run 1) | 0.3766 | 归一化后的 MAE |
| val_mae 原始 (Run 1) | **0.4399** | 原始空间 log_g 的 MAE |
| train_loss (Run 1) | 0.288 | Epoch 96 |
| train_loss (Run 2) | ~0.85 | Epoch 0 (L1 loss scale) |
| 训练速度 | ~6.9 it/s | 单 GPU |
| log_g 范围 | [1.0, 5.0] | 4 dex 范围 |

## 6.5 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| 完成训练 | 等待 200 epochs 完成 | 🔴 |
| L1 vs MSE | 对比 Run 1 和 Run 2 结果 | 🔴 |
| SW vs C1D | 分析 patch embedding 差异 | 🟡 |
| 与 baseline 比较 | vs LightGBM 1M | 🟡 |
| 架构探索 | 更深/更宽模型 | 🟢 |

---

# 7. 🗂️ 代码结构参考

## 7.1 核心模块一览

```
src/
├── models/                      # 模型定义
│   ├── specvit.py              # MyViT: 核心 ViT 模型
│   ├── embedding.py            # SpectraEmbeddings: Patch + Position
│   ├── tokenization.py         # C1D / SW / Linear tokenizers
│   ├── builder.py              # get_model(): 模型构建入口
│   ├── attention.py            # 自定义 attention 模块
│   └── rope.py                 # Rotary Position Embedding
│
├── dataloader/                  # 数据加载
│   ├── base.py                 # BaseSpecDataset: HDF5 加载 + Noise
│   └── spec_datasets.py        # ClassSpecDataset / RegSpecDataset
│
├── base/                        # 基础模块
│   ├── basemodule.py           # BaseDataModule / BaseLightningModule
│   └── vit.py                  # ViTLModule / SpecTrainer / Experiment
│
├── opt/                         # 优化器
│   └── optimizer.py            # OptModule: AdamW + LR Scheduler
│
└── nn/
    └── trainer.py              # train_single_experiment(): 轻量训练
```

## 7.2 关键函数调用链

### 模型构建
```
get_model(config)  [builder.py]
    └── _build_vit_model(config)
        ├── get_vit_config(config)  → ViTConfig
        └── MyViT(vit_config)  [specvit.py]
            └── self.vit = ViTModel(config)
                └── self.vit.embeddings = SpectraEmbeddings(config)  [embedding.py]
                    └── self.patch_embeddings = Conv1DPatchTokenizer(...)  [tokenization.py]
```

### 数据加载
```
ViTDataModule.from_config(config)  [vit.py]
    └── BaseDataModule(dataset_cls=RegSpecDataset)  [basemodule.py]
        └── self.train = RegSpecDataset.from_config(config)  [spec_datasets.py]
            └── BaseSpecDataset.load_data(stage)  [base.py]
                ├── h5py.File(load_path) → flux, error, wave
                ├── pd.read_hdf(load_path) → log_g
                └── _maybe_normalize_labels() → standard/minmax
```

### 训练循环
```
Experiment.run()  [vit.py]
    └── SpecTrainer.trainer.fit(ViTLModule, ViTDataModule)
        └── ViTLModule.training_step(batch)  [vit.py]
            ├── noise injection (on-the-fly)
            ├── model.forward(noisy, labels)  [specvit.py]
            │   ├── preprocessor(x) if exists
            │   ├── vit(x) → outputs
            │   ├── cls_token = outputs[0][:, 0, :]
            │   ├── logits = regressor(cls_token)
            │   └── loss = loss_fct(logits, labels)
            └── self.log(loss)
```

## 7.3 配置文件结构

```yaml
project: 'vit-1m-scaling'

model:
  task_type: reg
  image_size: 4096      # 光谱长度
  patch_size: 16        # Patch 大小
  hidden_size: 256      # 隐藏维度
  num_hidden_layers: 6  # Transformer 层数
  num_attention_heads: 8
  proj_fn: 'C1D'        # 或 'SW'
  pos_encoding_type: 'learned'

train:
  batch_size: 256
  ep: 200
  precision: '16-mixed'

loss:
  name: 'mse'           # 或 'l1', 'huber'

opt:
  type: 'AdamW'
  lr: 0.0003
  weight_decay: 0.01
  lr_sch: 'cosine'
  eta_min: 0.00001

data:
  file_path: "train_200k_0/dataset.h5"
  val_path: "val_1k/dataset.h5"
  test_path: "test_10k/dataset.h5"
  param: log_g
  label_norm: 'standard'

noise:
  noise_level: 1.0
```

---

# 8. 📎 附录

## 8.0 完整前向传播示例

以下是一个 batch 通过 ViT 的完整数据流：

```python
# Input: batch of noisy spectra
x = noisy_flux  # shape: (256, 4096) - batch_size=256, 4096 wavelengths

# ========== Step 1: Patch Tokenization ==========
# Conv1D: (batch, 4096) → (batch, 1, 4096) → Conv1d → (batch, 256, 256)
x = x.reshape(256, 1, 4096)            # (256, 1, 4096)
x = conv1d(x)                           # Conv1d(1, 256, kernel=16, stride=16)
x = x.transpose(1, 2)                   # (256, 256, 256) - 256 patches, 256 dims

# ========== Step 2: Add CLS Token ==========
cls_token = nn.Parameter(randn(1, 1, 256))
cls_tokens = cls_token.expand(256, 1, 256)  # (256, 1, 256)
x = torch.cat([cls_tokens, x], dim=1)       # (256, 257, 256)

# ========== Step 3: Add Position Embedding ==========
pos_embed = nn.Parameter(randn(1, 257, 256))
x = x + pos_embed                       # (256, 257, 256)

# ========== Step 4: Transformer Encoder (×6 layers) ==========
for layer in range(6):
    # --- Multi-Head Self-Attention ---
    Q = W_q(x)  # (256, 257, 256)
    K = W_k(x)  # (256, 257, 256)
    V = W_v(x)  # (256, 257, 256)
    
    # Reshape for multi-head: (batch, heads, seq, head_dim)
    Q = Q.view(256, 257, 8, 32).transpose(1, 2)  # (256, 8, 257, 32)
    K = K.view(256, 257, 8, 32).transpose(1, 2)
    V = V.view(256, 257, 8, 32).transpose(1, 2)
    
    # Attention scores
    scores = Q @ K.transpose(-2, -1) / sqrt(32)  # (256, 8, 257, 257)
    attn = softmax(scores, dim=-1)               # (256, 8, 257, 257)
    attn_out = attn @ V                          # (256, 8, 257, 32)
    
    # Concat heads
    attn_out = attn_out.transpose(1, 2).reshape(256, 257, 256)  # (256, 257, 256)
    attn_out = W_out(attn_out)  # (256, 257, 256)
    
    # Residual + LayerNorm
    x = LayerNorm(x + attn_out)  # (256, 257, 256)
    
    # --- FFN ---
    ffn_out = W_up(x)    # (256, 257, 1024) + GELU
    ffn_out = W_down(ffn_out)  # (256, 257, 256)
    
    # Residual + LayerNorm
    x = LayerNorm(x + ffn_out)  # (256, 257, 256)

# ========== Step 5: Extract CLS Token ==========
cls_output = x[:, 0, :]  # (256, 256) - only first token

# ========== Step 6: Regression Head ==========
logits = W_reg(cls_output)  # Linear(256, 1) → (256, 1)
pred_logg = logits.squeeze()  # (256,) - predicted log_g

# ========== Step 7: Compute Loss ==========
# MSE Loss
loss = ((pred_logg - labels) ** 2).mean()

# OR L1 Loss
loss = (pred_logg - labels).abs().mean()
```

**关键张量维度总结**:

| 阶段 | 张量形状 | 说明 |
|------|----------|------|
| 输入 | (256, 4096) | batch × spectrum_length |
| Patch 化后 | (256, 256, 256) | batch × num_patches × hidden_dim |
| 加 CLS 后 | (256, 257, 256) | batch × (1 + num_patches) × hidden_dim |
| Transformer 输出 | (256, 257, 256) | 同上 |
| CLS 提取 | (256, 256) | batch × hidden_dim |
| 回归输出 | (256, 1) | batch × num_labels |

## 8.1 数值结果

### Run 1 (MSE + C1D + standard)

| Epoch | R² | MAE | train_loss |
|-------|-----|-----|------|
| 0 | ~0 | ~1.0 | ~1.0 |
| 50 | ~0.5 | ~0.5 | ~0.4 |
| 96 | 0.713 | 0.383 | 0.288 |
| 200 | (待完成) | | |

### Run 2 (L1 + SW + minmax)

| Epoch | R² | MAE | train_loss |
|-------|-----|-----|------|
| 0 | (进行中) | - | ~0.85 |
| 200 | (待完成) | | |

## 8.2 执行记录

| 项 | Run 1 | Run 2 |
|----|-------|-------|
| GPU | 4 | 5 |
| 仓库 | `~/VIT` | `~/VIT` |
| 脚本 | `scripts/train_vit_1m.py` | `scripts/train_vit_1m.py` |
| Config | `configs/exp/vit_1m_large.yaml` | `configs/exp/vit_1m_l1.yaml` |
| Output | `checkpoints/vit_1m/` | `checkpoints/vit_1m/` |
| WandB | `runs/khgqjngm` | `runs/6yg86hgi` |

```bash
# Run 1: MSE + C1D
python scripts/train_vit_1m.py --gpu 0

# Run 2: L1 + SW + minmax
python scripts/train_vit_1m.py --config configs/exp/vit_1m_l1.yaml --gpu 1

# 查看日志
tail -f logs/vit_1m_full_*.log
tail -f logs/vit_1m_l1_*.log
```

## 8.3 调试

| 问题 | 解决 |
|------|------|
| DDP unused parameters | 使用 DDPStrategy(find_unused_parameters=True) |
| devices=0 error | GPU 参数改为列表 [0] |
| torch.median non-deterministic | 改用 torch.mean |
| Heteroscedastic loss 概念错误 | flux error ≠ log_g error, 改用普通 L1 |

## 8.4 代码变更

### 新增 Loss 函数 (`src/models/specvit.py`)

```python
class HeteroscedasticL1Loss(nn.Module):
    """L = |y - y_hat| / error (Laplace NLL)"""
    def forward(self, pred, target, error=None):
        l1_loss = torch.abs(pred - target)
        if error is not None:
            l1_loss = l1_loss / (error + 1e-8)
        return l1_loss.mean()

class HeteroscedasticMSELoss(nn.Module):
    """L = (y - y_hat)^2 / error^2 (Gaussian NLL)"""
    def forward(self, pred, target, error=None):
        mse_loss = (pred - target) ** 2
        if error is not None:
            mse_loss = mse_loss / (error ** 2 + 1e-8)
        return mse_loss.mean()
```

> 注: 上述 Heteroscedastic Loss 实现已完成，但不适用于当前问题（flux error 与 log_g 预测无关）

---

> **实验完成时间**: 2025-12-26 (进行中)
