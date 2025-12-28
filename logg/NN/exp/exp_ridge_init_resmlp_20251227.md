<!--
📝 Agent 书写规范（不出现在正文）:
- Header 全英文
- 正文中文
- 图表文字全英文（中文会乱码）
- 公式用 LaTeX: $inline$ 或 $$block$$
-->

# 🍃 Ridge-Initialized ResMLP for log g Regression
> **Name:** Ridge-Init ResMLP  
> **ID:** `VIT-20251227-nn-01`  
> **Topic:** `NN` | **MVP:** MVP-2.1 | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-27 | **Status:** 📋 企划  
> **Root:** `logg_1m_hub` | **Parent:** `NN_main` | **Child:** -

> 🎯 **Target:** 验证 Ridge 权重初始化 + 深层 ResNet MLP 能否结合线性先验与非线性表达，超越现有 MLP baseline  
> 🚀 **Next:** 如果成功 → 作为新的 NN baseline；如果失败 → 分析 Ridge 权重利用方式

## ⚡ 核心结论速览

> **一句话**: 待验证

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H2.1: Ridge 初始化能否加速收敛？ | ⏳ | 待验证 |
| H2.2: 深层 ResNet 能否学习有效残差？ | ⏳ | 待验证 |
| H2.3: 能否超越 MLP baseline (R²=0.498)？ | ⏳ | 待验证 |

| 指标 | 值 | 启示 |
|------|-----|------|
| Target R² | > 0.55 (32k) | 超越现有 MLP |
| Best R² | TODO | |
| vs MLP baseline | TODO | |

| Type | Link |
|------|------|
| 🧠 Hub | `logg/logg_1m/logg_1m_hub_20251222.md` |
| 📘 Topic Main | `logg/NN/NN_main_20251130.md` |
| 🗺️ Roadmap | `logg/logg_1m/logg_1m_roadmap_20251222.md` |

---
# 1. 🎯 目标

**问题**: 现有 MLP (R²=0.498) 弱于 LightGBM (R²=0.536)，能否通过 Ridge 初始化 + 深层 ResNet 结构突破？

**验证**: 
- H2.1: Ridge 权重初始化是否提供有效线性先验
- H2.2: ResNet 残差结构是否能学习非线性修正
- H2.3: 整体架构是否超越现有 baseline

| 预期 | 判断标准 |
|------|---------|
| 超越 MLP baseline | R² > 0.498 → 架构有效，继续优化 |
| 超越 LightGBM | R² > 0.536 → 深层 NN 可替代树模型 |
| 接近 ViT | R² > 0.60 → MLP 架构潜力巨大 |

**动机**:
1. Ridge 回归已证明能提取线性信息 (R²=0.458)
2. MLP 第一层可视为线性投影，用 Ridge 权重初始化可注入先验
3. 残差连接允许深层网络稳定学习非线性修正
4. 4-5 层深度应足够学习复杂非线性模式

---

# 2. 🦾 算法

## 2.1 Ridge 初始化策略

**Ridge 回归权重**：
$$
\mathbf{w}_{\text{ridge}} = (\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}
$$

**第一层初始化**：
- Ridge 权重 $\mathbf{w}_{\text{ridge}} \in \mathbb{R}^{4096}$ 作为第一层的一个（或多个）输出神经元的权重
- 策略选项：
  - **Strategy A**: 第一层复制 Ridge 权重到所有 hidden 神经元（需要扩展维度）
  - **Strategy B**: 第一层用 Ridge 预测值作为额外输入通道 (concat)
  - **Strategy C**: 第一层一个神经元用 Ridge 权重，其他随机初始化

**推荐 Strategy B**（最稳定）：
$$
\mathbf{h}_0 = [\mathbf{x}, \hat{y}_{\text{ridge}}] \in \mathbb{R}^{4097}
$$

## 2.2 ResNet MLP 结构

**核心设计**（4-5 层）：

```
Input x ∈ ℝ^4096
    │
    ├──────────────────────────────────┐
    ↓                                  │ (shortcut)
[Linear(4096→512) + LN + GELU]         │
    │                                  │
    ├──────────────────────────────────┤
    ↓                                  │ (shortcut)
[ResBlock: Linear→LN→GELU→Linear + skip]
    │                                  │
    ├──────────────────────────────────┤
    ↓                                  │ (shortcut)
[ResBlock: Linear→LN→GELU→Linear + skip]
    │                                  │
    ├──────────────────────────────────┤
    ↓                                  │ (shortcut)
[ResBlock: Linear→LN→GELU→Linear + skip]
    │                                  │
    ↓                                  │
[Linear(512→1)]  ←─────────────────────┘ (可选: + Ridge pred shortcut)
    │
Output ŷ ∈ ℝ
```

**ResBlock 定义**：
$$
\text{ResBlock}(\mathbf{h}) = \mathbf{h} + \text{Linear}_2(\text{GELU}(\text{LN}(\text{Linear}_1(\mathbf{h}))))
$$

## 2.3 可选：Ridge 残差学习

**目标改为学习 Ridge 残差**：
$$
\text{target} = y - \hat{y}_{\text{ridge}}
$$

**最终预测**：
$$
\hat{y} = \hat{y}_{\text{ridge}} + \text{ResMLP}(\mathbf{x})
$$

这与 MoE 中 Expert 学习残差的策略类似。

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | BOSZ → PFS MR |
| 路径 | `~/VIT/data/mag205_225_lowT_1M.h5` |
| Train/Val/Test | 32k / 10k / 10k (初始) 或 100k+ (扩展) |
| 特征维度 | 4096 |
| 目标 | log_g |

## 3.2 噪声

| 项 | 值 |
|----|-----|
| 类型 | heteroscedastic (PFS realistic) |
| σ | noise_level=1.0 |
| 范围 | train + val + test |

## 3.3 模型

| 参数 | 值 |
|------|-----|
| 模型 | Ridge-Init ResMLP |
| 总层数 | 4-5 层 (1 stem + 3-4 ResBlocks + 1 head) |
| 隐藏维度 | 512 (主干) / 256 (ResBlock bottleneck) |
| 激活函数 | GELU |
| 归一化 | LayerNorm |
| Dropout | 0.1-0.3 |

**架构变体扫描**：

| 变体 | 描述 |
|------|------|
| **V1: Baseline ResMLP** | 无 Ridge 初始化，验证 ResNet 结构本身效果 |
| **V2: Ridge-Concat** | 输入 concat Ridge 预测值 (4097 维) |
| **V3: Ridge-Init Layer1** | 第一层用 Ridge 权重初始化 |
| **V4: Ridge-Residual** | 学习 Ridge 残差，最后加回 Ridge 预测 |
| **V5: Ridge-Shortcut** | 输出层有 Ridge 预测的 skip connection |

## 3.4 训练

| 参数 | 值 |
|------|-----|
| epochs | 200 |
| batch | 2048 |
| lr | 1e-4 → 3e-4 (warmup) |
| optimizer | AdamW |
| scheduler | CosineAnnealing / OneCycleLR |
| weight_decay | 1e-4 |
| seed | 42 |

**训练策略**：
1. 先训练 Ridge 获得权重和预测值
2. 用 Ridge 结果初始化/增强 MLP
3. 可选：冻结 Ridge 相关参数几个 epoch

## 3.5 扫描参数

| 扫描 | 范围 | 固定 |
|------|------|------|
| 初始化策略 | [V1, V2, V3, V4, V5] | - |
| 隐藏维度 | [256, 512, 1024] | depth=4 |
| ResBlock 数量 | [2, 3, 4] | hidden=512 |
| Dropout | [0.1, 0.2, 0.3] | 最佳架构 |
| lr | [1e-4, 3e-4, 1e-3] | 最佳架构 |

---

# 4. 📊 图表

> ⚠️ 图表文字必须全英文！

### Fig 1: Ridge Initialization Strategy Comparison
**待生成**: 柱状图比较 V1-V5 的 R² 性能

### Fig 2: Training Curves
**待生成**: 训练/验证 loss 曲线，对比有无 Ridge 初始化的收敛速度

### Fig 3: Depth vs Performance
**待生成**: ResBlock 数量 vs R² 曲线

### Fig 4: Ridge Residual Analysis
**待生成**: 真实残差 vs 模型预测残差的散点图

---

# 5. 💡 预期洞见

## 5.1 宏观
- Ridge 初始化应该提供稳定的线性起点，避免随机初始化的不稳定
- ResNet 结构应该能稳定训练 4-5 层深度，避免梯度消失

## 5.2 模型层
- GELU + LayerNorm + Dropout 组合应该优于 ReLU
- 512 隐藏维度可能是性价比最高的选择

## 5.3 待验证
- V2 (Ridge-Concat) vs V4 (Ridge-Residual) 哪个更有效
- 是否需要冻结 Ridge 相关权重

---

# 6. 📝 待验证结论

## 6.1 假设验证目标

| # | 假设 | 验证标准 |
|---|------|---------|
| H2.1 | Ridge 初始化加速收敛 | 收敛 epoch 减少 ≥30% |
| H2.2 | 深层 ResNet 有效 | 4-5 层 > 2-3 层 R² |
| H2.3 | 超越 MLP baseline | R² > 0.498 (32k) |
| H2.4 | 接近或超越 LightGBM | R² > 0.536 (32k) |

## 6.2 关键数字参考

| 指标 | 值 | 条件 | 来源 |
|------|-----|------|------|
| MLP baseline | 0.498 | 32k, noise=1.0 | exp_mlp_baseline |
| MLP (100k) | 0.551 | 100k, noise=1.0 | exp_nn_comprehensive |
| Ridge | 0.458 | 32k, noise=1.0 | ridge_main |
| LightGBM | 0.536 | 32k, noise=1.0 | benchmark |
| ViT (1M) | 0.713 | 1M, noise=1.0 | vit_hub |
| Fisher ceiling | 0.89 | noise=1.0, mag=21.5 | scaling_hub |

## 6.3 预期设计启示

| 原则 | 预期建议 |
|------|---------|
| Ridge 初始化 | 如果有效 → 成为 MLP 训练标准流程 |
| ResNet 深度 | 如果 4-5 层 > 2-3 层 → 深层 MLP 值得投资 |
| 残差学习 | 如果 V4 > V2 → 学习残差比 concat 更有效 |

---

# 7. 📎 附录

## 7.1 数值结果（待填充）

| 变体 | R² | MAE | RMSE | Epochs |
|------|-----|-----|------|--------|
| V1: Baseline ResMLP | TODO | | | |
| V2: Ridge-Concat | TODO | | | |
| V3: Ridge-Init | TODO | | | |
| V4: Ridge-Residual | TODO | | | |
| V5: Ridge-Shortcut | TODO | | | |

## 7.2 执行记录

| 项 | 值 |
|----|-----|
| 仓库 | `~/VIT` |
| 脚本 | `scripts/train_ridge_resmlp.py` |
| Config | `configs/ridge_resmlp.yaml` |
| Output | `results/ridge_resmlp/` |

```bash
# Step 1: 训练 Ridge 获得权重
python scripts/train_ridge.py --alpha 200 --save_weights

# Step 2: 训练 Ridge-Init ResMLP
python scripts/train_ridge_resmlp.py --config configs/ridge_resmlp.yaml

# Step 3: 评估
python scripts/eval.py --ckpt results/ridge_resmlp/best.pt
```

## 7.3 模型代码骨架（参考）

```python
class ResBlock(nn.Module):
    def __init__(self, dim, bottleneck_dim=None, dropout=0.1):
        super().__init__()
        bottleneck_dim = bottleneck_dim or dim // 2
        self.block = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return x + self.block(x)

class RidgeResMLP(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=512, n_blocks=3, 
                 dropout=0.1, ridge_weights=None, strategy='concat'):
        super().__init__()
        self.strategy = strategy
        self.ridge_weights = ridge_weights  # 预训练 Ridge 权重
        
        # Stem
        in_dim = input_dim + 1 if strategy == 'concat' else input_dim
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # ResBlocks
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, hidden_dim//2, dropout)
            for _ in range(n_blocks)
        ])
        
        # Head
        self.head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Ridge prediction
        if self.ridge_weights is not None:
            ridge_pred = F.linear(x, self.ridge_weights.unsqueeze(0))
            if self.strategy == 'concat':
                x = torch.cat([x, ridge_pred], dim=-1)
        
        # Forward
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        
        # Optional: add back ridge prediction
        if self.strategy == 'residual' and self.ridge_weights is not None:
            out = out + ridge_pred
            
        return out.squeeze(-1)
```

## 7.4 相关文件

| 类型 | 路径 |
|------|------|
| MLP Baseline | `logg/NN/exp/exp_mlp_baseline_20251130.md` |
| Ridge 权重分析 | `logg/ridge/exp/exp_ridge_alpha_sweep_20251127.md` |
| NN 架构设计 | `logg/NN/exp/exp_nn_architecture_design_20251129.md` |
| logg 1M Hub | `logg/logg_1m/logg_1m_hub_20251222.md` |

---

> **立项时间**: 2025-12-27  
> **预估工作量**: 1-2 天（含代码实现和实验）  
> **优先级**: P1（已在 NN_main 下一步计划中）
