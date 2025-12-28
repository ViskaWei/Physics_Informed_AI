# 🤖 实验 Coding Prompt

> **Experiment ID:** `LOGG-DUAL-TOWER-01`  
> **日期:** 2025-12-28 | **来源:** `logg/moe/moe_snr_roadmap.md` MVP-4.0  
> **MVP:** MVP-4.0 (Gate-5: 双塔融合)  
> **Status:** 🔴 P0

---

## ⚠️ 核心规则

| 规则 | 说明 |
|------|------|
| **nohup 后台运行** | 所有训练必须 `nohup ... &`，>5分钟不持续追踪 |
| **跨仓库用终端** | 写入 Physics_Informed_AI 用 `cat/echo/cp`，禁止 IDE 工具 |
| **图片必须入报告** | 所有图表必须在报告 §3 中引用，路径 `logg/moe/exp/img/` |
| **语言** | Header 英文 \| 正文中文 \| 图表文字英文 |

---

## 🚀 仓库路由

| Topic | 仓库 | 前缀 |
|-------|------|------|
| **dual-tower-moe** | `~/VIT` | VIT- |

---

## 🎯 实验目标

验证 **物理 9-gate + quality gate 双塔融合** 能否叠加增益：

| 验证问题 | 验收标准 | 下一步 |
|---------|---------|--------|
| 双塔 > 单塔 phys gate? | ΔR² ≥ +0.005 | 继续 MVP-4.1/4.2 |
| 低 SNR 子集改善? | per-SNR ΔR² ≥ +0.01 | quality gate 有效 |
| 无叠加增益 | ΔR² < 0 | fallback 单塔 |

**核心思路**：物理轴与质量轴正交
- **物理轴 (Teff×[M/H])**：解释"光谱长什么样"（9 experts 不变）
- **质量轴 (SNR)**：解释"我们能看清多少"（10D quality_features）

---

## 🧪 实验设计

### 1. 数据配置

```yaml
data:
  source: "BOSZ simulated spectra"
  root: "/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M"
  train_shards: ["train_200k_{0..4}/dataset.h5"]  # 1M samples
  test_file: "test_1k_0/dataset.h5"  # 或 test_10k
  feature_dim: 4096  # MR arm spectral pixels
  target: "log_g"

noise:
  level: 1.0  # heteroscedastic Gaussian noise
  apply: "train"
```

### 2. 特征设计

#### 2.1 物理窗特征 (phys_features, ~13 维)

```python
def extract_phys_features(flux_noisy: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    """
    从 noisy flux 提取物理窗特征（用于 gate）
    
    特征列表 (13 维):
    - Ca II triplet: depth_8498, EW_8498, depth_8542, EW_8542, 
                     depth_8662, EW_8662, EW_CaT (7 维)
    - Na I: depth_Na, EW_Na (2 维)
    - Teff proxy: PCA1, PCA2, PCA3, PCA4 (4 维)
    """
    # 参考：exp_moe_9expert_phys_gate_20251204.md §2.2
    pass
```

#### 2.2 质量特征 (quality_features, 10 维) — 已冻结 ✅

```python
def quality_features(error: np.ndarray) -> np.ndarray:
    """10 aggregate statistics - de-leaked, SNR-preserving.
    
    已验证：logg R²=0.042 (无泄露)，SNR R²=0.995 (完美保留)
    """
    from scipy import stats
    return np.column_stack([
        np.mean(error, axis=-1),      # 0: mean
        np.std(error, axis=-1),       # 1: std
        np.min(error, axis=-1),       # 2: min
        np.max(error, axis=-1),       # 3: max
        np.median(error, axis=-1),    # 4: median
        np.sum(error, axis=-1),       # 5: sum
        np.percentile(error, 25, axis=-1),  # 6: q25
        np.percentile(error, 75, axis=-1),  # 7: q75
        stats.skew(error, axis=-1),   # 8: skew
        stats.kurtosis(error, axis=-1),     # 9: kurtosis
    ])
```

### 3. 模型配置

```yaml
experts:
  type: "Ridge"
  num: 9  # 按 Teff×[M/H] 分 3×3 grid
  bins:
    teff: [3750, 4500, 5250, 6000]  # 3 bins
    mh: [-2.0, -1.0, 0.0, 0.5]       # 3 bins
  alpha: 100000  # 高噪声最优
  training: "沿用已有预训练 experts（如果有）"

gate:
  architecture: "MLP"
  input_dim: 23  # phys_features (13) + quality_features (10)
  hidden_layers: [32, 16]
  output_dim: 9  # 对应 9 个 experts
  activation: "ReLU"
  training_loss: "MSE"  # 回归最优，不是分类 CE！
  routing: "soft"  # softmax 加权

preprocessing:
  scaler: "StandardScaler"
  fit_on: "train"
```

### 4. 双塔 Gate 架构

```python
class DualTowerGate(nn.Module):
    """
    双塔 Gate：物理特征 + 质量特征 → 9 expert weights
    
    方案 A（推荐先做）：简单 concat + MLP
    """
    def __init__(self, phys_dim=13, qual_dim=10, num_experts=9):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(phys_dim + qual_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_experts),
        )
    
    def forward(self, phys_feat, qual_feat):
        x = torch.cat([phys_feat, qual_feat], dim=-1)
        logits = self.mlp(x)
        weights = F.softmax(logits, dim=-1)
        return weights
```

### 5. 训练流程

```python
# 回归最优训练（MSE loss，不是分类 CE）
def train_gate_regression(gate, experts, train_loader, epochs=100, lr=1e-3):
    """
    直接最小化 final logg MSE 来学 gate weights
    
    Loss = MSE(Σ w_k * expert_k(flux), y_true)
    """
    optimizer = Adam(gate.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for flux, error, y_true, phys_feat, qual_feat in train_loader:
            # 1. Gate 输出权重
            weights = gate(phys_feat, qual_feat)  # [B, 9]
            
            # 2. 每个 expert 的预测（预训练 Ridge，不更新）
            expert_preds = []
            for k, expert in enumerate(experts):
                pred_k = expert.predict(flux)  # [B]
                expert_preds.append(pred_k)
            expert_preds = torch.stack(expert_preds, dim=-1)  # [B, 9]
            
            # 3. 加权预测
            final_pred = (weights * expert_preds).sum(dim=-1)  # [B]
            
            # 4. MSE loss
            loss = F.mse_loss(final_pred, y_true)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 📊 要生成的图表

| # | 图表类型 | X轴 | Y轴 | 保存路径 |
|---|---------|-----|-----|---------|
| 1 | Bar (对比) | Method | Test R² | `dual_tower_comparison.png` |
| 2 | Bar (per-SNR) | SNR Bin | R² | `dual_tower_persnr_r2.png` |
| 3 | Scatter | True log_g | Predicted log_g | `dual_tower_pred_vs_true.png` |
| 4 | Heatmap | Gate Entropy vs SNR | - | `dual_tower_entropy_vs_snr.png` |
| 5 | Bar | Feature ablation | ΔR² | `dual_tower_ablation.png` |

### 图表要求

- 所有文字 **英文**
- 必须对比 4 个方法：
  1. Global Ridge
  2. Phys-only Gate (9 experts)
  3. Quality-only Gate (4 SNR experts, 如有)
  4. **Dual-Tower Gate (9 experts)**
- 必须按 SNR bins (>7/4-7/2-4/≤2) 分析
- 画 gate 熵/置信度 vs SNR：验证低 SNR 是否更"平"

---

## 🗂️ 参考代码

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/logg_snr_moe.py` | 数据加载、SNR 计算、quality_features | 添加 phys_features |
| `scripts/scaling_oracle_moe_noise1.py` | 9-expert 训练框架 | 不需要改 |
| `scripts/moe_phys_gate.py` | 物理窗特征提取 | 直接复用 |

### 关键复用函数

```python
# 从 logg_snr_moe.py:
load_shards()           # 加载多个 HDF5 shards
add_noise()             # 添加 heteroscedastic noise
quality_features()      # 10D 质量统计量（已冻结）
train_snr_experts()     # SNR-binned experts

# 从 moe_phys_gate.py:
extract_line_features()     # Ca II, Na I depth/EW
extract_pca_features()      # PCA1-4 for Teff proxy
train_9bin_experts()        # 9-bin Ridge experts
```

---

## 📋 执行流程

### Step 1: 创建实验脚本

```bash
cd ~/VIT
# 创建脚本：scripts/logg_dual_tower_moe.py
```

**脚本核心结构**：

```python
"""
Dual-Tower MoE: Physics Gate + Quality Gate
Experiment ID: LOGG-DUAL-TOWER-01
"""

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# === 1. 数据加载 ===
def load_data():
    """复用 logg_snr_moe.py 的数据加载"""
    pass

# === 2. 特征提取 ===
def extract_phys_features(flux_noisy, wavelength):
    """物理窗特征：Ca II, Na I, PCA1-4"""
    pass

def quality_features(error):
    """质量特征：10 aggregate stats（已冻结）"""
    pass

# === 3. Experts 训练 ===
def train_9bin_experts(X_train, y_train, teff, mh):
    """按 Teff×[M/H] 分 9 bin，各训练 Ridge"""
    pass

# === 4. Gate 训练 ===
def train_dual_tower_gate(experts, X_train, y_train, 
                          phys_feat_train, qual_feat_train):
    """回归最优训练：MSE loss on final logg"""
    pass

# === 5. 评估 ===
def evaluate(gate, experts, X_test, y_test, 
             phys_feat_test, qual_feat_test, snr_test):
    """
    输出：
    - overall R²
    - per-SNR bin R²
    - gate 熵 vs SNR
    """
    pass

# === 6. 消融实验 ===
def ablation_study():
    """
    对比：
    1. phys-only gate
    2. qual-only gate  
    3. dual-tower gate
    """
    pass

# === Main ===
if __name__ == "__main__":
    # 1. 加载数据
    # 2. 提取特征
    # 3. 训练 experts（或加载已有）
    # 4. 训练 dual-tower gate
    # 5. 评估 + 切片分析
    # 6. 消融实验
    # 7. 保存图表
    pass
```

### Step 2: 启动训练

```bash
cd ~/VIT && source init.sh
nohup python scripts/logg_dual_tower_moe.py > logs/LOGG-DUAL-TOWER-01.log 2>&1 &
echo $! > logs/LOGG-DUAL-TOWER-01.pid
```

**确认正常后输出**：
```
✅ 任务已启动 (PID: xxx)
📋 tail -f ~/VIT/logs/LOGG-DUAL-TOWER-01.log
⏱️ 预计 ~15-20min（1M train + gate 训练），完成后告诉我继续
```

### Step 3: 生成图表

图表保存到：
```bash
IMG_DIR="/home/swei20/Physics_Informed_AI/logg/moe/exp/img"
```

### Step 4: 写报告

📄 **报告位置**: `logg/moe/exp/exp_moe_dual_tower_20251228.md`

更新已有的立项报告，填写实验结果。

---

## ✅ 检查清单

- [ ] 脚本创建完成 (`scripts/logg_dual_tower_moe.py`)
- [ ] 训练完成
- [ ] 5 张图表生成 + 保存到 `logg/moe/exp/img/`
- [ ] 必须输出 per-SNR bin R²
- [ ] 必须输出 gate 熵 vs SNR
- [ ] 报告更新 `logg/moe/exp/exp_moe_dual_tower_20251228.md`
- [ ] 同步关键数字到 `moe_snr_roadmap.md`
- [ ] 同步假设验证到 `moe_hub_20251203.md` Q6

---

## 🔧 故障排除

| 问题 | 修复 |
|------|------|
| Gate 训练不收敛 | 降 lr (1e-4)，加 warmup |
| phys_features 提取慢 | 预计算并缓存 |
| experts 加载失败 | 重新训练（~5min） |
| R² 负值 | 检查 scaler 是否 fit 正确 |

---

## 📐 Decision Gate

**Gate-5 验收标准**：

| 结果 | 判定 | 下一步 |
|------|------|--------|
| ΔR² ≥ +0.005 (vs 单塔) | ✅ 通过 | 继续 MVP-4.1 (因子分解) |
| 低 SNR 改善 ≥ +0.01 | ✅ quality gate 有效 | 继续开发 |
| ΔR² < 0 | ❌ 不通过 | fallback 单塔 (Phys-MoE) |

---

## 📚 对比基线

| 方法 | R² | 配置 | 来源 |
|------|-----|------|------|
| Global Ridge | 0.4611 | noise=1, 1M | SCALING-20251223 |
| Oracle Phys MoE (9 exp) | 0.6249 | noise=1, 1M | SCALING-20251223 |
| Soft-gate Phys MoE | 0.59 | noise=1, ρ=0.805 | Scaling Hub |
| SNR-MoE (4 exp) | 0.54 | noise=1, ρ=1.04 | moe_snr_hub |
| **Dual-Tower (目标)** | **≥0.60** | noise=1 | 本实验 |

---

## 🎯 关键验证点

1. **双塔 > 单塔？** — 对比 Dual-Tower vs Phys-only Gate
2. **低 SNR 改善？** — 按 SNR ≤4 切片看 ΔR²
3. **Gate 熵随 SNR 变化？** — 低 SNR 应该更"平"（更 ensemble）
4. **消融：quality 单独贡献？** — phys-only vs dual，看 Δ

