# 🤖 实验 Coding Prompt: Ridge-Init ResMLP

> **日期:** 2025-12-27 | **ID:** `VIT-20251227-nn-01`  
> **来源:** `logg/NN/exp/exp_ridge_init_resmlp_20251227.md`

---

## ⚠️ 核心规则

| 规则 | 说明 |
|------|------|
| **nohup 后台运行** | 所有训练必须 `nohup ... &`，>5分钟不持续追踪 |
| **跨仓库用终端** | 写入 Physics_Informed_AI 用 `cat/echo/cp`，禁止 IDE 工具 |
| **图片必须入报告** | 所有图表必须在报告 §3 中引用，路径 `logg/NN/img/` |
| **语言** | Header 英文 \| 正文中文 \| 图表文字英文 |

---

## 🚀 仓库路由

| Topic | 仓库 | 前缀 |
|-------|------|------|
| **NN / ResMLP** | `~/VIT` | VIT- |

---

## 🎯 实验规格

```yaml
experiment_id: "VIT-20251227-nn-01"
experiment_name: "Ridge-Init ResMLP"
repo_path: "~/VIT"

data:
  source: "BOSZ → PFS MR"
  path: "~/VIT/data/mag205_225_lowT_1M" 
  # 使用 train_200k_0 (32k 样本) 初始验证，成功后扩展
  train_size: 32000
  val_size: 10000
  test_size: 10000

noise:
  type: "heteroscedastic"
  sigma: 1.0
  apply_to: "train + val + test"

model:
  type: "RidgeResMLP"
  variants:
    V1_baseline: { strategy: "none", description: "无Ridge, 纯ResMLP" }
    V2_concat: { strategy: "concat", description: "输入concat Ridge预测" }
    V3_init: { strategy: "init", description: "第一层Ridge权重初始化" }
    V4_residual: { strategy: "residual", description: "学习Ridge残差" }
    V5_shortcut: { strategy: "shortcut", description: "输出层skip Ridge" }
  architecture:
    hidden_dim: 512
    n_blocks: 3  # 1 stem + 3 ResBlocks + 1 head = 5层
    bottleneck_ratio: 0.5
    activation: "gelu"
    norm: "LayerNorm"
    dropout: 0.1

training:
  epochs: 200
  batch_size: 2048
  optimizer: "AdamW"
  lr: 3e-4
  weight_decay: 1e-4
  scheduler: "CosineAnnealingLR"
  warmup_epochs: 10
  seed: 42
  early_stopping: 50
  gradient_clip: 1.0

ridge_pretrain:
  alpha: 200  # 32k 数据最优 alpha
  model_path: "~/VIT/models/ridge/lnreg_l2_a200_n32k_nz1p0.pkl"

plots:
  - { type: "strategy_comparison", save: "ridge_resmlp_strategy_compare.png" }
  - { type: "training_curves", save: "ridge_resmlp_training_curves.png" }
  - { type: "depth_ablation", save: "ridge_resmlp_depth_ablation.png" }
  - { type: "residual_scatter", save: "ridge_resmlp_residual_scatter.png" }
```

---

## 📋 执行流程

### Step 0: 准备 Ridge 模型

```bash
cd ~/VIT && source init.sh

# 检查是否已有 Ridge 模型
ls -la models/ridge/*32k*nz1p0*

# 如果没有，先训练 Ridge (约 1-2 分钟)
python -c "
from sklearn.linear_model import Ridge
import pickle
import h5py
import numpy as np
import pandas as pd

# Load 32k data
DATA_PATH = '/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/train_200k_0/dataset.h5'
with h5py.File(DATA_PATH, 'r') as f:
    flux = f['dataset/arrays/flux/value'][:32000].astype(np.float32)
    error = f['dataset/arrays/error/value'][:32000].astype(np.float32)
df = pd.read_hdf(DATA_PATH)[:32000]
logg = df['log_g'].values.astype(np.float32)

# Add noise
np.random.seed(42)
noisy_flux = flux + np.random.randn(*flux.shape) * error * 1.0

# Train Ridge
ridge = Ridge(alpha=200)
ridge.fit(noisy_flux, logg)

# Save
data = {'model': ridge, 'alpha': 200, 'noise_level': 1.0}
with open('models/ridge/ridge_a200_n32k_nz1p0.pkl', 'wb') as f:
    pickle.dump(data, f)
print('Ridge model saved!')
print(f'Weights shape: {ridge.coef_.shape}')
"
```

### Step 1: 创建 ResMLP 模型

在 `~/VIT/src/nn/models/resmlp.py` 创建新模型：

```python
"""
Ridge-Initialized ResMLP for Spectroscopic Regression.

Combines Ridge linear prior with deep residual MLP.
Supports 5 strategies:
  - V1 (none): Pure ResMLP baseline
  - V2 (concat): Concat Ridge prediction to input
  - V3 (init): Initialize first layer with Ridge weights
  - V4 (residual): Learn Ridge residual, add back at output
  - V5 (shortcut): Skip connection from Ridge pred to output
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Optional
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ResMLP_Config:
    """Configuration for Ridge ResMLP."""
    input_dim: int = 4096
    hidden_dim: int = 512
    n_blocks: int = 3
    bottleneck_ratio: float = 0.5
    dropout: float = 0.1
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    
    # Ridge integration strategy
    strategy: Literal["none", "concat", "init", "residual", "shortcut"] = "concat"
    ridge_path: Optional[str] = None
    
    @property
    def bottleneck_dim(self) -> int:
        return int(self.hidden_dim * self.bottleneck_ratio)


class ResBlock(nn.Module):
    """Residual block with bottleneck structure."""
    
    def __init__(self, dim: int, bottleneck_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class RidgeResMLP(nn.Module):
    """
    Ridge-Initialized ResMLP.
    
    Architecture:
        Input (4096) → Stem → ResBlock×N → Head → Output (1)
    
    With optional Ridge integration via:
        - concat: Input becomes (4097) with Ridge prediction
        - init: First layer uses Ridge weights
        - residual: Output = Ridge_pred + MLP(x)
        - shortcut: Output += Ridge_pred
    """
    
    def __init__(self, config: ResMLP_Config):
        super().__init__()
        self.config = config
        self.strategy = config.strategy
        
        # Load Ridge model if needed
        self.ridge_weights = None
        self.ridge_bias = None
        if config.ridge_path and config.strategy != "none":
            self._load_ridge(config.ridge_path)
        
        # Input dimension (depends on strategy)
        in_dim = config.input_dim
        if config.strategy == "concat":
            in_dim += 1  # Add Ridge prediction
        
        # Stem: project to hidden_dim
        self.stem = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Initialize first layer with Ridge weights if strategy == "init"
        if config.strategy == "init" and self.ridge_weights is not None:
            self._init_stem_from_ridge()
        
        # ResBlocks
        self.blocks = nn.ModuleList([
            ResBlock(config.hidden_dim, config.bottleneck_dim, config.dropout)
            for _ in range(config.n_blocks)
        ])
        
        # Head: project to output
        self.head = nn.Linear(config.hidden_dim, 1)
        
        # Optional: learnable weight for shortcut
        if config.strategy == "shortcut":
            self.shortcut_weight = nn.Parameter(torch.tensor(0.5))
    
    def _load_ridge(self, path: str):
        """Load Ridge weights from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        ridge = data['model']
        weights = ridge.coef_.flatten().astype(np.float32)
        bias = float(ridge.intercept_) if np.isscalar(ridge.intercept_) else float(ridge.intercept_[0])
        
        self.register_buffer('ridge_weights', torch.from_numpy(weights))
        self.register_buffer('ridge_bias', torch.tensor([bias]))
        
        print(f"[RidgeResMLP] Loaded Ridge weights from {path}")
        print(f"  Shape: {weights.shape}, Strategy: {self.strategy}")
    
    def _init_stem_from_ridge(self):
        """Initialize stem's first layer with Ridge weights."""
        if self.ridge_weights is None:
            return
        
        with torch.no_grad():
            # Expand Ridge weights to all hidden neurons with scaling
            n_hidden = self.config.hidden_dim
            scale = 1.0 / np.sqrt(n_hidden)
            
            # Each neuron starts with scaled Ridge weights
            for i in range(n_hidden):
                self.stem[0].weight.data[i] = self.ridge_weights * scale
            
            self.stem[0].bias.data.fill_(self.ridge_bias.item() / n_hidden)
        
        print(f"[RidgeResMLP] Initialized stem with Ridge weights (scale={scale:.4f})")
    
    def _get_ridge_pred(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Ridge prediction."""
        if self.ridge_weights is None:
            return torch.zeros(x.size(0), 1, device=x.device)
        return F.linear(x, self.ridge_weights.unsqueeze(0), self.ridge_bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, input_dim) spectral flux
        
        Returns:
            (batch,) predicted log_g
        """
        # Get Ridge prediction if needed
        ridge_pred = None
        if self.strategy in ["concat", "residual", "shortcut"]:
            ridge_pred = self._get_ridge_pred(x)
        
        # Modify input for concat strategy
        if self.strategy == "concat":
            x = torch.cat([x, ridge_pred], dim=-1)
        
        # Forward through network
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        
        # Apply Ridge integration at output
        if self.strategy == "residual":
            out = ridge_pred + out
        elif self.strategy == "shortcut":
            out = self.shortcut_weight * ridge_pred + (1 - self.shortcut_weight) * out
        
        return out.squeeze(-1)
    
    def get_param_count(self) -> int:
        """Get total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return (
            f"RidgeResMLP(\n"
            f"  strategy={self.strategy},\n"
            f"  hidden_dim={self.config.hidden_dim},\n"
            f"  n_blocks={self.config.n_blocks},\n"
            f"  params={self.get_param_count():,}\n"
            f")"
        )


def create_ridge_resmlp(
    input_dim: int = 4096,
    hidden_dim: int = 512,
    n_blocks: int = 3,
    dropout: float = 0.1,
    strategy: str = "concat",
    ridge_path: Optional[str] = None,
) -> RidgeResMLP:
    """Factory function for RidgeResMLP."""
    config = ResMLP_Config(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        dropout=dropout,
        strategy=strategy,
        ridge_path=ridge_path,
    )
    return RidgeResMLP(config)
```

### Step 2: 创建训练脚本

创建 `~/VIT/scripts/train_ridge_resmlp.py`：

```python
#!/usr/bin/env python3
"""
Train Ridge-Initialized ResMLP for log_g prediction.

Usage:
    python scripts/train_ridge_resmlp.py --strategy concat --hidden 512 --blocks 3
    
Strategies:
    - none: Pure ResMLP (baseline)
    - concat: Concat Ridge prediction to input
    - init: Initialize first layer with Ridge weights
    - residual: Learn Ridge residual
    - shortcut: Output skip connection
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nn.models.resmlp import create_ridge_resmlp, ResMLP_Config


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/train_200k_0/dataset.h5")
RIDGE_PATH = Path(__file__).resolve().parents[1] / "models/ridge/ridge_a200_n32k_nz1p0.pkl"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results/ridge_resmlp"
IMG_DIR = Path("/home/swei20/Physics_Informed_AI/logg/NN/img")

TRAIN_SIZE = 32000
VAL_SIZE = 10000
TEST_SIZE = 10000
NOISE_LEVEL = 1.0
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="concat",
                        choices=["none", "concat", "init", "residual", "shortcut"])
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(path, n_train, n_val, n_test, noise_level, seed):
    """Load and split data with noise injection."""
    print(f"\n[Data] Loading from {path}...")
    
    with h5py.File(path, 'r') as f:
        total = n_train + n_val + n_test
        flux = f['dataset/arrays/flux/value'][:total].astype(np.float32)
        error = f['dataset/arrays/error/value'][:total].astype(np.float32)
    
    df = pd.read_hdf(path)[:total]
    logg = df['log_g'].values.astype(np.float32)
    
    # Add noise
    np.random.seed(seed)
    noisy_flux = flux + np.random.randn(*flux.shape).astype(np.float32) * error * noise_level
    
    # Split
    X_train = noisy_flux[:n_train]
    y_train = logg[:n_train]
    X_val = noisy_flux[n_train:n_train+n_val]
    y_val = logg[n_train:n_train+n_val]
    X_test = noisy_flux[n_train+n_val:]
    y_test = logg[n_train+n_val:]
    
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for X, y in loader:
        X = X.to(device)
        pred = model(X)
        preds.append(pred.cpu().numpy())
        targets.append(y.numpy())
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)
    return r2, mae, preds, targets


def main():
    args = parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Ridge-ResMLP Training: Strategy={args.strategy}")
    print(f"{'='*60}")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        DATA_PATH, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, NOISE_LEVEL, args.seed
    )
    
    # Create dataloaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    
    # Create model
    ridge_path = str(RIDGE_PATH) if args.strategy != "none" else None
    model = create_ridge_resmlp(
        input_dim=4096,
        hidden_dim=args.hidden,
        n_blocks=args.blocks,
        dropout=args.dropout,
        strategy=args.strategy,
        ridge_path=ridge_path,
    )
    model = model.to(device)
    print(f"\n{model}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_r2 = -float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_r2': [], 'val_mae': []}
    
    print(f"\n[Training] Starting {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_r2, val_mae, _, _ = evaluate(model, val_loader, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_r2'].append(val_r2)
        history['val_mae'].append(val_mae)
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f"best_{args.strategy}.pt")
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={train_loss:.4f}, val_R²={val_r2:.4f}, val_MAE={val_mae:.4f}")
        
        if patience_counter >= 50:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    elapsed = time.time() - start_time
    print(f"\n[Training] Completed in {elapsed:.1f}s")
    
    # Load best model and evaluate on test
    model.load_state_dict(torch.load(OUTPUT_DIR / f"best_{args.strategy}.pt"))
    test_r2, test_mae, test_preds, test_targets = evaluate(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"[RESULTS] Strategy: {args.strategy}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Best Val R²: {best_val_r2:.4f}")
    print(f"{'='*60}")
    
    # Save results
    results = {
        'strategy': args.strategy,
        'hidden_dim': args.hidden,
        'n_blocks': args.blocks,
        'dropout': args.dropout,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'best_val_r2': best_val_r2,
        'epochs_trained': len(history['train_loss']),
        'params': model.get_param_count(),
        'elapsed_seconds': elapsed,
    }
    
    # Append to CSV
    results_file = OUTPUT_DIR / "results.csv"
    df_results = pd.DataFrame([results])
    if results_file.exists():
        df_existing = pd.read_csv(results_file)
        df_results = pd.concat([df_existing, df_results], ignore_index=True)
    df_results.to_csv(results_file, index=False)
    
    print(f"\n[Output] Results saved to {results_file}")
    
    return results


if __name__ == "__main__":
    main()
```

### Step 3: 运行所有变体

```bash
cd ~/VIT && source init.sh

# 创建日志目录
mkdir -p logs results/ridge_resmlp

# 运行 5 个变体（每个约 5-10 分钟，总计 ~30 分钟）
EXP_ID="VIT-20251227-nn-01"

# V1: Baseline (无 Ridge)
nohup python scripts/train_ridge_resmlp.py --strategy none --hidden 512 --blocks 3 --gpu 0 \
    > logs/${EXP_ID}_v1.log 2>&1 &
echo "V1 PID: $!"

# V2: Ridge-Concat
nohup python scripts/train_ridge_resmlp.py --strategy concat --hidden 512 --blocks 3 --gpu 1 \
    > logs/${EXP_ID}_v2.log 2>&1 &
echo "V2 PID: $!"

# V3: Ridge-Init
nohup python scripts/train_ridge_resmlp.py --strategy init --hidden 512 --blocks 3 --gpu 2 \
    > logs/${EXP_ID}_v3.log 2>&1 &
echo "V3 PID: $!"

# V4: Ridge-Residual
nohup python scripts/train_ridge_resmlp.py --strategy residual --hidden 512 --blocks 3 --gpu 3 \
    > logs/${EXP_ID}_v4.log 2>&1 &
echo "V4 PID: $!"

# V5: Ridge-Shortcut
nohup python scripts/train_ridge_resmlp.py --strategy shortcut --hidden 512 --blocks 3 --gpu 4 \
    > logs/${EXP_ID}_v5.log 2>&1 &
echo "V5 PID: $!"

echo "All variants launched!"
```

**确认正常后输出**：
```
✅ 任务已启动 (5 个变体并行)
📋 查看日志:
   tail -f ~/VIT/logs/VIT-20251227-nn-01_v1.log  # V1: Baseline
   tail -f ~/VIT/logs/VIT-20251227-nn-01_v2.log  # V2: Concat
   tail -f ~/VIT/logs/VIT-20251227-nn-01_v3.log  # V3: Init
   tail -f ~/VIT/logs/VIT-20251227-nn-01_v4.log  # V4: Residual
   tail -f ~/VIT/logs/VIT-20251227-nn-01_v5.log  # V5: Shortcut
⏱️ 预计每个 ~5-10 min，完成后告诉我继续
```

### Step 4: 生成图表

创建 `~/VIT/scripts/plot_ridge_resmlp.py`：

```python
#!/usr/bin/env python3
"""Generate plots for Ridge-ResMLP experiments."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results/ridge_resmlp"
IMG_DIR = Path("/home/swei20/Physics_Informed_AI/logg/NN/img")
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Baselines
BASELINES = {
    'Ridge': 0.458,
    'MLP (2L)': 0.498,
    'LightGBM': 0.536,
}

def plot_strategy_comparison():
    """Fig 1: Bar chart comparing all strategies."""
    df = pd.read_csv(OUTPUT_DIR / "results.csv")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    strategies = df['strategy'].values
    r2_scores = df['test_r2'].values
    
    x = np.arange(len(strategies))
    bars = ax.bar(x, r2_scores, color=['#2ecc71' if s != 'none' else '#95a5a1' for s in strategies])
    
    # Add baseline lines
    colors = ['#e74c3c', '#3498db', '#9b59b6']
    for (name, val), color in zip(BASELINES.items(), colors):
        ax.axhline(y=val, linestyle='--', color=color, alpha=0.7, label=name)
    
    # Labels
    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel('Test R²', fontsize=12)
    ax.set_title('Ridge-ResMLP: Strategy Comparison (32k samples, noise=1.0)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['V1: Baseline', 'V2: Concat', 'V3: Init', 'V4: Residual', 'V5: Shortcut'])
    ax.legend(loc='lower right')
    ax.set_ylim(0.4, 0.6)
    
    # Add value labels on bars
    for bar, val in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / "ridge_resmlp_strategy_compare.png", dpi=150)
    print(f"Saved: {IMG_DIR / 'ridge_resmlp_strategy_compare.png'}")
    plt.close()


if __name__ == "__main__":
    plot_strategy_comparison()
    print("\nAll plots generated!")
```

运行绘图:
```bash
python scripts/plot_ridge_resmlp.py
```

### Step 5: 写报告

训练完成后，更新实验报告：

```bash
# 读取结果
cat ~/VIT/results/ridge_resmlp/results.csv

# 用终端命令更新报告的核心结论部分
# (根据实际结果填写)
```

---

## 🗂️ 参考代码

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `src/nn/models/mlp.py` | MLPConfig, 初始化逻辑 | 添加 ResBlock 结构 |
| `train_nn.py` | 训练框架, 参数解析 | 简化为单策略训练 |
| `scripts/train_ridge_1m_optimal.py` | 数据加载逻辑 | 使用 32k 数据 |
| `src/nn/trainer.py` | Trainer 类 | 可选复用 |

---

## ✅ 检查清单

- [ ] Ridge 模型准备完成 (`models/ridge/ridge_a200_n32k_nz1p0.pkl`)
- [ ] ResMLP 模型代码 (`src/nn/models/resmlp.py`)
- [ ] 训练脚本 (`scripts/train_ridge_resmlp.py`)
- [ ] 5 个变体训练完成
- [ ] 图表生成 (英文)
  - [ ] `ridge_resmlp_strategy_compare.png`
- [ ] 更新实验报告 `logg/NN/exp/exp_ridge_init_resmlp_20251227.md`

---

## 🔧 故障排除

| 问题 | 修复 |
|------|------|
| NaN / Loss 爆炸 | 降 lr 到 1e-4，加 warmup |
| OOM | 减 batch_size 到 1024 |
| Ridge 文件不存在 | 运行 Step 0 先训练 Ridge |
| 收敛慢 | 增加 blocks 到 4，或增大 hidden 到 1024 |
| 过拟合 | 增加 dropout 到 0.2-0.3 |

---

## 📊 预期结果参考

| 变体 | 预期 R² | 说明 |
|------|---------|------|
| V1: Baseline | ~0.50-0.52 | 纯 ResMLP，应接近 MLP baseline |
| V2: Concat | ~0.51-0.54 | 注入 Ridge 先验 |
| V3: Init | ~0.50-0.53 | 权重初始化可能收敛更快 |
| V4: Residual | ~0.52-0.55 | 学习残差，理论最优 |
| V5: Shortcut | ~0.51-0.54 | 介于 concat 和 residual |

**成功标准**:
- 任一变体 R² > 0.536 → 超越 LightGBM ✅
- 任一变体 R² > 0.498 → 超越 MLP baseline ✅
- Ridge 变体 > Baseline 变体 → Ridge 初始化有效 ✅
