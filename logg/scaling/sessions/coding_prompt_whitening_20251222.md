# 🤖 Coding Prompt: Whitening / SNR Input Experiment

---
> **Experiment ID:** `SCALING-20251222-whitening-01`  
> **MVP:** MVP-1.6 (Whitening/SNR Input)  
> **Date:** 2025-12-22  
> **来源:** `logg/scaling/exp/exp_scaling_whitening_snr_20251222.md`

---

## 🚨 跨仓库写入规则（置顶警告）

> **所有写入 `/home/swei20/Physics_Informed_AI/` 知识中心的操作，必须使用终端命令！**
>
> - ❌ **禁止**：使用 IDE 编辑功能（write/search_replace 等工具）直接写入知识中心
> - ✅ **必须**：使用终端命令（`cat << 'EOF' >`、`echo >>`、`cp`、`tee`）写入

---

# 📋 Prompt 正文

```text
你是实验执行助理。按以下规格执行实验。

🚨🚨🚨 跨仓库写入规则（最高优先级）🚨🚨🚨

所有写入 /home/swei20/Physics_Informed_AI/ 的操作：
❌ 禁止：使用 IDE 工具（write、search_replace、edit_file 等）
✅ 必须：使用终端命令（cat << 'EOF' >、echo >>、cp、tee）

═══════════════════════════════════════
📋 实验规格
═══════════════════════════════════════

experiment_id: SCALING-20251222-whitening-01
mvp_source: MVP-1.6
date: 2025-12-22
hypothesis: H1.7.1 - Whitening (flux/error) 能提升 R² > 0.02

═══════════════════════════════════════
🗂️ 参考代码
═══════════════════════════════════════

主要参考：
1. /home/swei20/VIT/scripts/scaling_ml_ceiling_experiment.py
   - load_shards(): 加载多个 HDF5 shard
   - load_test_data(): 加载 test/val 数据
   - add_noise(): 添加噪声
   - train_ridge(): Ridge 训练
   - train_lightgbm(): LightGBM 训练

2. /home/swei20/VIT/scripts/scaling_ridge_alpha_extended.py
   - 扫描参数的模式

从参考代码复用：
- 数据加载函数 (load_shards, load_test_data)
- 噪声添加函数 (add_noise)
- 训练函数 (train_ridge, train_lightgbm)
- 可视化模式

═══════════════════════════════════════
🔧 新增逻辑：Input Variants
═══════════════════════════════════════

需要实现 6 种输入变体：

```python
def prepare_input_variants(X_train, X_test, error_train, error_test):
    """准备不同的输入表示方式。
    
    Args:
        X_train: (N_train, 4096) 原始 flux
        X_test: (N_test, 4096) 原始 flux
        error_train: (N_train, 4096) 每像素误差
        error_test: (N_test, 4096) 每像素误差
        
    Returns:
        dict: {variant_name: (X_train_processed, X_test_processed)}
    """
    from sklearn.preprocessing import StandardScaler
    
    variants = {}
    
    # 1. raw: 原始 flux，不做任何处理
    variants['raw'] = (X_train.copy(), X_test.copy())
    
    # 2. standardized: StandardScaler (当前 baseline)
    # x' = (x - μ_train) / σ_train
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    variants['standardized'] = (X_train_std, X_test_std)
    
    # 3. centered_only: 只去均值，不缩放
    # x' = x - μ_train
    scaler_center = StandardScaler(with_std=False)
    X_train_center = scaler_center.fit_transform(X_train)
    X_test_center = scaler_center.transform(X_test)
    variants['centered_only'] = (X_train_center, X_test_center)
    
    # 4. std_only: 只缩放，不去均值 (with_mean=False)
    # x' = x / σ_train
    scaler_std = StandardScaler(with_mean=False)
    X_train_stdonly = scaler_std.fit_transform(X_train)
    X_test_stdonly = scaler_std.transform(X_test)
    variants['std_only'] = (X_train_stdonly, X_test_stdonly)
    
    # 5. snr: flux / error (🔥 核心测试)
    # x' = flux / error
    # 注意：error 可能有 0 值，需要 clip
    eps = 1e-8
    X_train_snr = X_train / np.clip(error_train, eps, None)
    X_test_snr = X_test / np.clip(error_test, eps, None)
    variants['snr'] = (X_train_snr, X_test_snr)
    
    # 6. snr_centered: (flux - μ) / error
    # x' = (flux - μ_train) / error
    mu_train = X_train.mean(axis=0, keepdims=True)  # per-feature mean
    X_train_snr_ctr = (X_train - mu_train) / np.clip(error_train, eps, None)
    X_test_snr_ctr = (X_test - mu_train) / np.clip(error_test, eps, None)
    variants['snr_centered'] = (X_train_snr_ctr, X_test_snr_ctr)
    
    return variants
```

═══════════════════════════════════════
📊 数据配置
═══════════════════════════════════════

| 项目 | 配置 |
|------|------|
| **数据路径** | `/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/` |
| **Ridge 训练集** | 5 × 200k shards = 1M samples (Ridge 耗时恒定 ~20s) |
| **LightGBM 训练集** | 100k samples (从第一个 shard 采样，因 1M 太慢) |
| **测试集** | `test_1k_0/dataset.h5` 前 500 (pre-noised) |
| **验证集** | `test_1k_0/dataset.h5` 后 500 (需加噪) |
| **噪声水平** | noise_level = 1.0 |
| **目标变量** | log_g |
| **特征维度** | 4096 |

**⚠️ 数据量差异化配置原因**：
- Ridge: 无论数据大小，训练时间恒定 ~20s，使用全量 1M 数据
- LightGBM: 1M 数据太慢，降为 100k 进行快速测试

**关键**：必须同时加载 `flux` 和 `error` 数据！

HDF5 数据结构：
- `dataset/arrays/flux/value`: (N, 4096) 原始 flux
- `dataset/arrays/error/value`: (N, 4096) 每像素误差
- `dataset/arrays/noisy/value`: (N, 4096) 预加噪的 flux (test 用)

═══════════════════════════════════════
🔬 模型配置
═══════════════════════════════════════

| 模型 | 训练数据量 | 配置 | 来源 |
|------|-----------|------|------|
| **Ridge** | 1M | α = 1e5 | MVP-1.4 (1M 最优) |
| **LightGBM** | 100k | lr=0.05, n_estimators=1400, num_leaves=63 | MVP-1.5 baseline |

注意：
- Ridge 无需再对 snr/snr_centered 做 StandardScaler（已经归一化）
- Ridge 使用全量 1M 数据（耗时恒定 ~20s）
- LightGBM 使用 100k 数据（1M 太慢，快速验证 input variant 效果）
- LightGBM 对 snr/snr_centered 也测试

═══════════════════════════════════════
📈 要画的图（⚠️ 所有文字必须英文！）
═══════════════════════════════════════

### 图 1: Input Variant vs R² (Bar Chart)
- **类型**: 分组柱状图
- **X 轴**: Input Variant (raw, standardized, centered_only, std_only, snr, snr_centered)
- **Y 轴**: Test R²
- **分组**: Ridge (1M), LightGBM (100k)
- **标注**: 每个柱子上方标注具体 R² 值
- **图例说明**: 标明数据量差异 "Ridge (1M)" / "LightGBM (100k)"
- **保存**: `scaling_whitening_comparison.png`

### 图 2: Improvement vs Baseline (Delta Bar)
- **类型**: 水平柱状图
- **X 轴**: ΔR² (vs standardized baseline)
- **Y 轴**: Input Variant
- **分组**: Ridge (1M), LightGBM (100k)
- **高亮**: 正提升绿色，负提升红色
- **保存**: `scaling_whitening_delta.png`

### 图 3: Prediction vs True (Scatter)
- **类型**: 2×3 子图 (6 variants)
- **内容**: y_pred vs y_true, 对角线
- **标注**: 每个子图标注 R² 和 MAE
- **模型**: 选 Ridge (简洁起见)
- **保存**: `scaling_whitening_scatter.png`

### 图 4: Residual Distribution (Histogram)
- **类型**: 2 组 histogram (snr vs standardized)
- **X 轴**: Residual (y_pred - y_true)
- **Y 轴**: Count
- **分组**: Ridge (1M), LightGBM (100k)
- **保存**: `scaling_whitening_residual.png`

═══════════════════════════════════════
🏃 执行步骤
═══════════════════════════════════════

【Step 1】创建实验脚本
```bash
cd ~/VIT
# 基于 scaling_ml_ceiling_experiment.py 创建新脚本
cp scripts/scaling_ml_ceiling_experiment.py scripts/scaling_whitening_experiment.py
# 修改脚本，添加 Input Variant 逻辑
```

【Step 2】运行实验
```bash
cd ~/VIT
source init.sh  # 激活环境

python scripts/scaling_whitening_experiment.py \
    --output ./results/scaling_whitening \
    --img-dir /home/swei20/Physics_Informed_AI/logg/scaling/img \
    --ridge-max-train 1000000 \
    --lgbm-max-train 50000
```

# 参数说明：
# --ridge-max-train 1000000  # Ridge 用 1M 数据（耗时恒定 ~20s）
# --lgbm-max-train 50000     # LightGBM 用 100k 数据（1M 太慢）

【Step 3】检查结果
```bash
# 检查输出
cat ./results/scaling_whitening/metadata.json
ls /home/swei20/Physics_Informed_AI/logg/scaling/img/scaling_whitening_*.png
```

【Step 4】撰写报告 🚨 必须用终端命令！
```bash
KNOWLEDGE_CENTER="/home/swei20/Physics_Informed_AI"

# 基于现有框架填充实验结果
# 文件已存在: logg/scaling/exp/exp_scaling_whitening_snr_20251222.md
# 需要更新 §2-§6 的内容

# 使用 cat 追加或 sed 替换来更新报告
```

【Step 5】更新追踪文件 🚨 必须用终端命令！
```bash
# 更新 kanban
echo "- [x] SCALING-20251222-whitening-01: [结论]" >> "$KNOWLEDGE_CENTER/status/kanban.md"

# 更新 roadmap §4.1 看板视图 (MVP-1.6 Done)
# 更新 hub §3 洞见汇合站 (如有重要发现)
```

═══════════════════════════════════════
📦 交付物清单
═══════════════════════════════════════

| 类型 | 路径 |
|------|------|
| 实验脚本 | `~/VIT/scripts/scaling_whitening_experiment.py` |
| 结果目录 | `~/VIT/results/scaling_whitening/` |
| 指标 CSV | `~/VIT/results/scaling_whitening/whitening_results.csv` |
| 摘要 JSON | `~/VIT/results/scaling_whitening/metadata.json` |
| 实验报告 | `/home/swei20/Physics_Informed_AI/logg/scaling/exp/exp_scaling_whitening_snr_20251222.md` |
| 图表 | `/home/swei20/Physics_Informed_AI/logg/scaling/img/scaling_whitening_*.png` |

**CSV 列说明**：
- `variant`: input variant 名称
- `model`: Ridge / LightGBM
- `train_size`: 训练数据量 (Ridge=1M, LightGBM=100k)
- `test_r2`, `test_mae`, `test_rmse`: 测试指标

🚨 完成后更新（必须用 run_terminal_cmd 执行 bash 命令）：
- `status/kanban.md` → Done 区域
- `logg/scaling/scaling_roadmap.md` §4.1 (MVP-1.6 → Done)
- `logg/scaling/scaling_hub.md` §3 (如有重要洞见)
- `logg/scaling/exp/exp_scaling_whitening_snr_20251222.md` (填充 §3-§6)

═══════════════════════════════════════
✅ 验收标准
═══════════════════════════════════════

| 假设 | 预期 | 验收标准 |
|------|------|---------|
| H1.7.1 | Whitening 能提升 R² | ΔR² > 0.02 (snr/snr_centered vs standardized) |
| (附加) | StandardScaler 不损害性能 | standardized R² ≥ raw R² |

**⚠️ 对比原则**：
- 由于 Ridge (1M) 和 LightGBM (100k) 使用不同数据量
- **只对比同一模型内不同 input variant 的相对差异**
- 不对比 Ridge vs LightGBM 的绝对 R² 值

**结论判定**：
- ✅ H1.7.1 成立：snr 或 snr_centered 比 standardized 提升 > 0.02（同模型内）
- ❌ H1.7.1 不成立：差异在统计误差内 (< 0.02)

═══════════════════════════════════════
🔧 代码骨架参考
═══════════════════════════════════════

```python
#!/usr/bin/env python3
"""
Whitening / SNR Input Experiment
Experiment ID: SCALING-20251222-whitening-01
Hypothesis: H1.7.1 - Whitening (flux/error) 能提升 R² > 0.02
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb

# Constants
EXPERIMENT_ID = "SCALING-20251222-whitening-01"
DATA_ROOT = "/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M"
TRAIN_SHARDS = [f"{DATA_ROOT}/train_200k_{i}/dataset.h5" for i in range(5)]
TEST_FILE = f"{DATA_ROOT}/test_1k_0/dataset.h5"

# 差异化数据量配置
RIDGE_MAX_TRAIN = 1_000_000    # Ridge 耗时恒定 ~20s，使用全量
LGBM_MAX_TRAIN = 50_000       # LightGBM 1M 太慢，使用 100k

RIDGE_ALPHA = 1e5  # Best from MVP-1.4
NOISE_LEVEL = 1.0
SEED = 42

INPUT_VARIANTS = ['raw', 'standardized', 'centered_only', 'std_only', 'snr', 'snr_centered']

LGBM_PARAMS = {
    'learning_rate': 0.05,
    'n_estimators': 5000,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': SEED,
    'verbose': -1,
    'n_jobs': -1,
}


def load_shards(shard_paths, max_samples=None):
    """Load multiple HDF5 shards and concatenate.
    Returns: X (flux), error, y (log_g)
    """
    # ... (复用 scaling_ml_ceiling_experiment.py 的实现)
    pass


def load_test_data():
    """Load test data (first 500 pre-noised) and error."""
    # ... (需要修改以返回 error 数据)
    pass


def add_noise(X, error, noise_level=1.0, seed=SEED):
    """Add heteroscedastic Gaussian noise."""
    np.random.seed(seed)
    noise = np.random.randn(*X.shape) * error * noise_level
    return (X + noise).astype(np.float32)


def prepare_input_variants(X_train, X_test, error_train, error_test):
    """准备不同的输入表示方式。"""
    # ... (按上述实现)
    pass


def train_and_evaluate(variants_ridge, variants_lgbm, y_train_ridge, y_train_lgbm, 
                       y_test, models=['Ridge', 'LightGBM']):
    """对每种输入变体训练模型并评估。
    
    Args:
        variants_ridge: Ridge 用的 input variants (1M 数据)
        variants_lgbm: LightGBM 用的 input variants (100k 数据)
        y_train_ridge: Ridge 用的 y_train (1M)
        y_train_lgbm: LightGBM 用的 y_train (100k)
        y_test: 测试集 y
        models: 要训练的模型列表
    
    Returns:
        pd.DataFrame: 所有结果
    """
    results = []
    
    for variant_name in INPUT_VARIANTS:
        print(f"\n[{variant_name}] Training...")
        
        # Ridge (使用 1M 数据)
        if 'Ridge' in models:
            X_train, X_test = variants_ridge[variant_name]
            # 对 snr 变体不再额外 standardize
            if variant_name in ['snr', 'snr_centered']:
                X_tr, X_te = X_train, X_test
            else:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train)
                X_te = scaler.transform(X_test)
            
            print(f"  Ridge: training with {len(X_tr):,} samples...")
            model = Ridge(alpha=RIDGE_ALPHA, solver='auto', random_state=SEED)
            model.fit(X_tr, y_train_ridge)
            y_pred = model.predict(X_te)
            
            results.append({
                'variant': variant_name,
                'model': 'Ridge',
                'train_size': len(y_train_ridge),
                'test_r2': r2_score(y_test, y_pred),
                'test_mae': mean_absolute_error(y_test, y_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            })
        
        # LightGBM (使用 100k 数据)
        if 'LightGBM' in models:
            X_train, X_test = variants_lgbm[variant_name]
            
            print(f"  LightGBM: training with {len(X_train):,} samples...")
            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            # LightGBM 不需要额外 standardize
            model.fit(X_train, y_train_lgbm)
            y_pred = model.predict(X_test)
            
            results.append({
                'variant': variant_name,
                'model': 'LightGBM',
                'train_size': len(y_train_lgbm),
                'test_r2': r2_score(y_test, y_pred),
                'test_mae': mean_absolute_error(y_test, y_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            })
    
    return pd.DataFrame(results)


def create_plots(df, y_test, predictions_dict, output_dir, img_dir):
    """Create all visualization plots."""
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Bar chart comparison
    # ...
    
    # Plot 2: Delta vs baseline
    # ...
    
    # Plot 3: Scatter plots
    # ...
    
    # Plot 4: Residual distribution
    # ...


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='./results/scaling_whitening')
    parser.add_argument('--img-dir', type=str, 
                        default='/home/swei20/Physics_Informed_AI/logg/scaling/img')
    parser.add_argument('--ridge-max-train', type=int, default=RIDGE_MAX_TRAIN,
                        help='Ridge training samples (default: 1M, time ~20s regardless)')
    parser.add_argument('--lgbm-max-train', type=int, default=LGBM_MAX_TRAIN,
                        help='LightGBM training samples (default: 100k, 1M too slow)')
    args = parser.parse_args()
    
    print("="*80)
    print(f"Whitening / SNR Input Experiment")
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print("="*80)
    print(f"Ridge train size: {args.ridge_max_train:,}")
    print(f"LightGBM train size: {args.lgbm_max_train:,}")
    print("="*80)
    
    # Load data with error - 两套数据量
    print("[1/5] Loading training data for Ridge (1M)...")
    X_train_ridge, error_ridge, y_train_ridge = load_shards(TRAIN_SHARDS, args.ridge_max_train)
    
    print("[2/5] Loading training data for LightGBM (100k)...")
    X_train_lgbm, error_lgbm, y_train_lgbm = load_shards(TRAIN_SHARDS, args.lgbm_max_train)
    
    print("[3/5] Loading test data...")
    X_test, error_test, y_test = load_test_data()
    
    # Add noise to train (两套)
    X_train_ridge_noisy = add_noise(X_train_ridge, error_ridge, NOISE_LEVEL, seed=SEED)
    X_train_lgbm_noisy = add_noise(X_train_lgbm, error_lgbm, NOISE_LEVEL, seed=SEED)
    
    # Prepare input variants (两套)
    print("[4/5] Preparing input variants...")
    variants_ridge = prepare_input_variants(X_train_ridge_noisy, X_test, error_ridge, error_test)
    variants_lgbm = prepare_input_variants(X_train_lgbm_noisy, X_test, error_lgbm, error_test)
    
    # Train and evaluate
    print("[5/5] Training and evaluating...")
    df_results = train_and_evaluate(variants_ridge, variants_lgbm, 
                                     y_train_ridge, y_train_lgbm, y_test)
    
    # Save results
    df_results.to_csv(os.path.join(args.output, 'whitening_results.csv'), index=False)
    
    # Create plots
    create_plots(df_results, y_test, {}, args.output, args.img_dir)
    
    # Hypothesis verification
    print("\n" + "="*80)
    print("HYPOTHESIS VERIFICATION: H1.7.1")
    print("="*80)
    
    # Compare SNR variants vs standardized
    baseline = df_results[(df_results['variant'] == 'standardized') & 
                          (df_results['model'] == 'Ridge')]['test_r2'].values[0]
    snr_r2 = df_results[(df_results['variant'] == 'snr') & 
                        (df_results['model'] == 'Ridge')]['test_r2'].values[0]
    delta = snr_r2 - baseline
    
    print(f"Baseline (standardized) R²: {baseline:.4f}")
    print(f"SNR R²: {snr_r2:.4f}")
    print(f"Delta: {delta:+.4f}")
    print(f"H1.7.1 (ΔR² > 0.02): {'✅ CONFIRMED' if delta > 0.02 else '❌ REJECTED'}")
    
    # Save metadata
    metadata = {
        'experiment_id': EXPERIMENT_ID,
        'timestamp': datetime.now().isoformat(),
        # ...
    }
    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    print(f"Plots saved to: {args.img_dir}")


if __name__ == "__main__":
    main()
```

═══════════════════════════════════════
💡 注意事项
═══════════════════════════════════════

1. **⚠️ 差异化数据量（重要）**
   - Ridge: 1M 数据（训练耗时恒定 ~20s，与数据量无关）
   - LightGBM: 100k 数据（1M 太慢，仅做 input variant 对比测试）
   - 因此 Ridge 和 LightGBM 的 R² 不可直接对比（数据量不同）
   - 对比重点是：**同一模型内，不同 input variant 的相对差异**

2. **Error 数据处理**
   - error 可能包含 0 或极小值
   - 使用 `np.clip(error, 1e-8, None)` 防止除零

3. **SNR 变体的 StandardScaler**
   - 对 `snr` 和 `snr_centered` 变体，Ridge 训练时不再额外 standardize
   - 因为 SNR 化本身就是一种归一化

4. **LightGBM 特性**
   - LightGBM 对 input scaling 不敏感
   - 但为了公平比较，仍然测试所有变体

5. **图表文字**
   - ⚠️ 所有图表中的文字必须使用英文！
   - 中文会显示为乱码方块

6. **跨仓库写入**
   - 实验脚本在 `~/VIT` 执行
   - 报告和图表保存到 `/home/swei20/Physics_Informed_AI/logg/scaling/`
   - 更新知识中心文件必须用终端命令
```

---

> **使用说明**：
> 1. 复制「Prompt 正文」给 Cursor Agent
> 2. Agent 会基于参考代码创建实验脚本
> 3. 执行实验并生成图表
> 4. 填充 exp.md 报告
> 5. 更新 roadmap 和 hub

