# 🤖 实验 Coding Prompt

> **Experiment ID:** `LOGG-ERR-REPR-01`  
> **日期:** 2025-12-26 | **来源:** `logg/moe/moe_snr_roadmap.md` MVP-0.2  
> **MVP:** MVP-0.2 (Gate-1: Error 表示去泄露)  
> **Status:** 🔴 P0

---

## ⚠️ 核心规则

| 规则 | 说明 |
|------|------|
| **nohup 后台运行** | 所有训练必须 `nohup ... &`，>5分钟不持续追踪 |
| **跨仓库用终端** | 写入 Physics_Informed_AI 用 `cat/echo/cp`，禁止 IDE 工具 |
| **图片必须入报告** | 所有图表必须在报告 §4 中引用，路径 `logg/moe/img/` |
| **语言** | Header 英文 \| 正文中文 \| 图表文字英文 |

---

## 🚀 仓库路由

| Topic | 仓库 | 前缀 |
|-------|------|------|
| **error-deleakage** | `~/VIT` | VIT- |

---

## 🎯 实验目标

**背景**（来自 MVP-0.1 LOGG-ERR-BASE-01）:
- error-only Ridge R² = **0.99**（极严重泄露！）
- Shuffle 后 R² = **-0.98**（依赖波长对齐信息）
- Agg-stats R² = 0.068（不是简单统计量）
- Top 泄露像素: 3277-3388, 3724-3869

**目标**: 构造"只表达观测质量"的 error 表示

**验收标准**:
- ✅ logg R² < 0.05 **且** SNR 预测 R² > 0.5 → 通过 Gate-1
- ❌ 无法同时满足 → 禁用 error，只用 flux 做 MoE

---

## 🦾 去泄露策略（按成本从低到高）

### S1: 同口径归一化

```python
# 让 error 与 flux 做相同归一化，破坏 error 独有信息
def s1_normalize(error, flux, method='median'):
    if method == 'median':
        scale = np.median(flux, axis=-1, keepdims=True)
    elif method == 'l2':
        scale = np.linalg.norm(flux, axis=-1, keepdims=True)
    return error / (scale + 1e-8)
```

### S2: Template × Scale

```python
# 假设 error ≈ s * e0 + δ，只保留 scale s
def s2_template_scale(error, template=None):
    if template is None:
        template = np.mean(error, axis=0)  # 训练集均值作为模板
    
    # 最小二乘拟合 scale
    scale = np.sum(error * template, axis=-1) / np.sum(template ** 2)
    
    # 返回 1D 特征 (或加 median, iqr)
    features = {
        'scale': scale,
        'median': np.median(error, axis=-1),
        'iqr': np.percentile(error, 75, axis=-1) - np.percentile(error, 25, axis=-1)
    }
    return features
```

### S3: 无波长对齐统计

```python
# 使用不依赖像素位置的统计量
def s3_agnostic_stats(error, n_quantiles=5):
    # 对每个样本的 error 排序后取分位数
    sorted_err = np.sort(error, axis=-1)
    quantile_idx = np.linspace(0, error.shape[-1]-1, n_quantiles).astype(int)
    quantiles = sorted_err[:, quantile_idx]
    
    # 可选：加直方图 bin 计数
    hist_features = []
    for e in error:
        hist, _ = np.histogram(e, bins=10)
        hist_features.append(hist / hist.sum())
    
    return np.hstack([quantiles, np.array(hist_features)])
```

### S4: 残差仅做异常检测

```python
# 不用于 logg 回归，只用于 SNR 预测和质量标志
def s4_quality_only(error, template):
    scale = np.sum(error * template, axis=-1) / np.sum(template ** 2)
    residual_norm = np.linalg.norm(error - scale[:, None] * template, axis=-1)
    
    # SNR 近似 = scale / residual_norm
    quality_features = {
        'scale': scale,
        'residual_norm': residual_norm,
        'approx_snr': scale / (residual_norm + 1e-8)
    }
    return quality_features
```

---

## 🧪 实验设计

```yaml
experiment_id: "LOGG-ERR-REPR-01"
repo_path: "~/VIT"

data:
  source: "BOSZ/PFS simulator"
  train_path: "/home/swei20/data/data-20-30-100k/train.h5"
  val_path: "/home/swei20/data/data-20-30-100k/val.h5"
  test_path: "/home/swei20/data/data-20-30-100k/test.h5"
  num_samples: 100000
  num_test_samples: 10000
  
strategies:
  - name: "S1_median"
    method: "s1_normalize"
    params: {method: "median"}
    output_dim: 4096
    
  - name: "S1_l2"
    method: "s1_normalize"
    params: {method: "l2"}
    output_dim: 4096
    
  - name: "S2_scale_only"
    method: "s2_template_scale"
    features: ["scale"]
    output_dim: 1
    
  - name: "S2_scale_median_iqr"
    method: "s2_template_scale"
    features: ["scale", "median", "iqr"]
    output_dim: 3
    
  - name: "S3_5quantile"
    method: "s3_agnostic_stats"
    params: {n_quantiles: 5}
    output_dim: 5
    
  - name: "S3_10quantile"
    method: "s3_agnostic_stats"
    params: {n_quantiles: 10}
    output_dim: 10
    
  - name: "S3_quantile_hist"
    method: "s3_agnostic_stats"
    params: {n_quantiles: 5, add_hist: true}
    output_dim: 15

evaluation:
  # 任务 1: logg 泄露测试（目标 R² < 0.05）
  logg_leakage:
    model: Ridge
    alpha: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    target: "log_g"
    threshold: 0.05
    
  # 任务 2: SNR 预测能力（目标 R² > 0.5）
  snr_prediction:
    model: Ridge
    alpha: [0.01, 0.1, 1.0]
    target: "snr"
    threshold: 0.5

plots:
  - type: "strategy_comparison"
    save: "logg_err_repr_01_strategy_comparison.png"
  - type: "snr_prediction"
    save: "logg_err_repr_01_snr_prediction.png"
  - type: "tradeoff"
    save: "logg_err_repr_01_tradeoff.png"
```

---

## 📊 要生成的图表

| # | 图表类型 | X轴 | Y轴 | 保存路径 |
|---|---------|-----|-----|---------|
| 1 | Bar (策略对比) | Strategy | logg R² | `logg_err_repr_01_strategy_comparison.png` |
| 2 | Bar (SNR 预测) | Strategy | SNR R² | `logg_err_repr_01_snr_prediction.png` |
| 3 | Scatter (权衡) | logg R² (↓) | SNR R² (↑) | `logg_err_repr_01_tradeoff.png` |

### 图表要求

- 所有文字 **英文**
- 策略对比图需标注 threshold 线（logg R² = 0.05）
- SNR 预测图需标注 threshold 线（SNR R² = 0.5）
- 权衡图需标注"通过区域"（左上角）

---

## 🗂️ 参考代码

| 参考脚本 | 可复用 | 说明 |
|---------|--------|------|
| `scripts/logg_error_leakage_audit.py` | 数据加载、Ridge 训练 | MVP-0.1 脚本 |
| `src/lnreg/core.py` | `load_dataset()`, `compute_metrics()` | 通用工具 |
| `src/dataloader/base.py` | `RegSpecDataset` | 数据集类 |

---

## 📋 执行流程

### Step 1: 创建实验脚本

创建 `~/VIT/scripts/logg_error_deleakage.py`：

```python
#!/usr/bin/env python
"""
LOGG-ERR-REPR-01: Error Representation De-Leakage
测试 4 种去泄露策略，找到同时满足：
1. logg R² < 0.05 (去泄露)
2. SNR R² > 0.5 (保持质量信息)
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# === 策略实现 ===

def s1_normalize(error, flux, method='median'):
    """S1: 同口径归一化"""
    if method == 'median':
        scale = np.median(flux, axis=-1, keepdims=True)
    elif method == 'l2':
        scale = np.linalg.norm(flux, axis=-1, keepdims=True)
    return error / (scale + 1e-8)

def s2_template_scale(error, template=None, features=['scale']):
    """S2: template × scale"""
    if template is None:
        template = np.mean(error, axis=0)
    
    scale = np.sum(error * template, axis=-1) / (np.sum(template ** 2) + 1e-8)
    median = np.median(error, axis=-1)
    iqr = np.percentile(error, 75, axis=-1) - np.percentile(error, 25, axis=-1)
    
    feat_dict = {'scale': scale, 'median': median, 'iqr': iqr}
    return np.column_stack([feat_dict[f] for f in features])

def s3_agnostic_stats(error, n_quantiles=5, add_hist=False):
    """S3: 无波长对齐统计"""
    sorted_err = np.sort(error, axis=-1)
    n_pixels = error.shape[-1]
    quantile_idx = np.linspace(0, n_pixels - 1, n_quantiles).astype(int)
    quantiles = sorted_err[:, quantile_idx]
    
    if add_hist:
        hist_features = []
        for e in error:
            hist, _ = np.histogram(e, bins=10, density=True)
            hist_features.append(hist)
        return np.hstack([quantiles, np.array(hist_features)])
    
    return quantiles

def evaluate_strategy(X_train, y_train, X_test, y_test, alphas=[0.001, 0.01, 0.1, 1.0]):
    """评估策略：返回最佳 R²"""
    best_r2 = -np.inf
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        if r2 > best_r2:
            best_r2 = r2
    return best_r2

def main():
    # 1. 加载数据
    from src.dataloader import RegSpecDataset
    cfg = {...}  # 配置
    
    train_ds = RegSpecDataset.from_config(cfg)
    train_ds.load_data(stage='train')
    train_ds.load_params(stage='train')
    train_ds.load_snr(stage='train')
    
    # 2. 提取数据
    error_train = train_ds.error.numpy()
    flux_train = train_ds.flux.numpy()
    logg_train = train_ds.logg
    snr_train = train_ds.snr_no_mask.numpy()
    
    # 3. 测试各策略
    strategies = [
        ('S1_median', lambda e, f: s1_normalize(e, f, 'median')),
        ('S1_l2', lambda e, f: s1_normalize(e, f, 'l2')),
        ('S2_scale', lambda e, f: s2_template_scale(e, features=['scale'])),
        ('S2_scale_med_iqr', lambda e, f: s2_template_scale(e, features=['scale', 'median', 'iqr'])),
        ('S3_5q', lambda e, f: s3_agnostic_stats(e, n_quantiles=5)),
        ('S3_10q', lambda e, f: s3_agnostic_stats(e, n_quantiles=10)),
        ('S3_5q_hist', lambda e, f: s3_agnostic_stats(e, n_quantiles=5, add_hist=True)),
    ]
    
    results = []
    for name, transform in strategies:
        X = transform(error_train, flux_train)
        logg_r2 = evaluate_strategy(X, logg_train, X, logg_train)  # 简化：用 train 评估
        snr_r2 = evaluate_strategy(X, snr_train, X, snr_train)
        
        passed = logg_r2 < 0.05 and snr_r2 > 0.5
        results.append({
            'strategy': name,
            'logg_r2': logg_r2,
            'snr_r2': snr_r2,
            'passed': passed
        })
        print(f"{name}: logg R²={logg_r2:.4f}, SNR R²={snr_r2:.4f}, PASS={passed}")
    
    # 4. 保存结果
    pd.DataFrame(results).to_csv('results/logg_snr_moe/logg_err_repr_01_results.csv', index=False)
    
    # 5. 绘图 (略)

if __name__ == '__main__':
    main()
```

### Step 2: 启动训练

```bash
cd ~/VIT && source init.sh
mkdir -p logs results/logg_snr_moe
nohup python scripts/logg_error_deleakage.py > logs/LOGG-ERR-REPR-01.log 2>&1 &
echo $! > logs/LOGG-ERR-REPR-01.pid
```

**确认正常后输出**：
```
✅ 任务已启动 (PID: xxx)
📋 tail -f ~/VIT/logs/LOGG-ERR-REPR-01.log
⏱️ 预计 ~10min，完成后告诉我继续
```

### Step 3: 生成图表 & 复制

```bash
IMG_DIR="/home/swei20/Physics_Informed_AI/logg/moe/img"
cp ~/VIT/results/logg_snr_moe/logg_err_repr_01_*.png "$IMG_DIR/"
```

### Step 4: 更新报告

```bash
# 填写 exp_logg_err_repr_01_20251226.md 中的 TODO 部分
```

---

## ✅ 检查清单

- [ ] 脚本创建完成 (`scripts/logg_error_deleakage.py`)
- [ ] 7 种策略全部测试完成
- [ ] 3 张图表生成 + 保存到 `logg/moe/img/`
- [ ] 报告更新 `logg/moe/exp/exp_logg_err_repr_01_20251226.md`
- [ ] 同步最佳策略到 `moe_snr_hub.md`
- [ ] 同步状态到 `moe_snr_roadmap.md` MVP-0.2
- [ ] 若通过 → 冻结 `quality_features()` 实现

---

## 🔧 故障排除

| 问题 | 修复 |
|------|------|
| 所有策略 logg R² 仍 > 0.05 | 尝试更激进的压缩（如只用 1 个标量） |
| SNR R² < 0.5 | 可能需要保留更多信息（如增加分位数） |
| SNR 属性不存在 | 调用 `load_snr(stage)` 或用 `snr_no_mask` |

---

## 📐 Decision Gate

**Gate-1 验收标准**：

| 结果 | 判定 | 下一步 |
|------|------|--------|
| 某策略同时满足 logg R² < 0.05 且 SNR R² > 0.5 | ✅ 通过 | 冻结该策略为 `quality_features()`，进入 MVP-2.0 |
| 所有策略都无法同时满足 | ❌ 失败 | 禁用 error 输入，只用 flux 做 MoE |

---

## 📚 相关实验

| Experiment ID | 关系 |
|---------------|------|
| `LOGG-ERR-BASE-01` | MVP-0.1: 泄露基线（R²=0.99） |
| `LOGG-SNR-ORACLE-01` | MVP-1.0: Oracle SNR MoE |
| `LOGG-SNR-GATE-01` | MVP-2.0: Deployable Gate（下一步） |

