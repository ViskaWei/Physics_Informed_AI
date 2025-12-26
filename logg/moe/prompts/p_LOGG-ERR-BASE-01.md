# 🤖 实验 Coding Prompt

> **Experiment ID:** `LOGG-ERR-BASE-01`  
> **日期:** 2025-12-26 | **来源:** `logg/moe/moe_snr_roadmap.md` MVP-0.1  
> **MVP:** MVP-0.1 (Gate-1: Leakage Audit & Error 表示冻结)  
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
| **error-leakage** | `~/VIT` | VIT- |

---

## 🎯 实验目标

量化 **error vector 预测 logg 的"泄露程度"**：
- 核心问题：error vector 是否携带天体参数信息（logg 泄露）？
- 验收标准：**error-only R² < 0.05** → 通过 Gate-1
- 若 R² ≥ 0.05 → error 泄露严重，需进入 MVP-0.2 去泄露

**背景**：
- 用户观察到 error-only 线性回归 R²=0.91（极高泄露）
- 96% error 像素相似，仅 **40/4096** 不同
- 这 40 个位置可能对应"随谱型/谱线深度变化的 Poisson 项 / mask / throughput 特征"

---

## 🧪 实验设计

### 1. 数据配置

```yaml
data:
  source: "BOSZ/PFS simulator"
  root: "/home/swei20/data/data-20-30-100k"
  train_file: "train.h5"
  val_file: "val.h5"
  test_file: "test.h5"
  num_samples: 100000  # train
  num_test_samples: 10000  # val/test
  feature_dim: 4096
  target: "log_g"

input:
  X: error  # ⚠️ 关键：只用 error，不用 flux
  y: log_g
```

### 2. 模型配置

```yaml
models:
  linear:
    - type: LinearRegression  # OLS baseline
    - type: Ridge
      alpha: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
  
  tree:
    - type: LightGBM
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
      
noise_levels: [0.0, 0.5, 1.0]  # 测试不同噪声下的泄露程度
seed: 42
```

### 3. Sanity Checks（必做）

| Check | 目的 | 方法 | 判断标准 |
|-------|------|------|---------|
| **Shuffle Test** | 检验是否用了波长对齐信息 | 在同一 mag/SNR 组内随机打乱 error 向量 | 性能几乎不变 → 只用整体尺度；大幅下降 → 用了位置细节（泄露） |
| **Mask-only Test** | 检验 mask 位置是否是泄露源 | 只用 mask 向量（binary: 有效=0, 坏像素=1）做回归 | R² 高 → mask 是泄露源 |
| **Top-40 Test** | 检验那 40 个异常像素 | 只用 Top-40 高 importance 像素做回归 | R² 高 → 这些像素是泄露核心 |

---

## 📊 要生成的图表

| # | 图表类型 | X轴 | Y轴 | 保存路径 |
|---|---------|-----|-----|---------|
| 1 | Bar (对比) | Model | Test R² | `logg_err_base_01_r2_models.png` |
| 2 | Spectrum | Wavelength (pixel index) | Feature Importance | `logg_err_base_01_importance_spectrum.png` |
| 3 | Histogram | Importance value | Count | `logg_err_base_01_importance_hist.png` |
| 4 | Bar (sanity) | Test Type | R² | `logg_err_base_01_sanity_checks.png` |

### 图表要求

- 所有文字 **英文**
- Spectrum 图需标注 Top-40 像素位置（用红色竖线）
- 包含 threshold 参考线（R² = 0.05）
- 必须显示 error-only 与 flux-only 对照

---

## 🗂️ 参考代码

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `src/lnreg/core.py` | `load_dataset()`, `add_noise()`, `compute_metrics()`, `get_importance()` | 直接使用 |
| `src/dataloader/base.py` | `RegSpecDataset` (含 `.flux`, `.error`, `.logg`) | 直接使用 |
| `scripts/scaling_oracle_moe_noise1.py` | 数据加载流程、可视化框架 | 参考 |
| `train_lightgbm.py` | LightGBM 训练模板 | 参考 |

### 关键复用函数

```python
# 从 src/lnreg/core.py:
load_dataset(data_config, stage)   # 加载数据集
add_noise(X, error, noise_level)   # 添加异方差噪声
compute_metrics(y_true, y_pred)    # 计算 R², MAE, RMSE
get_importance(model)              # 提取 |coef_| 或 feature_importances_

# 从 src/dataloader/base.py:
ds.flux     # 光谱 flux
ds.error    # 误差向量
ds.logg     # log_g 标签 (需先调用 load_params)
```

---

## 📋 执行流程

### Step 1: 创建实验脚本

创建 `~/VIT/scripts/logg_error_leakage_audit.py`：

```python
#!/usr/bin/env python
"""
LOGG-ERR-BASE-01: Error-Only Leakage Baseline
量化 error vector 预测 logg 的泄露程度
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb

# 关键点：
# 1. 只用 error 作为输入 X（不是 flux）
# 2. 训练多个模型对比
# 3. 提取 feature importance 定位泄露像素
# 4. 做 Shuffle Test 和 Mask-only Test

def main():
    # 1. 加载数据
    from src.dataloader import RegSpecDataset
    cfg = {
        'data': {
            'file_path': '/home/swei20/data/data-20-30-100k/train.h5',
            'val_path': '/home/swei20/data/data-20-30-100k/val.h5',
            'test_path': '/home/swei20/data/data-20-30-100k/test.h5',
            'num_samples': 100000,
            'num_test_samples': 10000,
        },
        'noise': {'noise_level': 0.0},
        'output_dir': './temp'
    }
    
    train_ds = RegSpecDataset.from_config(cfg)
    train_ds.load_data(stage='train')
    train_ds.load_params(stage='train')
    
    test_ds = RegSpecDataset.from_config(cfg)
    test_ds.load_data(stage='test')
    test_ds.load_params(stage='test')
    
    # ⚠️ 关键：只用 error，不用 flux
    X_train = train_ds.error.numpy()
    y_train = train_ds.logg
    X_test = test_ds.error.numpy()
    y_test = test_ds.logg
    
    # 2. 训练模型
    models = {
        'OLS': LinearRegression(),
        'Ridge_0.001': Ridge(alpha=0.001),
        'Ridge_0.01': Ridge(alpha=0.01),
        'Ridge_0.1': Ridge(alpha=0.1),
        'Ridge_1': Ridge(alpha=1.0),
        'Ridge_10': Ridge(alpha=10.0),
        'Ridge_100': Ridge(alpha=100.0),
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results.append({'model': name, 'r2': r2, 'mae': mae})
        print(f"{name}: R² = {r2:.4f}, MAE = {mae:.4f}")
    
    # 3. LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    r2_lgb = r2_score(y_test, y_pred_lgb)
    results.append({'model': 'LightGBM', 'r2': r2_lgb, 'mae': mean_absolute_error(y_test, y_pred_lgb)})
    
    # 4. Feature Importance
    best_ridge = Ridge(alpha=0.001).fit(X_train, y_train)
    importance = np.abs(best_ridge.coef_)
    top_40_idx = np.argsort(importance)[-40:]
    
    # 5. Sanity Checks
    # ... Shuffle Test, Mask-only Test, Top-40 Test
    
    # 6. 可视化
    # ... 生成 4 张图表
    
    # 7. 保存结果
    pd.DataFrame(results).to_csv('results/logg_snr_moe/logg_err_base_01_results.csv', index=False)

if __name__ == '__main__':
    main()
```

### Step 2: 启动训练

```bash
cd ~/VIT && source init.sh
mkdir -p logs results/logg_snr_moe
nohup python scripts/logg_error_leakage_audit.py > logs/LOGG-ERR-BASE-01.log 2>&1 &
echo $! > logs/LOGG-ERR-BASE-01.pid
```

**确认正常后输出**：
```
✅ 任务已启动 (PID: xxx)
📋 tail -f ~/VIT/logs/LOGG-ERR-BASE-01.log
⏱️ 预计 ~5min，完成后告诉我继续
```

### Step 3: 生成图表

图表保存到：
```bash
IMG_DIR="/home/swei20/Physics_Informed_AI/logg/moe/exp/img"
cp ~/VIT/results/logg_snr_moe/*.png "$IMG_DIR/"
```

### Step 4: 写报告

📄 **模板**: `_backend/template/exp.md`

```bash
cat << 'EOF' > "/home/swei20/Physics_Informed_AI/logg/moe/exp/exp_logg_err_base_01_20251226.md"
# 🧪 Experiment: Error-Only Leakage Baseline

**Experiment ID:** `LOGG-ERR-BASE-01`
**Date:** 2025-12-26
**Status:** ✅/❌
**MVP:** MVP-0.1 (Gate-1)

---

## 🔗 上游追溯

| Type | Link |
|------|------|
| Hub | `logg/moe/moe_snr_hub.md` §DG1 |
| Roadmap | `logg/moe/moe_snr_roadmap.md` MVP-0.1 |
| 验证假设 | Q2.1: error-only R² 能否压到 < 0.05？ |

---

## ⚡ 核心结论速览

> **一句话总结**: [TODO: error-only R² = ?, 泄露程度 = ?]

| 假设 | 预期 | 实际 | 验证 |
|------|------|------|------|
| H1: error-only R² < 0.05 | < 0.05 | [TODO] | ✅/❌ |

| 关键数字 | 值 |
|---------|-----|
| error-only R² (Ridge best) | [TODO] |
| error-only R² (LightGBM) | [TODO] |
| Top-40 像素贡献占比 | [TODO] |
| Shuffle Test ΔR² | [TODO] |
| Mask-only R² | [TODO] |

---

## 🎯 目标

量化 error vector 的"泄露程度"：
1. 如果 error-only R² 接近 0 → error 不携带天体参数信息，可直接用于 gate
2. 如果 error-only R² 很高 → error 携带泄露信息，需要去泄露后再用

---

## 🧪 实验设计

### 2.1 数据
- 训练集: 100k，验证集: 10k，测试集: 10k
- 输入: **error vector（4096 维）** ← 不是 flux
- 输出: log_g

### 2.2 模型
- LinearRegression (OLS)
- Ridge (alpha = 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
- LightGBM (n_estimators=100, max_depth=6)

### 2.3 Sanity Checks
| Test | 目的 | 结果 |
|------|------|------|
| Shuffle Test | 检验波长对齐依赖 | [TODO] |
| Mask-only Test | 检验 mask 是否是泄露源 | [TODO] |
| Top-40 Test | 检验 40 个像素是否核心泄露 | [TODO] |

---

## 📊 实验图表

### Figure 1: Error-Only R² Across Models
![r2_models](img/logg_err_base_01_r2_models.png)
**描述**: [TODO]
**关键观察**: [TODO]

### Figure 2: Feature Importance Spectrum
![importance](img/logg_err_base_01_importance_spectrum.png)
**描述**: 高重要性像素是否集中在特定位置？
**关键观察**: [TODO: 标注 Top-40 像素位置]

### Figure 3: Importance Histogram
![hist](img/logg_err_base_01_importance_hist.png)
**描述**: 重要性分布是否集中？
**关键观察**: [TODO]

### Figure 4: Sanity Check Results
![sanity](img/logg_err_base_01_sanity_checks.png)
**描述**: Shuffle Test, Mask-only Test, Top-40 Test 对比
**关键观察**: [TODO]

---

## 💡 关键洞见

| # | 洞见 | 证据 | 决策影响 |
|---|------|------|----------|
| I1 | [TODO] | [TODO] | [TODO] |
| I2 | [TODO] | [TODO] | [TODO] |

---

## 📝 结论

### 5.1 核心发现
[TODO]

### 5.2 Gate-1 判定
- [ ] 通过 (R² < 0.05) → 进入 Gate-2 (Oracle SNR headroom)
- [ ] 未通过 (R² ≥ 0.05) → 进入 MVP-0.2 (去泄露)

### 5.3 设计启示
[TODO]

### 5.4 关键数字速查
| 指标 | 值 | 意义 |
|------|-----|------|
| error-only R² | [TODO] | 泄露程度 |
| Top-40 占比 | [TODO] | 泄露是否集中 |

---

## 📎 附录

### 6.1 数值结果表

| Model | Train R² | Val R² | Test R² | MAE | RMSE |
|-------|----------|--------|---------|-----|------|
| OLS | | | | | |
| Ridge_0.001 | | | | | |
| Ridge_0.01 | | | | | |
| Ridge_0.1 | | | | | |
| Ridge_1 | | | | | |
| Ridge_10 | | | | | |
| Ridge_100 | | | | | |
| LightGBM | | | | | |

### 6.2 Sanity Check 详细结果

**Shuffle Test**:
- 原始 R²: [TODO]
- 打乱后 R²: [TODO]
- ΔR²: [TODO]
- 结论: [TODO]

**Mask-only Test**:
- Mask-only R²: [TODO]
- 结论: [TODO]

**Top-40 Test**:
- Top-40 only R²: [TODO]
- 占全部 R² 比例: [TODO]
- 结论: [TODO]

EOF
```

---

## ✅ 检查清单

- [ ] 脚本创建完成 (`scripts/logg_error_leakage_audit.py`)
- [ ] 训练完成
- [ ] 4 张图表生成 + 保存到 `logg/moe/exp/img/`
- [ ] 报告写入 `logg/moe/exp/exp_logg_err_base_01_20251226.md`
- [ ] 同步关键数字到 `moe_snr_roadmap.md` MVP-0.1 状态
- [ ] 同步假设验证到 `moe_snr_hub.md` §DG1

---

## 🔧 故障排除

| 问题 | 修复 |
|------|------|
| 数据路径错误 | 检查 `/home/swei20/data/data-20-30-100k/` 是否存在 |
| LightGBM 安装问题 | `pip install lightgbm` |
| 内存不足 | 减少 num_samples |
| logg 属性不存在 | 确保调用 `load_params()` 后再访问 `ds.logg` |

---

## 📐 Decision Gate

**Gate-1 验收标准**：

| 结果 | 判定 | 下一步 |
|------|------|--------|
| R² < 0.05 | ✅ 通过 | 继续 MVP-1.0 (Oracle SNR-binned Experts) |
| R² ≥ 0.05 | ❌ 不通过 | 进入 MVP-0.2 (error 表示去泄露) |

**去泄露策略（若不通过）**：
- S1: 同口径归一化（error 与 flux 同 scale）
- S2: template×scale（只保留标量 s）
- S3: 无对齐统计（sorted quantiles / histogram）
- S4: 残差仅做异常检测

---

## 📚 相关实验

| Experiment ID | 关系 |
|---------------|------|
| `LOGG-ERR-REPR-01` | MVP-0.2: error 表示去泄露 |
| `LOGG-SNR-ORACLE-01` | MVP-1.0: Oracle SNR-binned Experts |
| `models/linear_error_sweep/results.csv` | 已有 error 回归结果（对照） |
