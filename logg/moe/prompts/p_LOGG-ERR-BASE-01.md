# 🤖 Coding Prompt: LOGG-ERR-BASE-01

> **日期:** 2025-12-26 | **来源:** `logg/moe/moe_snr_roadmap.md` MVP-0.1
> **Exp ID:** `LOGG-ERR-BASE-01` | **Gate:** Gate-1 (Leakage Audit)

---

## ⚠️ 核心规则

| 规则 | 说明 |
|------|------|
| **nohup 后台运行** | 所有训练必须 `nohup ... &`，>5分钟不持续追踪 |
| **跨仓库用终端** | 写入 Physics_Informed_AI 用 `cat/echo/cp`，禁止 IDE 工具 |
| **图片必须入报告** | 所有图表必须在报告 §3 中引用，路径 `logg/moe/img/` |
| **语言** | Header 英文 \| 正文中文 \| 图表文字英文 |

---

## 🎯 实验目标

**一句话**: 量化 error vector 预测 logg 的"泄露程度"——如果 error-only 模型 R² 很高，说明 error 携带了天体参数信息，必须先去泄露再用于 MoE gate。

**验收标准**:
- ✅ 若 error-only R² < 0.05 → 通过，可进入 Gate-2 (Oracle SNR headroom)
- ❌ 若 error-only R² ≥ 0.05 → 需要进一步压缩/去泄露（进入 MVP-0.2）

**产物**:
1. error-only 的 R²（train/val/test）
2. feature importance 分析：哪些像素贡献最大？是否集中在特定 40 个位置？
3. Sanity checks: Shuffle Test, Mask-only Test

---

## 🗂️ 参考代码

| 参考脚本 | 可复用 | 说明 |
|---------|--------|------|
| `~/VIT/src/lnreg/core.py` | `load_dataset()`, `add_noise()`, `compute_metrics()` | 数据加载、噪声注入、指标计算 |
| `~/VIT/src/dataloader/base.py` | `RegSpecDataset` | 数据集类，含 `.flux`, `.error`, `.labels` |
| `~/VIT/scripts/scaling_oracle_moe_noise1.py` | 数据加载流程 | 参考数据配置 |
| `~/VIT/models/linear_error_sweep/results.csv` | 已有结果 | 对照（但那是用 error 做 flux 回归，不是本实验） |

---

## 🎯 实验规格

```yaml
experiment_id: "LOGG-ERR-BASE-01"
repo_path: "~/VIT"

data:
  source: "BOSZ/PFS simulator"
  train_path: "/home/swei20/data/data-20-30-100k/train.h5"
  val_path: "/home/swei20/data/data-20-30-100k/val.h5"  
  test_path: "/home/swei20/data/data-20-30-100k/test.h5"
  num_samples: 100000  # train
  num_test_samples: 10000  # val/test
  split: "100k/10k/10k"
  param: "log_g"
  
input:
  X: error  # ⚠️ 关键：只用 error，不用 flux
  y: log_g

models:
  - type: LinearRegression  # OLS baseline
  - type: Ridge
    alpha: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
  - type: LightGBM
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1

noise_levels: [0.0, 0.5, 1.0]  # 测试不同噪声下的泄露程度

seed: 42

plots:
  - type: "r2_bar_chart"
    desc: "Error-only R² across models"
    save: "logg_err_base_01_r2_models.png"
  - type: "feature_importance_spectrum"
    desc: "Feature importance vs wavelength (identify leaky pixels)"
    save: "logg_err_base_01_importance_spectrum.png"
  - type: "importance_histogram"
    desc: "Importance distribution (check if concentrated)"
    save: "logg_err_base_01_importance_hist.png"
```

---

## 📋 执行流程

### Step 0: 环境准备
```bash
cd ~/VIT && source init.sh
mkdir -p logs results/logg_snr_moe
```

### Step 1: 创建实验脚本

创建 `~/VIT/scripts/logg_error_leakage_audit.py`：

**核心逻辑**:
```python
# 1. 加载数据（只取 error，不取 flux）
train_ds = RegSpecDataset.from_config(cfg)
train_ds.load_data(stage='train')
train_ds.load_params(stage='train')

X_train = train_ds.error.numpy()  # ⚠️ 只用 error
y_train = train_ds.logg

# 2. 训练多个模型
models = {
    'OLS': LinearRegression(),
    'Ridge_0.001': Ridge(alpha=0.001),
    'Ridge_0.01': Ridge(alpha=0.01),
    'Ridge_0.1': Ridge(alpha=0.1),
    'Ridge_1': Ridge(alpha=1.0),
    'Ridge_10': Ridge(alpha=10.0),
    'Ridge_100': Ridge(alpha=100.0),
}

# 3. 评估并记录 R²
# 4. 提取 feature importance（|coef_| for linear, feature_importances_ for LightGBM）
# 5. 可视化：importance vs wavelength，找出高重要性像素
```

### Step 2: Sanity Checks

**Shuffle Test** (检验是否用了波长对齐信息):
```python
# 在同一 mag/SNR 组内随机打乱 error 向量
# 如果性能几乎不变 → 模型只用"整体尺度"
# 如果大幅下降 → 模型用了"波长对齐细节"（泄露风险高）
```

**Mask-only Test** (检验 mask 位置是否是泄露源):
```python
# 创建 binary mask：有效像素=0，mask 像素=1
# 用 mask 向量做回归
# 如果 R² 很高 → mask 位置是泄露源
```

### Step 3: 运行实验

```bash
cd ~/VIT
nohup python scripts/logg_error_leakage_audit.py \
    --exp-id LOGG-ERR-BASE-01 \
    --output-dir results/logg_snr_moe \
    > logs/LOGG-ERR-BASE-01.log 2>&1 &
echo $! > logs/LOGG-ERR-BASE-01.pid
```

**确认正常后输出**：
```
✅ 任务已启动 (PID: xxx)
📋 tail -f ~/VIT/logs/LOGG-ERR-BASE-01.log
⏱️ 预计 ~5min，完成后告诉我继续
```

### Step 4: 生成图表

```bash
# 图表会在脚本中直接生成
# 保存到: ~/VIT/results/logg_snr_moe/
```

### Step 5: 写报告

```bash
KNOWLEDGE_CENTER="/home/swei20/Physics_Informed_AI"
cat << 'EOF' > "$KNOWLEDGE_CENTER/logg/moe/exp/exp_logg_err_base_01_20251226.md"
# 📗 LOGG-ERR-BASE-01: Error-Only Leakage Baseline

> **MVP:** 0.1 | **Gate:** Gate-1 (Leakage Audit)
> **Author:** Viska Wei | **Date:** 2025-12-26 | **Status:** ✅/❌

---

## 🔗 上游追溯

| 类型 | 链接 |
|------|------|
| Hub | `moe_snr_hub.md` §DG1 |
| Roadmap | `moe_snr_roadmap.md` MVP-0.1 |
| Session | - |

---

## ⚡ 核心结论速览

**一句话**: [TODO: error-only R² = ?，泄露程度 = ?]

**假设验证**:
- [ ] H1: error vector 预测 logg 的 R² < 0.05 → [结果]

**关键数字**:
| 指标 | 值 |
|------|---|
| error-only R² (Ridge best) | TODO |
| error-only R² (LightGBM) | TODO |
| Top-40 像素贡献占比 | TODO |

---

## 🎯 目标

量化 error vector 的"泄露程度"：
1. 如果 error-only R² 接近 0 → error 不携带天体参数信息，可直接用于 gate
2. 如果 error-only R² 很高 → error 携带泄露信息，需要去泄露后再用

---

## 🧪 实验设计

### 2.1 数据
- 训练集: 100k，验证集: 10k，测试集: 10k
- 输入: error vector（4096 维）
- 输出: log_g

### 2.2 模型
- LinearRegression (OLS)
- Ridge (alpha = 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
- LightGBM

### 2.3 Sanity Checks
- Shuffle Test: 打乱波长对齐，检验是否依赖位置信息
- Mask-only Test: 只用 mask 位置，检验是否是泄露源

---

## 📊 实验图表

### Fig 1: Error-Only R² Across Models
![r2_models](img/logg_err_base_01_r2_models.png)
**描述**: [TODO]
**关键观察**: [TODO]

### Fig 2: Feature Importance Spectrum
![importance](img/logg_err_base_01_importance_spectrum.png)
**描述**: [TODO]
**关键观察**: 高重要性像素是否集中在特定 40 个位置？

### Fig 3: Sanity Check Results
[TODO: Shuffle Test, Mask-only Test 结果]

---

## 💡 关键洞见

| # | 洞见 | 证据 | 决策影响 |
|---|------|------|----------|
| 1 | [TODO] | [TODO] | [TODO] |

---

## 📝 结论

### 5.1 核心发现
[TODO]

### 5.2 Gate-1 判定
- [ ] 通过 (R² < 0.05) → 进入 Gate-2
- [ ] 未通过 (R² ≥ 0.05) → 进入 MVP-0.2 (去泄露)

### 5.3 设计启示
[TODO]

---

## 📎 附录

### 6.1 数值结果表

| Model | Train R² | Val R² | Test R² | MAE | RMSE |
|-------|----------|--------|---------|-----|------|
| OLS | | | | | |
| Ridge_0.001 | | | | | |
| ... | | | | | |

### 6.2 Sanity Check 详细结果

**Shuffle Test**:
- 原始 R²: 
- 打乱后 R²: 
- 结论: 

**Mask-only Test**:
- Mask-only R²: 
- 结论: 

EOF
```

---

## ✅ 检查清单

- [ ] 脚本创建并运行成功
- [ ] 训练完成，结果保存
- [ ] 图表生成（英文标签）+ 已在报告 §3 引用
- [ ] 报告撰写完成（中文）
- [ ] 同步结果到 hub.md §DG1
- [ ] 同步结果到 roadmap.md MVP-0.1 状态

---

## 🔧 故障排除

| 问题 | 修复 |
|------|------|
| 数据路径错误 | 检查 `/home/swei20/data/data-20-30-100k/` 是否存在 |
| LightGBM 安装问题 | `pip install lightgbm` |
| 内存不足 | 减少 num_samples |

---

## 📌 后续步骤

根据本实验结果决定：
- **若 R² < 0.05**: 直接进入 MVP-1.0 (Oracle SNR-binned Experts)
- **若 R² ≥ 0.05**: 进入 MVP-0.2 (error 表示去泄露)
  - 策略 S1: 同口径归一化
  - 策略 S2: template×scale
  - 策略 S3: 无对齐统计

