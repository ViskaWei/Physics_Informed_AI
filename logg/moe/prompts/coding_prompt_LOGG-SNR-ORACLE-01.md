# 🤖 实验 Coding Prompt

> **Experiment ID:** `LOGG-SNR-ORACLE-01`  
> **日期:** 2025-12-26 | **来源:** `logg/moe/moe_snr_roadmap.md` MVP-1.0  
> **MVP:** MVP-1.0 (Gate-2: Oracle SNR Split Headroom)  
> **Status:** 🔴 P1

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
| **snr-moe** | `~/VIT` | VIT- |

---

## 🎯 实验目标

验证 **"按 SNR 分专家"** 在 oracle routing 下能带来多少 headroom：
- 核心问题：SNR-based MoE 是否值得做？
- 验收标准：**Oracle ΔR² ≥ 0.02**（相对 Global 单模型）
- 若 ΔR² < 0.02 → MoE 不值得，转向 whitening/conditional (Gate-4)

---

## 🧪 实验设计

### 1. SNR Bins 定义（来自 Fisher Multi-mag）

| Bin | 名称 | SNR 范围 | 物理含义 | 预期样本比例 |
|-----|------|----------|---------|-------------|
| **H** | High | SNR > 7 | 信息富，R²_max~0.89 | ~30%+ |
| **M** | Medium | 4 < SNR ≤ 7 | 临界区域，R²_max~0.74 | ~30%+ |
| **L** | Low | 2 < SNR ≤ 4 | 困难区域，R²_max~0.37 | ~20%+ |
| **X** | Extreme | SNR ≤ 2 | 信息悬崖，R²_max~0 | ~10%+ |

### 2. 数据配置

```yaml
data:
  source: "BOSZ simulated spectra"
  root: "/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M"
  train_shards: ["train_200k_{0..4}/dataset.h5"]  # 1M samples
  test_file: "test_1k_0/dataset.h5"
  feature_dim: 4096  # MR arm spectral pixels
  target: "log_g"

snr:
  # SNR 计算方法：使用 snr 列（在 noise_level=1 时的 base SNR）
  # 或从 flux/error 实时计算：snr = median(flux/error) or norm(flux)/norm(error)
  bins:
    H: [7, inf]    # 高质量
    M: [4, 7]      # 中等质量
    L: [2, 4]      # 低质量
    X: [0, 2]      # 极低质量
```

### 3. 模型配置

```yaml
model:
  type: "Ridge"
  alpha: 100000  # 从 scaling_oracle_moe_noise1.py 优化值
  # 每个 SNR bin 训练独立 expert
  experts: 4  # H, M, L, X

preprocessing:
  scaler: "StandardScaler"  # 每个 expert 独立 scaler
  clip_flux: true  # 截断负值

noise:
  level: 1.0  # heteroscedastic Gaussian noise
  apply: "train"  # 训练时 on-the-fly 加噪
```

### 4. 验证方式

```yaml
routing:
  type: "oracle"  # 使用真实 SNR 进行路由
  # test 时每个样本按其真实 SNR 分配到对应 expert

evaluation:
  metrics: ["R²", "MAE", "RMSE"]
  per_bin: true  # 必须输出每个 bin 的 R²
  comparison: "Global Ridge"  # 对照组：不分 bin 的单模型
```

---

## 📊 要生成的图表

| # | 图表类型 | X轴 | Y轴 | 保存路径 |
|---|---------|-----|-----|---------|
| 1 | Bar (对比) | Method (Global vs Oracle) | Test R² | `snr_oracle_moe_comparison.png` |
| 2 | Bar (per-bin) | SNR Bin (H/M/L/X) | R² | `snr_oracle_moe_perbin_r2.png` |
| 3 | Scatter | True log_g | Predicted log_g | `snr_oracle_moe_pred_vs_true.png` |
| 4 | Heatmap | SNR Bin | Oracle vs Global ΔR² | `snr_oracle_moe_delta_heatmap.png` |

### 图表要求

- 所有文字 **英文**
- 包含 ΔR² 标注（判断是否 ≥ 0.02）
- 包含 decision threshold 参考线
- 必须显示每个 bin 的样本数量

---

## 🗂️ 参考代码

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/scaling_oracle_moe_noise1.py` | 数据加载、Ridge训练、可视化框架 | 改 bins 为 SNR-based（4 bins），改路由逻辑 |
| `src/utils/snr_utils.py` | `calculate_snr_norm()`, `calculate_snr_median()` | 直接使用 |
| `src/dataloader/base.py` | `add_noise()` 方法, 数据加载 | 直接使用 |

### 关键复用函数

```python
# 从 scaling_oracle_moe_noise1.py:
load_shards()           # 加载多个 HDF5 shards
add_noise()             # 添加 heteroscedastic noise
train_global_ridge()    # 训练全局 Ridge
create_plots()          # 可视化框架

# 从 src/utils/snr_utils.py:
calculate_snr_norm()    # SNR = ||flux|| / ||error * noise_level||
noise_level_to_snr()    # noise_level → SNR 转换
```

---

## 📋 执行流程

### Step 1: 创建实验脚本

基于 `scaling_oracle_moe_noise1.py`，修改为 SNR-based binning：

```python
# 关键修改点：

# 1. 将 Teff/[M/H] bins 替换为 SNR bins
SNR_BINS = {
    'H': (7.0, np.inf),   # High SNR
    'M': (4.0, 7.0),      # Medium SNR
    'L': (2.0, 4.0),      # Low SNR
    'X': (0.0, 2.0),      # Extreme low SNR
}

# 2. SNR 计算（每个样本）
def compute_sample_snr(flux, error, noise_level=1.0):
    """计算每个样本的 SNR"""
    flux_norm = np.linalg.norm(flux, axis=-1)
    error_norm = np.linalg.norm(error, axis=-1) * noise_level
    return flux_norm / error_norm

# 3. 分配 bin
def assign_snr_bin(snr, bins=SNR_BINS):
    """根据 SNR 值分配到对应 bin"""
    for name, (low, high) in bins.items():
        if low <= snr < high or (high == np.inf and snr >= low):
            return name
    return 'X'  # 默认归入极低

# 4. Oracle 路由
# test 时用真实 SNR 路由，而非 gate 预测
```

### Step 2: 启动训练

```bash
cd ~/VIT && source init.sh
nohup python scripts/logg_snr_oracle_moe.py > logs/LOGG-SNR-ORACLE-01.log 2>&1 &
echo $! > logs/LOGG-SNR-ORACLE-01.pid
```

**确认正常后输出**：
```
✅ 任务已启动 (PID: xxx)
📋 tail -f ~/VIT/logs/LOGG-SNR-ORACLE-01.log
⏱️ 预计 ~10-15min（1M train），完成后告诉我继续
```

### Step 3: 生成图表

图表保存到：
```bash
IMG_DIR="/home/swei20/Physics_Informed_AI/logg/moe/exp/img"
```

### Step 4: 写报告

📄 **模板**: `_backend/template/exp.md`

```bash
cat << 'EOF' > "/home/swei20/Physics_Informed_AI/logg/moe/exp/exp_logg_snr_oracle_01_20251226.md"
# 🧪 Experiment: Oracle SNR-binned MoE

**Experiment ID:** `LOGG-SNR-ORACLE-01`
**Date:** 2025-12-26
**Status:** ✅/❌
**MVP:** MVP-1.0 (Gate-2)

---

## 🔗 上游追溯

| Type | Link |
|------|------|
| Hub | `logg/moe/moe_snr_hub.md` |
| Roadmap | `logg/moe/moe_snr_roadmap.md` MVP-1.0 |
| 验证假设 | Q1: SNR 分域是否有 headroom？ |

---

## ⚡ 核心结论速览

> **一句话总结**: [TODO]

| 假设 | 预期 | 实际 | 验证 |
|------|------|------|------|
| H1: ΔR² ≥ 0.02 | ≥ 0.02 | [TODO] | ✅/❌ |

| 关键数字 | 值 |
|---------|-----|
| Global Ridge R² | [TODO] |
| Oracle MoE R² | [TODO] |
| ΔR² | [TODO] |
| Coverage | [TODO] |

---

## 🎯 目标

[按模板填写]

---

## 🧪 实验设计

[按模板填写]

---

## 📊 实验图表

### Figure 1: Oracle vs Global R² Comparison
![comparison](img/snr_oracle_moe_comparison.png)

### Figure 2: Per-bin R²
![perbin](img/snr_oracle_moe_perbin_r2.png)

### Figure 3: Prediction vs True
![pred_vs_true](img/snr_oracle_moe_pred_vs_true.png)

---

## 💡 关键洞见

[TODO: 填写观察到的洞见]

---

## 📝 结论

[TODO: 核心发现、设计启示、关键数字速查]

---

## 📎 附录

### 数值结果表

| Model | R² | MAE | RMSE | N_train | N_test |
|-------|-----|-----|------|---------|--------|
| Global Ridge | | | | | |
| Oracle MoE | | | | | |
| - Bin H | | | | | |
| - Bin M | | | | | |
| - Bin L | | | | | |
| - Bin X | | | | | |

EOF
```

---

## ✅ 检查清单

- [ ] 脚本创建完成 (`scripts/logg_snr_oracle_moe.py`)
- [ ] 训练完成
- [ ] 4 张图表生成 + 保存到 `logg/moe/exp/img/`
- [ ] 报告写入 `logg/moe/exp/exp_logg_snr_oracle_01_20251226.md`
- [ ] 同步关键数字到 `moe_snr_roadmap.md`
- [ ] 同步假设验证到 `moe_snr_hub.md`

---

## 🔧 故障排除

| 问题 | 修复 |
|------|------|
| 某 bin 样本太少 | 合并相邻 bin 或设置最小样本阈值（<100 则 skip） |
| SNR 计算 NaN | 检查 error=0 情况，clip to 1e-6 |
| R² 负值 | 模型预测太差，检查 scaling/正则化 |

---

## 📐 Decision Gate

**Gate-2 验收标准**：

| 结果 | 判定 | 下一步 |
|------|------|--------|
| ΔR² ≥ 0.02 | ✅ 通过 | 继续 MVP-2.0 (Deployable Gate) |
| ΔR² < 0.02 | ❌ 不通过 | 跳到 Gate-4 (Whitening/Conditional) |

---

## 📚 相关实验

| Experiment ID | 关系 |
|---------------|------|
| `SCALING-20251223-oracle-moe-noise1-01` | 参考实现（Teff/[M/H] binning） |
| `SCALING-20251224-fisher-multi-mag` | SNR 阈值来源 |

