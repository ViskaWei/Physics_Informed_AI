# 🤖 实验 Coding Prompt

> **Experiment ID:** `MOE-CEILING-C1-01`  
> **日期:** 2025-12-28 | **来源:** `logg/moe/moe_to_ceiling_roadmap_20251228.md` MVP-C1.0  
> **MVP:** MVP-C1.0 (Gate-C1: Oracle Uplift)  
> **Status:** 🔴 P0

---

## ⚠️ 核心规则

| 规则 | 说明 |
|------|------|
| **nohup 后台运行** | 所有训练必须 `nohup ... &`，>5分钟不持续追踪 |
| **跨仓库用终端** | 写入 Physics_Informed_AI 用 `cat/echo/cp`，禁止 IDE 工具 |
| **图片必须入报告** | 所有图表必须在报告 §3 中引用，路径 `logg/moe/exp/img/` |
| **figsize 统一** | 所有图表 `figsize=(6, 5)`，保持一致性 |
| **语言** | Header 英文 \| 正文中文 \| 图表文字英文 |

---

## 🚀 仓库路由

| Topic | 仓库 | 前缀 |
|-------|------|------|
| **per-bin-expert-sweep** | `~/VIT` | VIT- |

---

## 🎯 实验目标

**核心任务**：把 Oracle 从 "Ridge-Oracle ~0.627" 抬到 ≥0.70（追平 ViT），最终目标 ≥0.75

| 验证问题 | 验收标准 | 下一步 |
|---------|---------|--------|
| Per-bin 最优专家能抬升 oracle？ | ΔR² ≥ +0.05 vs Ridge-Oracle | 继续 MVP-C2 保 ρ |
| Oracle-Hybrid ≥ 0.70？ | R² ≥ 0.70 | 已追平 ViT，可行 |
| Metal-poor bins 改善？ | Bin3/6 ΔR² ≥ +0.05 | 瓶颈解锁 |

**核心思路**：
- 保持 9 物理 bin (Teff×[M/H]) 不变
- 每个 bin **独立选择最优专家类型**：Ridge / LightGBM / 1D-CNN
- 使用 **Oracle routing**（真值分配）先确定 headroom
- 组合成 **Oracle-Hybrid**：每个 bin 用各自最优专家

---

## 🧪 实验设计

### 1. 数据配置

```yaml
data:
  source: "BOSZ simulated spectra"
  root: "/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M"
  train_shards: ["train_200k_{0..4}/dataset.h5"]  # 1M samples
  test_file: "test_10k/dataset.h5"  # ⚠️ 必须用 10k（口径冻结）
  feature_dim: 4096  # MR arm spectral pixels
  target: "log_g"

noise:
  level: 1.0  # heteroscedastic Gaussian noise
  apply: "train + test"
```

### 2. Binning 配置（沿用 9-bin）

```yaml
binning:
  type: "Teff × [M/H] grid"
  teff_boundaries: [3750, 4500, 5250, 6000]  # 3 bins
  mh_boundaries: [-2.0, -1.0, 0.0, 0.5]       # 3 bins
  total_bins: 9
  
bin_names:
  0: "Cool Poor"    # Teff < 4500, [M/H] < -1
  1: "Cool Solar"   # Teff < 4500, -1 ≤ [M/H] < 0
  2: "Cool Rich"    # Teff < 4500, [M/H] ≥ 0
  3: "Mid Poor"     # 4500 ≤ Teff < 5250, [M/H] < -1  ⚠️ 瓶颈
  4: "Mid Solar"    # 4500 ≤ Teff < 5250, -1 ≤ [M/H] < 0
  5: "Mid Rich"     # 4500 ≤ Teff < 5250, [M/H] ≥ 0
  6: "Hot Poor"     # Teff ≥ 5250, [M/H] < -1  ⚠️ 瓶颈
  7: "Hot Solar"    # Teff ≥ 5250, -1 ≤ [M/H] < 0
  8: "Hot Rich"     # Teff ≥ 5250, [M/H] ≥ 0
```

### 3. 专家候选类型

```yaml
expert_candidates:
  - type: "Ridge"
    alpha_sweep: [1, 10, 100, 1000, 10000, 100000, 1000000]  # per-bin 选最优
    note: "当前 baseline，各 bin 统一用 α=100k"
    
  - type: "LightGBM"
    params:
      n_estimators: 500
      max_depth: 5
      learning_rate: 0.05
      num_leaves: 20
      feature_fraction: 0.3
      bagging_fraction: 0.7
      min_child_samples: 50
      reg_alpha: 1.0
      reg_lambda: 1.0
      device_type: "gpu"
    note: "已验证 Bin3 +0.056 (MVP-15)"
    
  - type: "1D-CNN"
    config:
      channels: [32, 64, 32]
      kernel_size: 5
      pool_size: 4
      fc_dim: 64
      epochs: 50
      batch_size: 256
      lr: 1e-3
    note: "可选，只在关键 bin 尝试"
```

### 4. 实验流程

```
Step 1: 数据加载
├── 加载 1M train (5 shards)
├── 加载 10k test
└── 按 Teff×[M/H] 分配 bin labels

Step 2: Per-bin Expert Training
├── For each bin (0-8):
│   ├── 提取该 bin 的训练数据
│   ├── 训练 3 种专家：
│   │   ├── Ridge (α sweep → 选最优 α)
│   │   ├── LightGBM
│   │   └── 1D-CNN (可选)
│   └── 在该 bin 的 test 数据上评估 R²
└── 记录每个 bin 的 best expert & R²

Step 3: Oracle-Hybrid 组装
├── 每个 bin 选择最优专家
├── Oracle routing（用真值 Teff/[M/H] 分配）
└── 计算 overall R² (weighted by bin size)

Step 4: 对比分析
├── Ridge-Oracle vs Hybrid-Oracle
├── Per-bin 改进分析
└── Metal-poor (Bin3/6) 特别关注
```

---

## 📊 要生成的图表

| # | 图表类型 | X轴 | Y轴 | 保存路径 |
|---|---------|-----|-----|---------|
| 1 | Grouped Bar | Bin (0-8) | R² | `ceiling_perbin_expert_comparison.png` |
| 2 | Heatmap | Expert Type | Bin | `ceiling_expert_selection_heatmap.png` |
| 3 | Bar | Method | Overall R² | `ceiling_oracle_hybrid_vs_baseline.png` |
| 4 | Bar | Bin (sorted by difficulty) | ΔR² vs Ridge | `ceiling_perbin_delta_r2.png` |
| 5 | Scatter | True log_g | Predicted (Oracle-Hybrid) | `ceiling_hybrid_pred_vs_true.png` |

### 图表详细说明

**图 1: Per-bin Expert Comparison**
- 每个 bin 显示 3 组 bar：Ridge / LightGBM / 1D-CNN
- 用不同颜色标注 best expert
- 重点标注 Bin3/6 (metal-poor)

**图 2: Expert Selection Heatmap**
- 行：9 个 bin
- 列：Expert 类型
- 值：R² (颜色深浅)
- 星号标注每个 bin 的最优选择

**图 3: Overall Comparison**
- 对比 4 种方法：
  1. Global Ridge (baseline)
  2. Ridge-Oracle (当前 oracle)
  3. **Oracle-Hybrid (本实验目标)**
  4. ViT (参考线，~0.70)

**图 4: Per-bin Delta R²**
- 按 difficulty 排序（低 → 高）
- 显示 Oracle-Hybrid vs Ridge-Oracle 的 ΔR²
- 绿色=正增益，红色=负增益

---

## 🗂️ 参考代码

> **强制规则**：
> - ❌ 禁止在此写任何代码块、代码骨架、示例代码
> - ✅ Agent 执行时必须先阅读下方路径中的代码，理解逻辑后再修改
> - 💡 这样做确保复用已有代码逻辑，避免不一致

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `~/VIT/scripts/scaling_oracle_moe_noise1.py` | 数据加载、9-bin 定义、Ridge 训练 | 添加 LightGBM/CNN 专家 |
| `~/VIT/scripts/moe_lgbm_expert.py` | LightGBM 配置、per-bin 训练框架 | 扩展到所有 9 bin |
| `~/VIT/scripts/train_lightgbm_1m.py` | LightGBM 全量训练 | 参考超参数 |
| `~/VIT/scripts/train_ridge_1m_optimal.py` | Ridge α sweep | 参考 per-bin α 选择 |

### 关键复用函数

```
# 从 scaling_oracle_moe_noise1.py:
load_shard_data()           # 加载单个 HDF5 shard
add_heteroscedastic_noise() # 添加异方差噪声
assign_bins()               # 按 Teff×[M/H] 分配 bin
BinSpec dataclass           # Bin 规格定义
train_bin_expert()          # 训练单个 bin 的 Ridge

# 从 moe_lgbm_expert.py:
LGBM_PARAMS                 # LightGBM 配置（防过拟合）
train_lgbm_expert()         # 训练 LightGBM 专家
```

---

## 📋 执行流程

### Step 1: 创建实验脚本

```bash
cd ~/VIT
# 创建脚本：scripts/moe_ceiling_expert_sweep.py
```

### Step 2: 启动训练

```bash
cd ~/VIT && source init.sh
nohup python scripts/moe_ceiling_expert_sweep.py > logs/MOE-CEILING-C1-01.log 2>&1 &
echo $! > logs/MOE-CEILING-C1-01.pid
```

**确认正常后输出**：
```
✅ 任务已启动 (PID: xxx)
📋 tail -f ~/VIT/logs/MOE-CEILING-C1-01.log
⏱️ 预计 ~30-45min（9 bin × 3 expert types），完成后告诉我继续
```

### Step 3: 生成图表

图表保存到：
```bash
IMG_DIR="/home/swei20/Physics_Informed_AI/logg/moe/exp/img"
```

### Step 4: 写报告

📄 **报告位置**: `logg/moe/exp/exp_moe_ceiling_expert_sweep_20251228.md`

---

## ✅ 检查清单

- [ ] 脚本创建完成 (`scripts/moe_ceiling_expert_sweep.py`)
- [ ] 训练完成（9 bin × 3 expert types）
- [ ] 5 张图表生成 + 保存到 `logg/moe/exp/img/`
- [ ] 必须输出 per-bin R² 表格
- [ ] 必须输出 Oracle-Hybrid overall R²
- [ ] 报告创建 `logg/moe/exp/exp_moe_ceiling_expert_sweep_20251228.md`
- [ ] 同步关键数字到 `moe_to_ceiling_roadmap_20251228.md`
- [ ] 同步假设验证到 `moe_hub_20251203.md`

---

## 🔧 故障排除

| 问题 | 修复 |
|------|------|
| LightGBM OOM | 减少 num_leaves (20→15)，增加 min_child_samples |
| 某 bin 样本太少 | 检查分布，考虑合并相邻 bin |
| 1D-CNN 不收敛 | 降 lr (1e-3 → 1e-4)，加 BatchNorm |
| R² 负值 | 检查 train/test 噪声是否一致 |

---

## 📐 Decision Gate

**Gate-C1 验收标准**：

| 结果 | 判定 | 下一步 |
|------|------|--------|
| Oracle-Hybrid ≥ 0.70 | ✅ 通过 L1 | 继续 MVP-C2 保 ρ |
| ΔR² ≥ +0.05 vs Ridge-Oracle | ✅ 通过 | Hybrid 有效 |
| Oracle-Hybrid < 0.65 | ⚠️ 不足 | 进入 C3 (共享 trunk) |
| Bin3/6 无改善 | ⚠️ 注意 | 继续 MVP-C1.1 专项救援 |

---

## 📚 对比基线

| 方法 | R² | 配置 | 来源 |
|------|-----|------|------|
| Global Ridge | 0.4957 | noise=1, 1M, test=10k | card_ridge_1m_optimal |
| Oracle MoE (9×Ridge) | 0.627 | noise=1, 1M, test=10k | LOGG-DUAL-TOWER-01 |
| Phys-only Gate | 0.601 | noise=1, 1M, ρ=0.84 | LOGG-DUAL-TOWER-01 |
| ViT (参考) | ~0.70 | noise=1 | exp_vit_scaling |
| **Oracle-Hybrid (目标)** | **≥0.70** | noise=1 | 本实验 |

---

## 🎯 关键输出

### 必须输出的表格

**表 1: Per-bin Expert R² 对比**

| Bin | Name | Ridge (α=?) | LightGBM | 1D-CNN | Best | ΔR² vs Ridge |
|-----|------|-------------|----------|--------|------|--------------|
| 0 | Cool Poor | ? | ? | - | ? | ? |
| 1 | Cool Solar | ? | ? | - | ? | ? |
| 2 | Cool Rich | ? | ? | - | ? | ? |
| **3** | **Mid Poor** | ? | ? | ? | ? | **?** |
| 4 | Mid Solar | ? | ? | - | ? | ? |
| 5 | Mid Rich | ? | ? | - | ? | ? |
| **6** | **Hot Poor** | ? | ? | ? | ? | **?** |
| 7 | Hot Solar | ? | ? | - | ? | ? |
| 8 | Hot Rich | ? | ? | - | ? | ? |

**表 2: Overall Summary**

| Method | Overall R² | ΔR² vs Baseline | Notes |
|--------|-----------|-----------------|-------|
| Global Ridge | 0.4957 | - | baseline |
| Ridge-Oracle | 0.627 | +0.131 | 当前 oracle |
| **Oracle-Hybrid** | **?** | **?** | 本实验目标 |

---

## 📝 重点关注

1. **Metal-poor bins (3, 6)** 是否有显著改善？
   - MVP-15 已验证：Bin3 LGBM +0.056 ✅，Bin6 LGBM -0.032 ❌
   - 本实验需重新验证 @ noise=1, test=10k

2. **Per-bin α 选择**
   - 不同 bin 可能需要不同的 Ridge α
   - 记录每个 bin 的最优 α

3. **1D-CNN 只在关键 bin 尝试**
   - 优先在 Bin3/6 尝试
   - 如果 LightGBM 效果好，可跳过 CNN

4. **样本分布**
   - 记录每个 bin 的 train/test 样本数
   - 如果某 bin 样本太少（<1000 train），标注风险

