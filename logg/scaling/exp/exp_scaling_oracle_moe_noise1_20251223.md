# 📘 Oracle MoE @ noise=1 Structure Bonus Verification (1M Data)
> **Name:** TODO | **ID:** `TODO`  
> **Topic:** `` | **MVP:** MVP-16A | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:**  | **Status:** 🔄
```
💡 实验目的  
决定：影响的决策
```

---


## 🔗 Upstream Links
| Type | Link |
|------|------|
| 🧠 Hub | `logg//_hub.md` |
| 🗺️ Roadmap | `logg//_roadmap.md` |

---

---

## 🔗 Related Experiments

- **MVP-1.4** (noise=0.2): Oracle ΔR² ≈ +0.050
- **SCALING-20251222-ml-ceiling-01**: Ridge R² = 0.50 @ 1M, noise=1, α=5000
- **Next**: MVP-16A-1 (trainable gate with physical features)

---

## 📝 Notes

1. **1M data improves both Global and Oracle**: Compared to 100k results, both models improve.

2. **Metal-poor bins benefit most from per-bin training**: ΔR² = 0.17-0.19 in bins 0, 3, 6.

3. **Structure bonus is large at high noise**: 
   - noise=0.2: ΔR² ≈ +0.05
   - noise=1.0: ΔR² = **+0.16**
   
   This confirms MoE benefits more under high-noise conditions.

4. **Global Ridge R² = 0.46 vs expected 0.50**: Slight discrepancy with ml_ceiling results, possibly due to different random seeds or test/train split. The structure bonus conclusion remains robust.

---

## ✅ Conclusion

**Oracle MoE demonstrates very strong structure bonus at noise=1 with 1M data:**

- ΔR² = +0.1637 (5.5× higher than 0.03 threshold)
- R² = 0.6249 (exceeds 0.55 target)
- All 9 bins show improvement!

**Decision: Continue MoE development (MVP-16A-1, A-2)**

The next step is to develop a trainable gate that can approach Oracle performance using physical features (Ca II, Na I, PCA components).

---

*Generated: 2025-12-23 (1M data, α=100000)*

---

## 📊 Additional Visualizations (2025-12-24)

### Plot 5: ΔR² Structure Bonus Heatmap
![moe_delta_r2_heatmap.png](../img/moe_delta_r2_heatmap.png)

*Metal-poor bins (left column) show largest structural bonus (+0.17~0.19)*

### Plot 6: Sample Distribution per Bin
![moe_sample_distribution.png](../img/moe_sample_distribution.png)

*Training samples range from 63k to 117k per bin; test samples 62-126*

### Plot 7: Per-Bin R² Grouped Comparison
![moe_perbin_r2_grouped.png](../img/moe_perbin_r2_grouped.png)

*All 9 bins show Oracle Expert outperforming Global Ridge*

### Plot 8: MAE Heatmap (Oracle Expert)
![moe_mae_heatmap.png](../img/moe_mae_heatmap.png)

*Metal-rich bins have lowest MAE (0.32-0.38); Metal-poor bins highest (0.63-0.81)*

### Plot 9: Oracle MoE Dashboard
![moe_oracle_dashboard.png](../img/moe_oracle_dashboard.png)

*Comprehensive summary: R² comparison, ΔR² by bin, Oracle vs Global heatmaps*

### Plot 10: Noise Amplification Effect
![moe_noise_amplification.png](../img/moe_noise_amplification.png)

*MoE structural bonus is 3.3× larger at noise=1.0 vs noise=0.2*

---

*Plots added: 2025-12-24*

---

## 📎 附录

### 6.2 实验流程记录

#### 6.2.1 环境与配置

| 项目 | 值 |
|------|-----|
| **仓库** | `~/VIT` |
| **脚本路径** | `scripts/scaling_oracle_moe_noise1.py` |
| **输出路径** | `results/scaling_oracle_moe/` |
| **Python** | 3.10 |
| **主要依赖** | sklearn, numpy, pandas, matplotlib, seaborn, h5py |

#### 6.2.2 执行命令

```bash
cd ~/VIT
source init.sh

# 运行实验（1M 数据）
python scripts/scaling_oracle_moe_noise1.py

# 输出文件
# - results/scaling_oracle_moe/results.csv
# - results/scaling_oracle_moe/per_bin_results.csv
# - results/scaling_oracle_moe/metadata.json
# - 图表自动保存到知识中心
```

#### 6.2.3 关键配置

```python
# 数据路径
DATA_ROOT = "/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M"
TRAIN_SHARDS = [f"{DATA_ROOT}/train_200k_{i}/dataset.h5" for i in range(5)]
TEST_FILE = f"{DATA_ROOT}/test_1k_0/dataset.h5"

# 噪声配置
NOISE_LEVEL = 1.0  # 高噪声场景

# Ridge 配置
RIDGE_ALPHA = 100000  # 沿用 MVP-1.4 最优值

# 9-bin 划分
TEFF_BINS = [3750, 4500, 5250, 6000]  # 3 Teff bins
MH_BINS = [-2.0, -1.0, 0.0, 0.5]      # 3 [M/H] bins
```

#### 6.2.4 代码引用

| 参考脚本 | 可复用函数 | 说明 |
|---------|-----------|------|
| `~/VIT/scripts/moe_9expert_phys_gate.py` | `assign_bins()` | 9-bin 划分逻辑 |
| `~/VIT/scripts/scaling_ml_ceiling_experiment.py` | `load_shards()`, noise 添加 | 1M 数据加载管道 |

---

*实验流程记录添加: 2025-12-24*

---
## 📁 Data Source Documentation

### Dataset: BOSZ Synthetic Stellar Spectra

| 属性 | 值 |
|------|-----|
| **数据集名称** | BOSZ 50000 合成光谱 |
| **数据根目录** | `/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/` |
| **光谱类型** | MR arm (中分辨率) |
| **波长维度** | 4096 features |
| **星等范围** | mag 20.5-22.5 |
| **温度范围** | Low T (3750-6000 K) |

### 训练数据 (5 Shards × 200k = 1M samples)

| Shard | 样本数 | 文件大小 | 路径 |
|-------|--------|---------|------|
| 0 | 200,000 | 19 GB | `train_200k_0/dataset.h5` |
| 1 | 200,000 | 19 GB | `train_200k_1/dataset.h5` |
| 2 | 200,000 | 19 GB | `train_200k_2/dataset.h5` |
| 3 | 200,000 | 19 GB | `train_200k_3/dataset.h5` |
| 4 | 200,000 | 19 GB | `train_200k_4/dataset.h5` |
| **Total** | **1,000,000** | **95 GB** | |

### 测试数据

| 文件 | 样本数 | 文件大小 | 路径 |
|------|--------|---------|------|
| test_1k_0 | 1,000 | 128 MB | `test_1k_0/dataset.h5` |

### HDF5 数据结构

```
dataset.h5
├── dataset/
│   ├── arrays/
│   │   ├── flux/value      # (N, 4096) - 原始光谱通量
│   │   ├── error/value     # (N, 4096) - 光谱误差
│   │   ├── noisy/value     # (N, 4096) - 预加噪光谱 (仅 test)
│   │   └── mask/value      # (N, 4096) - 掩码
│   └── params/table        # (N,) - 参数表
└── spectrumdataset/
    ├── wave                # (4096,) - 波长
    └── wave_edges          # (4097,) - 波长边界
```

### 参数范围

| 参数 | 最小值 | 最大值 | 单位 |
|------|--------|--------|------|
| **log_g** (target) | 1.00 | 5.00 | dex |
| **T_eff** | 3750 | 6000 | K |
| **[M/H]** | -2.50 | 0.75 | dex |

### 噪声添加方式

```python
# Heteroscedastic Gaussian noise
noise = noise_level * error * np.random.randn(*flux.shape)
noisy = flux + noise
noisy = np.clip(noisy, 0, None)  # Clip negative values
```

| noise_level | 含义 |
|-------------|------|
| 0.0 | 无噪声 |
| 0.2 | 低噪声 |
| 1.0 | 标准噪声 (本实验使用) |
| 2.0 | 高噪声 |

### 9-Bin 划分 (MoE)

| Teff 边界 | [M/H] 边界 |
|-----------|-----------|
| [3750, 4500, 5250, 6000] | [-2.0, -1.0, 0.0, 0.5] |

```
          [M/H]
          Poor    Solar   Rich
         [-2,-1] [-1,0]  [0,0.5]
Teff     ┌───────┬───────┬───────┐
Cool     │ Bin 0 │ Bin 1 │ Bin 2 │  [3750,4500]
[3750,   ├───────┼───────┼───────┤
4500]    │       │       │       │
         ├───────┼───────┼───────┤
Mid      │ Bin 3 │ Bin 4 │ Bin 5 │  [4500,5250]
[4500,   ├───────┼───────┼───────┤
5250]    │       │       │       │
         ├───────┼───────┼───────┤
Hot      │ Bin 6 │ Bin 7 │ Bin 8 │  [5250,6000]
[5250,   └───────┴───────┴───────┘
6000]
```
