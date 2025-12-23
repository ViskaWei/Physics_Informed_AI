# BOSZ 模拟光谱数据集生成报告

## 概述

本报告记录了用于 PFS (Prime Focus Spectrograph) 光谱分析的 BOSZ 模拟数据集的生成过程、参数配置和技术细节。

| 项目 | 值 |
|------|-----|
| **数据集名称** | mag205_225_lowT_1M |
| **生成日期** | 2025-12-08 至 2025-12-21 |
| **总样本数** | 1,000,000 |
| **总数据量** | 93 GB |
| **生成耗时** | 304 小时 13 分 30 秒 |

---

## 1. 数据集结构

```
/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/
├── train_200k_0/
│   └── dataset.h5          (19 GB, 200,000 样本)
├── train_200k_1/
│   └── dataset.h5          (19 GB, 200,000 样本)
├── train_200k_2/
│   └── dataset.h5          (19 GB, 200,000 样本)
├── train_200k_3/
│   └── dataset.h5          (19 GB, 200,000 样本)
├── train_200k_4/
│   └── dataset.h5          (19 GB, 200,000 样本)
├── shard_index.txt         (分片索引文件)
├── logs/                   (生成日志)
└── README.md               (本文档)
```

---

## 2. 恒星模型参数

### 2.1 BOSZ 模型网格

| 参数 | 值 | 说明 |
|------|-----|------|
| **模型源** | BOSZ (ATLAS9) | Bohlin, Rauch, & Sah 恒星大气模型 |
| **模型分辨率** | R = 50,000 | 高分辨率光谱模型 |
| **模型路径** | `${PFSSPEC_DATA}/models/stellar/grid/bosz/bosz_50000` | |

### 2.2 恒星物理参数范围

| 参数 | 符号 | 最小值 | 最大值 | 单位 |
|------|------|--------|--------|------|
| **有效温度** | T_eff | 3750 | 6000 | K |
| **表面重力** | log_g | 1.0 | 5.0 | dex (cgs) |
| **金属丰度** | [M/H] | -2.5 | 0.75 | dex |
| **红移** | z | 0 | 0 | (静止帧) |
| **消光** | E(B-V) | 0 | 0 | mag |

### 2.3 测光参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **星等范围** | 20.5 - 22.5 mag | i 波段视星等 |
| **星等滤光片** | HSC i-band | `${PFSSPEC_DATA}/subaru/hsc/filters/fHSC-i.txt` |
| **星等分布** | uniform | 均匀分布 |
| **归一化方式** | median | 中值归一化 |
| **归一化波长** | 6500 - 9500 Å | |

---

## 3. 仪器配置 (PFS MR)

### 3.1 光谱仪参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **谱臂** | MR (Medium Resolution) | 中分辨率红臂 |
| **探测器配置** | `${PFSSPEC_DATA}/subaru/pfs/arms/mr.json` | |
| **PSF 模型** | PCA PSF | `mr.2/pca.h5` |

### 3.2 观测条件

| 参数 | 最小值 | 最大值 | 单位 |
|------|--------|--------|------|
| **视宁度 (Seeing)** | 0.5 | 1.5 | arcsec |
| **目标天顶角** | 0 | 45 | degree |
| **目标场角** | 0 | 0.65 | degree |
| **月球天顶角** | 30 | 90 | degree |
| **月球-目标角** | 60 | 180 | degree |
| **月相** | 0 | 0 | (新月) |

### 3.3 曝光设置

| 参数 | 值 | 说明 |
|------|-----|------|
| **单次曝光时间** | 900 秒 | 15 分钟 |
| **曝光次数** | 12 次 | |
| **总曝光时间** | 10,800 秒 | 3 小时 |

### 3.4 天空背景模型

| 组件 | 文件路径 |
|------|----------|
| **天空发射** | `${PFSSPEC_DATA}/subaru/pfs/noise/import/sky/mr/sky.h5` |
| **月光散射** | `${PFSSPEC_DATA}/subaru/pfs/noise/import/moon/mr/moon.h5` |

---

## 4. 采样与插值配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **采样模式** | random | 随机采样 |
| **采样分布** | beta | Beta 分布 |
| **插值模式** | spline | 样条插值 |
| **插值参数** | random | 随机插值参数 |
| **数据块大小** | 1000 | 每块处理样本数 |
| **红化处理** | true | 应用银河消光 |

---

## 5. 生成过程

### 5.1 计算环境

| 项目 | 值 |
|------|-----|
| **主机名** | elephant1 |
| **操作系统** | CentOS 7 (Linux 3.10.0) |
| **CPU** | Intel Xeon E7-4830 @ 2.13GHz |
| **CPU 核心数** | 64 (4 Socket × 8 Core × 2 Thread) |
| **内存** | 1 TB |
| **Python 版本** | 3.10.9 |
| **Conda 环境** | astro-tf211 |

### 5.2 并行配置

为最大化利用 64 核心 CPU，采用以下并行策略：

| 参数 | 值 | 说明 |
|------|-----|------|
| **并发 Shard 数** | 5 | 同时生成 5 个数据分片 |
| **每 Shard 线程数** | 12 | 每个分片使用 12 个 CPU 线程 |
| **总线程数** | 60 | 占用 94% CPU 资源 |
| **系统保留** | 4 核心 | 保证系统响应 |

### 5.3 生成时间线

| 事件 | 时间 | 说明 |
|------|------|------|
| 开始生成 | 2025-12-08 23:47:24 | 启动所有 5 个 shard |
| Shard 2 完成 | 2025-12-21 05:08:02 | 第一个完成 |
| Shard 0 完成 | 2025-12-21 08:20:44 | |
| Shard 1 完成 | 2025-12-21 11:14:02 | |
| Shard 3 完成 | 2025-12-21 15:46:54 | |
| Shard 4 完成 | 2025-12-21 16:00:54 | 最后一个完成 |
| **总耗时** | **304h 13m 30s** | 约 12.7 天 |

### 5.4 生成脚本

生成脚本位置: `/home/swei20/ga_pfsspec_all/generate_sharded_data.sh`

核心命令:
```bash
./bin/sim model bosz pfs \
    --threads 12 \
    --config /datascope/subaru/user/swei20/data/bosz50000/z0/train.json \
             /datascope/subaru/user/swei20/data/bosz50000/z0/inst_pfs_mr.json \
    --out ${OUTPUT_DIR} \
    --sample-count 200000 \
    --seeing 0.5 1.5
```

---

## 6. 配置文件

### 6.1 训练配置 (train.json)

```json
{
    "config": ["common.json"],
    "out": "${PFSSPEC_DATA}/train/pfs_stellar_model/dataset/bosz/nowave/train_1M",
    "sample_count": 1000,
    "z_dist": "uniform",
    "z": [0],
    "mag_dist": "uniform",
    "mag": [20.5, 22.5],
    "ext_dist": "uniform",
    "ext": [0],
    "redden": true
}
```

### 6.2 仪器配置 (inst_pfs_mr.json)

```json
{
    "detector": "${PFSSPEC_DATA}/subaru/pfs/arms/mr.json",
    "detector_map": null,
    "detector_psf_pca": "${PFSSPEC_DATA}/subaru/pfs/psf/import/mr.2/pca.h5",
    "sky": "${PFSSPEC_DATA}/subaru/pfs/noise/import/sky/mr/sky.h5",
    "sky_level": null,
    "moon": "${PFSSPEC_DATA}/subaru/pfs/noise/import/moon/mr/moon.h5",
    "moon_level": null,
    "model_res": null
}
```

### 6.3 通用配置 (common.json)

```json
{
    "dataset": "bosz",
    "pipeline": "pfs",
    "in": "${PFSSPEC_DATA}/models/stellar/grid/bosz/bosz_50000",
    "model_res": 50000.0,
    "sample_mode": "random",
    "sample_dist": "beta",
    "chunk_size": 1000,
    "interp_mode": "spline",
    "interp_param": "random",
    "wave": null,
    "wave_bins": null,
    "wave_log": false,
    "mag_filter": "${PFSSPEC_DATA}/subaru/hsc/filters/fHSC-i.txt",
    "norm": "median",
    "norm_wave": [6500.0, 9500.0],
    "exp_count": 12,
    "exp_time": 900.0,
    "calib_bias": null,
    "noise": null,
    "noise_freeze": false,
    "cont": null,
    "cont_wave": null,
    "redden": false,
    "target_zenith_angle": [0.0, 45.0],
    "target_field_angle": [0.0, 0.65],
    "moon_zenith_angle": [30.0, 90.0],
    "moon_target_angle": [60.0, 180.0],
    "moon_phase": [0.0],
    "M_H": [-2.5, 0.75],
    "T_eff": [3750.0, 6000.0],
    "log_g": [1.0, 5.0]
}
```

---

## 7. 数据使用

### 7.1 加载数据

```python
import h5py

# 加载单个 shard
with h5py.File('/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/train_200k_0/dataset.h5', 'r') as f:
    # 查看数据集结构
    print(list(f.keys()))
    
    # 加载光谱数据
    spectra = f['flux'][:]
    wavelength = f['wave'][:]
    
    # 加载标签
    labels = f['labels'][:]
```

### 7.2 使用索引文件

```python
# 读取所有 shard 路径
with open('/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/shard_index.txt', 'r') as f:
    shard_paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# 遍历所有 shard
for shard_path in shard_paths:
    dataset_path = f"{shard_path}/dataset.h5"
    # 处理每个 shard...
```

---

## 8. 注意事项

1. **数据量大**: 每个 shard 约 19GB，请确保有足够的内存进行数据加载
2. **红移为零**: 本数据集所有样本红移 z=0 (静止帧光谱)
3. **无消光**: 本数据集不包含星际消光 (E(B-V)=0)
4. **月相新月**: 所有样本在新月条件下模拟 (moon_phase=0)
5. **MR 谱臂**: 仅包含中分辨率红臂 (MR) 数据

---

## 9. 相关文件

| 文件 | 路径 |
|------|------|
| 生成脚本 | `/home/swei20/ga_pfsspec_all/generate_sharded_data.sh` |
| 训练配置 | `/datascope/subaru/user/swei20/data/bosz50000/z0/train.json` |
| 仪器配置 | `/datascope/subaru/user/swei20/data/bosz50000/z0/inst_pfs_mr.json` |
| 通用配置 | `/datascope/subaru/user/swei20/data/bosz50000/z0/common.json` |
| 生成日志 | `/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/logs/` |

---

*报告生成时间: 2025-12-22*
*生成工具: pfsspec simulation pipeline*

