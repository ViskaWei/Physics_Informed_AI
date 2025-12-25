# BOSZ Regular Grid Datasets for Fisher/CRLB Analysis

**Generated:** 2024-12-24  
**Updated:** 2024-12-25 (added mag22, mag22.5)

**Purpose:** Error array generation on regular grid for Fisher Information Matrix / CRLB calculation

## Datasets

| Dataset | Magnitude | Path |
|---------|-----------|------|
| grid_mag18_lowT | 18.0 | `/datascope/subaru/user/swei20/data/bosz50000/grid/grid_mag18_lowT` |
| grid_mag20_lowT | 20.0 | `/datascope/subaru/user/swei20/data/bosz50000/grid/grid_mag20_lowT` |
| grid_mag215_lowT | 21.5 | `/datascope/subaru/user/swei20/data/bosz50000/grid/grid_mag215_lowT` |
| grid_mag22_lowT | 22.0 | `/datascope/subaru/user/swei20/data/bosz50000/grid/grid_mag22_lowT` |
| grid_mag225_lowT | 22.5 | `/datascope/subaru/user/swei20/data/bosz50000/grid/grid_mag225_lowT` |
| grid_mag23_lowT | 23.0 | `/datascope/subaru/user/swei20/data/bosz50000/grid/grid_mag23_lowT` |

## SNR 计算说明

### 数据来源
- **flux** 和 **error** 均由 **PFS 模拟器** 直接生成（非手动计算）
- error 表示每个 pixel 的预期测量误差（1σ）
- 模拟器考虑了：观测条件（seeing, zenith angle, moon phase）、仪器响应、光子噪声等

### SNR 定义
```
SNR = flux / error   (per pixel)
```

### 汇总统计方式
- **SNR Median**: 所有样本、所有 pixel 的 median(flux/error)
- **SNR Range (5-95%)**: 第 5 和第 95 百分位

## SNR Summary

| Magnitude | N Spectra | SNR Median | SNR Range (5-95%) | Error Median |
|-----------|-----------|------------|-------------------|--------------|
| 18.0 | 30,182 | 87.4 | [40.5, 94.8] | 0.0058 |
| 20.0 | 30,182 | 24.0 | [7.1, 27.4] | 0.0210 |
| 21.5 | 30,182 | 7.1 | [1.8, 8.4] | 0.0703 |
| 22.0 | 30,182 | 4.6 | [1.1, 5.5] | 0.1085 |
| 22.5 | 30,182 | 3.0 | [0.7, 3.5] | 0.1690 |
| 23.0 | 30,182 | 1.9 | [0.5, 2.3] | 0.2649 |

## Fisher/CRLB Results Summary

| Magnitude | SNR | R²_max (median) | R²_max (90%) | Schur Decay |
|-----------|-----|-----------------|--------------|-------------|
| 18.0 | 87.4 | **0.9994** | 0.9999 | 0.6641 |
| 20.0 | 24.0 | **0.9906** | 0.9983 | 0.6842 |
| 21.5 | 7.1 | **0.8914** | 0.9804 | 0.6906 |
| 22.0 | 4.6 | **0.7396** | 0.9530 | 0.6921 |
| 22.5 | 3.0 | **0.3658** | 0.8854 | 0.6922 |
| 23.0 | 1.9 | **0.0000** | 0.7180 | 0.6923 |

**Key Insight:** SNR~4 是 R²_max median > 0.5 的临界点；SNR < 2 时信息几乎完全丧失。

## Physical Parameters (Grid Mode)

| Parameter | Values | Step |
|-----------|--------|------|
| T_eff | 3500-7000 K | 250K |
| log_g | 0.0-5.0 | 0.5 dex |
| [M/H] | -2.5 to +0.75 | 0.25 dex |

**Total:** 30,182 valid grid points per dataset

## Observational Parameters (Fixed)

| Parameter | Value |
|-----------|-------|
| Seeing | 1.0 arcsec |
| Exp Count | 4 |
| Exp Time | 900s |
| Zenith Angle | 22.5° |
| Field Angle | 0.325° |
| Moon Phase | 0.0 (new moon) |

## Data Format

| Field | Shape | Path in HDF5 |
|-------|-------|--------------|
| flux | (30182, 4096) | `dataset/arrays/flux/value` |
| error | (30182, 4096) | `dataset/arrays/error/value` |
| wave | (4096,) | `spectrumdataset/wave` |
| params | (30182,) | `dataset/params/table` |

**Wave range:** 7100.2 - 8849.8 Å (MR arm)

## Generation Command

```bash
./bin/sim model bosz pfs \
    --config "train.json" "inst_pfs_mr.json" \
    --out "<output_dir>" \
    --sample-mode grid \
    --interp-mode grid \
    --chunk-size 2 \
    --mag <18|20|21.5|22|22.5|23> \
    --seeing 1.0 \
    --target_zenith_angle 22.5 \
    --target_field_angle 0.325 \
    --moon_phase 0.0 \
    --threads 4
```

## Usage for Fisher/CRLB

```python
import h5py
import numpy as np

# Load dataset
with h5py.File('.../grid_mag20_lowT/dataset.h5', 'r') as f:
    flux = f['dataset/arrays/flux/value'][:]   # (30182, 4096)
    error = f['dataset/arrays/error/value'][:] # (30182, 4096)
    wave = f['spectrumdataset/wave'][:]        # (4096,)

# SNR 计算
snr = flux / error  # per-pixel SNR

# Noise covariance for Fisher: Σ = diag(error²)
# Fisher: I(θ) = (∂μ/∂θ)ᵀ Σ⁻¹ (∂μ/∂θ)
# Use regular grid spacing for accurate ∂μ/∂θ via finite differences
```
