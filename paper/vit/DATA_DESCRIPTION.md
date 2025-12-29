# Data Description for VIT Paper

## Overview

This document describes the BOSZ simulated stellar spectra dataset used for training the Vision Transformer model for surface gravity (log g) estimation.

---

## 1. Dataset Summary

| Property | Value |
|----------|-------|
| **Dataset Name** | BOSZ 1M Simulated PFS Spectra |
| **Data Source** | BOSZ ATLAS9 Stellar Atmosphere Models |
| **Simulation Date** | December 8-21, 2025 |
| **Total Samples** | 1,000,000 |
| **Total Size** | ~93 GB |
| **Format** | HDF5 |
| **Generation Time** | ~304 hours (12.7 days) |

---

## 2. Data Location

### 2.1 Full Dataset Path
```
/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/
```

### 2.2 Splits Used for Training

| Split | Path | Samples |
|-------|------|---------|
| **Training** | `train_200k_0/dataset.h5` | 200,000 |
| **Validation** | `val_1k/dataset.h5` | 1,000 |
| **Test** | `test_10k/dataset.h5` | 10,000 |

### 2.3 Additional Shards (Available but not used)

```
train_200k_1/dataset.h5  (200,000 samples)
train_200k_2/dataset.h5  (200,000 samples)
train_200k_3/dataset.h5  (200,000 samples)
train_200k_4/dataset.h5  (200,000 samples)
```

---

## 3. Data Format

### 3.1 HDF5 Structure

```
dataset.h5
├── spectrumdataset/
│   └── wave                    # Wavelength array (4096,)
├── dataset/
│   └── arrays/
│       ├── flux/
│       │   └── value           # Flux array (N, 4096)
│       └── error/
│           └── value           # Error array (N, 4096)
└── (pandas DataFrame)          # Stellar parameters
    ├── T_eff                   # Effective temperature (K)
    ├── log_g                   # Surface gravity (dex)
    ├── M_H                     # Metallicity [Fe/H] (dex)
    ├── a_M                     # Alpha enhancement
    ├── C_M                     # Carbon abundance
    ├── mag                     # i-band magnitude
    ├── redshift                # Redshift (z=0)
    └── snr                     # Signal-to-noise ratio
```

### 3.2 Data Types

| Field | Type | Shape | Unit |
|-------|------|-------|------|
| wave | float32 | (4096,) | Å |
| flux | float32 | (N, 4096) | normalized |
| error | float32 | (N, 4096) | fractional |
| T_eff | float64 | (N,) | K |
| log_g | float64 | (N,) | dex |
| M_H | float64 | (N,) | dex |
| mag | float64 | (N,) | mag |

---

## 4. Stellar Parameter Ranges

### 4.1 Physical Parameters

| Parameter | Symbol | Min | Max | Unit | Distribution |
|-----------|--------|-----|-----|------|--------------|
| Effective Temperature | T_eff | 3,750 | 6,000 | K | Beta |
| Surface Gravity | log_g | 1.0 | 5.0 | dex | Beta |
| Metallicity | [Fe/H] | -1.0 | 0.0 | dex | Beta |
| Alpha Enhancement | [α/M] | Fixed | | dex | From grid |
| Carbon Abundance | [C/M] | Fixed | | dex | From grid |

### 4.2 Observational Parameters

| Parameter | Symbol | Min | Max | Unit | Distribution |
|-----------|--------|-----|-----|------|--------------|
| i-band Magnitude | mag | 20.5 | 22.5 | mag | Uniform |
| Redshift | z | 0 | 0 | - | Fixed |
| Extinction | E(B-V) | 0 | 0 | mag | Fixed |

### 4.3 Parameter Statistics (Training Set)

| Parameter | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| T_eff | ~4,900 | ~600 | 3,750 | 6,000 |
| log_g | ~3.0 | ~1.2 | 1.0 | 5.0 |
| [Fe/H] | ~-0.5 | ~0.3 | -1.0 | 0.0 |
| mag | ~21.5 | ~0.6 | 20.5 | 22.5 |

---

## 5. Spectral Properties

### 5.1 Wavelength Grid

| Property | Value |
|----------|-------|
| Wavelength Range | ~7100 - 8850 Å |
| Number of Pixels | 4,096 |
| Sampling | Non-uniform (PFS detector) |
| Spectral Arm | MR (Medium Resolution Red) |

### 5.2 Normalization

| Property | Value |
|----------|-------|
| Method | Median normalization |
| Normalization Range | 6500 - 9500 Å |
| Post-normalization | Flux clipped to min=0 |

### 5.3 Model Resolution

| Property | Value |
|----------|-------|
| Native BOSZ Resolution | R = 50,000 |
| PFS MR Resolution | R ≈ 5,000 |
| Observed Resolution | Seeing-dependent |

---

## 6. Instrumental Configuration

### 6.1 Subaru PFS Medium Resolution Arm

| Property | Value |
|----------|-------|
| Instrument | Subaru PFS |
| Arm | MR (Medium Resolution) |
| Detector Config | `${PFSSPEC_DATA}/subaru/pfs/arms/mr.json` |
| PSF Model | PCA-based (`mr.2/pca.h5`) |

### 6.2 Observing Conditions

| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| Seeing | 0.5 | 1.5 | arcsec |
| Target Zenith Angle | 0 | 45 | deg |
| Target Field Angle | 0 | 0.65 | deg |
| Moon Zenith Angle | 30 | 90 | deg |
| Moon-Target Angle | 60 | 180 | deg |
| Moon Phase | 0 | 0 | (new moon) |

### 6.3 Exposure Configuration

| Property | Value |
|----------|-------|
| Single Exposure | 900 s (15 min) |
| Number of Exposures | 12 |
| Total Exposure Time | 10,800 s (3 hours) |

---

## 7. Noise Properties

### 7.1 Noise Sources

1. **Photon Noise**: From target flux
2. **Sky Background**: PFS sky model
3. **Detector Noise**: Read noise, dark current
4. **Moon Contamination**: New moon conditions

### 7.2 Error Array

The `error` array contains the 1σ uncertainty per pixel, derived from the PFS noise model accounting for all noise sources.

### 7.3 Signal-to-Noise Ratio

| Magnitude | Approximate SNR |
|-----------|-----------------|
| 20.5 | ~50-100 |
| 21.5 | ~20-50 |
| 22.5 | ~10-20 |

---

## 8. Data Loading Example

### 8.1 Python Code

```python
import h5py
import pandas as pd
import torch

def load_spectral_data(h5_path, num_samples=None):
    """Load spectral data from HDF5 file."""
    
    with h5py.File(h5_path, 'r') as f:
        # Load wavelength grid
        wave = torch.tensor(f['spectrumdataset/wave'][()], dtype=torch.float32)
        
        # Load flux and error
        if num_samples is None:
            flux = torch.tensor(f['dataset/arrays/flux/value'][:], dtype=torch.float32)
            error = torch.tensor(f['dataset/arrays/error/value'][:], dtype=torch.float32)
        else:
            flux = torch.tensor(f['dataset/arrays/flux/value'][:num_samples], dtype=torch.float32)
            error = torch.tensor(f['dataset/arrays/error/value'][:num_samples], dtype=torch.float32)
    
    # Post-process flux
    flux = flux.clamp(min=0.0)
    
    # Load stellar parameters
    df = pd.read_hdf(h5_path)
    if num_samples is not None:
        df = df[:num_samples]
    
    log_g = torch.tensor(df['log_g'].values, dtype=torch.float32)
    
    return {
        'wave': wave,
        'flux': flux,
        'error': error,
        'log_g': log_g,
        'T_eff': df['T_eff'].values,
        'M_H': df['M_H'].values,
        'mag': df['mag'].values,
    }

# Example usage
data = load_spectral_data(
    '/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/train_200k_0/dataset.h5',
    num_samples=10000
)

print(f"Wavelength shape: {data['wave'].shape}")
print(f"Flux shape: {data['flux'].shape}")
print(f"log_g range: [{data['log_g'].min():.2f}, {data['log_g'].max():.2f}]")
```

### 8.2 Output

```
Wavelength shape: torch.Size([4096])
Flux shape: torch.Size([10000, 4096])
log_g range: [1.00, 5.00]
```

---

## 9. Data Generation

### 9.1 Generation Pipeline

The data was generated using the `pfsspec` simulation pipeline:

```bash
./bin/sim model bosz pfs \
    --threads 12 \
    --config train.json inst_pfs_mr.json \
    --out ${OUTPUT_DIR} \
    --sample-count 200000 \
    --seeing 0.5 1.5
```

### 9.2 Sampling Strategy

| Property | Value |
|----------|-------|
| Sampling Mode | Random |
| Parameter Distribution | Beta |
| Interpolation | Spline |
| Chunk Size | 1000 samples |

### 9.3 Generation Infrastructure

| Property | Value |
|----------|-------|
| Host | elephant1 |
| CPU | Intel Xeon E7-4830 (64 cores) |
| Memory | 1 TB |
| Parallel Shards | 5 |
| Threads per Shard | 12 |

---

## 10. Data Quality Notes

### 10.1 Known Issues

1. **Fixed Redshift**: All spectra at z=0 (rest frame)
2. **No Extinction**: E(B-V) = 0 for all samples
3. **New Moon Only**: Moon phase = 0

### 10.2 Limitations

1. **Temperature Range**: Limited to cool stars (3750-6000 K)
2. **Metallicity Range**: Limited to near-solar ([Fe/H] > -1)
3. **No Binary Stars**: Single star models only
4. **No Rotation**: Non-rotating stellar models

### 10.3 Validation

The dataset has been validated against:
- Template fitting procedures (see `template_fit_combined/`)
- Cross-validation with held-out test sets
- SNR consistency checks

---

## Appendix: Related Files

| File | Description |
|------|-------------|
| `README.md` | Original dataset documentation |
| `shard_index.txt` | List of all shard paths |
| `logs/` | Generation logs |
| `template_fit_combined/` | Template fitting results |

---

*Document generated: December 28, 2025*
