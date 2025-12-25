# BOSZ Regular Grid Dataset for Fisher/CRLB Analysis

**Dataset Path:** `/datascope/subaru/user/swei20/data/bosz50000/grid/grid_mag215_lowT`

**Purpose:** Error array generation on regular grid for Fisher Information Matrix calculation

## Physical Parameters (Grid Mode - No Interpolation)

| Parameter | Values | Step |
|-----------|--------|------|
| T_eff | 3500-7000 K | 250K (low T region) |
| log_g | 0.0-5.0 | 0.5 dex |
| [M/H] | -2.5 to +0.75 | 0.25 dex |
| [C/M] | Fixed at source grid | - |
| [α/M] | Fixed at source grid | - |

**Total Grid Points:** 30,182 spectra (all valid points in parameter range)

## Observational Parameters (Fixed)

| Parameter | Value |
|-----------|-------|
| Magnitude | 21.5 (sdss_g) |
| Seeing | 1.0 arcsec |
| Exp Count | 4 |
| Exp Time | 900s |
| Zenith Angle | 22.5° |
| Field Angle | 0.325° |
| Moon Phase | 0.0 (new moon) |

## Data Format

| Field | Shape | Description |
|-------|-------|-------------|
| wave | (4096,) | MR arm wavelength grid |
| flux | (30182, 4096) | Normalized flux (median ~0.5) |
| error | (30182, 4096) | Per-pixel noise estimate |
| cont | (30182, 4096) | Continuum |
| T_eff | (30182,) | Effective temperature |
| log_g | (30182,) | Surface gravity |
| M_H | (30182,) | Metallicity |

## Statistics

- **SNR Range:** ~3-15 (magnitude 21.5)
- **Flux Median:** ~0.5 (normalized)
- **Error/Flux Ratio:** ~0.1-0.3

## Generation Command

```bash
./bin/sim model bosz pfs \
    --config "train.json" "inst_pfs_mr.json" \
    --out "/datascope/subaru/user/swei20/data/bosz50000/grid/grid_mag215_lowT" \
    --sample-mode grid \
    --interp-mode grid \
    --chunk-size 2 \
    --mag 21.5 \
    --seeing 1.0 \
    --target_zenith_angle 22.5 \
    --target_field_angle 0.325 \
    --moon_phase 0.0 \
    --threads 12
```

## Key Configuration

- `--sample-mode grid`: Iterate over all valid grid points (no random sampling)
- `--interp-mode grid`: Use exact grid values (no interpolation)
- Fixed observational parameters for consistent noise model

## Usage for Fisher/CRLB

```python
import h5py

with h5py.File('.../grid_mag215_lowT/dataset.h5', 'r') as f:
    flux = f['flux'][:]      # (30182, 4096)
    error = f['error'][:]    # (30182, 4096) - for Σ = diag(error²)
    log_g = f['log_g'][:]    # (30182,)
    
# Fisher: I(θ) = (∂μ/∂θ)ᵀ Σ⁻¹ (∂μ/∂θ)
# Use regular grid spacing for accurate ∂μ/∂θ via finite differences
```
