# Test 10k Dataset SNR Statistics

**Data Source**: `/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/test_10k/dataset.h5`

**Generated**: 2025-12-28

---

## Basic Statistics

| Metric | Value |
|--------|-------|
| Count | 10,000 |
| Mean | 9.16 |
| Median | 8.21 |
| Std | 4.41 |
| Min | 2.79 |
| Max | 27.68 |

**Distribution**: Right-skewed (median < mean)

---

## Percentiles

| Percentile | SNR Value |
|------------|-----------|
| 0% (min) | 2.79 |
| 5% | 3.71 |
| 10% | 4.09 |
| 25% | 5.29 |
| 50% (median) | 8.21 |
| 75% | 12.53 |
| 90% | 15.81 |
| 95% | 17.22 |
| 100% (max) | 27.68 |

---

## Binning Schemes (8 bins)

### Option 1: Equal-width Bins

```python
bins = [2.8, 5.9, 9.0, 12.1, 15.2, 18.3, 21.5, 24.6, 27.7]
```

| Bin Range | Count |
|-----------|-------|
| 2.8 - 5.9 | 3125 |
| 5.9 - 9.0 | 2416 |
| 9.0 - 12.1 | 1761 |
| 12.1 - 15.2 | 1459 |
| 15.2 - 18.3 | 1011 |
| 18.3 - 21.5 | 217 |
| 21.5 - 24.6 | 8 |
| 24.6 - 27.7 | 3 |

**⚠️ Warning**: Last 3 bins have very few samples.

---

### Option 2: Quantile Bins (Equal Samples) ✅ RECOMMENDED

```python
bins = [2.8, 4.3, 5.3, 6.6, 8.2, 10.2, 12.5, 15.2, 27.7]
```

| Bin Range | Count | Center SNR |
|-----------|-------|------------|
| 2.8 - 4.3 | 1250 | ~3.5 |
| 4.3 - 5.3 | 1250 | ~4.8 |
| 5.3 - 6.6 | 1250 | ~6.0 |
| 6.6 - 8.2 | 1250 | ~7.4 |
| 8.2 - 10.2 | 1250 | ~9.2 |
| 10.2 - 12.5 | 1250 | ~11.4 |
| 12.5 - 15.2 | 1250 | ~13.9 |
| 15.2 - 27.7 | 1250 | ~21.4 |

**Advantage**: Each bin has sufficient samples (1250) for reliable bootstrap statistics.

---

### Option 3: Log-scale Bins

```python
bins = [2.8, 3.7, 5.0, 6.6, 8.8, 11.7, 15.6, 20.8, 27.7]
```

| Bin Range | Count |
|-----------|-------|
| 2.8 - 3.7 | 509 |
| 3.7 - 5.0 | 1571 |
| 5.0 - 6.6 | 1667 |
| 6.6 - 8.8 | 1653 |
| 8.8 - 11.7 | 1690 |
| 11.7 - 15.6 | 1825 |
| 15.6 - 20.8 | 1065 |
| 20.8 - 27.7 | 19 |

**⚠️ Warning**: First and last bins have fewer samples.

---

### Option 4: Round Number Bins

```python
bins = [2.5, 4, 5, 6, 7, 8, 10, 15, 30]
```

| Bin Range | Count | Center |
|-----------|-------|--------|
| 2.5 - 4.0 | 884 | 3.2 |
| 4.0 - 5.0 | 1246 | 4.5 |
| 5.0 - 6.0 | 1089 | 5.5 |
| 6.0 - 7.0 | 847 | 6.5 |
| 7.0 - 8.0 | 796 | 7.5 |
| 8.0 - 10.0 | 1277 | 9.0 |
| 10.0 - 15.0 | 2511 | 12.5 |
| 15.0 - 30.0 | 1350 | 22.5 |

**Advantage**: Easy to interpret and report.

---

## Recommendation

For R² vs SNR plots with error bars:

1. **Use Quantile bins** for statistical reliability (1250 samples/bin)
2. **Use Round number bins** for interpretability in publications
3. **Avoid equal-width bins** due to extreme sample imbalance at high SNR

---

## Python Code for Binning

```python
import numpy as np
import pandas as pd

# Load data
df = pd.read_hdf('dataset.h5', key='dataset/params')
snr = df['snr'].values

# Recommended: Quantile bins
bins_quantile = np.percentile(snr, np.linspace(0, 100, 9))
# [2.79, 4.30, 5.30, 6.60, 8.21, 10.22, 12.53, 15.20, 27.68]

# Alternative: Round number bins
bins_round = [3, 4, 5, 6, 7, 8, 10, 15, 28]

# Assign bin labels
df['snr_bin'] = pd.cut(snr, bins=bins_quantile, labels=range(8))
```
