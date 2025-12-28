# Card: Ridge 1M Optimal Model & All Methods Comparison

> **Experiment ID**: VIT-20251227-ridge-1m-optimal  
> **Date**: 2024-12-27  
> **Status**: ✅ Completed

---

## 📊 Ridge 1M Model Summary

| 属性 | 值 |
|------|-----|
| **Model Name** | `ridge_a100000_n1M_nz1p0` |
| **Model Path** | `/home/swei20/VIT/models/ridge_1M/ridge_a100000_n1M_nz1p0.pkl` |
| **Alpha** | 100000 |
| **Train Size** | 1,000,000 |
| **Test Size** | 10,000 |
| **Train R²** | 0.4944 |
| **Test R²** | **0.4957** |
| **Test MAE** | 0.6619 |

---

## 📈 All Methods Comparison (sorted by R²)

| Method | Overall R² | Test Data |
|--------|-----------|-----------|
| Fisher/CRLB 5D | ~0.99 | Theoretical |
| ViT | 0.698 | 10k |
| LightGBM | 0.614 | 10k |
| **Ridge 1M** | **0.496** | **10k** |
| MLP | 0.457 | 1k |
| TemplateFit | 0.45 | 1k |
| CNN | 0.429 | 1k |

---

## 📈 Ridge R² vs SNR (8 bins, 1250 samples/bin)

| SNR | R² | 10% | 90% | N |
|-----|-----|-----|-----|---|
| 3.81 | 0.195 | 0.153 | 0.228 | 1250 |
| 4.78 | 0.386 | 0.358 | 0.416 | 1250 |
| 5.90 | 0.466 | 0.439 | 0.493 | 1250 |
| 7.37 | 0.518 | 0.494 | 0.537 | 1250 |
| 9.13 | 0.556 | 0.537 | 0.575 | 1250 |
| 11.31 | 0.592 | 0.577 | 0.610 | 1250 |
| 13.85 | 0.611 | 0.596 | 0.625 | 1250 |
| 16.82 | 0.647 | 0.633 | 0.660 | 1250 |

---

## 📁 Files

| 文件 | 路径 |
|------|------|
| **Model** | `models/ridge_1M/ridge_a100000_n1M_nz1p0.pkl` |
| **Figure** | `logg/scaling/exp/img/r2_vs_snr_all_methods.png` |
| **Figure (alt)** | `logg/scaling/exp/img/r2_vs_snr_cnn_mlp_ridge_1m.png` |
| **Script** | `scripts/plot_r2_vs_snr_all_methods.py` |

---

## 📊 Figure: R² vs SNR - All Methods

![r2_vs_snr_all_methods](img/r2_vs_snr_all_methods.png)

**内容**:
- Fisher/CRLB 5D ceiling (with mag annotations)
- ViT, LightGBM (from ceiling script)
- Ridge 1M (new, 10k test)
- MLP, CNN (from predictions.csv)
- Template Fitting

---

## 🔬 Key Findings

1. **Fisher ceiling >> ML methods**: 理论上限远高于所有 ML 方法
2. **ViT > LightGBM > Ridge**: ViT 在所有 SNR 都最优
3. **Ridge 1M vs 32k**: α 从 200 变为 100000，R² 从 0.458 升至 0.496
4. **方差改进**: 10k 测试的 10-90% 范围 ~0.07 (vs 1k 的 ~0.27)

---

## 🏷️ Tags

`#ridge` `#1M` `#all-methods` `#fisher-ceiling` `#r2-vs-snr` `#scaling`
