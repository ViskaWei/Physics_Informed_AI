# 📋 SpecViT Paper: Experiments Checklist

> **Status:** 📝 Active  
> **Date:** 2025-12-27  
> **Purpose:** Track experiments needed for paper + search prompts for existing results

---

## Overview

| Priority | Count | Status |
|----------|-------|--------|
| 🔴 P0 Must-Have | 5 | 1 running, 4 pending |
| 🟡 P1 Should-Have | 4 | 1 running, 3 pending |
| 🟢 P2 Nice-to-Have | 3 | 0 started |

---

## 🔴 P0: Must-Have (Paper Submission Blockers)

### P0.1 Finish 1M Run + Report Test Metrics

**Why:** Paper cannot use only validation metrics; need test R²/MAE (overall + per-SNR).

**Current Status:** 🚀 Running (ep112/200, val_r2=0.713)

**Need:**
- [ ] Wait for 200 epochs to complete
- [ ] Best checkpoint (by val_r2)
- [ ] Test R² / MAE (overall)
- [ ] Test R² / MAE per SNR bin
- [ ] Learning curves (train/val loss, val R²)

**Search Prompts (舱内检索):**
```
"VIT-20251226-vit-1m-large-01 test_r2"
"vit_1m log_g test 10000"
"mag205_225_lowT_1M best checkpoint"
"khgqjngm test"
```

**Paper Artifact:**
- Table: Main test metrics
- Figure: Learning curves

---

### P0.2 LightGBM Baseline on Same 1M Dataset

**Why:** "vs baseline 待比较" in report will be challenged by reviewers.

**Current Status:** ⏳ Pending

**Need:**
- [ ] Train LightGBM on 1M train (same as ViT)
- [ ] Evaluate on same val/test splits
- [ ] Report overall + per-SNR bin R²

**Search Prompts:**
```
"LightGBM 1M log_g mag205_225_lowT_1M"
"lgbm log_g noise_level=1.0 1M"
"gbdt log_g 7100-8850A 1M"
```

**Paper Artifact:**
- Table: ViT vs LightGBM (overall + per-SNR)
- Figure: Same R²–SNR plot with both models

---

### P0.3 Dataset Scaling Curve (N = 10k → 1M)

**Why:** Core evidence explaining "why ViT failed before" - **Transformer data requirements**.

**Current Status:** ⏳ Pending

**Need:**
- [ ] Fixed architecture: L6H256 (same as 1M run)
- [ ] Fixed training: same epochs/steps, early stop
- [ ] N = 10k, 50k, 100k, 200k, 500k, 1M
- [ ] Compare ViT vs LightGBM vs CNN

**Search Prompts:**
```
"vit scaling log_g 10k 50k 100k 200k"
"dataset size log_g vit L6 H256"
"num_samples=10000 vit log_g"
"vit small data log_g R2"
```

**Paper Artifact:**
- Figure: Performance vs dataset size (log scale x-axis)
- Key finding: ViT needs ~500k+ to match tree models

---

### P0.4 SNR Sweep Evaluation Aligned with Fisher/CRLB Ceiling

**Why:** Main figure requires consistent data sources for ceiling and model curves.

**Current Status:** ⏳ Pending (need test results from P0.1)

**Need:**
- [ ] Test split by SNR (or mag) bins
- [ ] Per-bin R²
- [ ] Fisher/CRLB: per-SNR R²_max (median + 10–90% band)
- [ ] Gap: R²_max − R²_model vs SNR

**Search Prompts:**
```
"Fisher CRLB logg R2_max vs SNR"
"CRLB_logg Schur complement"
"Jacobian spectra Fisher information"
"r2 ceiling logg snr"
"plot_r2_vs_snr ceiling"
```

**Paper Artifact:**
- Figure: R² vs SNR with ceiling band (MAIN FIGURE)
- Figure inset: Gap vs SNR

---

### P0.5 Tokenization Ablation on 1M or ≥200k

**Why:** Paper claims "physics-informed tokenization is key" - must prove with ablation.

**Current Status:** ⏳ Pending

**Ablations (minimum):**
- [ ] C1D vs SW
- [ ] Overlap/stride (SW overlap)
- [ ] Patch size (8/16/32/64)
- [ ] Chunk normalization on/off

**Search Prompts:**
```
"proj_fn C1D SW log_g"
"patch_size=16 32 64 log_g vit"
"stride overlap sliding window tokenization spectra"
"chunk normalization spectra vit"
```

**Paper Artifact:**
- Table: Ablation results
- Figure: Ablation bar plot (R²)

---

## 🟡 P1: Should-Have (Strengthen Paper)

### P1.1 Loss & Label Normalization Study

**Why:** Already have 2 runs (MSE+standard, L1+minmax). Natural "training robustness" story.

**Current Status:** 🔆 Running (Run1 vs Run2)

**Need:**
- [ ] Same data/model, cross comparison:
  - MSE+standard vs MSE+minmax
  - L1+standard vs L1+minmax
  - MSE vs L1 (all else equal)

**Search Prompts:**
```
"vit_1m_l1.yaml"
"loss L1 vs MSE log_g vit 1M"
"label_norm standard minmax log_g"
```

**Paper Artifact:**
- Figure: Learning curves comparison
- Table: Final test metrics

---

### P1.2 Positional Embedding Ablation

**Why:** If PIPE is a selling point, must show gains; otherwise downgrade to "exploration".

**Current Status:** ⏳ Pending

**Need:**
- [ ] Learned vs Sinusoidal vs PIPE vs RoPE
- [ ] Same architecture, same data

**Search Prompts:**
```
"PIPE positional embedding spectra"
"physics-informed positional embedding wavelength"
"RoPE spectra vit"
"position embedding ablation log_g"
```

**Paper Artifact:**
- Table: PE ablation
- (Bonus) Figure: Attention map vs spectral line positions

---

### P1.3 Multi-task vs Single-task

**Why:** Astronomical tasks often involve coupled parameters; multi-task is more realistic.

**Current Status:** ⏳ Pending

**Need:**
- [ ] Predict Teff/logg/[M/H] jointly
- [ ] Compare single-task vs multi-task per-parameter R²

**Search Prompts:**
```
"multihead regression Teff logg MH vit"
"5D labels vit spectra"
"joint training stellar parameters transformer"
```

**Paper Artifact:**
- Table: Single-task vs multi-task (per-parameter R²)
- Figure: Correlation of errors / residual covariance

---

### P1.4 Robustness: Cross-Noise Generalization

**Why:** Prove model generalizes across noise levels, more like real observations.

**Current Status:** ⏳ Pending

**Need:**
- [ ] Train noise=1.0, test on {0.5, 2.0}
- [ ] Train noise={0.5, 2.0}, test on noise=1.0

**Search Prompts:**
```
"noise_level=0.5 log_g vit"
"noise_level=2.0 log_g vit"
"cross-noise generalization spectra"
```

**Paper Artifact:**
- Figure: Robustness matrix (train noise × test noise)

---

## 🟢 P2: Nice-to-Have (Paper Enhancement)

### P2.1 Interpretability: Attention Visualization

**Why:** Show attention aligns with known spectral lines.

**Search Prompts:**
```
"attention map spectra vit line"
"integrated gradients spectra transformer"
"saliency log_g spectrum"
```

---

### P2.2 Pretrain on 1M then Finetune on Small-N

**Why:** Demonstrate sample efficiency gain from pretraining.

**Search Prompts:**
```
"pretrain 1M finetune 10k log_g"
"transfer learning spectra vit"
```

---

### P2.3 Synthetic → Real (Domain Shift)

**Why:** Ultimate goal is real data; even preliminary results valuable.

**Search Prompts:**
```
"LAMOST logg vit finetune"
"APOGEE spectra transformer logg"
```

---

## 📦 Paper-Ready Checklist (Minimum for Submission)

Before submitting, ensure:

| # | Item | Status |
|---|------|--------|
| 1 | 1M best checkpoint + test metrics (overall + SNR) | ⏳ |
| 2 | LightGBM 1M baseline (same data/split) | ⏳ |
| 3 | Scaling curve (N → performance) | ⏳ |
| 4 | R²–SNR + Fisher/CRLB ceiling + gap (MAIN FIGURE) | ⏳ |
| 5 | Tokenization ablation (C1D/SW/patch/overlap/norm) | ⏳ |
| 6 | Reproducibility info (config, seed, paths, hyperparams) | ✅ |

---

## 🎯 Story Line

> **"ViT doesn't fail because it's wrong for spectra—it fails because data scale wasn't there. When N is large enough, the model approaches the information-theoretic limit."**

This is the paper's core narrative. All experiments should support this story.

---

## 📝 Notes

- All search prompts are for the experiment log repository (`logg/`)
- If result not found, need to run the experiment
- Prioritize P0 experiments for initial submission
- P1/P2 can be added during revision

---

> **Last Updated:** 2025-12-27
