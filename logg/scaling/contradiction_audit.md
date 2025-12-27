# 🔍 Contradiction Audit: logg/scaling Topic
> **ID:** AUDIT-20251227-scaling-01 | **Date:** 2025-12-27 | **Status:** ✅ Complete

---

## Executive Summary

| Category | Count | Severity | Action Required |
|----------|-------|----------|-----------------|
| **Critical Contradictions** | 2 | 🔴 | Immediate resolution needed |
| **Scope Mismatches** | 3 | 🟡 | Split into conditional conclusions |
| **Missing Information (BLOCKERs)** | 5 | 🟠 | Require minimal supplementation |
| **Protocol Inconsistencies** | 2 | 🟡 | Standardize going forward |

---

# 1. Evidence Index

## 1.1 Experiment Registry

| Exp ID | Name | Date | Status | Dataset | Train N | Test N | Noise | Metric | Key Result | Evidence File |
|--------|------|------|--------|---------|---------|--------|-------|--------|------------|---------------|
| `SCALING-20251222-ml-ceiling-01` | ML Ceiling | 2025-12-22 | ✅ | mag205_225_lowT_1M | 1M | 500 | σ=1.0 | R² | Ridge=0.46, LGB=0.57 | exp_scaling_ml_ceiling_20251222.md |
| `SCALING-20251222-ridge-alpha-01` | Ridge α Sweep | 2025-12-22 | ✅ | mag205_225_lowT_1M | 100k/1M | 500 | σ=1.0 | R² | Best α=1e5, R²=0.5017 | exp_scaling_ridge_alpha_extended_20251222.md |
| `SCALING-20251222-whitening-01` | Whitening/SNR | 2025-12-22 | ✅ | mag205_225_lowT_1M | 1M/100k | 500 | σ=1.0 | R² | SNR Δ=+0.015 | exp_scaling_whitening_snr_20251222.md |
| **BLOCKER** | Ridge 1k Test | 2025-12-24 | ✅ | mag205_225_lowT_1M | 1M | 1k | σ=1.0 | R² | R²=0.4551 | exp_scaling_ridge_1ktest_20251224.md |
| `SCALING-20251223-oracle-moe-noise1-01` | Oracle MoE | 2025-12-23 | ✅ | mag205_225_lowT_1M | 1M | 1k | σ=1.0 | R² | Ridge=0.4611, MoE=0.6249 | exp_scaling_oracle_moe_noise1_20251223.md |
| `SCALING-20251224-fisher-ceiling-02` | Fisher V2 | 2025-12-24 | ✅ | grid_mag215_lowT | 1,260 | - | σ=1.0 | R²_max | 0.8914 | exp_scaling_fisher_ceiling_v2_20251224.md |
| `SCALING-20251224-nn-baseline-framework-01` | NN Baseline | 2025-12-24 | ✅ | mag205_225_lowT_1M | 100k/1M | 1k | σ=1.0 | R² | MLP=0.467, CNN=0.412 | exp_scaling_nn_baseline_framework_20251224.md |
| `SCALING-20251223-soft-moe-noise1-01` | Soft MoE | 2025-12-25 | ✅ | mag205_225_lowT_1M | 1M | 1k | σ=1.0 | R² | R²=0.5930, ρ=0.805 | exp_scaling_soft_moe_noise1_20251225.md |
| `SCALING-20251223-fisher-ceiling-01` | Fisher V1 | 2025-12-23 | ❌ FAILED | BOSZ continuous | 5k | - | σ=1.0 | R²_max | Invalid | exp_scaling_fisher_ceiling_20251223.md |

## 1.2 BLOCKER: Missing Critical Information

| Exp ID | Missing Field | Impact | Minimal Fix |
|--------|---------------|--------|-------------|
| `oracle-moe-noise1-01` | **experiment_id in header** | Cannot trace officially | Add `SCALING-20251223-oracle-moe-noise1-01` to header |
| `ridge_1ktest_20251224` | **experiment_id** | Missing ID entirely | Assign `SCALING-20251224-ridge-1ktest-01` |
| ALL | **Random seed** | Cannot reproduce exact numbers | Document seed (default numpy seed or explicit) |
| `ml-ceiling-01` | **Test protocol** | First 500 vs last 500 unclear | Document: "first 500 of test_1k_0" |
| `nn-baseline-01` | **Pre-noised vs on-fly** | Inconsistent comparison | All should use pre-stored noisy from test_1k_0 |

---

# 2. Claim Graph

## 2.1 Ridge R² Claims

| ClaimKey | ClaimValue | Evidence | Confidence | Issue |
|----------|------------|----------|------------|-------|
| (Ridge, R², global, noise=1, 1M, **test=500**) | **0.50** | ml-ceiling-01 | Suspect | Old test protocol |
| (Ridge, R², global, noise=1, 1M, **test=500**) | **0.46** | ml-ceiling-01 §5.5 | Suspect | Same exp, different number |
| (Ridge, R², global, noise=1, 1M, **test=1k**) | **0.4551** | ridge_1ktest_20251224 | Verified | Standard protocol |
| (Ridge, R², global, noise=1, 1M, **test=1k**, pre-noised) | **0.4611** | oracle-moe-noise1-01 | Verified | Uses α=100k |
| (Ridge, R², global, noise=1, 1M, test=500) | **0.5017** | ridge-alpha-01 | Plausible | α=1e5, test=500 |

### ⚠️ CONTRADICTION C1: Ridge R² values inconsistent (0.46 vs 0.50 vs 0.5017)

## 2.2 LightGBM R² Claims

| ClaimKey | ClaimValue | Evidence | Confidence | Issue |
|----------|------------|----------|------------|-------|
| (LightGBM, R², global, noise=1, **1M**, test=500) | **0.5709** | ml-ceiling-01 | Verified | Canonical |
| (LightGBM, R², global, noise=1, **100k**, test=500) | **0.5533** | whitening-01 | Verified | Different train size |

### ✅ NO CONTRADICTION: Different train sizes clearly documented

## 2.3 Oracle MoE R² Claims

| ClaimKey | ClaimValue | Evidence | Confidence | Issue |
|----------|------------|----------|------------|-------|
| (Oracle MoE, R², global, noise=1, 1M, test=1k) | **0.6249** | oracle-moe-noise1-01 | Verified | ✅ |
| (Oracle MoE, ΔR², vs Ridge) | **+0.1637** | oracle-moe-noise1-01 | Verified | ✅ |

### ✅ NO CONTRADICTION

## 2.4 Soft MoE R² Claims

| ClaimKey | ClaimValue | Evidence | Confidence | Issue |
|----------|------------|----------|------------|-------|
| (Soft MoE, R², global, noise=1, 1M, test=1k) | **0.5930** | soft-moe-noise1-01 | Verified | ✅ |
| (Soft MoE, ρ, retention) | **0.8052** | soft-moe-noise1-01 | Verified | ✅ |

### ✅ NO CONTRADICTION

## 2.5 Fisher Ceiling Claims

| ClaimKey | ClaimValue | Evidence | Confidence | Issue |
|----------|------------|----------|------------|-------|
| (Fisher, R²_max, mag=21.5, 3D) | **0.8914** | fisher-v2 | Verified | ✅ |
| (Fisher, R²_max, mag=21.5, 5D) | **0.8742** | fisher-v3a | Verified | ✅ |
| (Fisher, R²_max, mag=21.5, V1) | **0.9661** | fisher-v1 | **Invalid** | Method failed |

### ⚠️ STATUS CONTRADICTION C2: Hub §Canonical references "0.89" but V1 report still exists with 0.97

## 2.6 NN Baseline Claims

| ClaimKey | ClaimValue | Evidence | Confidence | Issue |
|----------|------------|----------|------------|-------|
| (MLP, R², 100k, flux_only) | **0.4671** | nn-baseline-01 | Verified | ✅ |
| (CNN, R², 100k, flux_only) | **0.4122** | nn-baseline-01 | Verified | ✅ |
| (MLP, R², 1M, whitening) | **-0.0003** | nn-baseline-01 | **Invalid** | Whitening failed |
| (CNN, R², 1M, whitening) | **0.4337** | nn-baseline-01 | Suspect | Whitening mode |

### ⚠️ SCOPE MISMATCH S1: 100k vs 1M comparisons mixed in conclusions

---

# 3. Contradiction Cards

## Card C1: Ridge R² Values Contradictory (0.46 vs 0.50 vs 0.5017)

| Field | Content |
|-------|---------|
| **What Conflicts** | A: Ridge R²=0.50 (ml-ceiling §5.5) vs B: Ridge R²=0.46 (hub canonical) vs C: R²=0.5017 (alpha-sweep) vs D: R²=0.4551 (1k test) vs E: R²=0.4611 (oracle-moe) |
| **Why Likely** | **Protocol mismatch**: test=500 vs test=1k + α differences (5000 vs 100000) |
| **Root Cause** | 1) Original exp used test=500, later switched to test=1k; 2) Different α values used; 3) Pre-noised vs on-fly noise inconsistency |
| **Fix Strategy** | **Standardize to test=1k, α=100000, pre-noised** → Canonical Ridge R² = 0.46 |
| **Owner Action** | Update hub canonical to specify: "test=1k, α=100k, pre-noised → R²=0.46" |

**Resolution Rule Applied**: Protocol Mismatch → Standardize

---

## Card C2: Hub References Failed Fisher V1 Numbers

| Field | Content |
|-------|---------|
| **What Conflicts** | Hub §Canonical shows R²_max=0.89 (V2, correct) but changelog still references V1 "R²_max=0.97" in some places |
| **Why Likely** | **Status contradiction**: V1 was marked failed but historical references not cleaned |
| **Fix Strategy** | **Isolate V1 as Invalid** → Move all V1 references to "Rejected/Invalidated" section |
| **Owner Action** | Add V1 to hub §Rejected with "Method failed: non-grid data caused CRLB to span 20 orders of magnitude" |

**Resolution Rule Applied**: Invalid → Isolate

---

## Card S1: NN Baseline Compares 100k and 1M Apples-to-Oranges

| Field | Content |
|-------|---------|
| **What Conflicts** | Report compares MLP@100k (0.467) vs Oracle MoE@1M (0.62) with different train sizes |
| **Why Likely** | **Scope mismatch**: Valid comparison requires same train size |
| **Fix Strategy** | **Split conclusion**: "MLP@100k ≈ Ridge@100k" is valid; "MLP vs Oracle MoE" needs @1M data |
| **Owner Action** | Report should state: "MLP 1M flux_only not yet run; comparison to Oracle MoE pending" |

**Resolution Rule Applied**: Scope Mismatch → Split

---

## Card S2: Whitening Experiment Uses 100k for LightGBM but 1M for Ridge

| Field | Content |
|-------|---------|
| **What Conflicts** | whitening-01 uses Ridge@1M (0.5077) vs LightGBM@100k (0.5533), different train sizes |
| **Why Likely** | **Scope mismatch**: "LightGBM 1M too slow" — documented but affects cross-model comparison |
| **Fix Strategy** | Acknowledge in conclusions: "LightGBM number is @100k, not directly comparable to Ridge@1M" |
| **Owner Action** | Add footnote to hub when citing LightGBM whitening result |

**Resolution Rule Applied**: Scope Mismatch → Split

---

## Card S3: Hub Claims "LightGBM=0.57" but Multiple Values Exist

| Field | Content |
|-------|---------|
| **What Conflicts** | Hub says 0.57, exp shows 0.5709@1M and 0.5533@100k |
| **Why Likely** | **Rounding + Scope**: 0.5709 rounded to 0.57 is acceptable; 0.5533 is different scope |
| **Fix Strategy** | Canonical should be explicit: "LightGBM R²=0.57 (train=1M, test=500, raw input)" |
| **Owner Action** | Add scope card to hub canonical |

**Resolution Rule Applied**: Scope Mismatch → Annotate

---

# 4. Resolution Summary

## 4.1 Resolved via Isolation

| Claim | Action | New Location |
|-------|--------|--------------|
| Fisher V1 R²_max=0.97 | Move to §Rejected | Hub §Rejected/Invalidated |
| MLP 1M whitening R²=-0.0003 | Move to §Rejected | Hub §Rejected/Invalidated |

## 4.2 Resolved via Split (Conditional Conclusions)

| Original Claim | Split Into | Condition |
|----------------|------------|-----------|
| "Ridge R²=0.46" | Ridge R²=0.46 | test=1k, α=100k, pre-noised |
| "Ridge R²=0.50" | Historical (deprecated) | test=500, α=5000 |
| "LightGBM better than Ridge" | LightGBM@1M > Ridge@1M | Same train size |
| "MLP matches Ridge" | MLP@100k ≈ Ridge@100k | Same train size |

## 4.3 Resolved via Standardization

| Issue | Standard Protocol Adopted |
|-------|---------------------------|
| Test set size | **1k** (full test_1k_0) |
| Test data | **Pre-stored noisy** (not on-fly) |
| Ridge α | **100000** (for 1M train) |
| LightGBM input | **raw** (never standardized) |
| NN input | **flux_only** (never whitening) |

## 4.4 Requires Minimal Experiment (1 MVP)

| Gap | Minimal Experiment | Judgment Criterion |
|-----|-------------------|-------------------|
| MLP@1M flux_only not run | Run MLP 3L_1024 on 1M with flux_only | If R² > 0.50 → MLP benefits from scale; If ≈ 0.47 → MLP saturated |

---

# 5. Patch Summary

## 5.1 Files Modified

| File | Change Type | Specific Changes |
|------|-------------|------------------|
| `scaling_hub_20251222.md` | **Rewrite** | See §6 below |
| `exp_scaling_oracle_moe_noise1_20251223.md` | **Patch Header** | Add experiment_id |
| `exp_scaling_ridge_1ktest_20251224.md` | **Patch Header** | Add experiment_id |
| `exp_scaling_ml_ceiling_20251222.md` | **Annotate** | Add deprecation note for 500 test |

## 5.2 Patches Applied

### Patch 1: oracle_moe header fix
```diff
- > **Name:** TODO | **ID:** `TODO`  
+ > **Name:** Oracle MoE @ noise=1 | **ID:** `SCALING-20251223-oracle-moe-noise1-01`  
```

### Patch 2: ridge_1ktest header fix
```diff
- > **Name:** TODO | **ID:** `TODO`  
+ > **Name:** Ridge 1k Test Validation | **ID:** `SCALING-20251224-ridge-1ktest-01`  
```

### Patch 3: ml_ceiling deprecation note
```diff
+ > ⚠️ **DEPRECATED PROTOCOL**: This experiment used test=500. Canonical protocol is now test=1k.
+ > The authoritative Ridge R² for 1M/noise=1 is **0.46** (from oracle-moe-noise1-01 with test=1k).
```

---

# 6. Hub Revision (Paste-Ready)

See separate file: `scaling_hub_20251222_v3_audited.md`

---

# 7. Report Template v2

See separate file: `report_template_v2.md`

---

# 8. Minimal Disambiguation Experiments

> 当存在未消除矛盾时，提出最小 MVP 实验来裁决

## 8.1 MVP-DISAMB-1: MLP 1M flux_only

| Field | Content |
|-------|---------|
| **Purpose** | 裁决 "MLP 是否受益于 1M 数据规模" |
| **Background** | 当前 MLP@100k=0.467, MLP@1M(whitening)=-0.0003(失败), 无法判断 MLP scaling |
| **Experiment** | MLP 3L_1024, 1M train, 1k test, **flux_only** input (非 whitening) |
| **Config** | epochs=10, batch=2048, lr=5e-4, AdamW, CosineAnnealing |
| **Runtime** | ~15 min (based on 100k × 10 = 34 min, 1M should be ~5x) |
| **Script** | `~/VIT/scripts/run_scaling_nn_baselines.py -e MLP_1M --input flux_only` |

### Judgment Criteria

| If Result | Conclusion | Action |
|-----------|------------|--------|
| **R² > 0.50** | MLP benefits from 1M scale | Update hub: "MLP scales with data, Route A viable" |
| **R² ≈ 0.47 (± 0.02)** | MLP saturated at 100k | Update hub: "MLP does not benefit from scale, Route B preferred" |
| **R² < 0.45** | Implementation issue | Debug training loop |

### Expected Output

```markdown
## MVP-DISAMB-1 Result

| Metric | Value | vs 100k | vs Oracle MoE |
|--------|-------|---------|---------------|
| MLP 1M R² | [VALUE] | [DELTA] | [GAP] |

**Decision**: [If > 0.50: Route A viable] / [If ≈ 0.47: Route B preferred]
```

---

## 8.2 No Other Disambiguation Needed

All other contradictions have been resolved via:
- **Isolation** (Fisher V1, Whitening failures)
- **Scope Split** (test=500 vs test=1k)
- **Protocol Standardization** (pre-noised, α=100k, raw/flux_only input)

---

*Generated: 2025-12-27 by Contradiction Audit Agent*
