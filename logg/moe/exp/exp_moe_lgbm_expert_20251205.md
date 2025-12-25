# 📘 📗 实验报告：Hard Bins 小 LightGBM Expert
> **Name:** TODO | **ID:** `VIT-20251205-moe-01`  
> **Topic:** `moe` | **MVP:** MVP-15 | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-05 | **Status:** 🔄
```
💡 实验目的  
决定：影响的决策
```

---


## 🔗 Upstream Links
| Type | Link |
|------|------|
| 🧠 Hub | `logg/moe/moe_hub.md` |
| 🗺️ Roadmap | `logg/moe/moe_roadmap.md` |

---

## ⚡ 核心结论速览

> **一句话总结**：🟢 **成功** - 强正则化 LGBM 在 Bin3 大幅改善 (+0.056)，Full coverage R² 首次超越 Ridge-only
>
> **关键数字**：
> - Full coverage R²: **0.9314** (vs Ridge-only 0.9298) ✅
> - Bin3 R²: **0.840** (ΔR² = +0.056) ✅
> - Bin6 R²: **0.815** (ΔR² = -0.032) ❌

---

## 实验配置

- 训练集: 32,000 样本
- 测试集: 1,000 样本 (816 covered)
- 噪声: 0.2
- LGBM bins: [3, 6] (Mid/Hot Metal-poor)

### LGBM 关键配置 (强正则化)

```python
LGBM_PARAMS = {
    'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05,
    'num_leaves': 20, 'feature_fraction': 0.3, 'bagging_fraction': 0.7,
    'min_child_samples': 50, 'reg_alpha': 1.0, 'reg_lambda': 1.0,
}
```

## Overall Results

| Method | R² | MAE |
|--------|-----|-----|
| Ridge-only MoE | 0.9298 | 0.2163 |
| **Hybrid MoE** | **0.9314** | 0.2177 |
| Global LGBM | 0.9216 | - |

## Per-Bin Results

| Bin | Description | Expert | R²_Ridge | R²_Hybrid | ΔR² |
|-----|-------------|--------|----------|-----------|-----|
| 0 | Cool Metal-poor | Ridge | 0.938 | 0.929 | -0.010 |
| 1 | Cool Solar | Ridge | 0.967 | 0.968 | +0.000 |
| 2 | Cool Metal-rich | Ridge | 0.989 | 0.989 | -0.000 |
| 3 | Mid Metal-poor | **LGBM** | 0.783 | **0.840** | **+0.056** ✅ |
| 4 | Mid Solar | Ridge | 0.938 | 0.942 | +0.004 |
| 5 | Mid Metal-rich | Ridge | 0.980 | 0.979 | -0.001 |
| 6 | Hot Metal-poor | **LGBM** | 0.847 | 0.815 | **-0.032** ❌ |
| 7 | Hot Solar | Ridge | 0.958 | 0.954 | -0.004 |
| 8 | Hot Metal-rich | Ridge | 0.973 | 0.971 | -0.002 |

## 配置对比实验

| 配置 | Hybrid R² | Bin3 ΔR² | Bin6 ΔR² | 结论 |
|------|-----------|----------|----------|------|
| 原始 (n=150, 弱正则) | 0.9267 | +0.046 | -0.056 | 过拟合 |
| 全谱 n=1000 | 0.9237 | +0.031 | -0.062 | 更严重过拟合 |
| 13D gate features | 0.8986 | -0.084 | -0.159 | 信息不足 |
| **强正则化** | **0.9314** | **+0.056** | -0.032 | ✅ 最佳 |

## 图表

- Per-bin R² comparison: `img/moe_lgbm_expert_per_bin_r2.png`
- Full coverage comparison: `img/moe_lgbm_expert_full_coverage.png`
- Bin scatter plots: `img/moe_lgbm_expert_bin_scatter.png`
- Feature importance: `img/moe_lgbm_expert_feature_importance.png`

## 关键洞见

1. **强正则化是关键**：减少 max_depth (8→5), num_leaves (63→20), 增加 reg_alpha/lambda (0.1→1.0)
2. **Bin3 vs Bin6 差异**：
   - Bin3 (Mid Metal-poor): LGBM 能捕捉非线性，改善显著
   - Bin6 (Hot Metal-poor): 高温谱线稀疏，即使 LGBM 也难以改善
3. **Gate features (13D) 信息不足**：只适合分类，不适合回归

## 下一步建议

- [x] 确认 Bin3 用 LGBM 有效 (+0.056)
- [ ] **Bin6 保持 Ridge** - LGBM 反而更差
- [ ] 考虑只对 Bin3 做 LGBM 替换的版本

---

*实验 ID: VIT-20251205-moe-lgbm-expert-01*
*脚本: ~/VIT/scripts/moe_lgbm_expert.py*
*结果: ~/VIT/results/moe/lgbm_expert/*
