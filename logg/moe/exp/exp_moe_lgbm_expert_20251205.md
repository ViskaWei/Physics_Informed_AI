# 📗 实验报告：Hard Bins 小 LightGBM Expert

---
> **实验名称：** MVP-15: Hard Bins 小 LightGBM Expert  
> **对应 MVP：** MVP-15  
> **日期：** 2025-12-05 (执行: 2025-12-09)  
> **状态：** ✅ 完成  
> **验证假设：** H-15

---

## ⚡ 核心结论速览

> **一句话总结**：🟡 **部分成功** - Bin3 大幅改善 (+0.046)，但 Bin6 意外退步 (-0.056)，需分治策略
>
> **关键数字**：
> - Full coverage R²: **0.9267** (vs Ridge-only 0.9298)
> - Bin3 R²: **0.829** (ΔR² = +0.046) ✅
> - Bin6 R²: **0.791** (ΔR² = -0.056) ❌

---

## 实验配置

- 训练集: 32,000 样本
- 测试集: 1,000 样本 (816 covered)
- 噪声: 0.2
- LGBM bins: [3, 6] (Mid/Hot Metal-poor)

## Overall Results

| Method | R² | MAE |
|--------|-----|-----|
| Ridge-only MoE | **0.9298** | 0.2163 |
| Hybrid MoE | 0.9267 | 0.2235 |
| Global LGBM | 0.9116 | - |

## Per-Bin Results

| Bin | Description | Expert | R²_Ridge | R²_Hybrid | ΔR² |
|-----|-------------|--------|----------|-----------|-----|
| 0 | Cool Metal-poor | Ridge | 0.938 | 0.928 | -0.010 |
| 1 | Cool Solar | Ridge | 0.967 | 0.967 | +0.000 |
| 2 | Cool Metal-rich | Ridge | 0.989 | 0.988 | -0.001 |
| 3 | Mid Metal-poor | **LGBM** | 0.783 | **0.829** | **+0.046** ✅ |
| 4 | Mid Solar | Ridge | 0.938 | 0.940 | +0.002 |
| 5 | Mid Metal-rich | Ridge | 0.980 | 0.979 | -0.001 |
| 6 | Hot Metal-poor | **LGBM** | 0.847 | 0.791 | **-0.056** ❌ |
| 7 | Hot Solar | Ridge | 0.958 | 0.950 | -0.007 |
| 8 | Hot Metal-rich | Ridge | 0.973 | 0.971 | -0.002 |

## 图表

- Per-bin R² comparison: `img/moe_lgbm_expert_per_bin_r2.png`
- Full coverage comparison: `img/moe_lgbm_expert_full_coverage.png`
- Bin scatter plots: `img/moe_lgbm_expert_bin_scatter.png`
- Feature importance: `img/moe_lgbm_expert_feature_importance.png`

## 关键洞见

1. **LGBM 不是万能解**：对 Bin3 有效 (+0.046)，对 Bin6 无效 (-0.056)
2. **Bin6 退步原因**：
   - Hot Metal-poor 谱线稀疏，全谱特征对 LGBM 不友好
   - OOF R²=0.835 vs Test R²=0.791，过拟合迹象
3. **分治策略**：Bin3 用 LGBM，Bin6 保持 Ridge

## 下一步

- [x] 确认 Bin3 用 LGBM 有效
- [ ] Bin6 保持 Ridge 或探索专用特征
- [ ] 考虑只对 Bin3 做 LGBM 替换的版本

---

*实验 ID: VIT-20251205-moe-lgbm-expert-01*
*脚本: ~/VIT/scripts/moe_lgbm_expert.py*
*结果: ~/VIT/results/moe/lgbm_expert/*
