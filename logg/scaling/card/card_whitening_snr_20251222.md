# 📇 Knowledge Card: Whitening/SNR Input Strategy
> **Name:** Whitening/SNR Input | **ID:** `VIT-20251222-scaling-whitening-card`  
> **Topic:** `scaling` | **Source:** `exp_scaling_whitening_snr_20251222.md` | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-22
```
💡 SNR 化对 Ridge 仅 +1.5%；⚠️ StandardScaler 严重损害 LightGBM 性能 (-36%)  
适用：输入预处理策略选择
```

---

## 🎯 问题与设置

**问题**: SNR/Whitening 输入 vs StandardScaler vs raw 哪个更好？

**设置**: 
- 数据: BOSZ 1M (Ridge) / 100k (LightGBM), noise σ=1.0
- 模型: Ridge α=1e5, LightGBM
- 关键变量: 6 种输入变体 (raw, standardized, centered_only, std_only, snr, snr_centered)

---

## 📊 关键结果

| # | 结果 | 数值 | 配置 |
|---|------|------|------|
| 1 | Best Ridge (snr_centered) | 0.5222 | +1.5% vs std |
| 2 | Ridge raw vs std | 0.0000 | 无差异 |
| 3 | LightGBM raw | **0.5533** | 最佳！ |
| 4 | LightGBM standardized | 0.1966 | ❌ -36%! |
| 5 | LightGBM snr | 0.0074 | ❌ 几乎失效 |

---

## 💡 核心洞见

### 🏗️ 宏观层（架构设计）

- **Ridge 对 scaling 不敏感**: 线性模型，standardization 只改变权重尺度
- **⚠️ LightGBM 必须用 raw**: Standardization 严重损害树模型

### 🔧 模型层（调参优化）

- **SNR 化边际效果**: 对 Ridge +1.5%，但未达 0.02 阈值
- **SNR 不是银弹**: 模型可能已从数据中隐式学到类似信息

### ⚙️ 工程层（实现细节）

- Ridge: 可继续用 StandardScaler（无害且便于比较）
- LightGBM: 必须用 raw 输入！
- Whitening (flux/error) 对树模型有害

---

## ➡️ 下一步

| 优先级 | 任务 | 相关 experiment_id |
|--------|------|-------------------|
| ✅ Done | 修复 LightGBM baseline 使用 raw 输入 | - |
| - | SNR 策略不推荐继续探索 | - |

---

## 🔗 相关链接

| 类型 | 路径 |
|------|------|
| 训练仓库 | `~/VIT/` |
| 脚本 | `~/VIT/scripts/scaling_whitening_experiment.py` |
| 完整报告 | `logg/scaling/exp/exp_scaling_whitening_snr_20251222.md` |

