# 🗺️ ViT Roadmap: Scaling to Fisher Ceiling
> **Name:** ViT Scaling Roadmap | **ID:** `VIT-20251227-vit-roadmap`  
> **Topic:** `vit` | **Phase:** 1 (Scaling Validation) | **Project:** `VIT`  
> **Author:** Viska Wei | **Date:** 2025-12-27 | **Status:** 🔄

```
💡 当前阶段目标  
Gate：完成 1M 200ep 训练 + LightGBM baseline 对比 + Scaling curve → 论文主结果就绪
```

---

## 🔗 Related Files

| Type | File |
|------|------|
| 🧠 Hub | [`vit_hub_20251227.md`](./vit_hub_20251227.md) |
| 📋 Kanban | `status/kanban.md` |
| 📗 Experiments | `exp_vit_*.md` |
| 🧠 Fisher Hub | [`../scaling/fisher_hub_20251225.md`](../scaling/fisher_hub_20251225.md) |
| 📄 Paper | [`../../paper/vit/specvit_paper.md`](../../paper/vit/specvit_paper.md) |

---

# 1. 🚦 Decision Gates

> Roadmap 定义怎么验证，Hub 做战略分析

## 1.1 战略路线 (来自Hub)

| Route | 名称 | Hub推荐 | 验证Gate |
|-------|------|---------|----------|
| A | Tokenization 优化 | 🟡 | Gate-4 |
| **Scale** | **1M 训练 + baseline** | 🟢 **推荐** | Gate-1,2,3 |
| C | 结构化 (MoE) | 🔴 | - |

## 1.2 Gate定义

### Gate-1: 1M 200ep 完成 + Test 指标

| 项 | 内容 |
|----|------|
| 验证 | ViT 在 1M 数据上的最终性能 |
| MVP | MVP-1.0 (1M scaling run) |
| 若 R² > 0.70 | ✅ 论文主结果就绪 |
| 若 R² < 0.65 | 🔴 需要调整架构/训练策略 |
| 状态 | 🚀 运行中 (ep112/200) |

### Gate-2: LightGBM 1M Baseline 对比

| 项 | 内容 |
|----|------|
| 验证 | ViT vs LightGBM 同口径对比 |
| MVP | MVP-2.0 (LightGBM 1M) |
| 若 ViT > LGBM | ✅ Transformer 优势成立 |
| 若 ViT < LGBM | 🔴 需要分析 gap 来源 |
| 状态 | ⏳ 待启动 |

### Gate-3: Scaling Curve (N → Performance)

| 项 | 内容 |
|----|------|
| 验证 | 数据规模如何影响 ViT 性能 |
| MVP | MVP-3.0 (N-sweep) |
| 预期 | 证明 Transformer 需要大数据 |
| 关键图 | performance vs N (log scale) |
| 状态 | ⏳ 待启动 |

### Gate-4: Tokenization Ablation

| 项 | 内容 |
|----|------|
| 验证 | C1D/SW, patch_size, overlap, norm 的影响 |
| MVP | MVP-4.0 (ablation runs) |
| 预期 | 确认设计选择合理性 |
| 状态 | ⏳ 待启动 |

## 1.3 本周重点

| 优先级 | MVP | Gate | 状态 | 预计完成 |
|--------|-----|------|------|---------|
| 🔴 P0.1 | MVP-1.0 (finish 200ep) | Gate-1 | 🚀 | 2025-12-28 |
| 🔴 P0.2 | MVP-2.0 (LightGBM 1M) | Gate-2 | ⏳ | 2025-12-28 |
| 🔴 P0.3 | MVP-3.0 (Scaling curve) | Gate-3 | ⏳ | 2025-12-29 |
| 🔴 P0.4 | SNR sweep eval | - | ⏳ | 2025-12-28 |
| 🔴 P0.5 | MVP-4.0 (Tokenization ablation) | Gate-4 | ⏳ | 2025-12-30 |

---

# 2. 📋 MVP列表

## 2.1 总览

| MVP | 名称 | Phase | Gate | 状态 | exp_id | 报告 |
|-----|------|-------|------|------|--------|------|
| 1.0 | ViT 1M Scaling | 1 | Gate-1 | 🚀 | `VIT-20251226-vit-1m-large-01` | [exp_vit_1m_scaling](./exp_vit_1m_scaling_20251226.md) |
| 1.1 | ViT Sweep Analysis | 1 | - | ✅ | `VIT-20251227-vit-sweep-01` | [exp_vit_sweep_analysis](./exp_vit_sweep_analysis_20251227.md) |
| 2.0 | LightGBM 1M Baseline | 1 | Gate-2 | ⏳ | - | - |
| 3.0 | Scaling Curve (N-sweep) | 1 | Gate-3 | ⏳ | - | - |
| 4.0 | Tokenization Ablation | 1 | Gate-4 | ⏳ | - | - |
| 5.0 | Loss/Label Norm Study | 1 | - | 🔆 | Run1 vs Run2 | [exp_vit_1m_scaling](./exp_vit_1m_scaling_20251226.md) |
| 6.0 | PE Ablation | 2 | - | ⏳ | - | - |
| 7.0 | Multi-task | 2 | - | ⏳ | - | - |

**状态**: ⏳计划 | 🔴就绪 | 🚀运行 | 🔆分析中 | ✅完成 | ❌取消

## 2.2 配置速查

| MVP | 数据量 | 架构 | 关键变量 | GPU |
|-----|--------|------|---------|-----|
| 1.0 | 1M | p16_h256_L6 | MSE/L1, standard/minmax | 4,5 |
| 2.0 | 1M | LightGBM | raw input | - |
| 3.0 | 10k~1M | p16_h256_L6 | num_samples | - |
| 4.0 | 200k+ | 多种 | patch/overlap/proj_fn | - |

---

# 3. 🔧 MVP规格

## Phase 1: Scaling Validation

### MVP-1.0: ViT 1M Scaling (🚀 运行中)

| 项 | 配置 |
|----|------|
| 目标 | 验证 ViT 在 1M 数据上的 log_g 预测能力 |
| 数据 | 1M train, 1k val, 10k test, noise=1.0 |
| 模型 | p16_h256_L6_a8, ~4.9M params |
| 训练 | 200 epochs, AdamW, lr=3e-4, cosine |
| 验收 | R²_val > 0.70, R²_test 需报告 |
| 当前 | ep112, R²_val=0.713 |

**Runs**:
| Run | Loss | Label Norm | proj_fn | 状态 | WandB |
|-----|------|-----------|---------|------|-------|
| Run 1 | MSE | standard | C1D | 🚀 ep96+ | [khgqjngm](https://wandb.ai/viskawei-johns-hopkins-university/vit-1m-scaling/runs/khgqjngm) |
| Run 2 | L1 | minmax | SW | 🚀 ep0+ | [6yg86hgi](https://wandb.ai/viskawei-johns-hopkins-university/vit-1m-scaling/runs/6yg86hgi) |

### MVP-2.0: LightGBM 1M Baseline (⏳ 待启动)

| 项 | 配置 |
|----|------|
| 目标 | 同口径 LightGBM baseline |
| 数据 | **同 MVP-1.0**: 1M train, 1k val, 10k test, noise=1.0 |
| 模型 | LightGBM, raw 4096-dim input |
| 验收 | R²_test + per-SNR 分 bin |

**检索 Prompt** (舱内搜索):
- `"LightGBM 1M log_g mag205_225_lowT_1M"`
- `"lgbm log_g noise_level=1.0 1M"`

### MVP-3.0: Scaling Curve (⏳ 待启动)

| 项 | 配置 |
|----|------|
| 目标 | 证明 Transformer 的数据需求 |
| 数据 | N = 10k, 50k, 100k, 200k, 500k, 1M |
| 模型 | **固定** p16_h256_L6 |
| 训练 | 固定 epochs 或 early stop |
| 验收 | R² vs N 曲线 (log scale) |

**检索 Prompt**:
- `"vit scaling log_g 10k 50k 100k 200k"`
- `"dataset size log_g vit L6 H256"`

### MVP-4.0: Tokenization Ablation (⏳ 待启动)

| 项 | 配置 |
|----|------|
| 目标 | 确认 tokenization 设计选择 |
| 数据 | 200k+ (足够体现差异) |
| 变量 | C1D vs SW, patch_size (8/16/32/64), stride/overlap, chunk norm |
| 验收 | ablation 表 + bar plot |

**检索 Prompt**:
- `"proj_fn C1D SW log_g"`
- `"patch_size=16 32 64 log_g vit"`

---

## Phase 2: Enhancements (待定)

### MVP-5.0: Loss/Label Norm Study

| 项 | 配置 |
|----|------|
| 目标 | 确认最优 loss + label norm 组合 |
| 变量 | MSE vs L1; standard vs minmax |
| 依赖 | MVP-1.0 Run1 vs Run2 完成 |

### MVP-6.0: PE Ablation

| 项 | 配置 |
|----|------|
| 目标 | 验证 PIPE 是否有增益 |
| 变量 | learned vs sinusoidal vs PIPE vs RoPE |

### MVP-7.0: Multi-task

| 项 | 配置 |
|----|------|
| 目标 | Teff/logg/[M/H] 联合预测 |
| 对比 | single-task vs multi-task |

---

# 4. 📊 进度追踪

## 4.1 看板

```
⏳计划       🔴就绪       🚀运行       🔆分析       ✅完成
MVP-2.0      -            MVP-1.0      MVP-5.0      MVP-1.1
MVP-3.0                   (Run1,Run2)
MVP-4.0
MVP-6.0
MVP-7.0
```

## 4.2 Gate进度

| Gate | MVP | 状态 | 结果 |
|------|-----|------|------|
| Gate-1 | MVP-1.0 | 🚀 | R²_val=0.713 (ep112), 待 200ep + test |
| Gate-2 | MVP-2.0 | ⏳ | - |
| Gate-3 | MVP-3.0 | ⏳ | - |
| Gate-4 | MVP-4.0 | ⏳ | - |

## 4.3 结论快照

| MVP | 结论 | 关键指标 | 同步Hub |
|-----|------|---------|---------|
| 1.0 | ViT 在 1M 数据上有效学习 log_g | R²=0.713 (ep112) | ✅ §6.3 |
| 1.1 | p16_h256_L6 是最优架构 | sweep 21 runs | ✅ §4 |

## 4.4 时间线

| 日期 | 事件 |
|------|------|
| 2025-12-26 | MVP-1.0 启动 (1M scaling) |
| 2025-12-27 | MVP-1.1 完成 (sweep 分析) |
| 2025-12-27 | Roadmap 创建 |
| 2025-12-28 | (预期) MVP-1.0 200ep 完成 |

---

# 5. 🔗 跨仓库集成

## 5.1 实验索引

| exp_id | project | topic | 状态 | MVP |
|--------|---------|-------|------|-----|
| `VIT-20251226-vit-1m-large-01` | VIT | vit | 🚀 | MVP-1.0 |
| `VIT-20251227-vit-sweep-01` | VIT | vit | ✅ | MVP-1.1 |

## 5.2 仓库链接

| 仓库 | 路径 | 用途 |
|------|------|------|
| VIT | `~/VIT/` | 训练代码 |
| 本仓库 | `logg/vit/` | 知识库 |
| Paper | `paper/vit/` | 论文草稿 |

## 5.3 运行路径

| MVP | 脚本 | 配置 | 输出 |
|-----|------|------|------|
| 1.0 | `scripts/train_vit_1m.py` | `configs/exp/vit_1m_large.yaml` | `checkpoints/vit_1m/` |
| 1.1 | sweep analysis | - | `results/` |

---

# 6. 📎 附录

## 6.1 数值汇总

| MVP | 配置 | R² (val) | MAE | 状态 |
|-----|------|----------|-----|------|
| 1.0 Run1 | MSE+C1D+standard | **0.713** | 0.38 | 🚀 ep112 |
| 1.0 Run2 | L1+SW+minmax | - | - | 🚀 ep0 |
| 1.1 Sweep Best | p16_h256_L6 | 0.662 | 0.43 | ✅ 10ep |

## 6.2 Paper Experiments Checklist (P0 Must-Have)

| # | 实验 | 对应 MVP | 状态 | 论文 Artifact |
|---|------|---------|------|--------------|
| P0.1 | 1M run + Test metrics | MVP-1.0 | 🚀 | Table: main results |
| P0.2 | LightGBM 1M baseline | MVP-2.0 | ⏳ | Table: ViT vs baselines |
| P0.3 | Scaling curve | MVP-3.0 | ⏳ | Fig: N vs R² |
| P0.4 | SNR sweep + ceiling | - | ⏳ | Fig: R² vs SNR (主图) |
| P0.5 | Tokenization ablation | MVP-4.0 | ⏳ | Table: ablation |

## 6.3 文件索引

| 类型 | 路径 |
|------|------|
| Roadmap | `vit_roadmap_20251227.md` |
| Hub | `vit_hub_20251227.md` |
| 图表 | `img/` |
| 论文 | `../../paper/vit/` |

## 6.4 更新日志

| 日期 | 变更 | 章节 |
|------|------|------|
| 2025-12-27 | 创建 Roadmap | - |
| 2025-12-27 | 整合 MVP-1.0, 1.1 结果 | §2, §4 |
| 2025-12-27 | 定义 Gate 1-4 | §1 |
