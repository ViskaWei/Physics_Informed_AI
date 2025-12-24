# 🗺️ Experiment Roadmap

> **Topic:** TODO  
> **Author:** Viska Wei  
> **Created:** TODO | **Updated:** TODO  
> **Current Phase:** Phase X

<!-- 
📝 Language Convention:
- Headers & section titles: English (keep as-is)
- Content (objectives, conclusions, notes): Chinese OK
- Table column headers: English (keep as-is)
- Table cell content: Chinese OK
-->

## 🔗 Related Files

| Type | File | Description |
|------|------|-------------|
| 🧠 Hub | [`[topic]_hub.md`](./[topic]_hub.md) | Knowledge & strategy |
| 📋 Kanban | [`kanban.md`](../../status/kanban.md) | Global task board |
| 📗 Experiments | `exp_*.md` | Detailed reports |

## 📑 Contents

- [1. 🚦 Decision Gates](#1--decision-gates)
- [2. 📋 MVP List](#2--mvp-list)
- [3. 🔧 MVP Specifications](#3--mvp-specifications)
- [4. 📊 Progress Tracking](#4--progress-tracking)
- [5. 🔗 Cross-Repo Integration](#5--cross-repo-integration)
- [6. 📎 Appendix](#6--appendix)

---

# 1. 🚦 Decision Gates

> **Hub 推荐战略方向，Roadmap 定义怎么验证**
>
> ⚠️ **职责边界**: 只做验证计划，不做战略分析（→ Hub）

## 1.1 Current Strategic Route (from Hub)

> **来自 Hub §2 的战略推荐**

| Route | 路线名称 | Hub 推荐 | 需要验证 |
|-------|---------|---------|---------|
| Route I | [路线 I 名称] | 🟡 待验证 | Gate-1 |
| **Route M** | [路线 M 名称] | 🟢 **推荐** | Gate-2 |
| Route S | [路线 S 名称] | 🔴 高风险 | Gate-3 |

> 📖 **战略推荐理由**见 [Hub §2 Answer Key](./[topic]_hub.md#2--answer-key--strategic-route)

---

## 1.2 Gate Definitions

> **做什么实验能过哪个决策门？**

### Gate-1: [Gate 名称]

| Item | Content |
|------|---------|
| **验证什么** | [验证哪个假设/问题] |
| **对应 MVP** | MVP-X.X |
| **Outcome A** | If [条件] → [Route 选择] |
| **Outcome B** | If [条件] → [Route 选择] |
| **Status** | ⏳ Pending / 🚀 Running / ✅ Done |

### Gate-2: [Gate 名称]

| Item | Content |
|------|---------|
| **验证什么** | [验证哪个假设/问题] |
| **对应 MVP** | MVP-X.X, MVP-X.X |
| **Outcome A** | If [条件] → [Action] |
| **Outcome B** | If [条件] → [Action] |
| **Status** | ⏳ / 🚀 / ✅ |

### Gate-3: [Gate 名称]

| Item | Content |
|------|---------|
| **验证什么** | [验证哪个假设/问题] |
| **对应 MVP** | MVP-X.X |
| **Outcome A** | If [条件] → [Action] |
| **Outcome B** | If [条件] → [Action] |
| **Status** | ⏳ / 🚀 / ✅ |

---

## 1.3 This Week's Focus

> **本周要做的 2-3 个 MVP（对应 Gate 验证）**

| Priority | MVP | 对应 Gate | Why First | Status |
|----------|-----|-----------|-----------|--------|
| 🔴 P0 | MVP-X.X: [Name] | Gate-X | [理由] | ⏳ |
| 🔴 P0 | MVP-X.X: [Name] | Gate-X | [理由] | ⏳ |
| 🟡 P1 | MVP-X.X: [Name] | Gate-X | [理由] | ⏳ |

---

## 1.4 Gate Progress Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Gate Progress Flow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Hub 推荐: Route M (表征/模型)                                  │
│                    ↓                                            │
│   ┌─────────────────────────────────────┐                       │
│   │ Gate-1: [Gate 名称]                  │ Status: ⏳            │
│   │ MVP: MVP-X.X                         │                       │
│   └─────────────────────────────────────┘                       │
│          ↓ [Outcome A]        ↓ [Outcome B]                     │
│    [Action A]            [Action B]                             │
│          │                    │                                 │
│   ┌──────┴──────┐      ┌──────┴──────┐                         │
│   │ Gate-2      │      │ Gate-3      │                         │
│   │ Status: ⏳  │      │ Status: ⏳  │                         │
│   └─────────────┘      └─────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# 2. 📋 MVP List

> **Overview of all MVPs for quick lookup and tracking**

## 2.1 Experiment Summary

| MVP | Name | Phase | Gate | Status | experiment_id | Report |
|-----|------|-------|------|--------|---------------|--------|
| MVP-0.0 | [Baseline] | 0 | - | ✅ | `[ID]` | [Link](./exp_xxx.md) |
| MVP-1.0 | [Exp name] | 1 | - | ✅ | `[ID]` | [Link](./exp_xxx.md) |
| MVP-2.0 | [Exp name] | 2 | Gate-1 | ⏳ | - | - |
| MVP-2.1 | [Exp name] | 2 | Gate-2 | ⏳ | - | - |
| MVP-2.2 | [Exp name] | 2 | Gate-3 | ⏳ | - | - |

**Status Legend:**
- ⏳ Planned | 🔴 Ready | 🚀 Running | ✅ Done | ❌ Cancelled | ⏸️ Paused

## 2.2 Configuration Reference

> **Key configurations across all MVPs**

| MVP | Data Size | Features | Model | Key Variable | Acceptance |
|-----|-----------|----------|-------|--------------|------------|
| MVP-0.0 | [train/test] | [dim] | [model] | - | baseline |
| MVP-1.0 | [train/test] | [dim] | [model] | [var] | [criteria] |
| MVP-2.0 | [train/test] | [dim] | [model] | [var] | [criteria] |

---

# 3. 🔧 MVP Specifications

> **Detailed specs for each MVP, ready for execution**

## Phase 0: Baseline

### MVP-0.0: [Baseline Name]

| Item | Config |
|------|--------|
| **Objective** | [One-line goal] |
| **Data** | [Data config] |
| **Model** | [Model config] |
| **Acceptance** | [Expected range] |
| **Early Stop** | [When to stop and debug] |

**Troubleshooting Checklist** (if not meeting criteria):
- [ ] [Check item 1]
- [ ] [Check item 2]

---

## Phase 1: [Phase Name]

### MVP-1.0: [Experiment Name]

| Item | Config |
|------|--------|
| **Objective** | [What question to answer?] |
| **Data** | [Data config] |
| **Model** | [Model config] |
| **Features** | [Feature config] |
| **Acceptance** | [Expected range] |

---

## Phase 2: Gate Verification

> **用于验证 Decision Gates 的实验**

### MVP-2.0: [Gate-1 验证实验]

| Item | Config |
|------|--------|
| **Objective** | [验证 Gate-1 的问题] |
| **Gate** | Gate-1 |
| **Data** | [Data config] |
| **Model** | [Model config] |
| **Acceptance** | [Outcome A/B 的判定标准] |

**→ Gate Impact:** 
- If R² ≥ X.XX → [Outcome A: 选择 Route X]
- If R² < X.XX → [Outcome B: 选择 Route Y]

---

### MVP-2.1: [Gate-2 验证实验]

| Item | Config |
|------|--------|
| **Objective** | [验证 Gate-2 的问题] |
| **Gate** | Gate-2 |
| **Data** | [Data config] |
| **Model** | [Model config] |
| **Acceptance** | [Outcome A/B 的判定标准] |

---

# 4. 📊 Progress Tracking

## 4.1 Kanban View

```
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│  ⏳ Planned  │   🔴 Ready   │  🚀 Running  │    ✅ Done   │  ❌ Cancelled │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ MVP-2.1      │ MVP-2.0      │              │ MVP-0.0      │              │
│ MVP-2.2      │              │              │ MVP-1.0      │              │
│              │              │              │ MVP-1.1      │              │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

## 4.2 Gate Progress

| Gate | MVP | Status | Result | Outcome |
|------|-----|--------|--------|---------|
| Gate-1 | MVP-2.0 | ⏳ | - | - |
| Gate-2 | MVP-2.1, MVP-2.2 | ⏳ | - | - |
| Gate-3 | MVP-2.3 | ⏳ | - | - |

## 4.3 Key Conclusions Snapshot

> **One-line conclusion per completed MVP, synced to Hub**

| MVP | Conclusion | Key Metric | Synced to Hub |
|-----|------------|------------|---------------|
| MVP-0.0 | [Conclusion] | R²=X.XX | ✅ §2.1 A) |
| MVP-1.0 | [Conclusion] | ΔR²=+X.XX | ✅ §2.1 B) |

## 4.4 Timeline

| Date | Event | Notes |
|------|-------|-------|
| YYYY-MM-DD | MVP-0.0 done | baseline |
| YYYY-MM-DD | MVP-1.0 done | - |
| YYYY-MM-DD | Gate-1 验证开始 | - |

---

# 5. 🔗 Cross-Repo Integration

## 5.1 Experiment Index

> **Links to experiments_index/index.csv**

| experiment_id | project | topic | status | MVP |
|---------------|---------|-------|--------|-----|
| `[PROJECT]-[DATE]-[topic]-01` | VIT / BlindSpot | [topic] | ✅ | MVP-1.0 |
| `[PROJECT]-[DATE]-[topic]-02` | VIT / BlindSpot | [topic] | 🚀 | MVP-2.0 |

## 5.2 Repository Links

| Repo | Directory | Purpose |
|------|-----------|---------|
| VIT | `~/VIT/results/[topic]/` | Training results |
| BlindSpot | `~/BlindSpotDenoiser/evals/` | Evaluation results |
| This repo | `logg/[topic]/` | Knowledge base |

## 5.3 Run Path Records

> **Actual run paths for reproducibility**

| MVP | Repo | Script | Config | Output |
|-----|------|--------|--------|--------|
| MVP-1.0 | VIT | `~/VIT/scripts/xxx.py` | `configs/xxx.yaml` | `lightning_logs/vX` |

---

# 6. 📎 Appendix

## 6.1 Results Summary

> **Core metrics from all MVPs**

### Main Metrics Comparison

| MVP | Config | $R^2$ | MAE | RMSE | ΔR² vs Baseline |
|-----|--------|-------|-----|------|-----------------|
| MVP-0.0 | [config] | X.XXX | X.XX | X.XX | - |
| MVP-1.0 | [config] | X.XXX | X.XX | X.XX | +X.XXX |

### [Dimension] Sweep Results

| [Dim] | $R^2$ | MAE | Notes |
|-------|-------|-----|-------|
| [val 1] | X.XXX | X.XX | |
| [val 2] | X.XXX | X.XX | |

---

## 6.2 File Index

| Type | Path | Description |
|------|------|-------------|
| Roadmap | `logg/[topic]/[topic]_roadmap_YYYYMMDD.md` | This file |
| Hub | `logg/[topic]/[topic]_hub_YYYYMMDD.md` | Knowledge navigation |
| MVP-1.0 | `logg/[topic]/exp_xxx_YYYYMMDD.md` | [Experiment name] |
| Images | `logg/[topic]/img/` | Experiment figures |

---

## 6.3 Changelog

| Date | Change | Sections |
|------|--------|----------|
| YYYY-MM-DD | Created Roadmap | - |
| YYYY-MM-DD | Added Decision Gates | §1 |
| YYYY-MM-DD | MVP-2.0 done, Gate-1 passed | §1.2, §4 |

---

> **Template Usage:**
> 
> ## Hub vs Roadmap 职责分工
> 
> | 问题 | Hub | Roadmap |
> |------|-----|---------|
> | 我们知道什么？ | ✅ §2 Answer Key | |
> | 该往哪走？ | ✅ §2 Strategic Route | |
> | 怎么验证？（Decision Gates） | | ✅ §1 |
> | 做哪些实验？ | | ✅ §2, §3 |
> | 本周做什么？ | | ✅ §1.3 This Week's Focus |
> | 进度如何？ | | ✅ §4 |
> | 学到了什么洞见？ | ✅ §3 Confluence | |
> | 设计原则是什么？ | ✅ §4 Principles | |
> 
> ## Roadmap Scope
> - ✅ **Do:** Decision Gates, MVP specs, execution tracking, progress, cross-repo integration
> - ❌ **Don't:** Insight synthesis (→ hub.md), strategic reasoning (→ hub.md)
> 
> ## Update Triggers
> - Planning new MVP → update §2, §3
> - MVP status change → update §4
> - Gate result → update §1.2, §4.2, sync conclusion to Hub §2
> - New Gate needed → update §1.2

