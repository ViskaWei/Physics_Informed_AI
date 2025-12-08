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
| 🧠 Hub | [`[topic]_hub.md`](./[topic]_hub.md) | Knowledge navigation |
| 📋 Kanban | [`kanban.md`](../../status/kanban.md) | Global task board |
| 📗 Experiments | `exp_*.md` | Detailed reports |

## 📑 Contents

- [1. 🎯 Phase Overview](#1--phase-overview)
- [2. 📋 MVP List](#2--mvp-list)
- [3. 🔧 MVP Specifications](#3--mvp-specifications)
- [4. 📊 Progress Tracking](#4--progress-tracking)
- [5. 🔗 Cross-Repo Integration](#5--cross-repo-integration)
- [6. 📎 Appendix](#6--appendix)

---

# 1. 🎯 Phase Overview

> **Experiments organized by phase, each with clear objectives**

## 1.1 Phase List

| Phase | Objective | MVPs | Status | Key Output |
|-------|-----------|------|--------|------------|
| **Phase 0: Baseline** | Establish baseline | MVP-0.x | ⏳ | Baseline metrics |
| **Phase 1: [Topic]** | [Objective] | MVP-1.x | ⏳ | [Output] |
| **Phase 2: [Topic]** | [Objective] | MVP-2.x | ⏳ | [Output] |
| **Phase 3: [Topic]** | [Objective] | MVP-3.x | ⏳ | [Output] |

## 1.2 Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                   MVP Experiment Dependencies               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   [Phase 0: Baseline]                                       │
│         │                                                   │
│         ├──────────────┬──────────────┐                     │
│         ▼              ▼              ▼                     │
│   [Phase 1]      [Phase 2]      [Phase 3]                  │
│         │              │              │                     │
│         └──────────────┼──────────────┘                     │
│                        ▼                                    │
│              [Phase Final: Integration]                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 1.3 Decision Points

> **Key decision points based on experiment results**

| Point | Trigger | Option A | Option B |
|-------|---------|----------|----------|
| D1 | After MVP-1.0 | If ΔR² ≥ 0.03 → Phase 2 | If < 0.03 → Stop |
| D2 | After MVP-2.0 | If [condition] → [action] | If [condition] → [action] |

---

# 2. 📋 MVP List

> **Overview of all MVPs for quick lookup and tracking**

## 2.1 Experiment Summary

| MVP | Name | Phase | Status | experiment_id | Report |
|-----|------|-------|--------|---------------|--------|
| MVP-0.0 | [Baseline] | 0 | ⏳ | - | - |
| MVP-1.0 | [Exp name] | 1 | ⏳ | `[ID]` | [Link](./exp_xxx.md) |
| MVP-1.1 | [Exp name] | 1 | ⏳ | `[ID]` | - |
| MVP-2.0 | [Exp name] | 2 | ⏳ | `[ID]` | - |

**Status Legend:**
- ⏳ Planned | 🔴 Ready | 🚀 Running | ✅ Done | ❌ Cancelled | ⏸️ Paused

## 2.2 Configuration Reference

> **Key configurations across all MVPs**

| MVP | Data Size | Features | Model | Key Variable | Acceptance |
|-----|-----------|----------|-------|--------------|------------|
| MVP-0.0 | [train/test] | [dim] | [model] | - | baseline |
| MVP-1.0 | [train/test] | [dim] | [model] | [var] | [criteria] |
| MVP-1.1 | [train/test] | [dim] | [model] | [var] | [criteria] |

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
| **Hypothesis** | H1.1 |
| **Data** | [Data config] |
| **Model** | [Model config] |
| **Features** | [Feature config] |
| **Acceptance** | [Expected range] |
| **Exception** | [How to interpret anomalies] |

**→ Hypothesis Impact:** If result is [X], then [implication for hypothesis/design]

**Steps:**
1. [Step 1]
2. [Step 2]
3. [Step 3]

---

### MVP-1.1: [Experiment Name]

| Item | Config |
|------|--------|
| **Objective** | [One-line goal] |
| **Depends On** | MVP-1.0 |
| **Data** | [Data config] |
| **Model** | [Model config] |
| **Acceptance** | [Expected range] |

---

## Phase 2: [Phase Name]

### MVP-2.0: [Experiment Name]

(Continue with same format...)

---

# 4. 📊 Progress Tracking

## 4.1 Kanban View

```
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│  ⏳ Planned  │   🔴 Ready   │  🚀 Running  │    ✅ Done   │  ❌ Cancelled │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ MVP-2.0      │ MVP-1.1      │ MVP-1.0      │ MVP-0.0      │              │
│ MVP-2.1      │              │              │              │              │
│ MVP-3.0      │              │              │              │              │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

## 4.2 Key Conclusions Snapshot

> **One-line conclusion per completed MVP, synced to Hub**

| MVP | Conclusion | Key Metric | Synced to Hub |
|-----|------------|------------|---------------|
| MVP-0.0 | [Conclusion] | R²=X.XX | ✅ §3.1 |
| MVP-1.0 | [Conclusion] | ΔR²=+X.XX | ✅ §3.2 |

## 4.3 Timeline

| Date | Event | Notes |
|------|-------|-------|
| YYYY-MM-DD | MVP-0.0 done | baseline |
| YYYY-MM-DD | MVP-1.0 start | - |
| YYYY-MM-DD | MVP-1.0 done | ΔR²=+X.XX |
| YYYY-MM-DD | Decision D1 | Continue Phase 2 |

---

# 5. 🔗 Cross-Repo Integration

## 5.1 Experiment Index

> **Links to experiments_index/index.csv**

| experiment_id | project | topic | status | MVP |
|---------------|---------|-------|--------|-----|
| `[PROJECT]-[DATE]-[topic]-01` | VIT / BlindSpot | [topic] | ✅ | MVP-1.0 |
| `[PROJECT]-[DATE]-[topic]-02` | VIT / BlindSpot | [topic] | 🚀 | MVP-1.1 |

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
| YYYY-MM-DD | MVP-1.0 done | §4.1, §4.2 |
| YYYY-MM-DD | Added Phase 2 | §1, §2, §3 |

---

> **Template Usage:**
> 
> **Roadmap Scope:**
> - ✅ **Do:** MVP specs, execution tracking, kanban, cross-repo integration, metrics
> - ❌ **Don't:** Hypothesis management (→ hub.md), insight synthesis (→ hub.md), strategy (→ hub.md)
> 
> **Update Triggers:**
> - Planning new MVP → update §2, §3
> - MVP status change → update §4
> - After experiment → record conclusion to §4.2, sync to Hub
> 
> **Hub vs Roadmap:**
> - Hub = "What do we know? Where should we go?"
> - Roadmap = "What experiments are planned? What's the progress?"

