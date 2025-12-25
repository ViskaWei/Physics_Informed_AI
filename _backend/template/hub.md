# 🧠 <Topic> Hub
> **ID:** <PROJECT-YYYYMMDD-topic-hub>  
> **Scope:** <数据/任务/场景>  
> **Status:** Exploring | Stable | Pivoting  
> **Hub 职责:** 知识汇总（验证的/洞见/决策空白），非实验计划  
> **Roadmap:** [<topic>_roadmap.md](./<topic>_roadmap.md)

---

## 0) Executive Snapshot

### 一句话
<最确定的结论 + 关键数字>

### Canonical Scoreboard（唯一权威口径）

> **Protocol:** <train/test/noise/metric 定义>

| 模型/方法 | 指标值 | 配置 | 备注 |
|-----------|--------|------|------|
| Baseline A | <value> | <config> | <note> |
| Baseline B | <value> | <config> | <note> |
| Upper bound | <value> | <method> | 理论/Oracle |

### 当前信念（我们现在相信什么）
- <信念1：基于多实验的稳定结论>
- <信念2：关键风险/不确定性>
- <信念3：目前最值得投入的方向>

---

## 1) Canonical Evaluation Protocol（数字口径冻结）

| 项目 | 规格 |
|------|------|
| Dataset | <名称/版本> |
| Train size | <N> |
| Test size | <N> |
| Noise/regime | <定义> |
| Metric | <R²/MAE/...> |

**规则:** Hub 只允许一个 canonical 协议；任何变更写入 Changelog。

---

## 2) Knowledge Ledger（我们知道什么）

| Claim | 状态 | 证据摘要 | 含义（So What） |
|-------|------|---------|----------------|
| C1: <结论1> | ✅ 强 | <exp/来源> | <它改变了什么决策> |
| C2: <结论2> | ✅ 强 | <exp/来源> | <它改变了什么决策> |
| C3: <结论3> | 🟡 中 | <exp/来源> | <如果证伪会怎样> |
| C4: <结论4> | 🔴 待验证 | <来源> | <这是最大风险> |

> **写法建议:** Claim 数量控制在 5-9 条；每条必须回答 "So What"

---

## 3) Insights（读者应该记住的）

### I1 — <洞见标题>

| 项 | 内容 |
|----|------|
| **观察** | <事实> |
| **解释** | <为什么会这样> |
| **决策影响** | <它改变了什么决策/下一步> |

### I2 — <洞见标题>

| 项 | 内容 |
|----|------|
| **观察** | <事实> |
| **解释** | <为什么会这样> |
| **决策影响** | <它改变了什么决策/下一步> |

> **规则:** 一般 5-8 条洞见足够；每条必须有 "决策影响"

---

## 4) Decision Gaps（下一步要补的知识空白）

> 写"要回答什么"，不写"怎么做实验"

### DG1 — <Gap 标题>

| 项 | 内容 |
|----|------|
| **为什么重要** | <会改变哪个关键决策> |
| **什么能关闭它** | <什么证据/结果能关闭不确定性> |
| **决策规则** | If <result> → <decision>；Else <pivot> |

### DG2 — <Gap 标题>

| 项 | 内容 |
|----|------|
| **为什么重要** | <...> |
| **什么能关闭它** | <...> |
| **决策规则** | If ... → ... |

> **规则:** 一般 3-6 个 DG 就够

---

## 5) Parallel Lanes（多线程探索保持可读）

| Lane | 目标 | 当前状态 | 会改变什么 |
|------|------|---------|-----------|
| L-A: <名称> | <...> | ✅/🟡/⏳ | <...> |
| L-B: <名称> | <...> | ✅/🟡/⏳ | <...> |
| L-C: <名称> | <...> | ✅/🟡/⏳ | <...> |

> **规则:** Lane 是"并行研究线"的最小可读单元，防止内容互相穿插

---

## 6) Design Principles（可移植规则）

| # | 原则 | 建议 | 证据 | 适用 |
|---|------|------|------|------|
| P1 | **<原则名>** | <做/不做> | <exp_xxx> | <场景> |
| P2 | **<原则名>** | <做/不做> | <exp_xxx> | <场景> |

---

## 7) Pointers（详细内容在哪里）

| 类型 | 文件 | 说明 |
|------|------|------|
| 📍 Roadmap | [<topic>_roadmap.md](./<topic>_roadmap.md) | 实验规划与执行 |
| 📗 Experiments | `exp/exp_*.md` | 详细报告 |
| 📥 Child Hubs | <link> | 子主题 |
| 📤 Parent Hub | <link> | 上层战略 |

---

## 8) Changelog（仅记录知识改变）

| 日期 | 变更 | 影响 |
|------|------|------|
| YYYY-MM-DD | 创建 Hub | - |
| YYYY-MM-DD | <knowledge change> | <what it affects> |

---

## 📎 Appendix（可选）

### A1: Canonical Evaluation History
> 旧口径数字仅供追溯

### A2: 假设存档
> 已验证/否定的假设可折叠存档

<details>
<summary><b>战略假设 (H1-Hn)</b></summary>

| # | 假设 | 状态 | 证据 |
|---|------|------|------|
| H1 | <...> | ✅/❌ | exp_xxx |

</details>

### A3: 领域背景/术语表
> 按需添加

---

# ⚠️ Hub 职责边界

**Hub 做什么：**
- ✅ 知识汇总（我们知道什么）
- ✅ 洞见提炼（为什么重要 + 决策影响）
- ✅ 决策空白（下一步要补什么知识）
- ✅ 设计原则（可复用规则）
- ✅ 数字口径冻结（canonical scoreboard）

**Hub 不做什么：**
- ❌ 实验计划/执行跟踪 → Roadmap
- ❌ 细粒度 testable hypotheses → Roadmap/Exp
- ❌ 日常 backlog → Kanban
- ❌ 具体实验流程/代码 → Exp report

**核心区别：**
- **Hub** = "我们知道了什么？下一步该往哪走？"（**战略**）
- **Roadmap** = "我们计划跑哪些实验？进度如何？"（**执行**）
