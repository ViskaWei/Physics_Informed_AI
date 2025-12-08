# 🔍 Review Agent Template

> **Purpose:** 审查同一 topic 下多个 `exp_*.md` 的元信息+结论，检查一致性/冲突，发现缺失元数据

---

## Trigger Words

`review` / `审查` / `检查` / `check`

---

## Input Specification

```
review [topic]
review lightgbm
review moe --since 2025-12-01
```

**Required:**
- `topic`: 要审查的 topic 目录名（如 `lightgbm`, `moe`, `diffusion`）

**Optional:**
- `--since YYYY-MM-DD`: 只审查该日期之后的实验
- `--files exp1.md exp2.md`: 指定具体文件

---

## Output Structure

### 1️⃣ Experiment Summary Table

| exp_id | date | data_config | model_config | main_metric | one_sentence_finding |
|--------|------|-------------|--------------|-------------|----------------------|
| VIT-20251205-lgbm-01 | 2025-12-05 | 32k/512/σ=0.1 | LightGBM n=1000, lr=0.05 | R²=0.945 | [一句话结论] |
| VIT-20251204-lgbm-02 | 2025-12-04 | 32k/512/σ=0.5 | LightGBM n=100, lr=0.1 | R²=0.674 | [一句话结论] |

**Field Extraction Rules:**
- `exp_id`: 从 Header 的 `ID:` 或 `experiment_id` 提取
- `date`: 从文件名 `exp_xxx_YYYYMMDD.md` 提取
- `data_config`: 从 §2.1 数据表格提取 `训练样本数/测试样本数/noise_level`
- `model_config`: 从 §2.2 模型 + §2.3 超参数提取
- `main_metric`: 从 ⚡核心结论速览 的关键数字表提取
- `one_sentence_finding`: 从 ⚡核心结论速览 的「一句话总结」提取

### 2️⃣ Consistency & Conflict Check

#### 2.1 一致性发现（Consistent Findings）

> 在多个实验中方向一致的结论

| Theme | Supporting Exps | Conclusion | Confidence |
|-------|----------------|------------|------------|
| [主题] | exp_01, exp_02 | [结论] | 🟢 High (N≥3) / 🟡 Medium (N=2) |

**Example:**
- 「增加 train_size 从 32k → 100k，LightGBM R² 稳定提升 ~+0.05」（支持：E02, E03）
- 「在相同 noise level 下，LightGBM 始终优于 Ridge」（支持：E01, E02, E03）

#### 2.2 潜在冲突（Potential Conflicts）

| Theme | Exp A | Exp B | Conflict | Possible Cause |
|-------|-------|-------|----------|----------------|
| [主题] | [exp_id: 结论A] | [exp_id: 结论B] | [矛盾点] | [可能原因] |

**Example:**
- E01 说「lr=0.05 最优」，E02 说「lr=0.1 最优」
  - 原因分析：E01 数据无噪声 + n=1000+，E02 有噪声 + n≤100

### 3️⃣ Missing Metadata & Patch Suggestions

> 检查 Summary Table 中 `MISSING` 字段并给出补全建议

#### Patch Checklist

```markdown
- [ ] `exp_xxx_YYYYMMDD.md`: 
      Field: 训练样本数 (§2.1)
      Status: MISSING
      Suggestion: "32,000"（依据：正文第 X 段提到 "train=32k"）

- [ ] `exp_yyy_YYYYMMDD.md`:
      Field: Noise levels (§2.1)
      Status: MISSING  
      Suggestion: "σ ∈ {0.0, 0.1, 0.2, 0.5, 1.0}"（依据：§3 图表描述）

- [ ] `exp_zzz_YYYYMMDD.md`:
      Field: model_config (§2.2)
      Status: INCOMPLETE
      需要手动确认: learning_rate 值
```

**Rules:**
- 如果正文有信息 → 给出精确补全建议（可直接复制粘贴）
- 如果找不到 → 标记「需要手动确认」，**不编造数字**

### 4️⃣ Cross-Experiment Synthesis（面向 main/hub）

> 站在 topic 主线角度的总结，供 Merge Agent 使用

#### 4.1 稳定结论（可写入 main.md §1.4 / hub.md §5 设计原则）

```markdown
| Conclusion | Evidence | Ready for Hub |
|------------|----------|---------------|
| [结论 1] | exp_01, exp_02, exp_03 | ✅ Yes |
| [结论 2] | exp_02, exp_03 | ✅ Yes |
```

#### 4.2 待验证方向（应在 hub.md §2 假设金字塔标为「待验证」）

```markdown
| Hypothesis | Status | Needs MVP |
|------------|--------|-----------|
| [假设 1] | 🟡 Partial (只有 1 个实验支持) | MVP-X.X |
| [假设 2] | ⚠️ Conflicting | 需要消歧实验 |
```

#### 4.3 建议新增 MVP（可挂到 roadmap.md）

```markdown
| Priority | Suggested MVP | Rationale |
|----------|---------------|-----------|
| 🔴 P0 | [MVP 名称] | [为什么需要这个实验] |
| 🟡 P1 | [MVP 名称] | [为什么需要这个实验] |
```

---

## Prompt Template (for AI)

```text
你是「Experiment Review Agent」。

【任务】
对同一 topic 下的多个实验报告 exp_*.md 做系统审查和汇总。

【输入】
- topic: {topic_name}
- 实验列表：以下是该 topic 下所有 exp_*.md 的完整内容

---
{exp_file_1_content}
---
{exp_file_2_content}
---
...

【输出格式】
严格按照以下四个章节输出：

### 1. Experiment Summary Table
[表格]

### 2. Consistency & Conflict Check
#### 2.1 一致性发现
[列表]
#### 2.2 潜在冲突
[表格 + 原因分析]

### 3. Missing Metadata & Patch Suggestions
[Checklist 格式]

### 4. Cross-Experiment Synthesis
#### 4.1 稳定结论
#### 4.2 待验证方向
#### 4.3 建议新增 MVP

【约束】
- 不要随意更改原始结论含义
- 不要编造任何数字；如果找不到就写「未知 / 需人工补充」
- Patch 建议必须可直接复制粘贴
```

---

## Integration Points

| Output Section | Target File | Target Section |
|----------------|-------------|----------------|
| §4.1 稳定结论 | `hub.md` | §5.1 Confirmed Principles |
| §4.1 稳定结论 | `main.md` | §1.4.1 已验证结论 |
| §4.2 待验证方向 | `hub.md` | §2.3 L3 Testable Hypotheses |
| §4.3 建议新增 MVP | `roadmap.md` | §2.1 Experiment Summary |
| Patch Suggestions | 原 `exp_*.md` | 相应章节 |

---

## Example Usage

```
用户: review lightgbm

AI: 🔍 审查 lightgbm topic...
    📁 找到 4 个实验报告:
    - exp_lightgbm_hyperparam_sweep_20251129.md
    - exp_lightgbm_noise_sweep_lr_20251204.md
    - exp_lightgbm_100k_noise_sweep_20251205.md
    - exp_lightgbm_summary_20251205.md

    ### 1. Experiment Summary Table
    | exp_id | date | data_config | ... |
    |--------|------|-------------|-----|
    | ... | ... | ... | ... |

    ### 2. Consistency & Conflict Check
    ...

    ### 3. Missing Metadata
    ✅ 所有必需字段完整

    ### 4. Cross-Experiment Synthesis
    📌 稳定结论 (3): 
    - lr 是最敏感超参数 [E01, E02, E03]
    - ...
    
    ⚠️ 待验证 (1):
    - 高噪声下最优 lr 是否与模型规模相关
    
    💡 建议新增 MVP (1):
    - P1: 测试 n=500 在 noise=1.0 下的 lr 敏感性
```

---

> **Template Version:** 1.0  
> **Created:** 2025-12-07  
> **Author:** Viska Wei
