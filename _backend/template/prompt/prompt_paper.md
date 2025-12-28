# 📝 Paper 检查与 TODO 生成 Prompt

> **触发词:** `paper <topic>`  
> **示例:** `paper vit`, `paper moe`, `paper scaling`

---

## 🎯 目标

自动分析论文草稿，**同步 logg/ 中的最新实验结果**，检查缺失内容，生成结构化 TODO 列表，并**自动更新 paper 和 checklist**。

---

## 📋 执行步骤

### Step 1：定位文件

1. **论文正文**: `paper/[topic]/` 目录下的 `.md` 文件
2. **实验清单**: `paper/[topic]/experiments_checklist.md`（如果存在）
3. **实验日志**: `logg/[topic]/` 目录
4. **智库导航**: `logg/[topic]/[topic]_hub.md`（如果存在）

### Step 2：扫描论文正文

分析论文结构，识别以下内容：

| 检查项 | 查找模式 | 状态标记 |
|--------|----------|----------|
| 摘要 | `## 摘要` 或 `## Abstract` | ✅/❌ |
| 引言 | `## 1. 引言` 或 `## Introduction` | ✅/❌ |
| 相关工作 | `## 2. 相关工作` | ✅/❌ |
| 方法 | `## 方法` 或 `## Method` | ✅/❌ |
| 实验 | `## 实验` 或 `## Experiments` | ✅/❌ |
| 结论 | `## 结论` 或 `## Conclusion` | ✅/❌ |
| 参考文献 | `## 参考文献` 或 `## References` | ✅/❌ |

### Step 3：识别 TODO 标记

扫描论文正文中的待办项：

```
匹配模式:
- "TODO"
- "待完成"
- "待添加"
- "[P0]", "[P1]", "[P2]"
- "⏳"
- "🔆"
- 表格中的空单元格
```

### Step 4：检查实验清单（如果存在）

读取 `experiments_checklist.md`，提取：
- 所有 P0 实验的状态
- 未完成的 `[ ]` 项
- 搜索词用于后续检索

### Step 5：在 logg/ 中检索已有结果

使用实验清单中的搜索词，在 `logg/[topic]/` 中检索：

```bash
# 搜索模式
grep -r "[搜索词]" logg/[topic]/
```

标记搜索结果：
- ✅ 找到：链接到对应文件
- ❌ 未找到：需要执行实验

### Step 6：同步实验结果到 Paper 和 Checklist ⭐ 新增

**核心原则**: 用 `logg/[topic]/` 中的最新结果更新 paper 和 checklist，消除不一致。

**6.1 从 logg/ 提取最新数据**:

| 提取内容 | 来源文件 | 目标位置 |
|---------|---------|---------|
| 主要指标（R², MAE） | `[topic]_hub.md` §0 权威数字 | paper 主表格 |
| Baseline 对比 | `[topic]_hub.md` §6.3 关键数字速查 | paper 对比表 |
| Scaling 数据 | `exp_*_scaling*.md` 数值表 | paper §6.4 |
| 消融结果 | `exp_*_ablation*.md` | paper §6.6 |
| 理论上限 | `*_hub.md` ceiling 数据 | paper §5 |

**6.2 更新 Paper 中的 TODO**:

```
匹配并替换:
- 表格中的 "TODO" → 实际数值
- 表格中的 "⏳" → 实际状态
- 图表路径 → 确认文件存在
- 空表格行 → 填充数据
```

**6.3 同步 Checklist 状态**:

```
对比 hub.md 和 checklist.md:
- hub 中 ✅ 但 checklist 中 ⏳ → 更新 checklist 为 ✅
- hub 中有结果但 checklist 未记录 → 添加到 checklist
- 数值不一致 → 以 hub 为准
```

**6.4 写入更新**:
- 直接修改 `paper/[topic]/*.md`
- 直接修改 `paper/[topic]/experiments_checklist.md`
- 保持文件格式和结构不变

---

### Step 7：生成结构化 TODO

按优先级输出剩余 TODO 列表（已更新的项不再列出）：

---

## 📤 输出格式

```markdown
# 📋 [topic] 论文 TODO 清单

> **生成时间:** YYYY-MM-DD HH:MM  
> **论文文件:** `paper/[topic]/xxx.md`  
> **实验清单:** `paper/[topic]/experiments_checklist.md`

---

## 📊 完成度概览

| 类别 | 完成 | 待办 | 进度 |
|------|------|------|------|
| 论文章节 | X/Y | - | XX% |
| P0 实验 | X/Y | Z | XX% |
| P1 实验 | X/Y | Z | XX% |
| P2 实验 | X/Y | Z | XX% |

---

## 🔴 投稿阻塞项（必须解决）

### 缺失章节
- [ ] [章节名] - 缺少内容

### 未完成 P0 实验
- [ ] P0.X [实验名] - [状态] - [下一步行动]

### 论文正文 TODO
- [ ] [位置]: [TODO 内容]

---

## 🟡 应该完成（增强质量）

- [ ] P1.X [实验名] - [状态]
- [ ] [优化项]

---

## 🟢 可选完成

- [ ] P2.X [实验名]

---

## 🔍 检索结果

| 搜索词 | 状态 | 文件位置 |
|--------|------|----------|
| "[关键词]" | ✅/❌ | `logg/xxx.md:行号` |

---

## 🎯 推荐下一步

1. **最紧急:** [具体行动]
2. **次紧急:** [具体行动]
3. **可并行:** [具体行动]
```

---

## ⚠️ 执行规则

1. **⭐ 自动同步**: 用 `logg/[topic]/` 中的最新结果**自动更新** paper 和 checklist
2. **数据来源优先级**: `[topic]_hub.md` > `exp_*.md` > `roadmap.md`
3. **优先级排序**: P0 > P1 > P2，投稿阻塞项最先列出
4. **具体行动**: 每个 TODO 项应包含可执行的下一步
5. **链接文件**: 检索到的结果应链接到具体文件和行号
6. **检查实验清单**: 如果 `experiments_checklist.md` 不存在，提示创建
7. **保持格式**: 更新时保持原文件的 markdown 格式和结构

---

## 📁 文件依赖

| 文件 | 必需 | 用途 |
|------|------|------|
| `paper/[topic]/*.md` | ✅ | 论文正文 |
| `paper/[topic]/experiments_checklist.md` | 可选 | 实验追踪 |
| `logg/[topic]/` | 可选 | 检索已有结果 |
| `_backend/template/paper_checklist.md` | 模板 | 创建新 checklist |

---

## 🔄 特殊情况处理

### 情况 1：experiments_checklist.md 不存在

输出提示：
```
⚠️ 实验清单不存在
建议: 参考模板 `_backend/template/paper_checklist.md` 创建
位置: `paper/[topic]/experiments_checklist.md`
```

### 情况 2：论文正文为空或不完整

输出基础结构建议：
```
📝 论文结构建议:
1. 摘要 - 核心贡献 1-3 点
2. 引言 - 问题/动机/贡献
3. 相关工作 - 3-5 个方向
4. 方法 - 模型/算法描述
5. 实验 - 数据/基线/结果/消融
6. 结论 - 总结/局限/未来工作
```

### 情况 3：多个论文文件

列出所有找到的文件，让用户选择：
```
找到多个论文文件:
1. paper/[topic]/xxx_cn.md (中文)
2. paper/[topic]/xxx.md (英文)
请指定: `paper [topic] [1/2]`
```

---

## 📎 示例

### 输入
```
paper vit
```

### 执行
1. 定位 `paper/vit/specvit_paper_cn.md`
2. 定位 `paper/vit/experiments_checklist.md`
3. 扫描论文正文中的 TODO
4. 检查实验清单 P0-P2 状态
5. 在 `logg/vit/` 和 `logg/scaling/` 中检索
6. 生成 TODO 清单

### 输出
```markdown
# 📋 vit 论文 TODO 清单

> **生成时间:** 2025-12-28 10:00  
> **论文文件:** `paper/vit/specvit_paper_cn.md`

## 📊 完成度概览

| 类别 | 完成 | 待办 | 进度 |
|------|------|------|------|
| 论文章节 | 8/8 | 0 | 100% |
| P0 实验 | 2/5 | 3 | 40% |
| P1 实验 | 1/4 | 3 | 25% |

## 🔴 投稿阻塞项

- [ ] P0.3 规模化曲线 - ⏳ 待开始 - 跑 vit 10k/50k/100k
- [ ] P0.5 Tokenization 消融 - ⏳ 待开始 - 设计消融矩阵

## 🎯 推荐下一步

1. **最紧急:** 等待 P0.1 100万训练完成（进行中）
2. **次紧急:** 启动 P0.2 LightGBM 基线实验
3. **可并行:** 设计 P0.3 规模化曲线实验配置
```

---

> **最后更新:** 2025-12-28
