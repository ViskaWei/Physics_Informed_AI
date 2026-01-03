# 📖 实验管理系统 - 命令速查手册

> **最后更新**: 2025-12-03  
> **适用仓库**: VIT | BlindSpotDenoiser | SpecDiffusion | Physics_Informed_AI

---

## 🗂️ 目录

- [快捷命令总览](#-快捷命令总览)
- [仓库路由规则](#-仓库路由规则)
- [Physics_Informed_AI 命令](#-physics_informed_ai-知识中心)
- [VIT 命令](#-vit-训练实验室)
- [BlindSpot 命令](#-blindspotdenoiser-训练实验室)
- [SpecDiffusion 命令](#-specdiffusion-diffusion-专用)
- [Shell 命令](#-shell-命令)
- [完整工作流](#-完整工作流)

---

## 🎯 快捷命令总览

| 命令 | VIT | BlindSpot | SpecDiff | Physics_AI | 作用 |
|------|:---:|:---------:|:--------:|:----------:|------|
| `?` | ✅ | ✅ | ✅ | ✅ | 查看进度状态 |
| `study code` | ❌ | ❌ | ❌ | ✅ | 编程题拆解+索引+进度 |
| `sync` | ✅ | ✅ | ✅ | ✅ | 同步实验到索引 |
| `reg` | ✅ | ✅ | ✅ | ❌ | 登记单个实验 |
| `n` | ✅ | ✅ | ✅ | ✅ | 新建实验计划 |
| `a` | ❌ | ❌ | ❌ | ✅ | 归档实验结果 |
| `u [exp_id]` | ❌ | ❌ | ❌ | ✅ | 🆕 完整更新+同步hub/roadmap+git push |
| `report` | ❌ | ❌ | ❌ | ✅ | 生成周报 |
| `card` | ❌ | ❌ | ❌ | ✅ | 创建知识卡片 |
| `design` | ❌ | ❌ | ❌ | ✅ | 提取设计原则 |
| `todo` | ❌ | ❌ | ❌ | ✅ | 管理待办 |
| `results` | ✅ | ✅ | ✅ | ❌ | 查看本地结果 |

---

## 🚀 仓库路由规则

**根据实验 topic 自动选择代码仓库：**

| Topic | 代码仓库 | 说明 |
|-------|---------|------|
| `diffusion` | `~/SpecDiffusion` | Diffusion 专用仓库 |
| `cnn`, `swin`, `ridge`, `pca`, `lightgbm`, `gta`, `noise`, `topk`, `train` | `~/VIT` | 通用 ML 实验 |
| `distill`, `latent`, `probe`, `encoder` | `~/BlindSpotDenoiser` | 去噪 / Latent 实验 |

**Experiment ID 前缀**：

| 仓库 | 前缀 | 示例 |
|------|------|------|
| VIT | `VIT-` | `VIT-20251203-cnn-01` |
| BlindSpot | `BS-` | `BS-20251203-latent-01` |
| SpecDiffusion | `SD-` | `SD-20251203-diff-supervised-01` |

---

## 📚 Physics_Informed_AI (知识中心)

### ❓ 查看进度
```
触发词: ? | ？ | status | 进度 | 状态
```

**输出内容**:
- 📊 实验索引统计
- 📋 P0/P1 待办任务
- 📦 归档队列状态
- 📝 最近更新的文档

---

### 🧠 编程题整理
```
触发词: study code [标识]
```

**作用**: 从 `ML/coding.md` 自动拆解一题到 `ML/coding/<category>/YYYYMMDD_slug.{md,py}`，并更新：
- `ML/coding/index.md`（索引）
- `ML/progress.md`（编程1/2 计数、进度条、总进度、「按主题进度」）

**分类目录**:
- `ml_regression` | `ml_clustering` | `ml_tree` | `ml_feature`
- `dl_conv` | `dl_attention` | `dl_quant` | `dl_prune` | `dl_rnn`
- `nlp_tokenizer` | `moe` | `dp` | `graph_tree` | `sliding_window`

**示例**:
```
用户: study code P4274
AI:
✅ 已拆解：20260102_max_energy_path → dp/
📄 题解：ML/coding/dp/20260102_max_energy_path.md
💻 代码：ML/coding/dp/20260102_max_energy_path.py
📑 索引：ML/coding/index.md 已更新
📈 进度：编程1 +1 | 总进度 9/75
```

**提示**:
- 题目来源默认读取 `ML/coding.md` 中最近一次完整的“思路+答案”片段
- 二刷请在命令后加 `--revisit` 或备注“二刷”，计入编程2

---

### 🔄 同步实验
```
触发词: sync | 同步 | scan
```

**作用**: 扫描 VIT 和 BlindSpot 仓库，同步到 `experiments_index/index.csv`

**示例**:
```
用户: sync

AI: 🔄 同步实验...
    📁 扫描 VIT 仓库... ✨ 发现 5 个新实验
    📁 扫描 BlindSpot 仓库... ✨ 发现 2 个新实验
    ✅ 已更新: experiments_index/index.csv
```

---

### 🆕 新建实验
```
触发词: n | N | new | 新建
```

**作用**: 创建 `logg/[topic]/exp_*.md` 实验框架

**示例**:
```
用户: n 测试 MLP 隐藏层数量对 log g 预测的影响

AI: 📝 创建实验计划...
    ✅ 已创建: logg/NN/exp_mlp_hidden_layers_20251201.md
    ✅ 已更新: logg/NN/NN_main_20251130.md §3.1
```

---

### 📦 归档实验
```
触发词: a | A | 归档 | archive
变体: a 1 | a all | s (查看队列)
```

**作用**: 将 `raw/` 中的原始结果归档为结构化 `exp.md`

**示例**:
```
用户: a

AI: 📋 当前归档队列 (3个):
    1. raw/MVP_EXPERIMENTS_FULL_REPORT.md → logg/gta/
    2. raw/logg_from_encoder_experiment_report.md → logg/distill/
    要归档哪个？(输入序号或 "all")

用户: 1

AI: ✅ 已创建: logg/gta/exp_mvp_experiments_20251201.md
```

---

### 📝 更新文档 🆕
```
触发词: u | U | update | 更新
变体: 
  - u [experiment_id] - 完整更新：补全exp+同步hub/roadmap+git push
  - u [关键词] - 智能追加内容+git push
```

**作用**: 
1. **完整更新模式** (`u [experiment_id]`): 审查exp报告完整性 → 去代码仓补全遗漏 → 同步hub/roadmap → 自动git push
2. **智能追加模式** (`u [关键词]`): 匹配文件 → 更新相关章节 → 自动git push

**示例 1 - 完整更新**:
```
用户: u VIT-20251222-scaling-ridge-01

AI: 📝 完整更新实验报告...
    
    📖 Step 2: 审查报告完整性
    | 章节 | 状态 |
    |------|------|
    | ⚡ 核心结论速览 | ✅ |
    | §6.2 实验流程 | ❌ 缺少代码引用 |
    
    🔧 Step 3: 补全遗漏内容
    📁 搜索 ~/VIT/ 仓库...
    ✅ 找到: ~/VIT/scripts/scaling_ridge_experiment.py
    ✅ 已补充到 §6.2 实验流程记录
    
    📤 Step 4: 同步到 hub.md
    ✅ §2 假设金字塔: H1.1 状态 → ✅
    ✅ §3 洞见汇合站: 添加新洞见
    
    📤 Step 5: 同步到 roadmap.md
    ✅ §2.1 实验总览: MVP-1.1 状态 → ✅
    ✅ §4.2 核心结论快照: 添加条目
    
    📦 Step 6: Git Commit + Push
    ✅ git commit -m "update: VIT-20251222-scaling-ridge-01"
    ✅ git push
```

**示例 2 - 智能追加**:
```
用户: u CNN 实验发现 dilation=2 时感受野覆盖吸收线宽度

AI: 📝 更新文档...
    🔍 匹配到: logg/cnn/exp_cnn_dilated_kernel_sweep_20251201.md
    ✅ 已添加到 §4.1 宏观层洞见
    
    📦 Git Push...
    ✅ git commit + push 完成
```

**代码仓库路由**:
| experiment_id 前缀 | 代码仓库 |
|-------------------|---------|
| `VIT-*` | `~/VIT/` |
| `BS-*` | `~/BlindSpotDenoiser/` |
| `SD-*` | `~/SpecDiffusion/` |

---

### 📊 生成报告
```
触发词: report | 汇报 | 周报
变体: report -4h | report -24h (简短报告给 LLM)
```

**作用**: 生成周报/月报/简短报告

**示例**:
```
用户: report

AI: 📊 生成周报...
    ✅ 报告已生成: reports/drafts/weekly_2025-12-01.md
```

---

### 🆕 简短报告 (LLM 优化)
```
触发词: report -Nh (如 -4h, -24h)
```

**作用**: 生成过去 N 小时的简短摘要，优化给 ChatGPT/Claude 分析

**特点**:
- 自动提取核心结论
- 包含给 AI 的思考提示
- 直接输出到终端便于复制

**示例**:
```
用户: report -4h

AI: 📊 生成简短报告 (过去 4 小时)...
    
    ════════════════════════════════════════
    📋 COPY THE FOLLOWING TO ChatGPT/Claude:
    ════════════════════════════════════════
    
    # 🧠 实验进展摘要 (Past 4h)
    
    ## 📌 核心结论
    - [cnn] Best R²=0.992, dilation=2 最优
    
    ## 🤖 给 AI 的提示
    1. 结论验证: 上述结论是否合理？
    2. 下一步建议: 最值得尝试什么？
    ...
```

---

### 📐 设计原则 (Design Principles)

```
触发词: design | 设计原则 | 原则
```

**作用**: 从各 hub.md 文件中提取新增的设计原则，并整合到汇总文件

**文件位置**:
- **汇总文件**: `design/principles.md`（精简命名）
- **提取脚本**: `_backend/scripts/extract_design_principles.py`

**工作流程**:
```
用户: design

AI: 🔍 扫描hub文件中的设计原则...
    📅 上次同步时间: 2025-12-25
    
    📁 找到 8 个hub文件
      ✅ fisher_hub_20251225.md: 发现 5 个原则
      ✅ ridge_hub_20251223.md: 发现 7 个原则
    
    📊 总共发现 12 个新增设计原则
    
    📝 追加到 design/principles.md...
    ✅ 已更新: design/principles.md
    ✅ 已更新最后同步时间: 2025-12-26
```

**说明**:
- 自动扫描所有 `*_hub*.md` 文件
- 只提取"已确认原则"表格中的条目（跳过"待验证原则"）
- 基于文件修改时间和上次同步时间判断是否为新增
- 自动去重，避免重复添加
- 保持原有分类结构（Ridge、LightGBM、NN、MoE 等）

**检查模式**:
```
用户: design --check

AI: 🔍 检查模式：仅检测不写入
    📊 发现 5 个新增原则（预览）
    💡 运行 'design' 来实际写入
```

### 📇 知识卡片 (Card)
```
触发词: card | 卡片 | kc
```

**定义**: Card 是**可复用的阶段性知识**，不是实验报告，不是 hub，不是 roadmap
- ✅ **做**: 跨多个实验的结构性认知、理论依据、可指导决策的结论
- ❌ **不做**: 指导下一步实验（这是 hub 的职责）

**位置规则**:
- 单主题 (如 fisher) → 在实验父文件夹下: `logg/scaling/card/`
- 跨主题 (如 ridge 涉及多个文件夹) → 在 `logg/card/`

**模板**: `_backend/template/card.md`

**工作流程**:
```
用户: card [关键词/描述]
    │
    ├─ Step 1: 确定 Card 位置
    │   - 解析关键词 → 判断是单主题还是跨主题
    │   - 单主题 → logg/[parent_topic]/card/
    │   - 跨主题 → logg/card/
    │   - 若 card/ 目录不存在 → 自动创建
    │
    ├─ Step 2: 检索所有相关实验
    │   - 在相关 logg/[topic]/exp/ 中搜索
    │   - 提取核心结论、关键数字
    │
    ├─ Step 3: 按 card.md 模板生成卡片
    │   - 🧠 Card ID｜一句话结论
    │   - 1️⃣ 数学/理论依据
    │   - 2️⃣ 实验结果（关键证据）
    │   - 3️⃣ 实验链接
    │
    └─ Step 4: 保存 + Git Commit
        card_[name]_YYYYMMDD.md
```

**示例**:
```
用户: card Fisher 理论上限

AI: 📇 创建知识卡片...
    
    🔍 检索相关实验:
    - logg/scaling/exp/exp_scaling_fisher_ceiling_v2_20251224.md
    - logg/scaling/exp/exp_scaling_fisher_multi_mag_20251224.md
    
    📂 确定位置: logg/scaling/card/ (单主题 fisher → scaling)
    
    ✅ 已创建: logg/scaling/card/card_fisher_ceiling_20251225.md
    
    📝 Card 预览:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🧠 Card F1｜log g 理论上限 R²_max = 0.89，当前模型仅利用 64%
    
    1️⃣ 理论依据: Fisher/CRLB，高斯噪声下的最优估计
    2️⃣ 关键结果: R²_max(median)=0.89, LightGBM=0.57
    3️⃣ 来源: 2 个实验
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
用户: card ridge 最优 alpha

AI: 📇 创建知识卡片...
    
    🔍 检索相关实验:
    - logg/ridge/exp/exp_ridge_alpha_extended_20251224.md
    - logg/scaling/exp/exp_scaling_ridge_100k_20251222.md
    - logg/benchmark/exp/exp_ridge_100k_noise_sweep_20251205.md
    
    📂 确定位置: logg/card/ (跨主题: ridge, scaling, benchmark)
    
    ✅ 已创建: logg/card/card_ridge_alpha_optimal_20251225.md
```

**Card vs Hub vs Exp**:

| 类型 | 职责 | 指导下一步？ |
|------|------|-------------|
| Card | 可复用的阶段性知识、结构性认知 | ❌ 不指导 |
| Hub | 问题树、假设、战略导航 | ✅ 指导 |
| Exp | 单次实验的详细记录 | ⚠️ 仅建议 |

---

### 📌 下一步计划
```
触发词: next | 下一步 | 计划
变体: next add | next done | next plan
```

**作用**: 查看、添加、管理下一步计划

**基本操作**:
```
用户: next                    # 查看当前计划
用户: next add P0 完成文档     # 添加 P0 任务
用户: next add P1 测试新模型   # 添加 P1 任务
用户: next done 1             # 完成第 1 个任务
```

**智能计划**:
```
用户: next plan

AI: 🧠 分析最近实验...
    
    💡 建议的下一步:
    🔴 P0: CNN dilation=2 效果好 → 测试 multi-scale
    🟡 P1: 写 knowledge card 总结发现
    
    是否添加？(y/n)
```

**从实验生成**:
```
用户: next from logg/cnn/exp_cnn_xxx.md

AI: 📖 分析实验报告...
    💡 建议: 测试更大 dilation, 对比 Transformer
```

---

## 🔬 VIT (训练实验室)

### ❓ 查看状态
```
触发词: ? | status | 进度
```

**输出**: 本地实验状态 + 与知识中心同步状态

---

### 📤 登记实验
```
触发词: reg | register | 登记
```

**作用**: 将完成的实验登记到知识中心

**示例**:
```
用户: reg cnn dilated 实验完成，R2=0.992

AI: 📝 登记实验...
    Experiment ID: VIT-20251201-cnn-dilated-01
    ✅ 已登记到: experiments_index/index.csv
```

---

### 🔄 同步到知识中心
```
触发词: sync | 同步 | push
```

**作用**: 批量扫描 VIT 实验并同步

**示例**:
```
用户: sync

AI: 🔄 同步 VIT 实验...
    📁 扫描 lightning_logs/... ✨ 3 个新实验
    📁 扫描 results/... ✨ 5 个新实验
    ✅ 已同步 8 个实验
```

---

### 🆕 新建实验
```
触发词: n | new | 新建
```

**作用**: 生成 experiment_id + 创建配置

**示例**:
```
用户: n 测试不同 dilation rate

AI: 📝 创建新实验...
    Experiment ID: VIT-20251201-cnn-dilation-01
    建议配置: configs/exp/cnn_dilation_sweep.yaml
```

---

### 📊 查看结果
```
触发词: results | 结果 | r
```

**作用**: 展示最近的实验结果

**示例**:
```
用户: results

AI: 📊 最近实验结果:
    lightning_logs/
    ├── version_42 (2h ago) - R2=0.992
    └── version_41 (5h ago) - R2=0.985
```

---

## 🔬 BlindSpotDenoiser (训练实验室)

### ❓ 查看状态
```
触发词: ? | status | 进度
```

---

### 📤 登记实验
```
触发词: reg | register | 登记
```

**示例**:
```
用户: reg latent probe 完成，R2=0.598

AI: 📝 登记实验...
    Experiment ID: BS-20251201-latent-probe-01
    ✅ 已登记
```

---

### 🔄 同步到知识中心
```
触发词: sync | 同步 | push
```

---

### 🆕 新建实验
```
触发词: n | new | 新建
```

---

### 📊 查看结果
```
触发词: results | 结果 | r
```

---

## 🔬 SpecDiffusion (Diffusion 专用)

> **⚠️ 所有 diffusion 相关实验必须在此仓库执行**

### ❓ 查看状态
```
触发词: ? | status | 进度
```

**输出**: 本地 diffusion 实验状态 + 与知识中心同步状态

---

### 📤 登记实验
```
触发词: reg | register | 登记
```

**作用**: 将完成的 diffusion 实验登记到知识中心

**示例**:
```
用户: reg diffusion supervised 完成，MSE=0.0045

AI: 📝 登记实验...
    Experiment ID: SD-20251203-diff-supervised-01
    ✅ 已登记到: experiments_index/index.csv
```

---

### 🔄 同步到知识中心
```
触发词: sync | 同步 | push
```

**作用**: 批量扫描 SpecDiffusion 实验并同步

---

### 🆕 新建实验
```
触发词: n | new | 新建
```

**作用**: 生成 experiment_id + 创建配置

**示例**:
```
用户: n 测试 DPS 后验采样

AI: 📝 创建新实验...
    Experiment ID: SD-20251203-diff-dps-01
    建议配置: configs/diffusion/dps.yaml
```

---

### 📊 查看结果
```
触发词: results | 结果 | r
```

**作用**: 展示最近的 diffusion 实验结果

**示例**:
```
用户: results

AI: 📊 最近 diffusion 实验:
    lightning_logs/diffusion/
    ├── supervised (2h ago) - MSE=0.0045
    └── baseline (1d ago) - Loss=0.0072
```

---

## 🚀 训练自动化系统

> **位置**: `_backend/scripts/training/`  
> **功能**: 健康检查 + 训练监控 + 后处理自动化

### 核心理念

1. **前几分钟健康检查** - NaN、显存溢出、loss 爆炸
2. **通过后让它自己跑** - 不用一直看 log
3. **完成后自动触发下一步** - eval / 画图 / summary
4. **只给 Cursor 精简信息** - summary.json，而不是完整日志

---

### 🎯 快速启动（推荐）

```bash
# 进入训练仓库
cd ~/VIT

# 使用驱动器启动训练
python ~/Physics_Informed_AI/_backend/scripts/training/driver.py \
    --config configs/exp/moe.yaml \
    --exp-id VIT-20251204-moe-01

# 或使用 shell 包装
~/Physics_Informed_AI/_backend/scripts/training/train.sh \
    VIT-20251204-moe-01 configs/exp/moe.yaml
```

---

### 📋 驱动器参数

```bash
python driver.py \
    --config configs/xxx.yaml \       # 配置文件（或 --cmd "完整命令"）
    --exp-id VIT-20251204-xxx \       # 实验 ID
    --health-time 300 \               # 健康检查时长（秒），默认 5 分钟
    --check-interval 10 \             # 检查间隔（秒）
    --skip-post                       # 跳过后处理
```

---

### 🪝 在训练脚本中使用钩子

```python
from training.train_hooks import TrainingHooks

hooks = TrainingHooks("VIT-20251204-xxx", signals_dir="./signals")

# 在 warmup 后标记健康
if step == 100 and loss < 10.0:
    hooks.mark_healthy(step=step, loss=loss)

# 训练结束
hooks.mark_done(metrics={"r2": 0.99, "mae": 0.05})
```

---

### 📡 信号文件

```
signals/
├── {exp_id}.healthy    # 健康检查通过
├── {exp_id}.done       # 训练完成
└── {exp_id}.failed     # 训练失败
```

---

### 📊 后处理输出

训练完成后自动生成：

```
results/{exp_id}/
├── metrics.csv      # 训练指标 CSV
├── summary.json     # 实验摘要 JSON
└── report_draft.md  # exp.md 报告骨架
```

---

### 💡 减少 Cursor Token

```bash
# ❌ 不要把整个日志给 Cursor
cat logs/train.log  # 10000 行...

# ✅ 只给摘要
cat results/xxx/summary.json

# 或者让 Cursor 自己读
"实验结果在 results/xxx/summary.json，帮我分析"
```

---

### 🔄 完整训练流程

```
1. 启动训练
   python driver.py --config config.yaml --exp-id VIT-xxx
   
2. 驱动器自动执行：
   ├─ 启动训练进程
   ├─ 前 5 分钟健康检查
   │   ├─ 通过 → 继续
   │   └─ 失败 → 终止 + 记录原因
   ├─ 等待训练完成
   └─ 自动后处理
       ├─ 提取 metrics.csv
       ├─ 生成 summary.json
       └─ 生成 report_draft.md

3. 给 Cursor 精简信息
   cat results/xxx/summary.json
   
4. 归档到知识中心
   a VIT-xxx
```

---

## 💻 Shell 命令

### 同步脚本

```bash
# 一键同步所有仓库
./scripts/sync_experiments.sh

# 仅预览，不实际执行
./scripts/sync_experiments.sh --dry-run

# 只同步某日期之后的
./scripts/sync_experiments.sh --since "2025-11-28"
```

### 登记脚本

```bash
# 完整参数
python scripts/register_experiment.py \
    --experiment_id "VIT-20251201-cnn-dilated-01" \
    --project VIT \
    --topic cnn \
    --status completed \
    --entry_point "scripts/run.py" \
    --config_path "configs/cnn_dilated.yaml" \
    --output_path "lightning_logs/version_42" \
    --metrics_summary "R2=0.992, MAE=0.028"

# 简化版
python scripts/register_experiment.py \
    -e "VIT-20251201-cnn-01" \
    -p VIT -t cnn -s completed \
    -m "R2=0.992"

# 更新已有实验
python scripts/register_experiment.py \
    -e "VIT-20251201-cnn-01" \
    -p VIT -t cnn \
    -m "R2=0.995" \
    --update
```

### 扫描脚本

```bash
# 扫描 VIT
python scripts/scan_vit_experiments.py --vit-root ~/VIT

# 扫描 BlindSpot
python scripts/scan_blindspot_experiments.py --blindspot-root ~/BlindSpotDenoiser

# 预览模式
python scripts/scan_vit_experiments.py --vit-root ~/VIT --dry-run
```

### 报告生成

```bash
# 周报
python scripts/generate_report.py --type weekly

# 月报
python scripts/generate_report.py --type monthly

# 自定义时间段
python scripts/generate_report.py --type adhoc \
    --start "2025-11-25" --end "2025-12-01"

# 🆕 简短报告（给 LLM 分析）
python scripts/generate_report.py -4h    # 过去 4 小时
python scripts/generate_report.py -24h   # 过去 24 小时
python scripts/generate_report.py -2h    # 过去 2 小时
```

### 便捷 Alias (添加到 ~/.bashrc)

```bash
# Physics_Informed_AI
alias pai='cd ~/Physics_Informed_AI'
alias sync-all='cd ~/Physics_Informed_AI && ./scripts/sync_experiments.sh'
alias report='cd ~/Physics_Informed_AI && python scripts/generate_report.py --type weekly'

# VIT
alias vit='cd ~/VIT'
alias sync-vit='python ~/Physics_Informed_AI/scripts/scan_vit_experiments.py --vit-root ~/VIT'

# BlindSpot
alias bs='cd ~/BlindSpotDenoiser'
alias sync-bs='python ~/Physics_Informed_AI/scripts/scan_blindspot_experiments.py --blindspot-root ~/BlindSpotDenoiser'

# SpecDiffusion
alias sd='cd ~/SpecDiffusion'
alias sync-sd='python ~/Physics_Informed_AI/scripts/scan_specdiffusion_experiments.py --sd-root ~/SpecDiffusion'

# 通用
alias reg='python ~/Physics_Informed_AI/scripts/register_experiment.py'

# 🆕 训练自动化
TRAIN_SCRIPTS="$HOME/Physics_Informed_AI/_backend/scripts/training"
alias train-driver="python $TRAIN_SCRIPTS/driver.py"
alias train-check="python $TRAIN_SCRIPTS/health_check.py"
alias train-post="python $TRAIN_SCRIPTS/post_process.py"

# 训练快捷函数
train() {
    if [ $# -lt 2 ]; then
        echo "用法: train <exp_id> <config.yaml>"
        return 1
    fi
    python "$TRAIN_SCRIPTS/driver.py" --exp-id "$1" --config "$2"
}
```

---

## 🔄 完整工作流

### 日常工作流

```
┌─────────────────────────────────────────────────────────────┐
│  1. 在 VIT/BlindSpot 跑实验                                  │
│     └─ python scripts/run.py --config xxx.yaml              │
│                         ↓                                    │
│  2. 实验完成后登记                                           │
│     └─ reg cnn 实验完成，R2=0.992                           │
│                         ↓                                    │
│  3. 切换到知识中心                                           │
│     └─ cd ~/Physics_Informed_AI                             │
│                         ↓                                    │
│  4. 查看进度                                                 │
│     └─ ?                                                     │
│                         ↓                                    │
│  5. 创建实验文档                                             │
│     └─ n CNN dilation 实验                                  │
│                         ↓                                    │
│  6. 归档详细结果                                             │
│     └─ a                                                     │
│                         ↓                                    │
│  7. 生成周报                                                 │
│     └─ report                                                │
└─────────────────────────────────────────────────────────────┘
```

### 批量同步流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. 在知识中心执行同步                                       │
│     └─ sync                                                  │
│                         ↓                                    │
│  2. 查看同步结果                                             │
│     └─ ?                                                     │
│                         ↓                                    │
│  3. 为重要实验创建文档                                       │
│     └─ n [实验描述]                                         │
│                         ↓                                    │
│  4. 归档详细结果                                             │
│     └─ a                                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 文件位置速查

| 文件 | 路径 |
|------|------|
| 实验索引 | `~/Physics_Informed_AI/experiments_index/index.csv` |
| 待办清单 | `~/Physics_Informed_AI/status/next_steps.md` |
| 归档队列 | `~/Physics_Informed_AI/status/archive_queue.md` |
| 周报草稿 | `~/Physics_Informed_AI/reports/drafts/` |
| 知识文档 | `~/Physics_Informed_AI/logg/[topic]/` |
| exp 模板 | `~/Physics_Informed_AI/template/exp.md` |
| main 模板 | `~/Physics_Informed_AI/template/main.md` |
| 知识卡片模板 | `~/Physics_Informed_AI/template/knowledge_card.md` |

---

## 📝 Experiment ID 格式

| 仓库 | 格式 | 示例 |
|------|------|------|
| VIT | `VIT-YYYYMMDD-topic-XX` | `VIT-20251201-cnn-dilated-01` |
| BlindSpot | `BS-YYYYMMDD-topic-XX` | `BS-20251201-latent-probe-01` |
| SpecDiffusion | `SD-YYYYMMDD-topic-XX` | `SD-20251203-diff-supervised-01` |

### Topic 关键词 & 仓库路由

| topic | 适用场景 | 代码仓库 |
|-------|---------|---------|
| `diff-*` | Diffusion 所有实验 | `~/SpecDiffusion` |
| `cnn` | CNN, dilated, kernel | `~/VIT` |
| `swin` | Swin, Transformer, Attention | `~/VIT` |
| `noise` | 噪声, SNR, 鲁棒性 | `~/VIT` |
| `topk` | Top-K 特征选择 | `~/VIT` |
| `ridge` | Ridge, Linear | `~/VIT` |
| `lightgbm` | LightGBM | `~/VIT` |
| `gta` | Global Tower | `~/VIT` |
| `pca` | PCA 降维 | `~/VIT` |
| `distill` | Latent, Probe, Encoder | `~/BlindSpotDenoiser` |
| `train` | 训练策略 | `~/VIT` |

---

> 💡 **提示**: 将此文件加入书签，随时查阅命令用法

