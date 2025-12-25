# 🤖 实验 Coding Prompt

---
> **日期:** YYYY-MM-DD | **来源:** `logg/[topic]/sessions/session_*.md`
---

## 🚨 跨仓库写入规则

> 写入 `/home/swei20/Physics_Informed_AI/` 必须用**终端命令**！
> - ❌ 禁止 IDE 工具 (write/search_replace)
> - ✅ 用 `cat << 'EOF' >`、`echo >>`、`cp`

```bash
KNOWLEDGE_CENTER="/home/swei20/Physics_Informed_AI"
cat << 'EOF' > "$KNOWLEDGE_CENTER/logg/[topic]/exp_xxx.md"
[内容]
EOF
```

---

## 🚀 仓库路由

| Topic | 仓库 | 前缀 |
|-------|------|------|
| diffusion | `~/SpecDiffusion` | SD- |
| cnn/swin/ridge/pca/gta/moe | `~/VIT` | VIT- |
| distill/latent/probe | `~/BlindSpotDenoiser` | BS- |

**驱动器**: `Physics_Informed_AI/_backend/scripts/training/driver.py`

---

# 📋 Prompt 正文

```text
你是实验执行助理。

🚨 跨仓库写入: 用终端命令，禁止 IDE 工具
📝 语言: Header 全英文 | 正文中文 | 图表文字全英文

═══════════════════════════════════════
执行流程
═══════════════════════════════════════

【Step 1】启动训练
```bash
cd [repo]
source init.sh
python .../driver.py --cmd "[训练命令]" --exp-id [exp_id]
# 或
python .../driver.py --config xxx.yaml --exp-id [exp_id]
```

健康检查失败？根据修复建议调整后重试。

【Step 2】生成图表（⚠️ 文字全英文！）
```bash
python plot.py --exp_id [exp_id] --output .../logg/[topic]/img/
```

【Step 3】写报告（用终端命令！）
```bash
cat << 'EOF' > "$KNOWLEDGE_CENTER/logg/[topic]/exp_[name]_YYYYMMDD.md"
# 🍃 [实验名称]
> **Name:** [Name]  
> **ID:** \`[exp_id]\`  
> **Topic:** \`[topic]\` | **MVP:** MVP-X.X | **Project:** \`VIT\`  
> **Author:** Viska Wei | **Date:** YYYY-MM-DD | **Status:** ✅  
> **Root:** \`[Root]\` | **Parent:** \`[Branch]\` | **Child**: |

> 🎯 **Target:** [一句话实验目的]  
> 🦾 **Decide:** [影响的决策]

---
## ⚡ 核心结论速览
> **一句话**: [最重要发现 + 关键数字]

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| Q? | ✅/❌ | [简短] |

---
## 1. 🎯 目标
[中文描述]

## 2. 🧪 实验设计
| 项 | 值 |
|----|-----|
| 数据 | [来源/路径/train-val-test] |
| 噪声 | σ=[值] |
| 模型 | [类型+参数] |
| 训练 | epochs/batch/lr/optimizer/seed |

## 3. 📊 图表
![](./img/[exp_id]_xxx.png)
**观察**: [观察]

## 4. 💡 洞见
- [发现]

## 5. 📝 结论
[核心发现 + 设计启示]

## 6. 📎 附录
[数值结果 + 执行日志]
EOF
```

【Step 4】更新追踪文件
```bash
echo "- [x] [exp_id]: [结论]" >> "$KNOWLEDGE_CENTER/status/kanban.md"
```

═══════════════════════════════════════
驱动器参数
═══════════════════════════════════════

| 参数 | 说明 | 默认 |
|------|------|------|
| --cmd | 训练命令 | 必需 |
| --exp-id | 实验 ID | 必需 |
| --health-time | 健康检查(秒) | 300 |

═══════════════════════════════════════
交付物
═══════════════════════════════════════

| 类型 | 路径 |
|------|------|
| 报告 | `logg/[topic]/exp_[name]_YYYYMMDD.md` |
| 图表 | `logg/[topic]/img/` |

🚨 完成后更新: kanban.md, roadmap.md §2.1, hub.md §3
```

---

# 🗂️ 参考代码

> 不写代码骨架，只列参考脚本路径

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `[路径]` | `func()` | [说明] |

---

# 🎯 实验规格

```yaml
experiment_id: "[PROJECT]-[YYYYMMDD]-[topic]-[##]"
repo_path: "~/VIT"

data:
  source: ""
  path: ""
  train/val/test: N/N/N
  feature_dim: N
  target: "log_g"

noise:
  type: "gaussian"
  sigma: 0.1
  apply_to: "train"

model:
  type: ""

training:
  epochs: N
  batch_size: N
  lr: 1e-4
  optimizer: "Adam"
  seed: 42

plots:
  - type: loss_curve
    save: "[exp_id]_loss.png"
```

---

# ✅ 成功标准

| 检查项 | ⬜ |
|--------|---|
| 训练完成 | |
| 图表(英文) | |
| 报告(中文) | |
| kanban更新 | |

---

# 🔧 故障排除

| 问题 | 修复 |
|------|------|
| NaN | 降 lr / grad_clip |
| OOM | 减 batch_size |
| Loss爆炸 | 降 lr / warmup |
