# Overleaf + Paper Sync 工作流指南 (SpecViT)

本指南描述如何将 `paper/vit/SpecViT/` 同步到独立的 GitHub 仓库，并与 Overleaf 协作。

> **注意**: 由于系统未安装 `git subtree`，我们使用基于 clone/copy 的替代方案，功能等效。

## 📋 配置变量

```bash
# 主仓中的论文目录
PAPER_DIR="paper/vit/SpecViT"

# 独立论文仓的 remote 名称
REMOTE_NAME="specvit-paper"

# 独立论文仓的 GitHub URL（需替换）
REMOTE_URL="<FILL_ME_GITHUB_URL>"

# 分支名
BRANCH="main"
```

---

## 🚀 一次性初始化

### Step 1: 在 GitHub 创建空仓库

1. 打开 [GitHub New Repository](https://github.com/new)
2. Repository name: `physics_informed_ai-specvit-paper`（建议命名）
3. **不要** 添加 README、.gitignore 或 License（必须是空仓库）
4. Create repository
5. 复制仓库 URL，例如：`https://github.com/YourUsername/physics_informed_ai-specvit-paper.git`

### Step 2: 在主仓添加 remote

```bash
cd ~/Physics_Informed_AI

# 添加 remote（替换 URL）
git remote add specvit-paper https://github.com/YourUsername/physics_informed_ai-specvit-paper.git

# 验证
git remote -v
```

### Step 3: 首次推送

```bash
# 确保所有更改已提交
git add -A
git commit -m "Prepare SpecViT paper for subtree push"

# 使用推送脚本（会自动处理）
./tools/specvit_subtree_push.sh
```

### Step 4: 更新脚本配置

编辑以下脚本，将 `<FILL_ME_GITHUB_URL>` 替换为实际 URL：
- `tools/specvit_subtree_push.sh`
- `tools/specvit_subtree_pull.sh`

---

## 🔗 Overleaf 配置

### 导入项目到 Overleaf

1. 打开 [Overleaf](https://www.overleaf.com)
2. New Project → **Import from GitHub**
3. 授权 Overleaf 访问你的 GitHub（首次需要）
4. 选择 `physics_informed_ai-specvit-paper` 仓库
5. 等待导入完成

### ⚠️ 重要提醒

- **Overleaf GitHub sync 不是自动的！**
- 需要手动在 Overleaf 中点击 Menu → GitHub → Pull/Push
- Pull = 从 GitHub 拉取更新到 Overleaf
- Push = 从 Overleaf 推送更改到 GitHub

---

## 📝 日常工作流

### 推荐流程：主仓为真源

这是推荐的工作流，保持主仓 (`Physics_Informed_AI`) 作为唯一真源。

```
┌─────────────────┐     subtree push     ┌─────────────────┐
│   Main Repo     │ ─────────────────→   │   Paper Repo    │
│ Physics_AI/     │                      │ (GitHub)        │
│ paper/vit/      │     subtree pull     │                 │
│   SpecViT/      │ ←───────────────── │                 │
└─────────────────┘                      └────────┬────────┘
                                                  │
                                         GitHub Sync (manual)
                                                  │
                                                  ▼
                                         ┌─────────────────┐
                                         │    Overleaf     │
                                         └─────────────────┘
```

#### 本地修改 → Overleaf

```bash
# 1. 在主仓编辑论文
cd ~/Physics_Informed_AI
vim paper/vit/SpecViT/sections/intro.tex

# 2. 提交更改
git add paper/vit/SpecViT/
git commit -m "Update introduction section"

# 3. 推送到独立论文仓
./tools/specvit_subtree_push.sh

# 4. 在 Overleaf 中：Menu → GitHub → Pull
```

#### Overleaf 修改 → 本地

```bash
# 1. 在 Overleaf 中：Menu → GitHub → Push
#    （将 Overleaf 更改推送到 GitHub）

# 2. 在主仓拉取更改
cd ~/Physics_Informed_AI
./tools/specvit_subtree_pull.sh

# 3. 推送到主仓远程（可选）
git push origin main
```

### 导出图片流程

```bash
# 1. 将生成的图放到图源目录
cp your_figure.pdf assets/figures/specvit/

# 2. 运行导出脚本（复制到论文目录）
./tools/specvit_export_figs.sh

# 3. 提交并推送
git add paper/vit/SpecViT/figs/
git commit -m "Add new figures"
./tools/specvit_subtree_push.sh

# 4. Overleaf: Menu → GitHub → Pull
```

---

## ⚠️ 冲突处理

### 场景：Overleaf Push 失败

如果 Overleaf 和 GitHub 有冲突，Overleaf 可能会：
1. 拒绝 Push
2. 或创建一个新分支（如 `overleaf-YYYY-MM-DD-XXXX`）

### 解决方案

```bash
# 1. 在 GitHub 网页上查看新分支
# 2. 创建 Pull Request: overleaf-* → main
# 3. 解决冲突并 Merge

# 4. 在主仓拉取更新
cd ~/Physics_Informed_AI
./tools/specvit_subtree_pull.sh

# 5. 推送到主仓
git push origin main
```

### 预防冲突的最佳实践

1. **单一编辑源**：尽量在一个地方编辑（本地或 Overleaf），避免同时编辑
2. **频繁同步**：每次编辑前后都同步
3. **小批量提交**：避免大量积压的更改

---

## 📁 文件结构

```
Physics_Informed_AI/                 # 主仓
├── paper/vit/SpecViT/              # 论文 LaTeX 工程 (PAPER_DIR)
│   ├── main.tex
│   ├── refs.bib
│   ├── sections/
│   ├── figs/                       # 发布版图（从 assets 复制）
│   ├── .gitignore
│   ├── Makefile
│   └── README.md
├── assets/figures/specvit/         # 图源目录
├── tools/
│   ├── specvit_subtree_push.sh
│   ├── specvit_subtree_pull.sh
│   └── specvit_export_figs.sh
└── docs/
    └── overleaf_subtree_guide_specvit.md  # 本文档

physics_informed_ai-specvit-paper/   # 独立论文仓 (GitHub)
├── main.tex                         # = PAPER_DIR 的内容
├── refs.bib
├── sections/
├── figs/
└── ...
```

---

## 🔧 常用命令速查

| 操作 | 命令 |
|------|------|
| 添加论文仓 remote | `git remote add specvit-paper <URL>` |
| 推送到论文仓 | `./tools/specvit_subtree_push.sh` |
| 从论文仓拉取 | `./tools/specvit_subtree_pull.sh` |
| 导出图片 | `./tools/specvit_export_figs.sh` |
| 本地编译 | `cd paper/vit/SpecViT && make` |
| 查看 remote | `git remote -v` |

---

## ❓ FAQ

### Q: 为什么不用 git submodule？
A: Overleaf 的 GitHub sync 不支持 submodule。Subtree 将代码直接嵌入主仓，Overleaf 可以正常访问。

### Q: 为什么不用 Git LFS？
A: Overleaf 不支持 Git LFS。大文件应保留在主仓的 `assets/` 目录，只将小的发布版图片复制到论文目录。

### Q: 可以在 Overleaf 上添加新文件吗？
A: 可以，但需要通过 `subtree pull` 同步回主仓。

### Q: subtree push 很慢怎么办？
A: 首次 push 可能较慢（遍历历史）。之后会快很多。也可以使用 `--squash` 选项简化历史。

---

## 📚 参考资料

- [Git Subtree 官方文档](https://git-scm.com/book/en/v2/Git-Tools-Advanced-Merging#_subtree_merge)
- [Overleaf GitHub Integration](https://www.overleaf.com/learn/how-to/GitHub_Synchronization)
- [Atlassian Git Subtree Tutorial](https://www.atlassian.com/git/tutorials/git-subtree)
