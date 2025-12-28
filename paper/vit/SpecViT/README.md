# SpecViT Paper - LaTeX Project

## 设计原则

1. **自洽可编译**：`main.tex` + `sections/` + `refs.bib` + `figs/` 完整齐全，可独立编译
2. **禁止 symlink**：`figs/` 中的图必须是复制进来的文件，不能是指向主仓其他目录的软链接（Overleaf Git 系统不兼容）
3. **只放发布版最终图**：`figs/` 只存放 PDF/PNG 格式的最终导出图，不放大数据文件或中间产物
4. **图源与发布分离**：主仓 `assets/figures/specvit/` 存放图源/生成产物，通过 `tools/specvit_export_figs.sh` 复制到 `figs/`
5. **不使用 submodule / LFS**：Overleaf 同步不兼容这些功能

## 目录结构

```
SpecViT/
├── main.tex              # 主文档入口
├── refs.bib              # 参考文献
├── sections/
│   ├── intro.tex         # 引言
│   ├── method.tex        # 方法
│   ├── experiments.tex   # 实验
│   └── conclusion.tex    # 结论
├── figs/                 # 发布版图片（PDF/PNG）
├── .gitignore            # 忽略 LaTeX 临时文件
├── Makefile              # 本地编译命令
└── README.md             # 本文档
```

## 本地编译

```bash
# 使用 latexmk（推荐）
cd paper/vit/SpecViT
latexmk -pdf main.tex

# 或手动编译
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# 清理临时文件
make clean
```

## Git Subtree 工作流

本目录通过 git subtree 同步到独立的论文 GitHub 仓库，供 Overleaf 使用。

### 相关脚本（位于主仓 `tools/`）

| 脚本 | 功能 |
|------|------|
| `specvit_subtree_push.sh` | 将本目录推送到独立论文仓 |
| `specvit_subtree_pull.sh` | 从独立论文仓拉取更新（Overleaf 修改回流） |
| `specvit_export_figs.sh` | 将图源目录的图复制到 `figs/` |

### 详细指南

参见 `docs/overleaf_subtree_guide_specvit.md`

## 配置变量

```bash
PAPER_DIR="paper/vit/SpecViT"
REMOTE_NAME="specvit-paper"
REMOTE_URL="<FILL_ME_GITHUB_URL>"  # 需替换为实际 GitHub 仓库 URL
SUBTREE_BRANCH="main"
```
