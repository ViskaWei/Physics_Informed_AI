你在本地仓库 /home/swei20/Physics_Informed_AI 中工作（对应 GitHub: https://github.com/ViskaWei/Physics_Informed_AI）。
目标：把 ViT 论文的 LaTeX 工程作为主仓内一个普通目录维护，路径固定为：
  paper/vit/SpecViT/
并建立基于 git subtree 的发布/回流工作流，使该子目录可同步到一个“独立论文 GitHub 仓”（供 Overleaf 的 Import from GitHub 同步使用）。

重要：你只修改主仓本地文件；不要真的创建远程仓。远程仓名/URL用占位符即可。

全局变量（请在文档和脚本中统一使用）：
- PAPER_DIR = paper/vit/SpecViT
- REMOTE_NAME 默认 specvit-paper
- REMOTE_URL 占位符 <FILL_ME_GITHUB_URL>
- SUBTREE_BRANCH 默认 main

(0) 设计原则（必须写入 paper/vit/SpecViT/README.md）：
- PAPER_DIR 必须“自洽可编译”：main.tex + sections/ + refs.bib + figs/ 等齐全
- 禁止使用 symlink 指向主仓其他目录的图（避免 Overleaf Git 系统问题）
- figs/ 只放“论文发布版最终导出图”（pdf/png），不要放大数据/中间产物
- 主仓可以有图源与生成脚本，但进入 PAPER_DIR/figs/ 的必须是复制出来的发布版
- 不使用 git submodule / Git LFS（Overleaf 同步不兼容）

(1) 在 PAPER_DIR 生成 LaTeX 工程骨架（若已存在文件则尽量保留原内容，只补缺/增强，不要粗暴覆盖）：
- PAPER_DIR/main.tex
  - \input{sections/intro.tex}
  - \input{sections/method.tex}
  - \input{sections/experiments.tex}
  - \input{sections/conclusion.tex}
- PAPER_DIR/refs.bib（写 1-2 条占位 bibtex）
- PAPER_DIR/sections/intro.tex, method.tex, experiments.tex, conclusion.tex（每个包含 TODO）
- PAPER_DIR/figs/（创建目录；放一个 placeholder 文件并在 README 说明替换）
- PAPER_DIR/.gitignore（忽略 latex 临时文件：*.aux, *.log, *.out, *.bbl, *.blg, *.synctex.gz, *.fls, *.fdb_latexmk 等）
- PAPER_DIR/Makefile 或 latexmkrc（可选，至少给出一个“本地编译建议命令”）

(2) 添加 subtree 工作流脚本到主仓 tools/（脚本要可直接运行、输出清晰提示、失败时给下一步建议）：
- tools/specvit_subtree_push.sh
- tools/specvit_subtree_pull.sh

脚本要求：
- 内置配置区：
  PAPER_DIR="paper/vit/SpecViT"
  REMOTE_NAME="specvit-paper"
  REMOTE_URL="<FILL_ME_GITHUB_URL>"
  BRANCH="main"
- push 脚本：
  1) 检查 PAPER_DIR 存在，否则退出并提示
  2) 检查 remote 是否存在；若不存在则提示用户执行：
     git remote add ${REMOTE_NAME} ${REMOTE_URL}
  3) 执行：
     git subtree push --prefix ${PAPER_DIR} ${REMOTE_NAME} ${BRANCH}
- pull 脚本（用于把 Overleaf/GitHub 独立论文仓的更新回流到主仓）：
  执行：
     git subtree pull --prefix ${PAPER_DIR} ${REMOTE_NAME} ${BRANCH} --squash
- 两个脚本都要打印：当前目录、prefix、remote、branch、执行的命令、以及成功后建议（比如下一步去 Overleaf Pull）

(3) 添加一个“发布版图拷贝”脚本（建议）：
- tools/specvit_export_figs.sh
功能：
- 图源目录默认：
  SRC_DIR="assets/figures/specvit/"
  若不存在则创建并写一个 README 提示这里放“图源/生成产物”
- 目标目录：
  DST_DIR="${PAPER_DIR}/figs/"
- 行为：
  - 复制 SRC_DIR 下的 .pdf/.png/.jpg 到 DST_DIR（覆盖同名）
  - 输出复制清单（复制了哪些文件）
  - 强提醒：不要用 symlink；Overleaf 不友好

(4) 写一份简明指南文档（新建即可）：
- docs/overleaf_subtree_guide_specvit.md
内容必须包含：
- 一次性初始化：
  1) GitHub 建空仓（占位名：physics_informed_ai-specvit-paper）
  2) 在主仓添加 remote（REMOTE_NAME/REMOTE_URL）
  3) 首次 subtree push（prefix=paper/vit/SpecViT）
- Overleaf 流程：
  New Project → Import from GitHub → 选独立论文仓
  强调：Overleaf GitHub sync 不是自动的，需要手动 Pull/Push
- 日常推荐工作流：
  - 主仓为真源：本地改 PAPER_DIR → commit → subtree push → Overleaf Pull
  - Overleaf 改动回流：Overleaf Push → 主仓 subtree pull --squash
- 冲突处理：
  Overleaf 如果冲突可能会推一个新分支；需要在 GitHub 上 merge 回 main，再 pull 回主仓

(5) 最终输出要求：
- 给出你新增/修改后的文件树（只列相关部分即可）
- 给出每个新增文件的完整内容
- 给出用户需要手动执行的命令清单（含占位符 REMOTE_URL）
- 所有路径都用相对主仓路径（不要写 /home/... 进脚本内容）
