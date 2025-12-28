# 🤖 实验 Coding Prompt: Fisher CRLB Residual Overlay

> **日期:** 2025-12-28 | **来源:** `logg/scaling/exp/exp_scaling_fisher_residual_overlay_20251228.md`

---

## ⚠️ 核心规则

| 规则 | 说明 |
|------|------|
| **nohup 后台运行** | 所有训练必须 `nohup ... &`，>5分钟不持续追踪 |
| **跨仓库用终端** | 写入 Physics_Informed_AI 用 `cat/echo/cp`，禁止 IDE 工具 |
| **图片必须入报告** | 所有图表必须在报告 §3 中引用，路径 `logg/scaling/exp/img/` |
| **figsize 统一** | 所有图表 `figsize=(6, 5)`，保持一致性 |
| **语言** | Header 英文 \| 正文中文 \| 图表文字英文 |

---

## 🚀 仓库路由

| Topic | 仓库 | 前缀 |
|-------|------|------|
| **fisher** | `~/VIT` | SCALING- |

---

## 🎯 实验规格

```yaml
experiment_id: "SCALING-20251228-fisher-residual-overlay"
repo_path: "~/VIT"
mvp: "MVP-FU-3"

# 数据来源（复用已有 Fisher 计算结果）
data:
  fisher_results:
    - path: "~/VIT/results/SCALING-20251224-fisher-ceiling-02/fisher_results.csv"
      mag: 21.5
      description: "V2 规则网格 Fisher 结果，包含 crlb_logg_marginalized"
    - path: "~/VIT/results/SCALING-20251224-fisher-multi-mag/"
      mags: [18, 20, 21.5, 22, 22.5, 23]
      description: "Multi-mag Fisher 结果"
  grid_data:
    path: "/datascope/subaru/user/swei20/data/bosz50000/grid/grid_mag215_lowT/dataset.h5"
    fields: [flux, error, T_eff, log_g, M_H]

# 任务：理论分析 + 可视化（无训练）
task: "visualization"
training: null  # 无需训练

# 分箱配置
binning:
  variable: "log_g"
  bin_width: 0.2  # dex
  range: [0.5, 5.5]
  aggregation: "median"  # median 更稳定

# 图表配置
plots:
  - id: "fig_fu3a"
    type: "residual_vs_true_with_envelope"
    title: "Residual vs True log g with Fisher CRLB Envelope"
    xlabel: "True log g (dex)"
    ylabel: "Residual (dex)"
    elements:
      - scatter: "model_residual"
      - hline: "global_std_1sigma"  # 当前全局 ±1σ
      - envelope: "fisher_sigma_binned"  # ±σ_fisher(logg) 包络
    save: "fisher_residual_overlay.png"
    
  - id: "fig_fu3b"
    type: "parity_with_band"
    title: "Parity Plot with Fisher CRLB Band"
    xlabel: "True log g (dex)"
    ylabel: "Predicted log g (dex)"
    elements:
      - scatter: "pred_vs_true"
      - line: "y=x"  # 红色虚线
      - band: "y=x ± σ_fisher(logg)"  # 理论带
    save: "fisher_parity_overlay.png"
    
  - id: "fig_fu3c"
    type: "histogram_with_rmse"
    title: "Residual Histogram with Fisher RMSE Lower Bound"
    xlabel: "Residual (dex)"
    ylabel: "Count"
    elements:
      - histogram: "residual_distribution"
      - vline: "rmse_min"  # sqrt(E[CRLB])
      - vline: "model_rmse"  # 实际模型 RMSE
    save: "fisher_histogram_overlay.png"
```

---

## 🗂️ 参考代码（⚠️ 只写路径，禁止写代码）

> **强制规则**：
> - ❌ 禁止在此写任何代码块、代码骨架、示例代码
> - ✅ Agent 执行时必须先阅读下方路径中的代码，理解逻辑后再修改
> - 💡 这样做确保复用已有代码逻辑，避免不一致

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `~/VIT/scripts/scaling_fisher_ceiling_v2.py` | `compute_crlb_from_fisher()`, `load_grid_data()` | N/A（直接复用结果 CSV） |
| `~/VIT/scripts/scaling_fisher_ceiling_v2_multi_mag.py` | 多 mag 循环逻辑 | N/A |
| `~/VIT/scripts/plot_r2_vs_snr_ceiling_unified_snr_median.py` | 绑图风格、分位带绘制 | 改为 residual overlay |
| `~/VIT/results/SCALING-20251224-fisher-ceiling-02/fisher_results.csv` | 直接加载 | 提取 `log_g`, `crlb_logg_marginalized` |

---

## 📋 执行流程

### Step 1: 创建脚本

创建 `~/VIT/scripts/scaling_fisher_residual_overlay.py`

**输入**：
- Fisher 结果 CSV：`~/VIT/results/SCALING-20251224-fisher-ceiling-02/fisher_results.csv`
- 字段：`log_g`, `crlb_logg_marginalized`

**核心逻辑**：
1. 加载 Fisher 结果 CSV
2. 计算 `sigma_fisher = sqrt(crlb_logg_marginalized)`
3. 按 `log_g` 分箱（0.2 dex），每箱取 `median(sigma_fisher)`
4. 生成三张图（Fig FU3a/b/c）

### Step 2: 执行

```bash
cd ~/VIT && source init.sh
python scripts/scaling_fisher_residual_overlay.py --mag 21.5
```

**预计时间**: <1 min（纯绘图，无训练）

### Step 3: 复制图表到知识库

```bash
# 图表保存到
cp ~/VIT/results/fisher_residual_overlay/*.png \
   /home/swei20/Physics_Informed_AI/logg/scaling/exp/img/
```

### Step 4: 更新报告

```bash
# 用终端命令更新报告
KNOWLEDGE_CENTER="/home/swei20/Physics_Informed_AI"
# 在 exp_scaling_fisher_residual_overlay_20251228.md 中填写观察和结论
```

---

## 📊 图表规格详解

### Fig FU3a: Residual vs True with Fisher Envelope（核心图）

**布局**：
- 底层：散点图（模型 residual = pred - true）
- 中层：水平虚线（当前全局 ±1σ，如 ±0.63）
- 顶层：**Fisher 包络**（两条红色虚线）
  - 上包络：`+median(σ_fisher)` per logg bin
  - 下包络：`-median(σ_fisher)` per logg bin

**标注**：
- Legend: `"Fisher CRLB (marginal) ±1σ lower bound"`
- 口径说明：这不是"模型应该落在里面"，而是"任何方法的 residual 标准差不可能系统性低于这条曲线"

**可选增强**：
- 画两条线：median 和 90% 分位，展示异质性
- 用 `fill_between` 画 95% 理论带（±1.96σ）

---

### Fig FU3b: Parity with Fisher Band

**布局**：
- 散点：pred_logg vs true_logg
- 红色虚线：y = x (Perfect)
- **带状区域**：`y = x ± σ_fisher(logg)`

**视觉效果**：模型点云的厚度 vs Fisher 给的"理论最窄厚度"

---

### Fig FU3c: Histogram with RMSE_min

**布局**：
- 直方图：residual 分布
- 竖线 1（红色）：`RMSE_min = sqrt(mean(crlb_logg_marginalized))`
- 竖线 2（蓝色）：实际模型 RMSE

**标注**：
- `"Fisher RMSE lower bound = {RMSE_min:.3f} dex"`
- `"Model RMSE = {model_rmse:.3f} dex"`

---

## 🔑 关键数值提取

从 `fisher_results.csv` 中提取：

| 指标 | 计算方法 | 用于 |
|------|---------|------|
| `sigma_fisher[i]` | `sqrt(crlb_logg_marginalized[i])` | per-sample 理论误差 |
| `sigma_binned[bin]` | `median(sigma_fisher[logg in bin])` | 分箱后的包络 |
| `RMSE_min` | `sqrt(mean(crlb_logg_marginalized))` | 直方图竖线 |

---

## ✅ 检查清单

- [ ] 脚本创建：`~/VIT/scripts/scaling_fisher_residual_overlay.py`
- [ ] 图表生成（英文标注）：
  - [ ] `fisher_residual_overlay.png`
  - [ ] `fisher_parity_overlay.png`
  - [ ] `fisher_histogram_overlay.png`
- [ ] 图表复制到：`/home/swei20/Physics_Informed_AI/logg/scaling/exp/img/`
- [ ] 报告更新：`exp_scaling_fisher_residual_overlay_20251228.md` §4 观察填写
- [ ] 关键数字记录：`sigma_fisher (median)`, `RMSE_min`, `vs model RMSE`

---

## 🔧 故障排除

| 问题 | 修复 |
|------|------|
| CSV 加载失败 | 检查路径：`~/VIT/results/SCALING-20251224-fisher-ceiling-02/fisher_results.csv` |
| σ_fisher 为 NaN | 原始 CRLB 有问题，过滤 `dropna()` |
| 包络线不平滑 | 增加分箱数量或用样条插值 |
| 图例重叠 | 调整 `legend(loc='upper right')` |

---

## 📎 输出文件索引

| 类型 | 路径 |
|------|------|
| 脚本 | `~/VIT/scripts/scaling_fisher_residual_overlay.py` |
| 结果目录 | `~/VIT/results/fisher_residual_overlay/` |
| 图表（VIT） | `~/VIT/results/fisher_residual_overlay/*.png` |
| 图表（知识库） | `logg/scaling/exp/img/fisher_*.png` |
| 报告 | `logg/scaling/exp/exp_scaling_fisher_residual_overlay_20251228.md` |
| Hub 同步 | `logg/scaling/fisher_hub_20251225.md` § Q7 |
| Roadmap 同步 | `logg/scaling/fisher_roadmap_20251225.md` § MVP-FU-3 |
