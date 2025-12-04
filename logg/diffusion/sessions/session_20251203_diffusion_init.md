# 💬 GPT 设计会话归档 - Diffusion 1D 恒星光谱立项

---
> **会话 ID：** `session-20251203-diffusion-init`  
> **日期：** 2025-12-03  
> **参与：** Viska + GPT-Pro  
> **关联 main：** [logg/diffusion/diffusion_main_20251203.md](../diffusion_main_20251203.md)  
> **状态：** ✅ 已归档到 kanban

---

# 📑 目录

- [1. 🎯 起点：问题 & 动机](#1--起点问题--动机)
- [2. 💡 MVP 候选列表](#2--mvp-候选列表)
- [3. ✅ 选择要执行的实验](#3--选择要执行的实验)
- [4. 📝 Prompt 草稿区](#4--prompt-草稿区)
- [5. 📋 会话后 TODO](#5--会话后-todo)

---

# 1. 🎯 起点：问题 & 动机

## 1.1 原始动机

> **为什么今天在想这个问题？**

- **背景**：在做恒星光谱参数估计，低 SNR 谱质量差，想用生成式方法降噪
- **触发点**：看到 RAA 2025 spec-DDPM 的工作，说明 diffusion 做 1D 光谱降噪已有先例
- **核心关切**：diffusion 降噪会不会"幻觉式补线"，引入系统性参数偏差？

## 1.2 GPT 对话中的关键摘录

### 💎 关键洞见 1：两条降噪路线

> "路线 A：监督式条件 diffusion（直接学 p(x_clean | y_noisy)）... 
> 路线 B：把 diffusion 当作"干净谱先验"，做后验采样（p(x_clean | y_obs)）"

**启发**：不是只有监督式一种做法，DPS 后验采样可能更能抑制幻觉

### 💎 关键洞见 2：三层参数推断方案

> "方案 1（最稳）：先采样干净谱，再做参数后验
> 方案 2（更快）：直接训练 p(θ|y) 的扩散/score 模型
> 方案 3（最物理一致也最难）：训练条件 diffusion 做前向生成 p(x|θ)"

**启发**：风险-收益权衡，方案 1 最不容易出科学事故

### 💎 关键洞见 3：评价指标

> "评价指标不能只看 MSE/SSIM... 应包含关键谱线窗的等效宽度/线深偏差、RV 偏差、以及最重要的参数偏差 vs SNR 曲线 + 不确定性覆盖率"

**启发**：必须建立谱线级别的物理评价，不能只看全局 loss

### 💎 关键洞见 4：异方差噪声

> "巡天谱典型是异方差（每像素 ivar 不同）+ 坏像素 + 天空线残差。强烈建议把 per-pixel σ(λ) / ivar、mask、分辨率/LSF 参数作为 conditioning 输入"

**启发**：必须条件化 noise map，否则模型会用先验补全掩盖真实不确定性

### ⚠️ 需要警惕的假设

> "diffusion 的近似似然/ELBO 做推断可能出现校准偏差"

**风险**：后验不确定性可能严重失准，需要覆盖率测试

---

# 2. 💡 MVP 候选列表

> 从对话中结构化出的实验候选

| MVP ID | 一句话目标 | 要验证的假设/问题 | 粗略实验想法 | 预估复杂度 |
|--------|------------|-------------------|--------------|-----------|
| MVP-0.0 | 1D U-Net DDPM 基线 | 能在光谱上训起来 | 标准 DDPM 训练 | 🟢 简单 |
| MVP-1.0 | 监督式 DDPM 降噪 | H1.1 视觉改善 vs 参数偏置 | spec-DDPM 复现 | 🟡 中等 |
| MVP-1.1 | DPS 后验采样降噪 | H1.2 likelihood 约束抗幻觉 | 先验 + DPS guidance | 🟡 中等 |
| MVP-1.2 | +ivar 条件化 | H1.4 异方差场景 | 2-channel input | 🟢 简单 |
| MVP-2.0 | 采样谱 → 参数后验 | H1.3 覆盖率校准 | 样本传播 | 🟢 简单 |
| MVP-2.1 | 摊销后验 p(θ\|y) | 快速推理 | 条件 score | 🔴 复杂 |
| MVP-3.0 | 谱线级评价 | 科学可信度 | EW/RV 偏置分析 | 🟡 中等 |
| MVP-3.1 | 覆盖率测试 | 不确定性校准 | PIT/CI 覆盖 | 🟢 简单 |

### MVP 详细说明

#### MVP-1.0：监督式 DDPM (spec-DDPM 复现)

**目标**：复现 RAA 2025 spec-DDPM 思路，用成对低/高 SNR 谱训练

**假设**：
- H1: MSE/SSIM 会优于传统方法（PCA, DnCNN）
- H2: 但参数偏置可能增加（向训练集均值回归）

**实验设计草案**：
- 数据：LAMOST DR10 成对谱 (~20K)
- 模型：条件 1D U-Net，y_noisy 作为 conditioning
- 指标：MSE, SSIM-1D, 后续参数 bias
- 成功标准：MSE < DnCNN baseline

**预期结果**：

| 情况 | 预期 | 说明 |
|------|------|------|
| 正常 | MSE 下降，bias 小幅增加 | 继续 MVP-1.1 对比 |
| 异常 A | MSE 下降，bias 大幅增加 | 警告：视觉 ≠ 科学 |
| 异常 B | MSE 不下降 | 排查网络/数据问题 |

---

#### MVP-1.1：DPS 后验采样

**目标**：用 DPS 思路，训练无条件先验，推理时加 likelihood 约束

**假设**：
- H1: 显式 likelihood 约束能抑制幻觉
- H2: 偏置应该小于监督式 DDPM

**实验设计草案**：
- 数据：只需高 SNR 谱训练先验
- 模型：无条件 1D U-Net + DPS guidance
- Forward Operator：A(x) = x（或含 LSF/mask）
- 噪声模型：异方差高斯，σ(λ) from ivar
- 成功标准：偏置 < MVP-1.0

---

#### MVP-2.0：采样谱 → 参数后验

**目标**：最稳的参数推断方案，不确定性端到端传播

**方法**：
1. 用 MVP-1.1 得到 x^(s) ~ p(x|y)，采样 N=100 个谱
2. 对每个样本跑参数估计器（Ridge/MLP/The Payne）得到 θ^(s)
3. 样本集合即为 θ 后验近似

**验收标准**：68% CI 实际覆盖率在 60-75%

---

# 3. ✅ 选择要执行的实验

## 3.1 本轮决策

| 决策 | MVP | 理由 |
|------|-----|------|
| ✅ **立刻排队** | MVP-0.0, MVP-1.0 | 基础验证，确认能跑 |
| 🕒 **以后再说** | MVP-1.1, MVP-1.2, MVP-2.0 | 等 MVP-1.0 结果后决定 |
| 🕒 **以后再说** | MVP-2.1 | 复杂度高，非必需 |
| 🕒 **以后再说** | MVP-3.0, MVP-3.1 | 评价阶段，等降噪完成 |

## 3.2 生成 experiment_id

> 将选中的 MVP 转为具体的 experiment_id，复制到 `status/kanban.md`

| experiment_id | MVP | 优先级 | 预估时间 | 备注 |
|---------------|-----|--------|---------|------|
| `VIT-20251203-diff-baseline-01` | MVP-0.0 | 🔴 P0 | ~3h | 1D U-Net DDPM sanity check |
| `VIT-20251203-diff-supervised-01` | MVP-1.0 | 🔴 P0 | ~4h | spec-DDPM 复现 |

---

# 4. 📝 Prompt 草稿区

## 4.1 通用 System Prompt 片段

```text
你是我的实验助理，负责根据给定的 MVP 说明，在 VIT 仓库中实现 1D diffusion 模型。

你要做的事包括：
1) 实现 1D U-Net 结构（参考 lucidrains/denoising-diffusion-pytorch）
2) 适配恒星光谱数据（长度 ~3000，波长范围）
3) 输出清晰的训练日志和生成样本

仓库路径：~/VIT
数据路径：参考现有 LAMOST 数据加载器
```

## 4.2 针对具体 experiment_id 的 Prompt

### Prompt for `VIT-20251203-diff-baseline-01`

```text
实验 ID: VIT-20251203-diff-baseline-01
MVP: MVP-0.0 1D U-Net DDPM Baseline
仓库: ~/VIT

请：
1. 创建 models/diffusion/unet_1d.py，实现 1D U-Net：
   - 4 blocks, channels: 64 → 128 → 256 → 512
   - 时间嵌入 (sinusoidal)
   - 跳连 (skip connections)
   - 最终输出维度与输入相同

2. 创建 models/diffusion/ddpm.py，实现标准 DDPM：
   - T=1000 步
   - ε-prediction (预测噪声)
   - 线性 β schedule

3. 创建 scripts/train_diffusion_baseline.py：
   - 加载 LAMOST 高 SNR 谱（参考现有数据加载）
   - 训练 50 epochs
   - 每 10 epochs 采样 16 个样本可视化

4. 训练完成后报告：
   - 生成样本的可视化（是否像光谱）
   - 训练 loss 曲线
   - lightning_logs 路径
```

### Prompt for `VIT-20251203-diff-supervised-01`

```text
实验 ID: VIT-20251203-diff-supervised-01
MVP: MVP-1.0 监督式条件 DDPM
仓库: ~/VIT
依赖: VIT-20251203-diff-baseline-01 的 1D U-Net

请：
1. 修改 unet_1d.py 支持条件输入：
   - 额外 channel 接收 y_noisy
   - 或者用 cross-attention

2. 创建数据加载：成对低/高 SNR 谱
   - 选项 A：同目标多次观测
   - 选项 B：人工加噪（高斯，异方差）

3. 训练条件 DDPM：
   - 输入：y_noisy
   - 目标：x_clean
   - 50 epochs

4. 评估：
   - MSE vs DnCNN/PCA baseline
   - 随机选 100 个测试谱，跑参数估计（Ridge），报告 bias

5. 报告：
   - 降噪前后对比图（3 个样本）
   - MSE 改善幅度
   - 参数 bias（Teff, logg, [Fe/H]）
```

---

# 5. 📋 会话后 TODO

## 5.1 立即执行

- [x] 把 §3.2 的 experiment_id 填入 `status/kanban.md` 的「TODO - 待跑实验」
- [x] 创建 `diffusion_main_20251203.md`（MVP 路线图）
- [ ] 在 `status/next_steps.md` 设置 P0 实验

## 5.2 实验完成后

- [ ] 从 kanban 的「TODO」移到「Running」，补上运行路径
- [ ] 实验结束后移到「Done」，准备写 exp.md

## 5.3 归档检查

| 检查项 | 状态 |
|--------|------|
| experiment_id 已填入 kanban | ✅ |
| main.md 已创建 | ✅ |
| next_steps.md 已设置优先级 | ⬜ |

---

# 📎 会话元数据

| 字段 | 值 |
|------|-----|
| **GPT 模型** | GPT-Pro |
| **对话轮数** | ~5 轮 |
| **会话主题标签** | `#diffusion` `#denoising` `#posterior` |

---

*归档时间: 2025-12-03*

