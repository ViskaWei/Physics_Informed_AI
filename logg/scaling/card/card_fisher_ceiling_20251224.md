# 🧠 Card F1｜log g 理论上限 R²_max = 0.89，当前模型仅利用 64%

> **结论（可指导决策）**
> 深度学习投入值得：noise=1 时理论上限 R²=0.89，LightGBM 仅达 0.57，存在 +32% 提升空间

---

## 1️⃣ 数学 / 理论依据

* **假设**：线性观测 + 高斯噪声
* **关键结论**：Fisher Information / CRLB 给出无偏估计的最优下界 → R²_max = 0.89 (median)
* **含义**：任何模型（包括非线性）在此噪声下最多可解释 89% 的 log_g 方差

---

## 2️⃣ 实验结果（关键证据）

* **图**：Fisher Ceiling 热力图 (log_g × T_eff)
* **关键结果**：
  - R²_max (median) = **0.8914**（规则网格, noise=1）
  - R²_max (90%) = 0.9804（高分位上限）
  - Gap vs LightGBM = **+32%**（相对 0.57）
  - Gap vs Ridge = **+43%**（相对 0.46）
  - Schur decay = 0.69（69% 信息保留，degeneracy 中等）

* **V1 失败教训**：非规则网格用邻近点差分 → CRLB 跨 20 个数量级，不可靠

---

## 3️⃣ 实验链接

| 来源 | 路径 |
|------|------|
| exp_1 | `logg/scaling/exp/exp_scaling_fisher_ceiling_v2_20251224.md` |
| exp_2 (失败) | `logg/scaling/exp/exp_scaling_fisher_ceiling_20251223.md` |
| 脚本 | `~/VIT/scripts/scaling_fisher_ceiling_v2.py` |
