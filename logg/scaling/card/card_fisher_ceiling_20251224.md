# 🧠 Card F1｜log g 理论上限 R²_max = 0.89 @ SNR=7，临界 SNR ≈ 4

> **结论（可指导决策）**
> 1) SNR > 7 时理论上限 R²=0.89，当前模型仅利用 64%，继续投入深度学习值得
> 2) **SNR ≈ 4 是临界点**：低于此 R²_max 骤降，需要额外策略
> 3) Degeneracy 由物理决定（Schur ≈ 0.69 恒定），与 SNR 无关

---

## 1️⃣ 数学 / 理论依据

* **假设**：线性观测 + 高斯噪声
* **关键结论**：
  - Fisher Information / CRLB 给出无偏估计的最优下界
  - $R^2_{\max} = 1 - \frac{\text{CRLB}}{\text{Var}(\log g)}$
  - Schur complement 量化参数纠缠（degeneracy）
* **含义**：任何模型（包括非线性）在给定噪声下最多可解释 R²_max 的方差

---

## 2️⃣ 实验结果（关键证据）

### A. SNR-R²_max 关系（Multi-Magnitude Sweep）

| SNR | Magnitude | R²_max (median) | 解读 |
|-----|-----------|-----------------|------|
| 87.4 | 18.0 | **0.9994** | 近乎完美 |
| 24.0 | 20.0 | **0.9906** | 优秀 |
| 7.1 | 21.5 | **0.8914** | 优秀（基准） |
| **4.6** | **22.0** | **0.7396** | ⚠️ 临界区域 |
| 3.0 | 22.5 | 0.3658 | 困难 |
| 1.9 | 23.0 | 0.0000 | 信息悬崖 |

### B. 关键数字

| 指标 | 值 | 条件 |
|------|-----|------|
| R²_max (median) | **0.8914** | SNR=7.1, mag=21.5 |
| Gap vs LightGBM | **+32%** | 0.89 vs 0.57 |
| Gap vs Ridge | **+43%** | 0.89 vs 0.46 |
| **临界 SNR** | **≈ 4** | R²_max > 0.5 的阈值 |
| Schur decay | **0.69 ± 0.01** | 恒定，与 SNR 无关 |
| CRLB range | 2.9 orders | 数值稳定 ✅ |

### C. 关键洞见

1. **信息饱和区 (SNR > 20)**：R²_max ≈ 1，瓶颈是模型能力而非信息量
2. **临界 SNR ≈ 4**：mag=22 是有效估计的极限边界
3. **信息悬崖 (SNR < 2)**：50%+ 样本 Fisher 信息趋近 0
4. **Degeneracy 恒定**：Schur ≈ 0.69，参数纠缠由光谱物理决定

### D. V1 失败教训

⚠️ 非规则网格用邻近点差分 → CRLB 跨 20 个数量级，不可靠

---

## 3️⃣ 实验链接

| 来源 | 路径 | 说明 |
|------|------|------|
| exp_1 (V2 基准) | `logg/scaling/exp/exp_scaling_fisher_ceiling_v2_20251224.md` | 单 mag=21.5 |
| **exp_2 (Multi-Mag)** | `logg/scaling/exp/exp_scaling_fisher_multi_mag_20251224.md` | **6 个 magnitude sweep** |
| exp_3 (V1 失败) | `logg/scaling/exp/exp_scaling_fisher_ceiling_20251223.md` | 方法失败 |
| 脚本 | `~/VIT/scripts/scaling_fisher_ceiling_v2.py` | 单 mag |
| 脚本 | `~/VIT/scripts/scaling_fisher_ceiling_v2_multi_mag.py` | 多 mag |
| Fisher Hub | `logg/scaling/fisher_hub_20251225.md` | 专题汇合 |
