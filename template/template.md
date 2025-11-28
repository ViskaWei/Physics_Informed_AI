You are an AI Coding & Research Assistant.  
Your task is:

**Given the input experiment notes, tables (.pkl/.csv), logs, and summaries,  
fill in the following Markdown experiment template in a clean, structured, professional form.**

---

## Requirements

### Basic Format
- The final output **must be a single Markdown file**
- Keep all section headings, fill content logically from provided material
- Use clear, concise, professional academic writing style
- DO NOT invent nonexistent numbers; only use what appears in the input
- If information is missing, leave a clear `TODO` placeholder

### Mathematical Formulas (LaTeX)
**You MUST use correct LaTeX syntax to ensure formulas render properly:**

1. **Inline formulas**: Use single `$` delimiters
   - ✅ Correct: `$R^2 = 0.99$`, `$\alpha = 0.01$`
   - ❌ Wrong: `R² = 0.99` (using superscript characters directly)

2. **Block formulas**: Use `$$...$$` or `\[...\]`
   ```
   $$
   \hat{y} = X w + b
   $$
   ```

3. **Common symbol conventions**:
   | Symbol | LaTeX syntax | Rendered |
   |--------|-------------|----------|
   | R² | `$R^2$` | $R^2$ |
   | α | `$\alpha$` | $\alpha$ |
   | σ | `$\sigma$` | $\sigma$ |
   | ≥ | `$\geq$` | $\geq$ |
   | ≈ | `$\approx$` | $\approx$ |
   | subscript | `$x_i$` | $x_i$ |
   | fraction | `$\frac{a}{b}$` | $\frac{a}{b}$ |
   | summation | `$\sum_{i=1}^n$` | $\sum_{i=1}^n$ |
   | transpose | `$X^\top$` | $X^\top$ |

4. **Special notes**:
   - Underscores outside formulas become italics; variable names like `log_g` should be `$\log g$` or `log\_g`
   - Add blank lines before and after formula blocks for proper rendering

### Tables (Markdown)
Use standard Markdown table syntax:
```markdown
| Setting | R² | MAE | RMSE | Notes |
|---------|-----|------|-------|-------|
| PCA=5 | 0.95 | 0.05 | 0.07 | baseline |
```

### Content Guidelines
- **Key Insights**: Summarize hierarchically (high-level → model-level → experiment details)
- **Plot Suggestions**: Describe what to plot and why, not code
- **Conclusion**: Directly answer what the experiment proves and implications for design

---

Now generate the filled template below.

---------------------------------------------
# 📘 Universal Experiment Report Template
---------------------------------------------

---
> **Experiment Name:** TODO  
> **Author:** Viska Wei  
> **Date:** TODO  
> **Data Version:** TODO  
> **Model Version:** TODO

---

# 1. Objectives (目标)

## 1.1 Background & Motivation (背景与动机)
> Describe the *highest-level scientific or engineering goal*, typically:
- Understanding spectral physics  
- Guiding neural-network architecture design  
- Identifying linear vs nonlinear components  
- Evaluating information content of wavelength regions  

Example:
- Provide physical + statistical understanding to design an optimal neural architecture for predicting stellar parameters.

## 1.2 Hypothesis (核心假设)
> Describe the core hypothesis this experiment tests.

Typical examples:
- "log g is controlled by a low-dimensional linear direction in flux space"
- "Error(σ) contains exploitable information about stellar parameters"
- "Most of NN capacity is used for filtering irrelevant information"

## 1.3 Questions to Verify (验证问题)
> Describe the specific questions this experiment answers.

Examples:
- "Does PCA(10) achieve R² ≥ 0.98?"
- "Does Error(σ) as input improve R²?"
- "How does optimal α change with noise level?"

---

# 2. Experimental Design (实验设计)
Provide detailed but concise method:

### 2.1 Data
- Samples:  
- Dimensions:  
- Parameters:  
- Noise Model:  
  \[
  \text{noisy\_flux} = \text{flux} + \mathcal{N}(0, \sigma^2)
  \]

### 2.2 Features Used
- flux / error / concatenated  
- PCA components  
- Top-K selected wavelengths  
- etc.

### 2.3 Model / Algorithm
Include equations if linear:

#### If Linear Regression:
\[
\hat{y} = X w + b
\]
\[
w = (X^\top X + \alpha I)^{-1} X^\top y
\]

#### If LightGBM:
- (Explain depth, leaves, feature fraction…)

### 2.4 Sweep / Hyperparameters
- alpha = …  
- noise levels = …  
- PCA dims = …

---

# 3. Results Table (核心结果表)

> Include all important metrics: R², MAE, RMSE, coverage, etc.

Example template:

| Setting | R² | MAE | RMSE | Notes |
|--------|----|-----|------|-------|
| PCA=5  |    |     |      |       |
| PCA=10 |    |     |      |       |
| Full   |    |     |      |       |

(Agent should fill using provided results.)

---

# 4. Key Insights (最重要的发现)

Write bullet points:

### 4.1 High-Level Insight (指导 Neural Network 的物理规律)
- e.g., “log g dominated by 1D linear direction in feature space”
- e.g., “Error(σ) provides meaningful robustness improvement”

### 4.2 Model-Level Insight (指导模型设计)
- e.g., PCA(10) → R²=0.99 means network only needs <10 effective DoF  
- e.g., Ridge α too large introduces spectral oversmoothing  

### 4.3 Micro-Level Findings (此实验细节)
- e.g., “Most discriminative wavelengths cluster around CaT and Balmer wings”
- e.g., “Top-50 pixels fail under noise=1.0 → structure required”  

---

# 5. Recommended Figures (建议绘图)
Write down *what plots to make*, not the code:

### 5.1 PCA Reconstruction Plot
- Plot eigenvector shapes
- Highlight correspondence to Balmer wings  
- Show variance explained curve  

### 5.2 R² vs PCA Dimension
- Sweep D from 1 → 50
- Expect monotonic curve showing intrinsic dimensionality  

### 5.3 Top-K Feature Stability Plot
- Importance vs wavelength  
- Vertical lines for selected top-k  

### 5.4 Noise Robustness Curve
\[
\text{R}^2(\text{noise})
\]
across models  

### 5.5 Error(σ) Contribution Heatmap
- If using heteroscedastic error inputs  

---

# 6. Conclusion (结论)
A concise synthesis of:
- What this experiment proves  
- How it supports next-step model design  
- What physics it reveals  
- Remaining uncertainties  

---

# 7. Next Steps (下一步)
- Hyperparameter sweeps  
- Neural architecture modifications  
- Additional physical diagnostics  
- Ablation experiments  

---------------------------------------------
(End of Template)
---------------------------------------------

Now fill the template with the provided data.
