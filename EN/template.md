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

# 📑 Table of Contents

- [1. 🎯 Objectives](#1--objectives)
- [2. 🧪 Experimental Design](#2--experimental-design)
- [3. 📊 Figures](#3--figures)
- [4. 💡 Key Insights](#4--key-insights)
- [5. 📝 Conclusion](#5--conclusion)
- [6. 📎 Appendix](#6--appendix)

---

# 1. 🎯 Objectives

## 1.1 Background & Motivation
> Describe the *highest-level scientific or engineering goal*, typically:
- Understanding spectral physics  
- Guiding neural-network architecture design  
- Identifying linear vs nonlinear components  
- Evaluating information content of wavelength regions  

Example:
- Provide physical + statistical understanding to design an optimal neural architecture for predicting stellar parameters.

## 1.2 Core Hypothesis
> Describe the core hypothesis this experiment tests.

**Format**:
```
> **[One-sentence hypothesis statement]**

If the hypothesis holds:
- [Implication 1]
- [Implication 2]

If the hypothesis does not hold:
- [Alternative understanding/direction]
```

Typical examples:
- "log g is controlled by a low-dimensional linear direction in flux space"
- "Error(σ) contains exploitable information about stellar parameters"
- "Most of NN capacity is used for filtering irrelevant information"

## 1.3 Verification Questions

> Use a **table** to list 3-5 specific questions, each with a verification target. Fill in results after the experiment.

| # | Question | Verification Target | Result |
|---|----------|---------------------|--------|
| Q1 | [Specific, quantifiable question?] | [Which aspect of the hypothesis does this verify?] | [❌/✅ + value] |
| Q2 | ... | ... | ... |
| Q3 | ... | ... | ... |

**Writing tips**:
- Questions must be **specific and quantifiable** (e.g., "Can we achieve $R^2 \geq 0.98$?" not "Is it good?")
- Verification target explains **which hypothesis/design decision this links to**
- Results start with ❌/✅, followed by key values

## 1.4 Summary of Conclusions (fill after experiment)

### 1.4.1 Experimental Conclusions

| Conclusion | Explanation |
|------------|-------------|
| **[Conclusion keyword]** | [One-sentence explanation] |
| **[Conclusion keyword]** | [One-sentence explanation] |

### 1.4.2 Design Implications

| Design Principle | Specific Recommendation |
|------------------|-------------------------|
| **[Principle name]** | [Specific action] |
| **[Principle name]** | [Specific action] |

> **One-sentence summary**: [Summarize the core finding and implication of the entire experiment in one sentence]

---

# 2. 🧪 Experimental Design

## 2.1 Data
- Training samples:  
- Test samples:  
- Feature dimensions:  
- Label parameters:  
- Noise model:  

$$
\text{noisy\_flux} = \text{flux} + \mathcal{N}(0, \sigma^2)
$$

## 2.2 Features Used
- flux / error / concatenated  
- PCA components  
- Top-K selected wavelengths  
- Patch-based features
- Other:

## 2.3 Model / Algorithm
Include equations if linear:

### If Linear Regression / Ridge:

$$
\hat{y} = X w + b
$$

$$
w = (X^\top X + \alpha I)^{-1} X^\top y
$$

### If LightGBM:
- num_leaves, max_depth, feature_fraction, etc.

### If PCA:
Explained variance curve:

$$
\lambda_i / \sum_j \lambda_j
$$

## 2.4 Hyperparameters
- alpha:  
- PCA dimensions:  
- noise levels:  
- LightGBM parameters:  
- etc.

---

# 3. 📊 Figures

> Display existing experiment figures. Each figure should have a title, description, and key observations.

### Figure 1: [Figure Title]
![Image](path/to/image.png)

**Figure 1. [Figure description]**

**Key observations**:
- Observation 1
- Observation 2

---

# 4. 💡 Key Insights

## 4.1 High-Level Insights (for guiding Neural Network architecture design)
Examples:
- log g is primarily in a low-dimensional linear subspace (1–10 dims)  
- log g information mainly comes from Balmer wings / CaT / continuum slope  
- PCA>5 recovers >98% R² → NN doesn't need high-dimensional input  
- Heteroscedastic error σ provides additional discriminative power  

## 4.2 Model-Level Insights (for optimizing models)
Examples:
- Ridge α too large causes over-smoothing  
- Top-K single-point method unstable → need local window structure  
- Noise=1.0, PCA still maintains high R² → intrinsic dimensionality is low  

## 4.3 Experiment-Level Details
Examples:
- Top 50 points effective at low noise, completely fail at high noise  
- error σ changes model's decision behavior at high noise  
- PCA component 1 corresponds to equivalent width variation, strongly correlated with log g  

---

# 5. 📝 Conclusion

> **Writing principle**: Conclusion should follow the logic of **"Core Finding → Key Conclusions → Design Implications → Physical Explanation → Quick Reference Numbers → Next Steps"**.

## 5.1 Core Finding

> Use one sentence to summarize the most important finding (punch line), highlighted in quote format.

Then use ❌/✅ contrast format to show hypothesis verification results:
- ❌ Original hypothesis: [What we originally thought]
- ✅ Experimental result: [What we actually found]

## 5.2 Key Conclusions (2-4 items)

> Use numbered table, each conclusion with evidence, limit to 2-4 items.

| # | Conclusion | Evidence |
|---|------------|----------|
| 1 | **[Conclusion keyword]** | [Supporting data/observation] |
| 2 | **[Conclusion keyword]** | [Supporting data/observation] |
| 3 | **[Conclusion keyword]** | [Supporting data/observation] |

## 5.3 Design Implications

> Organize by category: Architecture principles, Regularization strategies, Common pitfalls.

### Architecture Principles

| Principle | Recommendation | Reason |
|-----------|----------------|--------|
| [Principle name] | [Specific action] | [Why] |

### ⚠️ Common Pitfalls

| Common Practice | Experimental Evidence |
|-----------------|----------------------|
| "[Wrong practice]" | [Why it's wrong] |

## 5.4 Physical Explanation (optional)

> Use 1-3 bullet points to explain **why** we see these results, linking to domain knowledge.

## 5.5 Quick Reference Numbers

> Extract 3-5 most important numbers for quick review.

| Metric | Value |
|--------|-------|
| Best performance | [Value + configuration] |
| Minimum condition to reach target | [Value] |
| Performance ceiling | [Value] |

## 5.6 Next Steps

> Use table to list 2-3 specific directions.

| Direction | Specific Task |
|-----------|---------------|
| [Direction name] | [What to do specifically] |

---

# 6. 📎 Appendix

## 6.1 Numerical Results Tables

> Organize input results into the following table format.

Example format:

| Setting | R² | MAE | RMSE | Notes |
|---------|-----|------|-------|-------|
| PCA=5 |     |      |       |       |
| PCA=10 |    |      |       |       |
| Full |     |      |       |       |

If there are Top-K / Patch / Noise Sweep results, organize into similar tables.

## 6.2 Plot Suggestions

> Describe what plots to make, not code.

### 6.2.1 [Plot Title]
- **Purpose**: What to show
- **X-axis**:
- **Y-axis**:
- **Annotations**:

## 6.3 Related Files

| Type | Path |
|------|------|
| Figures | `path` |
| Results | `path` |

---------------------------------------------
(End of Template)
---------------------------------------------

Now fill the template with the provided data.
