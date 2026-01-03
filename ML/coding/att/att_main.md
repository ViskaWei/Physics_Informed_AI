# Att 类题目汇总 [0/6 完成]

> 📊 **进度**: 0/6 完成 (0%)  
> 🔄 **最后更新**: 2026-01-02  
> 📁 **分类**: att (Attention、ViT、LoRA、Multi-Head、Sparse Attention)

---

## 📋 题目总览

> 🔥 **重刷优先级**: TODO（按重要程度排序）

| 出题日期 | # | P编号 | 题目 | 难度 | 状态 | 完成日期 |
|----------|---|-------|------|------|------|----------|
| 2025-11-20 | 1 | P4481 | ViT Patch Embedding层实现 | 中等 | ❌ | - |
| 2025-10-22 | 2 | P4275 | 基于空间连续块的稀疏注意力机制 | 中等 | ❌ | - |
| 2025-10-15 | 3 | P4227 | 动态注意力掩码调度问题 | 中等 | ❌ | - |
| 2025-09-28 | 4 | P3843 | Masked Multi-Head Self-Attention 实现 | 中等 | ❌ | - |
| 2025-09-17 | 5 | P3712 | 大模型Attention模块开发 | 中等 | ❌ | - |
| 2025-09-12 | 6 | P3658 | 支持LoRA的Attention实现 | 中等 | ❌ | - |

---

## 🔧 通用模板

```python
# Scaled Dot-Product Attention
import numpy as np

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores)
    return weights @ V

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
```

---

## 题目1: ViT Patch Embedding层实现（P4481）

- **难度**: 中等
- **源**: [core46#第2题-p4481](../AI_编程题_Python解答_核心46题.md#第2题-p4481)

### 题目描述
TODO

### 思路
TODO

### 复杂度
TODO

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目2: 基于空间连续块的稀疏注意力机制（P4275）

- **难度**: 中等
- **源**: [core46#第3题-p4275](../AI_编程题_Python解答_核心46题.md#第3题-p4275)

### 题目描述
TODO

### 思路
TODO

### 复杂度
TODO

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目3: 动态注意力掩码调度问题（P4227）

- **难度**: 中等
- **源**: [core46#第2题-p4227](../AI_编程题_Python解答_核心46题.md#第2题-p4227)

### 题目描述
TODO

### 思路
TODO

### 复杂度
TODO

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目4: Masked Multi-Head Self-Attention 实现（P3843）

- **难度**: 中等
- **源**: [core46#第3题-p3843](../AI_编程题_Python解答_核心46题.md#第3题-p3843)

### 题目描述
TODO

### 思路
TODO

### 复杂度
TODO

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目5: 大模型Attention模块开发（P3712）

- **难度**: 中等
- **源**: [core46#第2题-p3712](../AI_编程题_Python解答_核心46题.md#第2题-p3712)

### 题目描述
TODO

### 思路
TODO

### 复杂度
TODO

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目6: 支持LoRA的Attention实现（P3658）

- **难度**: 中等
- **源**: [core46#第3题-p3658](../AI_编程题_Python解答_核心46题.md#第3题-p3658)

### 题目描述
TODO

### 思路
TODO

### 复杂度
TODO

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 📌 易错点总结

1. TODO

---

## 🔗 相关文件

- 源文件：`../AI_编程题_Python解答_核心46题.md`
- 索引：`../ai_core46_index.md`
