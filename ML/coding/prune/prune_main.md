# Prune 类题目汇总 [0/1 完成]

> 📊 **进度**: 0/1 完成 (0%)  
> 🔄 **最后更新**: 2026-01-02  
> 📁 **分类**: prune (剪枝、模型压缩)

---

## 📋 题目总览

> 🔥 **重刷优先级**: -

| 出题日期 | # | P编号 | 题目 | 难度 | 状态 | 完成日期 |
|----------|---|-------|------|------|------|----------|
| 2025-12-03 | 1 | P4518 | 基于剪枝的神经网络模型压缩 | 中等 | ❌ | - |

---

## 🔧 通用模板

```python
keep[np.argpartition(np.abs(W).sum(1), k - 1)[:k]] = False

# 剪枝基础
import numpy as np

def magnitude_pruning(weights, sparsity):
    """按权重绝对值大小剪枝"""
    threshold = np.percentile(np.abs(weights), sparsity * 100)
    mask = np.abs(weights) >= threshold
    return weights * mask, mask

def structured_pruning(weights, ratio):
    """结构化剪枝（按通道）"""
    norms = np.linalg.norm(weights, axis=(1, 2, 3))
    k = int(len(norms) * (1 - ratio))
    indices = np.argsort(norms)[-k:]
    return weights[indices], indices
```

---

## 题目1: 基于剪枝的神经网络模型压缩（P4518）

- **难度**: 中等
- **源**: [core46#第2题-p4518](../AI_编程题_Python解答_核心46题.md#第2题-p4518)

### 题目描述
TODO

### 思路
TODO

### 复杂度
TODO

### 我的代码
```python
import sys, numpy as np

def read():
    a = np.fromstring(sys.stdin.buffer.read().decode(), sep=' ')
    if a.size == 0: return
    n, d, c = map(int, a[:3]); p = 3
    X = a[p:p+n*d].reshape(n, d); p += n*d
    W = a[p:p+d*c].reshape(d, c); p += d*c
    return X, W, float(a[p])

r = read(); X, W, ratio = r; d = W.shape[0]
k = max(int(ratio * d), int(ratio > 0))
keep = np.ones(d, bool); keep[np.argpartition(np.abs(W).sum(1), k - 1)[:k]] = False
X, W = X[:, keep], W[keep]
print(*((X @ W).argmax(1)))
```

---

## 📌 易错点总结

1. TODO

---

## 🔗 相关文件

- 源文件：`../AI_编程题_Python解答_核心46题.md`
- 索引：`../ai_core46_index.md`
