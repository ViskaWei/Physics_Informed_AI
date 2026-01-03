# Tree 类题目汇总 [0/4 完成]

> 📊 **进度**: 0/4 完成 (0%)  
> 🔄 **最后更新**: 2026-01-02  
> 📁 **分类**: tree (决策树、剪枝、F1优化、阈值优化)

---

## 📋 题目总览

> 🔥 **重刷优先级**: TODO（按重要程度排序）

| 出题日期 | # | P编号 | 题目 | 难度 | 状态 | 完成日期 |
|----------|---|-------|------|------|------|----------|
| 2025-11-12 | 1 | P4465 | 决策树的QAM调制符合检测 | 中等 | ❌ | - |
| 2025-09-24 | 2 | P3792 | 基于决策树的无线状态预测 | 中等 | ❌ | - |
| 2025-09-05 | 3 | P3528 | 阈值最优的决策树 | 中等 | ❌ | - |
| 2025-08-27 | 4 | P3480 | F1值最优的决策树剪枝 | 中等 | ❌ | - |

---

## 🔧 通用模板

```python
# 决策树基础
def gini(y):
    """计算基尼系数"""
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p ** 2)

def entropy(y):
    """计算信息熵"""
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p + 1e-10))

def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return 2 * precision * recall / (precision + recall + 1e-10)
```

---

## 题目1: 决策树的QAM调制符合检测（P4465）

- **难度**: 中等
- **源**: [core46#第3题-p4465](../AI_编程题_Python解答_核心46题.md#第3题-p4465)

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

## 题目2: 基于决策树的无线状态预测（P3792）

- **难度**: 中等
- **源**: [core46#第3题-p3792](../AI_编程题_Python解答_核心46题.md#第3题-p3792)

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

## 题目3: 阈值最优的决策树（P3528）

- **难度**: 中等
- **源**: [core46#第2题-p3528](../AI_编程题_Python解答_核心46题.md#第2题-p3528)

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

## 题目4: F1值最优的决策树剪枝（P3480）

- **难度**: 中等
- **源**: [core46#第3题-p3480](../AI_编程题_Python解答_核心46题.md#第3题-p3480)

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
