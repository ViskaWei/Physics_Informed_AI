# RNN 类题目汇总 [0/2 完成]

> 📊 **进度**: 0/2 完成 (0%)  
> 🔄 **最后更新**: 2026-01-02  
> 📁 **分类**: rnn (LSTM、反向传播、循环神经网络)

---

## 📋 题目总览

> 🔥 **重刷优先级**: TODO（按重要程度排序）

| 出题日期 | # | P编号 | 题目 | 难度 | 状态 | 完成日期 |
|----------|---|-------|------|------|------|----------|
| 2025-10-17 | 1 | P4239 | 反向传播实现 | 中等 | ❌ | - |
| 2025-10-10 | 2 | P3875 | 经典LSTM模型结构实现 | 中等 | ❌ | - |

---

## 🔧 通用模板

```python
# LSTM 基础
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def lstm_cell(x, h_prev, c_prev, Wf, Wi, Wc, Wo, bf, bi, bc, bo):
    """LSTM 单元前向传播"""
    concat = np.concatenate([h_prev, x])
    
    f = sigmoid(Wf @ concat + bf)  # 遗忘门
    i = sigmoid(Wi @ concat + bi)  # 输入门
    c_hat = tanh(Wc @ concat + bc) # 候选记忆
    o = sigmoid(Wo @ concat + bo)  # 输出门
    
    c = f * c_prev + i * c_hat     # 新记忆
    h = o * tanh(c)                # 新隐藏状态
    
    return h, c
```

---

## 题目1: 反向传播实现（P4239）

- **难度**: 中等
- **源**: [core46#第3题-p4239](../AI_编程题_Python解答_核心46题.md#第3题-p4239)

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

## 题目2: 经典LSTM模型结构实现（P3875）

- **难度**: 中等
- **源**: [core46#第3题-p3875](../AI_编程题_Python解答_核心46题.md#第3题-p3875)

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
