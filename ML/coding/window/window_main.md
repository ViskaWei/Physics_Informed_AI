# Window 类题目汇总 [0/2 完成]

> 📊 **进度**: 0/2 完成 (0%)  
> 🔄 **最后更新**: 2026-01-02  
> 📁 **分类**: window (滑动窗口、多尺寸窗口、特征转换)

---

## 📋 题目总览

> 🔥 **重刷优先级**: TODO（按重要程度排序）

| 出题日期 | # | P编号 | 题目 | 难度 | 状态 | 完成日期 |
|----------|---|-------|------|------|------|----------|
| 2025-09-10 | 1 | P3639 | 历史的窗口搜索 | 中等 | ❌ | - |
| 2025-09-10 | 2 | P3640 | 多尺寸窗口滑动的特征转换 | 中等 | ❌ | - |

---

## 🔧 通用模板

```python
# 滑动窗口基础
from collections import deque

def sliding_window_max(nums, k):
    """滑动窗口最大值"""
    dq = deque()  # 存索引，保持单调递减
    result = []
    
    for i, num in enumerate(nums):
        # 移除窗口外的元素
        while dq and dq[0] <= i - k:
            dq.popleft()
        # 保持单调性
        while dq and nums[dq[-1]] < num:
            dq.pop()
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

---

## 题目1: 历史的窗口搜索（P3639）

- **难度**: 中等
- **源**: [core46#第2题-p3639](../AI_编程题_Python解答_核心46题.md#第2题-p3639)

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

## 题目2: 多尺寸窗口滑动的特征转换（P3640）

- **难度**: 中等
- **源**: [core46#第3题-p3640](../AI_编程题_Python解答_核心46题.md#第3题-p3640)

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
