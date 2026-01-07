# Tree 类题目汇总 [0/5 完成]

> 📊 **进度**: 0/5 完成 (0%)  
> 🔄 **最后更新**: 2026-01-05  
> 📁 **分类**: tree (决策树、剪枝、F1优化、阈值优化、推理)

---

## 📋 题目总览

> 🔥 **重刷优先级**: 5 > 1 > 4 > 2 > 3（按难度和重要程度排序）

| 出题日期 | # | P编号 | 题目 | 难度 | 状态 | 完成日期 |
|----------|---|-------|------|------|------|----------|
| 2025-11-12 | 1 | P4465 | 决策树的QAM调制符合检测（CART+Gini） ⭐ | 困难 | ❌ | - |
| 2025-09-24 | 2 | P3792 | 基于决策树的无线状态预测（ID3+信息增益） | 中等 | ❌ | - |
| 2025-09-05 | 3 | P3528 | 阈值最优的决策树 | 中等 | ❌ | - |
| 2025-08-28 | 5 | P3492 | 基于决策树预判资源调配优先级（推理） ⭐ | 简单 | ❌ | - |
| 2025-08-27 | 4 | P3480 | F1值最优的决策树剪枝 ⭐ | 中等 | ❌ | - |

---

## 🔧 通用模板
### 决策树
```python
tree = [Node(x) for x in X]
def decide(x):
    cur = tree[0]
    while cur:
        if cur.fdx == -1: return cur.label
        if x[cur.fdx] <= cur.val:
            cur = tree[cur.left]
        else:
            cur = tree[cur.right]
```
### Gini 系数计算
```python
from collections import Counter

def gini(labels):
    """计算 Gini 系数"""
    n = len(labels)
    if n == 0:
        return 0.0
    cnt = Counter(labels)
    return 1.0 - sum((c/n)**2 for c in cnt.values())
```

### 信息熵计算
```python
import math

def entropy(labels):
    """计算信息熵"""
    n = len(labels)
    if n == 0:
        return 0.0
    cnt = Counter(labels)
    return -sum((c/n) * math.log2(c/n) for c in cnt.values() if c > 0)
```

### F1 分数计算
```python
def f1_score(y_true, y_pred):
    """计算 F1 分数（二分类，正类为 1）"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
```

### 多数类投票
```python
def majority_label(labels):
    """返回多数类（平票取最小值）"""
    cnt = Counter(labels)
    max_c = max(cnt.values())
    return min(k for k, v in cnt.items() if v == max_c)
```

---

## 题目1: 决策树的QAM调制符合检测（P4465）⭐

- **难度**: 困难
- **核心**: CART 决策树 + Gini 系数
- **源**: [core46#第3题-p4465](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- 输入：M 个 16QAM 符号（复数的实部和虚部）+ 标签（0-15）
- 使用 CART 决策树进行分类
- 特征：2 维（x1=实部，x2=虚部）
- 输出：训练集 Gini 系数 + 测试样本预测标签

### 关键规则
1. 划分标准：**Gini 系数**
2. 最大深度：**5**
3. **切分点限制**：{-3, -2, -1, 0, 1, 2, 3}
4. 划分规则：`x[f] < threshold` 走左，`>=` 走右
5. Gini 系数保留 4 位小数

### 样例
```
输入:
10
2.56 0.73 14
3.88 0.83 14
-0.32 2.93 7
...
-1.14 0.20

输出:
0.8600
6
```

### 思路
1. 计算训练集整体 Gini 系数
2. 递归建 CART 树：
   - 枚举 2 个特征 × 7 个阈值
   - 选择加权 Gini 最小的划分
   - 深度达 5 或无有效提升则生成叶子
3. 预测时沿树走到叶子

### 复杂度
- 时间: O(N × 深度 × 特征数 × 阈值数)
- 空间: O(树节点数)

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目2: 基于决策树的无线状态预测（P3792）

- **难度**: 中等
- **核心**: ID3 决策树 + 信息增益
- **源**: [core46#第3题-p3792](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- 输入：n 个样本，m 个二值特征（0/1），标签 0/1
- 使用信息增益构建决策树
- 预测 q 个查询样本

### 关键规则
1. 划分标准：**信息增益**
2. 特征值：只有 0 和 1
3. **信息增益相等时**：选索引更小的特征
4. **无法划分时**：返回多数类（平票返回 0）

### 样例
```
输入:
10 3
1 0 1 1
1 0 0 0
...
3
1 0 1
0 0 0
1 1 0

输出:
1
0
1
```

### 思路
1. 计算熵：$H(S) = -\sum p_i \log_2 p_i$
2. 信息增益 = 划分前熵 - 加权条件熵
3. 选择增益最大的特征划分
4. 递归建树

### 复杂度
- 时间: O(N × M²)（最坏情况每层扫描所有特征）
- 空间: O(树节点数)

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目3: 阈值最优的决策树（P3528）

- **难度**: 中等
- **核心**: 决策树 + 阈值优化
- **源**: [core46#第2题-p3528](../AI_编程题_Python解答_核心46题.md)

### 题目描述
TODO: 从源文件补充详细描述

### 关键规则
TODO

### 样例
TODO

### 思路
TODO

### 复杂度
TODO

### 我的代码
```python
import sys, numpy as np

def read():
    d = np.fromstring(sys.stdin.buffer.read(), dtype=np.int64, sep=' ')
    if d.size == 0:
        return None
    m = int(d[0])
    xy = d[1:1 + 2 * m].reshape(m, 2)
    L, R = map(int, d[1 + 2 * m:1 + 2 * m + 2])
    return xy[:, 0], xy[:, 1], L, R

def main():
    t = read()
    if t is None: return
    x, y, L, R = t
    idx = np.lexsort((y,x)); y=y[idx];x=x[idx];
    pref = np.cumsum(y==L); sur = np.cumsum((y==R)[::-1])[::-1]
    cut = (pref[:-1] + sur[1:]).max(initial=0)
    best = max(sur[0], pref[-1], cut)
    sys.stdout.write(f"{best / x.size:.3f}\n")
if __name__ == "__main__":
    main()
```

---

## 题目4: F1值最优的决策树剪枝（P3480）⭐

- **难度**: 中等
- **核心**: 后剪枝 + F1 优化
- **源**: [core46#第3题-p3480](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- 给定一棵未剪枝的二叉决策树
- 对验证集进行评估，通过剪枝使 F1 最优
- 输出最优 F1 值（保留 6 位小数）

### 关键规则
1. 节点格式：`l r f th label`（左子/右子/特征/阈值/标签）
2. 决策规则：`x[f] <= th` 走左，`> th` 走右
3. 叶子节点：l=0, r=0
4. F1 = 2×P×R / (P+R)
5. 可以将任意节点剪枝为叶子

### 样例
```
输入:
7 3 2
2 3 1 50 0
4 5 2 50 0
6 7 2 50 1
0 0 0 0 0
0 0 0 0 1
0 0 0 0 0
0 0 0 0 1
30 60 1
30 30 1
60 30 1

输出:
0.800000
```

### 思路
1. DFS 后序遍历每个节点
2. 对每个节点计算两种情况：
   - 保留子树时的 F1
   - 剪枝为叶子时的 F1
3. 选择更大的 F1
4. 实现：递归返回 (tp, fp, fn, best_f1)

### 复杂度
- 时间: O(N × M)（N 节点数，M 验证集大小）
- 空间: O(N)（递归栈）

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目5: 基于决策树预判资源调配优先级（P3492）⭐

- **难度**: 简单
- **核心**: 决策树推理（非构建）
- **源**: [0828coding.md](../../../0828coding.md)

### 题目描述
- 输入：已训练好的决策树模型 + 待推理样本
- 决策树模型用矩阵表示（每行：分裂特征下标、阈值、左右子节点行号、分类结果）
- 对每个样本，从根节点遍历到叶子节点输出分类结果

### 关键规则
1. 划分规则：`x[f] <= threshold` 走左，`> threshold` 走右
2. 叶子节点：`feature_index == -1`
3. 节点编号从 0 开始，首行为根节点

### 样例
```
输入:
2 5 2
0 2.5 1 2 -1
-1 -1 -1 -1 1
1 5.0 3 4 -1
-1 -1 -1 -1 2
-1 -1 -1 -1 3
1.2 3.4
5.6 6.0

输出:
1
3
```

### 思路
1. 解析树结构：每个节点存储 `(feature_index, threshold, left, right, label)`
2. 推理：从根节点开始
   - 若 `feature_index == -1`，输出 label
   - 否则比较 `x[feature_index]` 与 `threshold`，决定走左/右

### 复杂度
- 时间: O(n × h)（n 样本数，h 树高度）
- 空间: O(m)（m 节点数）

### 我的代码
```python
import sys
d = sys.stdin.buffer.read().split()
import numpy as np
F, M, N = list(map(int, d[:3])); X = [np.array(d[3 + 5 * i: 3 + 5 * (i+1)], float) for i in range(M)]
X1 = np.array(d[3 + 5 * (M):], float).reshape(N, F)

class Node():
    def __init__(self, a):
        self.fdx = int(a[0])
        self.val = float(a[1])
        self.left = int(a[2])
        self.right = int(a[3])
        self.label = int(a[4])
tree = [Node(x) for x in X]
def decide(x):
    cur = tree[0]
    while cur:
        if cur.fdx == -1: return cur.label
        if x[cur.fdx] <= cur.val:
            cur = tree[cur.left]
        else:
            cur = tree[cur.right]
out = [decide(x) for x in X1]
for o in out: print(o)
```

---

## 📌 易错点总结

1. **Gini vs 熵**：
   - Gini: $1 - \sum p_i^2$（更常用于 CART）
   - 熵: $-\sum p_i \log_2 p_i$（用于 ID3/C4.5）

2. **划分规则的边界**：
   - CART：`< threshold` vs `>= threshold`
   - ID3：按特征值（0/1）分

3. **平票处理**：题目通常要求选最小值/索引

4. **F1 计算**：
   - 正类为 1，负类为 0
   - 分母为 0 时返回 0

5. **剪枝策略**：后剪枝比预剪枝更常见

6. **深度限制**：记得在递归时传递并检查

---

## 🔗 相关文件

- 源文件：`../AI_编程题_Python解答_核心46题.md`
- 索引：`../ai_core46_index.md`

---

## 📝 代码答案

### 题目1: P4465 决策树的QAM调制符合检测
```python
import sys
from collections import Counter

def gini_of_labels(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    cnt = Counter(labels)
    return 1.0 - sum((c/n)**2 for c in cnt.values())

def majority_label(labels):
    cnt = Counter(labels)
    maxc = max(cnt.values())
    return min(k for k, v in cnt.items() if v == maxc)

class Node:
    def __init__(self):
        self.is_leaf = True
        self.label = 0
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None

THRESHOLDS = [-3, -2, -1, 0, 1, 2, 3]

def build_tree(X, y, idxs, depth_left):
    node = Node()
    curr_labels = [y[i] for i in idxs]
    curr_gini = gini_of_labels(curr_labels)
    
    if curr_gini == 0.0 or depth_left == 0:
        node.is_leaf = True
        node.label = majority_label(curr_labels)
        return node

    best_gini = float('inf')
    best_f, best_t = -1, None
    best_left, best_right = None, None

    for f in [0, 1]:
        for t in THRESHOLDS:
            left, right = [], []
            for i in idxs:
                (left if X[i][f] < t else right).append(i)
            if not left or not right:
                continue
            g_left = gini_of_labels([y[i] for i in left])
            g_right = gini_of_labels([y[i] for i in right])
            w = (len(left)/len(idxs))*g_left + (len(right)/len(idxs))*g_right
            if w < best_gini - 1e-12:
                best_gini, best_f, best_t = w, f, t
                best_left, best_right = left, right

    if best_left is None or best_gini >= curr_gini - 1e-12:
        node.is_leaf = True
        node.label = majority_label(curr_labels)
        return node

    node.is_leaf = False
    node.feature, node.threshold = best_f, best_t
    node.left = build_tree(X, y, best_left, depth_left - 1)
    node.right = build_tree(X, y, best_right, depth_left - 1)
    return node

def predict(root, x):
    node = root
    while not node.is_leaf:
        node = node.left if x[node.feature] < node.threshold else node.right
    return node.label

def main():
    data = sys.stdin.read().strip().split()
    it = iter(data)
    M = int(next(it))
    X, y = [], []
    for _ in range(M):
        X.append([float(next(it)), float(next(it))])
        y.append(int(next(it)))
    test = [float(next(it)), float(next(it))]

    G = gini_of_labels(y)
    root = build_tree(X, y, list(range(M)), depth_left=5)
    pred = predict(root, test)

    print(f"{G:.4f}")
    print(pred)

if __name__ == "__main__":
    main()
```

### 题目2: P3792 基于决策树的无线状态预测
```python
import sys
import math
from collections import Counter

def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    cnt = Counter(labels)
    return -sum((c/n) * math.log2(c/n) for c in cnt.values() if c > 0)

def majority_label(labels):
    c1 = sum(labels)
    c0 = len(labels) - c1
    return 1 if c1 > c0 else 0

def build_tree(X, y, idxs, features):
    labels = [y[i] for i in idxs]
    if all(l == labels[0] for l in labels):
        return {"leaf": True, "label": labels[0]}

    base_H = entropy(labels)
    best_gain, best_f = -1.0, -1
    eps = 1e-12

    for f in features:
        idx0, idx1 = [], []
        for i in idxs:
            (idx0 if X[i][f] == 0 else idx1).append(i)
        lab0 = [y[i] for i in idx0]
        lab1 = [y[i] for i in idx1]
        cond = (len(idx0)/len(idxs))*entropy(lab0) + (len(idx1)/len(idxs))*entropy(lab1)
        gain = base_H - cond
        if gain > best_gain + eps or (abs(gain - best_gain) <= eps and f < best_f):
            best_gain, best_f = gain, f

    if best_gain <= eps or best_f == -1:
        return {"leaf": True, "label": majority_label(labels)}

    idx0, idx1 = [], []
    for i in idxs:
        (idx0 if X[i][best_f] == 0 else idx1).append(i)

    next_features = [f for f in features if f != best_f]
    left = {"leaf": True, "label": majority_label(labels)} if not idx0 else build_tree(X, y, idx0, next_features)
    right = {"leaf": True, "label": majority_label(labels)} if not idx1 else build_tree(X, y, idx1, next_features)

    return {"leaf": False, "feat": best_f, "left": left, "right": right}

def predict(tree, x):
    node = tree
    while not node["leaf"]:
        node = node["left"] if x[node["feat"]] == 0 else node["right"]
    return node["label"]

def main():
    data = sys.stdin.read().strip().split()
    it = iter(data)
    n, m = int(next(it)), int(next(it))
    X, y = [], []
    for _ in range(n):
        row = [int(next(it)) for _ in range(m + 1)]
        X.append(row[:m])
        y.append(row[m])
    q = int(next(it))
    Q = [[int(next(it)) for _ in range(m)] for _ in range(q)]

    tree = build_tree(X, y, list(range(n)), list(range(m)))
    print("\n".join(str(predict(tree, x)) for x in Q))

if __name__ == "__main__":
    main()
```

### 题目3: P3528 阈值最优的决策树
```python
# TODO: 待补充完整代码
```

### 题目4: P3480 F1值最优的决策树剪枝
```python
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n, m, k = int(next(it)), int(next(it)), int(next(it))

    nodes = {}
    for i in range(1, n + 1):
        nodes[i] = {
            'left': int(next(it)),
            'right': int(next(it)),
            'feature': int(next(it)),
            'threshold': int(next(it)),
            'label': int(next(it)),
            'is_leaf': False
        }
        nodes[i]['is_leaf'] = nodes[i]['left'] == 0 and nodes[i]['right'] == 0

    validation = []
    for _ in range(m):
        features = [float(next(it)) for _ in range(k)]
        label = int(next(it))
        validation.append((features, label))

    def evaluate(pred_label, data_subset):
        tp = fp = fn = 0
        for _, true_label in data_subset:
            if pred_label == 1 and true_label == 1: tp += 1
            elif pred_label == 1 and true_label == 0: fp += 1
            elif pred_label == 0 and true_label == 1: fn += 1
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return tp, fp, fn, f1

    def prune(node_id, data_subset):
        node = nodes[node_id]
        tp_leaf, fp_leaf, fn_leaf, f1_leaf = evaluate(node['label'], data_subset)
        
        if node['is_leaf'] or not data_subset:
            return tp_leaf, fp_leaf, fn_leaf, f1_leaf

        left_data, right_data = [], []
        for features, true_label in data_subset:
            if features[node['feature'] - 1] <= node['threshold']:
                left_data.append((features, true_label))
            else:
                right_data.append((features, true_label))

        left_tp, left_fp, left_fn, _ = prune(node['left'], left_data)
        right_tp, right_fp, right_fn, _ = prune(node['right'], right_data)

        tp_sub = left_tp + right_tp
        fp_sub = left_fp + right_fp
        fn_sub = left_fn + right_fn
        precision_sub = tp_sub / (tp_sub + fp_sub) if tp_sub + fp_sub > 0 else 0
        recall_sub = tp_sub / (tp_sub + fn_sub) if tp_sub + fn_sub > 0 else 0
        f1_sub = 2 * precision_sub * recall_sub / (precision_sub + recall_sub) if precision_sub + recall_sub > 0 else 0

        if f1_leaf > f1_sub:
            return tp_leaf, fp_leaf, fn_leaf, f1_leaf
        return tp_sub, fp_sub, fn_sub, f1_sub

    _, _, _, best_f1 = prune(1, validation)
    print(f"{best_f1:.6f}")

if __name__ == "__main__":
    main()
```

### 题目5: P3492 基于决策树预判资源调配优先级
```python
# 定义节点类
class Node:
    def __init__(self, feature_index, threshold, left, right, label):
        self.feature_index = feature_index  # 分裂特征下标
        self.threshold = threshold          # 分裂阈值
        self.left = left                    # 左子节点行号
        self.right = right                  # 右子节点行号
        self.label = label                  # 分类结果

# 读取输入
f, m, n = map(int, input().split())
tree = []
for _ in range(m):
    fi, thr, l, r, lbl = input().split()
    tree.append(Node(int(fi), float(thr), int(l), int(r), int(lbl)))

# 推理过程
for _ in range(n):
    features = list(map(float, input().split()))
    current = 0  # 从根节点开始
    while True:
        node = tree[current]
        if node.feature_index == -1:  # 到叶子节点
            print(node.label)
            break
        if features[node.feature_index] <= node.threshold:
            current = node.left
        else:
            current = node.right
```
