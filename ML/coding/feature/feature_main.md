# Feature 类题目汇总 [0/6 完成]

> 📊 **进度**: 0/6 完成 (0%)  
> 🔄 **最后更新**: 2026-01-04  
> 📁 **分类**: feature (特征工程、数据分配、实体匹配、关键点对齐)

---

## 📋 题目总览

> 🔥 **重刷优先级**: 1 > 4 > 2 > 5 > 3 > 6（按难度和重要程度排序）

| 出题日期 | # | P编号 | 题目 | 难度 | 状态 | 完成日期 |
|----------|---|-------|------|------|------|----------|
| 2025-11-05 | 1 | P4441 | 多目标推荐排序模型优化 ⭐ | 困难 | ❌ | - |
| 2025-10-29 | 2 | P4343 | 实体匹配结果合并问题（并查集） | 中等 | ❌ | - |
| 2025-10-23 | 3 | P4277 | 人脸关键点对齐（仿射变换） | 简单 | ❌ | - |
| 2025-10-10 | 4 | P3871 | 磁盘故障检测的特征工程（统计指标） ⭐ | 困难 | ❌ | - |
| 2025-09-04 | 5 | P3561 | 大模型训练数据均衡分配算法（LPT贪心） | 中等 | ❌ | - |
| 2025-08-27 | 6 | P3479 | 标签样本数量（KNN） | 中等 | ❌ | - |

---

## 🔧 通用模板
### Heap
```python
import heapq
h=[0] * N; heapq.heapify(h); 
val=heapq.heappop(h); heapq.heappush(h, x); 
```
### 并查集模板
```python
class UnionFind:
    def __init__(self):
        self.parent = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[py] = px
```

### 特征统计模板
```python
import math

def compute_stats(col):
    """计算一列数据的 8 项统计指标"""
    n = len(col)
    mu = sum(col) / n
    cmax, cmin = max(col), min(col)
    ptp = cmax - cmin
    
    # 二/三/四阶中心矩
    m2 = m3 = m4 = 0.0
    for x in col:
        d = x - mu
        m2 += d * d
        m3 += d * d * d
        m4 += d * d * d * d
    
    var = m2 / n
    std = math.sqrt(var)
    skew = (m3 / n) / (std ** 3) if std > 0 else 0.0
    kurt = (m4 / n) / (std ** 4) - 3.0 if std > 0 else 0.0
    
    return mu, cmax, cmin, ptp, std, var, skew, kurt
```

### LPT 贪心模板（负载均衡）
```python
import heapq

def lpt_schedule(tasks, n_machines):
    """LPT 贪心：将任务分配到 n 台机器，最小化最大负载"""
    tasks = sorted(tasks, reverse=True)  # 从大到小排序
    load = [0] * n_machines
    heapq.heapify(load)
    
    for t in tasks:
        cur = heapq.heappop(load)
        cur += t
        heapq.heappush(load, cur)
    
    return max(load)
```

---

## 题目1: 多目标推荐排序模型优化（P4441）⭐

- **难度**: 困难
- **核心**: 多任务学习 + 共享权重 + 联合损失
- **源**: [core46#第2题-p4441](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- 同时预测 CTR 和 CVR
- 共享特征权重 w，独立偏置 b_ctr, b_cvr
- 联合损失：$L = MSE_{CTR} + \alpha \cdot MSE_{CVR}$
- 输出：迭代 N 次后的平均联合损失值 × 10^10（四舍五入）

### 关键规则
1. 初始化：权重和偏置全为 0
2. 批量梯度下降
3. 输出格式：损失值 × 10^10 后四舍五入为整数

### 样例
```
输入:
1,1,1;2,2,2;3,3,3
1,0.5;2,1.0;3,1.5
500
0.01
0.5

输出:
27356237
```

### 思路
1. 解析输入（分号分隔样本，逗号分隔特征）
2. 前向计算 CTR 和 CVR 预测值
3. 计算梯度（联合损失对 w 和两个偏置的梯度）
4. 迭代更新
5. 最终损失 × 10^10

### 复杂度
- 时间: O(N × n × d)
- 空间: O(n × d)

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目2: 实体匹配结果合并问题（P4343）

- **难度**: 中等
- **核心**: 并查集
- **源**: [core46#第2题-p4343](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- N 个实体匹配系统的输出
- 如果两个系统有交集，合并结果
- 输出：合并后的实体组（按字典序排序）

### 关键规则
1. 每行输出按字典序排序
2. 组之间也按字典序排序
3. 使用并查集实现连通分量合并

### 样例
```
输入:
5
1 2 3
4 5
11 22
33 44 55 1
3 66

输出:
1 2 3 33 44 55 66
11 22
4 5
```

### 思路
1. 初始化并查集
2. 对每行，将所有实体与第一个实体合并
3. 按根节点分组
4. 排序输出

### 复杂度
- 时间: O(T × α(N))（T 为总实体出现次数）
- 空间: O(N)

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目3: 人脸关键点对齐（P4277）

- **难度**: 简单
- **核心**: 仿射变换 + 逆映射
- **源**: [core46#第2题-p4277](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- 给定输入图像 A、变换矩阵 M、输出尺寸
- 变换公式：
  - $x' = ax + by + t_x$
  - $y' = cx + dy + t_y$
- 输出变换后的图像

### 关键规则
1. 使用**逆映射**：对每个输出点 (x',y') 找源点 (x,y)
2. 最近邻插值（round）
3. 越界填 0
4. **线性部分不可逆**时返回全 0 图像

### 样例
```
输入:
3 2 1
10 20 30
40 50 60
70 80 90
0 1 0
-1 0 2
3 3

输出:
30 60 90 20 50 80 10 40 70
```

### 思路
1. 计算线性部分的逆矩阵
2. 对每个输出点 (x',y')：
   - 去除平移：dx = x' - tx, dy = y' - ty
   - 乘逆矩阵得源坐标
   - 四舍五入取整
3. 输出展平

### 复杂度
- 时间: O(H × W)
- 空间: O(H × W)

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目4: 磁盘故障检测的特征工程（P3871）⭐

- **难度**: 困难
- **核心**: 统计特征提取（8 项指标 × 19 列）
- **源**: [core46#第2题-p3871](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- 输入：多个样本，每样本 19 个特征
- 对每列计算 8 项统计指标：
  1. 均值 (Mean)
  2. 最大值 (Max)
  3. 最小值 (Min)
  4. 极差 (Ptp = Max - Min)
  5. 标准差 (Std)
  6. 方差 (Var)
  7. 偏度 (Skew)
  8. 峰度 (Kurt，Fisher 过度峰度，-3）

### 关键规则
1. 使用**总体**公式（分母为 n，不是 n-1）
2. 偏度：$\frac{\sum(x-\mu)^3/n}{\sigma^3}$
3. 峰度：$\frac{\sum(x-\mu)^4/n}{\sigma^4} - 3$
4. σ=0 时，skew=0, kurt=0
5. 输出保留 2 位小数

### 样例
```
输入:
1623456000 100.0 800000.0 ... （5个样本 × 19列）

输出:
1623456002.00 1623456004.00 ... （19列 × 8个指标 = 152个数）
```

### 思路
1. 解析输入，每 19 个数为一个样本
2. 按列重排
3. 对每列计算 8 项统计指标
4. 格式化输出

### 复杂度
- 时间: O(N × 19)
- 空间: O(N)

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目5: 大模型训练数据均衡分配算法（P3561）

- **难度**: 中等
- **核心**: LPT 贪心 + 最小堆
- **源**: [core46#第2题-p3561](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- m 个样本分配到 n 个 NPU
- 每个样本有长度，NPU 运行时间与样本长度和成正比
- 目标：最小化最大负载（min-max）

### 关键规则
1. 每个 NPU 至少分到一个样本
2. 样本不能切分
3. NP-hard 问题，使用 LPT 贪心近似

### 样例
```
输入:
4
7
89 245 64 128 79 166 144

输出:
245
```

### 思路
1. 样本按长度从大到小排序
2. 使用最小堆维护每个 NPU 的当前负载
3. 每次将最大样本分配给最空闲的 NPU
4. 返回最大负载

### 复杂度
- 时间: O(m log m + m log n)
- 空间: O(n)

### 我的代码
```python
import sys
it = iter(sys.stdin.read().strip().split())
N,M = int(next(it)),int(next(it)); X=[int(next(it)) for _ in range(M)]; X.sort(reverse=True)
load=[0] * N;
for i in range(M): load[min(range(N), key=lambda x: load[x])] += X[i]
print(max(load))
# 如果用heap
import heapq
h=[0] * N; heapq.heapify(h); maxx=-float('inf')
for x in X: val=heapq.heappop(h);heapq.heappush(h, val + x); maxx=max(val+x, maxx)
```

---

## 题目6: 标签样本数量（P3479）

- **难度**: 中等
- **核心**: KNN 分类
- **源**: [core46#第2题-p3479](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- 给定 m 个样本（n 维特征 + 标签）
- 对待分类点，找 k 个最近邻
- 输出：多数类标签 + 该标签在 k 近邻中的数量

### 关键规则
1. 欧氏距离（可用平方距离排序）
2. **并列第一**：选距离最近的那个邻居的标签
3. 输出格式：`标签 数量`

### 样例
```
输入:
3 10 2 3
0.81 0.64
0.19 0.2 1.0
0.18 0.14 0.0
...

输出:
0 2
```

### 思路
1. 计算待分类点到所有样本的距离
2. 排序取前 k 个
3. 统计标签频次
4. 找最高频次，若并列则选最近邻居的标签

### 复杂度
- 时间: O(m log m + m × n)
- 空间: O(m)

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 📌 易错点总结

1. **并查集的路径压缩**：别忘了递归时更新 parent

2. **仿射变换的逆映射**：
   - 不是直接应用变换，而是从输出找源
   - 线性部分行列式为 0 时不可逆

3. **统计量公式**：
   - 总体 vs 样本（分母 n vs n-1）
   - 峰度要 -3（Fisher 过度峰度）

4. **LPT 贪心**：
   - 从大到小排序
   - 分配给当前负载最小的机器

5. **KNN 并列处理**：
   - 先按距离排序
   - 并列时选最近邻居的标签

6. **多目标学习的梯度**：
   - 共享权重的梯度是两个任务梯度之和
   - 注意 α 的位置

---

## 🔗 相关文件

- 源文件：`../AI_编程题_Python解答_核心46题.md`
- 索引：`../ai_core46_index.md`

---

## 📝 代码答案

### 题目1: P4441 多目标推荐排序模型优化
```python
from ast import literal_eval
from decimal import Decimal, ROUND_HALF_UP
import sys

def parse_matrix(line: str):
    s = '[[' + line.strip().replace(';', '],[') + ']]'
    mat = literal_eval(s)
    return [[float(v) for v in row] for row in mat]

def train_and_loss(X, Y, iters, lr, alpha):
    n, d = len(X), len(X[0])
    w = [0.0] * d
    b_ctr, b_cvr = 0.0, 0.0

    for _ in range(iters):
        yhat_ctr = [sum(w[j] * X[i][j] for j in range(d)) + b_ctr for i in range(n)]
        yhat_cvr = [sum(w[j] * X[i][j] for j in range(d)) + b_cvr for i in range(n)]
        e_ctr = [yhat_ctr[i] - Y[i][0] for i in range(n)]
        e_cvr = [yhat_cvr[i] - Y[i][1] for i in range(n)]

        grad_w = [(2.0/n) * sum((e_ctr[i] + alpha*e_cvr[i]) * X[i][j] for i in range(n)) for j in range(d)]
        grad_b_ctr = (2.0/n) * sum(e_ctr)
        grad_b_cvr = alpha * (2.0/n) * sum(e_cvr)

        for j in range(d):
            w[j] -= lr * grad_w[j]
        b_ctr -= lr * grad_b_ctr
        b_cvr -= lr * grad_b_cvr

    # 最终损失
    yhat_ctr = [sum(w[j] * X[i][j] for j in range(d)) + b_ctr for i in range(n)]
    yhat_cvr = [sum(w[j] * X[i][j] for j in range(d)) + b_cvr for i in range(n)]
    mse_ctr = sum((yhat_ctr[i] - Y[i][0])**2 for i in range(n)) / n
    mse_cvr = sum((yhat_cvr[i] - Y[i][1])**2 for i in range(n)) / n
    return mse_ctr + alpha * mse_cvr

def main():
    lines = [line.rstrip('\n') for line in sys.stdin if line.strip()]
    X = parse_matrix(lines[0])
    Y = parse_matrix(lines[1])
    iters = int(lines[2])
    lr = float(lines[3])
    alpha = float(lines[4])

    loss = train_and_loss(X, Y, iters, lr, alpha)
    val = Decimal(str(loss)) * Decimal('10000000000')
    print(int(val.to_integral_value(rounding=ROUND_HALF_UP)))

if __name__ == "__main__":
    main()
```

### 题目2: P4343 实体匹配结果合并问题
```python
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[py] = px

def merge_entities(n, systems):
    uf = UnionFind()
    for line in systems:
        for entity in line:
            if entity not in uf.parent:
                uf.parent[entity] = entity

    for line in systems:
        base = line[0]
        for entity in line[1:]:
            uf.union(base, entity)

    groups = {}
    for entity in uf.parent:
        root = uf.find(entity)
        groups.setdefault(root, set()).add(entity)

    result = [sorted(group) for group in groups.values()]
    result.sort()
    return result

if __name__ == "__main__":
    n = int(input())
    systems = [input().strip().split() for _ in range(n)]
    for group in merge_entities(n, systems):
        print(" ".join(group))
```

### 题目3: P4277 人脸关键点对齐
```python
import sys

def affine_transform(A, M, H, W):
    a, b, tx = M[0]
    c, d, ty = M[1]
    det = a * d - b * c
    hA, wA = len(A), len(A[0]) if A else 0
    O = [[0] * W for _ in range(H)]
    
    if abs(det) < 1e-12 or hA == 0 or wA == 0:
        return O

    inv00, inv01 = d / det, -b / det
    inv10, inv11 = -c / det, a / det

    for y2 in range(H):
        for x2 in range(W):
            dx, dy = x2 - tx, y2 - ty
            x = inv00 * dx + inv01 * dy
            y = inv10 * dx + inv11 * dy
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < hA and 0 <= xi < wA:
                O[y2][x2] = A[yi][xi]
    return O

if __name__ == "__main__":
    lines = sys.stdin.read().strip().splitlines()
    a, m, _ = map(int, lines[0].split())
    idx = 1
    A = [list(map(int, lines[idx+i].split())) for i in range(a)]
    idx += a
    M = [list(map(float, lines[idx].split())), list(map(float, lines[idx+1].split()))]
    idx += m
    H, W = map(int, lines[idx].split())

    O = affine_transform(A, M, H, W)
    print(" ".join(str(x) for row in O for x in row))
```

### 题目4: P3871 磁盘故障检测的特征工程
```python
import sys, math
from ast import literal_eval

COLS = 19

def compute_col_stats(col):
    n = len(col)
    mu = sum(col) / n
    cmax, cmin = max(col), min(col)
    ptp = cmax - cmin
    m2 = m3 = m4 = 0.0
    for x in col:
        d = x - mu
        m2 += d * d
        m3 += d * d * d
        m4 += d * d * d * d
    var = m2 / n
    std = math.sqrt(var)
    skew = (m3 / n) / (std ** 3) if std > 0 else 0.0
    kurt = (m4 / n) / (std ** 4) - 3.0 if std > 0 else 0.0
    return (mu, cmax, cmin, ptp, std, var, skew, kurt)

def main():
    text = sys.stdin.read().strip()
    if not text:
        print("")
        return
    
    nums = []
    try:
        parsed = literal_eval(text)
        def flat(it):
            for v in it:
                if isinstance(v, (list, tuple)):
                    yield from flat(v)
                else:
                    yield float(v)
        nums = list(flat(parsed))
    except:
        text = text.replace(',', ' ')
        nums = [float(t) for t in text.split()]

    n = len(nums) // COLS
    out = []
    for j in range(COLS):
        col = [nums[j + i * COLS] for i in range(n)]
        out.extend(compute_col_stats(col))
    print(' '.join(f"{x:.2f}" for x in out))

if __name__ == "__main__":
    main()
```

### 题目5: P3561 大模型训练数据均衡分配算法
```python
import heapq

def group_samples(n, m, lens):
    if m == 0:
        print(0)
        return
    
    lens.sort(reverse=True)
    load = [0] * n
    heapq.heapify(load)
    ans = 0
    
    for x in lens:
        cur = heapq.heappop(load)
        cur += x
        ans = max(ans, cur)
        heapq.heappush(load, cur)
    
    print(ans)

if __name__ == "__main__":
    n = int(input().strip())
    m = int(input().strip())
    lens = list(map(int, input().strip().split()))
    group_samples(n, m, lens)
```

### 题目6: P3479 标签样本数量
```python
import sys
from collections import Counter

def main():
    tokens = sys.stdin.read().strip().split()
    it = iter(tokens)
    k, m, n, s = int(next(it)), int(next(it)), int(next(it)), int(next(it))
    q = [float(next(it)) for _ in range(n)]
    
    X, y = [], []
    for _ in range(m):
        row = [float(next(it)) for _ in range(n + 1)]
        X.append(row[:n])
        y.append(int(row[-1]))

    # 计算距离
    dists = []
    for i in range(m):
        dist2 = sum((q[j] - X[i][j])**2 for j in range(n))
        dists.append((dist2, i))
    dists.sort()

    # 前 k 个邻居
    top_labels = [y[dists[i][1]] for i in range(min(k, m))]
    cnt = Counter(top_labels)
    max_freq = max(cnt.values())
    tie_labels = {lab for lab, c in cnt.items() if c == max_freq}

    # 选最近邻居的标签
    for i in range(min(k, m)):
        lab = y[dists[i][1]]
        if lab in tie_labels:
            print(lab, cnt[lab])
            return

if __name__ == '__main__':
    main()
```
