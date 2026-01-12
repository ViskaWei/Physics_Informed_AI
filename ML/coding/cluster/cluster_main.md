# Cluster 类题目汇总 [0/7 完成]

> 📊 **进度**: 0/7 完成 (0%)  
> 🔄 **最后更新**: 2026-01-04  
> 📁 **分类**: cluster (KMeans、DBSCAN、聚类分析、噪声识别、轮廓系数)

---

## 📋 题目总览

> 🔥 **重刷优先级**: 7 > 1 > 3 > 5 > 4 > 6 > 2（按难度和重要程度排序）

| 出题日期 | # | P编号 | 题目 | 难度 | 状态 | 完成日期 |
|----------|---|-------|------|------|------|----------|
| 2025-12-03 | 1 | P4519 | 智能客户分群与新用户定位(KMeans均衡分区版) ⭐ | 困难 | ❌ | - |
| 2025-11-19 | 2 | P4475 | 终端款型聚类识别 | 中等 | ❌ | - |
| 2025-10-17 | 3 | P4238 | 预训练模型智能告警聚类与故障诊断（并查集） | 中等 | ❌ | - |
| 2025-10-15 | 4 | P4228 | 基于二分KMeans的子网分割 | 中等 | ❌ | - |
| 2025-10-10 | 5 | P3874 | 数据聚类及噪声点识别（DBSCAN） | 中等 | ❌ | - |
| 2025-09-28 | 6 | P3842 | Yolo检测器中的anchor聚类（IOU距离） | 中等 | ❌ | - |
| 2025-09-24 | 7 | P3791 | 无线网络优化中的基站聚类分析（轮廓系数） ⭐ | 困难 | ❌ | - |

---

## 🔧 通用模板

### KMeans 基础模板
```python
def kmeans(X, K, iters=100):
    N, D = X.shape
    C = X[np.random.choice(N, K, replace=False)]   # 初始化中心

    for _ in range(iters):
        # E-step：分配簇（广播算距离）
        dist = ((X[:, None, :] - C[None, :, :])**2).sum(axis=2)
        labels = dist.argmin(axis=1)

        # M-step：更新中心
        C = np.array([X[labels == k].mean(axis=0) for k in range(K)])

    return C, labels
```
```python
import math
step_E = lambda C0: [min(range(K), key=lambda k: dist(XY[n],C0[k])) for n in range(N)]
def step_M(root):
    sumx = [0]* K; sumy = [0] * K; l = [0] * K;
    for (x,y), k in zip(XY, root): sumx[k] += x; sumy[k] += y; l[k] += 1;    
    return [[math.floor(sumx[k]/l[k]), math.floor(sumy[k]/l[k])] for k in range(K)]

def kmeans():
    C0 = XY[:K]; 
    for t in range(T):
        root = step_E(C0); C1 = step_M(root); 
        if not any([dist(C0[k], C1[k]) >= 1e-4 for k in range(K)]): break
        C0= C1
    return C0
```

### 并查集模板（用于连通图聚类）
```python
class UF():
    def __init__(self, n):
        self.n = n
        self.root = [i for i in range(n)]
        self.size = [1] * n

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: # 别漏！！
            sa, sb = self.size[ra], self.size[rb]
            if sa >= sb:
                self.size[ra] += sb
                self.root[rb] = ra
            else:
                self.size[rb] += sa
                self.root[ra] = rb
        
    def find(self, a):
        if self.root[a] != a:  self.root[a] = self.find(self.root[a])
        return self.root[a]
```

### DBSCAN 模板
```python
from collections import deque

def dbscan(points, eps, min_samples):
    n = len(points)
    eps2 = eps * eps
    # 预计算邻居
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if dist2(points[i], points[j]) <= eps2:
                neighbors[i].append(j)
    
    core = [len(neighbors[i]) >= min_samples for i in range(n)]
    labels = [-1] * n
    cluster_id = 0
    
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        # BFS 扩展
        labels[i] = cluster_id
        q = deque(neighbors[i])
        while q:
            j = q.popleft()
            if labels[j] == -1:
                labels[j] = cluster_id
                if core[j]:
                    q.extend(neighbors[j])
        cluster_id += 1
    
    return cluster_id, sum(1 for v in labels if v == -1)
```

---

## ⭐ 题目1: 智能客户分群与新用户定位(KMeans均衡分区版)（P4519）

- **难度**: 困难
- **标签**: KMeans, 均衡分区, 容量约束
- **源**: [core46#第3题-p4519](../AI_编程题_Python解答_核心46题.md#第3题-p4519)

### 题目描述

某大型企业需对客户进行自动化分群，要求每个群组容量尽可能均衡。需实现：
1. 采用 KMeans 变种聚类，将所有客户分为 K 个群组，且保证每组人数相等或只相差 1
2. 当人数无法均分时，将多出来的客户依次分配给聚类中心编号更小的组
3. 对于新客户，利用最终分群中心点，确定其最合适归属的群组

**关键规则**：
- 初始聚类中心：前 K 个客户的数据
- 分配时：距离相等则选中心编号更小者；若最近中心已满，分配给下一个最近的可收组
- 群组容量：若 N=11, K=3，则各组容量为 [4,4,3]
- 更新中心：各维度特征均值（向下取整）
- 终止条件：分配及聚类中心均未发生变化
- 输出中心：按字典序升序排列

### 样例
```
输入：
8 2 3
10 10
12 9
11 11
100 100
102 99
97 98
50 51
53 49
45 46

输出：
11 10
51 50
99 99
2
```

### 思路

1. **容量计算**：q = N // K，r = N % K，前 r 个组容量为 q+1，其余为 q
2. **分配阶段**：依次处理每个客户，计算到每个中心的距离，按 (距离, 编号) 排序，找到第一个未满的簇
3. **更新中心**：每维特征和除以簇内人数，向下取整
4. **新用户归属**：计算到每个排序后中心的距离，选最近的（距离相等选字典序小的）

### 复杂度

- **时间复杂度**：$O(T \cdot N \cdot K \cdot M)$，T 为迭代次数，M 为特征维度
- **空间复杂度**：$O(N \cdot M + K \cdot M)$

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目2: 终端款型聚类识别（P4475）

- **难度**: 中等
- **标签**: KMeans, 欧氏距离
- **源**: [core46#第2题-p4475](../AI_编程题_Python解答_核心46题.md#第2题-p4475)

### 题目描述

通过终端的 4 个特征（包间隔时长、连接持续时长、漫游前信号强度、漫游后信号强度），使用 KMeans 算法对终端型号进行聚类，输出各类型终端数量（从小到大排序）。

**规则**：
- 初始 k 个质心：数据集前 k 个点
- 距离函数：$d_{x,y} = \sqrt{\sum_{k=1}^{4}(x_k - y_k)^2}$
- 终止条件：质心移动值 < 1e-8 或达到最大迭代次数

### 样例
```
输入：
3 20 1000
0.11 0.79 0.68 0.97
1.0 0.8 0.13 0.33
... (共20个点)

输出：
4 6 10
```

### 思路

标准 KMeans 实现：
1. 初始化：前 k 个点作为质心
2. 分配：每个点归到最近质心
3. 更新：质心 = 簇内点的均值
4. 迭代直到收敛或达到最大次数

### 复杂度

- **时间复杂度**：$O(T \cdot n \cdot k \cdot d)$，d=4 为特征维度
- **空间复杂度**：$O(n \cdot d + k \cdot d)$

### 我的代码
```python
import sys
it = iter(sys.stdin.read().strip().split())
K, M, T = int(next(it)),int(next(it)),int(next(it));
X = [[float(next(it)) for _ in range(4)] for _ in range(M)]
import math
from collections import Counter
def dist(x, y):
    return math.sqrt(sum([(x[i]-y[i])**2 for i in range(4)]))
Efn = lambda C0: [min(range(K), key=lambda k: dist(X[im], C0[k])) for im in range(M)]
def Mfn(root):
    summ, ll = [[0]*4 for _ in range(K)], [0]*K;
    for x, r in zip(X,root): summ[r]=list(map(lambda a, b: a+b,summ[r],x)); ll[r]+=1; 
    return [[summ[k][i]/ll[k] for i in range(4)] for k in range(K)]
def kmeans():
    C0 = X[:K]
    for t in range(T):
        root=Efn(C0); C1=Mfn(root);
        if all([dist(C0[k],C1[k]) < 1e-8 for k in range(K)]): break
        C0 = C1
    return list(Counter(root).values())
a =kmeans()
a.sort()
print(*a)
```

---

## 题目3: 预训练模型智能告警聚类与故障诊断（P4238）

- **难度**: 中等
- **标签**: 余弦相似度, 并查集, 连通图聚类
- **源**: [core46#第2题-p4238](../AI_编程题_Python解答_核心46题.md#第2题-p4238) https://codefun2000.com/p/P4238

### 题目描述

通过语义向量（embedding）对告警信息进行聚类：
- 相似度阈值：余弦相似度 ≥ 0.95 时判定为语义相似
- 弱传递聚类：若 A 与 B 相似，B 与 C 相似，则 A、B、C 属于同一聚类
- 返回数量最大的聚类的告警数量

**特殊情况**：
- 输入为空列表：返回 0
- 向量维度不一致：返回 0

### 样例
```
输入：
1 1.0 0.0 0.0
2 0.99 0.01 0.0
3 0.0 1.0 0.0
4 0.0 1.0 0.01
5 0.1 0.0 0.0

输出：
3

输入（维度不一致）：
1 1.000000 0.000000 0.000000 0.000000
2 0.990000 0.010000 0.000000 0.980000
3 0.000000 1.000000 0.000000

输出：
0
```
说明：告警 1、2、5 构成一个聚类（相似度传递）

### 思路

1. **数据验证**：检查向量维度一致性
2. **并查集**：用于高效合并相似告警
3. **余弦相似度**：$\cos(A,B) = \frac{A \cdot B}{|A| \times |B|}$
4. **遍历所有告警对**：相似度 ≥ 0.95 则合并
5. **统计最大聚类大小**

### 复杂度

- **时间复杂度**：$O(n^2 \cdot d)$，n 为告警数，d 为向量维度
- **空间复杂度**：$O(n)$

### 我的代码
```python
class UF():
    def __init__(self, n):
        self.n = n
        self.root = [i for i in range(n)]
        self.size = [1] * n

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: # 别漏！！
            sa, sb = self.size[ra], self.size[rb]
            if sa >= sb:
                self.size[ra] += sb
                self.root[rb] = ra
            else:
                self.size[rb] += sa
                self.root[ra] = rb
        
    def find(self, a):
        if self.root[a] != a:  self.root[a] = self.find(self.root[a])
        return self.root[a]

def read():
    D = []; check = None; n=0
    for line in sys.stdin:
        a = list(map(float, line.strip().split()[1:]))
        if check is None:
            check = len(a)
        else:
            if check != len(a): return [], 0
        D.append(a)
    return D, len(D) if D is not None else 0

def main():
    D, n = read()
    if not D or n == 0: return 0
    k = len(D[0]); norm = [math.sqrt(sum([a * a for a in D[nn]])) for nn in range(n)]
    uf = UF(n)
    for i in range(n):
        for j in range(i+1, n):
            if norm[i] > 0 and norm[j] > 0 and sum([D[i][kk] * D[j][kk] for kk in range(k)]) >= 0.95 * norm[i] * norm[j]:
                uf.union(i, j) 
    return max(uf.size)
print(main())
```

---

## 题目4: 基于二分KMeans的子网分割（P4228）

- **难度**: 中等
- **标签**: 二分KMeans, SSE, 递归分割
- **源**: [core46#第3题-p4228](../AI_编程题_Python解答_核心46题.md#第3题-p4228)

### 题目描述

使用二分 KMeans 算法（Bi-KMeans）进行子网分割：
1. 首先将全网按 KMeans（K=2）聚类成两个子网
2. 每次迭代只选择一个子网进一步划分，选择标准是能最大程度降低全局 SSE
3. 直到子网个数达到预期数量

**规则**：
- 初始值选取：子网中 x 坐标最小和最大的两个站点
- 迭代终止：聚类结果相同（移动 < 1e-6）或迭代 1000 次
- SSE 计算：簇内所有站点到簇心距离的平方之和

### 样例
```
输入：
3
3
0 0
2 2
5 5

输出：
2 1
1 1 1
```

### 思路

1. **KMeans 二分**：选 x 坐标最小/最大的点作为初始中心
2. **SSE 计算**：$SSE = \sum_{x \in C} ||x - \mu||^2$
3. **贪心选择**：每次选择分割后 SSE 减少最多的簇
4. **迭代直到达到目标簇数**

### 复杂度

- **时间复杂度**：$O(N^2 \cdot K \cdot T)$，T 为迭代次数
- **空间复杂度**：$O(N)$

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目5: 数据聚类及噪声点识别（P3874）

- **难度**: 中等
- **标签**: DBSCAN, 密度聚类, 噪声点
- **源**: [core46#第2题-p3874](../AI_编程题_Python解答_核心46题.md#第2题-p3874)

### 题目描述

实现 DBSCAN 算法，识别簇的个数和噪声点的个数。

**核心概念**：
- **eps-邻域**：与点 P 距离小于 eps 的所有样本点集合
- **核心点**：eps 邻域内样本数 ≥ min_samples
- **密度可达**：通过核心点链接可以到达
- **噪声点**：不属于任何簇的样本点

### 样例
```
输入：
1 5 20
5.05 1.36
-8.19 -6.47
... (共20个点)

输出：
2 2
```
说明：2 个簇，2 个噪声点

### 思路

1. **预计算邻居**：两两距离判断（含自身）
2. **标记核心点**：邻居数 ≥ min_samples
3. **BFS 扩展**：从未访问的核心点开始扩展簇
4. **统计结果**：簇数和噪声点数

### 复杂度

- **时间复杂度**：$O(n^2 \cdot d)$
- **空间复杂度**：$O(n^2)$（邻接表）

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目6: Yolo检测器中的anchor聚类（P3842）

- **难度**: 中等
- **标签**: KMeans, IOU距离, 目标检测
- **源**: [core46#第2题-p3842](../AI_编程题_Python解答_核心46题.md#第2题-p3842)

### 题目描述

基于 k-means 聚类算法生成 YOLO 目标检测中的 Anchor 框：
- 距离度量：$d = 1 - IOU$
- IOU 计算：$IOU = \frac{intersection}{union}$
- 初始化：前 K 个框作为初始中心
- 更新时向下取整
- 终止条件：迭代次数 T 或新旧中心 d 值之和 < 1e-4

### 样例
```
输入：
12 4 20
12 23
34 21
... (共12个框)

输出：
133 94
121 27
36 22
12 50
```

### 思路

1. **IOU 计算**：$intersection = \min(w_1,w_2) \times \min(h_1,h_2)$
2. **KMeans 变体**：用 d = 1 - IOU 作为距离
3. **更新中心**：均值向下取整
4. **输出**：按面积从大到小排序

### 复杂度

- **时间复杂度**：$O(T \cdot N \cdot K)$
- **空间复杂度**：$O(N + K)$

### 我的代码
```python
import sys 
d = iter(sys.stdin.read().strip().split())
N, K, T = int(next(d)), int(next(d)), int(next(d)); 
XY = [[float(next(d)), float(next(d))] for n in range(N)]
import math
def dist(x, x0):
    inter = min(x[0], x0[0]) * min(x[1], x0[1]); union = x[0] * x[1] + x0[0] * x0[1] - inter
    return 1 - inter / (union + 1e-16)
step_E = lambda C0: [min(range(K), key=lambda k: dist(XY[n],C0[k])) for n in range(N)]
def step_M(root):
    sumx = [0]* K; sumy = [0] * K; l = [0] * K;
    for (x,y), k in zip(XY, root): sumx[k] += x; sumy[k] += y; l[k] += 1;    
    return [[math.floor(sumx[k]/l[k]), math.floor(sumy[k]/l[k])] for k in range(K)]

def kmeans():
    C0 = XY[:K]; 
    for t in range(T):
        root = step_E(C0); C1 = step_M(root); 
        if not any([dist(C0[k], C1[k]) >= 1e-4 for k in range(K)]): break
        C0= C1
    return C0

C1 = kmeans()
C1.sort(key= lambda x: x[0] * x[1], reverse=True)
print("\n".join([f"{C1[k][0]} {C1[k][1]}" for k in range(K)]))
```

---

## ⭐ 题目7: 无线网络优化中的基站聚类分析（P3791）

- **难度**: 困难
- **标签**: KMeans, 轮廓系数, 银行家舍入
- **源**: [core46#第2题-p3791](../AI_编程题_Python解答_核心46题.md#第2题-p3791)

### 题目描述

使用 K-Means 算法将基站划分为 k 个簇，通过计算每个簇的轮廓系数识别信号覆盖最差的簇（轮廓系数最低），并输出该簇中心作为新增基站位置。

**轮廓系数公式**：
- $a(i)$：点 i 与同簇其他点的平均距离
- $b(i)$：点 i 与最近其他簇内所有点的平均距离
- $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$
- 若簇内只有一个点，则 $s(i) = 0$

**KMeans 终止条件**：最大迭代 100 次或所有中心移动 ≤ 1e-6

### 样例
```
输入：
6 2
0 0
1 1
2 2
10 10
11 11
5 5

输出：
8.67,8.67
```
说明：簇 0 轮廓系数 0.82，簇 1 轮廓系数 0.35，输出簇 1 中心

### 思路

1. **KMeans 聚类**：标准实现
2. **轮廓系数计算**：
   - a(p)：与同簇点的平均距离
   - b(p)：与最近其他簇的平均距离
3. **找最差簇**：轮廓系数最低
4. **输出**：银行家舍入保留两位小数

### 复杂度

- **时间复杂度**：$O(n^2)$（轮廓系数计算）
- **空间复杂度**：$O(n)$

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 📌 易错点总结

1. **初始化方式**：大多数题目用前 k 个点，部分题目用 x 坐标最小/最大的点
2. **距离度量**：欧氏距离 vs IOU 距离 vs 余弦相似度
3. **终止条件**：移动阈值、最大迭代次数、分配不变
4. **更新方式**：均值 vs 向下取整
5. **容量约束**：均衡分区需要考虑容量限制
6. **轮廓系数**：单点簇的轮廓系数为 0
7. **舍入方式**：银行家舍入（HALF_EVEN）vs 四舍五入
8. **传递性聚类**：并查集处理相似度传递

---

## 🔗 相关文件

- 源文件：`../AI_编程题_Python解答_核心46题.md`
- 索引：`../ai_core46_index.md`

---

## 📝 代码答案

### 题目1: 智能客户分群与新用户定位(KMeans均衡分区版)（P4519）

```python
import sys
from math import inf

def dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))

def balanced_kmeans(customers, K):
    N, M = len(customers), len(customers[0])
    base, rem = N // K, N % K
    capacity = [base + (1 if i < rem else 0) for i in range(K)]
    centers = [customers[i][:] for i in range(K)]
    assign = [-1] * N
    
    while True:
        new_assign = [-1] * N; sizes = [0] * K
        for i in range(N):
            dist_list = sorted([(dist2(customers[i], centers[k]), k) for k in range(K)])
            for _, k in dist_list:
                if sizes[k] < capacity[k]:
                    new_assign[i] = k; sizes[k] += 1; break
        
        new_centers = [[0] * M for _ in range(K)]; counts = [0] * K
        for i in range(N):
            c = new_assign[i]; counts[c] += 1
            for d in range(M): new_centers[c][d] += customers[i][d]
        for k in range(K):
            for d in range(M): new_centers[k][d] //= counts[k]
        
        if new_assign == assign and new_centers == centers: break
        centers, assign = new_centers, new_assign
    return centers

N, M, K = map(int, input().split())
customers = [[int(x) for x in input().split()] for _ in range(N)]
new_cust = [int(x) for x in input().split()]
centers = balanced_kmeans(customers, K); centers.sort()
for c in centers: print(" ".join(map(str, c)))
best = min(range(K), key=lambda i: (dist2(new_cust, centers[i]), centers[i]))
print(best + 1)
```

### 题目2: 终端款型聚类识别（P4475）

```python
import sys

def dist2(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(4))

def kmeans(points, k, max_iter):
    m = len(points)
    centers = [points[i][:] for i in range(k)]
    cluster_size = [0] * k
    
    for _ in range(max_iter):
        cluster_size = [0] * k
        sums = [[0.0] * 4 for _ in range(k)]
        
        for i in range(m):
            best_idx = min(range(k), key=lambda j: dist2(points[i], centers[j]))
            cluster_size[best_idx] += 1
            for t in range(4): sums[best_idx][t] += points[i][t]
        
        max_move = 0.0
        for j in range(k):
            new_center = [sums[j][t] / cluster_size[j] if cluster_size[j] > 0 
                          else centers[j][t] for t in range(4)]
            max_move = max(max_move, dist2(centers[j], new_center))
            centers[j] = new_center
        
        if max_move < 1e-8: break
    
    return cluster_size

data = sys.stdin.read().strip().split()
k, m, max_iter = int(data[0]), int(data[1]), int(data[2])
points = [[float(data[3 + i*4 + j]) for j in range(4)] for i in range(m)]
sizes = kmeans(points, k, max_iter)
sizes.sort()
print(" ".join(map(str, sizes)))
```

### 题目3: 预训练模型智能告警聚类与故障诊断（P4238）

```python
import sys
import math
from collections import Counter

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

def solve(alerts):
    if not alerts: return 0
    n, dim = len(alerts), len(alerts[0][1])
    if any(len(a[1]) != dim for a in alerts): return 0
    
    parent = list(range(n))
    def find(x):
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        px, py = find(x), find(y)
        if px != py: parent[px] = py
    
    for i in range(n):
        for j in range(i + 1, n):
            if cosine_similarity(alerts[i][1], alerts[j][1]) >= 0.95:
                union(i, j)
    
    return max(Counter(find(i) for i in range(n)).values())

alerts = []
for line in sys.stdin:
    parts = line.strip().split()
    if parts: alerts.append((parts[0], [float(x) for x in parts[1:]]))
print(solve(alerts))
```

### 题目4: 基于二分KMeans的子网分割（P4228）

```python
import numpy as np

def calculate_sse(points):
    if len(points) == 0: return 0
    center = np.mean(points, axis=0)
    return np.sum((points - center) ** 2)

def kmeans_split(points):
    if len(points) <= 1: return [points]
    min_idx, max_idx = np.argmin(points[:, 0]), np.argmax(points[:, 0])
    centers = np.array([points[min_idx], points[max_idx]])
    
    for _ in range(1000):
        distances = np.sum((points[:, np.newaxis] - centers) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([
            np.mean(points[labels == i], axis=0) if np.any(labels == i) else centers[i]
            for i in range(2)])
        if np.sum((centers - new_centers) ** 2) < 1e-12: break
        centers = new_centers
    
    distances = np.sum((points[:, np.newaxis] - centers) ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    return [points[labels == 0], points[labels == 1]]

def bi_kmeans(points, n):
    clusters = [points]; results = []
    clusters = kmeans_split(clusters[0])
    results.append(sorted([len(c) for c in clusters], reverse=True))
    
    while len(clusters) < n:
        max_sse_reduction, best_idx = -1, -1
        for i, c in enumerate(clusters):
            if len(c) <= 1: continue
            current_sse = calculate_sse(c)
            new_clusters = kmeans_split(c)
            new_sse = sum(calculate_sse(nc) for nc in new_clusters)
            if current_sse - new_sse > max_sse_reduction:
                max_sse_reduction = current_sse - new_sse
                best_idx = i
        
        new_clusters = kmeans_split(clusters[best_idx])
        clusters = clusters[:best_idx] + new_clusters + clusters[best_idx + 1:]
        results.append(sorted([len(c) for c in clusters], reverse=True))
    return results

n = int(input()); m = int(input())
points = np.array([list(map(int, input().split())) for _ in range(m)], dtype=float)
for r in bi_kmeans(points, n): print(' '.join(map(str, r)))
```

### 题目5: 数据聚类及噪声点识别（P3874）

```python
from collections import deque

def dist2(a, b):
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))

def dbscan(points, eps, min_samples):
    n = len(points)
    if n == 0: return 0, 0
    eps2 = eps * eps
    
    neighbors = [[j for j in range(n) if dist2(points[i], points[j]) <= eps2] for i in range(n)]
    core = [len(neighbors[i]) >= min_samples for i in range(n)]
    labels = [-1] * n; visited = [False] * n; cluster_id = 0
    
    for i in range(n):
        if visited[i] or not core[i]: continue
        visited[i] = True; labels[i] = cluster_id
        q = deque(neighbors[i])
        while q:
            j = q.popleft()
            if labels[j] == -1: labels[j] = cluster_id
            if not visited[j]:
                visited[j] = True
                if core[j]: q.extend(neighbors[j])
        cluster_id += 1
    
    return cluster_id, sum(1 for v in labels if v == -1)

data = input().split()
eps, min_samples, x = float(data[0]), int(data[1]), int(data[2])
points = [list(map(float, input().split())) for _ in range(x)]
clusters, noise = dbscan(points, eps, min_samples)
print(f"{clusters} {noise}")
```

### 题目6: Yolo检测器中的anchor聚类（P3842）

```python
import sys
import math

def iou_wh(w1, h1, w2, h2):
    inter = min(w1, w2) * min(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-16)

def kmeans_anchors(boxes, K, T):
    centers = [(float(boxes[i][0]), float(boxes[i][1])) for i in range(K)]
    n = len(boxes)
    
    for _ in range(T):
        assign = [min(range(K), key=lambda k: 1.0 - iou_wh(w, h, centers[k][0], centers[k][1])) 
                  for w, h in boxes]
        
        sums = [[0.0, 0.0] for _ in range(K)]; cnts = [0] * K
        for (w, h), k in zip(boxes, assign):
            sums[k][0] += w; sums[k][1] += h; cnts[k] += 1
        
        new_centers = [(math.floor(sums[k][0]/cnts[k]), math.floor(sums[k][1]/cnts[k])) 
                       if cnts[k] > 0 else centers[k] for k in range(K)]
        
        change = sum(1.0 - iou_wh(centers[k][0], centers[k][1], new_centers[k][0], new_centers[k][1]) 
                     for k in range(K))
        centers = [(float(c[0]), float(c[1])) for c in new_centers]
        if change < 1e-4: break
    
    final = [(int(w), int(h)) for w, h in centers]
    final.sort(key=lambda x: x[0] * x[1], reverse=True)
    return final

data = sys.stdin.read().strip().split()
N, K, T = int(data[0]), int(data[1]), int(data[2])
boxes = [(float(data[3+i*2]), float(data[4+i*2])) for i in range(N)]
for w, h in kmeans_anchors(boxes, K, T): print(f"{w} {h}")
```

### 题目7: 无线网络优化中的基站聚类分析（P3791）

```python
import sys
from decimal import Decimal, ROUND_HALF_EVEN

def kmeans(pts, k):
    n = len(pts)
    centers = [list(pts[i]) for i in range(k)]
    labels = [0] * n
    
    for _ in range(100):
        for i in range(n):
            labels[i] = min(range(k), key=lambda c: (pts[i][0]-centers[c][0])**2 + (pts[i][1]-centers[c][1])**2)
        
        sx, sy, cnt = [0.0]*k, [0.0]*k, [0]*k
        for i in range(n):
            c = labels[i]; sx[c] += pts[i][0]; sy[c] += pts[i][1]; cnt[c] += 1
        
        moved = 0.0
        for c in range(k):
            nx, ny = (sx[c]/cnt[c], sy[c]/cnt[c]) if cnt[c] > 0 else (centers[c][0], centers[c][1])
            moved += abs(nx - centers[c][0]) + abs(ny - centers[c][1])
            centers[c] = [nx, ny]
        if moved <= 1e-6: break
    
    return labels, centers

def silhouette(pts, labels, k):
    n = len(pts)
    groups = [[] for _ in range(k)]
    for i, c in enumerate(labels): groups[c].append(i)
    
    def dist(i, j):
        return ((pts[i][0]-pts[j][0])**2 + (pts[i][1]-pts[j][1])**2) ** 0.5
    
    avg = [0.0] * k
    for c in range(k):
        idx = groups[c]
        if not idx: continue
        ssum = 0.0
        for i in idx:
            a = sum(dist(i, j) for j in idx if j != i) / (len(idx)-1) if len(idx) > 1 else 0.0
            b = min((sum(dist(i, j) for j in groups[c2]) / len(groups[c2]) 
                     for c2 in range(k) if c2 != c and groups[c2]), default=float('inf'))
            m = max(a, b)
            ssum += (b - a) / m if m > 0 else 0.0
        avg[c] = ssum / len(idx)
    return avg

def rnd2(v):
    return f"{Decimal(str(v)).quantize(Decimal('0.00'), rounding=ROUND_HALF_EVEN):.2f}"

data = list(map(float, sys.stdin.read().strip().split()))
n, k = int(data[0]), int(data[1])
pts = [(data[2+i*2], data[3+i*2]) for i in range(n)]
labels, centers = kmeans(pts, k)
sil = silhouette(pts, labels, k)
bad = min(range(k), key=lambda c: (sil[c], c))
print(f"{rnd2(centers[bad][0])},{rnd2(centers[bad][1])}")
```
