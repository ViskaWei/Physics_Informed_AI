# Reg 类题目汇总 [4/6 完成]

> 📊 **进度**: 4/6 完成 (67%)  
> 🔄 **最后更新**: 2026-01-06  
> 📁 **分类**: reg (线性回归、逻辑回归、故障预测)

---

## 📋 题目总览

> 🔥 **重刷优先级**: 1 > 3 > 2 > 5 > 4 > 6（按难度和重要程度排序）

| 出题日期 | # | P编号 | 题目 | 难度 | 状态 | 完成日期 |
|----------|---|-------|------|------|------|----------|
| 2025-12-17 | 1 | P4532 | 使用线性回归预测手机售价 ⭐ | 中等 | ✅ | 2026-01-05 |
| 2025-11-06 | 2 | P4447 | 医疗诊断模型的训练与更新 | 中等 | ✅ | 2026-01-05 |
| 2025-10-29 | 3 | P4344 | 商品购买预测（逻辑回归+L2正则） | 中等 | ✅ | 2026-01-05 |
| 2025-10-10 | 4 | P3872 | 基于逻辑回归的意图分类器 | 中等 | ✅ | 2026-01-05 |
| 2025-09-18 | 5 | P3719 | 数据中心水温调节档位决策 | 中等 | ❌ | - |
| 2025-09-03 | 6 | P3552 | 云存储设备故障预测（数据清洗+逻辑回归） | 中等 | ❌ | - |

---

## 🔧 通用模板

### 线性回归模板（正规方程 + 高斯消元）
```python
H1=X @ W1; H2=H1 @ W2; Y=H2.mean(0)
dY=Y - Y0; loss=(dY**2).mean()
gY=2/K*dY; gH2=np.ones((L,1)) * gY/L;
gW2 = H1.T @ gH2; gH1 =gH2 @ W2.T; gW1 = X.T @ gH1;
W2 -= eta * gW2; W1 -= eta * gW1;
# linear solve
X, X1 = list(map(lambda x: np.hstack([np.ones((len(x), 1)), x]), [X,X1])); X,Y = X[:,:4],X[:,-1]
W = np.linalg.solve(X.T @ X, X.T @ Y)   # (4,)
y_pred = np.rint(X1 @ W).astype(int)

def linear_regression(X, y):
    """
    最小二乘法线性回归
    X: n x d 特征矩阵（不含偏置列）
    y: n x 1 标签
    返回: (d+1,) 权重向量 [w0, w1, ..., wd]
    """
    n, d = len(X), len(X[0])
    # 添加偏置列
    X_aug = [[1.0] + list(X[i]) for i in range(n)]
    
    # 计算 X^T X 和 X^T y
    dim = d + 1
    XTX = [[0.0] * dim for _ in range(dim)]
    XTy = [0.0] * dim
    
    for i in range(n):
        for a in range(dim):
            XTy[a] += X_aug[i][a] * y[i]
            for b in range(dim):
                XTX[a][b] += X_aug[i][a] * X_aug[i][b]
    
    # 高斯消元求解
    A = [XTX[i] + [XTy[i]] for i in range(dim)]
    for i in range(dim):
        pivot = A[i][i]
        for j in range(i, dim + 1):
            A[i][j] /= pivot
        for k in range(dim):
            if k != i:
                factor = A[k][i]
                for j in range(i, dim + 1):
                    A[k][j] -= factor * A[i][j]
    
    return [A[i][dim] for i in range(dim)]
```

### 逻辑回归模板
```python
import math

def sigmoid(z):
    """数值稳定的 sigmoid"""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def logistic_regression(X, y, lr=0.01, epochs=100, lam=0.0):
    """
    批量梯度下降训练逻辑回归
    X: n x d, y: n (0/1)
    返回: w (d,), b (scalar)
    """
    n, d = len(X), len(X[0])
    w = [0.0] * d
    b = 0.0
    
    for _ in range(epochs):
        grad_w = [0.0] * d
        grad_b = 0.0
        for i in range(n):
            z = b + sum(w[j] * X[i][j] for j in range(d))
            p = sigmoid(z)
            diff = p - y[i]
            for j in range(d):
                grad_w[j] += diff * X[i][j]
            grad_b += diff
        # 加 L2 正则
        for j in range(d):
            grad_w[j] = grad_w[j] / n + (lam / n) * w[j]
        grad_b /= n
        # 更新
        for j in range(d):
            w[j] -= lr * grad_w[j]
        b -= lr * grad_b
    
    return w, b
```

---

## 题目1: 使用线性回归预测手机售价（P4532）⭐

- **难度**: 中等
- **核心**: 正规方程 + 高斯消元
- **源**: [core46#第2题-p4532](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- 给定 K 部手机的三项评分（硬件能力、系统流畅度、AI能力）和售价
- 使用最小二乘法建立线性回归模型：$y = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3$
- 预测 N 部待上市手机的售价

### 关键规则
1. 使用正规方程法：$(X^T X) W = X^T Y$
2. 高斯消元求解 4×4 线性方程组
3. **四舍五入取整数**

### 样例
```
输入:
10
86 99 20 3595 175 171 90 6596 ...（K=10个手机，每个4个数字）
2
159 135 173 120 144 59

输出:
7116 5120
```

### 思路
1. 构造增广矩阵 $X$（添加全1偏置列）
2. 计算 $X^T X$（4×4）和 $X^T Y$（4×1）
3. 高斯消元求解权重 $W$
4. 对新数据预测并四舍五入

### 复杂度
- 时间: O(K + N)（矩阵运算为常数级 4×4）
- 空间: O(1)

### 我的代码
```python
import sys
it = iter(sys.stdin.read().strip().split())
K = int(next(it)); X=[[int((next(it))) for _ in range(4)] for _ in range(K)]; M=int(next(it)); X1=[[int((next(it))) for _ in range(3)] for _ in range(M)];
import numpy as np
X, X1 = list(map(lambda x: np.hstack([np.ones((len(x), 1)), x]), [X,X1])); X,Y = X[:,:4],X[:,-1]
W = np.linalg.solve(X.T @ X, X.T @ Y)   # (4,)
y_pred = np.rint(X1 @ W).astype(int)
print(*y_pred)
```

---

## 题目2: 医疗诊断模型的训练与更新（P4447）

- **难度**: 中等
- **核心**: 两层 MLP + 反向传播 + SGD
- **源**: [core46#第2题-p4447](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- 输入：L 个时刻的症状序列（每时刻 D 维特征）
- 模型：两层 MLP（无偏置，无激活函数）
  - 第一层：$h_t = x_t W_{mlp}$（D×D）
  - 分类层：$p_t = h_t W_{cls}$（D×K）
- 输出取序列平均：$\hat{y} = \frac{1}{L}\sum_t p_t$
- 损失：MSE = $\frac{1}{K}\sum_i (\hat{y}_i - y_i)^2$
- SGD 更新权重

### 关键规则
1. 无 softmax、无偏置、无激活函数
2. 输出格式：逗号分隔，保留 2 位小数
3. 梯度：利用链式法则推导

### 样例
```
输入:
4,2,5,1.0
0.10,0.20,0.30,0.25,0.15
0.0,1.0,-1.5,2.5,3.0,-0.5,0.7,0.3
0.6,-0.4,0.2,0.9
0.5,0.1,-0.3,0.8,0.0,-0.2,0.4,0.6,-0.5,1.0

输出:
0.14,0.26,0.16,0.13,0.52
0.04
0.61,-0.48,0.21,0.78
0.49,0.09,-0.27,0.82,-0.07,-0.21,0.39,0.63,-0.48,0.92
```

### 思路
1. 前向：计算 $h_t$、$p_t$、$\hat{y}$
2. 损失：MSE
3. 反向：$g = \frac{2}{K}(\hat{y} - y)$，链式求 $\nabla W_{cls}$ 和 $\nabla W_{mlp}$
4. SGD 更新

### 复杂度
- 时间: O(L·D² + L·D·K)
- 空间: O(L·D + D² + D·K)

### 我的代码
```python
import sys 
for i, line in enumerate(sys.stdin):
    if i == 0: 
        it=iter(line.strip().split(','))
        L,D,K,eta=int(next(it)),int(next(it)),int(next(it)),float(next(it))
    if i == 1:
        Y0K = list(map(float, line.strip().split(',')))
    if i == 2:
        LD_all = list(map(float, line.strip().split(',')))
        LD = [LD_all[i*D : (i+1)*D] for i in range(L) ]
    if i == 3:
        Wdd_all = list(map(float, line.strip().split(',')))
        Wdd = [Wdd_all[i*D : (i+1)*D] for i in range(D) ]
    if i == 4:
        Wdk_all = list(map(float, line.strip().split(',')))
        Wdk = [Wdk_all[i*K : (i+1)*K] for i in range(D) ]
import numpy as np
Y0, X, W1, W2 = map(lambda a: np.asarray(a, float), [Y0K, LD, Wdd, Wdk])
H1=X @ W1; H2=H1 @ W2; Y=H2.mean(0)
dY=Y - Y0; loss=(dY**2).mean()
gY=2/K*dY; gH2=np.ones((L,1)) * gY/L;
gW2 = H1.T @ gH2; gH1 =gH2 @ W2.T; gW1 = X.T @ gH1;
W2 -= eta * gW2; W1 -= eta * gW1;

print(",".join(f"{y:.2f}" for y in Y))
print(f"{loss:.2f}")
print(",".join(f"{x:.2f}"  for x in W1.ravel()))
print(",".join(f"{x:.2f}"  for x in W2.ravel()))
```

---

## 题目3: 商品购买预测（P4344）

- **难度**: 中等
- **核心**: 逻辑回归 + L2正则 + 梯度下降
- **源**: [core46#第3题-p4344](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- 特征：年龄、月收入、浏览时长（3维）
- 标签：是否购买（0/1）
- 模型：$p = \sigma(w^T x + b)$
- 损失：交叉熵 + L2正则：$\frac{\lambda}{2n}\|w\|_2^2$

### 关键规则
1. 批量梯度下降
2. 终止条件：达到 max_iter 或损失变化 < tol
3. 阈值 0.5 判断类别
4. **输出格式**：`类别 概率`，概率保留 4 位小数

### 样例
```
输入:
10 1000 0.01 0.1 0.0001
25 8 5 0
30 15 15 1
...
3
32 18 12
48 33 22
62 48 10

输出:
1 0.7539
1 0.9966
0 0.0004
```

### 思路
1. 初始化 w=0, b=0
2. 每次迭代计算梯度（含 L2 正则项）
3. 更新参数
4. 收敛检查

### 复杂度
- 时间: O(max_iter × n × d)
- 空间: O(n × d)

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目4: 基于逻辑回归的意图分类器（P3872）

- **难度**: 中等
- **核心**: One-hot 编码 + 逻辑回归 + SGD
- **源**: [core46#第3题-p3872](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- 输入：由大写字母 ABCDEFG 组成的字符串
- One-hot 编码：长度 7，字母存在则为 1
- 逻辑回归：$p = \sigma(w \cdot x + b)$
- 训练：学习率 0.1，轮数 20，batch=1（SGD）

### 关键规则
1. 字母只记录是否存在，**不计次数**
2. 阈值 0.5：p > 0.5 输出 1，否则 0
3. 初始 w、b 全为 0

### 样例
```
输入:
10 2
CBG 0
AFE 0
FGD 1
...
DBA
DAD

输出:
0
0
```

### 思路
1. 编码：将字符串转为 7 维 one-hot
2. SGD 训练 20 轮
3. 预测并输出

### 复杂度
- 时间: O(20 × N × 7) = O(N)
- 空间: O(7)

### 我的代码
```python
import sys
it = iter(sys.stdin.read().strip().split())
N,M=int(next(it)),int(next(it));D=[[next(it), int(next(it))] for _ in range(N)];S= [next(it) for _ in range(M)]
Y =[0]*N; X=[[0]*7 for _ in range(N)];Z=[[0]*7 for _ in range(M)]
def encode(s,TT):
    for sj in s: TT[ord(sj)-ord('A')]=1
for i, (s, y) in enumerate(D): Y[i] = y; encode(s, X[i])
_=[encode(s, z) for s,z in zip(S,Z)]
import numpy as np
X, Y, Z = list(map(np.array, [X,Y,Z]))
sigmoid = lambda z: 1/(1+np.exp(-np.clip(z,-50,50)))
W=np.zeros(7);b=0;lr=0.1
for e in range(20):
    for x,y in zip(X,Y):
        loss = sigmoid(W @ x + b) - y
        W -= lr * (loss) * x; b -= lr * (loss)
out =[int(sigmoid(W @ z + b) > 0.5) for z in Z]
print(*out, sep='\n')
```

---

## 题目5: 数据中心水温调节档位决策（P3719）

- **难度**: 中等
- **核心**: 回归/决策问题（非 DP）
- **源**: [core46#第3题-p3719](../AI_编程题_Python解答_核心46题.md)

### 题目描述
TODO

### 思路
TODO

### 复杂度
TODO

```
10 3 5 2 3 10
-41.53 13.54 -51.57 -0.90 -17.71 31.90 -24.43 88.34 74.12 -47.06 
23.09 -22.95 -28.74 21.31 -19.01 -31.90 -14.89 -78.25 -72.45 -15.58 
7.52 -94.70 -16.86 -13.02 -46.77 11.37 -38.38 -85.37 -52.83 -63.79 
-28.32 21.18 99.19 49.71 -34.97 -11.34 -37.89 47.18 9.20 -81.93 
-41.51 -32.35 -92.51 -32.15 -13.43 -23.66 20.44 -75.82 44.22 8.53 
47.37 9.90 -63.44 -59.53 -83.86 75.39 -25.82 65.88 -34.71 -19.44 
76.63 40.99 -59.21 68.59 -67.38 -42.16 -50.53 -97.76 -6.43 19.52 
-87.62 10.46 45.26 -25.53 -33.45 19.97 -10.22 18.23 -3.22 77.33 
-60.09 -59.57 1.14 38.16 11.20 72.59 67.72 1.23 20.46 10.16 
-91.39 89.93 -39.10 -46.39 -75.25 -60.65 -35.24 -42.82 -35.79 72.72 
87.64 -95.78 62.83 76.01 -64.21 -4.10 -57.60 -28.13 19.70 -53.57 
-44.83 -60.03 -21.49 8.07 92.63 -49.45 -92.33 82.55 66.42 88.41 
-14.72 -25.79 -61.13 -61.84 12.56 -9.34 0.41 12.45 -97.73 -28.63 
26.63 -53.13 -69.54 -36.86 -60.05 69.01 -40.20 14.10 4.46 -96.35 
-65.04 49.95 76.52 -59.36 -5.32 6.41 -19.52 70.04 -60.98 -69.90 
-12.80 -61.39 90.16 -24.62 30.56 -26.98 8.65 -80.11 -36.63 55.30 
-36.89 -77.94 -35.68 68.50 -82.66 -90.73 -58.08 21.55 -41.73 -46.05 
-84.69 -79.97 86.37 -41.67 -28.87 -69.67 6.72 73.15 -11.07 -38.84 
-89.53 -46.11 6.77 86.64 12.59 -81.60 -48.59 -99.16 73.70 -56.71 
-16.67 -86.89 -89.41 90.62 -57.18 78.30 -81.28 -76.13 40.99 43.49 


0
0
0
0
0
2
0
0
0
1
```
### 我的代码
```python
# X, y, X1, n, k = read()
# y = y.astype(int)
# N = len(X)
# eps = 1e-8

# # 标准化（只用训练集统计）
# mu = X.mean(0) if N else np.zeros(n)
# d = X - mu
# sig = np.sqrt((d*d).sum(0) / (N-1 if N > 1 else 1)) if N else np.ones(n)
# sig = np.where(sig < eps, 1.0, sig)
# X  = (X  - mu) / (sig + eps)
# X1 = (X1 - mu) / (sig + eps)

# def softmax(z):
#     z = z - z.max(1, keepdims=True)
#     e = np.exp(z)
#     return e / (e.sum(1, keepdims=True) + eps)

# W = np.zeros((n, k))
# b = np.zeros(k)
# lr, reg = 0.1, 1e-4

# for e in range(600 if N else 0):
#     P = softmax(X @ W + b)
#     P[np.arange(N), y] -= 1
#     dZ = P / N
#     W -= lr * (X.T @ dZ + reg * W)
#     b -= lr * dZ.sum(0)
#     if (e + 1) % 150 == 0: lr *= 0.9
# # print(W.round(),b)
# Y1 = softmax(X1 @ W + b).argmax(1)
# print(*Y1, sep="\n")
```

---

## 题目6: 云存储设备故障预测（P3552）

- **难度**: 中等
- **核心**: 数据清洗 + 逻辑回归
- **源**: [core46#第3题-p3552](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- 5 维特征：写入次数、读取次数、写延迟、读延迟、使用年限
- 数据清洗：
  - 缺失值（NaN）→ 用均值填充
  - 异常值 → 用中位数替换
- 逻辑回归：100 次迭代，学习率 0.01

### 异常值规则
| 特征 | 异常条件 |
|------|---------|
| 写入/读取次数 | < 0 |
| 延迟 | < 0 或 > 1000 |
| 年限 | < 0 或 > 20 |

### 样例
```
输入:
5
dev1,NaN,-50,NaN,-2.0,25,0
dev2,180,90,18.0,9.0,4,0
...
2
dev_predict1,80,40,NaN,2.0,2,0
dev_predict2,210,105,18.0,9.8,4,0

输出:
0
0
```

### 思路
1. 按列统计有效值的均值和中位数
2. 缺失值填均值，异常值填中位数
3. 逻辑回归训练 100 次
4. 预测

### 复杂度
- 时间: O(N log N)（排序求中位数）+ O(100 × N × 5)
- 空间: O(N)

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 📌 易错点总结

1. **正规方程的高斯消元**：注意主元归一化和消元顺序
2. **逻辑回归的 sigmoid 溢出**：使用数值稳定写法（分 z ≥ 0 和 z < 0）
3. **L2 正则的梯度**：$\frac{\partial}{\partial w_j}(\frac{\lambda}{2n}\|w\|^2) = \frac{\lambda}{n}w_j$
4. **数据清洗顺序**：先处理缺失值，再处理异常值
5. **MLP 反向传播**：注意矩阵乘法的转置方向
6. **One-hot 编码**：只记录存在性，不计次数

---

## 🔗 相关文件

- 源文件：`../AI_编程题_Python解答_核心46题.md`
- 索引：`../ai_core46_index.md`

---

## 📝 代码答案

### 题目1: P4532 线性回归预测手机售价
```python
import sys
import math

def linear_regression_predict(K, train_data, N, test_data):
    # X 是 K×4 矩阵，Y 是 K×1 向量
    X = []
    Y = []
    idx = 0
    for _ in range(K):
        x1, x2, x3, y = train_data[idx:idx+4]
        idx += 4
        X.append([1.0, x1, x2, x3])
        Y.append(y)

    # 计算 X^T * X 和 X^T * Y
    XT_X = [[0.0]*4 for _ in range(4)]
    XT_Y = [0.0]*4
    for i in range(K):
        for a in range(4):
            XT_Y[a] += X[i][a] * Y[i]
            for b in range(4):
                XT_X[a][b] += X[i][a] * X[i][b]

    # 高斯消元法求解 (X^T X)W = X^T Y
    # 构造增广矩阵
    A = [XT_X[i] + [XT_Y[i]] for i in range(4)]

    # 消元
    for i in range(4):
        pivot = A[i][i]
        for j in range(i, 5):
            A[i][j] /= pivot
        for k in range(4):
            if k != i:
                factor = A[k][i]
                for j in range(i, 5):
                    A[k][j] -= factor * A[i][j]

    W = [A[i][4] for i in range(4)]

    # 预测
    res = []
    idx = 0
    for _ in range(N):
        x1, x2, x3 = test_data[idx:idx+3]
        idx += 3
        y_pred = W[0] + W[1]*x1 + W[2]*x2 + W[3]*x3
        res.append(str(int(round(y_pred))))
    return res

def main():
    data = sys.stdin.read().strip().split()
    pos = 0
    K = int(data[pos]); pos += 1
    train_data = list(map(int, data[pos:pos+4*K]))
    pos += 4*K
    N = int(data[pos]); pos += 1
    test_data = list(map(int, data[pos:pos+3*N]))

    ans = linear_regression_predict(K, train_data, N, test_data)
    print(" ".join(ans))

if __name__ == "__main__":
    main()
```

### 题目2: P4447 医疗诊断模型的训练与更新
```python
import sys
import ast

def parse_line(line: str):
    return list(ast.literal_eval("[" + line.strip() + "]"))

def solve_once(L, D, K, eta, y_true, seq_flat, Wmlp_flat, Wcls_flat):
    X = [seq_flat[i*D:(i+1)*D] for i in range(L)]
    Wmlp = [Wmlp_flat[i*D:(i+1)*D] for i in range(D)]
    Wcls = [Wcls_flat[i*K:(i+1)*K] for i in range(D)]

    # 前向
    H_sum = [0.0]*D
    P_avg = [0.0]*K
    for t in range(L):
        x = X[t]
        h = [sum(x[d] * Wmlp[d][j] for d in range(D)) for j in range(D)]
        for j in range(D):
            H_sum[j] += h[j]
        p = [sum(h[j] * Wcls[j][k] for j in range(D)) for k in range(K)]
        for k in range(K):
            P_avg[k] += p[k]
    P_avg = [v / L for v in P_avg]

    # 损失
    loss = sum((P_avg[k] - y_true[k])**2 for k in range(K)) / K

    # 反向
    g = [(2.0 / K) * (P_avg[k] - y_true[k]) for k in range(K)]
    H_bar = [v / L for v in H_sum]
    dWcls = [[H_bar[j] * g[k] for k in range(K)] for j in range(D)]
    v = [sum(Wcls[j][k] * g[k] for k in range(K)) for j in range(D)]
    X_sum = [sum(X[t][d] for t in range(L)) for d in range(D)]
    X_bar = [v_ / L for v_ in X_sum]
    dWmlp = [[X_bar[i] * v[j] for j in range(D)] for i in range(D)]

    # SGD 更新
    for i in range(D):
        for j in range(D):
            Wmlp[i][j] -= eta * dWmlp[i][j]
    for j in range(D):
        for k in range(K):
            Wcls[j][k] -= eta * dWcls[j][k]

    Wmlp_new = [Wmlp[i][j] for i in range(D) for j in range(D)]
    Wcls_new = [Wcls[j][k] for j in range(D) for k in range(K)]
    return P_avg, loss, Wmlp_new, Wcls_new

def fmt_line(arr):
    return ",".join(f"{x:.2f}" for x in arr)

def main():
    lines = sys.stdin.read().strip().splitlines()
    L, D, K, eta = parse_line(lines[0])
    L, D, K = int(L), int(D), int(K)
    y_true = parse_line(lines[1])
    seq = parse_line(lines[2])
    Wmlp = parse_line(lines[3])
    Wcls = parse_line(lines[4])

    P_avg, loss, Wmlp_new, Wcls_new = solve_once(L, D, K, eta, y_true, seq, Wmlp, Wcls)
    print(fmt_line(P_avg))
    print(f"{loss:.2f}")
    print(fmt_line(Wmlp_new))
    print(fmt_line(Wcls_new))

if __name__ == "__main__":
    main()
```

### 题目3: P4344 商品购买预测
```python
import sys
import math

def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def compute_loss_and_grad(X, y, w, b, lam):
    n, d = len(X), len(w)
    eps = 1e-15
    loss, grad_w, grad_b = 0.0, [0.0]*d, 0.0

    for i in range(n):
        z = b + sum(w[j] * X[i][j] for j in range(d))
        p = sigmoid(z)
        loss += -(y[i] * math.log(max(p, eps)) + (1-y[i]) * math.log(max(1-p, eps)))
        diff = p - y[i]
        for j in range(d):
            grad_w[j] += diff * X[i][j]
        grad_b += diff

    loss /= n
    for j in range(d):
        grad_w[j] = grad_w[j] / n + (lam / n) * w[j]
    grad_b /= n
    l2 = sum(w[j]**2 for j in range(d))
    loss += (lam / (2*n)) * l2

    return loss, grad_w, grad_b

def train_logreg(X, y, max_iter, alpha, lam, tol):
    d = len(X[0])
    w, b = [0.0]*d, 0.0
    loss, _, _ = compute_loss_and_grad(X, y, w, b, lam)

    for _ in range(max_iter):
        _, grad_w, grad_b = compute_loss_and_grad(X, y, w, b, lam)
        for j in range(d):
            w[j] -= alpha * grad_w[j]
        b -= alpha * grad_b
        new_loss, _, _ = compute_loss_and_grad(X, y, w, b, lam)
        if abs(loss - new_loss) < tol:
            break
        loss = new_loss
    return w, b

def predict_one(x, w, b):
    z = b + sum(w[j] * x[j] for j in range(len(w)))
    p = sigmoid(z)
    return (1 if p >= 0.5 else 0), p

def main():
    data = sys.stdin.read().strip().split()
    it = iter(data)
    n, max_iter = int(next(it)), int(next(it))
    alpha, lam, tol = float(next(it)), float(next(it)), float(next(it))

    X, y = [], []
    for _ in range(n):
        X.append([float(next(it)), float(next(it)), float(next(it))])
        y.append(int(next(it)))

    m = int(next(it))
    test = [[float(next(it)), float(next(it)), float(next(it))] for _ in range(m)]

    w, b = train_logreg(X, y, max_iter, alpha, lam, tol)
    for x in test:
        lab, p = predict_one(x, w, b)
        print(f"{lab} {p:.4f}")

if __name__ == "__main__":
    main()
```

### 题目4: P3872 基于逻辑回归的意图分类器
```python
import sys, math

def encode(seq):
    x = [0.0] * 7
    for ch in set(seq.strip()):
        idx = ord(ch) - ord('A')
        if 0 <= idx < 7:
            x[idx] = 1.0
    return x

def train(X, y, lr=0.1, epochs=20):
    w, b = [0.0]*7, 0.0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            z = sum(w[j] * xi[j] for j in range(7)) + b
            p = 1.0 / (1.0 + math.exp(-z))
            dz = p - yi
            for j in range(7):
                w[j] -= lr * dz * xi[j]
            b -= lr * dz
    return w, b

def predict(w, b, xi):
    z = sum(w[j] * xi[j] for j in range(7)) + b
    p = 1.0 / (1.0 + math.exp(-z))
    return 1 if p > 0.5 else 0

def main():
    data = sys.stdin.read().strip().split()
    it = iter(data)
    N, M = int(next(it)), int(next(it))

    X, y = [], []
    for _ in range(N):
        X.append(encode(next(it)))
        y.append(int(next(it)))

    w, b = train(X, y)

    for _ in range(M):
        print(predict(w, b, encode(next(it))))

if __name__ == "__main__":
    main()
```

### 题目5: P3552 云存储设备故障预测
```python
import sys, math

def parse_line(line):
    parts = [p.strip() for p in line.strip().split(',')]
    if len(parts) < 7: return None
    feats = parts[1:6]
    y = int(float(parts[-1]))
    def to_num(s):
        return None if s == "NaN" else float(s)
    return [to_num(v) for v in feats], y

def valid(col, v):
    if v is None: return False
    if col in (0,1): return v >= 0
    if col in (2,3): return 0 <= v <= 1000
    if col == 4: return 0 <= v <= 20
    return True

def median(vals):
    if not vals: return 0.0
    vals = sorted(vals)
    n = len(vals)
    return vals[n//2] if n % 2 else 0.5*(vals[n//2-1]+vals[n//2])

def sigmoid(z):
    z = max(-30, min(30, z))
    return 1.0 / (1.0 + math.exp(-z))

def main():
    lines = sys.stdin.read().strip().splitlines()
    n = int(lines[0])
    train = [parse_line(lines[i+1]) for i in range(n)]
    m = int(lines[n+1])
    test = [parse_line(lines[n+2+i]) for i in range(m)]

    # 统计均值和中位数
    means, meds = [0.0]*5, [0.0]*5
    for j in range(5):
        valid_vals = [t[0][j] for t in train if t[0][j] is not None and valid(j, t[0][j])]
        means[j] = sum(valid_vals)/len(valid_vals) if valid_vals else 0.0
        meds[j] = median(valid_vals)

    # 清洗
    def clean(row):
        x = []
        for j in range(5):
            v = row[0][j]
            if v is None: v = means[j]
            elif not valid(j, v): v = meds[j]
            x.append(v)
        return x

    X_train = [clean(t) for t in train]
    y_train = [t[1] for t in train]
    X_test = [clean(t) for t in test]

    # 训练逻辑回归
    w, b = [0.0]*5, 0.0
    for _ in range(100):
        grad_w, grad_b = [0.0]*5, 0.0
        for i in range(n):
            z = b + sum(w[j]*X_train[i][j] for j in range(5))
            p = sigmoid(z)
            diff = p - y_train[i]
            for j in range(5):
                grad_w[j] += diff * X_train[i][j]
            grad_b += diff
        for j in range(5):
            w[j] -= 0.01 * grad_w[j] / n
        b -= 0.01 * grad_b / n

    # 预测
    for x in X_test:
        z = b + sum(w[j]*x[j] for j in range(5))
        p = sigmoid(z)
        print(1 if p >= 0.5 else 0)

if __name__ == "__main__":
    main()
```
