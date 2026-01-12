# MLE Interview Prep（Deep-ML）[0/41 完成]

> 📊 **进度**: 0/41 完成 (0%)  
> 🔄 **最后更新**: 2026-01-11  
> 📁 **来源**: [Deep-ML MLE Interview Prep](https://www.deep-ml.com/)  
> 💡 **说明**: 41 essential problems for MLE interviews

---

## 📋 题目总览

| 分类 | 题目数 | 完成 | 进度 |
|------|--------|------|------|
| 1. Core ML Algorithms | 5 | 0 | 0% |
| 2. Loss & Regularization | 4 | 0 | 0% |
| 3. Model Evaluation | 6 | 0 | 0% |
| 4. Neural Networks | 6 | 0 | 0% |
| 5. Optimizers | 3 | 0 | 0% |
| 6. CNNs | 2 | 0 | 0% |
| 7. Sequences | 2 | 0 | 0% |
| 8. Transformers | 3 | 0 | 0% |
| 9. Production & MLOps | 10 | 0 | 0% |

---

## 🔧 通用模板速查

> 💡 按类别组织，跳转：[1.回归/分类](#t1-回归分类) | [2.聚类/降维](#t2-聚类降维) | [3.损失函数](#t3-损失函数) | [4.评估指标](#t4-评估指标) | [5.激活函数](#t5-激活函数) | [6.优化器](#t6-优化器) | [7.NN层](#t7-神经网络层) | [8.CNN](#t8-cnn) | [9.Transformer](#t9-transformer)

---

### T1. 回归/分类

#### Linear Regression (梯度下降 + 正规方程)
```python
def linear_regression_gd(X, y, lr=0.01, epochs=1000):
    n, d = X.shape
    X = np.hstack([np.ones((n, 1)), X])  # 添加 bias
    W = np.zeros(d + 1)
    for _ in range(epochs):
        pred = X @ W
        grad = 2 / n * X.T @ (pred - y)
        W -= lr * grad
    return W

# 正规方程解
W = np.linalg.solve(X.T @ X, X.T @ y)
```

#### Logistic Regression
```python
def sigmoid(z):
    return 1 / (1 + np.exp(np.clip(-z, -500, 500)))
def sigmoid_stable(z):
    out = np.empty_like(z); 
    pos = z >= 0
    out[pos]  = 1.0/(1.0 + np.exp(-z[pos])) #z[pos] >=0 --> e^{-z} 不会爆炸
    ez = np.exp(z[~pos])           # z<0 区间更稳定
    out[~pos] = ez/(1.0 + ez)  # 计算 σ(z) 用稳定写法，而不是直接 1/(1+exp(-z)) 因为z <0可能会爆炸
    return out

def logistic_regression(X, y, lr=0.01, epochs=100, lam=0.0):
    n, d = X.shape
    W = np.zeros(d); b = 0
    for _ in range(epochs):
        z = X @ W + b; p = sigmoid(z); dy = p - y; dg = dy / n
        dW = X.T @ dg + lam * W; db = dg
        W -= lr * dW; b -= lr * db
    return W, b
```

#### Ridge/Lasso (正则化梯度)
```python
# Ridge (L2)
loss = 1/2 * (dy**2) /n +  lam/2 * W ** 2
dW = X.T @ dg + lam * W

# Lasso (L1)  
loss = 1/2 * (dy**2) /n +  lam * np.abs(W)
dW = X.T @ dg + lam * np.sign(W)
```

#### Decision Tree (ID3/C4.5)
```python
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p + 1e-10))

def info_gain(X, y, feature):
    H = entropy(y); x = X[:, feature]; 
    H_cond_v = [(x == v).mean() * entropy(y[x == v]) for v in np.unique(x)]
    return H - sum(H_cond_v)

def majority(y): 
    c = Counter(y)
    return c.most_common(1)[0][0] #不考虑平票、评测数据也不出 tie
    # 平票时选最早出现的:
    # return max(c, key=lambda k: (c[k], -y.index(k)))

def tree(X, y, feature):
    b = max(feature, key=lambda k: gain(X,y,feature))
```

---

### T2. 聚类/降维

#### KMeans
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

#### PCA
```python
# 方法1: SVD
X = (data - data.mean(0)) / data.std(0)  # 标准化
_, _, Vt = np.linalg.svd(X, full_matrices=False)
P = Vt[:k].T  # (d, k) 主成分方向

# 方法2: 协方差矩阵特征分解
C = np.cov(X, rowvar=False)
w, V = np.linalg.eigh(C)
I = np.argsort(w)[::-1]
P = V[:, I[:k]]
```

---

### T3. 损失函数

#### Cross-Entropy Loss (Binary + Multi-class)
```python
def binary_cross_entropy(y_true, y_pred):
    eps = 1e-15; y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean(0)

def bce_loss_from_logits(z, y):
    """数值稳定版 BCE (从 logits 直接计算)"""
    absz = np.abs(z)
    # softplus(z) = log(1 + exp(-|z|)) + max(z, 0) 防止溢出
    softplus = np.log1p(np.exp(-absz)) + np.maximum(z, 0) 
    BCEloss_total = softplus - y * z
    return BCEloss_total.mean()    

def logistic_reg_loss(X, W, y, lam=0.0):
    z = X @ W; p = sigmoid_stable(z)
    loss = bce_loss_from_logits(z, y) + 0.5*lam*np.sum(W*W)
    return loss

def multi_class_ce(y_true, y_pred):
    """y_true: one-hot, y_pred: softmax output"""
    eps = 1e-15
    loss_total = -np.sum(y_true * np.log(y_pred + eps), axis=1)
    return loss_total.mean(0)
```

---

### T4. 评估指标

#### K-fold Cross Validation
```python
def k_fold_cross_validation(X, y, k=5, shuffle=True):
    n = len(X); idx = np.arange(n)
    if shuffle: np.random.shuffle(idx)
    folds = np.array_split(idx, k)  # 自动把 n 分成 k 段（尽量均匀）
    return [(np.concatenate([*folds[:i], *folds[i+1:]]), folds[i]) for i in range(k)]
```

#### Confusion Matrix + Precision/Recall/F1/AUC
```python
def 
    cm = np.bincount(true*K + pred, minlength=K*K).reshape(K, K)
    TP = np.diag(cm)
    FP = cm.sum(0) - TP
    FN = cm.sum(1) - TP
    TN = cm.sum() - TP - FP - FN


def confusion_matrix(y_true, y_pred):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    TN = ((y_pred == 0) & (y_true == 0)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    return TP, TN, FP, FN

def confusion_matrix(data): # binary only
    counts = Counter(tuple(pair) for pair in data) 
    TP, FN, FP, TN = counts[(1, 1)], counts[(1, 0)], counts[(0, 1)], counts[(0, 0)]
    return [[TP, FN], [FP, TN]]

def multi_cat_confusion(Y, Y0):
    TP = np.bincount(Y[Y == Y0], minlength=K)
    FP = np.bincount(Y[Y != Y0], minlength=K) # FP[c]+=1 when 预测成Y[i] = c，但真实Y0[i]!= c
    FN = np.bincount(Y0[Y != Y0], minlength=K) #FN[c]+=1 when 真实Y0[i]== c, 但预测成别的Y[i]!=c

    P = np.divide(TP, TP + FP, out=np.zeros(K), where=(TP + FP) != 0)
    R = np.divide(TP, TP + FN, out=np.zeros(K), where=(TP + FN) != 0)
    F1 = np.divide(2 * P * R, P + R, out=np.zeros(K), where=(P + R) != 0)

def auc(y, p):
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0: return 0.0
    y = y[np.argsort(-p)]
    tpr = np.r_[0, np.cumsum(y) / P] # np.r_ 按行拼接
    fpr = np.r_[0, np.cumsum(1 - y) / N]
    return np.trapz(tpr, fpr)

def precision(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FP + 1e-10)

def recall(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FN + 1e-10)

def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-10)

def auc_roc(y_true, y_scores):
    """简易实现：梯形法则"""
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tpr, fpr = [0], [0]
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
        tpr.append(TP / (TP + FN + 1e-10))
        fpr.append(FP / (FP + TN + 1e-10))
    return np.trapz(tpr, fpr)
```

---

### T5. 激活函数

#### Softmax
```python
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)  # 数值稳定
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
```

#### ReLU
```python
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)
```

---

### T6. 优化器

### sigmoid activation
```python
z = X @ W + b; p = sigmoid(z)
dz = 2/n * (p - y) * p * (1 - p)   # MSE 的 dz
dW = X.T @ dz
db = dz.sum()
```

#### Gradient Descent (BGD/SGD/Mini-batch)
```python
def reg(W, X, Y, lr):
    n = len(X)
    p = X @ W; dy = p - Y; dg = 2/n * dy
    dW = X.T @ dg; W -= lr * dW
    return W

def gradient_descent(X, y, lr, epoch, Bs, method):
    n = len(X)
    W = np.zeros(X.shape[1])
    for _ in range(epoch):
        if method == 'batch':
            W = reg(W, X, y, lr)
        elif method == 'stochastic':
            for i in range(n): 
                W = reg(W, X[i:i+1], y[i:i+1], lr)
        elif method == 'mini_batch':
            for i in range(0, n, Bs):
                W = reg(W, X[i:i+Bs], y[i:i+Bs], lr)
    return W
```

#### Adam Optimizer
```python
def adam(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    for i in range(len(params)):
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        v[i] = beta2 * v[i] + (1 - beta2) * grads[i]**2
        m_hat = m[i] / (1 - beta1**t)  # bias correction
        v_hat = v[i] / (1 - beta2**t)
        params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, m, v
```

---

### T7. 神经网络层

#### Batch Normalization (BCHW)
BN 的核心是：对每个“通道/特征维”单独算统计量，统计量来自batch 以及其它非通道维。
```python
def batch_norm(x, gamma, beta, eps=1e-5):
    # x: (N, C, H, W) for BCHW
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    var = x.var(axis=(0, 2, 3), keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```
#### Layer Normalization (BLD)
def layer_norm_ch(x, gamma, beta, eps=1e-5):
    # x: (N, C, H, W)
    mean = x.mean(axis=1, keepdims=True)          # (N, 1, H, W)
    var  = x.var(axis=1, keepdims=True)           # (N, 1, H, W)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta  


#### Dropout (Inverted)
```python
def dropout(x, p=0.5, training=True):
    if not training or p == 0:
        return x
    mask = np.random.binomial(1, 1-p, x.shape) / (1-p)
    return x * mask
```

---

### T8. CNN

#### Conv2D
```python
def conv2d(X, K, stride=1, padding=0):
    # X: (C, H, W), K: (OC, IC, KH, KW)
    X = np.pad(X, ((0,0), (padding, padding), (padding, padding)))
    _, H, W = X.shape
    OC, IC, KH, KW = K.shape
    Ho = (H - KH) // stride + 1
    Wo = (W - KW) // stride + 1
    out = np.zeros((OC, Ho, Wo))
    for oc in range(OC):
        for h in range(Ho):
            for w in range(Wo):
                out[oc, h, w] = np.sum(
                    X[:, h*stride:h*stride+KH, w*stride:w*stride+KW] * K[oc])
    return out
```

#### Global Average Pooling
```python
def global_avg_pool(x):
    # x: (N, C, H, W) → (N, C)
    return x.mean(axis=(2, 3))
```

---

### T9. Transformer

#### Self-Attention
```python
def self_attention(Q, K, V, d_k):
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    return weights @ V
```

#### Positional Encoding
```python
def positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe
```

---

## 1. Core ML Algorithms（0/5）

| # | 题目 | 难度 | 状态 | 链接 |
|---|------|------|------|------|
| 1.1 | Linear Regression Using Gradient Descent | Easy | ❌ | [P15](https://www.deep-ml.com/problems/15) |
| 1.2 | Train Logistic Regression with Gradient Descent | Medium | ❌ | [P106](https://www.deep-ml.com/problems/106) |
| 1.3 | K-Means Clustering | Medium | ❌ | [P17](https://www.deep-ml.com/problems/17) |
| 1.4 | Principal Component Analysis (PCA) Implementation | Medium | ❌ | [P19](https://www.deep-ml.com/problems/19) |
| 1.5 | Decision Tree Learning | Medium | ❌ | [P20](https://www.deep-ml.com/problems/20) |

### 1.1 Linear Regression Using Gradient Descent (P15)

**题目描述**: 使用梯度下降实现线性回归

**关键公式**:
- 预测: $\hat{y} = X \cdot W$
- 损失: $L = \frac{1}{n}\sum(y - \hat{y})^2$
- 梯度: $\nabla W = -\frac{2}{n}X^T(y - \hat{y})$

```python
# TODO: 填写你的代码
```

---

### 1.2 Train Logistic Regression with Gradient Descent (P106)

**题目描述**: 使用梯度下降训练逻辑回归

**关键公式**:
- Sigmoid: $\sigma(z) = \frac{1}{1+e^{-z}}$
- 损失: $L = -\frac{1}{n}\sum[y\log(p) + (1-y)\log(1-p)]$
- 梯度: $\nabla W = \frac{1}{n}X^T(p - y)$

```python
# TODO: 填写你的代码
```

---

### 1.3 K-Means Clustering (P17)

**题目描述**: 实现 K-Means 聚类算法

**核心步骤**:
1. 初始化 K 个中心
2. E-step: 分配每个点到最近中心
3. M-step: 更新中心为簇内均值
4. 重复直到收敛

```python
# TODO: 填写你的代码
```

---

### 1.4 Principal Component Analysis (PCA) Implementation (P19)

**题目描述**: 实现 PCA 降维

**核心步骤**:
1. 数据标准化
2. 计算协方差矩阵 或 SVD 分解
3. 取前 k 个主成分

```python
# TODO: 填写你的代码
```

---

### 1.5 Decision Tree Learning (P20)

**题目描述**: 实现决策树学习算法

**核心概念**:
- 熵: $H(Y) = -\sum p_i \log_2 p_i$
- 信息增益: $IG = H(Y) - H(Y|X)$

```python
import numpy as np
from collections import Counter

def majority(y):
    y = list(y)
    c = Counter(y)
    return max(c, key=lambda k: (c[k], -y.index(k)))

def entropy(y):
    _,c=np.unique(y, return_counts=True)
    p=c/c.sum()
    return -(p*np.log2(p)).sum()

def gain(x,y):
    H=entropy(y); U=np.unique(x)
    return H - sum((x==u).mean()*entropy(y[x==u]) for u in U)

def g(x,k,t):
    y=np.array([r[t] for r in x], object)
    z=np.array([r[k] for r in x], object)
    return gain(z,y)

def d(x,a,t):
    if not x: return 'No examples'
    y=np.array([r[t] for r in x], object)
    if (y==y[0]).all(): return x[0][t]
    if not a: return majority(y)
    b=max(a, key=lambda k:g(x,k,t))
    return {b:{v:d([q for q in x if q[b]==v],[k for k in a if k!=b],t)
               for v in np.unique([q[b] for q in x])}}

def learn_decision_tree(examples, attributes, target_attr):
    return d(examples, attributes, target_attr)

```

---

## 2. Loss & Regularization（0/4）

| # | 题目 | 难度 | 状态 | 链接 |
|---|------|------|------|------|
| 2.1 | Implement Gradient Descent Variants with MSE Loss | Medium | ❌ | [P47](https://www.deep-ml.com/problems/47) |
| 2.2 | Compute Multi-class Cross-Entropy Loss | Easy | ❌ | [P134](https://www.deep-ml.com/problems/134) |
| 2.3 | Implement Ridge Regression Loss Function | Easy | ❌ | [P43](https://www.deep-ml.com/problems/43) |
| 2.4 | Implement Lasso Regression using Gradient Descent | Medium | ❌ | [P50](https://www.deep-ml.com/problems/50) |

### 2.1 Gradient Descent Variants with MSE Loss (P47)

**题目描述**: 实现不同变种的梯度下降（BGD, SGD, Mini-batch）

**关键公式**:
- MSE: $L = \frac{1}{n}\sum(y - \hat{y})^2$

```python
def reg(W,X,Y,lr):
    n = len(X)
    p = X @ W; dy=p-Y; dg=2/n * dy
    dW = X.T @ dg; W -= lr * dW
    return W
def gradient_descent(X, y, lr, epoch, Bs_batch_size, method):
    for _ in range(epoch):
        if method == 'batch':
            W = reg(W,X,y,lr)
        elif method == 'stochastic':
            for i in range(n): W = reg(W,X[i:i+1],y[i:i+1],lr)
        elif method == 'mini_batch':
            for i in range(0, n, Bs): # Bs = batchsize
                W = reg(W, X[i:i+Bs],y[i:i+Bs],lr)
    return W
```

---

### 2.2 Multi-class Cross-Entropy Loss (P134)

**题目描述**: 计算多分类交叉熵损失

**关键公式**:
$$L = -\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C} y_{ic} \log(p_{ic})$$

```python
# TODO: 填写你的代码
```

---

### 2.3 Ridge Regression Loss Function (P43)

**题目描述**: 实现带 L2 正则的损失函数

**关键公式**:
$$L = \frac{1}{n}\|y - Xw\|^2 + \lambda\|w\|^2$$

```python
# TODO: 填写你的代码
```

---

### 2.4 Lasso Regression using Gradient Descent (P50)

**题目描述**: 使用梯度下降实现 Lasso 回归

**关键公式**:
$$L = \frac{1}{n}\|y - Xw\|^2 + \lambda\|w\|_1$$

**注意**: L1 范数不可导，需用 subgradient

```python
# TODO: 填写你的代码
```

---

## 3. Model Evaluation（0/6）

| # | 题目 | 难度 | 状态 | 链接 |
|---|------|------|------|------|
| 3.1 | Implement K-Fold Cross-Validation | Medium | ❌ | [P18](https://www.deep-ml.com/problems/18) |
| 3.2 | Generate a Confusion Matrix for Binary Classification | Easy | ❌ | [P75](https://www.deep-ml.com/problems/75) |
| 3.3 | Implement Precision Metric | Easy | ❌ | [P46](https://www.deep-ml.com/problems/46) |
| 3.4 | Implement Recall Metric in Binary Classification | Easy | ❌ | [P52](https://www.deep-ml.com/problems/52) |
| 3.5 | Implement F-Score Calculation for Binary Classification | Easy | ❌ | [P61](https://www.deep-ml.com/problems/61) |
| 3.6 | Calculate AUC (Area Under ROC Curve) | Medium | ❌ | [P277](https://www.deep-ml.com/problems/277) |

### 3.1 K-Fold Cross-Validation (P18)

**题目描述**: 实现 K 折交叉验证

**核心步骤**:
1. 将数据分成 K 份
2. 每次用 1 份做验证，K-1 份做训练
3. 返回 K 次验证的平均结果

```python
# TODO: 填写你的代码
```

---

### 3.2 Confusion Matrix (P75)

**题目描述**: 生成二分类的混淆矩阵

**公式**:
|  | Pred=1 | Pred=0 |
|--|--------|--------|
| True=1 | TP | FN |
| True=0 | FP | TN |

```python
# TODO: 填写你的代码
```

---

### 3.3 Precision Metric (P46)

**题目描述**: 实现精确率

**公式**: $Precision = \frac{TP}{TP + FP}$

```python
# TODO: 填写你的代码
```

---

### 3.4 Recall Metric (P52)

**题目描述**: 实现召回率

**公式**: $Recall = \frac{TP}{TP + FN}$

```python
# TODO: 填写你的代码
```

---

### 3.5 F-Score Calculation (P61)

**题目描述**: 实现 F1 分数

**公式**: $F_1 = \frac{2 \cdot P \cdot R}{P + R}$

```python
# TODO: 填写你的代码
```

---

### 3.6 AUC (Area Under ROC Curve) (P277)

**题目描述**: 计算 ROC 曲线下面积

**核心思想**:
1. 计算不同阈值下的 TPR 和 FPR
2. 用梯形法则积分

```python
# TODO: 填写你的代码
```

---

## 4. Neural Networks（0/6）

| # | 题目 | 难度 | 状态 | 链接 |
|---|------|------|------|------|
| 4.1 | Single Neuron with Backpropagation | Easy | ❌ | [P25](https://www.deep-ml.com/problems/25) |
| 4.2 | Implementing a Custom Dense Layer in Python | Medium | ❌ | [P40](https://www.deep-ml.com/problems/40) |
| 4.3 | Implement Batch Normalization for BCHW Input | Medium | ❌ | [P115](https://www.deep-ml.com/problems/115) |
| 4.4 | Dropout Layer | Easy | ❌ | [P151](https://www.deep-ml.com/problems/151) |
| 4.5 | Implement ReLU Activation Function | Easy | ❌ | [P42](https://www.deep-ml.com/problems/42) |
| 4.6 | Softmax Activation Function Implementation | Easy | ❌ | [P23](https://www.deep-ml.com/problems/23) |

### 4.1 Single Neuron with Backpropagation (P25)

**题目描述**: 实现单神经元的前向和反向传播

**公式**:
- 前向: $y = \sigma(w \cdot x + b)$
- 反向: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \sigma'(z) \cdot x$

```python
# TODO: 填写你的代码
```

---

### 4.2 Custom Dense Layer (P40)

**题目描述**: 实现全连接层

**公式**:
- 前向: $Y = XW + b$
- 反向: $\nabla W = X^T \nabla Y$, $\nabla X = \nabla Y \cdot W^T$

```python
# TODO: 填写你的代码
```

---

### 4.3 Batch Normalization for BCHW (P115)

**题目描述**: 实现 BCHW 格式的 BatchNorm

**公式**:
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

```python
# TODO: 填写你的代码
```

---

### 4.4 Dropout Layer (P151)

**题目描述**: 实现 Dropout 层

**公式**: $y = x \cdot \text{mask} / (1 - p)$ (inverted dropout)

```python
# TODO: 填写你的代码
```

---

### 4.5 ReLU Activation Function (P42)

**题目描述**: 实现 ReLU 激活函数

**公式**: $\text{ReLU}(x) = \max(0, x)$

```python
# TODO: 填写你的代码
```

---

### 4.6 Softmax Activation Function (P23)

**题目描述**: 实现 Softmax 函数

**公式**: $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$

**技巧**: 减去最大值防止溢出

```python
# TODO: 填写你的代码
```

---

## 5. Optimizers（0/3）

| # | 题目 | 难度 | 状态 | 链接 |
|---|------|------|------|------|
| 5.1 | Implement Adam Optimization Algorithm | Medium | ❌ | [P49](https://www.deep-ml.com/problems/49) |
| 5.2 | Momentum Optimizer | Easy | ❌ | [P146](https://www.deep-ml.com/problems/146) |
| 5.3 | Gradient Clipping by Global Norm | Easy | ❌ | [P197](https://www.deep-ml.com/problems/197) |

### 5.1 Adam Optimization Algorithm (P49)

**题目描述**: 实现 Adam 优化器

**公式**:
- $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
- $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
- $\hat{m}_t = m_t / (1 - \beta_1^t)$
- $\hat{v}_t = v_t / (1 - \beta_2^t)$
- $\theta_t = \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$

```python
# TODO: 填写你的代码
```

---

### 5.2 Momentum Optimizer (P146)

**题目描述**: 实现动量优化器

**公式**:
- $v_t = \gamma v_{t-1} + \alpha \nabla L$
- $\theta_t = \theta_{t-1} - v_t$

```python
# TODO: 填写你的代码
```

---

### 5.3 Gradient Clipping by Global Norm (P197)

**题目描述**: 按全局范数裁剪梯度

**公式**: 若 $\|g\| > \text{max\_norm}$，则 $g = g \cdot \frac{\text{max\_norm}}{\|g\|}$

```python
# TODO: 填写你的代码
```

---

## 6. CNNs（0/2）

| # | 题目 | 难度 | 状态 | 链接 |
|---|------|------|------|------|
| 6.1 | Simple Convolutional 2D Layer | Medium | ❌ | [P41](https://www.deep-ml.com/problems/41) |
| 6.2 | Implement Global Average Pooling | Easy | ❌ | [P114](https://www.deep-ml.com/problems/114) |

### 6.1 Simple Convolutional 2D Layer (P41)

**题目描述**: 实现 2D 卷积层

**输出尺寸**: $H_{out} = (H + 2P - K) / S + 1$

```python
# TODO: 填写你的代码
```

---

### 6.2 Global Average Pooling (P114)

**题目描述**: 实现全局平均池化

**公式**: 对每个通道取空间维度的平均值

```python
def global_avg_pool(x):
    # x: (N, C, H, W) → (N, C)
    return x.mean(axis=(2, 3))
```

---

## 7. Sequences（0/2）

| # | 题目 | 难度 | 状态 | 链接 |
|---|------|------|------|------|
| 7.1 | Implement LSTM Network | Hard | ❌ | [P59](https://www.deep-ml.com/problems/59) |
| 7.2 | Implement GRU Cell | Medium | ❌ | [P287](https://www.deep-ml.com/problems/287) |

### 7.1 LSTM Network (P59)

**题目描述**: 实现 LSTM 网络

**门控公式**:
- 遗忘门: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
- 输入门: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
- 候选值: $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
- 细胞状态: $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
- 输出门: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
- 隐状态: $h_t = o_t * \tanh(C_t)$

```python
# TODO: 填写你的代码
```

---

### 7.2 GRU Cell (P287)

**题目描述**: 实现 GRU 单元

**门控公式**:
- 更新门: $z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$
- 重置门: $r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$
- 候选隐状态: $\tilde{h}_t = \tanh(W \cdot [r_t * h_{t-1}, x_t])$
- 隐状态: $h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$

```python
# TODO: 填写你的代码
```

---

## 8. Transformers（0/3）

| # | 题目 | 难度 | 状态 | 链接 |
|---|------|------|------|------|
| 8.1 | Implement Self-Attention Mechanism | Medium | ❌ | [P53](https://www.deep-ml.com/problems/53) |
| 8.2 | Implement Multi-Head Attention | Hard | ❌ | [P94](https://www.deep-ml.com/problems/94) |
| 8.3 | Positional Encoding Calculator | Easy | ❌ | [P85](https://www.deep-ml.com/problems/85) |

### 8.1 Self-Attention Mechanism (P53)

**题目描述**: 实现自注意力机制

**公式**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

```python
# TODO: 填写你的代码
```

---

### 8.2 Multi-Head Attention (P94)

**题目描述**: 实现多头注意力

**公式**:
- $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
- $\text{MultiHead} = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$

```python
# TODO: 填写你的代码
```

---

### 8.3 Positional Encoding Calculator (P85)

**题目描述**: 计算位置编码

**公式**:
- $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$
- $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$

```python
def positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe
```

---

## 9. Production & MLOps（0/10）

| # | 题目 | 难度 | 状态 | 链接 |
|---|------|------|------|------|
| 9.1 | Implement Stratified Train-Test Split | Easy | ❌ | [P275](https://www.deep-ml.com/problems/275) |
| 9.2 | Implement Grid Search | Medium | ❌ | [P288](https://www.deep-ml.com/problems/288) |
| 9.3 | Implement Early Stopping Based on Validation Loss | Easy | ❌ | [P135](https://www.deep-ml.com/problems/135) |
| 9.4 | Feature Drift Detection using PSI | Medium | ❌ | [P253](https://www.deep-ml.com/problems/253) |
| 9.5 | A/B Test Statistical Analysis | Medium | ❌ | [P269](https://www.deep-ml.com/problems/269) |
| 9.6 | Calculate P50/P95/P99 Latency Percentiles | Easy | ❌ | [P293](https://www.deep-ml.com/problems/293) |
| 9.7 | Implement INT8 Quantization | Medium | ❌ | [P294](https://www.deep-ml.com/problems/294) |
| 9.8 | Implement Prediction Distribution Monitoring | Medium | ❌ | [P295](https://www.deep-ml.com/problems/295) |
| 9.9 | Calculate Statistical Power for Experiment Design | Medium | ❌ | [P296](https://www.deep-ml.com/problems/296) |
| 9.10 | Implement Request Batching for Inference | Medium | ❌ | [P297](https://www.deep-ml.com/problems/297) |

### 9.1 Stratified Train-Test Split (P275)

**题目描述**: 实现分层划分训练/测试集

**核心**: 保持各类别在训练集和测试集中的比例一致

```python
# TODO: 填写你的代码
```

---

### 9.2 Grid Search (P288)

**题目描述**: 实现网格搜索超参调优

**核心**: 遍历所有超参组合，用交叉验证评估

```python
# TODO: 填写你的代码
```

---

### 9.3 Early Stopping (P135)

**题目描述**: 基于验证损失实现早停

**核心**: 当验证损失连续 patience 轮不下降时停止训练

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
```

---

### 9.4 Feature Drift Detection using PSI (P253)

**题目描述**: 使用 PSI 检测特征漂移

**公式**:
$$PSI = \sum_i (A_i - E_i) \cdot \ln\frac{A_i}{E_i}$$

其中 $A_i$ 是实际分布，$E_i$ 是期望分布

```python
# TODO: 填写你的代码
```

---

### 9.5 A/B Test Statistical Analysis (P269)

**题目描述**: 模型比较的 A/B 测试统计分析

**核心**: 计算 p-value，判断显著性

```python
# TODO: 填写你的代码
```

---

### 9.6 Calculate P50/P95/P99 Latency Percentiles (P293)

**题目描述**: 计算延迟百分位数

```python
def percentiles(data):
    data = sorted(data)
    n = len(data)
    p50 = data[int(n * 0.50)]
    p95 = data[int(n * 0.95)]
    p99 = data[int(n * 0.99)]
    return p50, p95, p99
```

---

### 9.7 INT8 Quantization (P294)

**题目描述**: 实现 INT8 量化

**公式**: $x_{int8} = \text{round}(x / scale) + zero\_point$

```python
# TODO: 填写你的代码
```

---

### 9.8 Prediction Distribution Monitoring (P295)

**题目描述**: 监控预测分布变化

**核心**: 比较训练时和推理时的预测分布

```python
# TODO: 填写你的代码
```

---

### 9.9 Statistical Power for Experiment Design (P296)

**题目描述**: 计算实验设计的统计功效

**公式**: $n = \frac{(z_\alpha + z_\beta)^2 \cdot 2\sigma^2}{\delta^2}$

```python
# TODO: 填写你的代码
```

---

### 9.10 Request Batching for Inference (P297)

**题目描述**: 实现推理请求批处理

**核心**: 将多个请求合并成 batch 以提高吞吐

```python
# TODO: 填写你的代码
```

---

## 📌 易错点总结

1. **数值稳定性**: sigmoid/softmax 要减最大值防溢出
2. **正则化梯度**: L2 梯度是 $\lambda w$，L1 是 $\lambda \cdot \text{sign}(w)$
3. **BatchNorm 维度**: BCHW 格式沿 (N, H, W) 维度求均值
4. **Dropout 缩放**: 训练时除以 $(1-p)$ (inverted dropout)
5. **Adam bias correction**: 初始阶段 $1-\beta^t$ 接近 0，必须做修正
6. **卷积填充**: 输出尺寸公式别忘了 padding
7. **Attention 缩放**: 除以 $\sqrt{d_k}$ 防止梯度消失
8. **交叉熵 log(0)**: 加 eps 防止取对数时出现 -inf

---

## 🔗 相关资源

- [Deep-ML MLE Interview Prep](https://www.deep-ml.com/)
- 本地题库: `ML/coding/` 下各分类文件
