# Conv 类题目汇总 [4/5 完成]

> 📊 **进度**: 4/5 完成 (80%)  
> 🔄 **最后更新**: 2026-01-07  
> 📁 **分类**: conv (卷积操作、零填充、多通道、空洞卷积、能量路径、Group卷积)

---

## 🇺🇸 US 留学生场 Conv 预测

### 已考过的 Conv 类型（4/6 场出现）

| 日期 | 题目 | 类型 | 核心考点 |
|------|------|------|---------|
| 11-20 | P4482 | 基础 Padding | 零填充 + 输出尺寸 |
| 11-06 | P4448 | 多通道 | stride + padding + 通道累加 |
| 10-23 | P4278 | **Dilation** ⭐ | 空洞卷积 + 有效核尺寸 |
| 09-18 | P3718 | Conv + DP | 卷积能量图 + 动态规划 |

### 📊 预测：下次 US 场可能出的 Conv 变形

| 优先级 | 题型 | 预测概率 | 理由 |
|--------|------|---------|------|
| 🔴 P0 | **Depthwise Conv** | 60% | MobileNet 热点，Group Conv 的特例 |
| 🔴 P0 | **Group Conv** | 50% | CN 场考过(P3493)，US 场还没考 |
| 🟡 P1 | **Transposed Conv** | 30% | 上采样场景，语义分割必备 |
| 🟡 P1 | **1×1 Conv** | 25% | 通道混合，简单但重要 |

### 🔥 预测1: Depthwise Separable Conv

**题目描述**：实现 Depthwise Separable Convolution（MobileNet 核心）
- **Depthwise Conv**：每个通道独立卷积（groups = C）
- **Pointwise Conv**：1×1 卷积混合通道

```python
import sys, numpy as np
lines = [l for l in sys.stdin.buffer.read().splitlines()]
C, H, W = np.fromstring(lines[0], int, sep=' ')
Img = np.fromstring(lines[1], float, sep=' ').reshape((C, H, W))
K = int(np.fromstring(lines[2], int, sep=' ')[0])
Ker = np.fromstring(lines[3], float, sep=' ').reshape((C, K, K))  # 每通道一个核
S, P = np.fromstring(lines[4], int, sep=' ')

X = np.pad(Img, ((0,0),(P,P),(P,P)))
win = np.lib.stride_tricks.sliding_window_view(X, (K,K), axis=(1,2))[:,::S,::S]
# win: (C, Ho, Wo, K, K) → 每通道独立卷积
out = np.einsum('ckw,chokw->cho', Ker, win)
print(" ".join(f"{v:.4f}" for v in out.ravel()))
```

### 🔥 预测2: Group Conv（简化版）

**核心**：`groups=G` 时，每组输入通道 `C//G`，每组输出通道 `OC//G`

```python
import sys, numpy as np
lines = [l for l in sys.stdin.buffer.read().splitlines()]
C, H, W = np.fromstring(lines[0], int, sep=' ')
Img = np.fromstring(lines[1], float, sep=' ').reshape((C, H, W))
OC, KC, K, _ = np.fromstring(lines[2], int, sep=' ')
Ker = np.fromstring(lines[3], float, sep=' ').reshape((OC, KC, K, K))
G, S, P = np.fromstring(lines[4], int, sep=' ')

X = np.pad(Img, ((0,0),(P,P),(P,P)))
Ho = (H + 2*P - K) // S + 1; Wo = (W + 2*P - K) // S + 1
out = np.zeros((OC, Ho, Wo))
Cg = C // G; OCg = OC // G  # 每组通道数

for g in range(G):
    Xg = X[g*Cg : (g+1)*Cg]
    Kg = Ker[g*OCg : (g+1)*OCg]
    win = np.lib.stride_tricks.sliding_window_view(Xg, (K,K), axis=(1,2))[:,::S,::S]
    out[g*OCg:(g+1)*OCg] = np.tensordot(Kg, win, axes=([1,2,3], [0,3,4]))

print(" ".join(f"{v:.4f}" for v in out.ravel()))
```

### 🟡 预测3: Transposed Conv (Deconv)

**核心**：上采样，输出尺寸 = `(H-1)*S - 2P + K`

```python
import sys, numpy as np
lines = [l for l in sys.stdin.buffer.read().splitlines()]
C, H, W = np.fromstring(lines[0], int, sep=' ')
Img = np.fromstring(lines[1], float, sep=' ').reshape((C, H, W))
OC, IC, K, _ = np.fromstring(lines[2], int, sep=' ')
Ker = np.fromstring(lines[3], float, sep=' ').reshape((OC, IC, K, K))
S, P = np.fromstring(lines[4], int, sep=' ')

# 插入零（stride间隔）
X_dilated = np.zeros((C, (H-1)*S+1, (W-1)*S+1))
X_dilated[:, ::S, ::S] = Img
# 翻转核 + 常规卷积
Ker_flip = Ker[:, :, ::-1, ::-1]
X_pad = np.pad(X_dilated, ((0,0),(K-1-P,K-1-P),(K-1-P,K-1-P)))
win = np.lib.stride_tricks.sliding_window_view(X_pad, (K,K), axis=(1,2))
out = np.tensordot(Ker_flip, win, axes=([1,2,3], [0,3,4]))

print(" ".join(f"{v:.4f}" for v in out.ravel()))
```

### 📋 备考 Checklist

| 题型 | 核心变化 | 模版调整 |
|------|---------|---------|
| 基础 Conv | P = K//2 | `np.pad(..., P)` |
| Dilation | Keff = D*(K-1)+1 | `win[..., ::D, ::D]` |
| Stride | 输出尺寸变化 | `win[:, ::S, ::S]` |
| **Depthwise** | groups = C | `einsum('ckw,chokw->cho')` |
| **Group** | 分组计算 | 循环 G 组 |
| **Transposed** | 先插零再卷 | 翻转核 + 大 padding |

---

## 📋 题目总览

> 🔥 **重刷优先级**: 5 > 4 > 1 > 3 > 2（Group卷积和带 dilation 的卷积最重要）

| 出题日期 | # | P编号 | 题目 | 难度 | 状态 | 完成日期 |
|----------|---|-------|------|------|------|----------|
| 2025-10-23 | 4 | P4278 | 卷积结构实现（带dilation）⭐ | 中等 | ✅ | 2026-01-02 |
| 2025-10-22 | 1 | P4274/P3718 | 最大能量路径 | 中等 | ✅ | 2026-01-02 |
| 2025-11-06 | 3 | P4448 | 卷积操作（多通道） | 中等 | ✅ | 2026-01-02 |
| 2025-11-20 | 2 | P4482 | 带Padding的卷积计算 | 中等 | ✅ | 2026-01-02 |
| 2025-08-28 | 5 | P3493 | Group卷积实现（分组/深度卷积）⭐ | 困难 | ❌ | - |

---

## 🔧 通用模板
### 最重要的cov
```python
X = np.pad(Img, ((0,0),(P,P),(P,P))); Kh=Kw=D*(K-1)+1
win=np.lib.stride_tricks.sliding_window_view(X, (Kh,Kw), axis=(1,2))[:,::S,::S,::D,::D]
a = np.tensordot(Ker, win, axes=([1,2,3], [0,3,4])) 

win = np.lib.stride_tricks.sliding_window_view(Pad, (K, K))  # (H, W, K, K)
a = np.tensordot(Ker, win)
```


### I/O 模板
```python
import sys
data = sys.stdin.read().strip().split()
it = iter(data)
K = int(next(it)); C = R = int(next(it));
Ker = [[ int(next(it)) for _ in range(K)] for _ in range(K)]
Img = [[ int(next(it)) for _ in range(C)] for _ in range(R)]
...
sys.stdout.write("\n".join(" ".join(map(str, row)) for row in E))
```

### Cov numpy
```python
import sys
import numpy as np
d = sys.stdin.read().strip().split(); H,W,K,K2=map(int,d[:4]);Img=np.array(d[4:4+H*W],float);Ker=np.array(d[4+H*W:],float);
Img = Img.reshape((H,W));Ker=Ker.reshape((K,K));
P = K//2; Img_pad = np.zeros((H+2*P,W+2*P)); Img_pad[P:P+H, P:P+W]=Img
E = sum(
        Ker[i, j] * Img_pad[i:i+H, j:j+W]
        for i in range(K) for j in range(K)
    )
R=H;C=W
dp = np.full((R + 2, C), -1e300); dp[1:R+1, 0] = E[:, 0]
for c in range(1, C):
    dp[1:R+1, c] = np.maximum.reduce([
        dp[0:R, c-1], dp[1:R+1, c-1], dp[2:R+2, c-1]
        ]) + E[:, c]
print(f"{dp[1:R+1, C-1].max():.1f}")
```
### 基础 Conv 模板（零填充）
```python
k2 = K // 2;
E = [[0.0] * C for _ in range(R)]
Img_pad = [[0] * (C + 2 * k2) for _ in range(R + 2 * k2)]
for r in range(R): Img_pad[r+k2][k2:k2+C] = Img[r][:] # r+k2 别忘
for r in range(R):
    for c in range(C):
        summ = 0
        for kr in range(K):
            for kc in range(K):
                summ += Img_pad[r+kr][c+kc] * Ker[kr][kc]
        E[r][c] = summ
```

### Conv 模板 + P (padding) + S (stride)
```python
Img_pad = [[[0] * (C + 2 * P) for _ in range(R+ 2 * P)] for _ in range(CH)]
for i in range(CH):
    for r in range(R): Img_pad[i][r + P][P:C+P] = Img[i][r][:]
OR = (R + 2 * P -KR )// S  + 1; OC = (C + 2 * P -KC )// S + 1; 
Out = [[0] * OC for _ in range(OR)]
for r in range(OR):
    for c in range(OC):
        summ = 0; br = r * S; bc = c * S; # stride 别忘了
        for i in range(CH):
            for kr in range(KR):
                for kc in range(KC):
                    summ += Ker[i][kr][kc] * Img_pad[i][br+kr][bc+kc]
        Out[r][c]=summ
```

---

## ⭐ 题目4: 卷积结构实现（P4278）- 带 Dilation【最重要】

- **难度**: 中等
- **标签**: conv, dilation, stride, padding, bias
- **源**: [core46#第3题-p4278](../AI_编程题_Python解答_核心46题.md#第3题-p4278)

### 题目描述

实现完整的 Conv2D，支持 stride、padding、dilation（空洞卷积）和 bias。

**参数**：
- input: 输入数据 (C, H, W)
- weight: 卷积核权重 (Out, In, K, K)
- bias: 卷积核偏置
- stride: 移动步长
- padding: 边缘填充像素数
- dilation: 卷积核元素间隔

**输出尺寸**（有效核尺寸 K_eff = dilation × (K-1) + 1）：
- $H_{out} = (H + 2 \times padding - K_{eff}) // stride + 1$
- $W_{out} = (W + 2 \times padding - K_{eff}) // stride + 1$

### 输入输出
- **输入**：
  - 第1行：c, x, y（输入形状）
  - 第2行：输入数据（c×x×y 个实数，行优先）
  - 第3行：out, in, k, k（卷积核形状）
  - 第4行：权重数据
  - 第5行：bias, stride, padding, dilation
  - 第6行：若 bias=1，为偏置值
- **输出**：卷积结果（保留4位小数）

### 样例
```
输入：
1 4 4
1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0
1 1 3 3
1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
1 1 0 1
1.0

输出：
55.0000 64.0000 91.0000 100.0000

3 8 6
-9.681700 -1.225267 4.855898 -4.517088 -6.002298 -2.289715 -5.678056 2.687776 2.420510 8.706787 -2.703423 -5.658304 6.312009 -7.888033 2.574984 2.396062 5.593134 8.151232 -6.918301 7.263687 -1.740501 1.437934 -9.927827 -1.597677 5.113558 5.984594 -5.502819 4.870729 3.360235 0.765364 8.090027 6.981866 3.449864 5.606130 5.157577 5.234392 -4.595610 -3.799532 3.089381 4.948492 -4.696732 0.489777 0.700693 9.080402 -2.702745 2.311531 1.654295 -3.925515 -9.523355 -3.858679 -1.286172 3.941982 6.441708 -4.443602 -8.281733 7.109346 5.437960 1.888211 -5.563640 -9.621842 -3.752320 0.557832 1.154090 -0.009600 3.679149 7.290318 7.507953 -2.729441 -4.141175 -3.907302 8.567237 3.194404 -9.364252 -7.421373 -7.958169 0.700507 -1.735869 -1.217550 5.393894 -2.123726 -1.366703 -5.047527 5.167020 -0.345517 -9.610172 -9.943266 4.428585 -2.145473 -8.743316 2.284460 -5.001495 -4.480437 5.783614 5.702390 9.190735 -3.881093 1.191216 5.467695 4.065187 7.091082 -2.812743 -8.342174 -6.979921 4.314517 6.930344 -1.630534 5.601771 -8.343892 3.467642 5.090948 7.535327 9.194397 8.836545 7.277806 -8.057580 6.288732 2.232077 9.285559 -4.802664 -2.398237 -9.008882 8.389951 -3.076771 7.877721 7.208428 -7.403027 -5.380954 -5.452948 -3.895849 -6.390004 8.340990 -1.935897 -6.528508 9.728299 -2.412988 1.003941 8.348140 -0.125019 -2.882522 9.003009 5.270408 -0.213994 8.699161 0.881338
3 3 4 4
2.043897 0.934829 1.225998 -1.399549 -2.279209 2.894312 -0.666331 2.024284 -0.636906 1.399990 2.315300 0.958217 1.637000 2.571774 -0.115992 2.699467 1.157632 -2.471613 2.502034 1.982215 2.042929 -1.182332 1.131664 -2.669795 -1.996167 -0.762841 0.195999 -2.734632 -2.744678 -2.656605 -0.678030 1.753225 -2.683200 2.638286 -1.808314 -0.650323 2.123624 0.827965 0.164358 0.256908 -1.178956 1.298880 -2.157198 -2.712148 2.463362 -2.076717 1.836560 0.953111 1.600111 -2.341401 -2.895654 -2.854697 1.954202 1.279206 -1.785161 -2.459264 -2.443874 0.831965 -0.144273 -2.747814 0.237903 2.030550 -0.422507 0.516884 -1.512888 -1.208423 2.637667 1.215032 -0.637099 -0.685513 2.837039 -2.622818 -2.691147 -0.603503 -1.912125 -2.568694 2.134190 -0.419296 1.879572 0.499709 -2.775104 0.588457 2.588775 2.212669 2.621813 2.418440 -1.488904 -0.865636 -1.926768 -2.687026 0.627235 2.714798 2.987713 0.511977 -0.294916 -2.461153 1.824735 2.524419 2.788733 1.987073 2.797777 1.062618 -1.078918 1.799078 0.926060 0.953642 -1.956020 -0.814256 0.546442 -1.751023 -2.901298 2.791034 -1.805165 2.301056 2.564148 -1.280051 0.464911 -0.289808 0.277514 2.767331 -1.833276 -2.831540 1.702544 -2.064365 -1.019762 -1.544963 2.950908 0.331794 0.784649 0.536466 2.001205 -1.037774 0.700756 -0.594687 2.491566 2.196744 0.380966 2.467647 2.331801 1.076793 2.913248 2.499885 1.256469 1.458246
0 2 3 2

输出：
24.3500 102.9196 13.9372 -0.4666 47.9315 84.2458 113.3018 58.9370 68.5063 -21.9355 73.3948 0.9606 -78.5932 13.1708 21.8428 34.1725 50.7713 -49.3644 -65.9254 -17.8579 40.7550 -14.3218 -27.3649 -19.4848 -0.8054 -12.7652 28.1154 11.1127 -85.4465 12.0784 18.9459 58.5616 1.2112 -33.8436 75.7512 45.7590
```

### 思路
1. **计算有效核尺寸**：K_eff = dilation × (K-1) + 1
2. **输出尺寸计算**：考虑 dilation 的影响
3. **卷积计算**：
   - ih = oh × stride + kh × dilation - padding
   - iw = ow × stride + kw × dilation - padding
   - 越界视为 0

### 复杂度
- 时间：$O(Out \times In \times H_{out} \times W_{out} \times K^2)$
- 空间：$O(C \times H \times W + Out \times In \times K^2)$

### 我的代码 ✅ (numpy)
```python
import sys,numpy as np 
lines = [l for l in sys.stdin.buffer.read().splitlines()]
CH, R, C = np.fromstring(lines[0], int, sep=' ')
Img = np.fromstring(lines[1], float, sep=' ').reshape((CH,R,C))
O, I, K, K2 = np.fromstring(lines[2], int, sep=' ')
Ker = np.fromstring(lines[3], float, sep=' ').reshape((O,I,K,K2))
B1, S, P, D = np.fromstring(lines[4], int, sep=' ')
B = np.fromstring(lines[5], float, sep=' ') if B1 == 1 else np.zeros(1)

X = np.pad(Img, ((0,0),(P,P),(P,P))); Kh=Kw=D*(K-1)+1
win=np.lib.stride_tricks.sliding_window_view(X, (Kh,Kw), axis=(1,2))[:,::S,::S,::D,::D]
a = np.tensordot(Ker, win, axes=([1,2,3], [0,3,4])) 
if B1 == 1: a+= B[:,None,None]
print(" ".join(f"{ii:.4f}" for ii in a.ravel()))
```
### 我的代码 ✅ (不用numpy)
```python
import sys
d = iter(sys.stdin.read().strip().split())
CH = int(next(d)); R = int(next(d)); C = int(next(d)); 
Img = [[[float(next(d)) for _ in range(C)] for _ in range(R) ] for _ in range(CH)]
O = int(next(d)); I  = int(next(d)); KR = int(next(d)); KC = int(next(d)); 
Ker = [[[[float(next(d)) for _ in range(KC)] for _ in range(KR) ] for _ in range(I)] for _ in range(O)]
B1 = int(next(d)); S  = int(next(d)); P = int(next(d)); D = int(next(d));
B = [float(next(d)) if B1 == 1 else 0 for _ in range(O)]

Img_Pad = [[[0] * (C+2*P) for _ in range(R+2*P)] for _ in range(CH)]
for i in range(CH):
    for r in range(R): Img_Pad[i][r+P][P : C+P] = Img[i][r][:]
# (D * (KR - 1) + 1) 实际kernel size
AR = (R + 2 * P - (D * (KR - 1) + 1)) // S + 1; AC = (C + 2 * P - (D * (KR - 1) + 1)) // S + 1; res = [];
for o in range(O):
    for r in range(AR):
        for c in range(AC):
            s = 0; br = r * S; bc = c * S ;
            for i in range(CH):
                for kr in range(KR):
                    for kc in range(KC):
                        s += Img_Pad[i][br+kr * D][bc+kc * D] * Ker[o][i][kr][kc]
            res.append(s+B[o])
print(" ".join([f"{v:.4f}" for v in res]))
```

---

## 题目1: 最大能量路径（P4274/P3718）

- **难度**: 中等
- **标签**: conv + dp
- **源**: [core46#第2题-p4274](../AI_编程题_Python解答_核心46题.md#第2题-p4274)

### 题目描述

在自动驾驶系统中，车道线识别是核心功能之一。给定一个 H×W 的图像以及一个 K×K 的策略矩阵，你需要从图像的第一列任意像素出发，走到最后一列任意像素，每一步只能向右、右上、右下移动一格。

**定义**：每个位置的能量值 = 策略矩阵与该位置周边信号值的乘积和（零填充卷积）

### 输入输出
- **输入**：H W K K，接下来 H 行图像矩阵，K 行策略矩阵
- **输出**：最大能量值（保留1位小数）

### 思路
1. **预处理能量图**：零填充卷积计算整张图的能量矩阵 E，复杂度 $O(H \cdot W \cdot K^2)$
2. **动态规划**：
   - 边界：$f_{i,0} = E_{i,0}$
   - 转移：$f_{i,j} = E_{i,j} + \max(f_{i-1,j-1}, f_{i,j-1}, f_{i+1,j-1})$
   - 答案：$\max_{0 \le i < H} f_{i,W-1}$

### 复杂度
- 时间：$O(H \cdot W \cdot K^2)$
- 空间：$O(H \cdot W)$（可滚动数组降到 $O(H)$）

### 我的代码 ✅
```python
import sys
import numpy as np
d = sys.stdin.buffer.read().split(); H,W,K,K2=map(int, d[:4]); Img = np.array(d[4:4+H*W],float); Ker = np.array(d[4+H*W:],float);
Img = Img.reshape((H,W)); Ker=Ker.reshape((K,K)); P=K//2;
Pad = np.pad(Img,((P,P),(P,P)));
# E = sum(Ker[i][j] * Pad[i:i+K,j:j+K]  for i in range(K) for j in range (K))
win = np.lib.stride_tricks.sliding_window_view(Pad, (K, K))  # (H, W, K, K)
E = (win * Ker).sum(axis=(-1, -2))
dp=np.ones((H+2,W)) * -1e100; dp[1:H+1,0] = E[:,0];
for c in range(1,W):
    dp[1:1+H, c] = np.maximum.reduce([dp[0:H, c-1],dp[1:H+1, c-1],dp[2:(H+2), c-1]]) + E[:,c]
out=dp[1:1+H,-1].max()
print(out)

import sys
data = sys.stdin.read().strip().split()
it = iter(data)
R = int(next(it)); C = int(next(it)); K1 = int(next(it)); K2=int(next(it)); K = K1;
Img = [ [float(next(it)) for _ in range(C)] for _ in range(R)];
Ker = [ [float(next(it)) for _ in range(K)] for _ in range(K)];
k2 = K // 2;
E = [[0.0] * C for _ in range(R)]
Img_pad = [[0] * (C + 2 * k2) for _ in range(R + 2 * k2)]
for r in range(R): Img_pad[r+k2][k2:k2+C] = Img[r][:]
for r in range(R):
    for c in range(C):
        summ = 0
        for kr in range(K):
            for kc in range(K):
                # if 0 <= r + kr - k2 < R and 0 <= c + kc - k2 < C:
                summ += Img_pad[r+kr][c+kc] * Ker[kr][kc]
        E[r][c] = summ
dp = [[-float('inf')] * (C) for _ in range(R+2)]
for r in range(R): dp[1+r][0] = E[r][0]
for c in range(1,C):
    for r in range(1,R+1):
        dp[r][c] = max(dp[r-1][c-1], dp[r][c-1], dp[r+1][c-1]) + E[r-1][c]
res = max([dp[r][C-1] for r in range(1,R+1)])
print(f"{res:.1f}")
```

---

## 题目3: 卷积操作（P4448）- 多通道

- **难度**: 中等
- **标签**: conv, multi-channel, stride, padding
- **源**: [core46#第3题-p4448](../AI_编程题_Python解答_核心46题.md#第3题-p4448)

### 题目描述

实现多通道卷积操作，支持 stride 和 padding。

**公式**：
$$\text{output}(i, j) = \sum_{c=0}^{C-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} \text{input}_c(i \times stride + m, j \times stride + n) \times \text{kernel}_c(m, n)$$

**输出尺寸**：
- $H_{out} = (H_{in} + 2 \times padding - K_h) // stride + 1$
- $W_{out} = (W_{in} + 2 \times padding - K_w) // stride + 1$

### 输入输出
- **输入**：
  - 第一行：C, H_in, W_in（输入张量形状）
  - 接下来 C×H_in 行：张量元素值
  - 一行：C, K_h, K_w（卷积核形状）
  - 接下来 C×K_h 行：卷积核元素值
  - 最后一行：stride, padding
- **输出**：H_out × W_out 的特征图（整数）

### 样例
```
输入：
2 3 3
1 2 3
4 5 6
7 8 9
2 3 4
5 6 7
8 9 10
2 2 2
1 0
0 1
2 0
0 2
1 0

输出：
22 28
40 46
```
```
输入：
1 3 3
5 2 5
5 3 4
1 9 2
1 2 3
2 -1 -1
1 1 0
2 1
输出：
5 7
-7 13
```

### 思路
1. **填充**：在输入张量四周补零
2. **滑动窗口**：以 stride 步长移动卷积核
3. **逐通道累加**：对每个位置，所有通道做乘加求和

### 复杂度
- 时间：$O(H_{out} \cdot W_{out} \cdot C \cdot K_h \cdot K_w)$
- 空间：$O(C \cdot (H_{in}+2p) \cdot (W_{in}+2p))$

### 我的代码 ✅
```python
import sys
d = iter(sys.stdin.read().strip().split())
CH = int(next(d)); R = int(next(d)); C = int(next(d));
Img = [[[ int(int(next(d))) for _ in range(C)] for _ in range(R)] for _ in range(CH)]
KCH = int(next(d)); KR = int(next(d)); KC = int(next(d));
Ker = [[[ int(int(next(d))) for _ in range(KC)] for _ in range(KR)] for _ in range(KCH)]
S = int(next(d)); P = int(next(d));
# print(C0, R, C, C1, KR, KC, Img, Ker, Std, Pad)
Img_pad = [[[0] * (C + 2 * P) for _ in range(R+ 2 * P)] for _ in range(CH)]
for i in range(CH):
    for r in range(R): Img_pad[i][r + P][P:C+P] = Img[i][r][:]
# print(Img_pad)
OR = (R + 2 * P -KR )// S  + 1; OC = (C + 2 * P -KC )// S + 1; 
# print(OR, OC)
Out = [[0] * OC for _ in range(OR)]
for r in range(OR):
    for c in range(OC):
        summ = 0; br = r * S; bc = c * S;
        for i in range(CH):
            for kr in range(KR):
                for kc in range(KC):
                    summ += Ker[i][kr][kc] * Img_pad[i][br+kr][bc+kc]
        Out[r][c]=summ
# print(Out)
sys.stdout.write("\n".join(" ".join(map(str, row)) for row in Out))
```

---

## 题目2: 带Padding的卷积计算（P4482）

- **难度**: 中等
- **标签**: conv, padding
- **源**: [core46#第3题-p4482](../AI_编程题_Python解答_核心46题.md#第3题-p4482)

### 题目描述

实现无核翻转的卷积计算（cross-correlation），使用 Padding 确保输出尺寸与输入一致。

**公式**：$(S \cdot K)(i, j) = \sum_{m} \sum_{n} S(i+m, j+n) \cdot K(m, n)$

### 输入输出
- **输入**：
  - 第一行：卷积核尺寸 m×m，图像尺寸 n×n（m为大于1的奇数）
  - 接下来 m 行：卷积核数据，值范围 [-10, 10]
  - 接下来 n 行：图像数据，值范围 [0, 255]
- **输出**：卷积后结果矩阵 n×n（整数）

### 样例
```
输入：
3 5
-5 4 0
0 -3 -2
3 2 0
231 112 85 120 114
154 237 168 55 35
203 204 160 70 7
194 32 36 99 181
64 185 251 30 115

输出：
-609 430 552 26 -107
394 -737 98 440 -25
-13 -108 -965 -538 503
294 195 371 -366 -543
214 -1899 -829 -106 -119
```

### 思路
1. **零填充**：在输入图像外围填充 `t = m//2` 圈 0
2. **卷积计算**：对每个输出位置 (i,j)，让卷积核以 (i,j) 为中心覆盖，做逐元素乘加
3. **边界检查法**（无需显式构造填充数组）：越界时视为0

### 复杂度
- 时间：$O(n^2 \cdot m^2)$
- 空间：$O(n^2)$

### 我的代码 ✅
```python
import sys
it = iter(sys.stdin.read().strip().split())
K = int(next(it)); C = R = int(next(it));
Ker = [[ int(next(it)) for _ in range(K)] for _ in range(K)]
Img = [[ int(next(it)) for _ in range(C)] for _ in range(R)]
# print(R,C, K, Img, Ker)
k2 = K // 2
Img_pad = [[0] * (C + 2 * k2) for _ in range(R+2*k2)]
for r in range(R): Img_pad[r+k2][k2:C+k2] = Img[r][:]
E = [[0] * (C) for _ in range(R)]
for r in range(R):
    for c in range(C):
        s = 0
        for kr in range(K):
            for kc in range(K):
                s += Img_pad[r+kr][c+kc] * Ker[kr][kc]
        E[r][c] = s
sys.stdout.write("\n".join(" ".join(map(str, row)) for row in E))
```

---

## 题目5: Group卷积实现（P3493）⭐

- **难度**: 困难
- **标签**: conv, group convolution, depthwise convolution
- **源**: [0828coding.md](../../../0828coding.md)

### 题目描述

实现分组卷积（Group Convolution）和深度卷积（Depthwise Convolution）的前向传播。分组卷积将输入张量和卷积核分组后，分别执行卷积计算，然后拼接输出。

**参数**：
- input: 输入数据 (N, C, H, W)
- kernel: 卷积核权重 (OC, KC, KH, KW)
- groups: 分组数

**约束条件**：
- `in_channels % groups == 0`
- `out_channels % groups == 0`
- `k_channels == in_channels // groups`

**输出尺寸**（stride=1, padding=0, dilation=1）：
- $H_{out} = H - K_h + 1$
- $W_{out} = W - K_w + 1$

### 输入输出
- **输入**：
  - 第1行：in_data（展开后的输入张量）
  - 第2行：in_shape（N C H W）
  - 第3行：kernel_data（展开后的卷积核）
  - 第4行：kernel_shape（OC KC KH KW）
  - 第5行：groups
- **输出**：
  - 第1行：out_data（展开后的输出张量）
  - 第2行：out_shape（N OC Ho Wo）
- **错误情况**：若形状与 groups 不合法，输出 `-1`

### 样例
```
输入：
1 2 3 4 5 6 7 8
1 2 2 2
1 0 0 1 -1 0 0 -1
2 1 2 2
2

输出：
5 -13
1 2 1 1
```

### 思路
1. **校验合法性**：检查 C%G==0, OC%G==0, KC==C//G, Ho>0, Wo>0
2. **分组计算**：
   - 每组输入通道数 `KC_g = C // G`
   - 每组输出通道数 `OC_g = OC // G`
   - 对每个 (n, g, oc, oh, ow)，累加该组对应输入通道与核窗口的乘积和
3. **按 N→C→H→W 展开输出**

### 复杂度
- 时间：$O(N \cdot OC \cdot H_o \cdot W_o \cdot (C/G) \cdot K_h \cdot K_w)$
- 空间：$O(N \cdot OC \cdot H_o \cdot W_o)$

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 📌 易错点总结

1. **零填充索引**：`Img_pad[r+k2]` 别忘 +k2
2. **边界检查**：越界时视为 0，不要访问非法索引
3. **卷积 vs 相关**：题目通常是无核翻转（cross-correlation），不是真正的卷积
4. **Dilation 公式**：有效核尺寸 = dilation × (K-1) + 1
5. **输出格式**：注意小数位数要求（1位/4位）
6. **多通道求和**：所有通道的结果要累加
7. **Group 卷积约束**：`KC == C // G`，不是 `KC == C`
8. **Group 卷积分组**：每组只处理对应的通道，不是全部通道

---

## 📝 代码答案

### 题目5: P3493 Group卷积实现
```python
import sys

def parse_line_to_ints(s: str):
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split() if x]

def main():
    lines = sys.stdin.read().splitlines()
    if len(lines) < 5:
        print("-1")
        print("-1")
        return

    line1, line2, line3, line4, line5 = lines[:5]
    in_data = parse_line_to_ints(line1)
    in_shape = parse_line_to_ints(line2)
    ker_data = parse_line_to_ints(line3)
    ker_shape = parse_line_to_ints(line4)
    groups_list = parse_line_to_ints(line5)

    if len(in_shape) != 4 or len(ker_shape) != 4 or len(groups_list) != 1:
        print("-1")
        print("-1")
        return

    N, C, H, W = in_shape
    OC, KC, KH, KW = ker_shape
    G = groups_list[0]

    # 基本合法性
    if N <= 0 or C <= 0 or H <= 0 or W <= 0 or OC <= 0 or KC <= 0 or KH <= 0 or KW <= 0 or G <= 0:
        print("-1")
        print("-1")
        return

    in_need = N * C * H * W
    ker_need = OC * KC * KH * KW
    if len(in_data) != in_need or len(ker_data) != ker_need:
        print("-1")
        print("-1")
        return

    if C % G != 0 or OC % G != 0:
        print("-1")
        print("-1")
        return

    if KC != C // G:
        print("-1")
        print("-1")
        return

    Ho = H - KH + 1
    Wo = W - KW + 1
    if Ho <= 0 or Wo <= 0:
        print("-1")
        print("-1")
        return

    # 预计算步长
    HW = H * W
    CHW = C * HW
    out_stride_n = OC * Ho * Wo
    out_stride_c = Ho * Wo
    ker_stride_oc = KC * KH * KW
    ker_stride_kc = KH * KW

    OCg = OC // G  # 每组输出通道数
    KCg = KC       # 每组输入通道数（核的通道数）

    y = [0] * (N * OC * Ho * Wo)

    for n in range(N):
        base_n_in = n * CHW
        base_n_out = n * out_stride_n

        for g in range(G):
            ic_start = g * KCg
            oc_start = g * OCg

            for ocg in range(OCg):
                oc = oc_start + ocg
                base_oc_out = base_n_out + oc * out_stride_c
                base_oc_ker = oc * ker_stride_oc

                for oh in range(Ho):
                    for ow in range(Wo):
                        acc = 0

                        for kc in range(KCg):
                            ic = ic_start + kc
                            base_ic_in = base_n_in + ic * HW
                            base_kc_ker = base_oc_ker + kc * ker_stride_kc

                            for kh in range(KH):
                                ih = oh + kh
                                row_in = base_ic_in + ih * W + ow
                                row_ker = base_kc_ker + kh * KW

                                for kw in range(KW):
                                    acc += in_data[row_in + kw] * ker_data[row_ker + kw]

                        y[base_oc_out + oh * Wo + ow] = acc

    print(" ".join(str(v) for v in y))
    print(N, OC, Ho, Wo)

if __name__ == "__main__":
    main()
```

---

## 🔗 相关文件

- 源文件：`../AI_编程题_Python解答_核心46题.md`
- 索引：`../ai_core46_index.md`
