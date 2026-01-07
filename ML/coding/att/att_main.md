# Att 类题目汇总 [7/7 完成] ✅

> 📊 **进度**: 7/7 完成 (100%) 🎉  
> 🔄 **最后更新**: 2026-01-05  
> 📁 **分类**: att (Attention、ViT、LoRA、Multi-Head、Sparse Attention、Self-Attention)

---

## 📋 题目总览

> 🔥 **重刷优先级**: 4 > 3 = 2 > 7 > 6 > 5 > 1（按重要程度排序）

| 出题日期 | # | P编号 | 题目 | 难度 | 状态 | 完成日期 |
|----------|---|-------|------|------|------|----------|
| 2025-11-20 | 1 | P4481 | ViT Patch Embedding层实现 | 中等 | ✅ | 2026-01-02 |
| 2025-10-22 | 2 | P4275 | 基于空间连续块的稀疏注意力机制 | 中等 | ✅ | 2026-01-02 |
| 2025-10-15 | 3 | P4227 | 动态注意力掩码调度问题 | 中等 | ✅ | 2026-01-02 |
| 2025-09-28 | 4 | P3843 | Masked Multi-Head Self-Attention 实现 | 中等 | ✅ | 2026-01-02 |
| 2025-09-17 | 5 | P3712 | 大模型Attention模块开发 | 中等 | ✅ | 2026-01-02 |
| 2025-09-12 | 6 | P3658 | 支持LoRA的Attention实现 | 中等 | ✅ | 2026-01-02 |
| 2025-09-04 | 7 | P3562 | 传感器数据分析（Self-Attention+FC） | 中等 | ✅ | 2026-01-05 |

🏆 **全部完成！** 难度/有价值/二刷重点： 4 > 3 = 2 > 7 > 6 > 5 > 1
---

## 🔧 通用模板

```python
# Scaled Dot-Product Attention
import numpy as np

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores)
    return weights @ V

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
```

---

## 题目1: ViT Patch Embedding层实现（P4481）

- **难度**: 中等
- **源**: [core46#第2题-p4481](../AI_编程题_Python解答_核心46题.md#第2题-p4481)

### 题目描述

Vision Transformer(ViT) 是视觉领域应用非常广泛的基础网络结构，经典的 ViT 结构包含了 Patch＆Position embedding、Transformer embedding、Transformer Encoder 等多个关键模块组成。这几个模块中，将图像分割为固定大小的 patch 并进行线性嵌入是一个关键步骤，也即 Patch Embedding 层，其主要实现步骤为：

**Step 1**：将输入图像分割为多个非重叠的 patch，也即将图片切分为 N×N 个 patch，如 3×3 个 2D 图像块；

**Step 2**：将每个 patch 展平为向量，也即将每个切分后的 2D Patch 展平为 1D 向量;

**Step 3**：对展平的 patch 进行线性变换(嵌入)，也即对每个展平后的 1D 向量做一个线性变换，使用一个可学习的权重矩阵 E 和偏置向量 B 进行线性变换，公式为：$Z=X*E+b$

**Step 4**：添加可学习的位置编码；

请根据以上提示步骤，实现 Patch Embedding 层。

**特别注意**：本实现过程中，无法使用深度学习框架，如 pytorch、tensorflow 等

**输入描述**：输入参数包括 `img_size`、`patch_size`、`channel`、`embedding_dim`，分别表示：
- 图像尺寸（图像长、宽默认相等）`img_size`
- patch 大小 `patch_size`
- 图像通道数 `channels`
- 嵌入维度 `embedding_dim`

**输出描述**：输出 `patch_embedding` 后的维度信息 `embedding_shape`，其中需要包含 cls token，具体可见样例。

### 样例
```
输入：
448 32 3 384

输出：
197 384
```
```
输入：
224 16 3 768

输出：
197 768
```

### 思路

Patch Embedding 的本质是一个分块 + 展开 + 线性变换的过程，可以理解为对图像做一次"卷积核为 patch，步长为 patch_size 的卷积 + reshape"，再加上一个 cls token。这里我们只需要计算输出向量序列的维度，而不是具体做矩阵运算。

1. **计算 patch 的个数**：图像被均匀切成大小为 patch_size × patch_size 的不重叠 patch
   - 每一维上的 patch 个数：$N = \frac{\text{img\_size}}{\text{patch\_size}}$
   - 总 patch 数目：$\text{num\_patches} = N \times N = \left(\frac{\text{img\_size}}{\text{patch\_size}}\right)^2$

2. **展开并线性变换**：每个 patch 的原始维度为 $\text{patch\_dim} = \text{patch\_size} \times \text{patch\_size} \times \text{channel}$，经过线性嵌入后每个 patch 变成一个长度为 embedding_dim 的向量。

3. **添加 CLS Token**：ViT 会额外添加一个可学习的 cls token，其维度和单个 patch 的嵌入相同，为 (embedding_dim,)。拼接到序列前面之后，序列长度变为：$\text{num\_tokens} = \text{num\_patches} + 1$

4. **最终输出维度**：$\text{embedding\_shape} = (\text{num\_patches} + 1, \text{embedding\_dim})$

### 复杂度

- **时间复杂度**：O(1)，只需要进行简单的除法运算
- **空间复杂度**：O(1)，只需要存储几个变量

### 我的代码
```python
I, P, CH, E = map(int, input().split())
num_patch = ((I-1) // P + 1)
print(num_patch **2 + 1, E)
```

---

## 题目2: 基于空间连续块的稀疏注意力机制（P4275）

- **难度**: 中等
- **源**: [core46#第3题-p4275](../AI_编程题_Python解答_核心46题.md#第3题-p4275)

### 题目描述

在大语言模型推理过程中，随着上下文长度增加，标准 Attention 的计算开销以 $O(n^2)$ 增长，成为性能瓶颈。为提升长序列处理效率，提出一种基于空间连续块的稀疏注意力机制。

**具体流程**：
1. 一个长度为 n 的历史 token 序列，每个 token 表示为 1 个 d 维特征向量 $x_j \in \mathbb{R}^d$
2. 按固定块大小 b，将序列划分为 $m = \lceil n/b \rceil$ 个空间连续块（最后一个块可不满）$B_1, B_2, ..., B_m$
3. 对每个块 $B_k$：
   - 计算平均池化向量：$\mathbf{h}_k = \frac{1}{|B_k|} \sum_{x \in B_k} \mathbf{x}$
   - 使用一个两层多层感知机（MLP）进行非线性压缩（隐藏维度 $d_l = 1$）：$\mathbf{c}_k = W_2 \cdot \sigma(W_1 \cdot \mathbf{h}_k + b_1) + b_2$
     - 其中 $W_1 \in \mathbb{R}^{1 \times d}$，$W_2 \in \mathbb{R}^{d \times 1}$，输出 $c_k \in \mathbb{R}^d$
     - $b_1 = 2$，$b_2 = 1$
     - $\sigma(x) = \max(0, x)$（即 ReLU 激活函数）
4. 给定查询向量 $\mathbf{q} \in \mathbb{R}^d$（题目中固定为全 1 向量：$q_i = 1$），计算每个压缩块的注意力得分：$a_k = \frac{\mathbf{q} \cdot \mathbf{c}_k}{\sqrt{d}}$
5. 将序列 A 划分为恰好 2 个连续非空子数组，目标是最大化这两个子数组和中的最小值 S
6. 最终输出该最大化的最小值 S 的整数化得分，即 $round(100 \cdot S)$

**输入描述**：第 1 行：n d b；接下来 n 行：每行 d 个数，表示 $x_i$；倒数第 2 行：d 个数，表示 $W_1$；最后 1 行：d 个数，表示 $W_2$

**输出描述**：返回一个整数，即上述步骤 5 的整数化得分

### 样例
```
输入：
3 1 1
2.0
4.0
6.0
1.0
2.0

输出：
1700
```
```
输入：
3 2 1
2.0 1.0
3.0 2.0
4.0 3.0
1.0 0.5
2.0 1.0

输出：
1732

Input:
6 2 2
5.000000 -2.000000
7.000000 4.000000
-1.000000 5.000000
-3.000000 2.000000
3.000000 6.000000
-2.000000 3.000000
3.000000 -2.000000
-3.000000 -2.000000
Out:
-6081
```

### 思路

整体可分为两部分：

1. **数值构造**：分块 + 池化 + MLP + 打分
   - 设序列长度为 n、维度为 d、块大小为 b，块数 $m = \lceil n/b \rceil$
   - 第 k 个块的均值池化：$h_k = \frac{1}{|B_k|}\sum_{x \in B_k} x \in \mathbb{R}^d$
   - 两层 MLP（隐藏维度为 1）：$t_k = W_1 \cdot h_k + b_1$，$r_k = \sigma(t_k) = \max(0, t_k)$，$c_k = W_2 \cdot r_k + b_2 \in \mathbb{R}^d$
   - 因为 $q = \mathbf{1}$，注意力得分：$a_k = \frac{\sum_{i=1}^{d} c_k^{(i)}}{\sqrt d}$

2. **最优划分**：前缀和 + 贪心
   - 目标是 $\max_{1 \le s \le m-1} \min(\sum_{i=1}^{s}a_i, \sum_{i=s+1}^{m}a_i)$
   - 记总和 $T = \sum_{i=1}^{m}a_i$，前缀和 $P_s = \sum_{i=1}^{s}a_i$
   - 最优 s 使得两段尽量"均衡"，即 $P_s$ 最接近 $T/2$
   - 实现上只需一次线性扫描：维护前缀和，逐个计算 $\min(P_s, T-P_s)$ 的最大值即可

### 复杂度

- **时间复杂度**：$O(n \cdot d)$
  - 计算所有块均值与 MLP：遍历每个 token 各维度，$O(n \cdot d)$
  - 计算打分并寻找最优切分点：$O(m)$，其中 $m = \lceil n/b \rceil \le n$
- **空间复杂度**：$O(d + m)$，可降到 $O(d)$（边算边累计，不必存整列）

### 我的代码
```python
import numpy as np
N, D, B = map(int, input().split()); M = (N-1) // B + 1;
X = np.array([list(map(float, input().split())) for _ in range(N)])
W1 = np.array(list(map(float, input().split()))); W2 = np.array(list(map(float, input().split()))); 
b1 = 2; b2 = 1; A = [0] * M
for m in range(M):
    Bk = X[m*B: (m+1) * B]  # B x D
    hk = Bk.mean(axis=0) # D
    ck = W2 * max(W1 @ hk + b1, 0) + b2
    A[m] = ck.sum() / np.sqrt(D)
total = sum(A); maxx = -float('inf'); prefix = 0;
for i in range(M-1):
    prefix += A[i]
    other = total - prefix
    maxx = max(maxx,  min(prefix, other))
print(f"{maxx * 100:.0f}")
```

---

## 题目3: 动态注意力掩码调度问题（P4227）

- **难度**: 中等
- **源**: [core46#第2题-p4227](../AI_编程题_Python解答_核心46题.md#第2题-p4227)

### 题目描述

你正在设计一种跨模态知识的大模型精准度机制，给定一个长度为 n 的输入 token 序列，每个位置 j 拥有一个 d 维特征向量 $X_j \in \mathbb{R}^d$ 和一个正整数计算容量 $c_j$，表示该位置最多可接收来自前 j 位置的信息连接数。

**系统需完成以下步骤**：

1. **RMSNorm 归一化**：对所有特征向量进行 RMSNorm 归一化（本题取 $\gamma = 1, \epsilon = 0$）：
   - 每个特征向量记为 $x_i \in \mathbb{R}^d$，其第 k 个分量为 $x_i[k]$
   - RMSNorm 定义为：$\hat{X_i} = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{k=1}^{d}x_i[k]^2 + \epsilon}} \cdot \gamma$

2. **注意力得分计算**：计算每对位置 $i < j$ 的注意力得分，使用标准缩放点积公式（基于 RMSNorm 归一化向量）：
   - $A_{ij} = \frac{\hat{x_i} \cdot \hat{x_j}}{\sqrt{d}}$

3. **掩码矩阵构造**：构造下三角注意力掩码矩阵 $M \in \{0,1\}^{n \times n}$，满足入度约束：
   - $\forall j \in [0, n), \sum_{i=0}^{j-1} M_{ij} \leq c_j$

4. **目标函数最大化**：最大化全局注意力信息总量，定义为所有激活连接的平方注意力得分之和：
   - $S = \sum_{j=0}^{n-1} \sum_{i=0}^{j-1} M_{ij} \cdot A_{ij}^2$

5. **输出整数化得分**：最终返回将最大化 S 乘以 100 后四舍五入得到的整数：$round(100 \cdot S)$

**输入描述**：第 1 行：n d；接下来 n 行：每行 d 个浮点数，表示 $x_j$；最后 1 行：n 个正整数，表示 $c_j$

**输出描述**：返回一个整数，即上述步骤 5 的整数化得分

### 样例
```
输入：
4 2
2.0 2.0
3.0 0.0
0.0 4.0
1.0 1.0
1 2 1 3

输出：
600
```
```
输入：
3 2
1.0 0.0
0.0 1.0
1.0 1.0
1 1 2

输出：
200
```

### 思路

本题的核心是在资源约束下最大化注意力信息总量。问题可以分解为以下几个步骤：

1. **RMSNorm 归一化**：对于每个 d 维特征向量，计算其均方根值，然后将向量的每个分量除以该均方根值。

2. **计算注意力得分**：对于任意两个位置 i 和 j（其中 $i < j$），使用归一化后的向量进行缩放点积运算，得到注意力得分 $A_{ij}$，并计算其平方值 $A_{ij}^2$。

3. **贪心选择**：对于每个位置 j，需要从前面的所有位置中选择最多 $c_j$ 个位置建立连接。为了最大化目标函数 S，应当采用贪心策略：对于每个位置 j，将所有前置位置按照 $A_{ij}^2$ 的值从大到小排序，然后选择前 $c_j$ 个最大的值。

4. **贪心策略的正确性**：目标函数 S 是所有激活连接的 $A_{ij}^2$ 之和，每个位置的选择是相互独立的，因此局部最优解（每个位置选择最大的 $c_j$ 个值）必然能导致全局最优解。

### 复杂度

- **时间复杂度**：$O(n^2 \cdot d)$
  - RMSNorm 归一化：$O(n \cdot d)$
  - 计算所有注意力得分平方：$O(n^2 \cdot d)$
  - 贪心选择：$O(n^2)$（每个位置最多 n 个前置位置，需要排序）
- **空间复杂度**：$O(n^2)$，用于存储注意力得分矩阵

### 我的代码
```python
import numpy as np
N, D = map(int, input().split())
Xj = np.array([list(map(float, input().split())) for _ in range(N)])
Cj = np.array(list(map(int, input().split())))
M = 1 - np.triu(np.ones((N,N)))
dom = np.sqrt((Xj**2).mean(axis=1, keepdims=True))
Xnorm = np.divide(Xj, dom+1e-12)
A = Xnorm @ Xnorm.T / np.sqrt(D); A2 = M * A**2
S = 0
for i in range(1, N):
    max_idx = min(i, Cj[i])
    S+= np.partition(A2[i][:i], -max_idx)[-max_idx:].sum()
print(f"{100*S:.0f}")
```

---

## 题目4: Masked Multi-Head Self-Attention 实现（P3843）

- **难度**: 中等
- **源**: [core46#第3题-p3843](../AI_编程题_Python解答_核心46题.md#第3题-p3843)

### 题目描述

在 Transformer 模型中，Multi-Head Self-Attention 是核心组件，用于捕捉序列中的依赖关系。你需要从头实现一个 Masked Multi-Head Self-Attention 函数，支持自注意力（即 queries、keys 和 values 来自同一输入序列），并处理编码（mask）以防止未来位置的信息泄露（常见于 Decoder 中）。

**具体要求**：

1. **支持多头注意力**：将注意力机制并行分成多个"头"，每个头学习不同的注意力模式
2. **计算过程**：
   - 生成 Q、K、V 矩阵：对输入序列 X（维度：[batch_size, seq_len, d_model]）通过 3 个线性层分别生成查询（Query, Q）、键（Key, K）、值（Value, V）矩阵：$Q = X \cdot W_Q$，$K = X \cdot W_K$，$V = X \cdot W_V$
   - 将 Q、K、V 拆分为多个头：分割为 num_heads 个并行的子矩阵（每个头的维度为 $d_k = d_{model} / num_{heads}$）
   - 对于每个头，计算注意力分数：$attention\_scores = (Q \cdot K^T) / \sqrt{d_k}$
   - 提供 mask（一个 (batch_size, seq_len, seq_len) 的布尔数组，其中 True 表示需要掩码的位置），则将 masked 位置的注意力分数设置为负无穷（-inf）
   - 对掩码后的分数应用 softmax 得到注意力权重
   - 计算注意力输出：$attention = softmax\_scores \cdot V$
   - 拼接多头输出，并通过一个线性投影得到最终结果：$output = concat(attention_1, ..., attention_{num\_heads}) \cdot W_O$

**输入描述**：以";"分隔，分别为 num_heads, X, Q、K、V，$W_O$

**输出描述**：输出为最终结果 output，输出保留两位有效小数，并且为 List

### 样例
```
输入：
2;[[[ 1.92, 1.48], [0.67, -1.23], [0.35, -0.68]], [[-1.11, 0.09], [-0.3, -0.39], [-0.59, -0.06]]];[[1.0, 2.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]]

输出：
[[[14.64, 14.64], [-5.36, -5.36], [-4.44, -4.44]], [[-2.79, -2.79], [-3.04, -3.04], [-2.79, -2.79]]]
```
```
输入：
2;[[[ 1.92, 1.48], [0.67, -1.23], [0.35, -0.68]], [[-1.11, 0.09], [-0.3, -0.39], [-0.59, -0.06]]];[[1.0,1.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]]

输出：
[[[14.64, 14.64], [-5.37, -5.37], [-4.62, -4.62]], [[-2.79, -2.79], [-3.03, -3.03], [-2.77, -2.77]]]
输入：
2;[[[1.17, 0.14], [-0.18, -0.17]], [[0.73, 0.15], [-1.95, 0.02]]];[[-0.07, -1.26], [0.64, 1.41]];[[-1.77, -0.78], [-1.03, 0.42]];[[0.82, 1.50], [0.32, 1.25]];[[0.39, 0.09], [2.00, -1.66]]
输出：
[[[4.25, -3.11], [1.65, -1.17]], [[2.82, -2.07], [-6.08, 4.69]]]

2;[[[0.81, 1.47, 0.82, 0.40], [-1.74, -0.73, 1.43, -1.08], [1.88, 1.30, 0.04, 1.28], [0.35, 1.32, -1.12, 0.12], [-1.05, -0.60, -1.78, -0.10]], [[0.74, -1.81, 1.72, 0.10], [0.89, -1.37, 1.10, 1.71], [-1.06, -1.51, -0.37, -1.96], [0.06, -0.34, -0.51, 0.10], [0.19, 0.52, -1.03, -1.50]], [[-0.26, 0.22, 0.77, 0.37], [-0.75, -1.97, 1.18, -0.02], [0.10, -1.93, -0.76, -1.73], [-0.80, 0.27, 0.07, -1.53], [-0.83, 1.99, 0.33, 0.14]]];[[-1.35, -0.93, 1.49, 0.97], [-0.34, -0.34, 0.62, -1.45], [-0.54, 1.67, 1.23, -0.75], [0.55, 0.87, -0.94, 1.91]];[[-1.85, 1.17, -0.59, 0.37], [1.09, 1.84, 1.01, 0.45], [0.24, -1.49, -0.43, 1.44], [0.67, 0.47, 1.68, 1.75]];[[0.49, 1.17, 1.10, -0.03], [-0.40, 1.99, -1.51, 1.71], [1.87, -0.74, -0.86, -1.23], [-0.65, -1.48, -1.72, -0.14]];[[-1.10, 1.23, 1.24, 0.45], [-0.72, 0.94, -0.28, -0.28], [-1.21, 1.88, -0.24, -0.96], [1.28, -0.56, 0.98, -0.67]]

[[[2.00, -2.07, 2.64, 1.40], [-6.48, 5.07, -2.11, 1.79], [4.00, -4.28, 6.29, 2.65], [-0.05, -0.76, 4.49, 2.84], [-4.33, 1.91, 1.62, 4.15]], [[-10.68, 7.81, 0.82, 4.77], [-2.57, -1.69, 0.13, 5.74], [-8.34, 9.96, -0.23, -2.00], [-7.13, 3.96, -1.34, 4.11], [-7.67, 9.68, 4.17, -0.79]], [[1.16, -2.97, 1.35, 2.97], [-4.75, 2.32, -1.79, 2.87], [-5.63, 4.47, 1.43, 2.13], [-10.22, 12.57, -0.16, -3.01], [-11.78, 15.12, -4.82, -5.75]]]

```

### 思路

本题要求手写「带因果掩码」的多头自注意力（Decoder 常用），整体流程：

1. **线性映射生成 Q/K/V**：$Q = X W_Q, K = X W_K, V = X W_V$，维度：[B, S, d_model]

2. **分头**：将最后一维 d_model 均分为 num_heads 个头，每头维度 $d_k = d_{model} / num_{heads}$，并重排为 $Q_h, K_h, V_h \in [B, H, S, d_k]$

3. **每头计算注意力分数**：$\text{scores} = \frac{Q_h K_h^\top}{\sqrt{d_k}} \in [B,H,S,S]$

4. **因果掩码（防未来信息泄露）**：构造下三角 Mask（[S,S]，下三角为 1，上三角为 0），广播到 [B,H,S,S]。将上三角（不允许关注的）位置置为 $-\infty$：$\text{masked\_scores} = \text{where}(mask=0, -\infty, \text{scores})$

5. **Softmax 得注意力权重**：按最后一维 S 做归一化，数值稳定：减去行最大值，$\alpha = \text{softmax}(\text{masked\_scores})$

6. **聚合得到每头输出**：$\text{head} = \alpha V_h \in [B,H,S,d_k]$

7. **拼接各头并做输出投影**：先将各头在 $d_k$ 维拼接回 $d_{model}$：[B,S,H·$d_k$] = [B,S,$d_{model}$]，再乘以 $W_O$：$\text{output} = \text{concat(heads)} W_O \in [B,S,d_{model}]$

### 复杂度

设批次 B、序列长度 S、模型维度 $D = d_{model}$、头数 H、每头维度 $d_k = D/H$。

- **时间复杂度**：$O(B \cdot S \cdot D^2 + B \cdot S^2 \cdot D)$
  - 线性映射：$O(B \cdot S \cdot D^2)$
  - 注意力 QK^T：每头 $O(S^2 \cdot d_k)$，总计 $O(B \cdot H \cdot S^2 \cdot d_k) = O(B \cdot S^2 \cdot D)$
  - 乘 V 聚合：同阶 $O(B \cdot S^2 \cdot D)$
  - 输出投影：$O(B \cdot S \cdot D^2)$

- **空间复杂度**：$O(B \cdot S \cdot D + B \cdot H \cdot S^2)$，主要存储 Q/K/V、注意力分数与权重

### 我的代码
```python
import numpy as np
from ast import literal_eval
H, XX, Wq, Wk, Wv, Wo= list(map(lambda x: np.array(literal_eval(x), dtype=float), input().strip().split(';'))); 
H = int(H); B, L, D = XX.shape; Dk = D // H 
QQ, KK, VV = [np.transpose((XX @ WW).reshape(B, L, H, Dk), (0,2,1,3)) for WW in [Wq, Wk, Wv]]
scores = (QQ @ np.transpose(KK, (0,1,3,2))/ np.sqrt(Dk)); 

MM = np.tril(np.ones((L, L)))[None, None, :, :]; 
scores = np.where(MM == 1, scores, -np.inf)

exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
softmax = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True); 
A = (np.transpose(softmax @ VV, (0,2,1,3)).reshape(B, L, D) @ Wo).round(2)
s = np.array2string(A, precision=2, separator=', ', formatter = {'float_kind': lambda x: f"{x:.2f}" if abs(x) >=0.005 else "0.00"})
print(' '.join(s.split()))
```

---

## 题目5: 大模型Attention模块开发（P3712）

- **难度**: 中等
- **源**: [core46#第2题-p3712](../AI_编程题_Python解答_核心46题.md#第2题-p3712)

### 题目描述

已知大模型常用的 Attention 模块定义如下：

$$Y = \text{softmax}\left(\frac{QK^T}{\sqrt{h}}\right)V$$

此处考虑二维情况，其中：
- $Q, K, V = XW_1, XW_2, XW_3 \in \mathbb{R}^{n \times h}$
- $X \in \mathbb{R}^{n \times m}$
- $W_1, W_2, W_3 \in \mathbb{R}^{m \times h}$

**注意**：
- 为简便起见，所有输入初始化为全 1 矩阵，所有权重矩阵初始化为上三角全 1 矩阵
- 对任意矩阵 M 的 softmax 计算简化为：$\text{softmax}(M)_{ij} = \frac{M_{ij}}{M_i}$，其中 $M_i = \sum_j M_{ij}$

**输入描述**：输入为维度参数 n, m 和 h，参数间使用空格隔开，均为小于 100 的正整数

**输出描述**：输出为结果矩阵 $Y \in \mathbb{R}^{n \times h}$ 的所有元素之和，在四舍五入后保留整数

### 样例
```
输入：
3 3 3

输出：
18
```
```
输入：
2 3 1

输出：
2
```
```
输入：
91 100 71

输出：
232596
```

### 思路

按题意用"暴力模拟"完整走一遍计算图：

1. **构造矩阵**：构造 X 为 n×m 的全 1；构造 $W_1$、$W_2$、$W_3$ 为 m×h 的上三角全 1（上三角矩阵：主对角线及以上的元素为 1，以下为 0）

2. **计算 Q, K, V**：$Q = X \cdot W_1$，$K = X \cdot W_2$，$V = X \cdot W_3$（普通三重循环矩阵乘法）

3. **计算注意力分数**：$M = (Q \cdot K^T) / \sqrt{h}$

4. **简化 softmax**：把 M 的每一行做归一化：$A[i][j] = M[i][j] / (\text{该行元素和})$

5. **计算输出**：$Y = A \cdot V$

6. **求和并输出**：将 Y 全部元素求和，四舍五入输出整数

### 复杂度

- **时间复杂度**：$O(n \cdot m \cdot h + n^2 \cdot h)$
  - 计算 Q、K、V：$O(n \cdot m \cdot h)$
  - 计算 $M = Q \cdot K^T$：$O(n^2 \cdot h)$
  - 行归一化：$O(n^2)$
  - 计算 $Y = A \cdot V$：$O(n^2 \cdot h)$

- **空间复杂度**：$O(n \cdot h + n^2)$（保存 Q、K、V、A 或 M 等中间结果）

### 我的代码
```python
import numpy as np
L, D, H = map(int, input().split())
X = np.ones((L, D)); W = np.triu(np.ones((D, H))); Q = K = V = X @ W
M = Q @ K.T/ np.sqrt(H)
A = np.divide(M , M.sum(axis=1) + 1e-12)
Y = A @ V
print(f"{Y.sum():.0f}")
```

---

## 题目6: 支持LoRA的Attention实现（P3658）

- **难度**: 中等
- **源**: [core46#第3题-p3658](../AI_编程题_Python解答_核心46题.md#第3题-p3658)

### 题目描述

相对于全量微调，LoRA 微调提出了一种低秩分解的方法，只需在原模型参数基础上增加少量的可训练参数，大幅降低计算成本和内存占用。具体而言，对于原始的预训练权重矩阵 W，LoRA 做以下改进：

$$W' = W + B \times A$$

其中：
- W 为原始权重（冻结不变）
- $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times d}$ 为新增的低秩矩阵，$r \ll d$，秩 r 一般很小
- 微调时只更新 A、B 这两个矩阵，显著减少训练的参数数量

请实现支持 LoRA 的 Attention 计算函数 `LoRA_Attention(x, W_q, W_k, W_v, A, B)`。为简化实现，仅需支持 Attention 中 Q 的 LoRA 结构实现即可。实现时请使用 float64 位精度。

**输入描述**：
- 第 1 行：b, d, r，其中 b 为 batch size，d 为特征的长度，r 为 LoRA 矩阵的秩，$b \geq 1, d \geq 1, r \geq 0$
- 第 2 行：输入 x，长度为 $b \times d$
- 第 3-5 行：$W_q, W_k, W_v$，长度为 $d \times d$
- 若 $r > 0$，则：
  - 第 6 行：A，长度为 $r \times d$
  - 第 7 行：B，长度为 $d \times r$

**输出描述**：LoRA Attention 计算的结果，输出保留四位小数，不足四位小数的补 0

### 样例
```
输入：
2 5 3
-0.58 -0.52 -0.02 0.56 0.79 0.06 -0.64 -0.04 -0.20 -0.38
0.24 -0.72 -0.66 0.96 0.02 -0.43 -0.24 0.19 -0.85 -0.35 0.69 -0.09 0.99 0.21 -0.06 0.55 0.57 0.97 0.58 -0.16 0.64 0.02 -0.71 0.53 -0.90
0.07 -0.16 -0.47 -0.32 -0.92 0.13 -0.74 -0.87 0.05 0.33 0.37 0.75 0.57 0.14 -0.62 0.67 -0.62 -0.85 0.09 -0.90 0.22 0.97 -0.68 0.61 0.48
0.39 -0.74 0.84 0.21 0.44 -0.59 -0.07 -0.84 -0.70 0.86 -0.12 -0.06 0.45 -0.43 -0.09 -0.73 0.56 -0.62 0.36 -0.87 -0.97 -0.48 0.71 0.07 -0.28
0.25 0.58 -0.04 -0.94 0.45 -0.60 0.89 0.94 0.35 -0.76 -0.47 -0.40 0.10 0.23 0.25
-0.18 -0.11 0.60 0.37 0.75 0.51 -0.76 -0.39 -0.81 -0.88 -0.43 -0.88 0.15 -0.46 -0.24

输出：
0.3499 0.0803 0.0376 -0.1791 0.3952 0.4112 0.2240 -0.0239 -0.2177 0.4478
```
```
输入：
1 3 2
0.58 -0.65 -0.63
-0.74 -0.71 0.65 0.70 -0.14 0.01 -0.84 0.20 0.25
-0.60 0.51 -0.12 -0.35 0.57 -0.38 -0.44 -0.82 0.53
0.14 0.03 -0.27 0.10 -0.12 0.85 -0.55 0.10 -0.43
0.65 0.32 -0.42 -0.62 -0.88 -0.70
-0.66 0.49 0.09 -0.21 0.48 0.41

输出：
0.2318 -0.3995 -0.1131
```

### 思路

1. **LoRA 思路**：
   - 原始权重 $W_q$ 冻结
   - 新增低秩矩阵 $A \in \mathbb{R}^{r \times d}, B \in \mathbb{R}^{d \times r}$，形成：$W_q' = W_q + B \times A$
   - 若 $r = 0$，直接用原始 $W_q$

2. **Attention 计算步骤**：
   - 计算 $Q = XW_q'^T$，$K = XW_k^T$，$V = XW_v^T$
   - 打分并缩放：$S = \frac{QK^T}{\sqrt{d}}$
   - 对每一行做稳定 softmax（减去行最大值）
   - 输出：$O = \text{softmax}(S) \cdot V$

3. **实现要点**：
   - float64 精度，避免溢出
   - softmax 时对每行减去最大值
   - 输出拉平，保留四位小数，-0.0000 特判为 0.0000

### 复杂度

- **时间复杂度**：$O(b \cdot d^2 + b^2 \cdot d)$
  - LoRA 权重计算：$O(d \cdot r \cdot d) = O(d^2 \cdot r)$（当 $r \ll d$ 时可忽略）
  - 计算 Q、K、V：$O(b \cdot d^2)$
  - 计算注意力分数：$O(b^2 \cdot d)$
  - Softmax 和输出：$O(b^2 \cdot d)$

- **空间复杂度**：$O(b \cdot d + b^2)$，主要存储 Q、K、V 和注意力分数矩阵

### 我的代码
```python
import numpy as np
B, D, R = map(int, input().split())
X = np.array(list(map(float, input().split()))).reshape(B, D) # 2x5
Wq = np.array(list(map(float, input().split()))).reshape(D, D)
Wk = np.array(list(map(float, input().split()))).reshape(D, D) # 5x5
Wv = np.array(list(map(float, input().split()))).reshape(D, D)
if R > 0:
    A1 = np.array(list(map(float, input().split()))).reshape(R, D) # 3 x 5
    A2 = np.array(list(map(float, input().split()))).reshape(D, R) # 5 x 3
Q = X @ (Wq + A2 @ A1 if R > 0 else 0).T; K = X @ Wk.T; V = X @ Wv.T; QK = Q @ K.T / np.sqrt(D); 
def softmax(x, axis=-1):
    x = x.astype(np.float64)                # 题目常要求 float64
    x = x - np.max(x, axis=axis, keepdims=True)  # 数值稳定
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
A = softmax(QK); O = (A @ V).reshape(-1)
print(" ".join([f"{xx:.4f}" for xx in O]))
```

---

## 题目7: 传感器数据分析（P3562）

- **难度**: 中等
- **核心**: Self-Attention + FC 推理
- **源**: [core46#第3题-p3562](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- 网络结构：两层 Self-Attention → 两层 FC
- 输入：L×D 的序列
- Attention：$\text{softmax}(\frac{QK^T}{\sqrt{d}})V$
- 无非线性激活函数

### 关键规则
1. 每层有 Wq, Wk, Wv（D×D）
2. FC 有权重 W 和偏置 b
3. 输出格式：逗号分隔，保留 2 位小数

### 样例
```
输入:
4,1
1.00,-3.00,9.50,6.50
-0.20
0.45
...

输出:
0.04,0.04,0.05,0.05
```

### 思路
1. 解析输入（逗号分隔）
2. 两层 Attention + FC
3. softmax 需要数值稳定（减最大值）

### 复杂度
- 时间: O(L² · D + L · D²)
- 空间: O(L² + L · D)

### 我的代码
```python
import sys
import numpy as np
import math

def softmax_rows(M):
    mx = np.max(M, axis=1, keepdims=True)
    E = np.exp(M - mx)
    return E / np.sum(E, axis=1, keepdims=True)

def attn(X, Wq, Wk, Wv, D):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = (Q @ K.T) / math.sqrt(D)
    return softmax_rows(S) @ V

def main():
    lines = [sys.stdin.readline().strip() for _ in range(12)]
    L, D = map(int, lines[0].split(','))

    def parse(idx, cnt):
        return np.array(list(map(float, lines[idx].split(',')))), idx + 1

    idx = 1
    seq, idx = parse(idx, L*D)
    seq = seq.reshape(L, D)

    Wq1, idx = parse(idx, D*D); Wq1 = Wq1.reshape(D, D)
    Wk1, idx = parse(idx, D*D); Wk1 = Wk1.reshape(D, D)
    Wv1, idx = parse(idx, D*D); Wv1 = Wv1.reshape(D, D)
    Wfc1, idx = parse(idx, D*D); Wfc1 = Wfc1.reshape(D, D)
    bfc1, idx = parse(idx, D)

    Wq2, idx = parse(idx, D*D); Wq2 = Wq2.reshape(D, D)
    Wk2, idx = parse(idx, D*D); Wk2 = Wk2.reshape(D, D)
    Wv2, idx = parse(idx, D*D); Wv2 = Wv2.reshape(D, D)
    Wfc2, idx = parse(idx, D*D); Wfc2 = Wfc2.reshape(D, D)
    bfc2, idx = parse(idx, D)

    Y1 = attn(seq, Wq1, Wk1, Wv1, D)
    Z1 = Y1 @ Wfc1 + bfc1
    Y2 = attn(Z1, Wq2, Wk2, Wv2, D)
    Z2 = Y2 @ Wfc2 + bfc2

    out = Z2.flatten()
    print(",".join(f"{x:.2f}" for x in out))

if __name__ == "__main__":
    main()
```

---

## 📌 易错点总结

1. **Self-Attention 缩放**：除以 $\sqrt{d}$，不是 $d$
2. TODO

---

## 🔗 相关文件

- 源文件：`../AI_编程题_Python解答_核心46题.md`
- 索引：`../ai_core46_index.md`

---

## 📝 代码答案

### 题目1: ViT Patch Embedding层实现（P4481）

```python
# 计算 Patch Embedding 输出维度的函数
def get_embedding_shape(img_size, patch_size, channel, embedding_dim):
    # 每一维上的 patch 个数
    num_per_dim = img_size // patch_size
    # 总的 patch 个数
    num_patches = num_per_dim * num_per_dim
    # 加上一个 cls token
    num_tokens = num_patches + 1
    # 返回 (序列长度, 嵌入维度)
    return num_tokens, embedding_dim

def main():
    # 读取输入：img_size patch_size channel embedding_dim
    img_size, patch_size, channel, embedding_dim = map(int, input().split())
    # 调用函数计算结果
    tokens, dim = get_embedding_shape(img_size, patch_size, channel, embedding_dim)
    # 按题目要求输出
    print(tokens, dim)

if __name__ == "__main__":
    main()
```

### 题目2: 基于空间连续块的稀疏注意力机制（P4275）

```python
import sys
import math
import numpy as np

# 核心功能：根据题意计算最终整数化得分
def solve(n: int, d: int, b: int, X: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> int:
    m = (n + b - 1) // b  # 块数
    A = []  # 压缩注意力得分序列

    sqrt_d = math.sqrt(d)

    # 逐块计算 a_k
    for k in range(m):
        start = k * b
        end = min((k + 1) * b, n)
        block = X[start:end]  # 该块的所有 token，形状 (len, d)

        # 平均池化 h_k
        h_k = block.mean(axis=0)

        # 两层 MLP：t = W1·h + b1，r = ReLU(t)，c = W2*r + b2(逐维加1)
        t = float(W1.dot(h_k)) + 2.0
        r = max(0.0, t)
        c = W2 * r + 1.0  # 广播加 1
        a_k = float(c.sum()) / sqrt_d
        A.append(a_k)

    # 线性扫描寻找最优切分点，使 min(左和, 右和) 最大
    T = sum(A)
    best = -1e100
    pref = 0.0
    for s in range(1, m):  # 必须切成两个非空段
        pref += A[s - 1]
        best = max(best, min(pref, T - pref))

    S = best
    return int(round(S * 100.0))

def main():
    data = sys.stdin.read().strip().split()
    it = iter(data)

    # 读入 n d b
    n = int(next(it)); d = int(next(it)); b = int(next(it))

    # 读入 n 行，每行 d 个浮点
    xs = [ [float(next(it)) for _ in range(d)] for _ in range(n) ]
    X = np.array(xs, dtype=float)

    # 读入 W1, W2（各 d 个数）
    W1 = np.array([float(next(it)) for _ in range(d)], dtype=float)
    W2 = np.array([float(next(it)) for _ in range(d)], dtype=float)

    ans = solve(n, d, b, X, W1, W2)
    print(ans)

if __name__ == "__main__":
    main()
```

### 题目3: 动态注意力掩码调度问题（P4227）

```python
import numpy as np

def solve(n, d, vectors, capacities):
    # 步骤1：对所有特征向量进行RMSNorm归一化
    normalized = []
    for vec in vectors:
        # 计算均方根值
        rms = np.sqrt(np.mean(np.array(vec) ** 2))
        # 归一化
        normalized.append(np.array(vec) / rms)

    # 步骤2：计算注意力得分的平方
    A_squared = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            # 计算点积
            dot_product = np.dot(normalized[i], normalized[j])
            # 缩放并计算平方
            A_ij = dot_product / np.sqrt(d)
            A_squared[i][j] = A_ij ** 2

    # 步骤3：贪心选择，最大化全局注意力信息总量S
    S = 0.0
    for j in range(1, n):
        # 收集位置j的所有前置位置的注意力得分平方
        scores = []
        for i in range(j):
            scores.append(A_squared[i][j])

        # 降序排序，选择最大的c_j个
        scores.sort(reverse=True)
        S += sum(scores[:capacities[j]])

    # 步骤4：输出整数化得分
    return round(100 * S)

if __name__ == "__main__":
    # 读取n和d
    n, d = map(int, input().split())

    # 读取特征向量
    vectors = []
    for _ in range(n):
        vec = list(map(float, input().split()))
        vectors.append(vec)

    # 读取计算容量
    capacities = list(map(int, input().split()))

    # 计算并输出结果
    result = solve(n, d, vectors, capacities)
    print(result)
```

### 题目4: Masked Multi-Head Self-Attention 实现（P3843）

```python
import sys
import numpy as np
from ast import literal_eval

def to_str(arr):
    """递归把嵌套 list 转成字符串，数值固定两位小数且无引号；把 -0.00 规整为 0.00"""
    if isinstance(arr, list):
        return "[" + ", ".join(to_str(x) for x in arr) + "]"
    else:
        # 数值分支
        v = float(arr)
        s = f"{v:.2f}"
        # 规整 -0.00 -> 0.00
        if s == "-0.00":
            s = "0.00"
        return s

def softmax_stable(x, axis=-1):
    # 数值稳定 softmax
    m = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - m)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def multi_head_self_attention(X, WQ, WK, WV, WO, num_heads):
    B, S, D = X.shape
    assert D % num_heads == 0, "d_model 必须能被 num_heads 整除"
    d_k = D // num_heads

    # 1) 线性映射
    Q = X @ WQ     # [B,S,D]
    K = X @ WK
    V = X @ WV

    # 2) 分头 -> [B,H,S,d_k]
    def split_heads(t):
        t = t.reshape(B, S, num_heads, d_k)     # [B,S,H,d_k]
        return np.transpose(t, (0, 2, 1, 3))    # [B,H,S,d_k]
    Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

    # 3) 注意力分数 [B,H,S,S]
    scores = (Qh @ np.transpose(Kh, (0,1,3,2))) / np.sqrt(d_k)

    # 4) 因果掩码：允许关注自己及之前位置 => 下三角为1，其余为0
    mask = np.tril(np.ones((S, S), dtype=np.float32))  # [S,S]
    mask = mask[None, None, :, :]                      # [1,1,S,S] 广播到 [B,H,S,S]
    scores = np.where(mask == 1, scores, -np.inf)

    # 5) softmax
    attn = softmax_stable(scores, axis=-1)  # [B,H,S,S]

    # 6) 加权求和
    heads = attn @ Vh                       # [B,H,S,d_k]

    # 7) 拼回 + 输出投影
    heads = np.transpose(heads, (0, 2, 1, 3))      # [B,S,H,d_k]
    concat = heads.reshape(B, S, D)                # [B,S,D]
    out = concat @ WO                              # [B,S,D]

    return out

def main():
    raw = sys.stdin.read().strip()
    # 按分号分割：num_heads;X;Q;K;V;W_O
    parts = [p.strip() for p in raw.split(';')]
    if len(parts) != 6:
        raise ValueError("输入应包含6段参数：num_heads;X;Q;K;V;W_O")

    num_heads = int(parts[0])
    X = np.array(literal_eval(parts[1]), dtype=float)
    WQ = np.array(literal_eval(parts[2]), dtype=float)
    WK = np.array(literal_eval(parts[3]), dtype=float)
    WV = np.array(literal_eval(parts[4]), dtype=float)
    WO = np.array(literal_eval(parts[5]), dtype=float)

    out = multi_head_self_attention(X, WQ, WK, WV, WO, num_heads)
    out = np.around(out, 2)                 # 保留两位小数
    print(to_str(out.tolist()))

if __name__ == "__main__":
    main()
```

### 题目5: 大模型Attention模块开发（P3712）

```python
import sys
import ast
import numpy as np

def solve(n, m, h):
    # 1) 构造 X 全 1，W 上三角全 1
    X = np.ones((n, m), dtype=float)
    W = np.triu(np.ones((m, h), dtype=float))  # W1=W2=W3 相同

    # 2) 计算 Q, K, V（矩阵乘法）
    Q = X @ W
    K = X @ W
    V = X @ W

    # 3) 计算 M=(Q·K^T)/sqrt(h)
    M = (Q @ K.T) / np.sqrt(float(h))

    # 4) "简化 softmax"：按行除以行和
    row_sum = M.sum(axis=1, keepdims=True)
    A = M / (row_sum + 1e-12)

    # 5) 计算 Y=A·V 并求和
    Y = A @ V
    total = float(Y.sum())

    # 6) 四舍五入输出整数
    return int(np.rint(total))

def main():
    s = sys.stdin.read().strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)) and len(val) == 3:
            n, m, h = map(int, val)
        else:
            n, m, h = map(int, s.split())
    except Exception:
        n, m, h = map(int, s.split())

    print(solve(n, m, h))

if __name__ == "__main__":
    main()
```

### 题目6: 支持LoRA的Attention实现（P3658）

```python
import sys
import numpy as np

def softmax(x):
    """
    计算softmax函数
    """
    x = x.astype(np.float64)
    max_vals = np.max(x, axis=1, keepdims=True)
    exp_vals = np.exp(x - max_vals)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

def LoRA_Attention(x, wq, wk, wv, A, B):
    """
    实现带有LoRA适配器的线性注意力机制
    """
    d = x.shape[1]

    # 应用LoRA适配器（如果提供）
    if A is not None and B is not None and A.size > 0 and B.size > 0:
        effective_wq = wq + B @ A
    else:
        effective_wq = wq

    # 计算查询、键和值
    Q = x @ effective_wq.T
    K = x @ wk.T
    V = x @ wv.T

    # 计算注意力分数
    scale_factor = 1.0 / np.sqrt(d)
    attention_scores = (Q @ K.T) * scale_factor

    # 应用softmax得到注意力权重
    attention_weights = softmax(attention_scores)

    # 计算输出
    output = attention_weights @ V
    return output

def format_output(values):
    """
    格式化输出，确保-0.0000显示为0.0000
    """
    formatted_values = []
    for value in values:
        formatted = f"{value:.4f}"
        if formatted == "-0.0000":
            formatted = "0.0000"
        formatted_values.append(formatted)
    return formatted_values

def main():
    # 读取输入数据
    data = list(map(float, sys.stdin.read().strip().split()))
    it = iter(data)

    # 读取维度参数
    b = int(next(it))
    d = int(next(it))
    r = int(next(it))

    # 读取输入矩阵
    x = np.array([next(it) for _ in range(b * d)]).reshape(b, d)

    # 读取权重矩阵
    wq = np.array([next(it) for _ in range(d * d)]).reshape(d, d)
    wk = np.array([next(it) for _ in range(d * d)]).reshape(d, d)
    wv = np.array([next(it) for _ in range(d * d)]).reshape(d, d)

    # 读取LoRA适配器参数（如果存在）
    if r > 0:
        A = np.array([next(it) for _ in range(r * d)]).reshape(r, d)
        B = np.array([next(it) for _ in range(d * r)]).reshape(d, r)
    else:
        A = None
        B = None

    # 计算输出
    output = LoRA_Attention(x, wq, wk, wv, A, B)

    # 格式化和打印结果
    flat_output = output.reshape(-1)
    formatted_output = format_output(flat_output)
    print(" ".join(formatted_output))

if __name__ == "__main__":
    main()
```

### 题目7: 传感器数据分析（P3562）

```python
import sys # 8,55
for i, line in enumerate(sys.stdin):
    it = iter(line.strip().split(','))
    if i == 0: 
        L, D = int(next(it)),int(next(it))
    if i == 1:
        X = [[float(next(it)) for _ in range(D)] for _ in range(L)]
    if i == 2:
        Wq1 = [[float(next(it)) for _ in range(D)] for _ in range(D)]
    if i == 3:
        Wk1 = [[float(next(it)) for _ in range(D)] for _ in range(D)]
    if i ==4:
        Wv1 = [[float(next(it)) for _ in range(D)] for _ in range(D)]
    if i == 5:
        W1 = [[float(next(it)) for _ in range(D)] for _ in range(D)]
    if i == 6:
        b1 = [float(next(it)) for _ in range(D)] 
    if i == 7:
        Wq2 = [[float(next(it)) for _ in range(D)] for _ in range(D)]
    if i == 8:
        Wk2 = [[float(next(it)) for _ in range(D)] for _ in range(D)]
    if i == 9:
        Wv2 = [[float(next(it)) for _ in range(D)] for _ in range(D)]
    if i == 10:
        W2 = [[float(next(it)) for _ in range(D)] for _ in range(D)]
    if i == 11:
        b2 = [float(next(it)) for _ in range(D)] 
import numpy as np
def softmax(x):
    exps = np.exp(x-np.max(x,axis=-1, keepdims=True)) 
    return exps/exps.sum(axis=-1, keepdims=True)
X,Wq1,Wk1,Wv1,W1,b1,Wq2,Wk2,Wv2,W2,b2 = list(map(np.array, [X,Wq1,Wk1,Wv1,W1,b1,Wq2,Wk2,Wv2,W2,b2]))
Q1,K1,V1 = X @ Wq1,X @ Wk1,X @ Wv1; A=softmax(Q1 @ K1.T/np.sqrt(D)) @ V1;
H1 = A @ W1 + b1;
Q2,K2,V2 = H1 @ Wq2,H1 @ Wk2,H1 @ Wv2; A2=softmax(Q2 @ K2.T/np.sqrt(D)) @ V2;
H2 = A2 @ W2 + b2
print(",".join(f"{a:.2f}" for a in H2.ravel()))
```
