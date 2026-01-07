2025年8月28日


# 第1题-基于决策树预判资源调配优先级（P3492）

## 题目内容

在无线通信系统中为了资源调配更加合理和及时，需要对未来一段时间网络的负载做出预判以确定资源调配的优先级。决策树是一种常用的模型，可以根据采集到的状态数据，结合时间段、天气等因素来推测合理的资源调配优先级。现在将一个训练好的分类决策树模型 **Tree** 安装在系统中，请实现决策树的推算算法，根据输入特征返回资源调配优先级。

---

## 输入描述

输入包含多行数据：

* 首行是属性参数；
* 后续 (m) 行是决策树模型的结构数据；
* 再后续 (n) 行是待推理的样本。

### 1. 属性参数（首行）

包含 3 个 `int` 类型数据，分别表示：

* 特征数量 (f)
* 树模型的节点数 (m)
* 待推理样本的行数 (n)

### 2. 决策树模型 (T)（(m) 行 5 列）

一行数据表示决策树的一个节点，首行表示根节点（即 (T) 矩阵中的第 0 行），共有 (m) 个节点。
每行 5 列分别表示：

1. 分裂特征的下标（`int`）
2. 分裂特征的阈值（`float`）
3. 当前节点左子节点的行号（`int`，即 (T) 矩阵中第几行）
4. 当前节点右子节点的行号（`int`，即 (T) 矩阵中第几行）
5. 分类结果（`int`，即资源调配优先级）

补充说明：

* 下标和行号均从 0 开始。
* 模型结构中有意义的数据均 (\ge 0)，无意义数据统一用 (-1) 表示。
  例如：叶子节点无分裂特征，因此该字段填入 (-1)。
* 决策规则：若样本特征的取值 (\le) 分裂特征的阈值，则进入左子树，否则进入右子树。

### 3. 待推理样本（(n) 行 (f) 列）

一行数据表示一个待推理的样本；每行包含 (f) 个特征，数据类型为 `float`。

---

## 输出描述

返回分类结果。
注：一行一个分类结果，数据类型为 `int`。

---

## 样例 1

### 输入

```
2 5 2
0 2.5 1 2 -1
-1 -1 -1 -1 1
1 5.0 3 4 -1
-1 -1 -1 -1 2
-1 -1 -1 -1 3
1.2 3.4
5.6 6.0
```

### 输出

```
1
3
```

### 说明

* 第 1 行：特征数量为 2，模型节点数为 5，待推理样本为 2 条。
* 第 2 到第 6 行：决策树模型，共 5 个节点。首行（第 2 行输入数据）为根节点。
* 第 7 到第 8 行：2 个待推理样本，每个样本包含 2 个特征。
* 输出：两条样本的推理结果分别为分类 1、分类 3。

---

## 样例 2

### 输入

```
3 9 2
0 6.1 1 2 -1
2 7.0 3 4 -1
1 6.5 5 6 -1
-1 -1 -1 -1 1
1 10.3 7 8 -1
-1 -1 -1 -1 5
-1 -1 -1 -1 6
-1 -1 -1 -1 3
-1 -1 -1 -1 4
3.2 9.2 6.2
6.3 3.2 12.0
```

### 输出

```
1
5
```

### 说明

* 第 1 行：特征数量为 3，模型节点数为 9，待推理样本为 2 条。
* 第 2 到第 10 行：决策树模型，共 9 个节点。首行（第 2 行输入数据）为根节点。
* 第 11 到第 12 行：2 个待推理样本，每个样本包含 3 个特征。
* 输出：两条样本的推理结果分别为分类 1、分类 5。

---

## 解题思路

### 数据结构设计

用一个结构体（或类）存储单个节点的全部信息：

* `feature_index`：分裂特征下标
* `threshold`：分裂阈值
* `left`：左子节点行号
* `right`：右子节点行号
* `label`：分类结果（仅在叶子节点有效）

### 读取模型

读取 (m) 个节点并存入数组或列表，**编号即其行号**。

### 推理过程

对每个样本，从根节点（0）开始：

1. 如果当前节点是叶节点（`feature_index == -1`），输出该节点分类结果；
2. 否则比较样本在 `feature_index` 特征上的值与 `threshold`：

   * 若 (\text{feature} \le \text{threshold})，走左子节点；
   * 否则走右子节点；
3. 重复直到到达叶节点。

### 复杂度

* 对每个样本，最多遍历树的高度 (h)，时间复杂度为：(;O(n\cdot h))
* 节点总数为 (m)，一般 (h \ll m)
* 空间复杂度为：(;O(m))

---

## 参考代码（Python）

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

# P3493-第2题-Group卷积实现

## 题目内容

卷积（Convolution）是计算视觉中常用的计算算子，广泛应用于图像分类、检测、跟踪等多领域。

如下图所示，以 2 个三维张量卷积计算为例，取输入张量（通道数、高度、宽度），卷积核（通道数、高度、宽度），二者执行卷积计算要求其通道数相同。

当取卷积计算步长、填充、膨胀、无偏置项（bias）时，卷积核在输入张量上从左至右，从上至下滑动，分别与滑窗所重叠的输入张量切片，逐元素相乘求和后，得到输出张量的各元素。

例如：
[
\begin{aligned}
y_{0,0}=&x_{0,0,0}k_{0,0,0}+x_{0,0,1}k_{0,0,1}+x_{0,1,0}k_{0,1,0}+x_{0,1,1}k_{0,1,1}\
&+x_{1,0,1}k_{1,0,1}+x_{1,1,0}k_{1,1,0}+x_{1,1,1}k_{1,1,1}\
&+x_{2,0,0}k_{2,0,0}+x_{2,1,0}k_{2,1,0}+x_{2,1,1}k_{2,1,1}=72
\end{aligned}
]

面向不同的应用需求，卷积存在多类变种。分组卷积（Group Convolution）即是随 2012 年 AlexNet 提出的一种变种，其将输入张量和卷积核分组后，分别执行卷积计算，然后把多个输出张量进行融合。

例如，输入张量尺寸为 (1\times32\times32\times32)（其中首个维度 1 为样本数），卷积核尺寸为 (4\times16\times3\times3)（其中首个维度 4 为输出张量通道数，亦可理解为卷积核个数），分组数为 2 时：输入张量被切分为两组 (1\times16\times32\times32)，卷积核被切分为两组 (2\times16\times3\times3)，分组进行无 padding 的卷积计算后，将两组尺寸为 (1\times2\times30\times30) 的计算结果在第 2 个维度拼接，得到尺寸为 (1\times4\times30\times30) 的输出张量。

请不使用 PyTorch、MindSpore、PaddlePaddle 等 AI 框架，使用编程语言原生库，编写一个支持分组卷积和深度卷积前向传播的函数，根据输入张量、卷积核、分组数，计算输出张量。

## 输入描述

输入包含 5 行数据：

* **in_data**：4 维输入张量展开后的数据序列，以空格分隔的正整数；
* **in_shape**：4 维输入张量的形状，以空格分隔的 4 个正整数，依次为

  * `batch_size`（样本数）
  * `in_channels`（输入张量通道数）
  * `height`（高度）
  * `width`（宽度）
* **kernel_data**：4 维卷积核展开后的数据序列，以空格分隔的正整数；
* **kernel_shape**：4 维卷积核的形状，以空格分隔的 4 个正整数，依次为

  * `out_channels`（输出张量通道数）
  * `k_channels`（卷积核通道数）
  * `kernel_h`（卷积核高度）
  * `kernel_w`（卷积核宽度）
* **groups**：分组数，需满足
  [
  in_channels \bmod groups = 0,\quad
  out_channels \bmod groups = 0,\quad
  k_channels = \frac{in_channels}{groups}.
  ]

## 输出描述

* **out_data**：4 维输出张量展开后的数据序列，以空格分隔的正整数；
* **out_shape**：4 维输出张量的形状，以空格分隔的 4 个正整数，依次为

  * `batch_size`（样本数）
  * `out_channels`（输出张量通道数）
  * `height`（高度）
  * `width`（宽度）

若输入张量和卷积核的形状与 group 的取值存在冲突，或出现其它取值冲突导致无法执行卷积计算，则 **out_data 和 out_shape 均返回 -1**。

## 样例

### 样例 1

**输入**

```
1 2 3 4 5 6 7 8
1 2 2 2
1 0 0 1 -1 0 0 -1
2 1 2 2
2
```

**输出**

```
5 -13
1 2 1 1
```

**说明**

输入张量为：
[
\left[\left[\begin{array}{ll}
1 & 2 \
3 & 4
\end{array}\right],\left[\begin{array}{ll}
5 & 6 \
7 & 8
\end{array}\right]\right]
]
输入张量形状为 (1\times2\times2\times2)。

卷积核为：
[
\left[\left[\begin{array}{ll}
1 & 0 \
0 & 1
\end{array}\right],\left[\begin{array}{cc}
-1 & 0 \
0 & -1
\end{array}\right]\right]
]
卷积核形状为 (2\times1\times2\times2)。

分组数为 2，输出张量为 ([5,-13])，输出张量形状为 (1\times2\times1\times1)。

### 样例 2

**输入**

```
1 2 3 4 5 6 7 8 9
1 1 3 3
1 0 0 -1
1 1 2 2
2
```

**输出**

```
-1
-1
```

**说明**

由于 (in_channels=1)、(out_channels=1)，不满足
[
in_channels \bmod groups = 0,\quad out_channels \bmod groups = 0
]
因此 out_data 和 out_shape 均返回 -1。

## 解题思路

（整理自页面“题解/题面概述”，仅做排版修复）

* 给定展开后的输入张量与卷积核及其形状，和分组数 `groups`，实现分组卷积（包含深度卷积的特例）前向计算。默认 `stride=1`、`padding=0`、`dilation=1`。
* 若形状与 `groups` 不合法或输出空间维度非正，则输出 `-1`。
* 关键条件：
  [
  in_channels \bmod groups = 0,\quad
  out_channels \bmod groups = 0,\quad
  k_channels = \frac{in_channels}{groups}.
  ]
* 输出尺寸：
  [
  H_{out} = H - K_h + 1,\quad W_{out} = W - K_w + 1,
  ]
  需要 (H_{out}>0,;W_{out}>0)。
* 深度卷积：是分组卷积的特例，`groups=in_channels, k_channels=1`，允许
  [
  out_channels = groups \times depth_multiplier.
  ]
* 思路：

  1. 解析 5 行输入，校验数据长度与形状乘积一致；
  2. 校验分组与通道约束；
  3. 计算 (H_{out},W_{out}) 并校验为正；
  4. 按 `N、组 g、组内输出通道 oc、空间位置 (oh,ow)、组内输入通道 kc、核 (kh,kw)` 六重循环累加；
  5. 扁平化顺序为 (N\rightarrow C\rightarrow H\rightarrow W) 输出。

## 数据结构设计

* 采用一维数组存储展开后的张量数据，并用步长（stride）/索引函数实现四维坐标到一维下标的映射：

  * 输入张量索引：((n,c,h,w)\mapsto n\cdot(C!HW)+c\cdot(HW)+h\cdot W+w)
  * 卷积核索引：((oc,kc,kh,kw)\mapsto oc\cdot(KC!KH!KW)+kc\cdot(KH!KW)+kh\cdot KW+kw)
  * 输出张量索引：((n,oc,oh,ow)\mapsto n\cdot(OC!H_o!W_o)+oc\cdot(H_o!W_o)+oh\cdot W_o+ow)

## 推理/算法流程

1. 读入 5 行：`in_data, in_shape, kernel_data, kernel_shape, groups`。
2. 校验：

   * `in_shape` 与 `kernel_shape` 均为 4 个正整数，`groups` 为 1 个正整数；
   * `len(in_data)==N*C*H*W`，`len(kernel_data)==OC*KC*KH*KW`；
   * `C%G==0`，`OC%G==0`，且 `KC==C//G`；
   * (H_o=H-KH+1>0)，(W_o=W-KW+1>0)。
3. 分组计算：

   * 每组输出通道数 (OC_g=OC/G)；
   * 每组输入通道数 (KC_g=C/G)（也即 `KC`）；
   * 对每个 (n,g,oc,oh,ow)，累加该组对应输入通道与核窗口的乘积和。
4. 将输出按 (N\rightarrow C\rightarrow H\rightarrow W) 展开输出，并输出形状 `N OC Ho Wo`。

## 复杂度

* 时间复杂度：
  [
  O\bigl(N\cdot OC \cdot H_o \cdot W_o \cdot (C/G)\cdot KH \cdot KW\bigr)
  ]
* 空间复杂度：输出缓冲为 (O(N\cdot OC\cdot H_o\cdot W_o))（不含输入与卷积核本身）。

## Python 参考代码

（整理自页面 Python 参考实现，仅做排版修复；不依赖第三方库）

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
