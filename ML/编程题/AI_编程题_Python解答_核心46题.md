# AI方向 - 编程题（Python 解答版）

> **仅包含 AI 方向核心编程题（46 道）**
>
> ⚠️ **注意**：由于原始数据中代码缺少缩进信息，Python 代码需要手动调整缩进后才能运行

---

## 📚 目录

### 2025年12月17日-AI方向

- [第2题-使用线性回归预测手机售价](#第2题-p4532) (中等) [[P4532](https://codefun2000.com/p/P4532)]
- [第3题-模型量化最小误差](#第3题-p4533) (困难) [[P4533](https://codefun2000.com/p/P4533)]

### 2025年12月3日-AI方向

- [第2题-基于剪枝的神经网络模型压缩](#第2题-p4518) (中等) [[P4518](https://codefun2000.com/p/P4518)]
- [第3题-智能客户分群与新用户定位(KMeans均衡分区版)](#第3题-p4519) (困难) [[P4519](https://codefun2000.com/p/P4519)]

### 2025年11月20日-留学生AI方向

- [第2题-Vision Transformer中的Patch Embdding层实现](#第2题-p4481) (简单) [[P4481](https://codefun2000.com/p/P4481)]
- [第3题-带Padding的卷积计算](#第3题-p4482) (中等) [[P4482](https://codefun2000.com/p/P4482)]

### 2025年11月19日-AI方向

- [第2题-终端款型聚类识别](#第2题-p4475) (中等) [[P4475](https://codefun2000.com/p/P4475)]
- [第3题-Prompt上下文信息精简:找出二叉树中的最大值子树](#第3题-p4476) (困难) [[P4476](https://codefun2000.com/p/P4476)]

### 2025年11月12日-AI方向

- [第2题-全连接层INT8非对称量化实现](#第2题-p4464) (中等) [[P4464](https://codefun2000.com/p/P4464)]
- [第3题-基于决策树的QAM调制符合检测](#第3题-p4465) (困难) [[P4465](https://codefun2000.com/p/P4465)]

### 2025年11月6日-留学生AI方向

- [第2题-医疗诊断模型的训练与更新](#第2题-p4447) (中等) [[P4447](https://codefun2000.com/p/P4447)]
- [第3题-卷积操作](#第3题-p4448) (中等) [[P4448](https://codefun2000.com/p/P4448)]

### 2025年11月5日-AI方向

- [第2题-多目标推荐排序模型优化](#第2题-p4441) (困难) [[P4441](https://codefun2000.com/p/P4441)]
- [第3题-须从规矩出方圆](#第3题-p4442) (困难) [[P4442](https://codefun2000.com/p/P4442)]

### 2025年10月29日-AI方向

- [第2题-实体匹配结果合并问题](#第2题-p4343) (中等) [[P4343](https://codefun2000.com/p/P4343)]
- [第3题-商品购买预测](#第3题-p4344) (中等) [[P4344](https://codefun2000.com/p/P4344)]

### 2025年10月23日-AI方向(留学生)

- [第2题-人脸关键点对齐](#第2题-p4277) (简单) [[P4277](https://codefun2000.com/p/P4277)]
- [第3题-卷积结构实现](#第3题-p4278) (中等) [[P4278](https://codefun2000.com/p/P4278)]

### 2025年10月22日-AI方向

- [第2题-最大能量路径](#第2题-p4274) (中等) [[P4274](https://codefun2000.com/p/P4274)]
- [第3题-基于空间连续块的稀疏注意力机制](#第3题-p4275) (中等) [[P4275](https://codefun2000.com/p/P4275)]

### 2025年10月17日-AI方向

- [第2题-利用大规模预训练模型实现智能告警聚类与故障诊断](#第2题-p4238) (中等) [[P4238](https://codefun2000.com/p/P4238)]
- [第3题-反向传播实现](#第3题-p4239) (困难) [[P4239](https://codefun2000.com/p/P4239)]

### 2025年10月15日-AI方向

- [第2题-动态注意力掩码调度问题](#第2题-p4227) (中等) [[P4227](https://codefun2000.com/p/P4227)]
- [第3题-基于二分Kmeans算法的子网分割问题](#第3题-p4228) (中等) [[P4228](https://codefun2000.com/p/P4228)]

### 2025年10月10日-AI方向

- [第2题-数据聚类及噪声点识别](#第2题-p3874) (中等) [[P3874](https://codefun2000.com/p/P3874)]
- [第3题-经典LSTM模型结构实现](#第3题-p3875) (中等) [[P3875](https://codefun2000.com/p/P3875)]

### 2025年10月10日(留学生)-AI岗

- [第2题-磁盘故障检测的特征工程](#第2题-p3871) (困难) [[P3871](https://codefun2000.com/p/P3871)]
- [第3题-基于逻辑回归的意图分类器](#第3题-p3872) (中等) [[P3872](https://codefun2000.com/p/P3872)]

### 2025年9月28日-AI方向

- [第2题-Yolo检测器中的anchor聚类](#第2题-p3842) (中等) [[P3842](https://codefun2000.com/p/P3842)]
- [第3题-Masked Multi-Head Self-Attention 实现](#第3题-p3843) (困难) [[P3843](https://codefun2000.com/p/P3843)]

### 2025年9月24日-AI岗

- [第2题-无线网络优化中的基站聚类分析](#第2题-p3791) (困难) [[P3791](https://codefun2000.com/p/P3791)]
- [第3题-基于决策树的无线状态预策](#第3题-p3792) (中等) [[P3792](https://codefun2000.com/p/P3792)]

### 2025年9月18日(留学生)-AI岗

- [第2题-最大能量路径](#第2题-p3718) (中等) [[P3718](https://codefun2000.com/p/P3718)]
- [第3题-数据中心水温调节档位决策](#第3题-p3719) (中等) [[P3719](https://codefun2000.com/p/P3719)]

### 2025年9月17日-AI岗

- [第2题-大模型Attention模块开发](#第2题-p3712) (中等) [[P3712](https://codefun2000.com/p/P3712)]
- [第3题-大模型分词](#第3题-p3713) (中等) [[P3713](https://codefun2000.com/p/P3713)]

### 2025年9月12日-AI岗

- [第2题-二叉树中序遍历的第k个祖先节点](#第2题-p3657) (中等) [[P3657](https://codefun2000.com/p/P3657)]
- [第3题-支持LoRA的Attention实现](#第3题-p3658) (困难) [[P3658](https://codefun2000.com/p/P3658)]

### 2025年9月10日-国内-AI

- [第2题-历史的窗口搜索](#第2题-p3639) (困难) [[P3639](https://codefun2000.com/p/P3639)]
- [第3题-多尺寸窗口滑动的特征转换](#第3题-p3640) (困难) [[P3640](https://codefun2000.com/p/P3640)]

### 2025年9月4日-留学生-AI

- [第2题-大模型训练数据均衡分配算法](#第2题-p3561) (中等) [[P3561](https://codefun2000.com/p/P3561)]
- [第3题-传感器数据分析](#第3题-p3562) (中等) [[P3562](https://codefun2000.com/p/P3562)]

### 2025年9月3日-国内-AI

- [第2题-大模型训练MOE场景路由优化算法](#第2题-p3553) (中等) [[P3553](https://codefun2000.com/p/P3553)]
- [第3题-云存储设备故障预测](#第3题-p3552) (中等) [[P3552](https://codefun2000.com/p/P3552)]

### 2025年8月27日-国内-AI

- [第2题-标签样本数量](#第2题-p3479) (中等) [[P3479](https://codefun2000.com/p/P3479)]
- [第3题-F1值最优的决策树剪枝](#第3题-p3480) (中等) [[P3480](https://codefun2000.com/p/P3480)]

---

## 2025年12月17日-AI方向

<a id="第2题-p4532"></a>

### 第2题-使用线性回归预测手机售价（P4532）- 中等





手机的售价跟手机的软硬件特性有关系。硬件规格越高、软件特性越丰富，则手机给消费者提供的价值越大，同时手机的售价越高。我们在市面上收集了若干款手机，从硬件能力、系统流畅度、AI能力3个方面对这些手机进行打分，并记录这些手机的分数和售价。请你使用最小二乘法建立线性回归模型，对这3个特征和手机售价的关系进行线性回归，然后预测若干款待上市的手机型号应该卖多少价钱。
该题目的数据保证最小二乘法有解析解。建议使用正规方程法，即矩阵求解。如果使用梯度下降法，请迭代至预测值的小数点后第一位稳定不变，以保证精度满足题目要求。
输入描述
第1行，正整数K，已知的手机个数。
第2行，K个手机的特征和售价记录，均为整数。用空格分割，一共4K数字是售个数字。每4个数字为一组，第1-3个数字为特征值,第4个数字是售价。。
第3行，正整数N，待估价的手机数量。
第4行，N个手机型号对应的特征，均为整数。用空格分割，一共3N个数字。每3个数字为一组，分别为3个特征值。
输出描述
N个正整数，代表每个手机的价格，使用空格分割，四舍五入取整数。
样例1
输入

10
86 99 20 3595 175 171 90 6596 194 42 47 4691 192 172 26 5927 44 20 168 4169 61 138 64 4348 161 42 85 4791 197 181 99 7126 170 55 95 5208 26 158 142 5231
2
159 135 173 120 144 59

输出
7116 5120

说明
已知10台手机的评分和售价，以第1台手机型号为例，硬件能力评分为86、系统流畅度评分为99、AI能力评分为20，售价为3595。以此类推。
需要求解2台手机的预期售价，其中第1台手机的硬件能力评分为159、系统流畅度评分为135、AI能力评分为173，使用正规方程法求解，得到的预期售价求整结果是7116。以此类推。
样例2
输入
4
30 23 24 1999 55 53 46 2999 68 85 78 3999 113 90 103 4999
1
126 114 143

输出
6009

说明
已知4台手机的评分和售价，以第1台手机为例，硬件能力评分为30、系统流畅度评分为23、AI能力评分为24，售价为1999。以此类推。需要求解1台手机的预期售价，这台手机的硬件能力评分为126、系统流畅度评分为114、AI能力评分为143，使用正规方程法求解，得到的预期售价的求整结果是6009。


#### 解答


解题思路
本题是一个多元线性回归建模问题：已知 K 部手机的三项评分特征与售价，要求拟合出线性关系，并用该关系预测新手机的价格。
1. 线性模型建立
设第 i 个样本的三维特征为 $x^{(i)}=(x_1^{(i)}, x_2^{(i)}, x_3^{(i)})$，对应售价为 $y^{(i)}$。
假设售价与特征满足线性关系：
$y^{(i)} = w_0 + w_1 x^{(i)}_1 + w_2 x^{(i)}_2 + w_3 x^{(i)}_3$
其中 $w_0$ 是偏置项，$w_1, w_2, w_3$ 是三个特征的权重。
为方便统一表示，把偏置并入特征，定义扩展特征：
$\tilde{x}^{(i)} = (1, x^{(i)}_1, x^{(i)}_2, x^{(i)}_3)$，参数向量为 $W=(w_0, w_1, w_2, w_3)$，则：
$y^{(i)} = W \cdot \tilde{x}^{(i)}$
2. 最小二乘目标
由于样本通常不能被一条直线（超平面）完全拟合，我们用最小二乘法，让预测值与真实值的平方误差之和最小：
$\min\limits_W \sum_{i=1}^{K}\left(W\cdot \tilde{x}^{(i)} - y^{(i)}\right)^2$
3. 求解方法（高斯消元/正规方程）
对上式求极值可得到一组线性方程（正规方程的结果），题目保证可解且数值稳定时，可以直接通过解线性方程组得到 W。工程实现上，不必显式写出复杂矩阵形式，只需构造一个 $4 \times 4$ 的系数矩阵和长度为 4 的常数向量，然后使用 高斯消元（Gauss Elimination） 求解：

扫描所有样本，累加得到系数矩阵中的各项（相当于统计不同特征乘积的和）
用高斯消元解出 w0,w1,w2,w3w_0,w_1,w_2,w_3w0,w1,w2,w3
对每个待预测手机特征代入 y=w0+w1x1+w2x2+w3x3y = w_0+w_1x_1+w_2x_2+w_3x_3y=w0+w1x1+w2x2+w3x3 输出结果

复杂度分析
设：

已知手机数量为 (K)
特征维度为常数 4（含偏置）

时间复杂度

构造矩阵与计算 (X^T X)：(O(K))
矩阵求逆（4×4）：(O(1))
预测 (N) 台手机：(O(N))

总时间复杂度：(O(K + N))
空间复杂度

存储矩阵 (X, Y) 以及中间矩阵，规模固定

空间复杂度：(O(1))
代码实现

**Python 代码：**

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

---

<a id="第3题-p4533"></a>

### 第3题-模型量化最小误差（P4533）- 困难





在一个深度神经网络中，网络的权重通常以浮点数的形式存储。为了减少内存占用和提高计算效率，需要将这些浮点数量化为整数,例如可通过int(Wfloat∗28)int (W _{float }*2^8)int(Wfloat∗28)将一个小于1的浮点数量化为INT8INT8INT8。
假设我们有一组[N,H]的模型权重，其中：N表示网络的层数，H表示每一层的维度。现在需要将网络权重进行量化，已知权重已经过预处理缩放到合适的值，可通过Wq=int(Wfloat∗2Qi)W_q=int(W_{float} *2^{Q_i})Wq=int(Wfloat∗2Qi)直接量化到对应的比特位QiQ_iQi，同时定义量化误差为$\Delta = \left| W_{\text{float}} - \frac{W_q}{2q_i} \right|$
同一层选用的量化比特是相同的，不同层之间可选择不同的量化比特。定义整个模型每一层的量化比特数为[Q1,Q2,...,QN][Q_1,Q_2,...,Q_N][Q1,Q2,...,QN],并限定 Qi∈[2,4,8]Q_i∈ [2,4,8]Qi∈[2,4,8]，为了保证整体空间压缩足够小，需满足∑i=1NQi≤Qmax\sum_{i=1}^{N} Q_i \leq Q_{\text{max}}∑i=1NQi≤Qmax。请给出最优的量化方案，使得所有层的量化误差总和最小。
输入描述
第一行:N,H,QmaxN,H,Q_{max}N,H,Qmax
接下来N行是模型权重，每行H个系数，系数间用空格分隔(0<N<=300，0<H<=100，0<Qmax<=2400)(0<N<=300，0<H<=100，0<Q_{max} <=2400)(0<N<=300，0<H<=100，0<Qmax<=2400)
输出描述
请输出在最优方案下，整个网络的最小总量化误差。请将答案*100后取整输出(例如最小总量化误差为12.34567812.34567812.345678时，输出1234)
样例1
输入
3 10 6
0.669342691379556 0.6232664728193106 0.009648814115477689 0.25655923835608296 0.8542091541905418 0.22734652633918107 0.3856022177718754 0.4735219607872916 0.7352822546717339 0.8810700172773613
0.8998864964296006 0.5355025966489801 0.9114305820079228 0.7237159502129922 0.8114010729538255 0.5647698690173886 0.5656036144842292 0.2915636526042238 0.4633626072815791 0.4933586717844284
0.5681407125745037 0.972337640852664 0.33248445308239827 0.8870229039214033 0.2869760304712957 0.5912444652782809 0.2513253965878265 0.8945001503120086 0.7217848272492855 0.21360959764416299

输出
384

说明
N=3,H=10,∑Qi=6N=3,H=10,\sum Q_i=6N=3,H=10,∑Qi=6
每层只能使用2比特量化，量化误差分别为1.365,1.260,1.2191.365,1.260,1.2191.365,1.260,1.219,总量化误差3.8443.8443.844,*100后输出整数为384。
样例2
输入
3 10 24
0.669342691379556 0.6232664728193106 0.009648814115477689 0.25655923835608296 0.8542091541905418 0.22734652633918107 0.3856022177718754 0.4735219607872916 0.7352822546717339 0.8810700172773613
0.8998864964296006 0.5355025966489801 0.9114305820079228 0.7237159502129922 0.8114010729538255 0.5647698690173886 0.5656036144842292 0.2915636526042238 0.4633626072815791 0.4933586717844284
0.5681407125745037 0.972337640852664 0.33248445308239827 0.8870229039214033 0.2869760304712957 0.5912444652782809 0.2513253965878265 0.8945001503120086 0.7217848272492855 0.21360959764416299

输出
5

说明
N=3,H=10,∑Qi=24N=3,H=10,\sum Q_i=24N=3,H=10,∑Qi=24
每层只能使用8比特量化，量化误差分别为0.018,0.018,0.020.018,0.018,0.020.018,0.018,0.02,总量化误差0.0560.0560.056,*100后输出整数为5。


#### 解答


解题思路
本题本质是一个 带约束的最优化问题，可以转化为 多重选择的动态规划（Multiple Choice Knapsack）。
1. 问题拆解

网络共有 N 层，每层有 H 个权重
每一层只能选择一种量化比特数
Qi∈2,4,8
Q_i \in {2, 4, 8}
Qi∈2,4,8
整体约束：
∑i=1NQi≤Qmax⁡
\sum_{i=1}^N Q_i \le Q_{\max}
∑i=1NQi≤Qmax
目标：最小化所有层的量化误差总和

2. 单层量化误差的计算（预处理）
对于某一层 i，如果选择量化比特 Q：
Wq=int(Wfloat×2Q)W_q = \text{int}(W_{\text{float}} \times 2^Q)
Wq=int(Wfloat×2Q)
量化还原后为：
W^=Wq2Q\hat{W} = \frac{W_q}{2^Q}
W^=2QWq
该层的总误差为：
$$\text{err}[i][Q] = \sum_{j=1}^{H} \left| W_{i,j} - \hat{W}_{i,j} \right|$$由于 Q 只有 {2,4,8} 三种取值，我们可以 预先计算 每一层在三种量化比特下的误差。
3. 动态规划建模
状态定义
设：
$$dp[i][q] = \text{前 } i \text{ 层，总比特数为 } q \text{ 时的最小误差}$$状态转移
对于第 i 层，尝试选择 {2,4,8} 中的一种：
$$dp[i][q] = \min \Big(
dp[i-1][q-2] + err[i][2],;
dp[i-1][q-4] + err[i][4],;
dp[i-1][q-8] + err[i][8]
\Big)$$前提是 q - Q >= 0。
初始化

dp[0] = 0
其他状态初始化为极大值

最终答案
min⁡q≤Qmax⁡dp[N][q]
\min_{q \le Q_{\max}} dp[N][q]
minq≤Qmaxdp[N][q]
4. 输出处理
题目要求输出：
⌊最小误差×100⌋
\lfloor \text{最小误差} \times 100 \rfloor
⌊最小误差×100⌋
复杂度分析
时间复杂度

误差预处理：
O(N×H×3)O(N \times H \times 3)O(N×H×3)
动态规划：
O(N×Qmax⁡×3)O(N \times Q_{\max} \times 3)O(N×Qmax×3)

空间复杂度

使用滚动数组优化后：
O(Qmax⁡)O(Q_{\max})O(Qmax)

代码实现

**Python 代码：**

```python
import math
import sys

def solve():
data = sys.stdin.read().strip().split()
idx = 0

N = int(data[idx]); idx += 1
H = int(data[idx]); idx += 1
Qmax = int(data[idx]); idx += 1

# 读取权重
weights = []
for _ in range(N):
layer = []
for _ in range(H):
layer.append(float(data[idx]))
idx += 1
weights.append(layer)

# 预计算每一层在 2/4/8 bit 下的误差
bits = [2, 4, 8]
err = [[0.0]*3 for _ in range(N)]

for i in range(N):
for k, Q in enumerate(bits):
scale = 2 ** Q
e = 0.0
for w in weights[i]:
wq = int(w * scale)
wr = wq / scale
e += abs(w - wr)
err[i][k] = e

INF = 1e100
dp = [INF] * (Qmax + 1)
dp[0] = 0.0

# 动态规划
for i in range(N):
ndp = [INF] * (Qmax + 1)
for q in range(Qmax + 1):
if dp[q] >= INF:
continue
for k, Q in enumerate(bits):
if q + Q <= Qmax:
ndp[q + Q] = min(ndp[q + Q], dp[q] + err[i][k])
dp = ndp

ans = min(dp)
print(int(ans * 100))

if __name__ == "__main__":
solve()

```

---

## 2025年12月3日-AI方向

<a id="第2题-p4518"></a>

### 第2题-基于剪枝的神经网络模型压缩（P4518）- 中等





在端侧设备部署神经网络模型时，需解决模型参数量过大的问题。本题目要求实现神经网络模型的结构化剪枝，通过移除冗余输入通道降低模型复杂度，同时保持分类性能。给定输入矩阵 X 、模型权重 W 以及剪枝比例 ratioratioratio，对 W 进行结构化剪枝，并使用剪枝后的结果计算模型预测结果。以下是相关计算流程及指标定义说明。

输入矩阵: X (维度: n×dn×dn×d， n 为样本数，d 为输入特征数)

权重矩阵: W (维度:d×cd×cd×c，c 为输出类别数)

计算过程:

线性变换: h=XWh=XWh=XW(维度:n×cn×cn×c)
SoftmaxSoftmaxSoftmax 激活 : y=softmax(h)y= softmax(h)y=softmax(h)(输出概率分布)
预测标签: label=arglabel = arglabel=arg max(y)max(y)max(y)

剪枝目标 : 对权重矩阵 W 按行剪枝(移除整行权重)，剪枝率为 ratioratioratio，剪枝指标为 L1L1L1 范数。
提示:
1、y=softmax(h)y= softmax(h)y=softmax(h) 其中 $y_{i j}=\frac{\exp \left(h_{i j}-\max \left(h_{i}\right)\right)}{\sum_{j} \exp \left(h_{i j}-\max \left(h_{i}\right)\right)}$ ，按行计算概率分布，每个元系减去最大值防止外溢,
2、labeli=argmax(yi)label_i = argmax(y_i)labeli=argmax(yi) 按行计算，输出每行最大值对应的列下标，范围为 [0,c)[0,c)[0,c) 。
1.剪枝定义

按行剪枝 : 移除权重矩阵 W 中不重要的行(对应输入特征)，保留重要行。

物理意义 : 移除对输出影响较小的输入特征，压缩模型输入维度。

剪枝后维度：

权重矩阵 W′W'W′ 维度：(d−k)×c(d-k)×c(d−k)×c ( k 为剪枝行数)。
输入矩阵 X′X'X′ 维度: n×(d−k)n×(d-k)n×(d−k) (需移除对应特征列)。

2.剪枝指标

L1L1L1 范数:对权重矩阵 W 的每一行计算绝对值之和。

第 i 行的 L1L1L1 范数: $\left\|W_{i,:}\right\|_{1}=\sum_{j=1}^{c}\left|W_{i j}\right|$

剪枝规则 : 保留 L1 范数较大的行(重要性高)，移除 L1 范数较小的行(重要性低)。

3.剪枝步骤
1.计算每行 L1L1L1 范数：$row\_norms=[\left\|W_{0,:}\right\|_{1},\left\|W_{i,:}\right\|_{1},...\left\|W_{d-1,:}\right\|_{1}]$
2.确定剪枝行数: k=⌊ratio×d⌋k = \lfloor ratio \times d \rfloork=⌊ratio×d⌋ (需剪掉的行数)
3.选择 L1L1L1 范数最小的 k 行移除，得到剪枝后权重矩阵 W′W'W′ 。
4.调整输入矩阵 X :移除与剪枝行对应的列，得到 X’X’X’ 。
说明:
k=⌊ratio×d⌋k = \lfloor ratio \times d \rfloork=⌊ratio×d⌋
表示 k 是向下取整后的结果。如果 ratio>0ratio>0ratio>0 并且向下取整后 k 为 0 ，则取 k 为 1 (至少剪枝 1 行)
输入描述
输入内容如下：
第一行三个整数：n d c
接下来 n 行，每行 d 个浮点数：X 矩阵
接下来 d 行，每行 c 个浮点数：W 矩阵
最后一行：剪枝率 ratioratioratio
输入范围：
1、1<=n,d,c<=641<= n, d, c <= 641<=n,d,c<=64
2、0<=ratio<=1.00 <= ratio <= 1.00<=ratio<=1.0
输出描述
输出为使用剪枝后矩阵计算得到的预测 labellabellabel 结果。
样例1
输入
4 5 2
1.89 1.88 0.87 0.19 0.62
0.75 0.75 1.45 0.24 0.65
1.26 0.4 0.69 0.54 0.93
0.11 0.61 0.25 1.47 1.96
0.89 2.44
0.97 2.61
2.24 0.72
1.64 0.38
2.29 0.69
0.3

输出
1 0 1 0

说明
样例2
输入
2 2 2
1.0 2.0
3.0 4.0
0.1 0.2
0.3 0.4
0.5

输出
1 1

说明
表示 X 矩阵为:
1.01.01.0 2.02.02.0
3.03.03.0 4.04.04.0
W 矩阵为：
0.10.10.1 0.20.20.2
0.30.30.3 0.40.40.4
剪枝率 ratioratioratio 为 0.50.50.5


#### 解答


解题思路
题目本质是：

先对权重矩阵 (W) 做按行的结构化剪枝（整行去掉），根据每行的 L1 范数决定保留与删除哪些输入特征；
然后用剪枝后的矩阵 (X')、(W') 做线性变换并预测类别。

1. 剪枝部分

对于每一行 (i)（对应第 (i) 个输入特征）计算：
∣Wi,:∣∗1=∑∗j=0c−1∣Wij∣|W_{i,:}|*1 = \sum*{j=0}^{c-1} |W_{ij}|
∣Wi,:∣∗1=∑∗j=0c−1∣Wij∣

剪枝行数：
k=⌊ratio×d⌋k = \lfloor ratio \times d \rfloor
k=⌊ratio×d⌋
如果 (ratio > 0) 且 (k = 0)，则令 (k = 1)，保证至少剪掉 1 行。

将所有行按 L1 范数从小到大排序，取前 (k) 行作为“要删除”的行。

对应删除：

在权重矩阵中删除这些行，得到 (W')（维度 (d−k)×c(d-k) \times c(d−k)×c）。
在输入矩阵中删除对应的列，得到 (X')（维度 n×(d−k)n \times (d-k)n×(d−k)）。

实现上可以这样做：

先计算 row_norms[i]，长度为 d。
建立一个下标数组 idx = [0, 1, ..., d-1]，按照 row_norms[idx[i]] 升序排序。
前 k 个下标加入一个“剪掉集合”，剩下的就是要保留的行（列）。

2. 前向计算与预测
剪枝后，线性部分为：
$$h = X' W' \quad (n \times (d-k)) \cdot ((d-k) \times c) = n \times c$$第 (i) 行第 (j) 列：
$$h_{ij} = \sum_{t=0}^{d-k-1} X'*{i,t} \cdot W'*{t,j}$$逻辑上，题目给出了 softmax：
$$y = \text{softmax}(h), \quad \text{label}*i = \arg\max_j y*{ij}$$但注意：

softmax 是对每个分量做单调递增变换，且不改变各维之间的大小关系。
因此：
$$\arg\max_j \text{softmax}(h_{i,:}) = \arg\max_j h_{ij}$$
也就是说，为了得到预测标签，我们完全可以跳过 softmax 的显式计算，直接对每行的 (h) 做 argmax 即可。
这样实现更简单，也避免不必要的指数运算和数值问题。
步骤总结：

用保留的特征索引，计算每个样本到每个类别的加权和 h[i][j]。
对每行 i 找到最大值所在的列下标 j，即为该样本的预测标签。
将所有 label 以空格分隔输出一行。

代码实现

**Python 代码：**

```python
import sys

def structured_pruning_prediction(n, d, c, X, W, ratio):
# 1. 计算每行 L1 范数
row_norms = []
for i in range(d):
s = 0.0
for j in range(c):
s += abs(W[i][j])
row_norms.append(s)

# 2. 计算剪枝行数 k
k = int(ratio * d)  # 向下取整
if ratio > 0 and k == 0:
k = 1  # 至少剪一行

# 3. 找到 L1 范数最小的 k 行（要剪掉的行）
indices = list(range(d))
indices.sort(key=lambda idx: row_norms[idx])  # 按范数从小到大排序
prune_set = set(indices[:k])  # 要剪掉的下标集合

# 4. 构造保留的特征索引（按原顺序）
keep_indices = [i for i in range(d) if i not in prune_set]
kept_d = len(keep_indices)

# 5. 使用保留特征计算 h = X' W'
# 为了节省空间，这里不显式构造 X'、W'，直接根据 keep_indices 访问原矩阵
labels = []
for i in range(n):
# scores[j] 表示样本 i 对类别 j 的线性得分 h_ij
scores = [0.0] * c
for t in range(kept_d):
feat_idx = keep_indices[t]
x_val = X[i][feat_idx]
if x_val == 0.0:
continue
for j in range(c):
scores[j] += x_val * W[feat_idx][j]

# 6. 对每行 scores 求 argmax，得到预测标签
max_j = 0
max_val = scores[0]
for j in range(1, c):
if scores[j] > max_val:
max_val = scores[j]
max_j = j
labels.append(max_j)

return labels

def main():
data = sys.stdin.read().strip().split()
if not data:
return

ptr = 0
n = int(data[ptr]); ptr += 1
d = int(data[ptr]); ptr += 1
c = int(data[ptr]); ptr += 1

# 读取 X 矩阵
X = []
for _ in range(n):
row = []
for _ in range(d):
row.append(float(data[ptr]))
ptr += 1
X.append(row)

# 读取 W 矩阵
W = []
for _ in range(d):
row = []
for _ in range(c):
row.append(float(data[ptr]))
ptr += 1
W.append(row)

# 读取 ratio
ratio = float(data[ptr]); ptr += 1

labels = structured_pruning_prediction(n, d, c, X, W, ratio)
print(" ".join(str(x) for x in labels))

if __name__ == "__main__":
main()

```

---

<a id="第3题-p4519"></a>

### 第3题-智能客户分群与新用户定位(KMeans均衡分区版)（P4519）- 困难





某大型企业在运营智能营销平台，需对数干位老客户根据购买偏好数据进行自动化人群分群，以便实现高度针对性的商品推荐和精准推广。然而公司规定每个群组容量需尽可能均衡，避免资源失衡。因此，你需要帮平台开发自动化分群与新用户智能归属方案，具体需求如下：
1.采用 KMeansKMeansKMeans 变种聚类，将所有客户分为 K 个群组，且保证每组人数相等或只相差 1 。
2.当人数无法均分时，将多出来的客户依次分配给聚类中心编号更小的组，确保分配唯一。
3.对于新客户数据，利用最终分群中心点，确定其最合适归属的群组。
具体描述
1.客户输入：有 N 位已注册客户，每位客户的购买习惯由 M 个正整数特征描述。
2.分群个数：分为 K 个群组 (2≤K≤min(20,N))(2≤K≤min(20,N))(2≤K≤min(20,N)) 。
3.初始聚类中心：为输入前 K 个客户的数据。
4.聚类算法流程

每一轮分配，依次处理每个客户(客户编号从小到大，即输入顺序)。

对于每个客户，计算其到每个中心的欧几里得距离(如有多个中心距离同为最小，选中心编号更小者)。

客户被分配到距离自己最近且该组未满规定容量的中心;若容量已满，则分配给下一个最近、编号更小的可收组。

群组容量分配规则为：每组分配人数为 ⌊N/K⌋ 或「N/K ⌉\lfloor N / K\rfloor \text { 或「N/K }\rceil⌊N/K⌋ 或「N/K ⌉ ，多出的依次分配给编号较小的中心组；例如 N=11,K=3N=11,K=3N=11,K=3，则各组容量为 [4,4,3] 。

每一轮分配结束后，更新聚类中心为各自组内所有成员特征均值（向下取整）。

若所有客户分配及聚类中心均未发生变化，则算法终止。

5.输出中心排序：最终输出 K 个中心点的特征，按字典序升序排列
6.新用户归属：给定新客户的特征后，判定其归属到与其距离最近的聚类中心(如距离相等，优先字典序最小的中心)，输出其中心编号(排序后中心中的序号，从 1 开始)。
输入描述
输入格式

第 1 行：N M K (空格分隔)

第 2 ~ N+1N+1N+1 行：每行 M 个非负整数，表示一个客户的特征

第 N+2N+2N+2 行：M 个非负整数，为新客户的特征

输入参数范围

2≤K≤min(20,N)2 ≤K ≤ min(20, N)2≤K≤min(20,N)
1≤N≤20001≤N ≤20001≤N≤2000
1≤M≤101≤M≤101≤M≤10
0≤0≤0≤ 特征值 ≤104≤10^4≤104

输出描述
输出格式

前 K 行：每行 M 个整数，表示聚类中心的特征，按字典序升序排列
第 K+1K+1K+1 行：新客户归属的中心编号(按字典序排序后的位置，1 为首位)

样例1
输入
4 1 2
0
10
0
10
5

输出
0
10
1

说明

聚类目标人数为 [1,1]，前两组各 1 人。
按题算法唯一执行，中心结果唯一。
新客户同时与多个中心点等距离，字典序优先，分到第一组。

样例2
输入
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

输出
11 10
51 50
99 99
2

说明

聚类目标人数为 [3,3,2] ，前两组各 3 人，最后 1 组 2 人。

按题算法唯一执行，中心结果唯一。

新客户 [45[45[45 46]46]46] 归属中间分组，输出 2 。

提示
1.欧几里得距离(Euclidean distance)

对于两个 M 维特征的客户 A 和 B(A=[a1,a2,…,aM]，B=[b1,b2,…,bM])B(A=[a_1,a_2,…,a_M ]，B=[b_1,b_2,…,b_M ])B(A=[a1,a2,…,aM]，B=[b1,b2,…,bM])，它们之间的欧几里得距离定义为：$\sqrt{\left(a_{1}-b_{1}\right)^{2}+\left(a_{2}-b_{2}\right)^{2}+\cdots+\left(a_{M}-b_{M}\right)^{2}}$

建议先计算平方和，再进行平方根(或用于距离大小比较时可直接用平方和，省略开方，顺序不会影响)。

2.均衡分组(Balanced partitioning)

若 N 个客户分到 K 组，原则上每组应有 q=⌊N/K⌋q=\lfloor N / K\rfloorq=⌊N/K⌋人。若 N 不能被 K 整除，则有 (N(N(N mod k)k)k) 个组会多一人。多出的名额分配给编号较小的分组。

3.字典序(Lexicographical order)

多维特征排序时，先比较第 1 维，若相同比第 2 维，依此类推。例如：[2,3,4]<[2,4,1]<[5,0,0][2,3,4]< [2,4,1]< [5,0,0][2,3,4]<[2,4,1]<[5,0,0] 。

4.唯一性规则说明

每个客户依次(输入顺序)分配到当前距离自己最近、且该组人数没到上限的中心，如有多个距离同为最小，选择中心编号最小(即输入时编号更小)的中心。

分组人数严格按照上面的第二点分配，保证每组人数最多相差 1 ，且人数多的始终是中心编号更小的组。

5.新客户KNN归属

将新客户特征与所有最终中心比距离，归属最近中心的分组。

若距离有多个中心同为最小，分到字典序最小的中心


#### 解答


解题思路
1. 整体算法
本题本质是一个带容量约束的 KMeans 聚类 + 新样本最近中心归属问题：

已有 N 个客户，每个是 M 维非负整数特征；

需要分成 K 个簇，并且每个簇人数尽量均衡（容量上限提前固定）；

聚类过程类似 KMeans：

用前 K 个样本作为初始中心；
反复执行「按当前中心分配 → 重新计算中心」直到收敛；

聚类完成后：

将 K 个中心按字典序排序并输出；
对新用户，用“最近中心 + 字典序优先”来决定归属中心，并输出其在排序后的位置。

这里的关键在于：分配阶段的容量约束和选择规则与普通 KMeans 不同。
2. 容量（目标人数）计算
先计算每个簇最大容量（题目保证最终都是满的）：

设

q = N // K（每组至少 q 人）
r = N % K（前 r 个组多一个人）

则组 0..K-1 的目标容量为：

若 i < r，容量为 q + 1
否则容量为 q

例如：N = 11, K = 3 → q = 3, r = 2 → 容量 [4, 4, 3]。
这样就保证：

每组人数只差 0 或 1；
人数更多的一定是编号更小的组（前面的组先拿到额外名额）。

3. 聚类迭代过程（变种 KMeans）

初始化中心

直接把前 K 个客户的特征当作初始中心 centers[0..K-1]。

一次分配轮次（Assignment Step）
依次处理每个客户 i = 0..N-1：

计算它到每个中心 k 的欧氏距离的平方（不必开根号，比较大小即可）：
$$dist^2 = \sum_{d=0}^{M-1} (x_{i,d} - center_{k,d})^2$$

对所有簇 (k) 按 (距离, k) 升序排序（先比距离，再比簇编号）；

从排序后的簇中，找到第一个未达容量上限的簇，把客户 i 分给它：

如果最优簇已满，则尝试下一个距离最近的簇；
总会找到一个，因为整体容量之和就是 N。

这样就满足了：

优先选择距离最近的中心；
距离相等时，优先中心编号更小；
容量被严格限制在预先计算好的上限。

更新中心（Update Step）
分配完成后，对每个簇 k：

累加该簇内所有客户在各维度的特征和；
再对每维特征做整数平均（向下取整）：$$center_{k,d} = \left\lfloor \frac{\sum x_{i,d}}{\text{簇内人数}} \right\rfloor$$

终止条件（收敛）

如果本轮分配得到的 assign[] 与上一轮完全一样，
且本轮计算出的 centers[][] 也与上一轮完全一样，
则认为算法收敛，终止循环。
否则继续下一轮。

注：状态空间有限（中心取值有限、分配方式有限），最终一定会收敛。

4. 输出中心 + 新用户归属

中心排序
收敛后得到最终 K 个中心。
按字典序升序排序：

先比第 1 维，不同则小者在前；
相同再比第 2 维，以此类推。

输出排序后的 K 行中心向量。

新用户归属（KNN 到中心）
用排序后的中心数组 sorted_centers 对新用户进行一次简单的「最近中心判断」：

对每个中心 c，计算新用户与其的欧氏距离平方；

选择距离最小的中心；

如果有多个中心距离相等，则优先选择字典序更小的中心。

因为我们已经按照字典序排序，所以在遍历中“先出现”的中心自然就是字典序更小的；

输出该中心在排序数组中的位置（1 开始编号）。

代码实现

**Python 代码：**

```python
import sys
from math import inf

# 计算两点间欧氏距离的平方
def dist2(a, b):
s = 0
for x, y in zip(a, b):
diff = x - y
s += diff * diff
return s

def balanced_kmeans(customers, K):
N = len(customers)
M = len(customers[0])

# 计算每个簇的目标容量
base = N // K
rem = N % K
capacity = [base + (1 if i < rem else 0) for i in range(K)]

# 初始中心：前 K 个客户
centers = [customers[i][:] for i in range(K)]

# 初始分配，全部设为 -1
assign = [-1] * N

while True:
# 新一轮分配
new_assign = [-1] * N
sizes = [0] * K

# 按客户编号从小到大分配
for i in range(N):
point = customers[i]

# 计算到每个中心的距离
dist_list = []
for k in range(K):
d2 = dist2(point, centers[k])
dist_list.append((d2, k))

# 按 (距离, 中心编号) 排序
dist_list.sort()

# 找到第一个未满容量的中心
for _, k in dist_list:
if sizes[k] < capacity[k]:
new_assign[i] = k
sizes[k] += 1
break

# 根据新分配结果更新中心
new_centers = [[0] * M for _ in range(K)]
counts = [0] * K
for i in range(N):
c = new_assign[i]
counts[c] += 1
for d in range(M):
new_centers[c][d] += customers[i][d]

for k in range(K):
# 每个簇一定有容量，所以 counts[k] > 0
for d in range(M):
new_centers[k][d] //= counts[k]

# 判断是否收敛（分配和中心都不变）
if new_assign == assign and new_centers == centers:
centers = new_centers
assign = new_assign
break
else:
centers = new_centers
assign = new_assign

return centers

def main():
data = sys.stdin.read().strip().split()
if not data:
return
it = iter(data)
N = int(next(it))
M = int(next(it))
K = int(next(it))

customers = []
for _ in range(N):
point = [int(next(it)) for _ in range(M)]
customers.append(point)

new_customer = [int(next(it)) for _ in range(M)]

# 执行均衡 KMeans
centers = balanced_kmeans(customers, K)

# 按字典序排序中心
centers.sort()

# 输出中心
out_lines = []
for c in centers:
out_lines.append(" ".join(str(x) for x in c))

# 新用户归属：最近中心，距离相等时字典序更小（排序后自然满足）
best_idx = 0
best_dist = dist2(new_customer, centers[0])
for i in range(1, K):
d2 = dist2(new_customer, centers[i])
if d2 < best_dist:
best_dist = d2
best_idx = i
# 若距离相等，由于中心已按字典序排序，保留原来的较小下标即可

out_lines.append(str(best_idx + 1))

sys.stdout.write("\n".join(out_lines))

if __name__ == "__main__":
main()

```

---

## 2025年11月20日-留学生AI方向

<a id="第2题-p4481"></a>

### 第2题-Vision Transformer中的Patch Embdding层实现（P4481）- 简单





VisionVisionVision Transformer(ViT)Transformer(ViT)Transformer(ViT) 是视觉领域应用非常广泛的基础网络结构，经典的 ViT 结构如图所示，
其包含了 PatchPatchPatch＆PositionPositionPosition embedding、Transformerembedding、Transformerembedding、Transformer EncoderEncoderEncoder 等多个关键模块组成。这几个模块中，将图像分割为固定大小的 patchpatchpatch 并进行线性嵌入是一个关键步骤，也即 PatchPatchPatch EmbeddingEmbeddingEmbedding 层，其主要实现步骤为：
StepStepStep 1：将输入图像分割为多个非重叠的 patchpatchpatch ，也即将图片切分为 N∗NN*NN∗N 个 patchpatchpatch ，如 3∗33*33∗3 个 2D 图像块；
StepStepStep 2：将每个 patchpatchpatch 展平为向量，也即将每个切分后的 2D PatchPatchPatch 展平为 1D 向量;
StepStepStep 3：对展平的 patchpatchpatch 进行线性变换(嵌入)，也即对每个展平后的 1D 向量做一个线性变换，使用一个可学习的权重矩阵 E 和 偏置向量 B 进行线性变换，公式为：Z=X∗E+bZ=X*E+bZ=X∗E+b
StepStepStep 4：添加可学习的位置编码；
请根据以上提示步骤，实现 PatchPatchPatch EmbeddingEmbeddingEmbedding 层。
特别注意：本实现过程中，无法使用深度学习框架，如 pytorch、tensorflowpytorch、tensorflowpytorch、tensorflow 等
输入描述
输入参数包括：imp_size、patch_size、channel、embedding_dimimp\_size、patch\_size、channel、embedding\_dimimp_size、patch_size、channel、embedding_dim，分别表示：
图像尺寸（图像长、宽默认相等）img_sizeimg\_sizeimg_size ；
patchpatchpatch 大小 patch_sizepatch\_sizepatch_size ；
图像通道数 channelschannelschannels ；
嵌入维度 embedding_dimembedding\_dimembedding_dim
输出描述
输出 patch_embeddingpatch\_embeddingpatch_embedding 后的维度信息 embedding_shapeembedding\_shapeembedding_shape，其中需要包含 cis tokentokentoken，具体可见样例。
样例1
输入
448 32 3 384

输出
197 384

说明
输入：448 32 3 384
分别表示：
图像尺寸（图像长、宽默认相等）img_size=448img\_size=448img_size=448 ；
patchpatchpatch 大小 patch_size=32patch\_size=32patch_size=32 ；
图像通道数 channels=3channels=3channels=3 ；
嵌入维度 embedding_dim=384embedding\_dim=384embedding_dim=384
输出：197 384
分别表示：
经过 patch_embeddingpatch\_embeddingpatch_embedding 层后得到的 embedding_shapeembedding\_shapeembedding_shape ，其中第一维 197 表示 patchpatchpatch token+cistoken+cistoken+cis tokentokentoken ，第二维表示 patch_embeddingpatch\_embeddingpatch_embedding 后的 enbeddingenbeddingenbedding 维度
样例2
输入
224 16 3 768

输出
197 768

说明
输入：224 16 3 768
分别表示：
图像尺寸（图像长、宽默认相等）img_size=224img\_size=224img_size=224 ；
patchpatchpatch 大小 patch_size=16patch\_size=16patch_size=16 ；
图像通道数 channels=3channels=3channels=3 ；
嵌入维度 embedding_dim=768embedding\_dim=768embedding_dim=768
输出：197 768
分别表示：
经过 patch_embeddingpatch\_embeddingpatch_embedding 层后得到的 embedding_shapeembedding\_shapeembedding_shape ，其中第一维 197 表示 patchpatchpatch token+cistoken+cistoken+cis tokentokentoken ，第二维表示 patch_embeddingpatch\_embeddingpatch_embedding 后的 enbeddingenbeddingenbedding 维度


#### 解答


解题思路
Patch Embedding 的本质是一个分块 + 展开 + 线性变换的过程，可以理解为对图像做一次“卷积核为 patch，步长为 patch_size 的卷积 + reshape”，再加上一个 cls token。这里我们只需要计算输出向量序列的维度，而不是具体做矩阵运算。
设输入参数为：

图像尺寸：img_size（高 = 宽）
patch 大小：patch_size
通道数：channel
嵌入维度：embedding_dim

1. 计算 patch 的个数
图像被均匀切成大小为 patch_size × patch_size 的不重叠 patch：

每一维上的 patch 个数：
N=img_sizepatch_sizeN = \frac{\text{img\_size}}{\text{patch\_size}}
N=patch_sizeimg_size

总 patch 数目：
$$\text{num\_patches} = N \times N = \left(\frac{\text{img\_size}}{\text{patch\_size}}\right)^2$$

题目默认输入是合法的，所以可以认为 img_size 能整除 patch_size。
2. 展开并线性变换
每个 patch 的原始维度为：
$$\text{patch\_dim} = \text{patch\_size} \times \text{patch\_size} \times \text{channel}$$线性嵌入用一个权重矩阵 E 和偏置 b：

E 的形状：patch_dim×embedding_dim\text{patch\_dim} \times \text{embedding\_dim}patch_dim×embedding_dim
b 的形状：embedding_dim\text{embedding\_dim}embedding_dim

对每个 patch 展开后的向量 X 做线性变换：
Z=X×E+bZ = X \times E + b
Z=X×E+b
因此每个 patch 最终变成一个长度为 embedding_dim 的向量。
所有 patch 经过线性变换后得到：
(num_patches, embedding_dim)(\text{num\_patches},\ \text{embedding\_dim})
(num_patches, embedding_dim)
3. 添加 CLS Token
ViT 会额外添加一个可学习的 cls token，其维度和单个 patch 的嵌入相同，为 (embedding_dim,)。
拼接到序列前面之后，序列长度变为：
num_tokens=num_patches+1\text{num\_tokens} = \text{num\_patches} + 1
num_tokens=num_patches+1
因此最终的 patch embedding 输出维度（不含 batch 维）为：
$$\text{embedding\_shape} = (\text{num\_patches} + 1,\ \text{embedding\_dim})$$4. 套用样例
样例输入：448 32 3 384

每边 patch 数：N=448/32=14N = 448 / 32 = 14N=448/32=14
总 patch 数：num_patches=142=196\text{num\_patches} = 14^2 = 196num_patches=142=196
加 cls token：num_tokens=196+1=197\text{num\_tokens} = 196 + 1 = 197num_tokens=196+1=197
每个 token 维度：384

所以输出为：
embedding_shape=(197, 384)\text{embedding\_shape} = (197,\ 384)
embedding_shape=(197, 384)
即：197 384
代码实现

**Python 代码：**

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

---

<a id="第3题-p4482"></a>

### 第3题-带Padding的卷积计算（P4482）- 中等





卷积计算是人工智应用的常见计算操作，在深度学习中，卷积一般使用无核翻转的方式，通过如下公式计算:
$(S \cdot K)(i, j)-\sum_{m} \sum_{n} S(i+m, j+n) \cdot K(m, n)$
其中:
S 是输入图像(或二维信号)
K 是卷积核(或滤波器)
(i,j)(i,j)(i,j) 是输出图像的坐标
(m,n)(m,n)(m,n) 是卷积核的坐标
所谓 PaddingPaddingPadding 是根据卷积的尺寸，在输入图像外围填充若干圈 0 ，确保卷积计算后的辅出数据尺寸与原始输入一致。
如：

输入描述
第一行 为输入卷积尺寸 m∗mm*mm∗m 和图像尺寸 n∗nn*nn∗n，m 为大于 1 的奇数，n 为大于 1 的整数
然后是卷积矩阵数据，共 m 行，每行为卷积核的某一行，值的取值范围 [−10,10][-10,10][−10,10] 整数
最后是图像数据，共 n 行，每行为图像数据的某一行，值的取值范围 [0,255] 整数

如：

3 3
1 0 -1
1 0 -1
1 0 -1
1 2 3
4 5 6
7 8 9

输出描述
输出为带 paddingpaddingpadding 卷积操作后结果数据，大小为输入图像的尺寸 n∗nn*nn∗n ，如：

-7 -4 7
-15 -6 15
-13 -4 13

样例1
输入
3 5
-5 4 0
0 -3 -2
3 2 0
231 112 85 120 114
154 237 168 55 35
203 204 160 70 7
194 32 36 99 181
64 185 251 30 115

输出
-609 430 552 26 -107
394 -737 98 440 -25
-13 -108 -965 -538 503
294 195 371 -366 -543
214 -1899 -829 -106 -119

样例2
输入
3 5
1 0 -1
1 0 -1
1 0 -1
3 2 5 4 8
1 8 2 4 1
3 7 5 2 9
8 2 1 4 3
1 5 7 2 8

输出
-10 -3 2 -2 8
-17 -5 7 -6 10
-17 4 7 -5 10
-14 -1 6 -7 8
-7 1 1 -3 6


#### 解答


解题思路
本题要求实现二维卷积 + 零填充（padding），并且题面明确说明是“无核翻转”的卷积，本质上就是二维相关运算（cross-correlation）：
对每个输出像素 (i,j)(i,j)(i,j)，让卷积核以 (i,j)(i,j)(i,j) 为中心覆盖在原图上，做逐元素乘加；超出边界的位置视为 0（即在图像外层补零）。
设：

卷积核尺寸：m×mm \times mm×m，且 m 为奇数（方便居中）
图像尺寸：n×nn \times nn×n
核心公式（按题意“无核翻转”）：

$$(S * K)(i, j) = \sum_{u=0}^{m-1} \sum_{v=0}^{m-1} S'(i + u - t,\, j + v - t)\cdot K(u, v)$$其中：

t=m−12t = \frac{m-1}{2}t=2m−1 为卷积核“半径”，即中心偏移
S′S'S′ 为对原图 S 进行零填充后的概念扩展：索引越界时视为 0

在实现时有两种等价写法：

显式构造一个 (n+2t)×(n+2t)(n+2t) \times (n+2t)(n+2t)×(n+2t) 的新数组，把原图放在中间，其余补 0，再做普通卷积。
不新建数组，在计算时判断坐标是否越界，越界就跳过（视为 0）。

为了节省空间，本题更适合使用方法 2：
对每个输出位置 (i,j)(i,j)(i,j)：

初始化 sum = 0

枚举卷积核坐标 (u, v)，范围 0 .. m-1

将其映射到图像坐标：

x = i + u - t
y = j + v - t

若 0 <= x < n 且 0 <= y < n，则 sum += image[x][y] * kernel[u][v]

最后 ans[i][j] = sum

按行输出结果即可。
代码实现

**Python 代码：**

```python
import sys

# 卷积函数，输入核 kernel 和图像 img，返回卷积结果
def convolve2d(kernel, img):
m = len(kernel)          # 卷积核大小 m
n = len(img)             # 图像大小 n
t = m // 2               # 核的半径 (padding 策略使用)

# 初始化结果矩阵
res = [[0] * n for _ in range(n)]

# 遍历每一个输出像素 (i, j)
for i in range(n):
for j in range(n):
s = 0
# 遍历卷积核 (u, v)
for u in range(m):
for v in range(m):
x = i + u - t   # 映射到图像中的行坐标
y = j + v - t   # 映射到图像中的列坐标
# 判断是否在原图范围内，越界视为 0
if 0 <= x < n and 0 <= y < n:
s += img[x][y] * kernel[u][v]
res[i][j] = s
return res

def main():
data = list(map(int, sys.stdin.read().split()))
# 读入 m, n
m, n = data[0], data[1]
idx = 2

# 读入卷积核 m 行，每行 m 个数
kernel = []
for _ in range(m):
row = data[idx:idx + m]
idx += m
kernel.append(row)

# 读入图像 n 行，每行 n 个数
img = []
for _ in range(n):
row = data[idx:idx + n]
idx += n
img.append(row)

# 进行卷积计算
res = convolve2d(kernel, img)

# 按题目要求输出，行内用空格分隔
out_lines = []
for i in range(n):
out_lines.append(" ".join(str(x) for x in res[i]))
sys.stdout.write("\n".join(out_lines))

if __name__ == "__main__":
main()

```

---

## 2025年11月19日-AI方向

<a id="第2题-p4475"></a>

### 第2题-终端款型聚类识别（P4475）- 中等





某部门需要对终端的漫游业务体验进行保障，不同的终端对于网络的配置要求不同。现在需要通过终端的网络流量等特征，识别该终端的型号是什么。
通过包间隔时长、连接持续时长、漫游前信号强度及漫游后信号强度 4 个特征，对终端的型号进行聚类。已知终端型号类别为 K 类，采用 KmeansKmeansKmeans 算法进行聚类，识别终端类型，并输出各类型终端数量。
KmeansKmeansKmeans 算法说明:
初始化: k 个初始质心
分配:将每个数据点分配到距离最近的质心，形成 k 个簇。其中距离需要根据数据类型选择上文给定的度量方式
更新:用簇内所有点的均值，重新计算每个簇的质心
迭代:重复步骤 2 和 3 ，直到质心不再发生变化(质心移动值小于 10−810^{-8}10−8 )或达到最大迭代次数
本题说明:
1、给定数据集中，默认 K 类终端都存在，不存在某款型终端个数为 0 的场景;
2、为消除不同特征权重问题，给出数据均已做好归一化处理，并保留两位小数;
3、为消除随机性，初始 k 个质心统一采用给定数据集前 k 个点;
4、距离函数定义为: $d_{x, y}=\sqrt{\sum_{k=1}^{4}\left(x_{k}-y_{k}\right)^{2}}$
输入描述
第 1 行: k m n : k 代表终端款型聚类个数，m 代表终端数量，n 代表迭代次数;
第 2 行 ~ 第 m+1m+1m+1 行:每一行 4 列，分别代表某个终端的包间隔时长、连接持续时长、漫游前信号强度及漫游后信号强度 4 个变量
输出描述
输出 k 款终端数量，从小到大排序。
样例1
输入
3 20 1000
0.11 0.79 0.68 0.97
1.0 0.8 0.13 0.33
0.27 0.02 0.5 0.46
0.83 0.29 0.23 0.75
0.97 0.08 0.84 0.55
0.29 0.71 0.17 0.83
0.03 0.6 0.88 0.28
0.24 0.26 0.82 0.03
0.96 0.12 0.82 0.36
0.13 0.12 0.86 0.44
0.23 0.7 0.35 0.06
0.42 0.49 0.67 0.84
0.8 0.49 0.47 0.7
0.68 0.03 0.11 0.07
0.77 0.19 0.95 0.44
0.25 0.12 0.98 0.04
0.7 0.11 0.53 0.3
0.73 0.67 0.46 0.96
0.11 0.31 0.91 0.57
0.43 0.61 0.13 0.1

输出
4 6 10

说明
输入: 20 个终端，其中包含 3 种款式，用 KmeansKmeansKmeans 算法最高选代 1000 次计算每款终端个数
输出: 3 款终端数量从小到大排序为 4 6 10
样例2
输入
4 32 800
0.73 0.96 0.2 0.53
0.01 0.19 0.42 0.46
0.27 0.24 0.87 0.8
0.97 0.77 0.42 0.04
0.41 0.69 0.96 0.56
0.27 0.4 0.56 0.56
0.28 0.04 0.74 0.82
0.17 0.2 0.95 0.1
0.2 0.1 0.14 0.93
0.86 0.59 0.42 0.52
0.35 0.77 0.37 0.08
0.52 0.48 0.16 0.56
0.59 0.97 0.21 0.05
0.67 0.94 0.28 0.08
0.09 0.65 0.55 1.0
0.77 0.14 0.35 0.01
0.02 0.18 0.72 0.26
0.71 0.78 0.86 0.11
0.54 0.02 0.75 0.2
0.15 0.76 0.59 0.23
0.71 0.66 0.43 0.32
0.17 0.57 0.53 0.42
0.04 0.34 0.66 0.28
0.79 0.14 0.11 0.6
0.04 0.48 0.05 0.04
0.62 0.43 0.28 0.6
0.47 0.13 0.35 0.17
0.9 0.82 0.97 0.71
0.99 0.53 0.24 0.56
0.83 0.44 0.7 0.4
0.71 0.45 0.64 0.53
0.6 0.54 0.86 0.11

输出
6 8 9 9

说明
输入: 32 个终端，其中包含 4 种款式，用 KmeansKmeansKmeans 算法最高迭代 800 次计算每款终端个数
输出: 4 款终端数量从小到大排序为 6899


#### 解答


解题思路
本题本质是一个典型的 KMeans 聚类 问题：给定每个终端的 4 维特征（包间隔时长、连接持续时长、漫游前信号强度、漫游后信号强度），将这些终端划分成 K 个簇，每个簇对应一种终端款型，最后输出每个簇中终端数量（从小到大排序）。
1. KMeans 算法回顾与本题设定
KMeans 的标准流程为：

初始化质心

从数据集中选择 K 个点作为初始质心。
本题明确要求：初始化质心使用数据集中前 k 个点。

分配（Assignment）

对每个样本点，计算其到所有质心的距离，将该点分配给最近的质心所属的簇。

距离定义为 4 维欧氏距离：
d(x,y)=∑i=14(xi−yi)2d(x,y)=\sqrt{\sum_{i=1}^4 (x_i-y_i)^2}
d(x,y)=i=1∑4(xi−yi)2

实现时可直接用 平方距离（去掉根号），比较大小结果相同，少一次 sqrt，更高效：
d2(x,y)=∑i=14(xi−yi)2d^2(x,y)=\sum_{i=1}^4 (x_i-y_i)^2
d2(x,y)=i=1∑4(xi−yi)2

更新质心（Update）

对每个簇，计算簇内所有样本的 4 维特征均值，将其作为新的质心：
μj=1∣Cj∣∑x∈Cjx\mu_j = \frac{1}{|C_j|}\sum_{x \in C_j} x
μj=∣Cj∣1x∈Cj∑x

迭代与收敛条件

重复“分配 + 更新”步骤，直到：

质心移动距离足够小（本题要求：质心移动值 < 10−810^{-8}10−8），或
达到给定的最大迭代次数 n。

具体实现中，我们可以对每个质心计算新旧质心的平方距离，取所有质心中 最大移动量：
$$\text{move}_j = \sum_{i=1}^4(\mu_{j,i}^{\text{new}} - \mu_{j,i}^{\text{old}})^2$$若所有质心的最大移动量 < 10−810^{-8}10−8，则认为收敛。

空簇问题的处理

理论上，KMeans 可能会出现某个簇为空的情况。

本题说明中给出：数据集中 默认 K 类终端都存在。
但在迭代过程中仍有极小概率产生空簇，因此实现时可以做一个稳健处理：

若某个簇在当前分配中没有任何点（数量为 0），则 保持该簇的质心不变，避免除零错误。

最终输出

算法停止后，我们已经有每个终端的簇归属，统计每个簇内终端数量，得到长度为 K 的数组。
按题意要求，将这 K 个数量 从小到大排序，用空格分隔输出。

2. 实现要点

读入数据

第 1 行：k m n

k：簇的个数（终端款型数）
m：样本个数（终端数）
n：最大迭代次数

接下来 m 行，每行 4 个浮点数，表示该终端的 4 维特征（已归一化，保留两位小数）。

数据结构设计

points[m][4]：保存所有终端的 4 维特征。
centroids[k][4]：当前 k 个质心。
assign[m]：每个终端所属的簇编号（0 ~ k-1）。
clusterSize[k]：每个簇中点的个数。
sum[k][4]：每个簇中各维度特征之和，用于计算新质心。

初始化

centroids[i] = points[i] 对于 i ∈ [0, k-1]。

每次迭代流程

将 clusterSize 和 sum 清零。

遍历每个点：

依次计算该点到每个质心的 平方距离，找到最小值对应的簇 j。
assign[i] = j，clusterSize[j]++，并将当前点特征累加到 sum[j] 中。

使用 sum 和 clusterSize 更新每个质心：

若 clusterSize[j] > 0：

新质心 = sum[j][d] / clusterSize[j]。

若 clusterSize[j] == 0：

保持原质心不变。

同时在更新时，计算新旧质心之间的最大移动距离，用于判断收敛。

结束后统计结果

上一轮分配的 clusterSize 即为最终每个簇的点数。
将 clusterSize 拷贝到数组，排序，然后输出。

代码实现

**Python 代码：**

```python
import sys
import math

# 计算两点之间的平方欧氏距离（4 维）
def dist2(a, b):
s = 0.0
for i in range(4):
diff = a[i] - b[i]
s += diff * diff
return s

def kmeans(points, k, max_iter):
m = len(points)
# 初始化质心：前 k 个点
centroids = [points[i][:] for i in range(k)]
assign = [0] * m
cluster_size = [0] * k

for _ in range(max_iter):
# 分配步骤
for j in range(k):
cluster_size[j] = 0
sums = [[0.0] * 4 for _ in range(k)]

for i in range(m):
# 找到最近质心
best_idx = 0
best_dist = dist2(points[i], centroids[0])
for j in range(1, k):
d = dist2(points[i], centroids[j])
if d < best_dist:
best_dist = d
best_idx = j
assign[i] = best_idx
cluster_size[best_idx] += 1
# 累加簇内点特征
for t in range(4):
sums[best_idx][t] += points[i][t]

# 更新质心
max_move = 0.0
for j in range(k):
if cluster_size[j] > 0:
new_centroid = [sums[j][t] / cluster_size[j] for t in range(4)]
else:
# 若该簇为空，保持旧质心不变
new_centroid = centroids[j][:]

# 计算质心移动距离（平方）
move = dist2(centroids[j], new_centroid)
if move > max_move:
max_move = move
centroids[j] = new_centroid

# 收敛判断：最大质心移动小于 1e-8
if max_move < 1e-8:
break

# cluster_size 即为各簇最终数量
return cluster_size

def main():
data = sys.stdin.read().strip().split()
if not data:
return
k = int(data[0])
m = int(data[1])
max_iter = int(data[2])

points = []
idx = 3
for _ in range(m):
# 依次读入 4 个浮点数
row = list(map(float, data[idx:idx+4]))
idx += 4
points.append(row)

sizes = kmeans(points, k, max_iter)
sizes.sort()
print(" ".join(str(x) for x in sizes))

if __name__ == "__main__":
main()

```

---

<a id="第3题-p4476"></a>

### 第3题-Prompt上下文信息精简:找出二叉树中的最大值子树（P4476）- 困难





描述: PromptPromptPrompt 应用面临的一个首要问题就是 TokenTokenToken 的长度和精确度问题，如何精简 PromptPromptPrompt 的 tokentokentoken 长度一直是大模型应用中的难题。假设 PromptPromptPrompt 的 tokentokentoken 序列是一颗二叉树，给定这样一颗二叉树，该二叉树的每个节点都有一个值，可以是正负值，也可以是 0 ，请返回该二又树的最大值子树。每颗子树的值为该子树所有节点值的和。
注意:
输入和输出数据的格式要求:(1)二叉树是完全二叉树;(2)二叉树节点数据是通过宽度优先搜索遍历获取;(3)遍历出的二叉树节点数据是以一维数组的形式存储。(4)如果一颗二叉树的左节点不存在，就以 nullnullnull 补齐。
举例:如果节点 A 和 B 是兄弟节点，它们两个的父节点是 C ，A 无子节点，B 有子节点 D 和 E ，那么这棵树的数组为 tree=[C,A,B,null,null,D,E]tree=[C,A,B,null,null,D,E]tree=[C,A,B,null,null,D,E];如果 B 只有左子节点 D ，则 tree=[C,A,B,null,null,D]tree =[C,A,B,null,null,D]tree=[C,A,B,null,null,D] ; 如果 B 只有右子节点 E ,则 tree=[C,A,B,null,null,null,E]tree =[C,A,B,null,null,null,E]tree=[C,A,B,null,null,null,E];
示例1

输入:[3,2,5]
输出: [3,2,5]
示例2

输入: [−5,−1,3,null,null,4,7][-5,-1,3,null,null,4,7][−5,−1,3,null,null,4,7]
输出: [3,4,7]
示例3

输入: [5,−1,3,null,null,4,7][5,-1,3,null,null,4,7][5,−1,3,null,null,4,7]
输出: [5,null,3,null,null,4,7]
输入描述
输入:二叉树是一颗完全二叉树，节点数据是通过宽度优先搜索遍历的，以一维数组结构表示，nullnullnull 代表为空的叶子节点。
以示例 2 为例，输入为 [−5,−1,3,null,null,4,7][-5,-1,3,null,null,4,7][−5,−1,3,null,null,4,7]，−1-1−1 节点虽然是叶子节点，但在完全二叉树中需要明确它的两个子节点，这两个子节点为 nullnullnull 。
输出描述
输出:输出最大值子树，也以宽度优先搜索完全二叉树后的数组结构表示的方式作为输出。
以示例 2 为例，由于 [−5,−1,3][-5,-1,3][−5,−1,3] 该子树为负值，不应当与子树 [3,4,7] 合一起，所以 [3,4,7] 是最大子树。
以示例 3 为例，最大值子树是 [5,null,3,null,null,4,7]，根节点 5 的所有左子树节点用 nullnullnull 补
齐。
样例1
输入
[-5,-1,3,null,null,4,7]

输出
[3,4,7]

说明
最大子树 max−sub−treemax-sub-treemax−sub−tree 是 [3,4,7]
样例2
输入
[-1,null,1,null,null,-1,-1,null,null,null,null,2,1,-3,-1,null,null,null,null,null,null,null,null,2,1,3,8]

输出
[1,-1,null,2,1,null,null,2,1,3,8]

说明
最大子树 max−sub−treemax-sub-treemax−sub−tree 是 [1,−1,null,2,1,null,null,2,1,3,8][1,-1,null,2,1,null,null,2,1,3,8][1,−1,null,2,1,null,null,2,1,3,8]


#### 解答


解题思路
题目里“子树”的定义：

可以在一棵树中选择任意一个结点作为“根”；

对于这个根的左右子树，我们可以选择保留或裁掉：

如果某个子树对总和的贡献 ≤ 0，就可以整个裁掉（不保留这一支）；
只保留“贡献为正”的子树分支；

目标：找到和最大的“裁剪后子树”，并输出这棵裁剪后的子树（仍然用完全二叉树层序数组表示，被裁掉的分支用 null，最后去掉末尾多余的 null）。

这本质上就是“在树上找最大权连通子图（允许剪枝）”，非常经典的写法是 树形 DP + 可选子树：
一、数组表示的二叉树结构
与之前相同，输入是一棵“完全二叉树”的一维数组：

根：下标 0
左子：2 * i + 1
右子：2 * i + 2
null 表示该位置没有结点（空）

我们仍然先解析出：

vals[i]：结点值（对于 null，值随便填，反正不用）
valid[i]：该位置是否是一个真实结点（非 null）

二、核心 DP：允许裁剪子树的最大和
对每个真实结点 i，定义：

dp[i] = 以 i 为根，在允许删掉任意“贡献 ≤ 0 的子树”的前提下，
能得到的 最大子树和

状态转移（自底向上，从 n-1 到 0）：

如果 i 不是有效结点（valid[i] == false），

我们不以它为根建树，dp[i] 记为 0 即可（不会被真正使用）。

如果 i 是有效结点：

左子下标：l = 2*i + 1

右子下标：r = 2*i + 2

左子贡献：

如果 l 在范围内且 valid[l] == true，则可用 dp[l]
否则为 0

右子贡献同理

当前根的最佳和：
$$dp[i] = vals[i]
+ \max(0, \text{leftDp})
+ \max(0, \text{rightDp})$$

这里的 max(0, dp[child]) 正是“如果这个子树贡献为正，就保留；否则就整个裁掉”的含义。

计算完 dp[i] 后，我们用它去更新全局最大值：

维护：

bestSum：当前所有结点中最大的 dp[i]
bestRoot：达到 bestSum 的下标 i

最终：

bestRoot 就是那棵“最大值裁剪子树”的根
和为 bestSum

注意：如果整棵树都是负数，
这个 DP 仍然会选出某个单个结点（值最大的那个）作为答案。

三、如何还原“被裁剪后的子树”结构
有了 bestRoot 和整棵树的 dp[]，要构造输出数组：

从 bestRoot 开始做 BFS（广度优先），队列中保存：

(原数组下标 originalIndex, 新树下标 newIndex)
新树也用完全二叉树规则：左子 2*newIndex+1，右子 2*newIndex+2

对于队头 (oi, ni)：

确保结果数组 res 的长度大于 ni，不够就用 null（None）补齐

设置 res[ni] = vals[oi]

然后处理原树中 oi 的左右孩子：

child = 2 * oi + 1（左）或 2 * oi + 2（右）

条件：

child 在数组范围内
valid[child] == true（是真实结点）
且 dp[child] > 0（说明这棵子树被保留下来）

满足条件则将 (child, 2*ni+1 或 2*ni+2) 入队

如果 dp[child] <= 0，表示这棵子树被“剪掉”，在新树对应位置将保持为 null。

BFS 完成后，res 中有若干 null：

中间的 null 是被剪掉的子树位置，必须保留
末尾连续的 null 是“完全二叉树填充”的冗余，需要按题意 从尾部删掉

这样就得到最终输出
四、小结
和“必须保留所有后代”的版本不同，这里：

DP 转移多了 max(0, 子树和)，允许剪掉坏分支；
仍然是 O(n) 的一次自底向上的 DP；
再配合 BFS + 剪枝条件 dp[child] > 0 来还原树结构。

复杂度分析
设数组长度为 n：

计算 dp[i]（自底向上）：每个下标访问一次，O(n)
BFS 还原最大子树：每个被保留结点最多被访问一次，O(n)
去掉末尾多余的 null：O(n)

总体：

时间复杂度：O(n)

空间复杂度：O(n)

dp 数组 O(n)
valid / vals / 队列 / 结果数组 O(n)

代码实现

**Python 代码：**

```python
import sys
from ast import literal_eval
from collections import deque

def max_pruned_subtree(arr):
n = len(arr)
if n == 0:
return []

# 标记每个位置是否为真实结点（非 None）
valid = [x is not None for x in arr]

# dp[i]：以 i 为根，允许裁掉贡献不为正的子树后，能得到的最大子树和
dp = [0] * n

best_sum = None  # 全局最大值
best_root = -1   # 对应的根下标

# 自底向上 DP
for i in range(n - 1, -1, -1):
if not valid[i]:
dp[i] = 0
continue

left = 2 * i + 1
right = 2 * i + 2

left_dp = dp[left] if left < n and valid[left] else 0
right_dp = dp[right] if right < n and valid[right] else 0

# 允许裁掉贡献不为正的子树
cur = arr[i]
if left_dp > 0:
cur += left_dp
if right_dp > 0:
cur += right_dp

dp[i] = cur

if best_sum is None or cur > best_sum:
best_sum = cur
best_root = i

if best_root == -1:
return []

# BFS 构造被裁剪后的最大子树（完全二叉树形式）
res = []
q = deque()
# 队列元素：(原数组下标, 新树下标)
q.append((best_root, 0))

while q:
oi, ni = q.popleft()

# 保证 res 长度足够
while len(res) <= ni:
res.append(None)

res[ni] = arr[oi]

left = 2 * oi + 1
right = 2 * oi + 2

# 左子树保留条件：存在、为真实结点、dp > 0
if left < n and valid[left] and dp[left] > 0:
q.append((left, 2 * ni + 1))
# 右子树保留条件：存在、为真实结点、dp > 0
if right < n and valid[right] and dp[right] > 0:
q.append((right, 2 * ni + 2))

# 去掉末尾多余的 None
while res and res[-1] is None:
res.pop()

return res

def main():
s = sys.stdin.readline().strip()
if not s:
return

# 将输入中的 'null' 替换为 Python 的 None，便于 literal_eval 解析
s = s.replace('null', 'None')
arr = literal_eval(s)  # 得到包含 int 和 None 的列表

ans = max_pruned_subtree(arr)

# 输出格式：[1,-1,null,2,...]
out = '[' + ','.join('null' if x is None else str(x) for x in ans) + ']'
print(out)

if __name__ == "__main__":
main()

```

---

## 2025年11月12日-AI方向

<a id="第2题-p4464"></a>

### 第2题-全连接层INT8非对称量化实现（P4464）- 中等





【背景】在移动设备部署深度学习模型时，浮点运算会消耗大量计算资源。通过 INT8 非对称量化，可将全连接层的浮点运算转化为整数运算，显著提高推理速度。实际应用中：

量化后模型大小缩小 4 倍（32bit→8bit）
整数运算指令比浮点指令快 2-4 倍
广泛应用于移动端 NLP 模型（如 BERT 最后一层分类头）
在物联网设备上可降低能耗并减少内存占用

【题目要求】请实现以下功能：

量化和全连接层计算：对输入向量 x 和权重矩阵 W 执行 INT8 非对称量化，使用量化后的整数值 xquantx_{quant}xquant和WquantW_{quant}Wquant 进行全连接层计算，输出计算结果。为简化起见，本题中全连接层不考虑偏置。
计算量化误差：对量化的整数进行反量化得到 xdequantx_{dequant}xdequant和 WdequantW_{dequant}Wdequant并进行全连接层计算，与原始浮点 x、W 的全连接层计算结果进行比较，计算两个全连接层输出之间的均方误差（MSE），并将 MSE × 100000 后四舍五入后输出。

【算法原理】
1、INT8 非对称量化：
1）尺度：scalev=(max(v)−min(v))/255scale_v=(max(v)-min(v))/255scalev=(max(v)−min(v))/255，当max(v)==min(v) max(v)==min(v)max(v)==min(v)，即张量 v 的所有值相等时，scalev=0scale_v=0scalev=0。
2）量化，对张量 v（向量 x 或矩阵 W）进行量化得到vquantv_{quant}vquant，量化后的整数区间为 [-128,127]：
$v_{quant} = clamp(round((v - min(v))/scale_v) - 128, -128,127)$
，当scalev=0时量化结果为vquant=−128当 scale_v = 0 时量化结果为v_{quant} = -128  当scalev=0时量化结果为vquant=−128。
其中 round () 采用就近取偶。
$\text{round}(x)= \begin{cases}  \lfloor x \rfloor, & \{x\} < \frac{1}{2}, \\ \lfloor x \rfloor + 1, & \{x\} > \frac{1}{2}, \\ 2 \cdot \lfloor \frac{x+1}{2} \rfloor, & \{x\} = \frac{1}{2}. \end{cases}$
其中：{x}=x−⌊x⌋\{x\} = x - \lfloor x \rfloor{x}=x−⌊x⌋，⌊x⌋\lfloor x \rfloor⌊x⌋ 表示向下取整。
$\text{clamp}(t, lo, hi)= \begin{cases}  lo, & t < lo \\ hi, & t > hi \\ t, & else \end{cases}$
3）反量化，对 vquantv_{\text{quant}}vquant 进行反量化后得到 vdequantv_{\text{dequant}}vdequant：
$v_{\text{dequant}} = (v_{\text{quant}} + 128) \cdot \text{scale}_v + \min(v)$，当 scalev=0\text{scale}_v = 0scalev=0 时，反量化值 vdequant=min⁡(v)v_{\text{dequant}} = \min(v)vdequant=min(v)，即为原始输入的最小值。
2、全连接层计算，以输入向量x和权重矩阵W为例，全连接层输出Y。
Y=x⋅WTY = x\cdot W^TY=x⋅WT
3、量化误差，计算原始浮点输入的全连接层输出 YfloatY_{\text{float}}Yfloat 和反量化数据的全连接层输出 YdequantY_{\text{dequant}}Ydequant 之间的均方误差（MSE）：
$MSE = \frac{1}{m} \sum_{i=0}^{m-1} (Y_{\text{float},i} - Y_{\text{dequant},i})^2$，m 为权重矩阵的行数。
输入描述
第一行: n (输入向量 x 的维度)第二行: n 个浮点数 (输入向量 x)第三行: m n (权重矩阵 W 的维度)接下来 m 行：每行 n 个浮点数 (权重矩阵 W)
输出描述
第一行: m 个整数 (使用量化数据 xquantx_{quant}xquant和 WquantW_{quant}Wquant计算的全连接层输出)
第二行: 1 个整数 (量化误差 MSE，注意是 MSE × 100000 后四舍五入输出整数)
样例1
输入
3
1.0 2.0 3.0
2 3
0.1 0.2 0.3
0.4 0.5 0.6

输出
13082 12929
0

说明
3 # n=3 (输入向量维度)
1.0 2.0 3.0 # x = [1.0, 2.0, 3.0]
2 3 # m=2, n=3 (权重矩阵 2×3)
0.1 0.2 0.3 # W 第 1 行 = [0.1, 0.2, 0.3]
0.4 0.5 0.6 # W 第 2 行 = [0.4, 0.5, 0.6]
量化输入向量 X: xquantx_{quant}xquant= [-128, 0, 127]
量化权重矩阵 W: WquantW_{quant}Wquant= [[-128, -77, -26], [25, 76, 127]]
量化域整数运算：输出第一行结果: 13082 12929
计算MSE
原始浮点输出:
Y_float [0] = 1.0×0.1 + 2.0×0.2 + 3.0×0.3 = 0.1 + 0.4 + 0.9 = 1.4
Y_float [1] = 1.0×0.4 + 2.0×0.5 + 3.0×0.6 = 0.4 + 1.0 + 1.8 = 3.2
反量化后: Y_dequant = Y_float
MSE: 输出第二行结果: 0
样例2
输入
7
0.3 -1.1 2.2 -3.3 4.4 -5.5 6.6
3 7
0.2 -0.3 0.4 -0.1 0 0.5 -0.6
-1.5 1.2 -0.9 0.6 -0.3 0.1 0
3 -2 1 -0.5 0.25 -0.125 0.0625

输出
-5476 -7406 8954
933

说明
7 # n=7 (输入向量维度)
0.3 -1.1 2.2 -3.3 4.4 -5.5 6.6 # x = [0.3, -1.1, 2.2, -3.3, 4.4, -5.5, 6.6]
3 7 # m=3, n=7 (权重矩阵 3×7)
0.2 -0.3 0.4 -0.1 0 0.5 -0.6 # W 第 1 行
-1.5 1.2 -0.9 0.6 -0.3 0.1 0 # W 第 2 行
3 -2 1 -0.5 0.25 -0.125 0.0625 # W 第 3 行
输出:
量化域整数运算输出: -5476 -7406 8954
MSE 输出: 933


#### 解答


解题思路
本题要求将输入向量与权重矩阵分别做 INT8 非对称量化（per-tensor），用量化后的整数直接做全连接（矩阵–向量乘），并用反量化后的结果评估与原始浮点计算之间的误差。
核心要点如下：

量化（asymmetric, INT8）
对张量 v（可为向量 x 或矩阵 W）做 per-tensor 量化：

尺度（scale）
$$\text{scale}_v = \frac{\max(v)-\min(v)}{255},\quad \text{若}\ \max(v)=\min(v)\ \text{则}\ \text{scale}_v=0$$

量化到 [−128,127][-128,127][−128,127]
$$v_{\text{quant}}=\text{clamp}\Big(\text{round}\Big(\frac{v-\min(v)}{\text{scale}_v}\Big)-128,\,-128,\,127\Big)$$其中 round 采用就近取偶（Banker’s Rounding）。当 scalev=0\text{scale}_v=0scalev=0 时，直接令 vquant=−128v_{\text{quant}}=-128vquant=−128。

反量化（dequant）
$$v_{\text{dequant}} = (v_{\text{quant}}+128)\cdot \text{scale}_v + \min(v)$$当 scalev=0\text{scale}_v=0scalev=0 时，令 vdequant=min⁡(v)v_{\text{dequant}}=\min(v)vdequant=min(v)。

全连接层计算
设输入为 x∈Rnx\in\mathbb{R}^nx∈Rn，权重 W∈Rm×nW\in\mathbb{R}^{m\times n}W∈Rm×n，输出
Y=x⋅W⊤∈RmY = x\cdot W^{\top}\in\mathbb{R}^m
Y=x⋅W⊤∈Rm

整数路径输出（第一行）：用 xquantx_{\text{quant}}xquant 与 WquantW_{\text{quant}}Wquant 直接做整数点积，得到 m 个整数。不添加偏置。

误差评估路径：分别将 xquant,Wquantx_{\text{quant}}, W_{\text{quant}}xquant,Wquant 反量化为 xdequant,Wdequantx_{\text{dequant}}, W_{\text{dequant}}xdequant,Wdequant，再做浮点全连接得到 YdequantY_{\text{dequant}}Ydequant。与原始浮点 YfloatY_{\text{float}}Yfloat 做均方误差
$$\text{MSE}=\frac{1}{m}\sum_{i=0}^{m-1}\big(Y_{\text{float},i}-Y_{\text{dequant},i}\big)^2$$输出时取 round(MSE×100000)\text{round}(\text{MSE}\times 100000)round(MSE×100000)，此处按“四舍五入”（half-up）。

实现细节

x 与 W 分别独立计算 min⁡,max⁡,scale\min,\max,\text{scale}min,max,scale（per-tensor 量化）。
量化时采用就近取偶；MSE 放大后采用四舍五入（half-up）。
矩阵乘法按行做点积；整数输出建议用较大整型累加避免溢出。

代码实现

**Python 代码：**

```python
import sys
import math

def quantize_tensor(values):
"""对一维列表进行INT8非对称量化，返回(q, scale, vmin)"""
vmin = min(values)
vmax = max(values)
if vmax == vmin:
# scale为0：全量化为-128
return [-128] * len(values), 0.0, vmin
scale = (vmax - vmin) / 255.0
q = []
for v in values:
t = (v - vmin) / scale  # 落在[0,255]
rq = round(t)           # 就近取偶
iv = int(rq) - 128
if iv < -128:
iv = -128
elif iv > 127:
iv = 127
q.append(iv)
return q, scale, vmin

def dequantize_tensor(q, scale, vmin):
"""反量化一维列表"""
if scale == 0.0:
return [vmin] * len(q)
return [(qi + 128) * scale + vmin for qi in q]

def fc_int_output(xq, Wq, n):
"""使用量化后的整数做全连接：返回长度m的整数输出"""
m = len(Wq) // n
y = []
for i in range(m):
s = 0
base = i * n
for j in range(n):
s += xq[j] * Wq[base + j]
y.append(s)
return y

def fc_float_output(x, W, n):
"""浮点全连接：返回长度m的浮点输出"""
m = len(W) // n
y = []
for i in range(m):
s = 0.0
base = i * n
for j in range(n):
s += x[j] * W[base + j]
y.append(s)
return y

def round_half_up(x):
"""四舍五入到最近整数（正数half-up，MSE>=0安全）"""
return int(math.floor(x + 0.5))

def main():
data = sys.stdin.read().strip().split()
it = iter(data)

n = int(next(it))
x = [float(next(it)) for _ in range(n)]

m = int(next(it)); n2 = int(next(it))  # 题面保证维度合法
W = []
for _ in range(m):
for _ in range(n):
W.append(float(next(it)))

# 量化（x 与 W 分别 per-tensor）
xq, sx, xmin = quantize_tensor(x)
Wq, sw, wmin = quantize_tensor(W)

# 整数路径输出
y_int = fc_int_output(xq, Wq, n)

# 误差评估：反量化 -> 浮点全连接
x_d = dequantize_tensor(xq, sx, xmin)
W_d = dequantize_tensor(Wq, sw, wmin)
y_float = fc_float_output(x, W, n)
y_deq = fc_float_output(x_d, W_d, n)

# MSE × 100000 四舍五入
msz = len(y_float)
mse = sum((y_float[i] - y_deq[i]) ** 2 for i in range(msz)) / msz
mse_scaled = round_half_up(mse * 100000.0)

# 输出
print(' '.join(str(v) for v in y_int))
print(mse_scaled)

if __name__ == "__main__":
main()

```

---

<a id="第3题-p4465"></a>

### 第3题-基于决策树的QAM调制符合检测（P4465）- 困难





在无线通信中使用QAM调制将信息通过无线信号从发送端传递到接收端。QAM调制后的信号可以使用一个复数表示。16QAM调制会生成16个不同的复数信号。在无线信号传输过程中，信号会受到高斯噪声污染，使得接收到的QAM信号与发送的QAM信号产生误差。该过程可以用如下公式表示：
Srx=Stx+nS_{rx} = S_{tx} + nSrx=Stx+n，其中，n为复数高斯噪声。
例如，一个发送16QAM16QAM16QAM调制符号为：
Stx=−1+1jS_{tx} = -1 + 1jStx=−1+1j
传输过程中受到的噪声信号为：
n=0.38−1.2jn = 0.38 - 1.2jn=0.38−1.2j
接收到的16QAM16QAM16QAM调制符号为：
Srx=−0.62−0.2jS_{rx} = -0.62 - 0.2jSrx=−0.62−0.2j
无线信号的符号检测过程，就是根据接收到的受噪声污染的QAM符号，判决输出其真实发送QAM符号。
下图所示为16QAM16QAM16QAM调制符号的星座图。图中，蓝色圆点表示发送的QAM符号，红色点表示受噪声污染后的接收QAM符号。
请使用CART决策树实现一个QAM符号检测器，完成16QAM16QAM16QAM调制的无线信号的接收检测。

要求：

根据输入的M个接收16QAM调制符号和真实标签构建CART决策树；
使用基尼系数（GiniGiniGini）作为划分标准；
决策树最大深度=5；
特征值切分点限制为{−3,−2,−1,0,1,2,3}\{-3,-2,-1,0,1,2,3\}{−3,−2,−1,0,1,2,3}；
输出训练集的GiniGiniGini系数；
输出验证QAM符号标签。

输入描述
第一行：一个整数 M ，表示训练样本集个数，取值范围[10~20]
接下来M行：两个实数 x1x1x1 ， x2x2x2 和一个整数 y ，以空格间隔。其中， x1x1x1 ， x2x2x2 分别表示复数QAM符号的实部和虚部，取值范围 [-10 ~ +10]，保留小数点后2位。 y 表示QAM符号的标签，取值范围[0~15]。
第 M+2M+2M+2 行：两个实数 x1x1x1 ， x2x2x2 ，分别表示测试用接收QAM符号的实部和虚部，取值范围 [-10 ~ +10]，保留小数点后2位。
输出描述
第一行：一个实数 G ，表示训练样本集合的Gini系数，四舍五入后保留小数点后4位。
第二行：一个整数 y ，表示测试QAM符号的分类标签。
样例1
输入
10
2.56 0.73 14
3.88 0.83 14
-0.32 2.93 7
-2.99 -3.56 0
3.36 -1.52 13
-2.70 -1.13 1
-0.57 0.97 6
2.71 3.22 15
2.35 -2.55 12
4.18 -1.25 13
-1.14 0.20

输出
0.8600
6

说明
上述输入第1行为训练样本集合中样本个数：10。
接下来10行为10个16QAM调制符号的接收信号（复数信号的实部、虚部），以及对应的原始发送符号的标签。
第12行为测试用的接收16QAM调制符号信号（复数信号的实部、虚部）。
输出第1行数值为使用这10个符号及对应原始符号标签作为训练样本集合，计算出的该集合Gini系数。数值四舍五入后保留四位小数。
输出第2行为基于上述构建的决策树对测试样本的原始发送符号标签的预测值。
样例2
输入
11
-3.24 0.96 2
2.79 0.95 14
2.99 -2.94 12
0.67 -2.55 8
-1.30 -0.71 5
0.73 -2.96 8
-3.04 1.30 2
-2.81 -0.68 1
2.88 3.33 15
-2.55 2.87 3
-1.01 -0.62 5
-3.24 -2.90

输出
0.8595
2

说明
上述输入第1行为训练样本集合中样本个数：11。
接下来11行为11个16QAM调制符号的接收信号（复数信号的实部、虚部），以及对应的原始发送符号的标签。
第13行为测试用的接收16QAM调制符号接收信号（复数信号的实部、虚部）。
输出第1行为使用这11个符号和对应的符号标签作为训练样本集合，计算出的该集合GiniGiniGini系数。数值四舍五入后保留四位小数。
输出第2行为基于上述构建的决策树对测试样本的原始发送符号标签的预测值。
提示
样本集合中的样本有K个类别，每个类别的样本，在样本集合中的概率分布为P=(P1,P2,...,PK)P = (P_1, P_2, ..., P_K)P=(P1,P2,...,PK)
给定样本集合D，计算其GiniGiniGini系数时，首先需要计算出样本集合中每个类别出现的比例PiP_iPi，然后基于如下GiniGiniGini系数计算公式计算：
Gini(D)=1−∑i=1KPi2Gini(D) = 1 - \sum_{i=1}^{K} P_i^2Gini(D)=1−∑i=1KPi2
其中，PiP_iPi是第i类样本出现的比例，K是样本中总类别数。
CART树实现步骤：

特征及切分点选择

遍历样本所有特征，对每一个特征值的特征值进行排序，以相邻特征值的中值作为切分点，计算以该切分点将样本划分为D1D1D1和D2D2D2两个子集后的加权基尼系数。
加权GiniGiniGini系数计算公式为：
Giniweight=W1Gini(D1)+W2Gini(D2)Gini_{weight} = W_1Gini(D_1) + W_2Gini(D_2)Giniweight=W1Gini(D1)+W2Gini(D2)
其中，$W_1$为子集D1D1D1中样本在集合D中占比，$W_2$为子集D2D2D2中样本在集合D中占比。

节点划分

选择使加权GiniGiniGini系数最小的特征和特征值切分点，将数据集划分为左右两个子集：左子集D1(<特征值划分点)D1(<特征值划分点)D1(<特征值划分点)和右子集D2(≥特征值划分点)D2(≥特征值划分点)D2(≥特征值划分点)。

递归构建树

对每个子集重复步骤1和2，直到满足停止条件（如节点样本数小于阈值，或达到最大深度）。


#### 解答


解题思路
本题要求用 CART 决策树 在二维特征（实部 x1、虚部 x2）上完成 16QAM 符号标签的分类，并使用 Gini 系数作为划分标准，树的最大深度为 5，且切分点仅允许取自集合 {-3,-2,-1,0,1,2,3}。训练后需要输出：
1）训练样本集合的整体 Gini 系数；2）对给定测试点的预测标签。
相关算法与实现要点

节点不纯度（Gini）
对任意样本集合 D，令第 i 类（标签）比例为 PiP_iPi，则
Gini(D)=1−∑iPi2Gini(D)=1-\sum_i P_i^2
Gini(D)=1−i∑Pi2
训练集整体 Gini 直接按全部训练样本的标签频率计算一次即可。

候选划分

特征：两维 (x1,x2)(x1,x2)(x1,x2)。

切分点：限定在 {−3,−2,−1,0,1,2,3}\{-3,-2,-1,0,1,2,3\}{−3,−2,−1,0,1,2,3}。

对每个特征与每个切分点，按规则将样本分为：
左子集 D1={x∣xf<t}D_1=\{x\mid x_f < t\}D1={x∣xf<t}，右子集 D2={x∣xf≥t}D_2=\{x\mid x_f \ge t\}D2={x∣xf≥t}。

计算加权 Gini：
$$Gini_{weight}=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2)$$

选择 加权 Gini 最小 且 两侧非空 的划分；若最优加权 Gini 未严格小于 当前集合的 Gini（或已纯/深度到限），则生成叶子结点。

叶子预测

叶子输出当前集合的多数类标签；若并列，取数值更小的标签，保证确定性。

构树

从根开始递归，深度上限为 5。
每个结点重复“枚举划分 → 选择最优 → 递归子树”的步骤。

预测

自根结点按“< 阈值走左，≥\ge≥ 阈值走右”到达叶子，输出叶子标签。

代码实现

**Python 代码：**

```python
import sys
from collections import Counter

# 计算集合的 Gini 系数
def gini_of_labels(labels):
n = len(labels)
if n == 0:
return 0.0
cnt = Counter(labels)
s = 0.0
for c in cnt.values():
p = c / n
s += p * p
return 1.0 - s

# 多数类（并列取更小标签）
def majority_label(labels):
cnt = Counter(labels)
maxc = max(cnt.values())
candidates = [k for k, v in cnt.items() if v == maxc]
return min(candidates)

# 决策树结点
class Node:
def __init__(self):
self.is_leaf = True
self.label = 0
self.feature = -1
self.threshold = 0.0
self.left = None
self.right = None

# 递归建树
THRESHOLDS = [-3, -2, -1, 0, 1, 2, 3]

def build_tree(X, y, idxs, depth_left):
node = Node()
curr_labels = [y[i] for i in idxs]
curr_gini = gini_of_labels(curr_labels)
# 叶子条件：纯、深度用尽
if curr_gini == 0.0 or depth_left == 0:
node.is_leaf = True
node.label = majority_label(curr_labels)
return node

best_gini = float('inf')
best_f, best_t = -1, None
best_left, best_right = None, None

# 枚举特征与阈值
for f in [0, 1]:
for t in THRESHOLDS:
left = []
right = []
for i in idxs:
if X[i][f] < t:
left.append(i)
else:
right.append(i)
# 两侧必须非空
if len(left) == 0 or len(right) == 0:
continue
g_left = gini_of_labels([y[i] for i in left])
g_right = gini_of_labels([y[i] for i in right])
w = (len(left) / len(idxs)) * g_left + (len(right) / len(idxs)) * g_right
# 选择加权 Gini 更小的划分；平手时用更小特征、更小阈值保证确定性
if w < best_gini - 1e-12 or (abs(w - best_gini) <= 1e-12 and (f < best_f or (f == best_f and t < (best_t if best_t is not None else 0)))):
best_gini = w
best_f, best_t = f, t
best_left, best_right = left, right

# 若无有效提升则设为叶子
if best_left is None or best_gini >= curr_gini - 1e-12:
node.is_leaf = True
node.label = majority_label(curr_labels)
return node

# 划分并递归
node.is_leaf = False
node.feature = best_f
node.threshold = best_t
node.left = build_tree(X, y, best_left, depth_left - 1)
node.right = build_tree(X, y, best_right, depth_left - 1)
return node

def predict(root, x):
node = root
while not node.is_leaf:
if x[node.feature] < node.threshold:
node = node.left
else:
node = node.right
return node.label

def main():
data = sys.stdin.read().strip().split()
it = iter(data)
M = int(next(it))
X = []
y = []
for _ in range(M):
x1 = float(next(it)); x2 = float(next(it)); yy = int(next(it))
X.append([x1, x2]); y.append(yy)
tx1 = float(next(it)); tx2 = float(next(it))
test = [tx1, tx2]

# 训练集整体 Gini
G = gini_of_labels(y)

# 训练 CART 树
idxs = list(range(M))
root = build_tree(X, y, idxs, depth_left=5)

# 预测
pred = predict(root, test)

# 输出
print(f"{G:.4f}")
print(pred)

if __name__ == "__main__":
main()

```

---

## 2025年11月6日-留学生AI方向

<a id="第2题-p4447"></a>

### 第2题-医疗诊断模型的训练与更新（P4447）- 中等





某智能医疗平台正在研发一套基于人工智能的自动疾病辅助诊断系统。例如，该系统通过对患者多次填写的症状问卷数据进行分析，帮助医生快速判断患者属子健康、感冒还是肺炎三类之一、每位患者在就诊前需填写一个包含多个症状的问题序列(如咳歌、发热、咽痛等);每条问卷的症状项被嵌入为特征向量，形成一个长度为 L 的症状序列，每个症状的特征维度为 D 。这些离散症状特征经过预处理后输入到诊断系统中。系统采用一层 MLP 进行特征映射，再使用一层 MLP 作为分类器输出各症状的预测概率，为简化考虑，输出率无需进行 sottmaxsottmaxsottmax 归一化。同时，MLP 层也无偏置项。请实现以下输出:

前向推理:输出预测概率( K 个，例如 K=3K=3K=3 时表示分类为健康/感冒/肺炎的概率)，并取症状维度的平均值作为输出

LOSSLOSSLOSS 计算:输出 MSE 损失 LOSSLOSSLOSS ; 定义为，$L_{m s c}=\frac{1}{K} \sum_{i=1}^{K}\left(y_{i}-\hat{y}_{i}\right)^{2}$
其中 K 为类别数， $y_1$ 为真实概率， y^i2\hat{y}_{i}^{2}y^i2 表示预测械率。

权重更新:输出单次反向传播后的权重。更新采用 SGD 优化器，定义为:
$W_{\text {new }}=W_{\text {old }}-\eta \nabla_{w} L$
其中 η\etaη 为学习率。

输入描述
第 1 行:序列长度 L∈[1,10]L∈[1,10]L∈[1,10]、特征维度 D∈[1,10]D∈[1,10]D∈[1,10]、分类数 K∈[2,5]K ∈[2,5]K∈[2,5]、学习率 η∈[0,1]\eta∈[0,1]η∈[0,1]
第 2 行:真实概率，K 个数
第 3 行:输入序列, L×DL×DL×D 个数
第 4 行:MLP 参数 WmlpW_{mlp}Wmlp，D×DD×DD×D 个数
第 5 行:分类层参数 WclsW_{cls}Wcls，D×KD×KD×K 个数
输出描述
第 1 行: K 个类别的预测概率
第 2 行:MSE LOSSLOSSLOSS，1 个数
第 3 行:MLP 更新后的参数 WmlpW_{mlp}Wmlp，D×DD×DD×D 个数
第 4 行:分类层更新后的参数 WclsW_{cls}Wcls，D×KD×KD×K 个数
注:数据间用运号隔开，输出结果均保留 2 位小数
样例1
输入
4,2,5,1.0
0.10,0.20,0.30,0.25,0.15
0.0,1.0,-1.5,2.5,3.0,-0.5,0.7,0.3
0.6,-0.4,0.2,0.9
0.5,0.1,-0.3,0.8,0.0,-0.2,0.4,0.6,-0.5,1.0

输出
0.14,0.26,0.16,0.13,0.52
0.04
0.61,-0.48,0.21,0.78
0.49,0.09,-0.27,0.82,-0.07,-0.21,0.39,0.63,-0.48,0.92

说明
输入:
第 1 行:序列长度 L=4L=4L=4、特征维度 D=2D=2D=2 、分类数 K=5K=5K=5、学习率 η=1.0η=1.0η=1.0
第 2 行:表示真实标签三分类的概率分别为 0.10,0.20,0.30,0.25,0.150.10,0.20,0.30,0.25,0.150.10,0.20,0.30,0.25,0.15
第 3 行:输入序列数据内容，4×2=84×2=84×2=8 个数
第 4 行: MLP 参数 WmlpW_{mlp}Wmlp，2×22×22×2 个数
第 5 行:分类层参数 WclsW_{cls}Wcls，2×52×52×5 个数
输出:
第 1 行:五分类的预测概率分别为 0.14,0.26,0.16,0.13,0.520.14,0.26,0.16,0.13,0.520.14,0.26,0.16,0.13,0.52
第 2 行: MSELOSSMSELOSSMSELOSS 为 0.040.040.04
第 3 行: MLP 更新后的参数 WmlpW_{mlp}Wmlp
第 4 行:分类层更新后的参数 WclsW_{cls}Wcls
样例2
输入
2,2,3,0.1
1.0,0.0,0.0
1.0,2.0,3.0,4.0
1.0,1.0,1.0,1.0
1.0,0.0,0.0,1.0,0.0,0.0

输出
5.00,0.00,0.00
5.33
0.47,-0.53,-0.80,0.20
0.47,0.00,0.00,0.20,0.00,0.00

说明
输入:
第 1 行:序列长度 L=2L=2L=2、特征维度 D=2D=2D=2 、分类数 K=3K=3K=3、学习率 η=0.1η=0.1η=0.1
第 2 行:表示真实标签三分类的概率分别为 1.0、0.0、0.01.0、0.0、0.01.0、0.0、0.0
第 3 行:输入序列数据内容，2×2=42×2=42×2=4 个数
第 4 行: MLP 参数 WmlpW_{mlp}Wmlp，2×22×22×2 个数
第 5 行:分类层参数 WclsW_{cls}Wcls，2×32×32×3 个数
输出:
第 1 行:三分类的预测概率分别为 5.00、0.00、0.005.00、0.00、0.005.00、0.00、0.00
第 2 行: MSELOSSMSELOSSMSELOSS 为 5.335.335.33
第 3 行: MLP 更新后的参数 WmlpW_{mlp}Wmlp
第 4 行:分类层更新后的参数 WclsW_{cls}Wcls


#### 解答


解题思路
该题要求实现一个极简的序列分类器：对长度为 L 的症状序列（每步维度 D）进行两层线性映射（均为无偏置的 MLP 层），得到每步的 K 维预测，再对“序列维度”取平均作为最终 K 类输出；损失函数为 MSE。随后基于单样本进行一次 SGD 更新，输出更新后的两层权重。

模型与前向传播

记第 t 步输入为向量 xt∈RDx_t\in\mathbb{R}^Dxt∈RD。

第一层 MLP（无偏置）：
$$h_t \;=\; x_t W_{\text{mlp}},\quad W_{\text{mlp}}\in\mathbb{R}^{D\times D}$$

分类层（无偏置）：
$$p_t \;=\; h_t W_{\text{cls}},\quad W_{\text{cls}}\in\mathbb{R}^{D\times K}$$

序列平均作为最终输出（无需 softmax）：
$$\hat{y} \;=\; \frac{1}{L}\sum_{t=1}^{L} p_t \;\in\; \mathbb{R}^{K}$$

损失函数
采用均方误差：

$$L \;=\; \frac{1}{K}\sum_{i=1}^{K}(\hat{y}_i - y_i)^2$$
反向传播与梯度

令
$$g \;=\; \frac{\partial L}{\partial \hat{y}} \;=\; \frac{2}{K}(\hat{y}-y) \;\in\; \mathbb{R}^{K}$$

因 y^=1L∑tpt\hat{y}=\frac{1}{L}\sum_t p_ty^=L1∑tpt，有
$$\frac{\partial L}{\partial p_t} \;=\; \frac{1}{L}g \quad(\forall t)$$

对分类层：
$$\frac{\partial L}{\partial W_{\text{cls}}}
\;=\;
\sum_{t} h_t^\top \left(\frac{1}{L}g\right)
\;=\;
\left(\frac{1}{L}\sum_t h_t\right)^\top g
\;\;\in\mathbb{R}^{D\times K}$$记 hˉ=1L∑tht \bar{h}=\frac{1}{L}\sum_t h_thˉ=L1∑tht，则 ∂L/∂Wcls=hˉ⊤g\partial L/\partial W_{\text{cls}}=\bar{h}^\top g∂L/∂Wcls=hˉ⊤g（外积）。

对第一层：
$$\frac{\partial L}{\partial h_t}
\;=\;
\left(\frac{1}{L}g\right) W_{\text{cls}}^\top
\;=\;
\frac{1}{L}\, (W_{\text{cls}}g) \;\in\; \mathbb{R}^{D}$$$$\frac{\partial L}{\partial W_{\text{mlp}}}
\;=\;
\sum_t x_t^\top \left(\frac{1}{L}\,W_{\text{cls}}g\right)
\;=\;
\left(\frac{1}{L}\sum_t x_t\right)^\top (W_{\text{cls}}g)
\;\;\in\mathbb{R}^{D\times D}$$记 xˉ=1L∑txt\bar{x}=\frac{1}{L}\sum_t x_txˉ=L1∑txt，  v=Wclsg\;v=W_{\text{cls}}gv=Wclsg，则 ∂L/∂Wmlp=xˉ⊤v\partial L/\partial W_{\text{mlp}}=\bar{x}^\top v∂L/∂Wmlp=xˉ⊤v（外积）。

SGD 更新

$$W_{\text{mlp}} \leftarrow W_{\text{mlp}} - \eta \frac{\partial L}{\partial W_{\text{mlp}}},\quad
W_{\text{cls}} \leftarrow W_{\text{cls}} - \eta \frac{\partial L}{\partial W_{\text{cls}}}$$
输出格式

第 1 行：y^\hat{y}y^（K 个，保留 2 位小数，逗号分隔）
第 2 行：MSE（1 个，保留 2 位小数）
第 3 行：更新后的 WmlpW_{\text{mlp}}Wmlp（按行展开 D×D，保留 2 位）
第 4 行：更新后的 WclsW_{\text{cls}}Wcls（按行展开 D×K，保留 2 位）

复杂度分析

前向：计算所有 hth_tht 与 ptp_tpt：
O(L⋅D2)O(L\cdot D^2)O(L⋅D2)（第一层） + O(L⋅D⋅K)O(L\cdot D\cdot K)O(L⋅D⋅K)（分类层）。
反向：
计算 g：O(K)O(K)O(K)；
计算 hˉ\bar{h}hˉ：O(L⋅D)O(L\cdot D)O(L⋅D)；
计算 ∂L/∂Wcls\partial L/\partial W_{\text{cls}}∂L/∂Wcls：O(D⋅K)O(D\cdot K)O(D⋅K)；
计算 v=Wclsgv=W_{\text{cls}}gv=Wclsg：O(D⋅K)O(D\cdot K)O(D⋅K)；
计算 xˉ\bar{x}xˉ：O(L⋅D)O(L\cdot D)O(L⋅D)；
计算 ∂L/∂Wmlp\partial L/\partial W_{\text{mlp}}∂L/∂Wmlp：O(D2)O(D^2)O(D2)。
总体时间复杂度：O(L⋅D2+L⋅D⋅K)O(L\cdot D^2 + L\cdot D\cdot K)O(L⋅D2+L⋅D⋅K)，在题目约束下（均 ≤10）非常合适。
空间复杂度：存储输入与权重及中间量，主要为 O(L⋅D+D2+D⋅K)O(L\cdot D + D^2 + D\cdot K)O(L⋅D+D2+D⋅K)。

代码实现

**Python 代码：**

```python
import sys
import ast

# 将一行形如 "1,2,3" 的输入安全解析为列表
def parse_line(line: str):
return list(ast.literal_eval("[" + line.strip() + "]"))

# 前向计算与一次SGD更新，返回(y_hat, loss, Wmlp_new, Wcls_new)
def solve_once(L, D, K, eta, y_true, seq_flat, Wmlp_flat, Wcls_flat):
# 还原形状
X = [seq_flat[i*D:(i+1)*D] for i in range(L)]           # L x D
Wmlp = [Wmlp_flat[i*D:(i+1)*D] for i in range(D)]       # D x D（按行）
Wcls = [Wcls_flat[i*K:(i+1)*K] for i in range(D)]       # D x K（按行）

# 前向：h_t = x_t @ Wmlp，p_t = h_t @ Wcls
H_sum = [0.0]*D
P_avg = [0.0]*K
for t in range(L):
x = X[t]  # 长度D
# h = x @ Wmlp
h = [0.0]*D
for j in range(D):
s = 0.0
for d in range(D):
s += x[d] * Wmlp[d][j]
h[j] = s
# 累加 H_sum
for j in range(D):
H_sum[j] += h[j]
# p = h @ Wcls
p = [0.0]*K
for k in range(K):
s = 0.0
for j in range(D):
s += h[j] * Wcls[j][k]
p[k] = s
# 累加到平均（最后再除以L）
for k in range(K):
P_avg[k] += p[k]
P_avg = [v / L for v in P_avg]  # 预测 \hat{y}

# 计算 MSE
loss = 0.0
for k in range(K):
diff = (P_avg[k] - y_true[k])
loss += diff * diff
loss /= K

# 反向：g = (2/K) * (y_hat - y_true)
g = [(2.0 / K) * (P_avg[k] - y_true[k]) for k in range(K)]

# dL/dWcls = ( (sum_t h_t)/L )^T * g   （外积）
H_bar = [v / L for v in H_sum]  # D
dWcls = [[H_bar[j] * g[k] for k in range(K)] for j in range(D)]

# v = Wcls @ g    （D）
v = [0.0]*D
for j in range(D):
s = 0.0
for k in range(K):
s += Wcls[j][k] * g[k]
v[j] = s

# dL/dWmlp = ( (sum_t x_t)/L )^T * v   （外积）
X_sum = [0.0]*D
for t in range(L):
for d in range(D):
X_sum[d] += X[t][d]
X_bar = [v_ / L for v_ in X_sum]  # D
dWmlp = [[X_bar[i] * v[j] for j in range(D)] for i in range(D)]

# SGD 更新
for i in range(D):
for j in range(D):
Wmlp[i][j] -= eta * dWmlp[i][j]
for j in range(D):
for k in range(K):
Wcls[j][k] -= eta * dWcls[j][k]

# 展平权重并返回
Wmlp_new = [Wmlp[i][j] for i in range(D) for j in range(D)]
Wcls_new = [Wcls[j][k] for j in range(D) for k in range(K)]
return P_avg, loss, Wmlp_new, Wcls_new

def fmt_line(arr):
return ",".join(f"{x:.2f}" for x in arr)

def main():
lines = [line for line in sys.stdin if line.strip() != ""]
# 读取五行
L, D, K, eta = parse_line(lines[0])
y_true = parse_line(lines[1])
seq_flat = parse_line(lines[2])
Wmlp_flat = parse_line(lines[3])
Wcls_flat = parse_line(lines[4])

y_hat, loss, Wmlp_new, Wcls_new = solve_once(L, D, K, eta, y_true, seq_flat, Wmlp_flat, Wcls_flat)

print(fmt_line(y_hat))
print(f"{loss:.2f}")
print(fmt_line(Wmlp_new))
print(fmt_line(Wcls_new))

if __name__ == "__main__":
main()

```

---

<a id="第3题-p4448"></a>

### 第3题-卷积操作（P4448）- 中等





卷积神经网络常用于图像分类、检测与分割等图像任务。多通道卷积的计算公式是卷积神经网络 (CNN)(CNN)(CNN) 中的核心运算，请用代码实现多通道卷积操作。
输入:
inputinputinput: 形状为 (C,H_in,W_in)(C,H\_in,W\_in)(C,H_in,W_in) 的输入张量，C 为输入通道数，H_inH\_inH_in 为输入高度，W_inW\_inW_in 为输入宽度
kernelkernelkernel: 形状为 (C,K_h,K_w)(C,K\_h,K\_w)(C,K_h,K_w) 的卷积核，C 为输入通道数(和 inputinputinput 的 C 数值相同)，K_hK\_hK_h 为输入高度，K_wK\_wK_w 为输入宽度
stridestridestride:卷积步长(大于等于 1 的整数)
其中
0<C<60<C<60<C<6
2<H_in<102<H\_in<102<H_in<10
2<W_in<102 <W\_in < 102<W_in<10
0<K_h<70<K\_h<70<K_h<7
0<K_w<70<K\_w<70<K_w<7
0<stride<40 < stride < 40<stride<4
0=<padding=<30 =< padding = < 30=<padding=<3
卷积操作过程说明:
1.填充:在输入张量的上下左右各填充 “0” ，填充后的形状为 (C,H_in+2padding,W_in+2padding)(C,H\_in+2padding,W\_in+2padding)(C,H_in+2padding,W_in+2padding)。
2 滑动窗口计算:从填充后的输入张量的左上角开始，按照 stridestridestride 步长滑动卷积核，每次取与卷积核形状相同的子区域。若剩余区域尺寸小于卷积核尺寸，则跳过该位置。
3,逐通道计算:对于每个子区域，将卷积核与对应位置的输入值逐通道相乘后求和(中间计算过程不做四舍五入)，得到输出张量的一个值，计算公式为:$\operatorname{output}(i, j)=\sum_{c=0}^{C-1} \sum_{m=0}^{K_{b}-1} \sum_{n=0}^{K_{n}-1} \text { input }_{c}(i \times \text { stride }+m, j \times \text { stride }+n) \times \operatorname{kernel}_{c}(m, n)$
4.输出结果:将所有子区域的计算结果组合成 (H_out,W_out)(H\_out,W\_out)(H_out,W_out) 的输出张量。
输出:
形状为 (H_out,W_out)(H\_out, W\_out)(H_out,W_out) 的 2D 数组，其中:
H_out=(H_in+2∗padding−k)//stride+1H\_out=(H\_in + 2 * padding -k) // stride + 1H_out=(H_in+2∗padding−k)//stride+1
W_out=(W_in+2∗padding−K)//stride+1W\_out= (W\_in + 2 * padding - K) // stride +1W_out=(W_in+2∗padding−K)//stride+1
注: ////// 代表除法后向下取整

上图为单通道卷积 paddingpaddingpadding 为 1，stridestridestride 为 1 时示意图，多通道卷积时还需对各通道进行求和，题中不再绘制示意图
输入描述
第一行 C,H_in,W_inC,H\_in, W\_inC,H_in,W_in 表示输入的张量的通道数、行数与列数;
接下来的 C×H_inC × H\_inC×H_in 行是此张量的元素值;
接下来一行 C,K_h,K_wC,K\_h,K\_wC,K_h,K_w 是卷积核的通道数、行数与列数;
接下来 C×H_inC×H\_inC×H_in是卷积核的元素值;
最后一行是 stridestridestride paddingpaddingpadding
输出描述
输出卷积后形状为 (H_out,W_out)(H\_out,W\_out)(H_out,W_out) 的特征图(二维矩阵)，元素均为整数。
样例1
输入
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

输出
22 28
40 46

说明
第一行 2 3 3 表示输入的张量为 2 通道，每个通道的行数和列数均为 3
第 2−42-42−4 行是此张量第一通道的元素值
第 5−75-75−7 行是此张量第二通道的元素值
第 8 行的 2 2 2 是卷积核的形状，其通道数与输入张量一致，都是 2 ，行数和列数都是 2
第 9、109、109、10 行是此卷积核第 1 通道的元素值
第 11、1211、1211、12 行是此卷积核第 2 通道的元素值
最后一行 2 1 代表卷积操作中卷积窗滑动的步长 stridestridestride 为 1 ，paddingpaddingpadding 为 0

计算过程:

输出尺寸: H_out=13+2×0−2+1=2,W_out=2H\_out=13+2×0-2+1=2,W\_out=2H_out=13+2×0−2+1=2,W_out=2

输出矩阵位置 (0,0)(0,0)(0,0) 的计算过程如下:
通道 1 :窗口 [[1,2],[4,5]]→(1×1)+(2×0)+(4×0)+(5×1)=6[[1,2],[4,5]]→(1×1)+(2×0)+(4×0)+(5×1)=6[[1,2],[4,5]]→(1×1)+(2×0)+(4×0)+(5×1)=6
通道 2 :窗口 [[2,3],[5,6]]→(2×2)+(3×0)+(5×0)+(6×2)=4+12=16[[2,3],[5,6]]→(2×2)+(3×0)+(5×0)+(6×2)=4+12=16[[2,3],[5,6]]→(2×2)+(3×0)+(5×0)+(6×2)=4+12=16
各通道求和: 6+16=226+16=226+16=22

输出矩阵位置 (0,1)(0,1)(0,1) 和 (1,0)(1,0)(1,0) 元素计算过程略。

位置 (1,1)(1,1)(1,1) 计算:
通道 1 :窗口 [[5,6],[8,9]]→(5×1)+(6×0)+(8×0)+(9×1)=14[[5,6],[8,9]]→(5×1)+(6×0)+(8×0)+(9×1)=14[[5,6],[8,9]]→(5×1)+(6×0)+(8×0)+(9×1)=14
通道 2 :窗口 [[6,71,[9,10]]→(6×2)+(7×0)+(9×0)+(10×2)=12+20=32[[6,71,[9,10]]→(6×2)+(7×0)+(9×0)+(10×2)=12+20=32[[6,71,[9,10]]→(6×2)+(7×0)+(9×0)+(10×2)=12+20=32
各通道总和: 14+32=4614+32=4614+32=46


#### 解答


解题思路
多通道卷积是卷积神经网络（CNN）中的核心运算。其本质是：在每个滑动位置，将输入在所有通道上的同形窗口与对应通道的卷积核做元素乘加（逐通道累加），得到一个标量作为该位置的输出值。

填充（padding）
按题意在输入张量四周补零，得到形状为
(C, Hin+2p, Win+2p)(C,\, H_{in}+2p,\, W_{in}+2p)(C,Hin+2p,Win+2p) 的新张量（p=paddingp=paddingp=padding）。

滑动窗口（stride）
以步长 s（s=strides=strides=stride）在高、宽方向移动卷积核窗口。仅当窗口完全落在填充后的输入内时才计算。输出尺寸为
$$H_{out}=\left\lfloor\frac{H_{in}+2p-K_h}{s}\right\rfloor+1,\quad
W_{out}=\left\lfloor\frac{W_{in}+2p-K_w}{s}\right\rfloor+1$$

逐通道累加（核心计算）
在位置 (i,j)(i,j)(i,j) 处的输出值：
$$\text{out}(i,j)=\sum_{c=0}^{C-1}\sum_{m=0}^{K_h-1}\sum_{n=0}^{K_w-1}
X_c(i\cdot s+m,\ j\cdot s+n)\cdot K_c(m,n)$$其中 X 为填充后的输入，K 为卷积核。中间计算用整数即可（题面输出也为整数）。

实现方法

读入参数与数据，构造三维输入与核。
先做零填充，得到新三维数组。
依据上式三重（或四重）循环完成滑动与乘加。
输出 Hout×WoutH_{out}\times W_{out}Hout×Wout 的二维矩阵。

本题无需使用快速傅里叶等高级算法，直接按定义实现即可，参数规模很小（C≤6,Hin,Win<10C\le 6, H_{in},W_{in}<10C≤6,Hin,Win<10）。
复杂度分析

时间复杂度：
对每个输出位置做 C×Kh×KwC\times K_h\times K_wC×Kh×Kw 次乘加，输出一共 Hout×WoutH_{out}\times W_{out}Hout×Wout 个位置，
$$O\big(H_{out}\cdot W_{out}\cdot C\cdot K_h\cdot K_w\big)$$在题目约束下规模很小，完全可行。

空间复杂度：
需要存储填充后的输入 O(C⋅(Hin+2p)⋅(Win+2p))O(C\cdot (H_{in}+2p)\cdot (W_{in}+2p))O(C⋅(Hin+2p)⋅(Win+2p))，卷积核 O(C⋅Kh⋅Kw)O(C\cdot K_h\cdot K_w)O(C⋅Kh⋅Kw)，以及输出 O(Hout⋅Wout)O(H_{out}\cdot W_{out})O(Hout⋅Wout)。总体为
$$O\big(C(H_{in}+2p)(W_{in}+2p)+C K_h K_w+H_{out}W_{out}\big)$$同样很小，合理。

代码实现

**Python 代码：**

```python
import sys

# 多通道卷积：输入input_tensor和kernel均为三维列表
def conv_multi_channel(input_tensor, kernel, stride, padding):
C = len(input_tensor)
Hin = len(input_tensor[0])
Win = len(input_tensor[0][0])

Kh = len(kernel[0])
Kw = len(kernel[0][0])

# 计算输出尺寸
Hout = (Hin + 2 * padding - Kh) // stride + 1
Wout = (Win + 2 * padding - Kw) // stride + 1

# 1) 先做零填充
Hp = Hin + 2 * padding
Wp = Win + 2 * padding
padded = [[[0 for _ in range(Wp)] for _ in range(Hp)] for _ in range(C)]
for c in range(C):
for i in range(Hin):
for j in range(Win):
padded[c][i + padding][j + padding] = input_tensor[c][i][j]

# 2) 按定义滑动窗口并逐通道累加
output = [[0 for _ in range(Wout)] for _ in range(Hout)]
for i in range(Hout):
for j in range(Wout):
s = 0
base_i = i * stride
base_j = j * stride
for c in range(C):
for m in range(Kh):
for n in range(Kw):
s += padded[c][base_i + m][base_j + n] * kernel[c][m][n]
output[i][j] = s
return output

def main():
data = []
for line in sys.stdin:
line = line.strip()
if line:
data.extend(map(int, line.split()))
ptr = 0

# 读输入张量
C, Hin, Win = data[ptr], data[ptr+1], data[ptr+2]; ptr += 3
input_tensor = []
for _ in range(C):
ch = []
for _ in range(Hin):
row = data[ptr:ptr+Win]; ptr += Win
ch.append(row)
input_tensor.append(ch)

# 读卷积核
Ck, Kh, Kw = data[ptr], data[ptr+1], data[ptr+2]; ptr += 3
# 题目保证 Ck == C
kernel = []
for _ in range(Ck):
ch = []
for _ in range(Kh):
row = data[ptr:ptr+Kw]; ptr += Kw
ch.append(row)
kernel.append(ch)

# 读 stride 和 padding
stride, padding = data[ptr], data[ptr+1]

# 计算卷积
out = conv_multi_channel(input_tensor, kernel, stride, padding)

# 输出
for i in range(len(out)):
print(" ".join(str(x) for x in out[i]))

if __name__ == "__main__":
main()

```

---

## 2025年11月5日-AI方向

<a id="第2题-p4441"></a>

### 第2题-多目标推荐排序模型优化（P4441）- 困难





多目标学习的推荐排序模型需同时预测点击率（CTR，Click−ThroughClick-ThroughClick−Through RateRateRate）和转化率（CVR，CVR，CVR，ConversionConversionConversion RateRateRate），可采用线性回归的方式完成多目标建模，常见方法包括共享特征权重但保留任务特定偏置；在此使用联合损失函数：$\text{Loss} = \text{MSE}_{\text{CTR}} + \alpha \cdot \text{MSE}_{\text{CVR}}$ 优化共享权重和两个偏置，其中，MSE 表示标准均方误差损失，α\alphaα 表示加权系数，权重和偏置初始值从 0 开始，返回迭代 N 次后的平均联合损失值乘以 10 的 10 次方后的四舍五入结果（注意是损失值结果，而非梯度的结果）。
输入描述
第一行，输入特征集合，1,2;3,4;5,61,2;3,4;5,61,2;3,4;5,6
第二行，预测的 ctr/cvrctr/cvrctr/cvr 指标集合，0.1,0.01;0.5,0.05;0.9,0.090.1,0.01;0.5,0.05;0.9,0.090.1,0.01;0.5,0.05;0.9,0.09
第三行，迭代次数 (iterationiterationiteration)，1000
第四行，学习率，0.010.010.01
第五行，加权系数，0.50.50.5
符号解释: 分号前后隔开不同的样本，逗号隔开样本内的不同值
输出描述
输出联合损失值 ∗10* 10∗10 的 10 次方的结果为 130106913010691301069
样例1
输入
1,1,1;2,2,2;3,3,3
1,0.5;2,1.0;3,1.5
500
0.01
0.5

输出
27356237

说明
输出联合损失值 10 的 10 次方的结果为 273562372735623727356237
样例2
输入
1,2;3,4;5,6
0.1,0.01;0.5,0.05;0.9,0.09
1000
0.01
0.5

输出
1301069

说明
输出联合损失值 10 的 10 次方 的结果为 130106913010691301069


#### 解答


No testdata at current.

解题思路
本题要求在多目标推荐系统中同时预测点击率（CTR）和转化率（CVR），使用一个共享特征权重的线性模型，通过最小化联合损失函数来优化模型参数。其核心思想是利用多任务学习（Multi-Task Learning）思想，让两个任务在共享信息的同时保留自身的差异。
整体过程可以分为以下几个步骤：

模型结构设计
使用一个共享的线性层（即相同的权重向量）来提取通用特征，同时为两个任务分别设置一个独立的偏置项。

CTR 和 CVR 的预测值都由同一组权重与输入特征相乘得到，但加上不同的偏置。
这种结构能够让两个任务在共享信息的同时保留一定的灵活性。

损失函数构建
分别计算 CTR 与 CVR 的预测误差（平方误差），并按给定的权重系数 α 进行加权求和，得到联合损失值。

当 α 较大时，模型会更关注 CVR 的优化；当 α 较小时，模型更偏向优化 CTR。

参数优化方法
使用批量梯度下降法（Batch Gradient Descent）更新参数。

每次迭代计算预测值与真实值的差异，得到梯度方向。
根据学习率对权重和偏置进行反向更新。
所有参数初始值设为 0，按给定的学习率和迭代次数更新。

结果计算与输出
在完成全部迭代后，重新计算一次最终的联合损失值。

将最终损失值乘以 10¹⁰ 并进行四舍五入。
输出结果为一个整数。

关键思想总结

利用多任务共享机制，提升模型对不同目标的综合学习能力。
通过联合损失函数平衡 CTR 与 CVR 的优化。
采用标准的线性回归与梯度下降优化方式，算法逻辑简单、可解释性强。

复杂度分析
设样本数为 n，特征维度为 d，迭代次数为 N。

时间复杂度：每次迭代需要一次前向与一次反向，均为 O(nd)O(nd)O(nd)。总计 O(Nnd)O(Nnd)O(Nnd)。
空间复杂度：存储 w\mathbf{w}w 与少量中间量，O(d)O(d)O(d)；外加输入数据 O(nd)O(nd)O(nd)。整体 O(nd)O(nd)O(nd)（若仅按模型参数计则 O(d)O(d)O(d)）。

代码实现

**Python 代码：**

```python
from ast import literal_eval
from decimal import Decimal, ROUND_HALF_UP
import sys

def parse_matrix(line: str):
# 将形如 "1,2;3,4" 转换为 "[[1,2],[3,4]]" 再 literal_eval
s = '[[' + line.strip().replace(';', '],[') + ']]'
mat = literal_eval(s)
# 转为 float
return [[float(v) for v in row] for row in mat]

def train_and_loss(X, Y, iters, lr, alpha):
n = len(X)
d = len(X[0])
# 参数初始化为 0
w = [0.0] * d
b_ctr = 0.0
b_cvr = 0.0

for _ in range(iters):
# 前向计算
yhat_ctr = [sum(w[j] * X[i][j] for j in range(d)) + b_ctr for i in range(n)]
yhat_cvr = [sum(w[j] * X[i][j] for j in range(d)) + b_cvr for i in range(n)]
e_ctr = [yhat_ctr[i] - Y[i][0] for i in range(n)]
e_cvr = [yhat_cvr[i] - Y[i][1] for i in range(n)]

# 梯度计算
grad_w = [0.0] * d
for j in range(d):
s = 0.0
for i in range(n):
s += (e_ctr[i] + alpha * e_cvr[i]) * X[i][j]
grad_w[j] = (2.0 / n) * s
grad_b_ctr = (2.0 / n) * sum(e_ctr)
grad_b_cvr = alpha * (2.0 / n) * sum(e_cvr)

# 参数更新
for j in range(d):
w[j] -= lr * grad_w[j]
b_ctr -= lr * grad_b_ctr
b_cvr -= lr * grad_b_cvr

# 最终损失
yhat_ctr = [sum(w[j] * X[i][j] for j in range(d)) + b_ctr for i in range(n)]
yhat_cvr = [sum(w[j] * X[i][j] for j in range(d)) + b_cvr for i in range(n)]
mse_ctr = sum((yhat_ctr[i] - Y[i][0]) ** 2 for i in range(n)) / n
mse_cvr = sum((yhat_cvr[i] - Y[i][1]) ** 2 for i in range(n)) / n
loss = mse_ctr + alpha * mse_cvr
return loss

def main():
lines = [line.rstrip('\n') for line in sys.stdin if line.strip() != '']
fea_line = lines[0]
lbl_line = lines[1]
iters = int(lines[2])
lr = float(lines[3])
alpha = float(lines[4])

X = parse_matrix(fea_line)
Y = parse_matrix(lbl_line)  # 每个样本形如 [ctr, cvr]

loss = train_and_loss(X, Y, iters, lr, alpha)

# 四舍五入到整数（HALF_UP）
val = Decimal(str(loss)) * Decimal('10000000000')
ans = val.to_integral_value(rounding=ROUND_HALF_UP)
print(f"{ans}")

if __name__ == "__main__":
main()

```

---

<a id="第3题-p4442"></a>

### 第3题-须从规矩出方圆（P4442）- 困难





钟师傅对像素画（PixelPixelPixel Art）有独特的品味理解，他最近沉迷于用胶带拼图形贴画面海报，但是竟然后发现，像素画，每个 pixelpixelpixel 的长宽都是固定的。

一种像素圆的画法是，将每个像素块看成一个独立正方形，如果正方形和圆的交集面积大于 10−1010^{-10}10−10，则涂黑像素块。例如左图就是一个典型 $25 \times 252$ 像素圆。可以看到每个像素的行宽和列宽是相同的，它占用原图的 533/(25∗25)=0.8528=85.28%533/(25*25)=0.8528=85.28\%533/(25∗25)=0.8528=85.28% 范围。而 π/4≈78.5398%\pi/4\approx78.5398\%π/4≈78.5398% 。
钟师傅喜欢精确的圆，因此他决定设计一种新的像素屏，专门为画圆而生，每行的间距和列间距不需要一样，为此它可以画出更精确的圆。
具体来说，如果给定像素是M×MM\times MM×M，那么钟师傅需要给出M−1M-1M−1个行宽变量和M−1M-1M−1个列高变量，  记出{xi},{yi}\{x_i\},\{y_i\}{xi},{yi}
固定 x0=y0=0,xM−1=yM−1=1x_0=y_0=0,x_{M-1}=y_{M-1}=1x0=y0=0,xM−1=yM−1=1，然后以 x=xi,y=yix=x_i,y=y_ix=xi,y=yi 作 2M−22M-22M−2条直线，将(0,0)−(1,1)(0,0)-(1,1)(0,0)−(1,1)这个正方形划分成M×MM\times MM×M个小矩形。每个矩形被涂色当且仅当这个矩形和以 (0.5,0.5)(0.5,0.5)(0.5,0.5) 为圆心，半径为 0.50.50.5 的单位圆交集面积 >10−10>10^{-10}>10−10 。
例如 M=25,xi=yi=i25M=25,x_i=y_i=\frac{i}{25}M=25,xi=yi=25i ，对应的像素逼近圆就是左图。
如果更精细地设置 x 和 y 的间距，就能得到更好的逼近，目标是在 M 固定的前提下，把染色块的面积量最小化。右图是一种更优的设定方案，具体数值参考样例。
钟师傅为了找到 M=25M=25M=25 的划分已经耗尽了全部心力，他希望你帮忙找到其他 M 的最优划分方案，特别的，为了方便计算，本题的 M 都是奇数。
输入描述
一行一个整数，M，输入保证 5≤M≤2005 \le M \le 2005≤M≤200，并且 M 是奇数。
输出描述
输出最优染色面积，精确到小数点后 4 位。
例如 M=25M=25M=25 的最优答案约为 0.8107678480.8107678480.810767848 ，那么四舍五入后，你应该输出 0.81080.81080.8108
样例1
输入
5

输出
0.8784

说明
M=5M=5M=5 的最优划分见下图：

面积约为 0.87841489310.87841489310.8784148931
划分方案：
x1=y1=0.08824681693923937x_1 = y_1 = 0.08824681693923937x1=y1=0.08824681693923937
x2=y2=0.2163464855861515x_2 = y_2 = 0.2163464855861515x2=y2=0.2163464855861515
x3=y3=0.7836535144138486x_3 = y_3 = 0.7836535144138486x3=y3=0.7836535144138486
x4=y4=0.911753183060766x_4 = y_4 = 0.911753183060766x4=y4=0.911753183060766
样例2
输入
25

输出
0.8108

说明
最优分布：
x1=y1=0.014219468994025153x_1 = y_1 = 0.014219468994025153x1=y1=0.014219468994025153
x2=y2=0.032278207527408176x_2 = y_2 = 0.032278207527408176x2=y2=0.032278207527408176
x3=y3=0.053248663959798335x_3 = y_3 = 0.053248663959798335x3=y3=0.053248663959798335
x4=y4=0.07678775198872612x_4 = y_4 = 0.07678775198872612x4=y4=0.07678775198872612
x5=y5=0.10277889747804814x_5 = y_5 = 0.10277889747804814x5=y5=0.10277889747804814
x6=y6=0.1312450507717095x_6 = y_6 = 0.1312450507717095x6=y6=0.1312450507717095
x7=y7=0.1623318383092049x_7 = y_7 = 0.1623318383092049x7=y7=0.1623318383092049
x8=y8=0.1963301205070792x_8 = y_8 = 0.1963301205070792x8=y8=0.1963301205070792
x9=y9=0.23374562325992837x_9 = y_9 = 0.23374562325992837x9=y9=0.23374562325992837
x10=y10=0.27547112225909104x_{10} = y_{10} = 0.27547112225909104x10=y10=0.27547112225909104
x11=y11=0.3232619881117089x_{11} = y_{11} = 0.3232619881117089x11=y11=0.3232619881117089
x12=y12=0.381605423707194x_{12} = y_{12} = 0.381605423707194x12=y12=0.381605423707194
x13=y13=0.618394576292806x_{13} = y_{13} = 0.618394576292806x13=y13=0.618394576292806
x14=y14=0.6767380118882911x_{14} = y_{14} = 0.6767380118882911x14=y14=0.6767380118882911
x15=y15=0.7245288777409089x_{15} = y_{15} = 0.7245288777409089x15=y15=0.7245288777409089
x16=y16=0.7662543767400716x_{16} = y_{16} = 0.7662543767400716x16=y16=0.7662543767400716
x17=y17=0.8036698794929208x_{17} = y_{17} = 0.8036698794929208x17=y17=0.8036698794929208
x18=y18=0.8376681616690795x_{18} = y_{18} = 0.8376681616690795x18=y18=0.8376681616690795
x19=y19=0.8687549492282904x_{19} = y_{19} = 0.8687549492282904x19=y19=0.8687549492282904
x20=y20=0.897221105219519x_{20} = y_{20} = 0.897221105219519x20=y20=0.897221105219519
x21=y21=0.9232122480112739x_{21} = y_{21} = 0.9232122480112739x21=y21=0.9232122480112739
x22=y22=0.94675136660420166x_{22} = y_{22} = 0.94675136660420166x22=y22=0.94675136660420166
x23=y23=0.9677217924725918x_{23} = y_{23} = 0.9677217924725918x23=y23=0.9677217924725918
x24=y24=0.9857805310059748x_{24} = y_{24} = 0.9857805310059748x24=y24=0.9857805310059748
提示

如果确定了所有 yiy_iyi 的值，那么 xix_ixi 的值也可以确定。考虑我们已经确定最优解需要用到轴线 y=ay=ay=a ，只要 a 不等于 0.50.50.5 ，那么 y=ay=ay=a 与圆会有两个交点 (b,a)(b,a)(b,a) 和 (c,a)(c,a)(c,a) ，那么最优解一定有轴线 x=bx=bx=b 和 x=cx=cx=c ，否则我们可以通过调整，得到面积更小的染色方案。即，可以通过 {yi}\{y_i\}{yi} 确定潜在的 {xi}\{x_i\}{xi} 。

这样我们可以列出式子，把最小化 lossXY({xi},{yi})lossXY(\{x_i\},\{y_i\})lossXY({xi},{yi}) 变成 lossY({yi})lossY(\{y_i\})lossY({yi}) 的问题。而且 lossYlossYlossY 会有比较容易求导。

可以考虑用某些编程语言自带的优化求解器（例如 numpynumpynumpy），或者自己实现梯度下降等优化方式，完成最优解求解。


#### 解答


题解思路
1. 对称化 + 角度参数化
圆心在 (12,12)(\tfrac12,\tfrac12)(21,21)，半径 r=12r=\tfrac12r=21。最优网格一定关于水平 [0,r]×[0,r]\times[0,r]×[0,r] 里描述边界，再用对称扩展到整圆。
设我们在第一象限里取若干条水平线 y=rsin⁡θy=r\sin\thetay=rsinθ 与竖直线 x=rcos⁡θx=r\cos\thetax=rcosθ（θ∈(0,π4)\theta\in(0,\tfrac\pi4)θ∈(0,4π)）。给定一组严格递增的角度
$$0<\theta_1<\theta_2<\cdots<\theta_p<\frac{\pi}{4},$$就得到一组“关键高度/宽度”
$$U=\bigl[r\sin\theta_1,\ldots,r\sin\theta_p\bigr]
\quad(+\ r/\sqrt2\text{ 若 }k\text{ 为奇})\quad
\cup\ \bigl[r\cos\theta_p,\ldots,r\cos\theta_1\bigr],$$并在两端补上 0 与 r。这里

M 为像素边长，令 k=M−12k=\frac{M-1}{2}k=2M−1（第一象限中除去边界的“层数”），
p=⌊k2⌋p=\left\lfloor \frac{k}{2}\right\rfloorp=⌊2k⌋ 是需要优化的角度个数；若 k 为奇，还需在中间插入 r/2r/\sqrt2r/2（对应主对角线）。

关键事实（题面提示 1）： 若最优解使用了水平线 y=a≠r/2y=a\ne r/\sqrt2y=a=r/2，则必有与圆的两个交点 (b,a)(b,a)(b,a), (c,a)(c,a)(c,a) 的竖线 x=b,x=cx=b,x=cx=b,x=c 同时出现，否则可微调使面积更小。因此用一组 {θi}\{\theta_i\}{θi} 就能同时确定全部水平/竖直分割线。
2. 面积公式（四倍化）
把上述 U 排好序，并在首尾补上 0 与 r，得到 Ufull=[0,U,r]U_\text{full}=[0, U, r]Ufull=[0,U,r]。
第一象限被分成 K+1K+1K+1 条水平条带（K=∣U∣K=|U|K=∣U∣），第 i 条条带高为
Δyi=Ufull[i]−Ufull[i−1],\Delta y_i=U_\text{full}[i]-U_\text{full}[i-1],
Δyi=Ufull[i]−Ufull[i−1],
对应可被“染色”的横向宽度上界等于和它关于主对角线镜像位置的“x-截距”，也就是
Xi=Ufull[K−i+2].X_i = U_\text{full}[K-i+2].
Xi=Ufull[K−i+2].
于是第一象限的像素覆盖面积为
S1/4=∑i=1K+1Δyi⋅Xi,S_{1/4}=\sum_{i=1}^{K+1}\Delta y_i\cdot X_i,
S1/4=i=1∑K+1Δyi⋅Xi,
全图面积为 S=4S1/4S=4S_{1/4}S=4S1/4。
3. 优化：坐标下降 + 1D 黄金分割
把目标写成 S(θ1,…,θp)S(\theta_1,\ldots,\theta_p)S(θ1,…,θp)。我们采用坐标下降：依次固定其他角，只对 θi\theta_iθi 在合法区间 (θi−1,θi+1)(\theta_{i-1},\theta_{i+1})(θi−1,θi+1) 内做一维极小化。
一维极小化使用黄金分割搜索。反复扫若干轮，若一轮没有任何角更新则停止。

初值：θi=i+1p+1⋅π4\theta_i=\dfrac{i+1}{p+1}\cdot \dfrac{\pi}{4}θi=p+1i+1⋅4π 均匀分布。
收敛：通常数十轮内稳定，对 M≤200M\le200M≤200 足够。

4. 输出
用四舍五入到 4 位小数（HALF_UP） 输出最小面积。
复杂度

每次评估 total_area 为 O(K)=O(M)O(K)=O(M)O(K)=O(M)。
每轮对每个 θi\theta_iθi 做常数次函数评估（黄金分割 < ⁣120<\!120<120 次）。
总复杂度约 O(迭代轮数⋅p⋅120⋅M)O(\text{迭代轮数}\cdot p \cdot 120 \cdot M)O(迭代轮数⋅p⋅120⋅M)，对 M≤200M\le 200M≤200 运行很快。


**Python 代码：**

```python
import sys
import math
from decimal import Decimal, ROUND_HALF_UP

def build_U_from_thetas(thetas, r, k):

U = [r * math.sin(t) for t in thetas]
if k % 2 == 1:
U.append(r / math.sqrt(2.0))
U += [r * math.cos(t) for t in reversed(thetas)]
return [0.0] + U + [r]

def total_area(thetas, r, k):
U_full = build_U_from_thetas(thetas, r, k)
K = len(U_full) - 2  # because U_full = [0] + U + [r]
s = 0.0
# i in [1, K+1]
for i in range(1, K + 2):
dy = U_full[i] - U_full[i - 1]
x_cap = U_full[K - i + 2]
s += dy * x_cap
return 4.0 * s

def golden_section_minimize(f, lo, hi, max_it=120, tol=1e-13):
golden = (math.sqrt(5.0) - 1.0) / 2.0
a, b = lo, hi
c = b - golden * (b - a)
d = a + golden * (b - a)
fc, fd = f(c), f(d)
it = 0

while (b - a) > tol and it < max_it:
if fc > fd:
a, c, fc = c, d, fd
d = a + golden * (b - a)
fd = f(d)
else:
b, d, fd = d, c, fc
c = b - golden * (b - a)
fc = f(c)
it += 1

return (a + b) / 2.0

def optimize_thetas(M):
r = 0.5
k = (M - 1) // 2
p = k // 2

if p == 0:
return total_area([], r, k)

thetas = [ (i + 1) * (math.pi / 4.0) / (p + 1) for i in range(p) ]

# outer coordinate-descent rounds
for _ in range(30):
changed = False
for i in range(p):
lo = thetas[i - 1] + 1e-12 if i > 0 else 1e-12
hi = thetas[i + 1] - 1e-12 if i < p - 1 else (math.pi / 4.0) - 1e-12

def f(x):
tmp = list(thetas)
tmp[i] = x
return total_area(tmp, r, k)

new_theta = golden_section_minimize(f, lo, hi, max_it=120, tol=1e-13)
if abs(new_theta - thetas[i]) > 1e-15:
thetas[i] = new_theta
changed = True

if not changed:
break

return total_area(thetas, r, k)

def main():
M = int(sys.stdin.readline().strip())
ans = optimize_thetas(M)
out = Decimal(str(ans)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
print(out)

if __name__ == "__main__":
main()

```

---

## 2025年10月29日-AI方向

<a id="第2题-p4343"></a>

### 第2题-实体匹配结果合并问题（P4343）- 中等





某业务部门有多个数据来源，现在需要对多个来源的实体数据进行去重、消歧、合并。有多个实体匹配系统（假设系统的匹配结果完全正确），每个系统从不同角度进行匹配，匹配结果是相同实体列表。
这些匹配结果中往往存在交叉重复的问题，需要对所有匹配结果进行合并去重。例如系统 A 的匹配结果是 ["1", "2"] ，系统 B 的匹配结果是 ["2", "3"]，那么合并后的匹配结果是 ["1", "2", "3"]。请你按照上述逻辑，编写代码实现对匹配结果的合并去重。
输入描述

第一行输入是整数 N，表示有 N 个实体匹配系统，1<=N<=100001<=N<=100001<=N<=10000 ；

接下来 N 行是每个实体匹配系统的匹配结果（相同实体列表，每行实体数量不超过 100 ）；实体使用数字字符串表示（字符长度不超过 10），实体之间通过空格分开。实体种类数量不超过 100000100000100000 。

输出描述
输出 M 行（M<=NM <= NM<=N），每行内容是合并后的匹配结果。
注意，每行的输出结果需要按照字典顺序进行排序；合并后的匹配结果列表之间，也需要按字典顺序进行排序。
样例1
输入
5
1 2 3
4 5
11 22
33 44 55 1
3 66

输出
1 2 3 33 44 55 66
11 22
4 5

说明
匹配结果 "1 2 3 "、"33 44 55 1"、"3 66"，存在重复实体 "1" 和 "3"，故可以合并，合并后按字典序为 "1 2 3 33  44 55 66"；
另外两组结果与其他组不存在重复；
合并后的匹配结果列表之间，按字典序排序后输出，即样例输出所示内容。
样例2
输入
2
1 2
2 3

输出
1 2 3

说明
有两组匹配结果，即 "1 2"和"2 3"，存在重复实体 "2"，故可以合并为 "1 2 3"。


#### 解答


解题思路
题意要求将多个实体匹配系统输出的实体结果进行去重、合并与排序。
每个系统的输出可视为一个集合，若两个集合存在交集，则它们应被合并为一个更大的集合。
这实际上是一个并查集（Union-Find） 的典型应用场景：

每个实体可以看作一个节点；
若两个实体在同一系统结果中出现，则它们属于同一连通分量；
通过并查集合并所有出现在同一行的实体；
最后，将所有节点根据其根节点分组，输出每个连通分量（集合）内的实体。

实现步骤

读取输入的系统数 N；

对每一行实体结果：

将该行所有实体放入列表；
使用并查集，将该行所有实体合并到同一集合中；

最终遍历所有实体，按根节点进行归类；

对每个集合内的实体进行字典序排序；

对所有集合按字典序整体排序；

输出结果。

复杂度分析

时间复杂度：
每次合并操作近似为常数时间（路径压缩优化），总复杂度约为 O(T * α(N))，其中 T 为所有实体出现次数（≤100000），α(N) 为阿克曼函数，几乎可视为常数。
排序部分复杂度为 O(K log K)，K 为最终集合内的元素数。
综合复杂度为 O(N log N)。

空间复杂度：
存储并查集及映射关系，约为 O(N)，符合题目要求。

代码实现

**Python 代码：**

```python
# 并查集实现类
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
# 初始化所有实体
for line in systems:
for entity in line:
if entity not in uf.parent:
uf.parent[entity] = entity

# 按每行进行合并
for line in systems:
base = line[0]
for entity in line[1:]:
uf.union(base, entity)

# 收集每个集合
groups = {}
for entity in uf.parent:
root = uf.find(entity)
groups.setdefault(root, set()).add(entity)

# 按题意要求排序
result = []
for group in groups.values():
sorted_group = sorted(group)
result.append(sorted_group)
result.sort()

return result

if __name__ == "__main__":
n = int(input())
systems = []
for _ in range(n):
line = input().strip().split()
if line:
systems.append(line)

merged = merge_entities(n, systems)
for group in merged:
print(" ".join(group))

```

---

<a id="第3题-p4344"></a>

### 第3题-商品购买预测（P4344）- 中等





实现一个二分类逻辑回归模型，用于预测用户是否会购买某商品（1 表示购买，0 表示不购买）。已知用户特征包括年龄（岁）、月收入（千元）和浏览时长（分钟），需通过这些特征建立预测模型。
我们构建一个逻辑回归模型来预测购买：ypred=sigmoid(wx+b)y_{pred} = sigmoid(wx + b)ypred=sigmoid(wx+b)，
w 和 b 分别为特征权重和偏置。
要求：

基于训练数据，使用梯度下降法训练逻辑回归模型参数。要求采用的具体函数方法为：损失函数为预测结果与标签值的交叉熵

$L = \frac{1}{n} \sum_{i=1}^{n} CrossEntropy(y_i^{pred}, y_i^{label})$，使用 L2L2L2 正则约束权重 L2(w)=∣∣w∣∣2L_2(w) = ||w||_2L2(w)=∣∣w∣∣2，激活函数为 SigmoidSigmoidSigmoid 。
训练终止条件：迭代次数达到最大次数（max_itermax\_itermax_iter）或损失函数变化量小于阈值（tol）。
对测试数据进行预测，输出预测结果（概率 ≥0.5≥0.5≥0.5 预测为 1，否则为 0），和对应的概率值，四舍五入保留四位小数。
输入描述
第一行：5 个数字，分别为训练样本数 n 、最大迭代次数 max_itermax\_itermax_iter、学习率 ααα（浮点型）、正则化系数 λλλ（浮点型）、损失阈值 tol（浮点型）。
接下来 n 行：每行 4 个数值，前 3 个为特征（年龄、月收入、浏览时长），第 4 个为标签（0 或 1）。
第 n+2n+2n+2 行：整数 m（测试样本数）。
接下来 m 行：每行 3 个数值，为测试特征。
输出描述
m 行，每行 1 个整数（ 0 或 1），表示预测结果；接着是一个空格；紧接着输出对应的商品购买小数概率值（保留四位小数）
样例1
输入
10 1000 0.01 0.1 0.0001
25 8 5 0
30 15 15 1
35 20 10 0
40 25 20 1
45 30 25 1
50 35 18 1
55 40 12 0
60 45 8 0
65 50 5 0
70 55 30 1
3
32 18 12
48 33 22
62 48 10

输出
1 0.7539
1 0.9966
0 0.0004

说明
对于第一个样例，年龄（32 岁）、月收入（18 千元）和浏览时长（12 分钟），输出结果 1 表示该用户会购买该商品，0.75390.75390.7539 表示用户购买该商品的概率是 75.3975.3975.39%。
样例2
输入
10 1000 0.01 0.1 0.0001
25 8 5 0
30 15 15 1
35 20 10 0
40 25 20 1
45 30 25 1
50 35 18 1
55 40 12 0
60 45 8 0
65 50 5 0
70 55 30 1
1
48 30 10

输出
0 0.0081

说明
对于第一个样例，年龄（48 岁）、月收入（30 千元）和浏览时长（10 分钟），输出结果 0 表示该用户不会购买该商品，0.00810.00810.0081 表示该用户购买该商品的概率是 0.810.810.81%。


#### 解答


解题思路
本题要求用二分类逻辑回归（Logistic Regression）对“是否购买”进行预测。模型形式为
y^=σ(z)=σ(w⊤x+b)\hat y = \sigma(z)=\sigma(w^\top x + b)y^=σ(z)=σ(w⊤x+b)，其中 σ(t)=11+e−t\sigma(t)=\frac{1}{1+e^{-t}}σ(t)=1+e−t1 为 Sigmoid。
优化目标（带 L2 正则的交叉熵）：
对 n 个样本，特征维度为 d=3d=3d=3（年龄、月收入、浏览时长），标签 y∈{0,1}y\in\{0,1\}y∈{0,1}。
损失函数取平均交叉熵并加入 L2 正则（采用更常见的平方范数形式）：
$$J(w,b)=\frac{1}{n}\sum_{i=1}^n\Big[-y_i\log \hat y_i-(1-y_i)\log(1-\hat y_i)\Big]+\frac{\lambda}{2n}\|w\|_2^2$$其中 y^i=σ(w⊤xi+b)\hat y_i=\sigma(w^\top x_i+b)y^i=σ(w⊤xi+b)。
梯度推导： 记 pi=y^ip_i=\hat y_ipi=y^i，则
$$\frac{\partial J}{\partial w}=\frac{1}{n}\sum_{i=1}^n (p_i-y_i)\,x_i+\frac{\lambda}{n}w,\qquad
\frac{\partial J}{\partial b}=\frac{1}{n}\sum_{i=1}^n (p_i-y_i).$$训练（批量梯度下降）：

初始化 w=0,b=0w=\mathbf{0}, b=0w=0,b=0；
迭代更新
w←w−α ∂J/∂ww \leftarrow w-\alpha\,\partial J/\partial ww←w−α∂J/∂w，
b←b−α ∂J/∂bb \leftarrow b-\alpha\,\partial J/\partial bb←b−α∂J/∂b；
终止条件：达到最大迭代次数 max_iter或 先更新参数，再计算一次新损失，用相邻两次损失之差判断是否 < tol；
数值稳定性：交叉熵内的 log⁡\loglog 对概率做 ε\varepsilonε 截断（如 1e−151e{-15}1e−15），Sigmoid 采用分段公式避免上溢/下溢。

预测：

概率 p=σ(w⊤x+b)p=\sigma(w^\top x+b)p=σ(w⊤x+b)；
阈值 0.5：p≥0.5p\ge 0.5p≥0.5 判为 1，否则 0；
输出格式为“预测结果 概率（四位小数）”。

复杂度分析

时间复杂度：每次迭代需扫一遍数据并做 d 次累计，故为 O(max_iter×n×d)O(\text{max\_iter}\times n \times d)O(max_iter×n×d)。本题 d=3d=3d=3，复杂度适宜。
空间复杂度：存储数据与参数，O(n×d)+O(d)O(n\times d)+O(d)O(n×d)+O(d)。参数仅 O(d)O(d)O(d)；数据为输入必须项。

代码实现

**Python 代码：**

```python
import sys
import math

# Sigmoid 函数，数值稳定写法
def sigmoid(z):
if z >= 0:
ez = math.exp(-z)
return 1.0 / (1.0 + ez)
else:
ez = math.exp(z)
return ez / (1.0 + ez)

# 计算带 L2 正则（平方范数）的平均交叉熵损失及其梯度
def compute_loss_and_grad(X, y, w, b, lam):
n = len(X)
d = len(w)
eps = 1e-15

loss = 0.0
grad_w = [0.0] * d
grad_b = 0.0

# 累加交叉熵与梯度
for i in range(n):
z = b
for j in range(d):
z += w[j] * X[i][j]
p = sigmoid(z)
yi = y[i]
# 交叉熵
loss += -(yi * math.log(max(p, eps)) + (1 - yi) * math.log(max(1 - p, eps)))
# 梯度累加
diff = p - yi
for j in range(d):
grad_w[j] += diff * X[i][j]
grad_b += diff

# 取平均
loss /= n
for j in range(d):
grad_w[j] /= n
grad_b /= n

# L2 正则项：lambda/(2n)*||w||^2，并相应修正梯度
l2 = 0.0
for j in range(d):
l2 += w[j] * w[j]
grad_w[j] += (lam / n) * w[j]
loss += (lam / (2 * n)) * l2

return loss, grad_w, grad_b

# 训练逻辑回归（批量梯度下降）
def train_logreg(X, y, max_iter, alpha, lam, tol):
d = len(X[0])
w = [0.0] * d
b = 0.0

# 先算一次初始损失
loss, _, _ = compute_loss_and_grad(X, y, w, b, lam)

for _ in range(max_iter):
# 基于当前参数计算梯度并更新
_, grad_w, grad_b = compute_loss_and_grad(X, y, w, b, lam)
for j in range(d):
w[j] -= alpha * grad_w[j]
b -= alpha * grad_b

# 更新后再计算一次新损失，用于提前停止判断
new_loss, _, _ = compute_loss_and_grad(X, y, w, b, lam)
if abs(loss - new_loss) < tol:
break
loss = new_loss

return w, b

# 单样本预测
def predict_one(x, w, b):
z = b
for j in range(len(w)):
z += w[j] * x[j]
p = sigmoid(z)
label = 1 if p >= 0.5 else 0
return label, p

def main():
data = sys.stdin.read().strip().split()
it = iter(data)

# 读取第一行
n = int(next(it))
max_iter = int(next(it))
alpha = float(next(it))
lam = float(next(it))
tol = float(next(it))

# 训练数据
X = []
y = []
for _ in range(n):
a = float(next(it))
inc = float(next(it))
dur = float(next(it))
lab = int(next(it))
X.append([a, inc, dur])
y.append(lab)

# 测试数据
m = int(next(it))
test = []
for _ in range(m):
a = float(next(it)); inc = float(next(it)); dur = float(next(it))
test.append([a, inc, dur])

# 训练
w, b = train_logreg(X, y, max_iter, alpha, lam, tol)

# 预测与输出
for x in test:
lab, p = predict_one(x, w, b)
print(f"{lab} {p:.4f}")

if __name__ == "__main__":
main()

```

---

## 2025年10月23日-AI方向(留学生)

<a id="第2题-p4277"></a>

### 第2题-人脸关键点对齐（P4277）- 简单





人脸关键点对齐是人脸识别算法过程中非常重要的一步，其方法是基于检测人脸关键点及模板人脸关键点获得变换矩阵 M ，使得最小二乘意义下把原图的关键点贴到模板关键点位置，其基本原理是对图像得仿射变换。现在你将实现一个图像的仿射变换函数，该函数接收一个二维图像矩阵 A 、一个变换矩阵 M 和输出图像的尺寸 0 ，返回变换后的图像。
变换公式为，其中 x,yx,yx,y 为原坐标， x′,y′x',y'x′,y′ 为变换后的坐标， a,b,c,da,b,c,da,b,c,d 是线性变换部分的系数， tx,tyt_x,t_ytx,ty 是平移向量:
x′=a×x+b×y+txx'=a×x+b×y+t_xx′=a×x+b×y+tx
y′=c×x+d×y+tyy'=c×x+d×y+t_yy′=c×x+d×y+ty
如果变换后的坐标超出原图像范围，则不赋值(保留为 0 )。
输入描述
输入图像 A :一个二维列表，表示输入图像，每个元素是一个像素值。
交换矩阵 M :一个二维列表，格式如下:
[[a,b,tx],[c,d,ty]][[a, b, t_ x], [c,d,t_y]][[a,b,tx],[c,d,ty]]
输出图像的尺寸 height,widthheight, widthheight,width
输入第一行分别为 A、MA、MA、M 列表的长度 a,ma,ma,m 以及输出列表 O 所占用的行，接下来的 a 行为输入图像 A ，然后 m 行是输入变换矩阵，最后一行是输出图像的大小
输出描述
返回一个二维列表，表示变换后的图像
样例1
输入
3 2 1
10 20 30
40 50 60
70 80 90
0 1 0
-1 0 2
3 3

输出
30 60 90 20 50 80 10 40 70

说明
第一行 3 2 1 表示:
从第二行起接下来的三行是图像 A 的输入，然后下面两行是变换矩阵的输入，最后一行是输出图像的高度及宽度
输出图像矩阵为 [[30,60,90],[20,50,80],[10,40,70][[30,60,90],[20,50,80],[10,40,70][[30,60,90],[20,50,80],[10,40,70] ,
展开后最终的输出为 30 60 90 20 50 80 10 40 70
样例2
输入
3 2 1
10 20 30
40 50 60
70 80 90
-1 0 2
0 1 0
3 4

输出
30 20 10 0 60 50 40 0 90 80 70 0

提示
1.如果变换矩阵的线性部分 (a,b,c,d)(a,b,c,d)(a,b,c,d) 不可逆，则返回一个全 0 的图像


#### 解答


解题思路

算法：对给定图像做二维仿射变换。仿射模型为
x′=ax+by+tx,y′=cx+dy+tyx' = a x + b y + t_x,\quad y' = c x + d y + t_yx′=ax+by+tx,y′=cx+dy+ty。
为避免“空洞”，采用逆映射 + 最近邻插值：对每个输出像素 (x′,y′)(x',y')(x′,y′)，先减去平移，再乘线性部分的逆矩阵，得到源坐标 (x,y)(x,y)(x,y)。若 (x,y)(x,y)(x,y) 在原图范围内，则取最近邻像素；否则置 0。

核心实现：

从矩阵 M=[abtxcdty]M=\begin{bmatrix}a&b&t_x\\c&d&t_y\end{bmatrix}M=[acbdtxty] 取出 a,b,c,d,tx,tya,b,c,d,t_x,t_ya,b,c,d,tx,ty。
计算线性部分的逆:

枚举输出图像尺寸 H×WH\times WH×W 的每个像素 (x′,y′)(x',y')(x′,y′)，

取 (round(x),round(y))(\text{round}(x),\text{round}(y))(round(x),round(y)) 作为最近邻位置，越界则填 0。

坐标约定：x 为列、y 为行，左上角为 (0,0)(0,0)(0,0)。

输入输出：
第一行给出行数：a m o_lines。接着 a 行是图像 A；随后 m 行是 2×3 的变换矩阵 M；最后一行是输出尺寸 H W。
输出按行优先展平成一行空格分隔的数列。

复杂度分析

时间复杂度：枚举全部输出像素，O(H×W)O(H \times W)O(H×W)。
空间复杂度：保存输出图像，O(H×W)O(H \times W)O(H×W)（除输入外）。

代码实现

**Python 代码：**

```python
# 题面功能封装在函数里；主函数做输入输出（ACM 风格）

from typing import List
import sys
import math

def affine_transform(A: List[List[int]], M: List[List[float]], H: int, W: int) -> List[List[int]]:
# 解析仿射参数
a, b, tx = M[0]
c, d, ty = M[1]
# 线性部分行列式
det = a * d - b * c
hA, wA = len(A), len(A[0]) if A else 0
# 输出初始化为 0
O = [[0 for _ in range(W)] for _ in range(H)]
if abs(det) < 1e-12 or hA == 0 or wA == 0:
return O

# 预计算逆矩阵
inv00 =  d / det
inv01 = -b / det
inv10 = -c / det
inv11 =  a / det

for y2 in range(H):          # y'
for x2 in range(W):      # x'
# 去掉平移再乘逆矩阵 -> 源坐标 (x, y)
dx = x2 - tx
dy = y2 - ty
x = inv00 * dx + inv01 * dy
y = inv10 * dx + inv11 * dy
xi = int(round(x))
yi = int(round(y))
if 0 <= yi < hA and 0 <= xi < wA:
O[y2][x2] = A[yi][xi]
return O

def main():
data = sys.stdin.read().strip().split()
it = iter(data)
a = int(next(it)); m = int(next(it)); _ = int(next(it))  # O 占 1 行
A = []
if __name__ == "__main__":
import sys

lines = sys.stdin.read().strip().splitlines()
if not lines:
sys.exit(0)
a, m, _ = map(int, lines[0].split())
idx = 1
A = [list(map(int, lines[idx+i].split())) for i in range(a)]
idx += a
M = [list(map(float, lines[idx].split())), list(map(float, lines[idx+1].split()))]
idx += m
H, W = map(int, lines[idx].split())

O = affine_transform(A, M, H, W)
# 按行优先展平输出
out = []
for r in O:
out.extend(map(str, r))
print(" ".join(out))

```

---

<a id="第3题-p4278"></a>

### 第3题-卷积结构实现（P4278）- 中等





卷积神经网络 (CNN)(CNN)(CNN) 是计算机视觉领域的核心模型，ResNedResNedResNed 通过残差连接 (Residual(Residual(Residual Connection)Connection)Connection) 进一步解决了深层神经网络梯度消失的问题，本题要求实现 CNN 基础的卷积函数 $Conv2D(input, weight, bias, stride,padding, dilation)$，相关参数描述如下:
inputinputinput:输入数据;
weightweightweight:卷积核的权重;
biasbiasbias:卷积核的偏置
stridestridestride:卷积核的移动步长;
paddingpaddingpadding:输入数据边缘填充的像素数(填充 0 );
dilationdilationdilation:卷积核元素之间的间隔;
输入描述
第 1 行:输入数据的形状 c,x,yc,x,yc,x,y，以空格隔开
第 2 行:输入数据，为 c∗x∗yc*x*yc∗x∗y 个实数，按照先行后列排序。
第 3 行:卷积核的形状 out,in,k,kout,in,k,kout,in,k,k
第 4 行:卷积的权重，数量为 out∗in∗k∗kout*in*k*kout∗in∗k∗k ，按照先行后列排序
第 5 行: bias,stride,padding,dilationbias, stride, padding, dilationbias,stride,padding,dilation
第 6 行:若 biasbiasbias 为 1 ，则该行为 biasbiasbias 的具体值，长度为 out ，否则该行为空
其中 0<x,y<1000,0<k<1000<x,y<1000,0<k<1000<x,y<1000,0<k<100
输出描述
卷积的计算结果，输出为一行，保留 4 位小数，不足四位小数补 0
样例1
输入
1 4 4
1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0
1 1 3 3
1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
0 1 0 1

输出
54.0000 63.0000 90.0000 99.0000

说明
输入的形状为 (1.4,4)(1.4,4)(1.4,4) ，故第二行的数据 1.01.01.0 2.02.02.0 3.03.03.0 4.04.04.0 5.05.05.0 6.06.06.0 7.07.07.0 8.08.08.0 9.09.09.0 10.010.010.0 11.011.011.0 12.012.012.0 13.013.013.0 14.014.014.0 15.015.015.0 16.016.016.0 的数据排列方式为:
[[[1.0,2.0,3.0,4.0],[[[1.0, 2.0, 3.0, 4.0],[[[1.0,2.0,3.0,4.0],
[5.0.6.0,7.0,8.0],[5.0.6.0,7.0,8.0],[5.0.6.0,7.0,8.0],
[9.0,10.0,11.0,12.0]
[13.0,14.0,15.0,16.0]]][13.0,14.0,15.0,16.0]]][13.0,14.0,15.0,16.0]]]
卷积计算结果为:
[[[54.0000,63.0000],[[[54.0000, 63.0000],[[[54.0000,63.0000],
[90.0000,99.0000]
输出为:54.000054.000054.0000 63.000063.000063.0000 90.000090.000090.0000 99.000099.000099.0000
样例2
输入
1 4 4
1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0
1 1 3 3
1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
1 1 0 1
1.0

输出
55.0000 64.0000 91.0000 100.0000


#### 解答


解题思路
本题需要手写二维卷积 Conv2D(input, weight, bias, stride, padding, dilation)，支持步幅、零填充与空洞（扩张）卷积。
设输入形状为 (C, H, W)，卷积核形状为 (Out, In, K, K)。

输出尺寸计算
令有效核尺寸 K_eff = dilation * (K - 1) + 1
H_out = floor((H + 2*padding - K_eff) / stride) + 1
W_out = floor((W + 2*padding - K_eff) / stride) + 1

卷积计算
对每个输出通道 oc、输出位置 (oh, ow)：
y[oc, oh, ow] = (bias[oc] if 有偏置 else 0)
+ Σ_{ic=0..In-1} Σ_{kh=0..K-1} Σ_{kw=0..K-1}
x[ic, ih, iw] * w[oc, ic, kh, kw]
其中：
ih = oh*stride + kh*dilation - padding
iw = ow*stride + kw*dilation - padding
超出边界的 (ih, iw) 视为 0（零填充）。

读写顺序

输入与权重均按“先行后列”展开；多维时采用 通道优先 再行、再列，即：

input 展开顺序：ic -> ih -> iw
weight 展开顺序：oc -> ic -> kh -> kw

输出打印顺序：oc -> oh -> ow，全部保留 4 位小数，以空格分隔。

复杂度分析

时间复杂度：O(Out * In * H_out * W_out * K * K)
空间复杂度：O(C*H*W + Out*In*K*K + Out*H_out*W_out)（主要为存储输入、权重与结果），额外辅助空间为 O(1)。

代码实现

**Python 代码：**

```python
# -*- coding: utf-8 -*-
# 题面要求：主函数内做输入输出，功能在外部函数；ACM 风格；中文注释

import sys

def conv2d(input_arr, C, H, W, weight_arr, Out, InC, K, bias_arr, has_bias, stride, padding, dilation):
# 计算输出尺寸
K_eff = dilation * (K - 1) + 1
H_out = (H + 2 * padding - K_eff) // stride + 1
W_out = (W + 2 * padding - K_eff) // stride + 1

# 索引函数（行优先）
def idx_input(ic, ih, iw):
return ic * (H * W) + ih * W + iw

def idx_weight(oc, ic, kh, kw):
return (((oc * InC + ic) * K + kh) * K + kw)

# 结果数组（扁平存储：oc -> oh -> ow）
out_size = Out * H_out * W_out
out_arr = [0.0] * out_size

# 卷积主循环
for oc in range(Out):
b = bias_arr[oc] if has_bias else 0.0
for oh in range(H_out):
for ow in range(W_out):
s = b
base_h = oh * stride - padding
base_w = ow * stride - padding
for ic in range(InC):
for kh in range(K):
ih = base_h + kh * dilation
if ih < 0 or ih >= H:
continue
for kw in range(K):
iw = base_w + kw * dilation
if iw < 0 or iw >= W:
continue
s += input_arr[idx_input(ic, ih, iw)] * weight_arr[idx_weight(oc, ic, kh, kw)]
out_index = (oc * H_out + oh) * W_out + ow
out_arr[out_index] = s
return out_arr, H_out, W_out

def main():
data = sys.stdin.read().strip().split()
it = iter(data)

# 读取输入形状 C H W
C = int(next(it)); H = int(next(it)); W = int(next(it))
# 读取输入数据（行优先，通道优先）
input_cnt = C * H * W
input_arr = [float(next(it)) for _ in range(input_cnt)]

# 读取卷积核形状 Out In K K
Out = int(next(it)); InC = int(next(it)); K1 = int(next(it)); K2 = int(next(it))
K = K1  # 题目保证为方核

# 读取权重
weight_cnt = Out * InC * K * K
weight_arr = [float(next(it)) for _ in range(weight_cnt)]

# 读取 bias 标志、stride、padding、dilation
has_bias_flag = int(next(it))
stride = int(next(it)); padding = int(next(it)); dilation = int(next(it))

# 读取 bias
if has_bias_flag == 1:
bias_arr = [float(next(it)) for _ in range(Out)]
has_bias = True
else:
bias_arr = [0.0] * Out
has_bias = False

# 计算卷积
out_arr, H_out, W_out = conv2d(input_arr, C, H, W, weight_arr, Out, InC, K, bias_arr, has_bias, stride, padding, dilation)

# 输出一行，四位小数
res = ["{:.4f}".format(v) for v in out_arr]
print(" ".join(res))

if __name__ == "__main__":
main()

```

---

## 2025年10月22日-AI方向

<a id="第2题-p4274"></a>

### 第2题-最大能量路径（P4274）- 中等





在自动驾驶系统中，车道线识别是核心功能之一。车道线通常具有连续性，从图像左侧到右侧逐渐展开。
为了识别出最可能的车道线路径，我们可以在图像中找到一条路径，使得路径上所有像素的信号值与策略矩阵的乘积之和最大。
现定义每个位置的能量值为策略矩阵与该位置周边信号值的乘积和。
给定一个 H×WH×WH×W 的图像以及一个 K×KK×KK×K 的策略矩阵，用于模拟不同方向的路径选择策略。
你需要从图像的第一列任意像素出发，走到最后一列任意像素，每一步只能向右、右上、右下移动一格。
在行进的过程中，需要实时的收集能量值，请找到一条路径，使得路径上的能量值之和最大。
输入描述
第一行输入 H W K K ，分表表示给定图像及策略矩阵的维度
接下来
H 行输入图像矩阵
K 行输入策略矩阵
输出描述
输出最大能量值
样例1
输入
1 1 1 1
5
1

输出
5.0

说明
有且仅有一条路径，最大能量值为 5∗15*15∗1 为 5.05.05.0
样例2
输入
3 3 3 3
1 2 3
4 5 6
7 8 9
1 2 2
1 1 1
1 1 1

输出
119.0

说明
输入第一行是一个 3×33×33×3 的图像以及 3×33×33×3 的策略矩阵
每个位置的能量图：
[[12.21.16.]
[30.50.36.]
[33.50.34.]][33.50.34.]][33.50.34.]]
最大能量路径的值：119.0119.0119.0 最大能量路径：(2,0)−>(1,1)−>(1,2)(2,0)->(1,1)->(1,2)(2,0)−>(1,1)−>(1,2)
提示
1.1.1.策略矩阵为奇数，边缘处用零填充
2.2.2.输出保留一位小数


#### 解答


思路

预处理能量： 先按上式计算整张图的能量矩阵 E，复杂度 O(H⋅W⋅K2)O(H\cdot W\cdot K^2)O(H⋅W⋅K2)。

动态规划建模： 用 fi,jf_{i,j}fi,j 表示走到位置 (i,j)(i,j)(i,j) 的最大能量和：

边界： fi,0=Ei,0f_{i,0}=E_{i,0}fi,0=Ei,0，对所有 i∈[0,H−1]i\in[0,H-1]i∈[0,H−1]。

转移：

答案： max⁡0≤i<Hfi,W−1\max\limits_{0\le i<H} f_{i,W-1}0≤i<Hmaxfi,W−1。
动规部分复杂度 O(H⋅W)O(H\cdot W)O(H⋅W)，总复杂度 O(H⋅W⋅K2)O(H\cdot W\cdot K^2)O(H⋅W⋅K2)，空间 O(H⋅W)O(H\cdot W)O(H⋅W)（可滚动数组降到 O(H)O(H)O(H)）。

实现细节：

第一行输入可能是用空格或反斜杠分隔，实际读取时可将反斜杠替换为空格再解析 H,W,KH, W, KH,W,K。
输出用固定小数位格式保留 1 位小数。


**Python 代码：**

```python
import sys

def main():
data = sys.stdin.read().strip().split()
it = iter(data)
H = int(next(it)); W = int(next(it)); K1 = int(next(it)); K2 = int(next(it))
K = K1  # 题面给了两个K，这里取第一个；通常两者相等

# 读图像矩阵
I = [[float(next(it)) for _ in range(W)] for _ in range(H)]
# 读策略矩阵
P = [[float(next(it)) for _ in range(K)] for _ in range(K)]

# 计算能量图（零填充卷积）
r = K // 2
E = [[0.0]*W for _ in range(H)]
for i in range(H):
for j in range(W):
s = 0.0
for u in range(K):
ii = i + (u - r)
if 0 <= ii < H:
rowI = I[ii]
rowP = P[u]
for v in range(K):
jj = j + (v - r)
if 0 <= jj < W:
s += rowP[v] * rowI[jj]
E[i][j] = s

# 动态规划
NEG = -1e300
prev = [NEG]*H
for i in range(H):
prev[i] = E[i][0]

for j in range(1, W):
cur = [NEG]*H
for i in range(H):
best = prev[i]
if i-1 >= 0:
best = max(best, prev[i-1])
if i+1 < H:
best = max(best, prev[i+1])
cur[i] = E[i][j] + best
prev = cur

ans = max(prev)
print(f"{ans:.1f}")

if __name__ == "__main__":
main()

```

---

<a id="第3题-p4275"></a>

### 第3题-基于空间连续块的稀疏注意力机制（P4275）- 中等





在大语言模型推理过程中，随着上下文长度增加，标准 AttentionAttentionAttention 的计算开销以 O(n2)O(n^2)O(n2) 增长，成为性能瓶颈。为提升长序列处理效率，提出一种基于空间连续块的稀疏注意力机制。
具体流程如下：

一个长度为 n 的历史 tokentokentoken 序列，每个 tokentokentoken 表示为 1 个 d 维特征向量 xj∈Rd\mathbf{x}_j \in \mathbb{R}^dxj∈Rd
。按固定块大小b，将序列划分为 m=ceil(n/b)m = ceil(n/b)m=ceil(n/b)个空间连续块（最后一个块可不满）
B1,B2,...,BmB_1, B_2, ..., B_mB1,B2,...,Bm，其中：Bk=[x(k−1)b,...,xmin(kn)−1]B_k = [x_{(k-1)b}, ..., x_{min(kn)-1}]Bk=[x(k−1)b,...,xmin(kn)−1]

对每个块 BkB_kBk：
(1) 计算平均池化向量：$\mathbf{h}_k = \frac{1}{B_k} \sum_{x \in B_k} \mathbf{x}$
(2) 使用一个两层多层感知机（MLP）进行非线性压缩（隐藏维度dl=1d_l = 1dl=1）：$\mathbf{c}_k = W_2 \cdot \sigma(W_1 \cdot \mathbf{h}_k + b_1) + b_2$
其中：
① W1∈R1×dW_1 \in \mathbb{R}^{1 \times d}W1∈R1×d，W2∈Rd×1W_2 \in \mathbb{R}^{d \times 1}W2∈Rd×1，输出 ck∈Rd\mathbf{c}_k \in \mathbb{R}^dck∈Rd
②b1=2b_1 = 2b1=2，b2=1b_2 = 1b2=1
③σ(x)=max(0,x)\sigma(x) = max(0, x)σ(x)=max(0,x)（即 ReLU 激活函数）

给定查询向量q∈Rd\mathbf{q} \in \mathbb{R}^dq∈Rd（题目中固定为全 1 向量：qi=1q_i = 1qi=1），计算每个压缩块的注意力得分：
$a_k = \frac{\mathbf{q} \cdot \mathbf{c}_k}{\sqrt{d}}$
得到压缩块注意力得分序列 A=(a1,a2,...,am)A = (a_1, a_2, ..., a_m)A=(a1,a2,...,am)

将序列 A 划分为恰好 2 个连续非空子数组，目标是最大化这两个子数组和中的最小值 S 。

最终输出该最大化的最小值 S 的整数化得分，该子数组对应的 tokentokentoken 块将跳过细粒度 attentionattentionattention 计算，实现稀疏推理。
其中，整数化得分即 S SS  乘以 100 后四舍五入得到的整数，以实现保留两位小数精度的整数化表示：
round(100⋅S)round(100 \cdot S)round(100⋅S)

输入描述
第 1 行：n d b，以空格分隔，分别为序列长度、tokentokentoken 向量维度、块大小
接下来 n 行：每行 d  个数，以空格分隔，表示 xix_ixi
倒数第 2 行：d 个数，以空格分隔，表示 $W_1$
最后 1 行：d 个数，以空格分隔，表示 $W_2$
约束条件：
1≤n≤10001 \leq n\leq 10001≤n≤1000
1≤b≤n1 \leq b\leq n1≤b≤n
1≤d≤1001 \leq d \leq 1001≤d≤100
所有向量非零
输出描述
返回一个整数，即上述步骤 5 的整数化得分
样例1
输入
3 1 1
2.0
4.0
6.0
1.0
2.0

输出
1700

说明
①分块：B1=[2.0]B_1 = [2.0]B1=[2.0]，B2=[4.0]B_2 = [4.0]B2=[4.0]，B3=[6.0]B_3 = [6.0]B3=[6.0]
②平均池化：h1=[2.0]h_1 = [2.0]h1=[2.0]，h2=[4.0]h_2 = [4.0]h2=[4.0]，h3=[6.0]h_3 = [6.0]h3=[6.0]
③MLP 压缩：c1=[9.0]c_1 = [9.0]c1=[9.0]，c2=[13.0]c_2 = [13.0]c2=[13.0]，c3=[17.0]c_3 = [17.0]c3=[17.0]
④注意力得分：A=[9,13,17]A=[9,13,17]A=[9,13,17]
⑤划分为 2 个连续非空子数组，最大化min(sum)min(sum)min(sum)：
[9]∣[13,17][9]|[13,17][9]∣[13,17] →→→ 和：9,30→min=99, 30 → min = 99,30→min=9
[9,13]∣[17][9,13]|[17][9,13]∣[17] →→→ 和：22,17→min=17→S=1722, 17 → min = 17→ S=1722,17→min=17→S=17，输出 1700
样例2
输入
3 2 1
2.0 1.0
3.0 2.0
4.0 3.0
1.0 0.5
2.0 1.0

输出
1732

说明
①分块：B1=[2.0,1.0]B_1 = [2.0, 1.0]B1=[2.0,1.0]，B2=[3.0,2.0]B_2 = [3.0, 2.0]B2=[3.0,2.0]，B3=[4.0,3.0]B_3 = [4.0, 3.0]B3=[4.0,3.0]
②平均池化：h1=[2.0,1.0]h_1 = [2.0, 1.0]h1=[2.0,1.0]，h2=[3.0,2.0]h_2 = [3.0, 2.0]h2=[3.0,2.0]，h3=[4.0,3.0]h_3 = [4.0, 3.0]h3=[4.0,3.0]
③MLP 压缩：c1=[10.0,5.5]c_1 = [10.0, 5.5]c1=[10.0,5.5]，c2=[13.0,7.0]c_2 = [13.0, 7.0]c2=[13.0,7.0]，c3=[16.0,8.5]c_3 = [16.0, 8.5]c3=[16.0,8.5]
④注意力得分：$A = [\frac{15.5}{\sqrt{2}}, \frac{20.0}{\sqrt{2}}, \frac{24.5}{\sqrt{2}}]$
⑤划分为 2 个连续非空子数组，最大化(min(sum))：
$[\frac{15.5}{\sqrt{2}}] ， [\frac{20.0}{\sqrt{2}}, \frac{24.5}{\sqrt{2}}] →$和：15.52\frac{15.5}{\sqrt{2}}215.5，44.52\frac{44.5}{\sqrt{2}}244.5 →min=→ min =→min= 15.52\frac{15.5}{\sqrt{2}}215.5
$[\frac{15.5}{\sqrt{2}}, \frac{20.0}{\sqrt{2}}] ， [\frac{24.5}{\sqrt{2}}] →$ 和：35.52\frac{35.5}{\sqrt{2}}235.5，24.52\frac{24.5}{\sqrt{2}}224.5 $→ min = \frac{24.5}{\sqrt{2}}→ S = \frac{24.5}{\sqrt{2}}$，输出 1732


#### 解答


解题思路
长序列下，将历史 token 划分为固定大小的空间连续块，每块做均值池化后经一个两层 MLP 压缩成向量，再用固定查询向量 q=1q=\mathbf{1}q=1 与压缩结果做打分。得到的压缩分数序列 A=(a1,…,am)A=(a_1,\dots,a_m)A=(a1,…,am) 之后，需要将其划分为恰好两个连续非空子数组，使两段和的最小值最大。整体可分为两部分：

数值构造：分块 + 池化 + MLP + 打分

设序列长度为 n、维度为 d、块大小为 b，块数 m=⌈n/b⌉m=\lceil n/b\rceilm=⌈n/b⌉。

第 k 个块的均值池化
$$h_k=\frac{1}{|B_k|}\sum_{x\in B_k} x\in\mathbb{R}^d$$

两层 MLP（隐藏维度为 1）：
$$t_k=W_1\cdot h_k+b_1,\quad r_k=\sigma(t_k)=\max(0,t_k)$$ck=W2⋅rk+b2    ∈Rdc_k=W_2\cdot r_k + b_2\;\;\in\mathbb{R}^d
ck=W2⋅rk+b2∈Rd
其中 $W_1\in\mathbb{R}^{1\times d},\, W_2\in\mathbb{R}^{d\times 1},\, b_1=2,\, b_2=1$（标量，按维度广播）。

因为 q=1q=\mathbf{1}q=1，注意力得分
$$a_k=\frac{q\cdot c_k}{\sqrt d}=\frac{\sum_{i=1}^{d} c_k^{(i)}}{\sqrt d}$$顺序得到 A。

最优划分：前缀和 + 贪心

目标是 $\max_{1\le s\le m-1} \min\Big(\sum_{i=1}^{s}a_i,\sum_{i=s+1}^{m}a_i\Big)$。
记总和 T=∑i=1maiT=\sum_{i=1}^{m}a_iT=∑i=1mai，前缀和 Ps=∑i=1saiP_s=\sum_{i=1}^{s}a_iPs=∑i=1sai。显然最优 s 使得两段尽量“均衡”，即 PsP_sPs 最接近 T/2T/2T/2。
实现上只需一次线性扫描：维护前缀和，逐个计算 min⁡(Ps,T−Ps)\min(P_s,T-P_s)min(Ps,T−Ps) 的最大值即可。
这是典型的前缀和 + 单遍扫描贪心，时间 O(m)O(m)O(m)，优于通用的“二分答案 + 可行性判断”。

最终答案为 S=max⁡smin⁡(⋅)S=\max_s \min(\cdot)S=maxsmin(⋅)，题目要求输出 round(100⋅S)\text{round}(100\cdot S)round(100⋅S) 的整数（四舍五入，保留两位小数的整数化）。
复杂度分析

时间复杂度：

计算所有块均值与 MLP：遍历每个 token 各维度，O(n⋅d)O(n\cdot d)O(n⋅d)。
计算打分并寻找最优切分点：O(m)O(m)O(m)，其中 m=⌈n/b⌉≤nm=\lceil n/b\rceil\le nm=⌈n/b⌉≤n。
总计 O(n⋅d)O(n\cdot d)O(n⋅d)。

空间复杂度：

保存一块的中间向量与常量参数，外加得分序列（或仅累计总和与前缀），为 O(d+m)O(d+m)O(d+m)，可降到 O(d)O(d)O(d)（边算边累计，不必存整列）。

代码实现

**Python 代码：**

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

---

## 2025年10月17日-AI方向

<a id="第2题-p4238"></a>

### 第2题-利用大规模预训练模型实现智能告警聚类与故障诊断（P4238）- 中等





【背景信息】在现代运维体系中，大量告警可能指向同一故障根源（如 “服务器 CPU 利用率过高” 和 “应用响应超时” 可能由同一硬件资源不足导致）。若能将语义相似的告警归为一类，不仅可以减少重复信息的干扰，还能帮助运维人员快速定位故障核心，缩短故障修复时间。
行业内普遍采用自然语言处理（NLP）技术对告警文本进行语义理解，采用基于预训练语言模型（如 BERT、sBERT 等）的语义向量（embedding）转化技术：通过模型处理，每条告警文本被转化为一个高维数值向量，向量的数学特征能够准确反映告警的语义信息，使得两条描述相同故障的告警（即使措辞略有差异），其对应的向量在空间中的距离会非常近；而语义无关的告警，向量距离则较远。
【任务目标】通过语义向量（embedding）对给定的告警信息进行聚类：每条告警包含唯一的 ID 和对应的向量 embedding，要求将余弦相似度≥0.95 的告警归为同一个聚类，并返回数量最大的聚类的告警数量
【规则要求】
聚类判定标准：
1）相似度阈值：当两条告警的余弦相似度 ≥ 0.95 时，判定为语义相似。
2）弱传递聚类（连通图聚类）规则：初始状态：每条告警单独构成一个类别。归入规则：若告警 X 与某类别 C 中的任意一条告警 的余弦相似度 ≥ 0.95，则将 X 归入类别 C。
合并规则：若告警 X 同时满足归入多个类别的条件（即与多个类别中的告警均相似），则这些类别需合并为一个新类别，X 归入该新类别。
传递性保证：聚类过程需确保所有满足相似条件的告警最终被合并到同一类别中。例如：若 A 与 B 相似（余弦相似度 ≥ 0.95），且 B 与 C 相似（余弦相似度 ≥ 0.95），则 A、B、C 必须属于同一类别（即使 A 与 C 的相似度可能 < 0.95）。
输入描述
每一行为一个告警信息，其中第一个字段是告警 ID，后面的字段是告警的嵌入向量。告警信息的总行数不会超过 1000 条。（请注意，测试集中可能包含如样例 2 所示的那种异常情况）
输出描述
找到包含告警数量最多的聚类，输出该聚类的告警数量。若所有告警均无法聚类（即每个类别仅包含 1 条告警），则返回 1；若输入为空列表（无任何告警），或者输入告警信息的向量维度不一致（即不同告警的 embedding 长度不同），则返回 0。
样例1
输入
1 1.0 0.0 0.0
2 0.99 0.01 0.0
3 0.0 1.0 0.0
4 0.0 1.0 0.01
5 0.1 0.0 0.0

输出
3

说明
每一行输入的第一个字段是告警 id，后面的字段是告警的嵌入向量，根据余弦相似度≥0.95 的规则，我们得到以下聚类关系：

告警 1、2、5 构成一个聚类；

告警 3、4 构成一个聚类；
所有聚类的大小分别为 3、2。其中数量最大的为 3，因此输出为 3。

样例2
输入
1 1.0 0.0 0.0
2 0.99 0.01 0.0 0.98
3 0.0 1.0 0.0

输出
0

说明
第 2 个告警的嵌入向量维度与其他告警不一致，属于异常情况，返回 0
样例3
输入
1 0.878434 -0.068245 -0.46237 0.099552
2 0.33961 -0.083281 -0.348141 0.869786
3 0.326485 -0.071012 -0.353166 0.873865
4 0.330106 -0.085155 -0.338106 0.87719
5 0.340185 -0.066865 -0.339054 0.874554
6 0.482266 -0.483077 0.539977 -0.492423
7 0.491966 -0.485237 0.526674 -0.495104
8 0.48426 -0.477249 0.531019 -0.505711
9 -0.669925 -0.330461 0.409454 -0.523778
10 -0.668543 -0.331692 0.403806 -0.529123

输出
4

说明
每一行输入的第一个字段是告警 id，后面的字段是告警的嵌入向量，根据余弦相似度≥0.95 的规则，我们得到以下聚类关系：

告警 2、3、4、5 构成一个聚类；

告警 6、7、8 构成一个聚类；

告警 9、10 构成一个聚类；

告警 1 独立成一个聚类；
综上所述，所有聚类的大小分别为 4、3、2、1。其中数量最大的为 4，因此输出为 4。

▶️


#### 解答


video solution

解题思路
本题要求对告警信息进行语义聚类，核心是基于余弦相似度的连通分量问题。题目的关键在于理解弱传递聚类规则：如果告警A与B相似，B与C相似，即使A与C不相似，它们也应属于同一聚类。这实际上是一个典型的并查集问题。
首先需要处理输入数据的合法性验证。输入为空或向量维度不一致时直接返回0。对于合法输入，我们需要计算所有告警对之间的余弦相似度。余弦相似度的计算公式为：
cos(A,B) = (A·B) / (|A| × |B|)
其中A·B表示向量点积，|A|和|B|表示向量的欧几里得范数。当余弦相似度大于等于0.95时，认为两条告警语义相似，需要将它们归入同一聚类。
为了高效地处理聚类合并操作，我们采用并查集数据结构。并查集支持两个核心操作：查找元素所属集合的根节点，以及合并两个集合。通过路径压缩优化，可以使查找操作接近常数时间复杂度。
算法流程如下：首先初始化并查集，每个告警独立成一个集合。然后遍历所有告警对，计算它们的余弦相似度，如果相似度达到阈值则合并两个集合。最后统计每个集合的大小，返回最大的集合大小即为答案。
复杂度分析
时间复杂度：O(n²·d + n²·α(n))，其中n是告警数量，d是向量维度，α是反阿克曼函数（可视为常数）。计算所有告警对的余弦相似度需要O(n²·d)时间，每次相似度计算涉及向量点积和范数计算均为O(d)。并查集的查找和合并操作经过路径压缩优化后均摊时间复杂度为O(α(n))，总共需要O(n²)次操作。
空间复杂度：O(n)，主要用于存储并查集的父节点数组以及统计聚类大小的哈希表。输入数据的存储空间为O(n·d)，但这是必须的输入开销。
代码实现

**Python 代码：**

```python
import sys
import math
from collections import Counter

def solve(alerts):
# 处理空输入
if not alerts:
return 0

n = len(alerts)
if n == 0:
return 0

# 检查向量维度一致性
dim = len(alerts[0][1])
for i in range(n):
if len(alerts[i][1]) != dim:
return 0

# 初始化并查集
parent = list(range(n))

def find(x):
# 路径压缩
if parent[x] != x:
parent[x] = find(parent[x])
return parent[x]

def union(x, y):
# 合并两个集合
px, py = find(x), find(y)
if px != py:
parent[px] = py

def cosine_similarity(v1, v2):
# 计算余弦相似度
dot_product = sum(a * b for a, b in zip(v1, v2))
norm1 = math.sqrt(sum(a * a for a in v1))
norm2 = math.sqrt(sum(b * b for b in v2))
if norm1 == 0 or norm2 == 0:
return 0
return dot_product / (norm1 * norm2)

# 遍历所有告警对，合并相似告警
for i in range(n):
for j in range(i + 1, n):
sim = cosine_similarity(alerts[i][1], alerts[j][1])
if sim >= 0.95:
union(i, j)

# 统计每个聚类的大小
clusters = Counter(find(i) for i in range(n))

# 返回最大聚类大小
return max(clusters.values())

def main():
alerts = []
for line in sys.stdin:
line = line.strip()
if not line:
continue
parts = line.split()
alert_id = parts[0]
embedding = [float(x) for x in parts[1:]]
alerts.append((alert_id, embedding))

result = solve(alerts)
print(result)

if __name__ == "__main__":
main()

```

---

<a id="第3题-p4239"></a>

### 第3题-反向传播实现（P4239）- 困难





给定K层前馈网络模型的权重矩阵M[i]M[i]M[i]、偏移向量b[i]b[i]b[i]，以及一批输入数据X和对应的真实分类标签Y_true_labelsY\_true\_labelsY_true_labels，请计算出总损失L对每一个权重矩阵M[i]M[i]M[i]和每一个偏移向量b[i]b[i]b[i]的梯度。
模型架构
模型有K个权重矩阵，第i个矩阵是M[i]。还有K个偏移向量，第i个向量是b[i]。
网络的计算过程（前向传播）如下：

输入是一个 N×wN \times wN×w 的矩阵 x，其中 N 是 batch size （本次输入数据的个数），w 是初始特征的数量。
网络的计算分为K层。我们用 A[i]A[i]A[i] 代表第 i 层的输出（激活值）。初始输入 A[0]=XA[0] = XA[0]=X。
对于第 i 层（从 i=1i = 1i=1 到 K−1K - 1K−1 ）：

线性计算：Z[i]=A[i−1]⋅M[i]+b[i]Z[i] = A[i - 1] \cdot M[i] + b[i]Z[i]=A[i−1]⋅M[i]+b[i]
激活函数：A[i]=ReLU(Z[i])A[i] = ReLU(Z[i])A[i]=ReLU(Z[i])

对于最后一层（第 K 层），使用 Softmax 激活函数来输出每个类别的概率：

线性计算：Z[K]=A[K−1]⋅M[K]+b[K]Z[K] = A[K - 1] \cdot M[K] + b[K]Z[K]=A[K−1]⋅M[K]+b[K]
激活函数：A[K]=Softmax(Z[K])A[K] = Softmax(Z[K])A[K]=Softmax(Z[K])

最终的输出为 A[K]A[K]A[K]，我们称之为 output_probabilities。这是一个 N×10N \times 10N×10 的矩阵。

为了衡量模型预测的好坏，使用了交叉熵损失（Cross-Entropy Loss），具体计算公式参见提示部分。
输入描述

第一行是一个整数K，代表网络的层数。

第二行是K个整数，依次代表每层的维度h[0],h[1],h[2],…,h[K−1]h[0], h[1], h[2], \ldots, h[K-1]h[0],h[1],h[2],…,h[K−1]。
最后一层的输出维度固定为10，代表10个类别。其中h[0]=W,h[K]=10h[0] = W, h[K] = 10h[0]=W,h[K]=10。

M[i]M[i]M[i]的形状是h[i−1]×h[i]h[i-1] \times h[i]h[i−1]×h[i]（其中h即为w），1≤i≤K1 \le i \le K1≤i≤K。

b[i]b[i]b[i]的形状是1×h[i]1 \times h[i]1×h[i]。

接下来是 K 个权重矩阵 M[1],…,M[K]M[1], \ldots, M[K]M[1],…,M[K] 的数据。第 i 矩阵的数字分为 h[i−1]h[i - 1]h[i−1] 行、每行 h[i]h[i]h[i] 个浮点数、第 i 行j列表示 M[i]M[i]M[i] 矩阵的第 i 行j列 1≤i≤K1 \leq i \leq K1≤i≤K）。

接下来是K个偏移向量b[1],…,b[K]b[1], \ldots, b[K]b[1],…,b[K]的数据。
分为K行，第i行有h[i]h[i]h[i]个数字，表示b[i]b[i]b[i]的权重（1≤i≤K1 \le i \le K1≤i≤K）。

之后是一个整数N，代表batch size。

接下来是N行，每行w个浮点数，代表输入矩阵X。

最后是N行，每行一个整数y∈{0,1,…,9}y \in \{0,1,\ldots,9\}y∈{0,1,…,9}，代表真实的类别标签。

输入保证输入的所有浮点数绝对值不超过100，且小数点后最多2位数字。
即任意输入浮点数f，有∣f∣≤100|f| \le 100∣f∣≤100，且100⋅f100 \cdot f100⋅f是整数。
输出描述

依次输出K个权重矩阵的梯度和K个偏移向量的梯度。

对于第i层（从1到K）：

首先，输出梯度矩阵∂L∂M[i]\dfrac{\partial L}{\partial M[i]}∂M[i]∂L，分为h[i−1]h[i-1]h[i−1]行输出，每行h[i]h[i]h[i]个实数，用一个空格分隔，行末不能有空格。
然后，输出平均梯度向量∂L∂b[i]\dfrac{\partial L}{\partial b[i]}∂b[i]∂L，一行有h[i]h[i]h[i]个数字，用一个空格分隔，行末不能有空格。

所有浮点数保留4位小数。

输入保证计算出来的输出梯度不会有NAN，且可以被双精度浮点数表示（即不会炸梯度，不会出现NAN/INF）。
样例1
输入
2
4 5
1.0 1.0 1.0 1.0 1.0
1.0 1.0 1.0 1.0 1.0
1.0 1.0 1.0 1.0 1.0
1.0 1.0 1.0 1.0 1.0
0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
0.1 0.1 0.1 0.1 0.1
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
2
0.5 0.5 0.5 0.5
0.5 0.5 0.5 0.5
2
8

输出
0.0000 0.0000 0.0000 0.0000 0.0000
0.0000 0.0000 0.0000 0.0000 0.0000
0.0000 0.0000 0.0000 0.0000 0.0000
0.0000 0.0000 0.0000 0.0000 0.0000
0.0000 0.0000 0.0000 0.0000 0.0000
0.2100 0.2100 -0.8400 0.2100 0.2100 0.2100 0.2100 0.2100 -0.8400 0.2100
0.2100 0.2100 -0.8400 0.2100 0.2100 0.2100 0.2100 0.2100 -0.8400 0.2100
0.2100 0.2100 -0.8400 0.2100 0.2100 0.2100 0.2100 0.2100 -0.8400 0.2100
0.2100 0.2100 -0.8400 0.2100 0.2100 0.2100 0.2100 0.2100 -0.8400 0.2100
0.2100 0.2100 -0.8400 0.2100 0.2100 0.2100 0.2100 0.2100 -0.8400 0.2100
0.1000 0.1000 -0.4000 0.1000 0.1000 0.1000 0.1000 0.1000 -0.4000 0.1000

说明
输入样例是2个样本，特征相同，但是标签不同。
样例2
输入
1
2
1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1
1
50.0 60.0
4

输出
0.0000 0.0000 0.0000 0.0000 -50.0000 0.0000 0.0000 0.0000 0.0000 50.0000
0.0000 0.0000 0.0000 0.0000 -60.0000 0.0000 0.0000 0.0000 0.0000 60.0000
0.0000 0.0000 0.0000 0.0000 -1.0000 0.0000 0.0000 0.0000 0.0000 1.0000

说明
一层网络，一个样本。
Z[1] = [56.01 112.02 168.03 224.04 280.05 336.06 392.07 448.08 504.09 560.1]
A[1] = [1.1925994847839173e-219, 2.519582300056682e-195, 5.323073712302622e-171,
1.1245956818306658e-146, 2.3759119560363354e-122, 5.019544102861018e-98,
1.060469557238993e-73, 2.240433909505195e-49, 4.733322204863164e-25, 1.0]
Ground Truth是[0,0,0,0,1,0,0,0,0,0]
提示
Loss计算公式
对于一批（batch）大小为N的输入，每个输入都有一个真实的类别标签
yj∈{0,1,…,9}y_j \in \{0,1,\ldots,9\}yj∈{0,1,…,9}
首先，我们将真实标签yjy_jyj转换为one-hot向量(Ytrue)j(Y_{true})_j(Ytrue)j。例如，如果真实标签为2，其one-hot向量为
[0,0,1,0,0,0,0,0,0,0]。
总损失L的计算方式是批次中所有样本损失的平均值：
$$L = -\frac{1}{N} \sum_{j=1}^{N} \sum_{l=0}^{9} (Y_{true})_{jl} \log(output\_probabilities_{jl})$$其中(output_probabilities)jl(output\_probabilities)_{jl}(output_probabilities)jl是第j个样本的预测输出向量（经过Softmax后）的第l个元素。
数据范围
1≤K≤51 \le K \le 51≤K≤5
1≤w,h[i]≤1001 \le w, h[i] \le 1001≤w,h[i]≤100
1≤N≤1001 \le N \le 1001≤N≤100
所有输入浮点数的绝对值不超过100，且浮点数小数点后最多两位小数。
提示
提示

提示1：

$\frac{\partial L}{\partial b[i]_j} = \frac{\partial Z[i]_j}{\partial b[i]_j} \cdot \frac{\partial L}{\partial Z[i]_j} = \frac{\partial L}{\partial Z[i]_j}$

提示2：

$\frac{\partial L}{\partial M[i]_{kj}} = \frac{\partial Z[i]_j}{\partial M[i]_{kj}} \cdot \frac{\partial L}{\partial Z[i]_j} = A[i-1][k] \cdot \frac{\partial L}{\partial Z[i]_j}$

提示3：

根据链式法则，dM[i]=A[i−1]T×dZ[i]dM[i] = A[i-1]^T \times dZ[i]dM[i]=A[i−1]T×dZ[i]（T表示转置）


#### 解答


解题思路
这道题要求实现一个多层前馈神经网络的反向传播算法，计算损失函数对每层权重矩阵和偏置向量的梯度。
核心算法是反向传播（Backpropagation），其基本思想是利用链式法则从输出层向输入层逐层计算梯度。具体实现包括以下步骤：
第一步是前向传播。从输入层开始，逐层计算每一层的线性组合结果Z和激活值A。前K-1层使用ReLU激活函数，最后一层使用Softmax激活函数得到类别概率分布。在前向传播过程中，需要保存所有的Z和A值，供反向传播使用。
第二步是计算输出层的梯度。对于交叉熵损失函数配合Softmax激活函数的组合，其对Z[K]的梯度有简化形式：dZ[K] = (output_probabilities - Y_true) / N，其中Y_true是真实标签的one-hot编码。
第三步是反向传播计算各层梯度。从输出层向输入层反向遍历，对于每一层，根据链式法则计算梯度。对于使用ReLU激活的层，需要根据前向传播时的Z值判断ReLU的导数（大于0为1，否则为0）。每层的权重梯度等于上一层激活值的转置与该层Z梯度的矩阵乘积，偏置梯度等于该层Z梯度在样本维度上的求和。
第四步是将计算得到的梯度按指定格式输出。需要注意梯度矩阵的形状必须与对应的权重矩阵形状一致。
实现过程中需要特别注意的是Softmax函数的数值稳定性问题，应该采用减去最大值的技巧避免指数运算溢出。
复杂度分析
时间复杂度为O(N × sum(h[i] × h[i+1]))，其中N是批次大小，h[i]是每层的维度。前向传播和反向传播都需要遍历所有层，每层的矩阵乘法运算复杂度为O(N × h[i] × h[i+1])。
空间复杂度为O(K × N × max(h[i]))，需要存储K层的所有中间激活值和梯度值，每层最多需要存储N × h[i]大小的矩阵。
代码实现

**Python 代码：**

```python
import sys
import numpy as np

# 打印矩阵，保留 4 位小数
def print_formatted_matrix(matrix):
for row in matrix:
print(" ".join(f"{x:.4f}" for x in row))

# 打印向量，保留 4 位小数
def print_formatted_vector(vector):
print(" ".join(f"{x:.4f}" for x in vector))

# 数值稳定的 Softmax 实现
def stable_softmax(z):
# 减去每行的最大值，防止指数溢出
z_shifted = z - np.max(z, axis=1, keepdims=True)
exps = np.exp(z_shifted)
return exps / np.sum(exps, axis=1, keepdims=True)

# 反向传播函数
def backprop(dZK, K, A, M, Z):
"""
dZK: 最后一层 softmax+交叉熵 的梯度
K: 网络层数
A: 每层激活输出（包含输入 A[0]）
M: 每层权重矩阵
Z: 每层线性变换结果
"""
grad_M = [None] * (K + 1)  # 存放每层对 M 的梯度
grad_b = [None] * (K + 1)  # 存放每层对 b 的梯度

# 最后一层梯度计算
grad_M[K] = A[K - 1].T @ dZK           # dL/dM[K] = A[K-1]^T * dZ[K]
grad_b[K] = np.sum(dZK, axis=0)        # dL/db[K] = sum(dZ[K])
dA_prev = dZK @ M[K].T                 # 传播到前一层

# 逐层反向传播（从 K-1 到 1）
for i in range(K - 1, 0, -1):
# ReLU 的导数：Z>0 时为 1，否则为 0
dZi = dA_prev * (Z[i] > 0).astype(float)
grad_M[i] = A[i - 1].T @ dZi       # dL/dM[i]
grad_b[i] = np.sum(dZi, axis=0)    # dL/db[i]
dA_prev = dZi @ M[i].T             # 向前传播梯度

return grad_M, grad_b

def solve():
try:
# 读取层数 K
K_str = sys.stdin.readline()
if not K_str.strip():
return
K = int(K_str)

# 读取每层维度（例如：4 5）
h_str = sys.stdin.readline()
h = list(map(int, h_str.split()))
dims = h + [10]  # 最后一层固定输出维度为 10 类

# 初始化权重矩阵和偏置向量
M = [None] * (K + 1)
b = [None] * (K + 1)

# 读取每层权重矩阵 M[i]
for i in range(1, K + 1):
rows, cols = dims[i-1], dims[i]
M[i] = np.array(
[list(map(float, sys.stdin.readline().split())) for _ in range(rows)],
dtype=float
).reshape(rows, cols)

# 读取每层偏置向量 b[i]
for i in range(1, K + 1):
b[i] = np.array(
list(map(float, sys.stdin.readline().split())),
dtype=float
).reshape(1, dims[i])

# 读取 batch size N
N = int(sys.stdin.readline())
# 读取输入样本矩阵 x
x = np.array([list(map(float, sys.stdin.readline().split())) for _ in range(N)], dtype=float)
# 读取真实标签（整数形式）
Y_labels = [int(sys.stdin.readline()) for _ in range(N)]

# 转换为 one-hot 编码
Y_true_onehot = np.zeros((N, 10), dtype=float)
Y_true_onehot[np.arange(N), Y_labels] = 1.0

# 前向传播
A = [None] * (K + 1)
Z = [None] * (K + 1)
A[0] = x

# 前 K-1 层使用 ReLU
for i in range(1, K):
Z[i] = A[i - 1] @ M[i] + b[i]
A[i] = np.maximum(0.0, Z[i])

# 最后一层使用 Softmax
Z[K] = A[K - 1] @ M[K] + b[K]
A[K] = stable_softmax(Z[K])
output_probabilities = A[K]

# Softmax + CrossEntropy 的梯度
dZK = (output_probabilities - Y_true_onehot) / N

# 反向传播计算梯度
grad_M, grad_b = backprop(dZK, K, A, M, Z)

# 输出每层梯度矩阵和偏置向量
for i in range(1, K + 1):
print_formatted_matrix(grad_M[i])
print_formatted_vector(grad_b[i])

except (IOError, ValueError, IndexError):
# 捕获输入或计算错误，防止程序崩溃
return

# 程序入口
if __name__ == "__main__":
solve()

```

---

## 2025年10月15日-AI方向

<a id="第2题-p4227"></a>

### 第2题-动态注意力掩码调度问题（P4227）- 中等





你正在设计一种跨模态知的大模型精准度机制，给定一个长度为 n 的输入 tokentokentoken 序列，每个位置 j 拥有一个 dd d维特征向量 Xj∈RdX_j \in \mathbb{R}^dXj∈Rd和一个正整数计算容量 cjc_jcj，表示该位置最多可接收来自前 j 位置的信息连接数。
系统需完成以下步骤：

RMSNormRMSNormRMSNorm 归一化：对所有特征向量进行 RMSNormRMSNormRMSNorm 归一化本题取(γ=1,ϵ=0)(\gamma = 1, \epsilon = 0)(γ=1,ϵ=0)：
每个特征向量记为xi∈Rdx_i \in \mathbb{R}^dxi∈Rd，其第 k kk 个分量为 xi[k]x_i[k]xi[k]。RMSNormRMSNormRMSNorm 定义为：
$\hat{X_i}  = \frac{ x_i}{\sqrt{\frac{1}{d}\sum_{k=1}^{d}x_i[k]^2 + \epsilon}}\cdot\gamma$

注意力得分计算：计算每对位置 i<ji<ji<j 的注意力得分，使用标准缩放点积公式（基于 RMSNormRMSNormRMSNorm 归一化向量）：
$A_{ij} = \frac{\hat{x_i}  \cdot \hat{x_j}}{\sqrt{d}}$

掩码矩阵构造：构造下三角注意力掩码矩阵M∈{0,1}n×nM \in \{0,1\}^{n \times n}M∈{0,1}n×n，满足入度约束：
$\forall j \in [0, n), \sum_{i=0}^{j-1} M_{ij} \leq c_j$

目标函数最大化：最大化全局注意力信息总量，全局注意力信息总量定义为所有激活连接的平方注意力得分之和：
$S = \sum_{j=0}^{n-1} \sum_{i=0}^{j-1} M_{ij} \cdot A_{ij}^2$

输出整数化得分：最终返回将最大化 S 乘以 100 后四舍五入得到的整数，以实现保留两位小数精度的整数化表示：
round(100⋅S)\text{round}(100 \cdot S)round(100⋅S)

输入描述

第 1 行: n d，以空格分隔，分别表示 tokentokentoken 序列长度和向量维度。
接下来 n 行：每行 d 个浮点数，以空格分隔，表示 xjx_jxj。
最后 1 行: nn n 个正整数，以空格分隔，表示 cjc_jcj。

约束条件

1≤n≤10001 \leq n \leq 10001≤n≤1000
1≤d≤1001 \leq d \leq 1001≤d≤100
所有向量非零

输出描述
返回一个整数，即上述步骤 5 的整数化得分
样例1
输入
4 2
2.0 2.0
3.0 0.0
0.0 4.0
1.0 1.0
1 2 1 3

输出
600

说明
位置 0：RMSNormRMSNormRMSNorm 归一化为 [1,1][1, 1][1,1]；无前置位置→→→对信息总量贡献 0
位置 1：RMSNormRMSNormRMSNorm 归一化为 [2,0][\sqrt{2}, 0][2,0]；前置位置 j=0j=0j=0，A012=1；c1=2A_{01}^2 = 1；c_1 = 2A012=1；c1=2，选择接收来自 j=0j=0j=0 的信息→→→对信息总量贡献 1
位置 2：RMSNorm2：RMSNorm2：RMSNorm 归一化为 [0,2][0, \sqrt{2}][0,2]；前置位置 j=0j=0j=0 和 j=1j=1j=1，计算 A022=1A_{02}^2=1A022=1,A122=0A_{12}^2 = 0A122=0；c2=1c_2 = 1c2=1，选择接收来自 j=0j=0j=0 的信息→→→对信息总量贡献 1
位置 3：RMSNorm3：RMSNorm3：RMSNorm 归一化为 [1,1][1, 1][1,1]；前置位置 j=0j=0j=0 和 j=1j=1j=1 和 j=2j=2j=2，计算 A032=2A_{03}^2 = 2A032=2，A132=1A_{13}^2 = 1A132=1，A232=1A_{23}^2 = 1A232=1；c2=3c_2 = 3c2=3，选择接收来自 j=0j=0j=0 和 j=1j=1j=1 和 j=2j=2j=2 的信息→→→对信息总量贡献 4
最大化 S=6S=6S=6，输出整数化得分 600
样例2
输入
3 2
1.0 0.0
0.0 1.0
1.0 1.0
1 1 2

输出
200

说明
位置 0：RMSNorm0：RMSNorm0：RMSNorm 归一化为 [2,0][\sqrt{2}, 0][2,0]；无前置位置→→→对信息总量贡献 0
位置 1：RMSNorm1：RMSNorm1：RMSNorm 归一化为 [0,2][0, \sqrt{2}][0,2]；前置位置 j=0j=0j=0，A012=0A_{01}^2 = 0A012=0；c1=1c_1 = 1c1=1，选择接收来自 i=0i=0i=0 的信息→→→对信息总量贡献 0
位置 2：RMSNorm2：RMSNorm2：RMSNorm 归一化为 [1,1][1, 1][1,1]；前置位置 j=0j=0j=0 和 j=1j=1j=1，计算 A022=1A_{02}^2 = 1A022=1，A122=1A_{12}^2 = 1A122=1；c2=2c_2 = 2c2=2，选择接收来自 j=0j=0j=0 和 j=1j=1j=1 的信息→→→对信息总量贡献 2
最大化 S=2S=2S=2，输出整数化得分 200

▶️


#### 解答


video solution

解题思路
本题的核心是在资源约束下最大化注意力信息总量。问题可以分解为以下几个步骤进行求解：
首先需要对所有特征向量进行RMSNorm归一化处理。对于每个d维特征向量，计算其均方根值，然后将向量的每个分量除以该均方根值。这一步保证了后续注意力得分计算的标准化基础。
接着计算所有位置对之间的注意力得分。对于任意两个位置i和j（其中i<j），使用归一化后的向量进行缩放点积运算，得到注意力得分AijA_{ij}Aij，并计算其平方值Aij2A_{ij}^2Aij2。由于最终目标函数中使用的是平方值，因此可以直接存储平方值以便后续使用。
问题的关键在于构造路径矩阵M。对于每个位置j，需要从前面的所有位置中选择最多cjc_jcj个位置建立连接。为了最大化目标函数S，应当采用贪心策略：对于每个位置j，将所有前置位置按照Aij2A_{ij}^2Aij2的值从大到小排序，然后选择前cjc_jcj个最大的值。这样可以保证每个位置获得的注意力信息量最大。
贪心策略的正确性在于：目标函数S是所有激活连接的Aij2A_{ij}^2Aij2之和，每个位置的选择是相互独立的，因此局部最优解（每个位置选择最大的cjc_jcj个值）必然能导致全局最优解。
最后将计算得到的S乘以100并四舍五入得到整数输出。
代码实现

**Python 代码：**

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

---

<a id="第3题-p4228"></a>

### 第3题-基于二分Kmeans算法的子网分割问题（P4228）- 中等





背景：在网络规划中，经常涉及子网分割问题，子网分割的目的是将距离相近的网络站点划分为一个子网，从而便于管理。
问题：聚类算法可以很好的解决子网分割问题，但是，聚类问题容易陷入局部最优。因此，本题期望采用优化版的聚类算法二分 KmeansKmeansKmeans 算法（Bi−KmeansBi-KmeansBi−Kmeans）进行子网分割。
方案概述：Bi−KmeansBi-KmeansBi−Kmeans 算法首先将全网按照常规的 KmeansKmeansKmeans 算法聚类成两个子网（也就是 K=2K=2K=2，两簇），然后，Bi−KmeansBi-KmeansBi−Kmeans 算法会基于 SSE（Sum of SquaredSquaredSquared ErrorErrorError）最小化原理，每次迭代只选择一个子网进一步划分，选择子网的原则是对该子网的进一步划分能够最大程度的降低全局 SSE，划分方法依旧是常规的 KmeansKmeansKmeans 算法（K=2K=2K=2），直到子网个数达到预期数量时，停止 Bi−KmeansBi-KmeansBi−Kmeans 算法的迭代（算法实现细节参见下述备注 1/2/31/2/31/2/3）。
备注 1-初始值选取：在进行常规 KmeansKmeansKmeans 聚类（K=2K=2K=2）二分子网时，选取子网中 x 坐标最小和最大的两个站点作为初始簇心进行划分（网络站点拥有不同的 x 坐标，本题中 x 坐标的最小值和最大值唯一）。
备注 2-算法迭代：在进行常规 KmeansKmeansKmeans 聚类（K=2K=2K=2）二分子网时，以簇中全部站点的平均坐标作为更新簇心；如果相邻的两次迭代聚类结果相同（各簇心迭代前后之间的距离小于 1e−61e^{-6}1e−6 则视为结果相同），则停止迭代，或者当迭代次数达到 1000 次时停止迭代。
备注 3−SSE3-SSE3−SSE 计算：以子网中全部站点的平均坐标为簇心，SSE 的计算方式是簇内所有站点到簇心距离的平方之和。
输入描述
输入包括三部分信息：
1）第一行数据表示期望分割的子网数量，用 N 表示，也就是聚类结果中簇的数量；N 是整数，范围 1<=N<=1001<=N<=1001<=N<=100 。
2）第二行数据表示全网站点总数，用 M 表示；M 是整数，范围 1<=M<=10001<=M<=10001<=M<=1000 。
3）从第三行开始的数据表示网络站点坐标，每一行代表一个站点的二维坐标，用空格分隔 x 轴坐标和 y 轴坐标； x 轴坐标和 y 轴坐标均为整数，0<=x0<=x0<=x 轴坐标 <=1000<=1000<=1000，0<=y0<=y0<=y 轴坐标 <=1000< =1000<=1000 。
输出描述
输出用二维数组记录划分的最终结果和划分过程，其中，第 k 行记录第 k 次划分后的结果，结果包含第 k 次划分后每个子网的站点数量，用空格分隔，并按照降序排列。
样例1
输入
3
3
0 0
2 2
5 5

输出
2 1
1 1 1

说明
输入表示我们期望将坐标分别为 (0,0)、(2,2)、(5,5)(0,0)、(2,2)、(5,5)(0,0)、(2,2)、(5,5) 的 3 个网络站点划分为 3 个子网。
按照题目要求，需要经过两次划分，第一次划分的结果为：
簇 1 1=[(0,0),(2,2)]1=[(0,0),(2,2)]1=[(0,0),(2,2)]
簇 1 2=[(5,5)]2=[(5,5)] 2=[(5,5)]
因此，期望输出的第一行是 2 1，其中，2 表示划分结果簇 1 1 有 2 个站点，1 表示划分结果簇 1 2 有 1 个站点，降序排列。
第二次划分的结果为：
簇 2 1=[(0,0)]1=[(0,0)]1=[(0,0)]
簇 2 2=[(2,2)]2=[(2,2)]2=[(2,2)]
簇 2 3=[(5,5)]3=[(5,5)]3=[(5,5)]
因此，期望输出的第二行是 1 1 1，其中，第一个 1 表示划分结果簇 2 1 有 1 个站点，第二个 1 表示划分结果簇 2 2 有 1 个站点，第三个 1 表示划分结果簇 2 3 有 1 个站点，降序排列。
样例2
输入
2
3
0 0
2 2
5 5

输出
2 1

说明
输入表示我们期望将坐标分别为 (0,0)、(2,2)、(5,5)(0,0)、(2,2)、(5,5)(0,0)、(2,2)、(5,5) 的 3 个网络站点划分为 2 个子网。
按照题目要求，需要经过一次划分，划分的结果为：
簇 1=[(0,0),(2,2)]1=[(0,0),(2,2)]1=[(0,0),(2,2)]
簇 2=[(5,5)]2=[(5,5)]2=[(5,5)]
因此，期望输出是 2 1，其中，2 表示划分结果簇 1 有 2 个站点，1 表示划分结果簇 2 有 1 个站点，降序排列。

▶️


#### 解答


video solution

解题思路
本题要求实现二分 K-means (Bi-Kmeans) 算法来解决网络子网分割问题。该算法是对传统 K-means 算法的优化，能够有效避免陷入局部最优解。
算法的核心思想是采用自顶向下的分裂策略，每次选择一个簇进行二分，直到达到目标簇数量。具体流程如下：
首先，将所有网络站点作为一个整体，使用标准 K-means 算法（K=2）将其分割成两个子网。在进行 K-means 聚类时，选取子网中 x 坐标最小和最大的两个站点作为初始簇心，然后迭代更新簇心（使用簇内所有站点的平均坐标），直到簇心变化小于阈值或达到最大迭代次数。
接下来，算法进入主循环，每次迭代都需要从现有的所有簇中选择一个进行进一步划分。选择的标准是基于 SSE（误差平方和）最小化原则：计算每个簇被划分前后的 SSE 差值，选择能够最大程度降低全局 SSE 的簇进行划分。SSE 的计算方式是以簇的平均坐标为簇心，计算簇内所有站点到簇心的欧氏距离平方和。
每次划分后，都需要输出当前所有簇的站点数量，并按降序排列。重复这个过程，直到簇的数量达到预期值 N。
需要注意的边界情况包括：只有一个站点的簇无法继续划分；簇的分配需要基于站点到簇心的最小欧氏距离；迭代过程中需要处理空簇的情况。
代码实现

**Python 代码：**

```python
import numpy as np

def calculate_sse(points):
# 计算簇的SSE（误差平方和）
if len(points) == 0:
return 0
center = np.mean(points, axis=0)
return np.sum((points - center) ** 2)

def kmeans_split(points):
# 使用K-means算法将点集分成两个簇
if len(points) <= 1:
return [points]

# 选择x坐标最小和最大的点作为初始簇心
min_idx = np.argmin(points[:, 0])
max_idx = np.argmax(points[:, 0])
centers = np.array([points[min_idx], points[max_idx]])

# 迭代更新簇心
for _ in range(1000):
# 计算每个点到两个簇心的距离
distances = np.sum((points[:, np.newaxis] - centers) ** 2, axis=2)
labels = np.argmin(distances, axis=1)

# 更新簇心
new_centers = np.array([
np.mean(points[labels == 0], axis=0) if np.any(labels == 0) else centers[0],
np.mean(points[labels == 1], axis=0) if np.any(labels == 1) else centers[1]
])

# 检查是否收敛
if np.sum((centers - new_centers) ** 2) < 1e-12:
break

centers = new_centers

# 最终分配
distances = np.sum((points[:, np.newaxis] - centers) ** 2, axis=2)
labels = np.argmin(distances, axis=1)

return [points[labels == 0], points[labels == 1]]

def bi_kmeans(points, n):
# 二分K-means算法主函数
clusters = [points]
results = []

# 第一次划分
clusters = kmeans_split(clusters[0])
sizes = sorted([len(c) for c in clusters], reverse=True)
results.append(sizes)

# 继续划分直到达到目标簇数量
while len(clusters) < n:
max_sse_reduction = -1
best_idx = -1

# 选择划分后SSE减少最多的簇
for i in range(len(clusters)):
if len(clusters[i]) <= 1:
continue

current_sse = calculate_sse(clusters[i])
new_clusters = kmeans_split(clusters[i])
new_sse = sum(calculate_sse(c) for c in new_clusters)
sse_reduction = current_sse - new_sse

if sse_reduction > max_sse_reduction:
max_sse_reduction = sse_reduction
best_idx = i

# 划分选中的簇
new_clusters = kmeans_split(clusters[best_idx])
clusters = clusters[:best_idx] + new_clusters + clusters[best_idx + 1:]

sizes = sorted([len(c) for c in clusters], reverse=True)
results.append(sizes)

return results

# 主函数
n = int(input())
m = int(input())
points = np.array([list(map(int, input().split())) for _ in range(m)], dtype=float)

results = bi_kmeans(points, n)
for result in results:
print(' '.join(map(str, result)))

```

---

## 2025年10月10日-AI方向

<a id="第2题-p3874"></a>

### 第2题-数据聚类及噪声点识别（P3874）- 中等





DBSCANDBSCANDBSCAN（Density−BasedDensity-BasedDensity−Based SpatialSpatialSpatial ClusteringClusteringClustering of ApplicationsApplicationsApplications withwithwith NoiseNoiseNoise）是一种基于密度的聚类算法。它能够识别出噪声点，发现任意形状的簇，其核心概念包括：

eps-邻域：样本点 P 与 点 Q 的距离小于 eps 的所有样本点的集合，即点 P 周围以 eps 为半径内所有样本点的集合。

核心点：若点 P 的 eps 邻域内的样本点数量大于 min_samplesmin\_samplesmin_samples 的阈值，则称点 P 为核心点。

直接密度可达：若点 P 是核心点，且点 Q 处于点 P 的 eps 邻域内，则称点 Q 由点 P 直接密度可达。

密度可达：若有一串点 P1,P2,...,PnP_1, P_2, ..., P_nP1,P2,...,Pn，对于任意 i，Pi+1P_{i+1}Pi+1 可由 PiP_iPi 直接密度可达，则称点 PnP_nPn 可由点 $P_1$ 密度可达。

密度相连：若存在点 O ，使得点 P 和点 Q 都可由 O 密度可达，则称点 P 和点 Q 是密度相连的。

簇：对于任意的点 P 和点 Q ，若点 P 属于某个簇，且点 Q 由点 P 密度可达，则点 Q 也属于这个簇。同理，由点 Q 密度可达的点也属于这个簇。即同一簇内的点，是密度相连的。

噪声点：不属于样本集内任何簇的样本点。

请你根据以上要求实现 DBSCANDBSCANDBSCAN 算法，要求根据给定的数据集、eps 和 min_samplesmin\_samplesmin_samples 值，输出簇的个数和噪声点的个数。
输入描述
第一行输入 3 个数 eps, min_samplesmin\_samplesmin_samples, x，以空格隔开，分别表示：

eps：计算 eps-邻域的半径值，可为小数
min_samplesmin\_samplesmin_samples：核心点的 eps-邻域内样本点数量阈值，整数
x：数据集行数，即样本数量，整数

第二行开始为输入的样本数据，数值之间用空格隔开，共输入 x 行，每行输入数值数量为 y，2≤y≤32 \leq y \leq 32≤y≤3
输出描述
输出两个值，用空格隔开，第一个值表示得到的簇的个数，第二个值表示识别到的噪声点的个数
样例1
输入
2 2 10
0 0
3 3
6 6
9 9
12 12
2 6
6 2
9 5
5 9
10 2

输出
0 10

说明
eps 设为 2，min_samplesmin\_samplesmin_samples 为 2，但给出的数据集中任意两个点之间的距离都大于2，因此即使 min_samplemin\_samplemin_sample 设置为 2，也没有形成任何簇，即所有的点都是噪声，如下图：

样例2
输入
1 5 20
5.05 1.36
-8.19 -6.47
4.5 2.5
5.01 2.06
4.30 2.28
4.22 1.82
4.58 1.82
4.81 2.46
4.81 1.09
4.80 1.78
5.16 2.44
-6.92 -6.38
-6.84 -7.03
-6.70 -7.20
-6.83 -7.87
-6.47 -6.20
-6.70 -6.11
-6.90 -6.10
-6.99 -6.70
5 5

输出
2 2

说明
样例数据集中共 20 个点，给定 eps 为 1 ， min_samplesmin\_samplesmin_samples 为 5 ，聚类后得到两个簇： 一个在坐标系第一象限，一个在第三象限，且存在两个噪声点 (5,5)(5, 5) (5,5) 和 (−8.19,−6.47)(-8.19, -6.47) (−8.19,−6.47)不属于任何簇，如下图：

▶️


#### 解答


video solution

解题思路
DBSCAN 基于“密度”成簇：对每个点找 eps 邻域（欧氏距离 ≤ eps 的点集合），若邻域样本数 ≥ min_samples，该点为核心点；从未访问的核心点出发，用 BFS/DFS 扩展，将其邻域内的点并入当前簇；若被并入的点本身也是核心点，则继续把它的邻域加入队列，直到不再扩展。最终没有被任何簇吸纳的点即为噪声点。
实现细节：

计算并缓存所有点的邻居列表（两两距离判断，含自身）；维度不写死，自动支持二维或三维输入。
core[i] = (len(neighbors[i]) >= min_samples) 判定核心点。
逐点扫描：若是未访问的核心点，创建新簇并用队列扩展；扩展时把邻居标成当前簇，遇到核心点则把它的邻居继续入队。
统计得到的簇个数与仍为 -1 的样本数（噪声点）。

复杂度分析

设样本数为 n，维度 d∈{2,3}。
预计算邻居：O(n^2 * d)；扩展遍历整体 O(n^2)。
总时间复杂度 O(n^2 * d)；空间复杂度（邻接表）O(n^2)，标记与标签 O(n)。在题目给定规模下可接受。

代码实现

**Python 代码：**

```python
import sys
from collections import deque
from ast import literal_eval

def dist2(a, b):
# 欧氏距离平方，维度自适应（2D/3D均可）
return sum((ai - bi) ** 2 for ai, bi in zip(a, b))

def dbscan(points, eps, min_samples):
n = len(points)
if n == 0:
return 0, 0
eps2 = eps * eps

# 预计算邻居（含自身），阈值用 <=
neighbors = [[] for _ in range(n)]
for i in range(n):
for j in range(n):
if dist2(points[i], points[j]) <= eps2:
neighbors[i].append(j)

core = [len(neighbors[i]) >= min_samples for i in range(n)]
labels = [-1] * n
visited = [False] * n
cluster_id = 0

for i in range(n):
if visited[i]:
continue
visited[i] = True
if not core[i]:
continue

labels[i] = cluster_id
q = deque(neighbors[i])  # 从核心点的邻居开始扩展
in_q = [False] * n
for nb in neighbors[i]:
in_q[nb] = True

while q:
j = q.popleft()
if labels[j] == -1:
labels[j] = cluster_id
if not visited[j]:
visited[j] = True
if core[j]:
for nb in neighbors[j]:
if not in_q[nb]:
q.append(nb)
in_q[nb] = True
cluster_id += 1

noise = sum(1 for v in labels if v == -1)
return cluster_id, noise

def main():
data = sys.stdin.read().strip().splitlines()
if not data:
return
a, b, c = data[0].split()
eps = float(a)
min_samples = int(b)
x = int(c)

points = [list(map(float,data[i].split())) for i in range(1, 1 + x)]
clusters, noise = dbscan(points, eps, min_samples)
print(f"{clusters} {noise}")

if __name__ == "__main__":
main()

```

---

<a id="第3题-p3875"></a>

### 第3题-经典LSTM模型结构实现（P3875）- 中等





【问题说明】长短期记忆网络（LongLongLong Short−TermShort-TermShort−Term MemoryMemoryMemory, LSTMLSTMLSTM）是一种特殊的循环神经网络（RNN），旨在解决传统 RNN 中存在的梯度消失和梯度爆炸问题，使其能够有效地学习长期依赖关系。
一个 LSTMLSTMLSTM 单元（CellCellCell）的核心由三个关键的门和一个细胞状态（CellCellCell StateStateState）组成：
细胞状态 (CellCellCell StateStateState):这是 LSTMLSTMLSTM 的“记忆高速公路"，信息沿着这条路径从一个时间步传递到下一个。它的更新是一个简单的线性操作(加法和乘法)，这使得梯度可以更直接地流动，从而避免了梯度消失。
遗忘门 (ForgetForgetForget GateGateGate):遗忘门决定从上一时间步的细胞状态中丢弃哪些信息。它通过一个 SigmoidSigmoidSigmoid 激活函数，对上一个隐藏状态和当前输入进行处理，输出一个介于 0 和 1 之间的向量。0 表示完全遗忘，1 表示完全保留。
输入门（InputGateInput GateInputGate）：输入门控制新信息写入到细胞状态中。它包含两个部分：

一个 SigmoidSigmoidSigmoid 层，用于决定哪些值需要更新。
一个 TanhTanhTanh 层，用于创建新的候选细胞状态（C ~ t）。

输出门（OutputOutputOutput GateGateGate）：输出门决定当前时刻的隐藏状态（HiddenHiddenHidden StateStateState）将输出哪些信息。它首先通过一个 SigmoidSigmoidSigmoid 层来决定细胞状态的哪些部分会被输出，然后对当前的细胞状态应用 TanhTanhTanh 函数，最后两者相乘得到新的隐藏状态。
【任务要求】请根据下图的 LSTMLSTMLSTM 结构示意图，实现一个 LSTMLSTMLSTM 模型的关键函数，并按下列要求输出计算结果。

该 LSTMLSTMLSTM 模型包含了 5 个 LSTMLSTMLSTM CellCellCell（上图中 A 单元），每个 LSTMLSTMLSTM CellCellCell 中的权重的定义如下图所示，分别为 wf,wi,wg,wowf, wi, wg, wowf,wi,wg,wo, 对应的偏置为 bf,bi,bg,bobf, bi, bg, bobf,bi,bg,bo。已在 pythonpythonpython 代码模板中提供了 5 个 LSTMLSTMLSTM CellCellCell 的权重和偏置的数据。如果使用非 PythonPythonPython 语言, 需沿用 Python3Python3Python3 代码模板中的参数设置。

该 LSTMLSTMLSTM 模型会循环地作用于输入序列中的每一个时间步（从 t=1t=1t=1 到 t=sequence_lengtht=sequence\_lengtht=sequence_length），每个时间步的计算都会产生一个 5 维的隐藏状态。请针对不同输入矩阵运行 LSTMLSTMLSTM 模型，计算对应每个时间步隐藏层状态 h 的首元素。其中输入 X 矩阵的形状为 [4,7][4, 7][4,7] ，即输入数据序列时间步长 sequence_lengthsequence\_lengthsequence_length 为 4 ，输入数据维度 X_dimX\_dimX_dim 为 7 。
输入描述
一共一行数据，用于描述输入矩阵。
其中前两个为整型数据，分别为 sequence_lengthsequence\_lengthsequence_length 行数和 x_dimx\_dimx_dim 列数，后面数据为输入矩阵的参数，均为浮点数，按行平铺 flattenflattenflatten 形式展开为一维序列，数据间以一个空格间隔。
输出描述
一共一行数据，输出每个时间步隐藏状态的首元素，按时间步顺序组成，数据之间以一个空格间隔。
数据精度要求:且均四舍五入精确到小数点后 3 位，同时若尾部存在 0 结尾需进行舍弃如 0.2000.2000.200 0.3100.310 0.310 0.8910.8910.891 0.0070.0070.007 需要舍弃尾部，变为 0.20.20.2 0.310.310.31 0.8910.8910.891 0.0070.0070.007 。特殊情况：0 或 0.0000.0000.0 或 0.0.0.00 或 0.00.00.0 需输出为 0.00.00.0 。
样例1
输入
4 7 -1.153285 -0.081943 0.464549 3.411137 0.594197 1.21088 -0.234899 -0.272196 0.279498 -0.289765 -0.826989 -0.224368 0.711969 -0.067545 0.80226 0.574793 2.458116 0.733628 0.698731 -0.816701 0.533741 -1.756603 -0.123113 -0.550757 0.273727 0.249046 -1.165406 -0.31581

输出
0.001 -0.002 0.012 -0.006

样例2
输入
4 7 -1.609352 -0.165708 -0.494005 1.980481 0.316188 -0.005439 -1.108964 0.576463 -0.048573 -0.384642 -1.112576 0.351411 0.698983 0.607453 0.364154 -0.220041 0.345962 -0.274185 -0.784176 -1.740389 1.118046 0.794949 2.249595 -0.038455 0.037336 -0.652332 1.491228 -0.248807

输出
-0.006 -0.012 -0.013 0.014

说明
样例的输入输出均为一行数据，具体格式及输出规范参考上述输入输出描述。经典LSTM模型结构实现


#### 解答


思路与公式

记忆单元数量（隐藏维度）为 m=5m=5m=5，输入维度为 x_dim=7x\_dim=7x_dim=7，因此每个门的权重矩阵形状均为 (m,  x_dim+m)=(5,12)(m,\; x\_dim+m)=(5,12)(m,x_dim+m)=(5,12)。在每个时间步 t，把当前输入 xt∈R7x_t\in\mathbb{R}^{7}xt∈R7 与上一步隐藏向量 ht−1∈R5h_{t-1}\in\mathbb{R}^{5}ht−1∈R5 级联成 xc=[xt;ht−1]∈R12\mathrm{xc}=[x_t;h_{t-1}]\in\mathbb{R}^{12}xc=[xt;ht−1]∈R12。

经典 LSTM 的前向计算如下（与题面描述一致）：
$$\begin{aligned}
g_t &= \tanh(W_g \, \mathrm{xc} + b_g),\\
i_t &= \sigma(W_i \, \mathrm{xc} + b_i),\\
f_t &= \sigma(W_f \, \mathrm{xc} + b_f),\\
o_t &= \sigma(W_o \, \mathrm{xc} + b_o),\\
s_t &= g_t \odot i_t + s_{t-1} \odot f_t,\\
h_t &= \tanh(s_t) \odot o_t,
\end{aligned}$$其中 σ(⋅)\sigma(\cdot)σ(⋅) 为 Sigmoid，tanh⁡(⋅)\tanh(\cdot)tanh(⋅) 为双曲正切，⊙\odot⊙ 为按元素乘法；初始 s0=0, h0=0s_0=\mathbf{0},\,h_0=\mathbf{0}s0=0,h0=0。

输出：对每个时间步的 hth_tht 取首元素 ht[0]h_t[0]ht[0]，四舍五入到 3 位小数并去除尾随 0；仅当数值为 0 时输出 0.0。

代码
import numpy as np

def sigmoid(x):
return 1 / (1 + np.exp(-x))

class LetterParam:
def __init__(self, mem_cell_ct, x_dim):
self.mem_cell_ct = mem_cell_ct
self.x_dim = x_dim
# Weight Matrices Shape(mem_cell_ct, x_dim + mem_cell_ct)
self.wg = np.array([
[0.009763, 0.043038, 0.020553, 0.008977, -0.015209, 0.000179, -0.012636, 0.017535, -0.022032, 0.06664, 0.06077, 0.02607],
[0.013609, 0.095119, -0.085793, -0.095274, -0.099966, 0.086524, 0.056641, 0.074002, 0.06774, -0.00704, 0.00686, -0.01013],
[-0.076345, 0.027984, -0.071329, 0.088934, 0.00437, -0.017065, -0.047089, 0.046647, -0.06871, 0.00387, -0.00661, -0.01133],
[0.022419, 0.023837, 0.08375, 0.036964, -0.028098, -0.012594, 0.039626, -0.03785, 0.03383, 0.00188, -0.00723, -0.06378],
[-0.036914, -0.027258, 0.014093, -0.01228, 0.007675, -0.078691, -0.058225, -0.05922, -0.04942, -0.06378, -0.01133, 0.00689]
], dtype=float)

self.wi = np.array([
[-0.012801, -0.094815, 0.009932, -0.012938, -0.015208, -0.03393, -0.06907, 0.02384, -0.04069, -0.04695, 0.04227, 0.00689],
[-0.073084, 0.002716, -0.063112, 0.057087, 0.017075, -0.001143, 0.06291, -0.06470, 0.00196, -0.06943, -0.04476, -0.00694],
[-0.074568, 0.019349, -0.054798, -0.076611, -0.053938, -0.030035, -0.00644, -0.06951, 0.02981, -0.03384, 0.00647, -0.02581],
[0.058727, 0.016001, -0.06754, 0.04015, 0.09291, 2e-06, 0.077904, -0.031577, 0.01303, -0.01410, -0.01398, 0.05501],
[0.007121, 0.090749, 0.005842, -0.053581, -0.025732, 0.07017, -0.018745, -0.09456, -0.05945, -0.06571, 0.06871, 0.00411]
], dtype=float)

self.wf = np.array([
[-0.084738, 0.055984, -0.012318, 0.044693, 0.065598, 0.007089, 0.000224, -0.06559, -0.04612, -7.3e-05, 0.03686, 0.06116],
[-0.023812, -0.086313, -0.042371, 0.081919, -0.057323, -0.009575, 0.086241, -0.05902, 0.03011, 0.00626, -0.05909, 0.00688],
[0.081826, -0.073366, 0.004863, 0.050082, 0.033803, -0.005449, -0.05630, -0.00347, -0.02532, -0.0456, -0.02682, 0.03758],
[0.05373, -0.037201, 0.014525, -0.04479, -0.009431, -0.029404, 0.03148, -0.02329, -0.03618, 0.03586, -0.01707, 0.01834],
[-0.06391, 0.048224, -0.015252, -0.014709, 0.028876, 0.004581, -0.017023, -0.09715, -0.03168, 0.04157, 0.04680, 0.05221]
], dtype=float)

self.wo = np.array([
[0.022434, -0.066186, -0.012788, 0.053852, -0.040935, -0.070167, -0.05504, -0.01525, -0.05224, -0.05249, 0.03916, -0.05965],
[-0.083761, 0.03392, 0.024249, -0.045149, -0.006756, -0.076326, -0.085208, 0.08015, 0.05878, 0.06814, 0.05341, 0.06931],
[0.015455, 0.062753, -0.015736, -0.09451, -0.009173, -0.079835, 0.063444, 0.039546, 0.03367, -0.05155, 0.09695, 0.07272],
[0.023083, -0.002993, -0.015995, 0.04557, -0.035437, -0.019891, -0.039207, 0.08944, 0.08376, 0.06708, -0.09135, 0.06871],
[0.090088, 0.061318, -0.003744, 0.093352, -0.016804, -0.036232, -0.096711, -0.05294, -0.06852, -0.01468, -0.03823, -0.06719]
], dtype=float)

# bias terms
self.bg = np.array([-0.017119, -0.010762, -0.01027, -0.075269, -0.065529], dtype=float)
self.bi = np.array([0.075116, 0.059407, 0.049271, -0.074094, 0.054991], dtype=float)
self.bf = np.array([0.018351, -0.01307, -0.014564, 0.009966, 0.066618], dtype=float)
self.bo = np.array([-0.054807, -0.077083, -0.014593, 0.047107, 0.007309], dtype=float)

class LstmState:
def __init__(self, mem_cell_ct, x_dim):
self.g = np.zeros(mem_cell_ct, dtype=float)
self.i = np.zeros(mem_cell_ct, dtype=float)
self.f = np.zeros(mem_cell_ct, dtype=float)
self.o = np.zeros(mem_cell_ct, dtype=float)
self.s = np.zeros(mem_cell_ct, dtype=float)
self.h = np.zeros(mem_cell_ct, dtype=float)

class LstmNode:
def __init__(self, lstm_param, lstm_state):
# store reference to parameters and to activations
self.state = lstm_state
self.param = lstm_param
# non-recurrent input concatenated with recurrent input
self.xc = None

class LstmNetwork():
def __init__(self, lstm_param):
self.lstm_param = lstm_param
self.lstm_node_list = []
# input sequence
self.x_list = []

def x_list_clear(self):
self.x_list = []

def x_list_add(self, x):
self.x_list.append(x)

def forward(self):
mem_cell_ct = self.lstm_param.mem_cell_ct
h_prev = np.zeros(mem_cell_ct, dtype=float)
s_prev = np.zeros(mem_cell_ct, dtype=float)
h_list = []

for t in range(len(self.x_list)):
x = self.x_list[t]
xc = np.hstack((x, h_prev))

g = np.tanh(np.dot(self.lstm_param.wg, xc) + self.lstm_param.bg)
i = sigmoid(np.dot(self.lstm_param.wi, xc) + self.lstm_param.bi)
f = sigmoid(np.dot(self.lstm_param.wf, xc) + self.lstm_param.bf)
o = sigmoid(np.dot(self.lstm_param.wo, xc) + self.lstm_param.bo)

s = f * s_prev + i * g
h = o * np.tanh(s)

h_list.append(h.copy())
h_prev = h
s_prev = s

return h_list

def format_float(x):
s = f"{x:.3f}"
if '.' in s:
s = s.rstrip('0').rstrip('.')
if s == '' or s == '-0':
return '0.0'
if '.' not in s:
s += '.0'
return s

def func():
data = list(map(float, input().split()))
seq_len = int(data[0])
x_dim = int(data[1])
vals = data[2:]
x_list = np.array(vals, dtype=float).reshape(seq_len, x_dim)

mem_cell_ct = 5
lstm_param = LetterParam(mem_cell_ct, x_dim)
lstm_net = LstmNetwork(lstm_param)

for i in range(seq_len):
lstm_net.x_list_add(x_list[i])

h_list = lstm_net.forward()
first_elems = [h[0] for h in h_list]
formatted = [format_float(x) for x in first_elems]
print(' '.join(formatted))

if __name__ == "__main__":
func()


---

## 2025年10月10日(留学生)-AI岗

<a id="第2题-p3871"></a>

### 第2题-磁盘故障检测的特征工程（P3871）- 困难





输入是磁盘检测中使用的典型的 "SMARTSMARTSMART" 数据集的部分信息，请根据下面的特征工程，提取相关特征。
特征工程的处理过程如下：

读取输入数据：

从标准输入读取一行数据，将其拆分成一个浮点数列表。

计算统计特征指标：

对个数据特征，计算以下统计指标：

均值 (MeanMeanMean)：计算每列数据的平均值。

最大值 (Max)：找出每列中的最大值。

最小值 (Min)：找出每列中的最小值。

极差 (Ptp)：计算每列的最大值与最小值之差。

标准差 (Std)：计算每列数据的标准差，反映数据的高程度。

方差 (Var)：计算每列数据的方差，反映数据的离散程度。

偏度 (SkewSkewSkew)：计算数据的偏度，反映数据分布的不对称性。

峰度 (KurtKurtKurt)：计算数据的峰度，反映数据分布的程度。
统计指标计算公式：
1.均值 (MeanMeanMean)：计算列据的平均值。
公式：means(summeans(summeans(sum of all values)/numbervalues)/numbervalues)/number of valuesvaluesvalues
2.最大值 (Max)：找出每列中的最大值。
3.最小值 (Min)：找出每列中的最小值。
4.极差 (Ptp)：计算每列的最大值与最小值之差。
公式：ptp=max−minptp=max-minptp=max−min
5.标准差 (Std) ：计算每列数据的标准差，反映数据的离散程度。
公式：std=sqrt(variance)std=sqrt(variance)std=sqrt(variance)
6.方差 (Var)：计算每列数据的方差，反映数据的离散程度。
公式：variance=meanvariance = meanvariance=mean of squaredsquaredsquared difierencesdifierencesdifierences fromfromfrom the MeanMeanMean
7.偏度 (SkewSkewSkew)：计算数据的偏度，反映数据分布的不对称性。
公式：skewness=(sum((x−mean)3)/n)/std3skewness =(sum((x-mean)^3)/n) /std^3skewness=(sum((x−mean)3)/n)/std3
8.峰度 (KurtKurtKurt)：计算数据的峰度，反映数据分布的陡峭程度。
公式：kurtosis=(sum((x−mean)4)/n)/std4−3kurtosis=(sum((x-mean)^4)/n)/std^{4}-3kurtosis=(sum((x−mean)4)/n)/std4−3

输出结果：

将所有统计指标按顺序排列

输入描述
输入数据为一行，包含多个样本信息，其中米格样本是 19 个浮点数，代表不同的存储设备指标。
每个样本的数据描述如下：
数据含义 说明
时间戳 UnixUnixUnix 时间戳(秒级或毫秒级)
容量 存储设备的总容量(字节)
已用容量 存储设备的已用容量(字节)
空闲容量 存储设备的空闲容量(字节)
读取操作计数 单位时间内完成的读取操作次数
写入操作计数 单位时间内完成的写入操作次数
读取吞吐量 读取数据的吞吐量(字节/秒)
写入吞吐量 写入数据的吞吐量(字节/秒)
读取延迟 读取操作的平均延迟(毫秒)
写入延迟 写入操作的平均延迟(毫秒)
读取错误计数 单位时间内发生的读取错误次数
写入错误计数 单位时间内发生的写入错误次数
硬盘温度 硬盘的当前温度(摄氏度)
转速 硬盘的转速 (RPM)(RPM)(RPM)
读取带宽 读取操作的带宽 (MB/s)(MB/s)(MB/s)
写入带宽 写入操作的带宽 (MB/s)(MB/s)(MB/s)
读取队列深度 读取操作的队列深度
写入队列深度 写入操作的队列
深度硬盘健康状态 硬盘的健康状态
输出描述
输出结果格式：
1、统计结果以一行输出，包含所有统计指标，按以下题序列，便用空格作为分隔符：
mean_0 max_0 min_0 ptp_0 std_0 var_0 skew_0 kurt 0 mean_1 max_1 min_1 ptp_1 std_1 var_1 skew_1 kurt_1... mean_18 max_18 min_18 ptp_18 std_18 var_18 skew_18 kurt_18
2、每个统计指标保留 2 位小数，便于阅读和理解。
样例1
输入
1623456000 100.0 800000.0 200000.0 100.0 200.0 500.0 1000.0 10.0 20.0 0.0 0.0 40.0 7200.0 50.0 100.0 5.0 10.0 0.0 1623456001 100.0 800000.0 200000.0 100.0 200.0 500.0 1000.0 10.0 20.0 0.0 0.0 40.0 7200.0 50.0 100.0 5.0 10.0 0.0 1623456002 100.0 800000.0 200000.0 100.0 200.0 500.0 1000.0 10.0 20.0 0.0 0.0 40.0 7200.0 50.0 100.0 5.0 10.0 0.0 1623456003 100.0 800000.0 200000.0 100.0 200.0 500.0 1000.0 10.0 20.0 0.0 0.0 40.0 7200.0 50.0 100.0 5.0 10.0 0.0 1623456004 100.0 800000.0 200000.0 100.0 200.0 500.0 1000.0 10.0 20.0 0.0 0.0 40.0 7200.0 50.0 100.0 5.0 10.0 0.0

输出
1623456002.00 1623456004.00 1623456000.00 4.00 1.41 2.00 0.00 -1.30 100.00 100.00 100.00 0.00 0.00 0.00 0.00 0.00 800000.00 800000.00 800000.00 0.00 0.00 0.00 0.00 0.00 200000.00 200000.00 200000.00 0.00 0.00 0.00 0.00 0.00 100.00 100.00 100.00 0.00 0.00 0.00 0.00 0.00 200.00 200.00 200.00 0.00 0.00 0.00 0.00 0.00 500.00 500.00 500.00 0.00 0.00 0.00 0.00 0.00 1000.00 1000.00 1000.00 0.00 0.00 0.00 0.00 0.00 10.00 10.00 10.00 0.00 0.00 0.00 0.00 0.00 20.00 20.00 20.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 40.00 40.00 40.00 0.00 0.00 0.00 0.00 0.00 7200.00 7200.00 7200.00 0.00 0.00 0.00 0.00 0.00 50.00 50.00 50.00 0.00 0.00 0.00 0.00 0.00 100.00 100.00 100.00 0.00 0.00 0.00 0.00 0.00 5.00 5.00 5.00 0.00 0.00 0.00 0.00 0.00 10.00 10.00 10.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00

说明
输入样本是 10 个，每个样本的内容是 19 列特征：样本 1−1-1− 特征 0 样本 2−2-2− 特征 2 ... 样本 1 特征 18 样本 2−2-2− 特征 0 ... 样本 2−2-2− 特征 18 ... 样本 10−10-10− 特征 0 样本 10−10-10− 特征 18
输出是每个特征的度量值：特征 0−0-0− 均值 特征 0−0-0− 最大值 ... 特征 0−0-0− 峰度 特征 1−1-1− 均值 特征 1−1-1− 最大值 … 特征 1−1-1− 峰度 ... 特征 18−18-18− 均值 特征 18−18-18− 最大值 … 特征 18−18-18− 峰度
提示
输入样本解释：
1、实际用例中，输入无注释，仅数值
2、实际用例中，多个样本拼接成一行输入，用户需要自己隔离样本( 19 个特征为一个样本)
输出结果解释：
1、实际输出中，仅需保留数值，(参考样例的输出解释)
2、结果显示小数点后 2 位


#### 解答


解题思路

输入解析：从标准输入读取一整行（或多行拼接）数字序列。每 19 个数字为一个样本，共有 N 个样本。将数据按列重排，得到 19 列，每列长度为 N。

Python：优先尝试 literal_eval 解析为列表，失败则按空白分割；同时兼容逗号分隔。
C++/Java：读取整段文本，替换换行/逗号为空格后用输入流解析为 double。

统计指标（逐列计算，采用总体定义，即分母为 n）：

均值：μ=1n∑xi\mu=\frac{1}{n}\sum x_iμ=n1∑xi
最大值 / 最小值
极差：ptp=max⁡−min⁡\mathrm{ptp}=\max-\minptp=max−min
方差：σ2=1n∑(xi−μ)2\sigma^2=\frac{1}{n}\sum (x_i-\mu)^2σ2=n1∑(xi−μ)2
标准差：σ=σ2\sigma=\sqrt{\sigma^2}σ=σ2
偏度（总体三阶标准化矩）：

峰度（Fisher过度峰度，均匀分布为负）

当 σ=0\sigma=0σ=0（整列常数）时，规定 skew=0、kurt=0。
输出：按列从 0 到 18，依次输出
mean_i max_i min_i ptp_i std_i var_i skew_i kurt_i，保留两位小数并以空格分隔。

复杂度分析

设样本数为 N，特征列数固定为 F=19F=19F=19。
时间复杂度：两趟扫描/列（先均值，再高阶矩）为 O(NF)O(NF)O(NF)，其中 F 常数，可视为 O(N)O(N)O(N)。
空间复杂度：除读入数据外，仅用常数级临时变量，O(1)O(1)O(1)（若按列临时拷贝，仍为 O(N)O(N)O(N) 以内）。

代码实现

**Python 代码：**

```python
# -*- coding: utf-8 -*-
# 题意：从 stdin 读入若干浮点数，每 19 个为一条样本，按列计算 8 项总体统计量并输出两位小数。
import sys, math
from ast import literal_eval

COLS = 19

def compute_col_stats(col):
"""对单列数据计算 mean, max, min, ptp, std, var, skew, kurt（总体 & Fisher 过度峰度）"""
n = len(col)
# 基本量
s = sum(col)
mu = s / n
cmax = max(col)
cmin = min(col)
ptp = cmax - cmin
# 二/三/四阶中心矩
m2 = 0.0
m3 = 0.0
m4 = 0.0
for x in col:
d = x - mu
d2 = d * d
m2 += d2
m3 += d2 * d
m4 += d2 * d2
var = m2 / n
std = math.sqrt(var)
if std == 0.0:
skew = 0.0
kurt = 0.0
else:
skew = (m3 / n) / (std ** 3)
kurt = (m4 / n) / (std ** 4) - 3.0
return (mu, cmax, cmin, ptp, std, var, skew, kurt)

def main():
text = sys.stdin.read().strip()
nums = []
if not text:
print("")  # 无输入时输出空行
return
# 优先尝试 literal_eval（如输入为 Python 列表/嵌套列表）
parsed = None
try:
parsed = literal_eval(text)
except Exception:
parsed = None

if isinstance(parsed, (list, tuple)):
# 可能是嵌套列表，做扁平化
def flat(it):
for v in it:
if isinstance(v, (list, tuple)):
for u in flat(v):
yield float(u)
else:
yield float(v)
nums = list(flat(parsed))
else:
# 兼容逗号/换行/多空格
text = text.replace(',', ' ')
for tok in text.split():
nums.append(float(tok))

# 计算样本数
if len(nums) % COLS != 0:
# 默认输入合法；这里宽容处理，截断到最近的整样本
total = (len(nums) // COLS) * COLS
nums = nums[:total]
n = len(nums) // COLS

# 按列计算
out = []
for j in range(COLS):
col = [nums[j + i * COLS] for i in range(n)]
out.extend(compute_col_stats(col))

# 按要求格式化输出（两位小数，空格分隔）
print(' '.join(f"{x:.2f}" for x in out))

if __name__ == "__main__":
main()

```

---

<a id="第3题-p3872"></a>

### 第3题-基于逻辑回归的意图分类器（P3872）- 中等





意图分类是一种常见的 NLP 任多，现在需要实现一个基于逻辑回归的意图分类系统(二分类)。本系统的输入是一个已处理的特征序列，输出是意图标签 ( 0 或 1 )。请根据以下步骤实现该分类器：
1、数据预处理：对输入(特征序列，即仅包含大写字母 ABCDEFGABCDEFGABCDEFG 的序列)进行 one−hotone-hotone−hot 编码，编码时的特征顺序是 ABCDEFGABCDEFGABCDEFG；某个特征存在取 1 ，不存在取 0 ;
2、模型初始化：构造一个单层的训练网络，初始化权重 w 和偏置 b 初始化的值全部设为 0 ;
3、模型训练：使用 SigmoidSigmoidSigmoid 函数计算预测值，使用交叉熵损失函数训练，使用梯度下降算法更新参数；训练过程中使用训练数据进行训练，相关超参数为：学习率 0.10.10.1，训练轮数 20 ，batchbatchbatch 大小为 1 ;
（1）SigmoidSigmoidSigmoid 函数公式:
σ(z)=11+e−zσ(z)=\frac{1}{1+e^{-z}}σ(z)=1+e−z1 ，其中 z=w⋅x+bz=w·x+bz=w⋅x+b （ x 为输入特征向量）。
4.模型预测：对测试数据进行预测，首先使用 sigmoidsigmoidsigmoid 函数计算预测值，然后做二值化处理(大于 0.50.50.5 认为是类别 1 ，否则是类别 0 )
输入描述
第一行输入 N 和 M，N(10<=N<=100)M，N(10<=N<=100)M，N(10<=N<=100) 表示训练数据条数，M(2<=M<=10)M(2<=M<=10)M(2<=M<=10) 表示测试数据条数；
接下来 N 行是训练数据，每行包含两部分内容(空格隔开)，即代表输入的特征序列(由大写字母 ABCDEFGABCDEFGABCDEFG 构成的字符串，长度范围是 [3,7] )和该条数据的意图标签(使用数字 0 或 1 表示)；
在接下来 M 行是测试数据的特征序列。
输出描述
输出有 M 行，每行是测试输出，即用数字 0 或 1 表示的意图标签。
样例1
输入
10 2
CBG 0
AFE 0
FGD 1
BFG 0
BBA 0
BDD 0
BEG 1
EGE 0
CAF 0
DGD 1
DBA
DAD

输出
0
0

说明
该样例有 10 条训练数据，2 条测试数据，2 个测试输出均 0
样例2
输入
10 3
GDEE 0
BDFEA 1
BDFE 0
GECD 0
DDCEE 1
ADA 0
EECBC 0
BACBC 1
D 1
FEE 0
AADC
BBAE
ECEC

输出
1
0
0

说明
该样例有 10 条训练数据，3 条测试数据，测试输出分别为 1、0、01、0、01、0、0 。


#### 解答


解题思路

将每条样本的“特征序列”看成由大写字母 A~G 组成的集合。按固定顺序 A B C D E F G 做 one-hot 编码：只要某个字母在序列中出现（次数不计），对应位置记为 1，否则为 0，得到长度为 7 的特征向量。
训练一个单层逻辑回归（Logistic Regression） 分类器：
预测值 p = sigmoid(w·x + b)；损失函数采用交叉熵；用**梯度下降（SGD，batch=1）**更新参数。题面给定超参：学习率 0.1，训练轮数 20，初始 w、b 全为 0。
预测阶段：对测试数据计算 p，阈值 0.5 二值化，p>0.5 输出 1，否则输出 0。
读入格式：第一行 N M；接着 N 行为训练样本（“字符串 标签”）；之后 M 行为仅含字符串的测试样本；输出 M 行预测标签。

复杂度分析

设训练样本数为 N、特征维度为 d=7、轮数为 E=20。

时间复杂度：O(E * N * d)，题目范围下极小。
空间复杂度：O(d)，仅存放权重与一个样本向量。

代码实现

**Python 代码：**

```python
# -*- coding: utf-8 -*-
# 逻辑回归 + one-hot(A~G) 的ACM风格实现
import sys, math

# 将字母串编码为长度7的one-hot（出现即1，不计次数）
def encode(seq):
x = [0.0] * 7
seen = set(seq.strip())
for ch in seen:
idx = ord(ch) - ord('A')
if 0 <= idx < 7:
x[idx] = 1.0
return x

# 训练：SGD，交叉熵损失，学习率0.1，轮数20
def train(X, y, lr=0.1, epochs=20):
w = [0.0] * 7
b = 0.0
for _ in range(epochs):
for xi, yi in zip(X, y):
z = sum(w[j] * xi[j] for j in range(7)) + b
p = 1.0 / (1.0 + math.exp(-z))
dz = p - yi  # 交叉熵对z的导数
# 参数更新（batch=1）
for j in range(7):
w[j] -= lr * dz * xi[j]
b -= lr * dz
return w, b

# 预测：p>0.5 -> 1，否则0
def predict(w, b, xi):
z = sum(w[j] * xi[j] for j in range(7)) + b
p = 1.0 / (1.0 + math.exp(-z))
return 1 if p > 0.5 else 0

def main():
data = sys.stdin.read().strip().split()
if not data:
return
it = iter(data)
N = int(next(it)); M = int(next(it))

X, y = [], []
# 读取N条训练数据
for _ in range(N):
seq = next(it)
label = int(next(it))
X.append(encode(seq))
y.append(label)

# 训练
w, b = train(X, y, lr=0.1, epochs=20)

# 读取并预测M条测试数据
outs = []
for _ in range(M):
seq = next(it)
xi = encode(seq)
outs.append(str(predict(w, b, xi)))

print("\n".join(outs))

if __name__ == "__main__":
main()

```

---

## 2025年9月28日-AI方向

<a id="第2题-p3842"></a>

### 第2题-Yolo检测器中的anchor聚类（P3842）- 中等





【背景信息】YOLO (You Only Look Once) 系列算法在目标检测领域采用了基于Anchor的机制。Anchor是预定义在图像上的一组固定尺寸和比例的参考框，在特征图的每个位置上预设多个Anchor框作为物体位置和尺寸预测的基准。通过模型预测Anchor与真实框的偏移量(Δx,Δy,Δw,Δh)(\Delta x, \Delta y, \Delta w, \Delta h)(Δx,Δy,Δw,Δh)，而非直接输出坐标，避免了直接回归绝对坐标的困难。
【任务目标】
基于k-means聚类算法生成YOLO目标检测中的Anchor框：给定N个检测框的宽和高，聚类得到K个Anchor尺寸，并按照面积从大到小的顺序输出Anchor尺寸。
【任务目标】
基于k-means聚类算法生成YOLO目标检测中的Anchor框：给定N个检测框的宽和高，聚类得到K个Anchor尺寸，并按照面积从大到小的顺序输出Anchor尺寸。
聚类的流程如下：
初始化：为保证聚类结果的稳定性，采用稳定初始化策略，直接取前K个框作为初始中心。
分配阶段：计算每个框到所有聚类中心的距离，分配到最近的中心。
更新阶段：计算每个簇中所有框的宽高均值作为新的聚类中心（在每次迭代计算聚类中心时，均进行向下取整）。
迭代终止条件：当达到设定迭代次数T或新旧聚类中心之间的d值之和小于1e-4时停止迭代。
注：聚类使用 d=1−IOUd = 1 - IOUd=1−IOU 作为距离度量，d和IOU的计算均使用浮点数。
其中IOU的核心公式为交并面积比，即
IOU=交集面积并集面积IOU = \frac{\text{交集面积}}{\text{并集面积}}
IOU=并集面积交集面积
对于检测框B1(w1,h1)B1(w_1,h_1)B1(w1,h1) 和 B2(w2,h2)B2(w_2,h_2)B2(w2,h2)，交集面积计算为：
$$intersection = \min(w_1, w_2) \times \min(h_1, h_2)$$并集面积则为两框总面积减去交集：
$$union = w_1 \times h_1 + w_2 \times h_2 - intersection$$最终：IOU = intersection/(union + 1e-16) (加极小值避免除零)
输入描述
第一行：N,K,TN, K, TN,K,T，以空格分开。
其中：

N 为训练集中检测框的个数，10≤N≤8010 \leq N \leq 8010≤N≤80

K 为聚类中心个数，3≤K≤93 \leq K \leq 93≤K≤9

T 为聚类迭代次数，2≤T≤302 \leq T \leq 302≤T≤30

接下来 N 行：每行为检测框的宽与高，用空格分开。
输出描述
按照聚类中心的面积从大到小的顺序，输出聚类后的中心。
（聚类中心的面积 = 聚类中心的宽 ×\times× 聚类中心的高）
样例1
输入
12 4 20
12 23
34 21
43 23
199 23
34 23
108 12
200 107
12 78
123 110
34 23
56 48
78 66

输出
133 94
121 27
36 22
12 50

说明

输入第一行为 12  4  2012 \; 4 \; 2012420，代表12个检测框，要聚成4类，最大迭代次数为20，接下来的12行是检测框的宽与高。

取前4个框 [12,23],[34,21],[43,23],[199,23][12, 23], [34, 21], [43, 23], [199, 23][12,23],[34,21],[43,23],[199,23] 作为初始聚类中心。

迭代更新计算聚类中心，注意每次迭代时聚类中心都做向下取整。

按照聚类中心的面积（宽 × 高）从大到小排序，输出4个Anchor聚类中心。

样例2
输入
12 3 10
12 23
34 21
43 23
199 23
34 23
108 12
200 107
12 78
123 110
34 23
56 48
78 66

输出
150 76
51 25
12 50

说明

输入第一行为 12  3  1012 \; 3 \; 1012310，代表12个检测框，要聚成3类，最大迭代次数为10，接下来的12行是检测框的宽与高。

取前3个框 [12,23],[34,21],[43,23][12, 23], [34, 21], [43, 23][12,23],[34,21],[43,23] 作为初始聚类中心。

迭代更新计算聚类中心，注意每次迭代时聚类中心都做向下取整。

按照聚类中心的面积（宽 × 高）从大到小排序，输出3个Anchor聚类中心。

提示
注：每次迭代的距离度量 d 和交并比 IOU 都是用浮点数计算，但每次迭代和最终输出的聚类中心都要做向下取整。

▶️


#### 解答


video solution

解题思路
本题要求用 K-means 在宽高空间对检测框进行聚类，以获得 YOLO 的 Anchor 尺寸。与欧氏距离不同，这里采用 d = 1 − IOU 作为距离度量，能更贴近目标检测中对宽高匹配的要求。
算法选择与要点

初始化（稳定初始化）：直接取前 K 个框作为初始聚类中心（宽、高）。

分配阶段：对每个样本框 (w,h)(w, h)(w,h) 计算到所有中心 (Wk,Hk)(W_k, H_k)(Wk,Hk) 的距离
d=1−IOU((w,h),(Wk,Hk))d = 1 - \text{IOU}((w,h),(W_k,H_k))
d=1−IOU((w,h),(Wk,Hk))
其中
$$\text{IOU}=\frac{\min(w,W_k)\cdot \min(h,H_k)}{w\cdot h + W_k\cdot H_k - \min(w,W_k)\cdot \min(h,H_k) + 1e-16}$$将样本分配给距离最小的簇。

更新阶段：对每个簇计算所有成员在宽、高上的均值并向下取整（floor），作为新的中心。如果某簇为空，则保留原中心不变。

终止条件：最多迭代 T 次；或当新旧中心配对的 d 值之和
$$\sum_{k=1}^{K} \left(1-\text{IOU}\big((W_k^{old},H_k^{old}),(W_k^{new},H_k^{new})\big)\right) < 1e-4$$时提前停止。

输出：将最终 K 个中心按面积（宽×高）从大到小排序输出。
注意：迭代中每次更新的中心与最终输出都要向下取整，但 IOU 与 d 的计算始终用浮点数（避免精度问题，分母加 1e−161e{-16}1e−16）。

该流程等价于在“宽高-IOU”空间上的 K-means 变体，常用于 Anchor 聚类以提升先验框与真实框的匹配度。
代码实现

**Python 代码：**

```python
import sys
import math

# 计算两个宽高框的 IOU（基于宽高，无位置信息）
def iou_wh(w1, h1, w2, h2):
inter = min(w1, w2) * min(h1, h2)
union = w1 * h1 + w2 * h2 - inter
return inter / (union + 1e-16)

# 使用 d = 1 - IOU 的 K-means 聚类，返回最终中心（整数）
def kmeans_anchors(boxes, K, T):
# 初始化：取前 K 个框为初始中心（转为浮点便于计算）
centers = [(float(boxes[i][0]), float(boxes[i][1])) for i in range(K)]
n = len(boxes)

for _ in range(T):
# 分配阶段：为每个样本选择最近中心
assign = [0] * n
for i, (w, h) in enumerate(boxes):
best_k = 0
best_d = 1.0 - iou_wh(w, h, centers[0][0], centers[0][1])
for k in range(1, K):
d = 1.0 - iou_wh(w, h, centers[k][0], centers[k][1])
if d < best_d:
best_d = d
best_k = k
assign[i] = best_k

# 更新阶段：计算新中心（对均值向下取整）
sums = [[0.0, 0.0] for _ in range(K)]
cnts = [0] * K
for (w, h), k in zip(boxes, assign):
sums[k][0] += w
sums[k][1] += h
cnts[k] += 1

new_centers = []
for k in range(K):
if cnts[k] == 0:
# 空簇：保留原中心
new_centers.append((centers[k][0], centers[k][1]))
else:
W = math.floor(sums[k][0] / cnts[k])
H = math.floor(sums[k][1] / cnts[k])
new_centers.append((float(W), float(H)))

# 终止条件：新旧中心 d 值之和
change = 0.0
for k in range(K):
change += (1.0 - iou_wh(centers[k][0], centers[k][1], new_centers[k][0], new_centers[k][1]))
centers = new_centers
if change < 1e-4:
break

# 最终结果向下取整并按面积从大到小排序
final_centers = [(int(math.floor(w)), int(math.floor(h))) for (w, h) in centers]
final_centers.sort(key=lambda x: x[0] * x[1], reverse=True)
return final_centers

def main():
data = sys.stdin.read().strip().split()
if not data:
return
it = iter(data)
N = int(next(it)); K = int(next(it)); T = int(next(it))
boxes = []
for _ in range(N):
w = float(next(it)); h = float(next(it))
boxes.append((w, h))

centers = kmeans_anchors(boxes, K, T)
out = []
for w, h in centers:
out.append(f"{w} {h}")
print("\n".join(out))

if __name__ == "__main__":
main()

```

---

<a id="第3题-p3843"></a>

### 第3题-Masked Multi-Head Self-Attention 实现（P3843）- 困难





在Transformer模型中，Multi-Head Self-Attention是核心组件，用于捕捉序列中的依赖关系。你需要从头实现一个Masked Multi-Head Self-Attention函数，支持自注意力（即queries、keys和values来自同一输入序列），并处理编码（mask）以防止未来位置的信息泄露（常见于Decoder中）。
具体要求：

支持多头注意力：将注意力机制并行分成多个"头"，每个头学习不同的注意力模式，增强模型对多维度特征的捕捉能力。
计算过程：

生成Q、K、V矩阵
对输入序列X（维度：[batch_size, seq_len, d_model]）通过3个线性层分别生成查询（Query, Q）、键（Key, K）、值（Value, V）矩阵：（Q=X⋅WQQ = X \cdot W_QQ=X⋅WQ，K=X⋅WKK = X \cdot W_KK=X⋅WK，V=X⋅WVV = X \cdot W_VV=X⋅WV），其中 $W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_{model}}$。
将Q、K、V拆分为多个头
将Q、K、V分割为num_heads个并行的子矩阵（每个头的维度为d_k = d_model / num_heads）。
分割后维度为[batch_size, num_heads, seq_len, d_k]。
对于每个头，计算注意力分数：attention_scores = (  Q⋅KTQ \cdot K^TQ⋅KT ) / sqrt(d_k)。
提供mask（一个(batch_size, seq_len, seq_len)的布尔数组，其中True表示需要掩码的位置），则将masked位置的注意力分数设置为负无穷（-inf），以确保softmax后为0。掩码后的分数为masked_scores。
对掩码后的分数应用softmax得到注意力权重。
softmax_scores=softmax(masked_scores)。
计算注意力输出：attention=softmax_scores  ·  V。
拼接多头输出，并通过一个线性投影得到最终结果。
$output =
concat(attention_1, ..., attention_{num_heads}) · W_O$
，其中 WO∈Rdmodel×dmodel W_O \in \mathbb{R}^{d_{model} \times d_{model}} WO∈Rdmodel×dmodel 是可学习参数，输出维度为
[batch_size, seq_len, d_model].

注意：
1、需处理批次（batch_size > 1）和变长序列。
2、输入参数以分号分隔。第一个参数为多头数量num_heads；
第二个参数为Q矩阵；第三个参数为K矩阵；第四个参数为V矩阵；第五个参数为 WO W_O WO。
3、输出为List，需要将np.ndarray转为List
输入描述
以";"分隔，分别为 num_heads, X, Q、K、V，WOW_OWO
输出描述
输出为最终结果 outputoutputoutput，输出保留两位有效小数，并且为 List。
样例1
输入
2;[[[ 1.92, 1.48], [0.67, -1.23], [0.35, -0.68]], [[-1.11, 0.09], [-0.3, -0.39], [-0.59, -0.06]]];[[1.0, 2.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]]

输出
[[[14.64, 14.64], [-5.36, -5.36], [-4.44, -4.44]], [[-2.79, -2.79], [-3.04, -3.04], [-2.79, -2.79]]]

样例2
输入
2;[[[ 1.92, 1.48], [0.67, -1.23], [0.35, -0.68]], [[-1.11, 0.09], [-0.3, -0.39], [-0.59, -0.06]]];[[1.0,1.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]]

输出
[[[14.64, 14.64], [-5.37, -5.37], [-4.62, -4.62]], [[-2.79, -2.79], [-3.03, -3.03], [-2.77, -2.77]]]

提示

手动实现softmax：exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))；softmax = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)。确保数值稳定性，减去每行最大值
使用np.around(np.ndarray, 2)将输出保留2位小数
通过下三角矩阵实现序列掩码mask，确保每个位置只能关注自身及之前的位置。下三角为1，上三角为0。
处理-inf：可以使用np.where(mask == 0, -np.inf, attention_scores)


#### 解答


解题思路
本题要求手写「带因果掩码」的多头自注意力（Decoder常用），输入为：

num_heads
X：形状 [batch_size, seq_len, d_model]
Q, K, V, W_O：均为形状 [d_model, d_model] 的线性投影矩阵（对应 $W_Q, W_K, W_V, W_O$）

整体流程（Scaled Dot-Product Attention + 多头并行）：

线性映射生成 Q/K/V
Q=XWQ,K=XWK,V=XWVQ = X W_Q,\quad K = X W_K,\quad V = X W_V
Q=XWQ,K=XWK,V=XWV
维度：[B, S, d_model]。

分头
将最后一维 d_model 均分为 num_heads 个头，每头维度 d_k = d_model / num_heads，并重排为
Q_h, K_h, V_h ∈ [B, H, S, d_k]。

每头计算注意力分数
$$\text{scores} = \frac{Q_h K_h^\top}{\sqrt{d_k}}\quad\in [B,H,S,S]$$

因果掩码（防未来信息泄露）
构造 下三角 Mask（[S,S]，下三角为1，上三角为0），广播到 [B,H,S,S]。
将上三角（不允许关注的）位置置为 −∞-\infty−∞：
$$\text{masked\_scores} = \text{where}(mask=0,\; -\infty,\; \text{scores})$$

Softmax 得注意力权重（按最后一维 S 做归一化，数值稳定：减去行最大值）
α=softmax(masked_scores)\alpha = \text{softmax}(\text{masked\_scores})
α=softmax(masked_scores)

聚合得到每头输出
head=αVh∈[B,H,S,dk]\text{head} = \alpha V_h \quad\in [B,H,S,d_k]
head=αVh∈[B,H,S,dk]

拼接各头并做输出投影
先将各头在 d_k 维拼接回 d_model：[B,S,H\cdot d_k] = [B,S,d_model]，
再乘以 W_O：
$$\text{output} = \text{concat(heads)}\; W_O \quad\in [B,S,d_{model}]$$

输出格式
题目要求保留两位小数，并以 List 形式输出（即常见的嵌套列表），需要把 ndarray/数组转换为列表。

变长序列：常见做法是先对批次内对齐（padding），然后结合 因果掩码 与 padding mask（本题未给出padding mask输入），本实现提供标准因果掩码；若有padding，可在同维度位置再叠加一个padding掩码（将padding位置设为 −∞-\infty−∞）。
复杂度分析
设批次 B、序列长度 S、模型维度 D=d_model、头数 H、每头维度 d_k=D/H。

时间复杂度

线性映射：X * W_Q/W_K/W_V 各为 O(B*S*D^2)（若使用分块/并行可等价为 O(B*S*D*D)）。
注意力 QK^T：每头 O(S^2*d_k)，总计 O(B*H*S^2*d_k) = O(B*S^2*D)。
乘 V 聚合：同阶 O(B*S^2*D)。
输出投影 * W_O：O(B*S*D^2)。
综合为 O(B*S*D^2 + B*S^2*D)，与标准Transformer一致。

空间复杂度
主要存储 Q/K/V、注意力分数与权重：O(B*S*D + B*H*S*S)，即 O(B*S*D + B*S^2*H)。

代码实现

**Python 代码：**

```python
# 题意：读入 "num_heads;X;Q;K;V;W_O"（用分号分隔），实现因果掩码多头自注意力
# 要求：输出为 List（嵌套列表），保留两位小数

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
# scores[b,h,i,j] = Qh[b,h,i,:] dot Kh[b,h,j,:] / sqrt(d_k)
# 利用矩阵乘法： (B,H,S,d_k) x (B,H,d_k,S) -> (B,H,S,S)
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
# 转为嵌套列表输出
out = np.around(out, 2)
print(to_str(out.tolist()))

if __name__ == "__main__":
main()

```

---

## 2025年9月24日-AI岗

<a id="第2题-p3791"></a>

### 第2题-无线网络优化中的基站聚类分析（P3791）- 困难





【问题背景】在无线网络优化中，基站的位置分布直接影响信号覆盖质量。密集区域的基站可能造成资源浪费，而稀疏区域则会出现信号覆盖不足。
【任务要求】给定n个基站的二维坐标，使用K-Means算法将其划分为k个簇，再通过计算每个簇的轮廓系数（Silhouette Coefficient），识别信号覆盖最差的簇（轮廓系数最低），并在该簇中心新增基站以优化信号覆盖。
【算法过程】

使用前k个基站作为初始聚类中心，执行K-Means算法。K-means的结束条件为：最大迭代次数100或者所有簇中心点移动距离都不大于1e−61e-61e−6。
计算每个簇的轮廓系数（簇内所有点的轮廓系数平均值）。
找出轮廓系数最低的簇。
输出该簇的中心坐标（保留两位小数），作为新增基站的位置。

K-Means和轮廓系数的详细介绍见“提示”。
输入描述
第一行：基站数量 n 和聚类簇数 k，之间以空格分开，其中 n 取值范围为 [1,500]，k 取值范围为 [1,120]。
接下来 n 行：每行两个整数，表示基站的坐标 x 和 y，其中 x 取值范围为 [0,5000]，y 取值范围为 [0,3000]。
输出描述
新增基站的坐标：x,yx,yx,y （输出结果四舍五入保留两位小数，采用 RoundingMode.HALF_EVEN）
样例1
输入
6 2
0 0
1 1
2 2
10 10
11 11
5 5

输出
8.67,8.67

说明
簇划分结果：簇 0:[(0,0),(1,1),(2,2)]0:[(0,0),(1,1),(2,2)]0:[(0,0),(1,1),(2,2)]，中心 (1,1);(1,1);(1,1); 簇 1:[(5,5),(10,10)(11,11)]1:[(5,5),(10,10)(11,11)]1:[(5,5),(10,10)(11,11)]，中心 (8.67,8.67)(8.67,8.67)(8.67,8.67) 轮廓系数:簇 0 的轮廓系数: 0.820.820.82 ;簇 1 轮廓系数: 0.350.350.35 答案:输簇 1 的中心点: (8.67,8.67)(8.67,8.67)(8.67,8.67)
样例2
输入
4 2
0 0
0 1
1 0
10 10

输出
0.33,0.33

说明
簇划分结果：簇 0 : [(0,0),(0,1)(1,0)] ，中心点: (0.33,0.33)(0.33,0.33)(0.33,0.33) ; 簇 [(10.10)] ，中心点: (10,10)(10,10)(10,10)
轮廓系数:线 0 的轮廓系数: 0.920.920.92 ;簇 1 的轮廓系数: 1.01.01.0 答案:输出簇 0 的中心点: 0.33,0.330.33,0.330.33,0.33
簇 0:[A(0,0),B(0,1),C(1,0)]0:[A(0,0),B(0,1),C(1, 0)]0:[A(0,0),B(0,1),C(1,0)] ;
簇 1 : [(10.10)][(10. 10)][(10.10)]
簇 0 的轮廓系数计算:
计算点 A(0,0)A(0,0)A(0,0) :
1、A 同簇平均距离为 1 : A 到 B(0,1)B(0,1)B(0,1) 距离 1 ，A 到 C(1,0)C(1,0)C(1,0) 距离 1
2、A 到簇 1 平均距离为 14.14214.14214.142 : A 到 D(10,10)D(10,10)D(10,10) 距离 14.14214.14214.142
3、A 的轮廓系数 s(A):0.929s(A):0.929s(A):0.929
计算点 B(0,1)B(0,1)B(0,1) :
1、B 同簇平均距离为 1.207:B1.207:B1.207:B 到 A(0,0)A (0,0)A(0,0) 距离 1 ，B 到 C(1,0)C(1,0)C(1,0) 距离 1.4141.4141.414
2、B 到簇 1 平均距离为 13.454:B13.454:B13.454:B 到 D(10,10)D(10,10)D(10,10) 距离 13.45413.45413.454
3、B 的轮廓系数 s(B):0.910s(B):0.910s(B):0.910
计算点 C(1,0):C(1,0):C(1,0):
1、C 同簇平均距离为 1.207:C1.207:C1.207:C 到 A(0,0)A(0,0)A(0,0) 距离 1，C1，C1，C 到 B(0,1)B(0,1)B(0,1) 距离 1.4141.4141.414
2、C 到簇 1 平均距离为 13.454:C13.454:C13.454:C 到 D(10,10)D(10,10)D(10,10) 距离 13.45413.45413.454
3、C 的轮廓数 s(C):0.910s(C):0.910s(C):0.910
簇 0 轮廓系数: (s(A)+s(b)+s(b))/3=2.749/3=0.92(s(A)+s(b)+s(b))/3=2.749/3=0.92(s(A)+s(b)+s(b))/3=2.749/3=0.92
提示
K-means的算法步骤为：

选择初始化的前 k 个样本作为初始聚类中心。

针对数据集中每个样本 xix_ixi，计算它到 k 个聚类中心的距离并将其分到距离最小的聚类中心所对应的类中。

针对每个类别 aja_jaj，重新计算它的聚类中心
aj=1ci∑x∈cixa_j = \frac{1}{c_i} \sum_{x \in c_i} x
aj=ci1x∈ci∑x
（即属于该类的所有样本的质心）。

重复上面2、3两步操作，直到达到某个中止条件（迭代次数、最小误差变化等）。

轮廓系数 (Silhouette Coefficient Index)：
1、对于一个数据点 i，先计算它和簇内其他数据点的平均距离 aia_iai。
2、然后计算该点与不包含该点所在簇的其他簇内数据点的平均距离 bib_ibi（簇间相似度），选取其中距离最小的那个作为 i 的簇间平均距离。
3、最后，计算数据点 i 的轮廓系数：
si=bi−aimax⁡(ai,bi)
s_i = \frac{b_i - a_i}{\max(a_i, b_i)}
si=max(ai,bi)bi−ai
将所有数据点的轮廓系数取平均值，即得到聚类算法的整体轮廓系数。
若某个数据点所在簇的数据点数量小于等于1，则该点的轮廓系数为 0。
RoundingMode.HALF_EVEN：
1、Python默认的HALF_EVEN模式，其他语言按照如下规则处理：
HALF_EVEN也称为“银行家舍入”或“向偶数舍入”。这种模式下，当小数部分恰好为0.50.50.5时，round()会将结果舍入到最近的偶数。

round(2.55, 1) 会返回 2.62.62.6 （因为6是偶数）
round(2.65, 1) 会返回 2.62.62.6 （因为6是偶数）
round(2.75, 1) 会返回 2.82.82.8 （因为8是偶数）
round(1.15, 1) 会返回 1.21.21.2 （因为2是偶数）

▶️


#### 解答


video solution

解题思路
相关算法与实现要点

K-Means

初始中心：取前 k 个点作为初始中心（简单且可重复）。

迭代过程：

分配：每个点归到欧氏距离最近的中心；
更新：每个簇的新中心为簇内点坐标的均值；

终止条件：迭代次数上限 100 次，或所有中心移动距离之和 ≤1e−6\le 1e{-6}≤1e−6。

若出现空簇，保持其中心不变（在本题给定范围下可稳定收敛；也可按需要将最远点“拉”成新簇中心，本文实现采用保持不变的简洁策略）。

轮廓系数（Silhouette）
对于样本 p：

a(p)a(p)a(p)：与同簇其它点的平均距离；若该簇仅单个样本，则 a(p)=0a(p)=0a(p)=0；
b(p)b(p)b(p)：与最近的其它簇内全部点的平均距离；
轮廓值：$s(p)=\dfrac{b(p)-a(p)}{\max\{a(p),\,b(p)\}} \in [-1,1]$。
簇的轮廓系数为簇内样本的 s(p)s(p)s(p) 平均值。
实现时按定义直接枚举计算，时间复杂度 O(n2)O(n^2)O(n2)，对 n≤500n\le 500n≤500 可接受。

输出与舍入

输出为被选中簇中心的 (x,y)(x,y)(x,y)，采用银行家舍入保留两位：

Python：Decimal(...).quantize(Decimal('0.00'), ROUND_HALF_EVEN)
Java：BigDecimal.setScale(2, RoundingMode.HALF_EVEN)

**Python 代码：**

```python
import sys
from decimal import Decimal, ROUND_HALF_EVEN

def kmeans(pts, k):
n = len(pts)
centers = [list(pts[i]) for i in range(k)]           # 前 k 个点做初始中心
labels = [0] * n
for _ in range(100):                                  # 最多 100 轮
# 分配：找最近中心
for i in range(n):
bi, bd = 0, (pts[i][0]-centers[0][0])**2 + (pts[i][1]-centers[0][1])**2
for c in range(1, k):
d = (pts[i][0]-centers[c][0])**2 + (pts[i][1]-centers[c][1])**2
if d < bd:
bd, bi = d, c
labels[i] = bi
# 更新：按均值重算中心
sx = [0.0]*k; sy = [0.0]*k; cnt = [0]*k
for i in range(n):
c = labels[i]; sx[c] += pts[i][0]; sy[c] += pts[i][1]; cnt[c] += 1
moved = 0.0
for c in range(k):
nx = centers[c][0]; ny = centers[c][1]
if cnt[c] > 0:
nx, ny = sx[c]/cnt[c], sy[c]/cnt[c]
moved += abs(nx - centers[c][0]) + abs(ny - centers[c][1])
centers[c][0], centers[c][1] = nx, ny
if moved <= 1e-6:                                 # 中心几乎不动则停止
break
return labels, centers

def silhouette(pts, labels, k):
n = len(pts)
groups = [[] for _ in range(k)]
for i, c in enumerate(labels): groups[c].append(i)

def dist(i, j):
dx = pts[i][0]-pts[j][0]; dy = pts[i][1]-pts[j][1]
return (dx*dx+dy*dy) ** 0.5

avg = [0.0]*k
for c in range(k):
idx = groups[c]
if not idx:
avg[c] = 0.0
continue
ssum = 0.0
for i in idx:
# a(p)
if len(idx) == 1: a = 0.0
else:
t = 0.0
for j in idx:
if j != i: t += dist(i, j)
a = t / (len(idx)-1)
# b(p)
b = float('inf')
for c2 in range(k):
if c2 == c or not groups[c2]:
continue
t = 0.0
for j in groups[c2]: t += dist(i, j)
b = min(b, t/len(groups[c2]))
if b == float('inf'): sp = 0.0
else:
m = max(a, b)
sp = 0.0 if m == 0 else (b - a) / m
ssum += sp
avg[c] = ssum / len(idx)
return avg

def rnd2(v):
return f"{Decimal(str(v)).quantize(Decimal('0.00'), rounding=ROUND_HALF_EVEN):.2f}"

def main():
data = list(map(float, sys.stdin.read().strip().split()))
if not data: return
n = int(data[0]); k = int(data[1])
pts = [(data[i], data[i+1]) for i in range(2, 2+2*n, 2)]
labels, centers = kmeans(pts, k)
sil = silhouette(pts, labels, k)
bad = min(range(k), key=lambda c: (sil[c], c))        # 轮廓系数最低
x, y = centers[bad]
print(f"{rnd2(x)},{rnd2(y)}")

if __name__ == "__main__":
main()

```

---

<a id="第3题-p3792"></a>

### 第3题-基于决策树的无线状态预策（P3792）- 中等





通过分析基站的关键性能指标（如信息强度，干扰水平，用户数量等）。可以预测网络是否处于正常状态（标签0）或劣化状态（标签1）.决策树算法因其直观的判断逻辑和快速的响应能力，被广泛应用于无线网络智能运维场景。
给定一组基站性能特征数据和对应的网络质量标签，请实现一个基于信息增益的决策树分类器。用于判断网络质量是否劣化。信息熵定义为：
H(S)=−∑i∈Spilog⁡2(pi)H(S) = -\sum_{i \in S} p_i \log_2(p_i)
H(S)=−i∈S∑pilog2(pi)
信息熵 = 划分前熵 - 划分后条件熵
特殊情况处理：

当多个特征对应的信息增益相等时，优先选择索引较小的特征进行样本划分；
当没有特征对样本进一步划分时，该节点预测为样本数较多的标签（若标签0和标签1数量一致，则默认该节点预测为0）。

输入描述
第一行：整数 n (1≤n≤1000)n \ (1 \leq n \leq 1000)n (1≤n≤1000)，表示训练样本数量，整数 m (2≤m≤10)m \ (2 \leq m \leq 10)m (2≤m≤10) 表示特征数。
接下来 n 行，每行包含 m+1m + 1m+1 个整数，前  m 个是特征 1 ~ 特征 m 对应的特征值（取值为 0 或 1），最后一个是要预测的标签（ 0 或 1）。
下一行，整数 q(1≤q≤100)q (1 \le q \le 100)q(1≤q≤100)，表示查询样本数量。
接下来 q 行，每行包括 m 个整数，表示查询样本的特征值 (0 或 1)。
输出描述
输出 q 行，每行为一个整数（0或1），表示对应查询样本的预测类别。
样例1
输入
10 3
1 0 1 1
1 0 0 0
1 1 1 1
1 1 0 1
0 0 1 0
0 0 0 0
0 1 1 1
0 1 0 0
1 0 1 1
0 1 1 1
3
1 0 1
0 0 0
1 1 0

输出
1
0
1

说明
数据包含 10 个样本，每个样本包含三个特征。基于训练数据，可以构建出如下决策树：

1.
[特征3]

2.
/                                                                   \

3.
/                                                                       \

4.
[特征1]                                                              [特征1]

5.
/                   \                                                     /                   \

6.
/                          \                                               /                         \

7.
标签=0                [特征2]                                   [特征2]                 标签=1

8.
/                \                            /                \

9.
/                    \                        /                    \

10.
标签=0             标签=1             标签=0           标签=1

样例2
输入
6 2
1 1 1
0 0 0
1 0 1
0 1 0
1 1 1
0 0 0
2
1 1
0 0

输出
1
0

说明
训练数据包含 6 个样本，每个样本包含两个特征(特征 1 和特征 2 )，构建得到的决策树如下：

1.
特征1

2.
/                     \

3.
/                           \

4.
标签=0                 标签=1

原始信息熵：−1/2-1/2−1/2 log(1/2)−1/2log(1/2)=1log(1/2)-1/2log(1/2)=1log(1/2)−1/2log(1/2)=1
使用特征 1 进行划分后：信息熵 =1log(1)=0=1log(1)=0=1log(1)=0 ，增益为 1
使用特征 2 进行划分后：信息熵 =0.5×0.9183+0.5×0.9183=0.9183=0.5×0.9183+0.5×0.9183=0.9183=0.5×0.9183+0.5×0.9183=0.9183，增益为 0.08170.08170.0817


#### 解答


解题思路
本题要求用训练样本（m 个二值特征，标签为 0/1）构建一个二叉决策树（ID3），再对 q 个查询样本进行预测。
核心要点：

划分准则：使用信息增益（Entropy + Information Gain）。

节点样本集合 S 的熵 H(S)=−∑pilog⁡2piH(S)=-\sum p_i\log_2 p_iH(S)=−∑pilog2pi（二分类时就是看 0/1 的占比）。
选择尚未使用的特征 f 进行二分（按 0/1 分到 S0,S1S_0,S_1S0,S1），条件熵为
$H(S|f)=\frac{|S_0|}{|S|}H(S_0)+\frac{|S_1|}{|S|}H(S_1)$，
信息增益 IG=H(S)−H(S∣f)IG=H(S)-H(S|f)IG=H(S)−H(S∣f)。
在信息增益最大的特征上划分；若增益相等，选索引更小的特征。

停止与叶子规则（与题面特殊情况一致）：

若当前样本全为同一标签 → 返回该标签。
若没有可带来进一步划分的特征（所有剩余特征信息增益 ≤0\le 0≤0 或已用尽）→ 返回多数标签（平票返回 0）。
实现时，为了保证预测分支完整：当某次划分出现一侧子集为空，也令该子结点为多数标签叶子（平票 0）。

预测：从根开始，按结点记录的特征索引读查询样本的该特征值（0/1）走向子结点，直到叶子输出标签。

实现细节：

递归建树，传入：样本下标集合、剩余可用特征（或 used 标记）。
计算信息增益时用双计数即可（统计在该特征取 0/1 时两类数量）。
为避免浮点误差，用一个很小的 ε=1e−12\varepsilon=1e{-12}ε=1e−12 比较大小。

代码实现

**Python 代码：**

```python
import sys
import math
from typing import List, Dict, Any

# 计算集合 S 的熵（S 给出为标签列表 0/1）
def entropy(labels: List[int]) -> float:
n = len(labels)
if n == 0:
return 0.0
c1 = sum(labels)
c0 = n - c1
res = 0.0
for c in (c0, c1):
if c == 0:
continue
p = c / n
res -= p * math.log2(p)
return res

# 返回多数标签（平票返回 0）
def majority_label(labels: List[int]) -> int:
c1 = sum(labels)
c0 = len(labels) - c1
return 1 if c1 > c0 else 0

# 递归建树；features 为可用特征下标列表
def build_tree(X: List[List[int]], y: List[int], idxs: List[int], features: List[int]) -> Dict[str, Any]:
# 若纯节点，直接返回叶子
labels = [y[i] for i in idxs]
if all(l == labels[0] for l in labels):
return {"leaf": True, "label": labels[0]}

# 若没有可用特征，或所有增益 <= 0，则返回多数标签
base_H = entropy(labels)
best_gain = -1.0
best_f = -1
eps = 1e-12

for f in features:
# 按特征 f 二分，统计 0/1 两侧标签
idx0, idx1 = [], []
lab0, lab1 = [], []
for i in idxs:
if X[i][f] == 0:
idx0.append(i)
lab0.append(y[i])
else:
idx1.append(i)
lab1.append(y[i])
cond = (len(idx0) / len(idxs)) * entropy(lab0) + (len(idx1) / len(idxs)) * entropy(lab1)
gain = base_H - cond
if gain > best_gain + eps or (abs(gain - best_gain) <= eps and f < best_f):
best_gain = gain
best_f = f

if best_gain <= eps or best_f == -1:
return {"leaf": True, "label": majority_label(labels)}

# 根据最优特征划分
idx0, idx1 = [], []
for i in idxs:
(idx0 if X[i][best_f] == 0 else idx1).append(i)

# 子结点：空子集用多数标签叶子填充，保证可预测
next_features = [f for f in features if f != best_f]
if len(idx0) == 0:
left_node = {"leaf": True, "label": majority_label(labels)}
else:
left_node = build_tree(X, y, idx0, next_features)

if len(idx1) == 0:
right_node = {"leaf": True, "label": majority_label(labels)}
else:
right_node = build_tree(X, y, idx1, next_features)

return {"leaf": False, "feat": best_f, "left": left_node, "right": right_node}

# 用树进行预测
def predict(tree: Dict[str, Any], x: List[int]) -> int:
node = tree
while not node["leaf"]:
f = node["feat"]
node = node["left"] if x[f] == 0 else node["right"]
return node["label"]

def main():
data = sys.stdin.read().strip().split()
it = iter(data)
n = int(next(it))
m = int(next(it))
X, y = [], []
for _ in range(n):
row = [int(next(it)) for _ in range(m + 1)]
X.append(row[:m])
y.append(row[m])
q = int(next(it))
Q = [[int(next(it)) for _ in range(m)] for _ in range(q)]

idxs = list(range(n))
features = list(range(m))
tree = build_tree(X, y, idxs, features)

out = []
for x in Q:
out.append(str(predict(tree, x)))
print("\n".join(out))

if __name__ == "__main__":
main()

```

---

## 2025年9月18日(留学生)-AI岗

<a id="第2题-p3718"></a>

### 第2题-最大能量路径（P3718）- 中等





在自动驾驶系统中，车道线识别是核心功能之一。车道线通常具有连续性，从图像左侧到右侧逐渐展开。
为了识别出最可能的车道线路径，我们可以在图像中找到一条路径，使得路径上所有像素的信号值与策略矩阵的乘积之和最大。
现定义每个位置的能量值为策略矩阵与该位置周边信号值的乘积和。
给定一个 H×WH×WH×W 的图像以及一个 K×KK×KK×K 的策略矩阵，用于模拟不同方向的路径选择策略。
你需要从图像的第一列任意像素出发，走到最后一列任意像素，每一步只能向右、右上、右下移动一格。
在行进的过程中，需要实时的收集能量值，请找到一条路径，使得路径上的能量值之和最大。
输入描述
第一行输入 H W K K ，分表表示给定图像及策略矩阵的维度
接下来
H 行输入图像矩阵
K 行输入策略矩阵
输出描述
输出最大能量值
样例1
输入
1 1 1 1
5
1

输出
5.0

说明
有且仅有一条路径，最大能量值为 5∗15*15∗1 为 5.05.05.0
样例2
输入
3 3 3 3
1 2 3
4 5 6
7 8 9
1 2 2
1 1 1
1 1 1

输出
119.0

说明
输入第一行是一个 3×33×33×3 的图像以及 3×33×33×3 的策略矩阵
每个位置的能量图：
[[12.21.16.]
[30.50.36.]
[33.50.34.]][33.50.34.]][33.50.34.]]
最大能量路径的值：119.0119.0119.0 最大能量路径：(2,0)−>(1,1)−>(1,2)(2,0)->(1,1)->(1,2)(2,0)−>(1,1)−>(1,2)
提示
1.1.1.策略矩阵为奇数，边缘处用零填充
2.2.2.输出保留一位小数


#### 解答


解题思路
本题可分两步完成：

能量图计算（二维相关）
将策略矩阵视为卷积核，但无需翻转（即做二维相关，correlation）：
设图像为 I（H×W），策略矩阵为 S（K×K，K 为奇数，p = K//2）。
对于每个像素 (r,c)，其能量定义为
$$E[r][c] = \sum_{i=0}^{K-1}\sum_{j=0}^{K-1} S[i][j]\cdot I[r+i-p][c+j-p]$$超出边界的图像像素按 0 填充。这样得到一幅 H×W 的能量图 E。

最大能量路径（列向动态规划）
从第一列任意行出发，到最后一列任意行，允许的移动：右上 (r-1,c+1)、右 (r,c+1)、右下 (r+1,c+1)。
设 dp[r][c] 为到达 (r,c) 的最大能量和：

初始化：dp[r][0] = E[r][0]

转移：
$$dp[r][c] = E[r][c] + \max\big(dp[r][c-1],\ dp[r-1][c-1]\ (\text{若}r>0),\ dp[r+1][c-1]\ (\text{若}r<H-1)\big)$$

答案：max⁡0≤r<Hdp[r][W−1]\max_{0\le r<H} dp[r][W-1]max0≤r<Hdp[r][W−1]


**Python 代码：**

```python
import sys

def main():
data = sys.stdin.read().strip().split()
it = iter(data)
H = int(next(it)); W = int(next(it)); K1 = int(next(it)); K2 = int(next(it))
K = K1  # 题面给了两个K，这里取第一个；通常两者相等

# 读图像矩阵
I = [[float(next(it)) for _ in range(W)] for _ in range(H)]
# 读策略矩阵
P = [[float(next(it)) for _ in range(K)] for _ in range(K)]

# 计算能量图（零填充卷积）
r = K // 2
E = [[0.0]*W for _ in range(H)]
for i in range(H):
for j in range(W):
s = 0.0
for u in range(K):
ii = i + (u - r)
if 0 <= ii < H:
rowI = I[ii]
rowP = P[u]
for v in range(K):
jj = j + (v - r)
if 0 <= jj < W:
s += rowP[v] * rowI[jj]
E[i][j] = s

# 动态规划
NEG = -1e300
prev = [NEG]*H
for i in range(H):
prev[i] = E[i][0]

for j in range(1, W):
cur = [NEG]*H
for i in range(H):
best = prev[i]
if i-1 >= 0:
best = max(best, prev[i-1])
if i+1 < H:
best = max(best, prev[i+1])
cur[i] = E[i][j] + best
prev = cur

ans = max(prev)
print(f"{ans:.1f}")

if __name__ == "__main__":
main()

```

---

<a id="第3题-p3719"></a>

### 第3题-数据中心水温调节档位决策（P3719）- 中等





数据中心的机房需要散热，一般通过水冷系统调控机房的温度。出于节能的目的，需要收集数据中心最近一段时间的业务负裁，同时结合历史信息，外界气温、湿度等因素调高或调低冷机的出水温度。
逻辑回归是比轻量的模型，其输出 0 或 1 可以分别表示调低或调高温度。
在实际使用场景中，仅判断调低或调高不能满足业务要求，需要细化到调低 1 度、调高 0.50.50.5度、维持不变、调高 0.50.50.5 度、调高 1 度等不同的"档位"。
因此，可以对逻辑回归进行改造，将其输出更改为 softmaxsoftmaxsoftmax 可满足要求。
请根据提供的观测数据，训练改造后的模型，并根据给定样本数据预测调节的档位。
优化后的模型：O=XW+b，P=softmax(O)O=XW+b，P=softmax(O)O=XW+b，P=softmax(O)。
X∈R(m,n)X∈R(m,n)X∈R(m,n) 表示 m 条样本，每个样本有 n 个特征；
W∈R(n,k)W∈R(n,k)W∈R(n,k) 为权重；k 是档位数(即 k 个分类)；
b∈R(1,k)b∈R(1,k)b∈R(1,k) 为偏置。W 和 b 通过训练得到。
P∈R(m,k)P∈R(m,k)P∈R(m,k) 表示预测的概率，取概率最大的作为输出档位。
输入描述
1.第一行是数据 schemaschemaschema，分别表示特征数 n ，分类数 k ，第 0 类样本数(即档位 0 )，第 1 类样本数，…，，…，，…，第 k−1k-1k−1 类样本数，待预测样本数 m ；数据均为 int 类型
2.后续的多行是 k 个分类的训练样本(按照分类 0 的多条样本、分类 1 的多条样本 、…、…、… 依次排列，一行一个样本)和 m 条待预测样本(一行个样本)；数据均为 floatfloatfloat 类型
输出描述
每个待预测样本所所属的分类，一行输出一个样本的预测结果
样例1
输入
2 3 2 3 2 3
9 95
33 53
53 55
69 21
68 31
70 85
80 83
25 70
45 30
79 86

输出
0
1
2

说明
1.第一行数据 2 3 2 3 2 3 ，表示每条样本 2 个特征，共有 3 个分类；分类 0 样本 2 条，分类 1 样本 3 条，分类 2 样本 2 条；测试用例 3 条
2.第 2~3 行为分类 0 样本(标签为 0 )，4~6 行为分类 1 样本(标签 1 )，7~8 行为分类 2 样本(标签为 2 )
3.最后 3 行是待预测样本
输出：3 个待预测样本的预测结果为分类 0 、分类 1 、分类 2
样例2
输入
3 3 4 3 4 5
9 95 7
33 53 13
45 43 6
40 50 11
53 55 36
69 21 55
68 31 43
70 85 23
80 83 46
70 73 55
76 78 53
25 70 6
20 69 16
45 30 50
79 86 51
70 76 36

输出
0
0
1
2
2

说明
1、第一行数据 3 3 4 3 4 5 ，表示每条样本 3 个特征，共有 3 个分类；分类 0 样本 4 条，分类 1 样本 3 条，分类 2 样本 4 条；测试用例 5 条
2.第 2~5 行为分类 0 样本(标签为 0)，6~8 行为分类 1 样本(标签 1 )，9~12 行为分类 2 样本(标签为 2 )
3.最后 5 行是待预测样本
输出：
5 个待预测样本的预测结果为分类 0 (即档位 0 )、分类 0 、分类 1 、分类 2 、分类 2
提示
1.使用交叉熵损失函数，标签值需转为 one−hotone-hotone−hot 编码；梯度求解函数如下方公式所示，Y 为真实标签(即 one−hotone-hotone−hot 编码)；P 为预测的概率(即 softmaxsoftmaxsoftmax 的输出)
2.采用批梯度下降优化参数；选择比较好的学习率(根据经验，为了加快收敛速度，初始学习率可设置较大的数值，例如 5 )；适当增加迭代次数有助于获得更精确的结果；
3.原始数据值域一般在 100 以内，为避免计算 softmaxsoftmaxsoftmax 越界，需要对数据集执行归一化操作。
交叉熵函数
$-\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{k} Y_{i, j} \log \left(P_{i, j}\right)$
损失函数对 W 的梯度
1mXT(P−Y)\frac{1}{m} X^{T}(P-Y)m1XT(P−Y)
损失函数对 b 的梯度
$\frac{1}{m} \sum_{i=1}^{m}\left(P_{i,:}-Y_{i,:}\right)$


#### 解答


模型与目标

线性部分：O=XW+bO=XW+bO=XW+b，其中 X∈RN×nX\in\mathbb{R}^{N\times n}X∈RN×n、W∈Rn×kW\in\mathbb{R}^{n\times k}W∈Rn×k、b∈R1×kb\in\mathbb{R}^{1\times k}b∈R1×k。

概率：P=softmax(O)\mathrm{softmax}(O)softmax(O)，对每行做
（减去按行最大值做数值稳定）。

损失（带 $L_2$ 正则）：

梯度：

预测：对每个样本取 arg⁡max⁡jpij\arg\max_j p_{ij}argmaxjpij 得到分类 0∼k−10\sim k-10∼k−1。


**Python 代码：**

```python
# 说明：多分类 softmax 回归，使用全批量梯度下降与L2正则
# 输入输出同题面描述

import sys
import math

def read_ints():
return list(map(float, sys.stdin.readline().strip().split()))

# 为了稳妥处理空白行，使用迭代读取
tokens = []
for line in sys.stdin:
parts = line.strip().split()
if parts:
tokens.extend(parts)

it = iter(tokens)
def next_int():
return int(next(it))
def next_float():
return float(next(it))

n = next_int()
k = next_int()
cnt = [next_int() for _ in range(k)]
N = sum(cnt)
m = next_int()

# 读取训练与预测
X_train = [[next_float() for _ in range(n)] for _ in range(N)]
y_train = []
for c in range(k):
for _ in range(cnt[c]):
y_train.append(c)
X_test = [[next_float() for _ in range(n)] for _ in range(m)]

# 标准化
eps = 1e-8
mu = [0.0]*n
sigma = [0.0]*n

for j in range(n):
s = sum(X_train[i][j] for i in range(N)) if N > 0 else 0.0
mu[j] = s / N if N > 0 else 0.0
for j in range(n):
ss = 0.0
for i in range(N):
d = X_train[i][j] - mu[j]
ss += d*d
sigma[j] = math.sqrt((ss / max(1, N-1))) if N > 1 else 1.0
if sigma[j] < eps: sigma[j] = 1.0

def standardize(X):
for i in range(len(X)):
row = X[i]
for j in range(n):
row[j] = (row[j] - mu[j]) / (sigma[j] + eps)

standardize(X_train)
standardize(X_test)

# 参数
W = [[0.0]*k for _ in range(n)]
b = [0.0]*k

# 超参
epochs = 600
lr = 0.1
reg = 1e-4

# 训练
for epoch in range(epochs):
dW = [[0.0]*k for _ in range(n)]
db = [0.0]*k

for i in range(N):
# logits
logits = [b[j] + sum(X_train[i][f]*W[f][j] for f in range(n)) for j in range(k)]
mx = max(logits)
exps = [math.exp(v - mx) for v in logits]
s = sum(exps) + eps
probs = [v/s for v in exps]

y = y_train[i]
for j in range(k):
diff = probs[j] - (1.0 if j == y else 0.0)
for f in range(n):
dW[f][j] += X_train[i][f] * diff
db[j] += diff

invN = (1.0 / N) if N > 0 else 1.0
for f in range(n):
for j in range(k):
dW[f][j] = dW[f][j] * invN + reg * W[f][j]
for j in range(k):
db[j] *= invN

for f in range(n):
for j in range(k):
W[f][j] -= lr * dW[f][j]
for j in range(k):
b[j] -= lr * db[j]

if (epoch + 1) % 150 == 0:
lr *= 0.9  # 轻微衰减

# 预测
out_lines = []
for i in range(m):
logits = [b[j] + sum(X_test[i][f]*W[f][j] for f in range(n)) for j in range(k)]
argmax = max(range(k), key=lambda j: logits[j])
out_lines.append(str(argmax))

print("\n".join(out_lines))

```

---

## 2025年9月17日-AI岗

<a id="第2题-p3712"></a>

### 第2题-大模型Attention模块开发（P3712）- 中等





已知大模型常用的 Attention 模块定义如下：
$Y = \text{softmax}\left(\frac{QK^T}{\sqrt{h}}\right)V$
此处考虑二维情况，其中
$Q, K, V = XW_1, XW_2, XW_3 \in \mathbb{R}^{n \times h}, \quad X \in \mathbb{R}^{n \times m}, \quad W_1, W_2, W_3 \in \mathbb{R}^{m \times h}$
注意：

为简便起见，所有输入初始化为全1 1 1矩阵，所有权重矩阵初始化为上三角全 1 矩阵。

对任意矩阵 ( M ) 的 softmaxsoftmaxsoftmax 计算简化为：

$\text{softmax}(M)_{ij} = \frac{M_{ij}}{M_i}, \quad M_i = \sum_j M_{ij}$
输入描述
输入为维度参数 n,mn, mn,m和h hh，参数间使用空格隔开，均为小于 100 的正整数
输出描述
输出为结果矩阵  Y∈Rn×hY \in \mathbb{R}^{n \times h}Y∈Rn×h的所有元素之和，例如 15，输出在四舍五入后保留整数
样例1
输入
3 3 3

输出
18

说明
$X =
\begin{pmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{pmatrix}, \quad
W_1, W_2, W_3 =
\begin{pmatrix}
1 & 1 & 1 \\
0 & 1 & 1 \\
0 & 0 & 1
\end{pmatrix}$
$Q, K, V =
\begin{pmatrix}
1 & 2 & 3 \\
1 & 2 & 3 \\
1 & 2 & 3
\end{pmatrix}, \quad
Y =
\begin{pmatrix}
1 & 2 & 3 \\
1 & 2 & 3 \\
1 & 2 & 3
\end{pmatrix}$
输出为：18
样例2
输入
2 3 1

输出
2

说明
$X =
\begin{pmatrix}
1 & 1 & 1 \\
1 & 1 & 1
\end{pmatrix}, \quad
W_1, W_2, W_3 =
\begin{pmatrix}
1 \\
0 \\
0
\end{pmatrix}$
$Q, K, V =
\begin{pmatrix}
1 \\
1
\end{pmatrix}, \quad
Y =
\begin{pmatrix}
1 \\
1
\end{pmatrix}$
输出为：2
提示
输入参数不包含 0，为正整数


#### 解答


解题思路

按题意用“暴力模拟”完整走一遍计算图：

构造 X 为 n×m 的全 1；构造 W1、W2、W3 为 m×h 的上三角全 1。
计算 Q=X·W1，K=X·W2，V=X·W3（普通三重循环矩阵乘法）。
计算 M=(Q·K^T)/sqrt(h)。
按“简化 softmax”把 M 的每一行做归一化：A[i][j]=M[i][j]/(该行元素和)。
计算 Y=A·V。
将 Y 全部元素求和，四舍五入输出整数。

算法类型：暴力模拟/矩阵运算。

由于 n、m、h < 100，直接模拟即可在时空限制内通过。

复杂度分析

矩阵乘法开销：

计算 Q、K、V：O(n·m·h)
计算 M=Q·K^T：O(n²·h)
行归一化：O(n²)
计算 Y=A·V：O(n²·h)

总时间复杂度：O(n·m·h + n²·h)。

空间复杂度：O(n·h + n²)（保存 Q、K、V、A 或 M 等中间结果）。

代码实现

**Python 代码：**

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

# 4) “简化 softmax”：按行除以行和
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

---

<a id="第3题-p3713"></a>

### 第3题-大模型分词（P3713）- 中等





您正在为一种罕见的语言构建一个专用的大语言模型。由于训练样本缺失，传统BPE BPE BPE等标准的分词器效果不佳，使得大模型推理生成的句子不理想。
幸运的是，一位语言学家为罕见的语言的已知词根和词缀(我们统称为“词元"或“TokenTokenToken”)都标注了一个“置信度”分数，这个分数代表了该词元作为一个“独立单位”的合理性，同时，语言学家还总结出了一个转移分数表，表示当前词元选择对下一个词元"置信度"的影响。
您的任务是设计并实现一个“最优分词器”，它能将输入的罕见语言句子(一个不含空格的英文小写字符多也串)切分成一系列词元，并使得所有词元的置信度分数之和达到最大，从而帮助大语言模型在后续处理中能够输出更合理的句了
输入描述
第一行输入待分词的字符串 texttexttext，假设只包含英文小写字母；
接着输入词典词条数 n；
然后输入nn n行，每一行包含一个单词和对应的分值，以空格分隔。
第 n+3n+3n+3 行为转移分数的个数 m。
随后m行为转移分数数据。包括起始词、下一个词、转移分数加分X。以空格分隔。
参数范围说明：

0<len(text)≤1000 < len(text) \leq 1000<len(text)≤100
−100≤-100 \leq−100≤ 词典中单词的得分 ≤100 \leq 100≤100
−100≤-100 \le−100≤ 词汇表置信度分数P≤100P \le 100P≤100
输入的字符串都是英文小写字母
0<0 < 0<词汇表大小 n≤100n \le 100n≤100

输出描述
返回最高的分词得分，若根据已知间汇表无法拆分则返回0、我们约定若切分成一系列词元中含有不在已知词汇表中的词，则最终得分为0。
样例1
输入
applepie
2
pen 3
apple 10
2
pen apple 5
pie apple 2

输出
0

说明
texttexttext中的字符不能和词典词条匹配出切分结果，无法计算得分。
样例2
输入
goodeats
4
good 15
goo 12
deats 14
eats 10
1
good eats -5

输出
26

说明
切分为["goodgoodgood","eatseatseats"] 的总分=15+10−5=20 = 15 + 10 - 5 = 20=15+10−5=20；
切分为 ["goo","deatsdeatsdeats"] 的总分=12+14=26 = 12 + 14 = 26=12+14=26；
所以最大得分为 26。

▶️


#### 解答


video solution

解题思路
本题要求把一段不含空格的小写英文字符串切分为若干“词元”，并最大化：
所有词元的置信度分数之和 + 相邻词元之间的转移加分之和。
若无法完全用词表中的词覆盖整句，则输出 0。
核心算法：动态规划（DP）

设原串为 text，长度为 L。

用哈希表保存词表 score[w]（词 w 的置信度分数）与转移加分 bonus[u][v]（从词 u 到 v 的转移分）。

令 dp[i] 表示能切到下标 i（前缀 text[0:i]）的所有方案中，“以某个词结尾”的最优分数集合。

具体用：dp[i] 是一个映射 {last_word -> best_score}，表示前缀 text[0:i] 且最后一个词为 last_word 的最优总分。

转移：枚举以 i 结尾的词 w = text[j:i]（w 必须在词表中），再看前一段的结尾：

若 j == 0：这是首个词，dp[i][w] = max(dp[i][w], score[w])。
若 j > 0：需要从 dp[j] 的每个候选 u 转移，
dp[i][w] = max(dp[i][w], dp[j][u] + score[w] + bonus[u][w](若无则为0))。

答案是 dp[L] 中所有值的最大值；若 dp[L] 为空（无法完整切分），输出 0。

为降复杂度，可预先计算词表中最长词长 maxLen，只枚举长度不超过 maxLen 的后缀。

为什么不能用贪心？

因为转移加分依赖于相邻两个词，局部最优（当前词分高）并不一定带来全局最优（可能与下一词的转移分差）。因此需要 DP 统筹考虑上下文。

复杂度分析

设原串长度 L ≤ 100，词表大小 N ≤ 100，最长词长 maxLen ≤ L。
外层位置 i 共 L 次；对每个 i，仅尝试长度 ≤ maxLen 的后缀，近似 O(maxLen)；
对每个合法后缀 w，需遍历 dp[j] 的状态数（不超过词表大小 N）。
因此时间复杂度近似为 O(L × maxLen × N)，在给定数据范围内完全可行。
空间复杂度：dp 存每个位置最多 N 个词状态，故 O(L × N)。

代码实现

**Python 代码：**

```python
import sys
from ast import literal_eval

def solve(text, vocab_list, trans_list):
# 构建词表分数
score = {}
max_len = 0
for w, p in vocab_list:
score[w] = p
if len(w) > max_len:
max_len = len(w)
# 构建转移加分表
bonus = {}
for u, v, x in trans_list:
if u not in bonus:
bonus[u] = {}
bonus[u][v] = x

L = len(text)
# dp[i]: dict {last_word: best_score} 覆盖 text[0:i]
dp = [dict() for _ in range(L + 1)]

# 枚举前缀终点 i
for i in range(1, L + 1):
# 只需要尝试不超过词表最长长度的后缀
up = min(max_len, i)
for l in range(1, up + 1):
w = text[i - l:i]
if w not in score:
continue
base = score[w]
j = i - l
if j == 0:
# 首词
dp[i][w] = max(dp[i].get(w, float("-inf")), base)
else:
if not dp[j]:
continue
# 从所有可能的前一词转移
for u, val in dp[j].items():
add = bonus.get(u, {}).get(w, 0)
cand = val + base + add
if cand > dp[i].get(w, float("-inf")):
dp[i][w] = cand

if not dp[L]:
return 0
return max(dp[L].values())

def main():
data = sys.stdin.read().strip().splitlines()
idx = 0
text = data[idx].strip(); idx += 1
n = literal_eval(data[idx].strip()); idx += 1

vocab_list = []
for _ in range(n):
parts = data[idx].strip().split()
w = parts[0]
p = literal_eval(parts[1])
vocab_list.append((w, p))
idx += 1

m = literal_eval(data[idx].strip()); idx += 1
trans_list = []
for _ in range(m):
parts = data[idx].strip().split()
u, v = parts[0], parts[1]
x = literal_eval(parts[2])
trans_list.append((u, v, x))
idx += 1

ans = solve(text, vocab_list, trans_list)
print(ans)

if __name__ == "__main__":
main()

```

---

## 2025年9月12日-AI岗

<a id="第2题-p3657"></a>

### 第2题-二叉树中序遍历的第k个祖先节点（P3657）- 中等





在大规模语言模型 (LLM(LLM(LLM MOE)MOE)MOE) 架构中，每一层的 MLP 模块中有若干个专家。用一颗二叉树把这些专家组织起来，二叉树的每个节点是一个专家。
现给定一个二叉树的根节点，以及两个整数 u 和 k 。任务是找出节点在二叉树中序遍历序列中的第 k 个祖先节点的值。
一个节点的祖先节点是指从根节点到该节点路径上的所有节点(不包括该节点本身)。
这里，“第 k 个祖先”指的是在中序遍历序列中，位于节点 u 前面的所有祖先节点中的第 k 个位置祖先节点。如果这样的祖先节点不存在，则返回 −1-1−1。
输入描述
输入将包含两行。
第一行是一个字符串，表示一棵二叉树：
<1>空节点用 '#' 表示;
<2>非空节点的值为整数;
<3>节点之间用一个空格分隔;
<4>树的层次遍历顺序给出比如，"123##45“ 表示根节点为 1 ，左子节点为 2 ，右子节点为 3 ; 2 没有子节点; 3 的左子节点为 4 ，右子节点为 5 。
第二行包含两个整数 u 和 k ，分别表示目标节点的值和要查找的祖先节点的偏移量, 一个空格分隔这两个值。
输出描述
一个整数，表示在中序遍历序列中节点 u 的第 k 个祖先节点的值。如果不存在，则返回 −1-1−1 。
样例1
输入
30 15 45 7 20 35 50 # # 18 # # 40
40 3

输出
-1

说明
二叉树结构如下：

中序遍历的顺序是：7,15,18,20,30,35,40,45,507,15,18,20,30,35,40,45,507,15,18,20,30,35,40,45,50。
节点 u=40u=40u=40 。在中序遍历序列中，位于 40 前面的节点有 7,15,18,20,30,357,15,18,20,30,357,15,18,20,30,35 。
节点 40 的祖先节点有 30,45,3530,45,3530,45,35 。
在祖先节点中，在中序遍历序列中位于 40 前面的有 30,3530,3530,35 。
按照中序遍历顺序，其前面的祖先节点有： 30,3530,3530,35。
第 K=3K=3K=3 个祖先节点(即在 40 前面第三个位置的祖先节点)不存在。
样例2
输入
10 5 15 2 7 12 18
7 1

输出
5

说明
二叉树结构如下：

中序遍历的顺序是：2,5,7,10,12,15,182,5,7,10,12,15,182,5,7,10,12,15,18 。
节点 u=7u=7u=7 。在中序遍历序列中，位于 7 前面的节点有 2,52,52,5 。
第 k=1k=1k=1 个祖先(即在 7 前面第二个位置的祖先)是 5 。

▶️


#### 解答


video solution

解题思路
关键定义回顾

祖先：根到节点 u 路径上的所有节点（不含 u）。
“中序前面的祖先”：把整棵树做一次中序遍历，沿序列从左到右走，记录在到达 u 之前出现的、同时又是 u 的祖先的那些节点，按出现顺序计数，第 k 个即答案；若不足 k 个返回 -1。

总体算法（一次建树 + 一次中序，O(n)）

按层次序列建树

输入是层次遍历（BFS）序列，空节点为 #。

用队列逐层挂接左右孩子；同时维护：

parent：节点 → 父节点 指针（便于之后找祖先）；
val2node：值 → 节点 指针（便于定位 u）。

说明：题目通常默认结点值唯一，若不唯一则需用“节点身份”而非“值”来定位；本题按样例与常规题型 默认值唯一。

若不存在值为 u 的节点：直接输出 -1。

得到祖先集合

从 val2node[u] 沿 parent 回溯到根，放入一个 anc 哈希集合，表示 u 的全部祖先。

一次中序遍历并计数

进行中序遍历（左—根—右），维护一个计数器 cnt。

每访问到一个节点 x：

若 x == u，停止遍历：若 cnt >= k 则答案在“前面的祖先”中第 k 个；反之 -1。
若 x 在 anc 集合中，则 cnt += 1，当 cnt == k 时把当前节点值记为 ans_k（因为之后还可能没到 u，先保存，等遇到 u 再输出）。

结束条件：遍历到 u 即可（不用遍历全树）。

该策略直接把“在中序序列里位于 u 之前且为祖先”的定义转化为一次遍历的在线计数，无需额外数组与二分。

复杂度分析

设节点数为 n。
建树：O(n)；构造 parent/val2node：O(n)；
中序遍历直到遇到 u：最坏 O(n)；
总时间复杂度：O(n)；空间复杂度：O(n)（队列、映射、祖先集合与递归/显式栈）。

代码实现

**Python 代码：**

```python
import sys
from collections import deque

class Node:
def __init__(self, v):
self.v = v
self.l = None
self.r = None

def build_tree(tokens):
# 空或首元素为# => 空树
if not tokens or tokens[0] == '#':
return None, {}, {}
it = 0
root = Node(int(tokens[it]))
it += 1
q = deque([root])
parent = {root: None}
val2node = {root.v: root}
# 逐个为队首节点挂接左右孩子
while q and it < len(tokens):
cur = q.popleft()
# 左孩子
if it < len(tokens):
t = tokens[it]; it += 1
if t != '#':
left = Node(int(t))
cur.l = left
parent[left] = cur
val2node[left.v] = left
q.append(left)
# 右孩子
if it < len(tokens):
t = tokens[it]; it += 1
if t != '#':
right = Node(int(t))
cur.r = right
parent[right] = cur
val2node[right.v] = right
q.append(right)
return root, parent, val2node

def kth_ancestor_in_inorder_before_u(root, parent, val2node, u, k):
# u不存在
if u not in val2node:
return -1
u_node = val2node[u]
# 收集u的全部祖先到集合
anc = set()
p = parent.get(u_node)
while p is not None:
anc.add(p)
p = parent.get(p)
# 中序遍历（显式栈），计数祖先并在遇到u时判定
stack = []
cur = root
cnt = 0
ans_k = None  # 提前保存第k个祖先的值
while stack or cur:
while cur:
stack.append(cur)
cur = cur.l
cur = stack.pop()
# 访问cur
if cur is u_node:
# 到达u，返回结果（若此前出现了第k个祖先）
return ans_k if cnt >= k else -1
if cur in anc:
cnt += 1
if cnt == k:
ans_k = cur.v
# 继续右子树
cur = cur.r
return -1

def main():
data = sys.stdin.read().strip().splitlines()
if len(data) < 2:
print(-1)
return
tokens = data[0].strip().split()
u, k = map(int, data[1].strip().split())
root, parent, val2node = build_tree(tokens)
ans = kth_ancestor_in_inorder_before_u(root, parent, val2node, u, k)
print(ans)

if __name__ == "__main__":
main()

```

---

<a id="第3题-p3658"></a>

### 第3题-支持LoRA的Attention实现（P3658）- 困难





相对于全量微调，LoRALoRALoRA微调提出了一种低秩分解的方法，只需在原模型参数基础上增加少量的可训练参数，大幅降低计算成本和内存占用。具体而言，对于原始的预训练权重矩阵W，LORALORALORA做以下改进：
W′=W+B×AW'=W+B×AW′=W+B×A
W为原始权重(冻结不变)，B∈Rd×rB∈R^{d×r}B∈Rd×r和 A∈Rr×dA ∈R^{r×d}A∈Rr×d为新增的低秩矩阵，r<<dr<<dr<<d，秩r一般很小。微调时只更新 A、BA、BA、B这两个矩阵，显著减少训练的参数数量。请实现支持LoRALoRALoRA的AttentionAttentionAttention计算
函数LoRA_Attention(x,Wa,Wk,Wv,A,B) LoRA\_Attention(x,W_a,W_k,W_v,A,B)LoRA_Attention(x,Wa,Wk,Wv,A,B) 。为简化实现，仅需支持AttentionAttentionAttention中Q QQ的LoRALoRALoRA结构实现即可。实现时请使用float64float64float64位精度。
输入描述
第1行： b,d,rb,d,rb,d,r，其中b为batch sizeb为batch\ sizeb为batch size,d为特征的长度，r为LoRALoRALoRA矩阵的秩，b≥1,d≥1,r≥0b≥1,d≥1,r≥0b≥1,d≥1,r≥0
第2行：输入x，长度为b×db×db×d
第3−53-53−5行: Wq,Wk,WvW_q,W_k,W_vWq,Wk,Wv,长度为d×dd×dd×d
若r>0r>0r>0，则:
第6行：A，长度为r×dr×dr×d
第7行：B，长度为d×rd×rd×r
输出描述
LoRAAttentionLoRA AttentionLoRAAttention计算的结果，输出保留四位小数，不足四位小数的补0
样例1
输入
2 5 3
-0.58 -0.52 -0.02 0.56 0.79 0.06 -0.64 -0.04 -0.20 -0.38
0.24 -0.72 -0.66 0.96 0.02 -0.43 -0.24 0.19 -0.85 -0.35 0.69 -0.09 0.99 0.21 -0.06 0.55 0.57 0.97 0.58 -0.16 0.64 0.02 -0.71 0.53 -0.90
0.07 -0.16 -0.47 -0.32 -0.92 0.13 -0.74 -0.87 0.05 0.33 0.37 0.75 0.57 0.14 -0.62 0.67 -0.62 -0.85 0.09 -0.90 0.22 0.97 -0.68 0.61 0.48
0.39 -0.74 0.84 0.21 0.44 -0.59 -0.07 -0.84 -0.70 0.86 -0.12 -0.06 0.45 -0.43 -0.09 -0.73 0.56 -0.62 0.36 -0.87 -0.97 -0.48 0.71 0.07 -0.28
0.25 0.58 -0.04 -0.94 0.45 -0.60 0.89 0.94 0.35 -0.76 -0.47 -0.40 0.10 0.23 0.25
-0.18 -0.11 0.60 0.37 0.75 0.51 -0.76 -0.39 -0.81 -0.88 -0.43 -0.88 0.15 -0.46 -0.24

输出
0.3499 0.0803 0.0376 -0.1791 0.3952 0.4112 0.2240 -0.0239 -0.2177 0.4478

样例2
输入
1 3 2
0.58 -0.65 -0.63
-0.74 -0.71 0.65 0.70 -0.14 0.01 -0.84 0.20 0.25
-0.60 0.51 -0.12 -0.35 0.57 -0.38 -0.44 -0.82 0.53
0.14 0.03 -0.27 0.10 -0.12 0.85 -0.55 0.10 -0.43
0.65 0.32 -0.42 -0.62 -0.88 -0.70
-0.66 0.49 0.09 -0.21 0.48 0.41

输出
0.2318 -0.3995 -0.1131


#### 解答


1. LoRA 思路

原始权重 WqW_qWq 冻结；

新增低秩矩阵 $A\in \mathbb{R}^{r\times d}, B\in \mathbb{R}^{d\times r}$，形成：
Wq′=Wq+BAW_q' = W_q + BA
Wq′=Wq+BA

若 r=0r=0r=0，直接用原始 WqW_qWq。

2. Attention 计算步骤

计算
$$Q = XW_q'^\top,\quad K = XW_k^\top,\quad V = XW_v^\top$$

打分并缩放
S=QK⊤dS = \frac{QK^\top}{\sqrt{d}}
S=dQK⊤

对每一行做 稳定 softmax。

输出
O=softmax(S)VO = \text{softmax}(S)V
O=softmax(S)V

题目样例输入输出描述不清，使用矩阵转置才能通过

3. 实现要点

float64 精度，避免溢出；
softmax 时对每行减去最大值；
输出拉平，保留四位小数，-0.0000 特判为 0.0000。

代码实现
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


---

## 2025年9月10日-国内-AI

<a id="第2题-p3639"></a>

### 第2题-历史的窗口搜索（P3639）- 困难





传统TF−IDFTF-IDFTF−IDF方法对突发新闻的敏感度不足，为提高热点词识别效果，某新闻热点追踪平台提出基于历史窗口的TF-IDF方法，具体地，在某个时间点t提出一个查询q时，系统不应该在整个历史文档库中进行大海捞针式的搜索。
相反，它需要智能地聚焦于查询发生时间点 t 前的一段”历史时间窗口“内的文档，并且只需考虑这个窗口内最新的信息。
您的任务就是实现这个“历史窗口”检索引擎的核心逻辑。
历史窗口：仅计算从查询时间点 t 开始之前的 K 篇文档的词频，而非全部历史文档。
动态权重：为了使该搜索模型更关注短期趋势，窗口内越新(文档编号m越大)的文档权重越高，窗口内第j篇文档的权重为(K−j+1)/K(K-j+1)/K(K−j+1)/K(最新文档权重=1=1=1，最旧文档权重=1/K=1/K=1/K)。
筛选与输出：计算查询内容q与窗口内每一篇文档向量之间的余弦相似度(CosineSimilarity)(Cosine Similarity)(CosineSimilarity),返回本次查询中余弦相似度>=0.6>=0.6>=0.6且余弦相似度最高的文档编号m，若未找到满足条件的文档编号(余弦相似度<0.6<0.6<0.6)，则本次查询返回−1-1−1；若存在多个相同最高相似度的文档，返回时间窗口中最早的一篇文档编号。
相似度计算方法：查询q向量(向量A)的第i维计算公式为，qi=TF(wi,q)×IDF(wi)q_i = TF(w_i,q)×IDF(w_i)qi=TF(wi,q)×IDF(wi) ，其中 TF(wi,q)TF(w_i,q) TF(wi,q)表示词 wiw_iwi在所有查询内容queryqueryquery中的词频，IDF(wi)IDF(w_i) IDF(wi)表示词wi w_iwi在窗口文档集合中的中逆文档频率。窗口文档集合中第n篇文档(向量B)的第i维向量计算公式为:di=TF(wi,doc)×IDF(wi)×weightnd_i = TF(w_i,doc) × IDF(w_i) × weight_ndi=TF(wi,doc)×IDF(wi)×weightn,其中 TF(wi,doc)TF(w_i,doc)TF(wi,doc) 表示词 wiw_iwi在查询窗口文档集合中的词频，IDF(wi)IDF(w_i) IDF(wi)表示词 wiw_iwi在窗口文档中的中逆文档频率，weightnweight_nweightn表示第n篇文档的动态权重。
提示:向量A.BA.BA.B的余弦相似度:$\cos (A, B)=\frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|\|\mathbf{B}\|}$
输入描述
输入第一行表示文档集合corpuscorpuscorpus 总数N
第二行开始每一行为从时间点0开始的文档，时间点和文档编号按行递增
(注:文档编号、查询时间点均可理解为数组下标，下标从0开始)
之后的一行为查询窗口的大小K
接下来的下一行表示总查询次数P
然后紧跟每个查询，格式为搜索时间点t具体查询内容q，t和q中间用空格隔开
参数限制
1.1<=K<=1.1<=K<=1.1<=K<=文档总数N
2.0<2.0<2.0<文档总数N<=100N<=100N<=100
3.0<3.0<3.0<总查询次数P<=100P<=100P<=100
在处理完基本题目的I/OI/OI/O操作后，你可能需要实现的函数原型
是historyhistoryhistory_searchsearchsearch(corpus,K，query)(corpus,K，query)(corpus,K，query)，参数说明如下:
1.corpuscorpus corpus：从时间点0开始的文档集合，corpus[i]corpus[i]corpus[i]表示第i篇文档，文档编号m和时间点t均对应数组下标，下标从0开始。
2.K:窗口大小。
3.queryqueryquery：查询列表。每个查询是一个二元组(查询时间点t，查询内容q)。
输出描述
输出空格分隔的最匹配的文档编号m，文档编号和查询顺序 一 一 对应，若两篇文档最高相似度相同，则返回窗口中最早的文档编号
若无匹配则对应位置返回−1-1−1
样例1
输入
5
long short term memory
data science
natural language processing
python for data science
a tutorial on python
3
3
2 natural language
4 python data science
4 long short term memory

输出
2 3 -1

说明
第一行5表示新闻文档集合corpuscorpus corpus的总数N
然后紧跟从时间点0开始的具体的每篇文档，时间点和文档编号按行递增:
时间点0文档编号0："longlonglong shortshortshort termtermterm memorymemorymemory"
时间点1文档编号1：“datadatadata sciencesciencescience”
时间点2文档编号2：“naturalnaturalnatural languageprocessinglanguage processinglanguageprocessing"
时间点3文档编号3：“pythonpythonpython for datadatadata sciencesciencescience"
时间点4文档编号4：“a tutorialtutorialtutorial on pythonpythonpython"
下一行的“3”为窗口的大小K
接下来的“3”表示查询queryqueryquery的个数
然后紧跟每个查询，格式为搜索时间点t具体查询内容q
第一次查询"naturalnaturalnatural languagelanguagelanguage"在时间点2，我们正好可以看到0−20-20−2号文档，因为只有2号文档"naturalnaturalnatural languagelanguagelanguage processingprocessingprocessing"出现了关键词，因此第一个查询返回2
第二次查询"pythonpythonpython datadatadata sciencesciencescience"在时间点4，由于窗口大小是3，我们可以看到2,3,42,3,42,3,4号文档我们无法看到0,10,10,1号文档，本次查询最匹配的是文档编号 3
第三次查询“longlonglong shortshortshort termtermterm memorymemorymemory"在时间点4，同样的我们只能看到2,3,42,3,42,3,4号文档，无法看到0,10,10,1号文档，且2,3,42,3,42,3,4号文档中没有与查询相似文档，
所以本次查询返回−1-1−1
提示
1.qq q可能存在不在窗口文档集合中的查询词，因此在计算逆文档频率的时候你需要进行平滑操作，公式如下IDF(x)=log⁡(N+1N(x)+1)+1I D F(x)=\log \left(\frac{N+1}{N(x)+1}\right)+1IDF(x)=log(N(x)+1N+1)+1其中N代表窗口文档的总数，而N(x)N(x)N(x)代表包含词x的文档总数
2.新闻文档集合corpuscorpuscorpus 所包含的文档内容是空格分隔的小写英文文档字符串


#### 解答


解题思路
问题抽象

文档集合 corpus[0…N-1] 按时间递增（下标既是时间点/编号）。

查询 (t, q) 的检索窗口只看 L = max(0, t-K+1) 到 R = min(t, N-1) 的 M=R-L+1 篇文档。

动态权重：窗口内从新到旧编号第 j 篇（最新 j=1）的权重
weight = (K - j + 1) / K（最新 1，最旧 1/K）。

向量定义（关键）

查询向量：q_i = TF(w_i,q) * IDF(w_i)
文档向量（方案一按题面）：d_i = TF(w_i,doc) * IDF(w_i) * weight_n

相似度：余弦相似度
cos⁡(A,Bn)=A⋅Bn∥A∥∥Bn∥\cos(A,B_n)=\frac{A\cdot B_n}{\|A\|\|B_n\|}
cos(A,Bn)=∥A∥∥Bn∥A⋅Bn
由于 weight_n 乘在文档向量每一维，在分子与分母中会抵消，因而最终分数与不加权相同；但我们仍按题面实现把权重乘进文档分量与其范数的计算中。

筛选：仅返回窗口内相似度≥0.6 的文档中最高者；若并列，取窗口中最早（编号最小）。若无则返回 -1。

细节实现

TF(w,doc) = count(w)/len(doc)；查询同理。
IDF(w) = ln((M+1)/(df(w)+1)) + 1（仅在窗口内，带平滑）。
仅对查询出现的词计算点积与范数，提升效率。
计算时把 weight 乘到文档分量与文档范数里，不再额外乘相似度分子。

复杂度

时间：O(K·L + K·Q)（通常记作 O(K·L)）
空间：O(Q + K)

代码实现

**Python 代码：**

```python
import sys, math
from collections import Counter, defaultdict
input = sys.stdin.readline

def main():
N = int(input().strip())
docs_raw = [input().strip() for _ in range(N)]
K = int(input().strip())
P = int(input().strip())
queries = []
for _ in range(P):
q = input().strip().split(' ')
queries.append((int(q[0]), q[1:]))

# 预处理每篇文档的分词与词频
doc_tokens = [s.split() for s in docs_raw]
doc_lens = [len(tks) for tks in doc_tokens]
doc_cnt = [Counter(tks) for tks in doc_tokens]

ans = []
eps = 1e-12

for t, q_words in queries:
# 窗口 [L, R]
R = min(max(0, t), N - 1)
L = max(0, R - K + 1)
M = R - L + 1

if not q_words or M <= 0:
ans.append(-1)
continue

q_cnt = Counter(q_words)
q_len = len(q_words)

# df & idf（仅对查询词；平滑）
df = defaultdict(int)
q_set = set(q_cnt.keys())
for i in range(L, R + 1):
for w in q_set:
if doc_cnt[i].get(w, 0) > 0:
df[w] += 1
idf = {w: math.log((M + 1.0) / (df[w] + 1.0)) + 1.0 for w in q_set}

# 查询范数
q_norm_sq = 0.0
for w, c in q_cnt.items():
tfq = c / q_len
x = tfq * idf[w]
q_norm_sq += x * x
q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0

best_score, best_id = -1.0, -1

# 新->旧遍历
for idx in range(R, L - 1, -1):
dl = doc_lens[idx] if doc_lens[idx] > 0 else 1
# 位置权重：j=1 最新
j = (R - idx) + 1
weight = (K - j + 1) / K

dot = 0.0
d_norm_sq = 0.0
for w, qc in q_cnt.items():
tfd = doc_cnt[idx].get(w, 0) / dl
tfq = qc / q_len
dq = tfq * idf[w]
dd = tfd * idf[w] * weight   # 方案一：把weight乘进文档维度
dot += dq * dd
d_norm_sq += dd * dd

sim = 0.0
if d_norm_sq > 0:
d_norm = math.sqrt(d_norm_sq)
sim = dot / (q_norm * d_norm)  # 不再额外乘weight

if sim >= 0.6 - eps:
if sim > best_score + eps:
best_score, best_id = sim, idx
elif abs(sim - best_score) <= eps:
best_id = idx if (best_id == -1 or idx < best_id) else best_id

ans.append(best_id if best_id != -1 else -1)

print(' '.join(str(x) for x in ans))

if __name__ == "__main__":
main()

```

---

<a id="第3题-p3640"></a>

### 第3题-多尺寸窗口滑动的特征转换（P3640）- 困难





题目背景:
数据治理阶段经常会碰到特征转换，当前有一个时间序列数据(用 1 维整数数组表示)，和一个窗口序列( 1 维整数数组)，窗口序列中每个元素表示 1 个窗口 w 。
转换要求:
实现一个多尺寸滑动窗口特征转换函数，每个窗口提取 5 个特征，计算出来如果是整数不带小数点，则小数点后最多保留 3 位(四舍五入)
小数点举例说明:
1.0−>11.0->11.0−>1 整数不带小数点
1.10−>1.11.10->1.11.10−>1.1 结尾不带 0
1.1116−>1.1121.1116->1.1121.1116−>1.112 最多保留 3 位小数，最后一位四舍五入
特征包含:
均值(meanmeanmean)
标准差(样本标准差，ddof=1ddof=1ddof=1 ;分母为 0 时，返回 0 )
最小值(min)
最大值(max)
线性趋势斜率(使用线性回归拟合)
线性趋势斜率算法:
时间索引: x=[0,1,2,...,w−1]x = [0, 1, 2,...,w-1]x=[0,1,2,...,w−1]
对应的值: y=[y0,y1,y2,...,yw−1]y = [y_0,y_1, y_2,...,y_{w-1}]y=[y0,y1,y2,...,yw−1]
拟合一条直线:y=βx+αy=βx+αy=βx+α
βββ 是斜率(我们需要的趋势斜率)
ααα 是截距
使用最小二乘法，斜率的计算公式为:
β=[n∗∑(xy)−∑x∗∑y]/[n∗∑(x2)−(∑x)2]β = [n* ∑(xy) - ∑x*∑y] / [n*∑(x^2) - (∑x)^2]β=[n∗∑(xy)−∑x∗∑y]/[n∗∑(x2)−(∑x)2]
n 是窗口大小
标准差计算公式:
$s=\sqrt{\frac{1}{n-1} \sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}$
转换规则:
对于每个窗口大小 w(window_arrayw(window\_arrayw(window_array 中的一个元素)，从索引(索引从 0 开始)i=max(window_arravy)−wi=max(window\_arravy)-wi=max(window_arravy)−w ，开始滑动窗口
所有窗口的特征按 window_arravwindow\_arravwindow_arrav 中的元素顺序排列
如果数组长度小于任意一个窗口大小,则结果数组为空
输入描述
inputinputinput_arrayarrayarray :一维数组,表示时间序列数据
windowwindowwindow_arrayarrayarray:一维数组,多个窗口，每个元素为窗口 w
输出描述
二维数组，形状为 (n,m)(n,m)(n,m) ，其中:
n=len(input_array)−max(window_array)+1n = len(input\_array) - max(window\_array) +1n=len(input_array)−max(window_array)+1
m=len(window_array)∗5m = len(window\_array) * 5m=len(window_array)∗5
每一行说明:
$[mean1, std1, min1, max1, slope1, mean2, std2, min2, max2, slope2,...]$
样例1
输入
[10, 20], [3, 4]

输出
[]

说明
输入长度小于窗口最小大小，输出空数组
样例2
输入
[1, 2, 3, 4, 5], [2, 3]

输出
[2.5, 0.707, 2, 3, 1, 2, 1, 1, 3, 1]
[3.5, 0.707, 3, 4, 1, 3, 1, 2, 4, 1]
[4.5, 0.707, 4, 5, 1, 4, 1, 3, 5, 1]

说明
1、窗口大小 2 的特征:
[2,3]: mean=2.5,std=0.707, min=2, max=3, slope=1
[3,4]: mean=3.5, std=0.707, min=3, max=4, slope=1
[4,5]: mean=4.5, std=0.707, min=4, max=5, slope=1
2、窗口大小 3 的特征:
[1,2,3]: mean=2, std=1, min=1, max=3, slope=1
[2,3,4]: mean=3, std=1, min=2, max=4, slope=1
[3,4,5]: mean=4, std=1, min=3, max=5, slope=1


#### 解答


算法思路

均值 & 标准差

用前缀和与前缀平方和可快速计算：

sum = prefix[i+w] - prefix[i]
mean = sum / w
var = (Σy² - w*mean²) / (w-1)
std = sqrt(max(var,0))

最小值 & 最大值

简单做法：每个窗口遍历一遍 O(w)。
高效做法：用单调队列，O(n) 解决所有窗口的 min/max。

斜率 slope

公式：
slope = (w * Σ(x*y) - Σx * Σy) / (w * Σx² - (Σx)²)

其中 x = 0..w-1，Σx 和 Σx² 可直接公式计算。

特殊情况：w=1 时，分母为 0，slope=0。

对齐方式

输出行数 n = len(a) - max(wins) + 1。
对于窗口大小 w，第 r 行的数据对应起点 idx = r + (Wmax - w)。

复杂度分析

朴素实现：每个窗口 O(w)，总复杂度 O(N * Σw)。
优化实现：前缀和+单调队列+公式，复杂度 O(N * |wins|)。
空间复杂度 O(N)。

代码实现

**Python 代码：**

```python
import sys, re
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

txt = eval(sys.stdin.read())

a = np.array(txt[0], dtype=float)
wins = np.array(txt[1], dtype=int)

if a.size == 0 or wins.size == 0:
print("[]"); sys.exit(0)

Wmax = int(wins.max())
n = a.size - Wmax + 1
if n <= 0:
print("[]"); sys.exit(0)

# —— 工具：数值格式化（最多3位小数，去掉多余0和小数点） ——
def fmt_arr(x: np.ndarray):
y = np.round(x, 3)                                 # 四舍五入到3位
s = np.char.mod('%.3f', y)                         # 统一三位
s = np.char.rstrip(np.char.rstrip(s, '0'), '.')    # 去尾0与点
s = np.where(s == '', '0', s)                      # 处理 -0 -> 0
return s.tolist()

# —— 主逻辑：对每个 w 计算整批窗口特征（均值/标准差/最小/最大/斜率） ——
# 斜率：对 x=0..w-1，用中心化公式 beta = sum((x-x̄)*(y-ȳ)) / sum((x-x̄)^2)
features_per_w = {}  # w -> (mean, std, min, max, slope), 每个都是 shape=(N-w+1,)
N = a.size

for w in wins:
W = sliding_window_view(a, w)            # 形状: (N-w+1, w)
mean = W.mean(axis=1)
# 样本标准差 ddof=1；w=1 时结果为 nan，按题意置 0
std = W.std(axis=1, ddof=1)
std = np.nan_to_num(std, nan=0.0)

mn = W.min(axis=1)
mx = W.max(axis=1)

x = np.arange(w, dtype=float)
xc = x - x.mean()                         # x 中心化
denom = np.sum(xc * xc)                   # 标准化分母
# 将每个窗口的 y 做中心化，再与 xc 点积即可得到所有斜率
yc = W - mean[:, None]
slope = yc @ xc / (denom if denom != 0 else 1.0)
if denom == 0: slope[:] = 0.0            # w=1 的情况

features_per_w[int(w)] = (mean, std, mn, mx, slope)

# —— 对齐规则：第 r 行使用每个 w 的索引 r + (Wmax - w) ——
rows = []
for r in range(n):
parts = []
for w in wins:
idx = r + (Wmax - int(w))
mean, std, mn, mx, slope = features_per_w[int(w)]
parts.extend(fmt_arr(np.array([mean[idx], std[idx], mn[idx], mx[idx], slope[idx]])))
rows.append("[" + ", ".join(parts) + "]")

print("\n".join(rows))

```

---

## 2025年9月4日-留学生-AI

<a id="第2题-p3561"></a>

### 第2题-大模型训练数据均衡分配算法（P3561）- 中等





大模型训练通常采用数据并行的训练方式，处理大规模数据集(样本)，加速训练过程，具休的:
假设有n个NPU，m个样本，把m个样本分给n个NPU，每个NPU上有一份完整模型，各自计算自己的样本数据，其中m>=nm>=nm>=n，保证每个NPU至少分到一个样本，且样本不能切分，一个样本必须完整的被分到个NPUNPU NPU上
每个NPU的运行时间跟所分到的样本的长度和呈正相关。如果每个NPU上的样本长度和相差较大，会形成木桶效应，执行快的NPU等待执行慢的NPU，最终执行时间由最大样本和长度的NPU决定。
试着编号一段程序对样本进行均衡分配，设n个NPU上分得的最大的样本和为lmaxl_{max}lmax，使lmaxl_{max}lmax最小，即求min(lmax)min(l_{max})min(lmax)
输入描述
第一行为1个整数n(0<n<1000)n(0< n< 1000)n(0<n<1000)，表示NPU的个数
第二行为1个整数m(0<m<10000)m(0 < m< 10000)m(0<m<10000)，表示样本的个数
第三行有m个处于区间[1,100000]之内的整数，表示m个样本中每个样本的长度
输出描述
输出1个整数(行尾没有空格)，该数字表示min(lmax)min(l_{max})min(lmax)的值,
样例1
输入
4
7
89 245 64 128 79 166 144

输出
245

说明
样本根据NPU个数进行分组，一共有4个NPU，所以有4个分组，最优样本分配如下:
$(1)79,144 \  (2)245 \      (3)64，166  \      (4)128，89$
求和分别为:223，245，230，217223，245，230，217223，245，230，217；4个NPU中最大的样本长度和为245、所以输出245
样例2
输入
2
3
145 274 100

输出
274

样本根据NPU个数进行分组，一共有2个NPU，3个样本；所以有2个分出，有以下3种分法:
(1)145,274+100；(2)274,145+100；(3)100，145+274(1)145,274+100；(2)274,145+100；(3)100，145+274(1)145,274+100；(2)274,145+100；(3)100，145+274
3种分法的最大样本和分别为:374，274，419374，274，419374，274，419;
所以第2种分法超均衡，最大样本和长度最小，为274，所以输出274

▶️


#### 解答


video solution

本题为np-hard问题，以下给出近似解法

解题思路
算法选择：LPT 贪心 + 最小堆

将样本时长按从大到小排序。
使用一个大小为 n 的最小堆（优先队列）来维护每个 NPU 的当前负载，初值全为 0。
依次取出当前最大的样本，把它放到当前负载最小的 NPU 上（堆顶），更新该 NPU 的负载并压回堆。
全部分配完成后，所有 NPU 负载中的最大值即为答案。

直觉：把大的任务优先安排，并始终让它落到当前最“空闲”的设备，可有效压低最大负载。
该策略在工程中广泛使用，复杂度低、效果稳定；且在测试样例中取得最优值。
正确性要点

任何可行分配的最大负载下界为 max(a[i])（因为最大的任务至少要放到某一台机器上）。
LPT 通过“最大任务优先 + 把它给最空的机器”的组合，尽量不让某一台机器独自承担巨大的额外负载，从而逼近最优。

复杂度分析

排序：O(m log m)
每个样本做一次“取堆顶 + 更新 + 入堆”：O(log n)，总计 O(m log n)
整体复杂度：O(m log m + m log n)，在 m ≤ 1e4, n ≤ 1e3 下完全可行。
额外空间：O(n)（维护堆）。

代码实现

**Python 代码：**

```python
from typing import List
import heapq

def group_samples(group_num: int, sample_num: int, sample_lens: List[int]):
n, m = group_num, sample_num
a = sample_lens

# 特判：没有样本
if m == 0:
print(0)
return

# 将样本按从大到小排序（LPT 的“Longest”）
a.sort(reverse=True)

# 最小堆保存每个 NPU 的当前负载，初始全 0
load = [0] * n
heapq.heapify(load)

# 逐个样本分配到当前最空闲的 NPU（堆顶）
ans = 0
for x in a:
cur = heapq.heappop(load)   # 取出当前最小负载
cur += x                    # 分配该样本
ans = max(ans, cur)         # 维护最大负载
heapq.heappush(load, cur)   # 放回堆

print(ans)

if __name__ == "__main__":
# 读取输入：n, m, 接着一行 m 个数字
n = int(input().strip())
m = int(input().strip())
lens = list(map(int, input().strip().split()))
group_samples(n, m, lens)

```

---

<a id="第3题-p3562"></a>

### 第3题-传感器数据分析（P3562）- 中等





某工业制造企业在其生产线上部署了多台传感器以监控关键设备(如电机、泵、压缩机等)的运行状态。这些传感器周期性地采集设备的多维度运行数据(如温度、振动、压力、电流、转速等)，每隔固定时间窗口会生成一组时序特征数据。为了实现设备早期故障预警，需要对每一组采集到时序数据进行异常检测和评分。工程师们通过人工标记历史数据集，训练出一套多层自注意力(Self−Attention)(Self-Attention)(Self−Attention)+多层全连接层(FC)(FC)(FC)结构的神经网管模型。现在，为了模型的快速部罢与测试，需要根据题目中给定的网络权重参数，编写代码完成端到端推理，输出每一组传感器时序数据的最终导常分数。结构如下图所示:

具体说明如下：

每一组采集数据为一个二维矩阵，尺寸为L，L采样时序长度，D为每次采样包含的特征数(如:10个时间点、每点5个特征)。

网络结构为:两层Self−AttentionSelf-AttentionSelf−Attention，每层后接全连接层FC，最终输出异常分数。为简化起见，网络中无非线性激活函数。

Self−AttentonSelf-AttentonSelf−Attenton采用Dot−productAttentionDot-product AttentionDot−productAttention，计算公式如下:$\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V$

输入描述
第一行:序列长度 L∈[1,10]L∈[1,10]L∈[1,10]、特征维度D∈[1,10] D∈ [1,10]D∈[1,10]
第二行:输入序列，L×DL×DL×D个数
第3~5行:第一层SeIfAttentionSeIfAttentionSeIfAttention参数Wq1W_{q1}Wq1，Wk1W_k1Wk1，Wv1W_v1Wv1每行D×DD×DD×D个数
第6行:第-层FC参数Wfc1W_{fc1}Wfc1 ，D×DD×DD×D 个数
第7行:第一层FC偏置 bfc1b_{fc1}bfc1，D 个数
第8~10行:第二层Self−AttentionSelf-AttentionSelf−Attention参数Wq2W_{q2}Wq2，Wk2W_{k2}Wk2，Wv2W_{v2}Wv2，每行D×DD×DD×D个数
第11行:第层FC参数Wfc2W_{fc2}Wfc2，D×DD×DD×D个数
第12行:第二层FC编置bfc2b_{fc2}bfc2，D个数
输出描述
一行，即最终FC输出，L×DL×DL×D个数
注:数据间用返号隔开，输出结果均保留2位小数
样例1
输入
4,1
1.00,-3.00,9.50,6.50
-0.20
0.45
-0.20
0.60
0.15
0.23
-0.34
0.50
-0.32
0.05

输出
0.04,0.04,0.05,0.05

说明
输入：
首行:4 14\ 14 1表示序列长度L=4L=4L=4，特征维度D=1D=1D=1
第二行:1.00,−3.00,9.50,6.501.00,-3.00,9.50,6.501.00,−3.00,9.50,6.50
输入序列为4×1=4 4×1=44×1=4，即4个时刻的传感器数据
第3~5行:第一层SelfAttentionSelfAttentionSelfAttention参数Wq1W_{q1}Wq1,Wk1W_{k1}Wk1，Wv1W_{v1}Wv1(每行1个数)
第6行:第一层FC参数Wfc1W_{fc1}Wfc1(1个数)
第7行:第一层FC偏置 bfc1b_{fc1}bfc1(1个数)
第8~10行:第二层Self−AttentionSelf-AttentionSelf−Attention参数Wq2,Wk2,WV2W_{q2},W_{k2},W_{V2}Wq2,Wk2,WV2(每行1个数)
第11行:第二层FC参数 Wfc2W_{fc2}Wfc2(1个数)
第12行:第二层FC偏置 bfc2b_{fc2}bfc2(1个数)
输出：
最终FC输出的4×1=44×1=44×1=4个数，英文逗号分隔。
样例2
输入
2,2
1.00,2.00,3.00,4.00
0.10,0.20,0.30,0.40
-0.10,-0.20,-0.30,-0.40
0.50,0.60,0.70,0.80
-0.50,-0.60,-0.70,-0.80
0.01,0.02
0.11,0.12,0.13,0.14
-0.11,-0.12,-0.13,-0.14
0.21,0.22,0.23,0.24
-0.21,-0.22,-0.23,-0.24
0.03,0.04

输出
0.66,0.69,0.66,0.69

说明
输入：
首行:2 22\ 22 2表示序列长度L=2L=2L=2，特征维度D=2D=2D=2
第二行:1.00,2.00,3.00,4.001.00,2.00,3.00,4.001.00,2.00,3.00,4.00
输入序列为2×2 2×22×2，分别为第1时刻(1.00,2.00)(1.00,2.00)(1.00,2.00)，第2时刻(3.00,4.00)(3.00,4.00)(3.00,4.00)
第3~5行:第一层SelfAttentionSelfAttentionSelfAttention参数Wq1W_{q1}Wq1，Wk1W_{k1}Wk1，Wv1W_{v1}Wv1(每行2×22×22×2个数)
第6行:第一层FC参数Wfc1W_{fc1}Wfc1(2×2 2×22×2个数)
第7行:第一层FC偏置 bfc1b_{fc1}bfc1(2个数)
第8~10行:第二层Self−AttentionSelf-AttentionSelf−Attention参数Wq2,Wk2,WV2W_{q2},W_{k2},W_{V2}Wq2,Wk2,WV2(每行2×22×22×2个数)
第11行:第二层FC参数 Wfc2W_{fc2}Wfc2(2×22×22×2个数)
第12行:第二层FC偏置 bfc2b_{fc2}bfc2(2个数)
输出：
最终FC输出的2×2=42×2=42×2=4个数，英文逗号分隔。


#### 解答


解题思路
1) 关键算法

Scaled Dot-Product Self-Attention
对输入 X (L×D)：

线性映射：Q=XWq, K=XWk, V=XWv（均为 L×D）。
打分：S = QK^T / sqrt(D)（L×L）。
归一化：对 S 每行做 softmax 得到注意力矩阵 A。
聚合：Y = A V（L×D）。

全连接(FC)：Y = XW + b，其中 b 对 L 行广播。

2) 实现要点

只涉及矩阵乘、softmax、广播，加上两次拼装即可。
softmax 为数值稳定：对每行先减去行最大值。
按输入顺序读入、按形状 reshape 成矩阵。
输出格式化为两位小数并用英文逗号连接。

3) 正确性说明（简要）
Attention 依公式对每个时刻的表征与所有时刻交互；两层堆叠后再经 FC，完全符合题面给出的结构图与公式，因此结果唯一确定。
4) 复杂度分析

两次注意力各需：

QK^T：O(L^2·D)
AV：O(L^2·D)

两次 FC：O(L·D^2)
综合：时间复杂度 O(L^2·D + L·D^2)；空间复杂度 O(L^2 + L·D)（存储注意力矩阵与中间结果）。

参考实现

**Python 代码：**

```python
import sys
import numpy as np
import math

class Solution:
def analyze_data(self, L: int, D: int,
seq: np.ndarray,
Wq1: np.ndarray, Wk1: np.ndarray, Wv1: np.ndarray,
Wmlp1: np.ndarray, bmlp1: np.ndarray,
Wq2: np.ndarray, Wk2: np.ndarray, Wv2: np.ndarray,
Wmlp2: np.ndarray, bmlp2: np.ndarray) -> np.ndarray:
# —— 行 softmax（数值稳定）——
def softmax_rows(M: np.ndarray) -> np.ndarray:
mx = np.max(M, axis=1, keepdims=True)   # 每行最大值
E = np.exp(M - mx)                      # 减去最大值避免上溢
S = np.sum(E, axis=1, keepdims=True)    # 每行求和
return E / S

# —— 单层 Scaled Dot-Product Self-Attention ——
def attn(X: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray) -> np.ndarray:
Q = X @ Wq                  # (L,D)
K = X @ Wk                  # (L,D)
V = X @ Wv                  # (L,D)
S = (Q @ K.T) / math.sqrt(D)  # (L,L)  /sqrt(D) 缩放
A = softmax_rows(S)         # (L,L)
return A @ V                # (L,D)

X = seq                         # (L,D)

# 第一层 Self-Attention -> FC
Y1 = attn(X, Wq1, Wk1, Wv1)     # (L,D)
Z1 = Y1 @ Wmlp1 + bmlp1         # (L,D) + (D,)  行广播

# 第二层 Self-Attention -> FC
Y2 = attn(Z1, Wq2, Wk2, Wv2)    # (L,D)
Z2 = Y2 @ Wmlp2 + bmlp2         # (L,D)

return Z2                       # 返回最终 (L,D)

if __name__ == "__main__":
# 固定读取 12 行（与模板保持一致）
lines = [sys.stdin.readline().strip() for _ in range(12)]
L, D = map(int, lines[0].split(','))

def parse_line(idx: int, count: int):
# 按英文逗号分隔（模板要求），读取指定数量的浮点
values = list(map(float, lines[idx].split(',')))
assert len(values) == count, f"Line {idx} expected {count} values"
return np.array(values, dtype=np.float64), idx + 1

idx = 1
seq_flat, idx = parse_line(idx, L * D)
seq = seq_flat.reshape(L, D)

Wq1_flat, idx = parse_line(idx, D * D)
Wk1_flat, idx = parse_line(idx, D * D)
Wv1_flat, idx = parse_line(idx, D * D)
Wmlp1_flat, idx = parse_line(idx, D * D)
bmlp1, idx = parse_line(idx, D)

Wq2_flat, idx = parse_line(idx, D * D)
Wk2_flat, idx = parse_line(idx, D * D)
Wv2_flat, idx = parse_line(idx, D * D)
Wmlp2_flat, idx = parse_line(idx, D * D)
bmlp2, idx = parse_line(idx, D)

# reshape all matrices（名称与模板一致）
Wq1 = Wq1_flat.reshape(D, D)
Wk1 = Wk1_flat.reshape(D, D)
Wv1 = Wv1_flat.reshape(D, D)
Wmlp1 = Wmlp1_flat.reshape(D, D)

Wq2 = Wq2_flat.reshape(D, D)
Wk2 = Wk2_flat.reshape(D, D)
Wv2 = Wv2_flat.reshape(D, D)
Wmlp2 = Wmlp2_flat.reshape(D, D)

# bias 保持 (D,)；numpy 与 (L,D) 相加会做按行广播
solver = Solution()
result = solver.analyze_data(L, D, seq,
Wq1, Wk1, Wv1, Wmlp1, bmlp1,
Wq2, Wk2, Wv2, Wmlp2, bmlp2)

# 输出结果：保留两位小数
flat = result.flatten()
print(','.join(f"{x:.2f}" for x in flat))

```

---

## 2025年9月3日-国内-AI

<a id="第2题-p3553"></a>

### 第2题-大模型训练MOE场景路由优化算法（P3553）- 中等





MOE 模型训练时，tokentokentoken 根据概率发送到 topktopktopk 个不同的专家进行计算。这些专家分布在多个 NPU 卡上。Device−LimitedrDevice-LimitedrDevice−Limitedr outingoutingouting 算法将 tokentokentoken 的路由目标限制在 P 个 NPU 上，可以有效降低通信成本。具体的：

把 n 个专家平均分配在 m 个 NPU 上，每个 NPU 上的专家为一个组;设 n 个专家的编号为 N=[0,1,2,…,n−1]N=[0,1,2,…,n-1]N=[0,1,2,…,n−1] ，同一个专家组上的专家编号是连续的;

每个专家对应一个概率，表示被路由到的可能性;用每个组中的最大概率作为本组代表，从所有组中选择概率最大的 p 个组，其所在的 NPU 即为路由目标限制 NPU ;

再从上述 p 个 NPU 对应的所有专家概率中选择 k 个最大的概率对应的专家编号作为最终路由目标。

试着编写一段程序，实现以上路由算法。
输入描述
第一行有 4 个处于区间 [1,10000] 之内的整数，第 1 个表示专家的个数 n ，第 2 个表示 NPU 个数 m ，第 3 个表示路由目标限制 NPU 个数 p ，第 4 个表示目标路由专家个数 k ;
第二行有 n 个处于区间 (0,1)(0,1)(0,1) 之内的浮点数，表示每个专家对应的概率值，这 n 个数对应的专家的编号为 [0,1,2,...,n−1][0,1,2,...,n-1][0,1,2,...,n−1] ;
输出描述
如果，n 不能被 m 整除或者获取不到 k 个专家编号，输出 errorerrorerror ;
否则，按照从小到大的顺序，输出 k 个专家编号,任意相邻两数之间有空格，最后一个数字(行尾没有空格)
样例1
输入
8 4 4 2
0.5 0.01 0.09 0.023 0.027 0.05 0.1 0.2

输出
0 7

说明
将专家分成 4 组，分别为：(1)0.5(1)0.5(1)0.5 0.010.010.01 (2)0.09(2)0.09(2)0.09 0.0230.0230.023 (3)0.027(3)0.027(3)0.027 0.050.050.05 (4)0.1(4)0.1(4)0.1 0.20.20.2
限定专家为 4 ，则 4 组都被选定，从选定的 4 组中，选择 2 个专家，分别是 0.50.50.5 和 0.20.20.2 对应的专家，对应的编号分别是 0 和 7 ，按照升序排个列，输出: 0 7
样例2
输入
8 4 5 2
0.3 0.04 0.06 0.45 0.05 0.01 0.03 0.06

输出
error

说明
NPU 一共只有 4 个，需要限定 5 个 NPU ，不满足条件，输出 errorerrorerror

▶️


#### 解答


video solution

思路与方法
给定 n 个专家，平均分布在 m 张 NPU 上（每张卡上一组，组内专家编号连续）。算法分三步：

按组取代表
组大小为 g=n/mg = n/mg=n/m。对每一组，找到组内最大概率以及对应的专家编号，作为该组代表值。

选路由目标 NPU（选 p 个组）
将所有组按“代表概率”从大到小排序，取前 p 个组（对应的 NPU）。若 p>mp>mp>m，直接输出 error。

在选定的 p 个组里选 k 个专家
将这 p 个组中的所有专家收集起来，按概率从大到小挑选前 k 个专家编号作为最终路由目标。若可选专家数 p⋅g<kp\cdot g < kp⋅g<k，输出 error。
为了结果可复现，概率相同时按专家编号小的优先。

最后把选出的 k 个专家编号按升序输出（空格分隔，行尾无空格）。
复杂度分析

计算每组最大值：遍历一次，O(n)O(n)O(n)。
选出前 p 个组：对 m 个代表排序，O(mlog⁡m)O(m\log m)O(mlogm)（或用大小为 p 的堆 O(mlog⁡p)O(m\log p)O(mlogp)）。
在 p 个组里选出前 k 个专家：对 p⋅gp\cdot gp⋅g 个元素排序，O((p⋅g)log⁡(p⋅g))O((p\cdot g)\log (p\cdot g))O((p⋅g)log(p⋅g))（或用堆 O((p⋅g)log⁡k)O((p\cdot g)\log k)O((p⋅g)logk)）。
总体：在数据范围 n≤104n\le 10^4n≤104 下，直接排序实现已足够高效，代码更简洁。

实现要点

组索引：第 i 组覆盖的专家编号区间为 [i⋅g, (i+1)⋅g−1][i\cdot g,\ (i+1)\cdot g-1][i⋅g, (i+1)⋅g−1]。

排序键：

选组时：按 (组代表概率 desc, 组索引 asc) 稳定选择。
选专家时：按 (概率 desc, 专家编号 asc)。

输出：最终 k 个编号再升序打印。

参考实现

**Python 代码：**

```python
import sys

def main():
data = sys.stdin.read().strip().split()
if len(data) < 4:
print("error")
return
it = iter(data)
try:
n = int(next(it)); m = int(next(it)); p = int(next(it)); k = int(next(it))
except:
print("error"); return

# 读取 n 个概率
probs = []
for _ in range(n):
try:
probs.append(float(next(it)))
except:
print("error"); return

# 基本校验
if n % m != 0:
print("error"); return
if p > m:
print("error"); return

g = n // m  # 每组大小

# 1) 计算每组代表（最大概率及其专家编号）
group_repr = []  # (max_prob, group_id, expert_idx_of_max)
for gi in range(m):
L = gi * g
R = L + g
max_prob = -1.0
max_idx = -1
# 组内扫描找最大值；并用较小编号打破平局
for idx in range(L, R):
pr = probs[idx]
if pr > max_prob or (abs(pr - max_prob) < 1e-18 and idx < max_idx):
max_prob = pr
max_idx = idx
group_repr.append((max_prob, gi, max_idx))

# 2) 选择前 p 个组（按代表概率降序；组索引升序打破平局）
group_repr.sort(key=lambda x: (-x[0], x[1]))
chosen_groups = set([gi for _, gi, _ in group_repr[:p]])

# 3) 收集这些组的所有专家并选前 k 个（按概率降序，编号升序）
pool = []
for gi in chosen_groups:
L = gi * g
R = L + g
for idx in range(L, R):
pool.append((probs[idx], idx))

if len(pool) < k:
print("error"); return

pool.sort(key=lambda x: (-x[0], x[1]))
chosen = [idx for _, idx in pool[:k]]

chosen.sort()
print(" ".join(map(str, chosen)))

if __name__ == "__main__":
main()

```

---

<a id="第3题-p3552"></a>

### 第3题-云存储设备故障预测（P3552）- 中等





在云存储系统中，需要预测存储设备故障以提前迁移数据。每条设备日志包含:
设备 ID ，写入次数，读取次数，平均写入延迟 (ms)(ms)(ms) ，平均读取延迟 (ms)(ms)(ms) ，设备使用年限(年)，设备状态(0正常/1故障)
你需要实现一个设备故障预测系统。包含以下功能:
1、数据清洗:

缺失值标记为"NaN"，用该字段有效值的均值填充

异常值范围:
写入/读取次数:<0<0<0
平均写入/读取延迟:<0<0<0或>1000>1000>1000
使用年限:<0<0<0或>20>20>20
异常值用该字段有效值的中位数替换

2、逻辑回归模型:

使用批量梯度下降法 (Batch(Batch(Batch GD)GD)GD) 训练，每次迭代使用全部样本

特征:[写入次数，读取次数，平均写入延迟，平均读取延迟，设备使用年限]

标签:设备状态

参数:迭代 100 次，学习率 α=0.01α=0.01α=0.01，初始权重全 0

3、预测输出:
预测结果: 0 (正常)或 1 (故障)
输入描述
第一行为训练总个数 N，(2<=N<=100)N，(2<=N <= 100)N，(2<=N<=100)
第二行起连续 N 行训练数据，每个训练数据包含:设备ID，写入次数，读取次数，平均写入延迟，平均读取延迟，设备使用年限，状态
第 N+2N+2N+2 行为预测数据总个数 M，(1<=M<=10)M，(1<=M<=10)M，(1<=M<=10)
第 N+3N+3N+3 行起连续 M 行预测数据，每个预测数据包含:设备 ID ，写入次数，读取次数，平均写入延迟，平均读取延迟，设备使用年限，状态
输出描述
M 行预测结果
样例1
输入
5
dev1,NaN,-50,NaN,-2.0,25,0
dev2,180,90,18.0,9.0,4,0
dev3,NaN,80,1500.0,800.0,NaN,0
dev4,-100,-50,-5.0,-2.0,-1,0
dev5,200,NaN,20.0,NaN,5,1
2
dev_predict1,80,40,NaN,2.0,2,0
dev_predict2,210,105,18.0,9.8,4,0

输出
0
0

说明
1、预测数据包含缺失值"NaN"，需要数据清洗
2、M 值为 2 ，输出分为 2 行，第一行表示“dev_predict1"设备的预测结果为 0 ，第二行表示 “dev_predict2” 设备的预期结果为 0
样例2
输入
3
dev1,100,50,20.1,10.2,2,0
dev2,150,80,25.3,NaN,3,1
dev3,120,60,22.4,15.0,1,0
1
dev_predict1,130,70,21.0,12.0,2,0

输出
1

说明
输出"dev_predict1“设备的预测结果为 1
提示
线性组合 z：
z=w0+∑i=15wixiz=w_0+\sum^5_{i=1}w_ix_iz=w0+∑i=15wixi
概率函数 P(y=1)P(y=1)P(y=1) ：
P(y=1)=11+e−zP(y=1)=\frac{1}{1+e^{-z}}P(y=1)=1+e−z1
预测规则：

▶️


#### 解答


video solution

解题思路
1) 数据清洗（按列统计 ➜ 按行替换）

将每条日志按逗号切分：第 1 列为设备ID；最后一列为标签 y∈{0,1}；中间前 5 列依次为特征：
写入次数、读取次数、平均写入延迟(ms)、平均读取延迟(ms)、设备使用年限(年)。
（若行里意外多出字段，取前 5 个数值作特征、最后一个作标签，兼容样例1）

缺失值填充：若特征中出现字符串 "NaN"，视为缺失，用该列有效值的均值填充（仅用训练集估计）。

异常值矫正：按规则判定异常并用该列有效值（在合法区间内）中位数替换：

写入/读取次数：< 0
平均写入/读取延迟：< 0 或 > 1000
设备使用年限：< 0 或 > 20

测试集用训练集的均值/中位数进行同样处理，保证一致性。

2) 模型与训练

使用带偏置项的逻辑回归（Logistic Regression），损失为对数损失。

优化：批量梯度下降（Batch GD）

学习率 α = 0.01，迭代 100 次；
参数初始化为 0；
梯度：对每次迭代，用全量样本累加梯度再更新。

预测：sigmoid(z) ≥ 0.5 判为 1，否则 0。

3) 复杂度分析

设训练样本数 N (≤100)，特征数 d=5，迭代 T=100：

统计均值/中位数：O(N*d log N)（中位数排序或用选择算法可降到线性）
训练：O(T * N * d)
预测：O(M * d)（M ≤ 10）

在本题数据范围内，时间与内存都非常充裕。

4) 边界与实现细节

标签永远取最后一列；特征只取ID 后的前 5 个数值（与样例1兼容）。
若某列“有效值”为空（极端情况），中位数回退为该列均值，再不行则 0。
数值转换时忽略空格；"NaN"（大小写敏感）按缺失处理。
sigmoid 计算做简单溢出保护（例如截断 z）。

参考实现

**Python 代码：**

```python
import sys, math

def parse_line(line):
parts = [p.strip() for p in line.strip().split(',')]
if not parts: return None
id_ = parts[0]
if len(parts) < 7:
# 不足字段，直接跳过（题面不会出现）
return None
# 特征：紧跟在ID后的前5个数值；标签：最后一个
feats_raw = parts[1:6]
y_raw = parts[-1]
def to_num(s):
if s == "NaN": return None
try:
return float(s)
except:
return None
x = [to_num(v) for v in feats_raw]
# 标签按最后一列，容忍浮点写法
y = 0
try:
y = int(float(y_raw))
except:
y = 0
return id_, x, y

# 合法区间判断
def valid(col, v):
if v is None: return False
if col in (0,1):  # 写/读次数
return v >= 0
if col in (2,3):  # 延迟
return 0 <= v <= 1000
if col == 4:      # 年限
return 0 <= v <= 20
return True

def median(vals):
n = len(vals)
if n == 0: return 0.0
vals2 = sorted(vals)
mid = n // 2
if n % 2 == 1:
return vals2[mid]
else:
return 0.5 * (vals2[mid - 1] + vals2[mid])

def sigmoid(z):
# 简单数值稳定
if z > 30: z = 30
if z < -30: z = -30
return 1.0 / (1.0 + math.exp(-z))

def clean_matrix(X, means, meds):
# 替换缺失 -> 均值；异常 -> 中位数
n = len(X)
d = len(X[0]) if n else 5
out = []
for i in range(n):
row = []
for j in range(d):
v = X[i][j]
if v is None:
v = means[j]
# 异常替换
if not valid(j, v):
v = meds[j]
row.append(v)
out.append(row)
return out

def main():
data = sys.stdin.read().strip().splitlines()
if not data:
return
it = 0
# 读 N
while it < len(data) and data[it].strip() == "":
it += 1
N = int(data[it].strip()); it += 1

# 读训练集
trainX_raw, trainY = [], []
for _ in range(N):
while it < len(data) and data[it].strip() == "":
it += 1
id_, x, y = parse_line(data[it]); it += 1
trainX_raw.append(x)
trainY.append(y)

# —— 统计每列“有效值”的均值与中位数（仅用训练集的有效值）——
d = 5
means = [0.0] * d
meds  = [0.0] * d

for j in range(d):
valid_vals = [row[j] for row in trainX_raw if valid(j, row[j])]
if valid_vals:
means[j] = sum(valid_vals) / len(valid_vals)
# 中位数
s = sorted(valid_vals)
n = len(s)
meds[j] = s[n//2] if n % 2 == 1 else 0.5 * (s[n//2 - 1] + s[n//2])
else:
# 没有任何有效值时的回退
means[j] = 0.0
meds[j]  = 0.0

# 清洗训练集
trainX = clean_matrix(trainX_raw, means, meds)

# 读 M
while it < len(data) and data[it].strip() == "":
it += 1
M = int(data[it].strip()); it += 1

# 读测试集（忽略其提供的状态列，仅用于输入格式）
testX_raw = []
for _ in range(M):
while it < len(data) and data[it].strip() == "":
it += 1
parts = [p.strip() for p in data[it].strip().split(',')]
it += 1
feats_raw = parts[1:6]  # 取前5个特征
def to_num(s):
if s == "NaN": return None
try:
return float(s)
except:
return None
testX_raw.append([to_num(v) for v in feats_raw])

testX = clean_matrix(testX_raw, means, meds)

# 训练逻辑回归（批量GD）
n = len(trainX)
w = [0.0]*(d+1)  # w[0] 为偏置
alpha = 0.01
T = 100

for _ in range(T):
g = [0.0]*(d+1)
for i in range(n):
z = w[0]
for j in range(d):
z += w[j+1] * trainX[i][j]
p = sigmoid(z)
diff = p - trainY[i]
g[0] += diff
for j in range(d):
g[j+1] += diff * trainX[i][j]
# 参数更新（平均梯度）
for k in range(d+1):
w[k] -= alpha * g[k] / n

# 预测
out_lines = []
for i in range(M):
z = w[0]
for j in range(d):
z += w[j+1] * testX[i][j]
p = sigmoid(z)
pred = 1 if p >= 0.5 else 0
out_lines.append(str(pred))
print("\n".join(out_lines))

if __name__ == "__main__":
main()

```

---

## 2025年8月27日-国内-AI

<a id="第2题-p3479"></a>

### 第2题-标签样本数量（P3479）- 中等





KNN 算法的核心思想是，如果一个样本在特征空间中的 K 个最相邻的样本中的大多数属于某一个类别，则该样本也属于这个类别，并具有这个类别上样本的特性。请按照下面的步理，实现 KNNKNN
KNN 算法。
KNN 算法说明：
计算待分类点到其他样本点的距离；
通过距离进行排序，选择距离最小的 K 个点；提取这 K 个临近点的类别，根据少数服从多数的原则，将占比最多的那个标签赋值给待分类样本点的 labellabellabel 。
本题说明：
1、给定数据集中，默认每一类标签都存在数据，不存在某类型数量为 0 的场景；
2、为消除不同特征权重问题，给出数据均已做好归一化处理，并保留两位小数；
3、出现并列第一的情形时，取并列第一的样本中，最近邻居的标签返回；
4、距离函数定义为: dx,y=∑i=1n(xi−yi)2d_{x,y}=\sqrt{\sum^n_{i=1}(x_i-y_i)^2}dx,y=∑i=1n(xi−yi)2。
输入描述
第 1 行：k m n s ：k 代表每次计算时选取的最近邻居个数(不大于 20 )，m 代表样本数量(不大于 200 )，n 代表样本维度(不包括标签，不大于 5 )，s 代表类别个数(不于 5 )；
第 2 行：待分类样本
第 3 行~第 m+2m+2m+2 行：m 个样本，每一行 n+1n+1n+1 列，最后一列为类别标签 labellabellabel
输出描述
输出待分类样本的类别标签及距离最小的 K 个点中的该标签样本数量
样例1
输入
3 10 2 3
0.81 0.64
0.19 0.2 1.0
0.18 0.14 0.0
0.76 0.58 1.0
0.4 0.16 1.0
0.98 0.85 0.0
0.42 0.97 1.0
0.75 0.26 1.0
0.24 0.06 1.0
0.97 0.8 0.0
0.21 0.1 2.0

输出
0 2

说明
第 1 行输入说明输入了 m=10m=10m=10 个样本，每个样本有 n=2n=2n=2 个维度的数据(去除最后一列标签)，共有 s=3s=3s=3 种类别
第 2 行输入待分类样本的 n 维数据
从第 3 行到第 12 行的前两列数据为输入的 m=10m=10m=10 个样本，每个样本有 n=2n=2n=2 个维度的数据+最后一列的标签数据
待分类样本 [0.81[0.81[0.81 0.64]0.64]0.64] 最近的前 k=3k=3k=3 个邻居分别为：[0.76[0.76[0.76 0.58],[0.980.58],[0.980.58],[0.98 0.85],[0.970.85],[0.970.85],[0.97 0.8]0.8]0.8] ，分别有 2 个 0 号标签和 1 个 1 号标签 0 号标签占多，返回 0 以及标签 0 的样本数量 2
样例2
输入
6 10 2 4
0.78 0.63
0.57 0.07 1.0
0.5 0.13 1.0
0.83 0.07 3.0
0.27 0.87 3.0
0.81 0.44 2.0
0.21 0.73 3.0
0.45 0.91 1.0
0.12 0.22 2.0
0.25 0.48 0.0
0.54 0.87 1.0

输出
1 2

说明
本样例的距离最小的 6 个样本中，标签 1 和标签 3 出现次数都是 2 次，并列第一；虽然 [0.8[0.8[0.8 0.44]0.44]0.44] 距离样本最近，但其标签 2 不是出现最多的，排除在下一轮统计样本中此时需要从标签 1 和标签 3 中的样本中，选取距离最近的 [0.54[0.54[0.54 0.87]0.87]0.87] 的标签 1 作为返回值，并同时返回标签 1 的样本数量 2 。

▶️


#### 解答


video solution

解题思路
核心步骤

读入参数：k,m,n,sk, m, n, sk,m,n,s；读入待分类样本向量 q（维度 n）；读入 m 条样本（前 n 列为特征，最后一列为标签）。

计算距离：对每个样本 x，计算与 q 的欧氏距离
d(q,x)=∑i=1n(qi−xi)2d(q,x)=\sqrt{\sum_{i=1}^{n}(q_i-x_i)^2}
d(q,x)=i=1∑n(qi−xi)2
为了效率与不影响排序，可直接用平方距离（省去开方，单调性一致）。

排序取前 k：按距离从小到大排序，取前 k 个邻居。

投票与并列规则：统计前 k 个邻居的标签频次，找出最高频数。若有多个标签并列第一，则在这几个标签中，选择距离最近的那个邻居的标签（即在已排序的前 k 邻居中，从前往后找到第一个其标签属于“并列集合”的样本）。

输出：输出最终预测标签与在前 k 中该标签出现的次数，格式：“label count”。

正确性说明

归一化保证各维度量纲一致，欧氏距离可直接比较。
使用平方距离与开方距离等价于排序目的。
并列处理遵循题意“序列第一（最近邻）优先”。

复杂度分析

距离计算：O(m⋅n)O(m\cdot n)O(m⋅n)
排序：O(mlog⁡m)O(m\log m)O(mlogm)
统计投票：O(k)O(k)O(k)
总复杂度：O(mlog⁡m+m⋅n)O(m\log m + m\cdot n)O(mlogm+m⋅n)，在 m≤200,n≤5m\le 200, n\le 5m≤200,n≤5 的限制下完全可行。
额外空间：存距离与样本索引 O(m)O(m)O(m)。


**Python 代码：**

```python
import sys
from collections import Counter

def main():
# 读入所有标记，适配行内/换行混排
tokens = sys.stdin.read().strip().split()
it = iter(tokens)

# 基本参数
k = int(next(it)); m = int(next(it)); n = int(next(it)); s = int(next(it))  # s未直接使用

# 待分类样本 q
q = [float(next(it)) for _ in range(n)]

# 读入 m 个样本（n 个特征 + 1 个标签）
X = []
y = []
for _ in range(m):
row = [float(next(it)) for __ in range(n + 1)]
X.append(row[:n])
# 标签以 float 给出，输出需要整数格式
y.append(int(row[-1]))

# 计算平方欧氏距离，保存 (dist2, idx)
dists = []
for i in range(m):
xi = X[i]
# 平方距离即可用于排序
dist2 = 0.0
for j in range(n):
diff = q[j] - xi[j]
dist2 += diff * diff
dists.append((dist2, i))

# 按距离升序排序
dists.sort(key=lambda t: t[0])

# 取前 k 个邻居的索引与标签
top_idx = [dists[i][1] for i in range(min(k, m))]
top_labels = [y[i] for i in top_idx]

# 统计频次
cnt = Counter(top_labels)
max_freq = max(cnt.values())

# 找出并列第一的标签集合
tie_labels = {lab for lab, c in cnt.items() if c == max_freq}

# 若并列，按距离顺序选择第一个属于并列集合的邻居的标签
# dists 已整体排序，这里只需在前 k 中寻找
chosen = None
for i in range(min(k, m)):
lab = y[dists[i][1]]
if lab in tie_labels:
chosen = lab
break

# 输出：标签 与 在前 k 中该标签出现次数
print(chosen, cnt[chosen])

if __name__ == '__main__':
main()

```

---

<a id="第3题-p3480"></a>

### 第3题-F1值最优的决策树剪枝（P3480）- 中等





决策树生成算法递归地产生决策树，直到不能继续下去为止，这样产生的树往往对训练数据的分类很准确，但对未知的测试数据的分类却没有那么准确，即出现过拟合现象。
在决策树学习中将已生成的树进行简化的过程称为剪枝。具体地，剪枝从已生成的树上裁掉一些子树或叶节点，并将其根节点或父节点作为新的叶节点，从而简化分类树模型。
小A希望通过决策树的方法解决一个二分类任务。在该二分类的任务中，标签 1 是正分类，标签 0 是负分类。现在小A已经训练了一个未剪枝的二分类的决策树。他希望对该决策树进行剪枝，能够在验证集上达到最优的 F1F1F1 值。
给定一个二叉树为待剪枝的二分类决策树，每个节点有 3 个参数 fi、thi、labelif_i、th_i、label_ifi、thi、labeli 。当节点非叶节点时，fi、thif_i、th_ifi、thi 表示该节点决策应用的特征编号和阈值。在数据的第 fif_ifi 个特征小于等于 thith_ithi 时决策走左节点，大于 thith_ithi 时走右节点。决策树的预测通过该规则推理到叶节点时，叶节点的 labelilabel_ilabeli 为该条数据的预测结果。
请输出小A通过剪枝在验证集上可以达到的最优 F1F1F1 值。
输入描述
第一行为一个 N、M、KN、M、KN、M、K 。其中，N(1<=N<=100)N(1<=N<=100)N(1<=N<=100) 表示决策树的节点个数。M(1<=M<=300)M(1<=M<=300)M(1<=M<=300) 表示验证集条数。K(1<=K<=100)K(1<=K<=100)K(1<=K<=100) 表示每条验证集特征个数。
随后 N 行，第 i 行表示第 i 个节点，根节点编号为 1 ，每行包括 5 个整数 li、ri、fi、thi、labelil_i、r_i、f_i、th_i、label_ili、ri、fi、thi、labeli 。其中 li、ril_i、r_ili、ri 分别表示节点的左右子节点编号 (0<=liri<=100)(0<=l_ir_i<=100)(0<=liri<=100) 。若 li=0、ri=0l_i =0、r_i=0li=0、ri=0 则表示无子节点，不存在只有一个子节点的情况。当节点非叶节点时，fi、thif_i、th_ifi、thi 表示该节点的特征编号和阔值，否则 fi、thif_i、th_ifi、thi 为 0 。labelilabel_ilabeli 表示当该节点作为叶节点时的分类结果( labelilabel_ilabeli 取值为 0 或 1 )。
随后 M 行为验证集特征和 labellabellabel，每行 K+1K+1K+1 个整数，前 K 个整数为该条数据的特征，最后一个整数位该条数据的 labellabellabel 。
输出描述
请输出一个浮点数，为验证集可达到的最优 F1F1F1 值，四舍五入保留小数点后 6 位。
样例1
输入
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

输出
0.800000

说明
原始决策树为

第一条数据的终止节点为 5 ，节点 predictlabelpredict_labelpredictlabel 为 1 ，预测正确。
第二条数据的终止节点为 4 ，节点 predictlabelpredict_labelpredictlabel 为 0 ，预测错误。
第三条数据的终止节点为 6 ，节点 predictlabelpredict_labelpredictlabel 为 0 ，预测错误。
PrecisionPrecisionPrecision 为 1 , RecallRecallRecall 为 1/3,F11/3,F11/3,F1 ScoreScoreScore 为 1/21/21/2 。
决策树可将节点 6、76、76、7 裁剪掉，裁剪后的决策树为：

第一条数据的终止节点为 5 ，节点 predictlabelpredict_labelpredictlabel 为 1 ，预测正确。
第二条数据的终止节点为 4 ，节点 predictlabelpredict_labelpredictlabel 为 0 ，预测错误。
第三条数据的终止节点为 3 ，节点 predictlabelpredict_labelpredictlabel 为 1 ，预测正确。
PrecisionPrecisionPrecision 为 1,Recall1,Recall1,Recall 为 2/3,F12/3,F12/3,F1 ScoreScoreScore 为 4/5=0.8000004/5=0.8000004/5=0.800000 。
样例2
输入
7 3 3
2 3 3 87 1
0 0 1 3 0
4 5 1 38 1
0 0 2 8 1
6 7 2 94 1
0 0 2 44 1
0 0 2 9 0
30 78 73 0
73 99 99 1
72 3 2 0

输出
1.00

提示
F1F1F1 值计算公式：
F1=2∗(Precision∗Recall)/(Precision+Recall)F1=2*(Precision *Recall) /(Precision + Recall)F1=2∗(Precision∗Recall)/(Precision+Recall)

▶️


#### 解答


video solution

题解思路
方法思路

问题分析：决策树可能过拟合训练数据，需要通过剪枝提高泛化能力。对于每个节点，考虑将其转换为叶节点后的F1值，选择最优方案。
算法选择：使用深度优先搜索(DFS)后序遍历决策树。对于每个节点，计算：

保留子树时的F1值（递归处理左右子树）
剪枝为叶节点时的F1值

关键操作：比较两种方案的F1值，选择较大的一个，实现贪心剪枝。
复杂度分析：每个节点处理一次，每次处理需要遍历验证数据子集。时间复杂度为O(N*M)，其中N为节点数，M为验证集大小。

解题代码

**Python 代码：**

```python
import sys

def main():
data = sys.stdin.read().split()
it = iter(data)

n = int(next(it)); m = int(next(it)); k = int(next(it))

# 读取节点信息
nodes = {}
for i in range(1, n + 1):
left_id = int(next(it))
right_id = int(next(it))
feature = int(next(it))
threshold = int(next(it))
label = int(next(it))
nodes[i] = {
'left': left_id,
'right': right_id,
'feature': feature,
'threshold': threshold,
'label': label,
'is_leaf': left_id == 0 and right_id == 0
}

# 读取验证数据
validation_data = []
for _ in range(m):
features = [float(next(it)) for _ in range(k)]
true_label = int(next(it))
validation_data.append((features, true_label))

def evaluate_with_label(pred_label, data_subset):
tp = fp = fn = 0
for _, true_label in data_subset:
if pred_label == 1 and true_label == 1:
tp += 1
elif pred_label == 1 and true_label == 0:
fp += 1
elif pred_label == 0 and true_label == 1:
fn += 1
precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
return tp, fp, fn, f1

def prune(node_id, data_subset):
node = nodes[node_id]

# 作为叶子评估（剪枝为叶）
tp_leaf, fp_leaf, fn_leaf, f1_leaf = evaluate_with_label(node['label'], data_subset)
if node['is_leaf'] or not data_subset:
return tp_leaf, fp_leaf, fn_leaf, f1_leaf

# 分割数据
left_data, right_data = [], []
for features, true_label in data_subset:
if features[node['feature'] - 1] <= node['threshold']:
left_data.append((features, true_label))
else:
right_data.append((features, true_label))

left_stat = prune(node['left'], left_data)
right_stat = prune(node['right'], right_data)

left_tp, left_fp, left_fn, _ = left_stat
right_tp, right_fp, right_fn, _ = right_stat

tp_sub = left_tp + right_tp
fp_sub = left_fp + right_fp
fn_sub = left_fn + right_fn
precision_sub = tp_sub / (tp_sub + fp_sub) if tp_sub + fp_sub > 0 else 0
recall_sub = tp_sub / (tp_sub + fn_sub) if tp_sub + fn_sub > 0 else 0
f1_sub = 2 * precision_sub * recall_sub / (precision_sub + recall_sub) if precision_sub + recall_sub > 0 else 0

# 选更优方案
if f1_leaf > f1_sub:
return tp_leaf, fp_leaf, fn_leaf, f1_leaf
else:
return tp_sub, fp_sub, fn_sub, f1_sub

# 从根节点开始剪枝
_, _, _, best_f1 = prune(1, validation_data)
print("{:.6f}".format(best_f1))

if __name__ == "__main__":
main()

```

---
