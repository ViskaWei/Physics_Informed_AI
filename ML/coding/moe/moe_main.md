# MoE 类题目汇总 [1/1 完成]

> 📊 **进度**: 1/1 完成 (100%) ✅  
> 🔄 **最后更新**: 2026-01-06  
> 📁 **分类**: moe (Mixture of Experts、路由优化)

---

## 📋 题目总览

> 🔥 **重刷优先级**: -

| 出题日期 | # | P编号 | 题目 | 难度 | 状态 | 完成日期 |
|----------|---|-------|------|------|------|----------|
| 2025-09-03 | 1 | P3553 | 大模型训练MOE场景路由优化算法 | 中等 | ✅ | 2026-01-06 |

---

## 🔧 通用模板

```python
# MoE 基础
import numpy as np

def top_k_gating(x, gate_weights, k=2):
    """Top-K 门控选择专家"""
    # x: (batch, dim), gate_weights: (dim, num_experts)
    logits = x @ gate_weights  # (batch, num_experts)
    
    # 选择 top-k 专家
    top_k_indices = np.argsort(logits, axis=-1)[:, -k:]
    top_k_logits = np.take_along_axis(logits, top_k_indices, axis=-1)
    
    # softmax 归一化
    top_k_gates = np.exp(top_k_logits) / np.exp(top_k_logits).sum(axis=-1, keepdims=True)
    
    return top_k_indices, top_k_gates

def load_balance_loss(gate_probs, num_experts):
    """负载均衡损失"""
    # 每个专家被选择的平均概率
    expert_load = gate_probs.mean(axis=0)
    # 理想均匀分布
    uniform = 1.0 / num_experts
    return np.sum((expert_load - uniform) ** 2)
```

---

## 题目1: 大模型训练MOE场景路由优化算法（P3553）

- **难度**: 中等
- **源**: [core46#第2题-p3553](../AI_编程题_Python解答_核心46题.md#第2题-p3553)

### 题目描述

MOE 模型训练时，token 根据概率发送到 top-k 个不同的专家进行计算。这些专家分布在多个 NPU 卡上。Device-Limited routing 算法将 token 的路由目标限制在 P 个 NPU 上，可以有效降低通信成本。具体的：

1. 把 n 个专家平均分配在 m 个 NPU 上，每个 NPU 上的专家为一个组；设 n 个专家的编号为 $N=[0,1,2,…,n-1]$，同一个专家组上的专家编号是连续的；
2. 每个专家对应一个概率，表示被路由到的可能性；用每个组中的最大概率作为本组代表，从所有组中选择概率最大的 p 个组，其所在的 NPU 即为路由目标限制 NPU；
3. 再从上述 p 个 NPU 对应的所有专家概率中选择 k 个最大的概率对应的专家编号作为最终路由目标。

**输入描述**：
- 第一行有 4 个处于区间 [1,10000] 之内的整数：n（专家个数）、m（NPU个数）、p（路由目标限制NPU个数）、k（目标路由专家个数）
- 第二行有 n 个处于区间 (0,1) 之内的浮点数，表示每个专家对应的概率值

**输出描述**：
- 如果 n 不能被 m 整除或者获取不到 k 个专家编号，输出 `error`
- 否则，按照从小到大的顺序，输出 k 个专家编号（空格分隔，行尾无空格）

**样例**：
```
输入：8 4 4 2
     0.5 0.01 0.09 0.023 0.027 0.05 0.1 0.2
输出：0 7
说明：将专家分成 4 组：(0.5, 0.01), (0.09, 0.023), (0.027, 0.05), (0.1, 0.2)
     限定 4 个 NPU，选择概率最大的 2 个专家：0.5→编号0, 0.2→编号7
```

### 思路

**三步走算法**：
1. **按组取代表**：组大小 $g=n/m$，对每组找组内最大概率作为该组代表值
2. **选路由目标 NPU（选 p 个组）**：将所有组按代表概率降序排序，取前 p 个组。若 $p>m$，输出 error
3. **在选定的 p 个组里选 k 个专家**：将这 p 个组中的所有专家收集起来，按概率降序挑选前 k 个。若可选专家数 $p \cdot g < k$，输出 error

**排序键**：
- 选组时：按 (组代表概率 desc, 组索引 asc)
- 选专家时：按 (概率 desc, 专家编号 asc)
- 输出：最终 k 个编号再升序打印

### 复杂度

- **时间复杂度**：$O(n + m \log m + pg \log(pg))$
  - 计算每组最大值：$O(n)$
  - 选出前 p 个组：$O(m \log m)$
  - 在 p 个组里选出前 k 个专家：$O(pg \log(pg))$
- **空间复杂度**：$O(n)$

### 我的代码
```python
import sys
import numpy as np

def read():
    d = sys.stdin.buffer.read().split()
    if len(d) < 4:
        return
    try:
        n, m, p, k = map(int, d[:4])
        a = np.array(d[4:4 + n], float)
    except:
        return
    return (n, m, p, k, a) if a.size == n else None

def main():
    r = read()
    if not r: print("error"); return
    n, m, p, k, a = r
    if n % m or p > m: print("error"); return
    g = n // m; mx = a.reshape((m, g)).max(1);
    if g * p < k:  print("error"); return
    npu_idx = np.argsort(-mx)[:p]
    zdx = (npu_idx[:, None] * g + np.arange(g)).ravel()
    zdxk = np.sort(zdx[np.argsort(-a[zdx])[:k]])
    print(*zdxk)

if __name__ == "__main__":
    main()
```

---

## 📌 易错点总结

1. **error 条件检查**：`n % m != 0` 或 `p > m` 或 `p * g < k` 都要输出 error
2. **排序稳定性**：概率相同时按专家编号小的优先（升序打破平局）
3. **最终输出要升序**：选出 k 个专家后，需要再按编号升序排列输出
4. **组索引计算**：第 i 组覆盖的专家编号区间为 $[i \cdot g, (i+1) \cdot g - 1]$

---

## 🔗 相关文件

- 源文件：`../AI_编程题_Python解答_核心46题.md`
- 索引：`../ai_core46_index.md`
