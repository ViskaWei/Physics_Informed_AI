# Graph 类题目汇总 [0/2 完成]

> 📊 **进度**: 0/2 完成 (0%)  
> 🔄 **最后更新**: 2026-01-04  
> 📁 **分类**: graph (图论、树、祖先节点、子树)

---

## 📋 题目总览

> 🔥 **重刷优先级**: 1 > 2（按难度和重要程度排序）

| 出题日期 | # | P编号 | 题目 | 难度 | 状态 | 完成日期 |
|----------|---|-------|------|------|------|----------|
| 2025-11-19 | 1 | P4476 | 最大值子树（树形DP + 可选剪枝） ⭐ | 困难 | ❌ | - |
| 2025-09-12 | 2 | P3657 | 二叉树中序遍历的第k个祖先节点 | 中等 | ❌ | - |

---

## 🔧 通用模板

### 二叉树层序构建模板
```python
from collections import deque

class Node:
    def __init__(self, v):
        self.v = v
        self.l = None
        self.r = None

def build_tree(tokens):
    """从层序遍历序列构建二叉树，# 表示空节点"""
    if not tokens or tokens[0] == '#':
        return None, {}, {}
    
    root = Node(int(tokens[0]))
    q = deque([root])
    parent = {root: None}
    val2node = {root.v: root}
    idx = 1
    
    while q and idx < len(tokens):
        cur = q.popleft()
        # 左孩子
        if idx < len(tokens) and tokens[idx] != '#':
            left = Node(int(tokens[idx]))
            cur.l = left
            parent[left] = cur
            val2node[left.v] = left
            q.append(left)
        idx += 1
        # 右孩子
        if idx < len(tokens) and tokens[idx] != '#':
            right = Node(int(tokens[idx]))
            cur.r = right
            parent[right] = cur
            val2node[right.v] = right
            q.append(right)
        idx += 1
    
    return root, parent, val2node
```

### 树的遍历模板
```python
def inorder(root):
    """中序遍历（左-根-右）"""
    if not root:
        return []
    return inorder(root.l) + [root.v] + inorder(root.r)

def preorder(root):
    """前序遍历（根-左-右）"""
    if not root:
        return []
    return [root.v] + preorder(root.l) + preorder(root.r)

def postorder(root):
    """后序遍历（左-右-根）"""
    if not root:
        return []
    return postorder(root.l) + postorder(root.r) + [root.v]
```

### 树形 DP 模板
```python
def tree_dp(root):
    """自底向上的树形 DP"""
    if root is None:
        return 0
    
    left_val = tree_dp(root.l)
    right_val = tree_dp(root.r)
    
    # 当前节点的最优值
    cur_val = root.v + max(0, left_val) + max(0, right_val)
    return cur_val
```

---

## 题目1: 最大值子树（P4476）⭐

- **难度**: 困难
- **核心**: 树形 DP + 可选剪枝
- **源**: [core46#第3题-p4476](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- 给定一棵完全二叉树（层序数组表示，null 表示空）
- 可以选择任意节点作为根，并裁剪贡献 ≤ 0 的子树
- 求裁剪后的最大子树和

### 关键规则
1. 子树的值 = 所有保留节点值之和
2. 可以裁剪贡献 ≤ 0 的子树分支
3. 输出格式：层序遍历的数组，null 表示被裁剪，末尾多余 null 删除

### 样例
```
输入:
[-5,-1,3,null,null,4,7]

输出:
[3,4,7]

说明:
- 根节点 -5 的左子树 -1 贡献为负，裁剪
- 以 3 为根的子树 [3,4,7] 和为 14，是最大的
```

### 思路
1. **数组索引**：根=0，左子=2i+1，右子=2i+2
2. **树形 DP**（自底向上）：
   - $dp[i] = val[i] + \max(0, dp[left]) + \max(0, dp[right])$
   - 若子树贡献 ≤ 0，则裁剪
3. 找到 dp 值最大的节点作为新根
4. **BFS 还原**：只保留 dp > 0 的子树

### 复杂度
- 时间: O(n)
- 空间: O(n)

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 题目2: 二叉树中序遍历的第k个祖先节点（P3657）

- **难度**: 中等
- **核心**: 建树 + 祖先集合 + 中序遍历计数
- **源**: [core46#第2题-p3657](../AI_编程题_Python解答_核心46题.md)

### 题目描述
- 给定二叉树（层序遍历，# 表示空）
- 找节点 u 在中序遍历中，位于 u 前面的所有祖先中的第 k 个

### 关键规则
1. 祖先 = 从根到 u 路径上的节点（不含 u）
2. "第 k 个" = 在中序序列中，u 前面的祖先，按出现顺序第 k 个
3. 不存在返回 -1

### 样例
```
输入:
30 15 45 7 20 35 50 # # 18 # # 40
40 3

输出:
-1

说明:
- 中序遍历：7,15,18,20,30,35,40,45,50
- 节点 40 的祖先：30,45,35
- 在 40 前面的祖先：30,35（按中序顺序）
- 第 3 个不存在，返回 -1
```

### 思路
1. **建树**：层序遍历构建，同时记录 parent 和 val2node
2. **收集祖先**：从 u 沿 parent 回溯到根
3. **中序遍历计数**：
   - 遍历到 u 前，统计出现的祖先节点
   - 第 k 个祖先记为 ans_k
   - 到达 u 时返回结果

### 复杂度
- 时间: O(n)
- 空间: O(n)

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 📌 易错点总结

1. **完全二叉树的数组表示**：
   - 根：索引 0
   - 左子：2i + 1
   - 右子：2i + 2
   - 父节点：(i-1) // 2

2. **层序遍历建树**：
   - 空节点用 null/# 表示
   - 队列逐层挂接

3. **树形 DP 的可选子树**：
   - $\max(0, dp[child])$ 表示可以不选

4. **中序遍历 + 祖先**：
   - 祖先不一定都在 u 前面
   - 需要同时满足"是祖先"和"在 u 前面"

5. **输出格式**：
   - 末尾多余的 null 要删除
   - 中间的 null 要保留

---

## 🔗 相关文件

- 源文件：`../AI_编程题_Python解答_核心46题.md`
- 索引：`../ai_core46_index.md`

---

## 📝 代码答案

### 题目1: P4476 最大值子树
```python
import sys
from ast import literal_eval
from collections import deque

def max_pruned_subtree(arr):
    n = len(arr)
    if n == 0:
        return []

    valid = [x is not None for x in arr]
    dp = [0] * n
    best_sum = None
    best_root = -1

    # 自底向上 DP
    for i in range(n - 1, -1, -1):
        if not valid[i]:
            dp[i] = 0
            continue

        left = 2 * i + 1
        right = 2 * i + 2
        left_dp = dp[left] if left < n and valid[left] else 0
        right_dp = dp[right] if right < n and valid[right] else 0

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

    # BFS 构造结果
    res = []
    q = deque([(best_root, 0)])

    while q:
        oi, ni = q.popleft()
        while len(res) <= ni:
            res.append(None)
        res[ni] = arr[oi]

        left = 2 * oi + 1
        right = 2 * oi + 2
        if left < n and valid[left] and dp[left] > 0:
            q.append((left, 2 * ni + 1))
        if right < n and valid[right] and dp[right] > 0:
            q.append((right, 2 * ni + 2))

    # 去掉末尾 None
    while res and res[-1] is None:
        res.pop()

    return res

def main():
    s = sys.stdin.readline().strip()
    if not s:
        return

    # 解析输入：将 null 转换为 None
    s = s.replace('null', 'None')
    arr = literal_eval(s)
    
    result = max_pruned_subtree(arr)
    
    # 输出格式化
    out = []
    for v in result:
        out.append('null' if v is None else str(v))
    print('[' + ','.join(out) + ']')

if __name__ == "__main__":
    main()
```

### 题目2: P3657 二叉树中序遍历的第k个祖先节点
```python
import sys
from collections import deque

class Node:
    def __init__(self, v):
        self.v = v
        self.l = None
        self.r = None

def build_tree(tokens):
    if not tokens or tokens[0] == '#':
        return None, {}, {}
    
    root = Node(int(tokens[0]))
    q = deque([root])
    parent = {root: None}
    val2node = {root.v: root}
    idx = 1
    
    while q and idx < len(tokens):
        cur = q.popleft()
        # 左孩子
        if idx < len(tokens):
            t = tokens[idx]
            idx += 1
            if t != '#':
                left = Node(int(t))
                cur.l = left
                parent[left] = cur
                val2node[left.v] = left
                q.append(left)
        # 右孩子
        if idx < len(tokens):
            t = tokens[idx]
            idx += 1
            if t != '#':
                right = Node(int(t))
                cur.r = right
                parent[right] = cur
                val2node[right.v] = right
                q.append(right)
    
    return root, parent, val2node

def kth_ancestor_in_inorder_before_u(root, parent, val2node, u, k):
    if u not in val2node:
        return -1
    
    u_node = val2node[u]
    
    # 收集 u 的全部祖先
    anc = set()
    p = parent.get(u_node)
    while p is not None:
        anc.add(p)
        p = parent.get(p)
    
    # 中序遍历计数
    stack = []
    cur = root
    cnt = 0
    ans_k = None
    
    while stack or cur:
        while cur:
            stack.append(cur)
            cur = cur.l
        cur = stack.pop()
        
        if cur is u_node:
            return ans_k if cnt >= k else -1
        
        if cur in anc:
            cnt += 1
            if cnt == k:
                ans_k = cur.v
        
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
