#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最大能量路径（DP/滚动数组）
输入：
  H W K K
  H 行图像 I
  K 行策略矩阵 P
输出：最大能量值（保留 1 位小数）
"""
import sys


def main():
    data = sys.stdin.read().strip().split()
    it = iter(data)
    H = int(next(it)); W = int(next(it)); K1 = int(next(it)); K2 = int(next(it))
    K = K1  # 通常两个 K 相等

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

    # 动态规划：只能向右、右上、右下
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

