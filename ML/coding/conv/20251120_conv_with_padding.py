#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带Padding的卷积计算
Topic: conv
Date: 2025-11-20

I/O:
  stdin:
    ...
  stdout:
    ...
"""
import sys
from typing import List


def solve(tokens: List[str]) -> str:
    it = iter(tokens)
    # TODO: parse input
    # TODO: implement convolution with zero-padding
    return ""


def main() -> None:
    data = sys.stdin.read().strip().split()
    out = solve(data)
    if out is not None:
        print(out)


if __name__ == "__main__":
    main()

