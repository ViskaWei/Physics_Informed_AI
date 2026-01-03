#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卷积操作
Topic: conv
Date: 2025-11-06

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
    # TODO: implement convolution
    return ""


def main() -> None:
    data = sys.stdin.read().strip().split()
    out = solve(data)
    if out is not None:
        print(out)


if __name__ == "__main__":
    main()

