#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template: {title}
Topic: {topic}
Date: {YYYY-MM-DD}

I/O:
  stdin:
    ...
  stdout:
    ...
"""
import sys
from typing import List, Tuple


def solve(data: List[str]) -> str:
    it = iter(data)
    # TODO: parse input
    # TODO: implement
    return ""


def main() -> None:
    data = sys.stdin.read().strip().split()
    out = solve(data)
    if out is not None:
        print(out)


if __name__ == "__main__":
    main()

