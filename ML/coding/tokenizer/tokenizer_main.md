# Tokenizer 类题目汇总 [0/1 完成]

> 📊 **进度**: 0/1 完成 (0%)  
> 🔄 **最后更新**: 2026-01-02  
> 📁 **分类**: tokenizer (分词、BPE、大模型分词)

---

## 📋 题目总览

> 🔥 **重刷优先级**: -

| 出题日期 | # | P编号 | 题目 | 难度 | 状态 | 完成日期 |
|----------|---|-------|------|------|------|----------|
| 2025-09-17 | 1 | P3713 | 大模型分词 | 中等 | ❌ | - |

---

## 🔧 通用模板

```python
# BPE 分词基础
def get_stats(vocab):
    """统计相邻 token 对出现频率"""
    pairs = {}
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs

def merge_vocab(pair, vocab):
    """合并最高频 token 对"""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab
```

---

## 题目1: 大模型分词（P3713）

- **难度**: 中等
- **源**: [core46#第3题-p3713](../AI_编程题_Python解答_核心46题.md#第3题-p3713)

### 题目描述
TODO

### 思路
TODO

### 复杂度
TODO

### 我的代码
```python
# TODO: 填写你的代码
```

---

## 📌 易错点总结

1. TODO

---

## 🔗 相关文件

- 源文件：`../AI_编程题_Python解答_核心46题.md`
- 索引：`../ai_core46_index.md`
