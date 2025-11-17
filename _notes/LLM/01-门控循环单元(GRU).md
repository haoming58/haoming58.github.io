---
layout: note_with_toc
title: 3. 
description: Gated Recurrent Unit - advanced RNN architecture with reset and update gates
category: Deep Learning
subcategory: Advanced RNN
tags: [RNN, GRU, Gated Networks, Deep Learning, Neural Networks]
permalink: /notes/gated-recurrent-unit/
redirect_from:
  - /notes/门控循环单元（）/
  - /notes/sequence-modeling-basics/
---

# 2. 文本数据处理

## 2.1 词嵌入

LLM 无法处理原始文本，需要实现数学运算。为了，实现数据转化到向量，称为词嵌入，不同的数据格式需要转换到不同的向量格式或者模型。

其中，较早且受欢迎的例子是 Word2Vec 方法。但是在实际的应用中，使用的是更高级的维度，维度越高，所耗费的。

1. 先训练一个嵌入模型

2. 再把嵌入提供给别的模型使用

3. 嵌入不会随任务变化


## 2.2 分词

由于模型不能直接理解文本，需要数字。需要下面两步：
1. 文本写成token
2. token 转化为向量

这跟深度学习，关于如何创建词表的章节十分相似

那什么是token ？

  - 一个完整的英文单词
  例：“dog”“happy”

  - 一个词的一部分（子词 subword）
  例：“work”“##ing”（在英文 BPE 中很常见）

  - 标点符号
  例：“,” “.” “!”

  - 特殊字符
  如 <eos>（句子结束）、<pad> 等

文本是伊迪丝·华顿（Edith Wharton）的短篇小说, 去预训练
《判决》

```python



```




