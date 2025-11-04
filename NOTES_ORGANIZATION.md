# 笔记分类组织结构

生成时间：2024-11-04

## 📂 当前分类结构

### 1. Deep Learning
主分类，包含深度学习相关的所有笔记

#### 子分类：RNN
基础 RNN 笔记（共6篇）

| # | 文件名 | 标题 | URL |
|---|--------|------|-----|
| 1 | `01-序列建模基础.md` | 1. 序列建模基础 | `/notes/sequence-modeling-basics/` |
| 2 | `02-文本数据处理.md` | 2. 文本数据处理 | `/notes/text-data-processing/` |
| 3 | `03-语言模型和数据集.md` | 3. 语言模型和数据集 | `/notes/language-models-and-datasets/` |
| 4 | `04-循环神经网络从0开始实现 copy.md` | 4. 循环神经网络从0开始实现 | `/notes/rnn-from-scratch/` |
| 5 | `05-循环神经网络简洁实现 copy.md` | 5. 循环神经网络简洁实现 | `/notes/rnn-concise-implementation/` |
| 6 | `06 - 通过时间反向传播.md` | 6 通过时间反向传播 | `/notes/backpropagation-through-time/` |

#### 子分类：Advanced RNN
进阶 RNN 笔记（共1篇）

| # | 文件名 | 标题 | URL |
|---|--------|------|-----|
| 1 | `01-门控循环单元（GRU）.md` | 1. 门控循环单元（GRU） | `/notes/gated-recurrent-unit/` |

---

### 2. Data Structures
数据结构与算法笔记（共1篇）

| # | 文件名 | 标题 | URL |
|---|--------|------|-----|
| 1 | `1. 哈希表.md` | 深入理解哈希表 (Hash Table) | `/notes/hash-table-explained/` |

---

### 3. 天线
天线设计笔记（共1篇）

| # | 文件名 | 标题 | URL |
|---|--------|------|-----|
| 1 | `天线设计基础.md` | 天线设计基础 | `/notes/天线设计基础/` |

---

## 🌐 网站显示效果

访问网站时的预期层级结构：

```
Notes (笔记)
│
├── Deep Learning (主分类)
│   ├── RNN (子分类)
│   │   ├── 1. 序列建模基础
│   │   ├── 2. 文本数据处理
│   │   ├── 3. 语言模型和数据集
│   │   ├── 4. 循环神经网络从0开始实现
│   │   ├── 5. 循环神经网络简洁实现
│   │   └── 6. 通过时间反向传播
│   │
│   └── Advanced RNN (子分类)
│       └── 1. 门控循环单元（GRU）
│
├── Data Structures
│   └── 深入理解哈希表
│
└── 天线
    └── 天线设计基础
```

---

## ⚠️ 注意事项

### 文件名问题
以下文件仍包含 " copy" 后缀：
- `04-循环神经网络从0开始实现 copy.md`
- `05-循环神经网络简洁实现 copy.md`

**说明**：这不影响网站显示，因为 `permalink` 已经正确设置为英文 URL。

### Permalink 问题
天线笔记的 permalink 仍使用中文：
- `permalink: /notes/天线设计基础/`

**建议**：改为英文 URL，例如 `/notes/antenna-design-basics/`

---

## 📋 Front Matter 标准格式

每个笔记的 Front matter 应包含：

```yaml
---
layout: note_with_toc
title: 笔记标题
description: 笔记描述（英文）
category: 主分类
subcategory: 子分类（可选）
tags: [标签1, 标签2, ...]
permalink: /notes/english-url/
redirect_from:
  - /notes/旧链接/
---
```

---

## ✅ 已完成的优化

1. ✅ 统一所有 RNN 笔记的 permalink 为英文 URL
2. ✅ 添加 `redirect_from` 保留旧链接兼容性
3. ✅ 建立清晰的分类层级（Deep Learning → RNN / Advanced RNN）
4. ✅ 修复 GRU 笔记的格式问题
5. ✅ 所有修改已推送到 GitHub

---

## 🔧 待优化项

1. ⚠️ 文件名移除 " copy" 后缀（手动操作）
2. ⚠️ 天线笔记的 permalink 改为英文
3. 💡 未来可扩展 Advanced RNN 子分类（LSTM、Seq2Seq、Attention 等）
