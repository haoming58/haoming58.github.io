---
layout: note_with_toc
title: 7. 序列到序列学习（seq2seq）
description: Gated Recurrent Unit - advanced RNN architecture with reset and update gates
category: Deep Learning
subcategory: Advanced RNN
tags: [RNN, GRU, Gated Networks, Deep Learning, Neural Networks]
permalink: /notes/gated-recurrent-unit/
redirect_from:
  - /notes/长短期记忆网络（LSTM）/
  - /notes/sequence-modeling-basics/
---

# 7. 序列到序列学习

## 7.1 机制

基于机器翻译中，输入和输出系列的长度是可变的。在上一节提出了编码的结构，然后应用到序列到序列的任务

![alt text](../../../assets/img/notes/new_rnn/使用循环神经网络编码器和循环神经网络解码器的序列到序列学习¶.png)

## 7.2 编码器

所谓的编码器件，实际上就是RNN系列的输入和输出。

$N$: 批量大小 (Batch Size)
$T$: 时间步数 / 序列长度 (Number of Time Steps)
$V$: 词表大小 (Vocab Size)
$D$: 嵌入特征维度 (Embed Size)
$H$: 隐藏状态维度 (Hidden Size)

1. 输入序列 $X$ ： $(N, T)$

加入嵌入层： $(T, N, D)$

2. 循环层：

应该是从 X 取出的切片：$(N, D)$ ， $h_{t-1}$ (上一时间步隐状态):形状: $(N, H)$

$W_{xh}$ (输入到隐藏层): 形状 $(D, H)
$$W_{hh}$ (隐藏层到隐藏层): 形状 $(H, H)
$$b_h$ (偏置): 形状 $(H)$

$h_t$ 的形状为 $(N, H)$

3. 输出与上下文变量：

由于每个 $h_t$ 是 $(N, H)$，共有 $T$ 个，

Output 形状: $(T, N, H)$

通常情况下，上下文变量取的是最后一个变量，还要加上使用多层结构。

State 形状: $(L, N, H)$


## 7.3 解码器

这里就有一个很大的变化，或者说值得注意的就是将上下文进行拼接

$N$: 批量大小 (Batch Size)
$T$: 时间步数 / 序列长度 (Number of Time Steps)
$V$: 词表大小 (Vocab Size)
$D$: 嵌入特征维度 (Embed Size)
$H$: 隐藏状态维度 (Hidden Size)$L$

1. 当前时刻输入 ($y_{t'-1}$) ： $(N, D)$
2. 上下文变量 ($\mathbf{c}$)：  $(N, H)$
3. 进行拼接操作：$$(N, D) \oplus (N, H) \rightarrow (N, D + H)$$

这里稍微解释一下：拼接”通过矩阵运算

将当前维度的词嵌入向量与上下文向量结合，更好地预测。

等价于 $$\text{Output} = [W_x, W_c] \times \begin{bmatrix} x \\ c \end{bmatrix}$$

但是为了计算效率，做 一次 大矩阵乘法（拼接后的宽矩阵），比做 两次 小矩阵乘法再做一次加法，速度要快得多。

这里叫做特征融合

4. 输出层计算（全连接层）：

$$(N, H) \times (H, V') \rightarrow (N, V')$$

## 7.4 损失函数

### 7.4.1Mask演示：

假设：序列最大长度 maxlen = 3，即位置索引为 [0, 1, 2]。有 2 个样本，有效长度分别为 1 和 2。左边（位置索引）：它是一个 (1, 3) 的行向量，比较时会向下复制扩充：$$\begin{bmatrix} 0 & 1 & 2 \end{bmatrix} \xrightarrow{\text{广播}} \begin{bmatrix} 0 & 1 & 2 \\ 0 & 1 & 2 \end{bmatrix}$$右边（有效长度）：它是一个 (2, 1) 的列向量，比较时会向右复制扩充：$$\begin{bmatrix} 1 \\ 2 \end{bmatrix} \xrightarrow{\text{广播}} \begin{bmatrix} 1 & 1 & 1 \\ 2 & 2 & 2 \end{bmatrix}$$执行 < 运算：将上面两个矩阵对应位置进行对比：$$\begin{bmatrix} 0 & 1 & 2 \\ 0 & 1 & 2 \end{bmatrix} < \begin{bmatrix} 1 & 1 & 1 \\ 2 & 2 & 2 \end{bmatrix}$$第一行对比：$0 < 1$ ($\text{True}$)$1 < 1$ ($\text{False}$) $\rightarrow$ 超出长度 1，无效$2 < 1$ ($\text{False}$) $\rightarrow$ 超出长度 1，无效第二行对比：$0 < 2$ ($\text{True}$)$1 < 2$ ($\text{True}$)$2 < 2$ ($\text{False}$) $\rightarrow$ 超出长度 2，无效最终结果 (Mask)：$$\begin{bmatrix} \text{True} & \text{False} & \text{False} \\ \text{True} & \text{True} & \text{False} \end{bmatrix}$$

### 7.4.2 损失函数的Mask解释：

一个 Batch，包含 2 个样本，最大长度为 3

正确答案 (Labels):

样本 1（有效长度 3）：[A, B, C]

样本 2（有效长度 1）：[D, \<Pad>, \<Pad>]

的预测 (Preds)：

样本 1：预测 [A, B, F] （前两个对，第三个错）

样本 2：预测 [D, Z, X] （第一个对，后面两个瞎猜的，因为是凑数位）

有效长度 (valid_len)：[3, 1]


- PyTorch 的 CrossEntropyLoss(reduction='none') 会给每个位置打分。

样本 1：样本 1 原始 Loss 向量：[0.1, 0.1, 2.5]

样本 2：样本 2 原始 Loss 向量：[0.1, 5.0, 6.2] 

在样本2中，实际上是存在pad的，但是你如果不忽略，直接求平均的话，会有问题，

于是采用掩码的方法去处理这个问题：

根据 valid_len = [3, 1] 生成“免死金牌”：

样本 1 (长度3)：[1, 1, 1] （全都算分）

样本 2 (长度1)：[1, 0, 0] （只有第1个算分，后面免单），最终就做了正确的损失函数计算。




## 7.5 训练

1. 权重初始化
2. 数据加载
3. 前向传播
4. 损失计算
5. 反向传播
6. 梯度裁剪
7. 参数更新
8. 进行可视化
















