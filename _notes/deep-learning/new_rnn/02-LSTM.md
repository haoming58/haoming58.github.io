---
layout: note_with_toc
title: 2. 长短期记忆网络（LSTM）
description: Long Short-Term Memory (LSTM) - advanced RNN with input, forget, and output gates for capturing long-range dependencies
category: Deep Learning
subcategory: Advanced RNN
tags: [RNN, LSTM, Gated Networks, Deep Learning, Neural Networks]
permalink: /notes/long-short-term-memory/
redirect_from:
  - /notes/长短期记忆网络（LSTM）/
---

# 2. 长短期记忆网络（LSTM）

## 2.1 问题

隐变量模型始终存在长期信息保存和短期输入缺失问题。比GRU的设计早了快20年

它的总体设计有三个门，灵感来源于计算机的逻辑门。


## 2.2 设计

![长短期记忆模型中的输入门、遗忘门和输出门]({{ '/assets/img/notes/new_rnn/长短期记忆模型中的输入门、遗忘门和输出门.png' | relative_url }})

### 2.2.1 门


输入门（Input Gate）：决定何时将新数据读入记忆元。

遗忘门（Forget Gate）：决定何时重置/忽略记忆元中的旧内容。

输出门（Output Gate）：决定何时将记忆元的内容输出到隐状态。

这三个门（$I_t$, $F_t$, $O_t$）的计算方式类似，都使用 $Sigmoid$ 激活函数 ($\sigma$) 将值约束在 $(0, 1)$ 范围内，实现门控功能。

$$F_t = \sigma(X_t W_{xf} + H_{t-1} W_{hf} + b_f)$$

$$I_t = \sigma(X_t W_{xi} + H_{t-1} W_{hi} + b_i)$$

$$O_t = \sigma(X_t W_{xo} + H_{t-1} W_{ho} + b_o)$$


### 2.2.3 候选记忆门

候选记忆元 $\tilde{C}_t$ 代表了当前时间步可能添加到记忆元的新信息。它使用 $\tanh$ 激活函数，将值约束在 $(-1, 1)$ 范围内：$$\tilde{C}_t = \tanh(X_t W_{xc} + H_{t-1} W_{hc} + b_c)$$


### 2.2.4 记忆元


记忆元 $C_t$ 的更新是 LSTM 最核心的部分，它结合了旧记忆和新信息，并由遗忘门和输入门精确控制：$$C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t$$


遗忘门 ($F_t$) $\odot$ 旧记忆 ($C_{t-1}$): 决定保留多少过去的记忆元 $C_{t-1}$ 的内容。

如果 $F_t$ 接近 $1$，保留旧记忆；接近 $0$，遗忘旧记忆。

输入门 ($I_t$) $\odot$ 候选记忆元 ($\tilde{C}_t$): 决定采用多少来自新数据 $\tilde{C}_t$。

如果 $I_t$ 接近 $1$，完全采纳新信息；接近 $0$，忽略新信息。

这种机制使得信息可以长久地保存在记忆元中（如果 $F_t \approx 1$ 且 $I_t \approx 0$），有效缓解了传统 RNN 中的梯度消失问题，从而捕获长距离依赖关系。


### 2.2.4 记忆元

最终的隐状态 $H_t$ 是模型的输出之一:

$$H_t = O_t \odot \tanh(C_t)$$

输出门 ($O_t$): 决定将记忆元 $C_t$ 中的多少信息暴露给隐状态 $H_t$

如果 $O_t$ 接近 $1$，有效地将所有记忆信息传递给预测部分；接近 $0$，则隐状态几乎不包含记忆信息，只保留记忆元内部的信息。


![在长短期记忆模型中计算隐状态]({{ '/assets/img/notes/new_rnn/在长短期记忆模型中计算隐状态.png' | relative_url }})


### 2.2.5 总结

LSTM 的核心在于其**双轨机制**：一条用于存储**长期信息（记忆元 $C_t$）**，另一条用于处理**短期信息（隐状态 $H_t$）**。在每一个时间步，网络的目标是精确地控制这两种信息。

整个过程始于**接收输入**：当前时刻的输入 $X_t$ 和上一时刻的短期信息 $H_{t-1}$ 被送入网络，用于计算所有的“门”。

#### 1. 遗忘旧知识

流程首先遇到**遗忘门 ($F_t$)**。这个门充当一个过滤器，它基于当前的输入 $X_t$ 和 $H_{t-1}$，决定上一时刻的**长期记忆 $C_{t-1}$** 中哪些内容是**不再重要的**。遗忘门会生成一个 $0$ 到 $1$ 之间的数值向量，然后通过按元素相乘的方式，**选择性地保留** $C_{t-1}$ 中的信息。可以想象，这是对旧知识进行一次**清理**。

#### 2. 采纳新知识

清理完毕后，网络开始准备接收新信息。这需要两个步骤：
1.  首先，**候选记忆元 ($\tilde{C}_t$)** 基于 $X_t$ 和 $H_{t-1}$ 计算出所有**潜在的新知识**。
2.  紧接着，**输入门 ($I_t$)** 充当另一个过滤器，它决定 $\tilde{C}_t$ 中**哪些部分是值得采纳的**。只有当 $I_t$ 的值接近 $1$ 时，对应的 $\tilde{C}_t$ 元素才会被考虑写入长期记忆。

#### 3. 长期记忆的更新

至此，我们有了**经过过滤的旧记忆**（由 $F_t$ 控制的 $C_{t-1}$）和**经过过滤的新信息**（由 $I_t$ 控制的 $\tilde{C}_t$）。最新的**记忆元 $C_t$** 就是将这两部分**直接相加**得到。正是这种直接的加法更新机制，保证了信息和梯度能够相对完整地在时间步上流动，从而成功维护了长期依赖。

$$C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t$$

#### 4. 产生输出

最后，我们需要将更新后的**长期记忆 $C_t$** 转化为**短期输出 $H_t$**。这里由**输出门 ($O_t$)** 发挥作用。输出门决定了 $C_t$ 中的哪些信息应该被**公开**。网络首先对 $C_t$ 进行 $\text{tanh}$ 激活来规范化数值，然后用 $O_t$ 对其进行**按元素筛选**。最终的结果就是**隐状态 $H_t$**，它既是当前时间步的输出，也是下一个时间步的输入，继续参与下一轮的门控计算。

$$H_t = O_t \odot \tanh(C_t)$$

这个过程在每个时间步周而复始，使得 LSTM 能够高效地决定何时遗忘、何时输入、以及何时输出信息。




## 2.3 代码实践

基本库加载

```python

import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

```

同样参数的基本初始化

```python
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    # 还是一样的输入和输出维度，one hot 向量
    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    # 类似GRU的，每个门都有三组参数
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

```


```python

def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

```
$H$ (隐状态): $(N, H)$

$C$ (记忆元): $(N, H)$

下面一步，进行维度计算:

批量大小 (Batch Size): $N$
输入特征数 (Input Features): $D$, 这里应该是词表的大小
隐藏单元数 (Hidden Units): $H$

X (输入): 在循环内，X 是当前时间步的输入，维度为 $(N, D)$

| 参数         | 形状 (Shape) | 作用                |
| :--------- | :--------- | :---------------- |
| $W_{x*}$   | $(D, H)$   | 输入 $X$ → 门/候选记忆元  |
| $W_{h*}$   | $(H, H)$   | 隐状态 $H$ → 门/候选记忆元 |
| $W_{hq}$   | $(H, Q)$   | 隐状态 $H$ → 输出层     |
| $b_*, b_c$ | $(H)$      | 门或候选记忆元的偏置        |
| $b_q$      | $(Q)$      | 输出层偏置             |



```python

def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
```  


```python
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```  


![LSTM训练结果]({{ '/assets/img/notes/new_rnn/LSTM.png' | relative_url }}) 


```python
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```  
![LSTM简洁实现]({{ '/assets/img/notes/new_rnn/LSTM1.png' | relative_url }})


## 2.4 问题

### 2.4.1 调整和分析超参数对运行时间、困惑度和输出顺序的影响。

![超参数影响]({{ '/assets/img/notes/new_rnn/超参数.png' | relative_url }})

隐藏单元数	num_hiddens	模型容量	决定了模型能“记住”多少信息。

太小会导致欠拟合（学不会），太大会导致过拟合（死记硬背）且计算极慢。

时间步数	num_steps	记忆长度	决定了模型往回看多远。

越大能捕捉越长的依赖关系，但计算量和内存消耗线性增加，且可能导致梯度问题。

学习率	lr	收敛速度	决定了梯度下降的步长。太大容易震荡甚至发散（NaN），太小收敛极慢。

批量大小	batch_size	训练稳定性	决定了梯度估计的准确性。

越大训练越快（并行度高）但可能陷入局部最优；越小噪声越大但有时能带来更好的泛化。


### 2.4.2 如何更改模型以生成适当的单词，而不是字符序列？

我的理解是有个函数可以：

```python
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)

    # 将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]

    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab
```

把这个需要调整为  tokens = tokenize(lines, 'word'), 这个下面就跟着改

```python
class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
```

### 2.4.4 在给定隐藏层维度的情况下，比较门控循环单元、长短期记忆网络和常规循环神经网络的计算成本。要特别注意训练和推断成本。

![RNN vs LSTM对比]({{ '/assets/img/notes/new_rnn/rnn vs lstm.png' | relative_url }})


### 2.4.5 既然候选记忆元通过使用函数来确保值范围在之间，那么为什么隐状态需要再次使用函数来确保输出值范围在之间呢？ ？。

答案的核心在于：虽然**候选记忆元 $\tilde{C}_t$** 的值确实被限制在 $[-1, 1]$ 之间，但真正的**记忆元 $C_t$** 的值**并没有**被限制在这个范围内，它可能会累积得非常大。

让我们逐步剖析这个原因：


请重新看记忆元的更新公式：

$$C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t$$

* **$\tilde{C}_t$ 是受限的**：确实，由于 $\text{tanh}$ 的存在，当前这一步想写入的新信息 $I_t \odot \tilde{C}_t$ 最大只能是 $1$（或最小是 $-1$）。
* **但 $C_t$ 是累积的**：注意公式中的第一项 $F_t \odot C_{t-1}$。这是一种**线性自连接**（Linear Self-Connection）。
    * 如果遗忘门 $F_t$ 一直保持打开状态（接近 $1$），那么过去的信息 $C_{t-1}$ 就会几乎无损地保留下来。
    * 如果网络在多个时间步连续收到正向输入（$\tilde{C}_t > 0$），记忆元 $C_t$ 的值就会不断叠加。

> **举个例子：**
> 假设 $F_t$ 始终为 $1.0$，$I_t$ 始终为 $1.0$，而输入每次都让 $\tilde{C}_t$ 为 $0.5$。
> * $t=1: C_1 = 1.0 \times 0 + 1.0 \times 0.5 = 0.5$
> * $t=2: C_2 = 1.0 \times 0.5 + 1.0 \times 0.5 = 1.0$
> * $t=3: C_3 = 1.0 \times 1.0 + 1.0 \times 0.5 = 1.5$
> * ...
> * $t=100: C_{100} = 50.0$
>
> 看到了吗？**$C_t$ 的值可以远远超出 $[-1, 1]$ 的范围。** 这正是 LSTM 能够记住长距离信息的关键——它像一个计数器一样，可以不受干扰地不断累积信息。


既然 $C_t$ 可以变得很大（例如 $50.0$ 或 $-100.0$），如果我们直接把它传给下一层或下一个时间步：

$$H_t = O_t \odot C_t \quad (\text{假设没有第二个 tanh})$$

那么 $H_t$ 的值也会变得巨大。这会带来两个严重问题：
1.  **数值不稳定**：巨大的值传入后续的网络层，容易导致梯度爆炸，使得训练难以收敛。
2.  **不匹配**：神经网络的其他部分（如其他门的输入）通常期望接收范围在 $[-1, 1]$ 或 $[0, 1]$ 左右的标准化输入。

因此，我们需要第二个 $\text{tanh}$ 在输出之前对 $C_t$ 进行**“挤压”（Squashing）**：

$$H_t = O_t \odot \tanh(C_t)$$

无论 $C_t$ 累积到多大（比如 $50.0$），$\tanh(50.0)$ 都会把它稳稳地限制回 $\approx 1.0$。
这确保了**隐状态 $H_t$ 始终保持在 $[-1, 1]$ 的稳定范围内**，可以安全地传递给网络的其他部分。


* **第一个 $\tanh$ (在 $\tilde{C}_t$ 中)**：是为了规范化**当前步的新输入**，确保每次写入的信息量是受控的。
* **记忆元 $C_t$**：是一个**无界的累积器**，负责长距离携带信息，它的值可以很大。
* **第二个 $\tanh$ (在 $H_t$ 中)**：是为了规范化**最终输出**，防止累积过大的记忆值破坏后续计算的稳定性。


### 2.4.5 实现一个能够基于时间序列进行预测而不是基于字符序列进行预测的长短期记忆网络模型。

```python

```