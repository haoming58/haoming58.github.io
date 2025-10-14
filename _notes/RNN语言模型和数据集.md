---
layout: note_with_toc
title: 3. 语言模型和数据集
description: Language models and dataset processing for RNN
category: Machine Learning
tags: [RNN, Language Model, N-gram, Dataset, Deep Learning]
permalink: /notes/RNN语言模型和数据集/
---

# 3. 语言模型和数据集

## 3.1 核心思想

语言模型将处理好的词元（可以是单个字母或单词）看作一个长度为 $T$ 的序列，每个位置的词都是一个观测值：

$$
x_1, x_2, \dots, x_T
$$

语言模型的任务是估计该序列的联合概率分布：

$$
P(x_1, x_2, \dots, x_T)
$$

根据链式法则，这个联合概率可以分解为一系列条件概率的乘积：

$$
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \dots, x_{t-1})
$$

这样分解是因为联合概率的维度过高，无法一次性预测整篇文本。通过一步步预测下一个词，模型将复杂的大问题拆解成多个小问题。模型通过学习条件概率分布，就能在给定上下文的情况下生成文本。

**核心思想**：通过统计语言规律，预测序列中每个词出现的概率。

## 3.2 参数与概率估计

最简单的频率估计方法：

$$
P(\text{deep}) = \frac{\text{单词 "deep" 出现的次数}}{\text{语料库中所有单词的总次数}}
$$

表示单词出现的相对频率。

### 3.2.2 条件概率

计算连续两个词的概率，例如"learning"在"deep"之后出现的概率：

$$
P(\text{learning} \mid \text{deep}) = \frac{\text{"deep learning" 出现的次数}}{\text{"deep" 出现的次数}}
$$

随着序列长度增加，会出现许多组合从未出现过，导致概率为 0，这就是**零概率问题**。

### 3.2.3 平滑技术

引入 **拉普拉斯平滑（Laplace Smoothing）** 给未出现的词分配非零概率：

$$
P_{\text{smooth}}(w) = \frac{\text{出现次数} + 1}{\text{总词数} + V}
$$

更一般的形式：

$$
P(w) = \frac{\text{次数} + \alpha}{\text{总词数} + \alpha V}
$$

- $V$ 是词表大小  
- $\alpha$ 控制平滑强度  
  - $\alpha = 0$ 时无平滑  
  - $\alpha$ 很大时接近均匀分布

语言模型通过统计词与词之间的共现规律，利用概率和条件概率描述语言结构；平滑技术保证模型面对未见序列时仍能给出合理概率估计。

## 3.3 马尔可夫模型与 n-gram

### 马尔可夫模型

如果序列满足 **一阶马尔可夫性质**：

$$
P(w_n \mid w_1, w_2, \dots, w_{n-1}) \approx P(w_n \mid w_{n-1})
$$

当前词只依赖前一个词。阶数越高，模型捕捉的依赖关系越长，为序列建模提供近似公式。


根据上下文长度不同，可定义：

- **一元语法（Unigram）**  
  只考虑单个词出现概率，不包含上下文。  
  示例：句子"猫 喜欢 睡觉"中  
  $$
  P(\text{猫}) = \frac{1}{3}
  $$

- **二元语法（Bigram）**  
  考虑每个词在前一个词条件下出现概率。  
  示例：
  $$
  P(\text{睡觉} \mid \text{猫}) = 0
  $$
  $$
  P(\text{喜欢} \mid \text{猫}) = 1
  $$

- **三元语法（Trigram）**  
  考虑每个词在前两个词条件下出现概率。  
  示例：
  $$
  P(\text{睡觉} \mid \text{猫 喜欢}) = 1
  $$

**特点**：

- 通过统计固定长度词组合出现频率，捕捉词之间依赖关系  
- 随着 n 增大：  
  - 模型复杂度增加  
  - 数据稀疏问题加剧  
- 实际应用中需在上下文长度与计算代价间平衡

## 3.4 齐普夫定律

自然语言中，词频遵循 **幂律分布**（Zipf's Law）：

- 高频词（如 `the`、`of`）出现非常频繁  
- 绝大多数词出现频率低  
- 双对数坐标下，词序号与频率点近似落在直线上

**对语言模型的影响**：

- 频率分布极不均衡，未平滑的 n-gram 模型表现差  
- 低频词或未出现的组合概率为 0，需要平滑技术解决

## 3.5 多元语法与组合建模

- 任意词理论上都有可能出现，因此通常使用 **n-gram** 考虑词组合  
- 示例：词集 `吃、苹果、桌子、电脑、香蕉、手机`  
  - 两两组合总数 \(6^2 = 36\)  
  - 但并非所有组合合理（如“吃桌子”不常见）  
- 数据训练可学习语言规律，使模型生成合理概率分布


## 3.6 代码实践

### 3.6.1 自然语言统计

**基本统计**：
```python
import random
import torch
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())
# 将所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
vocab.token_freqs[:10]
```

![词频统计结果](figures/image-23.png)


**词频图绘制**：

```python
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

![词频分布图](figures/image-24.png)

**Zipf 定律分析**：

在自然语言中，词频与排名的关系可表示为：

$$
n(x) \propto \frac{1}{x}
$$

其中，$n(x)$ 表示排名为 $x$ 的词的出现频率。在对数–对数坐标下，取对数后：

$$
\log n(x) = \log k - \log x
$$

这样，原本的双曲线关系变成了**一条直线**，方便观察和分析。这说明词频分布遵循幂律规律——少数高频词出现频繁，而多数低频词出现稀少。

这告诉我们想要通过计数统计和平滑来建模单词是不可行的，因为这样建模的结果会大大高估尾部单词的频率。


### 3.6.2 二元语法统计

```python
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

![二元语法词频](figures/image-26.png)

### 3.6.3 三元语法统计

```python
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

![三元语法词频](figures/image-27.png)

### 3.6.4 对比分析

```python
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

![一元、二元、三元语法对比](figures/image-28.png)

图表显示，n-gram 的阶数越高，词频分布越稀疏。这意味着我们可以通过这个模型学习和预测。

## 3.7 长序列数据处理

在实际语言建模中，文本序列往往非常长，无法一次性全部输入模型。因此，需要对序列进行拆分，以便模型能够高效读取和训练。

### 1. 序列拆分

- **基本思路**：将长序列切成若干段，然后依次取出训练。
- **问题**：如果每次切分都是固定起点，模型总是看到相同的序列，容易丢失某些数据规律。
- **解决方法**：引入**随机起点** \(k\)。
  - \(k\) 的取值范围在 \((0, \tau)\)，其中 \(\tau\) 是窗口长度。
  - 通过随机起点，确保所有序列顺序都能被模型学习到，增加数据的覆盖性和随机性。

### 2. 窗口滑动与样本生成

- **窗口大小**：保持一致，以便并行训练。
- **生成样本**：
  - 假设序列为 `[1, 2, 3, 4]`，窗口大小为 3：
    ```
    X: [1, 2, 3]
    Y: [2, 3, 4]
    ```
  - 通过滑动窗口，每次生成对应的输入 X 和目标 Y。

### 3. 索引构建

- 数据从语料库加载时，核心在于**创建良好的索引**。
- 索引的作用：
  - 确定每个窗口的起点和终点
  - 保证数据加载顺序正确
  - 支持并行训练时的数据分配
- 利用索引，可以灵活生成训练样本，而不需要每次都操作原始序列。

### 4. 批次处理策略

有两种常见方法：

1. **先分批再切分**：
   - 先将数据分成若干批次
   - 在每个批次内切分序列
   - 可以在切分后打乱批次和序列索引，增加随机性

2. **先切分再分批**：
   - 先按顺序切好所有序列
   - 然后再分成批次
   - 每个批次中再进行序列切分
   - 切分次数与批次数量一致
   - 最终生成训练样本

### 5. 并行训练考虑

- 窗口长度相等保证每个批次可以并行处理
- 随机起点和批次打乱确保训练数据覆盖充分，避免模型过度依赖固定序列模式
- 这种处理方式既保证了数据的**覆盖性**，又保持**随机性**，有利于模型更好地学习长序列中的规律


### 代码实现： 

### 随机采样


```python

corpus 是处理好的语料库，之前上面生成的是词表，不要搞混乱或者搞错。

def seq_data_iter_random(corpus, batch_size, num_steps):
    """
    随机顺序的序列数据迭代器
    
    参数:
    corpus: 原始序列（list或array），比如 [0,1,2,3,4,5,6,...]
    batch_size: 每个小批量中包含的子序列数量
    num_steps: 窗口的大小，每个子序列包含的时间步数

    功能：
    - 将原始序列切成长度为 num_steps 的多个子序列
    - 对每个子序列对应的标签 Y 为右移一位的版本
    - 随机打乱顺序，按 batch_size 返回每个批次的 X 和 Y
    """

    # 从 corpus 中随机丢掉最前面几个元素，使得起始位置随机
    corpus = corpus[random.randint(0, num_steps - 1):]  
    # 注意：random.randint(0, num_steps - 1) 会返回一个整数，前面的数据会被永久丢弃

    # 计算可以切出多少个完整子序列
    # 整数除法，向下取整，保证 Y 序列能对应上
    num_subseqs = (len(corpus) - 1) // num_steps

    # 生成每个子序列对应的起始索引
    # 比如 [0, num_steps, 2*num_steps, ...]
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))

    # 随机打乱起始索引顺序，保证每次训练样本顺序不同
    random.shuffle(initial_indices)

    # 内部函数：根据起始位置 pos 返回长度为 num_steps 的子序列
    def data(pos):
        return corpus[pos: pos + num_steps]

    # 计算总共有多少个 batch，小样本集合
    # 每个小样本集合， 包含 batch_size 个子序列，窗口
    # 总的窗口数，
    num_batches = num_subseqs // batch_size  

    # 迭代每个 batch，装数据的时候，
    首先看在哪个批次，从第一批次开始，然后每个批次，
    有batch_size个窗口 或者序列，每个窗口的大小是num_steps.
    一次训练中，有2个序列/2个窗口，batch_size 也是数量

    for i in range(0, batch_size * num_batches, batch_size): 下面有个图可以解释
        # 选择当前 batch 的初始索引，针对每一个批次
        initial_indices_per_batch = initial_indices[i: i + batch_size]


        initial_indices 是生成每个子序列对应的**起始索引**，里面是随机的.
        间隔肯定就是窗口的大小，假设是 5

        initial_indices = [0，5，10，15....]

        initial_indices = [25,5,20,......] 打乱

        假设第一个批次，有2个窗口，batch_size = 2. num_batche = 3（没有关系）

        initial_indices_per_batch = initial_indices[0,2] = [25,5]

        然后，这个就是第一个批次，2个窗口的起始索引，来自corpus

        # 构建当前 batch，批次 的 X 和 Y，看batch_size 有 2个，就循环2次，
        #这也是为什么initial_indices[i: i + batch_size] 的原因，然后在等于给initial_indices_per_batch

        X = [data(j) for j in initial_indices_per_batch]      # 输入序列

        针对第一个窗口，25，返回就是原始数据[25,25+5]
        针对第二个窗口，5，返回就是原始数据[5,5+5]

        Y = [data(j + 1) for j in initial_indices_per_batch]  # 标签序列，右移一位
        最后，返回一个批次，batch_size 大小的窗口的原数据
        # 使用 yield 返回当前 batch 的 X 和 Y，迭代器方式，每次返回一个 batch
        yield torch.tensor(X), torch.tensor(Y)

```
![批次处理逻辑示意图](figures/batch_logic_clear-1.png)

```python
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]

    offset = random.randint(0, num_steps - 1)
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
```

### 顺序分区
```python
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps) 

    从 0 到 窗口的大小之间选择一个数字，用于跳过
    
    为了避免模型每次都从相同位置开始，随机跳过前面一小段。

    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    
    '''

    整体的序列长度减去 offset， 用于构造 X 和 Y（对应的减1）

    向下整除得到可以完整分成 batch_size 的组数， 这个是每一个批次中，要训练的窗口数。

    是先把语料切成几行（批次的行），然后再按时间窗口（num_steps）滑动.

    如何构造，通过最终的维度进行反推： 每次迭代的是一个批次，每个批次中的维度是 
    
    X.shape == (batch_size, num_steps)

    这里和上面的不同就在于，它是竖着切。以一个例子来说明： 

    corpus = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    batch_size = 2
    num_steps = 3

    num_tokens = ((18 - 1) // 2) * 2 = 16

    [[0, 1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13, 14, 15]]

    num_batches = 8 // 3 = 2 个批次

    X = [[0, 1, 2],
     [8, 9, 10]]
    Y = [[1, 2, 3],
       [9, 10, 11]]

    X = [[3, 4, 5],
     [11, 12, 13]]
    Y = [[4, 5, 6],
     [12, 13, 14]]

    '''

    Xs = np.array(corpus[offset: offset + num_tokens]) 

    创建数组的函数

    Ys = np.array(corpus[offset + 1: offset + 1 + num_tokens])

    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)

    是 NumPy 数组的操作，不是 Python 列表的

    num_batches = Xs.shape[1] // num_steps 切分得到批次

    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

为了帮助理解，制作如下图，去理解整体路程。


![顺序分区示意图](figures/seq_partition.png)


```python
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

```


最终合并在一起


```python
class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        
        定义了基本的参数：
        
        '''

        batch_size, 小批次中有几个样本
        num_steps, 窗口的大小
        use_random_iter 使用乱序加载器
        max_tokens: 最大的序列长度
        '''
        判断使用随机还是顺序加载

        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential

        使用之前的加载函数，获取原始语料以及对应的词表

        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        
        将参数定义为对象参数，方便调用和使用

        self.batch_size, self.num_steps = batch_size, num_steps


    def __iter__(self): 
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
        # 这里就定义了如何是使用顺序数据加载，还是乱序数据加载。
        # d2l.seq_data_iter_random / d2l.seq_data_iter_sequential



def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):

    """返回时光机器数据集的迭代器和词表"""
    建设一个对象，返回对象及其词表
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
```


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
---
## 3.7 问题


## 1️⃣ 四元语法需要存储多少词频和相邻的词频？

假设训练数据集中有 100,000 个单词，词频 = 单个词出现的次数，
几个连续相邻的词出现的频率，也就是所谓的 n-gram 频率（n元语法频率）

- **四元语法（4-gram）**：  词频 + 相邻多词频

即 1-gram、2-gram、3-gram、4-gram 的频数总数

  \[
  Nmax​=V+V^2+V^3+V^4
  \]
  
  理论上所有可能组合为  

  \[
  100{,}000^4 = 10^{20}
  \]



 但实际语料中极少出现，因此只存出现过的 n-gram，有些是不会出现的，比如说 
 ‘你 爱 我 嘛’  不可能是‘嘛 你 爱 我’ 并用平滑、截断、压缩、哈希或神经网络等方法处理稀疏性。


## 2️⃣ 如何对一系列对话建模？

### 🔹 统计语言模型（n-gram）
使用概率模型：
\[
P(w_t | w_{t-1}, w_{t-2}, \dots)
\]
- 优点：简单直观  
- 缺点：上下文受限、稀疏问题严重

### 🔹 神经语言模型（RNN / LSTM / Transformer）
- 通过神经网络学习上下文依赖关系  
- 参数共享，能建模长距离依赖  
- 适合多轮对话建模

---

## 3️⃣ 一元、二元、三元语法的齐普夫定律指数是否不同？

齐普夫定律：
\[
f_i \propto \frac{1}{i^\alpha}
\]
其中 \( f_i \) 为词频，\( i \) 为词排名。

| 语法类型 | 典型指数 α |
|-----------|-------------|
| 一元语法 | ≈ 1.0 |
| 二元语法 | ≈ 1.1–1.2 |
| 三元语法 | ≈ 1.3–1.5 |

**估计方法：**  
在 log–log 坐标中拟合直线：
\[
\log f_i = -\alpha \log i + C
\]
斜率的绝对值即为 α


## 4️⃣ 读取长序列数据的其他方法

- 滑动窗口采样（可重叠）
- 按句子或段落分割
- 按对话轮次采样
- 动态长度采样，改变窗口的大小（可变 `num_steps`）


## 5️⃣ 为什么使用随机偏移量是个好主意？

- 防止模型记住固定的分割点  
- 提高样本多样性  
- 减少过拟合，增强泛化能力


## 6️⃣ 它能实现完美的均匀分布吗？

❌ 不完全。  
偏移量通常只在 `[0, num_steps - 1]` 范围内随机，  
分布仍会略有偏差，词出现的频率相同，这就是“均匀分布”



## 7️⃣ 如何让分布更均匀？

模型能更好地学到低频但重要的词组合，极度不均匀的数据会导致模型“偏向高频类”，效果差，模型学习得更稳定、更全面。



- 扩大随机偏移范围  
- 每个 epoch 打乱语料顺序  
- 随机选择句子起点  
- 或使用掩码语言建模（masking）



## 8️⃣ 如果序列样本是完整句子，会有什么问题？如何解决？

### ❌ 问题：
- 每个句子长度不同，无法组成统一批次（batch）

### ✅ 解决方案：
1. **Padding（填充）**：短句补 0  
2. **Bucketing（分桶）**：按句长分桶  
3. **Dynamic Batching（动态批次）**：动态调整 batch 大小  
4. **Packed Sequence（打包序列）**：如 PyTorch 的 `pack_padded_sequence`
