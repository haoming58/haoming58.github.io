---
layout: note_with_toc
title: 1. 序列建模基础
description: Basic concepts and principles of sequence modeling
category: Machine Learning
tags: [RNN, Sequence Modeling, Deep Learning, Neural Networks]
permalink: /notes/RNN序列建模基础/
---

# 1. 序列建模基础

## 1.1 什么是序列模型？

有些数据是“有顺序”的，比如：

* 股票价格每天的变化
* 一个人说话时的语音
* 天气的温度变化

这些数据的**顺序不能打乱**，否则意义就全变了。
👉 所以我们需要**序列模型**，来理解“过去会怎样影响未来”。


## 1.2 常见的统计方法

### 1.2.1 自回归模型（AR）

**直观理解**：
预测未来，靠的是“过去的自己”。

* 比如预测今天的股票价格，我们可以参考前几天的价格。
* 但数据太多会很麻烦，所以只看最近的一段（比如最近 3 天），这就叫**窗口 $\tau$**。

公式写出来：

$$
x_t = P(x_t \mid x_{t-1}, x_{t-2}, \ldots, x_{t-\tau})
$$

📌 **生活类比**
就像你复习考试，只需要看最近几天的复习笔记，不可能把所有学过的都重新背一遍。

### 1.2.2 隐变量自回归模型（Latent AR）

问题来了：
如果历史特别长（比如看 1 年的股价），光靠窗口还是太复杂。

于是我们想了个办法：
👉 用一个“总结笔记” $h_t$，把过去的内容都总结进去。
以后预测的时候，就直接参考这份总结，而不是看所有历史。

![1758904905325](image/2025_8_26/1758904905325.png)

公式：

$$
x_t \sim P(x_t \mid h_t), \quad h_t = f(h_{t-1}, x_{t-1})
$$

📌 **生活类比**
就像上课时，你不会把老师说的每一句话都记下来，而是做一份“课堂总结”。以后复习考试时，你看的是总结，而不是完整课堂录音。

### 1.2.3 马尔可夫模型

马尔可夫模型的思想很简单：
👉 **未来只依赖最近的情况，不依赖更久远的历史。**

* 如果只依赖“上一刻”的状态，这叫“一阶马尔可夫”。
* 比如预测明天天气，只看今天的天气，不看更久之前的。

公式：

$$
P(x_{t+1} \mid x_1,\ldots,x_t) = P(x_{t+1} \mid x_t)
$$

📌 **生活类比**
你决定明天要不要带伞，通常只看今天的天气，不会去管一个月前的天气。

### 1.2.4 因果关系

概率的公式可以随便换顺序，但时间可不是。

👉 现实生活里，**过去影响未来，但未来不能影响过去**。

* 解释 $P(z_{t+1} \mid z_t)$（未来依赖于现在）是合理的；
* 解释 $P(z_t \mid z_{t+1})$（现在依赖未来）就不符合因果关系。

📌 **生活类比**
你今天是否会下雨，不能由“明天会不会下雨”来决定。

## 1.3 总结

这样讲下来：

* 自回归模型：看过去几步
* 隐变量模型：做总结笔记
* 马尔可夫模型：未来只看最近
* 因果关系：只能“由前推后”

---



## 1.3 代码实践

### 1.3.1 原始数据
在代码实践部分，有2块，一是代码解释无法直接复制运行，另外一个是源代码可以直接复制。

使用正弦函数和一些可加性噪声来生成序列数据。

```python
%matplotlib inline 这个用于将代码的执行图片结果直接放在jupyter里面展示，而不是直接以为窗口的形式展示
import torch 导入torch库
from torch import nn 导入nn库
from d2l import torch as d2l 导入d2l库，用于绘制

#共产生 1000 个点
因为，电脑的数据是离散的，当数据点很多，就成为了连续数据。
T = 1000

#生成一个生成一个一维张量
time = torch.arange(1, T + 1, dtype=torch.float32) 

张量的数据类型是 32 位浮点数，这里的也可理解arange生成一系列值【1，T+1）区间是左闭右开。 
总的长度就是T，因为，这里取不到 T+1, 所以生成的值是1，2，3，.....T 。因此，长度为T

# 生成带噪声的正弦信号
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
因为，要相加，维度 T 需要保持一致。
x = sin(0.01 * t) + ε,   其中 ε ~ N(0, 0.2²)
torch.normal(0, 0.2, (T,))
生成一个长度为 T=1000 的噪声张量
每个元素都服从 均值 0、标准差 0.2 的正态分布

# 绘图
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
绘制图像  横坐标是time 长度为T ， 纵坐标是 y 轴数据，需要是列表，即使只有一条曲线也要放在列表里， 后面的是标签，xlim=[1, 1000] 是横坐标的显示范围
```

```python
%matplotlib inline
import torch
from torch import nn
from d2l import torch as d2l

T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```


![1758907002850](image/2025_8_26/1758907002850.png)

### 1.3.2 转化


原始数据有了之后，对于监督学习模型，需要将时间序列切分为特征（feature）和标签（label）。每个特征使用前 `tau` 个连续的值作为输入，这里的 `tau` 就是嵌入维度（窗口大小）。数学上，可以表示为：

\[
X_i = [x_i, x_{i+1}, \dots, x_{i+\tau-1}], \quad Y_i = x_{i+\tau}
\]

例如，如果 `tau = 5`，序列为 `x = [x1, x2, x3, ..., x10]`，则：

```python
# 第1个数据对
X1 = [x1, x2, x3, x4, x5]
Y1 = x6

# 第2个数据对
X2 = [x2, x3, x4, x5, x6]
Y2 = x7

# 第3个数据对
X3 = [x3, x4, x5, x6, x7]
Y3 = x8
```


前 `tau` 个数据点无法作为标签，因为对应的特征需要在它之前的 `tau` 个数据点。例如，如果想将 `x5` 当作标签，则需要特征 `[x0, x1, x2, x3, x4]`，但 `x0` 不存在，因此无法形成完整的特征－标签对。

这里以标签作为基础。 这也就意味着，有多少标签，就有多少对，而最初的前面窗口的大小不能当标签，因此，

样本总数可以用公式计算：

$$
\text{总样本数} = \text{序列长度} - \tau
$$


### 1.3.3 基本数据处理

这里，我们仅使用前600个“特征－标签”对进行训练。

```python

# ============================================================
# 1. 定义特征矩阵
# ============================================================
features = torch.zeros((T - tau, tau))

# 维度说明：
   行数 = (多少对)
   列数 = (每对的序列长度，即窗口长度)

# 举例：
 features = {
   [x1, x2, x3, x4, x5],
   [x2, x3, x4, x5, x6],
   ...
 }

# ============================================================
# 2. 把一维时间序列 𝑥 转换成一个特征矩阵  features，构造所谓的「滞后特征」(lag features)
# ============================================================
  使用按列填充方式（不是按行），总长度为 T - tau
  每一列由原始序列中一段连续数据组成

for i in range(tau):
    features[:, i] = x[i : T - tau + i]

# 说明：
     features取所有行，i 列  
   - X 是一维长序列
   - 通常来说，以左边的第一元素对应的索引i， 和右边第一个原始的索引对齐i，若是无法理解，请看下面的例子
   - 索引i从 0 开始，到tau-1 结束，共tau列。
   - 也可以按行构造，取决于选择


# ============================================================
# 3. 构造标签
# ============================================================
labels = x[tau:].reshape((-1, 1))

# 标签说明：
   - 第一个窗口之后的所有原始数据都是标签，为了对应feature的维度对应
   - 把一维数组转为二维列向量
   - -1 表示自动推算行数
   -  1 表示只有一列


# ============================================================
# 4. 示例
# ============================================================
x   = [x1, x2, x3, x4, x5, x6]
T   = 6
tau = 3

 i = 0
 x[0 : 3] = [x1, x2, x3] → 填到 features[:, 0]

 i = 1
 x[1 : 4] = [x2, x3, x4] → 填到 features[:, 1]

 i = 2
 x[2 : 5] = [x3, x4, x5] → 填到 features[:, 2]


# ============================================================
# 5. 构造训练集迭代器
# ============================================================

batch_size, n_train = 16, 600



# 只有前 n_train 个样本用于训练
train_iter = d2l.load_array(
    (features[:n_train], labels[:n_train]),
    batch_size,
    is_train=True
)
 只有前n_train个样本用于训练，这里维度不是2维的shape， 因此，相当于1维度的 前n_train，因为没有，逗号
 从数据集中取前 600 个样本作为训练集。可以理解600个
```


```python
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)
```
### 1.3.4 模型架构

 - 初始化

```python
init_weights 是一个 函数，用于初始化网络层的权重

def init_weights(m):
    if type(m) == nn.Linear: 判断这个层是不是 nn.Linear（全连接层，只对全连接层初始化，卷积层或其他层不处理）
        nn.init.xavier_uniform_(m.weight) 
        
  用 Xavier 均匀分布初始化权重。
  Xavier 初始化的作用：保持每层输入输出的方差一致，避免梯度消失或爆炸
  _ 结尾表示这个函数会 直接修改权重，而不是返回新值。
  

def get_net():
    nn.Sequential(...)：按顺序组合层
    net = nn.Sequential(nn.Linear(2,10),
                        nn.ReLU(),
                        nn.Linear(10,1))
    net.apply(init_weights) 会递归地对网络中每个层调用 init_weights(m) 
    nn.Linear(2,10) 和 nn.Linear(10,1) 的权重都会用 Xavier 初始化
    return net

loss = nn.MSELoss(reduction = 'none') 计算每个样本的平方误差，不平均
```

```python
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def get_net():
    net = nn.Sequential(nn.Linear(2,10),
                        nn.ReLU(),
                        nn.Linear(10,1))
    net.apply(init_weights)
    return net

loss = nn.MSELoss(reduction = 'none')
```
- 模型训练

```python

def train(net, train_iter, loss, epochs, lr):
#############
# -net → 你的神经网络模型

# -train_iter → 小批量数据迭代器

# -loss → 损失函数（这里是 MSELoss(reduction='none')）

# -epochs → 训练轮数

# -lr → 学习率
##############
    trainer = paddle.optimizer.Adam(learning_rate=lr, parameters=net.parameters())
    使用 Adam 优化器，学习率为 lr
    告诉优化器去更新网络里的所有参数（权重和偏置）

    for epoch in range(epochs):
        for i,(X, y) in enumerate (train_iter()):
            trainer.clear_grad() PyTorch 的梯度是累加的，每次更新前要清零，否则会把前一次的梯度累加上去。只针对每一个批次训练后，使用的梯度更新
            l = loss(net(X), y)
            l.sum().backward() backward()：自动计算梯度，存储在每个参数的 .grad 中。
            trainer.step() 根据梯度更新参数。
        print(f'epoch {epoch + 1}, ' 每个 epoch 打印一次平均训练损失， 因为epoch 是下标0开始
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```
```python
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```


![1759174346320](image/2025_8_26/1759174346320.png)


### 1.3.4 预测 



- **单步预测**是指基于已有的历史观测值预测下一个时间点的数值，每次预测后都有真实值可以直接进行对比，误差不会累积。

```python

onestep_preds = net(features)
输入你的特征值，然后得到预测值，因为，网络是我们以及训练好的，因为在上面的代码，我们已经使用了train。

features: [x1,x2,x3,x4 ]   onestep_preds = [x5
           x2,x3,x4,x5                     x6    ]
               .
               .                            .
               .                            .                             
                                          
d2l.plot([time, time[tau:]], 
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))

绘图时需要两条线：一条是原始数据 x 对应时间 time（用 x.detach().numpy() 转成 numpy），
另一条是模型预测 onestep_preds 对应时间 time[tau:]（同样用 detach().numpy() 转成 numpy），detach() 是把 tensor 从计算图中分离，不再跟踪梯度。
```

```python
onestep_preds = net(features)
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
```


![1759178315073](image/2025_8_26/1759178315073.png)

- **多步预测**是基于历史值预测未来多个时间点，通常通过递归（预测值作为下一步输入）或直接（一次性输出多步结果）方式实现，但递归方式容易导致误差逐步放大。


在代码设计上，可以采用两种方式进行预测：一种是基于完整时间序列进行预测，另一种是使用滑动窗口批量预测。在滑动窗口方式中，每一行代表一个独立的起始时间点，例如第 0 行输入 ([1,2,3]) 预测未来第 1、2 步，第 1 行输入 ([2,3,4]) 预测未来第 1、2 步，行与行之间不具有时间连续性。这种设计的目的不是为了训练模型或生成完整的未来序列，而是用于测试模型在不同预测步长下的性能，分析误差随预测步数变化的规律，并绘制对比曲线，从而评估模型的短期与长期预测能力。仅仅只是为了测试和教学，以一个例子解释。

```python

multistep_preds = torch.zeros(T) 生成一个长度为 T 的张量

multistep_preds[: n_train + tau] = x[: n_train + tau]

之前提到一共有1000步。只用了604步，因为窗口，设置的是4.

这里的数据到604 后面全是0

从数据中取出前 n_train + tau 个元素

for i in range(n_train + tau, T): 也就是标签的后面预测结果是605 
    multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1)))
    
    i的取值是604, 605, 606, ..., 999。
    
    当 i = 604 的时候，对应训练数据就应该是600 601 602 603， 就是 i - tau。这个区间，然后写成net的输入数据的格式 1 行 4 列。 

    循环到999次

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))


绘图时需要3条线：一条是原始数据 x 对应时间 time（用 x.detach().numpy() 转成 numpy），

另一条是模型预测 onestep_preds 对应时间 time[tau:]（同样用 detach().numpy() 转成 numpy），detach() 是把 tensor 从计算图中分离，不再跟踪梯度。

最后一条是多步模型预测 multistep_preds 对应时间 time[n_train + tau]（同样用 detach().numpy() 转成 numpy），detach() 是把 tensor 从计算图中分离，不再跟踪梯度。从604步开始

```


```python
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
```
![1759247745951](image/2025_8_26/1759247745951.png)


- **K步预测** k = 1，4，16，64 通过对整个序列预测的计算， 让我们更仔细地看一下
步预测的困难。在此之前，通过一个例子去理解：




| 历史1 | 历史2 | 1-step | 2-step | 3-step |
| ----- | ----- | ------ | ------ | ------ |
| t1    | t2    | t3     | t4     | t5     |
| t2    | t3    | t4     | t5     | t6     |
| t3    | t4    | t5     | t6     | t7     |


假设总的时间步长为 \(T = 6\)，历史窗口大小 \(\tau = 2\)，数据序列为 t1, t2, t3, t4, t5, t6，最早的标签从 t3 开始。对于 K 步预测：  

 **1-step 预测**：使用前两个时间步 t1、t2 预测下一个值 t3，然后用 t2、t3 预测 t4，以此类推。  
 **2-step 预测**：首先用 t1、t2 预测 t3，然后继续使用 t2、t3 预测 t4，即每一步预测都依赖于前面的历史或预测值。  
 **3-step 预测**：尝试使用 t3、t4 预测 t5，再用 t4、t5 预测 t6，但当需要预测 t7 时失败，因为总序列长度只有 6， 没有真实标签可以训练或验证。

 1-step： t3 t4 t5
 2-step： t4 t5 t6
 3-step： t5 t6 t7

#### 为什么出现公式 \(T - \tau - \text{max\_steps} + 1\)

当我们从一个序列长度 \(T\) 中采样时：

- 有效的预测窗口必须满足：  

  \[
  t + \text{max\_steps} \leq T
  \]

因为预测 \(t+1, \dots, t+\text{max\_steps}\) 都必须落在已有的观测范围内,也就是已经有的数据。

结合输入长度 \(\tau\)：  
- 输入窗口从 \(t-\tau+1\) 开始，到 \(t\) 结束。  
- 最后一次能用来预测的时间点是：  
  \[
  t_{\max} = T - \text{max\_steps}
  \]

- 能采样的窗口数量就是：  
  \[
  N = T - \tau - \text{max\_steps} + 1
  \]

---
```python
# 多步时间序列预测特征构造与可视化

# 1. 背景
# 在时间序列预测中，我们不仅可以进行单步预测（预测下一个时间点），
# 也可以进行多步预测（预测未来多个时间点）。
#
# 参数设定：
# - 最大预测步数：max_steps = 64
# - 历史窗口长度：tau
#
# 为了生成训练样本，我们构造一个特征矩阵 features。

features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))

# 2. 特征矩阵构造
# 行数：训练样本数
# - 单步预测时：T - tau
#   例如序列 (t1, t2, t3, t4, t5, t6)，tau=2 时，可以生成 4 个样本 (t3, t4, t5, t6)。
# - 多步预测时：T - tau - max_steps + 1
#   例如 tau=2, max_steps=3 时，样本数为 2。
#
# 列数：tau + max_steps
# - 前 tau 列：历史观测值
# - 后 max_steps 列：逐步预测得到的未来值

# 3. 填充方式
# 3.1 历史部分
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 每一列是一个时间点的滑动窗口。

# 3.2 预测部分
for i in range(tau, tau + max_steps): 
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

# 从第 tau 列开始逐步预测：
# - tau 列 = 1-step预测
# - tau+1 列 = 2-step预测
# - ...
# 每一步预测都依赖于前 tau 个输入（包括历史和已预测的未来值）。

# 4. 示例说明
# 假设序列为 (t1, t2, t3, t4, t5, t6)，tau=2，max_steps=3：
#
# | 历史1 | 历史2 | 1-step | 2-step | 3-step |
# |-------|-------|--------|--------|--------|
# | t1    | t2    | t3     | t4     | t5     |
# | t2    | t3    | t4     | t5     | t6     |

# 5. 可视化多步预测
steps = (1, 4, 16, 64)

d2l.plot(
    [time[tau + i - 1 : T - max_steps + i] for i in steps],   # 时间点
    [features[:, tau + i - 1].detach().numpy() for i in steps],  # 对应预测值
    'time', 'x',
    legend=[f'{i}-step preds' for i in steps],
    xlim=[5, 1000],
    figsize=(6, 3)
)

# - 横轴：time = [1, 2, ..., T]
#   Python 索引从 0 开始，因此需要 tau + i - 1 来对齐。
# - 纵轴：预测值
# - 图例：1-step preds, 4-step preds, ...

# 6. 总结
# 1. 构造 features 矩阵：
#    - 前 tau 列 = 历史观测
#    - 后 max_steps 列 = 递归预测未来
# 2. 样本数受 T, tau, max_steps 共同限制
# 3. 绘图时根据 steps 选择预测步长，注意时间对齐

```