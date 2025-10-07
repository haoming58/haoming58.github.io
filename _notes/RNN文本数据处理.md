---
layout: note_with_toc
title: 2. 文本数据处理
description: Text data preprocessing and tokenization techniques
category: Machine Learning
tags: [RNN, Text Processing, Tokenization, Natural Language Processing]
permalink: /notes/RNN文本数据处理/
---

文本数据是一种**序列数据**，因此可以使用**序列建模**的方法进行处理。在建模之前，需要理解一些基本概念。

### 核心要点

1. **文本转换**
   计算机无法直接理解原始文本，需要将文本转换为计算机可处理的**字符串格式**。

2. **分词（Tokenization）**
   将字符串拆分为更小的单位，称为**词元（token）**。词元可以是单词、子词或者字符。

3. **词典构建（Lexicon）**
   基于词元构建**词典**，通常会用到**词形还原**来规范化词形。这样有助于模型理解文本，也便于后续的数值转换。

4. **数值化（Numerical Conversion）**
   词元需要被转换为**数值索引（整数）**。计算机最终以二进制存储数值，模型也只能处理数值输入。

---

## 2.1 词元（Token）

文本预处理一般包括以下步骤：

1. **文本清洗**

   * 将所有字母转换为**小写**。
   * 移除所有**非字母字符**。
   * 将单词间的分隔符（如标点符号）替换为空格。

   可使用 Python 的 `re.sub()` 函数实现。

2. **词元定义**
   词元的基本单位可以是**单词**或者**字符**。

   示例：

   * 单词级词元：`"word"` → `"word"`
   * 字符级词元：`"word"` → `"w", "o", "r", "d"`

3. **词典创建**
   分词完成后，构建**词典**，将每个词元映射为**数值索引**，用于模型训练。

   示例：

   | 词元  | 索引 |
   | --- | -- |
   | the | 1  |
   | cat | 2  |
   | sat | 3  |
   | on  | 4  |
   | mat | 5  |

---

## 2.2 代码实践

### 2.2.1 基本库 
```python
import collections 

提供了很多方便的数据结构，比如：

Counter：可以用来统计词频，非常适合做文本数据处理。
defaultdict：带默认值的字典。

import re 
用于文本清洗、模式匹配、替换等操作。
把文本中所有非字母字符替换为空格：

from d2l import torch as d2l
```
```python
import collections
import re
from d2l import torch as d2l
```


### 2.2.2 读取数据 

```python
import re
from d2l import torch as d2l  # 确保你已安装 d2l 库

# ------------------------------
# 1. 配置数据集下载信息
# ------------------------------
d2l.DATA_HUB['time_machine'] = (
    d2l.DATA_URL + 'timemachine.txt',  # 下载地址
    '090b5e7e70c295757f55df93cb0a180b9691891a'  # SHA-1 校验值
)

# 解释：
# 从右往左看：
# d2l.DATA_URL = 'https://d2l-data.s3-accelerate.amazonaws.com/'
# d2l.DATA_URL + 'timemachine.txt' = 'https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
# '090b5e7e70c295757f55df93cb0a180b9691891a' 是 SHA-1 校验值，用于验证文件完整性
# 当下载文件时，d2l.download 会自动计算本地文件的 SHA-1，如果不一致会重新下载

# DATA_HUB 是一个字典，用来存储数据集信息
DATA_HUB = {
    'time_machine': (
        'https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt',
        '090b5e7e70c295757f55df93cb0a180b9691891a'
    ),
    'dataset_name2': ('url2', 'sha1_checksum2'),  # 可以继续添加其他数据集
}

# ------------------------------
# 2. 定义读取函数
# ------------------------------
def read_time_machine():  # @save
    """将《时间机器》数据集加载到文本行列表中，并进行简单预处理"""
    
    # 使用 with 语句（上下文管理器）打开文件
    # 上下文管理器自动在使用完文件后关闭文件，避免占用系统资源
    # open(file_path, 'r') 中 'r' 表示只读模式
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()  # 按行读取文件，返回列表，每行是列表元素

    # 数据预处理：
    # 1. 只保留英文字母，将其他字符替换为空格
    # 2. 去掉行首尾空格
    # 3. 全部转换为小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

    # 解析说明：
    # re.sub(pattern, repl, string)  -> 用 repl 替换 string 中所有匹配 pattern 的部分
    # '[^A-Za-z]+' -> 匹配所有非字母字符（连续一次或多次）
    # strip()      -> 去掉字符串首尾空白
    # lower()      -> 转换为小写
    # 列表推导式   -> 遍历 lines 列表，每行进行处理并生成新的列表

    总的来说，除了 字母以外的其他字符，用空格去表示或者代替


# ------------------------------
# 3. 使用函数读取并查看数据
# ------------------------------

lines = read_time_machine() 使用函数，获得干净的文本列表。

print(f'# 文本总行数: {len(lines)}')
print('第1行:', lines[0])
print('第11行:', lines[10])
```

```python
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])
```

### 2.2.3 词元化

在lines列表中，每一行就是一行文本，现在要做的就是，将一行的文本，转换到最小的基本的单位，词元token： WORD /W O R D


```python 
def tokenize(lines, token='word'):  #@save 
    """将文本行拆分为单词或字符词元""" # 默认 是 word


    if token == 'word': 如果 token = word， 一个词语， the，world，之类

        return [line.split() for line in lines]

        将字符串按照空白字符（空格、制表符 \t、换行 \n 等）拆分成一个列表

        line = "  Hello   world  Python "

        ['Hello', 'world', 'Python']

    elif token == 'char': 这个就是针对单个的字母， A,a 之类

        return [list(line) for line in lines] 

        将字符串拆分成单个字符，生成一个列表

        line = "Hello"

        ['H', 'e', 'l', 'l', 'o']

    else:

        print('错误：未知词元类型：' + token)


tokens = tokenize(lines) 这里默认的word

for i in range(11):
    print(tokens[i])
['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
[]
[]
[]
[]

这里的空格就是之前提到的不是字母类别的东西，就会替换成[].

```

```python 
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

### 2.2.3 词表化

词表化函数有点复杂，拆分讲解，以及不要希望通过一次就能完整地理解应用

```python 
class Vocab:
    """文本词表"""
    
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        初始化函数
        
        self:
          让变量变成实例变量（属于对象，不是临时的）
          让方法变成实例方法（必须通过对象来调用）
        
        参数:
        tokens: 输入的文本序列，通常是分好词的列表，例如 ["我", "爱", "学习", "学习"]
        min_freq: 词的最小出现频率，如果一个词出现次数小于这个值，就不加入词表
        reserved_tokens: 预留的特殊符号列表，比如 ["<pad>", "<bos>", "<eos>"]
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # 按出现频率排序
        counter = count_corpus(tokens)  # 返回字典 {'词': 频率}
        
        # sorted(iterable, key=None, reverse=False)
        # iterable：可迭代对象（列表、元组、字典、字符串等）
        # key：一个函数，告诉Python按什么规则排序
        # key=lambda x: x[1] 表示按照元组里的第二个元素（词频）排序
        # counter.items()： [('学习', 2), ('我', 1), ('爱', 1)] 
        # reverse：是否反转（默认False→升序；True→降序）
        # 根据词频降序排序，得到[(词, 频率), ...]的列表
        # _ 表示内部使用，但是也可调用
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
    
        """
        {key_expression: value_expression for item in iterable if condition}
        """
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        # 创建词元到索引的映射字典
        # enumerate() 的作用是在遍历可迭代对象时，同时得到元素的索引和值
        # enumerate(self.idx_to_token) 返回：[(0, '<unk>'), (1, '<pad>'), (2, '<bos>'), (3, '<eos>')]
        # token: idx for idx, token 是字典推导式的快捷方式


        # 将高频词加入词表
        for token, freq in self._token_freqs: # {'词': 频率} 一系列的字典，提取到token, freq，每一个词元
            if freq < min_freq: # 这里的目的是对于那些不怎么用的词语直接就不管
                break

            if token not in self.token_to_idx: # 这里就是把没有在词表里面的加入进去重复2步操作步骤。

                self.idx_to_token.append(token) 
      # 先是加入词元，假设列表长度原来是 N，那么新加入的词元索引就是 N（因为 Python 的列表索引从 0 开始）
                self.token_to_idx[token] = len(self.idx_to_token) - 1 动态添加
                # 因为列表索引从 0 开始，最后一个元素的索引是 len-1，添加索引到对应的字典的字符
    """
          idx_to_token = ['<pad>', '<unk>', 'the', 'cat']
          token_to_idx = {'<pad>':0, '<unk>':1, 'the':2, 'cat':3}

          self.idx_to_token.append('dog')
          # idx_to_token = ['<pad>', '<unk>', 'the', 'cat', 'dog']

          self.token_to_idx['dog'] = len(self.idx_to_token) - 1
          # len(idx_to_token) = 5
          # len(idx_to_token) - 1 = 4
          # token_to_idx = {'<pad>':0, '<unk>':1, 'the':2, 'cat':3, 'dog':4}

   """
        总结：总的来说，有3个关键点或存储结构，所用到的数据结构是列表、元组、字典的相互结合：

        1. self._token_freqs：存储词频信息，格式为 [('词', 频率), ('词', 频率), ...]
        2. self.idx_to_token：存储索引到词元的映射，包括特殊词元
        3. self.token_to_idx：存储词元到索引的映射

        处理流程：
        - 首先基于完整词频统计创建基础词表
        - 然后通过for循环逐个分析高频词并添加到词表
        - 每次添加新词时，同时更新两个映射结构以保持同步

        举例说明添加新词 'hello' 的过程：

        # 第一步：将词元加入列表
        idx_to_token.append('hello')
        # idx_to_token 现在 = ['<unk>', '<pad>', '<bos>', '<eos>', 'hello']

        # 第二步：在字典中记录对应的索引
        token_to_idx['hello'] = len(idx_to_token) - 1
        # len(idx_to_token) = 5 → 索引 = 4
        # token_to_idx = {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3, 'hello': 4}

       这里，手动把数值输入进去给字典，然后就是hello 没有value，所以就被赋予值了。索引。
       特殊词元有固定的优先索引（0,1,2,3）
```

```python
    def __len__(self):
        return len(self.idx_to_token)

       这就是直接返回词表整体的长度是多少。

       既然有了词表，下面要做的就是2点：

       1. 根据索引找到词元
       2. 根据词元找到索引
      

    def __getitem__(self, tokens): 根据词元找到索引

        首先看是一个词，还是多个词。先判断
            if not isinstance(tokens, (list, tuple)):
                # 判断传入的是单个 token 还是 token 列表。
                # 如果 tokens 不是 list 或 tuple，就认为是单个 token

                # tokens = "I"
                # 或者二维列表的情况
                # tokens = [["I", "am", "this"], ["You", "are", "that"]]

                return self.token_to_idx.get(tokens, self.unk)
                ict.get(key, default) 的作用:value = some_dict.get(key, default)
                # 使用词元到数字的映射表 self.token_to_idx 来查找 tokens 对应的索引
                # 如果 tokens 不在映射表中，就返回 self.unk 就是设置的defaut（通常是未知 token 的索引）
                ict.get(key, default) 的作用:value = some_dict.get(key, default)
                token_to_idx 这个变量是字典
                self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
                这里0就代表未知。
            
            return [self.__getitem__(token) for token in tokens] 针对多个词，获取多个索引。
    
            # 如果 tokens 是列表或元组，就递归调用 __getitem__，就是循环，一行中的每一个词。

            # 将列表里的每个 token（或子列表）都转换成索引 整理
            下面是例子：
            vocab[["I", ["am", "you"]]]  # 返回 [1, [2, 3]]，列表中可以嵌套列表

    def to_tokens(self, indices): 根据索引到对应的词，其规则和上面类似
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0，这里就刚好用到def __getitem__(self, tokens)
        return 0

    @property 外部可以读取 token_freqs，但不能直接修改 _token_freqs，就是self._token_freqs 是私有变量

    def token_freqs(self):
        return self._token_freqs

    def count_corpus(tokens):  # @save 这个程序就是统计词元的频率
        """统计词元的频率"""
        # tokens 可以是 1D 列表（["I", "am", "you"]）或 2D 列表（[["I","am"],["you","are"]])
        
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # 如果 tokens 为空，或者 tokens[0] 是列表，说明 tokens 是二维列表
            # 需要把二维列表展平成一维列表
            tokens = [token for line in tokens for token in line]
            # 这里使用了列表推导式：
            # line 遍历 tokens 中的每一行（每个子列表）
            # token 遍历每一行中的 token
            # 最终生成一个一维的 tokens 列表

        # 使用 collections.Counter 来统计每个 token 出现的次数
        return collections.Counter(tokens)

                tokens1 = ["I", "am", "you", "I"]
        print(count_corpus(tokens1))  
        # 输出: Counter({'I': 2, 'am': 1, 'you': 1})

        tokens2 = [["I", "am"], ["you", "are", "I"]]
        print(count_corpus(tokens2))
        # 输出: Counter({'I': 2, 'am': 1, 'you': 1, 'are': 1})
        严格来说 Counter 是字典的子类，所以可以把它当作字典来使用：
``` 

```python
class Vocab:  #@save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

### 2.2.4 整合

```python
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
是一个字典，所以 .items() 返回的是字典的键值对视图，list() 把字典视图转换成列表
```

```python
有10个子列表
for i in [0, 10]: 这个是表示取[0] [10] 两个
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])
```


```python
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine() 读数据集
    tokens = tokenize(lines, 'char') 清理干净和词元化
    vocab = Vocab(tokens) 生成词表

    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中

    corpus = [vocab[token] for line in tokens for token in line] 
    vocab[token] 是 vocab.__getitem__(token)
    当你使用 [] 访问对象时，Python 自动调用对象的 __getitem__ 方法
    这些就属于语法的一些基本功
    tokens = [["t", "h", "e"], ["t", "i", "m", "e"]]
    {"<unk>":0, "t":1, "h":2, "e":3, "i":4, "m":5}
    corpus = [1, 2, 3, 1, 4, 5, 3]  # 一维索引列表


    if max_tokens > 0: 如果 max_tokens 设定了最大长度，就只保留前 max_tokens 个 token。，默认为 -1，表示使用全部词元
        corpus = corpus[:max_tokens] 取前max_tokens 的数字，索引
    return corpus, vocab
1
corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

### 2.2.4 问题
 - 问题一 常见的词元化
 1. 正则表达式词元化（Regex Tokenization）

```python
import re
text = "ChatGPT is great, isn't it?"
tokens = re.findall(r"\b\w+\b", text) 去掉标点、识别单词、数字
print(tokens)

['ChatGPT', 'is', 'great', 'isn', 't', 'it']
```

 2. NLTK 分词器（Word Tokenizer）

```python
import nltk
from nltk.tokenize import word_tokenize 
nltk.download('punkt')
text = "ChatGPT is great, isn't it?"
tokens = word_tokenize(text) 支持缩写、标点
print(tokens)

['ChatGPT', 'is', 'great', ',', "isn't", 'it', '?']
```

 3. 中文分词：jieba

```python
import jieba
text = "我爱自然语言处理"
tokens = jieba.lcut(text) 针对中文设计，效果优秀。
print(tokens)

['我', '爱', '自然语言处理']
```

 - 问题二 改变Vocab实例的min_freq参数
为了，方便，我直接修改下面的
 ```python
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens，min_freq=100/1000/10000) 自己换啊
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

明显看出，会对词表的大小有影响：


下面我们来详细解释它对 **词表大小（vocabulary size）** 的影响👇

举个例子：

假设语料中的词频统计结果如下：

| 单词       | 频次  |
| -------- | --- |
| the      | 120 |
| deep     | 30  |
| learning | 20  |
| python   | 5   |
| awesome  | 1   |
| amazing  | 1   |

### 情况 1：`min_freq=1`

→ 所有单词都保留
**词表大小 = 6**

### 情况 2：`min_freq=5`

→ 只保留出现≥5次的单词
**词表大小 = 4**（即 `['the', 'deep', 'learning', 'python']`）

### 情况 3：`min_freq=20`

→ 只保留出现≥20次的单词
**词表大小 = 2**（即 `['the', 'deep']`）

---
